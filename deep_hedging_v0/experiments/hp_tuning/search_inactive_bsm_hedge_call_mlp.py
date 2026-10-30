"""Resumable, shared-warmup inactor search for inactive_bsm_hedge_call_mlp only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torchrl
from torchrl.envs import GymWrapper

from experiments.utils.inactor_search import (
    PPOUpdater,
    SearchJointPolicy,
    collect_episode,
    evaluate_policy,
    make_search_actor,
    make_search_inactor,
    rollout_metrics,
)
from experiments.utils.sim_config import EnvConfig, PPOConfig, TrainingConfig, train_test_split
from hedging.envs import HedgeCallBS


ACTOR_LR = 1e-4
DEFAULT_RATES = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

# These manifest fields may change between a run and its --resume: the trial grid
# (lrs/strategies and its expansion), pruning knobs, thread count, and source
# fingerprint. Everything else must match or resume is refused.
FLEXIBLE_SETTING_KEYS = frozenset({
    "inactor_lrs", "strategies", "reduction", "finalists", "threads",
    "trials", "source_digest", "torch_version", "torchrl_version",
})


@dataclass(frozen=True)
class Trial:
    strategy: str
    inactor_lr: float = 1e-3
    initial_hold: float | None = None
    actor_scale: float = 1.0
    actor_every: int = 3
    actor_freeze: int = 0
    ramp_batches: int = 0
    critic_burnin: int = 0
    entropy: float = 0.001
    clip: float = 0.1
    normalize_advantage: bool = False
    target_kl: float = 0.0
    passes: int = 1
    anneal: bool = False
    contextual: bool = False

    @property
    def name(self):
        if self.strategy == "actor_only":
            return self.strategy
        return f"{self.strategy}_lr{self.inactor_lr:.8g}"

    def strength(self, completed):
        if completed < self.critic_burnin:
            return 0.0
        if self.ramp_batches:
            return min(1.0, (completed - self.critic_burnin) / self.ramp_batches)
        return 1.0


def strategies():
    gentle = Trial("init05", initial_hold=0.05, actor_scale=0.1, actor_freeze=32)
    ramp = replace(gentle, strategy="ramp05", actor_freeze=64, ramp_batches=64)
    trust = replace(
        ramp, strategy="trust_region", normalize_advantage=True, clip=0.05, target_kl=0.02
    )
    return {
        trial.strategy: trial for trial in (
            Trial("reference"),
            Trial("frozen", actor_scale=0.0),
            Trial("slow_actor", actor_scale=0.1),
            replace(gentle, strategy="init01", initial_hold=0.01),
            gentle,
            replace(gentle, strategy="init15", initial_hold=0.15),
            ramp,
            replace(ramp, strategy="no_entropy", entropy=0.0),
            trust,
            replace(trust, strategy="critic_warmup", critic_burnin=32, actor_freeze=96),
            replace(trust, strategy="contextual_frozen", contextual=True, actor_scale=0.0),
            replace(trust, strategy="contextual_joint", contextual=True),
            replace(trust, strategy="ppo3", passes=3),
            replace(trust, strategy="annealed", anneal=True),
        )
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for record in records for key in record))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def source_digest():
    sources = sorted((ROOT / "hedging").glob("*.py"))
    sources += [Path(__file__)]
    sources += [ROOT / "experiments" / "utils" / name for name in (
        "inactor_search.py", "ppo_mlp_actor.py", "sim_config.py", "data.py",
    )]
    sources += [ROOT / "experiments" / "option_parameter_estimation" / "bsm_data_sp500_2017_2021.json"]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class InactorSearch:
    def __init__(self, arguments, trials):
        self.args = arguments
        self.trials = {trial.name: trial for trial in trials}
        self.output = arguments.output.resolve()
        self.env_config = EnvConfig()
        self.ppo_config = PPOConfig()
        self.train_parameters, self.test_parameters = train_test_split("bsm", 4, "sp500")
        self.hidden_dim = int(TrainingConfig().hidden_size / np.sqrt(2))
        settings = {
            key: value for key, value in vars(arguments).items()
            if key not in ("output", "resume", "dry_run")
        }
        settings.update({
            "actor_lr": ACTOR_LR, "hidden_dim": self.hidden_dim,
            "environment": {
                key: getattr(self.env_config, key) for key in (
                    "maturity", "r", "num_steps", "transaction_cost", "transaction_fee_rate",
                )
            },
            "ppo": asdict(self.ppo_config),
            "trials": [asdict(trial) for trial in trials],
            "source_digest": source_digest(), "torch_version": str(torch.__version__),
            "torchrl_version": str(torchrl.__version__),
        })
        core = {key: value for key, value in settings.items() if key not in FLEXIBLE_SETTING_KEYS}
        self.signature = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        self.manifest_path = self.output / "manifest.json"
        self.signature_history = []
        if arguments.resume:
            if not self.manifest_path.is_file():
                raise ValueError("--resume requires an existing search output with manifest.json")
            manifest_data = read_json(self.manifest_path)
            old_settings = manifest_data["settings"]
            mismatches = {
                key: (old_settings.get(key), settings.get(key))
                for key in set(old_settings) | set(settings)
                if key not in FLEXIBLE_SETTING_KEYS and old_settings.get(key) != settings.get(key)
            }
            if mismatches:
                details = "; ".join(
                    f"{key}: {old!r} -> {new!r}"
                    for key, (old, new) in sorted(mismatches.items())
                )
                raise ValueError(
                    f"Resume settings changed ({details}); repeat the original flags or use a new output."
                )
            if manifest_data["signature"] != self.signature:
                history = [manifest_data["signature"], *manifest_data.get("signature_history", [])]
                self.signature_history = list(dict.fromkeys(history))
                print(
                    "RESUME: trial grid or search source changed; existing warm-up and completed "
                    "trials are reused, new configurations start from the shared warm-up.",
                    flush=True,
                )
                write_json(self.manifest_path, {
                    **manifest_data, "signature": self.signature, "settings": settings,
                    "signature_history": self.signature_history,
                    "command": shlex.join([sys.executable, *sys.argv]),
                    "resumed_at": datetime.now().isoformat(),
                })
            else:
                self.signature_history = manifest_data.get("signature_history", [])
        else:
            if self.output.exists():
                raise ValueError("Output already exists. Use --resume or choose a new --output directory.")
            write_json(self.manifest_path, {
                "signature": self.signature, "settings": settings, "signature_history": [],
                "command": shlex.join([sys.executable, *sys.argv]),
                "created_at": datetime.now().isoformat(),
            })

    def event(self, directory, payload):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")

    def load(self, path):
        state = torch.load(path, map_location=self.args.device, weights_only=True)
        if state["signature"] != self.signature and state["signature"] not in self.signature_history:
            raise ValueError(f"Checkpoint settings mismatch: {path}")
        return state

    def environment(self, *, evaluation=False, test=False):
        initial_prices, strikes, volatility = self.test_parameters if test else self.train_parameters
        base = HedgeCallBS(
            initial_prices, strikes, self.env_config.maturity, self.env_config.r, volatility,
            self.args.eval_paths if evaluation else self.args.num_paths,
            self.env_config.num_steps, history_len=1,
            transaction_cost=self.env_config.transaction_cost,
            transaction_fee_rate=self.env_config.transaction_fee_rate,
        )
        return GymWrapper(base, device=self.args.device)

    def actor(self, seed):
        torch.manual_seed(seed)
        return make_search_actor(11, self.hidden_dim, self.args.device)

    def path_seed(self, seed, completed, joint=False):
        return seed * 10_000_000 + int(joint) * 1_000_000 + completed // self.args.episodes_per_epoch

    def rollout(self, env, policy, seed, completed, joint=False):
        return collect_episode(
            env, policy, self.env_config.num_steps,
            self.path_seed(seed, completed, joint),
            seed * 10_000_000 + int(joint) * 1_000_000 + 500_000 + completed,
        )

    def validation(self, env, policy):
        strength = policy.gate_strength
        policy.gate_strength = 1.0
        try:
            return evaluate_policy(env, policy, self.env_config.num_steps, self.args.validation_seeds)
        finally:
            policy.gate_strength = strength

    def warmup(self, seed):
        directory = self.output / "warmups" / f"seed_{seed}"
        checkpoint = directory / "last.pt"
        actor = self.actor(seed)
        updater = PPOUpdater(actor, self.ppo_config, ACTOR_LR)
        completed = 0
        if checkpoint.exists():
            state = self.load(checkpoint)
            actor.load_state_dict(state["actor"])
            updater.optimizer.load_state_dict(state["actor_optimizer"])
            completed = state["completed"]
        target = self.args.warmup_epochs * self.args.episodes_per_epoch
        env = self.environment()
        policy = SearchJointPolicy(actor.get_policy_operator())
        started = time.monotonic()
        initial_completed = completed
        try:
            for batch_index in range(completed, target):
                batch = self.rollout(env, policy, seed, batch_index)
                metrics = rollout_metrics(batch)
                losses = updater.update(batch, self.args.sub_batches)
                completed = batch_index + 1
                if completed % self.args.log_every == 0 or completed == target:
                    seconds_per_batch = (time.monotonic() - started) / (completed - initial_completed)
                    remaining_minutes = seconds_per_batch * (target - completed) / 60
                    print(
                        f"WARMUP seed={seed} {completed}/{target} reward={metrics['avg_reward']:.6f} "
                        f"saturation={metrics['actor_saturation_rate']:.3f} "
                        f"remaining~{remaining_minutes:.1f}min", flush=True,
                    )
                    self.event(directory, {"phase": "warmup", "batch": completed, **metrics, "actor": losses})
                if completed % self.args.checkpoint_every == 0 or completed == target:
                    save_checkpoint(checkpoint, {
                        "signature": self.signature, "completed": completed,
                        "actor": actor.state_dict(), "actor_optimizer": updater.optimizer.state_dict(),
                    })
        finally:
            env.close()
        baseline_path = directory / "validation.json"
        if not baseline_path.exists():
            env = self.environment(evaluation=True)
            try:
                write_json(baseline_path, self.validation(env, policy))
            finally:
                env.close()
        print(f"WARMUP READY seed={seed}: {checkpoint}", flush=True)

    def build_trial(self, trial, seed):
        warmup = self.load(self.output / "warmups" / f"seed_{seed}" / "last.pt")
        actor = self.actor(seed)
        actor.load_state_dict(warmup["actor"])
        actor_updater = PPOUpdater(actor, self.ppo_config, ACTOR_LR)
        actor_updater.optimizer.load_state_dict(warmup["actor_optimizer"])
        gate = gate_updater = None
        if trial.strategy != "actor_only":
            torch.manual_seed(seed + 10_000)
            gate = make_search_inactor(
                11, self.hidden_dim, self.args.device, trial.initial_hold, trial.contextual
            )
            gate_updater = PPOUpdater(
                gate, self.ppo_config, trial.inactor_lr, inactor=True,
                entropy_coeff=trial.entropy, clip_epsilon=trial.clip,
                normalize_advantage=trial.normalize_advantage, target_kl=trial.target_kl,
            )
        policy = SearchJointPolicy(
            actor.get_policy_operator(), None if gate is None else gate.get_policy_operator()
        )
        return actor, actor_updater, gate, gate_updater, policy

    def train_trial(self, trial, seed, budget):
        directory = self.output / "trials" / trial.name / f"seed_{seed}"
        rung_path = directory / f"rung_{budget}.json"
        if rung_path.exists():
            return read_json(rung_path)
        actor, actor_updater, gate, gate_updater, policy = self.build_trial(trial, seed)
        checkpoint = directory / "last.pt"
        completed = 0
        best_score = -math.inf
        best_batch = 0
        best_metrics = None
        if checkpoint.exists():
            state = self.load(checkpoint)
            actor.load_state_dict(state["actor"])
            actor_updater.optimizer.load_state_dict(state["actor_optimizer"])
            if gate is not None:
                gate.load_state_dict(state["inactor"])
                gate_updater.optimizer.load_state_dict(state["inactor_optimizer"])
            completed, best_score = state["completed"], state["best_score"]
            best_batch, best_metrics = state["best_batch"], state["best_metrics"]
        for group in actor_updater.optimizer.param_groups:
            group["lr"] = ACTOR_LR * trial.actor_scale
        train_env = self.environment()
        validation_env = self.environment(evaluation=True)

        def snapshot():
            return {
                "signature": self.signature, "trial": asdict(trial), "completed": completed,
                "best_score": best_score, "best_batch": best_batch, "best_metrics": best_metrics,
                "actor": actor.state_dict(), "actor_optimizer": actor_updater.optimizer.state_dict(),
                "inactor": None if gate is None else gate.state_dict(),
                "inactor_optimizer": None if gate_updater is None else gate_updater.optimizer.state_dict(),
            }

        def validate():
            nonlocal best_score, best_batch, best_metrics
            metrics = self.validation(validation_env, policy)
            if metrics["avg_reward"] > best_score:
                best_score, best_batch, best_metrics = metrics["avg_reward"], completed, metrics
                save_checkpoint(directory / "best.pt", snapshot())
            self.event(directory, {"phase": "validation_full_gate", "batch": completed, **metrics})
            return metrics

        try:
            if not checkpoint.exists():
                full_gate = validate()
                policy.gate_strength = trial.strength(0)
                scheduled_gate = evaluate_policy(
                    validation_env, policy, self.env_config.num_steps, self.args.validation_seeds
                ) if trial.strength(0) != 1.0 else full_gate
                baseline = read_json(self.output / "warmups" / f"seed_{seed}" / "validation.json")
                write_json(directory / "activation.json", {
                    "warmup_actor": baseline, "full_gate": full_gate, "scheduled_gate": scheduled_gate,
                    "full_gate_reward_delta": full_gate["avg_reward"] - baseline["avg_reward"],
                    "scheduled_gate_reward_delta": scheduled_gate["avg_reward"] - baseline["avg_reward"],
                })
                save_checkpoint(checkpoint, snapshot())
            for batch_index in range(completed, budget):
                policy.gate_strength = trial.strength(batch_index)
                batch = self.rollout(train_env, policy, seed, batch_index, joint=True)
                metrics = rollout_metrics(batch)
                gate_losses = actor_losses = None
                if gate_updater is not None:
                    factor = 1 - 0.9 * batch_index / max(self.args.joint_batches - 1, 1) if trial.anneal else 1.0
                    for group in gate_updater.optimizer.param_groups:
                        group["lr"] = trial.inactor_lr * factor
                    gate_losses = gate_updater.update(
                        batch, self.args.sub_batches, trial.passes, batch_index < trial.critic_burnin
                    )
                actor_due = (
                    trial.actor_scale > 0 and batch_index >= trial.actor_freeze
                    and (batch_index % self.args.episodes_per_epoch) % trial.actor_every == 0
                )
                if actor_due:
                    actor_losses = actor_updater.update(batch, self.args.sub_batches)
                completed = batch_index + 1
                if completed % self.args.eval_every == 0 or completed == budget:
                    validate()
                if completed % self.args.log_every == 0 or completed == budget:
                    self.event(directory, {
                        "phase": "train", "batch": completed, "gate_strength": policy.gate_strength,
                        "actor_lr": ACTOR_LR * trial.actor_scale,
                        "inactor_lr": None if gate_updater is None else gate_updater.optimizer.param_groups[0]["lr"],
                        **metrics, "actor": actor_losses, "inactor": gate_losses,
                    })
                    print(
                        f"TRIAL {trial.name} seed={seed} {completed}/{budget} "
                        f"reward={metrics['avg_reward']:.6f} hold={metrics['hold_rate']:.3f} "
                        f"best_validation={best_score:.6f}", flush=True,
                    )
                if completed % self.args.checkpoint_every == 0 or completed == budget:
                    save_checkpoint(checkpoint, snapshot())
            result = {
                "trial": trial.name, "seed": seed, "budget": budget,
                "best_batch": best_batch, "best_validation": best_metrics,
                "checkpoint": str(directory / "best.pt"),
            }
            write_json(rung_path, result)
            return result
        finally:
            train_env.close()
            validation_env.close()

    def ranking(self, results):
        rows = []
        for trial_name in sorted({result["trial"] for result in results}):
            records = [result for result in results if result["trial"] == trial_name]
            if len(records) != len(self.args.seeds):
                continue
            deltas = [
                record["best_validation"]["avg_reward"]
                - read_json(self.output / "warmups" / f"seed_{record['seed']}" / "validation.json")["avg_reward"]
                for record in records
            ]
            rows.append({
                "trial": trial_name, "budget": records[0]["budget"], "seeds": len(records),
                "validation_reward": float(np.mean([
                    record["best_validation"]["avg_reward"] for record in records
                ])),
                "validation_delta_vs_warmup": float(np.mean(deltas)),
                "delta_std_across_training_seeds": float(np.std(deltas, ddof=1 if len(deltas) > 1 else 0)),
                "validation_cvar95": float(np.mean([
                    record["best_validation"]["tracking_cvar95"] for record in records
                ])),
                "validation_hold_rate": float(np.mean([
                    record["best_validation"]["hold_rate"] for record in records
                ])),
            })
        return sorted(rows, key=lambda row: row["validation_reward"], reverse=True)

    def final_evaluation(self, names, ranking):
        rows = []
        for seed in self.args.seeds:
            baseline_actor = self.actor(seed)
            warmup = self.load(self.output / "warmups" / f"seed_{seed}" / "last.pt")
            baseline_actor.load_state_dict(warmup["actor"])
            baseline_policy = SearchJointPolicy(baseline_actor.get_policy_operator())
            env = self.environment(evaluation=True, test=True)
            try:
                baselines = {
                    mode: evaluate_policy(
                        env, baseline_policy, self.env_config.num_steps, self.args.test_seeds,
                        deterministic=mode == "deterministic",
                    ) for mode in ("stochastic", "deterministic")
                }
                for mode, metrics in baselines.items():
                    rows.append({
                        "trial": "warmup_actor", "seed": seed, "mode": mode,
                        "test_reward_delta": 0.0,
                        **{key: value for key, value in metrics.items() if key != "replicates"},
                    })
                for name in names:
                    trial = self.trials[name] if name != "actor_only" else Trial("actor_only", actor_every=1)
                    directory = self.output / "trials" / name / f"seed_{seed}"
                    report_path = directory / "test.json"
                    if report_path.exists():
                        rows.extend(read_json(report_path)["rows"])
                        continue
                    actor, _, gate, _, policy = self.build_trial(trial, seed)
                    state = self.load(directory / "best.pt")
                    actor.load_state_dict(state["actor"])
                    if gate is not None:
                        gate.load_state_dict(state["inactor"])
                    trial_rows = []
                    details = {}
                    for mode in ("stochastic", "deterministic"):
                        metrics = evaluate_policy(
                            env, policy, self.env_config.num_steps, self.args.test_seeds,
                            deterministic=mode == "deterministic",
                        )
                        details[mode] = metrics
                        trial_rows.append({
                            "trial": name, "seed": seed, "mode": mode,
                            "best_batch": state["completed"],
                            "test_reward_delta": metrics["avg_reward"] - baselines[mode]["avg_reward"],
                            **{key: value for key, value in metrics.items() if key != "replicates"},
                        })
                    write_json(report_path, {"rows": trial_rows, "details": details})
                    rows.extend(trial_rows)
            finally:
                env.close()
        controls = {
            (row["seed"], row["mode"]): row["avg_reward"]
            for row in rows if row["trial"] == "actor_only"
        }
        for row in rows:
            row["test_delta_vs_actor_only"] = row["avg_reward"] - controls[row["seed"], row["mode"]]
        write_csv(self.output / "test_results.csv", rows)
        summary = []
        for name in ["warmup_actor", *names]:
            for mode in ("stochastic", "deterministic"):
                selected = [row for row in rows if row["trial"] == name and row["mode"] == mode]
                deltas = [row["test_reward_delta"] for row in selected]
                summary.append({
                    "trial": name, "mode": mode, "training_seeds": len(selected),
                    "test_reward": float(np.mean([row["avg_reward"] for row in selected])),
                    "test_delta_vs_warmup": float(np.mean(deltas)),
                    "test_delta_std": float(np.std(deltas, ddof=1 if len(deltas) > 1 else 0)),
                    "test_delta_vs_actor_only": float(np.mean([
                        row["test_delta_vs_actor_only"] for row in selected
                    ])),
                    "test_cvar95": float(np.mean([row["tracking_cvar95"] for row in selected])),
                    "test_hold_rate": float(np.mean([row["hold_rate"] for row in selected])),
                    "test_turnover": float(np.mean([row["turnover"] for row in selected])),
                })
        write_csv(self.output / "summary.csv", summary)
        winner = ranking[0]["trial"]
        write_json(self.output / "best_config.json", {
            "selected_by": "mean stochastic validation reward, never test reward",
            "trial": asdict(self.trials[winner]), "actor_lr": ACTOR_LR,
            "validation": ranking[0],
            "checkpoints": {
                str(seed): str(self.output / "trials" / winner / f"seed_{seed}" / "best.pt")
                for seed in self.args.seeds
            },
        })
        print(f"\nVALIDATION WINNER: {winner}", flush=True)
        for row in summary:
            if row["mode"] == "stochastic":
                print(
                    f"{row['trial']}: test_reward={row['test_reward']:.6f} "
                    f"delta_vs_warmup={row['test_delta_vs_warmup']:+.6f} "
                    f"delta_vs_actor_only={row['test_delta_vs_actor_only']:+.6f} "
                    f"CVaR95={row['test_cvar95']:.6f}", flush=True,
                )
        print(f"RESULTS: {self.output / 'summary.csv'}", flush=True)

    def activation_report(self):
        rows = []
        for path in sorted((self.output / "trials").glob("*/seed_*/activation.json")):
            report = read_json(path)
            rows.append({
                "trial": path.parent.parent.name, "seed": int(path.parent.name.removeprefix("seed_")),
                "warmup_reward": report["warmup_actor"]["avg_reward"],
                "full_gate_reward": report["full_gate"]["avg_reward"],
                "scheduled_gate_reward": report["scheduled_gate"]["avg_reward"],
                "full_gate_reward_delta": report["full_gate_reward_delta"],
                "scheduled_gate_reward_delta": report["scheduled_gate_reward_delta"],
                "full_gate_hold_rate": report["full_gate"]["hold_rate"],
                "scheduled_gate_hold_rate": report["scheduled_gate"]["hold_rate"],
            })
        write_csv(self.output / "activation.csv", rows)

    def run(self):
        for seed in self.args.seeds:
            self.warmup(seed)
        active = list(self.trials)
        protected = [name for name, trial in self.trials.items()
                     if trial.strategy == "reference" and trial.inactor_lr == 1e-3]
        all_rows = []
        ranking = []
        for rung_index, budget in enumerate(self.args.rungs):
            rung_file = self.output / f"rung_{rung_index}.json"
            if rung_file.exists():
                record = read_json(rung_file)
                active = [name for name in record["promoted"] if name in self.trials]
                dropped = [name for name in record["promoted"] if name not in self.trials]
                if dropped:
                    print(
                        f"Skipping {len(dropped)} promoted trials no longer in the grid: {dropped}",
                        flush=True,
                    )
                ranking = record["ranking"]
                all_rows.extend(ranking)
                continue
            print(f"\nRUNG {rung_index + 1}: {len(active)} trials, {budget} total batches each", flush=True)
            results = []
            for name in active:
                try:
                    for seed in self.args.seeds:
                        results.append(self.train_trial(self.trials[name], seed, budget))
                except Exception:
                    failure = {"trial": name, "budget": budget, "traceback": traceback.format_exc()}
                    self.event(self.output / "failures", failure)
                    print(f"FAILED {name}: see {self.output / 'failures' / 'history.jsonl'}", flush=True)
            ranking = self.ranking(results)
            if not ranking:
                raise RuntimeError("No trial completed this rung on every requested seed; see failures/history.jsonl")
            keep = max(1, math.ceil(len(ranking) / self.args.reduction))
            active = [row["trial"] for row in ranking[:keep]]
            active += [name for name in protected if name not in active and any(row["trial"] == name for row in ranking)]
            write_json(rung_file, {"budget": budget, "ranking": ranking, "promoted": active})
            all_rows.extend(ranking)
            write_csv(self.output / "leaderboard.csv", all_rows)
        control = Trial("actor_only", actor_every=1)
        for seed in self.args.seeds:
            self.train_trial(control, seed, self.args.joint_batches)
        write_csv(self.output / "leaderboard.csv", all_rows)
        self.activation_report()
        names = [row["trial"] for row in ranking if row["trial"] in self.trials][:self.args.finalists]
        names += [name for name in protected if name not in names and any(row["trial"] == name for row in ranking)]
        self.final_evaluation([*names, "actor_only"], ranking)


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args():
    env_config, training_config = EnvConfig(), TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("inactor_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the search plan without training or creating output.")
    parser.add_argument("--smoke", action="store_true", help="Tiny plumbing check; not a meaningful optimization run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--inactor-lrs", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--strategies", nargs="+", choices=list(strategies()), default=list(strategies()))
    parser.add_argument("--warmup-epochs", type=positive_int, default=training_config.policy_epochs)
    parser.add_argument("--episodes-per-epoch", type=positive_int, default=training_config.num_episodes,
                        help="Batches per warm-up epoch and between training-path refreshes.")
    parser.add_argument("--joint-batches", type=positive_int, help="Final joint budget; defaults to inaction_epochs * episodes_per_epoch.")
    parser.add_argument("--rungs", type=positive_int, nargs="+", help="Increasing cumulative screening budgets, ending at joint-batches.")
    parser.add_argument("--reduction", type=positive_int, default=3, help="Keep roughly 1/reduction after each rung; 1 disables pruning.")
    parser.add_argument("--finalists", type=positive_int, default=4)
    parser.add_argument("--num-paths", type=positive_int, default=env_config.num_paths)
    parser.add_argument("--eval-paths", type=positive_int, default=100)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[100001, 100003, 100019])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[200003, 200009, 200017])
    parser.add_argument("--sub-batches", type=positive_int, default=training_config.sub_batch_num)
    parser.add_argument("--eval-every", type=positive_int, default=200)
    parser.add_argument("--checkpoint-every", type=positive_int, default=200)
    parser.add_argument("--log-every", type=positive_int, default=20)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.seeds = [0]
        arguments.inactor_lrs = [1e-3]
        arguments.warmup_epochs = 1
        arguments.episodes_per_epoch = 2
        arguments.joint_batches = 6
        arguments.rungs = [2, 6]
        arguments.num_paths = 2
        arguments.eval_paths = 3
        arguments.validation_seeds = [100001]
        arguments.test_seeds = [200003]
        arguments.eval_every = arguments.checkpoint_every = arguments.log_every = 2
    arguments.joint_batches = arguments.joint_batches or training_config.inaction_epochs * arguments.episodes_per_epoch
    if arguments.rungs is None:
        arguments.rungs = sorted({min(budget, arguments.joint_batches) for budget in (200, 800, 2400)} | {arguments.joint_batches})
    if sorted(set(arguments.rungs)) != arguments.rungs or arguments.rungs[-1] != arguments.joint_batches:
        parser.error("--rungs must be strictly increasing and end at --joint-batches")
    if any(not math.isfinite(rate) or rate <= 0 for rate in arguments.inactor_lrs):
        parser.error("inactor learning rates must be finite and positive")
    for name in ("seeds", "validation_seeds", "test_seeds", "inactor_lrs", "strategies"):
        values = getattr(arguments, name)
        if len(set(values)) != len(values):
            parser.error(f"duplicate values in --{name.replace('_', '-')}")
    for name in ("seeds", "validation_seeds", "test_seeds"):
        if any(seed < 0 for seed in getattr(arguments, name)):
            parser.error("seeds must be nonnegative")
    training_seeds = {
        seed * 10_000_000 + epoch for seed in arguments.seeds
        for epoch in range(arguments.warmup_epochs)
    } | {
        seed * 10_000_000 + 1_000_000 + epoch for seed in arguments.seeds
        for epoch in range(math.ceil(arguments.joint_batches / arguments.episodes_per_epoch))
    }
    if set(arguments.validation_seeds) & set(arguments.test_seeds) or training_seeds & set(arguments.validation_seeds + arguments.test_seeds):
        parser.error("training, validation, and test path seeds must be disjoint")
    if arguments.device == "auto":
        arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    return arguments


def main():
    arguments = parse_args()
    templates = strategies()
    trials = [replace(templates[name], inactor_lr=rate)
              for name in arguments.strategies for rate in arguments.inactor_lrs]
    if arguments.smoke:
        trials = [replace(
            trial, actor_freeze=min(trial.actor_freeze, 1),
            ramp_batches=min(trial.ramp_batches, 1), critic_burnin=min(trial.critic_burnin, 1),
        ) for trial in trials]
    print(
        f"Actor LR={ACTOR_LR:g}; {len(trials)} configurations; seeds={arguments.seeds}; "
        f"warm-up={arguments.warmup_epochs * arguments.episodes_per_epoch} batches/seed; "
        f"rungs={arguments.rungs}; device={arguments.device}", flush=True,
    )
    if arguments.dry_run:
        print(json.dumps([asdict(trial) for trial in trials], indent=2))
        return
    torch.set_num_threads(arguments.threads)
    os.environ["DEEP_HEDGING_PATH_CACHE"] = "0"
    search = InactorSearch(arguments, trials)
    search.run()


if __name__ == "__main__":
    main()