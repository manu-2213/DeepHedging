"""One experiment = one (dynamics, product, arch, net) cell.

  1. load / build the cached market banks (train, validation, test)
  2. Optuna study over the learning rate(s); each trial trains with early
     stopping on the VALIDATION split (fresh paths of the training years)
  3. retrain the best learning rate(s) on the final seeds and evaluate on the
     TEST split (held-out calibration year), which is never used for selection
  4. write summary.json (+ per-seed json, checkpoints, optuna journal)

The Optuna journal makes the job resumable: resubmitting the same command
continues the study and skips final seeds that already finished.

With --time-budget-min the run also fits short Slurm jobs: it saves its full training state in <out>/resume
and exits with code 75 when the budget is used, and the same command continues exactly there.

Example:
  python -m fast_hedging.experiment --dynamics bsm --product call --arch rnn --net single \
      --out results/bsm_call_rnn_single
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .market import MarketSpec, get_bank, default_device
from .ppo import (Agent, OutOfTime, RunCheckpoint, Task, TrainConfig, TrialPruned, _plain, env_index, evaluate, evaluate_baseline, trace_rollout,
                  train_phase, tune_ww_band)

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}       # identical for every experiment
EXIT_RESUME = 75       # the Slurm script resubmits itself on this exit code
DEFAULT_PATHS = {"bsm": {"train": 2000, "val": 500, "test": 2000},
                 "heston": {"train": 1000, "val": 250, "test": 1000}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dynamics", choices=("bsm", "heston"), required=True)
    p.add_argument("--product", choices=("call", "doc", "con"), required=True)
    p.add_argument("--arch", choices=("mlp", "rnn", "transformer"), default="rnn",
                   help="transformer: attention over the same --window as the window LSTM")
    p.add_argument("--advantage", choices=("gae", "grpo"), help="gae (critic, default) | grpo (no critic, group-relative)")
    p.add_argument("--grpo-group", type=int, help="grpo: copies of each path per iteration (default 4)")
    p.add_argument("--grpo-norm", choices=("group", "none"), help="grpo: divide by the group std (default group)")
    p.add_argument("--rnn-mode", choices=("window", "recurrent"), default="window",
                   help="window: LSTM over the last --window observations (history_len, default 8); "
                        "recurrent: one LSTM pass over the whole path (about window-times less LSTM work)")
    p.add_argument("--net", choices=("single", "band", "double"), default="single",
                   help="single: actor-critic; band: actor with a learned no-trade band; double: actor + inaction net")
    p.add_argument("--reward", choices=("mean_cvar", "tracking"), default="mean_cvar",
                   help="mean_cvar: E[PnL] - lambda*CVaR95 via a Rockafellar-Uryasev terminal penalty (gamma 1); "
                        "tracking: original per-step -|PV - L| (gamma 0.99)")
    p.add_argument("--cvar-lambda", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--digital-payout-frac", type=float, default=0.1,
                   help="cash-or-nothing pays this fraction of S0 (<= 0: pay 1, the degenerate v1 setting)")
    p.add_argument("--name", help="experiment name (default: <dynamics>_<product>_<arch>_<net>[_<reward>])")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--market", default="sp500")
    p.add_argument("--underlying-fee", type=float, default=0.01)
    p.add_argument("--option-fee", type=float, default=0.05)
    p.add_argument("--train-paths", type=int); p.add_argument("--val-paths", type=int)
    p.add_argument("--test-paths", type=int)
    # Optuna
    p.add_argument("--n-trials", type=int, default=30, help="finished (complete+pruned) trials to reach")
    p.add_argument("--tuning-hours", type=float, default=24.0, help="wall-clock cap for the study")
    p.add_argument("--actor-lr-range", type=float, nargs=2, default=(1e-6, 3e-3))
    p.add_argument("--inactor-lr-range", type=float, nargs=2, default=(1e-5, 3e-2))
    p.add_argument("--joint-lr-range", type=float, nargs=2, default=(1e-7, 1e-3))
    p.add_argument("--tune-joint-lr", action="store_true",
                   help="tune the joint-phase actor LR separately (default: joint LR = actor LR, old protocol)")
    p.add_argument("--no-pruning", action="store_true")
    p.add_argument("--prune-warmup", type=int,
                   help="validation checks before a trial may be pruned, counted within each phase (default: 8, first phase only)")
    p.add_argument("--tuning-seeds", type=int, nargs="+", default=[0, 1000],
                   help="each trial is trained once per seed; the objective is the mean validation metric "
                        "(pruning uses the first seed). Several seeds make LR selection robust to seed luck")
    p.add_argument("--final-seeds", type=int, nargs="*", default=list(range(1, 11)))
    p.add_argument("--trial-log-every", type=int, default=5, help="keep every k-th iteration in trial train logs")
    # double-net protocol
    p.add_argument("--actor-init", choices=("train", "from-single"), default="train",
                   help="double net stage 1: train its own actor (old protocol) or load the single-net result")
    p.add_argument("--single-dir", type=Path, help="results dir of the matching single-net run (from-single)")
    # training knobs (see ppo.TrainConfig for defaults)
    for name in ("paths-per-iter", "ppo-epochs", "minibatch", "eval-every", "patience", "max-iters",
                 "min-iters", "window", "hidden", "actor-update-every", "lr-patience", "lr-decays"):
        p.add_argument(f"--{name}", type=int)
    p.add_argument("--min-rel-improvement", type=float)
    p.add_argument("--lr-decay-factor", type=float, help="learning-rate multiplier at each plateau cut (default 0.5)")
    p.add_argument("--value-coef", type=float, help="critic loss weight (default 0.1; the old pipeline effectively used 1.0)")
    p.add_argument("--call-action-low", type=float, help="lowest position on a call (default 0; the old single net used -1)")
    p.add_argument("--target-kl", type=float, help="KL limit on the PPO epochs of a rollout (default 0.03; 0 = none, as the old code)")
    p.add_argument("--init-std", type=float, help="policy noise of an untrained net (default: about 1.0)")
    p.add_argument("--time-budget-min", type=float,
                   help="save the training state and exit with code 75 after this many minutes (for chained short jobs)")
    p.add_argument("--no-agent-state", action="store_true",
                   help="old inputs only: hide holdings, tracking error, costs, time since trade, proposed trade")
    p.add_argument("--inactor-init-hold-prob", type=float)
    p.add_argument("--reward-norm", choices=("premium", "spot", "none"))
    p.add_argument("--no-obs-norm", action="store_true"); p.add_argument("--no-adv-norm", action="store_true")
    p.add_argument("--credit-held-actions", action="store_true",
                   help="old behaviour: train the actor on proposals that the inactor overrode")
    p.add_argument("--lmbda", type=float, help="GAE lambda (default 0.95; 1.0 = Monte Carlo advantages)")
    p.add_argument("--cvar-pooled", action="store_true",
                   help="one running VaR over all training assets (v2.0) instead of one per asset (= year)")
    p.add_argument("--value-loss", choices=("mse", "huber"), help="critic loss (default mse; huber = v2.0)")
    p.add_argument("--pad-mode", choices=("edge", "zero"), help="window padding before t=0 (default edge)")
    p.add_argument("--trace-paths", type=int, default=100,
                   help="test paths (x assets x strikes) whose full step-by-step trajectories are saved")
    p.add_argument("--device", default=None)
    p.add_argument("--bank-only", action="store_true", help="only build the market banks (pre-generation)")
    return p.parse_args(argv)


def make_config(a) -> TrainConfig:
    cfg = TrainConfig(arch=a.arch, rnn_mode=a.rnn_mode, net=a.net, reward=a.reward, gamma=a.gamma,
                      underlying_fee=a.underlying_fee,
                      option_fee=a.option_fee, agent_state=not a.no_agent_state, normalize_obs=not a.no_obs_norm,
                      normalize_adv=not a.no_adv_norm, mask_held_actor_loss=not a.credit_held_actions,
                      cvar_per_asset=not a.cvar_pooled)
    over = {k.replace("-", "_"): getattr(a, k.replace("-", "_")) for k in
            ("paths-per-iter", "ppo-epochs", "minibatch", "eval-every", "patience", "max-iters", "min-iters",
             "window", "hidden", "actor-update-every", "lr-patience", "lr-decays", "lr-decay-factor", "min-rel-improvement", "inactor-init-hold-prob",
             "reward-norm", "cvar-lambda", "lmbda", "value-loss", "pad-mode", "value-coef", "init-std", "call-action-low",
             "advantage", "grpo-group", "grpo-norm")}
    cfg = replace(cfg, **{k: v for k, v in over.items() if v is not None})
    if a.target_kl is not None:
        cfg = replace(cfg, target_kl=a.target_kl if a.target_kl > 0 else None)
    return cfg


def load_tasks(a, cfg, device):
    spec = MarketSpec(dynamics=a.dynamics, product=a.product, market=a.market,
                      digital_payout_frac=a.digital_payout_frac)
    sizes = dict(DEFAULT_PATHS[a.dynamics])
    for split in sizes:
        if getattr(a, f"{split}_paths"):
            sizes[split] = getattr(a, f"{split}_paths")
    banks = {s: get_bank(spec, s, sizes[s], SPLIT_SEEDS[s], device=device) for s in ("train", "val", "test")}
    checks = {s: b.check for s, b in banks.items()}
    if a.bank_only:
        return None, spec, sizes, checks
    train = Task(banks["train"], a.product, spec.r, cfg, device)
    val = Task(banks["val"], a.product, spec.r, cfg, device, norm=train.norm)
    test = Task(banks["test"], a.product, spec.r, cfg, device, norm=train.norm)
    return (train, val, test), spec, sizes, checks


class Logger:
    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, msg):
        print(msg, flush=True)
        self.fh.write(msg + "\n")


def save_ckpt(path, agent, cfg, extra=None):
    tmp = Path(str(path) + ".tmp")
    torch.save({"state": agent.state_dict(), "config": asdict(cfg), **(extra or {})}, tmp)
    os.replace(tmp, path)


def _compatible(saved_cfg, cfg):
    keys = ("arch", "rnn_mode", "agent_state", "window", "normalize_obs", "underlying_fee", "option_fee", "reward",
            "reward_norm", "pad_mode")
    return all(saved_cfg.get(k) == getattr(cfg, k) for k in keys)


def run_training(a, cfg, tasks, params, seed, log, report=None, init_actor=None, logs=None, train_log_every=1, ckpt=None):
    """Train one model with fixed hyper-parameters.
    Returns (agent, val_metric, val_metrics, logs) with logs = {"eval": [...], "train": [...]}.
    Pass `logs` to keep the entries even if training is pruned or crashes.
    Pass `ckpt` (RunCheckpoint) to continue a run that an earlier job started."""
    train, val, _ = tasks
    torch.manual_seed(seed); np.random.seed(seed)
    agent = Agent(cfg, a.product, train.device)
    logs = logs if logs is not None else {"eval": [], "train": []}
    saved = ckpt.load() if ckpt is not None else None
    if saved is not None:
        logs.update(saved["logs"])
        if saved["phase"] == "done":
            agent.load_state_dict(saved["agent"])
            return agent, saved["best"], saved["best_metrics"], logs
    kw = dict(report=report, log=log, logs=logs, train_log_every=train_log_every, ckpt=ckpt)
    if cfg.net != "double":          # single and band nets: one phase
        best, m, _ = train_phase(agent, train, val, phase="actor", lr_actor=params["actor_lr"], seed=seed, **kw)
    else:
        if init_actor is not None:
            agent.actor.load_state_dict(init_actor)
        elif saved is None or saved["phase"] != "joint":
            train_phase(agent, train, val, phase="actor", lr_actor=params["actor_lr"], seed=seed, **kw)
        best, m, _ = train_phase(agent, train, val, phase="joint", lr_actor=params["joint_lr"],
                                 lr_inactor=params["inactor_lr"], seed=seed + 10_000, step_offset=100_000,
                                 iter_offset=n_iters(logs), **kw)
    if ckpt is not None:             # survives the job ending before the caller has stored the result
        ckpt.save({"phase": "done", "agent": agent.state_dict(), "best": best, "best_metrics": _plain(m),
                   "logs": _plain(logs)})
    return agent, best, m, logs


def n_iters(logs):
    return int(logs.get("last_global_iter", 0))


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def resolve_protocol(a, cfg, out, log):
    """Decide (once, persisted) how the double net's stage-1 actor is obtained."""
    proto_path = out / "protocol.json"
    if proto_path.exists():
        return json.loads(proto_path.read_text())
    proto = {"actor_init": "train" if cfg.net == "double" else "n/a"}
    if cfg.net == "double" and a.actor_init == "from-single":
        if a.single_dir and (a.single_dir / "best_trial.pt").exists() and (a.single_dir / "summary.json").exists():
            proto = {"actor_init": "from-single", "single_dir": str(a.single_dir.resolve()),
                     "single_actor_lr": json.loads((a.single_dir / "summary.json").read_text())["best_params"]["actor_lr"]}
        else:
            log(f"WARNING: --actor-init from-single but no finished single-net run in {a.single_dir}; "
                "falling back to training the stage-1 actor inside each trial")
    proto_path.write_text(json.dumps(proto, indent=2))
    return proto


def load_single_actor(proto, cfg, name):
    path = Path(proto["single_dir"]) / name
    if not path.exists():
        path = Path(proto["single_dir"]) / "best_trial.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)      # own checkpoints: tensors + primitives only
    if not _compatible(ck["config"], cfg):
        raise RuntimeError(f"single-net checkpoint {path} was trained with an incompatible config")
    return ck["state"]["actor"], str(path)


def main(argv=None):
    try:
        run(parse_args(argv))
    except OutOfTime:
        print("time budget used; state saved, submit the same command again to continue", flush=True)
        sys.exit(EXIT_RESUME)


def run(a):
    device = a.device or default_device()
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True          # LSTM shapes are fixed: autotune once
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / "log.txt")
    cfg = make_config(a)
    name = a.name or f"{a.dynamics}_{a.product}_{a.arch}_{a.net}_{a.reward}"
    t_start = time.time()
    log(f"=== {name} | fees underlying={a.underlying_fee} option={a.option_fee} | reward={cfg.reward} "
        f"gamma={cfg.gamma} | device={device}")
    tasks, spec, sizes, checks = load_tasks(a, cfg, device)
    (a.out / "bank_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    for split, c in checks.items():
        for w in c["warnings"]:
            log(f"WARNING bank[{split}]: {w}")
    log(f"banks ready in {time.time() - t_start:.0f}s: {sizes}")
    if a.bank_only:
        return
    proto = resolve_protocol(a, cfg, a.out, log)
    chained = a.time_budget_min is not None
    deadline = t_start + a.time_budget_min * 60 if chained else None
    resume_dir = a.out / "resume"

    def ck(tag):
        return RunCheckpoint(resume_dir / f"{tag}.pt", deadline) if chained else None

    def clear(pattern):
        for f in resume_dir.glob(pattern):
            f.unlink()

    def once(tag, fn):
        """Chained jobs compute a test-set block once and keep it for later jobs."""
        path = a.out / f"{tag}.json"
        if chained and path.exists():
            return json.loads(path.read_text())
        if chained and time.time() >= deadline:
            raise OutOfTime()
        res = fn()
        if chained:
            path.write_text(json.dumps(res, default=str))
        return res

    from_single = proto["actor_init"] == "from-single"
    if from_single:   # the loaded actor keeps the single-net width (int(91/sqrt 2) = 64)
        cfg = replace(cfg, hidden=int(math.ceil(cfg.hidden * math.sqrt(2))))
    if from_single:
        init_tune, init_src = load_single_actor(proto, cfg, "best_trial.pt")
        log(f"stage-1 actor loaded from {init_src}")

    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from optuna.trial import TrialState

    storage = JournalStorage(JournalFileBackend(str(a.out / "optuna.journal")))
    # each experiment explores its own sequence of start-up points (v1 used seed 0 everywhere)
    sampler_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    pruner = optuna.pruners.NopPruner() if a.no_pruning else optuna.pruners.MedianPruner(
        n_startup_trials=6, n_warmup_steps=8 if a.prune_warmup is None else a.prune_warmup, interval_steps=1)
    study = optuna.create_study(study_name="study", storage=storage, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=8),
                                pruner=pruner)
    if chained:      # a sampler restarted with the same seed would redraw the learning rates of the first job
        study.sampler = optuna.samplers.TPESampler(seed=sampler_seed + len(study.get_trials(deepcopy=False)),
                                                   n_startup_trials=8)
    for tr in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        # A saved trial goes back to the queue with its parameters and reports; anything else was killed.
        paused = chained and any(resume_dir.glob(f"trial{tr.number}_seed*.pt"))
        study._storage.set_trial_state_values(tr._trial_id, TrialState.WAITING if paused else TrialState.FAIL)

    def suggest(trial):
        params = {}
        if cfg.net == "single" or not from_single:
            params["actor_lr"] = trial.suggest_float("actor_lr", *a.actor_lr_range, log=True)
        if cfg.net == "double":
            params["inactor_lr"] = trial.suggest_float("inactor_lr", *a.inactor_lr_range, log=True)
            if a.tune_joint_lr or from_single:
                params["joint_lr"] = trial.suggest_float("joint_lr", *a.joint_lr_range, log=True)
            else:
                params["joint_lr"] = params["actor_lr"]
        return params

    def objective(trial):
        params = suggest(trial)
        log(f"--- trial {trial.number}: {params}")

        def report(step, value):
            trial.report(value, step)
            # the double net's joint phase reports from step 100000, past the pruner's own warmup
            if step % 100_000 >= (a.prune_warmup or 0) and trial.should_prune():
                raise TrialPruned()
        state, t0, runs, agent0 = "failed", time.time(), [], None
        try:
            for i, seed in enumerate(a.tuning_seeds):
                logs = {"eval": [], "train": []}
                runs.append({"seed": seed, "value": None, "val_metrics": None, "logs": logs})
                agent, best_s, m_s, _ = run_training(
                    a, cfg, tasks, params, seed, log, report if i == 0 else None,
                    init_actor=init_tune if from_single else None, logs=logs,
                    train_log_every=a.trial_log_every, ckpt=ck(f"trial{trial.number}_seed{seed}"))
                runs[-1].update(value=best_s, val_metrics=m_s)
                if i == 0:
                    agent0 = agent
            state = "complete"
        except OutOfTime:
            state = "paused"
            raise
        except TrialPruned:
            state = "pruned"
            log(f"--- trial {trial.number} pruned")
            raise optuna.TrialPruned()
        finally:   # full per-trial record, whatever happened (a paused trial carries on in the next job)
            vals = [r["value"] for r in runs if r["value"] is not None]
            value = float(np.mean(vals)) if vals and all(math.isfinite(v) for v in vals) else None
            first = runs[0] if runs else {"logs": {"eval": [], "train": []}, "val_metrics": None}
            trials_dir = a.out / "trials"
            trials_dir.mkdir(exist_ok=True)
            if state != "paused":
                (trials_dir / f"trial_{trial.number:03d}.json").write_text(json.dumps(
                    {"trial": trial.number, "state": state, "params": params,
                     "value": value if state == "complete" else _finite(runs[0]["value"]) if runs else None,
                     "val_metrics": first["val_metrics"], "iters": sum(n_iters(r["logs"]) for r in runs),
                     "seconds": time.time() - t0,
                     "eval_log": first["logs"]["eval"], "train_log": first["logs"]["train"],
                     "per_seed": [{"seed": r["seed"], "value": _finite(r["value"]), "val_metrics": r["val_metrics"],
                                   "iters": n_iters(r["logs"]), "eval_log": r["logs"]["eval"],
                                   "train_log": r["logs"]["train"]} for r in runs]}, default=str))
                if chained:
                    clear(f"trial{trial.number}_seed*.pt")
        best = value if value is not None else math.inf
        m = runs[0]["val_metrics"]
        trial.set_user_attr("val_metrics", m)
        trial.set_user_attr("per_seed_values", [_finite(r["value"]) for r in runs])
        trial.set_user_attr("params_full", params)
        trial.set_user_attr("iters", sum(n_iters(r["logs"]) for r in runs))
        done = [t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))]
        if not done or best < min(t.value for t in done):
            # weights of the first tuning seed; test numbers come from the final seeds anyway
            save_ckpt(a.out / "best_trial.pt", agent0, cfg, {"params": params, "trial": trial.number})
        log(f"--- trial {trial.number} done: val {cfg.metric}={best:.4f}")
        return best

    finished = lambda: len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))
    remaining = a.n_trials - finished()
    clock = a.out / "tuning_clock.json"      # chained jobs share the --tuning-hours cap
    spent = json.loads(clock.read_text())["seconds"] if chained and clock.exists() else 0.0
    if remaining > 0 and a.tuning_hours * 3600 > spent:
        t_tune = time.time()
        try:
            study.optimize(objective, n_trials=remaining, timeout=a.tuning_hours * 3600 - spent,
                           catch=(torch.cuda.OutOfMemoryError,),   # real bugs should crash, not be skipped
                           gc_after_trial=True)
        finally:
            if chained:
                clock.write_text(json.dumps({"seconds": spent + time.time() - t_tune}))
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete:
        raise SystemExit("no completed Optuna trial; nothing to evaluate")
    best_trial = study.best_trial
    best_params = best_trial.user_attrs["params_full"]
    (a.out / "best_params.json").write_text(json.dumps(best_params, indent=2))
    log(f"best trial {best_trial.number}: {best_params} val {cfg.metric}={best_trial.value:.4f}")

    test_index = env_index(tasks[2])

    def test_report(agent, tag):
        """Test metrics; per-env arrays and trajectories go to <tag>_test_paths.npz."""
        use_i = cfg.net == "double"
        m_det, arr_det = evaluate(agent, tasks[2], use_inactor=use_i)
        res = {"deterministic": m_det}
        arrays = {f"deterministic_{k}": v for k, v in arr_det.items()}
        arrays.update(trace_rollout(agent, tasks[2], a.trace_paths, use_inactor=use_i))
        if use_i:
            m_st, arr_st = evaluate(agent, tasks[2], use_inactor=True, inactor_stochastic=True, seed=0)
            res["stochastic_inaction"] = m_st
            arrays.update({f"stochastic_inaction_{k}": v for k, v in arr_st.items()})
        np.savez_compressed(a.out / f"{tag}_test_paths.npz", **arrays, **test_index)
        return res

    # Test-set evaluation of the best tuning trial (selected on validation only).
    # If the job died between saving a checkpoint and Optuna recording the trial, the file can
    # belong to a different trial; retrain the best trial (same seed) in that case.
    def best_trial_block():
        ck_path = a.out / "best_trial.pt"
        ck_best = torch.load(ck_path, map_location=device, weights_only=True) if ck_path.exists() else None
        if ck_best is None or ck_best.get("trial") != best_trial.number:
            log(f"best_trial.pt does not hold trial {best_trial.number}; retraining it")
            agent, _, _, _ = run_training(a, cfg, tasks, best_params, a.tuning_seeds[0], log,
                                          init_actor=init_tune if from_single else None, ckpt=ck("best_trial_retrain"))
            save_ckpt(ck_path, agent, cfg, {"params": best_params, "trial": best_trial.number})
        else:
            agent = Agent(cfg, a.product, device)
            agent.load_state_dict(ck_best["state"])
        return test_report(agent, "best_trial")
    best_trial_test = once("best_trial_test", best_trial_block)

    def baselines_block():
        baselines, base_arrays = {}, {}
        plan = [("no_hedge", None)]
        if a.product == "call":
            plan.append(("model_delta", None))          # BS N(d1) or Heston P1
        if a.product == "call" and a.dynamics == "heston":
            plan.append(("mv_delta", None))             # Heston minimum-variance delta (spot + correlated vol move)
        if a.product == "call" and a.dynamics == "bsm":
            k_ww, ww_scores = tune_ww_band(tasks[1])     # band scale chosen on validation only
            log(f"Whalley-Wilmott band scale chosen on validation: {k_ww:.3g}")
            plan.append(("ww_band", k_ww))
        for kind, param in plan:
            baselines[kind], arr = evaluate_baseline(tasks[2], kind, param)
            base_arrays.update({f"{kind}_{k}": v for k, v in arr.items()})
        if a.product == "call" and a.dynamics == "bsm":
            baselines["ww_band"]["validation_scores"] = {f"{k:.4g}": v for k, v in ww_scores.items()}
        np.savez_compressed(a.out / "baselines_test_paths.npz", **base_arrays, **test_index)
        return baselines
    baselines = once("baselines_test", baselines_block)
    log("test baselines: " + ", ".join(f"{k} cvar_95={v['cvar_95']:.2f}" for k, v in baselines.items()))

    # Final seeds with the best hyper-parameters.
    finals = {}
    for seed in a.final_seeds:
        jpath = a.out / f"final_seed{seed}.json"
        if jpath.exists():
            finals[seed] = json.loads(jpath.read_text()); continue
        log(f"=== final seed {seed}")
        init = None
        src = None
        if from_single:
            init, src = load_single_actor(proto, cfg, f"final_seed{seed}.pt")
        t0 = time.time()
        agent, best, m, logs = run_training(a, cfg, tasks, best_params, seed, log, init_actor=init,
                                            ckpt=ck(f"final_seed{seed}"))
        res = {"seed": seed, "val_metric": best, "val": m, "test": test_report(agent, f"final_seed{seed}"),
               "iters": n_iters(logs), "train_seconds": sum(logs.get("phase_seconds", {}).values()) or time.time() - t0,
               "actor_init_source": src, "eval_log": logs["eval"], "train_log": logs["train"]}
        save_ckpt(a.out / f"final_seed{seed}.pt", agent, cfg, {"params": best_params, "seed": seed})
        jpath.write_text(json.dumps(res, default=str))
        if chained:
            clear(f"final_seed{seed}.pt")
        finals[seed] = res
        log(f"    test cvar_95={res['test']['deterministic']['cvar_95']:.4f} "
            f"mean={res['test']['deterministic']['mean']:.4f} ({res['train_seconds']:.0f}s)")

    keys = finals[next(iter(finals))]["test"]["deterministic"].keys() if finals else []
    agg = {k: {"mean": float(np.mean([f["test"]["deterministic"][k] for f in finals.values()])),
               "std": float(np.std([f["test"]["deterministic"][k] for f in finals.values()]))} for k in keys}
    all_trials = study.get_trials(deepcopy=False)
    summary = {
        "experiment": {"name": name, "dynamics": a.dynamics, "product": a.product, "arch": a.arch, "net": a.net,
                       "reward": cfg.reward, "digital_payout_frac": a.digital_payout_frac,
                       "tuning_seeds": a.tuning_seeds,
                       "market": a.market, "underlying_fee": a.underlying_fee, "option_fee": a.option_fee,
                       "paths": sizes, "split_seeds": SPLIT_SEEDS},
        "protocol": proto, "config": asdict(cfg), "selection_metric": f"validation {cfg.metric}",
        "best_params": best_params, "best_trial": best_trial.number, "best_val_metric": best_trial.value,
        "trials": {s.name: sum(t.state == s for t in all_trials) for s in TrialState},
        "bank_checks": checks,
        "trial_table": [{"trial": t.number, "state": t.state.name, "value": _finite(t.value),
                         "per_seed_values": t.user_attrs.get("per_seed_values"),
                         "params": t.user_attrs.get("params_full", t.params), "iters": t.user_attrs.get("iters"),
                         "seconds": t.duration.total_seconds() if t.duration else None} for t in all_trials],
        "best_trial_test": best_trial_test,
        "baselines_test": baselines,
        "final_seeds_test_deterministic": agg,
        "final_seeds": {s: {k: v for k, v in f.items() if k not in ("eval_log", "train_log")}
                        for s, f in finals.items()},
        "files": {"trials": "trials/trial_NNN.json (params, eval_log, train_log)",
                  "final_seed": "final_seedN.json (eval_log, train_log, test metrics)",
                  "paths": "*_test_paths.npz (per-env arrays + trace_* trajectories), baselines_test_paths.npz"},
        "wall_hours": (time.time() - t_start) / 3600,
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"summary written to {a.out / 'summary.json'}")


if __name__ == "__main__":
    main()
