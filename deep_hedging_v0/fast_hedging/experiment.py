"""One experiment = one (dynamics, product, arch, net) cell.

  1. load / build the cached market banks (train, validation, test)
  2. Optuna study over the learning rate(s); each trial trains with early
     stopping on the VALIDATION split (fresh paths of the training years)
  3. retrain the best learning rate(s) on the final seeds and evaluate on the
     TEST split (held-out calibration year), which is never used for selection
  4. write summary.json (+ per-seed json, checkpoints, optuna journal)

The Optuna journal makes the job resumable: resubmitting the same command
continues the study and skips final seeds that already finished.

Example:
  python -m fast_hedging.experiment --dynamics bsm --product call --arch rnn --net single \
      --out results/bsm_call_rnn_single
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .market import MarketSpec, get_bank, default_device
from .ppo import (Agent, Task, TrainConfig, TrialPruned, env_index, evaluate, evaluate_baseline, trace_rollout,
                  train_phase)

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}       # identical for every experiment
DEFAULT_PATHS = {"bsm": {"train": 2000, "val": 500, "test": 2000},
                 "heston": {"train": 1000, "val": 250, "test": 1000}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dynamics", choices=("bsm", "heston"), required=True)
    p.add_argument("--product", choices=("call", "doc", "con"), required=True)
    p.add_argument("--arch", choices=("mlp", "rnn"), default="rnn")
    p.add_argument("--rnn-mode", choices=("window", "recurrent"), default="window",
                   help="window: LSTM over the last --window observations (history_len, default 8); "
                        "recurrent: one LSTM pass over the whole path (about window-times less LSTM work)")
    p.add_argument("--net", choices=("single", "double"), default="single")
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
    p.add_argument("--tuning-seed", type=int, default=0)
    p.add_argument("--final-seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    # double-net protocol
    p.add_argument("--actor-init", choices=("train", "from-single"), default="train",
                   help="double net stage 1: train its own actor (old protocol) or load the single-net result")
    p.add_argument("--single-dir", type=Path, help="results dir of the matching single-net run (from-single)")
    # training knobs (see ppo.TrainConfig for defaults)
    for name in ("paths-per-iter", "ppo-epochs", "minibatch", "eval-every", "patience", "max-iters",
                 "min-iters", "window", "hidden", "actor-update-every"):
        p.add_argument(f"--{name}", type=int)
    p.add_argument("--min-rel-improvement", type=float)
    p.add_argument("--no-agent-state", action="store_true",
                   help="old inputs only: hide holdings, tracking error, costs, time since trade, proposed trade")
    p.add_argument("--inactor-init-hold-prob", type=float)
    p.add_argument("--reward-norm", choices=("premium", "spot", "none"))
    p.add_argument("--no-obs-norm", action="store_true"); p.add_argument("--no-adv-norm", action="store_true")
    p.add_argument("--credit-held-actions", action="store_true",
                   help="old behaviour: train the actor on proposals that the inactor overrode")
    p.add_argument("--trace-paths", type=int, default=100,
                   help="test paths (x assets x strikes) whose full step-by-step trajectories are saved")
    p.add_argument("--device", default=None)
    p.add_argument("--bank-only", action="store_true", help="only build the market banks (pre-generation)")
    return p.parse_args(argv)


def make_config(a) -> TrainConfig:
    cfg = TrainConfig(arch=a.arch, rnn_mode=a.rnn_mode, net=a.net, underlying_fee=a.underlying_fee,
                      option_fee=a.option_fee, agent_state=not a.no_agent_state, normalize_obs=not a.no_obs_norm,
                      normalize_adv=not a.no_adv_norm, mask_held_actor_loss=not a.credit_held_actions)
    over = {k.replace("-", "_"): getattr(a, k.replace("-", "_")) for k in
            ("paths-per-iter", "ppo-epochs", "minibatch", "eval-every", "patience", "max-iters", "min-iters",
             "window", "hidden", "actor-update-every", "min-rel-improvement", "inactor-init-hold-prob",
             "reward-norm")}
    return replace(cfg, **{k: v for k, v in over.items() if v is not None})


def load_tasks(a, cfg, device):
    spec = MarketSpec(dynamics=a.dynamics, product=a.product, market=a.market)
    sizes = dict(DEFAULT_PATHS[a.dynamics])
    for split in sizes:
        if getattr(a, f"{split}_paths"):
            sizes[split] = getattr(a, f"{split}_paths")
    banks = {s: get_bank(spec, s, sizes[s], SPLIT_SEEDS[s], device=device) for s in ("train", "val", "test")}
    if a.bank_only:
        return None, spec, sizes
    train = Task(banks["train"], a.product, spec.r, cfg, device)
    val = Task(banks["val"], a.product, spec.r, cfg, device, norm=train.norm)
    test = Task(banks["test"], a.product, spec.r, cfg, device, norm=train.norm)
    return (train, val, test), spec, sizes


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
    keys = ("arch", "rnn_mode", "agent_state", "window", "normalize_obs", "underlying_fee", "option_fee",
            "reward_norm")
    return all(saved_cfg.get(k) == getattr(cfg, k) for k in keys)


def run_training(a, cfg, tasks, params, seed, log, report=None, init_actor=None, logs=None):
    """Train one model with fixed hyper-parameters.
    Returns (agent, val_metric, val_metrics, logs) with logs = {"eval": [...], "train": [...]}.
    Pass `logs` to keep the entries even if training is pruned or crashes."""
    train, val, _ = tasks
    torch.manual_seed(seed); np.random.seed(seed)
    agent = Agent(cfg, a.product, train.device)
    logs = logs if logs is not None else {"eval": [], "train": []}
    if cfg.net == "single":
        best, m, _ = train_phase(agent, train, val, phase="actor", lr_actor=params["actor_lr"], seed=seed,
                                 report=report, log=log, logs=logs)
        return agent, best, m, logs
    if init_actor is not None:
        agent.actor.load_state_dict(init_actor)
    else:
        train_phase(agent, train, val, phase="actor", lr_actor=params["actor_lr"], seed=seed,
                    report=report, log=log, logs=logs)
    best, m, _ = train_phase(agent, train, val, phase="joint", lr_actor=params["joint_lr"],
                             lr_inactor=params["inactor_lr"], seed=seed + 10_000, report=report,
                             step_offset=100_000, log=log, iter_offset=n_iters(logs), logs=logs)
    return agent, best, m, logs


def n_iters(logs):
    return logs["train"][-1]["global_iter"] if logs["train"] else 0


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
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not _compatible(ck["config"], cfg):
        raise RuntimeError(f"single-net checkpoint {path} was trained with an incompatible config")
    return ck["state"]["actor"], str(path)


def main(argv=None):
    a = parse_args(argv)
    device = a.device or default_device()
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True          # LSTM shapes are fixed: autotune once
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)
    log = Logger(a.out / "log.txt")
    cfg = make_config(a)
    t_start = time.time()
    log(f"=== {a.dynamics} {a.product} {a.arch} {a.net} | fees underlying={a.underlying_fee} "
        f"option={a.option_fee} | device={device}")
    tasks, spec, sizes = load_tasks(a, cfg, device)
    log(f"banks ready in {time.time() - t_start:.0f}s: {sizes}")
    if a.bank_only:
        return
    proto = resolve_protocol(a, cfg, a.out, log)
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
    pruner = optuna.pruners.NopPruner() if a.no_pruning else optuna.pruners.MedianPruner(
        n_startup_trials=6, n_warmup_steps=8, interval_steps=1)
    study = optuna.create_study(study_name="study", storage=storage, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=a.tuning_seed, n_startup_trials=8),
                                pruner=pruner)
    for tr in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study._storage.set_trial_state_values(tr._trial_id, TrialState.FAIL)   # left over from a killed job

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
            if trial.should_prune():
                raise TrialPruned()
        logs, state, best, m, t0 = {"eval": [], "train": []}, "failed", None, None, time.time()
        try:
            agent, best, m, _ = run_training(a, cfg, tasks, params, a.tuning_seed, log, report,
                                             init_actor=init_tune if from_single else None, logs=logs)
            state = "complete"
        except TrialPruned:
            state = "pruned"
            log(f"--- trial {trial.number} pruned")
            raise optuna.TrialPruned()
        finally:   # full per-trial record, whatever happened
            trials_dir = a.out / "trials"
            trials_dir.mkdir(exist_ok=True)
            (trials_dir / f"trial_{trial.number:03d}.json").write_text(json.dumps(
                {"trial": trial.number, "state": state, "params": params, "value": _finite(best),
                 "val_metrics": m, "iters": n_iters(logs), "seconds": time.time() - t0,
                 "eval_log": logs["eval"], "train_log": logs["train"]}, default=str))
        trial.set_user_attr("val_metrics", m)
        trial.set_user_attr("params_full", params)
        trial.set_user_attr("iters", n_iters(logs))
        done = [t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))]
        if not done or best < min(t.value for t in done):
            save_ckpt(a.out / "best_trial.pt", agent, cfg, {"params": params, "trial": trial.number})
        log(f"--- trial {trial.number} done: val {cfg.metric}={best:.4f}")
        return best

    finished = lambda: len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)))
    remaining = a.n_trials - finished()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=a.tuning_hours * 3600, catch=(torch.cuda.OutOfMemoryError,),   # real bugs should crash, not be skipped
                       gc_after_trial=True)
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
    ck_path = a.out / "best_trial.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False) if ck_path.exists() else None
    if ck is None or ck.get("trial") != best_trial.number:
        log(f"best_trial.pt does not hold trial {best_trial.number}; retraining it")
        agent, _, _, _ = run_training(a, cfg, tasks, best_params, a.tuning_seed, log,
                                      init_actor=init_tune if from_single else None)
        save_ckpt(ck_path, agent, cfg, {"params": best_params, "trial": best_trial.number})
    else:
        agent = Agent(cfg, a.product, device)
        agent.load_state_dict(ck["state"])
    best_trial_test = test_report(agent, "best_trial")

    baselines, base_arrays = {}, {}
    kinds = ["no_hedge"] + (["bs_delta"] if a.dynamics == "bsm" and a.product == "call" else [])
    for kind in kinds:
        baselines[kind], arr = evaluate_baseline(tasks[2], kind)
        base_arrays.update({f"{kind}_{k}": v for k, v in arr.items()})
    np.savez_compressed(a.out / "baselines_test_paths.npz", **base_arrays, **test_index)
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
        agent, best, m, logs = run_training(a, cfg, tasks, best_params, seed, log, init_actor=init)
        res = {"seed": seed, "val_metric": best, "val": m, "test": test_report(agent, f"final_seed{seed}"),
               "iters": n_iters(logs), "train_seconds": time.time() - t0,
               "actor_init_source": src, "eval_log": logs["eval"], "train_log": logs["train"]}
        save_ckpt(a.out / f"final_seed{seed}.pt", agent, cfg, {"params": best_params, "seed": seed})
        jpath.write_text(json.dumps(res, default=str))
        finals[seed] = res
        log(f"    test cvar_95={res['test']['deterministic']['cvar_95']:.4f} "
            f"mean={res['test']['deterministic']['mean']:.4f} ({res['train_seconds']:.0f}s)")

    keys = finals[next(iter(finals))]["test"]["deterministic"].keys() if finals else []
    agg = {k: {"mean": float(np.mean([f["test"]["deterministic"][k] for f in finals.values()])),
               "std": float(np.std([f["test"]["deterministic"][k] for f in finals.values()]))} for k in keys}
    all_trials = study.get_trials(deepcopy=False)
    summary = {
        "experiment": {"dynamics": a.dynamics, "product": a.product, "arch": a.arch, "net": a.net,
                       "market": a.market, "underlying_fee": a.underlying_fee, "option_fee": a.option_fee,
                       "paths": sizes, "split_seeds": SPLIT_SEEDS},
        "protocol": proto, "config": asdict(cfg), "selection_metric": f"validation {cfg.metric}",
        "best_params": best_params, "best_trial": best_trial.number, "best_val_metric": best_trial.value,
        "trials": {s.name: sum(t.state == s for t in all_trials) for s in TrialState},
        "trial_table": [{"trial": t.number, "state": t.state.name, "value": _finite(t.value),
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
