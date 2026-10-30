"""One run = one experiment x one net type.

    single : Optuna over the actor learning rate (training seed 0, early stopping on validation
             CVaR95), then the best rate is retrained on the final seeds and scored on the test year.
    double : starts every training from the matching single-net checkpoint (same seed), adds the
             inaction gate, Optuna over (gate_lr, actor_lr) for the joint phase, then final seeds.
             Needs the single run of the same experiment to have finished.

Selection only ever uses the validation split; the test year (2021) is only reported.
Re-submitting the same command resumes: finished trials and finished seeds are kept.

Example:
    python -u deep_hedging_v0/experiments/fast/run.py --experiment bsm_call_mlp --mode single
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]          # deep_hedging_v0/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
from optuna.trial import TrialState
import torch

from experiments.fast.config import (BankConfig, MarketConfig, PPOConfig, StopConfig, default_bank_config,
                                     parse_experiment, smoke_overrides)
from experiments.fast.path_bank import SPLITS, Bank, get_bank
from experiments.fast.train import save_result, test_metrics, train

LR_RANGES = {"mlp": (1e-6, 3e-3), "rnn": (1e-7, 1e-3)}
GATE_LR_RANGE = (1e-5, 1e-1)
SUMMARY_KEYS = ("cvar_95", "cvar_99", "var_95", "mean", "std", "mean_abs_tracking_error",
                "mean_transaction_cost", "trade_frequency", "hold_rate")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_banks(bank_dir, dynamics, product, market, bank_cfg, device):
    return {s: Bank(get_bank(bank_dir, dynamics, product, s, market, bank_cfg, device), device) for s in SPLITS}


def aggregate(rows: list[dict], split="test") -> dict:
    out = {}
    for key in SUMMARY_KEYS:
        vals = np.array([r[split][key] for r in rows if split in r], dtype=np.float64)
        if vals.size:
            out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                        "n": int(vals.size)}
    return out


def paired_comparison(single_rows: dict, double_rows: dict) -> dict:
    """Same seed -> same pretrained actor, so single vs double is a paired comparison."""
    seeds = sorted(set(single_rows) & set(double_rows))
    if not seeds:
        return {}
    s = np.array([single_rows[k]["test"]["cvar_95"] for k in seeds])
    d = np.array([double_rows[k]["test"]["cvar_95"] for k in seeds])
    diff = s - d                                     # > 0 means the double net has the smaller tail loss
    out = {"seeds": seeds, "single_test_cvar_95": s.tolist(), "double_test_cvar_95": d.tolist(),
           "improvement_mean": float(diff.mean()),
           "improvement_std": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
           "double_wins": int((diff > 0).sum()), "n": int(diff.size)}
    if diff.size >= 5:
        try:
            from scipy.stats import wilcoxon
            out["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
        except Exception as exc:  # all-zero differences, missing scipy, ...
            out["wilcoxon_p"] = None
            out["wilcoxon_error"] = str(exc)
    return out


class WandbLogger:
    def __init__(self, enabled, project, group, config):
        self.enabled, self.project, self.group, self.config = enabled, project, group, config

    def log_run(self, name, result, metrics=None, params=None):
        if not self.enabled:
            return
        import wandb
        run = wandb.init(project=self.project, group=self.group, name=name, reinit=True,
                         config={**self.config, **(params or {})})
        for row in result.history:
            wandb.log(row, step=row["iter"])
        if metrics:
            wandb.log({f"final/{split}/{k}": v for split, m in metrics.items() for k, v in m.items()})
        run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="<bsm|heston>_<call|doc|con>_rnn")
    p.add_argument("--mode", required=True, choices=("single", "double"))
    p.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "fast" / "results")
    p.add_argument("--bank-dir", type=Path, default=ROOT.parent / ".path_banks")
    p.add_argument("--n-trials", type=int, default=None, help="default: 20 single, 30 double")
    p.add_argument("--tuning-hours", type=float, default=36.0, help="stop launching new trials after this")
    p.add_argument("--tuning-seed", type=int, default=0)
    p.add_argument("--final-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--no-trade-features", action="store_true", help="drop the trading-history features (env.trade_features)")
    p.add_argument("--smoke", action="store_true", help="tiny banks and runs, to check the pipeline")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "deephedging_fast_1_5"))
    args = p.parse_args()

    dynamics, product, arch = parse_experiment(args.experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    market = MarketConfig()
    bank_cfg: BankConfig = default_bank_config(dynamics, product, smoke=args.smoke)
    ppo, stop = PPOConfig(trade_features=not args.no_trade_features), StopConfig()
    if args.smoke:
        smoke_overrides(ppo, stop)
        args.results_dir = args.results_dir / "smoke"
    n_trials = args.n_trials or (2 if args.smoke else (20 if args.mode == "single" else 30))

    out = args.results_dir / args.experiment / args.mode
    (out / "trials").mkdir(parents=True, exist_ok=True)
    (out / "final").mkdir(parents=True, exist_ok=True)
    run_config = {"experiment": args.experiment, "mode": args.mode, "device": str(device),
                  "market": asdict(market), "bank": asdict(bank_cfg), "ppo": asdict(ppo), "stop": asdict(stop),
                  "n_trials": n_trials, "tuning_seed": args.tuning_seed, "final_seeds": args.final_seeds}
    write_json(out / "config.json", run_config)
    log(f"{args.experiment}/{args.mode} on {device}; fees: underlying {market.underlying_fee:.0%}, "
        f"options {market.option_fee:.0%}; results -> {out}")

    banks = load_banks(args.bank_dir, dynamics, product, market, bank_cfg, device)
    log(f"banks: train {banks['train'].num_paths} x {banks['train'].options_per_path} options, "
        f"val {banks['val'].num_paths} x {banks['val'].options_per_path}, "
        f"test {banks['test'].num_paths} x {banks['test'].options_per_path}")
    wb = WandbLogger(args.wandb, args.wandb_project, f"{args.experiment}_{args.mode}", run_config)

    single_dir = args.results_dir / args.experiment / "single"
    tuning_init = None
    if args.mode == "double":
        summary_path = single_dir / "summary.json"
        if not summary_path.exists():
            sys.exit(f"Missing {summary_path}: run the single-net job for {args.experiment} first.")
        single_summary = read_json(summary_path)
        if single_summary["ppo"].get("trade_features") != ppo.trade_features:
            sys.exit("--no-trade-features must match the single-net run")
        single_lr = single_summary["best_params"]["actor_lr"]
        tuning_init = torch.load(single_dir / single_summary["best_trial_checkpoint"], weights_only=False)["actor_state"]
        actor_lr_range = (single_lr * 1e-3, single_lr * 2.0)
        log(f"double net warm-starts from single best trial (actor_lr={single_lr:.3g})")

    common = dict(arch=arch, product=product, banks=banks, market=market, ppo=ppo, stop=stop, device=device)

    # ------------------------------------------------------------------ tuning
    study = optuna.create_study(
        study_name=f"{args.experiment}_{args.mode}", storage=f"sqlite:///{out / 'study.db'}",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.tuning_seed, n_startup_trials=6),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=stop.eval_every * stop.min_evals),
    )
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)       # left over from a killed job

    def objective(trial):
        if args.mode == "single":
            params = {"actor_lr": trial.suggest_float("actor_lr", *LR_RANGES[arch], log=True)}
            res = train("single", seed=args.tuning_seed, actor_lr=params["actor_lr"], trial=trial,
                        log=log, tag=f"[trial {trial.number}]", **common)
        else:
            params = {"gate_lr": trial.suggest_float("gate_lr", *GATE_LR_RANGE, log=True),
                      "actor_lr": trial.suggest_float("actor_lr", *actor_lr_range, log=True)}
            res = train("double", seed=args.tuning_seed, actor_lr=params["actor_lr"], gate_lr=params["gate_lr"],
                        init_actor_state=tuning_init, trial=trial, log=log, tag=f"[trial {trial.number}]", **common)
        ckpt = out / "trials" / f"trial_{trial.number:03d}.pt"
        save_result(ckpt, res, {"params": params, "seed": args.tuning_seed})
        for key, value in (("checkpoint", str(ckpt.relative_to(out))), ("best_iter", res.best_iter),
                           ("iters", res.iters), ("stop_reason", res.stop_reason), ("seconds", res.seconds)):
            trial.set_user_attr(key, value)
        wb.log_run(f"trial_{trial.number:03d}", res, params=params)
        log(f"[trial {trial.number}] {params} -> best val CVaR95 {res.best_val:.4f} at it {res.best_iter} "
            f"({res.stop_reason}, {res.iters} its, {res.seconds / 60:.1f} min)")
        return res.best_val if math.isfinite(res.best_val) else float("nan")

    finished = [t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]
    remaining = n_trials - len(finished)
    if remaining > 0:
        log(f"tuning: {len(finished)} trials done, running up to {remaining} more")
        study.optimize(objective, n_trials=remaining, timeout=args.tuning_hours * 3600, gc_after_trial=True)
    best = study.best_trial
    log(f"best trial {best.number}: {best.params} val CVaR95 {best.value:.4f}")

    # ------------------------------------------------------------- final seeds
    rows = {}
    for seed in args.final_seeds:
        row_path = out / "final" / f"seed_{seed}.json"
        if row_path.exists():
            rows[seed] = read_json(row_path)
            continue
        if args.mode == "single":
            res = train("single", seed=seed, actor_lr=best.params["actor_lr"], log=log, tag=f"[seed {seed}]", **common)
        else:
            init_path = single_dir / "final" / f"seed_{seed}.pt"
            if not init_path.exists():
                sys.exit(f"Missing {init_path}: the single-net job must finish seed {seed} first.")
            init = torch.load(init_path, weights_only=False)["actor_state"]
            res = train("double", seed=seed, actor_lr=best.params["actor_lr"], gate_lr=best.params["gate_lr"],
                        init_actor_state=init, log=log, tag=f"[seed {seed}]", **common)
        metrics = test_metrics(args.mode, arch, product, banks, market, ppo, res, device)
        save_result(out / "final" / f"seed_{seed}.pt", res, {"params": best.params, "seed": seed})
        rows[seed] = {"seed": seed, "params": best.params, "best_val": res.best_val, "best_iter": res.best_iter,
                      "iters": res.iters, "stop_reason": res.stop_reason, "seconds": res.seconds, **metrics}
        write_json(row_path, rows[seed])
        wb.log_run(f"final_seed_{seed}", res, metrics=metrics, params=best.params)
        log(f"[seed {seed}] test CVaR95 {metrics['test']['cvar_95']:.4f} "
            f"cost {metrics['test']['mean_transaction_cost']:.3f} hold {metrics['test']['hold_rate']:.3f}")

    summary = {
        "experiment": args.experiment, "mode": args.mode, "best_params": best.params, "best_val_cvar_95": best.value,
        "best_trial": best.number, "best_trial_checkpoint": best.user_attrs["checkpoint"],
        "trial_states": {s.name: sum(t.state == s for t in study.trials) for s in TrialState},
        "ppo": asdict(ppo), "fees": {"underlying": market.underlying_fee, "options": market.option_fee},
        "test": aggregate(list(rows.values()), "test"), "val": aggregate(list(rows.values()), "val"),
        "seeds": {str(k): v for k, v in rows.items()},
    }
    if args.mode == "double":
        single_rows = {int(f.stem.split("_")[1]): read_json(f)
                       for f in (single_dir / "final").glob("seed_*.json") if "." not in f.stem}
        summary["test_sampled_gate"] = aggregate(list(rows.values()), "test_sampled_gate")
        summary["comparison_vs_single"] = paired_comparison(single_rows, rows)
        log(f"single vs double (test CVaR95, paired): {summary['comparison_vs_single']}")
    write_json(out / "summary.json", summary)
    log(f"done -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
