"""Load fast_hedging results for analysis (pandas / numpy).

Typical use, from deep_hedging_v0/ (on your laptop after rsyncing the results back):

    from fast_hedging import analysis as fa
    root = "experiments/results/liquid_1_5"
    runs   = fa.load_summaries(root)              # one row per (run, seed): test metrics
    trials = fa.load_trials(root)                 # one row per Optuna trial
    curves = fa.load_curves(root, kind="eval")    # validation curves of the final seeds
    train  = fa.load_curves(root, kind="train")   # per-iteration PPO diagnostics
    p = fa.load_paths(f"{root}/bsm_call_rnn_double", seed=1)    # per-path arrays + trajectories
    fa.compare_nets(root, "bsm", "call", seed=1)  # paired single-vs-double test with bootstrap CI

All runs of the same (dynamics, product) use the same test paths in the same order,
so per-path arrays of the single and double net are directly paired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN_FIELDS = ("dynamics", "product", "arch", "net")


def _runs(root):
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")
                  and (p / "protocol.json").exists())


def _meta(run_dir):
    parts = run_dir.name.split("_")
    if len(parts) >= 4:
        return dict(zip(RUN_FIELDS, parts[:4]), run=run_dir.name)
    return {"run": run_dir.name}


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        elif not isinstance(v, list):
            out[f"{prefix}{k}"] = v
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def load_summaries(root):
    """One row per (run, final seed) with test metrics (test/deterministic/..., etc.)."""
    rows = []
    for run in _runs(root):
        for f in sorted(run.glob("final_seed*.json")):
            res = json.loads(f.read_text())
            rows.append({**_meta(run), "seed": res["seed"], "iters": res.get("iters"),
                         "train_seconds": res.get("train_seconds"), "val_metric": res.get("val_metric"),
                         **_flat(res.get("test"), "test/"), **_flat(res.get("val"), "val/")})
    return pd.DataFrame(rows)


def load_run_summaries(root):
    """One row per run: best learning rates, trial counts, seed mean/std, baselines."""
    rows = []
    for run in _runs(root):
        f = run / "summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        rows.append({**_meta(run), **_flat(s.get("best_params"), "best/"), "best_val_metric": s.get("best_val_metric"),
                     **_flat(s.get("trials"), "trials/"), **_flat(s.get("final_seeds_test_deterministic"), "test/"),
                     **_flat(s.get("baselines_test"), "baseline/"), "wall_hours": s.get("wall_hours")})
    return pd.DataFrame(rows)


def load_trials(root):
    """One row per Optuna trial (complete, pruned and failed)."""
    rows = []
    for run in _runs(root):
        for f in sorted((run / "trials").glob("trial_*.json")):
            t = json.loads(f.read_text())
            rows.append({**_meta(run), "trial": t["trial"], "state": t["state"], "value": t["value"],
                         "iters": t["iters"], "seconds": t["seconds"], **_flat(t["params"], "param/"),
                         **_flat(t.get("val_metrics"), "val/")})
    return pd.DataFrame(rows)


def load_curves(root, kind="eval", source="final"):
    """Learning curves. kind: 'eval' (validation checks) or 'train' (every iteration).
    source: 'final' (final seeds) or 'trials' (Optuna trials)."""
    key = f"{kind}_log"
    frames = []
    for run in _runs(root):
        files = sorted(run.glob("final_seed*.json")) if source == "final" else sorted((run / "trials").glob("trial_*.json"))
        for f in files:
            d = json.loads(f.read_text())
            df = pd.DataFrame(d.get(key, []))
            if df.empty:
                continue
            df = df.assign(**_meta(run))
            df["seed" if source == "final" else "trial"] = d.get("seed", d.get("trial"))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Per-path arrays
# --------------------------------------------------------------------------
def load_paths(run_dir, seed=None, tag=None):
    """Per-env test arrays and trajectories of one model as a dict of numpy arrays.

    Keys: env_path/env_asset/env_strike/env_K, deterministic_{pnl,abs_err,cost,hold_rate,n_trades},
    stochastic_inaction_* (double net), trace_* (first --trace-paths paths, shape (B, N[+1], ...)).
    tag defaults to final_seed<seed>; use tag='best_trial' for the best tuning trial."""
    tag = tag or f"final_seed{seed}"
    with np.load(Path(run_dir) / f"{tag}_test_paths.npz") as d:
        return {k: d[k] for k in d.files}


def load_baselines(run_dir):
    with np.load(Path(run_dir) / "baselines_test_paths.npz") as d:
        return {k: d[k] for k in d.files}


def paths_frame(run_dir, seed=None, tag=None, mode="deterministic"):
    """Per-env totals as a DataFrame (one row per test env)."""
    p = load_paths(run_dir, seed, tag)
    cols = {k: p[k] for k in ("env_path", "env_asset", "env_strike", "env_K")}
    cols.update({k[len(mode) + 1:]: v for k, v in p.items() if k.startswith(mode + "_")})
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def cvar_loss(pnl, level=0.95):
    """Expected shortfall of the loss (-pnl), same convention as training/selection."""
    losses = -np.asarray(pnl, np.float64)
    var = np.percentile(losses, level * 100)
    return float(losses[losses >= var].mean())


def paired_bootstrap(pnl_a, pnl_b, env_path, stat=cvar_loss, n_boot=5000, seed=0):
    """CI for stat(a) - stat(b) on paired test envs, resampling whole simulated paths
    (envs on the same path, i.e. all strikes, are correlated and are kept together)."""
    pnl_a, pnl_b, env_path = map(np.asarray, (pnl_a, pnl_b, env_path))
    if pnl_a.shape != pnl_b.shape:
        raise ValueError("arrays are not paired (different shapes)")
    uniq, inv = np.unique(env_path, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        diffs[i] = stat(pnl_a[idx]) - stat(pnl_b[idx])
    point = stat(pnl_a) - stat(pnl_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": float(
        min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def compare_nets(root, dynamics, product, seed=1, arch="rnn", mode="deterministic", n_boot=5000):
    """Paired comparison of single vs double net on identical test paths.
    diff < 0 means the double net has the lower (better) CVaR95 / cost / error."""
    root = Path(root)
    s = load_paths(root / f"{dynamics}_{product}_{arch}_single", seed)
    d = load_paths(root / f"{dynamics}_{product}_{arch}_double", seed)
    if not np.array_equal(s["env_path"], d["env_path"]):
        raise ValueError("single and double runs were evaluated on different test envs")
    key = lambda p, k: p[f"deterministic_{k}"] if f"{mode}_{k}" not in p else p[f"{mode}_{k}"]
    out = {"cvar_95": paired_bootstrap(key(d, "pnl"), key(s, "pnl"), s["env_path"], n_boot=n_boot)}
    for stat in ("cost", "abs_err"):
        out[f"mean_{stat}"] = paired_bootstrap(key(d, stat), key(s, stat), s["env_path"],
                                               stat=lambda x: float(np.mean(x)), n_boot=n_boot)
    out["mean_pnl"] = paired_bootstrap(-key(d, "pnl"), -key(s, "pnl"), s["env_path"],
                                       stat=lambda x: float(np.mean(x)), n_boot=n_boot)   # mean loss
    return pd.DataFrame(out).T
