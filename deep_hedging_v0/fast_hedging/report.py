"""Quick text report of (possibly unfinished) runs. Reads saved files only, trains nothing.

    python -m fast_hedging.report experiments/results/v2_1_5            # status + tables
    python -m fast_hedging.report experiments/results/v2_1_5 --paired   # + per-seed path bootstrap
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as fa

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
NET_ORDER = ("single", "band", "double")
GROUP = ["dynamics", "product", "reward"]


def status(root):
    rows = []
    for run in fa._runs(root):
        states = [json.loads(f.read_text())["state"] for f in sorted((run / "trials").glob("trial_*.json"))]
        best = json.loads((run / "best_params.json").read_text()) if (run / "best_params.json").exists() else {}
        warn = 0
        if (run / "bank_checks.json").exists():
            warn = sum(len(c.get("warnings", [])) for c in json.loads((run / "bank_checks.json").read_text()).values())
        rows.append({"run": run.name, "complete": states.count("complete"), "pruned": states.count("pruned"),
                     "failed": states.count("failed"), "final_seeds": len(list(run.glob("final_seed*.json"))),
                     "summary": (run / "summary.json").exists(), "bank_warnings": warn,
                     **{k: f"{v:.2e}" for k, v in best.items()}})
    return pd.DataFrame(rows)


def test_table(root):
    df = fa.load_summaries(root)
    if df.empty:
        return df
    cols = {"test/deterministic/cvar_95": "cvar95", "test/deterministic/mean": "mean_pnl",
            "test/deterministic/std": "std_pnl", "test/deterministic/mean_cost": "cost",
            "test/deterministic/no_trade_rate": "no_trade", "test/deterministic/mean_trades": "trades",
            "test/stochastic_inaction/cvar_95": "cvar95_stoch_inaction"}
    df = df.rename(columns=cols)
    keep = [c for c in cols.values() if c in df]
    g = df.groupby(GROUP + ["net"])
    out = g[keep].mean().round(3)
    out.insert(0, "seeds", g.size())
    out.insert(2, "cvar95_sd", g["cvar95"].std(ddof=0).round(3))
    return out


def baselines(root):
    rows, seen = [], set()
    for run in fa._runs(root):
        meta = fa._meta(run)
        key = (meta.get("dynamics"), meta.get("product"))
        f = run / "summary.json"
        if key in seen or not f.exists():
            continue
        seen.add(key)
        for kind, m in json.loads(f.read_text()).get("baselines_test", {}).items():
            rows.append({"dynamics": key[0], "product": key[1], "baseline": kind, "cvar95": m["cvar_95"],
                         "mean_pnl": m["mean"], "cost": m.get("mean_cost"), "no_trade": m.get("no_trade_rate"),
                         "param": m.get("param")})
    return pd.DataFrame(rows).round(3)


def seed_level(root, n_boot=10000, metrics=("cvar_95", "mean_cost", "mean")):
    """Method-level comparison: the training seeds of each net are the samples (training noise),
    not the test paths. Bootstrap CI of the difference in means over seeds + Mann-Whitney U."""
    from scipy.stats import mannwhitneyu
    df = fa.load_summaries(root)
    rows = []
    if df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    for keys, g in df.groupby(GROUP):
        nets = [n for n in NET_ORDER if n in set(g["net"])]
        for net_b, net_a in combinations(nets, 2):          # net_a is the more structured one
            for met in metrics:
                col = f"test/deterministic/{met}"
                b = g.loc[g["net"] == net_b, col].to_numpy(float)
                a_ = g.loc[g["net"] == net_a, col].to_numpy(float)
                if len(a_) < 2 or len(b) < 2:
                    continue
                sign = -1.0 if met == "mean" else 1.0       # every metric reported as 'lower is better'
                a_, b = sign * a_, sign * b
                boots = rng.choice(a_, (n_boot, a_.size)).mean(1) - rng.choice(b, (n_boot, b.size)).mean(1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append({**dict(zip(GROUP, keys)), "comparison": f"{net_a}-{net_b}",
                             "metric": met if met != "mean" else "mean_loss", "mean_b": b.mean(), "mean_a": a_.mean(),
                             "diff": a_.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
                             "P(a<b)": float((a_[:, None] < b[None, :]).mean()),
                             "mwu_p": float(mannwhitneyu(a_, b, alternative="two-sided").pvalue),
                             "n": f"{a_.size}v{b.size}"})
    out = pd.DataFrame(rows)
    return out.round(3) if not out.empty else out


def paired(root, n_boot):
    rows = []
    root = Path(root)
    groups = {}
    for r in fa._runs(root):
        m = fa._meta(r)
        groups.setdefault((m["dynamics"], m["product"], m["reward"]), set()).add(m["net"])
    seed_of = lambda p: int(p.stem.split("seed")[1].split("_")[0])
    for (dyn, prod, rew), nets in sorted(groups.items()):
        nets = [n for n in NET_ORDER if n in nets]
        for net_b, net_a in combinations(nets, 2):
            da, db = fa.run_dir(root, dyn, prod, net_a, rew), fa.run_dir(root, dyn, prod, net_b, rew)
            seeds = sorted({seed_of(p) for p in da.glob("final_seed*_test_paths.npz")}
                           & {seed_of(p) for p in db.glob("final_seed*_test_paths.npz")})
            for seed in seeds:
                c = fa.compare_nets(root, dyn, prod, seed=seed, n_boot=n_boot, net_a=net_a, net_b=net_b, reward=rew)
                r = c.loc["cvar_95"]
                rows.append({"dynamics": dyn, "product": prod, "reward": rew, "comparison": f"{net_a}-{net_b}",
                             "seed": seed, "cvar95_diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                             "cost_diff": c.loc["mean_cost", "diff"]})
    out = pd.DataFrame(rows)
    return out.round(3) if not out.empty else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--paired", action="store_true", help="per-seed paired bootstrap on test paths")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    print("=== STATUS\n" + status(a.root).to_string(index=False))
    t = test_table(a.root)
    print("\n=== TEST (mean over finished seeds; loss convention: lower cvar95 is better)\n"
          + (t.to_string() if not t.empty else "no finished seeds yet"))
    b = baselines(a.root)
    if not b.empty:
        print("\n=== BASELINES (same test paths and costs; ww_band scale tuned on validation)\n" + b.to_string(index=False))
    sl = seed_level(a.root)
    if not sl.empty:
        print("\n=== SEED-LEVEL (samples = training seeds; lower is better for every metric; "
              "diff = a - b; P(a<b) = share of seed pairs where a is better)\n" + sl.to_string(index=False))
    if a.paired:
        print(f"\n(computing paired bootstrap, {a.n_boot} resamples per seed ...)", flush=True)
        pr = paired(a.root, a.n_boot)
        print("\n=== PAIRED per seed (CI covers TEST-PATH noise only, for one pair of trained models)\n"
              + (pr.to_string(index=False) if not pr.empty else "no seed finished for two nets yet"))


if __name__ == "__main__":
    main()
