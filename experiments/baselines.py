"""Model-based reference strategies for one cell of the paper's tables (the "model band" column).

Builds the validation and test banks, computes the model targets (hedging_engine.model_band), chooses the band scale k
on the validation bank only and reports the test bank. For the BSM call it also evaluates the analytic Whalley-Wilmott
band as a cross-check.

usage (from the repository root):
  python -m experiments.baselines --dynamics bsm --product doc --fees 01 1 [--device cuda] [--val-paths N --test-paths N]
  python -m experiments.baselines --dynamics heston --product doc --fees 1 --market sp500x --out experiments/results/main_years_baselines
Writes <out>/<dynamics>_<product>_fee<fee>.json (default out: experiments/results/main_baselines).
"""
import argparse
import json
from pathlib import Path

import torch

from hedging_engine.market import MarketSpec, get_bank
from hedging_engine.model_band import BUMP, compute_targets, evaluate_target_band, per_asset_metrics
from hedging_engine.ppo import WW_GRID, Task, TrainConfig, evaluate_baseline

from .protocol import BASELINE_PATHS, FEES, SPLIT_SEEDS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamics", choices=("bsm", "heston"), required=True)
    ap.add_argument("--product", choices=("call", "doc", "con"), required=True)
    ap.add_argument("--fees", nargs="+", default=["01", "1"], choices=list(FEES))
    ap.add_argument("--val-paths", type=int, default=BASELINE_PATHS["val"]); ap.add_argument("--test-paths", type=int, default=BASELINE_PATHS["test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bump", type=float, default=BUMP, help="relative spot bump of the finite differences")
    ap.add_argument("--mc-paths", type=int, help="inner paths of the Heston barrier Monte Carlo (default: the bank's 2,000)")
    ap.add_argument("--out", type=Path, default=Path("experiments/results/main_baselines"))
    ap.add_argument("--market", default="sp500", help="calibration set (sp500x: years 2017-2023, test = 2021-2023)")
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)
    spec = MarketSpec(dynamics=a.dynamics, product=a.product, market=a.market)
    sizes = {"val": a.val_paths, "test": a.test_paths}
    banks = {s: get_bank(spec, s, sizes[s], SPLIT_SEEDS[s], device=a.device) for s in ("val", "test")}
    greeks = {s: compute_targets(spec, s, banks[s], a.device, log, a.bump, a.mc_paths, seed=SPLIT_SEEDS[s]) for s in ("val", "test")}
    a.out.mkdir(parents=True, exist_ok=True)
    for fee in a.fees:
        uf, of = FEES[fee]
        cfg = TrainConfig(underlying_fee=uf, option_fee=of, normalize_obs=False, reward_norm="none", agent_state=False)
        val = Task(banks["val"], a.product, spec.r, cfg, a.device)
        test = Task(banks["test"], a.product, spec.r, cfg, a.device)
        scores = {k: evaluate_target_band(val, *greeks["val"], k)["mean_abs_tracking_error"] for k in WW_GRID}
        k_star = min(scores, key=scores.get)
        log(f"[{a.dynamics} {a.product} fee {fee}] band scale on validation: k={k_star:.3g} (tracking {scores[k_star]:.3f})")
        res = {"dynamics": a.dynamics, "product": a.product, "fee": fee, "fees": [uf, of], "paths": sizes, "bump": a.bump, "mc_paths": a.mc_paths,
               "band_scale": k_star, "validation_scores": {f"{k:.4g}": v for k, v in scores.items()},
               "validation": {"model_target": evaluate_target_band(val, *greeks["val"], None)["mean_abs_tracking_error"],
                              "model_band": scores[k_star]},
               "test": {"no_hedge": evaluate_baseline(test, "no_hedge")[0],
                        "model_target": evaluate_target_band(test, *greeks["test"], None),
                        "model_band": evaluate_target_band(test, *greeks["test"], k_star)}}
        if a.dynamics == "bsm" and a.product == "call":          # cross-check against the analytic Whalley-Wilmott band
            res["test"]["ww_band_analytic"] = evaluate_baseline(test, "ww_band", k_star)[0]
            res["test"]["model_delta"] = evaluate_baseline(test, "model_delta")[0]
        if a.dynamics == "heston" and a.product == "call":
            res["test"]["model_delta"] = evaluate_baseline(test, "model_delta")[0]
            res["test"]["mv_delta"] = evaluate_baseline(test, "mv_delta")[0]
        if test.A > 1:   # several test years (e.g. market sp500x): also report each year separately, in sorted-year order
            res["test_by_asset"] = {}
            for kname, kk in (("model_target", None), ("model_band", k_star)):
                _, arr = evaluate_target_band(test, *greeks["test"], kk, return_arrays=True)
                res["test_by_asset"][kname] = per_asset_metrics(test, arr)
            _, arr = evaluate_baseline(test, "no_hedge")
            res["test_by_asset"]["no_hedge"] = per_asset_metrics(test, arr)
        for kname, m in res["test"].items():
            log(f"   test {kname:18} track={m['mean_abs_tracking_error']:.3f} cost={m['mean_cost']:.2f} trades={m['mean_trades']:.1f}")
        (a.out / f"{a.dynamics}_{a.product}_fee{fee}.json").write_text(json.dumps(res, indent=1))
    log("MODEL BAND DONE")


if __name__ == "__main__":
    main()
