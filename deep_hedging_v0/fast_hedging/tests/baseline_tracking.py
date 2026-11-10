"""Tracking error (mean_t |PV_t+1 - L_t+1|, the agents' 'track' metric) and CVaR95 of the classical
baselines on the v2.4 BSM call banks, for both fee settings. The WW band scale is tuned on validation
twice: once for CVaR95 (as in the experiment reports) and once for tracking error.

Usage (from deep_hedging_v0/):  python -m fast_hedging.tests.baseline_tracking [cpu|cuda]
"""
import json
import sys

import torch

from fast_hedging.experiment import load_tasks, make_config, parse_args
from fast_hedging.ppo import evaluate_baseline, tune_ww_band

FEES = {"0.1%": ["--underlying-fee", "0.001", "--option-fee", "0.002"],
        "1%": ["--underlying-fee", "0.01", "--option-fee", "0.05"]}
BASE = ["--dynamics", "bsm", "--product", "call", "--net", "band", "--arch", "rnn", "--rnn-mode", "window",
        "--window", "10", "--reward", "tracking", "--no-obs-norm", "--no-agent-state", "--name", "baseline_tracking", "--out", "/tmp/baseline_tracking"]
KEYS = ("cvar_95", "mean", "mean_abs_tracking_error", "mean_cost", "mean_trades")


def main():
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = {}
    for fee, flags in FEES.items():
        a = parse_args(BASE + flags)
        (_, val, test), *_ = load_tasks(a, make_config(a), device)
        k_cvar, _ = tune_ww_band(val, metric="cvar_95")
        k_track, _ = tune_ww_band(val, metric="mean_abs_tracking_error")
        res = {"no_hedge": evaluate_baseline(test, "no_hedge")[0],
               "model_delta": evaluate_baseline(test, "model_delta")[0],
               f"ww_band (cvar-tuned, k={k_cvar:.3g})": evaluate_baseline(test, "ww_band", k_cvar)[0],
               f"ww_band (track-tuned, k={k_track:.3g})": evaluate_baseline(test, "ww_band", k_track)[0]}
        out[fee] = {n: {k: m[k] for k in KEYS} for n, m in res.items()}
        for n, m in out[fee].items():
            print(f"{fee:5s} {n:32s} " + "  ".join(f"{k}={m[k]:.2f}" for k in KEYS), flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
