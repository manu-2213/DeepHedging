"""Calibrated market parameters: the Black-Scholes and Heston fits the engine simulates from.

One JSON per market and model in hedging_engine/data, keyed by calibration year. A market's years are used in sorted
order: the first `train_size` years are the training (and validation) calibrations, the rest are the test calibrations.
The scripts that produced the files are in experiments/calibration.

  bsm_data_<market>_2017_2021.json     per year: S0, mean, std (annualised log-return volatility), K_sampled (5 strikes)
  heston_params_QL_v2_<market>.json    per year: v0, kappa, theta, rho, xi (+ fit diagnostics), S0, K_sampled

Markets: sp500 (2017-2021, the paper's main table), sp500x (2017-2023, the out-of-sample years), and four more
(alphabet, apple, microsoft, nasdaq100) that the paper does not use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FLOAT_DTYPE = np.float32


def data_dir() -> Path:
    return Path(os.getenv("HEDGING_ENGINE_DATA", str(Path(__file__).resolve().parent / "data")))


def _years(data: dict):
    return sorted(data.keys())          # deterministic order


def convert_bsm(data: dict):
    """Yearly dict -> S0 (N,), K (N, M), sigma (N,)."""
    years = _years(data)
    S0 = np.array([data[y]["S0"] for y in years], dtype=FLOAT_DTYPE)
    K = np.array([data[y]["K_sampled"] for y in years], dtype=FLOAT_DTYPE)
    sigma = np.array([data[y]["std"] for y in years], dtype=FLOAT_DTYPE)
    return S0, K, sigma


def convert_heston(data: dict):
    """Yearly dict -> params {kappa, theta, rho, sigma (= vol of vol xi), lda}, S0 (N,), K (N, M), v0 (N,)."""
    years = _years(data)
    kappa = np.array([data[y]["kappa"] for y in years], dtype=FLOAT_DTYPE)
    theta = np.array([data[y]["theta"] for y in years], dtype=FLOAT_DTYPE)
    rho = np.array([data[y]["rho"] for y in years], dtype=FLOAT_DTYPE)
    sigma = np.array([data[y]["xi"] for y in years], dtype=FLOAT_DTYPE)
    params = {"kappa": kappa, "theta": theta, "rho": rho, "sigma": sigma, "lda": np.zeros_like(kappa)}
    S0 = np.array([data[y]["S0"] for y in years], dtype=FLOAT_DTYPE)
    v0 = np.array([data[y]["v0"] for y in years], dtype=FLOAT_DTYPE)
    K = np.array([data[y]["K_sampled"] for y in years], dtype=FLOAT_DTYPE)
    return params, S0, K, v0


def load_bsm_data(market: str, path: Path | None = None):
    with open(path or data_dir() / f"bsm_data_{market}_2017_2021.json", "r", encoding="utf-8") as f:
        return convert_bsm(json.load(f))


def load_heston_data(market: str, path: Path | None = None):
    with open(path or data_dir() / f"heston_params_QL_v2_{market}.json", "r", encoding="utf-8") as f:
        return convert_heston(json.load(f))


def train_test_split(dynamics: str, train_size: int, market: str):
    """((params,) S0, K, sigma-or-v0) for the training years and for the test years."""
    if dynamics == "heston":
        params, S0, K, v0 = load_heston_data(market=market)
        params_train = {k: v[:train_size] for k, v in params.items()}
        params_test = {k: v[train_size:] for k, v in params.items()}
        return ((params_train, S0[:train_size], K[:train_size], v0[:train_size]),
                (params_test, S0[train_size:], K[train_size:], v0[train_size:]))
    if dynamics == "bsm":
        S0, K, sigma = load_bsm_data(market=market)
        return ((S0[:train_size], K[:train_size], sigma[:train_size]),
                (S0[train_size:], K[train_size:], sigma[train_size:]))
    raise ValueError("dynamics must be 'heston' or 'bsm'")
