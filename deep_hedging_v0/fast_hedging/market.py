"""Market banks: simulated paths plus every price the hedging problem needs.

A bank is generated once per (dynamics, product, split, size, seed) and cached
as .npz, so every Optuna trial and every seed reuses identical market data.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from . import pricing

PKG_ROOT = Path(__file__).resolve().parents[1]           # deep_hedging_v0/
if str(PKG_ROOT) not in sys.path:
    sys.path.append(str(PKG_ROOT))

from experiments.utils.sim_config import train_test_split  # noqa: E402

BANK_VERSION = 4   # v4: adaptive Heston integration limit, pooled DOC premium, device-independent MC,
                   #     minimum-variance delta stored, one-step drift checks (v3: fixed phi_max = 50)
PRODUCTS = ("call", "doc", "con")
DYNAMICS = ("bsm", "heston")


@dataclass(frozen=True)
class MarketSpec:
    dynamics: str = "bsm"
    product: str = "call"
    market: str = "sp500"
    train_size: int = 4          # first 4 calibration years train/val, the rest test
    maturity: float = 1.0
    r: float = 0.01
    num_steps: int = 250
    barrier_alpha: float = 0.9   # DOC barrier = alpha * K
    digital_payout: float = 1.0  # CON pays this amount when digital_payout_frac <= 0
    digital_payout_frac: float = 0.1   # CON pays this fraction of the asset's S0 (v3 default)
    heston_phi_min: float = 50.0      # Gil-Pelaez integration limit: at least this ...
    heston_phi_cap: float = 3200.0    # ... and at most this; adaptive per point (pricing.heston_phi_max)
    heston_dphi: float = 0.05         # grid spacing of the integral
    heston_mc_paths: int = 2000       # inner paths of the barrier nested MC per start point (v3: 500)


@dataclass
class Bank:
    """All arrays have leading dims (P, A, ...). M = strikes per asset."""
    S: np.ndarray            # (P, A, N+1) underlying
    liability: np.ndarray    # (P, A, M, N+1) price of the claim being hedged
    instruments: np.ndarray  # (P, A, M, N+1, I) prices of hedging instruments
    call: np.ndarray         # (P, A, M, N+1) vanilla call (feature input, doc/con)
    put: np.ndarray          # (P, A, M, N+1) put used as hedge (feature input, doc/con)
    K: np.ndarray            # (A, M)
    T: np.ndarray            # (N+1,) time to maturity grid
    instrument_kind: tuple   # e.g. ("underlying",) or ("option", "option")
    sigma: np.ndarray | None = None   # (A,) BSM volatility, used only for the delta-hedge baseline
    delta: np.ndarray | None = None   # (P, A, M, N+1) model delta of the call (call product only, else empty)
    delta_mv: np.ndarray | None = None  # (P, A, M, N+1) minimum-variance delta (Heston call only, else empty)
    check: dict | None = None         # result of check_bank (not stored in the .npz)

    @property
    def num_paths(self):
        return self.S.shape[0]


def _split_params(spec: MarketSpec, split: str):
    train, test = train_test_split(dynamics=spec.dynamics, train_size=spec.train_size, market=spec.market)
    part = train if split in ("train", "val") else test
    if spec.dynamics == "bsm":
        S0, K, sigma = part
        return {"S0": np.asarray(S0, np.float64), "K": np.asarray(K, np.float64),
                "sigma": np.asarray(sigma, np.float64)}
    params, S0, K, v0 = part
    return {"S0": np.asarray(S0, np.float64), "K": np.asarray(K, np.float64),
            "v0": np.asarray(v0, np.float64), "kappa": np.asarray(params["kappa"], np.float64),
            "theta": np.asarray(params["theta"], np.float64), "rho": np.asarray(params["rho"], np.float64),
            "xi": np.asarray(params["sigma"], np.float64)}


def _simulate_bsm(p, n_paths, spec, rng):
    A, N = len(p["S0"]), spec.num_steps
    dt = spec.maturity / N
    z = rng.standard_normal((n_paths, A, N))
    sig = p["sigma"][None, :, None]
    log_ret = (spec.r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z
    logS = np.log(p["S0"])[None, :, None] + np.cumsum(log_ret, axis=-1)
    S = np.concatenate([np.broadcast_to(p["S0"][None, :, None], (n_paths, A, 1)), np.exp(logS)], axis=-1)
    return S.astype(np.float32), None


def _simulate_heston(p, n_paths, spec, rng):
    """Euler full-truncation scheme, identical to HestonSimulator._generate_paths."""
    A, N = len(p["S0"]), spec.num_steps
    dt = spec.maturity / N
    z_s = rng.standard_normal((n_paths, A, N))
    z_v = rng.standard_normal((n_paths, A, N))
    rho = p["rho"][None, :]
    z_v = rho[..., None] * z_s + np.sqrt(1 - rho[..., None] ** 2) * z_v
    v = np.broadcast_to(p["v0"][None, :], (n_paths, A)).astype(np.float64).copy()
    logS = np.broadcast_to(np.log(p["S0"])[None, :], (n_paths, A)).copy()
    V = np.empty((n_paths, A, N + 1)); LS = np.empty((n_paths, A, N + 1))
    V[..., 0], LS[..., 0] = v, logS
    for t in range(N):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp * dt)
        v = v + p["kappa"] * (p["theta"] - vp) * dt + p["xi"] * sq * z_v[..., t]
        logS = logS + (spec.r - 0.5 * vp) * dt + sq * z_s[..., t]
        V[..., t + 1], LS[..., t + 1] = v, logS
    return np.exp(LS).astype(np.float32), V.astype(np.float32)


def build_bank(spec: MarketSpec, split: str, n_paths: int, seed: int, device="cpu") -> Bank:
    p = _split_params(spec, split)
    rng = np.random.default_rng(seed)
    N = spec.num_steps
    T = np.linspace(spec.maturity, 0.0, N + 1)
    K = p["K"]
    H = spec.barrier_alpha * K
    if spec.digital_payout_frac > 0:   # payout scaled to the market: comparable to the call premium
        Q = np.broadcast_to(spec.digital_payout_frac * p["S0"][:, None], K.shape).copy()
    else:
        Q = np.full_like(K, spec.digital_payout)
    delta = np.zeros((0,), np.float32)
    delta_mv = np.zeros((0,), np.float32)
    r = spec.r
    if spec.dynamics == "bsm":
        S, _ = _simulate_bsm(p, n_paths, spec, rng)
        sig = p["sigma"]
        call = pricing.bs_euro(S, K, T, r, sig, 1)
        if spec.product == "call":
            liab, put = call, np.zeros_like(call)
            delta = pricing.bs_call_delta(S, K, T, r, sig)
        elif spec.product == "doc":
            liab = pricing.bs_down_out_call(S, K, H, T, r, sig)
            put = pricing.bs_euro(S, H**2 / K, T, r, sig, -1)
        else:
            liab = pricing.bs_cash_or_nothing(S, K, Q, T, r, sig)
            put = pricing.bs_euro(S, K, T, r, sig, -1)
    else:
        S, V = _simulate_heston(p, n_paths, spec, rng)
        hp = {k: p[k] for k in ("kappa", "theta", "rho", "xi")}
        kw = dict(phi_min=spec.heston_phi_min, phi_cap=spec.heston_phi_cap, dphi=spec.heston_dphi, device=device)
        if spec.product == "doc":
            strikes = np.concatenate([K, H**2 / K], axis=1)
            calls, _ = pricing.heston_vanilla_and_digital(S, V, T, hp, r, strikes, **kw)
            M = K.shape[1]
            call = calls[:, :, :M]
            put = pricing.heston_put_from_call(calls[:, :, M:], S, H**2 / K, T, r)
            liab = pricing.heston_down_out_call(S, V, T, hp, r, K, H, call, mc_paths=spec.heston_mc_paths,
                                                seed=seed + 7, device=device)
        else:
            call, dig, hdelta, hdelta_mv = pricing.heston_vanilla_and_digital(S, V, T, hp, r, K, return_delta=True, **kw)
            if spec.product == "call":
                liab, put = call, np.zeros_like(call)
                delta, delta_mv = hdelta, hdelta_mv
            else:
                liab = (dig * Q[None, :, :, None]).astype(np.float32)
                put = pricing.heston_put_from_call(call, S, K, T, r)
    if spec.product == "call":
        instruments = np.broadcast_to(S[:, :, None, :, None], call.shape + (1,)).copy()
        kind = ("underlying",)
    else:
        instruments = np.stack([call, put], axis=-1)
        kind = ("option", "option")
    return Bank(S=S, liability=liab.astype(np.float32), instruments=instruments.astype(np.float32),
                call=call.astype(np.float32), put=put.astype(np.float32), K=K.astype(np.float32),
                T=T.astype(np.float32), instrument_kind=kind,
                sigma=p["sigma"].astype(np.float32) if spec.dynamics == "bsm" else np.full(len(K), np.nan, np.float32),
                delta=np.asarray(delta, np.float32), delta_mv=np.asarray(delta_mv, np.float32))


class BankError(RuntimeError):
    pass


def check_bank(bank: Bank, r: float, z_warn=5.0, rel_warn=0.01) -> dict:
    """Sanity checks that catch broken market data before any training.

    * every array finite (hard error otherwise)
    * prices non-negative
    * martingale test (unconditional): under the simulated risk-neutral dynamics every discounted
      price must satisfy E[e^{-r t} X_t] = X_0; tested at t in {N/4, N/2, 3N/4, N-10, N-5, N-1, N}.
    * one-step drift test at EVERY step: E[e^{-r dt} X_{t+1} - X_t] = 0 across paths. Its standard
      error is one daily move over sqrt(P), so it detects a surface that is mispriced on a few dates
      (the fixed phi_max = 50 bug lived in the last ten trading days) which the unconditional test
      averages away, and which t = N cannot see because the terminal column is the exact payoff.
    Large z-scores flag mispriced surfaces or a discretisation bias between the simulated paths
    and the pricing formula. A warning needs both |z| > z_warn and a relative error > rel_warn
    (relative to the t = 0 price), so MC noise on a cheap option is not reported.
    """
    out = {"finite": True, "warnings": []}
    for name in ("S", "liability", "instruments", "call", "put"):
        if not np.isfinite(getattr(bank, name)).all():
            raise BankError(f"non-finite values in bank.{name}")
    for name in ("delta", "delta_mv"):
        arr = getattr(bank, name, None)
        if arr is not None and arr.size and not np.isfinite(arr).all():
            raise BankError(f"non-finite values in bank.{name}")
    if (bank.liability < -1e-3).any():
        out["warnings"].append(f"negative liability prices (min {bank.liability.min():.4g})")
    N = bank.T.shape[0] - 1
    dt = float(bank.T[0] - bank.T[1])
    elapsed = (bank.T[0] - bank.T).astype(np.float64)
    series = {"liability": bank.liability}
    for i, kind in enumerate(bank.instrument_kind):
        series[f"instrument{i}_{kind}"] = bank.instruments[..., i]
    mart = {}
    times = sorted({N // 4, N // 2, 3 * N // 4, max(N - 10, 1), max(N - 5, 1), N - 1, N})
    for name, X in series.items():               # X: (P, A, M, N+1)
        X64 = X.astype(np.float64)
        X0 = X64[:, :, :, 0].mean(0)              # identical across paths at t=0
        ref = np.maximum(np.abs(X0), 1e-8)
        worst_z, worst_rel = 0.0, 0.0
        for t in times:
            disc = np.exp(-r * elapsed[t]) * X64[:, :, :, t]
            se = disc.std(0) / np.sqrt(disc.shape[0])
            diff = disc.mean(0) - X0
            z = np.abs(diff) / np.maximum(se, 1e-12)
            rel = np.abs(diff) / ref
            worst_z, worst_rel = max(worst_z, float(z.max())), max(worst_rel, float(rel.max()))
            bad = (z > z_warn) & (rel > rel_warn)
            if bad.any():
                out["warnings"].append(f"{name}: martingale violation at t={t} (max z {z.max():.1f}, "
                                       f"rel {rel.max():.2%})")
        step = np.exp(-r * dt) * X64[..., 1:] - X64[..., :-1]             # (P, A, M, N)
        mu = step.mean(0)
        se1 = step.std(0) / np.sqrt(step.shape[0])
        z1 = np.abs(mu) / np.maximum(se1, 1e-12)
        rel1 = np.abs(mu) / ref[..., None]
        bad1 = (z1 > z_warn) & (rel1 > rel_warn)
        if bad1.any():
            ts = np.unique(np.nonzero(bad1)[-1])
            shown = ts[:8].tolist() + (["..."] if ts.size > 8 else [])
            out["warnings"].append(f"{name}: one-step drift at {ts.size} step(s) t={shown} "
                                   f"(max z {z1.max():.1f}, rel {rel1.max():.2%})")
        mart[name] = {"max_z": worst_z, "max_rel_err": worst_rel,
                      "step_max_z": float(z1.max()), "step_max_rel_err": float(rel1.max()),
                      "step_worst_t": int(np.unravel_index(int(z1.argmax()), z1.shape)[-1])}
    out["martingale"] = mart
    return out


def cache_dir() -> Path:
    return Path(os.getenv("FAST_HEDGING_CACHE", str(PKG_ROOT.parent / ".fast_hedging_cache")))


def get_bank(spec: MarketSpec, split: str, n_paths: int, seed: int, device="cpu", verbose=True) -> Bank:
    key = json.dumps({"v": BANK_VERSION, "spec": asdict(spec), "split": split, "n": n_paths, "seed": seed},
                     sort_keys=True)
    name = f"{spec.dynamics}_{spec.product}_{split}_{n_paths}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"
    path = cache_dir() / name
    fields = ("S", "liability", "instruments", "call", "put", "K", "T", "sigma", "delta", "delta_mv")
    bank = None
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as d:
                kind = tuple(str(x) for x in d["instrument_kind"])
                bank = Bank(**{f: d[f] for f in fields}, instrument_kind=kind)
        except (OSError, ValueError, KeyError):
            bank = None
    if bank is not None:
        bank.check = check_bank(bank, spec.r)
        _report_check(name, bank.check, verbose)
        return bank
    if verbose:
        print(f"[bank] building {name} on {device} ...", flush=True)
    bank = build_bank(spec, split, n_paths, seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, **{f: getattr(bank, f) for f in fields}, instrument_kind=np.array(bank.instrument_kind))
    os.replace(tmp, path)
    if verbose:
        print(f"[bank] saved {path}", flush=True)
    bank.check = check_bank(bank, spec.r)
    _report_check(name, bank.check, verbose)
    return bank


def _report_check(name, check, verbose):
    if not verbose:
        return
    worst = max(v["max_z"] for v in check["martingale"].values())
    worst1 = max(v["step_max_z"] for v in check["martingale"].values())
    print(f"[bank] check {name}: martingale max |z| = {worst:.1f}, one-step drift max |z| = {worst1:.1f}"
          + ("" if not check["warnings"] else "  WARNINGS: " + "; ".join(check["warnings"])), flush=True)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
