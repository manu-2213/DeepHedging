"""Model-based reference strategies for every cell of the main table (2026-11-11).

For the call (one hedging instrument, the underlying) this is the Whalley-Wilmott band around the model delta:
BS delta or Heston delta, gamma by finite differences. For the down-and-out call and the cash-or-nothing, which are
hedged with two vanilla options (a call and a put), the target is the model delta-gamma matching portfolio in those
two options, and the no-trade band around each position follows the Whalley-Wilmott form with the instrument's own
price and sensitivity: half-width H_i = k (exp(-r tau) fee_i X_i (d target_i / dS)^2)^(1/3). Greeks come from 5-point
finite differences of the same pricers that built the banks (Heston barrier: the nested Monte Carlo with common random
numbers). The band scale k is chosen on the validation bank only; the test bank gives the reported numbers.

usage: python -m fast_hedging.model_band --dynamics bsm --product doc --fees 01 1 [--device cuda] [--val-paths N --test-paths N]
Writes experiments/results/main_baselines/<dynamics>_<product>_fee<fee>.json and caches the greeks next to the banks.
"""
import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import pricing
from .market import BANK_VERSION, MarketSpec, _simulate_heston, _split_params, cache_dir, get_bank
from .ppo import WW_GRID, Task, TrainConfig, evaluate_baseline, risk_metrics

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}
SIZES = {"val": 500, "test": 2000}
FEES = {"01": (0.001, 0.002), "1": (0.01, 0.05)}
BUMP = 0.01                     # relative spot bump of the finite differences
MULT = (-2, -1, 0, 1, 2)


def bank_name(spec, split, n, seed):
    key = json.dumps({"v": BANK_VERSION, "spec": asdict(spec), "split": split, "n": n, "seed": seed}, sort_keys=True)
    return f"{spec.dynamics}_{spec.product}_{split}_{n}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def prices_bsm(spec, p, S_b, S_orig):
    """Liability and instrument prices along the paths S_b (bumped spot, original history for the barrier)."""
    K, sig, r = p["K"], p["sigma"], spec.r
    T = np.linspace(spec.maturity, 0.0, spec.num_steps + 1)
    if spec.product == "call":
        L = pricing.bs_euro(S_b, K, T, r, sig, 1).astype(np.float64)
        X = np.broadcast_to(S_b[:, :, None, :, None], L.shape + (1,)).astype(np.float64)
        return L, X
    H = spec.barrier_alpha * K
    if spec.product == "doc":
        S, Kx, Tx, sg = pricing._expand(S_b, K, T, sig)
        Hx = np.asarray(H, np.float64)[None, :, :, None]
        vanilla = pricing.bs_euro(S_b, K, T, r, sig, 1).astype(np.float64)
        alpha = 0.5 * (1 - (r / (0.5 * sg**2)))
        S_new = Hx**2 / S
        body = pricing._bs_price(Kx, Tx[..., :-1], S_new[..., :-1], r, sg, 1)
        last = np.maximum(0.0, S_new - Kx)[..., -1:]
        term2 = (S / Hx) ** (2 * alpha) * np.concatenate([body, last], axis=-1)
        alive = np.cumprod(S_orig[:, :, None, :] > Hx, axis=-1)          # knock-out status of the real path
        L = (vanilla - term2) * alive
        put = pricing.bs_euro(S_b, H**2 / K, T, r, sig, -1).astype(np.float64)
        return L, np.stack([vanilla, put], axis=-1)
    Q = np.broadcast_to(spec.digital_payout_frac * p["S0"][:, None], K.shape).copy() if spec.digital_payout_frac > 0 \
        else np.full_like(K, spec.digital_payout)
    L = pricing.bs_cash_or_nothing(S_b, K, Q, T, r, sig).astype(np.float64)
    call = pricing.bs_euro(S_b, K, T, r, sig, 1).astype(np.float64)
    put = pricing.bs_euro(S_b, K, T, r, sig, -1).astype(np.float64)
    return L, np.stack([call, put], axis=-1)


def prices_heston(spec, p, S_b, S_orig, V, seed, device, mc_paths=None):
    K, r = p["K"], spec.r
    T = np.linspace(spec.maturity, 0.0, spec.num_steps + 1)
    hp = {k: p[k] for k in ("kappa", "theta", "rho", "xi")}
    kw = dict(phi_min=spec.heston_phi_min, phi_cap=spec.heston_phi_cap, dphi=spec.heston_dphi, device=device)
    H = spec.barrier_alpha * K
    if spec.product == "doc":
        M = K.shape[1]
        calls, _ = pricing.heston_vanilla_and_digital(S_b, V, T, hp, r, np.concatenate([K, H**2 / K], axis=1), **kw)
        call = calls[:, :, :M].astype(np.float64)
        put = pricing.heston_put_from_call(calls[:, :, M:], S_b, H**2 / K, T, r).astype(np.float64)
        L = pricing.heston_down_out_call(S_b, V, T, hp, r, K, H, call, mc_paths=mc_paths or spec.heston_mc_paths,
                                         seed=seed + 7, device=device).astype(np.float64)
        alive = np.cumprod(S_orig[:, :, None, :] > np.asarray(H)[None, :, :, None], axis=-1)
        return L * alive, np.stack([call, put], axis=-1)
    call, dig = pricing.heston_vanilla_and_digital(S_b, V, T, hp, r, K, **kw)
    call = call.astype(np.float64)
    if spec.product == "call":
        return call, np.broadcast_to(S_b[:, :, None, :, None], call.shape + (1,)).astype(np.float64)
    Q = np.broadcast_to(spec.digital_payout_frac * p["S0"][:, None], K.shape).copy() if spec.digital_payout_frac > 0 \
        else np.full_like(K, spec.digital_payout)
    L = dig.astype(np.float64) * Q[None, :, :, None]
    put = pricing.heston_put_from_call(call, S_b, K, T, r).astype(np.float64)
    return L, np.stack([call, put], axis=-1)


def ridge_solve(A, y, eps=1e-6):
    """x = argmin |A x - y|^2 + eps |x|^2 for stacks of 2x2 systems. A: (..., 2, 2), y: (..., 2)."""
    AtA = np.einsum("...ji,...jk->...ik", A, A)
    Aty = np.einsum("...ji,...j->...i", A, y)
    lam = eps * np.maximum(1.0, np.einsum("...ii->...", AtA))[..., None, None]
    AtA = AtA + lam * np.eye(2)
    det = AtA[..., 0, 0] * AtA[..., 1, 1] - AtA[..., 0, 1] * AtA[..., 1, 0]
    inv = np.stack([np.stack([AtA[..., 1, 1], -AtA[..., 0, 1]], -1), np.stack([-AtA[..., 1, 0], AtA[..., 0, 0]], -1)], -2) / det[..., None, None]
    return np.einsum("...ij,...j->...i", inv, Aty)


def compute_targets(spec, split, bank, device, log, bump=BUMP, mc_paths=None):
    """Model target positions and their spot sensitivity along the bank's paths: (P, A, M, N+1, I) each."""
    name = bank_name(spec, split, bank.S.shape[0], SPLIT_SEEDS[split])
    cache = cache_dir() / (f"{name}.greeks_h{bump:g}" + (f"_mc{mc_paths}" if mc_paths else "") + ".npz")
    if cache.exists():
        with np.load(cache) as d:
            return d["target"], d["dtarget"]
    p = _split_params(spec, split)
    S = bank.S.astype(np.float64)
    V = None
    if spec.dynamics == "heston":
        rng = np.random.default_rng(SPLIT_SEEDS[split])
        S2, V = _simulate_heston(p, S.shape[0], spec, rng)
        assert np.allclose(S2, S, rtol=1e-5, atol=1e-6), "could not reproduce the bank's Heston paths"
    h = bump
    Ls, Xs = {}, {}
    for m in MULT:
        S_b = S * (1.0 + m * h)
        if spec.dynamics == "bsm":
            Ls[m], Xs[m] = prices_bsm(spec, p, S_b, S)
        else:
            Ls[m], Xs[m] = prices_heston(spec, p, S_b, S, V, SPLIT_SEEDS[split], device, mc_paths)
        log(f"[greeks] {split} bump {m:+d} priced")
    Sx = S[:, :, None, :]                                     # (P, A, 1, N+1)
    lo, hi = (-1.0, 1.0)
    targets = {}
    for m in (-1, 0, 1):
        dS = h * Sx
        dL = (Ls[m + 1] - Ls[m - 1]) / (2 * dS)
        gL = (Ls[m + 1] - 2 * Ls[m] + Ls[m - 1]) / dS**2
        dX = (Xs[m + 1] - Xs[m - 1]) / (2 * dS[..., None])
        gX = (Xs[m + 1] - 2 * Xs[m] + Xs[m - 1]) / dS[..., None] ** 2
        if Xs[m].shape[-1] == 1:                             # the underlying: target = model delta
            tgt = dL[..., None]
        else:                                                # two options: match delta and (spot-scaled) gamma
            Sm = Sx * (1.0 + m * h)
            A = np.stack([dX, gX * Sm[..., None]], axis=-2)              # (..., 2 equations, 2 instruments)
            y = np.stack([dL, gL * Sm], axis=-1)
            tgt = ridge_solve(A, y)
        targets[m] = np.clip(np.nan_to_num(tgt, nan=0.0, posinf=lo, neginf=hi), lo, hi)
    dtarget = (targets[1] - targets[-1]) / (2 * h * Sx[..., None])
    target = targets[0]
    np.savez_compressed(cache, target=target.astype(np.float32), dtarget=np.nan_to_num(dtarget).astype(np.float32))
    log(f"[greeks] cached {cache.name}")
    return target.astype(np.float32), np.nan_to_num(dtarget).astype(np.float32)


@torch.no_grad()
def evaluate_target_band(task, target, dtarget, k, chunk=256, return_arrays=False):
    """Rebalance to `target` only outside the band of half-width k (exp(-r tau) fee_i X_i dtarget_i^2)^(1/3); k=None: daily."""
    tgt_all = torch.as_tensor(target, device=task.device)
    dt_all = torch.as_tensor(dtarget, device=task.device)
    pnls, costs, trades, errs = [], [], [], []
    disc = torch.exp(-task.r * task.T[: task.N]).to(task.device)       # (N,)
    for lo in range(0, task.P, chunk):
        idx = torch.arange(lo, min(task.P, lo + chunk), device=task.device)
        B = idx.numel() * task.A * task.M
        X = task.X[idx].reshape(B, task.N1, task.I)
        L = task.L[idx].reshape(B, task.N1)
        tg = tgt_all[idx].reshape(B, task.N1, task.I)[:, : task.N]
        dg = dt_all[idx].reshape(B, task.N1, task.I)[:, : task.N]
        band = None
        if k is not None:
            band = float(k) * (disc[None, :, None] * task.fees[None, None, :] * X[:, : task.N].abs() * dg**2).clamp_min(0).pow(1.0 / 3.0)
        pos = torch.zeros(B, task.I, device=task.device)
        cash = L[:, 0].clone()
        cost_tot = torch.zeros(B, device=task.device)
        n_tr = torch.zeros(B, device=task.device)
        abs_err = torch.zeros(B, device=task.device)
        for t in range(task.N):
            final = tg[:, t]
            if band is not None:
                final = torch.maximum(torch.minimum(pos, final + band[:, t]), final - band[:, t])
            trade = final - pos
            flow = trade * X[:, t]
            cost = (task.fees * flow.abs()).sum(-1)
            cash = (cash - flow.sum(-1) - cost) * task.growth
            pos = final
            cost_tot += cost
            n_tr += (trade.abs().amax(-1) > 1e-6).float()
            abs_err += (cash + (pos * X[:, t + 1]).sum(-1) - L[:, t + 1]).abs()
        pv = cash + (pos * X[:, task.N]).sum(-1)
        pnls.append((pv - L[:, task.N]).cpu()); costs.append(cost_tot.cpu()); trades.append(n_tr.cpu()); errs.append((abs_err / task.N).cpu())
    pnl, cost, n_trades, abs_err = (torch.cat(v).numpy() for v in (pnls, costs, trades, errs))
    m = risk_metrics(pnl)
    m.update({"mean_abs_tracking_error": float(abs_err.mean()), "mean_cost": float(cost.mean()), "mean_trades": float(n_trades.mean()),
              "no_trade_rate": float(1.0 - n_trades.mean() / task.N)})
    if k is not None:
        m["param"] = float(k)
    if return_arrays:
        return m, {"pnl": pnl, "cost": cost, "n_trades": n_trades, "abs_err": abs_err}
    return m


def _per_asset(task, arr):
    """Metrics per asset (= calibration year) from per-env arrays in (path, asset, strike) order."""
    asset = np.repeat(np.tile(np.arange(task.A), task.P), task.M)
    out = {}
    for a in range(task.A):
        sel = asset == a
        m = risk_metrics(arr["pnl"][sel])
        m.update({"mean_abs_tracking_error": float(arr["abs_err"][sel].mean()), "mean_cost": float(arr["cost"][sel].mean()),
                  "mean_trades": float(arr["n_trades"][sel].mean())})
        out[str(a)] = m
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamics", choices=("bsm", "heston"), required=True)
    ap.add_argument("--product", choices=("call", "doc", "con"), required=True)
    ap.add_argument("--fees", nargs="+", default=["01", "1"], choices=list(FEES))
    ap.add_argument("--val-paths", type=int, default=SIZES["val"]); ap.add_argument("--test-paths", type=int, default=SIZES["test"])
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
    greeks = {s: compute_targets(spec, s, banks[s], a.device, log, a.bump, a.mc_paths) for s in ("val", "test")}
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
                res["test_by_asset"][kname] = _per_asset(test, arr)
            _, arr = evaluate_baseline(test, "no_hedge")
            res["test_by_asset"]["no_hedge"] = _per_asset(test, arr)
        for kname, m in res["test"].items():
            log(f"   test {kname:18} track={m['mean_abs_tracking_error']:.3f} cost={m['mean_cost']:.2f} trades={m['mean_trades']:.1f}")
        (a.out / f"{a.dynamics}_{a.product}_fee{fee}.json").write_text(json.dumps(res, indent=1))
    log("MODEL BAND DONE")


if __name__ == "__main__":
    main()
