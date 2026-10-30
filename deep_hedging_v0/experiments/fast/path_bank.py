"""Pre-simulated path banks: asset paths plus every price the hedging env needs.

A bank is built once per (dynamics, product, split) and cached on disk, so no
training run ever re-simulates or re-prices. The old envs repriced every epoch,
and the Heston pricer looped in Python over every (path, asset, strike, step).

Split layout:
    train : training calibration years (2017-2020), seed_train
    val   : the same years, fresh paths (seed_val)  -> early stopping + Optuna
    test  : held-out year (2021), seed_test          -> reported only, never used for selection

Bank tensors (float32):
    S  (P, A, N+1)       underlying paths
    V  (P, A, M, N+1)    liability price (call / down-and-out call / cash-or-nothing call)
    C  (P, A, M, N+1)    hedge call price           (doc, con only)
    Pt (P, A, M, N+1)    hedge put price            (doc, con only)
    K  (A, M), S0 (A,), T (N+1,) time to maturity
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.fast.config import BankConfig, MarketConfig, config_hash

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- data
def load_market_params(dynamics: str, market: MarketConfig) -> tuple[dict, dict]:
    from experiments.utils.sim_config import train_test_split

    train, test = train_test_split(dynamics=dynamics, train_size=market.train_size, market=market.market)

    def pack(split):
        if dynamics == "bsm":
            S0, K, sigma = split
            return {"S0": S0, "K": K, "sigma": sigma}
        params, S0, K, v0 = split
        return {"S0": S0, "K": K, "v0": v0, "kappa": params["kappa"], "theta": params["theta"],
                "rho": params["rho"], "xi": params["sigma"]}

    return pack(train), pack(test)


# ---------------------------------------------------------------------------- BSM
def _bsm_bank(prm: dict, product: str, market: MarketConfig, n_paths: int, seed: int) -> dict:
    """Uses the existing (already vectorised) BlackScholesSimulator so prices match the old envs."""
    os.environ["DEEP_HEDGING_PATH_CACHE"] = "0"
    from hedging.simulators import BlackScholesSimulator

    S0 = np.asarray(prm["S0"], dtype=np.float32)
    K = np.asarray(prm["K"], dtype=np.float32)
    sigma = np.asarray(prm["sigma"], dtype=np.float32)
    sim = BlackScholesSimulator(S0, float(market.r), sigma, float(market.maturity),
                                market.num_steps, n_paths, np.random.default_rng(seed))
    out = {"S": sim.paths, "T": sim.T}
    if product == "call":
        out["V"] = sim.euro_call(K)
    elif product == "doc":
        H = (market.barrier_alpha * K).astype(np.float32)
        out["V"] = sim.down_out_call(K, H)
        out["C"] = sim.euro_call(K)
        out["Pt"] = sim.euro_put(H ** 2 / K)
    elif product == "con":
        Q = np.full_like(K, market.digital_payout)
        out["V"] = sim.cash_or_nothing_call(K, Q)
        out["C"] = sim.euro_call(K)
        out["Pt"] = sim.euro_put(K)
    else:
        raise ValueError(product)
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for k, v in out.items()}


# ------------------------------------------------------------------------- Heston
def _heston_paths(prm, market, n_paths, seed, device):
    """Euler full-truncation scheme, identical to HestonSimulator._generate_paths."""
    f64 = torch.float64
    A, N = len(prm["S0"]), market.num_steps
    dt = market.maturity / N
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)[None, :]
    S0, v0, kap, th, rho, xi = (t(prm[k]) for k in ("S0", "v0", "kappa", "theta", "rho", "xi"))
    gen = torch.Generator(device=device).manual_seed(seed)
    z_s = torch.randn(N, n_paths, A, generator=gen, device=device, dtype=f64)
    z_p = torch.randn(N, n_paths, A, generator=gen, device=device, dtype=f64)
    z_v = rho * z_s + torch.sqrt(1.0 - rho ** 2) * z_p
    v = v0.expand(n_paths, A).clone()
    log_s = torch.log(S0).expand(n_paths, A).clone()
    S_hist, v_hist = [torch.exp(log_s)], [v]
    for n in range(N):
        v_pos = v.clamp(min=0.0)
        sq = torch.sqrt(v_pos * dt)
        v = v + kap * (th - v_pos) * dt + xi * sq * z_v[n]
        log_s = log_s + (market.r - 0.5 * v_pos) * dt + sq * z_s[n]
        S_hist.append(torch.exp(log_s))
        v_hist.append(v)
    return torch.stack(S_hist, -1), torch.stack(v_hist, -1)  # (P, A, N+1) float64


def _heston_prob(u, S, K, T, v, kap, th, rho, xi, r, phi, dx):
    """P1 (u=0.5) / P2 (u=-0.5) by trapezoid integration; same 'trap' formula as HestonSimulator."""
    S, K, T, v, kap, th, rho, xi = (x[:, None] for x in (S, K, T, v, kap, th, rho, xi))
    ph = phi[None, :]
    iph = 1j * ph
    b = kap - rho * xi if u > 0 else kap
    a = kap * th
    d = torch.sqrt((rho * xi * iph - b) ** 2 - xi ** 2 * (2 * u * iph - ph ** 2))
    bm = b - rho * xi * iph
    g = (bm + d) / (bm - d)
    c = 1.0 / g
    edt = torch.exp(-d * T)
    D = ((bm - d) / xi ** 2) * ((1 - edt) / (1 - c * edt))
    G = (1 - c * edt) / (1 - c)
    C = r * iph * T + (a / xi ** 2) * ((bm - d) * T - 2.0 * torch.log(G))
    cf = torch.exp(C + D * v + iph * torch.log(S))
    integrand = torch.real(torch.exp(-iph * torch.log(K)) * cf / iph)
    return 0.5 + torch.trapezoid(integrand, dx=dx, dim=-1) / math.pi


def _heston_grid(S, v, K, T, prm):
    """Flatten every (p, a, m, t<N) pricing point into 1-D tensors."""
    P, A, N1 = S.shape
    M, N = K.shape[1], N1 - 1
    shape = (P, A, M, N)
    dev = S.device
    per_asset = lambda k: torch.as_tensor(np.asarray(prm[k], np.float64), device=dev)[None, :, None, None].expand(shape)
    flat = lambda x: x.reshape(-1)
    return dict(
        S=flat(S[:, :, None, :N].expand(shape)),
        v=flat(v[:, :, None, :N].clamp(min=0.0).expand(shape)),
        K=flat(K[None, :, :, None].expand(shape)),
        T=flat(T[None, None, None, :N].expand(shape)),
        kap=flat(per_asset("kappa")), th=flat(per_asset("theta")),
        rho=flat(per_asset("rho")), xi=flat(per_asset("xi")),
    ), shape


def _heston_integrate(S, v, K, T, prm, market, bank, which):
    grid, shape = _heston_grid(S, v, K, T, prm)
    dev = S.device
    phi = torch.linspace(1e-8, bank.heston_phi_max, bank.heston_phi_points, dtype=torch.float64, device=dev)
    dx = float(phi[1] - phi[0])
    out = torch.empty(grid["S"].numel(), dtype=torch.float64, device=dev)
    for i in range(0, out.numel(), bank.heston_cf_chunk):
        sl = slice(i, i + bank.heston_cf_chunk)
        g = {k: x[sl] for k, x in grid.items()}
        args = (g["S"], g["K"], g["T"], g["v"], g["kap"], g["th"], g["rho"], g["xi"], market.r, phi, dx)
        p2 = _heston_prob(-0.5, *args)
        if which == "call":
            p1 = _heston_prob(0.5, *args)
            out[sl] = g["S"] * p1 - g["K"] * torch.exp(-market.r * g["T"]) * p2
        else:  # digital: discounted P2
            out[sl] = torch.exp(-market.r * g["T"]) * p2
    return out.reshape(shape)


def heston_call(S, v, K, T, prm, market, bank):
    prices = _heston_integrate(S, v, K, T, prm, market, bank, "call")
    S4, K4, T4 = S[:, :, None, :], K[None, :, :, None], T[None, None, None, :]
    last = torch.relu(S4[..., -1] - K4[..., -1])
    full = torch.cat([prices, last[..., None]], -1)
    lower = torch.relu(S4 - K4 * torch.exp(-market.r * T4))
    return torch.minimum(torch.maximum(full, lower), S4.expand_as(full))


def heston_put(S, v, K, T, prm, market, bank, call=None):
    call = heston_call(S, v, K, T, prm, market, bank) if call is None else call
    return call - S[:, :, None, :] + K[None, :, :, None] * torch.exp(-market.r * T[None, None, None, :])


def heston_digital(S, v, K, T, prm, market, bank, payout):
    prices = _heston_integrate(S, v, K, T, prm, market, bank, "digital") * payout
    S4, K4, T4 = S[:, :, None, :], K[None, :, :, None], T[None, None, None, :]
    last = payout * (S4[..., -1] > K4[..., -1]).to(prices.dtype)
    full = torch.cat([prices, last[..., None]], -1)
    return torch.clamp(full, min=0.0).minimum(payout * torch.exp(-market.r * T4).expand_as(full))


@torch.no_grad()
def heston_doc_mc(S, v, K, H, T, prm, market, bank, seed, vanilla):
    """Down-and-out call by nested MC (as HestonSimulator.down_out_call), vectorised on the GPU.

    Two changes versus the old code, both lowering noise and cost:
      * common random numbers: the inner draws are shared across start dates and
        outer paths, so MC error is smooth in t instead of fresh noise each step
        (fresh noise made the day-to-day price change mostly MC noise);
      * inner paths are shared across strikes (only the barrier/payoff differ).
    """
    dev, f32 = S.device, torch.float32
    P, A, N1 = S.shape
    M, N, mc = K.shape[1], N1 - 1, bank.heston_mc_paths
    dt, r = market.maturity / N, market.r
    S, v, K, H, T = S.to(f32), v.to(f32), K.to(f32), H.to(f32), T.to(f32)
    a_idx = torch.arange(A, device=dev).repeat(P)                          # column j = p*A + a
    pa = lambda k: torch.as_tensor(np.asarray(prm[k], np.float32), device=dev)[a_idx][None, :]
    kap, th, rho, xi = pa("kappa"), pa("theta"), pa("rho"), pa("xi")
    sq1m = torch.sqrt(1.0 - rho ** 2)
    Kf, Hf = K[a_idx][None], H[a_idx][None]                                # (1, PA, M)
    gen = torch.Generator(device=dev).manual_seed(seed + 7919)
    z_s = torch.randn(N, mc, 1, generator=gen, device=dev, dtype=f32)
    z_p = torch.randn(N, mc, 1, generator=gen, device=dev, dtype=f32)

    alive = torch.cumprod((S[:, :, None, :] > H[None, :, :, None]).to(f32), dim=-1)  # (P, A, M, N+1)
    out = torch.zeros(P, A, M, N1, dtype=f32, device=dev)
    out[..., N] = torch.relu(S[:, :, None, N] - K[None]) * alive[..., N]
    for t in range(N):
        s = S[:, :, t].reshape(1, P * A).expand(mc, P * A).clone()
        var = v[:, :, t].reshape(1, P * A).clamp(min=0.0).expand(mc, P * A).clone()
        run_min = torch.full_like(s, float("inf"))
        for k in range(N - t):
            v_pos = var.clamp(min=0.0)
            sq = torch.sqrt(v_pos * dt)
            var = var + kap * (th - v_pos) * dt + xi * sq * (rho * z_s[k] + sq1m * z_p[k])
            s = s * torch.exp((r - 0.5 * v_pos) * dt + sq * z_s[k])
            run_min = torch.minimum(run_min, s)
        payoff = torch.relu(s[..., None] - Kf) * (run_min[..., None] > Hf).to(f32)  # (mc, PA, M)
        est = math.exp(-r * float(T[t])) * payoff.mean(0)
        out[..., t] = est.reshape(P, A, M) * alive[..., t]
    out[..., :N] = torch.minimum(out[..., :N], vanilla[..., :N].to(f32))
    return out


def _heston_bank(prm, product, market, bank, n_paths, seed, device):
    S, v = _heston_paths(prm, market, n_paths, seed, device)
    T = torch.linspace(market.maturity, 0.0, market.num_steps + 1, dtype=torch.float64, device=device)
    K = torch.as_tensor(np.asarray(prm["K"], np.float64), device=device)
    call_K = heston_call(S, v, K, T, prm, market, bank)
    out = {"S": S, "T": T}
    if product == "call":
        out["V"] = call_K
    elif product == "doc":
        H = market.barrier_alpha * K
        out["V"] = heston_doc_mc(S, v, K, H, T, prm, market, bank, seed, vanilla=call_K)
        out["C"] = call_K
        out["Pt"] = heston_put(S, v, H ** 2 / K, T, prm, market, bank)
    elif product == "con":
        out["V"] = heston_digital(S, v, K, T, prm, market, bank, market.digital_payout)
        out["C"] = call_K
        out["Pt"] = heston_put(S, v, K, T, prm, market, bank, call=call_K)
    else:
        raise ValueError(product)
    return {k: x.detach().to("cpu", torch.float32).contiguous() for k, x in out.items()}


# ------------------------------------------------------------------- build / cache
def build_split(dynamics, product, split, market: MarketConfig, bank: BankConfig, device) -> dict:
    train_prm, test_prm = load_market_params(dynamics, market)
    prm = test_prm if split == "test" else train_prm
    n_paths = {"train": bank.train_paths, "val": bank.val_paths, "test": bank.test_paths}[split]
    seed = {"train": bank.seed_train, "val": bank.seed_val, "test": bank.seed_test}[split]
    t0 = time.time()
    if dynamics == "bsm":
        data = _bsm_bank(prm, product, market, n_paths, seed)
    else:
        data = _heston_bank(prm, product, market, bank, n_paths, seed, device)
    data["K"] = torch.as_tensor(np.asarray(prm["K"], np.float32))
    data["S0"] = torch.as_tensor(np.asarray(prm["S0"], np.float32))
    data["meta"] = {"dynamics": dynamics, "product": product, "split": split, "seed": seed,
                    "n_paths": n_paths, "market": asdict(market), "bank": asdict(bank),
                    "build_seconds": time.time() - t0}
    for name in ("V", "C", "Pt"):
        if name in data and not torch.isfinite(data[name]).all():
            raise FloatingPointError(f"non-finite {name} in {dynamics}/{product}/{split} bank")
    return data


def bank_path(bank_dir: Path, dynamics, product, split, market, bank) -> Path:
    return Path(bank_dir) / f"{dynamics}_{product}_{split}_{config_hash(market, bank)}.pt"


def _acquire_lock(lock: Path, target: Path, stale_seconds=6 * 3600) -> bool:
    """True if we own the lock; False if another job finished the file while we waited."""
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            if target.exists():
                return False
            try:
                if time.time() - lock.stat().st_mtime > stale_seconds:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            print(f"[bank] waiting for another job to finish {target.name}", flush=True)
            time.sleep(30)


def get_bank(bank_dir, dynamics, product, split, market, bank, device, build_if_missing=True) -> dict:
    path = bank_path(bank_dir, dynamics, product, split, market, bank)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock")
        if _acquire_lock(lock, path):
            try:
                if not path.exists():
                    print(f"[bank] building {path.name} on {device}", flush=True)
                    data = build_split(dynamics, product, split, market, bank, device)
                    tmp = path.with_suffix(".tmp")
                    torch.save(data, tmp)
                    os.replace(tmp, path)
                    print(f"[bank] saved {path} ({data['meta']['build_seconds']:.0f}s)", flush=True)
            finally:
                lock.unlink(missing_ok=True)
    return torch.load(path, map_location="cpu", weights_only=False)


class Bank:
    """A loaded split moved to the training device."""

    def __init__(self, data: dict, device):
        self.meta = data["meta"]
        self.S = data["S"].to(device)
        self.V = data["V"].to(device)
        self.C = data["C"].to(device) if "C" in data else None
        self.Pt = data["Pt"].to(device) if "Pt" in data else None
        self.K = data["K"].to(device)
        self.S0 = data["S0"].to(device)
        self.T = data["T"].to(device)
        self.num_paths, self.num_assets, n1 = self.S.shape
        self.num_strikes = self.K.shape[1]
        self.num_steps = n1 - 1
        self.has_options = self.C is not None
        self.action_dim = 2 if self.has_options else 1

    @property
    def options_per_path(self) -> int:
        return self.num_assets * self.num_strikes
