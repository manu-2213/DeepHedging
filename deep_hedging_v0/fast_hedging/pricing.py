"""Vectorised option pricing under Black-Scholes and Heston (numpy / torch).

The formulas reproduce hedging/simulators.py exactly, but evaluate every
(path, asset, strike, time) point in one batched call instead of Python loops.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

NP = np.float32


# --------------------------------------------------------------------------
# Black-Scholes (closed form, numpy; identical maths to BlackScholesSimulator)
# --------------------------------------------------------------------------
def _bs_d1_d2(K, T, S, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def _bs_price(K, T, S, r, sigma, flag):
    d1, d2 = _bs_d1_d2(K, T, S, r, sigma)
    return flag * S * norm.cdf(flag * d1) - flag * K * np.exp(-r * T) * norm.cdf(flag * d2)


def _expand(paths, K, T_grid, sigma):
    S = paths[:, :, None, :].astype(np.float64)          # (P, A, 1, N+1)
    Kx = np.asarray(K, np.float64)[None, :, :, None]       # (1, A, M, 1)
    T = np.asarray(T_grid, np.float64)[None, None, None, :]
    sig = np.asarray(sigma, np.float64)[None, :, None, None]
    return S, Kx, T, sig


def bs_euro(paths, K, T_grid, r, sigma, flag):
    S, Kx, T, sig = _expand(paths, K, T_grid, sigma)
    body = _bs_price(Kx, T[..., :-1], S[..., :-1], r, sig, flag)
    last = np.maximum(0.0, flag * (S - Kx))[..., -1:]
    return np.concatenate([body, last], axis=-1).astype(NP)


def bs_down_out_call(paths, K, H, T_grid, r, sigma):
    S, Kx, T, sig = _expand(paths, K, T_grid, sigma)
    Hx = np.asarray(H, np.float64)[None, :, :, None]
    vanilla = bs_euro(paths, K, T_grid, r, sigma, 1).astype(np.float64)
    alpha = 0.5 * (1 - (r / (0.5 * sig**2)))
    S_new = Hx**2 / S
    body = _bs_price(Kx, T[..., :-1], S_new[..., :-1], r, sig, 1)
    last = np.maximum(0.0, S_new - Kx)[..., -1:]
    term2 = (S / Hx) ** (2 * alpha) * np.concatenate([body, last], axis=-1)
    alive = np.cumprod(S > Hx, axis=-1)
    return ((vanilla - term2) * alive).astype(NP)


def bs_cash_or_nothing(paths, K, Q, T_grid, r, sigma):
    S, Kx, T, sig = _expand(paths, K, T_grid, sigma)
    Qx = np.broadcast_to(np.asarray(Q, np.float64), np.asarray(K).shape)[None, :, :, None]
    _, d2 = _bs_d1_d2(Kx, T[..., :-1], S[..., :-1], r, sig)
    body = np.exp(-r * T[..., :-1]) * norm.cdf(d2) * Qx
    last = ((S - Kx)[..., -1:] > 0) * Qx
    return np.concatenate([body, last], axis=-1).astype(NP)


# --------------------------------------------------------------------------
# Heston: characteristic-function pricing, batched in torch
# --------------------------------------------------------------------------
def _heston_cf(phi, x, T, v, kappa, theta, rho, xi, r, u):
    """Same 'trap=1' formulation as HestonSimulator._heston_cf.

    phi: (F,) real; x, T, v, kappa, theta, rho, xi: (n, 1) real. Returns (n, F) complex.
    """
    iphi = 1j * phi
    b = kappa - rho * xi if u == 0.5 else kappa
    a = kappa * theta
    d = torch.sqrt((rho * xi * iphi - b) ** 2 - xi**2 * (2 * u * iphi - phi**2))
    g = (b - rho * xi * iphi + d) / (b - rho * xi * iphi - d)
    c = 1.0 / g
    e = torch.exp(-d * T)
    D = ((b - rho * xi * iphi - d) / xi**2) * ((1 - e) / (1 - c * e))
    G = (1 - c * e) / (1 - c)
    C = r * iphi * T + (a / xi**2) * ((b - rho * xi * iphi - d) * T - 2.0 * torch.log(G))
    return torch.exp(C + D * v + iphi * x)


def heston_probabilities(S, v, T, params, strikes, r, *, n_points=1000, phi_max=50.0,
                         device="cpu", chunk=4096):
    """Heston P1/P2 for every point and every strike.

    S, v, T: (n,) arrays (one entry per (path, asset, time) point) and
    params: dict of (n,) arrays kappa/theta/rho/xi. strikes: (n, M).
    Returns P1, P2 as numpy float64 arrays of shape (n, M).
    Computed in complex64, the precision of the original JAX code (x64 disabled):
    much faster than complex128 on GPUs with weak float64 throughput.
    The characteristic function does not depend on the strike, so it is
    evaluated once per point and reused for all M strikes.
    """
    f32 = torch.float32
    phi = torch.linspace(1e-8, phi_max, n_points, dtype=f32, device=device)
    dx = float(phi[1] - phi[0])
    n, M = strikes.shape
    P1 = np.empty((n, M)); P2 = np.empty((n, M))
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        col = lambda a: torch.as_tensor(np.asarray(a[lo:hi], np.float32), device=device)[:, None]
        x, TT, vv = torch.log(col(S)), col(T), torch.clamp(col(v), min=0.0)
        k, th, rh, xi = col(params["kappa"]), col(params["theta"]), col(params["rho"]), col(params["xi"])
        logK = torch.log(torch.as_tensor(np.asarray(strikes[lo:hi], np.float32), device=device))
        for u, out in ((0.5, P1), (-0.5, P2)):
            cf = _heston_cf(phi, x, TT, vv, k, th, rh, xi, r, u) / (1j * phi)   # (c, F)
            # integrand for each strike: Re(exp(-i phi lnK) cf / (i phi))
            ph = torch.exp(-1j * phi[None, None, :] * logK[:, :, None])         # (c, M, F)
            integrand = torch.real(ph * cf[:, None, :])
            integral = torch.trapezoid(integrand, dx=dx, dim=-1)
            out[lo:hi] = (0.5 + integral / math.pi).double().cpu().numpy()
    return P1, P2


def heston_vanilla_and_digital(S_paths, V_paths, T_grid, params_asset, r, strikes, *,
                               n_points=1000, phi_max=50.0, device="cpu"):
    """Calls and cash-or-nothing (Q=1) prices for an arbitrary strike matrix.

    S_paths, V_paths: (P, A, N+1); strikes: (A, M). Returns calls, digitals of
    shape (P, A, M, N+1), terminal column = payoff, same clipping as HestonSimulator.
    """
    P, A, N1 = S_paths.shape
    N = N1 - 1
    M = strikes.shape[1]
    S = S_paths[:, :, :N]; V = V_paths[:, :, :N]
    T = np.broadcast_to(np.asarray(T_grid[:N], np.float64)[None, None, :], (P, A, N))
    rep = lambda p: np.broadcast_to(np.asarray(p, np.float64)[None, :, None], (P, A, N)).reshape(-1)
    params = {k: rep(params_asset[k]) for k in ("kappa", "theta", "rho", "xi")}
    K_pts = np.broadcast_to(np.asarray(strikes, np.float64)[None, :, None, :], (P, A, N, M)).reshape(-1, M)
    P1, P2 = heston_probabilities(S.reshape(-1).astype(np.float64), V.reshape(-1), T.reshape(-1), params,
                                  K_pts, r, n_points=n_points, phi_max=phi_max, device=device)
    P1 = P1.reshape(P, A, N, M).transpose(0, 1, 3, 2); P2 = P2.reshape(P, A, N, M).transpose(0, 1, 3, 2)
    S4 = S_paths[:, :, None, :].astype(np.float64)
    K4 = np.asarray(strikes, np.float64)[None, :, :, None]
    T4 = np.asarray(T_grid, np.float64)[None, None, None, :]
    disc = np.exp(-r * T4)
    call = np.concatenate([S4[..., :N] * P1 - K4 * disc[..., :N] * P2,
                           np.maximum(0.0, S4[..., N:] - K4)], axis=-1)
    call = np.clip(call, np.maximum(S4 - K4 * disc, 0.0), np.broadcast_to(S4, call.shape))
    dig = np.concatenate([disc[..., :N] * P2, (S4[..., N:] > K4).astype(np.float64)], axis=-1)
    dig = np.clip(dig, 0.0, np.broadcast_to(disc, dig.shape))
    return call.astype(NP), dig.astype(NP)


def heston_put_from_call(call, S_paths, strikes, T_grid, r):
    S4 = S_paths[:, :, None, :].astype(np.float64)
    K4 = np.asarray(strikes, np.float64)[None, :, :, None]
    T4 = np.asarray(T_grid, np.float64)[None, None, None, :]
    return (call - S4 + K4 * np.exp(-r * T4)).astype(NP)


def heston_down_out_call(S_paths, V_paths, T_grid, params_asset, r, K, H, vanilla, *,
                         mc_paths=500, seed=0, device="cpu", outer_chunk=256):
    """Nested-MC down-and-out call, same estimator as HestonSimulator.down_out_call.

    Speed-ups versus the original: (1) all start times are simulated in one
    pass (inner paths started at t only move once the global clock passes t),
    (2) one inner simulation serves every strike via the running minimum,
    (3) everything runs as batched torch ops on `device`.
    Inner paths that start at different times share the same random draws at each
    clock step (common random numbers). Each estimate is still unbiased; the error
    is correlated across t, which makes the price path smoother than independent MC.
    """
    P, A, N1 = S_paths.shape
    N = N1 - 1
    M = K.shape[1]
    dt = float(T_grid[0] - T_grid[1])
    gen = torch.Generator(device=device).manual_seed(int(seed))
    out = np.zeros((P, A, M, N1), dtype=np.float64)
    Kt = torch.as_tensor(np.asarray(K, np.float64), device=device)
    Ht = torch.as_tensor(np.asarray(H, np.float64), device=device)
    T_rem = torch.as_tensor(np.asarray(T_grid[:N], np.float64), device=device)
    ftype = torch.float32
    for a in range(A):
        kap, th, rh, xi = (float(params_asset[k][a]) for k in ("kappa", "theta", "rho", "xi"))
        for lo in range(0, P, outer_chunk):
            hi = min(P, lo + outer_chunk)
            S0 = torch.as_tensor(S_paths[lo:hi, a, :N], dtype=ftype, device=device)       # (c, N)
            v0 = torch.as_tensor(V_paths[lo:hi, a, :N], dtype=ftype, device=device)
            c = hi - lo
            S = S0[:, None, :].expand(c, mc_paths, N).clone()
            v = torch.clamp(v0, min=0.0)[:, None, :].expand(c, mc_paths, N).clone()
            run_min = S.clone()
            for s_ in range(1, N + 1):
                # at clock s_ only inner paths started at t < s_ move: update the slice [:s_]
                z_s = torch.randn(c, mc_paths, 1, generator=gen, device=device, dtype=ftype)
                z_v = torch.randn(c, mc_paths, 1, generator=gen, device=device, dtype=ftype)
                z_v = rh * z_s + math.sqrt(1.0 - rh**2) * z_v
                v_pos = torch.clamp(v[..., :s_], min=0.0)
                sq = torch.sqrt(v_pos * dt)
                S[..., :s_] *= torch.exp((r - 0.5 * v_pos) * dt + sq * z_s)
                v[..., :s_] += kap * (th - v_pos) * dt + xi * sq * z_v
                run_min[..., :s_] = torch.minimum(run_min[..., :s_], S[..., :s_])
            S64, mn = S.double(), run_min.double()
            for m in range(M):
                pay = torch.clamp(S64 - Kt[a, m], min=0.0) * (mn > Ht[a, m])
                est = torch.exp(-r * T_rem)[None, :] * pay.mean(dim=1)                    # (c, N)
                out[lo:hi, a, m, :N] = est.cpu().numpy()
    S4 = S_paths[:, :, None, :]
    alive = np.minimum.accumulate(S4 > np.asarray(H)[None, :, :, None], axis=-1)
    out[..., N] = np.maximum(S4[..., N] - np.asarray(K)[None, :, :], 0.0)
    out = np.minimum(out, vanilla) * alive
    return out.astype(NP)
