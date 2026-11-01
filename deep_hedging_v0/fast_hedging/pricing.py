"""Vectorised option pricing under Black-Scholes and Heston (numpy / torch).

The Black-Scholes formulas reproduce hedging/simulators.py exactly. The Heston pricer uses the
same characteristic function as the original JAX code but an adaptive integration limit
(see `heston_phi_max`): the original fixed limit phi_max = 50 truncated the Gil-Pelaez
integral far too early close to maturity (ATM call 6 instead of 15 dollars one day before
expiry on the 2021 S&P fit) and gave negative out-of-the-money prices that the intrinsic
clip hid. Every (path, asset, strike, time) point is priced in one batched call.
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


def bs_call_delta(paths, K, T_grid, r, sigma):
    """Black-Scholes call delta N(d1) on every point; terminal column = 1{S_N > K}."""
    S, Kx, T, sig = _expand(paths, K, T_grid, sigma)
    d1, _ = _bs_d1_d2(Kx, T[..., :-1], S[..., :-1], r, sig)
    last = (S[..., -1:] > Kx).astype(np.float64)
    return np.concatenate([norm.cdf(d1), last], axis=-1).astype(NP)


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
PHI_MIN, PHI_CAP, DPHI = 50.0, 3200.0, 0.05


def _heston_cf(phi, x, T, v, kappa, theta, rho, xi, r, u):
    """Same 'trap=1' formulation as HestonSimulator._heston_cf.

    phi: (F,) real; x, T, v, kappa, theta, rho, xi: (n, 1) real.
    Returns the characteristic function f (n, F) complex and D (n, F) with df/dv = D f.
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
    return torch.exp(C + D * v + iphi * x), D


def heston_phi_max(v, T, kappa, theta, rho, xi, *, phi_min=PHI_MIN, phi_cap=PHI_CAP,
                   c_gauss=30.0, c_exp=18.0):
    """Upper integration limit for every point, rounded up to phi_min * 2^k.

    |f(phi)| falls like exp(-var_int phi^2 / 2) for moderate phi, with var_int = E int_0^T v_s ds,
    but only like exp(-rate phi) for large phi, rate = sqrt(1 - rho^2) (v + kappa theta T) / xi
    (fat tails from the vol of vol). The integrand is negligible only once both exponents are
    large, so phi_max = max(sqrt(c_gauss / var_int), c_exp / rate), clipped to [phi_min, phi_cap].
    With v = 0 and T of a few days both criteria exceed the cap; there the option is almost
    intrinsic and the residual error is about S / (pi phi_cap) (0.4 dollars on S = 3700).
    Validated against a complex128 reference with phi_max = 4000 on all five S&P fits:
    errors below 0.002 dollars for T >= 25 days, below 0.5 dollars on the last days.
    """
    v = np.maximum(np.asarray(v, np.float64), 0.0)
    T = np.asarray(T, np.float64)
    kappa, theta, rho, xi = (np.asarray(p, np.float64) for p in (kappa, theta, rho, xi))
    ek = (1.0 - np.exp(-kappa * T)) / kappa
    var_int = v * ek + theta * (T - ek)
    phi_g = np.sqrt(c_gauss / np.maximum(var_int, 1e-12))
    rate = np.sqrt(1.0 - rho**2) * (v + kappa * theta * T) / xi
    phi_e = c_exp / np.maximum(rate, 1e-12)
    need = np.clip(np.maximum(phi_g, phi_e), phi_min, phi_cap)
    k = np.ceil(np.log2(need / phi_min) - 1e-9)
    return np.minimum(phi_min * 2.0**k, phi_cap)


def heston_probabilities(S, v, T, params, strikes, r, *, device="cpu", dphi=DPHI, phi_min=PHI_MIN,
                         phi_cap=PHI_CAP, return_dv=False, budget=2**25):
    """Heston P1/P2 (and with return_dv the derivatives dP1/dv, dP2/dv) for every point and strike.

    S, v, T: (n,) arrays (one entry per (path, asset, time) point); params: dict of (n,) arrays
    kappa/theta/rho/xi; strikes: (n, M). Returns numpy float64 arrays of shape (n, M).
    Points are grouped by their integration limit (`heston_phi_max`), the grid spacing is dphi
    everywhere, and the chunk size keeps the (chunk, M, F) integrand under `budget` elements.
    complex64 is used throughout: on the production grid it agrees with complex128 to 1e-6.
    The characteristic function does not depend on the strike, so it is evaluated once per
    point and reused for all M strikes.
    """
    f32 = torch.float32
    S, v, T = (np.asarray(a, np.float64) for a in (S, v, T))
    n, M = strikes.shape
    P1, P2 = np.empty((n, M)), np.empty((n, M))
    dP1, dP2 = (np.empty((n, M)), np.empty((n, M))) if return_dv else (None, None)
    pm_all = heston_phi_max(v, T, params["kappa"], params["theta"], params["rho"], params["xi"],
                            phi_min=phi_min, phi_cap=phi_cap)
    for pm in np.unique(pm_all):
        sel = np.flatnonzero(pm_all == pm)
        F = int(round(pm / dphi)) + 1
        phi = torch.linspace(1e-8, float(pm), F, dtype=f32, device=device)
        dx = float(phi[1] - phi[0])
        chunk = max(1, int(budget // (F * M)))
        for lo in range(0, sel.size, chunk):
            idx = sel[lo:lo + chunk]
            col = lambda a: torch.as_tensor(np.asarray(a[idx], np.float32), device=device)[:, None]
            x, TT, vv = torch.log(col(S)), col(T), torch.clamp(col(v), min=0.0)
            k, th, rh, xi = col(params["kappa"]), col(params["theta"]), col(params["rho"]), col(params["xi"])
            logK = torch.log(torch.as_tensor(np.asarray(strikes[idx], np.float32), device=device))
            ph = torch.exp(-1j * phi[None, None, :] * logK[:, :, None])             # (c, M, F)
            for u, out, dout in ((0.5, P1, dP1), (-0.5, P2, dP2)):
                f, D = _heston_cf(phi, x, TT, vv, k, th, rh, xi, r, u)
                cf = f / (1j * phi)                                                 # (c, F)
                integral = torch.trapezoid(torch.real(ph * cf[:, None, :]), dx=dx, dim=-1)
                out[idx] = (0.5 + integral / math.pi).double().cpu().numpy()
                if return_dv:
                    dint = torch.trapezoid(torch.real(ph * (D * cf)[:, None, :]), dx=dx, dim=-1)
                    dout[idx] = (dint / math.pi).double().cpu().numpy()
    return (P1, P2, dP1, dP2) if return_dv else (P1, P2)


def heston_vanilla_and_digital(S_paths, V_paths, T_grid, params_asset, r, strikes, *,
                               device="cpu", return_delta=False, **kw):
    """Calls and cash-or-nothing (Q=1) prices for an arbitrary strike matrix.

    S_paths, V_paths: (P, A, N+1); strikes: (A, M). Returns calls, digitals of
    shape (P, A, M, N+1), terminal column = payoff, same clipping as HestonSimulator.
    return_delta=True also returns the Heston call delta dC/dS = P1 and the minimum-variance
    delta dC/dS + (rho xi / S) dC/dv, which also hedges the vol move correlated with the spot
    (terminal column 1{S_N > K} for both). Extra kw go to `heston_probabilities`.
    """
    P, A, N1 = S_paths.shape
    N = N1 - 1
    M = strikes.shape[1]
    S = S_paths[:, :, :N]; V = V_paths[:, :, :N]
    T = np.broadcast_to(np.asarray(T_grid[:N], np.float64)[None, None, :], (P, A, N))
    rep = lambda p: np.broadcast_to(np.asarray(p, np.float64)[None, :, None], (P, A, N)).reshape(-1)
    params = {k: rep(params_asset[k]) for k in ("kappa", "theta", "rho", "xi")}
    K_pts = np.broadcast_to(np.asarray(strikes, np.float64)[None, :, None, :], (P, A, N, M)).reshape(-1, M)
    res = heston_probabilities(S.reshape(-1).astype(np.float64), V.reshape(-1), T.reshape(-1), params,
                               K_pts, r, device=device, return_dv=return_delta, **kw)
    shp = lambda a: a.reshape(P, A, N, M).transpose(0, 1, 3, 2)
    P1, P2 = shp(res[0]), shp(res[1])
    S4 = S_paths[:, :, None, :].astype(np.float64)
    K4 = np.asarray(strikes, np.float64)[None, :, :, None]
    T4 = np.asarray(T_grid, np.float64)[None, None, None, :]
    disc = np.exp(-r * T4)
    call = np.concatenate([S4[..., :N] * P1 - K4 * disc[..., :N] * P2,
                           np.maximum(0.0, S4[..., N:] - K4)], axis=-1)
    call = np.clip(call, np.maximum(S4 - K4 * disc, 0.0), np.broadcast_to(S4, call.shape))
    dig = np.concatenate([disc[..., :N] * P2, (S4[..., N:] > K4).astype(np.float64)], axis=-1)
    dig = np.clip(dig, 0.0, np.broadcast_to(disc, dig.shape))
    if return_delta:
        term = (S4[..., N:] > K4).astype(np.float64)
        delta = np.clip(np.concatenate([P1, term], axis=-1), 0.0, 1.0)
        dP1, dP2 = shp(res[2]), shp(res[3])
        dC_dv = S4[..., :N] * dP1 - K4 * disc[..., :N] * dP2
        rx = (np.asarray(params_asset["rho"], np.float64) * np.asarray(params_asset["xi"], np.float64))[None, :, None, None]
        delta_mv = np.clip(np.concatenate([P1 + rx * dC_dv / S4[..., :N], term], axis=-1), 0.0, 1.0)
        return call.astype(NP), dig.astype(NP), delta.astype(NP), delta_mv.astype(NP)
    return call.astype(NP), dig.astype(NP)


def heston_put_from_call(call, S_paths, strikes, T_grid, r):
    S4 = S_paths[:, :, None, :].astype(np.float64)
    K4 = np.asarray(strikes, np.float64)[None, :, :, None]
    T4 = np.asarray(T_grid, np.float64)[None, None, None, :]
    return (call - S4 + K4 * np.exp(-r * T4)).astype(NP)


def heston_down_out_call(S_paths, V_paths, T_grid, params_asset, r, K, H, vanilla, *,
                         mc_paths=2000, seed=0, device="cpu", budget=2**26):
    """Nested-MC down-and-out call, same estimator as HestonSimulator.down_out_call.

    Speed-ups versus the original: (1) all start times are simulated in one
    pass (inner paths started at t only move once the global clock passes t),
    (2) one inner simulation serves every strike via the running minimum,
    (3) everything runs as batched torch ops on `device`.
    Inner paths that start at different times share the same random draws at each
    clock step (common random numbers). Each estimate is still unbiased; the error
    is correlated across t, which makes the price path smoother than independent MC.
    The normal draws come from a CPU generator, so the bank does not depend on the device.
    At t = 0 every outer path starts from the same (S0, v0), so the P independent estimates
    are pooled: the premium is one number per strike (P x mc_paths inner paths), not a
    different noisy number per path.
    """
    P, A, N1 = S_paths.shape
    N = N1 - 1
    M = K.shape[1]
    dt = float(T_grid[0] - T_grid[1])
    gen = torch.Generator().manual_seed(int(seed))
    out = np.zeros((P, A, M, N1), dtype=np.float64)
    Kt = torch.as_tensor(np.asarray(K, np.float64), device=device)
    Ht = torch.as_tensor(np.asarray(H, np.float64), device=device)
    T_rem = torch.as_tensor(np.asarray(T_grid[:N], np.float64), device=device)
    ftype = torch.float32
    outer_chunk = max(8, int(budget // (mc_paths * N)))
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
            z = torch.randn(N, 2, c, mc_paths, 1, generator=gen, dtype=ftype).to(device)
            for s_ in range(1, N + 1):
                # at clock s_ only inner paths started at t < s_ move: update the slice [:s_]
                z_s = z[s_ - 1, 0]
                z_v = rh * z_s + math.sqrt(1.0 - rh**2) * z[s_ - 1, 1]
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
            del S, v, run_min, z, S64, mn
    out[..., 0] = out[..., 0].mean(axis=0, keepdims=True)      # pooled premium, see docstring
    S4 = S_paths[:, :, None, :]
    alive = np.minimum.accumulate(S4 > np.asarray(H)[None, :, :, None], axis=-1)
    out[..., N] = np.maximum(S4[..., N] - np.asarray(K)[None, :, :], 0.0)
    out = np.minimum(out, vanilla) * alive
    return out.astype(NP)
