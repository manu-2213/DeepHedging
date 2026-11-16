"""Pricing checks that need nothing but the engine.

  1. Black-Scholes closed forms on simulated paths: put-call parity, the digital as the strike derivative of the call,
     the down-and-out call bounded by the vanilla, worthless after a barrier hit and equal to the payoff at maturity.
  2. The vectorised Heston pricer against an independent complex128 Gil-Pelaez reference (phi_max 4000, dphi 0.02) on
     the 2021 S&P fit (rho -0.95), down to the last trading day and including v = 0, where the residual is bounded by
     S / (pi phi_cap), about 0.4 dollars on S = 3700.
  3. The Heston barrier nested Monte Carlo: the pooled t = 0 premium is the same number on every path, and the price
     never exceeds the vanilla call by more than Monte Carlo noise.

Until 2026-11 this file also compared prices, observations, rewards and P&L with the original Gym environments. That
comparison needs the old code; it is in the git history (tag pre-cleanup-2026-11-16, fast_hedging/tests/test_parity.py).

Usage (from the repository root):   python -m tests.test_pricing
"""
import sys

import numpy as np

from hedging_engine import pricing
from hedging_engine.market import MarketSpec, _simulate_bsm, _simulate_heston

R = 0.01


def close(name, a, b, rtol=2e-4, atol=2e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b) / (atol + rtol * np.abs(b)))
    status = "OK " if err <= 1.0 else "FAIL"
    print(f"  [{status}] {name:28s} max scaled err {err:.3g}")
    return err <= 1.0


def run_bsm_pricing():
    print("== black-scholes pricing")
    S0 = np.array([100.0, 120.0]); K = np.array([[95.0, 105.0], [110.0, 130.0]]); sig = np.array([0.2, 0.3])
    spec = MarketSpec(num_steps=20, r=R)
    S, _ = _simulate_bsm({"S0": S0, "sigma": sig}, 200, spec, np.random.default_rng(3))
    Tg = np.linspace(spec.maturity, 0.0, spec.num_steps + 1)
    S4, K4, T4 = S[:, :, None, :].astype(np.float64), K[None, :, :, None], Tg[None, None, None, :]
    call, put = pricing.bs_euro(S, K, Tg, R, sig, 1), pricing.bs_euro(S, K, Tg, R, sig, -1)
    ok = close("put-call parity", call - put, S4 - K4 * np.exp(-R * T4), rtol=1e-4, atol=2e-3)
    h = 0.25
    fd = (pricing.bs_euro(S, K - h, Tg, R, sig, 1).astype(np.float64) - pricing.bs_euro(S, K + h, Tg, R, sig, 1)) / (2 * h)
    dig = pricing.bs_cash_or_nothing(S, K, np.ones_like(K), Tg, R, sig)
    ok &= close("digital = -dC/dK (before maturity)", dig[..., :-1], fd[..., :-1], rtol=1e-2, atol=1e-2)
    H = 0.9 * K
    doc = pricing.bs_down_out_call(S, K, H, Tg, R, sig)
    alive = np.cumprod(S4 > H[None, :, :, None], axis=-1).astype(bool)
    ok &= close("down-and-out <= vanilla", np.maximum(doc - call, 0.0), np.zeros_like(doc), rtol=0.0, atol=1e-3)
    ok &= close("down-and-out >= 0", np.minimum(doc, 0.0), np.zeros_like(doc), rtol=0.0, atol=1e-3)
    ok &= close("down-and-out is 0 after a hit", doc[~alive], np.zeros((~alive).sum()), rtol=0.0, atol=1e-6)
    ok &= close("down-and-out payoff at maturity", doc[..., -1], (np.maximum(S4 - K4, 0.0) * alive)[..., -1], rtol=1e-5, atol=1e-4)
    return bool(ok)


def _reference_heston_call(S, K, T, v, p, r, phi_max=4000.0, dphi=0.02):
    """Independent complex128 Gil-Pelaez price on a long, fine grid (numpy only)."""
    import math
    trap = getattr(np, "trapezoid", None) or np.trapz
    phi = np.linspace(1e-8, phi_max, int(round(phi_max / dphi)) + 1).astype(np.complex128)
    rho, xi, kappa, theta = p["rho"], p["xi"], p["kappa"], p["theta"]
    out = []
    for u in (0.5, -0.5):
        iphi = 1j * phi
        b = kappa - rho * xi if u == 0.5 else kappa
        d = np.sqrt((rho * xi * iphi - b) ** 2 - xi**2 * (2 * u * iphi - phi**2))
        g = (b - rho * xi * iphi + d) / (b - rho * xi * iphi - d)
        c = 1.0 / g
        e = np.exp(-d * T)
        D = ((b - rho * xi * iphi - d) / xi**2) * ((1 - e) / (1 - c * e))
        G = (1 - c * e) / (1 - c)
        C = r * iphi * T + (kappa * theta / xi**2) * ((b - rho * xi * iphi - d) * T - 2.0 * np.log(G))
        f = np.exp(C + D * v + iphi * math.log(S)) / iphi
        integ = np.real(np.exp(-iphi * math.log(K)) * f)
        out.append(0.5 + trap(integ, dx=float(np.real(phi[1] - phi[0]))) / math.pi)
    return S * out[0] - K * math.exp(-r * T) * out[1]


def run_heston_pricing():
    print("== heston pricing")
    ok = True
    p21 = dict(kappa=0.9552, theta=0.0819, rho=-0.95, xi=0.5006)
    hp21 = {k: np.array([val]) for k, val in p21.items()}
    S0 = 3700.0
    Ks = np.array([[3330.0, 3589.0, 3700.0, 3811.0, 4070.0]])
    Tg = np.array([1.0, 0.25, 0.04, 0.02, 1 / 250, 0.0])
    n = len(Tg) - 1
    for v, atol in ((0.0271, 0.1), (0.1, 0.1), (0.0, 0.75)):
        Sp, Vp = np.full((1, 1, n + 1), S0), np.full((1, 1, n + 1), v)
        call, _ = pricing.heston_vanilla_and_digital(Sp, Vp, Tg, hp21, R, Ks)
        ref = np.array([[_reference_heston_call(S0, K, T, v, p21, R) for T in Tg[:n]] for K in Ks[0]])
        ok &= close(f"heston call vs reference (v={v})", call[0, 0, :, :n], ref, rtol=1e-3, atol=atol)
    # barrier: nested Monte Carlo on paths from the engine's own simulator
    prm = dict(S0=np.array([100.0, 110.0]), v0=np.array([0.04, 0.05]), theta=np.array([0.05, 0.04]), rho=np.array([-0.7, -0.5]),
               kappa=np.array([2.0, 3.0]), xi=np.array([0.4, 0.5]))
    Kh = np.array([[95.0, 105.0], [100.0, 120.0]]); H = 0.9 * Kh
    spec = MarketSpec(dynamics="heston", num_steps=6, r=R)
    S, V = _simulate_heston(prm, 4, spec, np.random.default_rng(1))
    Tg = np.linspace(spec.maturity, 0.0, spec.num_steps + 1)
    hp = {k: prm[k] for k in ("kappa", "theta", "rho", "xi")}
    call, _ = pricing.heston_vanilla_and_digital(S, V, Tg, hp, R, Kh)
    doc = pricing.heston_down_out_call(S, V, Tg, hp, R, Kh, H, call, mc_paths=4000, seed=5)
    ok &= close("heston DOC pooled premium (same on all paths)", doc[..., 0],
                np.broadcast_to(doc[:1, ..., 0], doc[..., 0].shape), rtol=0.0, atol=1e-6)
    ok &= close("heston DOC <= vanilla (MC, loose)", np.maximum(doc - call, 0.0), np.zeros_like(doc), rtol=0.0, atol=0.5)
    return bool(ok)


if __name__ == "__main__":
    results = [run_bsm_pricing(), run_heston_pricing()]
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
