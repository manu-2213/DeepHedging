"""Parity checks: the fast engine must reproduce the original environments.

Runs the old Gym environments (hedging/envs.py) and the new vectorised code on
the SAME simulated paths and the SAME action sequence, and compares prices,
observations, rewards and terminal P&L.  Usage (from deep_hedging_v0/):
    python -m fast_hedging.tests.test_parity
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("DEEP_HEDGING_PATH_CACHE", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from hedging.envs import HedgeCallBS, HedgeDocBS, HedgeConcBS          # noqa: E402
from fast_hedging import pricing                                        # noqa: E402
from fast_hedging.features import build_observations                    # noqa: E402
from fast_hedging.market import Bank                                    # noqa: E402
from fast_hedging.ppo import Task, TrainConfig, rollout, Agent          # noqa: E402

S0 = np.array([100.0, 120.0]); K = np.array([[95.0, 105.0], [110.0, 130.0]]); SIG = np.array([0.2, 0.3])
R, T_MAT, N, P = 0.01, 1.0, 20, 3
UF, OF = 0.01, 0.05


def close(name, a, b, rtol=2e-4, atol=2e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.max(np.abs(a - b) / (atol + rtol * np.abs(b)))
    status = "OK " if err <= 1.0 else "FAIL"
    print(f"  [{status}] {name:28s} max scaled err {err:.3g}")
    return err <= 1.0


def run(product):
    print(f"== {product}")
    H = 0.9 * K
    if product == "call":
        env = HedgeCallBS(S0, K, T_MAT, R, SIG, P, N, transaction_cost=True, transaction_fee_rate=UF)
    elif product == "doc":
        env = HedgeDocBS(S0, K, H, T_MAT, R, SIG, P, N, history_len=1, transaction_cost=True,
                         transaction_fee_rate=OF / 2)        # old env charges 2x the rate on options
    else:
        env = HedgeConcBS(S0, K, np.ones((1, 1)), T_MAT, R, SIG, P, N, transaction_cost=True,
                          transaction_fee_rate=OF / 2)
    obs0, _ = env.reset(seed=3)
    S = env.simulator.paths.copy()
    Tg = env.simulator.T
    call = pricing.bs_euro(S, K, Tg, R, SIG, 1)
    if product == "call":
        liab, put = call, np.zeros_like(call)
        inst = np.broadcast_to(S[:, :, None, :, None], call.shape + (1,)).copy(); kind = ("underlying",)
    elif product == "doc":
        liab = pricing.bs_down_out_call(S, K, H, Tg, R, SIG); put = pricing.bs_euro(S, H**2 / K, Tg, R, SIG, -1)
        inst = np.stack([call, put], -1); kind = ("option", "option")
    else:
        liab = pricing.bs_cash_or_nothing(S, K, np.ones_like(K), Tg, R, SIG); put = pricing.bs_euro(S, K, Tg, R, SIG, -1)
        inst = np.stack([call, put], -1); kind = ("option", "option")
    ok = close("liability price", liab, env.option_prices)
    if product != "call":
        ok &= close("hedge call price", call, env.call_prices)
        ok &= close("hedge put price", put, env.put_prices)
    bank = Bank(S=S, liability=liab, instruments=inst, call=call, put=put, K=K.astype(np.float32),
                T=Tg.astype(np.float32), instrument_kind=kind)
    tt = lambda x: torch.as_tensor(x)
    obs_new = build_observations(tt(S), tt(liab), tt(call), tt(put), tt(K.astype(np.float32)), tt(Tg), R, product)
    # random but fixed action sequence
    rng = np.random.default_rng(0)
    A = 1 if product == "call" else 2
    lo = 0.0 if product == "call" else -1.0
    acts = rng.uniform(lo, 1.0, size=(N, env.num_envs, A)).astype(np.float32)
    old_obs, old_rew = [obs0[:, -1].copy()], []
    for t in range(N):
        o, rwd, *_ = env.step(acts[t])
        old_obs.append(o[:, -1].copy()); old_rew.append(rwd.copy())
    # The old code adds np.finfo(dtype).eps where dtype is accidentally float64 for some price
    # arrays, so ratio features differ (and explode) when a price is ~0. Compare elsewhere only.
    prices = [liab, call, put] if product != "call" else [liab]
    flat = lambda x: x.reshape(-1, N + 1)
    ok_pts = np.ones((env.num_envs, N + 1), bool)
    for pr in prices:
        prev = np.concatenate([flat(pr)[:, :1], flat(pr)[:, :-1]], 1)
        ok_pts &= (np.abs(flat(pr)) > 1e-4) & (np.abs(prev) > 1e-4)
    ok &= close("observations (prices>1e-4)", obs_new.numpy()[ok_pts], np.stack(old_obs, 1)[ok_pts],
                rtol=1e-3, atol=1e-3)
    old_pnl = (env.portfolio_value[..., -1] - env.option_prices[..., -1]).reshape(-1)

    # Drive the new engine with the same actions via a stub actor.
    cfg = TrainConfig(arch="mlp", net="single", agent_state=False, reward="tracking", underlying_fee=UF, option_fee=OF, reward_norm="none",
                      normalize_obs=False)
    task = Task(bank, product, R, cfg, "cpu")
    agent = Agent(cfg, product, "cpu")
    seq = iter(torch.as_tensor(acts))
    low, high = (0.0, 1.0) if product == "call" else (-1.0, 1.0)

    class Stub:
        def __init__(self, a): self.a = a
        def squash(self, u): return self.a
        def mode(self): return self.a
        def log_prob(self, u): return torch.zeros(u.shape[0])
        loc = scale = None
    agent.actor.heads = lambda feat, extra=None: (Stub(next(seq)), torch.zeros(feat.shape[0]))
    data = task.batch(torch.arange(P))
    rec, out = rollout(agent, task, data, use_inactor=False, actor_stochastic=False)
    ok &= close("rewards", rec["reward"].numpy().T, np.stack(old_rew), rtol=1e-3, atol=1e-3)
    ok &= close("terminal P&L", out["pnl"].numpy(), old_pnl, rtol=1e-3, atol=1e-3)
    return ok


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
    """Heston pricing checks.
    (1) The vectorised pricer against an independent complex128 reference (phi_max 4000, dphi 0.02) on the
        2021 S&P fit (rho -0.95), down to the last trading day and including v = 0, where the residual is
        bounded by S / (pi phi_cap), about 0.4 dollars on S = 3700.
    (2) Against the original per-point JAX code far from maturity only: the original used a fixed
        phi_max = 50, which is badly wrong close to maturity, so the two must NOT agree there.
    (3) Barrier nested MC against the original (loose: MC noise), and the pooled t = 0 premium."""
    from hedging.simulators import HestonSimulator
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
    S0h = np.array([100.0, 110.0]); Kh = np.array([[95.0, 105.0], [100.0, 120.0]])
    prm = dict(v0=np.array([0.04, 0.05]), theta=np.array([0.05, 0.04]), rho=np.array([-0.7, -0.5]),
               kappa=np.array([2.0, 3.0]), xi=np.array([0.4, 0.5]))
    sim = HestonSimulator(S0h, R, prm["v0"], prm["theta"], prm["rho"], prm["kappa"], prm["xi"], 1.0, 6, 2,
                          seed=1, n_points=400)
    old_call = np.asarray(sim.euro_call(Kh)); old_dig = np.asarray(sim.cash_or_nothing_call(Kh, 1.0))
    hp = {k: prm[k] for k in ("kappa", "theta", "rho", "xi")}
    call, dig = pricing.heston_vanilla_and_digital(np.asarray(sim.paths), np.asarray(sim.v), np.asarray(sim.T), hp,
                                                   R, Kh)
    far = slice(0, 4)                      # T = 1, 5/6, 4/6, 3/6: the old fixed limit is still accurate there
    ok &= close("heston call vs original (T>=0.5)", call[..., far], old_call[..., far], rtol=2e-3, atol=1e-2)
    ok &= close("heston digital vs original (T>=0.5)", dig[..., far], old_dig[..., far], rtol=2e-3, atol=1e-2)
    H = 0.9 * Kh
    old_doc = sim.down_out_call(Kh, H, mc_paths=20000, seed=5)
    new_doc = pricing.heston_down_out_call(np.asarray(sim.paths), np.asarray(sim.v), np.asarray(sim.T), hp, R, Kh, H,
                                           call, mc_paths=20000, seed=5)
    ok &= close("heston DOC (MC, loose)", new_doc, old_doc, rtol=3e-2, atol=0.15)
    ok &= close("heston DOC pooled premium (same on all paths)", new_doc[..., 0],
                np.broadcast_to(new_doc[:1, ..., 0], new_doc[..., 0].shape), rtol=0.0, atol=1e-6)
    return ok


if __name__ == "__main__":
    results = [run(p) for p in ("call", "doc", "con")] + [run_heston_pricing()]
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
