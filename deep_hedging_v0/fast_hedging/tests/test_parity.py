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
    cfg = TrainConfig(arch="mlp", net="single", agent_state=False, underlying_fee=UF, option_fee=OF, reward_norm="none",
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


def run_heston_pricing():
    """New vectorised Heston call vs the original per-point JAX implementation (tiny grid)."""
    from hedging.simulators import HestonSimulator
    print("== heston pricing")
    S0h = np.array([100.0, 110.0]); Kh = np.array([[95.0, 105.0], [100.0, 120.0]])
    prm = dict(v0=np.array([0.04, 0.05]), theta=np.array([0.05, 0.04]), rho=np.array([-0.7, -0.5]),
               kappa=np.array([2.0, 3.0]), xi=np.array([0.4, 0.5]))
    sim = HestonSimulator(S0h, R, prm["v0"], prm["theta"], prm["rho"], prm["kappa"], prm["xi"], 1.0, 6, 2,
                          seed=1, n_points=400)
    old_call = sim.euro_call(Kh); old_dig = sim.cash_or_nothing_call(Kh, 1.0)
    hp = {k: prm[k] for k in ("kappa", "theta", "rho", "xi")}
    call, dig = pricing.heston_vanilla_and_digital(np.asarray(sim.paths), np.asarray(sim.v), np.asarray(sim.T), hp,
                                                   R, Kh, n_points=400)
    ok = close("heston call", call, old_call, rtol=2e-3, atol=2e-3)
    ok &= close("heston digital", dig, old_dig, rtol=2e-3, atol=2e-3)
    H = 0.9 * Kh
    old_doc = sim.down_out_call(Kh, H, mc_paths=20000, seed=5)
    new_doc = pricing.heston_down_out_call(np.asarray(sim.paths), np.asarray(sim.v), np.asarray(sim.T), hp, R, Kh, H,
                                           call, mc_paths=20000, seed=5)
    ok &= close("heston DOC (MC, loose)", new_doc, old_doc, rtol=3e-2, atol=0.15)
    return ok


if __name__ == "__main__":
    results = [run(p) for p in ("call", "doc", "con")] + [run_heston_pricing()]
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
