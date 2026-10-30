"""Vectorised GPU hedging environment.

Reproduces hedging/env_hedge_{call,doc,conc}.py exactly, with two intended changes:
  * costs are 1% on the underlying and 5% on option premium (old: rate and 2 x rate);
  * every env runs on the GPU in lock-step, with no Gym / TorchRL / NumPy round trips.

Per step t = 0..N-1 for each option e (flattened over path, asset, strike):
    trade   = a_t - h_t
    cash_t -= trade . X_t + sum(fee * |trade * X_t|)
    h_{t+1} = a_t ;  cash_{t+1} = cash_t * exp(r dt)
    PV_{t+1} = cash_{t+1} + h_{t+1} . X_{t+1}
    reward   = -|PV_{t+1} - V_{t+1}|
with cash_0 = V_0 (premium received). Terminal P&L = PV_N - V_N.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from experiments.fast.path_bank import Bank

EPS = float(torch.finfo(torch.float32).eps)
LOG_FLOOR = 1e-8
BASE_FEATURES = 11
OPTION_FEATURES = 6
TRADE_EPS = 1e-6


def _safe_log(x):
    return torch.log(torch.clamp(x, min=LOG_FLOOR))


def _changes(x):
    """(diff, ratio, log ratio) against the previous step; t=0 gives (0, 1, 0) as in feature_extractor."""
    prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    ratio = (x + EPS) / (prev + EPS)
    return x - prev, ratio, _safe_log(ratio)


def static_features(S, V, K, T, r, C=None, Pt=None):
    """All market features for every step at once. Same order as create_observation_hedge_*."""
    dS, rS, lS = _changes(S)
    dV, rV, lV = _changes(V)
    est_delta = torch.clamp(dV / (dS + EPS), 0.0, 1.0)
    moneyness = K[:, None] / S
    feats = [dS, rS, lS, dV, rV, lV, est_delta, moneyness, _safe_log(moneyness),
             T[None, :].expand_as(S), torch.full_like(S, r)]
    if C is not None:
        dC, rC, lC = _changes(C)
        dP, rP, lP = _changes(Pt)
        feats += [dC, dP, rC, rP, lC, lP]
    return torch.stack(feats, dim=-1)  # (E, N+1, F)


@dataclass
class HedgeBatch:
    feats: torch.Tensor   # (E, N+1, F)
    S: torch.Tensor       # (E, N+1) underlying
    V: torch.Tensor       # (E, N+1)
    X: torch.Tensor       # (E, N+1, D) hedge instrument prices
    fee: torch.Tensor     # (D,)
    num_steps: int

    @property
    def num_envs(self):
        return self.V.shape[0]

    @property
    def action_dim(self):
        return self.X.shape[-1]


def make_batch(bank: Bank, path_idx: torch.Tensor, r: float, underlying_fee: float, option_fee: float) -> HedgeBatch:
    S = bank.S[path_idx]                                  # (P, A, N+1)
    P, A, N1 = S.shape
    M = bank.num_strikes
    S_e = S[:, :, None, :].expand(P, A, M, N1).reshape(-1, N1)
    V_e = bank.V[path_idx].reshape(-1, N1)
    K_e = bank.K[None].expand(P, A, M).reshape(-1)
    dev = S.device
    if bank.has_options:
        C_e = bank.C[path_idx].reshape(-1, N1)
        P_e = bank.Pt[path_idx].reshape(-1, N1)
        feats = static_features(S_e, V_e, K_e, bank.T, r, C_e, P_e)
        X = torch.stack([C_e, P_e], dim=-1)
        fee = torch.tensor([option_fee, option_fee], device=dev)
    else:
        feats = static_features(S_e, V_e, K_e, bank.T, r)
        X = S_e[..., None]
        fee = torch.tensor([underlying_fee], device=dev)
    return HedgeBatch(feats=feats, S=S_e, V=V_e, X=X, fee=fee, num_steps=N1 - 1)


class HedgeState:
    def __init__(self, batch: HedgeBatch, r: float, dt: float):
        self.b = batch
        self.growth = math.exp(r * dt)
        E, D = batch.num_envs, batch.action_dim
        dev = batch.V.device
        self.cash = batch.V[:, 0].clone()
        self.pv = self.cash.clone()                      # portfolio value, starts at the premium
        self.hold = torch.zeros(E, D, device=dev)
        self.last_trade = torch.zeros(E, D, device=dev)  # signed size of the most recent rebalance
        self.traded = torch.zeros(E, D, device=dev)      # 1 if that instrument was traded last step
        self.since_trade = torch.zeros(E, device=dev)    # steps since any instrument was last traded
        self.costs = torch.zeros(E, device=dev)
        self.trade_steps = torch.zeros(E, device=dev)
        self.t = 0

    def step(self, action):
        X_t = self.b.X[:, self.t]
        trade = action - self.hold
        value = trade * X_t
        cost = (self.b.fee * value.abs()).sum(-1)
        self.cash = (self.cash - value.sum(-1) - cost) * self.growth
        self.hold = action
        self.costs += cost
        self.last_trade = trade
        self.traded = (trade.abs() > TRADE_EPS).float()
        any_trade = self.traded.amax(-1)
        self.trade_steps += any_trade
        self.since_trade = (self.since_trade + 1.0) * (1.0 - any_trade)
        self.t += 1
        self.pv = self.cash + (self.hold * self.b.X[:, self.t]).sum(-1)
        return -(self.pv - self.b.V[:, self.t]).abs(), self.pv


def trade_features(state: HedgeState) -> torch.Tensor:
    """What the agent has done so far, in scale-free units (3D + 3 columns):
    holdings, last trade, traded-last-step flags, steps since last trade / N,
    cumulative costs and current hedging error, both in % of initial spot."""
    b, t = state.b, state.t
    spot0 = b.S[:, 0]
    return torch.cat([
        state.hold,
        state.last_trade,
        state.traded,
        (state.since_trade / b.num_steps)[:, None],
        (100.0 * state.costs / spot0)[:, None],
        (100.0 * (state.pv - b.V[:, t]) / spot0)[:, None],
    ], dim=-1)


class ObsBuilder:
    """MLP: obs_t. RNN: sliding window of the last L observations, zero padded (old history_len).
    With trade features on, every entry of the window also carries that day's trading state, so the
    LSTM sees the recent trade / no-trade pattern."""

    def __init__(self, batch: HedgeBatch, arch: str, history: int, use_trade_features: bool):
        self.b, self.arch, self.L, self.use_tf = batch, arch, history, use_trade_features
        self.buf = None

    def _obs(self, state: HedgeState):
        o = self.b.feats[:, state.t]
        return torch.cat([o, trade_features(state)], dim=-1) if self.use_tf else o

    def reset(self, state: HedgeState):
        o = self._obs(state)
        if self.arch == "mlp":
            return o
        self.buf = torch.zeros(o.shape[0], self.L, o.shape[1], device=o.device)
        self.buf[:, -1] = o
        return self.buf

    def next(self, state: HedgeState):
        o = self._obs(state)
        if self.arch == "mlp":
            return o
        self.buf = torch.cat([self.buf[:, 1:], o[:, None]], dim=1)
        return self.buf


def obs_dim(bank: Bank, use_trade_features: bool) -> int:
    f = BASE_FEATURES + (OPTION_FEATURES if bank.has_options else 0)
    return f + (3 * bank.action_dim + 3 if use_trade_features else 0)
