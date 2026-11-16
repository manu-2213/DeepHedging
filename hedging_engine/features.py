"""Observation features for every time step at once (torch).

Reproduces hedging/feature_extractor.py: 11 features for the call, plus 6
call/put features for doc/con. Features never depend on the agent's actions,
so they are computed for the whole path in one shot.
"""
from __future__ import annotations

import torch

EPS32 = float(torch.finfo(torch.float32).eps)
LOG_EPS = 1e-8


def _safe_log(x):
    return torch.log(torch.clamp(x, min=LOG_EPS))


def _changes(x):
    """x: (..., N+1). Returns diff, ratio, log-ratio with step-0 conventions (0, 1, 0)."""
    prev = torch.cat([x[..., :1], x[..., :-1]], dim=-1)
    diff = x - prev
    ratio = (x + EPS32) / (prev + EPS32)
    diff[..., 0] = 0.0
    ratio[..., 0] = 1.0
    return diff, ratio, _safe_log(ratio)


def build_observations(S, liability, call, put, K, T, r, product):
    """S: (P, A, N+1); liability/call/put: (P, A, M, N+1); K: (A, M); T: (N+1,).

    Returns (P*A*M, N+1, F) float32, env order (path, asset, strike) like HedgeBase.
    """
    P, A, M, N1 = liability.shape
    Sx = S[:, :, None, :].expand(P, A, M, N1)
    dS, rS, lS = _changes(Sx)
    dO, rO, lO = _changes(liability)
    delta = torch.clamp(dO / (dS + EPS32), 0.0, 1.0)
    delta[..., 0] = 0.0
    money = K[None, :, :, None] / Sx
    feats = [dS, rS, lS, dO, rO, lO, delta, money, _safe_log(money),
             T[None, None, None, :].expand(P, A, M, N1), torch.full_like(money, float(r))]
    if product in ("doc", "con"):
        dC, rC, lC = _changes(call)
        dP, rP, lP = _changes(put)
        feats += [dC, dP, rC, rP, lC, lP]
    obs = torch.stack(feats, dim=-1)                     # (P, A, M, N+1, F)
    return obs.reshape(P * A * M, N1, obs.shape[-1]).float()


def feature_dim(product: str) -> int:
    return 11 if product == "call" else 17
