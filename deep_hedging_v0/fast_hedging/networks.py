"""Actor-critic and inaction networks (plain torch, same shapes as the TorchRL ones).

The policy and value heads see [encoder features, agent state]. The agent state
(current holdings, tracking error, costs paid, time since last trade, and for the
inaction net the proposed trade and its cost) depends on the agent's own past
actions, so it is fed to the heads rather than to the market encoder.

single net : encoder -> (TanhNormal policy head, value head)        hidden 64
double net : the same actor + an inaction net (encoder -> Bernoulli
             'hold' head, value head); both use hidden int(64/sqrt 2)=45
             so the pair has roughly the parameter count of one single net.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_BIAS = math.log(math.expm1(1.0 - 0.01))   # tensordict biased_softplus_1.0


class Encoder(nn.Module):
    """Market-feature encoder.

    'mlp'       : 2-layer MLP on the current observation, input (n, F) -> (n, h).
    'window'    : 2-layer LSTM over a sliding window of the last W observations,
                  re-run from scratch at every step (the old history_len=10 model),
                  input (n, W, F) -> (n, h).
    'recurrent' : 2-layer LSTM run once over the whole path with its hidden state
                  carried forward, input (n, T, F) -> (n, T, h). Same network as
                  'window', but each step costs one LSTM step instead of W.
    """

    def __init__(self, mode, in_dim, hidden, activation="tanh"):
        super().__init__()
        self.mode = mode
        if mode == "mlp":
            act = nn.Tanh if activation == "tanh" else nn.ReLU
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), act(), nn.Linear(hidden, hidden), act())
        elif mode in ("window", "recurrent"):
            self.net = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True)
        else:
            raise ValueError(mode)

    @property
    def sequential(self):
        return self.mode == "recurrent"

    def forward(self, x):
        if self.mode == "mlp":
            return self.net(x)
        out, _ = self.net(x)
        return out if self.mode == "recurrent" else out[:, -1]


class SquashedNormal:
    """Normal in pre-tanh space u, action = low + (high-low) * (tanh(u)+1)/2."""

    def __init__(self, loc, scale, low, high):
        self.loc, self.scale, self.low, self.high = loc, scale, low, high
        self.base = torch.distributions.Normal(loc, scale)

    def sample(self):
        return self.base.sample()

    def squash(self, u):
        return self.low + (self.high - self.low) * 0.5 * (torch.tanh(u) + 1.0)

    def log_prob(self, u):
        # log|da/du| = log((high-low)/2) + log(1 - tanh(u)^2)
        log_det = math.log((self.high - self.low) / 2.0) + 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return (self.base.log_prob(u) - log_det).sum(-1)

    def entropy(self):                           # entropy of the base Gaussian (bonus is tiny)
        return self.base.entropy().sum(-1)

    def mode(self):
        return self.loc


class ActorCritic(nn.Module):
    def __init__(self, mode, obs_dim, action_dim, hidden, low, high, extra_dim=0):
        super().__init__()
        self.encoder = Encoder(mode, obs_dim, hidden, "tanh")
        self.policy = nn.Linear(hidden + extra_dim, 2 * action_dim)
        self.value = nn.Sequential(nn.Linear(hidden + extra_dim, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.low, self.high, self.extra_dim = float(low), float(high), extra_dim

    def heads(self, feat, extra=None):
        h = feat if extra is None else torch.cat([feat, extra], dim=-1)
        loc, raw = self.policy(h).chunk(2, dim=-1)
        scale = (F.softplus(raw + _SCALE_BIAS) + 0.01).clamp_min(1e-4)
        return SquashedNormal(loc, scale, self.low, self.high), self.value(h).squeeze(-1)


class Inactor(nn.Module):
    """Bernoulli 'hold previous position' decision (1 = do not trade)."""

    def __init__(self, mode, obs_dim, hidden, extra_dim=0, init_hold_prob=None):
        super().__init__()
        self.encoder = Encoder(mode, obs_dim, hidden, "relu")
        self.policy = nn.Linear(hidden + extra_dim, 1)
        self.value = nn.Sequential(nn.Linear(hidden + extra_dim, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.extra_dim = extra_dim
        if init_hold_prob is not None:
            nn.init.zeros_(self.policy.weight)
            nn.init.constant_(self.policy.bias, math.log(init_hold_prob / (1 - init_hold_prob)))

    def heads(self, feat, extra=None):
        h = feat if extra is None else torch.cat([feat, extra], dim=-1)
        logits = self.policy(h).squeeze(-1)
        return torch.distributions.Bernoulli(logits=logits), self.value(h).squeeze(-1)
