"""Plain-PyTorch versions of the old TorchRL actor/critic and inaction nets.

Architectures follow experiments/utils/{ppo_mlp_actor,ppo_rnn_actor,actor_inactor_mlp}.py:
  * HedgeActor: MLP(Tanh, 2 layers) or 2-layer LSTM over the window -> (loc, scale) of a
    tanh-squashed Normal mapped to [low, high]; critic Linear-Tanh-Linear on the same features.
  * InactionGate: MLP(ReLU) or LSTM -> Bernoulli logits per hedge instrument
    (1 = keep yesterday's position); critic on its own features.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCALE_BIAS = math.log(math.e - 1.0)   # softplus(bias) == 1, like NormalParamExtractor's biased softplus
_MIN_SCALE = 1e-4


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), act(), nn.Linear(hidden, hidden), act())

    def forward(self, x):
        return self.net(x)


class LSTMEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)

    def forward(self, x):          # (B, L, F) -> (B, hidden), last step of the window
        out, _ = self.rnn(x)
        return out[:, -1]


def _encoder(arch, in_dim, hidden, act):
    return MLPEncoder(in_dim, hidden, act) if arch == "mlp" else LSTMEncoder(in_dim, hidden)


def _critic(hidden):
    return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))


class HedgeActor(nn.Module):
    def __init__(self, arch, obs_dim, act_dim, hidden, low, high):
        super().__init__()
        self.encoder = _encoder(arch, obs_dim, hidden, nn.Tanh)
        self.policy_head = nn.Linear(hidden, 2 * act_dim)
        self.value_head = _critic(hidden)
        self.low, self.half = float(low), (float(high) - float(low)) / 2.0

    def _params(self, obs):
        h = self.encoder(obs)
        loc, raw_scale = self.policy_head(h).chunk(2, dim=-1)
        scale = F.softplus(raw_scale + _SCALE_BIAS) + _MIN_SCALE
        return loc, scale, self.value_head(h).squeeze(-1)

    def squash(self, z):
        return self.low + (torch.tanh(z) + 1.0) * self.half

    def _log_prob(self, loc, scale, z):
        base = torch.distributions.Normal(loc, scale).log_prob(z)
        log_det = math.log(self.half) + 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        return (base - log_det).sum(-1)

    def act(self, obs, deterministic=False):
        """Returns (action, pre-tanh sample z, log_prob(z), value)."""
        loc, scale, value = self._params(obs)
        z = loc if deterministic else loc + scale * torch.randn_like(loc)
        return self.squash(z), z, self._log_prob(loc, scale, z), value

    def evaluate(self, obs, z):
        loc, scale, value = self._params(obs)
        entropy = torch.distributions.Normal(loc, scale).entropy().sum(-1)  # base-Normal entropy
        return self._log_prob(loc, scale, z), entropy, value


class InactionGate(nn.Module):
    def __init__(self, arch, obs_dim, act_dim, hidden, init_logit=0.0):
        super().__init__()
        self.encoder = _encoder(arch, obs_dim, hidden, nn.ReLU)
        self.policy_head = nn.Linear(hidden, act_dim)
        # Zero weights + negative bias: at the start every state has logit = init_logit, so the
        # deterministic gate never holds and the double net begins exactly at its single net.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.constant_(self.policy_head.bias, init_logit)
        self.value_head = _critic(hidden)

    def _params(self, obs):
        h = self.encoder(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs, deterministic=False):
        """Returns (hold mask in {0,1}, log_prob, value)."""
        logits, value = self._params(obs)
        if deterministic:
            mask = (logits > 0).float()
        else:
            mask = torch.bernoulli(torch.sigmoid(logits))
        logp = torch.distributions.Bernoulli(logits=logits).log_prob(mask).sum(-1)
        return mask, logp, value

    def evaluate(self, obs, mask):
        logits, value = self._params(obs)
        dist = torch.distributions.Bernoulli(logits=logits)
        return dist.log_prob(mask).sum(-1), dist.entropy().sum(-1), value


def action_bounds(product: str) -> tuple[float, float]:
    """Old scripts: call hedge ratio in [0, 1]; doc/con option positions in [-1, 1]."""
    return (0.0, 1.0) if product == "call" else (-1.0, 1.0)
