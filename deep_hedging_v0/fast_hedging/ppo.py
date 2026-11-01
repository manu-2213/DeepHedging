"""Vectorised hedging environment + PPO for the single net and the actor/inactor pair.

Environment semantics are those of hedging/env_hedge_*.py:
  t = 0..N-1: choose position a_t in each hedging instrument,
  cash_t  -= sum_i (a_t - h_t)_i X_t,i + fee_i |(a_t - h_t)_i X_t,i|
  cash_t+1 = cash_t * exp(r dt),  PV_t+1 = cash_t+1 + sum_i a_t,i X_t+1,i
  reward_t = -|PV_t+1 - L_t+1|   (optionally divided by the claim's premium)
Terminal hedging P&L = PV_N - L_N.

Rewards (cfg.reward):
  tracking  : r_t = -|PV_t+1 - L_t+1| * scale  (original objective, gamma 0.99). A cost paid at t
              lowers PV for every later step, so it is counted about N - t times, while the
              terminal tail counts once: early trades are over-penalised.
  mean_cvar : r_t = d(PV - L)_t * scale, plus at maturity -lambda/(1-alpha) * (loss - c)^+  with
              loss = -(PV_N - L_N) * scale and c a running VaR_alpha estimate (Rockafellar-Uryasev).
              With gamma = 1 the return is PnL - lambda/(1-alpha)(loss - c)^+, whose expectation,
              minimised over c, is E[PnL] - lambda * CVaR_alpha(loss): each cost is counted once and
              the training objective matches the CVaR95 used for selection and testing.

Networks (cfg.net): single (actor-critic), band (actor outputs target + no-trade half-width,
executed hedge = holdings projected onto [target - w, target + w]), double (actor + inaction net).

Agent state fed to the policy/value heads at step t (cfg.agent_state, on by default):
  actor   : h_t (current holdings = last executed hedge), (PV_t - L_t)/premium,
            costs paid so far/premium, steps since the last executed trade / N
  inactor : the actor's state + proposed position a_t, proposed trade a_t - h_t,
            cost of executing that trade/premium
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .features import build_observations, feature_dim
from .market import Bank
from .networks import ActorCritic, Inactor


class TrialPruned(Exception):
    """Raised by a report callback to abort training early (mapped to optuna.TrialPruned)."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    arch: str = "rnn"                 # rnn | mlp
    rnn_mode: str = "window"          # window: LSTM over the last `window` steps | recurrent: whole path
    net: str = "single"               # single | band | double
    window: int = 8                   # history_len: observations the window LSTM sees (t-7..t)
    hidden: int = 64
    underlying_fee: float = 0.01
    option_fee: float = 0.05
    agent_state: bool = True          # feed holdings, tracking error, costs, time since trade (+ proposed trade)
    reward_norm: str = "premium"      # premium | spot | none
    normalize_obs: bool = True
    normalize_adv: bool = True
    mask_held_actor_loss: bool = True # do not credit the actor for actions that were never executed
    inactor_init_hold_prob: float | None = None
    reward: str = "mean_cvar"         # mean_cvar | tracking (see module docstring)
    cvar_alpha: float = 0.95
    cvar_lambda: float = 1.0          # weight of CVaR against expected P&L in mean_cvar
    cvar_ema: float = 0.9             # smoothing of the running VaR estimate c
    cvar_per_asset: bool = True       # one running VaR per asset (= calibration year), like the per-year test metric
    value_loss: str = "mse"           # mse | huber. Huber(1) saturates on the rare, large terminal CVaR targets
                                      # and makes the critic under-estimate the tail (v2.0 used huber)
    pad_mode: str = "edge"            # window LSTM padding before t=0: edge (repeat obs_0) | zero (= mean observation)
    gamma: float | None = None        # None: 1.0 for mean_cvar, 0.99 for tracking
    lmbda: float = 0.95               # GAE lambda; 1.0 = Monte Carlo advantages (terminal penalty reaches every step)
    clip: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    target_kl: float | None = 0.03    # stop the PPO epochs of a rollout once approx. KL > 1.5 x this
    paths_per_iter: int = 64          # paths drawn per asset per iteration
    ppo_epochs: int = 4
    minibatch: int = 16384            # samples per gradient step (recurrent: minibatch // N whole paths)
    actor_update_every: int = 3       # joint phase: actor updated every k-th iteration (old protocol)
    eval_every: int = 10              # iterations between validation evaluations (a full validation
                                      # rollout costs several training rollouts with the window LSTM)
    patience: int = 10                # evaluations without improvement (= 100 iterations) before stopping
    min_rel_improvement: float = 1e-3
    max_iters: int = 3000             # per phase
    min_iters: int = 50
    metric: str = "cvar_95"
    eval_chunk_paths: int = 256
    max_bad_batches: int = 20         # consecutive non-finite rollouts before a phase is aborted

    def __post_init__(self):
        if self.gamma is None:
            self.gamma = 1.0 if self.reward == "mean_cvar" else 0.99
        if self.net not in ("single", "band", "double"):
            raise ValueError(f"unknown net {self.net}")
        if self.reward not in ("mean_cvar", "tracking"):
            raise ValueError(f"unknown reward {self.reward}")
        if self.value_loss not in ("mse", "huber"):
            raise ValueError(f"unknown value_loss {self.value_loss}")
        if self.pad_mode not in ("edge", "zero"):
            raise ValueError(f"unknown pad_mode {self.pad_mode}")


def action_bounds(product):
    return (0.0, 1.0) if product == "call" else (-1.0, 1.0)


def encoder_mode(cfg: TrainConfig) -> str:
    return "mlp" if cfg.arch == "mlp" else cfg.rnn_mode


def actor_state_dim(A):
    return A + 3


def inactor_state_dim(A):
    return actor_state_dim(A) + 2 * A + 1


# --------------------------------------------------------------------------
# Market data on device
# --------------------------------------------------------------------------
class Task:
    def __init__(self, bank: Bank, product: str, r: float, cfg: TrainConfig, device, norm=None):
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        self.S, self.L, self.X = t(bank.S), t(bank.liability), t(bank.instruments)
        self.C, self.Pu, self.K, self.T = t(bank.call), t(bank.put), t(bank.K), t(bank.T)
        self.sigma = None if bank.sigma is None or np.isnan(bank.sigma).any() else t(bank.sigma)
        self.D = t(bank.delta) if bank.delta is not None and bank.delta.size else None   # (P, A, M, N+1)
        dmv = getattr(bank, "delta_mv", None)
        self.Dmv = t(dmv) if dmv is not None and dmv.size else None                       # (P, A, M, N+1)
        self.pad_mode = cfg.pad_mode
        self.product, self.r, self.device = product, r, device
        self.P, self.A, self.M, self.N1, self.I = self.X.shape
        self.N = self.N1 - 1
        self.growth = math.exp(r * float(bank.T[0]) / self.N)
        kinds = bank.instrument_kind
        self.fees = torch.tensor([cfg.underlying_fee if k == "underlying" else cfg.option_fee for k in kinds],
                                 device=device, dtype=torch.float32)
        prem = self.L[:, :, :, 0].mean(dim=(0, 2))                          # (A,)
        self.prem_scale = (1.0 / prem.clamp_min(1e-8)).float()              # agent-state features
        if cfg.reward_norm == "premium":
            scale = self.prem_scale
        elif cfg.reward_norm == "spot":
            scale = 1.0 / self.S[0, :, 0]
        else:
            scale = torch.ones_like(prem)
        self.asset_scale = scale.float()
        self.window = cfg.window if encoder_mode(cfg) == "window" else 1
        if norm is None and cfg.normalize_obs:
            sub = self._raw_obs(torch.arange(min(self.P, 256), device=device))[:, : self.N]
            flat = sub.reshape(-1, sub.shape[-1])
            norm = (flat.mean(0), flat.std(0).clamp_min(1e-6))
        self.norm = norm

    def _raw_obs(self, idx):
        return build_observations(self.S[idx], self.L[idx], self.C[idx], self.Pu[idx], self.K, self.T,
                                  self.r, self.product)

    def batch(self, idx):
        """Everything one rollout needs for the paths `idx`, flattened over (path, asset, strike)."""
        obs = self._raw_obs(idx)
        if self.norm is not None:
            obs = ((obs - self.norm[0]) / self.norm[1]).clamp(-10.0, 10.0)
        k = idx.numel()
        B = k * self.A * self.M
        X = self.X[idx].reshape(B, self.N1, self.I)
        L = self.L[idx].reshape(B, self.N1)
        per_env = lambda v: v[None, :, None].expand(k, self.A, self.M).reshape(B)
        if self.window > 1:                 # history before t=0: repeat obs_0 (edge) or zeros (= the mean observation)
            if self.pad_mode == "edge":
                pad = obs[:, :1].expand(B, self.window - 1, obs.shape[-1])
            else:
                pad = torch.zeros(B, self.window - 1, obs.shape[-1], device=obs.device, dtype=obs.dtype)
            obs = torch.cat([pad, obs], dim=1)                              # (B, N+W, F)
        return {"obs": obs, "X": X, "L": L, "scale": per_env(self.asset_scale),
                "pscale": per_env(self.prem_scale), "B": B}

    def inputs(self, obs, b, t):
        """Per-sample encoder inputs (non-recurrent encoders): (n, F) or windows (n, W, F)."""
        if self.window == 1:
            return obs[b, t]
        offs = torch.arange(self.window, device=obs.device)
        return obs[b[:, None], t[:, None] + offs[None, :]]


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class Agent:
    def __init__(self, cfg: TrainConfig, product: str, device):
        self.cfg, self.device = cfg, device
        A = 1 if product == "call" else 2
        self.action_dim = A
        low, high = action_bounds(product)
        F_ = feature_dim(product)
        mode = encoder_mode(cfg)
        h = int(cfg.hidden / math.sqrt(2)) if cfg.net == "double" else cfg.hidden
        if cfg.net == "band":       # [targets in [low, high], half-widths in [0, (high-low)/2]]
            self.actor_out = 2 * A
            lo_v, hi_v = [low] * A + [0.0] * A, [high] * A + [(high - low) / 2.0] * A
        else:
            self.actor_out = A
            lo_v, hi_v = low, high
        self.actor = ActorCritic(mode, F_, self.actor_out, h, lo_v, hi_v,
                                 extra_dim=actor_state_dim(A) if cfg.agent_state else 0).to(device)
        self.inactor = None
        if cfg.net == "double":
            self.inactor = Inactor(mode, F_, h, extra_dim=inactor_state_dim(A) if cfg.agent_state else 0,
                                   init_hold_prob=cfg.inactor_init_hold_prob).to(device)

    def state_dict(self):
        return {"actor": copy.deepcopy(self.actor.state_dict()),
                "inactor": None if self.inactor is None else copy.deepcopy(self.inactor.state_dict())}

    def load_state_dict(self, sd, actor_only=False):
        self.actor.load_state_dict(sd["actor"])
        if not actor_only and self.inactor is not None and sd.get("inactor") is not None:
            self.inactor.load_state_dict(sd["inactor"])


def _encode_all(encoder, task, obs, B, N, chunk=131072):
    """Encoder features for every (env, t<N), no grad. Returns (B, N, h)."""
    if encoder.sequential:
        c = max(1, chunk // N)
        return torch.cat([encoder(obs[i:i + c, :N]) for i in range(0, B, c)])
    b = torch.arange(B, device=obs.device).repeat_interleave(N)
    t = torch.arange(N, device=obs.device).repeat(B)
    outs = [encoder(task.inputs(obs, b[i:i + chunk], t[i:i + chunk])) for i in range(0, B * N, chunk)]
    return torch.cat(outs).reshape(B, N, -1)


@torch.no_grad()
def rollout(agent: Agent, task: Task, data, *, use_inactor, actor_stochastic=True, inactor_stochastic=None,
            record=True, gen=None, trace=False):
    """trace=True also returns per-step trajectories (for analysis; keep B small)."""
    if inactor_stochastic is None:
        inactor_stochastic = actor_stochastic
    cfg, N, B = agent.cfg, task.N, data["B"]
    X, L, scale, ps = data["X"], data["L"], data["scale"], data["pscale"]
    feat_a = _encode_all(agent.actor.encoder, task, data["obs"], B, N)
    feat_i = _encode_all(agent.inactor.encoder, task, data["obs"], B, N) if use_inactor else None
    A = agent.action_dim
    dev = X.device
    pos = torch.zeros(B, A, device=dev)
    cash = L[:, 0].clone()
    pv = cash.clone()                       # portfolio value before trading at t
    cum_cost = torch.zeros(B, device=dev)
    since = torch.zeros(B, device=dev)      # steps since the last executed trade
    rec = {}
    if record:
        z = lambda *s: torch.zeros(B, N, *s, device=dev)
        rec = {"u": z(agent.actor_out), "logp": z(), "value": z(), "reward": z(), "executed": z()}
        if cfg.agent_state:
            rec["xa"] = z(actor_state_dim(A))
        if use_inactor:
            rec.update({"hold": z(), "hold_logp": z(), "i_value": z()})
            if cfg.agent_state:
                rec["xi"] = z(inactor_state_dim(A))
    abs_err = torch.zeros(B, device=dev)
    holds = torch.zeros(B, device=dev)
    n_trades = torch.zeros(B, device=dev)
    tr = None
    if trace:
        zt = lambda *s: torch.zeros(B, N, *s, device=dev)
        tr = {"position": zt(A), "proposed": zt(A), "band_width": zt(A), "hold": zt(), "hold_prob": zt(),
              "cost": zt(), "tracking_error": zt(), "portfolio_value": torch.zeros(B, N + 1, device=dev)}
        tr["portfolio_value"][:, 0] = pv
    for t in range(N):
        xa = None
        if cfg.agent_state:
            xa = torch.cat([pos, ((pv - L[:, t]) * ps)[:, None], (cum_cost * ps)[:, None],
                            (since / N)[:, None]], dim=-1).clamp(-10.0, 10.0)
        dist, value = agent.actor.heads(feat_a[:, t], xa)
        u = torch.normal(dist.loc, dist.scale, generator=gen) if actor_stochastic else dist.mode()
        a = dist.squash(u)
        width = None
        if cfg.net == "band":       # learned no-trade band: move to the nearest band edge only
            a, width = a[:, :A], a[:, A:]
            final = torch.maximum(torch.minimum(pos, a + width), a - width)
        else:
            final = a
        if use_inactor:
            xi = None
            if cfg.agent_state:
                prop = a - pos
                est = (task.fees * (prop * X[:, t]).abs()).sum(-1) * ps
                xi = torch.cat([xa, a, prop, est[:, None]], dim=-1).clamp(-10.0, 10.0)
            idist, ivalue = agent.inactor.heads(feat_i[:, t], xi)
            if inactor_stochastic:
                hold = torch.bernoulli(idist.probs, generator=gen)
            else:
                hold = (idist.logits > 0).float()
            final = torch.where(hold[:, None] > 0.5, pos, a)
            holds += hold
        trade = final - pos
        flow = trade * X[:, t]
        cost = (task.fees * flow.abs()).sum(-1)
        cash = (cash - flow.sum(-1) - cost) * task.growth
        pnl_before = pv - L[:, t]
        pv = cash + (final * X[:, t + 1]).sum(-1)
        err = (pv - L[:, t + 1]).abs()
        if record:
            rec["u"][:, t], rec["logp"][:, t], rec["value"][:, t] = u, dist.log_prob(u), value
            if cfg.reward == "tracking":
                rec["reward"][:, t] = -err * scale
            else:                   # P&L increment; terminal CVaR penalty added in train_phase
                rec["reward"][:, t] = ((pv - L[:, t + 1]) - pnl_before) * scale
            rec["executed"][:, t] = 1.0 if not use_inactor else 1.0 - hold
            if cfg.agent_state:
                rec["xa"][:, t] = xa
            if use_inactor:
                rec["hold"][:, t], rec["hold_logp"][:, t], rec["i_value"][:, t] = hold, idist.log_prob(hold), ivalue
                if cfg.agent_state:
                    rec["xi"][:, t] = xi
        traded = (trade.abs().amax(-1) > 1e-6).float()
        since = (since + 1.0) * (1.0 - traded)
        n_trades += traded
        if trace:
            tr["position"][:, t], tr["proposed"][:, t], tr["cost"][:, t] = final, a, cost
            if width is not None:
                tr["band_width"][:, t] = width
            tr["tracking_error"][:, t] = pv - L[:, t + 1]
            tr["portfolio_value"][:, t + 1] = pv
            if use_inactor:
                tr["hold"][:, t], tr["hold_prob"][:, t] = hold, idist.probs
        abs_err += err
        cum_cost += cost
        pos = final
    pnl = pv - L[:, N]
    out = {"pnl": pnl, "abs_err": abs_err / N, "cost": cum_cost, "hold_rate": holds / N, "n_trades": n_trades}
    if trace:
        out["trace"] = tr
    return rec, out


def gae(reward, value, gamma, lmbda):
    B, N = reward.shape
    adv = torch.zeros_like(reward)
    last = torch.zeros(B, device=reward.device)
    for t in range(N - 1, -1, -1):
        nxt = value[:, t + 1] if t + 1 < N else torch.zeros(B, device=reward.device)
        delta = reward[:, t] + gamma * nxt - value[:, t]
        last = delta + gamma * lmbda * last
        adv[:, t] = last
    return adv, adv + value


def _ppo_step(opt, params, loss, max_norm):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm)
    for p in params:
        if p.grad is not None:
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    opt.step()


def _masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _minibatches(model, task, obs, B, N, cfg, gen):
    """Yields (flat sample indices j = b*N + t, encoder features for those samples)."""
    dev = obs.device
    if model.encoder.sequential:
        env_mb = min(B, max(1, cfg.minibatch // N))          # whole paths, so the LSTM state is exact
        steps = torch.arange(N, device=dev)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=dev, generator=gen)
            for lo in range(0, B - env_mb + 1, env_mb):
                e = perm[lo:lo + env_mb]
                feat = model.encoder(obs[e, :N])                # (e, N, h)
                yield (e[:, None] * N + steps[None, :]).reshape(-1), feat.reshape(-1, feat.shape[-1])
    else:
        n = B * N
        mb = min(cfg.minibatch, n)
        b_all = torch.arange(B, device=dev).repeat_interleave(N)
        t_all = torch.arange(N, device=dev).repeat(B)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=dev, generator=gen)
            for lo in range(0, n - mb + 1, mb):
                j = perm[lo:lo + mb]
                yield j, model.encoder(task.inputs(obs, b_all[j], t_all[j]))


def update(agent: Agent, task: Task, data, rec, *, which, opt, gen=None):
    """PPO epochs over the rollout for 'actor' or 'inactor'."""
    cfg = agent.cfg
    B, N = rec["reward"].shape
    if which == "actor":
        model, value_old, logp_old = agent.actor, rec["value"], rec["logp"]
        mask = rec["executed"] if (cfg.mask_held_actor_loss and "hold" in rec) else torch.ones_like(value_old)
        state = rec.get("xa")
    else:
        model, value_old, logp_old = agent.inactor, rec["i_value"], rec["hold_logp"]
        mask = torch.ones_like(value_old)
        state = rec.get("xi")
    adv, ret = gae(rec["reward"], value_old, cfg.gamma, cfg.lmbda)
    flat = lambda x: x.reshape(B * N, *x.shape[2:])
    adv, ret, logp_old, mask = flat(adv), flat(ret), flat(logp_old), flat(mask)
    state = None if state is None else flat(state)
    u = flat(rec["u"])
    hold = flat(rec["hold"]) if "hold" in rec else None
    params = list(model.parameters())
    stats = {k: [] for k in ("loss_objective", "loss_value", "entropy", "approx_kl", "clip_fraction",
                             "policy_spread", "explained_variance")}
    var_ret = float(ret.var().clamp_min(1e-12))
    for j, feat in _minibatches(model, task, data["obs"], B, N, cfg, gen):
        dist, v = model.heads(feat, None if state is None else state[j])
        logp = dist.log_prob(u[j]) if which == "actor" else dist.log_prob(hold[j])
        m = mask[j]
        a = adv[j]
        if cfg.normalize_adv:
            mu = _masked_mean(a, m)
            sd = torch.sqrt(_masked_mean((a - mu) ** 2, m)).clamp_min(1e-8)
            a = (a - mu) / sd
        log_ratio = (logp - logp_old[j]).clamp(-20.0, 20.0)       # exp can never overflow
        with torch.no_grad():
            kl = _masked_mean(torch.exp(log_ratio) - 1 - log_ratio, m)
        if cfg.target_kl is not None and stats["loss_objective"] and float(kl) > 1.5 * cfg.target_kl:
            stats["early_stops"] = [torch.tensor(1.0)]
            break                                                   # policy moved far enough
        ratio = torch.exp(log_ratio)
        obj = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
        loss_obj = _masked_mean(obj, m)
        if cfg.value_loss == "huber":
            loss_v = cfg.value_coef * F.smooth_l1_loss(v, ret[j])
        else:
            loss_v = cfg.value_coef * ((v - ret[j]) ** 2).mean()
        loss_e = -cfg.entropy_coef * dist.entropy().mean()
        _ppo_step(opt, params, loss_obj + loss_v + loss_e, cfg.max_grad_norm)
        with torch.no_grad():
            spread = dist.scale.mean() if which == "actor" else dist.probs.mean()   # action std / mean hold prob
            for k, val in (("loss_objective", loss_obj), ("loss_value", loss_v), ("entropy", dist.entropy().mean()),
                         ("approx_kl", kl),
                         ("clip_fraction", _masked_mean(((ratio - 1).abs() > cfg.clip).float(), m)),
                         ("policy_spread", spread),
                         ("explained_variance", 1 - (ret[j] - v).var() / var_ret)):
                stats[k].append(val.detach())
    out = {k: float(torch.stack(v).mean()) if v else float("nan") for k, v in stats.items()}
    out.setdefault("early_stops", 0.0)          # 1.0 if this update stopped early on the KL limit
    out["minibatch_steps"] = float(len(stats["loss_objective"]))
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def risk_metrics(pnl: np.ndarray) -> dict:
    """Same conventions as experiments/utils/testing.py (loss = -pnl, positive is bad)."""
    x = pnl.astype(np.float64).ravel()
    losses = -x
    var95, var99 = np.percentile(losses, 95), np.percentile(losses, 99)
    return {"mean": float(x.mean()), "std": float(x.std()), "var_95": float(var95), "var_99": float(var99),
            "cvar_95": float(losses[losses >= var95].mean()), "cvar_99": float(losses[losses >= var99].mean()),
            "min": float(x.min()), "max": float(x.max())}


@torch.no_grad()
def evaluate(agent: Agent, task: Task, *, use_inactor, actor_stochastic=False, inactor_stochastic=False, seed=0):
    """Default: deterministic actor and deterministic hold (p > 0.5).
    inactor_stochastic=True reproduces the old 'stochastic inaction' report (actor still deterministic)."""
    stochastic = actor_stochastic or inactor_stochastic
    gen = torch.Generator(device=task.device).manual_seed(seed) if stochastic else None
    outs = {k: [] for k in ("pnl", "abs_err", "cost", "hold_rate", "n_trades")}
    for lo in range(0, task.P, agent.cfg.eval_chunk_paths):
        idx = torch.arange(lo, min(task.P, lo + agent.cfg.eval_chunk_paths), device=task.device)
        _, o = rollout(agent, task, task.batch(idx), use_inactor=use_inactor, actor_stochastic=actor_stochastic,
                       inactor_stochastic=inactor_stochastic, record=False, gen=gen)
        for k in outs:
            outs[k].append(o[k].cpu())
    arr = {k: torch.cat(v).numpy() for k, v in outs.items()}
    m = risk_metrics(arr["pnl"])
    m.update({"mean_abs_tracking_error": float(arr["abs_err"].mean()), "mean_cost": float(arr["cost"].mean()),
              "hold_rate": float(arr["hold_rate"].mean()), "mean_trades": float(arr["n_trades"].mean()),
              "no_trade_rate": float(1.0 - arr["n_trades"].mean() / task.N)})
    return m, arr          # arr: per-env numpy arrays in (path, asset, strike) order


# --------------------------------------------------------------------------
# Training phases with early stopping
# --------------------------------------------------------------------------
def train_phase(agent: Agent, train: Task, val: Task, *, phase, lr_actor, lr_inactor=None, seed=0,
                report=None, step_offset=0, log=print, iter_offset=0, logs=None, train_log_every=1):
    """phase 'actor': train the actor alone (the single net, or stage 1 of the double net).
    phase 'joint': train actor+inactor together (stage 2 of the double net).
    Stops when the validation metric has not improved for `patience` evaluations
    and restores the best weights.
    Returns (best_metric, best_val_metrics, logs); logs["eval"] has one entry per
    validation check, train_log one entry per iteration (rollout stats + PPO diagnostics).
    If `logs` = {"eval": [...], "train": [...]} is given, entries are appended there as they
    happen, so they survive a pruned or crashed trial."""
    cfg = agent.cfg
    use_inactor = phase == "joint"
    gen = torch.Generator(device=train.device).manual_seed(seed)
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)
    opt_i = torch.optim.Adam(agent.inactor.parameters(), lr=lr_inactor) if use_inactor else None
    logs = logs if logs is not None else {"eval": [], "train": []}
    history, train_log = logs["eval"], logs["train"]
    best, best_state, best_metrics, bad = math.inf, agent.state_dict(), None, 0
    k = min(cfg.paths_per_iter, train.P)
    t0 = time.time()
    n_eval, bad_batches, c_var = 0, 0, None
    for it in range(1, cfg.max_iters + 1):
        idx = torch.randperm(train.P, device=train.device, generator=gen)[:k]
        data = train.batch(idx)
        rec, out = rollout(agent, train, data, use_inactor=use_inactor, actor_stochastic=True, gen=gen)
        if not (torch.isfinite(rec["reward"]).all() and torch.isfinite(rec["logp"]).all()
                and torch.isfinite(out["pnl"]).all()):
            bad_batches += 1
            log(f"  [{phase}] it {it}: non-finite rollout skipped ({bad_batches}/{cfg.max_bad_batches})")
            if bad_batches >= cfg.max_bad_batches:
                log(f"  [{phase}] too many non-finite rollouts; stopping this phase with the best weights")
                break
            continue
        bad_batches = 0
        if cfg.reward == "mean_cvar":   # Rockafellar-Uryasev terminal penalty, c = running VaR_alpha
            loss_env = -out["pnl"] * data["scale"]
            if cfg.cvar_per_asset:      # envs are ordered (path, asset, strike): one VaR per asset (= year)
                per_asset = loss_env.view(k, train.A, train.M).transpose(0, 1).reshape(train.A, -1)
                q = torch.quantile(per_asset, cfg.cvar_alpha, dim=1)[None, :, None].expand(k, train.A, train.M).reshape(-1)
            else:
                q = torch.quantile(loss_env, cfg.cvar_alpha)
            c_var = q if c_var is None else cfg.cvar_ema * c_var + (1 - cfg.cvar_ema) * q
            rec["reward"][:, -1] -= cfg.cvar_lambda / (1 - cfg.cvar_alpha) * torch.relu(loss_env - c_var)
        entry = {"iter": it, "global_iter": iter_offset + it, "phase": phase,
                 "cvar_threshold": float(c_var.mean()) if c_var is not None else None,
                 "cvar_threshold_per_asset": (c_var.view(k, train.A, train.M)[0, :, 0].tolist()
                                              if c_var is not None and c_var.dim() else None),
                 "train_cvar_95": risk_metrics(out["pnl"].cpu().numpy())["cvar_95"],
                 "train_mean_pnl": float(out["pnl"].mean()), "train_abs_err": float(out["abs_err"].mean()),
                 "train_cost": float(out["cost"].mean()), "train_hold_rate": float(out["hold_rate"].mean()),
                 "train_trades": float(out["n_trades"].mean()), "train_reward": float(rec["reward"].sum(1).mean())}
        if use_inactor:
            for key, val_ in update(agent, train, data, rec, which="inactor", opt=opt_i, gen=gen).items():
                entry[f"inactor_{key}"] = val_
        if not use_inactor or cfg.actor_update_every == 1 or it % cfg.actor_update_every == 1:
            for key, val_ in update(agent, train, data, rec, which="actor", opt=opt_a, gen=gen).items():
                entry[f"actor_{key}"] = val_
        entry["elapsed_s"] = time.time() - t0
        logs["last_global_iter"] = iter_offset + it
        if it == 1 or it % train_log_every == 0:
            train_log.append(entry)
        if it % cfg.eval_every == 0:
            n_eval += 1
            m, _ = evaluate(agent, val, use_inactor=use_inactor)
            metric = m[cfg.metric]
            if not math.isfinite(metric):
                log(f"  [{phase}] non-finite validation metric at iter {it}; stopping")
                break
            history.append({"iter": it, "global_iter": iter_offset + it, "phase": phase,
                            "train_abs_err": float(out["abs_err"].mean()),
                            **{f"val_{a}": b for a, b in m.items()}, "elapsed_s": time.time() - t0})
            if metric < best - cfg.min_rel_improvement * abs(best if math.isfinite(best) else metric):
                best, best_state, best_metrics, bad = metric, agent.state_dict(), m, 0
            else:
                bad += 1
            history[-1]["best_so_far"], history[-1]["bad_evals"] = best, bad
            log(f"  [{phase}] it {it:5d}  val {cfg.metric}={metric:.4f}  best={best:.4f}  "
                f"hold={m['hold_rate']:.3f}  cost={m['mean_cost']:.2f}  bad={bad}/{cfg.patience}  "
                f"{(time.time() - t0) / it:.2f}s/it")
            if report is not None:
                report(step_offset + n_eval, metric)        # may raise TrialPruned
            if bad >= cfg.patience and it >= cfg.min_iters:
                break
    agent.load_state_dict(best_state)
    return best, best_metrics, logs


@torch.no_grad()
def trace_rollout(agent: Agent, task: Task, n_paths, *, use_inactor, inactor_stochastic=False, seed=0):
    """Step-by-step trajectories on the first `n_paths` paths (all assets/strikes), as numpy."""
    idx = torch.arange(min(n_paths, task.P), device=task.device)
    gen = torch.Generator(device=task.device).manual_seed(seed) if inactor_stochastic else None
    data = task.batch(idx)
    _, out = rollout(agent, task, data, use_inactor=use_inactor, actor_stochastic=False,
                     inactor_stochastic=inactor_stochastic, record=False, gen=gen, trace=True)
    res = {f"trace_{k}": v.cpu().numpy() for k, v in out.pop("trace").items()}
    res.update({f"trace_{k}": v.cpu().numpy() for k, v in out.items()})        # per-env totals
    k = idx.numel()
    res["trace_liability"] = data["L"].cpu().numpy()                             # (B, N+1)
    res["trace_instrument_prices"] = data["X"].cpu().numpy()                     # (B, N+1, I)
    res["trace_spot"] = (task.S[idx][:, :, None, :].expand(k, task.A, task.M, task.N1)
                         .reshape(-1, task.N1).cpu().numpy())                    # (B, N+1)
    return res


def env_index(task: Task):
    """(path, asset, strike, K) of every env in the (path, asset, strike) flattening order."""
    p, a, m = np.meshgrid(np.arange(task.P), np.arange(task.A), np.arange(task.M), indexing="ij")
    K = task.K.cpu().numpy()[a, m]
    return {"env_path": p.ravel(), "env_asset": a.ravel(), "env_strike": m.ravel(), "env_K": K.ravel()}


# --------------------------------------------------------------------------
# Reference strategies on the same paths (sanity baselines)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_baseline(task: Task, kind: str, param=None, chunk=256):
    """Classical strategies on the same paths and costs as the agents.

    'no_hedge'   : never trade.
    'model_delta': rebalance daily to the model delta of the call (BS N(d1) or Heston P1);
                   call products only (needs bank.delta).
    'mv_delta'   : Heston minimum-variance delta P1 + (rho xi / S) dC/dv, which also hedges the
                   variance move correlated with the spot (rho = -0.95 in the S&P fits);
                   Heston call only (needs bank.delta_mv).
    'ww_band'    : Whalley-Wilmott no-trade band around the BS delta (BSM call only): only trade
                   to the nearest band edge, half-width H = param * (exp(-r tau) lam S Gamma^2)^(1/3),
                   lam = underlying fee; param absorbs the unknown risk aversion (tune on validation).
    Returns (metrics, per-env arrays)."""
    pnls, costs, trades = [], [], []
    for lo in range(0, task.P, chunk):
        idx = torch.arange(lo, min(task.P, lo + chunk), device=task.device)
        k = idx.numel()
        B = k * task.A * task.M
        X = task.X[idx].reshape(B, task.N1, task.I)
        L = task.L[idx].reshape(B, task.N1)
        band = None
        if kind == "no_hedge":
            target = torch.zeros(B, task.N, task.I, device=task.device)
        elif kind == "model_delta":
            if task.D is None:
                raise ValueError("model_delta needs a call bank with stored deltas")
            target = task.D[idx][..., :task.N].reshape(B, task.N, 1)
        elif kind == "mv_delta":
            if task.Dmv is None:
                raise ValueError("mv_delta needs a Heston call bank with stored minimum-variance deltas")
            target = task.Dmv[idx][..., :task.N].reshape(B, task.N, 1)
        elif kind == "ww_band":
            if task.sigma is None or task.product != "call":
                raise ValueError("ww_band is defined for the Black-Scholes call only")
            S = task.S[idx][:, :, None, :task.N].expand(k, task.A, task.M, task.N).double()
            Kx = task.K[None, :, :, None].double()
            tau = task.T[None, None, None, :task.N].double().clamp_min(1e-6)
            sig = task.sigma[None, :, None, None].double()
            d1 = (torch.log(S / Kx) + (task.r + 0.5 * sig**2) * tau) / (sig * torch.sqrt(tau))
            target = (0.5 * (1 + torch.erf(d1 / math.sqrt(2.0)))).float().reshape(B, task.N, 1)
            gamma = torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) / (S * sig * torch.sqrt(tau))
            lam = float(task.fees[0])
            band = (float(param) * (torch.exp(-task.r * tau) * lam * S * gamma**2) ** (1.0 / 3.0))
            band = band.float().reshape(B, task.N, 1)
        else:
            raise ValueError(kind)
        pos = torch.zeros(B, task.I, device=task.device)
        cash = L[:, 0].clone()
        cost_tot = torch.zeros(B, device=task.device)
        n_tr = torch.zeros(B, device=task.device)
        for t in range(task.N):
            final = target[:, t]
            if band is not None:
                final = torch.maximum(torch.minimum(pos, final + band[:, t]), final - band[:, t])
            trade = final - pos
            flow = trade * X[:, t]
            cost = (task.fees * flow.abs()).sum(-1)
            cash = (cash - flow.sum(-1) - cost) * task.growth
            pos = final
            cost_tot += cost
            n_tr += (trade.abs().amax(-1) > 1e-6).float()
        pv = cash + (pos * X[:, task.N]).sum(-1)
        pnls.append((pv - L[:, task.N]).cpu()); costs.append(cost_tot.cpu()); trades.append(n_tr.cpu())
    pnl, cost, n_trades = (torch.cat(v).numpy() for v in (pnls, costs, trades))
    m = risk_metrics(pnl)
    m.update({"mean_cost": float(cost.mean()), "mean_trades": float(n_trades.mean()),
              "no_trade_rate": float(1.0 - n_trades.mean() / task.N)})
    if param is not None:
        m["param"] = float(param)
    return m, {"pnl": pnl, "cost": cost, "n_trades": n_trades}


WW_GRID = tuple(float(x) for x in np.geomspace(0.25, 64.0, 17))


def tune_ww_band(val: Task, grid=WW_GRID, metric="cvar_95"):
    """Pick the Whalley-Wilmott band scale on the validation bank (never on test)."""
    scores = {k: evaluate_baseline(val, "ww_band", k)[0][metric] for k in grid}
    best = min(scores, key=scores.get)
    return best, scores
