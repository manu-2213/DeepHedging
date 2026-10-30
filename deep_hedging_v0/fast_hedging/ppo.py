"""Vectorised hedging environment + PPO for the single net and the actor/inactor pair.

Environment semantics are those of hedging/env_hedge_*.py:
  t = 0..N-1: choose position a_t in each hedging instrument,
  cash_t  -= sum_i (a_t - h_t)_i X_t,i + fee_i |(a_t - h_t)_i X_t,i|
  cash_t+1 = cash_t * exp(r dt),  PV_t+1 = cash_t+1 + sum_i a_t,i X_t+1,i
  reward_t = -|PV_t+1 - L_t+1|   (optionally divided by the claim's premium)
Terminal hedging P&L = PV_N - L_N.

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
    rnn_mode: str = "recurrent"       # recurrent (LSTM over the whole path) | window (old sliding window)
    net: str = "single"               # single | double
    window: int = 10                  # sliding-window length for rnn_mode=window (old history_len_rnn)
    hidden: int = 64
    underlying_fee: float = 0.01
    option_fee: float = 0.05
    agent_state: bool = True          # feed holdings, tracking error, costs, time since trade (+ proposed trade)
    reward_norm: str = "premium"      # premium | spot | none
    normalize_obs: bool = True
    normalize_adv: bool = True
    mask_held_actor_loss: bool = True # do not credit the actor for actions that were never executed
    inactor_init_hold_prob: float | None = None
    gamma: float = 0.99
    lmbda: float = 0.95
    clip: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    paths_per_iter: int = 64          # paths drawn per asset per iteration
    ppo_epochs: int = 4
    minibatch: int = 16384            # samples per gradient step (recurrent: minibatch // N whole paths)
    actor_update_every: int = 3       # joint phase: actor updated every k-th iteration (old protocol)
    eval_every: int = 5               # iterations between validation evaluations
    patience: int = 20                # evaluations without improvement before stopping
    min_rel_improvement: float = 1e-3
    max_iters: int = 3000             # per phase
    min_iters: int = 50
    metric: str = "cvar_95"
    eval_chunk_paths: int = 256


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
        if self.window > 1:
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
        h = cfg.hidden if cfg.net == "single" else int(cfg.hidden / math.sqrt(2))
        self.actor = ActorCritic(mode, F_, A, h, low, high,
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


def _encode_all(encoder, task, obs, B, N, chunk=65536):
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
            record=True, gen=None):
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
        rec = {"u": z(A), "logp": z(), "value": z(), "reward": z(), "executed": z()}
        if cfg.agent_state:
            rec["xa"] = z(actor_state_dim(A))
        if use_inactor:
            rec.update({"hold": z(), "hold_logp": z(), "i_value": z()})
            if cfg.agent_state:
                rec["xi"] = z(inactor_state_dim(A))
    abs_err = torch.zeros(B, device=dev)
    holds = torch.zeros(B, device=dev)
    for t in range(N):
        xa = None
        if cfg.agent_state:
            xa = torch.cat([pos, ((pv - L[:, t]) * ps)[:, None], (cum_cost * ps)[:, None],
                            (since / N)[:, None]], dim=-1).clamp(-10.0, 10.0)
        dist, value = agent.actor.heads(feat_a[:, t], xa)
        u = torch.normal(dist.loc, dist.scale, generator=gen) if actor_stochastic else dist.mode()
        a = dist.squash(u)
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
        pv = cash + (final * X[:, t + 1]).sum(-1)
        err = (pv - L[:, t + 1]).abs()
        if record:
            rec["u"][:, t], rec["logp"][:, t], rec["value"][:, t] = u, dist.log_prob(u), value
            rec["reward"][:, t] = -err * scale
            rec["executed"][:, t] = 1.0 if not use_inactor else 1.0 - hold
            if cfg.agent_state:
                rec["xa"][:, t] = xa
            if use_inactor:
                rec["hold"][:, t], rec["hold_logp"][:, t], rec["i_value"][:, t] = hold, idist.log_prob(hold), ivalue
                if cfg.agent_state:
                    rec["xi"][:, t] = xi
        traded = (trade.abs().amax(-1) > 1e-6).float()
        since = (since + 1.0) * (1.0 - traded)
        abs_err += err
        cum_cost += cost
        pos = final
    pnl = pv - L[:, N]
    return rec, {"pnl": pnl, "abs_err": abs_err / N, "cost": cum_cost, "hold_rate": holds / N}


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
    stats = []
    for j, feat in _minibatches(model, task, data["obs"], B, N, cfg, gen):
        dist, v = model.heads(feat, None if state is None else state[j])
        logp = dist.log_prob(u[j]) if which == "actor" else dist.log_prob(hold[j])
        m = mask[j]
        a = adv[j]
        if cfg.normalize_adv:
            mu = _masked_mean(a, m)
            sd = torch.sqrt(_masked_mean((a - mu) ** 2, m)).clamp_min(1e-8)
            a = (a - mu) / sd
        ratio = torch.exp(logp - logp_old[j])
        obj = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
        loss_obj = _masked_mean(obj, m)
        loss_v = cfg.value_coef * F.smooth_l1_loss(v, ret[j])
        loss_e = -cfg.entropy_coef * dist.entropy().mean()
        _ppo_step(opt, params, loss_obj + loss_v + loss_e, cfg.max_grad_norm)
        stats.append(float(loss_obj.detach()))
    return float(np.mean(stats)) if stats else float("nan")


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
    outs = {k: [] for k in ("pnl", "abs_err", "cost", "hold_rate")}
    for lo in range(0, task.P, agent.cfg.eval_chunk_paths):
        idx = torch.arange(lo, min(task.P, lo + agent.cfg.eval_chunk_paths), device=task.device)
        _, o = rollout(agent, task, task.batch(idx), use_inactor=use_inactor, actor_stochastic=actor_stochastic,
                       inactor_stochastic=inactor_stochastic, record=False, gen=gen)
        for k in outs:
            outs[k].append(o[k].cpu())
    arr = {k: torch.cat(v).numpy() for k, v in outs.items()}
    m = risk_metrics(arr["pnl"])
    m.update({"mean_abs_tracking_error": float(arr["abs_err"].mean()), "mean_cost": float(arr["cost"].mean()),
              "hold_rate": float(arr["hold_rate"].mean())})
    return m, arr["pnl"]


# --------------------------------------------------------------------------
# Training phases with early stopping
# --------------------------------------------------------------------------
def train_phase(agent: Agent, train: Task, val: Task, *, phase, lr_actor, lr_inactor=None, seed=0,
                report=None, step_offset=0, log=print):
    """phase 'actor': train the actor alone (the single net, or stage 1 of the double net).
    phase 'joint': train actor+inactor together (stage 2 of the double net).
    Stops when the validation metric has not improved for `patience` evaluations
    and restores the best weights. Returns (best_metric, best_val_metrics, history)."""
    cfg = agent.cfg
    use_inactor = phase == "joint"
    gen = torch.Generator(device=train.device).manual_seed(seed)
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=lr_actor)
    opt_i = torch.optim.Adam(agent.inactor.parameters(), lr=lr_inactor) if use_inactor else None
    best, best_state, best_metrics, bad, history = math.inf, agent.state_dict(), None, 0, []
    k = min(cfg.paths_per_iter, train.P)
    t0 = time.time()
    n_eval = 0
    for it in range(1, cfg.max_iters + 1):
        idx = torch.randperm(train.P, device=train.device, generator=gen)[:k]
        data = train.batch(idx)
        rec, out = rollout(agent, train, data, use_inactor=use_inactor, actor_stochastic=True, gen=gen)
        if use_inactor:
            update(agent, train, data, rec, which="inactor", opt=opt_i, gen=gen)
        if not use_inactor or cfg.actor_update_every == 1 or it % cfg.actor_update_every == 1:
            update(agent, train, data, rec, which="actor", opt=opt_a, gen=gen)
        if it % cfg.eval_every == 0:
            n_eval += 1
            m, _ = evaluate(agent, val, use_inactor=use_inactor)
            metric = m[cfg.metric]
            if not math.isfinite(metric):
                log(f"  [{phase}] non-finite validation metric at iter {it}; stopping")
                break
            history.append({"iter": it, "phase": phase, "train_abs_err": float(out["abs_err"].mean()),
                            **{f"val_{a}": b for a, b in m.items()}, "elapsed_s": time.time() - t0})
            if metric < best - cfg.min_rel_improvement * abs(best if math.isfinite(best) else metric):
                best, best_state, best_metrics, bad = metric, agent.state_dict(), m, 0
            else:
                bad += 1
            log(f"  [{phase}] it {it:5d}  val {cfg.metric}={metric:.4f}  best={best:.4f}  "
                f"hold={m['hold_rate']:.3f}  cost={m['mean_cost']:.2f}  bad={bad}/{cfg.patience}  "
                f"{(time.time() - t0) / it:.2f}s/it")
            if report is not None:
                report(step_offset + n_eval, metric)        # may raise TrialPruned
            if bad >= cfg.patience and it >= cfg.min_iters:
                break
    agent.load_state_dict(best_state)
    return best, best_metrics, history


# --------------------------------------------------------------------------
# Reference strategies on the same paths (sanity baselines)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_baseline(task: Task, kind: str, chunk=256):
    """'no_hedge': never trade.  'bs_delta': daily Black-Scholes delta hedge (BSM call only)."""
    pnls, costs = [], []
    for lo in range(0, task.P, chunk):
        idx = torch.arange(lo, min(task.P, lo + chunk), device=task.device)
        k = idx.numel()
        B = k * task.A * task.M
        X = task.X[idx].reshape(B, task.N1, task.I)
        L = task.L[idx].reshape(B, task.N1)
        if kind == "no_hedge":
            pos_all = torch.zeros(B, task.N, task.I, device=task.device)
        elif kind == "bs_delta":
            S = task.S[idx][:, :, None, :task.N].expand(k, task.A, task.M, task.N)
            Kx = task.K[None, :, :, None]
            tau = task.T[None, None, None, :task.N].clamp_min(1e-8)
            sig = task.sigma[None, :, None, None]
            d1 = (torch.log(S / Kx) + (task.r + 0.5 * sig**2) * tau) / (sig * torch.sqrt(tau))
            pos_all = (0.5 * (1 + torch.erf(d1 / math.sqrt(2.0)))).reshape(B, task.N, 1)
        else:
            raise ValueError(kind)
        pos = torch.zeros(B, task.I, device=task.device)
        cash = L[:, 0].clone()
        cost_tot = torch.zeros(B, device=task.device)
        for t in range(task.N):
            final = pos_all[:, t]
            flow = (final - pos) * X[:, t]
            cost = (task.fees * flow.abs()).sum(-1)
            cash = (cash - flow.sum(-1) - cost) * task.growth
            pos = final
            cost_tot += cost
        pv = cash + (pos * X[:, task.N]).sum(-1)
        pnls.append((pv - L[:, task.N]).cpu()); costs.append(cost_tot.cpu())
    pnl = torch.cat(pnls).numpy()
    m = risk_metrics(pnl)
    m["mean_cost"] = float(torch.cat(costs).mean())
    return m
