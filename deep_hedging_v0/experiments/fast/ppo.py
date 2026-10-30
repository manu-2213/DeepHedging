"""Rollouts, GAE and clipped-PPO updates for the single net and the actor + inaction gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from experiments.fast.config import PPOConfig
from experiments.fast.env import HedgeBatch, HedgeState, ObsBuilder, make_batch
from experiments.fast.path_bank import Bank


@dataclass
class Rollout:
    obs: torch.Tensor          # (N, E, ...) observation fed to both nets
    z: torch.Tensor            # (N, E, D) actor pre-tanh sample (the proposed action)
    logp_a: torch.Tensor       # (N, E)
    value_a: torch.Tensor      # (N, E)
    reward: torch.Tensor       # (N, E) raw reward, -|PV - V|
    mask: torch.Tensor | None = None     # (N, E, D) gate decision, 1 = hold
    logp_g: torch.Tensor | None = None
    value_g: torch.Tensor | None = None


@dataclass
class EpisodeResult:
    pnl: torch.Tensor          # (E,) terminal P&L = PV_N - V_N
    costs: torch.Tensor        # (E,) total transaction costs paid
    trade_steps: torch.Tensor  # (E,) number of steps with a non-zero trade
    mean_abs_error: float      # average |PV_t - V_t| per step, the training objective
    hold_rate: float           # fraction of (step, instrument) decisions that kept the position
    rollout: Rollout | None = None


@torch.no_grad()
def run_episode(batch: HedgeBatch, actor, gate, arch, ppo: PPOConfig, r, dt,
                deterministic=False, gate_mode="sample", store=False) -> EpisodeResult:
    """gate_mode: 'sample' or 'deterministic' (ignored when gate is None)."""
    state = HedgeState(batch, r, dt)
    builder = ObsBuilder(batch, arch, ppo.rnn_history, ppo.trade_features)
    obs = builder.reset(state)
    N = batch.num_steps
    keep = {k: [] for k in ("obs", "z", "logp_a", "value_a", "reward", "mask", "logp_g", "value_g")}
    dev = batch.V.device
    abs_err_sum = torch.zeros((), device=dev)   # tensors, so the loop never syncs with the GPU
    hold_sum = torch.zeros((), device=dev)
    pv = None
    for t in range(N):
        action, z, logp_a, value_a = actor.act(obs, deterministic=deterministic)
        if gate is not None:
            mask, logp_g, value_g = gate.act(obs, deterministic=(gate_mode == "deterministic"))
            final = torch.where(mask > 0.5, state.hold, action)
            hold_sum += mask.mean()
        else:
            final = action
        reward, pv = state.step(final)
        abs_err_sum -= reward.mean()
        if store:
            keep["obs"].append(obs); keep["z"].append(z); keep["logp_a"].append(logp_a)
            keep["value_a"].append(value_a); keep["reward"].append(reward)
            if gate is not None:
                keep["mask"].append(mask); keep["logp_g"].append(logp_g); keep["value_g"].append(value_g)
        if t < N - 1:
            obs = builder.next(state)
    rollout = None
    if store:
        stacked = {k: torch.stack(v) for k, v in keep.items() if v}
        rollout = Rollout(**stacked)
    return EpisodeResult(
        pnl=pv - batch.V[:, N], costs=state.costs, trade_steps=state.trade_steps,
        mean_abs_error=float(abs_err_sum) / N, hold_rate=float(hold_sum) / N if gate is not None else 0.0,
        rollout=rollout,
    )


@torch.no_grad()
def gae(reward, value, gamma, lmbda):
    """GAE with a terminal state after the last step (value 0), as TorchRL's GAE with done=True."""
    N = reward.shape[0]
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(reward[0])
    for t in reversed(range(N)):
        next_value = value[t + 1] if t + 1 < N else torch.zeros_like(value[t])
        delta = reward[t] + gamma * next_value - value[t]
        last = delta + gamma * lmbda * last
        adv[t] = last
    return adv, adv + value


def ppo_update(net, optim, obs, act, old_logp, adv, ret, ppo: PPOConfig) -> dict:
    obs = obs.reshape(-1, *obs.shape[2:])
    act = act.reshape(-1, act.shape[-1])
    old_logp, adv, ret = old_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
    n = adv.numel()
    mb = max(1, n // ppo.num_minibatches)
    stats = {"loss_objective": 0.0, "loss_critic": 0.0, "entropy": 0.0, "clip_frac": 0.0, "skipped": 0}
    steps = 0
    for _ in range(ppo.ppo_epochs):
        perm = torch.randperm(n, device=adv.device)
        for i in range(ppo.num_minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            logp, entropy, value = net.evaluate(obs[idx], act[idx])
            a = adv[idx]
            if ppo.normalize_advantage:
                a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = torch.exp(logp - old_logp[idx])
            objective = -torch.min(ratio * a, ratio.clamp(1 - ppo.clip, 1 + ppo.clip) * a).mean()
            critic = F.smooth_l1_loss(value, ret[idx])          # TorchRL ClipPPOLoss default
            loss = objective + ppo.value_coef * critic - ppo.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                stats["skipped"] += 1
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), ppo.max_grad_norm)
            optim.step()
            steps += 1
            stats["loss_objective"] += float(objective)
            stats["loss_critic"] += float(critic)
            stats["entropy"] += float(entropy.mean())
            stats["clip_frac"] += float(((ratio - 1).abs() > ppo.clip).float().mean())
    for k in ("loss_objective", "loss_critic", "entropy", "clip_frac"):
        stats[k] /= max(1, steps)
    return stats


# ----------------------------------------------------------------------- evaluation
def risk_metrics(pnl: np.ndarray) -> dict:
    """Same conventions as experiments/utils/testing.py (loss = -P&L, positive = bad)."""
    x = pnl.astype(np.float64).ravel()
    loss = -x
    var95, var99 = np.percentile(loss, 95), np.percentile(loss, 99)
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        "var_95": float(var95), "var_99": float(var99),
        "cvar_95": float(loss[loss >= var95].mean()), "cvar_99": float(loss[loss >= var99].mean()),
        "min": float(x.min()), "max": float(x.max()),
    }


@torch.no_grad()
def evaluate(bank: Bank, actor, gate, arch, ppo, market, gate_mode="deterministic", chunk_paths=1000) -> dict:
    """Deterministic actor on every path of a bank (in chunks). Returns risk metrics + diagnostics."""
    was_training = [(m, m.training) for m in (actor, gate) if m is not None]
    for m, _ in was_training:
        m.eval()
    dt = market.maturity / market.num_steps
    pnl, costs, trades, err, hold, weights = [], [], [], 0.0, 0.0, 0
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(0)  # stochastic-gate evaluation is reproducible and does not disturb training RNG
    try:
        for start in range(0, bank.num_paths, chunk_paths):
            idx = torch.arange(start, min(start + chunk_paths, bank.num_paths), device=bank.S.device)
            batch = make_batch(bank, idx, market.r, market.underlying_fee, market.option_fee)
            res = run_episode(batch, actor, gate, arch, ppo, market.r, dt,
                              deterministic=True, gate_mode=gate_mode)
            n = batch.num_envs
            pnl.append(res.pnl.cpu()); costs.append(res.costs.cpu()); trades.append(res.trade_steps.cpu())
            err += res.mean_abs_error * n; hold += res.hold_rate * n; weights += n
    finally:
        torch.random.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for m, mode in was_training:
            m.train(mode)
    pnl = torch.cat(pnl).numpy()
    out = risk_metrics(pnl)
    if not np.isfinite(pnl).all():
        out["cvar_95"] = float("nan")
    out.update({
        "mean_abs_tracking_error": err / weights,
        "mean_transaction_cost": float(torch.cat(costs).mean()),
        "trade_frequency": float(torch.cat(trades).mean()) / market.num_steps,
        "hold_rate": hold / weights,
        "num_options": int(weights),
    })
    return out
