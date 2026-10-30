"""Training with early stopping on validation CVaR95, shared by Optuna trials and final seeds."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
import time

import torch

from experiments.fast.config import MarketConfig, PPOConfig, StopConfig
from experiments.fast.env import make_batch, obs_dim
from experiments.fast.models import HedgeActor, InactionGate, action_bounds
from experiments.fast.path_bank import Bank
from experiments.fast.ppo import evaluate, gae, ppo_update, run_episode


@dataclass
class TrainResult:
    best_val: float
    best_iter: int
    iters: int
    stop_reason: str
    seconds: float
    best_metrics: dict
    actor_state: dict
    gate_state: dict | None = None
    history: list = field(default_factory=list)


class EarlyStopper:
    def __init__(self, stop: StopConfig):
        self.cfg, self.best, self.bad, self.evals = stop, math.inf, 0, 0

    def update(self, value: float) -> bool:
        """Records one evaluation; True if it is a new best."""
        self.evals += 1
        if math.isfinite(value) and value < self.best - self.cfg.min_rel_improvement * abs(self.best if math.isfinite(self.best) else 0.0):
            self.best, self.bad = value, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.evals >= self.cfg.min_evals and self.bad >= self.cfg.patience


def build_actor(arch, product, bank: Bank, ppo: PPOConfig, device):
    low, high = action_bounds(product)
    return HedgeActor(arch, obs_dim(bank, ppo.trade_features), bank.action_dim, ppo.hidden, low, high).to(device)


def build_gate(arch, bank: Bank, ppo: PPOConfig, device):
    return InactionGate(arch, obs_dim(bank, ppo.trade_features), bank.action_dim, ppo.hidden, ppo.gate_init_logit).to(device)


def reward_scale(train_bank: Bank) -> float:
    """Rewards are in index points (S0 ~ 2000-3700). Train on % of spot so the critic is well scaled;
    a positive constant does not change the optimal policy. Reported metrics stay in raw units."""
    return 100.0 / float(train_bank.S0.mean())


def train(mode, arch, product, banks: dict, market: MarketConfig, ppo: PPOConfig, stop: StopConfig,
          seed: int, device, actor_lr: float, gate_lr: float | None = None,
          init_actor_state: dict | None = None, trial=None, log=print, tag="") -> TrainResult:
    """mode='single': train the hedge actor alone.
    mode='double': start from a trained single-net actor, add the inaction gate, train jointly
    (gate every iteration, actor every `actor_update_every` iterations, as in actor_inactor_training)."""
    torch.manual_seed(seed)
    train_bank, val_bank = banks["train"], banks["val"]
    actor = build_actor(arch, product, train_bank, ppo, device)
    if init_actor_state is not None:
        actor.load_state_dict(init_actor_state)
    gate = build_gate(arch, train_bank, ppo, device) if mode == "double" else None
    opt_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    opt_gate = torch.optim.Adam(gate.parameters(), lr=gate_lr) if gate is not None else None

    scale = reward_scale(train_bank)
    dt = market.maturity / market.num_steps
    sampler = torch.Generator().manual_seed(seed)
    stopper = EarlyStopper(stop)
    history, best_metrics = [], {}
    best_state = (copy.deepcopy(actor.state_dict()), copy.deepcopy(gate.state_dict()) if gate else None)
    best_iter, stop_reason, t0 = 0, "max_iters", time.time()

    def validate(it):
        nonlocal best_state, best_iter, best_metrics
        metrics = evaluate(val_bank, actor, gate, arch, ppo, market, gate_mode="deterministic")
        value = metrics["cvar_95"]
        improved = stopper.update(value)
        if improved:
            best_state = (copy.deepcopy(actor.state_dict()), copy.deepcopy(gate.state_dict()) if gate else None)
            best_iter, best_metrics = it, metrics
        row = {"iter": it, "val_cvar_95": value, "val_mean": metrics["mean"],
               "val_tracking_error": metrics["mean_abs_tracking_error"],
               "val_cost": metrics["mean_transaction_cost"], "val_trade_freq": metrics["trade_frequency"],
               "val_hold_rate": metrics["hold_rate"], "best": stopper.best, "seconds": time.time() - t0}
        history.append(row)
        log(f"{tag} it={it:5d} val_cvar95={value:10.4f} best={stopper.best:10.4f} "
            f"err={metrics['mean_abs_tracking_error']:.4f} cost={metrics['mean_transaction_cost']:.3f} "
            f"trade_freq={metrics['trade_frequency']:.3f} hold={metrics['hold_rate']:.3f} "
            f"bad={stopper.bad}/{stop.patience} {time.time() - t0:.0f}s")
        if trial is not None and math.isfinite(value):
            trial.report(value, step=it)
            if trial.should_prune():
                from optuna import TrialPruned
                raise TrialPruned()
        return value

    validate(0)  # double net: the starting point is the pretrained single-net actor
    it = 0
    for it in range(1, stop.max_iters + 1):
        idx = torch.randperm(train_bank.num_paths, generator=sampler)[:ppo.batch_paths].to(device)
        batch = make_batch(train_bank, idx, market.r, market.underlying_fee, market.option_fee)
        roll = run_episode(batch, actor, gate, arch, ppo, market.r, dt, store=True).rollout
        reward = roll.reward * scale
        if gate is not None:
            adv, ret = gae(reward, roll.value_g, ppo.gamma, ppo.lmbda)
            ppo_update(gate, opt_gate, roll.obs, roll.mask, roll.logp_g, adv, ret, ppo)
        if gate is None or (it - 1) % ppo.actor_update_every == 0:
            adv, ret = gae(reward, roll.value_a, ppo.gamma, ppo.lmbda)
            ppo_update(actor, opt_actor, roll.obs, roll.z, roll.logp_a, adv, ret, ppo)
        del roll, batch
        if it % stop.eval_every == 0:
            value = validate(it)
            if not math.isfinite(value) and not math.isfinite(stopper.best):
                stop_reason = "diverged"
                break
            if stopper.should_stop:
                stop_reason = "patience"
                break

    actor.load_state_dict(best_state[0])
    if gate is not None:
        gate.load_state_dict(best_state[1])
    return TrainResult(best_val=stopper.best, best_iter=best_iter, iters=it, stop_reason=stop_reason,
                       seconds=time.time() - t0, best_metrics=best_metrics,
                       actor_state={k: v.cpu() for k, v in best_state[0].items()},
                       gate_state={k: v.cpu() for k, v in best_state[1].items()} if gate else None,
                       history=history)


def test_metrics(mode, arch, product, banks, market, ppo, result: TrainResult, device) -> dict:
    """Held-out test year. Double net: deterministic gate (used for selection) and sampled gate."""
    actor = build_actor(arch, product, banks["train"], ppo, device)
    actor.load_state_dict(result.actor_state)
    gate = None
    if mode == "double":
        gate = build_gate(arch, banks["train"], ppo, device)
        gate.load_state_dict(result.gate_state)
    out = {"test": evaluate(banks["test"], actor, gate, arch, ppo, market, gate_mode="deterministic"),
           "val": evaluate(banks["val"], actor, gate, arch, ppo, market, gate_mode="deterministic")}
    if gate is not None:
        out["test_sampled_gate"] = evaluate(banks["test"], actor, gate, arch, ppo, market, gate_mode="sample")
    return out


def save_result(path, result: TrainResult, extra: dict):
    torch.save({"actor_state": result.actor_state, "gate_state": result.gate_state, **extra}, path)
    with open(path.with_suffix(".history.json"), "w", encoding="utf-8") as f:
        json.dump(result.history, f, indent=1, allow_nan=True)
