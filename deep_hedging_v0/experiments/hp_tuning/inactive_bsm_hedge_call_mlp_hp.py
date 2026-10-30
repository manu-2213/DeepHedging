"""
Hyperparameter sweep — inactive_bsm_hedge_call_mlp
===================================================
Drop-in replacement for inactive_bsm_hedge_call_mlp.py, parameterised via
CLI so that the SLURM array can vary learning rate and scheduler strategy
without touching any other training logic.

CLI arguments
-------------
--seed       int    reproducibility seed          (default 0)
--lr         float  learning rate for actor+inactor Adam  (default 2e-5)
--scheduler  str    {cosine, linear, step}         (default cosine)

Scheduler details (T_total = policy_epochs + inaction_epochs = 30)
-------------------------------------------------------------------
cosine  CosineAnnealingLR  eta_min  = lr / 10       (smooth decay to 10 %)
linear  LinearLR           end_factor = 0.1         (linear  decay to 10 %)
step    StepLR             step_size=6, gamma=0.7   (decay ×0.7 every 6 ep)
"""

import os
import sys
import numpy as np
import torch
import wandb
import argparse

# ── module root = deep_hedging_v0/ (same depth as simulations/) ──────
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ── Universal CUDA→NumPy safeguard ───────────────────────────────────
if not hasattr(torch.Tensor, "_orig_numpy"):
    torch.Tensor._orig_numpy = torch.Tensor.numpy

    def _safe_numpy(self, *args, **kwargs):
        if self.is_cuda:
            return self.detach().cpu().numpy()
        return self._orig_numpy(*args, **kwargs)

    torch.Tensor.numpy = _safe_numpy

from hedging.envs import HedgeCallBS
from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp
from experiments.utils.joint_policy import JointPolicy
from experiments.utils.training_loop import action_training, actor_inactor_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    EnvConfig,
    PPOConfig,
    TrainingConfig,
    train_test_split,
)

from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR

VALID_SCHEDULERS = ("cosine", "linear", "step")


# ─────────────────────────────────────────────────────────────────────
# Scheduler factory
# ─────────────────────────────────────────────────────────────────────

def build_actor_scheduler(name: str, optimizer, lr: float, total_steps: int):
    """
    Return a configured LR scheduler for the actor optimiser.

    Parameters
    ----------
    name        : one of ('cosine', 'linear', 'step')
    optimizer   : the Adam instance to attach the scheduler to
    lr          : initial learning rate (used to derive eta_min)
    total_steps : policy_epochs + inaction_epochs  (= 30 by default)
    """
    if name == "cosine":
        # Smooth cosine curve from lr → lr/10 over training
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=lr / 10.0,
        )
    elif name == "linear":
        # Strictly linear decay from lr → 0.1·lr
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=max(1, total_steps),
        )
    elif name == "step":
        # Multiply by 0.7 every  total_steps//5  epochs  (≈6 steps for T=30)
        step_size = max(1, total_steps // 5)
        return StepLR(optimizer, step_size=step_size, gamma=0.7)
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Valid choices: {VALID_SCHEDULERS}"
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BSM call inactive-network sweep: lr × scheduler"
    )
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--lr",          type=float, default=2e-5,
                        help="Actor Adam learning rate")
    parser.add_argument("--inactor_lr",  type=float, default=2e-5,
                        help="Inactor Adam learning rate (swept independently)")
    parser.add_argument("--scheduler",   type=str,   default="cosine",
                        choices=VALID_SCHEDULERS,
                        help="LR decay strategy for the actor scheduler")
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── WandB run name encodes the full config ────────────────────────
    run_name = (
        f"hp_bsm_call_mlp"
        f"_lr{args.lr:.0e}"
        f"_{args.scheduler}"
        f"_ilr{args.inactor_lr:.0e}"
        f"_s{args.seed}"
    )
    wandb.init(
        name=run_name,
        config={
            "lr":          args.lr,
            "inactor_lr":  args.inactor_lr,
            "scheduler":   args.scheduler,
            "seed":        args.seed,
        },
    )

    # ── Configs ──────────────────────────────────────────────────────
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    trn_cfg = TrainingConfig()

    train, test = train_test_split(dynamics="bsm", train_size=4, market="sp500")
    S0, K, sigma = train

    maturity             = env_cfg.maturity
    r                    = env_cfg.r
    num_paths            = env_cfg.num_paths
    num_steps            = env_cfg.num_steps
    history_len          = 1
    feature_dim          = 11
    hidden_dim           = int(trn_cfg.hidden_size / np.sqrt(2))
    action_dim           = 1
    transaction_cost     = env_cfg.transaction_cost
    transaction_fee_rate = env_cfg.transaction_fee_rate

    clip_param    = ppo_cfg.clip_param
    value_coef    = ppo_cfg.value_coef
    entropy_coeff = ppo_cfg.entropy_coeff
    gamma         = ppo_cfg.gamma
    lmbda         = ppo_cfg.lmbda

    # ── Environment ──────────────────────────────────────────────────
    base_env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GymWrapper(base_env, device=device)
    env.reset(seed=args.seed)

    # ── Models ───────────────────────────────────────────────────────
    action_model, inaction_model = generate_actor_inactor_mlp(
        feat_dim=feature_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        env=env,
        device=device,
        action_low=0.0,
        action_high=1.0,
    )
    action_model.to(device)
    inaction_model.to(device)

    # ── GAE modules ──────────────────────────────────────────────────
    actor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=action_model.get_value_operator(),
        shifted=True,
    )
    inactor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=inaction_model.get_value_operator(),
        shifted=True,
    )

    # ── PPO loss modules ─────────────────────────────────────────────
    actor_loss_module = ClipPPOLoss(
        actor_network=action_model.get_policy_operator(),
        critic_network=action_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )
    inactor_loss_module = ClipPPOLoss(
        actor_network=inaction_model.get_policy_operator(),
        critic_network=inaction_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    # ── Optimisers — actor and inactor use independently swept LRs ────
    lr = args.lr
    optim_actor   = torch.optim.Adam(actor_loss_module.parameters(),   lr=lr)
    optim_inactor = torch.optim.Adam(inactor_loss_module.parameters(), lr=args.inactor_lr)

    num_episodes    = trn_cfg.num_episodes
    policy_epochs   = trn_cfg.policy_epochs
    inaction_epochs = trn_cfg.inaction_epochs
    total_sched_steps = policy_epochs + inaction_epochs   # = 30

    # ── Scheduler (configurable) ──────────────────────────────────────
    actor_scheduler = build_actor_scheduler(
        args.scheduler, optim_actor, lr, total_sched_steps
    )

    frames_per_batch = env.num_envs * num_steps
    sub_batch_num    = trn_cfg.sub_batch_num
    sub_batch_size   = frames_per_batch // sub_batch_num

    # ── Phase 1: action-only pre-training ─────────────────────────────
    action_model, actor_scheduler = action_training(
        env,
        action_model,
        policy_epochs,
        num_episodes,
        device,
        actor_advantage_module,
        actor_loss_module,
        optim_actor,
        frames_per_batch,
        sub_batch_num,
        sub_batch_size,
        1,
        scheduler=actor_scheduler,
    )  # TorchRL bug: must pass scheduler here

    # ── Phase 2: joint actor + inactor training ──────────────────────
    train_stats = actor_inactor_training(
        env=env,
        action_model=action_model,
        inaction_model=inaction_model,
        actor_advantage_module=actor_advantage_module,
        inactor_advantage_module=inactor_advantage_module,
        actor_loss_module=actor_loss_module,
        inactor_loss_module=inactor_loss_module,
        optim_actor=optim_actor,
        optim_inactor=optim_inactor,
        frames_per_batch=frames_per_batch,
        num_epochs=inaction_epochs,
        num_episodes=num_episodes,
        sub_batch_num=sub_batch_num,
        sub_batch_size=sub_batch_size,
        device=device,
        action_dim=action_dim,
        initial_lr=args.inactor_lr,
        actor_scheduler=actor_scheduler,
    )

    # ── Test ─────────────────────────────────────────────────────────
    action_model   = train_stats["action_model"]
    inaction_model = train_stats["inaction_model"]

    S0, K, sigma = test
    base_env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate,
    )

    joint_policy = JointPolicy(
        action_model.get_policy_operator(),
        inaction_model.get_policy_operator(),
        action_dim,
        device,
    )

    test_model(
        base_env,
        joint_policy,
        num_steps,
        device,
        plotting=True,
        models_to_save={
            "action_model":   action_model,
            "inaction_model": inaction_model,
        },
    )


if __name__ == "__main__":
    main()
