import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import wandb
import argparse

# ---- Universal CUDA→NumPy safeguard ----
if not hasattr(torch.Tensor, "_orig_numpy"):
    torch.Tensor._orig_numpy = torch.Tensor.numpy

    def safe_numpy(self, *args, **kwargs):
        # transparently move tensor to CPU before numpy() if it's on GPU
        if self.is_cuda:
            return self.detach().cpu().numpy()
        return self._orig_numpy(*args, **kwargs)

    torch.Tensor.numpy = safe_numpy

import torch.nn as nn
from hedging.envs import HedgeCallHeston
from hedging.reward_utils import compute_discounted_cumsum_rewards
from hedging.logit_normal import LogitNormal
from hedging.plot_utils import plot_portfolio_vs_option_price
from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp
from experiments.utils.joint_policy import JointPolicy
from experiments.utils.training_loop import action_training, actor_inactor_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    TrainingConfig,
    PPOConfig,
    EnvConfig,
    train_test_split
)

from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


def main():
    print("Program started")
    wandb.init(name=os.path.splitext(os.path.basename(__file__))[0])
    print("wandb seems to be working")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    trn_cfg = TrainingConfig()

    # Load Heston parameters
    train, test = train_test_split(dynamics="heston", train_size=4, market="sp500")
    params, S0, K, v0 = train

    # Use config values for other parameters
    r = env_cfg.r
    maturity = env_cfg.maturity
    num_paths = env_cfg.num_paths
    num_steps = env_cfg.num_steps
    history_len = env_cfg.history_len
    feature_dim = 11  
    hidden_dim = int(trn_cfg.hidden_size / np.sqrt(2))
    action_dim = 1
    
    # PPO parameters from config
    clip_param = ppo_cfg.clip_param
    value_coef = ppo_cfg.value_coef
    entropy_coeff = ppo_cfg.entropy_coeff
    gamma = ppo_cfg.gamma
    lmbda = ppo_cfg.lmbda

    base_env = HedgeCallHeston(
        S0=S0, K=K, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
        kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
        num_steps=num_steps, num_paths=num_paths, history_len=history_len
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GymWrapper(base_env, device=device)
    env.reset(seed=0)

    action_model, inaction_model = generate_actor_inactor_mlp(
        feat_dim=feature_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        env=env,
        device=device,
        action_low=0.0,
        action_high=1.0
    )

    action_model.to(device)
    inaction_model.to(device)

    actor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=action_model.get_value_operator(),
        shifted=True
    )
    # Note: training_loop will set this to state_value for MLP models

    actor_loss_module = ClipPPOLoss(
        actor_network=action_model.get_policy_operator(),
        critic_network=action_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    # Note: training_loop will handle setting keys for loss module

    inactor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=inaction_model.get_value_operator(),
        shifted=True
    )

    inactor_loss_module = ClipPPOLoss(
        actor_network=inaction_model.get_policy_operator(),
        critic_network=inaction_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    # Note: training_loop will handle setting keys for loss module

    lr = ppo_cfg.learning_rate
    optim_actor = torch.optim.Adam(actor_loss_module.parameters(), lr=lr)
    optim_inactor = torch.optim.Adam(inactor_loss_module.parameters(), lr=lr)

    num_episodes = trn_cfg.num_episodes
    policy_epochs = trn_cfg.policy_epochs
    inaction_epochs = trn_cfg.inaction_epochs
    frames_per_batch = env.num_envs * num_steps
    sub_batch_num = trn_cfg.sub_batch_num
    sub_batch_size = frames_per_batch // sub_batch_num

    action_model = action_training(env, 
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
                    ) # TorchRL bug

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
                    initial_lr=ppo_cfg.learning_rate_ex,
                )
    # Test

    action_model, inaction_model = train_stats["action_model"], train_stats["inaction_model"]

    params, S0, K, v0 = test
    base_env = HedgeCallHeston(
        S0=S0, K=K, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
        kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
        num_steps=num_steps, num_paths=num_paths, history_len=history_len
    )

    joint_policy = JointPolicy(
        action_model.get_policy_operator(),
        inaction_model.get_policy_operator(),
        action_dim,
        device,
    )

    # Use test_model to evaluate and log results to wandb
    test_model(base_env, joint_policy, num_steps, device, plotting=True)

if __name__ == "__main__":
    main()
