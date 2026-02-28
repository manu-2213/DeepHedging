import os
import sys
import numpy as np
import torch
import wandb
import argparse

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# ---- Universal CUDA→NumPy safeguard ----

# save the original method once
if not hasattr(torch.Tensor, "_orig_numpy"):
    torch.Tensor._orig_numpy = torch.Tensor.numpy

    def safe_numpy(self, *args, **kwargs):
        # transparently move tensor to CPU before numpy() if it's on GPU
        if self.is_cuda:
            return self.detach().cpu().numpy()
        return self._orig_numpy(*args, **kwargs)

    torch.Tensor.numpy = safe_numpy

from hedging.envs import HedgeConcBS
from experiments.utils.actor_inactor_rnn import generate_actor_inactor_rnn_exotic
from experiments.utils.joint_policy import JointPolicy
from experiments.utils.training_loop import action_training, actor_inactor_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    EnvConfig,
    PPOConfig,
    TrainingConfig,
    train_test_split,
    default_concentration_matrix,
)


from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    wandb.init(name=os.path.splitext(os.path.basename(__file__))[0])
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    trn_cfg = TrainingConfig()

    train, test = train_test_split(dynamics="bsm", train_size=4, market="sp500")
    S0, K, sigma = train
    maturity = env_cfg.maturity
    r = env_cfg.r
    num_paths = env_cfg.num_paths
    num_steps = env_cfg.num_steps
    history_len = env_cfg.history_len_rnn
    feature_dim = None  # will be inferred from env
    hidden_dim = int(trn_cfg.hidden_size / np.sqrt(2))
    action_dim = 2
    transaction_cost = env_cfg.transaction_cost
    transaction_fee_rate = env_cfg.transaction_fee_rate
    P = default_concentration_matrix(size=1)

    clip_param = ppo_cfg.clip_param
    value_coef = ppo_cfg.value_coef
    entropy_coeff = ppo_cfg.entropy_coeff
    gamma = ppo_cfg.gamma
    lmbda = ppo_cfg.lmbda

    base_env = HedgeConcBS(
        S0, K, P, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GymWrapper(base_env, device=device)
    env.reset(seed=0)

    feature_dim = env.full_observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    action_model, inaction_model = generate_actor_inactor_rnn_exotic(
        feat_dim=feature_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        env=env,
        device=device,
    )

    action_model.to(device)
    inaction_model.to(device)

    actor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=action_model.get_value_operator(),
        shifted=True
    )
    # Note: training_loop will set this to a_state_value for RNN models

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
    actor_scheduler = CosineAnnealingLR(
        optim_actor,
        T_max=max(1, policy_epochs + inaction_epochs),
        eta_min=lr / 2,
    )
    frames_per_batch = env.num_envs * num_steps
    sub_batch_num = trn_cfg.sub_batch_num
    sub_batch_size = frames_per_batch // sub_batch_num

    action_model, actor_scheduler = action_training(env, 
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
                    )

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
                    actor_scheduler=actor_scheduler,
                )
    # Test

    action_model, inaction_model = train_stats["action_model"], train_stats["inaction_model"]

    S0, K, sigma = test
    base_env = HedgeConcBS(
        S0, K, P, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate
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
