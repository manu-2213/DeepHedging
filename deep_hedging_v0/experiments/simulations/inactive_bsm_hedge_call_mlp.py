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

from hedging.envs import HedgeCallBS
from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp
from experiments.utils.joint_policy import JointPolicy
from experiments.utils.training_loop import action_training, actor_inactor_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    EnvConfig,
    PPOConfig,
    TrainingConfig,
    load_bsm_data,
)


from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    trn_cfg = TrainingConfig()

    

    S0 = np.array([50.0, 100.0, 200.0])
    K = np.array([[45.0, 55.0], [90.0, 110.0], [180.0, 220.0]])
    maturity = 1.0
    r = 0.05
    sigma = np.array([0.15, 0.2, 0.25])
    num_paths = 30
    num_steps = 250
    history_len = 1
    feature_dim = 11
    hidden_dim = trn_cfg.hidden_size / np.sqrt(2)
    action_dim = 1 
    transaction_cost = env_cfg.transaction_cost
    transaction_fee_rate = env_cfg.transaction_fee_rate

    clip_param = 0.2
    value_coef = 0.1
    entropy_coeff = 0.001
    gamma = 0.99
    lmbda = 0.95

    base_env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate
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
    )

    action_model.to(device)
    inaction_model.to(device)

    actor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=action_model.get_value_operator(),
        shifted=True # make sure use this one for RNN
    )
    actor_advantage_module.set_keys(value="state_value")

    actor_loss_module = ClipPPOLoss(
        actor_network=action_model.get_policy_operator(),
        critic_network=action_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    actor_loss_module.set_keys(
        action="action",
        sample_log_prob="action_log_prob",
        value="state_value"
    )

    inactor_advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=inaction_model.get_value_operator(),
        shifted=True # make sure use this one for RNN
    )

    inactor_loss_module = ClipPPOLoss(
        actor_network=inaction_model.get_policy_operator(),
        critic_network=inaction_model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    inactor_loss_module.set_keys(
        action="inact",
        sample_log_prob="inact_log_prob",
        value="i_state_value"
    )

    lr=5e-5
    optim_actor = torch.optim.Adam(actor_loss_module.parameters(), lr=lr)
    optim_inactor = torch.optim.Adam(inactor_loss_module.parameters(), lr=lr)

    num_epochs = 1
    num_episodes = 20
    policy_epochs = 1
    inaction_epochs = 1
    frames_per_batch = env.num_envs * num_steps
    sub_batch_num = 10
    sub_batch_size = frames_per_batch // sub_batch_num
    frames_per_batch, sub_batch_size


    problem_name = "complex mlp"
    seed = 1

    wandb.init()

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
                )
    # Test

    action_model, inaction_model = train_stats["action_model"], train_stats["inaction_model"]

    base_env = HedgeCallBS(
        S0, K, maturity, r, sigma, 5, num_steps,
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
    test_model(base_env, joint_policy, num_steps, device, plotting=False)

if __name__ == "__main__":
    main()
