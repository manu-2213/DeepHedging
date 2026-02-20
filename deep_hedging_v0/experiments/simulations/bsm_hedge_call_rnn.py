import os
import sys
import numpy as np
import torch
import wandb
import argparse

# --- Add your project root to sys.path (like you already did) ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

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
from experiments.utils.ppo_rnn_actor import create_ppo_rnn_actor
from experiments.utils.training_loop import action_training
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
    # 1. Fixed seed, since we're just testing locally
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    S0, K, sigma = load_bsm_data()

    # 2. Your env parameters (copied from your original script)
    env_cfg = EnvConfig()
    train_cfg = TrainingConfig()
    maturity = env_cfg.maturity
    r = env_cfg.r
    num_paths = env_cfg.num_paths
    num_steps = env_cfg.num_steps
    history_len = env_cfg.history_len_rnn
    input_dim = 11
    hidden_size = train_cfg.hidden_size
    action_dim = 1
    transaction_cost = env_cfg.transaction_cost
    transaction_fee_rate = env_cfg.transaction_fee_rate

    base_env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps,
        history_len=history_len,
        transaction_cost=transaction_cost,
        transaction_fee_rate=transaction_fee_rate,
    )
    env = GymWrapper(base_env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_per_batch = env.num_envs * num_steps
    sub_batch_num = train_cfg.sub_batch_num
    sub_batch_size = frames_per_batch // sub_batch_num

    # PPO + GAE params
    ppo_cfg = PPOConfig()
    clip_param = ppo_cfg.clip_param
    value_coef = ppo_cfg.value_coef
    entropy_coeff = ppo_cfg.entropy_coeff
    gamma = ppo_cfg.gamma
    lmbda = ppo_cfg.lmbda

    model = create_ppo_rnn_actor(input_dim=input_dim, action_dim=action_dim,
                                 hidden_dim=hidden_size).to(device)

    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=model.get_value_operator(),
        shifted=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=model.get_policy_operator(),
        critic_network=model.get_value_operator(),
        clip_epsilon=clip_param,
        entropy_coeff=entropy_coeff,
        value_coef=value_coef,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=ppo_cfg.learning_rate)

    num_epochs = train_cfg.num_epochs
    num_episodes = train_cfg.num_episodes

    wandb.init(name=os.path.splitext(os.path.basename(__file__))[0])

    model = action_training(
        env,
        model,
        num_epochs,
        num_episodes,
        device,
        advantage_module,
        loss_module,
        optim,
        frames_per_batch,
        sub_batch_num,
        sub_batch_size,
        log_frquency=1,  # log every episode
    )

    test_env = HedgeCallBS(S0, K, maturity, r, sigma, num_paths, num_steps, history_len=history_len,
                           transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate)
    test_model(test_env, model, num_steps, device, plotting=False)

    # Close the W&B run cleanly
    wandb.finish()

if __name__ == "__main__":
    main()
