import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.envs import HedgeCallHeston
from experiments.utils.ppo_mlp_actor import create_ppo_mlp_actor
from experiments.utils.training_loop import action_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import load_heston_data
import wandb

import numpy as np
import torch


# save the original method once
if not hasattr(torch.Tensor, "_orig_numpy"):
    torch.Tensor._orig_numpy = torch.Tensor.numpy

    def safe_numpy(self, *args, **kwargs):
        # transparently move tensor to CPU before numpy() if it's on GPU
        if self.is_cuda:
            return self.detach().cpu().numpy()
        return self._orig_numpy(*args, **kwargs)

    torch.Tensor.numpy = safe_numpy



from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# --- Env Parameters ---
params, S0, K, v0 = load_heston_data()

r = 0.01
maturity = 1.0
trap = 1
num_paths = 100
num_steps = 250
history_len = 1
input_dim = 11
hidden_size = 64
action_dim = 1
transaction_cost = True
transaction_fee_rate = 1e-3


base_env = HedgeCallHeston(
    S0=S0, K = K, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=100, num_paths=1, history_len=1, transaction_cost=transaction_cost,
    transaction_fee_rate=transaction_fee_rate
)

env = GymWrapper(base_env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
act_spec = env.specs["input_spec", "full_action_spec", "action"].to(device)

frames_per_batch = env.num_envs * num_steps
sub_batch_num = 10
sub_batch_size = frames_per_batch // sub_batch_num
frames_per_batch, sub_batch_size

# Param for PPO
clip_param = 0.2
value_coef = 0.1
entropy_coeff = 0.001
# Param for GAE
gamma = 0.99
lmbda = 0.95

wandb.init()

model = create_ppo_mlp_actor(input_dim=input_dim, action_dim=action_dim,
                             hidden_dim=hidden_size, device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=model.get_value_operator(),
    shifted=True # make sure use this one for RNN
)

loss_module = ClipPPOLoss(
    actor_network=model.get_policy_operator(),
    critic_network=model.get_value_operator(),
    clip_epsilon=clip_param,
    entropy_coeff=entropy_coeff,
    value_coef=value_coef,
)

optim = torch.optim.Adam(loss_module.parameters(),lr=1e-4)

num_epochs = 20
num_episodes = 200
model = action_training(env, 
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
                        log_frquency=1
                        )

# Test

base_env = HedgeCallHeston(
    S0=S0, K = K, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=5, history_len=history_len
)
test_model(base_env, model, num_steps, device, plotting=True)
