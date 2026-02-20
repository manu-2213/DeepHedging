import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.envs import HedgeCallHeston
from experiments.utils.ppo_mlp_actor import create_ppo_mlp_actor
from experiments.utils.training_loop import action_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    load_heston_data,
    EnvConfig,
    PPOConfig,
    TrainingConfig
)
import wandb
import torch
import warnings
import numpy as np

if not hasattr(torch.Tensor, "_orig_numpy"):
    torch.Tensor._orig_numpy = torch.Tensor.numpy

    def safe_numpy(self, *args, **kwargs):
        # transparently move tensor to CPU before numpy() if it's on GPU
        if self.is_cuda:
            warnings.warn("Calling .numpy() on CUDA tensor", stacklevel=2)
            return self.detach().cpu().numpy()
        return self._orig_numpy(*args, **kwargs)

    torch.Tensor.numpy = safe_numpy

from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# --- Env Parameters ---
params, S0, K, v0 = load_heston_data()
env_cfg = EnvConfig()
ppo_cfg = PPOConfig()
train_cfg = TrainingConfig()

r = env_cfg.r
maturity = env_cfg.maturity
trap = env_cfg.trap
num_paths = env_cfg.num_paths_heston
num_steps = env_cfg.num_steps
history_len = env_cfg.history_len
input_dim = 11
hidden_size = train_cfg.hidden_size
action_dim = 1
transaction_cost = env_cfg.transaction_cost
transaction_fee_rate = env_cfg.transaction_fee_rate


base_env = HedgeCallHeston(
    S0=S0, K = K, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=num_paths, history_len=history_len, 
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)

env = GymWrapper(base_env)

frames_per_batch = env.num_envs * num_steps 
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

act_spec = env.specs["input_spec", "full_action_spec", "action"].to(device)

sub_batch_num = train_cfg.sub_batch_num
sub_batch_size = frames_per_batch // sub_batch_num
frames_per_batch, sub_batch_size

# Param for PPO
clip_param = ppo_cfg.clip_param
value_coef = ppo_cfg.value_coef
entropy_coeff = ppo_cfg.entropy_coeff
# Param for GAE
gamma = ppo_cfg.gamma
lmbda = ppo_cfg.lmbda

wandb.init(name=os.path.splitext(os.path.basename(__file__))[0])

model = create_ppo_mlp_actor(input_dim=input_dim, action_dim=action_dim,
                             hidden_dim=hidden_size, device=device, action_low=0.0, action_high=1.0)

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

optim = torch.optim.Adam(loss_module.parameters(),lr=ppo_cfg.learning_rate)

num_epochs = train_cfg.num_epochs
num_episodes = train_cfg.num_episodes

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
    num_steps=num_steps, num_paths=num_paths, history_len=history_len,
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)
test_model(base_env, model, num_steps, device, plotting=True)
