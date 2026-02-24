import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.envs import HedgeDocHeston
from experiments.utils.ppo_rnn_actor import create_ppo_rnn_actor_exotic
from experiments.utils.training_loop import action_training
from experiments.utils.testing import test_model
from experiments.utils.sim_config import (
    train_test_split,
    EnvConfig,
    PPOConfig,
    TrainingConfig,
    compute_barriers
)
import torch
import wandb

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


train, test = train_test_split(dynamics="heston", train_size=4, market="sp500")
params, S0, K, v0 = train
H = compute_barriers(K)

env_cfg = EnvConfig()
ppo_cfg = PPOConfig()
train_cfg = TrainingConfig()

r = env_cfg.r
maturity = env_cfg.maturity
trap = env_cfg.trap
num_paths = env_cfg.num_paths_heston
num_steps = env_cfg.num_steps
history_len = env_cfg.history_len_rnn
input_dim = 17
hidden_size = train_cfg.hidden_size
action_dim = 2
transaction_cost = env_cfg.transaction_cost
transaction_fee_rate = env_cfg.transaction_fee_rate

base_env = HedgeDocHeston(
    S0=S0, K = K, H=H, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=num_paths, history_len=history_len,
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)

env = GymWrapper(base_env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
act_spec = env.specs["input_spec", "full_action_spec", "action"].to(device)

frames_per_batch = env.num_envs * num_steps
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

model = create_ppo_rnn_actor_exotic(input_dim=input_dim, hidden_dim=hidden_size, 
                                    action_dim=action_dim)

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
                        )

# Test
params, S0, K, v0 = test
H = compute_barriers(K)
base_env = HedgeDocHeston(
    S0=S0, K = K, H=H, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=num_paths, history_len=history_len,
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)

test_model(base_env, model, num_steps, device, plotting=True)
