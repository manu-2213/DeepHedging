import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hedging.envs import HedgeDocHeston
from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp

from experiments.utils.joint_policy import JointPolicy
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


import numpy as np
import torch

from torchrl.envs import GymWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

S0 = np.array([50.0, 100.0, 200.0])
K = np.array([[52.5, 55.0], [105.0, 110.0], [210.0, 220.0]])
H = np.array([[42.5, 45.0], [85.0, 90.0], [170.0, 180.0]])  # barrier
maturity = 1.0
r = 0.05
v0 = np.array([0.15, 0.2, 0.25])
maturity = 1.0
trap = 1

num_paths = 10
num_steps = 25
history_len = 1
feature_dim = 17
hidden_dim = 64
action_dim = 2
transaction_cost = True
transaction_fee_rate = 1e-3

params = {
    "kappa": np.array([5.0, 2.5, 3.0]),
    "theta": np.array([0.05, 0.035, 0.045]),
    "rho": np.array([-0.8, -0.6, -0.5]),
    "sigma": np.array([0.5, 0.4, 0.55]),
    "lda": np.array([0.0, 0.0, 0.0]) # not ued for now
}

base_env = HedgeDocHeston(
    S0=S0, K = K, H=H, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=num_paths, history_len=history_len,
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = GymWrapper(base_env, device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
act_spec = env.specs["input_spec", "full_action_spec", "action"].to(device)

frames_per_batch = env.num_envs * num_steps
sub_batch_num = 10
sub_batch_size = frames_per_batch // sub_batch_num
frames_per_batch, sub_batch_size

# Param for PPO
clip_param = 0.2
value_coef = 0.1
entropy_coeff = 0.1
# Param for GAE
gamma = 0.99
lmbda = 0.95

env.reset(seed=0)

action_model, inaction_model = generate_actor_inactor_mlp(
    feat_dim=feature_dim,
    hidden_dim=hidden_dim,
    action_dim=action_dim,
    env=env,
    device=device,
    action_low=-1
)

action_model.to(device)
inaction_model.to(device)

actor_advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=action_model.get_value_operator(),
    shifted=True # make sure use this one for RNN
)
actor_advantage_module.set_keys(value="a_state_value")

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
    value="a_state_value"
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

lr=1e-4
optim_actor = torch.optim.Adam(actor_loss_module.parameters(), lr=lr)
optim_inactor = torch.optim.Adam(inactor_loss_module.parameters(), lr=lr)

num_epochs = 5
num_episodes = 20
policy_only_epochs = 2
frames_per_batch = env.num_envs * num_steps
sub_batch_num = 10
sub_batch_size = frames_per_batch // sub_batch_num
frames_per_batch, sub_batch_size


with set_exploration_type(ExplorationType.RANDOM):
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            if epoch < policy_only_epochs:

                env.reset(seed=epoch + 1000)
                collector = SyncDataCollector(
                    env,
                    action_model.get_policy_operator(),
                    frames_per_batch=frames_per_batch,
                    total_frames=frames_per_batch,
                    device=device,
                )
                replay_buffer = ReplayBuffer(
                    storage=LazyTensorStorage(max_size=frames_per_batch),
                    sampler=SamplerWithoutReplacement(),
                )
                for batch in collector:
                    actor_advantage_module(batch)
                    replay_buffer.extend(batch.reshape(-1).cpu())
                    for _ in range(sub_batch_num):
                        subdata = replay_buffer.sample(sub_batch_size)
                        optim_actor.zero_grad()
                        # Forward pass PPO loss
                        loss = actor_loss_module(subdata.to(device))
                        loss_critic, loss_objective, loss_entropy = (
                            loss["loss_critic"],
                            loss["loss_objective"],
                            loss["loss_entropy"],
                        )
                        loss_sum = loss_critic + loss_objective + loss_entropy
                        # Backward pass
                        loss_sum.backward()
                        torch.nn.utils.clip_grad_norm_(actor_loss_module.parameters(), max_norm=1.0)
                        for param in actor_loss_module.parameters():
                            if param.grad is not None:
                                param.grad = torch.nan_to_num(param.grad)
                        # Update the networks
                        optim_actor.step()

                if (episode + 1) % 1 == 0:
                    print(
                        f"""Epoch {epoch+1}/{num_epochs}, Episode {episode + 1}/{num_episodes}, Loss: {loss_sum.item()}, Loss Critic: {loss_critic.item()}, Loss Obj. {loss_objective.item()}, Loss Ent. {loss_entropy.item()}, Avg. Reward: {batch['next', 'reward'].mean().item()}"""
                    )

            else:
                
                actor_advantage_module.set_keys(value="a_state_value")
                inactor_advantage_module.set_keys(value="i_state_value")

                joint_policy = JointPolicy(
                    action_model.get_policy_operator(),
                    inaction_model.get_policy_operator(),
                    action_dim,
                    device,
                )

                
                collector = SyncDataCollector(
                    env,
                    joint_policy,
                    frames_per_batch=frames_per_batch,
                    total_frames=frames_per_batch,
                    device=device,
                )

                
                replay_buffer_actor = ReplayBuffer(
                    storage=LazyTensorStorage(max_size=frames_per_batch),
                    sampler=SamplerWithoutReplacement(),
                )
                replay_buffer_inactor = ReplayBuffer(
                    storage=LazyTensorStorage(max_size=frames_per_batch),
                    sampler=SamplerWithoutReplacement(),
                )

                for batch in collector:
                    
                    batch_actor = batch.clone(False)
                    batch_inactor = batch.clone(False)

                    if "original_action" in batch_actor.keys():
                        batch_actor.set_("action", batch_actor["original_action"])
                    
                    actor_advantage_module(batch_actor)
                    inactor_advantage_module(batch_inactor)

                    replay_buffer_actor.extend(batch_actor.reshape(-1).cpu())
                    replay_buffer_inactor.extend(batch_inactor.reshape(-1).cpu())

                    # Inaction Policy is Trained
                    for _ in range(sub_batch_num):
                        subdata = replay_buffer_inactor.sample(sub_batch_size).to(device)
                        optim_inactor.zero_grad()
                        loss = inactor_loss_module(subdata)
                        loss_critic = loss["loss_critic"]
                        loss_objective = loss["loss_objective"]
                        loss_entropy = loss["loss_entropy"]
                        loss_sum = loss_critic + loss_objective + loss_entropy
                        loss_sum.backward()
                        torch.nn.utils.clip_grad_norm_(inactor_loss_module.parameters(), max_norm=1.0)
                        for p in inactor_loss_module.parameters():
                            if p.grad is not None:
                                p.grad = torch.nan_to_num(p.grad)
                        optim_inactor.step()
                    # For logging
                    i_loss_critic = loss_critic.item()
                    i_loss_objective = loss_objective.item()
                    i_loss_entropy = loss_entropy.item()

                    inact_loss = loss_sum.item()

                    # Actor Policy Trained Every 3 Losses
                    if episode % 3 == 0:
                        for _ in range(sub_batch_num):
                            subdata = replay_buffer_actor.sample(sub_batch_size).to(device)
                            optim_actor.zero_grad()
                            loss = actor_loss_module(subdata)
                            loss_critic = loss["loss_critic"]
                            loss_objective = loss["loss_objective"]
                            loss_entropy = loss["loss_entropy"]
                            loss_sum = loss_critic + loss_objective + loss_entropy
                            loss_sum.backward()
                            torch.nn.utils.clip_grad_norm_(actor_loss_module.parameters(), max_norm=1.0)
                            for p in actor_loss_module.parameters():
                                if p.grad is not None:
                                    p.grad = torch.nan_to_num(p.grad)
                            optim_actor.step()
                        current_policy_loss = loss_sum.item()
                        # Logging
                        a_loss_critic = loss_critic.item()
                        a_loss_objective = loss_objective.item()
                        a_loss_entropy = loss_entropy.item()
                    else:
                        current_policy_loss = 0.0
                        a_loss_critic = float('nan')
                        a_loss_objective = float('nan')
                        a_loss_entropy = float('nan')

                # be nice and reset stateful policy
                joint_policy.reset()

                if (episode + 1) % 10 == 0:
                    avg_reward = batch["next", "reward"].mean().item()

                    # actor losses print logic
                    if episode % 3 == 0:
                        policy_info = (
                            f"Policy Loss: {current_policy_loss:.6f}, "
                            f"Critic: {loss_critic.item():.6f}, "
                            f"Obj: {loss_objective.item():.6f}, "
                            f"Ent: {loss_entropy.item():.6f}"
                        )
                    else:
                        policy_info = "Policy: No Update"

                    # inaction losses (always updated)
                    inaction_info = (
                        f"Inaction Loss: {inact_loss:.6f}, "
                        f"Critic: {i_loss_critic:.6f}, "
                        f"Obj: {i_loss_objective:.6f}, "
                        f"Ent: {i_loss_entropy:.6f}"
                    )

                    print(
                        f"\nEpoch {epoch + 1}/{num_epochs}, "
                        f"Episode {episode + 1}/{num_episodes}, "
                        f"\n{policy_info}, "
                        f"\n{inaction_info}, "
                        f"\nAvg. Reward: {avg_reward:.6f}"
                    )


# Test

base_env = HedgeDocHeston(
    S0=S0, K = K, H=H, r=r, v0=v0, theta=params["theta"], rho=params["rho"],
    kappa=params["kappa"], xi=params["sigma"], maturity=maturity,
    num_steps=num_steps, num_paths=5, history_len=history_len,
    transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
)

env = GymWrapper(base_env, device=device)
env.reset(seed=0)

joint_policy = JointPolicy(
                    action_model.get_policy_operator(),
                    inaction_model.get_policy_operator(),
                    action_dim,
                    device,
                )

with set_exploration_type(ExplorationType.RANDOM):
    rollout = env.rollout(max_steps=num_steps, policy=joint_policy)

rewards = rollout['next', 'reward'].detach().cpu().numpy()

print(f"Mean reward: {rewards.mean()}")


