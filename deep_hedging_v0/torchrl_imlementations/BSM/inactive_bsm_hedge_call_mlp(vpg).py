import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from experiments.utils.actor_inactor_mlp import generate_actor_inactor_mlp

import numpy as np
import torch
import torch.nn as nn
from hedging.envs import HedgeCallBS
from hedging.reward_utils import compute_discounted_cumsum_rewards
from hedging.logit_normal import LogitNormal

from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeProbabilisticModule, ProbabilisticActor, TanhNormal


import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict

S0 = np.array([50.0, 100.0, 200.0])
K = np.array([[45.0, 55.0], [90.0, 110.0], [180.0, 220.0]])
maturity = 1.0
r = 0.05
sigma = np.array([0.15, 0.2, 0.25])
num_paths = 100
num_steps = 250
history_len = 1
feature_dim = 11
hidden_dim = 64
action_dim = 1 
transaction_cost = True
transaction_fee_rate = 1e-3

base_env = HedgeCallBS(
    S0, K, maturity, r, sigma, num_paths, num_steps,
    history_len=history_len,
    transaction_cost=transaction_cost,
    transaction_fee_rate=transaction_fee_rate
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GymWrapper(base_env, device=device)
env.reset(seed=0)

actor, inactor = generate_actor_inactor_mlp(feat_dim=feature_dim, 
                                            hidden_dim=hidden_dim, 
                                            action_dim=action_dim,
                                            env=env,
                                            device=device,
                                            )

lr = 1e-3 
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
optimizer_inactor = torch.optim.Adam(inactor.parameters(), lr=lr)

num_epochs = 20
num_episodes = 200
policy_only_epochs = 8
joint_training_epochs = num_epochs - policy_only_epochs
gamma = 0.999

total_steps = num_epochs * num_episodes

scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_actor, T_max=total_steps, eta_min=1e-4
)
scheduler_inactor = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_inactor, T_max=total_steps, eta_min=1e-4
)


with set_exploration_type(ExplorationType.RANDOM):
    for epoch in range(num_epochs):
        
        if epoch < policy_only_epochs:
            print(f"\n Epoch: {epoch+1}/{num_epochs} - PHASE 1: Policy Network Training Only")

            for episode in range(num_episodes):
                td_episode = env.reset()
                
                td_episode = env.rollout(
                    policy=actor,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_all_done=True,
                    max_steps=num_steps,
                )

                rewards = td_episode["next", "reward"].squeeze(-1)

                R = compute_discounted_cumsum_rewards(np.array(rewards), gamma)
                R = R - R.mean(axis=1, keepdims=True)
                R = R / (R.std(axis=1, keepdims=True) + np.finfo(R.dtype).eps)
                R = torch.tensor(R, dtype=torch.float32).to(device)

                action_log_probs = td_episode.get("action_log_prob")

                optimizer_actor.zero_grad()
                actor_loss = (-R * action_log_probs).mean() # torch.stack(action_log_probs)?
                actor_loss.backward()
                optimizer_actor.step()
                

                if (episode + 1) % 20 == 0:
                    print(
                        f"  Episode {episode + 1}/{num_episodes}, "
                        f"Policy Loss: {actor_loss.item():.4f}, "
                        f"Avg. Reward: {np.array(rewards).mean():.4f}"
                    )
                    scheduler_actor.step()
        
        else:
            print(f"\n Epoch: {epoch+1}/{num_epochs} - PHASE 2: Joint Policy Training")

            for episode in range(num_episodes):
                td_episode = env.reset()
                prev_action = torch.zeros(env.batch_size[0], action_dim, device=device)
                log_probs, inact_log_probs, rewards = [], [], []

                for t in range(num_steps):
                    td_episode = actor(td_episode)    
                    td_episode = inactor(td_episode)  

                    mask = td_episode["inact"].squeeze(-1).squeeze(-1).bool().expand_as(td_episode["action"])
                    final_action = torch.where(mask, prev_action.squeeze(-1), td_episode["action"])
                    chosen_logp_inact = td_episode["inaction_log_prob"]
                    logp_action_t = td_episode["action_log_prob"]

                    td_episode = td_episode.clone(True)
                    td_episode.set_("action", final_action.detach())
                    td_episode = env.step(td_episode)

                    rewards.append(td_episode["next", "reward"])             
                    log_probs.append(logp_action_t)                  
                    inact_log_probs.append(chosen_logp_inact)        

                    prev_action = final_action.detach()
                
                rewards_t = torch.stack(rewards, dim=0)                        
                rewards_np = rewards_t.cpu().numpy().squeeze(-1)
                
                R = compute_discounted_cumsum_rewards(np.array(rewards_np), gamma)
                R = R - R.mean(axis=1, keepdims=True)
                R = R / (R.std(axis=1, keepdims=True) + np.finfo(R.dtype).eps)
                R = torch.tensor(R, dtype=torch.float32, device=device)        

                log_probs_t = torch.stack(log_probs, dim=0)                    
                inact_log_probs_t = torch.stack(inact_log_probs, dim=0)       

               
                optimizer_inactor.zero_grad()
                inact_loss = (-R * inact_log_probs_t.squeeze(-1)).mean()
                inact_loss.backward()
                optimizer_inactor.step()
                

                if episode % 3 == 0:
                    optimizer_actor.zero_grad()
                    actor_loss = (-R * log_probs_t).mean()
                    actor_loss.backward()
                    optimizer_actor.step()
                    current_policy_loss = actor_loss.item()
                else:
                    current_policy_loss = 0.0
                
                if (episode + 1) % 20 == 0:
                    policy_info = f"Policy Loss: {current_policy_loss:.4f}" if episode % 3 == 0 else "Policy: No Update"
                    print(
                        f"  Episode {episode + 1}/{num_episodes}, "
                        f"{policy_info}, "
                        f"Inaction Loss: {inact_loss.item():.4f}, "
                        f"Avg. Reward: {rewards_t.mean().item():.4f}"
                    )
                    scheduler_inactor.step()

class TestPolicy(nn.Module):
    def __init__(self, actor, inactor):
        super().__init__()
        self.actor = actor
        self.inactor = inactor

    def forward(self, td):
        # Deterministic action from the actor
        with torch.no_grad():
            td = self.actor(td)
            proposed_action = td["action"]  
        
        td = self.inactor(td)
        mask = td["inact"].squeeze(-1).bool().expand_as(proposed_action)

        prev_action = td.get("prev_action", torch.zeros_like(proposed_action))
        final_action = torch.where(mask, prev_action, proposed_action)

        td.set("action", final_action)
        td.set("prev_action", final_action.detach())  # save for next step

        return td

base_env = HedgeCallBS(S0, K, maturity, r, sigma, 5, num_steps, history_len=history_len)
env = GymWrapper(base_env, device=device)
env.reset(seed=0)

test_policy = TestPolicy(actor, inactor)

with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
    td_rollout = env.rollout(
        max_steps=num_steps,
        policy=test_policy,
        auto_reset=True,
        break_when_all_done=True,
    )

print("Test rollout complete!")
print("Average Reward:", td_rollout["next", "reward"].mean().item())

