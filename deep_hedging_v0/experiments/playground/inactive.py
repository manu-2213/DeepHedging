import os
import sys

# Go up one directory from this file's location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from hedging.envs import HedgeCallBS
from hedging.reward_utils import compute_discounted_cumsum_rewards
from hedging.plot_utils import plot_portfolio_vs_option_price
from hedging.tanh_normal import TanhNorm
from scipy import stats


# Training nets
class PolicyNetwork(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_size, 
        action_dim=1, 
        log_std_min=-20, 
        log_std_max=2
    ):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, action_dim)
        
        
        self.fc_log_std = nn.Linear(hidden_size, action_dim)
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, history_features):
        x = history_features[:, -1, :]
        x = torch.tanh(self.fc1(x))
        mu = self.fc_mu(x)  # Directly bound mu
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample_action(self, mu, dist_params, deterministic=False):
        
        log_std = dist_params
        std = torch.exp(log_std)
        distribution = TanhNorm(mu, std)           
        if deterministic:
            action = torch.tanh(mu)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action, log_prob

class InactionNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_size, 
        action_dim=1, 
        log_std_min=-20, 
        log_std_max=2
    ):
        super(InactionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, action_dim)
        
        
        self.fc_log_std = nn.Linear(hidden_size, action_dim)
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, history_features):
        x = history_features[:, -1, :]
        x = torch.tanh(self.fc1(x))
        mu = self.fc_mu(x)  # Directly bound mu
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample_action(self, mu, dist_params, deterministic=False):
        
        log_std = dist_params
        std = torch.exp(log_std)
        distribution = TanhNorm(mu, std)           
        if deterministic:
            action = torch.tanh(mu)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action, log_prob


# Collect statistics

action_1 = []
reward_1 = []
tot_reward1 = []
action_2 = []
reward_2 = []
tot_reward2 = []





def train1():
    """This trains the model with no transaction costs"""
    S0 = np.array([50.0, 100.0, 200.0])
    K = np.array([[45.0, 55.0], [90.0, 110.0], [180.0, 220.0]])
    maturity = 1.0
    r = 0.05
    sigma = np.array([0.15, 0.2, 0.25])
    num_paths = 100
    num_steps = 250
    history_len = 1
    transaction_cost = False # No transaction costs are being considered.
    transaction_fee_rate = 0.00

    env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps, history_len=history_len, 
                        transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
    )

    # --- Policy Network Parameters ---
    input_dim = 11
    hidden_size = 128

    policy_net = PolicyNetwork(input_dim, hidden_size)
    inaction_net = InactionNet(input_dim, hidden_size)

    # --- Optimization Parameters ---
    learning_rate = 4*1e-4 # Seems to work best for now: 1e-3 was a bit unstable with the dual network architecture

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_2 = torch.optim.Adam(inaction_net.parameters(), lr=learning_rate)

    # --- Other Parameters ---
    num_episodes = 200
    num_epochs = 20
    discount_factor = 0.999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    inaction_net.to(device)

    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            log_prob_history = []
            inaction_log_prob_history = []  # ADD: Track inaction decisions
            reward_history = []
            state_history = []

            state, _ = env.reset(seed=epoch + 1000)  # [num_envs, obs_dim]
            state_history.append(state)

            done = np.zeros(env.num_envs, dtype=bool)

            while not all(done):
                # Prepare input to policy and inaction net
                policy_net_input = np.concatenate(state_history[-history_len:], axis=1)
                policy_net_input_tensor = torch.tensor(policy_net_input, dtype=torch.float32).to(device)

                # Get action
                action_mu, action_sigma = policy_net(policy_net_input_tensor)
                action, log_prob = policy_net.sample_action(action_mu, action_sigma)
                log_prob_history.append(log_prob)
                action_np = action.detach().cpu().numpy()

                # Decide which envs to step using inaction_net
                inaction_mu, inaction_sigma = inaction_net(policy_net_input_tensor)
                inaction_action, inaction_log_prob = inaction_net.sample_action(inaction_mu, inaction_sigma)
                inaction_log_prob_history.append(inaction_log_prob)
                
                # Convert inaction action to boolean decision (> 0.5 means take action)
                should_step = (inaction_action > 0.0).cpu().numpy()  # boolean mask

                # Step all environments
                next_state_all, reward_all, done_all, _, _ = env.step(action_np)

                # Initialize new buffers
                next_state = np.copy(state[:, 0, :])  # [num_envs, obs_dim]
                reward = np.zeros(env.num_envs)
                new_done = np.copy(done)

                # Apply step results only where allowed
                for i in range(env.num_envs):
                    if not done[i]:
                        if should_step[i]:
                            next_state[i] = next_state_all[i]
                            reward[i] = reward_all[i]
                            new_done[i] = done_all[i]
                        else:
                            # Keep state, give same reward as if stepped, but still check if episode should end
                            reward[i] = reward_all[i]
                            new_done[i] = done_all[i]

                # Update trackers
                state = next_state[:, None, :]
                state_history.append(state)
                reward_history.append(reward)
                done = new_done

            # Compute and normalize rewards
            R = compute_discounted_cumsum_rewards(np.array(reward_history), discount_factor)  # shape: [time, num_envs]
            R = R - R.mean(axis=1, keepdims=True)
            R = R / (R.std(axis=1, keepdims=True) + np.finfo(R.dtype).eps)
            R = torch.tensor(R, dtype=torch.float32).to(device)

            # Compute loss and update
            optimizer.zero_grad()
            optimizer_2.zero_grad()
            
            # Policy loss
            policy_loss = (-R * torch.stack(log_prob_history)).mean()
            
            # ADD: Inaction loss  
            inaction_loss = (-R * torch.stack(inaction_log_prob_history)).mean()
            
            policy_loss.backward()
            inaction_loss.backward()
            
            optimizer.step()
            optimizer_2.step()

            if (episode + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Inaction Loss: {inaction_loss.item():.4f}, "
                    f"Avg. Reward: {np.array(reward_history).mean():.4f}"
                )
    env = HedgeCallBS(
    S0, K, maturity, r, sigma, 5, num_steps, history_len=history_len, 
                      transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
    )
    env.reset(seed=0)

    log_prob_history = []
    inaction_log_prob_history = []  # ADD: Also track inaction decisions in test
    reward_history = []
    state_history = []
    action_taken_history = []  # Track when actions were actually taken

    state, _ = env.reset(seed=0)
    state = state[:, None, :]
    state_history.append(state)

    done = np.zeros(env.num_envs, dtype=bool)

    while not all(done):
        policy_net_input = np.concatenate(state_history[-history_len:], axis=1)
        policy_net_input_tensor = torch.tensor(policy_net_input, dtype=torch.float32).to(device)
        
        # Get action from policy network
        action_mu, action_sigma = policy_net(policy_net_input_tensor)
        action, log_prob = policy_net.sample_action(action_mu, action_sigma, True)
        log_prob_history.append(log_prob)
        action_np = action.detach().cpu().numpy()
        
        # Get inaction decision - FIX: Properly unpack the tuple like in training
        inaction_mu, inaction_sigma = inaction_net(policy_net_input_tensor)
        inaction_action, inaction_log_prob = inaction_net.sample_action(inaction_mu, inaction_sigma, True)
        inaction_log_prob_history.append(inaction_log_prob)
        
    
        should_step = (inaction_action > 0.0).cpu().numpy()  # boolean mask (0.0 > Take action)
        action_taken_history.append(should_step.copy())  # Track for later analysis
        
        # Step all environments
        next_state_all, reward_all, done_all, _, _ = env.step(action_np)
        
        # Initialize new buffers
        next_state = np.copy(state[:, 0, :])
        reward = np.zeros(env.num_envs)
        new_done = np.copy(done)
        
        # Apply step results only where allowed
        for i in range(env.num_envs):
            if not done[i]:
                if should_step[i]:
                    next_state[i] = next_state_all[i]
                    reward[i] = reward_all[i]
                    new_done[i] = done_all[i]
                else:
                    # Keep state, give same reward as if stepped, but still check if episode should end
                    reward[i] = reward_all[i]
                    new_done[i] = done_all[i]
        
        # Update trackers
        state = next_state[:, None, :]
        state_history.append(state)
        reward_history.append(reward)
        done = new_done

    print(f"Test completed. Total reward: {np.array(reward_history).sum():.4f}")
    print(f"Average reward per step: {np.array(reward_history).mean():.4f}")
    print(f"Actions taken: {np.mean([step.sum() for step in action_taken_history]):.2f} out of {env.num_envs} environments per step")
    tot_reward1.append(np.array(reward_history).sum())
    reward_1.append(np.array(reward_history).mean())
    action_1.append(np.mean([step.sum() for step in action_taken_history]))


def train2():
    # --- Env. Parameters ---
    S0 = np.array([50.0, 100.0, 200.0])
    K = np.array([[45.0, 55.0], [90.0, 110.0], [180.0, 220.0]])
    maturity = 1.0
    r = 0.05
    sigma = np.array([0.15, 0.2, 0.25])
    num_paths = 100
    num_steps = 250
    history_len = 1
    transaction_cost = True # Transaction costs are being considered.
    transaction_fee_rate = 0.005 

    env = HedgeCallBS(
        S0, K, maturity, r, sigma, num_paths, num_steps, history_len=history_len, 
                        transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
    )

    # --- Policy Network Parameters ---
    input_dim = 11
    hidden_size = 128

    policy_net = PolicyNetwork(input_dim, hidden_size)
    inaction_net = InactionNet(input_dim, hidden_size)

    # --- Optimization Parameters ---
    learning_rate = 4*1e-4 # Seems to work best for now: 1e-3 was a bit unstable with the dual network architecture

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_2 = torch.optim.Adam(inaction_net.parameters(), lr=learning_rate)

    # --- Other Parameters ---
    num_episodes = 200
    num_epochs = 20
    discount_factor = 0.999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    inaction_net.to(device)

    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            log_prob_history = []
            inaction_log_prob_history = []  # ADD: Track inaction decisions
            reward_history = []
            state_history = []

            state, _ = env.reset(seed=epoch + 1000)  # [num_envs, obs_dim]
            state_history.append(state)

            done = np.zeros(env.num_envs, dtype=bool)

            while not all(done):
                # Prepare input to policy and inaction net
                policy_net_input = np.concatenate(state_history[-history_len:], axis=1)
                policy_net_input_tensor = torch.tensor(policy_net_input, dtype=torch.float32).to(device)

                # Get action
                action_mu, action_sigma = policy_net(policy_net_input_tensor)
                action, log_prob = policy_net.sample_action(action_mu, action_sigma)
                log_prob_history.append(log_prob)
                action_np = action.detach().cpu().numpy()

                # Decide which envs to step using inaction_net
                inaction_mu, inaction_sigma = inaction_net(policy_net_input_tensor)
                inaction_action, inaction_log_prob = inaction_net.sample_action(inaction_mu, inaction_sigma)
                inaction_log_prob_history.append(inaction_log_prob)
                
                # Convert inaction action to boolean decision (> 0.5 means take action)
                should_step = (inaction_action > 0.0).cpu().numpy()  # boolean mask

                # Step all environments
                next_state_all, reward_all, done_all, _, _ = env.step(action_np)

                # Initialize new buffers
                next_state = np.copy(state[:, 0, :])  # [num_envs, obs_dim]
                reward = np.zeros(env.num_envs)
                new_done = np.copy(done)

                # Apply step results only where allowed
                for i in range(env.num_envs):
                    if not done[i]:
                        if should_step[i]:
                            next_state[i] = next_state_all[i]
                            reward[i] = reward_all[i]
                            new_done[i] = done_all[i]
                        else:
                            # Keep state, give same reward as if stepped, but still check if episode should end
                            reward[i] = reward_all[i]
                            new_done[i] = done_all[i]

                # Update trackers
                state = next_state[:, None, :]
                state_history.append(state)
                reward_history.append(reward)
                done = new_done

            # Compute and normalize rewards
            R = compute_discounted_cumsum_rewards(np.array(reward_history), discount_factor)  # shape: [time, num_envs]
            R = R - R.mean(axis=1, keepdims=True)
            R = R / (R.std(axis=1, keepdims=True) + np.finfo(R.dtype).eps)
            R = torch.tensor(R, dtype=torch.float32).to(device)

            # Compute loss and update
            optimizer.zero_grad()
            optimizer_2.zero_grad()
            
            # Policy loss
            policy_loss = (-R * torch.stack(log_prob_history)).mean()
            
            # ADD: Inaction loss  
            inaction_loss = (-R * torch.stack(inaction_log_prob_history)).mean()
            
            policy_loss.backward()
            inaction_loss.backward()
            
            optimizer.step()
            optimizer_2.step()

            if (episode + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Inaction Loss: {inaction_loss.item():.4f}, "
                    f"Avg. Reward: {np.array(reward_history).mean():.4f}"
                )
    env = HedgeCallBS(
    S0, K, maturity, r, sigma, 5, num_steps, history_len=history_len, 
                      transaction_cost=transaction_cost, transaction_fee_rate=transaction_fee_rate
    )
    env.reset(seed=0)

    log_prob_history = []
    inaction_log_prob_history = []  # ADD: Also track inaction decisions in test
    reward_history = []
    state_history = []
    action_taken_history = []  # Track when actions were actually taken

    state, _ = env.reset(seed=0)
    state = state[:, None, :]
    state_history.append(state)

    done = np.zeros(env.num_envs, dtype=bool)

    while not all(done):
        policy_net_input = np.concatenate(state_history[-history_len:], axis=1)
        policy_net_input_tensor = torch.tensor(policy_net_input, dtype=torch.float32).to(device)
        
        # Get action from policy network
        action_mu, action_sigma = policy_net(policy_net_input_tensor)
        action, log_prob = policy_net.sample_action(action_mu, action_sigma, True)
        log_prob_history.append(log_prob)
        action_np = action.detach().cpu().numpy()
        
        # Get inaction decision - FIX: Properly unpack the tuple like in training
        inaction_mu, inaction_sigma = inaction_net(policy_net_input_tensor)
        inaction_action, inaction_log_prob = inaction_net.sample_action(inaction_mu, inaction_sigma, True)
        inaction_log_prob_history.append(inaction_log_prob)
        
    
        should_step = (inaction_action > 0.0).cpu().numpy()  # boolean mask (0.0 > Take action)
        action_taken_history.append(should_step.copy())  # Track for later analysis
        
        # Step all environments
        next_state_all, reward_all, done_all, _, _ = env.step(action_np)
        
        # Initialize new buffers
        next_state = np.copy(state[:, 0, :])
        reward = np.zeros(env.num_envs)
        new_done = np.copy(done)
        
        # Apply step results only where allowed
        for i in range(env.num_envs):
            if not done[i]:
                if should_step[i]:
                    next_state[i] = next_state_all[i]
                    reward[i] = reward_all[i]
                    new_done[i] = done_all[i]
                else:
                    # Keep state, give same reward as if stepped, but still check if episode should end
                    reward[i] = reward_all[i]
                    new_done[i] = done_all[i]
        
        # Update trackers
        state = next_state[:, None, :]
        state_history.append(state)
        reward_history.append(reward)
        done = new_done

    print(f"Test completed. Total reward: {np.array(reward_history).sum():.4f}")
    print(f"Average reward per step: {np.array(reward_history).mean():.4f}")
    print(f"Actions taken: {np.mean([step.sum() for step in action_taken_history]):.2f} out of {env.num_envs} environments per step")
    tot_reward2.append(np.array(reward_history).sum())
    reward_2.append(np.array(reward_history).mean())
    action_2.append(np.mean([step.sum() for step in action_taken_history]))

if __name__ == "__main__":
    for i in range(10):
        #train1()
        train2()


    def calculate_stats(data, name):
        """Calculate mean, std, and 95% CI for a dataset"""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)  # Sample standard deviation
        
        # 95% Confidence Interval using t-distribution
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        margin_error = t_critical * (std / np.sqrt(n))
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        print(f"\n{name}:")
        print(f"  Sample size: {n}")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': n
        }

    # Calculate statistics for all metrics
    print("=== STATISTICAL ANALYSIS ===")

    # Scenario 1 (No transaction costs)
    #print("\n--- SCENARIO 1 (No Transaction Costs) ---")
    #action_1_stats = calculate_stats(action_1, "Actions Taken per Step")
    #reward_1_stats = calculate_stats(reward_1, "Average Reward per Step")
    #tot_reward1_stats = calculate_stats(tot_reward1, "Total Reward")

    # Scenario 2 (With transaction costs)
    print("\n--- SCENARIO 2 (With Transaction Costs) ---")
    action_2_stats = calculate_stats(action_2, "Actions Taken per Step")
    reward_2_stats = calculate_stats(reward_2, "Average Reward per Step")
    tot_reward2_stats = calculate_stats(tot_reward2, "Total Reward")

    # Comparison between scenarios
    print("\n=== COMPARISON BETWEEN SCENARIOS ===")

    # Perform t-tests to check for significant differences
    def compare_scenarios(data1, data2, metric_name):
        """Perform independent t-test between two scenarios"""
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        print(f"\n{metric_name} Comparison:")
        print(f"  Scenario 1 Mean: {np.mean(data1):.4f}")
        print(f"  Scenario 2 Mean: {np.mean(data2):.4f}")
        print(f"  Difference: {np.mean(data1) - np.mean(data2):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
"""
    compare_scenarios(action_1, action_2, "Actions Taken")
    compare_scenarios(reward_1, reward_2, "Average Reward")
    #compare_scenarios(tot_reward1, tot_reward2, "Total Reward")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Metric':<25} {'Scenario':<12} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
    print("-" * 77)

    metrics = [
        ("Actions/Step", action_1_stats, action_2_stats),
        ("Avg Reward/Step", reward_1_stats, reward_2_stats),
        ("Total Reward", tot_reward1_stats, tot_reward2_stats)
    ]

    for metric_name, stats1, stats2 in metrics:
        print(f"{metric_name:<25} {'No TxnCost':<12} {stats1['mean']:<10.4f} {stats1['std']:<10.4f} [{stats1['ci_lower']:.4f}, {stats1['ci_upper']:.4f}]")
        print(f"{'':<25} {'With TxnCost':<12} {stats2['mean']:<10.4f} {stats2['std']:<10.4f} [{stats2['ci_lower']:.4f}, {stats2['ci_upper']:.4f}]")
        print()

"""