
from hedging.plot_utils import plot_portfolio_vs_option_price
from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
import wandb


def test_model(base_env, model, num_steps, device, plotting=False):
    env = GymWrapper(base_env, device=device)
    env.reset(seed=0)

    # Handle both models with get_policy_operator() and direct policies (like JointPolicy)
    if hasattr(model, 'get_policy_operator'):
        policy = model.get_policy_operator()
    else:
        policy = model

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(max_steps=num_steps, policy=policy)

    rewards = rollout['next', 'reward'].detach().cpu().numpy()
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    
    print(f" Mean of rewards: {mean_reward} Standard dev: {std_reward}")
    
    # Log final test statistics to wandb
    wandb.log({
        "final/mean_reward": float(mean_reward),
        "final/std_reward": float(std_reward),
    })

    if plotting:
        plot_portfolio_vs_option_price(env._env)

    return rewards



    
        