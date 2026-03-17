from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

FLOAT_DTYPE = np.float32


class HedgeBase(VectorEnv, ABC):
    def __init__(
        self,
        S0,
        K,
        maturity,
        r,
        sigma,
        num_paths,
        num_steps,
        reward_type,
        action_dim,
        action_low,
        action_high,
        feature_dim,
        feature_low,
        feature_high,
        history_len,
        lda,
        transaction_cost,
        transaction_fee_rate,
    ):
        """
        Initialize the Delta Hedging Environment

        Parameters:
        - S0: Initial stock prices (array of shape [num_asset])
        - K: Strike prices (array of shape [num_asset, num_strike_per_asset])
        - maturity: Time to maturity (scalar)
        - r: Risk-free rate (scalar)
        - num_paths: Number of simulations to run in parallel
        - num_steps: Number of time steps in each simulation
        """

        self.S0 = np.asarray(S0, dtype=FLOAT_DTYPE)
        self.K = np.asarray(K, dtype=FLOAT_DTYPE)
        self.maturity = maturity
        self.r = r
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.reward_type = reward_type #
        self.action_dim = action_dim
        self.lda = lda

        self.num_assets = len(self.S0)
        self.num_strikes = self.K.shape[1]
        self.num_total_options = self.num_paths * self.num_assets * self.num_strikes

        # Gym VectorEnv
        self.num_envs = self.num_total_options
        self.state = None
        self.single_action_space = spaces.Box(
            low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = spaces.Box(
            low=feature_low, high=feature_high, shape=(history_len, feature_dim,), dtype=np.float32
        ) # Dynamically adjust the number of features?
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.K_expanded = self.K[None, :, :].repeat(self.num_paths, axis=0)
        # (num_paths, num_assets, num_strikes)
        self.feature_dim = feature_dim
        self.history_len = history_len

        # Transaction cost
        self.transaction_cost = transaction_cost
        self.transaction_fee_rate = transaction_fee_rate

        # Autoreset
        self.autoreset_mode = gym.vector.AutoresetMode.DISABLED
        
        # Soft reset flag: when True, reset() will skip path regeneration
        # Set this to True after initial setup to make auto-resets fast
        self._soft_reset_enabled = False
        
        # Flag indicating whether current reset is a soft reset (skip pricing)
        self._is_soft_reset = False

    def reset_portfolio(self):
        # Reset portfolio
        self.cash_account = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=FLOAT_DTYPE,
        )
        self.portfolio_value = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=FLOAT_DTYPE,
        )
        # Add cash since the option was just sold
        self.cash_account[..., 0] = self.option_prices[..., 0] # This defined later?
        self.portfolio_value[..., 0] = self.cash_account[..., 0]

    def _calculate_option_prices(self):
        """
        Calculate option prices after reset. Override in subclasses for model-specific pricing.
        Default implementation assumes simulator has euro_call method.
        """
        self.option_prices = self.simulator.euro_call(self.K)

    def post_reset(self):
        """
        Post-reset routine: initialize state and portfolio.
        Skips expensive option pricing during soft resets.
        """
        self.stock_prices = self.simulator.paths
        
        # Only recalculate option prices on full reset
        if not self._is_soft_reset:
            self._calculate_option_prices()
        
        self.current_step = 0
        # Reset portfolio and cash account
        self.reset_portfolio()
        # Reset asset positions
        self.shares_held = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=FLOAT_DTYPE,
        )
        # Initial state features
        self.state[:, -1, :] = self._create_observations()
        return self.state, {}

    def step(self, action):
        """
        Take a step in the environment based on action

        Parameters:
        - action: Call position and put position
          (shape: [num_paths * num_assets * num_strikes, action_dim])
        """
        
        action = action.reshape(-1, self.action_dim)
        
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid" 
        assert self.state is not None, "Call reset before using step method."

        # Reshape actions to match our dimensions
        action_reshaped = action.reshape(
            self.num_paths, self.num_assets, self.num_strikes, self.action_dim
        )

        # Update portfolio based on actions
        self._update_portfolio(action_reshaped)

        # move to next step
        self.current_step += 1
        # Calculate rewards
        reward = self._calculate_rewards()
        self.state = np.concatenate(
            [self.state[:, 1:, :], self._create_observations()[:, None, :]], axis=1
        )

        # Check if done
        done = self.current_step == self.num_steps

        return (
            self.state,
            reward,
            np.full(self.num_envs, done), # Why do we return this twice?
            np.full(self.num_envs, done),
            {},
        )

    @abstractmethod
    def _create_observations(self):
        pass

    @abstractmethod
    def _update_portfolio(self, positions):
        pass

    def _calculate_rewards(self, lda = 0):
        """Calculate rewards based on the reward type"""
        # Reward is negative of absolute difference between portfolio and option value
        if self.reward_type == "abs_diff":
            step_reward = -np.abs(
                self.portfolio_value[:, :, :, self.current_step]
                - self.option_prices[:, :, :, self.current_step]
            )
        elif self.reward_type == "portfolio_return":
            step_reward = self.portfolio_value[:, :, :, self.current_step]
        
        elif self.reward_type == "risk_adjusted_return":
            step_reward = self.portfolio_value[:, :, :, self.current_step] - lda * np.abs(
                self.portfolio_value[:, :, :, self.current_step]
                - self.option_prices[:, :, :, self.current_step] 
            )
        else:
            raise ValueError(f"Invalid reward type: {self.reward}")
        # Flatten the reward to match the expected shape
        return step_reward.reshape(self.num_total_options)
