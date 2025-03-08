import numpy as np
from scipy.stats import norm
from gymnasium import spaces
from gymnasium.experimental.vector import VectorEnv  # ver. 0.29.1
from gymnasium.vector.utils import batch_space


class HedgingEnv(VectorEnv):
    def __init__(
        self,
        S0,
        K,
        sigma,
        r,
        num_simulation=100,
        history_len=None,
        num_step=250,
        reward_type="abs_diff",
    ):
        """
        Initialize the Delta Hedging Environment

        Parameters:
        - S0: Initial stock prices (array of shape [num_asset])
        - K: Strike prices (array of shape [num_asset, num_strike_per_asset])
        - sigma: Volatilities (array of shape [num_asset])
        - r: Risk-free rate (scalar)
        - num_simulation: Number of simulations to run in parallel
        - num_step: Number of time steps in each simulation
        """

        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.num_simulation = num_simulation
        self.history_len = history_len
        self.num_step = num_step
        self.reward_type = reward_type

        self.num_asset = len(S0)
        self.num_strike = K.shape[1]
        self.num_total_options = self.num_simulation * self.num_asset * self.num_strike

        # Time setup
        self.dt = 1.0 / num_step
        self.T = np.linspace(1, 0, num_step + 1)  # Time to maturity

        # Gym VectorEnv
        self.num_envs = self.num_total_options
        self.state = None
        self.single_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.feature_dim = 11
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        if self.history_len is not None:
            self.single_observation_space = spaces.Box(
                -np.inf, np.inf, shape=(history_len, self.feature_dim), dtype=np.float32
            )
        else:
            self.single_observation_space = spaces.Box(
                -np.inf, np.inf, shape=(self.feature_dim,), dtype=np.float32
            )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return initial state
        """

        super().reset(seed=seed)
        # Generate full simulations
        self._generate_simulations()

        self.current_step = 0

        if self.history_len is not None:
            self.observation_buff = np.zeros(
                (self.num_envs, self.history_len, self.feature_dim)
            )

        # Reset portfolio
        self.cash_account = np.zeros(
            (self.num_simulation, self.num_asset, self.num_strike, self.num_step + 1)
        )
        self.shares_held = np.zeros(
            (self.num_simulation, self.num_asset, self.num_strike, self.num_step + 1)
        )
        self.portfolio_value = np.zeros(
            (self.num_simulation, self.num_asset, self.num_strike, self.num_step + 1)
        )

        # Add cash since the option was just sold
        self.cash_account[..., 0] = self.option_prices[..., 0]
        self.portfolio_value[..., 0] = self.cash_account[..., 0]

        # Initial state features
        self.state = self._create_observations()

        return self.state, {}

    def step(self, action):
        """
        Take a step in the environment based on action

        Parameters:
        - action: Delta values to use for hedging
          (shape: [num_simulation * num_asset * num_strike])
        """
        action = action.reshape(-1, 1)
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        # Reshape actions to match our dimensions
        action_reshaped = action.reshape(
            self.num_simulation, self.num_asset, self.num_strike
        )

        # Update portfolio based on actions (stock positions)
        self._update_portfolio(action_reshaped)

        # move to next step
        self.current_step += 1
        # Calculate rewards
        reward = self._calculate_rewards()
        self.state = self._create_observations()

        # Check if done
        done = self.current_step == self.num_step

        return (
            self.state,
            reward,
            np.full(self.num_envs, done),
            np.full(self.num_envs, done),
            {},
        )

    def _generate_simulations(self):
        """Generate stock price simulations using Black-Scholes model

        S0 = np.tile(S0, (num_simulation,1))[...,None]
        z = np.random.normal(0, 1, (num_simulation, num_asset, num_step))
        log_returns = (r - 0.5 * sigma[None,:,None]**2) * dt + sigma[None,:,None] * np.sqrt(dt) * z
        log_S = np.log(S0) + np.cumsum(log_returns, axis=-1)
        stock_prices = np.exp(log_S)
        stock_prices = np.concatenate((S0, stock_prices), axis=-1)

        """
        # Expand S0 for simulations
        S0_expanded = np.tile(self.S0, (self.num_simulation, 1))[:, :, None]

        # Generate random normal shocks
        z = self.np_random.normal(
            0, 1, (self.num_simulation, self.num_asset, self.num_step)
        )

        # Calculate log returns
        log_returns = (
            self.r - 0.5 * self.sigma[None, :, None] ** 2
        ) * self.dt + self.sigma[None, :, None] * np.sqrt(self.dt) * z

        # Calculate log stock prices
        log_S = np.log(S0_expanded) + np.cumsum(log_returns, axis=-1)

        # Convert to stock prices
        stock_prices = np.exp(log_S)

        # Add initial price at time 0
        self.stock_prices = np.concatenate((S0_expanded, stock_prices), axis=-1)

        # Calculate option prices
        self.option_prices = self._calculate_call_prices()

        # Calculate ground truth deltas
        self.ground_truth_deltas = self._calculate_ground_truth_deltas()

    def _calculate_call_prices(self):
        """Calculate option prices using Black-Scholes formula

        S = S[..., None, :]  # (num_simulation, num_asset, 1, num_step)
        K = K[None, ..., None]  # (1, num_asset, num_strike, 1)
        T = T[None, None, None, :]  # (1, 1, 1, Step_num)
        sigma = sigma[None, :, None, None]  # (1, num_asset, 1, 1)
        call_prices = np.maximum(0, S - K) * (T <= 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_prices = np.where(T > 0, S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), call_prices)

        """
        S = self.stock_prices[:, :, None, :]
        # (num_simulation, num_asset, 1, num_step+1)
        K = self.K[None, :, :, None]
        # (1, num_asset, num_strike, 1)
        T = self.T[None, None, None, :]
        # (1, 1, 1, num_step+1)
        sigma = self.sigma[None, :, None, None]
        # (1, num_asset, 1, 1)

        # Handle expiration case
        last_call_prices = np.maximum(0, S - K)[..., -1]

        # Calculate d1 and d2
        d1 = (np.log(S[..., :-1] / K) + (self.r + 0.5 * sigma**2) * T[..., :-1]) / (
            sigma * np.sqrt(T[..., :-1])
        )
        d2 = d1 - sigma * np.sqrt(T[..., :-1])

        # Use Black-Scholes formula for non-expired options
        call_prices = S[..., :-1] * norm.cdf(d1) - K * np.exp(
            -self.r * T[..., :-1]
        ) * norm.cdf(d2)

        return np.concatenate((call_prices, last_call_prices[..., None]), axis=-1)
        # (num_simulation, num_asset, num_strike, num_step+1)

    def _calculate_ground_truth_deltas(self):
        """Calculate theoretical Black-Scholes deltas for options

        S = S[..., None, :]  # (num_simulation, num_asset, 1, num_step)
        K = K[None, ..., None]  # (1, num_asset, Num_unique_K, 1)
        T = T[None, None, None, :]  # (1, 1, 1, num_step)
        sigma = sigma[None, :, None, None]  # (1, num_asset, 1, 1)
        delta = np.where(T <= 0, np.where(S > K, 1.0, 0.0), 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = np.where(T > 0, norm.cdf(d1), delta)

        """
        S = self.stock_prices[:, :, None, :]
        # (num_simulation, num_asset, 1, num_step+1)
        K = self.K[None, :, :, None]
        # (1, num_asset, num_strike, 1)
        T = self.T[None, None, None, :]
        # (1, 1, 1, num_step+1)
        sigma = self.sigma[None, :, None, None]
        # (1, num_asset, 1, 1)

        # Handle expiration case
        last_deltas = np.where(S[..., -1] > K[..., -1], 1.0, 0.0)

        # Calculate d1 for non-expired options
        d1 = (np.log(S[..., :-1] / K) + (self.r + 0.5 * sigma**2) * T[..., :-1]) / (
            sigma * np.sqrt(T[..., :-1])
        )

        # Calculate deltas
        deltas = norm.cdf(d1)

        # Combine with expiration deltas
        return np.concatenate((deltas, last_deltas[..., None]), axis=-1)
        # (num_simulation, num_asset, num_strike, num_step+1)

    def _create_observations(self):
        """Create state features for the given step"""
        step_idx = self.current_step
        # Get current stock prices
        S_current = self.stock_prices[:, :, step_idx]  # (num_simulation, num_asset)
        K_expanded = self.K[None, :, :].repeat(
            self.num_simulation, axis=0
        )  # (num_simulation, num_asset, num_strike)
        T_current = self.T[step_idx]

        # Expand stock price to match K dimensions
        S_expanded = S_current[:, :, None].repeat(
            self.num_strike, axis=2
        )  # (num_simulation, num_asset, num_strike)

        # Calculate moneyness and other common features
        moneyness = K_expanded / S_expanded
        moneyness_log = np.log(moneyness)
        time_feature = np.full_like(moneyness, T_current)
        risk_free_rate = np.full_like(moneyness, self.r)

        if step_idx == 0:
            # For initial step, set price_change to 1 and log_change to 0
            asset_price_change = np.zeros_like(moneyness)
            asset_price_change_ratio = np.ones_like(moneyness)
            asset_price_change_log = np.zeros_like(moneyness)
            option_price_change = np.zeros_like(moneyness)
            option_price_change_ratio = np.ones_like(moneyness)
            option_price_change_log = np.zeros_like(moneyness)
            est_delta = np.zeros_like(moneyness)
        else:
            # Calculate price changes for steps after the first one
            S_prev = self.stock_prices[:, :, step_idx - 1]
            S_prev_expanded = S_prev[:, :, None].repeat(self.num_strike, axis=2)
            asset_price_change = S_expanded - S_prev_expanded
            asset_price_change_ratio = (S_expanded + np.finfo(S_expanded.dtype).eps) / (
                S_prev_expanded + np.finfo(S_expanded.dtype).eps
            )
            asset_price_change_log = np.log(asset_price_change_ratio)
            option_price_change = (
                self.option_prices[:, :, :, step_idx]
                - self.option_prices[:, :, :, step_idx - 1]
            )
            option_price_change_ratio = (
                self.option_prices[:, :, :, step_idx]
                + np.finfo(self.option_prices.dtype).eps
            ) / (
                self.option_prices[:, :, :, step_idx - 1]
                + np.finfo(self.option_prices.dtype).eps
            )
            option_price_change_log = np.log(option_price_change_ratio)
            est_delta = option_price_change / (
                asset_price_change + np.finfo(asset_price_change.dtype).eps
            )
            est_delta = np.clip(est_delta, 0.0, 1.0)

        # Stack features
        features = np.stack(
            [
                asset_price_change,  # Asset price change
                asset_price_change_ratio,  # Asset price change ratio
                asset_price_change_log,  # Asset price change (log)
                option_price_change,  # Option price change
                option_price_change_ratio,  # Option price change ratio
                option_price_change_log,  #  Option price change (log)
                est_delta,  # Estimated delta
                moneyness,  # Moneyness (K/S)
                moneyness_log,  # Log moneyness
                time_feature,  # Time to maturity
                risk_free_rate,  # Risk-free rate
            ],
            axis=-1,
        )

        # Reshape to have a flat batch dimension
        features = features.reshape(self.num_total_options, -1)
        if self.history_len is not None:
            self.observation_buff = np.concatenate(
                (self.observation_buff[:, 1:, :], features[:, None, :]), axis=1
            )
            return self.observation_buff
        else:
            return features

    def _update_portfolio(self, stock_position):
        """Update portfolio based on stock positions"""
        # Set the new positions based on actions
        shares_to_trade = stock_position - self.shares_held[:, :, :, self.current_step]
        self.shares_held[:, :, :, self.current_step] = stock_position

        # Update cash for the current period
        self.cash_account[:, :, :, self.current_step] -= (
            shares_to_trade * self.stock_prices[:, :, None, self.current_step]
        )

        # Update for the next step
        if self.current_step < self.num_step:
            # Carry cash with interests
            self.cash_account[:, :, :, self.current_step + 1] = self.cash_account[
                :, :, :, self.current_step
            ] * np.exp(self.r * self.dt)
            # Carry stocks
            self.shares_held[:, :, :, self.current_step + 1] = self.shares_held[
                :, :, :, self.current_step
            ]
            # Calculate portfolio value
            stock_value_next = (
                self.shares_held[:, :, :, self.current_step + 1]
                * self.stock_prices[:, :, None, self.current_step + 1]
            )
            self.portfolio_value[:, :, :, self.current_step + 1] = (
                self.cash_account[:, :, :, self.current_step + 1] + stock_value_next
            )

    def _calculate_rewards(self):
        """Calculate rewards based on the reward type"""
        # Reward is negative of absolute difference between portfolio and option value
        if self.reward_type == "abs_diff":
            step_reward = -np.abs(
                self.portfolio_value[:, :, :, self.current_step]
                - self.option_prices[:, :, :, self.current_step]
            )
        elif self.reward_type == "portfolio_return":
            step_reward = self.portfolio_value[:, :, :, self.current_step]
        else:
            raise ValueError(f"Invalid reward type: {self.reward}")
        # Flatten the reward to match the expected shape
        return step_reward.reshape(self.num_total_options)
