import numpy as np
from hedging.env_hedge_base import HedgeBase
from hedging.feature_extractor import create_observation_hedge_call


class HedgeCall(HedgeBase):
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
        super().__init__(
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
        )

    def post_reset(self):
        self.stock_prices = self.simulator.paths
        # Only reprice options on a full reset (new paths).
        # Soft resets (torchRL auto-resets between episodes) reuse cached prices.
        if not getattr(self, '_is_soft_reset', False):
            self.option_prices = self.simulator.euro_call(self.K)
        self.current_step = 0
        # Reset portfolio and cash account
        self.reset_portfolio()
        # Reset asset positions
        self.shares_held = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=np.float32,
        )
        # Initial state features
        self.state[:, -1, :] = self._create_observations()
        return self.state, {}

    def _create_observations(self):
        return create_observation_hedge_call(
            self.current_step,
            self.stock_prices,
            self.option_prices,
            self.K_expanded,
            self.r,
            self.simulator.T,
            self.num_strikes,
            self.num_total_options,
        )

    def _update_portfolio(self, stock_position):
        """Update portfolio based on stock positions"""
        # Set the new positions based on actions
        stock_position = stock_position[..., 0]
        shares_to_trade = stock_position - self.shares_held[:, :, :, self.current_step]
        self.shares_held[:, :, :, self.current_step] = stock_position

        # Update cash for the current period
        

        # Calculate and apply transaction costs if enabled
        if self.transaction_cost:
            # Calculate transaction costs
            stock_transaction_costs = np.abs(shares_to_trade * self.stock_prices[:, :, None, self.current_step]) * self.transaction_fee_rate
            
            
            # Subtract trading costs from cash account
            self.cash_account[:, :, :, self.current_step] -= (
                shares_to_trade * self.stock_prices[:, :, None, self.current_step] + stock_transaction_costs
            )
            
        else:
            self.cash_account[:, :, :, self.current_step] -= (
            shares_to_trade * self.stock_prices[:, :, None, self.current_step]
        )

        # Update for the next step
        if self.current_step < self.num_steps:
            # Carry cash with interests
            self.cash_account[:, :, :, self.current_step + 1] = self.cash_account[
                :, :, :, self.current_step
            ] * np.exp(self.r * self.simulator.dt)
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


