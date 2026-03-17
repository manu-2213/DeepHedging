import numpy as np
from hedging.env_hedge_base import HedgeBase
from hedging.feature_extractor import create_observation_hedge_doc


class HedgeDoc(HedgeBase):
    def __init__(
        self,
        S0,
        K,
        H,
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
            r, #
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
        self.H = H

    def post_reset(self):
        self.stock_prices = self.simulator.paths
        # Only reprice options on a full reset (new paths).
        # Soft resets (torchRL auto-resets between episodes) reuse cached prices.
        if not getattr(self, '_is_soft_reset', False):
            self.option_prices = self.simulator.down_out_call(self.K, self.H)  # DoC prices
            self.call_prices = self.simulator.euro_call(self.K)
            self.put_prices = self.simulator.euro_put(self.H**2 / self.K)
        self.current_step = 0
        # Reset portfolio and cash account
        self.reset_portfolio()
        # Reset asset positions
        self.call_held = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=np.float32,
        )
        self.put_held = np.zeros(
            (self.num_paths, self.num_assets, self.num_strikes, self.num_steps + 1),
            dtype=np.float32,
        )
        # Initial state features
        self.state[:, -1, :] = self._create_observations()
        return self.state, {}

    def _create_observations(self):
        return create_observation_hedge_doc(
            self.current_step,
            self.stock_prices,
            self.option_prices,
            self.call_prices,
            self.put_prices,
            self.K_expanded,
            self.r,
            self.simulator.T,
            self.num_strikes,
            self.num_total_options,
        )

        
    def _update_portfolio(self, positions):
        
        # Set the new positions based on actions
        call_position = positions[..., 0]  # Call position
        put_position = positions[..., 1]  # Put position
        call_to_trade = call_position - self.call_held[:, :, :, self.current_step]
        self.call_held[:, :, :, self.current_step] = call_position
        put_to_trade = put_position - self.put_held[:, :, :, self.current_step]
        self.put_held[:, :, :, self.current_step] = put_position

        # Update cash for the current period
        call_trade_value = call_to_trade * self.call_prices[..., self.current_step]
        put_trade_value = put_to_trade * self.put_prices[..., self.current_step]
        
        # Calculate and apply transaction costs if enabled
        if self.transaction_cost:
            # Calculate transaction costs
            call_transaction_costs = np.abs(call_trade_value) * self.transaction_fee_rate * 5
            put_transaction_costs = np.abs(put_trade_value) * self.transaction_fee_rate * 5
            
            # Subtract trading costs from cash account
            self.cash_account[:, :, :, self.current_step] -= (
                call_trade_value + call_transaction_costs
            )
            self.cash_account[:, :, :, self.current_step] -= (
                put_trade_value + put_transaction_costs
            )
        else:
            # Without transaction costs, just subtract trade values
            self.cash_account[:, :, :, self.current_step] -= call_trade_value
            self.cash_account[:, :, :, self.current_step] -= put_trade_value

        # Update for the next step
        if self.current_step < self.num_steps:
            # Carry cash with interests
            self.cash_account[:, :, :, self.current_step + 1] = self.cash_account[
                :, :, :, self.current_step
            ] * np.exp(self.r * self.simulator.dt)
            
            # Carry call option
            self.call_held[:, :, :, self.current_step + 1] = self.call_held[
                :, :, :, self.current_step
            ]
            
            # Carry put option
            self.put_held[:, :, :, self.current_step + 1] = self.put_held[
                :, :, :, self.current_step
            ]
            
            # Calculate portfolio value
            call_value_next = (
                self.call_held[:, :, :, self.current_step + 1]
                * self.call_prices[..., self.current_step + 1]
            )
            put_value_next = (
                self.put_held[:, :, :, self.current_step + 1]
                * self.put_prices[..., self.current_step + 1]
            )
            self.portfolio_value[:, :, :, self.current_step + 1] = (
                self.cash_account[:, :, :, self.current_step + 1]
                + call_value_next
                + put_value_next
            )
