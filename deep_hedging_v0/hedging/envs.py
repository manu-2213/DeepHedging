import numpy as np
from hedging.env_hedge_call import HedgeCall
from hedging.env_hedge_doc import HedgeDoc
from hedging.env_hedge_conc import HedgeConc
from hedging.simulators import BlackScholesSimulator
from hedging.simulators import HestonSimulator


class HedgeCallBS(HedgeCall):
    def __init__(
        self,
        S0,
        K,
        maturity,
        r,
        sigma,
        num_paths=128,
        num_steps=250,
        reward_type="abs_diff",
        action_dim=1,
        action_low=-1.0,
        action_high=1.0,
        feature_dim=11,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=1, # Does it make sense to define this here?
        lda=0.5,
        transaction_cost=False, 
        transaction_fee_rate=0.01,
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
        self.sigma = sigma

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        self.simulator = BlackScholesSimulator(
            self.S0,
            self.r,
            self.sigma,
            self.maturity,
            self.num_steps,
            self.num_paths,
            self.np_random,
        )
        return self.post_reset()


class HedgeDocBS(HedgeDoc):
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
        reward_type="abs_diff",
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        feature_dim=17,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=10, # Set to 1 for MLP
        lda=0.5,
        transaction_cost=False,
        transaction_fee_rate=0.001,
    ):
        super().__init__(
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
        )
        self.sigma = sigma

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        self.simulator = BlackScholesSimulator(
            self.S0,
            self.r,
            self.sigma,
            self.maturity,
            self.num_steps,
            self.num_paths,
            self.np_random,
        )
        return self.post_reset()


class HedgeConcBS(HedgeConc):
    def __init__(
        self,
        S0,
        K,
        P,
        maturity,
        r,
        sigma,
        num_paths=128,
        num_steps=250,
        reward_type="abs_diff",
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        feature_dim=17,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=1, # Does it make sense to define this here?
        lda=0.5,
        transaction_cost=False, 
        transaction_fee_rate=0.01,
    ):
        super().__init__(
            S0,
            K,
            P,
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
        self.sigma = sigma

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        self.simulator = BlackScholesSimulator(
            self.S0,
            self.r,
            self.sigma,
            self.maturity,
            self.num_steps,
            self.num_paths,
            self.np_random,
        )
        return self.post_reset()


class HedgeCallHeston(HedgeCall):
    def __init__(
        self,
        S0,
        K,
        maturity,
        r,
        v0,
        theta,
        rho,
        kappa,
        xi,
        num_paths=128,
        num_steps=250,
        reward_type="abs_diff",
        action_dim=1,
        action_low=-1.0, # Hope this makese sense. [0,1] for simple hedge
        action_high=1.0,
        feature_dim=11,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=1, # Does this fix assertion error about action dim
        lda=0.5,
        transaction_cost=False,
        transaction_fee_rate=0.001,
    ):
        super().__init__(
            S0,
            K,
            maturity,
            r,
            v0,
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
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self._last_reset_seed = None  # Track last seed for smart soft reset
        self._is_soft_reset = False  # Flag to tell post_reset whether to recalc option prices

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        # Smart soft reset: skip path regeneration if seed hasn't changed
        # This handles auto-resets at episode end (seed=None) efficiently
        soft_from_options = options.get("soft", False) if options else False
        seed_unchanged = (seed is None) or (seed == self._last_reset_seed)
        has_simulator = hasattr(self, 'simulator') and self.simulator is not None
        soft_enabled = getattr(self, '_soft_reset_enabled', False)
        
        if has_simulator and (soft_from_options or (seed_unchanged and soft_enabled)):
            # Skip path regeneration, just reset internal state
            self._is_soft_reset = True
            return self.post_reset()
        
        # Full reset: create simulator or regenerate paths
        self._is_soft_reset = False
        self._last_reset_seed = seed  # Remember this seed
        if not has_simulator:
            self.simulator = HestonSimulator(
                self.S0,
                self.r,
                self.v0,
                self.theta,
                self.rho,
                self.kappa,
                self.xi,
                self.maturity,
                self.num_steps,
                self.num_paths,
                self.np_random,
            )
        else:
            # Regenerate paths with new seed
            self.simulator.generate_asset_prices()
        # After first full reset, enable soft reset for auto-resets
        self._soft_reset_enabled = True
        return self.post_reset()


class HedgeDocHeston(HedgeDoc):
    def __init__(
        self,
        S0,
        K,
        H,
        maturity,
        r,
        v0,
        theta,
        rho,
        kappa,
        xi,
        num_paths=128,
        num_steps=250,
        reward_type="abs_diff",
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        feature_dim=17,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=10,
        lda=0.5,
        transaction_cost=False,
        transaction_fee_rate=0.001,
    ):
        super().__init__(
            S0,
            K,
            H,
            maturity,
            r,
            v0, 
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
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self._last_reset_seed = None  # Track last seed for smart soft reset
        self._is_soft_reset = False  # Flag to tell post_reset whether to recalc option prices

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        # Smart soft reset: skip path regeneration if seed hasn't changed
        # This handles auto-resets at episode end (seed=None) efficiently
        soft_from_options = options.get("soft", False) if options else False
        seed_unchanged = (seed is None) or (seed == self._last_reset_seed)
        has_simulator = hasattr(self, 'simulator') and self.simulator is not None
        soft_enabled = getattr(self, '_soft_reset_enabled', False)
        
        if has_simulator and (soft_from_options or (seed_unchanged and soft_enabled)):
            # Skip path regeneration, just reset internal state
            self._is_soft_reset = True
            return self.post_reset()
        
        # Full reset: create simulator or regenerate paths
        self._is_soft_reset = False
        self._last_reset_seed = seed  # Remember this seed
        if not has_simulator:
            self.simulator = HestonSimulator(
                self.S0,
                self.r,
                self.v0,
                self.theta,
                self.rho,
                self.kappa,
                self.xi,
                self.maturity,
                self.num_steps,
                self.num_paths,
                self.np_random,
            )
        else:
            # Regenerate paths with new seed
            self.simulator.generate_asset_prices()
        # After first full reset, enable soft reset for auto-resets
        self._soft_reset_enabled = True
        return self.post_reset()

    def _calculate_option_prices(self):
        """Calculate multiple option prices for down-out call hedging."""
        self.option_prices = self.simulator.down_out_call(self.K, self.H)
        self.call_prices = self.simulator.euro_call(self.K)
        self.put_prices = self.simulator.euro_put(self.H**2 / self.K)


class HedgeConcHeston(HedgeConc):
    def __init__(
        self,
        S0,
        K,
        P,
        maturity,
        r,
        v0,
        theta,
        rho,
        kappa,
        xi,
        num_paths=128,
        num_steps=250,
        reward_type="abs_diff",
        action_dim=2,
        action_low=-1.0,
        action_high=1.0,
        feature_dim=17,
        feature_low=-np.inf,
        feature_high=np.inf,
        history_len=1,
        lda=0.5,
        transaction_cost=False,
        transaction_fee_rate=0.001,
    ):
        super().__init__(
            S0,
            K,
            P,
            maturity,
            r,
            v0,  
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
        self.v0 = v0
        self.theta = theta
        self.rho = rho
        self.kappa = kappa
        self.xi = xi
        self._last_reset_seed = None  # Track last seed for smart soft reset
        self._is_soft_reset = False  # Flag to tell post_reset whether to recalc option prices

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(
            (self.num_total_options, self.history_len, self.feature_dim)
        )
        # Smart soft reset: skip path regeneration if seed hasn't changed
        # This handles auto-resets at episode end (seed=None) efficiently
        soft_from_options = options.get("soft", False) if options else False
        seed_unchanged = (seed is None) or (seed == self._last_reset_seed)
        has_simulator = hasattr(self, 'simulator') and self.simulator is not None
        soft_enabled = getattr(self, '_soft_reset_enabled', False)
        
        if has_simulator and (soft_from_options or (seed_unchanged and soft_enabled)):
            # Skip path regeneration, just reset internal state
            self._is_soft_reset = True
            return self.post_reset()
        
        # Full reset: create simulator or regenerate paths
        self._is_soft_reset = False
        self._last_reset_seed = seed  # Remember this seed
        if not has_simulator:
            self.simulator = HestonSimulator(
                self.S0,
                self.r,
                self.v0,
                self.theta,
                self.rho,
                self.kappa,
                self.xi,
                self.maturity,
                self.num_steps,
                self.num_paths,
                self.np_random,
            )
        else:
            # Regenerate paths with new seed
            self.simulator.generate_asset_prices()
        # After first full reset, enable soft reset for auto-resets
        self._soft_reset_enabled = True
        return self.post_reset()

    def _calculate_option_prices(self):
        """Calculate multiple option prices for cash-or-nothing call hedging."""
        self.option_prices = self.simulator.cash_or_nothing_call(self.K, self.P)
        self.call_prices = self.simulator.euro_call(self.K)
        self.put_prices = self.simulator.euro_put(self.K)