# environment.py
import gymnasium as gym
import numpy as np
from scipy.stats import norm

class DeepHedgingEnv(gym.Env):
    """
    Gymnasium environment for dynamic option hedging.
    
    At reset, the environment computes the initial option price and hedge (delta)
    using Black–Scholes formulas. At each step the agent supplies a new set of hedge ratios;
    the environment updates the cash account (with accrued interest and trading costs) and
    evolves the stock prices via geometric Brownian motion.
    
    The state is a flattened vector consisting of:
      [stock prices (num_asset),
       cash (flattened over assets & strikes),
       current hedge (flattened),
       current time-to-maturity (scalar)]
    
    The action is a matrix (num_asset × num_strike) of new hedge ratios.
    
    At terminal time the reward is defined as the (average) difference between the option payoff
    and the hedging portfolio’s value.
    The environment also stores a history of the theoretical option prices and portfolio values.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, S0, K, sigma, r, num_step=260):
        super().__init__()
        self.S0 = np.array(S0)          # (num_asset,)
        self.K = np.array(K)            # (num_asset, num_strike)
        self.sigma = np.array(sigma)    # (num_asset,)
        self.r = r
        self.num_step = num_step
        self.dt = 1.0 / num_step
        self.T_arr = np.linspace(1, 0, num_step+1)  # time-to-maturity vector
        self.num_asset = len(S0)
        self.num_strike = self.K.shape[1]
        
        obs_dim = self.num_asset + self.num_asset*self.num_strike*2 + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-100, high=100, shape=(self.num_asset, self.num_strike), dtype=np.float32
        )
        self.reset()
    
    def reset(self, seed=None, options=None):
        self.t = 0
        self.S = self.S0.copy()  # current stock prices, shape (num_asset,)
        # Compute initial option price and delta (using Black–Scholes)
        S_in = self.S[None, :, None]  # shape (1, num_asset, 1)
        T0 = np.array([self.T_arr[0]])
        option_prices = black_scholes_call(S_in, self.K, T0, self.r, self.sigma)  # (1, num_asset, num_strike, 1)
        option_prices = np.squeeze(option_prices, axis=(0, 3))  # (num_asset, num_strike)
        initial_delta = black_scholes_delta(S_in, self.K, T0, self.r, self.sigma)  # (1, num_asset, num_strike, 1)
        initial_delta = np.squeeze(initial_delta, axis=(0, 3))  # (num_asset, num_strike)
        self.current_delta = initial_delta.copy()  # hedge positions
        self.cash = option_prices - self.current_delta * self.S[:, None]  # (num_asset, num_strike)
        
        # For plotting
        self.history = {"option_prices": [], "portfolio_values": []}
        self._record_history()
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.concatenate([
            self.S, 
            self.cash.flatten(), 
            self.current_delta.flatten(),
            np.array([self.T_arr[self.t]])
        ])
        return obs.astype(np.float32)
    
    def step(self, action):
        # action: new hedge ratios (num_asset, num_strike)
        trade = action - self.current_delta  # shares traded per asset/strike
        self.cash = self.cash * np.exp(self.r * self.dt) - trade * self.S[:, None]
        self.current_delta = action.copy()
        # Update stock prices via geometric Brownian motion.
        z = np.random.normal(size=self.num_asset)
        self.S = self.S * np.exp((self.r - 0.5 * self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*z)
        self.t += 1
        done = self.t >= self.num_step
        if done:
            option_payoff = np.maximum(self.S[:, None] - self.K, 0)
            portfolio_value = self.cash + self.current_delta * self.S[:, None]
            reward = np.mean(option_payoff - portfolio_value)
        else:
            reward = 0.0
        self._record_history()
        # Gymnasium step returns: (obs, reward, terminated, truncated, info)
        return self._get_obs(), reward, done, False, {}
    
    def _record_history(self):
        # Record theoretical option price and portfolio value at current time.
        T_current = np.array([self.T_arr[self.t]])
        S_in = self.S[None, :, None]
        option_price = black_scholes_call(S_in, self.K, T_current, self.r, self.sigma)
        option_price = np.squeeze(option_price, axis=(0,3))  # (num_asset, num_strike)
        portfolio_value = self.cash + self.current_delta * self.S[:, None]
        self.history["option_prices"].append(option_price)
        self.history["portfolio_values"].append(portfolio_value)
    
    def render(self, mode="human"):
        print(f"Step {self.t}:")
        print(f"  Stock Prices: {self.S}")
        print(f"  Cash:\n{self.cash}")
        print(f"  Hedge:\n{self.current_delta}")
        print(f"  Time-to-maturity: {self.T_arr[self.t]}")

def black_scholes_call(S, K, T, r, sigma):
    S = S[..., None, :]  # (num_sim, num_asset, 1, num_step)
    K = K[None, ..., None]  # (1, num_asset, num_strike, 1)
    T = T[None, None, None, :]  # (1, 1, 1, num_step)
    sigma = sigma[None, :, None, None]  # (1, num_asset, 1, 1)
    call_prices = np.maximum(0, S - K) * (T <= 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_prices = np.where(T > 0, S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2), call_prices)
    return call_prices

def black_scholes_delta(S, K, T, r, sigma):
    S = S[..., None, :]
    K = K[None, ..., None]
    T = T[None, None, None, :]
    sigma = sigma[None, :, None, None]
    delta = np.where(T <= 0, np.where(S > K, 1.0, 0.0), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    delta = np.where(T > 0, norm.cdf(d1), delta)
    return delta
