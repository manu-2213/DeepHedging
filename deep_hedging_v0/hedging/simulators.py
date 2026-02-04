import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import jax.random as jrandom
# Enable 64-bit precision 
jax.config.update("jax_enable_x64", True)
from scipy.stats import norm
from abc import ABC, abstractmethod
from hedging.black_scholes_model import black_scholes_d1_d2, black_scholes_pricer

class Simulators(ABC):

    @abstractmethod
    def __init__(self, S0, r, maturity, num_steps, num_paths, random_generator):
        assert type(r) == float
        assert type(maturity) == float
        self.S0 = S0
        self.num_assets = len(S0)
        self.r = r
        self.maturity = maturity
        self.num_steps = num_steps
        self.num_paths = num_paths
        self.T = np.linspace(self.maturity, 0, self.num_steps + 1)
        self.dt = self.maturity / self.num_steps
        self.paths = np.zeros((self.num_paths, self.num_assets, self.num_steps + 1))
        self.paths[:, :, 0] = np.tile(self.S0, (self.num_paths, 1))
        self.np_random = random_generator

    @abstractmethod
    def generate_asset_prices(self):
        pass

    @abstractmethod
    def euro_call(self):
        pass

    @abstractmethod
    def euro_put(self):
        pass

    @abstractmethod
    def down_out_call(self):
        pass

    @abstractmethod
    def cash_or_nothing_call(self):
        pass


class BlackScholesSimulator(Simulators):
    def __init__(self, S0, r, sigma, maturity, num_steps, num_paths, random_generator):
        super().__init__(S0, r, maturity, num_steps, num_paths, random_generator)
        assert len(S0) == len(sigma)
        self.sigma = sigma
        self.generate_asset_prices()

    def generate_asset_prices(self):
        S0_expanded = self.paths[:, :, 0:1]
        z = self.np_random.normal(
            0, 1, (self.num_paths, self.num_assets, self.num_steps)
        )
        log_returns = (
            self.r - 0.5 * self.sigma[None, :, None] ** 2
        ) * self.dt + self.sigma[None, :, None] * np.sqrt(self.dt) * z
        log_S = np.log(S0_expanded) + np.cumsum(log_returns, axis=-1)
        asset_prices = np.exp(log_S)
        self.paths[:, :, 1:] = asset_prices

    def expand_dim(self, K):
        S = self.paths[:, :, None, :]
        # (num_simulation, num_asset, 1, num_step+1)
        K = K[None, :, :, None]
        # (1, num_asset, num_strike, 1)
        T = self.T[None, None, None, :]
        # (1, 1, 1, num_step+1)
        sigma = self.sigma[None, :, None, None]
        # (1, num_asset, 1, 1)
        return S, K, T, sigma

    def euro_option_price(self, K, call_flag):
        S, K, T, sigma = self.expand_dim(K)
        # Handle expiration case
        last_prices = np.maximum(0, call_flag * (S - K))[..., -1] # call flag is 1 or - 1
        option_prices = black_scholes_pricer(
            K, T[..., :-1], S[..., :-1], self.r, sigma, call_flag
        )
        return np.concatenate((option_prices, last_prices[..., None]), axis=-1)
        # (num_simulation, num_asset, num_strike, num_step+1)

    def euro_call(self, K):
        return self.euro_option_price(K, 1)

    def euro_put(self, K):
        return self.euro_option_price(K, -1)

    def down_out_call(self, K, H):
        term1 = self.euro_call(K)
        S, K, T, sigma = self.expand_dim(K)
        H = H[None, :, :, None]
        # (1, num_asset, num_strike, 1)
        alpha = 0.5 * (1 - (self.r / (0.5 * sigma**2)))
        S_new = H**2 / S
        last_prices = np.maximum(0, (S_new - K))[..., -1]
        option_prices = black_scholes_pricer(
            K, T[..., :-1], S_new[..., :-1], self.r, sigma, 1
        )
        term2 = (S / H) ** (2 * alpha) * np.concatenate(
            (option_prices, last_prices[..., None]), axis=-1
        )
        doc_price = term1 - term2
        not_hit_barrier = np.cumprod(S > H, axis=-1)
        return doc_price * not_hit_barrier

    def cash_or_nothing_call(self, K, P):
        S, K, T, sigma = self.expand_dim(K)
        P = P[None, :, :, None]
        # Handle expiration case
        last_prices = ((S - K)[..., -1] > 0)  * P[..., -1]
        # (1, num_asset, num_strike, 1)
        _, d2 = black_scholes_d1_d2(K, T[..., :-1], S[..., :-1], self.r, sigma)
        option_prices = np.exp(-self.r * T[..., :-1]) * norm.cdf(d2) * P
        return np.concatenate((option_prices, last_prices[..., None]), axis=-1)

class HestonSimulator:
    """
    JAX-accelerated Heston stochastic volatility simulator.
    
    Uses JAX for:
    - JIT compilation of hot paths
    - lax.scan for efficient loop-free path generation
    - Vectorized characteristic function integration
    - Automatic GPU/TPU acceleration when available
    
    Parameters
    ----------
    S0 : array-like
        Initial spot prices, shape (num_assets,)
    r : float
        Risk-free rate
    v0 : array-like
        Initial variance, shape (num_assets,)
    theta : array-like
        Long-term variance, shape (num_assets,)
    rho : array-like
        Correlation between spot and variance, shape (num_assets,)
    kappa : array-like
        Mean reversion speed, shape (num_assets,)
    xi : array-like
        Volatility of variance, shape (num_assets,)
    maturity : float
        Time to maturity in years
    num_steps : int
        Number of time steps
    num_paths : int
        Number of Monte Carlo paths
    seed : int
        Random seed for reproducibility
    n_points : int
        Number of integration points for characteristic function
    mc_paths : int
        Number of paths for nested Monte Carlo (barrier pricing)
    phi_max : float
        Upper limit of integration for characteristic function
        
    Attributes
    ----------
    paths : ndarray
        Simulated spot prices, shape (num_paths, num_assets, num_steps+1)
    v : ndarray
        Simulated variance paths, shape (num_paths, num_assets, num_steps+1)
    T : ndarray
        Time-to-maturity grid (decreasing from maturity to 0)
        
    Notes
    -----
    On HPC with GPU:
    - JAX automatically detects and uses available GPUs
    - Use `jax.devices()` to check available devices
    - Expected 10-100x speedup over CPU for large num_paths
    
    Example
    -------
    >>> sim = HestonSimulator(
    ...     S0=np.array([100.0]), r=0.03, v0=np.array([0.05]),
    ...     theta=np.array([0.05]), rho=np.array([-0.8]),
    ...     kappa=np.array([5.0]), xi=np.array([0.5]),
    ...     maturity=0.5, num_steps=100, num_paths=100, seed=42
    ... )
    >>> prices = sim.euro_call(K=np.array([[90, 100, 110]]))
    """
    
    def __init__(
        self,
        S0,
        r,
        v0,
        theta,
        rho,
        kappa,
        xi,
        maturity,
        num_steps,
        num_paths,
        seed=42,
        n_points=1000,
        mc_paths=5000,
        phi_max=50.0,
    ):
        # Validate inputs
        S0 = np.asarray(S0)
        v0 = np.asarray(v0)
        theta = np.asarray(theta)
        rho = np.asarray(rho)
        kappa = np.asarray(kappa)
        xi = np.asarray(xi)
        
        assert len(S0) == len(v0) == len(theta) == len(rho) == len(kappa) == len(xi), (
            f"All parameter arrays must have the same length"
        )
        
        # Store as JAX arrays for GPU compatibility
        self.S0 = jnp.asarray(S0, dtype=jnp.float64)
        self.r = float(r)
        self.v0 = jnp.asarray(v0, dtype=jnp.float64)
        self.theta = jnp.asarray(theta, dtype=jnp.float64)
        self.rho = jnp.asarray(rho, dtype=jnp.float64)
        self.kappa = jnp.asarray(kappa, dtype=jnp.float64)
        self.xi = jnp.asarray(xi, dtype=jnp.float64)
        
        self.num_assets = len(S0)
        self.maturity = float(maturity)
        self.num_steps = int(num_steps)
        self.num_paths = int(num_paths)
        self.dt = self.maturity / self.num_steps
        self.T = jnp.linspace(self.maturity, 0.0, self.num_steps + 1)
        
        self.n_points = n_points
        self.mc_paths = mc_paths
        self.phi_max = phi_max
        
        # Handle seed: convert NumPy Generator to integer if needed
        if hasattr(seed, 'integers'):
            # It's a NumPy Generator, extract a seed from it
            self.seed = int(seed.integers(0, 2**31 - 1))
        elif seed is None:
            self.seed = 42
        else:
            self.seed = int(seed)
        
        # Pre-compute phi grid for characteristic function integration
        self.phi_grid = jnp.linspace(1e-8, phi_max, n_points)
        self.dx = float(self.phi_grid[1] - self.phi_grid[0])
        
        # Generate paths using JAX
        self._generate_paths(self.seed)
        print(f"HestonSimulator initialized on {jax.devices()[0]}")
    
    def _generate_paths(self, seed):
        """
        Generate Heston paths using JAX with lax.scan.
        
        Uses Euler-Maruyama discretization for both variance and log-price.
        The variance is floored at 0 to prevent negative values.
        """
        key = jrandom.PRNGKey(seed)
        P, A, N = self.num_paths, self.num_assets, self.num_steps
        dt = self.dt
        
        # Generate all random draws upfront
        key, subkey = jrandom.split(key)
        z_s = jrandom.normal(subkey, shape=(P, A, N))
        key, subkey = jrandom.split(key)
        z_v = jrandom.normal(subkey, shape=(P, A, N))
        
        # Correlate z_v with z_s: dW_v = rho*dW_s + sqrt(1-rho^2)*dW_perp
        rho_exp = self.rho[None, :, None]
        z_v = rho_exp * z_s + jnp.sqrt(1.0 - rho_exp**2) * z_v
        
        # Broadcast parameters for vectorized operations
        kappa = self.kappa[None, :]
        theta = self.theta[None, :]
        xi = self.xi[None, :]
        
        def step(carry, inputs):
            """Single Euler-Maruyama step for Heston dynamics."""
            v_t, log_S = carry
            z_s_t, z_v_t = inputs
            
            v_pos = jnp.maximum(v_t, 0.0)
            sqrt_vdt = jnp.sqrt(v_pos * dt)
            
            # Euler step for variance: dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
            v_next = v_t + kappa * (theta - v_pos) * dt + xi * sqrt_vdt * z_v_t
            
            # Log-return step: d(log S) = (r - 0.5*v)*dt + sqrt(v)*dW_s
            log_return = (self.r - 0.5 * v_pos) * dt + sqrt_vdt * z_s_t
            log_S_next = log_S + log_return
            
            return (v_next, log_S_next), (v_next, log_S_next)
        
        # Initial state
        v0_exp = jnp.broadcast_to(self.v0[None, :], (P, A))
        log_S0 = jnp.log(jnp.broadcast_to(self.S0[None, :], (P, A)))
        
        # Transpose for scan: (P, A, N) -> (N, P, A)
        z_s_t = jnp.transpose(z_s, (2, 0, 1))
        z_v_t = jnp.transpose(z_v, (2, 0, 1))
        
        # Run scan (compiled loop)
        _, (v_history, log_S_history) = lax.scan(step, (v0_exp, log_S0), (z_s_t, z_v_t))
        
        # Reshape: (N, P, A) -> (P, A, N) and prepend initial values
        v_history = jnp.transpose(v_history, (1, 2, 0))
        log_S_history = jnp.transpose(log_S_history, (1, 2, 0))
        
        # Store variance paths
        self._V = jnp.concatenate([v0_exp[:, :, None], v_history], axis=-1)
        
        # Store spot paths
        S_history = jnp.exp(log_S_history)
        S0_exp = jnp.broadcast_to(self.S0[None, :, None], (P, A, 1))
        self._S = jnp.concatenate([S0_exp, S_history], axis=-1)
        
        # NumPy-compatible aliases for backward compatibility
        self.paths = np.array(self._S)
        self.v = np.array(self._V)
    
    @property
    def S(self):
        """Spot price paths as JAX array."""
        return self._S
    
    @property
    def V(self):
        """Variance paths as JAX array."""
        return self._V
    
    def _heston_cf(self, phi, S, T, v0, kappa, theta, rho, xi, r, trap, u):
        """
        Heston characteristic function for P1 (u=0.5) or P2 (u=-0.5) measure.
        
        Uses the 'trap' formulation (trap=1) for numerical stability.
        """
        x = jnp.log(S)
        b = kappa - rho * xi if u == 0.5 else kappa
        a = kappa * theta
        
        iφ = 1j * phi
        d = jnp.sqrt((rho * xi * iφ - b)**2 - xi**2 * (2 * u * iφ - phi**2))
        g = (b - rho * xi * iφ + d) / (b - rho * xi * iφ - d)
        
        if trap == 1:
            c = 1.0 / g
            D = ((b - rho * xi * iφ - d) / (xi**2)) * ((1 - jnp.exp(-d * T)) / (1 - c * jnp.exp(-d * T)))
            G = (1 - c * jnp.exp(-d * T)) / (1 - c)
            C = r * iφ * T + (a / xi**2) * ((b - rho * xi * iφ - d) * T - 2.0 * jnp.log(G))
        else:
            G = (1 - g * jnp.exp(d * T)) / (1 - g)
            D = ((b - rho * xi * iφ + d) / (xi**2)) * ((1 - jnp.exp(d * T)) / (1 - g * jnp.exp(d * T)))
            C = r * iφ * T + (a / xi**2) * ((b - rho * xi * iφ + d) * T - 2.0 * jnp.log(G))
        
        return jnp.exp(C + D * v0 + 1j * phi * x)
    
    def _compute_probability(self, phi, S, K, T, v0, kappa, theta, rho, xi, r, trap, u):
        """Compute P1 or P2 probability via numerical integration."""
        cf = self._heston_cf(phi, S, T, v0, kappa, theta, rho, xi, r, trap, u)
        integrand = jnp.real((jnp.exp(-1j * phi * jnp.log(K)) * cf) / (1j * phi))
        # Flatten integrand to 1D and integrate
        integrand_flat = np.asarray(integrand).flatten()
        integral = np.trapezoid(integrand_flat, dx=self.dx)
        return 0.5 + (1.0 / np.pi) * integral
    
    def euro_call(self, K, trap=1):
        """
        Price European call options using dual-integral Heston formula.
        
        Parameters
        ----------
        K : ndarray
            Strike prices, shape (num_assets, num_strikes)
        trap : int
            Formulation choice (1 = trap, 0 = standard)
            
        Returns
        -------
        ndarray
            Call prices, shape (num_paths, num_assets, num_strikes, num_steps+1)
        """
        K = np.asarray(K)
        P, A, N = self.num_paths, self.num_assets, self.num_steps
        M = K.shape[1]
        
        # Expand dimensions
        S = np.array(self.S)[:, :, None, :]  # (P, A, 1, N+1)
        K_exp = np.broadcast_to(K[None, :, :, None], (P, A, M, N + 1))
        T_exp = np.broadcast_to(np.array(self.T)[None, None, None, :], (P, A, M, N + 1))
        
        # Terminal payoff (intrinsic value at maturity)
        last_prices = np.maximum(0.0, S[..., -1] - K_exp[..., -1])
        
        if N == 0:
            return last_prices[..., None]
        
        # Price at each time step using characteristic function integration
        prices = np.zeros((P, A, M, N))
        phi = self.phi_grid
        
        for p in range(P):
            for a in range(A):
                kappa_a = float(self.kappa[a])
                theta_a = float(self.theta[a])
                rho_a = float(self.rho[a])
                xi_a = float(self.xi[a])
                v0_a = float(self.v0[a])
                
                for m in range(M):
                    K_am = float(K[a, m])
                    for t in range(N):
                        S_pamt = float(S[p, a, 0, t])
                        T_t = float(T_exp[p, a, m, t])
                        
                        if T_t <= 0:
                            prices[p, a, m, t] = max(0.0, S_pamt - K_am)
                        else:
                            P1 = self._compute_probability(
                                phi, S_pamt, K_am, T_t, v0_a,
                                kappa_a, theta_a, rho_a, xi_a, self.r, trap, u=0.5
                            )
                            P2 = self._compute_probability(
                                phi, S_pamt, K_am, T_t, v0_a,
                                kappa_a, theta_a, rho_a, xi_a, self.r, trap, u=-0.5
                            )
                            prices[p, a, m, t] = S_pamt * P1 - K_am * np.exp(-self.r * T_t) * P2
        
        return np.concatenate([prices, last_prices[..., None]], axis=-1)
    
    def euro_put(self, K, trap=1):
        """Price European put options via put-call parity."""
        C = self.euro_call(K, trap=trap)
        K = np.asarray(K)
        P, A, N = self.num_paths, self.num_assets, self.num_steps
        M = K.shape[1]
        
        S = np.array(self.S)[:, :, None, :]
        K_exp = np.broadcast_to(K[None, :, :, None], (P, A, M, N + 1))
        T_exp = np.broadcast_to(np.array(self.T)[None, None, None, :], (P, A, M, N + 1))
        
        return C - S + K_exp * np.exp(-self.r * T_exp)
    
    def cash_or_nothing_call(self, K, Q=1.0, trap=1):
        """
        Price cash-or-nothing call options.
        
        Pays Q if S_T > K, 0 otherwise.
        """
        # Ensure Q is a scalar
        Q = float(np.asarray(Q).flatten()[0]) if hasattr(Q, '__iter__') else float(Q)
        K = np.asarray(K)
        P, A, N = self.num_paths, self.num_assets, self.num_steps
        M = K.shape[1]
        
        S = np.array(self.S)[:, :, None, :]
        K_exp = np.broadcast_to(K[None, :, :, None], (P, A, M, N + 1))
        T_exp = np.broadcast_to(np.array(self.T)[None, None, None, :], (P, A, M, N + 1))
        
        # Terminal payoff
        last_prices = Q * (S[..., -1] > K_exp[..., -1]).astype(float)
        
        if N == 0:
            return last_prices[..., None]
        
        # Price via P2 probability
        prices = np.zeros((P, A, M, N))
        phi = self.phi_grid
        
        for p in range(P):
            for a in range(A):
                kappa_a = float(self.kappa[a])
                theta_a = float(self.theta[a])
                rho_a = float(self.rho[a])
                xi_a = float(self.xi[a])
                v0_a = float(self.v0[a])
                
                for m in range(M):
                    K_am = float(K[a, m])
                    for t in range(N):
                        S_pamt = float(S[p, a, 0, t])
                        T_t = float(T_exp[p, a, m, t])
                        
                        if T_t <= 0:
                            prices[p, a, m, t] = Q * (S_pamt > K_am)
                        else:
                            P2 = self._compute_probability(
                                phi, S_pamt, K_am, T_t, v0_a,
                                kappa_a, theta_a, rho_a, xi_a, self.r, trap, u=-0.5
                            )
                            prices[p, a, m, t] = Q * np.exp(-self.r * T_t) * P2
        
        return np.concatenate([prices, last_prices[..., None]], axis=-1)
    
    def down_out_call(self, K, H, mc_paths=None, enforce_upper_bound=True, seed=None):
        """
        Price down-and-out call options via nested Monte Carlo.
        
        Parameters
        ----------
        K : ndarray
            Strike prices, shape (num_assets, num_strikes)
        H : float or ndarray
            Barrier levels
        mc_paths : int, optional
            Number of nested MC paths (default: self.mc_paths)
        enforce_upper_bound : bool
            Clip to vanilla call price for stability
        seed : int, optional
            Random seed for nested MC
            
        Returns
        -------
        ndarray
            Barrier option prices, shape (num_paths, num_assets, num_strikes, num_steps+1)
        """
        mc_samples = self.mc_paths if mc_paths is None else mc_paths
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        
        P, A, N = self.num_paths, self.num_assets, self.num_steps
        dt = self.dt
        S_outer = np.array(self.S)
        v_outer = np.array(self.V)
        
        K = np.asarray(K)
        M = K.shape[1]
        H = np.asarray(H)
        if H.ndim == 0:
            H_full = np.full((A, M), float(H))
        else:
            H_full = np.broadcast_to(H if H.shape == (A, M) else H.reshape(A, -1), (A, M))
        
        S_exp = S_outer[:, :, None, :]
        v_exp = v_outer[:, :, None, :]
        K_exp = np.broadcast_to(K[None, :, :, None], (P, A, M, N + 1))
        H_exp = np.broadcast_to(H_full[None, :, :, None], (P, A, M, N + 1))
        alive_prefix = np.minimum.accumulate(S_exp > H_exp, axis=-1)
        
        V = np.zeros((P, A, M, N + 1))
        V[..., -1] = np.maximum(S_exp[..., -1] - K_exp[..., -1], 0.0) * alive_prefix[..., -1]
        
        if N == 0:
            return V
        
        if enforce_upper_bound:
            vanilla = self.euro_call(K)
        
        kappa = np.array(self.kappa)
        theta = np.array(self.theta)
        rho = np.array(self.rho)
        xi = np.array(self.xi)
        r = self.r
        
        def _cond_batch(S0_vec, v0_vec, K_vec, H_vec, asset_idx_vec, steps_rem, T_rem):
            B = S0_vec.shape[0]
            if B == 0:
                return np.zeros((0,), dtype=float)
            kappa_b = kappa[asset_idx_vec]
            theta_b = theta[asset_idx_vec]
            rho_b = rho[asset_idx_vec]
            xi_b = xi[asset_idx_vec]
            
            S = np.broadcast_to(S0_vec[None, :], (mc_samples, B)).copy()
            v = np.broadcast_to(np.maximum(v0_vec, 0.0)[None, :], (mc_samples, B)).copy()
            alive_mask = np.ones_like(S, dtype=bool)
            
            for _ in range(steps_rem):
                z_s = rng.normal(0.0, 1.0, (mc_samples, B))
                z_v = rng.normal(0.0, 1.0, (mc_samples, B))
                z_v = rho_b * z_s + np.sqrt(1.0 - rho_b**2) * z_v
                
                v_pos = np.maximum(v, 0.0)
                sqrt_vdt = np.sqrt(v_pos * dt)
                v += kappa_b * (theta_b - v_pos) * dt + xi_b * sqrt_vdt * z_v
                S *= np.exp((r - 0.5 * v_pos) * dt + sqrt_vdt * z_s)
                alive_mask &= (S > H_vec)
            
            payoff = np.maximum(S - K_vec, 0.0) * alive_mask
            return np.exp(-r * T_rem) * payoff.mean(axis=0)
        
        for t in range(N - 1, -1, -1):
            steps_rem = N - t
            T_rem = float(self.T[t])
            alive_mask_t = alive_prefix[..., t]
            idx_p, idx_a, idx_m = np.where(alive_mask_t)
            if idx_p.size == 0:
                V[..., t] = 0.0
                continue
            S0_vec = S_exp[idx_p, idx_a, 0, t]
            v0_vec = v_exp[idx_p, idx_a, 0, t]
            K_vec = K_exp[idx_p, idx_a, idx_m, t]
            H_vec = H_exp[idx_p, idx_a, idx_m, t]
            est = _cond_batch(S0_vec, v0_vec, K_vec, H_vec, idx_a, steps_rem, T_rem)
            V[idx_p, idx_a, idx_m, t] = est
            if enforce_upper_bound:
                V[idx_p, idx_a, idx_m, t] = np.minimum(V[idx_p, idx_a, idx_m, t], vanilla[idx_p, idx_a, idx_m, t])
        
        return V
    
    def generate_asset_prices(self):
        """Regenerate paths with a new random seed."""
        self._generate_paths(self.seed + 1)
        self.seed += 1