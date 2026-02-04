import pandas as pd
import numpy as np
from numba import njit
from scipy.integrate import trapezoid





def get_BSM_data(year: str, market: str):
    path = f"./option_data/{market}_{year}.json.bz2"
    df =  pd.read_json(path, compression='bz2', orient='index')

    S0 = df["forward_price"].iloc[0]

    idx = df.groupby("date")["tau"].idxmin()
    daily_spot = df.loc[idx, ["date", "forward_price"]].set_index("date").sort_index()
    daily_spot.shape

    daily_spot["log_ret"] = np.log(daily_spot["forward_price"] / daily_spot["forward_price"].shift(1))

    log_returns = daily_spot['log_ret'].dropna()

    annualized_return = log_returns.mean() * 252

    annualized_vol = log_returns.std() * np.sqrt(252)

    strikes = df["strike_price"].unique().tolist()

    return S0, annualized_return, annualized_vol, strikes


def sample_strikes(S0: float, K: list, n_strikes: int, sparcity: int) -> list:
    """
    Inputs:
        S0: initial asset price
        K: List of strike prices available from data
        n_strikes: Number of sampled strikes to return
        sparcity: Controls how different the strike prices are by skipping some elements of K
    
    Returns:
        return_list: ordered list with an odd number of elements, the one in the center corresponds
        to the closest ATM strike price given S0
    """
    K.sort()
    dist = float('inf')
    for i in range(len(K)):
        dist = min(dist, abs(S0 - K[i]))
        if abs(S0 - K[i]) != dist:
            # minimum dist has passed
            if i > 0:
                atm_K = K[i-1]
            else:
                print("Nope")
            
            range_k = min(len(K)-i-1, i-1)

            return_list = [atm_K]

            if n_strikes * sparcity < range_k * 2:
                j = 1
                for _ in range(n_strikes // 2):
                    return_list.append(K[i - 1 + j * sparcity])
                    return_list.append(K[i - 1 - j * sparcity])
                    j  += 1
            return_list.sort()
            return return_list
    return None



def heston_call_price(S, K, T, r, v0, kappa, theta, rho, xi, trap=1, phi_max=50.0, n_points=2000):
    """
    Price a European call under Heston using characteristic function.
    """
    if T <= 0:
        return max(S - K, 0.0)
    
    phi = np.linspace(1e-8, phi_max, n_points)
    dx = phi[1] - phi[0]
    
    a = kappa * theta
    x = np.log(S / K)
    
    P1 = _heston_probability(phi, x, T, r, v0, kappa, theta, rho, xi, a, j=1, trap=trap, dx=dx)
    P2 = _heston_probability(phi, x, T, r, v0, kappa, theta, rho, xi, a, j=2, trap=trap, dx=dx)
    
    call_price = S * P1 - K * np.exp(-r * T) * P2
    return max(call_price, 0.0)

def heston_put_price(S, K, T, r, v0, kappa, theta, rho, xi, trap=1, phi_max=50.0, n_points=2000):
    """Price a European put under Heston using put-call parity."""
    call = heston_call_price(S, K, T, r, v0, kappa, theta, rho, xi, trap, phi_max, n_points)
    put = call - S + K * np.exp(-r * T)
    return max(put, 0.0)

def _heston_probability(phi, x, T, r, v0, kappa, theta, rho, xi, a, j=1, trap=1, dx=1.0):
    """
    Compute Heston probability P_j for j=1,2.
    """
    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa
    
    iφ = 1j * phi
    
    d = np.sqrt((rho * xi * iφ - b)**2 - xi**2 * (2 * u * iφ - phi**2))
    g = (b - rho * xi * iφ + d) / (b - rho * xi * iφ - d)
    
    if trap == 1:
        c = 1.0 / g
        D = ((b - rho * xi * iφ - d) / (xi**2)) * ((1 - np.exp(-d * T)) / (1 - c * np.exp(-d * T)))
        G = (1 - c * np.exp(-d * T)) / (1 - c)
        C = r * iφ * T + (a / xi**2) * ((b - rho * xi * iφ - d) * T - 2.0 * np.log(G))
    else:
        G = (1 - g * np.exp(d * T)) / (1 - g)
        D = ((b - rho * xi * iφ + d) / (xi**2)) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        C = r * iφ * T + (a / xi**2) * ((b - rho * xi * iφ + d) * T - 2.0 * np.log(G))
    
    f = np.exp(C + D * v0 + 1j * phi * x)
    integrand = np.real((np.exp(-1j * phi * np.log(1.0)) * f) / (1j * phi))
    integral = trapezoid(integrand, dx=dx)
    prob = 0.5 + (1.0 / np.pi) * integral
    
    return prob



def convert_data_heston(data: dict):
    
    years = sorted(data.keys())  # deterministic order

    kappa = np.array([data[y]["kappa"] for y in years], dtype=float)
    theta = np.array([data[y]["theta"] for y in years], dtype=float)
    rho   = np.array([data[y]["rho"]   for y in years], dtype=float)
    sigma = np.array([data[y]["xi"]    for y in years], dtype=float)  
    lda   = np.zeros_like(kappa)  # lda 0s for now

    params = {
        "kappa": kappa,
        "theta": theta,
        "rho": rho,
        "sigma": sigma,
        "lda": lda
    }

    S0 = np.array([data[y]["S0"] for y in years], dtype=float)
    v0 = np.array([data[y]["v0"] for y in years], dtype=float)

    # K_sampled as 2D matrix
    K = np.array([data[y]["K_sampled"] for y in years], dtype=float)

    return params, S0, K, v0
