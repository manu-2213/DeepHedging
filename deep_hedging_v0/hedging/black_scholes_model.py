import numpy as np
from scipy.stats import norm


def black_scholes_d1_d2(K, T, S, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_pricer(K, T, S, r, sigma, call_flag):
    d1, d2 = black_scholes_d1_d2(K, T, S, r, sigma)
    term1 = call_flag * S * norm.cdf(call_flag * d1)
    term2 = call_flag * K * np.exp(-r * T) * norm.cdf(call_flag * d2)
    option_prices = term1 - term2
    return option_prices
