import numpy as np
import pandas as pd
from scipy.stats import norm


# Pricing Models

def black_scholes_price(S, K, T, r, sigma, is_call=True):

    S = np.asarray(S)
    T = np.asarray(T)
    
    # Use np.where to check  T <= 0 condition on numpy array:
    payoff = np.where(is_call, np.maximum(0.0, S - K), np.maximum(0.0, K - S))
    
    # Compute d1 and d2 if T > 0.
    d1 = np.where(
        T > 0,
        (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)),
        0.0 # T <= 0
    )
    d2 = np.where(T > 0, d1 - sigma * np.sqrt(T), 0.0)
    
    # Black-Scholes price where T > 0
    sign = 1 if is_call else -1
    bs_price = sign * (S * norm.cdf(sign * d1) - K * np.exp(-r * T) * norm.cdf(sign * d2))
    
    # Where T > 0, use the BS price; otherwise, use payoff.
    return np.where(T > 0, bs_price, payoff)


def black_scholes_delta(S, K, T, r, sigma, is_call=True):
    
    S = np.asarray(S)
    T = np.asarray(T)
    
    # For T <= 0, define delta as the derivative of the payoff:
    # For calls: 1 if S > K, else 0; for puts: 0 if S > K, else -1.
    delta_payoff = np.where(is_call, np.where(S > K, 1.0, 0.0), np.where(S > K, 0.0, -1.0))
    
    # Compute d1 where T > 0
    d1 = np.where(
        T > 0,
        (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)),
        0.0
    )
    
    # Compute the Black-Scholes delta for T > 0.
    if is_call:
        delta_bs = norm.cdf(d1)
    else:
        delta_bs = norm.cdf(d1) - 1  # For puts
    
    # Use BS delta for T > 0; otherwise, use the payoff delta.
    return np.where(T > 0, delta_bs, delta_payoff)


def black_scholes_price2(S, K, T, r, sigma, is_call=True):
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return (2 * is_call - 1) * S * norm.cdf((2 * is_call - 1) * d1) + \
        (- 2 * is_call + 1) *  K * np.e ** (- r * T) * norm.cdf((2 * is_call - 1) * d2)

def black_scholes_delta2(S, K, T, r, sigma, is_call=True):
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        else:
            return 0.0 if S > K else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return norm.cdf(d1) - (not is_call) # N(d1) - 1 if is_call = False


def bsm_until_maturity(dataframe, K, mu, sigma, T, option_pricer, is_call=True) -> pd.DataFrame:
    bsm_prices = []
    
    # Loop over the rows of the dataframe and compute bsm price for all assets
    for i, row in dataframe.iterrows():
        t = T - row["Time"]  # Time until maturity
        prices = row[1:].values  # Ignore time column
        
        # Call black_scholes_price for each price in the row
        bsm_row_prices = [option_pricer(price, K, mu, sigma, t, is_call) for price in prices]
        bsm_prices.append(bsm_row_prices)

    # Convert the list of bsm prices to a DataFrame
    bsm_prices_array = np.array(bsm_prices)
    df = pd.DataFrame(bsm_prices_array, columns=[f"Asset_{i+1}" for i in range(len(dataframe.columns) - 1)])
    
    # Add the "Time" column back to the DataFrame
    df.insert(0, "Time", dataframe["Time"])

    return df


def delta_until_maturity(dataframe, K, mu, sigma, T, delta, is_call=True) -> pd.DataFrame:
    
    deltas = []
    
    # Loop over the rows of the dataframe and compute the delta for all assets
    for i, row in dataframe.iterrows():
        t = T - row["Time"]  # Time until maturity
        prices = row[1:].values  # Ignore time column
        
        # Call black_scholes_price for each price in the row
        delta = [delta(price, K, t, mu, sigma, is_call) for price in prices]
        deltas.append(delta)

    # Convert the list of deltas to a DataFrame
    deltas_array = np.array(deltas)
    df = pd.DataFrame(deltas_array, columns=[f"Delta_Asset_{i+1}" for i in range(len(dataframe.columns) - 1)])
    
    # Add the "Time" column back to the DataFrame
    df.insert(0, "Time", dataframe["Time"])

    return df