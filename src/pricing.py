import numpy as np
import pandas as pd
from scipy.stats import norm
import numba as nb


# Pricing Models

@nb.njit
def black_scholes_price(S, K, mu, sigma, T, is_call = True):

    if T <= 0:
        return np.maximum(0, S - K)  # Handle expiration case
    
    d1 = ( np.log(S / K) + (mu + 0.5 * sigma ** 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return (2 * is_call - 1) * S * norm.cdf((2 * is_call - 1) * d1) + \
        (- 2 * is_call + 1) *  K * np.e ** (- mu * T) * norm.cdf((2 * is_call - 1) * d2)  



def bsm_until_maturity(dataframe, K, mu, sigma, T, is_call=True):
    bsm_prices = []
    
    # Loop over the rows of the dataframe and compute bsm price for all assets
    for i, row in dataframe.iterrows():
        t = T - row["Time"]  # Time until maturity
        prices = row[1:].values  # Ignore time column
        
        # Call black_scholes_price for each price in the row
        bsm_row_prices = [black_scholes_price(price, K, mu, sigma, t, is_call) for price in prices]
        bsm_prices.append(bsm_row_prices)

    # Convert the list of bsm prices to a DataFrame
    bsm_prices_array = np.array(bsm_prices)
    df = pd.DataFrame(bsm_prices_array, columns=[f"Asset_{i+1}" for i in range(len(dataframe.columns) - 1)])
    
    # Add the "Time" column back to the DataFrame
    df.insert(0, "Time", dataframe["Time"])

    return df

def bs_delta(S, K, T, r, sigma, is_call = True):
    if T <= 0:
        return 1.0 if S > K else 0.0 # Handle expiration case
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) - (not is_call) # Formula for put is N(d1) - 1
    return delta

def delta_until_maturity(dataframe, K, mu, sigma, T, is_call=True):
    
    deltas = []
    
    # Loop over the rows of the dataframe and compute the delta for all assets
    for i, row in dataframe.iterrows():
        t = T - row["Time"]  # Time until maturity
        prices = row[1:].values  # Ignore time column
        
        # Call black_scholes_price for each price in the row
        delta = [bs_delta(price, K, t, mu, sigma, is_call) for price in prices]
        deltas.append(delta)

    # Convert the list of deltas to a DataFrame
    deltas_array = np.array(deltas)
    df = pd.DataFrame(deltas_array, columns=[f"Delta_Asset_{i+1}" for i in range(len(dataframe.columns) - 1)])
    
    # Add the "Time" column back to the DataFrame
    df.insert(0, "Time", dataframe["Time"])

    return df