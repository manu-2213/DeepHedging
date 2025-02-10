import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# TO DO: Handle T = 0 in BSM

# Define functions to simulate data

def GBM(S0_array, mu, sigma, T, dt, seed=None):
    
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility
    
    N = int(T / dt)  # Num time steps
    time = np.linspace(0, T, N + 1)  # Time points from 0 to T
    S0_array = np.asarray(S0_array)  # Convert to numpy array if not already
    
    # Brownian motion increments
    W = np.random.normal(0, np.sqrt(dt), (len(S0_array), N)) # mean, std, rows, cols
    
    # Compute cumulative sum to simulate Brownian motion paths
    drift = (mu - 0.5 * sigma**2) * dt # drift component of BM
    diffusion = sigma * W # random component of BM
    
    log_returns = np.cumsum(drift + diffusion, axis=1)
    
    # Compute asset price paths
    S = S0_array[:, None] * np.exp(np.column_stack([np.zeros(len(S0_array)), log_returns]))
    
    # Create DataFrame with results
    df = pd.DataFrame(S.T, columns=[f"Asset_{i+1}" for i in range(len(S0_array))])
    df.insert(0, "Time", time)
    
    return df

def Heston():
    pass

def SABR():
    pass

def Merton_Jump():
    # Good tp simulate market crashes
    pass

def Variance_Gamma():
    # Heavy tails
    pass



# Define plotting of these functions

def plot_dataframe(df, title = "Price Evolution", x_label = "Time", y_label = "Price", show_labels = True):

    plt.figure(figsize=(10, 6))
    for col in df.columns[1:]:
        plt.plot(df["Time"], df[col], label=col)
    plt.xlabel(x_label)
    plt.ylabel("Price")
    plt.title(title)
    if show_labels:
        plt.legend()
    plt.grid(True)
    plt.show()



# Pricing Models

def black_scholes_price(S, K, mu, sigma, T, is_call = True):

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

def dynamic_delta_hedging(stock_prices,
                          deltas,):
    shares = 100
    
    
    pass
    





