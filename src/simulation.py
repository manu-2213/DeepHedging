import numpy as np
import pandas as pd


# Define functions to simulate synhtetic data

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
    # Good to simulate market crashes
    pass

def Variance_Gamma():
    # Heavy tails
    pass