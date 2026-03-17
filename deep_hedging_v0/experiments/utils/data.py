import numpy as np

FLOAT_DTYPE = np.float32

def convert_data(data: dict):
    """
    Convert the given yearly dict into arrays:
    S0: shape (N,)
    K:  shape (N, M)
    sigma: shape (N,)
    """
    years = sorted(data.keys())  # ensure deterministic order

    S0 = np.array([data[y]["S0"] for y in years], dtype=FLOAT_DTYPE)
    K  = np.array([data[y]["K_sampled"] for y in years], dtype=FLOAT_DTYPE)
    sigma = np.array([data[y]["std"] for y in years], dtype=FLOAT_DTYPE)

    return S0, K, sigma


def convert_data_heston(data: dict):
    
    years = sorted(data.keys())  # deterministic order

    kappa = np.array([data[y]["kappa"] for y in years], dtype=FLOAT_DTYPE)
    theta = np.array([data[y]["theta"] for y in years], dtype=FLOAT_DTYPE)
    rho   = np.array([data[y]["rho"]   for y in years], dtype=FLOAT_DTYPE)
    sigma = np.array([data[y]["xi"]    for y in years], dtype=FLOAT_DTYPE)  
    lda   = np.zeros_like(kappa)  # lda 0s for now

    params = {
        "kappa": kappa,
        "theta": theta,
        "rho": rho,
        "sigma": sigma,
        "lda": lda
    }

    S0 = np.array([data[y]["S0"] for y in years], dtype=FLOAT_DTYPE)
    v0 = np.array([data[y]["v0"] for y in years], dtype=FLOAT_DTYPE)

    # K_sampled as 2D matrix
    K = np.array([data[y]["K_sampled"] for y in years], dtype=FLOAT_DTYPE)

    return params, S0, K, v0

def compute_barriers(K, alpha=0.90):
    """
    Produce a barrier array HOW with the same shape as K.
    Standard choice: barrier is alpha * K (e.g. 90% of strike).
    """
    K = np.asarray(K, dtype=FLOAT_DTYPE)
    H = alpha * K
    return H
