"""Helpers shared by the calibration scripts: the Black-Scholes inputs of one year and the strike sampler."""
from pathlib import Path

import numpy as np
import pandas as pd


def get_BSM_data(year: str, market: str):
    path = Path(__file__).parent / "option_data" / f"{market}_{year}.json.bz2"
    df =  pd.read_json(path, compression='bz2', orient='index')

    S0 = df["forward_price"].iloc[0]

    idx = df.groupby("date")["tau"].idxmin()
    daily_spot = df.loc[idx, ["date", "forward_price"]].set_index("date").sort_index()
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
    if n_strikes < 1 or n_strikes % 2 == 0:
        raise ValueError("n_strikes must be a positive odd integer")
    if sparcity < 1:
        raise ValueError("sparcity must be at least 1")
    if len(K) == 0:
        return []

    strikes = sorted(K)
    atm_index = min(
        range(len(strikes)),
        key=lambda index: (abs(S0 - strikes[index]), strikes[index]),
    )
    half_window = n_strikes // 2
    sparse_start = atm_index - half_window * sparcity
    sparse_end = atm_index + half_window * sparcity

    if sparse_start >= 0 and sparse_end < len(strikes):
        return [strikes[index] for index in range(sparse_start, sparse_end + 1, sparcity)]

    count = min(n_strikes, len(strikes))
    start = min(max(atm_index - count // 2, 0), len(strikes) - count)
    return strikes[start:start + count]
