import pandas as pd
import numpy as np
import numba as nb

# Set pandas display option to show two decimal places
pd.options.display.float_format = '{:,.4f}'.format

def adjustments(delta_diff):
    """
    Just to showcase that adjustments with a negative number are associated with selling the udnerlying.
    Simmilarly, adjustments with a positive number are associated with buying the underlying.
    """
    return - delta_diff

def dynamic_delta_hedging(df, initial_options = 100,initial_cash=0):
    """
    df: Contains data on the time to expiration, Asset Price, Price of the option, and delta.
    We assume that we go long on the option and short on the underlying (to mantain our portfolio delta neutral)
    Assume we allow the buying and selling of fractions of the underlying.
    """
    
    # Compute hedge adjustments (change in delta
    df["Total_Delta_Position"] = initial_options * df["Delta"].diff().fillna(0)
    df["Adjustments_(Contracts)"] = df["Delta"].diff().fillna(0).apply(adjustments) # X > 0 -> Buy
    df["Total_Adjustments"] = df["Adjustments_(Contracts)"].cumsum()
    df["Ajustment_Cash_Flow"] = - df["Adjustments_(Contracts)"] * df["Asset_Price"]

    # Need to include interest on adjustments, for now we assume there are no borrowing or lending costs.
    # Need to include transaction costs, for now we assume there is none.
        

    return df

def PnL(df, K, initial_options = 100):
    adjustment_cash_flows = df["Ajustment_Cash_Flow"].sum()
    option_profit = initial_options * (max(df["Asset_Price"].iloc[-1] - K, 0) - df["Call_Price"][0])
    # As part of the original hedge, we were required to sell short df["Delta"] * initial_options / 100 contracts
    # At expiration, we are requires to buy them back for df["Asset_Price"].iloc[-1]
    underlying_position = df["Delta"][0] / 100 * initial_options * (df["Asset_Price"][0] - df["Asset_Price"].iloc[-1])
    print(f"Adjustment Cash Flows: {adjustment_cash_flows}")
    print(f"Option Profit: {option_profit}")
    print(f"Initial Underlying Hedge Position: {underlying_position}")
    return adjustment_cash_flows + option_profit + underlying_position