import pandas as pd
import numpy as np

def initialize_simulation(S0, K, T, r, sigma, option_pricer, option_delta, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    # Calculate initial option price and delta for each path
    initial_option_price = option_pricer(S0, K, T, r, sigma)
    initial_delta = option_delta(S0, K, T, r, sigma)
    
    # Set up initial hedge portfolio
    shares_held = initial_delta
    cash_account = initial_option_price - shares_held * S0
    portfolio_value = cash_account + shares_held * S0  # should equal initial_option_price

    # current_S will be updated during the simulation
    current_S = S0.copy()
    
    return initial_option_price, cash_account, shares_held, portfolio_value, current_S


def dynamic_delta_hedging(S0, K, T, r, sigma, steps, dt, option_pricer, option_delta):
    
    n = S0.shape[0]
    
    # Preallocate arrays for all simulation values:
    stock_prices   = np.empty((steps + 1, n))
    option_prices  = np.empty((steps + 1, n))
    portfolio_values = np.empty((steps + 1, n))
    deltas         = np.empty((steps + 1, n))
    cash_accounts  = np.empty((steps + 1, n))
    shares_held_list = np.empty((steps + 1, n))
    
    # Initialize at t = 0
    stock_prices[0, :] = S0
    initial_option_price = option_pricer(S0, K, T, r, sigma)
    option_prices[0, :] = initial_option_price
    initial_delta = option_delta(S0, K, T, r, sigma)
    deltas[0, :] = initial_delta
    shares_held_list[0, :] = initial_delta
    cash_accounts[0, :] = initial_option_price - initial_delta * S0
    portfolio_values[0, :] = cash_accounts[0, :] + shares_held_list[0, :] * S0
    
    # current_S holds the evolving stock price per simulation path
    current_S = S0.copy()
    
    for i in range(1, steps + 1):
        # Decrease time to expiry
        time_to_expiry = T - i * dt
        
        # Simulate stock price evolution using geometric Brownian motion
        z = np.random.normal(0, 1, size=n)
        current_S = current_S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        stock_prices[i, :] = current_S
        
        # Recalculate option price and delta for the new state
        current_option_price = option_pricer(current_S, K, time_to_expiry, r, sigma)
        option_prices[i, :] = current_option_price
        current_delta = option_delta(current_S, K, time_to_expiry, r, sigma)
        deltas[i, :] = current_delta
        
        # Compute the change in delta, which is the number of shares to trade
        delta_change = current_delta - shares_held_list[i - 1, :]
        shares_to_trade = delta_change
        
        # Update the cash account: trading cost and risk-free growth
        cash_accounts[i, :] = cash_accounts[i - 1, :] - shares_to_trade * current_S
        # Let the cash account grow with the risk-free rate over dt
        cash_accounts[i, :] *= np.exp(r * dt)
        
        # Update the shares held and portfolio value
        shares_held_list[i, :] = shares_held_list[i - 1, :] + shares_to_trade
        portfolio_values[i, :] = cash_accounts[i, :] + shares_held_list[i, :] * current_S
    
    return stock_prices, option_prices, portfolio_values, deltas, cash_accounts, shares_held_list

def PnL(current_S, portfolio_values, initial_option_price, K):
    
    # Option payoff at expiration (for a call option)
    option_payoff = np.maximum(0, current_S - K)
    final_portfolio_value = portfolio_values[-1, :]
    diff = final_portfolio_value - option_payoff
    
    # Print summary statistics (or you can return these arrays for further analysis)
    # print("Initial Option Prices:", initial_option_price)
    # print("Final Portfolio Values:", final_portfolio_value)
    # print("Option Payoffs at Expiration:", option_payoff)
    # print("Differences (PnL):", diff)
    
    return diff