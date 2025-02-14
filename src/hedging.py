import pandas as pd
import numpy as np

# Set pandas display option to show two decimal places
pd.options.display.float_format = '{:,.4f}'.format

def initialize_simulation(S0, K, T, r, sigma,
                          option_pricer, # for instance, bs_price
                          option_delta, # for instance, bs_delta
                          portfolio_value = 0,
                          cash_account = 0,
                          shares_held = 0,
                          seed = None):

    if seed is not None:
        np.random.seed(seed)

    initial_option_price = option_pricer(S0, K, T, r, sigma)
    initial_delta = option_delta(S0, K, T, r, sigma)
    
    # Set up initial hedge portfolio
    shares_held = initial_delta
    cash_account = initial_option_price - shares_held * S0
    portfolio_value = cash_account + shares_held * S0 # Should be equal to initial_option_price

    # Store values for plotting
    stock_prices = [S0]
    option_prices = [initial_option_price]
    portfolio_values = [portfolio_value]
    deltas = [initial_delta]
    cash_accounts = [cash_account]
    shares_held_list = [shares_held]

    current_S = S0

    return initial_option_price, stock_prices, option_prices, portfolio_values, deltas, cash_accounts, shares_held_list, current_S


def dynamic_delta_hedging(current_S, K, T, r, sigma,
                          steps, 
                          stock_prices,
                          option_prices, 
                          portfolio_values, 
                          deltas, 
                          cash_accounts, 
                          shares_held_list,
                          option_pricer,
                          option_delta):

    for i in range(1, steps + 1):
        time_to_expiry = T - i * dt

        z = np.random.normal(0, 1)
        current_S = current_S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        stock_prices.append(current_S)

        # Recalculate option price and delta

        current_option_price = option_pricer(current_S, K, time_to_expiry, r, sigma)
        option_prices.append(current_option_price)

        current_delta = option_delta(current_S, K, time_to_expiry, r, sigma) # This must be converted to a NN
        deltas.append(current_delta)

        delta_change = current_delta - shares_held_list[i - 1] # shares held is just shares_held_list[i - 1]
        shares_to_trade = delta_change

        # Trade shares and update cash account
        cash_accounts.append(cash_accounts[-1] - shares_to_trade * current_S)  # Buy shares (or sell if negative), cash decreases (or increases)
        shares_held_list.append(shares_held_list[-1] + shares_to_trade)

        # Let cash account grow with risk-free rate
        cash_accounts[i] *= np.exp(r * dt)

        # Update portfolio value
        portfolio_value = cash_accounts[i] + shares_held_list[i] * current_S
        portfolio_values.append(portfolio_value)

    return stock_prices, option_prices, portfolio_values, deltas, cash_accounts, shares_held_list, current_S
    

def PnL(current_S, portfolio_values, initial_option_price):
    # 3. Expiration Payoff
    option_payoff = np.maximum(0, current_S - K)
    final_portfolio_value = portfolio_values[-1]

    print(f"Initial Black-Scholes Option Price: {initial_option_price:.4f}")
    print(f"Final Portfolio Value (Hedge): {final_portfolio_value:.4f}")
    print(f"Option Payoff at Expiration: {option_payoff:.4f}")
    print(f"Difference between Final Portfolio Value and Option Payoff: {final_portfolio_value - option_payoff:.4f}")
