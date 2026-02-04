# Deep Reinforcement Learning for Delta Hedging

This repository implements and compares different approaches to delta hedging, from classical Black-Scholes-Merton (BSM) to modern deep reinforcement learning methods.

## Project Overview

The project is structured in three main phases:
1. Data Generation and Simulation Framework
2. Classical Delta Hedging Implementation 
3. Deep Reinforcement Learning Implementation (Coming Soon)

### Current Features

#### Synthetic Data Generation
- Geometric Brownian Motion (GBM) for multiple asset paths
- Planned: Heston, SABR, Merton Jump, and Variance Gamma models
- Customizable parameters for volatility, drift, and time horizon

#### Classical Delta Hedging
- BSM option pricing and Greeks calculation
- Dynamic delta hedging implementation
- P&L calculation and performance metrics
- Visualization tools for hedge performance

### Coming Soon: Deep Hedging

The ultimate goal is to implement deep reinforcement learning for hedging options. This approach offers several advantages over classical methods:

1. **Model-Free Hedging**
   - No assumptions about underlying price dynamics
   - Learns directly from market data
   - Can adapt to changing market conditions

2. **Transaction Costs**
   - Naturally incorporates transaction costs in the hedging strategy
   - Balances hedging error vs trading costs

3. **Market Frictions**
   - Handles discrete-time hedging
   - Accounts for bid-ask spreads
   - Manages inventory constraints
