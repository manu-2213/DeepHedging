
from hedging.plot_utils import plot_portfolio_vs_option_price
from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
import numpy as np
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm as sp_norm


def _var(x: np.ndarray, level: float) -> float:
    """Value at Risk at confidence level (e.g. 0.95). Loss convention: positive = loss."""
    return float(np.percentile(-x, level * 100))


def _cvar(x: np.ndarray, level: float) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level. Coherent risk measure."""
    var = _var(x, level)
    losses = -x
    return float(losses[losses >= var].mean())


def _sharpe(x: np.ndarray) -> float:
    """Sharpe ratio (mean / std). Returns NaN if std == 0."""
    std = x.std()
    return float(x.mean() / std) if std > 0 else float("nan")


def _sortino(x: np.ndarray) -> float:
    """Sortino ratio (mean / downside std). Returns NaN if no losses."""
    downside = x[x < 0]
    if len(downside) == 0:
        return float("nan")
    downside_std = np.sqrt(np.mean(downside ** 2))
    return float(x.mean() / downside_std) if downside_std > 0 else float("nan")


def compute_risk_metrics(pnl: np.ndarray) -> dict:
    """
    Compute risk and performance metrics on a cross-sectional distribution
    of terminal hedging P&L (one value per Monte Carlo path).

    Parameters
    ----------
    pnl : np.ndarray, shape (N,)
        Terminal hedging P&L per path: portfolio_value - option_payoff at maturity.

    Returns
    -------
    dict with keys: mean, std, sharpe, sortino,
                    var_95, var_99, cvar_95, cvar_99,
                    skewness, excess_kurtosis, min, max
    """
    r = pnl.flatten().astype(float)

    from scipy.stats import skew, kurtosis  # lazy import

    metrics = {
        "mean":             float(r.mean()),
        "std":              float(r.std()),
        "sharpe":           _sharpe(r),
        "sortino":          _sortino(r),
        "var_95":           _var(r, 0.95),
        "var_99":           _var(r, 0.99),
        "cvar_95":          _cvar(r, 0.95),
        "cvar_99":          _cvar(r, 0.99),
        "skewness":         float(skew(r)),
        "excess_kurtosis":  float(kurtosis(r, fisher=True)),  # Fisher: 0 = Gaussian
        "min":              float(r.min()),
        "max":              float(r.max()),
    }
    return metrics


def _log_pnl_distribution(terminal_pnl: np.ndarray) -> None:
    """Graph 2: Histogram of terminal P&L distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal_pnl.tolist(), nbinsx=60, name="Terminal P&L",
        marker_color="steelblue", opacity=0.75,
    ))
    fig.update_layout(
        title="Terminal Hedging P&L Distribution",
        xaxis_title="Terminal P&L",
        yaxis_title="Count",
        bargap=0.05,
    )
    wandb.log({"final/pnl_distribution": wandb.Plotly(fig)})


def _log_delta_paths(base_env, n_paths: int = 5) -> None:
    """Graph 3: Learned delta paths vs analytical BS delta (BSM envs only)."""
    if not hasattr(base_env, 'shares_held'):
        return

    shares = base_env.shares_held          # (P, A, K, T+1)
    num_show   = min(n_paths, base_env.num_paths)
    num_assets = base_env.num_assets
    num_strikes = base_env.num_strikes
    num_steps  = base_env.num_steps
    T_mat = base_env.maturity
    dt    = T_mat / num_steps
    times = (np.arange(num_steps + 1) * dt).tolist()

    has_bsm = hasattr(base_env, 'simulator') and hasattr(base_env.simulator, 'sigma')
    ncols = max(1, num_assets * num_strikes)

    fig = make_subplots(
        rows=num_show, cols=ncols,
        subplot_titles=[f"Path {p+1}" for p in range(num_show * ncols)],
        shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.05,
    )

    for p in range(num_show):
        for i in range(num_assets):
            for j in range(num_strikes):
                col_idx = i * num_strikes + j + 1
                show_legend = (p == 0 and i == 0 and j == 0)
                learned = shares[p, i, j, :].tolist()

                fig.add_trace(
                    go.Scatter(x=times, y=learned, mode='lines',
                               name='Learned δ', line=dict(color='steelblue'),
                               showlegend=show_legend),
                    row=p + 1, col=col_idx,
                )

                if has_bsm:
                    sigma_i = float(base_env.simulator.sigma[i])
                    K_ij    = float(base_env.K[i, j])
                    r       = base_env.r
                    bs_deltas = []
                    for t_idx in range(num_steps + 1):
                        S   = float(base_env.stock_prices[p, i, t_idx])
                        tau = T_mat - t_idx * dt
                        if tau <= 1e-6:
                            bs_deltas.append(1.0 if S > K_ij else 0.0)
                        else:
                            d1 = (np.log(S / K_ij) + (r + 0.5 * sigma_i**2) * tau) / (sigma_i * np.sqrt(tau))
                            bs_deltas.append(float(sp_norm.cdf(d1)))
                    fig.add_trace(
                        go.Scatter(x=times, y=bs_deltas, mode='lines',
                                   name='BS δ', line=dict(color='crimson', dash='dash'),
                                   showlegend=show_legend),
                        row=p + 1, col=col_idx,
                    )

    fig.update_layout(
        title="Learned Delta Paths vs BS Analytical Delta",
        height=max(300, 280 * num_show),
        xaxis_title="Time",
    )
    wandb.log({"final/delta_paths": wandb.Plotly(fig)})


def _log_transaction_cost_scatter(base_env, terminal_pnl: np.ndarray) -> None:
    """Graph 5: Per-path total transaction cost vs terminal P&L scatter."""
    if not hasattr(base_env, 'shares_held'):
        return

    shares   = base_env.shares_held                              # (P, A, K, T+1)
    S        = base_env.stock_prices                             # (P, A, T+1)
    fee_rate = getattr(base_env, 'transaction_fee_rate', 0.0)

    delta_shares = np.diff(shares, axis=-1)                      # (P, A, K, T)
    S_exp = S[:, :, np.newaxis, :delta_shares.shape[-1]]         # (P, A, 1, T)
    per_path_cost = (np.abs(delta_shares * S_exp) * fee_rate).sum(axis=(1, 2, 3))  # (P,)

    per_path_pnl = terminal_pnl.reshape(
        base_env.num_paths, base_env.num_assets, base_env.num_strikes
    ).mean(axis=(1, 2))                                          # (P,)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=per_path_cost.tolist(), y=per_path_pnl.tolist(),
        mode='markers',
        marker=dict(size=5, opacity=0.6, color='steelblue'),
        name='Paths',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(
        title="Transaction Costs vs Terminal P&L (per path)",
        xaxis_title="Total Transaction Cost",
        yaxis_title="Terminal P&L",
    )
    wandb.log({"final/cost_vs_pnl": wandb.Plotly(fig)})


def _log_action_magnitude_distribution(rollout) -> None:
    """Graph 6: Histogram of rebalancing trade sizes |δ_t − δ_{t-1}|."""
    actions = rollout["action"].detach().cpu().numpy()   # (num_envs, num_steps, A)

    if actions.ndim == 3:
        deltas = np.abs(np.diff(actions, axis=1)).flatten()
    elif actions.ndim == 2:
        deltas = np.abs(np.diff(actions, axis=0)).flatten()
    else:
        deltas = np.abs(np.diff(actions.flatten()))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=deltas.tolist(), nbinsx=60, name="|Δδ|",
        marker_color="steelblue", opacity=0.75,
    ))
    fig.update_layout(
        title="Action Magnitude Distribution |δ_t − δ_{t-1}|",
        xaxis_title="|δ_t − δ_{t-1}|",
        yaxis_title="Count",
        bargap=0.05,
    )
    wandb.log({"final/action_magnitude_distribution": wandb.Plotly(fig)})


def test_model(base_env, model, num_steps, device, plotting=False):
    env = GymWrapper(base_env, device=device)
    env.reset(seed=0)

    # Handle both models with get_policy_operator() and direct policies (like JointPolicy)
    if hasattr(model, 'get_policy_operator'):
        policy = model.get_policy_operator()
    else:
        policy = model

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(max_steps=num_steps, policy=policy)

    # --- RL reward statistics (training signal, logged for monitoring) ---
    rewards = rollout['next', 'reward'].detach().cpu().numpy()
    mean_reward = float(rewards.mean())
    std_reward = float(rewards.std())
    print(f"  RL reward  |  mean: {mean_reward:.4f}  std: {std_reward:.4f}")
    wandb.log({
        "final/mean_reward": mean_reward,
        "final/std_reward":  std_reward,
    })

    # --- Terminal hedging P&L across Monte Carlo paths ---
    # portfolio_value[..., -1] - option_prices[..., -1] gives the hedging error
    # at maturity for each simulated path. This is the financially meaningful
    # distribution on which risk metrics should be computed.
    terminal_portfolio = base_env.portfolio_value[..., -1]   # (num_paths, num_assets, num_strikes)
    terminal_option    = base_env.option_prices[..., -1]     # (num_paths, num_assets, num_strikes)
    terminal_pnl = (terminal_portfolio - terminal_option).flatten()  # (N,)

    metrics = compute_risk_metrics(terminal_pnl)

    print(
        f"\n{'─'*55}\n"
        f"  Terminal Hedging P&L  (N={len(terminal_pnl)} paths)\n"
        f"{'─'*55}\n"
        f"  Summary\n"
        f"    Mean P&L         : {metrics['mean']:>10.4f}\n"
        f"    Std  P&L         : {metrics['std']:>10.4f}\n"
        f"    Min / Max        : {metrics['min']:>10.4f} / {metrics['max']:>10.4f}\n"
        f"\n"
        f"  Risk measures  (loss = positive number)\n"
        f"    VaR  95%         : {metrics['var_95']:>10.4f}\n"
        f"    VaR  99%         : {metrics['var_99']:>10.4f}\n"
        f"    CVaR 95% (ES)    : {metrics['cvar_95']:>10.4f}\n"
        f"    CVaR 99% (ES)    : {metrics['cvar_99']:>10.4f}\n"
        f"\n"
        f"  Distribution\n"
        f"    Sharpe           : {metrics['sharpe']:>10.4f}\n"
        f"    Sortino          : {metrics['sortino']:>10.4f}\n"
        f"    Skewness         : {metrics['skewness']:>10.4f}\n"
        f"    Excess Kurtosis  : {metrics['excess_kurtosis']:>10.4f}\n"
        f"{'─'*55}"
    )

    # Log all metrics to wandb
    wandb.log({f"final/{k}": v for k, v in metrics.items()})

    if plotting:
        plot_portfolio_vs_option_price(env._env)
        _log_pnl_distribution(terminal_pnl)
        _log_delta_paths(env._env)
        _log_transaction_cost_scatter(env._env, terminal_pnl)
        _log_action_magnitude_distribution(rollout)

    return terminal_pnl, metrics



    
        