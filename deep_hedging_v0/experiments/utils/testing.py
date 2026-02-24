
from hedging.plot_utils import plot_portfolio_vs_option_price
from torchrl.envs import GymWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type
import numpy as np
import wandb


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

    return terminal_pnl, metrics



    
        