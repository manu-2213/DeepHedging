"""
Verify that soft-resets skip expensive option repricing.

Run from the project root:
    python -m experiments.utils.test_reset_behavior
"""

import time
import numpy as np

from experiments.utils.sim_config import train_test_split
from hedging.envs import HedgeCallHeston


def make_tiny_heston_env(train_data):
    params, S0, K, v0 = train_data
    return HedgeCallHeston(
        S0=S0, K=K, r=0.03, v0=v0,
        theta=params["theta"], rho=params["rho"],
        kappa=params["kappa"], xi=params["sigma"],
        maturity=0.5,
        num_steps=10,   # tiny – enough to exercise the logic
        num_paths=8,
    )


def run():
    train, _ = train_test_split(dynamics="heston", train_size=4, market="sp500")

    env = make_tiny_heston_env(train)

    print("\n=== FULL RESET (seed=0, first time) ===")
    t0 = time.time()
    env.reset(seed=0)
    full_reset_time = time.time() - t0
    prices_after_full = env.option_prices.copy()
    print(f"  time: {full_reset_time:.3f}s   _soft_reset_enabled={env._soft_reset_enabled}")

    print("\n=== SOFT RESET (seed=None, same paths) ===")
    t0 = time.time()
    env.reset(seed=None)
    soft_reset_time = time.time() - t0
    prices_after_soft = env.option_prices.copy()
    print(f"  time: {soft_reset_time:.3f}s   _is_soft_reset={env._is_soft_reset}")

    # Option prices must be identical after soft reset (cached)
    prices_match = np.allclose(prices_after_full, prices_after_soft)
    print(f"\n  Option prices unchanged after soft reset: {prices_match}")

    print("\n=== FORCED FULL RESET (simulate epoch boundary) ===")
    env._soft_reset_enabled = False
    env._last_reset_seed = None
    t0 = time.time()
    env.reset(seed=None)
    forced_full_time = time.time() - t0
    print(f"  time: {forced_full_time:.3f}s   _is_soft_reset={env._is_soft_reset}")

    print("\n=== SUMMARY ===")
    print(f"  Full  reset: {full_reset_time:.3f}s")
    print(f"  Soft  reset: {soft_reset_time:.4f}s  (speedup: {full_reset_time/max(soft_reset_time,1e-9):.0f}x)")
    print(f"  Forced full: {forced_full_time:.3f}s")

    if soft_reset_time > full_reset_time * 0.5:
        print("\n  WARNING: Soft reset is not much faster than full reset.")
        print("  Check that _is_soft_reset is True and option pricing was skipped.")
    else:
        print("\n  OK: Soft reset is significantly faster than full reset.")

    assert prices_match, "BUG: option prices changed after soft reset!"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    run()
