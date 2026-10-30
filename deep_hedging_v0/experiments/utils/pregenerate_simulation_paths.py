"""
Pre-generate and cache simulated asset paths so simulation scripts can skip
the expensive path-generation step on their next run (see the on-disk path
cache in hedging/simulators.py).

Path generation depends only on (S0, sigma/Heston params, r, maturity,
num_steps, num_paths, seed) — not on strikes/barriers/concentration matrices
or MLP-vs-RNN architecture. So one BSM call env + one Heston call env, each
built from the same "sp500" train/test split every simulation script uses,
is enough to prime the cache for all 24 simulation scripts at once.

Run this once, then run your simulation scripts as usual with the SAME
DEEP_HEDGING_* environment variables and --seed: they will hit the cache
instead of re-simulating.

Usage:
    PYTHONPATH=deep_hedging_v0 python -m experiments.utils.pregenerate_simulation_paths --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from experiments.utils.sim_config import EnvConfig, train_test_split
from hedging.envs import HedgeCallBS, HedgeCallHeston

# experiments/utils/testing.py:test_model always resets the test split with seed=0.
TEST_SPLIT_SEED = 0


def _prime(env, seed: int, label: str) -> None:
    env.reset(seed=seed)
    print(f"  primed: {label} (seed={seed})")


def pregenerate(seeds, market: str = "sp500", train_size: int = 4) -> None:
    env_cfg = EnvConfig()
    print(
        f"Config: num_paths={env_cfg.num_paths} num_paths_heston={env_cfg.num_paths_heston} "
        f"num_steps={env_cfg.num_steps} maturity={env_cfg.maturity} r={env_cfg.r}"
    )

    bsm_train, bsm_test = train_test_split(dynamics="bsm", train_size=train_size, market=market)
    S0_train, K_train, sigma_train = bsm_train
    S0_test, K_test, sigma_test = bsm_test

    for seed in seeds:
        env = HedgeCallBS(
            S0_train, K_train, env_cfg.maturity, env_cfg.r, sigma_train,
            env_cfg.num_paths, env_cfg.num_steps, history_len=1,
        )
        _prime(env, seed, f"bsm train split (paths={env_cfg.num_paths})")

    test_env = HedgeCallBS(
        S0_test, K_test, env_cfg.maturity, env_cfg.r, sigma_test,
        env_cfg.num_paths, env_cfg.num_steps, history_len=1,
    )
    _prime(test_env, TEST_SPLIT_SEED, f"bsm test split (paths={env_cfg.num_paths})")

    heston_train, heston_test = train_test_split(dynamics="heston", train_size=train_size, market=market)
    params_train, S0_train, K_train, v0_train = heston_train
    params_test, S0_test, K_test, v0_test = heston_test

    for seed in seeds:
        env = HedgeCallHeston(
            S0=S0_train, K=K_train, maturity=env_cfg.maturity, r=env_cfg.r, v0=v0_train,
            theta=params_train["theta"], rho=params_train["rho"], kappa=params_train["kappa"],
            xi=params_train["sigma"], num_paths=env_cfg.num_paths_heston,
            num_steps=env_cfg.num_steps, history_len=1,
        )
        _prime(env, seed, f"heston train split (paths={env_cfg.num_paths_heston})")

    test_env = HedgeCallHeston(
        S0=S0_test, K=K_test, maturity=env_cfg.maturity, r=env_cfg.r, v0=v0_test,
        theta=params_test["theta"], rho=params_test["rho"], kappa=params_test["kappa"],
        xi=params_test["sigma"], num_paths=env_cfg.num_paths_heston,
        num_steps=env_cfg.num_steps, history_len=1,
    )
    _prime(test_env, TEST_SPLIT_SEED, f"heston test split (paths={env_cfg.num_paths_heston})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[0],
        help="Training seed(s) to pre-generate paths for (test split always uses seed=0).",
    )
    parser.add_argument("--market", type=str, default="sp500")
    parser.add_argument("--train-size", type=int, default=4)
    args = parser.parse_args()

    pregenerate(args.seed, market=args.market, train_size=args.train_size)
    print("Done. Run your simulation scripts with matching DEEP_HEDGING_* variables and --seed to hit the cache.")


if __name__ == "__main__":
    main()
