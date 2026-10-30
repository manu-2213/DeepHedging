"""Configuration for the fast single-vs-double-net hedging pipeline.

There is exactly one market-friction regime in this pipeline:
    * 1% of traded notional when the hedge instrument is the underlying (call product)
    * 5% of traded premium when the hedge instruments are options (doc / con products)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

UNDERLYING_FEE = 0.01
OPTION_FEE = 0.05

DYNAMICS = ("bsm", "heston")
PRODUCTS = ("call", "doc", "con")
ARCHS = ("rnn", "mlp")      # the grid only runs rnn; mlp is kept for reference


def experiment_name(dynamics: str, product: str, arch: str) -> str:
    return f"{dynamics}_{product}_{arch}"


def parse_experiment(name: str) -> tuple[str, str, str]:
    dynamics, product, arch = name.split("_")
    if dynamics not in DYNAMICS or product not in PRODUCTS or arch not in ARCHS:
        raise ValueError(f"Unknown experiment {name!r}; expected <{'|'.join(DYNAMICS)}>_<{'|'.join(PRODUCTS)}>_<{'|'.join(ARCHS)}>")
    return dynamics, product, arch


@dataclass
class MarketConfig:
    market: str = "sp500"
    train_size: int = 4            # first 4 calibration years train, the rest (2021) is the test year
    maturity: float = 1.0
    r: float = 0.01
    num_steps: int = 250
    barrier_alpha: float = 0.9     # DOC barrier H = alpha * K (same as compute_barriers)
    digital_payout: float = 1.0    # CON payout (same as default_concentration_matrix)
    underlying_fee: float = UNDERLYING_FEE
    option_fee: float = OPTION_FEE


@dataclass
class BankConfig:
    """Number of simulated paths per asset (calibration year) in each split."""
    train_paths: int
    val_paths: int
    test_paths: int
    seed_train: int = 11
    seed_val: int = 22
    seed_test: int = 33
    heston_mc_paths: int = 2048        # nested MC paths for the Heston DOC price
    heston_phi_points: int = 1000      # same integration grid as HestonSimulator
    heston_phi_max: float = 50.0
    heston_cf_chunk: int = 8192        # pricing points per GPU chunk


def default_bank_config(dynamics: str, product: str, smoke: bool = False) -> BankConfig:
    if smoke:
        return BankConfig(train_paths=40, val_paths=20, test_paths=40, heston_mc_paths=128)
    if dynamics == "bsm":
        return BankConfig(train_paths=4000, val_paths=1000, test_paths=4000)
    if product == "doc":
        return BankConfig(train_paths=1000, val_paths=500, test_paths=2000)
    return BankConfig(train_paths=2000, val_paths=500, test_paths=2000)


@dataclass
class PPOConfig:
    hidden: int = 64
    rnn_history: int = 8            # sliding window length fed to the LSTM (old history_len_rnn was 10)
    clip: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-3
    gamma: float = 0.99
    lmbda: float = 0.95
    max_grad_norm: float = 1.0
    batch_paths: int = 100          # paths per asset per rollout -> 100 * 4 years * 5 strikes = 2000 envs (as before)
    ppo_epochs: int = 4             # old pipeline: 1 pass
    num_minibatches: int = 8        # old pipeline: 4
    normalize_advantage: bool = True
    actor_update_every: int = 3     # double net: actor updated every 3rd iteration (as before)
    gate_init_logit: float = -2.0   # double net: gate starts holding ~12% of the time
    trade_features: bool = True     # holdings, last trade, trade flags, time since trade, costs, error (env.py)


@dataclass
class StopConfig:
    eval_every: int = 20            # iterations between validation evaluations
    patience: int = 15              # evaluations without improvement before stopping
    min_evals: int = 10             # never stop before this many evaluations
    max_iters: int = 4000           # hard cap per training run
    min_rel_improvement: float = 1e-3


def smoke_overrides(ppo: PPOConfig, stop: StopConfig) -> None:
    ppo.batch_paths = 10
    stop.eval_every = 5
    stop.patience = 2
    stop.min_evals = 2
    stop.max_iters = 20


def config_hash(*configs) -> str:
    payload = json.dumps([asdict(c) for c in configs], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def to_dict(**configs) -> dict:
    return {name: asdict(cfg) for name, cfg in configs.items()}
