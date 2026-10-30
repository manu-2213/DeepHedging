"""Shared configuration utilities for simulations that log to Weights & Biases."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace, field
from os import getenv
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np

from experiments.utils.data import convert_data, compute_barriers, convert_data_heston

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_BSM = PROJECT_ROOT / "experiments" / "option_parameter_estimation"
DATA_PATH_HESTON = PROJECT_ROOT / "experiments" / "option_parameter_estimation"
DEFAULT_WANDB_PROJECT = getenv("WANDB_PROJECT", "deephedging_local_test")


def _positive_int_env(name: str, default: int) -> int:
    value = int(getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_float_env(name: str, default: float) -> float:
    value = float(getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _float_env_override(name: str, default: float) -> float:
    return _positive_float_env(name, default) if getenv(name) is not None else default


@dataclass
class EnvConfig:
    maturity: float = 1.0
    r: float = 0.01
    num_paths: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_NUM_PATHS", 100))
    num_paths_heston: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_NUM_PATHS_HESTON", 20))
    num_steps: int = 250  # Full year (daily hedging)
    history_len: int = 1
    history_len_rnn: int = 10
    transaction_cost: bool = True
    transaction_fee_rate: float = field(default_factory=lambda: _positive_float_env("DEEP_HEDGING_TRANSACTION_FEE_RATE", 1e-3))
    trap: int = 1
    P: np.ndarray = field(default_factory=lambda: np.array([[1.0]], dtype=np.float32))


@dataclass
class PPOConfig:
    clip_param: float = 0.1
    value_coef: float = 0.1
    entropy_coeff: float = 0.001
    gamma: float = 0.99
    lmbda: float = 0.95
    learning_rate: float = field(default_factory=lambda: _positive_float_env("DEEP_HEDGING_LEARNING_RATE", 2e-5))
    learning_rate_inactive: float = field(default_factory=lambda: _positive_float_env("DEEP_HEDGING_INACTOR_LEARNING_RATE", 1e-3))
    learning_rate_ex: float = field(default_factory=lambda: _positive_float_env("DEEP_HEDGING_JOINT_LEARNING_RATE", 5e-6))

    def __post_init__(self) -> None:
        self.learning_rate = _float_env_override("DEEP_HEDGING_LEARNING_RATE", self.learning_rate)
        self.learning_rate_inactive = _float_env_override("DEEP_HEDGING_INACTOR_LEARNING_RATE", self.learning_rate_inactive)
        self.learning_rate_ex = _float_env_override("DEEP_HEDGING_JOINT_LEARNING_RATE", self.learning_rate_ex)


@dataclass
class TrainingConfig:
    num_epochs: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_NUM_EPOCHS", 60))
    policy_epochs: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_POLICY_EPOCHS", 40))
    inaction_epochs: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_INACTION_EPOCHS", 20))
    num_episodes: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_NUM_EPISODES", 400))
    sub_batch_num: int = field(default_factory=lambda: _positive_int_env("DEEP_HEDGING_SUB_BATCH_NUM", 4))
    hidden_size: int = 64

def train_test_split(dynamics: str, train_size: int, market: str):
    if dynamics == "heston":
        params, S0, K, v0 = load_heston_data(market=market)
        params_train = {}
        params_test = {}
        for key, value in params.items():
            params_train[key] = value[:train_size]
            params_test[key] = value[train_size:]
    elif dynamics == "bsm":
        S0, K, v0 = load_bsm_data(market=market)
    else:
        raise ValueError("Dynamics must be either 'heston' or 'bsm' as of v0")
    
    S0_train = S0[:train_size]
    S0_test = S0[train_size:]

    K_train = K[:train_size]
    K_test = K[train_size:]

    v0_train = v0[:train_size]
    v0_test = v0[train_size:]

    if dynamics == "heston":
        train = (params_train, S0_train, K_train, v0_train)
        test = (params_test, S0_test, K_test, v0_test)
        
    elif dynamics == "bsm":
        train = (S0_train, K_train, v0_train)
        test = (S0_test, K_test, v0_test)

    return train, test


def load_bsm_data(market: str, path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads S0, K, sigma arrays used across the BSM experiments."""
    DATA_PATH = DATA_PATH_BSM / f"bsm_data_{market}_2017_2021.json"
    with open(path or DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return convert_data(data)

def load_heston_data(market: str, path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads S0, K, sigma arrays used across the BSM experiments."""
    DATA_PATH = DATA_PATH_HESTON / f"heston_params_QL_v2_{market}.json"
    with open(path or DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return convert_data_heston(data)

def default_barriers(strikes: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Computes default DOC barriers as a fraction of the strikes."""
    return compute_barriers(strikes, alpha=alpha)


def default_concentration_matrix(size: int = 1) -> np.ndarray:
    """Returns a simple concentration matrix used by the concentrated hedge setup."""
    return np.ones((size, size), dtype=np.float32)


def training_to_wandb_config(training: TrainingConfig, ppo: PPOConfig, extra: Dict[str, float] | None = None) -> Dict[str, float]:
    """Utility helper to merge training + PPO configs for wandb logging."""
    config = {**asdict(training), **asdict(ppo)}
    if extra:
        config.update(extra)
    return config


def _default_heston_params() -> Dict[str, np.ndarray]:
    """Internal helper returning canonical Heston stochastic-vol parameters."""
    return {
        "kappa": np.array([5.0, 2.5, 3.0], dtype=np.float32),
        "theta": np.array([0.05, 0.035, 0.045], dtype=np.float32),
        "rho": np.array([-0.8, -0.6, -0.5], dtype=np.float32),
        "sigma": np.array([0.5, 0.4, 0.55], dtype=np.float32),
        "lda": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }


def get_heston_call_config() -> Dict[str, object]:
    """Returns default spot, strike, and calibration values for call options under Heston."""
    return {
        "S0": np.array([100.0, 120.0, 80.0], dtype=np.float32),
        "K": np.array(
            [
                [90.0, 100.0, 110.0],
                [100.0, 120.0, 140.0],
                [70.0, 80.0, 90.0],
            ],
            dtype=np.float32,
        ),
        "v0": np.array([0.05, 0.04, 0.06], dtype=np.float32),
        "r": 0.03,
        "maturity": 1.0,
        "num_paths": 100,
        "num_steps": 250,
        "history_len": 1,
        "input_dim": 11,
        "hidden_size": 64,
        "action_dim": 1,
        "transaction_cost": True,
        "transaction_fee_rate": 1e-3,
        "params": _default_heston_params(),
    }


def get_heston_doc_config() -> Dict[str, object]:
    """Returns default parameters for down-and-out call (DOC) hedging under Heston."""
    return {
        "S0": np.array([50.0, 100.0, 200.0], dtype=np.float32),
        "K": np.array(
            [
                [52.5, 55.0],
                [105.0, 110.0],
                [210.0, 220.0],
            ],
            dtype=np.float32,
        ),
        "H": np.array(
            [
                [42.5, 45.0],
                [85.0, 90.0],
                [170.0, 180.0],
            ],
            dtype=np.float32,
        ),
        "v0": np.array([0.15, 0.2, 0.25], dtype=np.float32),
        "r": 0.05,
        "maturity": 1.0,
        "num_paths": 100,
        "num_steps": 250,
        "history_len": 1,
        "input_dim": 17,
        "hidden_size": 64,
        "action_dim": 2,
        "transaction_cost": True,
        "transaction_fee_rate": 1e-3,
        "params": _default_heston_params(),
    }


def get_heston_conc_config() -> Dict[str, object]:
    """Returns default parameters for concentrated book hedging under Heston."""
    config = get_heston_call_config()
    config.update(
        {
            "input_dim": 17,
            "action_dim": 2,
            "P": default_concentration_matrix(),
        }
    )
    return config