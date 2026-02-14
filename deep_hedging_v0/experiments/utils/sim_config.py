"""Shared configuration utilities for simulations that log to Weights & Biases."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace, field
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np

from experiments.utils.data import convert_data, compute_barriers, convert_data_heston

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_BSM = PROJECT_ROOT / "experiments" / "option_parameter_estimation" / "bsm_data_2017_2021_sp500.json"
DATA_PATH_HESTON = PROJECT_ROOT / "experiments" / "option_parameter_estimation" / "heston_params_CMA.json"
DEFAULT_WANDB_PROJECT = "deephedging_local_test"


@dataclass
class EnvConfig:
    maturity: float = 1.0
    r: float = 0.01
    num_paths: int = 100
    num_paths_heston: int = 20 # Very compute heavy
    num_steps: int = 250
    history_len: int = 1
    history_len_rnn: int = 5
    transaction_cost: bool = True
    transaction_fee_rate: float = 1e-3
    trap: int = 1
    P: np.ndarray = field(default_factory=lambda: np.array([[1.0]], dtype=np.float32))


@dataclass
class PPOConfig:
    clip_param: float = 0.2
    value_coef: float = 0.1
    entropy_coeff: float = 0.001
    gamma: float = 0.99
    lmbda: float = 0.95
    learning_rate: float = 8e-5 # -> 4e-5
    learning_rate_ex: float = 5e-6


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    policy_epochs: int = 20
    inaction_epochs: int = 10
    num_episodes: int = 200
    sub_batch_num: int = 16
    hidden_size: int = 128


def load_bsm_data(path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads S0, K, sigma arrays used across the BSM experiments."""
    with open(path or DATA_PATH_BSM, "r", encoding="utf-8") as file:
        data = json.load(file)
    return convert_data(data)

def load_heston_data(path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads S0, K, sigma arrays used across the BSM experiments."""
    with open(path or DATA_PATH_HESTON, "r", encoding="utf-8") as file:
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
        "kappa": np.array([5.0, 2.5, 3.0], dtype=np.float64),
        "theta": np.array([0.05, 0.035, 0.045], dtype=np.float64),
        "rho": np.array([-0.8, -0.6, -0.5], dtype=np.float64),
        "sigma": np.array([0.5, 0.4, 0.55], dtype=np.float64),
        "lda": np.array([0.0, 0.0, 0.0], dtype=np.float64),
    }


def get_heston_call_config() -> Dict[str, object]:
    """Returns default spot, strike, and calibration values for call options under Heston."""
    return {
        "S0": np.array([100.0, 120.0, 80.0], dtype=np.float64),
        "K": np.array(
            [
                [90.0, 100.0, 110.0],
                [100.0, 120.0, 140.0],
                [70.0, 80.0, 90.0],
            ],
            dtype=np.float64,
        ),
        "v0": np.array([0.05, 0.04, 0.06], dtype=np.float64),
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
        "S0": np.array([50.0, 100.0, 200.0], dtype=np.float64),
        "K": np.array(
            [
                [52.5, 55.0],
                [105.0, 110.0],
                [210.0, 220.0],
            ],
            dtype=np.float64,
        ),
        "H": np.array(
            [
                [42.5, 45.0],
                [85.0, 90.0],
                [170.0, 180.0],
            ],
            dtype=np.float64,
        ),
        "v0": np.array([0.15, 0.2, 0.25], dtype=np.float64),
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
