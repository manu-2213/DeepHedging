"""The protocol shared by every experiment of the paper: split seeds, bank sizes and the two cost regimes."""

SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}       # identical for every experiment
# Default number of simulated paths per split. The paper's runs pass 2,000 / 500 / 2,000 for both dynamics.
DEFAULT_PATHS = {"bsm": {"train": 2000, "val": 500, "test": 2000},
                 "heston": {"train": 1000, "val": 250, "test": 1000}}
BASELINE_PATHS = {"val": 500, "test": 2000}                  # banks of the model-based reference strategies
FEES = {"01": (0.001, 0.002), "1": (0.01, 0.05)}             # tag -> (underlying fee, option fee): low and high cost
EXIT_RESUME = 75       # exit code of a run that used its time budget; the Slurm script resubmits itself on it
