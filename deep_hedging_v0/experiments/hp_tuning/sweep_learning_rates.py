"""Run and rank a learning-rate sweep across all simulation entry points."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ACTOR_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
INACTOR_RATES = (5e-4, 1e-3, 5e-3, 1e-3, 5e-2, 1e-2)
BASELINE_MULTIPLIERS = (0.25, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
CVaR_PATTERN = re.compile(r"SWEEP_RESULT cvar_95=(\S+)")
ROOT = Path(__file__).resolve().parents[2]
SIMULATIONS_DIR = ROOT / "experiments" / "simulations"
RESULT_FIELDS = (
    "experiment", "trial", "actor_learning_rate", "inactor_learning_rate",
    "joint_learning_rate", "seed", "epochs", "policy_epochs", "inaction_epochs",
    "episodes", "cvar_95", "exit_code", "log_path", "output_tail",
)


def trial_rates(script: Path) -> list[tuple[float, float | None]]:
    actor_default = 5e-6 if script.stem.endswith("_rnn") else 5e-5
    if script.stem.startswith("inactive_"):
        return [(actor_default * multiplier, inactor_rate)
                for multiplier, inactor_rate in product(ACTOR_MULTIPLIERS, INACTOR_RATES)]
    return [(actor_default * multiplier, None) for multiplier in BASELINE_MULTIPLIERS]


def build_trial_command(script: Path, rate: float, seed: int = 0) -> list[str]:
    command = [sys.executable, str(script), "--seed", str(seed)]
    if "--actor_learning_rate" in script.read_text(encoding="utf-8"):
        command.extend(("--actor_learning_rate", str(rate)))
    return command


def trial_environment(
    actor_rate: float, inactor_rate: float | None,
    epochs: int, policy_epochs: int, inaction_epochs: int, episodes: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "WANDB_MODE": "disabled",
            "DEEP_HEDGING_SAVE_ARTIFACTS": "0",
            "DEEP_HEDGING_NUM_EPOCHS": str(epochs),
            "DEEP_HEDGING_POLICY_EPOCHS": str(policy_epochs),
            "DEEP_HEDGING_INACTION_EPOCHS": str(inaction_epochs),
            "DEEP_HEDGING_NUM_EPISODES": str(episodes),
            "DEEP_HEDGING_LEARNING_RATE": str(actor_rate),
            "DEEP_HEDGING_INACTOR_LEARNING_RATE": str(inactor_rate if inactor_rate is not None else 1e-3),
            "DEEP_HEDGING_JOINT_LEARNING_RATE": str(actor_rate),
        }
    )
    return environment


def run_trial(
    script: Path, actor_rate: float, inactor_rate: float | None,
    arguments: argparse.Namespace, trial: int, log_dir: Path,
) -> dict[str, object]:
    environment = trial_environment(
        actor_rate, inactor_rate, arguments.epochs,
        arguments.policy_epochs, arguments.inaction_epochs, arguments.episodes,
    )
    result = subprocess.run(
        build_trial_command(script, actor_rate, arguments.seed),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    matches = CVaR_PATTERN.findall(result.stdout)
    try:
        cvar = float(matches[-1]) if matches else None
    except ValueError:
        cvar = None
    if cvar is not None and not math.isfinite(cvar):
        cvar = None
    log_path = log_dir / f"{script.stem}_{trial:02d}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    return {
        "experiment": script.stem,
        "trial": trial,
        "actor_learning_rate": actor_rate,
        "inactor_learning_rate": inactor_rate,
        "joint_learning_rate": actor_rate if inactor_rate is not None else None,
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "policy_epochs": arguments.policy_epochs,
        "inaction_epochs": arguments.inaction_epochs,
        "episodes": arguments.episodes,
        "cvar_95": cvar,
        "exit_code": result.returncode,
        "log_path": str(log_path),
        "output_tail": result.stdout[-2000:],
    }


def save_best_config(
    script: Path, best: dict[str, object], arguments: argparse.Namespace,
    valid_trials: int,
) -> Path:
    destination = arguments.output.with_suffix(".best").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    actor_rate = best["actor_learning_rate"]
    inactor_rate = best["inactor_learning_rate"]
    settings = {
        "DEEP_HEDGING_LEARNING_RATE": str(actor_rate),
        "DEEP_HEDGING_JOINT_LEARNING_RATE": str(actor_rate),
        "DEEP_HEDGING_INACTOR_LEARNING_RATE": str(inactor_rate if inactor_rate is not None else 1e-3),
        "DEEP_HEDGING_NUM_EPOCHS": str(arguments.epochs),
        "DEEP_HEDGING_POLICY_EPOCHS": str(arguments.policy_epochs),
        "DEEP_HEDGING_INACTION_EPOCHS": str(arguments.inaction_epochs),
        "DEEP_HEDGING_NUM_EPISODES": str(arguments.episodes),
        "DEEP_HEDGING_NUM_PATHS": os.getenv("DEEP_HEDGING_NUM_PATHS", "50"),
        "DEEP_HEDGING_NUM_PATHS_HESTON": os.getenv("DEEP_HEDGING_NUM_PATHS_HESTON", "10"),
        "DEEP_HEDGING_SUB_BATCH_NUM": os.getenv("DEEP_HEDGING_SUB_BATCH_NUM", "4"),
    }
    config = {
        "experiment": script.stem,
        "actor_learning_rate": actor_rate,
        "inactor_learning_rate": inactor_rate,
        "joint_learning_rate": actor_rate if inactor_rate is not None else None,
        "seed": arguments.seed,
        "cvar_95": best["cvar_95"],
        "selected_trial": best["trial"],
        "valid_trials": valid_trials,
        "attempted_trials": 12,
        "source_csv": str(arguments.output.resolve()),
        "environment": settings,
    }
    config_path = destination / f"{script.stem}.json"
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    command = ["python", "-u", str(script.relative_to(ROOT.parent))]
    command.extend(build_trial_command(script, actor_rate, arguments.seed)[4:])
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in settings.items())
    launcher = f'''#!/bin/bash
#SBATCH --job-name=best_{script.stem}
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=240:00:00

set -eo pipefail
cd "${{DEEP_HEDGING_REPO:-{ROOT.parent}}}"
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate
set -u

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-12}}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export DEEP_HEDGING_PATH_CACHE=1
export DEEP_HEDGING_SAVE_ARTIFACTS=1
export WANDB_MODE="${{WANDB_MODE:-online}}"
{exports}

exec {shlex.join(command)} --seed "${{RUN_SEED:-{arguments.seed}}}"
'''
    launcher_path = destination / f"{script.stem}.slurm"
    launcher_path.write_text(launcher, encoding="utf-8")
    print(f"SAVED {script.stem}: {config_path}\nRUN: sbatch {shlex.quote(str(launcher_path))}", flush=True)
    return config_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=positive_int, default=60, help="Active baseline epochs.")
    parser.add_argument("--policy-epochs", type=positive_int, default=40)
    parser.add_argument("--inaction-epochs", type=positive_int, default=20)
    parser.add_argument("--episodes", type=positive_int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("learning_rate_sweep.csv"))
    parser.add_argument("--scripts", nargs="*", help="Optional simulation filenames to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without training or writing files.")
    arguments = parser.parse_args()
    available = {script.name: script for script in sorted(SIMULATIONS_DIR.glob("*.py"))}
    if arguments.scripts and any(name not in available for name in arguments.scripts):
        parser.error("--scripts must name existing simulation filenames, without directory components")
    scripts = (
        [available[name] for name in dict.fromkeys(arguments.scripts)]
        if arguments.scripts
        else list(available.values())
    )
    if not scripts:
        parser.error("No simulation scripts found")
    print(f"{len(scripts)} experiments x 12 configurations = {12 * len(scripts)} trials; seed={arguments.seed}")
    if arguments.epochs != arguments.policy_epochs + arguments.inaction_epochs:
        print("WARNING: active and inactive total epoch budgets differ.")
    if arguments.dry_run:
        for script in scripts:
            for trial, (actor_rate, inactor_rate) in enumerate(trial_rates(script), start=1):
                print(f"{script.stem} trial={trial:02d} actor_lr={actor_rate:g} inactor_lr={inactor_rate}")
        return
    if arguments.output.exists():
        parser.error(f"Output already exists: {arguments.output}; choose a new --output")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = arguments.output.with_suffix(".logs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best_trials = {}
    with arguments.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        output.flush()
        for script in scripts:
            for trial, (actor_rate, inactor_rate) in enumerate(trial_rates(script), start=1):
                print(f"Running {script.name} [{trial}/12] actor_lr={actor_rate:g} inactor_lr={inactor_rate}", flush=True)
                result = run_trial(script, actor_rate, inactor_rate, arguments, trial, log_dir)
                results.append(result)
                writer.writerow(result)
                output.flush()
            trials = [trial for trial in results if trial["experiment"] == script.stem and trial["exit_code"] == 0 and trial["cvar_95"] is not None]
            if trials:
                best = min(trials, key=lambda trial: trial["cvar_95"])
                best_trials[script.stem] = best
                save_best_config(script, best, arguments, len(trials))
                print(f"BEST {script.stem}: actor_lr={best['actor_learning_rate']:g}, "
                      f"inactor_lr={best['inactor_learning_rate']}, cvar_95={best['cvar_95']:.4f} "
                      f"({len(trials)}/12 valid trials)")
            else:
                print(f"FAILED {script.stem}: no valid CVaR result")
    for name, inactive in best_trials.items():
        if not name.startswith("inactive_"):
            continue
        active = best_trials.get(name.removeprefix("inactive_"))
        if active is not None:
            improvement = active["cvar_95"] - inactive["cvar_95"]
            print(f"COMPARE {name}: active-minus-inactive CVaR={improvement:.4f}; "
                  f"inactive {'wins' if improvement > 0 else 'ties' if improvement == 0 else 'loses'}")


if __name__ == "__main__":
    main()