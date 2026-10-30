"""Tune one experiment, then train its winning learning rates on seeds 1-10."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

if __package__:
    from .sweep_learning_rates import (
        ROOT, SIMULATIONS_DIR, CVaR_PATTERN, build_trial_command,
        positive_int, trial_rates,
    )
else:
    from sweep_learning_rates import (
        ROOT, SIMULATIONS_DIR, CVaR_PATTERN, build_trial_command,
        positive_int, trial_rates,
    )


def run_training(script, actor_rate, inactor_rate, seed, epochs, arguments, phase, index):
    environment = os.environ.copy()
    environment.update({
        "DEEP_HEDGING_NUM_EPOCHS": str(epochs),
        "DEEP_HEDGING_POLICY_EPOCHS": str(arguments.policy_epochs),
        "DEEP_HEDGING_INACTION_EPOCHS": str(arguments.inaction_epochs),
        "DEEP_HEDGING_NUM_EPISODES": str(arguments.episodes),
        "DEEP_HEDGING_LEARNING_RATE": str(actor_rate),
        "DEEP_HEDGING_JOINT_LEARNING_RATE": str(actor_rate),
        "DEEP_HEDGING_INACTOR_LEARNING_RATE": str(inactor_rate if inactor_rate is not None else 1e-3),
        "WANDB_MODE": "online",
        "WANDB_RUN_GROUP": arguments.output.name,
        "WANDB_JOB_TYPE": phase,
        "WANDB_TAGS": f"{phase},{script.stem},seed={seed},actor_lr={actor_rate},inactor_lr={inactor_rate},epochs={epochs}",
        "DEEP_HEDGING_SAVE_ARTIFACTS": "1" if phase == "final" else "0",
        "PYTHONUNBUFFERED": "1",
    })
    for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        environment.pop(name, None)
    log_path = arguments.output / f"{phase}_{index:02d}.log"
    cvar = None
    print(f"{phase.upper()} {index}: {script.stem} seed={seed} actor_lr={actor_rate:g} inactor_lr={inactor_rate} epochs={epochs}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        with subprocess.Popen(
            build_trial_command(script, actor_rate, seed), cwd=ROOT.parent,
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                matches = CVaR_PATTERN.findall(line)
                if matches:
                    try:
                        cvar = float(matches[-1])
                    except ValueError:
                        cvar = None
            exit_code = process.wait()
    if cvar is not None and not math.isfinite(cvar):
        cvar = None
    return {
        "phase": phase, "index": index, "seed": seed,
        "actor_learning_rate": actor_rate, "inactor_learning_rate": inactor_rate,
        "epochs": epochs, "policy_epochs": arguments.policy_epochs,
        "inaction_epochs": arguments.inaction_epochs, "episodes": arguments.episodes,
        "cvar_95": cvar, "exit_code": exit_code, "log_path": str(log_path),
    }


def run_experiment(script, arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    tuning_epochs = arguments.epochs if script.stem.startswith("inactive_") else math.ceil(0.6 * arguments.epochs)
    candidates = []
    with (arguments.output / "runs.csv").open("w", newline="", encoding="utf-8") as output:
        writer = None
        for index, (actor_rate, inactor_rate) in enumerate(trial_rates(script), 1):
            result = run_training(script, actor_rate, inactor_rate, 0, tuning_epochs, arguments, "tuning", index)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=list(result))
                writer.writeheader()
            writer.writerow(result)
            output.flush()
            if result["exit_code"] == 0 and result["cvar_95"] is not None:
                candidates.append(result)
        if not candidates:
            raise RuntimeError("All 12 tuning trials failed or returned no finite CVaR; final training was not started")
        best = min(candidates, key=lambda result: result["cvar_95"])
        config = {
            "experiment": script.stem, "actor_learning_rate": best["actor_learning_rate"],
            "inactor_learning_rate": best["inactor_learning_rate"],
            "joint_learning_rate": best["actor_learning_rate"],
            "selected_trial": best["index"], "cvar_95": best["cvar_95"],
            "valid_trials": len(candidates), "attempted_trials": 12,
            "tuning_seed": 0, "tuning_epochs": tuning_epochs,
            "final_epochs": arguments.epochs, "policy_epochs": arguments.policy_epochs,
            "inaction_epochs": arguments.inaction_epochs, "episodes": arguments.episodes,
            "final_seeds": list(range(1, 11)),
        }
        (arguments.output / "best_config.json").write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"BEST saved: {arguments.output / 'best_config.json'} ({len(candidates)}/12 valid trials)", flush=True)
        failed_seeds = []
        for seed in range(1, 11):
            result = run_training(
                script, best["actor_learning_rate"], best["inactor_learning_rate"],
                seed, arguments.epochs, arguments, "final", seed,
            )
            writer.writerow(result)
            output.flush()
            if result["exit_code"] != 0 or result["cvar_95"] is None:
                failed_seeds.append(seed)
        if failed_seeds:
            raise RuntimeError(f"Final runs failed or returned no finite CVaR for seeds {failed_seeds}; see runs.csv and logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, choices=sorted(script.name for script in SIMULATIONS_DIR.glob("*.py")))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=positive_int, default=60)
    parser.add_argument("--policy-epochs", type=positive_int, default=40)
    parser.add_argument("--inaction-epochs", type=positive_int, default=20)
    parser.add_argument("--episodes", type=positive_int, default=400)
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    if arguments.output.exists():
        parser.error("Output directory already exists; choose a new directory to preserve previous results")
    run_experiment(SIMULATIONS_DIR / arguments.script, arguments)


if __name__ == "__main__":
    main()