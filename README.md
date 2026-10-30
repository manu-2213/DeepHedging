# Deep Reinforcement Learning for Delta Hedging

This repository implements and compares different approaches to delta hedging, from classical Black-Scholes-Merton (BSM) to modern deep reinforcement learning methods.

## Project Overview

The project is structured in three main phases:
1. Data Generation and Simulation Framework
2. Classical Delta Hedging Implementation 
3. Deep Reinforcement Learning Implementation (Coming Soon)

### Current Features

#### Synthetic Data Generation
- Geometric Brownian Motion (GBM) for multiple asset paths
- Planned: Heston, SABR, Merton Jump, and Variance Gamma models
- Customizable parameters for volatility, drift, and time horizon

#### Classical Delta Hedging
- BSM option pricing and Greeks calculation
- Dynamic delta hedging implementation
- P&L calculation and performance metrics
- Visualization tools for hedge performance

### Coming Soon: Deep Hedging

The ultimate goal is to implement deep reinforcement learning for hedging options. This approach offers several advantages over classical methods:

1. **Model-Free Hedging**
   - No assumptions about underlying price dynamics
   - Learns directly from market data
   - Can adapt to changing market conditions

2. **Transaction Costs**
   - Naturally incorporates transaction costs in the hedging strategy
   - Balances hedging error vs trading costs

3. **Market Frictions**
   - Handles discrete-time hedging
   - Accounts for bid-ask spreads
   - Manages inventory constraints

## Tune Then Train (Current Slurm Workflow)

Submit the existing per-experiment Slurm files. Each job now runs 12 LR
candidates first, saves the lowest valid CVaR95 configuration, and then
automatically trains that configuration from scratch on seeds 1-10. No
separate sweep submission or post-sweep submission is needed.

| Stage | Active-only models | Inactive models |
| --- | --- | --- |
| 12 tuning runs, seed 0 | 36 epochs (60% of 60) | Full 40 actor + 20 joint epochs |
| 10 final runs, seeds 1-10 | Full 60 epochs | Full 40 actor + 20 joint epochs |

All runs use 400 episodes per epoch. The LR candidates are the same 12-entry
grids documented below: active-only models vary actor LR; inactive models
vary actor and inactor LRs independently. The actor LR is applied in both
pretraining and joint training. The existing model initialization is repeated
for each final seed; tuning checkpoints are not reused.

From the repository root on the cluster, submit one experiment:

```sh
sbatch deep_hedging_v0/experiments/simulations/slurm/inactive_bsm_hedge_call_mlp.slurm
```

Or submit all 24 (this does not impose a global concurrency limit):

```sh
for job in deep_hedging_v0/experiments/simulations/slurm/*hedge*.slurm; do
   sbatch "$job"
done
```

Both stages force W&B online; authenticate the cluster environment before
submission. The launchers use their original module-loading and `.venv`
activation commands. They do not install or upgrade any packages. Training
subprocesses preserve inherited `PYTHONPATH`, `PYTHONHOME`, and other module
environment settings, and run from the repository root as the original
direct commands did. `WANDB_PROJECT` is inherited when set; otherwise W&B
uses its existing configuration/default behavior.
Runs are grouped by experiment and Slurm job ID, with
job types `tuning` and `final` and tags showing phase, seed and LRs. Training
output streams live to Slurm stdout and individual log files. Final runs
save model/plot artifacts; tuning runs do not.

Each job saves `runs.csv`, `best_config.json`, and per-run logs under
`deep_hedging_v0/experiments/hp_tuning/results/<experiment>_<job-id>/`.
The winning config is saved before final training starts. Invalid/failed
tuning candidates cannot win; if all fail, final training does not start.
All ten final seeds are attempted, and the job exits nonzero if any fail.

The existing 240-hour allocation now covers 22 sequential runs; adjust
walltime if needed within cluster limits. Already-running jobs retain their
submitted scripts. Do not submit the legacy sweep arrays as well unless you
intend to repeat tuning. CVaR selection uses the existing test split as
validation data; reserve independent data for final outperformance claims.

## Legacy Standalone Learning-Rate Search

Run these commands from the repository root using the Python environment with
the project dependencies installed:

```sh
python deep_hedging_v0/experiments/hp_tuning/sweep_learning_rates.py --dry-run
python deep_hedging_v0/experiments/hp_tuning/sweep_learning_rates.py --output learning_rate_sweep.csv
```

The sweep covers all 24 scripts in `deep_hedging_v0/experiments/simulations`:
BSM/Heston, call/concentration/DOC, MLP/RNN, active/inactive. Each experiment
gets exactly 12 configurations, for 288 training runs in total.

| Model | Actor learning rates | Inactor learning rates |
| --- | --- | --- |
| Inactive MLP | `2.5e-5`, `5e-5`, `1e-4` | `1e-4`, `3e-4`, `1e-3`, `3e-3` |
| Inactive RNN | `2.5e-6`, `5e-6`, `1e-5` | `1e-4`, `3e-4`, `1e-3`, `3e-3` |

Inactive models use the full 3-by-4 Cartesian grid, including the current
default pair. The selected actor LR applies during both actor pretraining
and joint training; the inactor LR is independent. Active models have no
inactor, so they test 12 distinct actor rates: multipliers
`0.25, 0.4, 0.5, 0.65, 0.8, 1, 1.25, 1.5, 1.75, 2, 2.5, 3` around
`5e-5` (MLP) or `5e-6` (RNN).

Defaults preserve 60 active epochs versus 40 actor-pretraining + 20 joint
epochs, with 400 episodes per epoch and training seed 0. Epoch counts match,
but optimizer-update counts and runtime need not match. Override budgets with
`--epochs`, `--policy-epochs`, `--inaction-epochs`, and `--episodes`; unequal
total epochs produce a warning. `--seed` sets the training seed explicitly
for every script. Existing evaluation uses seed 0 independently.

The CSV is flushed after each trial and records rates, budgets, seed, final
CVaR95, exit status, and a log path. Full trial logs are placed beside the CSV
in a `.logs` directory. Existing CSV files are never overwritten; choose a
new output for another run. W&B and model/plot artifact saving are disabled.
Use `--scripts bsm_hedge_call_mlp.py inactive_bsm_hedge_call_mlp.py` to select
a matched subset. The console reports each best valid trial and the paired
active-minus-inactive CVaR difference: positive means inactive performed better.
Failed trials and missing/non-finite metrics cannot win.

This is a validation search, not evidence that inactive models outperform:
the scripts' existing test split becomes validation data when used to select
LRs. Reserve fresh held-out data for final reporting, and confirm the chosen
configurations across seeds. Inactive ranking uses the existing primary
evaluation (deterministic actor, stochastic inactor), not whichever evaluation
mode happens to score best.

Run the lightweight sweep tests without importing the ML stack:

```sh
PYTHONPATH=deep_hedging_v0 python -m unittest experiments.hp_tuning.test_sweep_learning_rates
```

### Legacy Sweep Array Submission

On the cluster, from the repository root:

```sh
sbatch deep_hedging_v0/experiments/hp_tuning/slurm/bsm_lr_search.slurm
sbatch deep_hedging_v0/experiments/hp_tuning/slurm/heston_lr_search.slurm
```

Each array has six tasks: `0=call_mlp`, `1=call_rnn`, `2=con_mlp`,
`3=con_rnn`, `4=doc_mlp`, `5=doc_rnn`. Each task runs the active and inactive
versions sequentially, with 12 configurations each. Together the arrays run
all 288 trials, using seed 0 (no extra seed loop). Each array permits two
concurrent tasks, so submitting both uses at most four GPUs.

Resources and modules match the existing cluster scripts: account
`pilot_sae_gpu`, partition `sae`, one GPU, 12 CPUs, 240 GB RAM and a 240-hour
walltime per task. Python comes from the cluster repository's `.venv`.
The default repository location is `/data/home/acw796/DeepHedging`; set
`DEEP_HEDGING_REPO` before submission to override it. Set `SWEEP_SEED` to
override the training seed. Path counts are fixed at 50 for BSM and 10 for
Heston, with four minibatches, to avoid inheriting local smoke-test settings.

Results and full trial logs are grouped under
`deep_hedging_v0/experiments/hp_tuning/results/lr_search_<array-job-id>/`,
with one CSV per active/inactive pair. Slurm stdout/stderr use the job name,
array job ID and task ID in the submission directory. The end of each stdout
file reports both best configurations and their CVaR comparison.

For a cluster dry run, append `--dry-run` to either `sbatch` command. To run
only a selected pair, override the array, for example `sbatch --array=0 ...`.
To change concurrency, use `sbatch --array=0-5%1 ...`. Cancel an array with
`scancel <array-job-id>`. These launchers have not been submitted locally.

### Run the Selected Learning Rates

As soon as an experiment finishes its 12 trials, the sweep saves its lowest
valid CVaR95 configuration in a `.best` directory beside the CSV. Each
experiment gets two files:

- `<experiment>.json`: actor/inactor/joint LRs, selected trial and CVaR,
   valid/attempted trial counts, training seed, training budgets, path counts,
   minibatch count, and source CSV.
- `<experiment>.slurm`: a ready-to-submit training job with those same LRs
   and budgets. Actor CLI arguments and config-only overrides are both handled;
   the selected actor LR also applies during joint training.

For example, after the BSM call MLP pair completes (replace `12345` with
the sweep's array job ID):

```sh
results=deep_hedging_v0/experiments/hp_tuning/results/lr_search_12345
sbatch "$results/bsm_hedge_call_mlp.best/bsm_hedge_call_mlp.slurm"
sbatch "$results/bsm_hedge_call_mlp.best/inactive_bsm_hedge_call_mlp.slurm"
```

To submit every saved winner from that array:

```sh
for job in "$results"/*.best/*.slurm; do
      sbatch "$job"
done
```

Each generated job trains once, defaulting to the sweep seed. For a new seed,
use `sbatch --export=ALL,RUN_SEED=1 <generated-job.slurm>`. Jobs re-enable
checkpoint/plot saving and default to online W&B (configure cluster W&B
credentials beforehand, or export `WANDB_MODE=offline`). They use the same
cluster resource/module settings as the search jobs and support
`DEEP_HEDGING_REPO` if the repository has moved.

Exports are isolated by CSV/array job, so parallel tasks cannot overwrite
each other's winners. A completed experiment's exports survive interruption
of a later experiment. No winner is generated if every trial failed; JSON
records the valid-trial count when only some succeeded. These files contain
selected hyperparameters, not trained weights: submitting a generated job
retrains the model without repeating the LR search.
