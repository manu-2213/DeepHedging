<div align="center">

# The Value of Not Acting

### Deep reinforcement learning with an inaction network for cost-aware option hedging

![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![Slurm](https://img.shields.io/badge/HPC-Slurm-0B7A75)
![Status](https://img.shields.io/badge/status-PhD%20research-8A2BE2)

</div>

---

## The idea

With transaction costs, the hard part of hedging is not only *how much* to hold but *whether to trade at all*. A deep
hedger that outputs a new position at every step has to learn inaction as a special case of action.

The **Actor–Inactor** separates the two decisions. An **actor** proposes a hedge. An **inaction network** (the
inactor) then decides, at every step, whether to execute the proposal or keep the current holding. The code compares
three designs of the same size:

| | Single policy | Band net | Actor–Inactor |
|---|---|---|---|
| **Networks** | one actor | one actor | actor + inaction network |
| **Decision at each step** | new hedge position | target and no-trade half-width | proposed position, then *trade* or *hold* |
| **Not trading** | must output the old position exactly | automatic inside the band | a Bernoulli "hold" keeps the previous hedge |
| **Classical analogue** | daily re-hedging | Whalley–Wilmott band | — |

All three are model-free: no price model, Greek or model-based strategy enters training. Model-based strategies appear
only as test references.

## What is in this repository

The code has two parts, and they are kept apart on purpose.

```
hedging_engine/          the hedging engine: a library, with no experiment in it
├── calibration.py       calibrated market parameters and the train / test split by year
├── data/                the calibrated parameters themselves (Black–Scholes and Heston, per year)
├── market.py            simulation and the cached market banks (paths + every price, built once)
├── pricing.py           vectorised Black–Scholes and Heston pricing, barrier Monte Carlo
├── features.py          observations
├── networks.py          encoders (causal transformer, LSTM), actor, inaction network
├── ppo.py               the hedging environment, PPO with GRPO advantages, training loop, evaluation
├── model_band.py        model-based reference strategy (delta / delta–gamma target with a no-trade band)
└── NOTES.md             design notes, bug log, information-leakage audit

experiments/             everything that runs an experiment of the paper
├── train.py             one training run (command line around the engine, resumable)
├── baselines.py         the model-based references for one cell
├── protocol.py          split seeds, bank sizes, the two cost regimes
├── calibration/         raw option data and the scripts that fit the parameters
├── slurm/               job generator, evaluation jobs, and the argument lists of every submitted batch
├── evaluation/          re-evaluation of trained nets: large test sample, out-of-sample years, per-path dumps
├── analysis/            collecting runs, the main table, the gate and transfer analyses
├── figures/             the paper's figures
└── README.md            which script produced which table and figure

tests/                   pricing checks, exact-resume checks, checks of the transformer and GRPO code
docs/hpc.md              how runs are launched and watched on the cluster
TRAINING.md              the working record: decisions, results, dated log
```

The engine never imports from `experiments`. An experiment is a set of arguments to `experiments.train`.

## The experiments

**Contracts and hedging instruments**

| Dynamics | Liability | Hedged with | Cost per trade, low / high |
|---|---|---|---|
| Black–Scholes, Heston | European call | the underlying | 0.1% / 1% of notional |
| Black–Scholes, Heston | Down-and-out call (barrier at 90% of strike) | a call and a put | 0.2% / 5% of premium |
| Black–Scholes, Heston | Cash-or-nothing call (pays 10% of spot) | a call and a put | 0.2% / 5% of premium |

Twelve cells (3 contracts × 2 dynamics × 2 cost regimes), five seeds each, for each architecture.

**Protocol**

* Parameters are fitted per year to S&P 500 data: the Heston parameters to the option surface (QuantLib), the
  Black–Scholes volatility to realised returns.
* Years 2017–2020 give the training and validation paths (different simulated paths of the same calibrations). 2021 is
  the test year. 2022 and 2023 are additional out-of-sample years for nets trained on 2017–2020 only.
* One measure: **tracking error**, the mean absolute gap between the hedging portfolio and the liability along the
  path. It is the training reward, the selection criterion on validation, and the reported test number. Transaction
  cost is reported next to it.
* Method, fixed for every run: a causal transformer over the last 10 observations, PPO clipping with critic-free
  group-relative advantages (GRPO, 8 copies of each path), training to convergence with early stopping on validation.
* The learning rate is the only tuned hyperparameter, per cell, on seed 1 and on validation only.
* The Actor–Inactor of a seed starts from that seed's converged single policy and then trains both networks jointly.

## Results

Test tracking error (2021), mean ± standard deviation over five seeds, and mean transaction cost per path.

| Cell | Cost | Single policy | Actor–Inactor | Model band | No hedge |
|---|---|---|---|---|---|
| BSM call | low | 9.86 ± 0.14 / 14.6 | 9.36 ± 0.13 / 13.1 | 8.36 / 12.7 | 153.8 |
| BSM call | high | 39.58 ± 0.69 / 57.3 | 37.75 ± 0.48 / 57.9 | 35.46 / 54.0 | 153.8 |
| BSM down-and-out call | low | 3.67 ± 0.20 / 0.7 | 3.69 ± 0.19 / 0.7 | 3.46 / 0.5 | 158.6 |
| BSM down-and-out call | high | 13.68 ± 0.14 / 13.6 | 13.59 ± 0.24 / 11.4 | 12.59 / 11.8 | 158.6 |
| BSM cash-or-nothing | low | 4.56 ± 0.24 / 3.8 | 4.38 ± 0.30 / 3.5 | 5.98 / 2.4 | 93.3 |
| BSM cash-or-nothing | high | 21.00 ± 2.19 / 37.9 | 14.79 ± 0.22 / 24.1 | 14.79 / 23.6 | 93.3 |
| Heston call | low | 28.63 ± 0.30 / 18.9 | 28.58 ± 0.19 / 16.7 | 84.37 / 15.1 | 157.5 |
| Heston call | high | 44.46 ± 0.67 / 57.0 | 43.98 ± 1.44 / 56.9 | 74.51 / 42.9 | 157.5 |
| Heston down-and-out call | low | 8.55 ± 0.26 / 2.4 | 8.56 ± 0.24 / 2.0 | 83.21 / 6.9 | 186.2 |
| Heston down-and-out call | high | 35.52 ± 0.99 / 40.3 | 29.98 ± 0.77 / 26.6 | 89.41 / 107.4 | 186.2 |
| Heston cash-or-nothing | low | 6.43 ± 0.45 / 5.0 | 6.28 ± 0.36 / 4.7 | 45.10 / 2.9 | 84.6 |
| Heston cash-or-nothing | high | 20.96 ± 1.46 / 32.7 | 16.74 ± 1.74 / 21.1 | 51.23 / 36.3 | 84.6 |

The model band is the model-based reference: the true model's delta (or delta–gamma portfolio) with a
Whalley–Wilmott-style no-trade band tuned on validation. Under Black–Scholes it is a strong reference; for the call it
is the Whalley–Wilmott band itself. Under Heston it does poorly in every cell, while the learned policies do not need
the model.

The table is produced by `python experiments/analysis/main_table.py`. Comparisons that are still running (the learned
band net, parameter-parity pairs, a proposal-aware inactor, and transfer of a frozen low-cost actor to the high-cost
regime) are logged in `TRAINING.md`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m tests.test_pricing          # seconds: Black–Scholes and Heston pricing checks
```

A toy run on a laptop CPU (a few paths and iterations, to see the pipeline end to end):

```bash
python -m experiments.train --dynamics bsm --product call --net single \
    --arch tfpath --advantage grpo --grpo-group 8 --reward tracking --metric mean_abs_tracking_error \
    --reward-norm none --no-obs-norm --no-adv-norm --no-agent-state --window 10 --pad-mode zero \
    --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --actor-lr-range 1e-4 1e-4 \
    --n-trials 1 --no-pruning --tuning-seeds 1 --final-seeds \
    --train-paths 24 --val-paths 12 --test-paths 12 --paths-per-iter 3 --max-iters 6 --min-iters 2 --eval-every 2 \
    --device cpu --out experiments/results/demo
```

It writes `summary.json`, the test results per path and the checkpoint to `experiments/results/demo`.

The real runs need a GPU and are launched as chains of one-hour Slurm jobs. `experiments/slurm/make_main.sh` writes
the job script of any run of the paper:

```bash
cd experiments/slurm
sbatch $(bash make_main.sh single heston doc 1 1 3e-4)     # single policy, Heston down-and-out call, high cost, seed 1
```

`experiments/README.md` lists every batch and the script behind each table and figure. `docs/hpc.md` covers the
cluster.

## Why it is fast

* **Prices once.** Paths and all option prices are simulated and priced once per (model, contract, split) and cached.
  Heston pricing is batched on the GPU with an integration limit chosen per point from the decay of the characteristic
  function. Every bank passes a martingale and a one-step drift test before it is used.
* **Hedges in tensors.** All paths step together on the GPU. There is no environment loop.
* **Encodes each path once.** The causal transformer makes one pass over a path instead of one per step.
* **Resumes exactly.** A run saves its full state and continues in the next job, so training to convergence fits
  one-hour jobs.

## History

Earlier stages of the project (the original Gym/TorchRL environments, the first vectorised pipeline, hyper-parameter
searches and the development batches that fixed the method) are not in the working tree any more. They are in the git
history under the tag `pre-cleanup-2026-11-16`.

## Background

* Buehler, Gonon, Teichmann, Wood (2019). *Deep hedging*. Quantitative Finance.
* Whalley, Wilmott (1997). *An asymptotic analysis of an optimal hedging model for option pricing with transaction
  costs*. Mathematical Finance.
* Schulman et al. (2017). *Proximal policy optimization algorithms*.
* Heston (1993). *A closed-form solution for options with stochastic volatility*. Review of Financial Studies.

<div align="center">
<sub>PhD research project · Queen Mary University of London</sub>
</div>
