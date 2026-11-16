# Experiments

Everything here runs, evaluates or analyses an experiment of the paper. The hedging engine itself is in
`hedging_engine/`; nothing in this folder is imported by it.

```
train.py          one training run: python -m experiments.train ... --out experiments/results/<batch>/<run>
baselines.py      model-based references for one cell: python -m experiments.baselines --dynamics ... --product ...
protocol.py       split seeds (train 101, validation 202, test 303), bank sizes, the two cost regimes
calibration/      option data and the parameter fits the engine simulates from
slurm/            job generator (make_main.sh), evaluation and check jobs, jobs/ = argument lists of submitted batches
evaluation/       re-evaluation of trained nets (no training)
analysis/         collecting runs into tables, the main table, the gate and transfer analyses
figures/          the paper's figures
results/          run outputs (not tracked): results/<batch>/<run>/{log.txt, summary.json, best_trial.pt, *_test_paths.npz}
```

Commands are run from the repository root unless a path says otherwise.

## Runs and their names

A run is one net on one cell at one cost level with one seed. `slurm/make_main.sh <recipe> ...` writes its job script;
the script trains in 50-minute pieces and resubmits itself until the run has converged (see `docs/hpc.md`).

| Recipe | Run name under `results/main/` | What it is |
|---|---|---|
| `single` | `<dyn>_<prod>_fee<fee>_single_lr<rate>_s<seed>` | single policy at one actor rate |
| `double` | `<dyn>_<prod>_fee<fee>_double_i<il>_j<jl>[_fromlr<rate>]_s<seed>` | Actor–Inactor started from a converged single; inactor rate `il`, joint actor rate `jl` |
| `band` | `..._band_lr<rate>_s<seed>` | learned band net (target and half-width), one stage |
| `pair` | `..._pair_i<il>_j<jl>_lr<rate>_s<seed>` | Actor–Inactor from scratch at parameter parity (two nets of width 45 against one of 64) |
| `gate` | `..._gate_i<il>_j<jl>_s<seed>` | Actor–Inactor whose inactor also sees holding, proposal, gap and estimated cost |
| `xfer` | `..._xfer_i<il>_s<seed>` | actor taken from a single trained at the *other* cost level and frozen; only the inactor is trained |

`<dyn>` is `bsm` or `heston`, `<prod>` is `call`, `doc` (down-and-out call) or `con` (cash-or-nothing), `<fee>` is `01`
(low cost) or `1` (high cost), rates are written without the minus sign (`3e4` = 3e-4).

The arguments common to every run are the `COMMON` line of `make_main.sh`. That line is the method of the paper.

## What produced each result of the paper

**Main table (single policy against Actor–Inactor, 12 cells, 5 seeds).**
1. Seed 1, tuning: `make_main.sh tune <dyn> <prod> <fee>` trains the single policy at five actor rates
   (3e-5 … 3e-3; 1e-2 was added by hand where 3e-3 won). When they have finished, `pick` takes the validation-best and
   submits its four Actor–Inactors (inactor rate {1e-3, 1e-4} × joint rate {1e-4, 1e-5}). The picks are recorded in
   `slurm/jobs/tuning_picks.txt`.
2. Seeds 2–5: `make_main.sh seedchain` trains the seed's single at the cell's first rate and then its Actor–Inactor at
   the chosen rates. `seed2chain` trains the single at the cell's second rate as well; an Actor–Inactor is started from
   it only if it beats the first on validation. The rate pairs per cell are the `MAIN` table in
   `analysis/main_table.py`.
3. BSM call, seed 1, was trained during method development with the same arguments under other names:
   `results/v2_4/tgsg8_fee{01,1}_s1` (single) and `results/v2_4/dbltg8_i<il>_j<jl>_fee{01,1}_s1` (Actor–Inactor).
   The job lists below refer to them by those names.
4. Model band and no hedge: `sbatch slurm/baselines_main.slurm` (`experiments.baselines` for every cell, both costs)
   writes `results/main_baselines/<dyn>_<prod>_fee<fee>.json`.
5. Table: `analysis/collect_runs.py` (run where the results are) gathers every run's validation curve, test results
   and settings into `results/collected/runs.json` and `curves.json`; `python experiments/analysis/main_table.py`
   prints the table. Every choice between runs uses validation numbers only. `analysis/selected_runs.json` is the
   resulting selection per cell and seed.

**Out-of-sample years 2021 / 2022 / 2023.** Market `sp500x` holds seven calibration years; its test split is the last
three. `slurm/eval_years.slurm` with one line of `slurm/jobs/years_jobs.txt` evaluates a cell's ten selected nets per
year (`evaluation/eval_years.py` → `results/main_years/<cell>/`) and the model band for that market
(`results/main_years_baselines/`). `slurm/band_years_hdoc01.slurm` reruns the band alone with a longer limit (the
Heston barrier's nested Monte Carlo does not fit one hour).

**Figures.**

| Figure | Script | Inputs |
|---|---|---|
| tracking error per cell (`fig_main_tracking_zoom`), Actor–Inactor over single (`fig_double_over_single`) | `figures/make_figs.py` | `results/collected/` |
| distribution of tracking error, high cost (`fig_track_hist_fee1`) | `figures/make_path_figs.py` | `results/main_big/`: 50,000 fresh test paths per cell, `slurm/eval_big.slurm` with a line of `slurm/jobs/big_cells.txt` (`evaluation/eval_big.py`) |
| through the life of the contract (`fig_life_fee1`), hold map (`fig_hold_map`), sample paths (`fig_sample_paths`), position against model delta (`fig_delta_ols`) | `figures/make_path_figs.py` | the test traces of the seed-1 runs (`slurm/jobs/figure_runs.txt`) and the model band's per-path dumps: `slurm/band_paths.slurm` (`evaluation/band_paths.py`) |

`PAPER=1` switches both figure scripts to the paper's wording and writes to `paper_notes/figures/paper`.

**Follow-up comparisons** (logged in `TRAINING.md`; job lists are the arguments after the recipe name):

| Batch | Launch | List |
|---|---|---|
| learned band net, 60 runs | `make_main.sh band <line>` | `slurm/jobs/band_jobs.txt` (`dyn prod fee seed rate`) |
| parameter parity, 60 runs | `make_main.sh pair <line>` | `slurm/jobs/pair_jobs.txt` (`dyn prod fee seed rate il jl`) |
| proposal-aware inactor and cost-level transfer, 30 + 30 runs | `make_main.sh <line>` | `slurm/jobs/gate_jobs.txt` (lines start with `gate` or `xfer`) |

* `analysis/hold_analysis.py`: is the inactor's hold decision a function of the trade size (a band) or of the market
  state? Logistic regressions on the test traces.
* `analysis/shrink_check.py`: how the single policy's position changes between the two cost levels (why a frozen
  low-cost actor transfers on the exotics and not on the call).

**Checks.** `slurm/check_tests.slurm` (the `tests/` folder on a GPU node), `slurm/check_main.slurm` (tiny single and
Actor–Inactor for every cell, chained resume),
`slurm/check_band.slurm` (the finite-difference band reproduces the analytic Whalley–Wilmott band on the BSM call),
`slurm/bank_heston2000.slurm` (pre-builds the Heston banks).

## Calibration

`calibration/option_data/<market>_<year>.json.bz2` are the option chains (one row per quote). From them:

* `calibration/calibrate_heston.py` fits the Heston parameters per year with QuantLib and writes
  `hedging_engine/data/heston_params_QL_v2_<market>.json`. Years already in the file are skipped.
* `calibration/fit_bsm.py <market> <years>` writes the Black–Scholes inputs (spot, realised volatility, five strikes)
  to `hedging_engine/data/bsm_data_<market>_2017_2021.json`.
* `calibration/convert_spx.py` converts the 2022–2023 OptionsDX end-of-day files into the same yearly format. Those two
  years are purchased data and are not tracked; `slurm/calib_2223.slurm` is the calibration job for them.

Market `sp500` (2017–2021) is the paper's main market and `sp500x` (2017–2023) adds the two out-of-sample years. The
first five years of `sp500x` are the `sp500` fits, so the main results' banks are unchanged. Four more markets
(Alphabet, Apple, Microsoft, Nasdaq-100) are calibrated but not used in the paper.
