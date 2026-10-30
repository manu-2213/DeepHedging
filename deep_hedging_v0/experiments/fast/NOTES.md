# Fast pipeline: notes, bugs found, and how to run

One friction regime only: **1% of traded notional on the underlying, 5% of traded premium on
options** (`config.py`). The call product hedges with the underlying, so only the 1% applies.
The barrier (doc) and digital (con) products hedge with a call and a put, so only the 5% applies.

None of this code has been executed yet. Run `slurm/smoke_test.slurm` (1 hour, tiny sizes) on
the HPC before submitting the grid.

## Why the old pipeline was so slow

| Cause | Where |
|---|---|
| Each slurm job ran 12 tuning trainings plus 10 final-seed trainings back to back | `hp_tuning/run_experiment.py` |
| One training is 60 epochs x 400 episodes = 24,000 rollouts of 250 Python env steps through Gym and TorchRL | `utils/training_loop.py` |
| Only 4 gradient steps per 500,000-frame batch, so huge amounts of simulation per update | `sub_batch_num=4`, one pass |
| Heston option prices are computed in a Python loop over every path, asset, strike and day, with two JAX calls each | `hedging/simulators.py` `euro_call` |
| Every epoch regenerates paths and reprices everything, including Heston DOC by nested Monte Carlo | `env.reset` with a new seed |

Measured once on the laptop (BSM call, MLP, 2,000 envs): about 1.5 s per episode, so about 10 h
per training on CPU, before the Heston repricing cost.

## What the fast pipeline does instead

1. **Path banks** (`path_bank.py`): train, validation and test paths and every price are simulated
   once per product and cached in `.path_banks/`. Heston pricing is vectorised on the GPU with the
   same characteristic-function formula and integration grid. BSM reuses `BlackScholesSimulator`.
2. **GPU environment** (`env.py`): all options step in lock-step as tensors. Features, portfolio
   update, reward and terminal P&L follow the old envs line by line.
3. **Lean PPO** (`ppo.py`): same networks, clip, value and entropy coefficients, gamma and lambda.
   Each rollout is reused for 4 epochs of 8 minibatches instead of one pass of 4.
4. **Patience** (`train.py`): validation CVaR95 every 20 iterations. Training stops after 15
   evaluations without a 0.1% improvement, and the best checkpoint is restored.
5. **Optuna** (`run.py`): TPE sampler plus median pruning, which kills bad learning rates early.
   The study is stored in SQLite, so re-submitting a job resumes it.
6. **Double net warm start**: the gate is added on top of the single net's checkpoint with the same
   seed. That mirrors the old two-phase training without paying for phase one twice. It also makes
   single vs double a paired comparison per seed. The gate starts at "never hold", so on
   validation the double net can only match or beat its single net.

Search spaces: single actor learning rate, log-uniform 1e-6 to 3e-3 for the MLP and 1e-7 to 1e-3
for the RNN. Double: gate learning rate 1e-5 to 1e-1 and joint actor learning rate from 0.001x to
2x the single net's best rate. Defaults are 20 trials for single and 30 for double.

## Bugs and problems found in the old code

Ranked by how much they affect the single vs double comparison.

1. **Learning rates were chosen on the test year.** `testing.py` prints `SWEEP_RESULT cvar_95` from
   the 2021 test env, and the sweeps pick the minimum. The reported test CVaR is optimistically
   biased. The new code selects on a validation split and only reports the test year.
2. **Heston call double-net scripts run with no transaction costs.**
   `inactive_heston_hedge_call_{mlp,rnn}.py` never pass `transaction_cost`, and the env default
   is `False`. The single-net Heston call scripts do pay costs.
3. **Unequal tuning budgets.** `sweep_learning_rates.py` gives the double net 36 configurations and
   the single net 12. `INACTOR_RATES` lists 1e-3 twice and `attempted_trials` is hard-coded to
   12. `run_experiment.py` tunes the double net for 60 epochs and the single net for 36.
4. **Test CVaR95 is estimated from very few samples.** The test year has 100 paths x 5 strikes for
   BSM and 20 x 5 for Heston. CVaR95 is then the mean of 25 or 5 losses, so learning-rate selection
   was mostly noise. The new test bank has 20,000 options for BSM and 10,000 for Heston.
5. **Neither net saw its current position or its trading history.** The 11 or 17 features were
   market-only. With proportional costs the optimal hedge depends on the current holding (a
   no-trade band). Both nets now get trading-history features by default (see below).
   `--no-trade-features` switches them off.
6. **Heston DOC prices are mostly Monte Carlo noise from day to day.** Every date used fresh inner
   paths (500), so the price error was about the size of a daily price move. The new pricer uses
   common random numbers across dates and 2,048 inner paths.
7. **Option fee was 2x the base rate, not 5x.** The comment in `env_hedge_conc.py` says 5x.
   `illiquid_hedging_all_experiments.slurm` says "rate=0.01 => 5%/10%" but sets 0.05. That run was
   5% on the underlying and 10% on options.
8. **The same paths are replayed for 400 episodes.** Paths only change at epoch boundaries, so a
   run sees 60 distinct path sets. The new code samples a fresh subset of a large bank every rollout.
9. **The digital (con) product is badly scaled.** It pays 1 index point, while the hedge calls are
   worth hundreds and positions are capped at [-1, 1]. Any trade costs far more than the liability,
   so "do nothing" is close to optimal. Consider `digital_payout` near the strike.
10. **Inconsistent action bounds and gate outputs.** The single-net RNN call actor has no action spec,
    so it trades in [-1, 1] while the MLP uses [0, 1]. The MLP gate has one logit and the RNN gate
    one per instrument. The new code uses [0, 1] for the call, [-1, 1] for the options, and one
    logit per instrument for both architectures.
11. **Objective mismatch.** Training minimises the sum of daily |portfolio - option|, selection uses
    terminal CVaR95, and there is no liquidation cost at maturity. This is unchanged, but worth
    stating in the paper.
12. **Credential in git.** `deep_hedging_v0/hpc_notes.txt` contains what looks like a password and
    is staged. Remove it from the index and rotate it.

## Scope: RNN only, window of 8

The grid runs only the LSTM nets: 6 products, each with a single and a double run. The LSTM reads
a sliding window of the last 8 observations (it was 10). The MLP code path still exists but has no
slurm files.

## Trading-history features

Both nets see the same 3D + 3 extra columns at every step of the window, with D = 1 for the call
and 2 for the option hedges. Because they sit inside the 8-day window, the LSTM sees the recent
trade / no-trade pattern, not just today's state.

| Feature | Meaning |
|---|---|
| holdings (D) | current position before today's decision |
| last trade (D) | signed size of the most recent rebalance |
| traded flag (D) | 1 if that instrument was traded on the previous step |
| steps since last trade / N | how long the position has been left alone |
| cumulative costs, % of initial spot | how much trading has cost so far |
| hedging error PV - V, % of initial spot | how far the hedge currently is from the liability |

For the single net the traded flag is almost always 1, because continuous actions rarely repeat
exactly. That is accurate information about its behaviour, not a bug.

## Other changes worth knowing

- Training rewards are scaled to % of spot (100 / mean S0) so the critic sees sensible magnitudes,
  and advantages are normalised per minibatch. Neither changes the optimal policy. Reported
  metrics are in raw index points as before.
- PPO entropy uses the base Normal's entropy rather than TorchRL's sample estimate. The
  coefficient is 1e-3, so the effect is small.
- The double net is reported with a deterministic gate (used for selection) and a sampled gate
  (the old default), under `test` and `test_sampled_gate`.
- In the double net the actor still gets PPO credit for proposals the gate overrode, as before.
  Masking held steps out of the actor loss is a sensible follow-up.

## Is 1% / 5% reasonable for a liquid asset?

The 5% on options is plausible: bid-ask spreads on index options are a few percent of premium,
more for out-of-the-money strikes. The 1% on the underlying is not "liquid". For the S&P 500,
trading costs are around 0.01% to 0.05%. At 1%, rebalancing every day costs more than the option
premium for most strikes. That is a useful high-friction regime where inaction should matter,
but call it "high friction" or "illiquid underlying" in the write-up. A 0.1% / 1% run would be
the natural liquid counterpart.

## How to run on the HPC

```bash
cd /data/home/acw796/DeepHedging
source .venv/bin/activate && pip install "optuna>=4.0"   # if the HPC venv lacks it
cd deep_hedging_v0/experiments/fast/slurm
sbatch smoke_test.slurm          # about 1 hour; check the .out file first
bash submit_all.sh               # banks -> single rnn -> double rnn, all chained
bash submit_all.sh bsm_call      # or only some products
```

There is one slurm file per run: `<dynamics>_<product>_rnn_<single|double>.slurm`, plus
`bank_<dynamics>_<product>.slurm`. If a job hits its walltime, `sbatch` the same file again and it
resumes.

Results land in `deep_hedging_v0/experiments/fast/results/<experiment>/<single|double>/`:

- `study.db` is the Optuna study.
- `trials/` holds one checkpoint and validation history per trial.
- `final/seed_<s>.json` holds the per-seed validation and test metrics.
- `summary.json` holds the best parameters and the test mean and std across seeds. For the double
  net it also holds the paired single-vs-double comparison: CVaR95 improvement, wins and the
  Wilcoxon p-value.
