# fast_hedging: notes, findings and how to run

## Why the old pipeline was slow
Measured on the original code (BSM call, MLP, 100 paths x 4 years x 5 strikes = 2,000 envs):

| item | value |
|---|---|
| one episode (collect 250 steps + GAE + 4 PPO steps), CPU | about 1.5 s |
| episodes per training run (60 epochs x 400) | 24,000 |
| one training run | about 10 h |
| training runs per slurm job (12 tuning + 10 seeds) | 22 |

Causes, in order of impact:
1. **Sequential trainings inside one job.** Each job ran 12 (single) or 36 (double) tuning
   runs plus 10 seeds, back to back, each with a fixed 60-epoch budget and no early stopping.
2. **Very little learning per simulated step.** 500k frames per batch, but only 4 gradient
   steps (one pass, minibatch 125k). Most wall time was spent stepping the environment.
3. **Python/TorchRL stepping.** GymWrapper + SyncDataCollector step 250 times per episode with
   numpy <-> torch copies, although the market does not depend on the agent at all.
4. **Re-pricing every epoch.** A full reset re-simulates and re-prices all options. Under Heston
   the pricing is a Python loop over every (path, asset, strike, time) with two JAX calls per
   point, `euro_put` re-prices all calls again, and the DOC nested Monte Carlo runs per time step
   in numpy. This is why the Heston runs used only 10 to 20 paths.
5. The same 100 paths were replayed for 400 episodes inside an epoch.

## What fast_hedging does instead
* **Market banks**: paths and every price (liability, hedge instruments, features) are built
  once per (dynamics, product, split) and cached as `.npz` in `.fast_hedging_cache/`.
  Heston pricing is batched in torch (the characteristic function is shared across strikes),
  and the DOC nested MC simulates all start times in one pass on the GPU.
* **Vectorised environment**: features for the whole path are computed in one call; the
  hedging recursion is a 250-step loop of batched tensor ops on the GPU.
* **Lean PPO**: plain torch, 4 PPO epochs over minibatches of 16k samples per rollout of
  64 paths per asset, GAE(0.99, 0.95), clip 0.1, value coef 0.1, entropy 0.001 (the old values).
* **Early stopping**: validation CVaR95 every 10 iterations, patience 10 evaluations, best
  weights restored. Long-learning settings keep training; flat ones stop early.
* **Optuna**: 30 trials per run for both nets (equal budget), TPE over log learning rates
  (8 random start-up trials), median pruning on the validation curve after 6 trials, a 180 h
  cap, journal storage so a killed job resumes where it stopped. Single net tunes 1 learning
  rate; double net tunes actor and inactor rates (joint-phase actor rate = actor rate unless
  `--tune-joint-lr`). Then 5 final seeds with the best rates.
* **Clean splits**: train and validation are different paths of the 2017-2020 calibrations;
  the 2021 calibration is the test set and is never used for selection.

Parity with the old environments is checked by `python -m fast_hedging.tests.test_parity`
(same paths, same actions: prices, features, rewards and terminal P&L agree).

Measured per-iteration cost on a laptop CPU, BSM call, 64 paths x 20 option slots:
MLP single about 1 s, MLP double about 1.5 s, sliding-window RNN single about 45 s.

**RNN only.** The grid runs RNNs only; the MLP simulation scripts were deleted. The default
is the old sliding-window design with a window of 8 (`--rnn-mode window --window 8`): a
2-layer LSTM reads observations t-7..t at every step. `--rnn-mode recurrent` runs the same LSTM
once over the whole path instead (about 8x less LSTM work, minibatches are whole paths).
Nothing has been timed on a GPU; `smoke_test.slurm` prints seconds per iteration.

**Optimisations in this version**
* Heston characteristic-function pricing in complex64, the precision of the original JAX code.
* Barrier Monte Carlo updates only the inner paths that have started (half the work), with
  larger GPU batches.
* Validation every 10 iterations, patience 10 (the same 100-iteration stopping window as
  before, half the validation cost).
* cuDNN autotuning and TF32 matmuls on the GPU; larger encoder chunks.
* Not done, because they need a test run first: `torch.compile` or CUDA graphs for the
  250-step rollout loop, bf16 autocast for the LSTM, caching validation observations.

## Information leakage audit
No input uses information from after the decision time:
* Market features at step t use prices at t and t-1, time to maturity and r only.
* Agent state at t uses holdings, cash and costs up to t and the liability price at t.
* The inaction net sees the proposed trade and its cost at prices of t.
* The window LSTM reads t-7..t (zero padding before the start); the recurrent LSTM is
  one-directional; the terminal observation is never fed to a policy.
* Knock-out flags use a running prefix of the path. Nested-MC prices at t use only S_t, v_t.
* Observation normalisation uses training-bank statistics only. Learning rates and early
  stopping use the validation bank (different paths of the training years). The 2021 test
  year is only used in the final report.
* Under Heston, option prices reveal the current variance v_t. That is realistic: market
  option prices are observable.

## Protocol kept from the old code
* Single net: actor-critic, hidden 64. Double net: actor + inaction net, each hidden 45.
* **Agent state** (on by default, `--no-agent-state` to remove), fed to the policy and value heads:
  both nets get current holdings (the last executed hedge), tracking error PV - liability,
  transaction costs paid so far (both divided by the claim's premium) and steps since the last
  executed trade. The inaction net also gets the proposed position, the proposed trade and the
  cost of executing it.
* Double net trains its actor alone first, then actor + inactor jointly; the actor is updated
  every third iteration in the joint phase. Default: joint-phase actor LR = stage-1 LR.
* Rewards: per-step `-|portfolio - liability|`. Test metric: CVaR95 of terminal P&L.

## Changes from the old code (switchable)
| change | why | flag to revert |
|---|---|---|
| reward divided by the claim's premium | dollar rewards on a 3,000-point index made LRs fragile | `--reward-norm none` |
| observations standardised with training statistics | raw features range from 1e-3 to 1e3 | `--no-obs-norm` |
| advantages normalised per minibatch | standard PPO practice | `--no-adv-norm` |
| actor not trained on proposals the inactor overrode | those actions were never executed | `--credit-held-actions` |
| separate underlying and option fees | the old option fee was hard-coded to 2x | n/a |
| stage-1 actor reused from the single-net run | optional, makes the comparison paired | `--actor-init from-single --single-dir ...` |

## Bugs and issues found in the old code
1. **LRs selected on the test set.** `run_experiment.py` and `sweep_learning_rates.py` pick the
   LR with the lowest *test* CVaR, so reported test numbers are biased in favour of whichever
   net had more trials.
2. **Unequal tuning budgets.** Double net 36 trials and 60 tuning epochs, single net 12 trials and
   36 epochs. `INACTOR_RATES` lists 1e-3 twice, and `attempted_trials` is hard-coded to 12.
3. **The Heston call double-net scripts (mlp and rnn) never pass `transaction_cost`**, so they
   trained without costs while their single-net counterparts had costs. All six Heston
   double-net scripts also take the joint-phase LR from an environment variable instead of
   `--actor_learning_rate` (harmless under `run_experiment.py`, which sets both equal).
4. **Option fee hard-coded to 2x the rate** in `env_hedge_doc.py` and `env_hedge_conc.py`
   (comment says 5x). The illiquid run therefore used 5% / 10%.
5. **Neither net saw its current position.** With proportional costs the optimal hedge depends
   on current holdings (no-trade band), and the inaction net decided "trade or not" without
   seeing the trade it would make. Fixed by the agent-state inputs above.
6. RNN call scripts have no action bounds (range -1..1) while the MLP call uses 0..1.
7. Feature ratios use `np.finfo(dtype).eps` of float64 arrays, so ratio features blow up to
   about 1e3 when a barrier or put price hits zero.
8. No liquidation cost at maturity, and CON pays 1 on a 3,000-point index, so its premium is
   tiny compared with any hedge trade.
9. `hpc_notes.txt` contains what looks like a password and is staged in git.

## Are 1% and 5% "liquid"?
For the S&P 500 index, 1% per trade on the underlying is far from liquid; index futures and
ETFs trade at around 1-5 basis points. 5% of premium for index options is plausible to high
(bid-ask spreads are a few percent of premium near the money, more out of the money).
Delta hedging the 2021 test call daily at 1% costs about the option's premium in fees
(test-bank BS delta hedge: mean cost about 204 against a premium of about 210), so 1%/5% is a
high-friction regime where trading less clearly pays. That is a sensible setting for showing
the double net's value, but call it "high transaction costs", not "liquid". A 0.1%/1% run
would be the natural liquid counterpart: change `FEES` in `make_slurm.sh`.

## How to run (HPC)
```bash
cd /data/home/acw796/DeepHedging && source .venv/bin/activate && uv pip install "optuna>=4.0"
cd deep_hedging_v0/experiments/simulations/slurm/liquid_1_5
mkdir -p logs && sbatch smoke_test.slurm      # up to 4 h; check logs/ and the parity output
./submit_all.sh                               # bank build, then 12 RNN jobs (one per run)
```
Outputs go to `deep_hedging_v0/experiments/results/liquid_1_5/<run>/`:

| file | contents |
|---|---|
| `summary.json` | best LRs, trial table, seed mean/std of test metrics, baselines |
| `trials/trial_NNN.json` | every Optuna trial (complete, pruned, failed): params, validation curve, per-iteration training log |
| `final_seedN.json` | validation curve, per-iteration training log, test metrics |
| `final_seedN_test_paths.npz` | per test env: P&L, cost, tracking error, hold rate, trade count; `trace_*`: step-by-step holdings, proposed trades, hold decisions and probabilities, costs, portfolio value, liability, prices for the first 100 test paths |
| `best_trial_test_paths.npz` | the same for the best tuning trial |
| `baselines_test_paths.npz` | per-env P&L and cost of no-hedge (and BS delta for the BSM call) |
| `best_trial.pt`, `final_seedN.pt` | weights |
| `optuna.journal`, `log.txt` | Optuna storage, full console log |

The per-iteration training log holds rollout stats (train CVaR, P&L, cost, hold rate, trades)
and PPO diagnostics per net (objective and value loss, entropy, approximate KL, clip fraction,
action std or mean hold probability, explained variance).

Load everything with `fast_hedging/analysis.py` (pandas):
`load_summaries`, `load_run_summaries`, `load_trials`, `load_curves`, `load_paths`,
`load_baselines`, and `compare_nets` for a paired single-vs-double bootstrap on identical test
paths (resampling whole simulated paths).
Resubmitting a run's slurm file resumes it.
