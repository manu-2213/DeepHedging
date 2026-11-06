# fast_hedging: notes, findings and how to run

## v2.2 (chained 1-hour jobs; slurm folder `v2_2_5`, results `experiments/results/v2_2_5/`)
- `--time-budget-min M`: after M minutes the run saves everything and exits with code 75. The same command continues.
  Without the flag, behaviour and results are unchanged.
- Saved per seed in `<out>/resume/`: weights, optimizers, random generators, running VaR, best weights, stopping
  counters and logs. Saved every 100 iterations and at the deadline, by atomic replace.
- An interrupted Optuna trial keeps its number, parameters and pruning reports (RUNNING goes back to WAITING).
  A trial with no saved state is marked failed, as before.
- Chained jobs reseed the Optuna sampler from the trial count. The same seed in each job would redraw the first
  job's learning rates. `--tuning-hours` adds up over the jobs (`tuning_clock.json`).
- Test-set blocks for the best trial and the baselines run once and are kept as JSON.
- Stopping in `v2_2_5`: no iteration cap and patience 2000 iterations (`--patience 200`, checks every 10). That is 5
  old epochs, as one old epoch is 400 episodes and one episode is one rollout, like one iteration here. By gradient
  steps the old pipeline's 5 epochs would be only about 105 iterations, because an old episode makes 4 steps and an
  iteration about 76. The rollout count was chosen.
- Learning rates in `v2_2_5` main runs are fixed, not tuned: actor and joint 1e-5 (the author's experience with these
  RNNs), inaction net 1e-3 (the old default). The old call script defaults to 5e-6, while v2.1 selected 5.85e-4 on BSM call
  under censored stopping, which favours fast rates. Runs go in stages, BSM call first, with 3 final seeds to start.
- `--prune-warmup N` stops pruning before N validation checks in each phase of the double net, for any later tuning.
- PyTorch: `v2_2_5` runs on 2.11.0+cu128, as `gpushort` node `sbg2` has driver 12.9. v2.1 ran on 2.15.0+cu132.
- New flags with old defaults: `--value-coef`, `--init-std` and `--prune-warmup`. Check with `python -m fast_hedging.tests.test_resume`.

## v2.1 (fixes after the second audit, 1 November 2026; slurm folder `v2_1_5`)
Found by reading the v2.0 code and checking the Heston pricer against a complex128 reference.
`BANK_VERSION` is 4: every cached bank is rebuilt. Full write-up: `paper_notes/findings.tex`.

| change | why | flag / field to revert |
|---|---|---|
| **adaptive integration limit for the Heston characteristic function** (`pricing.heston_phi_max`: per point, from the decay rate of the CF, 50 to 3200 in bins of 50 x 2^k, spacing 0.05) | the fixed `phi_max = 50` truncated the Gil-Pelaez integral far too early near maturity. 2021 S&P fit, ATM call on 3700: 44 vs 50 (10 days), 26 vs 35 (5 days), 6 vs 15 (1 day); OTM prices went negative and the intrinsic clip hid it. Heston doc/con agents traded calls and puts at these prices in the last two weeks: a systematic arbitrage against the exact terminal payoff. Inherited from the original JAX simulator, so the parity test could not see it. complex64 is not the cause (float64 on the same grid gives the same numbers) | `MarketSpec.heston_phi_min / heston_phi_cap / heston_dphi` |
| `check_bank`: one-step drift test `E[e^{-r dt} X_{t+1} - X_t] = 0` at every step, martingale test also at N-10, N-5, N-1 | the unconditional test at N/4, N/2, 3N/4, N averaged the late bias away, and t = N is the exact payoff | n/a |
| DOC premium pooled at t = 0; 2000 inner MC paths (was 500) | every outer path starts from the same (S0, v0) but received a different noisy premium (SE about 20 dollars on 250): pure noise in every P&L and CVaR, including the no-hedge baseline | `MarketSpec.heston_mc_paths` |
| DOC MC normal draws from a CPU generator | a bank is now the same on CPU and GPU (the cache key does not contain the device) | n/a |
| running VaR per asset (= calibration year) in the mean-CVaR penalty | the pooled quantile mixed four premium scales; the test metric is a per-year CVaR | `--cvar-pooled` |
| critic loss MSE (was smooth L1 = TorchRL default) | Huber(1) saturates on the rare terminal targets (20 x the excess loss, in premium units): the critic under-estimates the tail, so early steps under-weight the CVaR term, which is the v1 under-hedging failure mode | `--value-loss huber` |
| `--lmbda` flag and a `lambda` ablation group (BSM call x 3 nets, lambda = 1) | with GAE lambda 0.95 the terminal penalty reaches step t directly with weight 0.95^(N-1-t); earlier steps see it only through the critic | `--lmbda` |
| window LSTM padding before t = 0 repeats the first observation | zero padding after normalisation fed the *mean* observation for the first 7 steps (the old code zero-padded raw features) | `--pad-mode zero` |
| Heston minimum-variance delta baseline `mv_delta` (= P1 + rho xi / S dC/dv, stored in the bank) | the plain model delta P1 ignores the vega term, large at rho = -0.95; a weak reference overstates what the agents achieve | n/a |
| `torch.load(..., weights_only=True)` | checkpoints hold tensors and primitives only; closes the pickle path | n/a |
| parity test: Heston pricer checked against an independent complex128 reference down to the last trading day; pooled premium asserted | the old test compared against the old (equally wrong) code | n/a |

Smoke test covers a Heston call run now (model and MV delta baselines). Submit with
`bash submit_all.sh`, `bash submit_all.sh ablation`, `bash submit_all.sh lambda` or `all`.

## v2.0 (slurm folder `v2_1_5`, results `experiments/results/v2_1_5/`)
Changes after the first full run (v1, archived in `slurm/old_pipeline/liquid_1_5_run1`):

| change | why (v1 evidence) | flag |
|---|---|---|
| reward `mean_cvar`: P&L increments + Rockafellar-Uryasev CVaR95 terminal penalty, gamma 1 | the v1 tracking reward counts a cost paid at t about N - t times; both nets under-hedged and lost to BS delta on CVaR95 | `--reward tracking` (ablation) |
| `band` net: one actor that outputs a target AND a no-trade half-width per instrument | tests whether the double net's gain comes from the extra net or from being able to not trade | `--net band` |
| digital pays 10% of S0 | v1 paid $1 on a 3,700-point index: no hedge was optimal, the experiment was degenerate | `--digital-payout-frac 0` (v1) |
| 2 tuning seeds per Optuna trial (objective = mean) | v1 seed-to-seed spread was larger than the single/double gap | `--tuning-seeds` |
| 10 final seeds, seed-level statistics in `report.py` | path-level CIs only covered test-path noise | `--final-seeds` |
| per-experiment Optuna sampler seed | v1 runs re-used identical start-up points and picked identical LRs | n/a |
| target-KL early stop (0.03) and log-ratio clamp in PPO | v1 single-net DOC seeds blew up beyond no-hedge | `--` (TrainConfig.target_kl) |
| non-finite rollouts skipped (20 in a row stop the phase) | defensive | TrainConfig.max_bad_batches |
| bank checks: finiteness + martingale test of every discounted price | catches mispriced surfaces / Euler-vs-formula bias before training | `bank_checks.json` |
| baselines: model delta (BS or Heston P1) for calls, Whalley-Wilmott band (BSM call, scale tuned on validation) | stronger classical references | n/a |


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
* Heston characteristic-function pricing in complex64, the precision of the original JAX code, with an
  adaptive integration limit per point (v2.1; the fixed limit of the original was wrong near maturity).
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
* The window LSTM reads t-7..t (the first observation is repeated before the start); the recurrent
  LSTM is one-directional; the terminal observation is never fed to a policy.
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
9. ~~`hpc_notes.txt` contains what looks like a password and is staged in git.~~ Checked on 1 Nov 2026:
   it only says where the HPC password is issued; there is no credential in git. The slurm files do
   contain the cluster hostname and the login in the default repo path (override with `DEEP_HEDGING_REPO`).

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
cd deep_hedging_v0/experiments/simulations/slurm/v2_1_5
mkdir -p logs && sbatch smoke_test.slurm      # up to 6 h; check logs/ and the parity output
bash submit_all.sh                            # bank build, then the 18 main runs (one job per run)
bash submit_all.sh ablation                   # 6 tracking-reward ablations;  ... lambda: 3 GAE-lambda-1 runs
```
Outputs go to `deep_hedging_v0/experiments/results/v2_1_5/<run>/`:

| file | contents |
|---|---|
| `summary.json` | best LRs, trial table, seed mean/std of test metrics, baselines |
| `trials/trial_NNN.json` | every Optuna trial (complete, pruned, failed): params, validation curve, per-iteration training log |
| `final_seedN.json` | validation curve, per-iteration training log, test metrics |
| `final_seedN_test_paths.npz` | per test env: P&L, cost, tracking error, hold rate, trade count; `trace_*`: step-by-step holdings, proposed trades, hold decisions and probabilities, costs, portfolio value, liability, prices for the first 100 test paths |
| `best_trial_test_paths.npz` | the same for the best tuning trial |
| `baselines_test_paths.npz` | per-env P&L and cost of no-hedge, model delta (calls), Heston MV delta (Heston call), Whalley-Wilmott band (BSM call) |
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
