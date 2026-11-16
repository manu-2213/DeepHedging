# Training: goals, rules, state and plan

The working record for getting the hedging nets to train well. Update it with every decision and every finished batch of
runs. The paper-quality write-up is `paper_notes/findings.tex` (section "Converging the BSM call band net (v2.3–v2.4)").
HPC operations (rsync, sbatch, monitoring) are in `docs/hpc.md`.

Live curves and the full run table: https://claude.ai/artifact/TsvTrW5uWQUQotQJmCjJ5r
(data from `experiments/analysis/collect_runs.py`, which runs on the HPC and reads every run's log, summary and launch command).

> **Repository layout changed on 2026-11-16** (see the log entry of that date). Entries before it use the old paths.
> Old → new: `deep_hedging_v0/fast_hedging/{market,pricing,features,networks,ppo,model_band}.py` → `hedging_engine/`;
> `fast_hedging/experiment.py` → `experiments/train.py`; `python -m fast_hedging.model_band` → `python -m experiments.baselines`;
> `experiments/simulations/slurm/main/` → `experiments/slurm/`; `experiments/option_parameter_estimation/` →
> `experiments/calibration/` (parameter files in `hedging_engine/data/`); `new.MD` → `docs/hpc.md`; scratchpad scripts
> (`report_runs.py`, `table_now.py`, `eval_*.py`, `make_*figs.py`, `hold_analysis.py`, `shrink_check.py`) →
> `experiments/{analysis,evaluation,figures}/`. Everything older is at git tag `pre-cleanup-2026-11-16`.
> **The HPC copy still has the old layout** until its queue is empty (`docs/hpc.md`).

## What we want (the user's priorities, 2026-11-08)

0. **Band nets are dropped, and tracking error is the only measure** (2026-11-08 22:00). Report transaction cost next to it.
   **Chosen encoder (2026-11-09): fast transformer + GRPO, group 8.** The user expects it to win after tuning and wants state of the art and speed. Decision rule: adopt it if, after tuning and to convergence over seeds,
   its tracking error matches or beats LSTM + PPO on the single and double net. Speed decides ties. It has to be shown first: with GRPO group 8 it is at 15.8 against 9.7–10.0 at 0.1% and 36.4 against 40.6 at 1% (single net, seed 1, still running, 2026-11-09 02:10), so at 1% it now leads; 0.1% it does not yet.
1. **The double net must shine.** The double net (actor + inaction net, "inactor", which decides whether to trade)
   models trading costs directly. It is the main architecture. The band net is secondary.
2. **Every run must converge**, so comparisons are fair. Converged means early stopping ended the run (200 validation
   checks without improvement, after all three rate halvings). It does not mean a time or iteration budget ran out.
3. **The inactor can be hard to train.** Watch its hold rate and the joint phase, not only the final number.
   Failure modes: it holds almost always (never trades), or almost never (it does nothing).
4. **Learning rates matter.** Tune them first, for each net and phase: actor, inactor, and the joint-phase actor.
5. **Use the architecture that does best.** Compare encoders (LSTM, fast transformer) and advantages (PPO, GRPO) on the
   double net itself. Don't carry over the band-net ranking.
6. Afterwards, the same recipe goes to every product (call, down-and-out call, cash-or-nothing) and to the other dynamics.

## Fixed rules

- **Measure: tracking error, and nothing else.** The mean over steps of |PV_t − L_t| on the validation paths, used for selection, rate cuts and
  stopping (`--metric mean_abs_tracking_error`). Always report **transaction cost** next to it. CVaR95 is still written to the logs but is not used or shown.
- **Model-free.** No Whalley–Wilmott, delta or other model output may be used in training: no warm start, no target,
  no "diagnostic" init. WW and delta are only reference scores (WW is tuned on validation tracking error).
- **Compute.** Training runs only on the HPC, as chained one-hour gpushort jobs (exit 75 means resubmit). The cap is 2 GPUs per user.
  Cancel only job IDs I created. Never use `scancel -u`, and never hold jobs from a script that depends on the local SSH connection.
- **Old pipeline:** removed from the working tree on 2026-11-16 at the user's request (clean-up); it is at git tag `pre-cleanup-2026-11-16`. Before that date the rule was that it stays frozen.
- Compare runs at matched iterations while they train, and on test only once they have finished.

## Reference numbers (BSM call, same paths for every run)

| | val tracking 0.1% | val tracking 1% | test tracking 0.1% | test tracking 1% | test cost 0.1% / 1% |
|---|---|---|---|---|---|
| Whalley–Wilmott (tuned on tracking) | **7.42** | **28.45** | 8.36 | 35.48 | 12.7 / 54.2 |
| Model delta, daily | | | 11.26 | 112.3 | 20.4 / 203.6 |
| No hedge | 162.7 | 162.7 | 153.8 | 153.8 | 0 / 0 |

## What we have learned (BSM call)

- **v2.4 beat v2.3 LSTM + PPO on the band net.** The best validation tracking error went from 20.5–41 to about 10.2 (0.1%) and from 43–163
  to about 30 (1%).
- **At 1% the band net matches WW on test tracking error** (LSTM + GRPO 35.5 against 35.48), with 17% lower cost (45 against 54).
  At 0.1% it is still about 1.4 times WW, because it under-trades (about 30 trades per path against 127).
- **Correction (2026-11-08 21:00): the 9.7–10.0 at 0.1% is not a double-net result.** It comes from the *actor phase* of the v2.3
  double-net runs, so it is a single net trading on all 250 steps with no holding (val 9.72 / 9.98 / 9.96). Seed 3 is the only 0.1% run in the joint phase; its best
  joint check (9.86) has hold rate 0.00, and by iteration 3050 it holds 54% with tracking 15.7. There is **no evidence yet that the inactor helps at 0.1%**.
- **LSTM + PPO is the best single-net actor so far, and the newer methods have not beaten it there** (0.1%: 9.7–10.0; 1%: 40.6 converged, seed 1).
  Fast transformer + GRPO single: 18.0 at 0.1%, 43.1 at 1% (both still running, 2026-11-08 23:00). Fast transformer + PPO single: failed at 0.1% (98.6, best at iteration 180, stopped at 3680, test 107.9) and 80.5 at 1% (still running).
  The LSTM + PPO actor was *not* beaten; only its band version was.
- **The inactor collapsed in the only 1% joint run** (seed 1, inactor 1e-3): best check at iteration 130 (40.34, hold 0.78, 54 trades, cost 31.6 against the actor's 55.1), then
  hold 0.99 with 2 trades and tracking 57.7 by iteration 660. So holding did cut cost by 43% at equal tracking for a moment, which is the case for the double net; it must be made stable.
- **Transformer + GRPO has not had a proper go.** It has only used the LSTM's settings (rate 1e-4, group 4). The window transformer and transformer + GRPO band runs were stopped on 2026-11-07 on my recommendation, as "behind on both measures";
  that was wrong on tracking: tfgrpo_fee1_s2 was at 29.89, the best 1% band value of any run, at iteration 3790 when stopped. They were resumed and then cancelled with the other band runs; the single-net `wtgs` runs now test that encoder.
- Fast transformer: about 2 times cheaper per iteration than the LSTM (median 0.51 s against 1.10 s on the single net, 0.65 s against 1.12 s on the band net) and 5 times cheaper than the window transformer (3.3 s). Earlier notes saying 5 times against the LSTM were wrong. It was the best single band run at 0.1% (10.78). Its seed spread is large.
- Settings with no clear gain (band net): rate 3e-5 (slow) or 3e-4 (falls into never trading), agent state, gamma 0.995/1,
  4x batch, 10x data, 128-wide net, window transformer.
- The mean–CVaR reward was dropped when tracking error became the measure.
- **Probe 5 (2026-11-10, 1%, seed 1): none of the double-net training ideas moves test tracking** (36.9–37.7 for all five variants vs 37.48 control). Agent state alone: no change. Inactor starting at 5% holding: overrun within 200 iterations. Hold-dropout actor: same tracking, cost −6% (one seed). Both together: unstable joint phase. The 2,000-path validation set does not change outcomes either.
- **Pitfall:** grepping `track=` over a whole log.txt also matches the baseline line (WW's test tracking). Parse only the `] it ... val` lines.

## Current plan (2026-11-10): the main table

**(call, doc, con) × (bsm, heston) × (0.1%, 1%) × (single, double), 5 seeds each.** Method fixed from the BSM-call work: fast transformer + GRPO group 8, tracking reward, selection on validation tracking error, no agent state, chained 1-hour jobs. Heston at the BSM path counts (2,000 / 500 / 2,000). Generator: `experiments/simulations/slurm/main/make_main.sh`; results `experiments/results/main/`; job names `mn_*`.

1. **Per cell, tune on seed 1:** single actor lr {3e-5, 1e-4, 3e-4}; then from the validation-best single, the double grid inactor {1e-3, 1e-4} × joint {1e-4, 1e-5} (inactor 1e-5 dropped: never learned to hold). Pick on validation only.
2. **Seeds 2–5** with the chosen rates (`seedchain`: each seed's single submits that seed's double on finish).
3. Order: BSM call seeds (submitted 2026-11-10) → BSM doc, con → Heston call, doc, con. Prerequisites in flight: `check_main.slurm` (tiny single + double for each new dynamics × product, CPU) and `bank_heston2000.slurm`.
4. Report on test: tracking error with cost alongside; call has WW/delta references, doc/con have the single net and no hedge.
5. Budget: ~45 GPU-h per cell, ~500 GPU-h in all, ~10 days on the 2-GPU cap if the queue is kept full.

Superseded plan (2026-11-08): single nets to convergence, transformer + GRPO tuning, double grid from `tgsg8`, seeds — all done for the BSM call; see the log.

## Log of decisions

- 2026-11-05: v2.3, 18 BSM call runs (single/band/double × 2 fees × 3 seeds), LSTM + PPO.
- 2026-11-06: v2.4 probes (transformer, GRPO, scaling). v2.3 held.
- 2026-11-07: fast transformer (tfpath) built. WW warm start built and **rejected** (model-free rule).
- 2026-11-07 23:50: **tracking error is the measure**, and costs are reported. `--metric` flag added. CVaR-reward runs cancelled.
- 2026-11-08 morning: rate sweep (1e-4 is best) and agent state (no clear gain) on the band net. Gamma probe started.
- 2026-11-08 13:00: **the double net is the focus.** Learning rates first. Band runs moved to low priority (nice 500).
- 2026-11-08 13:20: stage A submitted (tfpsingle, tfpgrposingle at nice 0; grposingle at nice 50). The six v2.3 double nets
  were released at nice 100. Queue order: stage A, then grposingle, then the v2.3 doubles, then band runs. The v2.3 single
  and band runs stay held. Stage B is submitted by hand when the stage-A singles finish. Pick the source per fee by best
  validation tracking error.
- 2026-11-08 20:50: the HPC connection and the local watcher are down, last status read 13:30. Stage B has not been submitted. To do on reconnect: submit stage B from the finished singles, and make that hand-over run on the HPC side so a dropped connection cannot stall it.
- 2026-11-08 21:15: found the actor-phase misreading above. `lstmsingle` submitted (35610568/9); grid changed; band scripts carry `#SBATCH --nice=500` so resubmissions stay low;
  resumed tf/tfgrpo band chains carry `--nice=200`; 9 tuning runs submitted (35610602–10). 53 jobs in the queue (10 held) on 2 GPUs: expect several days.
- 2026-11-08 21:30: CPU check passed for the path the grid depends on (`check_dbl_from_single.slurm`): `--actor-init from-single`, separate inactor and joint rates, two exit-75 resumes (iteration and best value carried over), final test report. Mechanics only: 40 iterations on 200 paths from a poor actor.
- 2026-11-08 22:00: **band nets dropped.** 24 queued band jobs cancelled (v2.3 band, and every v2.4 band chain including the resumed window-transformer runs; none was running; logs and checkpoints kept). The `run_grpo_*`, `run_tfp_*`, `run_tfpgrpo_*` band scripts stay on disk as templates for `make_double.sh`.
  `run_tfpgrposingle_fee<f>_s1.slurm` now also submits a double-net grid from the fast transformer + GRPO actor when it finishes (a job already running keeps its old copy of the script, so check this hand-over happened).
  The v2.3 single and double runs were cancelled at the user's request at 21:15 (below).
- 2026-11-08 21:15: **12 v2.3 single and double jobs cancelled** (CVaR-selected; logs kept), at the user's request. `lstmsingle` and the grid cell inactor 1e-3 × joint 1e-4 replace them with tracking selection. LSTM + PPO now has seed 1 only until the seed runs.
  Queue re-evaluated, 16 jobs: fast single nets (running), `lstmsingle`, `grposingle`, tuning (`tgs*`), then `wtgs` last (`--nice=300`: at 3.3 s/iter it is the expensive one and speed matters). Each chain gets about one hour per queue cycle while the queue is long.
  Estimate: first LSTM + PPO grid results in about 1.5–2 days, everything in about 3–4 days.
  State of the fast transformer + GRPO single net: 18.0 (0.1%) and 53.9 (1%), all three rate cuts used and still improving at rate 1.25e-5. That the improvement continues only at low rate is why the 3e-5 tuning run matters.
- 2026-11-08 23:00: **fast transformer + GRPO at 1% moved from 53.9 to 43.1** (iteration 22,720; LSTM + PPO is 40.6). 0.1% is 18.0 (iteration 21,970). Both still improving.
  **The rate cuts fire too early.** All three were spent by iteration 2,590 at 1% (4,210 at 0.1%). At 1% the net was then still worse than no hedge (tracking 191 at iteration 3,000; no hedge is 162.7). So about 90% of training has been at the floor rate 1.25e-5. The schedule reflects a noisy start, not a plateau. If `tgslr3e5` also spends its cuts early, it will sit at 3.75e-6 and be slow for the same reason; check that before reading it as a rate result.
  **Per unit of time the transformer is not ahead yet.** At 1%, tf + GRPO reached 73 after about 15,000 iterations (2.2 h at 0.53 s), while the new LSTM + PPO run reached 73 after 3,120 (0.8 h at 0.96 s). At 0.1%, LSTM + PPO is at 13.3 after 2,730 iterations (0.8 h); tf + GRPO has not reached that after 3.2 h. Its speed advantage per iteration (2 times) does not cover its slower learning per iteration so far. This is seed 1, untuned, with the rate schedule above.
  Tuning, early: `tgslr3e5_fee01` 81.9 at iteration 2,110 and `tgslr3e4_fee01` 111.9 at 1,350 (the 1e-4 run had 85.7 at 1,500 and 73.0 at 3,000). No conclusion yet. Fast tf + PPO at 1%: 80.5, still running. No run finished or failed since the 21:15 entry.
  Times on the 22:00 and 21:15 entries above are approximate; the band drop came first.
- 2026-11-08 23:25: **`tgslr3e4_fee01` (fast tf + GRPO single, rate 3e-4, 0.1%) finished and failed.** Best validation tracking 98.5, test 92.1 (cost 120.4, 250 trades), early-stopped at iteration 4,810 with validation tracking at the end (140.9) well above its best. All three cuts used by iteration 2,500. Rate 3e-4 is too high for this method at 0.1%.
  `tgslr3e5_fee01` is ahead of the 1e-4 run so far: 61.8 at iteration 4,680 (the 1e-4 run had 68.3 at 4,500), first cut only at iteration 3,120, so 3e-5 does not spend its cuts in the noisy start.
- 2026-11-09 00:10: **`tgslr3e4_fee1` also failed:** best validation tracking 156.5 (no hedge 162.7), test 140.3, cost 197.8, stopped at iteration 4,310. Rate 3e-4 is out at both fees.
  `tgslr3e5_fee1` is well ahead of the 1e-4 run at matched iterations: 129.8 against 191.1 at iteration 3,000, and 118.9 at 5,000 against 174.9 at 4,500. After its first cut (iteration 2,420) it runs at 1.5e-5, close to the 1e-4 run's floor of 1.25e-5, so the gap comes from the first few thousand iterations: starting at 1e-4 costs a large setback at 1%.
- 2026-11-09 02:10: **GRPO group size is the biggest lever found so far.** `tgsg8` (group 8 × 12 paths, rate 1e-4, otherwise the same as `tfpgrposingle`) has no rate cut yet and reads 15.77 at 0.1% (iteration 4,910, cost about 22) and **36.36 at 1%** (iteration 6,470, cost about 48), while group 4 at the same rate read 68.3 at 0.1% (iteration 4,500) and 160.2 at 1% (iteration 6,000). At 1% this beats LSTM + PPO (40.6). It reached tracking 73 at about iteration 2,850, roughly 0.4 h against 0.8 h for LSTM + PPO, so the speed case for the transformer is back.
  **This corrects the 00:10 reading.** "Starting at 1e-4 costs a large early setback" holds only for group 4: at the same rate 1e-4 group 8 does better than group 4 at either rate. The noise in the group-4 advantages was the likely problem, and a lower rate was only compensating for it.
  Group 4 also moved: `tfpgrposingle` 14.94 at 0.1% (iteration 27,950) and 41.5 at 1% (iteration 23,990).
  Actions: `make_groups.sh` submitted `tgsg16` (16 × 6) and `tgsg32` (32 × 3) at both fees (35616845–8). The double-net grid hand-over moved from `run_tfpgrposingle_*` to `run_tgsg8_*`, source name `tg8`, checked by a dry run of the generated script (`--actor-init from-single --single-dir .../tgsg8_fee<f>_s1 --grpo-group 8 --paths-per-iter 12`). The group-4 grid can still be made by hand. The job for `tgsg8` queued now holds its older copy of the script, so if that exact job finishes the run the grid must be submitted by hand.
- 2026-11-09 04:10: no run finished or failed. **The 0.1% lead of rate 3e-5 over 1e-4 (group 4) has gone:** 39.6 at iteration 10,140 against about 38 for 1e-4 at the same point (36.7 at 10,500). At 1% it is still ahead (83.1 at iteration 9,790 against 120.8–133.1 for 1e-4 between 9,000 and 10,500). Group 4 at 1e-4 reached 37.9 at 1% (iteration 28,530), also below LSTM + PPO's 40.6. New LSTM + PPO single net: 10.5 at 0.1% (iteration 5,880, third cut used) and 57.3 at 1% (iteration 5,780). Group 8, 16 and 32 have had no new turn since 02:10.
- 2026-11-09 05:50: **first converged fast transformer + GRPO single net: `tgsg8_fee1` (group 8, rate 1e-4).** Early stopping ended it at iteration 11,960, two chained jobs. Validation tracking 34.66. **Test tracking 39.15, cost 55.64**, against WW 35.48 and cost 54.22: about 10% above WW's tracking at about the same cost. Trades 250 per path (single net, as designed). Rate cuts used: all three.
  `tgsg8_fee01` is at 10.58 at iteration 11,320 (cost about 13), level with the new LSTM + PPO run (10.52) and close to the old 9.7–10.0.
  The grid hand-over did not fire, as expected: that job held the older script. `make_double.sh B tg8 1` was run by hand and six `dbltg8_*_fee1_s1` jobs submitted. First double-net data from the group-8 actor should arrive after these reach the front of the queue.
- 2026-11-09 07:50: **Group size saturates at 8.** At matched iterations (best validation tracking so far) groups 8, 16 and 32 are the same within noise, and group 4 is far behind. 0.1%, iteration 6,000: 13.5 / 13.4 / 13.6, group 4 58.8. 1%, iteration 6,000: 38.3 / 38.0 / 38.1, group 4 160.2. Speed is also the same (0.47–0.53 s/iter). So group 8 stays the setting and the grid source is unchanged; `tgsg16` and `tgsg32` add no information that changes a decision.
  Group 4 at 0.1%: 13.1 at iteration 33,980. No run finished or failed since 05:50.
- 2026-11-09 09:50: no run finished or failed; the six `dbltg8` jobs have not started. New LSTM + PPO single net: 9.76 at 0.1% (iteration 8,980, cost 10.8), matching the old 9.7–10.0, and 47.1 at 1% (iteration 8,930, cost 57.4). Group 4 fast tf + GRPO: 13.1 at 0.1%, 35.5 at 1% (iteration 34,480, cost 49.4).
- 2026-11-09 10:00: **decision (the user): go with fast transformer + GRPO to save time; convergence of both the single and the double net is very important.** Basis at this point: 1% converged 34.66 validation (test 39.15) against the LSTM + PPO rerun's 47.1, still running; 0.1% 10.58 against 9.76, group 8 still running; about half the cost per iteration. So the setting is fast transformer + GRPO, group 8, rate 1e-4.
  12 jobs cancelled by ID, logs and checkpoints kept: `grposingle` ×2 (never started), `tfpsingle_fee1`, `tgslr3e5` ×2, `tgsstd02_fee01`, `tgsg16` ×2, `tgsg32` ×2, `wtgs` ×2 (never started). These are not converged and are recorded only as far as they went.
  11 jobs remain: six `dbltg8_*_fee1`, `tgsg8_fee01` (starts its own 0.1% grid when it finishes), `lstmsingle` ×2 as the converged reference, and group-4 `tfpgrposingle` ×2. `make_double.sh B lstm` now refuses unless `ALLOW_LSTM_GRID=1`, so the LSTM runs will not launch double-net grids.
  Next: after the grid, seeds 2 and 3 for the group-8 single net and the best double-net rates.
- 2026-11-09 11:15: **first double-net results from the group-8 actor at 1%: the inactor trains and is stable.** Validation, joint phase only (checked in the raw log; the actor is loaded from `tgsg8_fee1_s1`):
  inactor 1e-3 × joint 1e-4: best 29.73 at iteration 3,830, hold 0.87, 33 trades, cost 42.6; at iteration 4,400 it is 29.99, hold 0.87 (0.867–0.874 over the last 50 checks), bad 57/200, all three cuts used. Inactor 1e-4 × joint 1e-4: 30.76 at iteration 1,590, hold 0.86. Inactor 1e-3 × joint 1e-5: 30.73 at 1,500, hold 0.88, but less steady (46.2 at iteration 1,540, then 38.1).
  Against: single net from the same actor 34.66 (cost 47.7), WW 28.45 on validation (WW is tuned on those paths). So the double net cuts tracking by about 14% and cost by about 11% relative to its own single-net actor, and is within about 5% of WW. The LSTM inactor collapse at rate 1e-3 did not recur here. None of these has converged or been tested yet. Inactor 1e-5 runs have not started.
  `tgsg8_fee01`: 10.23 at iteration 17,920, all cuts used.
- 2026-11-09 11:30: **overfitting check (the user asked).** All tracking numbers quoted during training are validation (500 fresh paths of the training years); test is 2,000 paths of a held-out calibration year, so it also differs in market parameters. Converged `tgsg8_fee1`: validation 34.66, test 39.15 (ratio 1.13). WW, one fitted parameter, so it cannot overfit: 28.45 to 35.48 (1.25). The net's gap to WW shrinks from 22% on validation to 10% on test, so there is no sign of overfitting in this run; the rise is the harder test year. Training-path error (47.6, last 200 checks) is above validation (35.0), but it includes exploration noise, so it only shows the net is not fitting the training paths better. Best-checkpoint selection bias is small (best 34.66, mean of last 200 checks 35.04). The double net has no test result yet; it must be judged on test, since the grid and many checks select on the same 500 validation paths.
- 2026-11-09 13:50: **`tfpgrposingle_fee1` (group 4) converged:** iteration 37,190, validation 35.29, **test 40.62, cost 58.6**. Group 8 converged at iteration 11,960 with test 39.15 and cost 55.6, so group 8 is better on tracking and cost at about a third of the iterations. No hand-over grid launched from group 4, as intended.
  Double-net grid at 1% (validation, best so far): i1e3 j1e4 29.73; i1e3 j1e5 29.83; i1e4 j1e5 30.02; i1e4 j1e4 30.13 (holds 0.86–0.87, 32–36 trades, cost about 41–42). Inactor 1e-5 is clearly weaker on tracking: 34.19 (j1e5) and 34.60 (j1e4), holds 0.92–0.94, 14–20 trades, cost about 26. So inactor rate 1e-3 or 1e-4 is the range that lowers tracking; 1e-5 mostly cuts cost at the single net's tracking error. None has converged.
  `lstmsingle`: 9.72 at 0.1% (bad 150/200), 46.0 at 1%. Group 4 at 0.1%: 12.16.
- 2026-11-09 14:10: **`lstmsingle_fee01` converged:** iteration 11,280, validation 9.72, **test 10.45, cost 14.0** (WW 8.36, cost 12.7; 25% above WW on tracking). No LSTM grid launched, as intended (`make_double.sh` guard). `tgsg8_fee01` (0.1%) is at 10.20, still running. `lstmsingle_fee1` 44.97, still running.
- 2026-11-09 15:00: **first converged double net (1%, inactor 1e-3, joint 1e-4): test tracking 37.47, cost 57.67, 34.8 trades, hold 0.86.** Validation was 29.73 (iteration 3,830; stopped at 5,830).
  On test it improves on its own single net by only 4% on tracking (39.15) and costs 4% **more** (55.64). On validation the same comparison was 14% better tracking and 11% lower cost, so the double net's advantage over the single net largely did not carry to the held-out year. Against WW the gap is about the same on both sets: tracking 4.5% above on validation, 5.6% above on test (37.47 against 35.48); cost 6% above on test (57.67 against 54.22).
  Reading: test/validation tracking is 1.26 for this net, 1.25 for WW and 1.13 for the single net. The two strategies that hold (no trade on most steps) degrade alike in the test year, so this looks like a held-out-year effect on no-trade strategies at least as much as overfitting; 500 validation paths with heavy selection will also flatter it. Not settled by one run. One seed, one grid cell.
  To do: compare the remaining grid cells on test once converged, pick the cell on validation only, then seeds 2 and 3.
- 2026-11-09 15:40: second converged double net at 1% (inactor 1e-4, joint 1e-4): validation 30.13, **test 37.31, cost 56.08**, 34.5 trades, hold 0.86. Against the first (validation 29.73, test 37.47, cost 57.67) the validation order and the test order are reversed, so differences of this size are noise. Against the single net (39.15, 55.64): tracking 4.7% better, cost 0.8% higher. Against WW (35.48, 54.22): 5.2% and 3.4% above.
- 2026-11-09 15:50: two more double nets converged at 1%. Inactor 1e-3 × joint 1e-5: validation 29.53, test 37.57, cost 57.88, 37.8 trades. **Inactor 1e-5 × joint 1e-4: validation 34.60, test 37.97, cost 46.52, 68.1 trades, hold 0.73.**
  Four cells converged: test tracking 37.31–37.97 (1.8% spread) while validation spans 29.5–34.6. **Validation tracking does not separate these cells the way test does.** The 1e-5 inactor is worst on validation by far, level on test tracking, and 17–20% lower on test cost than the others (46.5 against 56.1–57.9), 14% below WW's cost (54.22) and 16% below the single net's (55.64), at 7% above WW and 3% below the single net on tracking.
  Do not pick the cell using test. By the validation rule the pick would be inactor 1e-3 × joint 1e-5 (29.53). The likely fix is a larger validation set than 500 paths, so that selection means more.
- 2026-11-09 16:45: inactor 1e-5 × joint 1e-5 converged: validation 34.16, test 37.61, cost 46.69, 89.5 trades, hold 0.64. Five of six cells done.
  **Correction on the inactor 1e-5 runs: they are not stable.** The raw log shows the inactor drifting toward always holding: both have their best check near iteration 1,860 (hold 0.72–0.78, cost about 35.5, tracking 34.2–34.6) and by the end (iterations 3,780–3,850) hold is 0.94–0.96, cost 24 and tracking 41.8. The low test cost comes from a checkpoint taken mid-drift, so the run ended by early stopping on a transient, not a plateau. The earlier note that this rate "mostly cuts cost at the single net's tracking" quoted the last checks (hold 0.92–0.94), not the selected checkpoint. Inactor 1e-3 and 1e-4 runs are stable (hold 0.855–0.873 over the last 50 checks).
  Current best validation: inactor 1e-4 × joint 1e-5 at 29.16 (iteration 6,050, still running).
- 2026-11-09 18:00: **the 1% double-net grid is complete (6 of 6).** Last cell, inactor 1e-4 × joint 1e-5: validation 29.18, test 37.28, cost 57.30, 38.7 trades. It is the validation pick and also the best on test, by a margin within noise. The four stable cells: test 37.28–37.57, cost 56.1–57.9, against the single net 39.15 / 55.6 and WW 35.48 / 54.2.
- 2026-11-09 18:00: **probe 5 prepared (the user said to run the two training ideas).** New flag `--actor-hold-dropout p` (ppo.py, experiment.py): in actor-phase training rollouts each market path draws a hold probability in [0, p] and the actor's trade is skipped at random steps; the mask is shared by a path's GRPO copies; skipped steps are masked in the actor loss; validation is unchanged. Off by default, and with it off no extra random numbers are drawn, so existing runs are unaffected. The earlier runs all use `--no-agent-state`, so the actor never saw its holdings; forced holds need that input on.
  `make_probe5.sh`, 1%, seed 1, group 8, all with `--val-paths 2000`: `p5ctl` (control, double from `tgsg8_fee1`, i1e4 j1e4), `p5init` (+ `--inactor-init-hold-prob 0.05`), `p5sstate` (single with agent state) -> `p5dstate`, `p5shold` (agent state + hold dropout 0.95) -> `p5dhold`, `p5dholdinit`. Waiting on `check_probe5.slurm` (CPU check, job 35681684) and `bank_val2000.slurm` (35681747) before submitting.
- 2026-11-09 18:20: `check_probe5` passed (forced holds active, mean training hold rate 0.48; inactor at 5% init gives hold 0.00 in deterministic validation; chained resume and from-single with agent state work). Validation bank of 2,000 paths built. Submitted `p5ctl`, `p5init`, `p5sstate`, `p5shold`; the two single nets submit `p5dstate`, `p5dhold`, `p5dholdinit` when they finish. Judge on test; compare `p5init` with `p5ctl`, `p5dhold` with `p5dstate`, and `p5dstate` with `p5ctl`.
- 2026-11-09 22:00: **`tgsg8_fee01` (fast transformer + GRPO, group 8, 0.1%) converged:** validation 9.58 (iteration 36,310), test 9.67, cost 13.98. Against LSTM + PPO (test 10.45 / 14.00) it is 7% better on tracking at equal cost; against WW (8.36 / 12.7) it is 16% above. So the chosen method now beats LSTM + PPO on test at both fees. The hand-over worked: the six-cell 0.1% double-net grid (`dbltg8_*_fee01_s1`) submitted itself. Queue is now 12 jobs on 2 GPUs; probe 5 was submitted earlier so it keeps priority.
  Probe 5 after one hour each (validation on 2,000 paths, so levels are not directly comparable to the 500-path runs): singles with agent state `p5sstate` 39.7 and `p5shold` 36.9 at iteration ~4,500, against 44.3 for `tgsg8_fee1` at the same iteration; doubles `p5ctl` 31.1 (hold 0.84) and `p5init` 31.8 (hold 0.85) at joint iteration ~2,400. The 5%-start inactor climbs to the same hold rate as the neutral start. 2,000 validation paths slow the double nets to 1.2 s/it (from 0.9).
- 2026-11-09 22:20: **`p5ctl` converged** (control double net, i1e4 j1e4, 2,000 validation paths): validation 31.05, test 37.48, cost 55.90, 42.5 trades, hold 0.83, joint iteration 4,610. The same cell with 500 validation paths gave test 37.31 / 56.08, so the larger validation set did not change the outcome; test/val ratio 1.21 (was 1.24). This is the reference for the other probe-5 runs.
- 2026-11-09 23:00: **`p5init` converged (idea 2, inactor starts at 5% holding): no gain.** Validation 31.85, test 37.69, cost 53.97, 41.4 trades, hold 0.83, against the control 37.48 / 55.90 / 0.83. The log shows the inactor at hold 0.88 by joint iteration 210 (with a transient to validation 55.7 on the way), so the starting point is overrun within 200 iterations at inactor rate 1e-4 and the two runs end at the same policy. Idea 2 is dropped. Remaining: `p5sstate` (val 35.5 at it 6,350, still improving) and `p5shold` (36.4 at it 4,550), then their double nets.
- 2026-11-10 00:10: **`lstmsingle_fee1` (LSTM + PPO, 1%) converged:** validation 39.89 (iteration 20,510), test 51.6, cost 84.75. Test/val ratio 1.29, far worse than the transformer's 1.13, so on test the chosen method (39.15 / 55.6) is 24% better on tracking and 34% cheaper. Both LSTM + PPO references are now done (0.1%: 10.45 / 14.00). Only `tfpgrposingle_fee01` (group 4) is still running of the old singles (10.44 at it 57,120).
  Probe 5 singles: `p5sstate` 35.41 at it 8,190 (all three cuts used, patience counter 82/200, so it will stop near 35.4); `p5shold` 34.07 at it 7,480 with only one cut used and still improving — already below the no-agent-state single (34.66 on 500 paths) despite training with ~half its steps forced to hold. 0.1% double grid: first two cells running; i1e3 j1e4 at joint it 220 holds 27% with validation 12.1 against the single net's 9.58.
- 2026-11-10 02:10: **0.1% double grid, first evidence the inactor helps at low fees.** After ~3,400 joint iterations: i1e3 j1e4 val 9.14 (hold 0.32, cost 10.0), i1e3 j1e5 9.13 (hold 0.30, cost 10.2), i1e4 j1e4 9.33 (hold 0.28, cost 10.8), against the single net 9.58 / 11.8. Unlike at 1% (hold 0.85) the inactor settles near 30% holding. Not converged; test numbers decide. Probe singles unchanged (queued behind the grid's GPU slots).
- 2026-11-10 03:00: **`p5sstate` converged (single, agent state on, 2,000 val paths):** validation 35.41 (it 9,370), test 39.56, cost 57.75, against 39.15 / 55.64 for the single net without agent state. Agent state alone changes nothing for the single net. `p5dstate` submitted itself. `p5shold` (agent state + forced holds) at 33.62 (it 8,900), still improving, below both singles.
- 2026-11-10 03:40: **`tfpgrposingle_fee01` (group 4, 0.1%) converged:** validation 10.44 (it 57,940), test 9.98, cost 14.94, against group 8 9.67 / 13.98 at it 36,310. Group 8 is better at both fees on test with fewer iterations. All six single-net references (LSTM + PPO, group 4, group 8; two fees) are converged.
  Single-net test table, seed 1 — 0.1%: group 8 9.67/13.98, group 4 9.98/14.94, LSTM + PPO 10.45/14.00, WW 8.36/12.7. 1%: group 8 39.15/55.64, group 4 40.62/58.60, LSTM + PPO 51.6/84.75, WW 35.48/54.2.
- 2026-11-10 04:40: **first 0.1% double net converged** (`dbltg8_i1e3_j1e5_fee01`): validation 9.13 (joint it 5,020), test 9.46, cost 13.32, hold 0.23, 192 trades, against the single net 9.67 / 13.98 and WW 8.36 / 12.7. Tracking 2% better and cost 5% lower than the single; 13% above WW on tracking. The hold rate is stable at 0.29–0.31 through the joint phase (no drift). Test/val ratio 1.04 at 0.1%, against 1.21–1.26 at 1%.
- 2026-11-10 05:40: `dbltg8_i1e5_j1e5_fee01` converged: validation 9.50, test 9.73, cost 14.22, hold 0.08, 230 trades — no gain over the single net (9.67 / 13.98). At inactor rate 1e-5 the hold rate only reaches 0.12 on validation in 5,000 joint iterations, so the inaction net barely learns; the 1e-3 cell (test 9.46 / 13.32, hold 0.23) remains the one that helps. `p5dstate` began its joint phase (val 33.72 at it 260, hold 0.77).
- 2026-11-10 07:30: **`dbltg8_i1e4_j1e5_fee01` converged, best 0.1% result so far:** validation 9.11 (joint it 7,360; current validation pick), test 9.23, cost 12.67, hold 0.35, 162 trades. Against the single net (9.67 / 13.98): 4.6% better tracking, 9% cheaper. Against WW (8.36 / 12.7): same cost, tracking 10% above. Hold rate stable at 0.39–0.41 on validation. 0.1% grid: 3 of 6 done (i1e3 j1e5 9.46 / 13.32; i1e5 j1e5 9.73 / 14.22). `p5shold` improved again to 32.93 at it 15,440 (counter 187/200). `p5dstate` at 31.16 (joint it 2,320, hold 0.86), tracking the control.
- 2026-11-10 08:30: **`p5shold` converged (single, agent state + hold dropout 0.95):** validation 32.93 (it 15,570; best single-net validation at 1%), but test 39.78, cost 62.82, against 39.15 / 55.64 for the plain single net. Test/val ratio 1.21 against 1.13 for the plain single; the validation gain does not transfer and the test cost is 13% higher. The single-net test (trading every step) is not the target of this actor, however; the joint phase decides. `p5dhold` and `p5dholdinit` submitted themselves. `p5dstate` at 30.74 (joint it 3,750, hold 0.86), on the control's trajectory.
- 2026-11-10 11:20 (SSH master dropped with the previous session ~09:00; re-established by the user): three runs converged meanwhile.
  **0.1% grid, 5 of 6 done.** `i1e3 j1e4`: validation 8.80 (new validation pick, joint it 13,710), test 9.24, cost 12.95, hold 0.37, 157 trades. `i1e4 j1e4`: validation 8.96, test 9.40, cost 13.06, hold 0.29. With `i1e4 j1e5` (9.23 / 12.67) and `i1e3 j1e5` (9.46 / 13.32), the four learning-inactor cells span test 9.23–9.46 while validation spans 8.80–9.13 — as at 1%, validation does not order them on test. Validation pick on test: 9.24 / 12.95, i.e. 4.4% better tracking and 7% cheaper than the single net (9.67 / 13.98), 10.5% above WW on tracking at 2% more cost. Last cell `i1e5 j1e4` still running (9.33 at it 18,490, hold 0.16, slow).
  **`p5dholdinit` converged (ideas 1 + 2 together):** validation 31.68, test 37.19, cost 55.58, hold 0.38, 155 trades, against the control 37.48 / 55.90 / hold 0.83. Marginally the best 1% double-net test tracking, within noise, at equal cost — but it reached it by a very different policy (holding 38% of steps instead of 85%) and **the joint phase was unstable**: validation swung between 33 and 67 and the hold rate between 0.00 and 0.87 from one check to the next; early stopping fired at it 3,650 and the selected checkpoint is one of the swings. Not a usable setting as is. `p5dhold` (idea 1 alone) at best 31.17 (it 4,100, hold 0.82, counter 172/200), about to stop; `p5dstate` at 30.65 (it 8,270, hold 0.85), the control's trajectory.
- 2026-11-10 12:30: **main table started (user: all experiments, lr tuned per cell, 5 seeds).** Wrote `slurm/main/make_main.sh`, `check_main.slurm`, `bank_heston2000.slurm`; all banks for the current code exist at the old sizes (BSM 2000/500/2000, Heston 1000/250/1000); the Heston 2,000-path banks are being built (job 35712552, the 1,000-path sets took 28–96 s each on 2026-11-02). Check job 35712551. Submitted BSM call seeds 2–5 singles at both fees (jobs 35712553–60); their doubles (1%: i1e-4 × j1e-5; 0.1%: i1e-3 × j1e-4) follow automatically. Probe 5 stays out of the main recipe.
- 2026-11-10 13:10: **compute options.** The account's `sacctmgr` associations list GPU quotas on `andrena` (8), `apini` (8), `sae` (12), `gpu` (4), `bciwiph` (1), but test submissions to all five fail with "invalid account/partition combination" — access would need an admin request; the 10-day `gpu` partition had 77 pending / 11 running. Submitted `bench_share.slurm` (job 35713224: the main-table single net 1, 2 and 4 at a time on one GPU, with utilisation sampling) and `bench_cpu.slurm` (35713225: same net on 8 CPU cores). The 8 BSM-call seed singles (35712553–60) are **held** so the benchmark takes the next slot; `bench_share.slurm` releases them when it starts (`scontrol release` at its top). If the benchmark never runs, release them by hand.
- 2026-11-10 13:30: **user decision: run the main table on the HPC with the 1-hour chained jobs for now**; cloud GPUs (RunPod/Lambda, ~$50–700 depending on sharing) and an admin request for other partitions are parked ("we will get back to this"). The sharing benchmark still runs: if several trainings fit on one GPU, the main-table jobs will pack several chains per 1-hour job.
- 2026-11-10 14:10: **`sae` partition works with `--account=pilot_sae_gpu`** (the user pointed to the v2_1_5 scripts). A 5-minute test started at once on an H100 node despite ~700 pending jobs there: short 1-GPU jobs backfill. Quota on `sae` is 12 GPUs, so the 2-GPU `gpushort` cap no longer binds. `make_main.sh` now emits `--partition=gpushort,sae --account=pilot_sae_gpu`; the 8 held BSM-call seed singles were cancelled and resubmitted (35713873–80); three started immediately on `xdg1` (H100). CPU benchmark: 3.96 s/it on 8 cores against ~0.5 on a GPU — CPU training is not worth it. `make_main.sh` also gained `tune` (the three lr singles) and `pick` (each tuning single calls it on finish; once all three are done it picks the validation-best and submits that cell's double grid, once).
  **`p5dhold` converged (idea 1 alone: hold-dropout actor, agent state, neutral inactor start):** validation 31.17, test 37.00, cost 52.58, hold 0.79, 52 trades, against the control 37.48 / 55.90 / 0.83. Stable joint phase (hold 0.79–0.84), unlike `p5dholdinit`. Tracking 1.3% better (within noise), cost 6% lower. `p5dstate` (agent state alone) still running; it decides whether the cost gain comes from the agent state or the forced holds.
  **0.1% double grid complete (6 of 6):** `i1e5 j1e4` validation 9.31, test 9.83, cost 14.53, hold 0.10 — no gain, as the other 1e-5 cell. Validation pick stays `i1e3 j1e4` (8.80; test 9.24 / 12.95).
- 2026-11-10 14:40: machinery check passed for BSM doc, BSM con, Heston call, Heston doc (Heston con still running); Heston 2,000-path banks built (10 min). **Submitted 24 tuning singles** (`make_main.sh tune`): {bsm doc, bsm con, heston call, heston doc} × {0.1%, 1%} × actor lr {3e-5, 1e-4, 3e-4}, seed 1; each calls `pick` on finish, which submits the cell's double grid from the validation-best single once all three are done. BSM-call seed singles: 4 running on `sae` (3 on an H100 node at ~0.27 s/it, one on an A100), 4 pending. Queue: 28 pending on gpushort,sae; sae quota 12.
- 2026-11-10 15:30: **check passed for all six (dynamics × product) cells**; Heston con tuning submitted (6 singles). **GPU-sharing benchmark** (job 35713224, V100-32GB node): 1 process 0.70 s/it; 2 processes 1.12 s/it each (throughput ×1.25); 4 processes 2.08 s/it each (×1.35); utilisation of one process peaks at 90–95%. Not worth packing, especially with `sae` available — one run per job stays. Queue: 9 running on sae, 2 on gpushort, ~28 pending.
- 2026-11-10 16:10: **`p5dstate` converged — probe 5 complete.** Agent state alone: validation 30.41, test 36.87, cost 57.63, hold 0.84, 41 trades (joint it 13,170). Probe 5 on test, 1%, seed 1, all double nets i1e-4 × j1e-4 with 2,000 validation paths:

  | Run | Test tracking | Test cost | Hold | Joint phase |
  |---|---|---|---|---|
  | control | 37.48 | 55.90 | 0.83 | stable |
  | + inactor starts at 5% holding (idea 2) | 37.69 | 53.97 | 0.83 | stable; start overrun by it 210 |
  | + agent state | 36.87 | 57.63 | 0.84 | stable |
  | + agent state + hold-dropout actor (idea 1) | 37.00 | 52.58 | 0.79 | stable |
  | + agent state + hold dropout + 5% start | 37.19 | 55.58 | 0.38 | unstable (val 33↔67) |

  Tracking spans 36.9–37.7 (2%), single seed: noise. Cost spans 52.6–57.6; the hold-dropout actor is the cheapest (−6% vs control, −9% vs agent state alone) at equal tracking — the only consistent signal, but one seed. **Decision: the main recipe stays (no agent state, no hold dropout); probe 5 closed.** Hold dropout is worth a seeded follow-up for the cost dimension later, not now.
- 2026-11-10 14:30 GMT: **main table progress.** 12 running (10 on sae), 23 pending, 0 failures. First tuning singles converged: `bsm_con_fee1_lr3e4` (22.3), `bsm_doc_fee1_lr3e4` (25.5), `bsm_con_fee01_lr1e4` (55.8 — stuck; early stopping fired on a bad plateau). Standings (best validation so far): BSM con 0.1%: 3e-4 9.80, 1e-4 55.8, 3e-5 53.5. BSM con 1%: 1e-4 19.3, 3e-4 22.3, 3e-5 37.4. BSM doc 0.1%: 3e-4 23.16 ≈ 1e-4 23.27, 3e-5 60.0 (stuck). BSM doc 1%: 3e-4 25.5, 1e-4 29.3, 3e-5 78. Heston call 0.1%: 3e-4 24.4, 1e-4 24.9, 3e-5 31.3. Heston call 1%: 3e-4 37.7, 1e-4 38.9, 3e-5 46.5. Heston doc 0.1%: 3e-4 7.65, 1e-4 15.5, 3e-5 44. Heston doc 1%: 1e-4 37.6 (others early). BSM call seeds 2–5: 0.1% at 10.1–11.6, 1% at 34.3–39.4, still running. **3e-4 wins or ties in most new cells (it failed on the BSM call), so the tuning set is extended to {3e-5, 1e-4, 3e-4, 1e-3}**: 10 `lr1e-3` singles submitted for the five new cells × 2 fees; `pick` now waits for four rates. Speed on H100 0.22–0.30 s/it.
- 2026-11-10 15:30 GMT: first converged tuning singles with test numbers (seed 1): Heston call 1% lr 3e-4: val 37.71, test 49.44, cost 68.9 (test/val 1.31; no hedge 157.5). BSM con 1% lr 3e-4: val 22.3, test 20.5, cost 37.2 (no hedge 93.3). BSM doc 1% lr 3e-4: val 25.45, test 13.65, cost 13.4 (no hedge 158.6; the held-out year is much easier for the barrier claim). BSM con 0.1% lr 1e-4: stuck at 55.8 / test 58.1. Test/validation ratios differ a lot by product and dynamics, so cells are only compared within themselves. 10 running, 34 pending, no failures, no `pick` fired yet.
- 2026-11-10 16:10 GMT: **dashboard extended for the main table** (user: curves per configuration and a preliminary final table, to judge convergence). New sections: "Main table: where every cell stands" (one row per cell × fee: tuning progress, single pick with val/test/cost, rate cuts used, best-check iteration vs run length, seeds done; the same for the double grid; WW and no-hedge test) and "Learning curves per cell" (12 small panels, colour = configuration, line style = seed 1–5, dot = best check, bar = converged). Reading convergence: a converged run always stops 2,000 iterations after its best check, so "best near the end" is normal; a best check very early (e.g. `bsm_con_fee01_lr1e4`: best at 750 of 3,740) marks a stuck policy. Seed-1 BSM call rows come from the v2.4 runs (`tgsg8`, `dbltg8`).
- 2026-11-10 16:25 GMT: 20 main runs done, 0 failures, 11 running, ~20 pending. BSM call 1% seeds converged: s2 32.84, s3 36.44, s5 34.04 (s4 at 32.81, running); 0.1% seeds at 9.67–10.65, running. **Rate 1e-3 now leads in 8 of the 10 new cells** (best val, seed 1): BSM con 0.1% 7.43 (3e-4: 9.64); BSM con 1% 17.8 (1e-4: 19.3); BSM doc 0.1% 11.8 (1e-4: 15.0); BSM doc 1% 41.7 (3e-4: 25.5 — the exception); Heston call 0.1% 23.47 (3e-4: 23.66); Heston call 1% 36.8 (3e-4: 37.7); Heston con 0.1% 5.99 (3e-5, 1e-4, 3e-4 all stuck at 52.8–54.1); Heston con 1% 17.6 (3e-4: 18.2); Heston doc 0.1% 5.16 (3e-4: 5.69); Heston doc 1% 19.2 at 3e-4 (1e-3 at 27.0, early). So the two-instrument products and Heston want much higher actor rates than the BSM call (where 3e-4 failed). **Tuning set extended to {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}**; 10 `lr3e-3` singles submitted; `pick` waits for five.
- 2026-11-10 16:30 GMT: first `pick` fired: Heston call 1% → single `lr1e-3` (val 36.79), 4-cell double grid submitted — before the 3e-3 run exists. If 3e-3 wins that cell, rerun its grid (`make_main.sh grid heston call 1 <run>` after removing `picked_heston_call_fee1`).
- 2026-11-10 22:20 GMT (SSH was down ~17:00–22:00; the cluster ran on): **96 main runs done, 0 failures; the queue drained.** All tuning singles done except two slow 3e-5 runs (bsm con 0.1%, heston doc 0.1%; both far behind their cell's best, finishing). Eight `pick`s fired and all eight 4-cell double grids are converged. **BSM call: all 5 seeds done for single and double**, both fees.
  Single picks (val): bsm doc 0.1% 1e-3 (7.98); bsm doc 1% 3e-4 (25.45); bsm con 0.1% 3e-3 (5.26, pick pending); bsm con 1% 1e-3 (17.80); heston call 0.1% 3e-3 (23.04); heston call 1% 1e-3 (36.76; 3e-3 came later at 36.25 — stale pick); heston doc 0.1% 3e-3 (4.94, pending); heston doc 1% 3e-4 (17.75); heston con 0.1% 3e-3 (5.11); heston con 1% 3e-3 (16.77). **3e-3 is best in 6 cells — edge again; 1e-2 singles submitted for those six** (heston call 0.1%/1%, heston con 0.1%/1%, bsm con 0.1%, heston doc 0.1%). If 1e-2 wins a cell, its grid is rerun.
  Double grid picks (val) and test gains vs the single pick (test track / cost): bsm doc 0.1% i1e-3 j1e-5: 3.78/0.7 vs 3.78/0.9. bsm doc 1% i1e-3 j1e-5: 13.55/11.3 vs 13.65/13.4 (i1e-3 j1e-4 has test 12.74). bsm con 1% i1e-3 j1e-4: **14.60/24.9 vs 19.81/32.7 (−26% / −24%)**. heston call 0.1% i1e-3 j1e-4: 28.84/16.8 vs 28.85/18.6. heston call 1% i1e-3 j1e-4: 42.78/54.0 vs 44.72/59.8. heston doc 1% i1e-3 j1e-4: **28.70/26.0 vs 35.00/38.1 (−18% / −32%)**. heston con 0.1%: all four 4.95–4.99, test 6.1–6.25 vs 6.06/4.6 (tie). heston con 1% i1e-3 j1e-4: **17.79/22.7 vs 21.98/35.2 (−19% / −35%)**. The double net shines at 1% on the exotic products; inactor 1e-3 is the pick in 7 of 8 grids.
  BSM call 5 seeds (test track / cost): 0.1% single 9.67, 9.84, 10.19, 9.82, 10.09 (mean 9.92 / 14.6); double i1e-3 j1e-4 9.24, 9.51, 10.95, 9.18, 9.50 (mean 9.68 / 12.9; seed 3 held 54% and tracked worse). 1% single 39.15, 38.63, 40.68, 39.59, 39.87 (mean 39.58 / 57.3); double i1e-4 j1e-5 37.28, 37.76, 38.35, 37.15, 38.23 (mean 37.75 / 57.9). WW 8.36 / 12.7 and 35.48 / 54.2.
  **Seeds 2–5 submitted** (16 seedchains) for the four cells whose pick is settled and not at the rate edge: bsm doc 0.1% (1e-3; i1e-3 j1e-5), bsm doc 1% (3e-4; i1e-3 j1e-5), bsm con 1% (1e-3; i1e-3 j1e-4), heston doc 1% (3e-4; i1e-3 j1e-4). The six edge cells wait for 1e-2.
- 2026-11-10 23:10 GMT: **rate 1e-2 fails**: heston con 1% val 105.6 (test 126.6) against 16.77 at 3e-3; heston call 0.1% 25.3 and heston con 0.1% 7.81 at ~9,000 iterations, behind 3e-3 (23.04, 5.11). So 3e-3 is an interior optimum. Seeds 2–5 submitted for heston con 1% (single 3e-3; double i1e-3 × j1e-4); the other edge cells follow as their 1e-2 run converges. 14 running, 9 pending.
- 2026-11-11 09:10 GMT: **overnight: 152 → all queued work done, 0 failures; queue empty at 09:00.** Five cells now have all 5 seeds for single and double (bsm call both fees, bsm doc both fees, bsm con 1%, heston doc 1%, heston con 1%). Rate 1e-2 lost in every cell but one: heston call 0.1% 24.8 (3e-3: 23.04), heston con 0.1% 6.15 (5.11), heston doc 0.1% 5.29 (4.94), heston call 1% 293 (diverged), **bsm con 0.1% 5.17 vs 5.26 at 3e-3 — 1.7% better**. Decision: keep 3e-3 for bsm con 0.1% (interior optimum; the margin is within seed noise — seeds of one setting spread 3–10%; 1e-2 diverged in two cells). Deviation from strict validation-best, recorded here.
  Grids of the last two tuned cells: bsm con 0.1% all four within 5.20–5.24 (pick i1e-4 j1e-5; test hold 0.00 — at 0.1% the digital's inactor never holds); heston doc 0.1% 4.90–4.93 (pick i1e-3 j1e-5). **Heston call 1%: the stale pick (1e-3, 36.76) is replaced by 3e-3 (36.25)** per protocol: grid rerun from `heston_call_fee1_single_lr3e3_s1` (4 doubles; old marker kept as `picked_heston_call_fee1_stale_lr1e3`), seed singles at 3e-3 submitted now, seed doubles after the new grid's pick.
  Submitted: 16 seedchains (heston call 0.1%: 3e-3, i1e-3 j1e-4; heston doc 0.1%: 3e-3, i1e-3 j1e-5; heston con 0.1%: 3e-3, i1e-4 j1e-4; bsm con 0.1%: 3e-3, i1e-4 j1e-5) + 4 grid doubles + 4 singles = 24 jobs. Remaining after these: 4 heston call 1% seed doubles. Expected finish: early afternoon.
- 2026-11-11 09:30 GMT: **five-seed results for the seven finished cells (test, mean ± sd over 5 seeds):**

  | Cell | Single track / cost | Double track / cost (hold) | WW / no hedge |
  |---|---|---|---|
  | BSM call 0.1% | 9.92 ± 0.21 / 14.6 ± 0.4 | 9.68 ± 0.73 / 12.9 ± 0.5 (0.39) | 8.36 / 12.7 · 153.8 |
  | BSM call 1% | 39.6 ± 0.8 / 57.3 ± 2.0 | 37.8 ± 0.5 / 57.9 ± 2.7 (0.79) | 35.5 / 54.2 · 153.8 |
  | BSM doc 0.1% | 3.81 ± 0.26 / 0.85 ± 0.17 | 3.85 ± 0.32 / 0.81 ± 0.18 (0.18) | – · 158.6 |
  | BSM doc 1% | 13.68 ± 0.15 / 13.6 ± 0.3 | 13.59 ± 0.27 / 11.4 ± 0.3 (0.94) | – · 158.6 |
  | BSM con 1% | 22.6 ± 7.6 / 32.8 ± 1.3 | 18.8 ± 9.2 / 24.6 ± 2.3 (0.77) | – · 93.3 |
  | Heston doc 1% | 35.9 ± 1.2 / 41.9 ± 4.7 | 29.8 ± 1.1 / 26.6 ± 1.0 (0.97) | – · 186.2 |
  | Heston con 1% | 38.0 ± 20.4 / 57.9 ± 34.5 | 29.0 ± 16.5 / 27.8 ± 16.1 (0.89) | – · 84.6 |

  **Seed failures in the cash-or-nothing cells at 1%.** BSM con: single seed 5 stuck at val 42.4 (others 16.8–19.8); its double inherits it (41.6). Heston con: single seeds 3 and 5 stuck at 41.8 and 50.9 (others 15.9–18.9); doubles inherit (18.4, 48.4). All failed runs used their 3 cuts early and stopped at 5–9k iterations: the high actor rate (1e-3 / 3e-3) that is needed to leave the no-trade plateau is unstable for some seeds. Without the failed seeds: BSM con 1% single 19.2, double 14.7; Heston con 1% single 23.2 (3 seeds), double 20.6. The barrier and call cells have no failures (sd 1–3%). Decision needed from the user: report as is (protocol), or run a second rate per seed and select per seed on validation.
- 2026-11-11 10:00 GMT: **user decision: per-seed rate selection (option 2) and WW-style bands for all cells.** `make_main.sh` gained `seed2chain` (the seed's second-best tuning rate) and `maybe_double` (on finish: compare validation with the seed's first-rate single; submit a double from the second-rate single only if it wins; double named `..._from<lr>_s<seed>`). Submitted 46 chains: seeds 2–5 of 11 cells (+ seed 1 for the BSM call, whose second rate 3e-5 was never run to convergence). Second rates: bsm call 3e-5; bsm doc 0.1% 3e-3, 1% 1e-4; bsm con 0.1% 1e-2, 1% 3e-5; heston call 0.1% 1e-3; heston doc 0.1% 1e-3, 1% 3e-3; heston con 1e-3 both. Heston call 1% waits for its grid rerun. Per-seed rule: single = validation-best of the two rates; double = the one started from that single. Queue: 9 running, 61 pending.
  WW for all cells: the barrier and digital are hedged with two options (call and put), so the reference becomes a model **delta–gamma matching portfolio** in those two instruments (greeks by 5-point finite differences of the bank's own pricers; Heston barrier via the nested MC with common random numbers), with a WW-style no-trade band per instrument, half-width k·(fee_i · X_i · (d target_i/dS)²)^(1/3), k tuned on validation; for the calls it reduces to WW (Heston: Heston delta, gamma by finite differences). Implementation in progress (`fast_hedging/model_band.py`).
- 2026-11-11 12:30 GMT: **model-band baselines.** `fast_hedging/model_band.py` written (delta / delta–gamma target in the cell's own hedging instruments, WW-style band per instrument, k on validation; greeks by 5-point finite differences of the bank pricers, cached next to the banks). Smoke test (job 35751645, 20-path banks, CPU) passed: on the BSM call the finite-difference band reproduces the analytic WW band (35.31 vs 35.34). Problem found: the Heston barrier's nested-MC greeks at a 1% bump are noise (model target tracked 542 against no hedge 213), so the barrier now runs with a 5% bump and 10,000 inner paths (`--bump`, `--mc-paths` flags; cache name carries them). Full job `baselines_main.slurm` resubmitted as 35757578 (GPU; barrier last; resumable). Note from the existing summaries: under Heston in the held-out year the model delta hedge is poor — call 1%: delta 118, minimum-variance delta 80.5, no hedge 157.5, against the RL single net 44 and double 43 — so the band reference for Heston cells will be weak; the paper should say why (the test year's dynamics vs the calibrated model).
- 2026-11-11 17:40 GMT: **model-band baselines for 10 of 12 cells** (job 35774500; the Heston barrier, nested MC at a 5% bump, is still pricing). Check: on the BSM call the finite-difference band reproduces the analytic WW band exactly (0.1%: 8.36 / 12.7 both; 1%: 35.46 / 54.0 vs 35.48 / 54.2). Test tracking error / cost of the model band (k tuned on validation), with the daily model target and no hedge:

  | Cell | Fee | Model band | Model target, daily | No hedge |
  |---|---|---|---|---|
  | BSM call | 0.1% / 1% | 8.36 / 12.7 · 35.46 / 54.0 | 11.26 / 20.3 · 112.2 / 203 | 153.8 |
  | BSM doc | 0.1% / 1% | 3.46 / 0.5 · 12.59 / 11.8 | 3.46 / 0.5 · 12.86 / 12.1 | 158.6 |
  | BSM con | 0.1% / 1% | 5.98 / 2.4 · 14.79 / 23.6 | 6.22 / 3.4 · 49.2 / 85.8 | 93.3 |
  | Heston call | 0.1% / 1% | 84.4 / 15.1 · 74.5 / 42.9 | 84.3 / 18.3 · 118.1 / 183 | 157.5 |
  | Heston con | 0.1% / 1% | 45.1 / 2.9 · 51.2 / 36.3 | 45.2 / 3.6 · 67.6 / 90.7 | 84.6 |

  Reading: under BSM the model band is the strongest reference and the RL nets are 8–20% behind it on the call and barrier; on the BSM digital the nets beat it at 0.1% (single 4.56, double 4.4–4.9 vs 5.98) and the double net is close at 1% (healthy seeds ~14.7 vs 14.79). Under Heston the model hedge is weak in the test year (band 45–84 against RL 6–43), the same effect as the Heston delta references noted at 12:30; the band's validation scores (42, 44 for the call; 20, 25 for the digital) are far below its test scores, so the gap is a test-year effect, not a tuning failure. The Heston references therefore mainly show the gap between a calibrated-model hedge and a learned one; the paper must say so.
  Dashboard: report_runs.py reads main_baselines (rows "<cell>: Whalley–Wilmott-style model band (k=…)", "model target", "minimum-variance delta", "no hedge"); overview column "Model band test"; the small multiples' dashed line is now each cell's band on validation. Page v14, runs v31 (asset cd770dc3f812af51d4fbc402437078fd), curves v45 (asset 4480ffe2104f548f708ad21622b3ae08).
  **Heston call 1%:** rerun grid converged: i1e-3 j1e-4 (val 34.66) > i1e-3 j1e-5 35.22 > i1e-4 × {j1e-4, j1e-5} 36.5–36.6. Seed singles at 3e-3 done (seed 2 failed: val 100.5; others 36.0–36.8). Submitted by hand: 4 seed doubles from the 3e-3 singles (jobs 35785723/25/27/29) and 4 second-rate chains at 1e-3 (35785724/26/28/30). Queue 39.
  Per-seed-selected table (single = validation-best of the two rates per seed; double = the one started from that single; cells still filling in, so counts are given):

  | Cell | Fee | Single track / cost | Double track / cost | Model band | Seeds S/D |
  |---|---|---|---|---|---|
  | BSM call | 0.1% | 9.99 ± 0.16 / 14.7 | 9.79 ± 0.68 / 12.8 | 8.36 / 12.7 | 4/4 |
  | BSM call | 1% | 39.81 ± 0.70 / 57.9 | 37.87 ± 0.47 / 58.1 | 35.46 / 54.0 | 5/4 |
  | BSM doc | 0.1% | 3.82 ± 0.23 / 0.8 | 3.91 ± 0.29 / 0.8 | 3.46 / 0.5 | 5/4 |
  | BSM doc | 1% | 13.68 ± 0.14 / 13.6 | 13.59 ± 0.24 / 11.4 | 12.59 / 11.8 | 5/5 |
  | BSM con | 0.1% | 4.56 ± 0.24 / 3.8 | 4.87 / 3.9 | 5.98 / 2.4 | 5/1 |
  | BSM con | 1% | 22.61 ± 6.77 / 32.8 | 18.76 ± 8.20 / 24.6 | 14.79 / 23.6 | 5/5 |
  | Heston call | 0.1% | 28.63 ± 0.30 / 18.9 | 28.58 ± 0.20 / 16.0 | 84.4 / 15.1 | 5/3 |
  | Heston call | 1% | 68.1 ± 47.5 / 83.5 | 42.78 / 54.0 | 74.5 / 42.9 | 5/1 |
  | Heston doc | 0.1% | 8.55 ± 0.26 / 2.4 | 8.42 ± 0.08 / 1.9 | pending | 5/2 |
  | Heston doc | 1% | 35.52 ± 0.99 / 40.3 | 30.10 ± 0.82 / 26.8 | pending | 5/4 |
  | Heston con | 0.1% | 6.43 ± 0.45 / 5.0 | 6.18 / 4.2 | 45.1 / 2.9 | 5/1 |
  | Heston con | 1% | 20.96 ± 1.46 / 32.7 | 18.51 ± 0.71 / 20.9 | 51.2 / 36.3 | 5/2 |

  The per-seed selection has already repaired Heston con 1% (second rate 1e-3 wins on seeds 3 and 5: single 20.96 ± 1.46 against 38.0 ± 20.4 before; the doubles from the 1e-3 singles are queued). BSM con 1% seed 5 (val 42.4 at 1e-3) has its 3e-5 single at 18.48 and still running; Heston call 1% seed 2 (val 100.5 at 3e-3) gets 1e-3 now. Tool: scratchpad `curves/table_now.py` (rates per cell hard-coded in `MAIN`).
- 2026-11-11 19:45 GMT: **baselines complete (job 35774500, 17 min).** Heston down-and-out call band: 0.1% k=0.25, val 41.1, test 83.2 / 6.9; 1% k=16, val 45.6, test 89.4 / 107.4 (daily target 83.2 / 7.1 and 132.7 / 176.3; no hedge 186.2). As for the other Heston cells, validation (41–46) is far below test (83–89) and the RL nets (8.4 and 30.1) are an order of magnitude better: the calibrated-model hedge does not transfer to the test year. Model bands exist for all 12 cells now; dashboard runs v32 (asset 5e4e299f94ae56454978f29f761bd4bc), curves v46 (asset cb2bb1bdadcf6e20a83b5037a3251985). Per-seed table at this time: BSM doc 0.1% single 3.67 ± 0.20 / 0.7 (3e-3 wins on 3 seeds), double 3.78 ± 0.19 / 0.8 (3 seeds); **BSM con 1% single 21.0 ± 2.2 / 37.9 (seed 5 at 3e-5: 18.5 replaces 42.4), double 14.63 ± 0.06 / 23.3 over 3 seeds, equal to the model band 14.79 / 23.6**; Heston con 0.1% double 6.32 ± 0.14 / 4.7 (2 seeds). 280 runs done, 0 failures.
  **Queue stalled:** 23 pending with reason QOSMaxGRESPerUser, 2 running (both on gpushort, whose QOS caps 2 GPUs per user). Our jobs run under QOS `normal`; the 12-GPU allowance comes from the `sae` partition QOS, and sae currently has 13 of 16 nodes in state `mix-` (planned: backfill has reserved them for queued multi-node jobs) and 63 jobs running, so there is no backfill room there. Nothing to change on our side; the finish estimate slips by however long sae stays reserved (tonight to tomorrow).
- 2026-11-11 21:15 GMT: **partition policy (user).** Jobs submitted as `gpushort,sae` sat with reason QOSMaxGRESPerUser once gpushort's 2-GPU cap was hit, even though they were also eligible for sae. The 22 pending jobs were moved to `sae` only (`scontrol update Partition=sae`; their scripts edited the same way so chain resubmissions stay there). sae is full tonight (reason Priority), so the watcher now also **feeds gpushort**: every 10 minutes, if fewer than 2 of our jobs run on gpushort, it moves the oldest pending sae jobs there one by one (`feed` in scratchpad `watch_night.sh`). Scripts stay at sae so resubmitted chain links return to the pool. The generator still emits `gpushort,sae` for new jobs; switch it to `sae` if the same block recurs.
- 2026-11-12 09:20 GMT: **all main runs finished (262 in results/main, 0 failures; queue empty at 09:00).** Final table with per-seed rate selection (single = validation-best of the two rates per seed; double = the one started from that single; test tracking error / cost, mean ± sd over 5 seeds):

  | Cell | Fee | Single | Double | Double/single | Model band | No hedge |
  |---|---|---|---|---|---|---|
  | BSM call | 0.1% | 9.86 ± 0.14 / 14.6 | 9.36 ± 0.13 / 13.1 | 0.95 | 8.36 / 12.7 | 153.8 |
  | BSM call | 1% | 39.58 ± 0.69 / 57.3 | 37.75 ± 0.48 / 57.9 | 0.95 | 35.46 / 54.0 | 153.8 |
  | BSM doc | 0.1% | 3.67 ± 0.20 / 0.7 | 3.69 ± 0.19 / 0.7 | 1.01 | 3.46 / 0.5 | 158.6 |
  | BSM doc | 1% | 13.68 ± 0.14 / 13.6 | 13.59 ± 0.24 / 11.4 | 0.99 | 12.59 / 11.8 | 158.6 |
  | BSM con | 0.1% | 4.56 ± 0.24 / 3.8 | 4.38 ± 0.30 / 3.5 | 0.96 | 5.98 / 2.4 | 93.3 |
  | BSM con | 1% | 21.00 ± 2.19 / 37.9 | 14.79 ± 0.22 / 24.1 | 0.71 | 14.79 / 23.6 | 93.3 |
  | Heston call | 0.1% | 28.63 ± 0.30 / 18.9 | 28.58 ± 0.19 / 16.7 | 1.00 | 84.4 / 15.1 | 157.5 |
  | Heston call | 1% | 44.46 ± 0.67 / 57.0 | 43.98 ± 1.44 / 56.9 | 0.99 | 74.5 / 42.9 | 157.5 |
  | Heston doc | 0.1% | 8.55 ± 0.26 / 2.4 | 8.56 ± 0.24 / 2.0 | 1.00 | 83.2 / 6.9 | 186.2 |
  | Heston doc | 1% | 35.52 ± 0.99 / 40.3 | 29.98 ± 0.77 / 26.6 | 0.85 | 89.4 / 107.4 | 186.2 |
  | Heston con | 0.1% | 6.43 ± 0.45 / 5.0 | 6.28 ± 0.36 / 4.7 | 0.98 | 45.1 / 2.9 | 84.6 |
  | Heston con | 1% | 20.96 ± 1.46 / 32.7 | 16.74 ± 1.74 / 21.1 | 0.80 | 51.2 / 36.3 | 84.6 |

  (BSM call seed 1 rows are the v2_4 runs tgsg8 / dbltg8; the "Double/single" column is the mean of the per-seed ratios.) Per-seed values and hold rates are in scratchpad `curves/table_now.py` output and the dashboard. How often the second rate won on validation: bsm con 0.1% 4/5 (1e-2), heston con 1% 3/5 (1e-3), bsm doc 0.1% 2/5, bsm con 1% 2/5, heston call 1% 2/5, heston doc 0.1% 2/5, the rest 0–1/5. **No failed seed remains:** every cell's single sd is ≤ 5% of the mean except bsm con 1% (10%: two seeds at 23.5–23.7 vs 18.4–19.8, both converged, no plateau).
  Reading. (1) Double net vs single: at 1% on the exotics the double net wins clearly and consistently — bsm con −29% tracking / −36% cost, heston con −20% / −35%, heston doc −15% / −34%; BSM call −5% at both fees (cost −12% at 0.1%); the remaining cells are ties on tracking (ratio 0.98–1.01) with 10–20% lower cost at 0.1% (heston call 16.7 vs 18.9, heston doc 2.0 vs 2.4, bsm doc 1% 11.4 vs 13.6). Hold rates: 0.73–0.99 at 1%, 0.0–0.5 at 0.1%. (2) Against the model band under BSM: the nets are 8–20% behind on the call and barrier; on the digital they beat it at 0.1% (4.42 vs 5.98) and the double net equals it at 1% (14.79 vs 14.79, cost 24.1 vs 23.6). (3) Under Heston the learned nets are 2–10× better than the calibrated-model hedge on test (test-year effect, explained 2026-11-11 17:40).
  Loose ends: bsm con 0.1% seed 1 — the per-seed rule picks the 1e-2 single (val 5.17) but the seed-1 grid doubles came from the 3e-3 single (the documented deviation); a double from the 1e-2 single (i1e-4 j1e-5, `_fromlr1e2_s1`) is submitted (job 35848709, ~1 h) so the cell is 5/5 under the uniform rule. Also found and fixed: `make_main.sh`'s `double)` dispatch dropped the from-tag argument (`$9`); it had overwritten `run_bsm_con_fee01_double_i1e4_j1e5_s1.slurm` with identical content before the fix (not submitted). Dashboard: runs v33 (asset 7633c5e7f86bcc9053bb45c1fdae5205), curves v47 (asset 878ba6e27704666fe1e29ac29664ff88; thinned to every 10th check, 11 MB, after the 20 MB cap was hit).
- 2026-11-12 10:40 GMT: last loose end closed: bsm con 0.1% seed-1 double from the 1e-2 single (job 35848709) converged at val 5.16, test 4.23 / 3.3, hold 0.00; the cell is now 5/5 under the uniform rule: double 4.38 ± 0.30 / 3.5. **The main table is final** (263 runs). Table above corrected; findings.tex updated; dashboard runs v34, curves v48.
- 2026-11-12 13:30 GMT: **paper figures** in `paper_notes/figures/` (PDF + PNG; scripts in the session scratchpad `figs/make_figs.py` from runs/curves.json and `figs/make_path_figs.py` from the per-path test outputs of the seed-1 selected runs plus `main_baselines/*_paths.npz`, dumped by `slurm/main/band_paths.py`, job 35862229, which reproduces the JSON numbers exactly). Aggregate: fig_main_tracking, fig_main_cost, fig_cost_vs_tracking, fig_double_over_single, fig_hold_rate, fig_learning_curves, fig_rate_selection, fig_rate_sweep. Path level: fig_pnl_dist_fee1/fee01 (terminal P&L densities with 5th-percentile ticks), fig_life_fee1/fee01 (median |error| and cumulative cost over the life), fig_sample_paths (one BSM path per product with hold steps shaded), fig_hold_map (hold probability over moneyness × time), fig_position_vs_delta.
  Three things the path figures show that the table does not. (1) **Terminal P&L and tracking error rank differently under BSM**: the model band has the lowest tracking error on call and barrier but a more negative P&L mode than the nets (BSM call band −60 vs double −30; barrier band a spike at −17 vs nets at 0), because the band's drift is its cumulative cost, which the nets partly avoid; the paper should report both. (2) **The nets found the static hedge of the down-and-out call**: ≈ +1 call and −0.93 put for the whole life (the reflection-principle hedge with the H²/K put), near-zero trading, which is why that cell's cost is the lowest of all; the model target instead unwinds the put when the spot drifts away from the barrier. (3) **At 1% the single net is a shrunk delta** (position ≈ 0.2 + 0.5·delta on the BSM call), the known cost-optimal shrinkage; under Heston the scatter is wider and tilted, the learned hedge is not a function of the model delta. Hold map: at 1% the inaction net holds everywhere except near the strike late in life and just above the barrier; at 0.1% the digital's inactor never holds.
- 2026-11-12 16:30 GMT: **correction to the 09:20 table (BSM call rows).** `table_now.py` had excluded the v2_4 seed-1 runs (tgsg8 / dbltg8) from the per-seed selection, so seed 1 of the BSM call was taken from the 3e-5 single instead of the validation-best 1e-4 one. Corrected: 0.1% single 9.86 ± 0.14 / 14.6, double 9.36 ± 0.13 / 13.1; 1% single 39.58 ± 0.69 / 57.3, double 37.75 ± 0.48 / 57.9 (figures were already right; findings.tex corrected). **Paper:** `paper_notes/paper/` updated to the current protocol (GRPO + causal transformer, two cost regimes, per-instrument fees, 5 seeds, per-seed rate rule, model-band references); results.tex filled with the final table and draft analysis; `live_preview.py` = live LaTeX preview (latexmk -pvc + pdf.js page at localhost:8765). Tracking-error histograms replace the P&L ones (`fig_track_hist_fee1`); 50,000-path evaluations of the seed-1 nets at 1% submitted (jobs 35863073–78, results in experiments/results/main_big) to redraw the histograms on fresh paths.
- 2026-11-12 18:00 GMT: **50,000-path evaluation (1%, seed-1 selected nets, fresh paths, split seed 404; jobs 35863073–78, 2–30 min each).** Test tracking error single / double: BSM call 38.79 / 36.94; BSM doc 13.68 / 13.62; BSM con 19.64 / 14.57; Heston call 44.66 / 44.11; Heston doc 33.03 / 27.20; Heston con 21.24 / 16.82 — within 1–6% of the 2,000-path numbers of the same nets, so the 2,000-path test set is not under-sampled for the means. Arrays in `experiments/results/main_big/<cell>_<net>.npz` (+ .json metrics; script `slurm/main/eval_big.py`, `eval_big.slurm`). `fig_track_hist_fee1` now uses these 250,000 environments per net; the paper version is regenerated.
- 2026-11-12 20:30 GMT: paper edits per the user: abstract claim matched to the measured gains; "Actor–Inactor" naming everywhere (text and all figures); negative-control sentence for the low-cost BSM digital; transformer + GRPO mentioned in related work and the GRPO advantage / PPO clip loss written out in the methodology; duplicates paper.tex / bibliography.bib deleted; corrected Greeks paragraph (digital delta ~ 1/sqrt(T−t) at the strike, barrier Greeks bounded but discontinuous hedge at knock-out); objective-choice remark citing Buehler et al.; **CVaR dropped from the paper** (mixed direction: Actor–Inactor tail worse on BSM call/doc at 1%, better on Heston doc/con; digitals' tail equals no-hedge; band dominates the tail under BSM) — numbers kept here: CVaR95 single/double/band: bsm call 56/53/45 (0.1%), 261/303/138 (1%); bsm doc 21/22/7, 67/81/20; bsm con 152/152/156, 186/184/184; heston call 144/145/290, 243/257/417; heston doc 48/47/721, 490/364/1082; heston con 132/132/188, 270/220/268. **Figure 1 is now `fig_main_tracking_zoom`** (BSM left, Heston right, product × fee rows, per-panel linear axis, band annotated when off-scale). Layout made robust: all result floats declared at the head of the results section, all double-column, dblfloatfix, raggedbottom.
- 2026-11-12 22:30 GMT: **reviewer pass on the paper** (TRAINING.md keeps the substance; the review itself was given in chat). Fixed in the paper: (1) the Heston reference explanation was wrong — the band uses the test year's own Greeks; it loses because the market is incomplete (spot-only hedge leaves volatility unspanned) and the 2021 calibration (θ=0.082 ≫ v0=0.027, κ=0.96, ξ=0.50, ρ=−0.95) makes that risk large; mv-delta 80 vs delta 118 and the tilted Fig. 7 support it; (5) the BSM DoC is statically hedgeable by the reflection pair (long call K, short K/H puts at H²/K) — stated in the preliminaries; (6) the two-instrument delta–gamma band is labelled a reference strategy of our own construction; (7) Limitations, Conclusion and a Compute paragraph added; (8) disclosure that the per-seed rule was introduced after stalled seeds were seen + Appendix A with the one-rate-protocol table (notable: heston doc 0.1% single 8.32 ± 0.08 under one rate vs 8.55 ± 0.26 per-seed — the second rate won on validation, lost on test). (3) Band net (Imaki et al.-style) added as a third architecture: `make_main.sh band <dyn> <prod> <fee> <seed> <lr>` → `<cell>_band_lr<lr>_s<seed>`, one stage, at each seed's selected single rate (list in scratchpad paths/band_jobs.txt, 60 runs); **submission pending: SSH master dropped before the sbatch loop ran** (make_main.sh updated locally; HPC copy not yet updated). (2) 2022/2023 test years blocked on option-chain data (repo has sp500_2017–2021.json.bz2 only); pipeline afterwards: calibrate (~15 min/year Heston), build 2,000-path banks, evaluate the 120 saved nets + bands. Point 4 (compute fairness) dropped per the user: early stopping to convergence makes the budget argument moot; optional control = warm-restart the converged single at the joint-stage actor rate.
- 2026-11-12 23:10 GMT: **band-net comparison submitted**: 60 runs (12 cells × 5 seeds), `<cell>_band_lr<lr>_s<seed>`, one stage, each seed at its selected single-policy rate, partition `sae` with the watcher feeding gpushort. Expected 1–2 days on sae. Protocol: validation early stopping as the other nets; the band net is `--net band` (actor outputs target + no-trade half-width per instrument; executed position = current holding projected onto the band). Report as a third column in the paper's main table once done.
- 2026-11-13 11:40 GMT: **band runs**: sae opened overnight; 12 running, 45 pending, 3 done, 0 failures. First band-net results (test track / cost; same seed's single, double): bsm con 0.1% s1 6.73/1.7 (4.24/3.4, 4.23/3.3); bsm con 1% s5 22.6/12.0 (18.4/31, 14.7/22); bsm doc 1% s5 14.4/11.5 (13.6/13.4, 13.2/11.0) — the band net trades least and tracks worst so far. **OLS of position on model delta** (call cells, seed 1, 500 envs × 250 steps): 0.1%: BSM single 0.02+0.97Δ (R² 0.99), double 0.02+0.97Δ; Heston single −0.11+0.91Δ (0.91). 1%: BSM single 0.25+0.48Δ (0.94), double 0.11+0.64Δ (0.82); Heston single 0.05+0.57Δ (0.93), double −0.03+0.67Δ (0.86). Figure `fig_delta_ols` (2×2, column width) replaces `fig_position_vs_delta` as fig:delta. Written into results 6.5; the "illiquid instruments / no extra cost at low cost" argument added to the abstract, intro and conclusion.
- 2026-11-13 13:30 GMT: **parameter parity was false in the main table**: checkpoints show every from-single Actor–Inactor is a 64+64 pair (70,681 + 70,482 params) against the 70,681-param single, because `from-single` raises cfg.hidden to 91 so the loaded actor keeps width 64 and the inactor is built at int(91/√2)=64. The 45+45 pair exists only when the double net trains from scratch. **Fix (user): parity runs submitted** — `make_main.sh pair <dyn> <prod> <fee> <seed> <lr> <il> <jl>` → `<cell>_pair_i<il>_j<jl>_lr<lr>_s<seed>`: `--net double --actor-init train --tune-joint-lr`, stage 1 trains its own 45-wide actor at the seed's selected single rate, stage 2 adds a 45-wide inactor (cell's il/jl pick); 60 runs on sae (job list scratchpad paths/pair_jobs.txt). Queue: 105 pending (60 pair + 45 band), 11 running. The paper's main Actor–Inactor column will be replaced by these when done; the 64+64 results stay as "same actor, gate added" evidence. Paper sentences fixed: BSM σ fitted to realised returns (not implied vol); joint actor rate "at or below" the inactor's; Monte Carlo noise in the barrier's mark affects both arms equally. 2022/2023: no option chains in the repo; yfinance not installed locally; options are vendor data (OptionMetrics/CBOE) or a VIX-term-structure proxy calibration (approximate; to discuss).
- 2026-11-13 15:30 GMT: **2022/2023 test years — data in hand.** The user bought OptionsDX SPX end-of-day chains (8 quarterly 7z files, one row per date/expiry/strike with call and put bid/ask, underlying last). Converted (scratchpad `convert_spx.py`) to the pipeline's yearly long format: 3.94 M / 3.77 M quotes, 256 / 249 days; spot = UNDERLYING_LAST; price = bid/ask mid (both > 0); rate = 13-week T-bill (^IRX) close of the day. Uploaded as `option_data/sp500_2022.json.bz2`, `sp500_2023.json.bz2`. **Heston calibration running** (job 35977970, `compute`, 4 h; same script/settings as 2017–2021; it appends to `heston_params_QL_v2_sp500.json` on the HPC — afterwards split into a new market `sp500x` (7 years → test split = 2021, 2022, 2023) and restore the 5-year `sp500` file so the main results' banks are unchanged). BSM side: realised vol from the converted spot series (computation running). Ready: `eval_years.py` (per-year metrics of saved nets on the sp500x test bank), `model_band.py --market`. Band/pair runs: 10 running, 104 pending, 6 band done.
- 2026-11-13 13:20 GMT: **2022/2023 Heston calibration done** (job 35977970, 60 min for both years, same script/settings as 2017–2021): 2022 v0 0.0336, κ 0.680, θ 0.094, ρ −0.95, ξ 0.446, WRMSE 0.0124 (date 2022-01-06); 2023 v0 0.0156, κ 1.598, θ 0.0335, ρ −0.863, ξ 0.335, WRMSE 0.0188 (date 2023-12-20); Feller not satisfied in either, as in 2017–2021. Strikes 4775–4815 and 3835–3875 (5-pt SPX grid). New market **`sp500x`** = 7 years (`heston_params_QL_v2_sp500x.json`, `bsm_data_sp500x_2017_2021.json`; BSM 2022 σ 0.239, 2023 σ 0.131 from realised returns), test split = 2021, 2022, 2023; the `sp500` files restored to 5 years so the main banks are unchanged. **Submitted 12 `eval_years` jobs** (one per cell): per-year metrics of the 10 selected nets (5 seeds × single/double) + no-hedge/delta references → `experiments/results/main_years/<cell>/`, and the model band per year → `main_years_baselines/` (model_band.py gained `--market` and `test_by_asset`). Note: 2021 is re-evaluated on a *new* 2,000-path bank (different paths from the main test set, same calibration) — a free robustness check of the 2021 numbers.
- 2026-11-13 18:45 GMT: **feeder bug.** The watcher's gpushort feeder counted only *running* jobs on gpushort, so while the moved jobs sat pending with `ReqNodeNotAvail` it kept moving more every 10 min: 63 band jobs + the 12 `years` jobs ended up pending on gpushort. All moved back to `sae`; feeder fixed to count pending+running (never more than 2 on gpushort); watcher restarted. The 2 oldest `years` jobs moved to gpushort by hand. Status 18:33: band 17/60 done, pair 0/60 (2 running), years 0/12; 0 failures. sae is saturated (all our sae jobs `Priority`), so throughput today is 2–6 GPUs.
- 2026-11-14 17:00 GMT: **status**: band 41/60 done, pair 0/60 (1 running, sae saturated), years 2/12 cells done; 0 failures. **New-year results, BSM call (5 seeds, test track / cost; no hedge; band)**: 0.1%: 2021 9.69±0.17/14.7 vs 9.19±0.14/13.3 (nh 147, band 8.3); 2022 22.5±1.4/33.8 vs 20.3±1.4/26.4 (nh 364, band 15.2); 2023 10.61±0.18/15.6 vs 10.06±0.17/14.0 (nh 162, band 9.0). 1%: 2021 37.4±0.6/57.5 vs 35.1±0.4/58.8 (band 34.9); 2022 74.9±5.1/128 vs 65.9±1.1/109 (band 56.2); 2023 42.0±0.7/60.9 vs 40.1±0.5/61.7 (band 37.9). The Actor–Inactor's gain holds in every year (−5%, −10%, −5% at 0.1%; −6%, −12%, −5% at 1%) and is largest in the 2022 drawdown year (vol 24%). 2021 on fresh paths reproduces the main-table numbers within 2%.
  **Band net (41 runs) — important**: at 1% the band net is a strong competitor: BSM call 1% band 33.9–37.6 (mean ≈35.1) vs single 39.6 / Actor–Inactor 37.75; BSM con 1% band 12.6–13.2 at 1e-3 (the two 3e-5 seeds 19.2, 22.6) vs 21.0 / 14.8; BSM doc 1% 13.3–14.4 vs 13.68 / 13.59; Heston call 1% 41.6–44.2 vs 44.5 / 44.0; Heston con 1% 15.3, 17.0 vs 21.0 / 16.7; Heston doc 1% 22.4–33.9 vs 35.5 / 30.0. At 0.1% it is worse or unstable: BSM call 13.7, 28.3 (vs 9.9); BSM con 6.0–7.7 (vs 4.56); Heston doc 8.3–9.1 (tie). It trades far less (13–44 trades/path at 1% vs ~250). Not yet per-seed-complete; full comparison when all 60 are in. This must change the paper's framing: the gate is not uniformly better than a learned band at 1%; the honest claim will be about low-cost robustness + simplicity, or the Actor–Inactor must be improved (e.g. the parity runs, or an actor that proposes less aggressively).

## Queue and priority notes

- **Two gpushort slots, always full (user rule, 2026-11-14).** The account has 2 GPUs on gpushort at all times. Whenever work is pending, exactly two of our jobs must sit on gpushort (`scontrol update JobId=<id> Partition=gpushort`), the rest on `sae`; never submit `gpushort,sae`. The watcher's `feed()` does this every 10 min and at start; after every SSH reconnect restart it and check `squeue -u acw796 -p gpushort` by hand. Short evaluation jobs go first.

- A resubmitted chain job reads `#SBATCH` lines from the script on disk, so priority must be set there (`#SBATCH --nice=N`); `scontrol update nice` lasts only for that job.
- A released held job keeps its old submit time, which gives it a high age priority. Set its nice before releasing it.

### 2026-11-14 18:30 GMT — band net on Heston (interim, 15/30 Heston band runs done)
Per-seed validation-selected test tracking / cost (n = seeds done). Single and Actor–Inactor (64+64) from the main table.

| Cell | Fee | Single | Actor–Inactor | Band net | Model band |
|---|---|---|---|---|---|
| Heston call | 0.1% | 28.63 / 18.9 | 28.58 / 16.7 | 28.59 / 12.9 (n=1) | 84.4 / 15.1 |
| Heston call | 1% | 44.46 / 57.0 | 43.98 / 56.9 | 42.71 ± 1.05 / 48.2 (n=4) | 74.5 / 42.9 |
| Heston DoC | 0.1% | 8.55 / 2.4 | 8.56 / 2.0 | 8.75 ± 0.29 / 1.0 (n=4) | 83.2 / 6.9 |
| Heston DoC | 1% | 35.52 / 40.3 | 29.98 / 26.6 | 29.10 ± 4.27 / 25.7 (n=4; rate 3e-3 seed: 22.4) | 89.4 / 107.4 |
| Heston CoN | 0.1% | 6.43 / 5.0 | 6.28 / 4.7 | — | 45.1 / 2.9 |
| Heston CoN | 1% | 20.96 / 32.7 | 16.74 / 21.1 | 16.14 ± 0.86 / 18.1 (n=2) | 51.2 / 36.3 |

Reading: the band net's advantage is not a BSM artefact (where a no-trade band around delta is the asymptotically optimal
shape). On Heston at 1% it ties or edges the Actor–Inactor on tracking and is cheaper; at 0.1% all three tie. Wait for all
60 band runs and the 60 parity (45+45) runs before changing the paper. Watcher feed bug fixed (zsh glob on `!seen[...]`);
feed now runs from `~/feed_gpushort.sh` on the HPC; two `mn_years` jobs on gpushort; years evals 4/12 cells done.

### 2026-11-14 18:40 GMT — what the inactor learned, and two new Actor–Inactor experiments (gate, xfer)
**Context.** The band net ties the Actor–Inactor at 1% on BSM *and* Heston (see 18:30 entry). The user's position: both
parameterise inaction; the question is what the decoupled design can do that a band cannot, and whether a frozen hedger plus a
retrained gate transfers across liquidity regimes. Per-instrument gating was discussed and dropped: every exotic cell hedges with
a call and a put at the same option fee (0.2% / 5%), the vanilla call with the underlying only (0.1% / 1%); no cell mixes fees.

**Trace analysis (seed 1, test traces, `scratchpad/gate/hold_analysis.py`).** Logistic regression of the deterministic hold
decision on (i) the proposed trade's notional, (ii) market state (S/K, S/K², t, t², 1- and 5-step returns and their
magnitudes), (iii) both; AUC on the last 30% of trace paths. Note the main-table inactor sees neither its holding nor the
actor's proposal (`--no-agent-state`), so it *cannot* implement a band; the question is whether its hold decisions nevertheless
line up with trade size.

| Cell, 1% | hold rate | AUC trade size | AUC market state | AUC both | hold rate, smallest / largest trade quintile |
|---|---|---|---|---|---|
| BSM call | 0.84 | 0.68 | 0.78 | 0.80 | 0.74 / 0.96 |
| BSM DoC | 0.95 | 0.90 | 0.91 | 0.96 | 0.82 / 0.97 |
| BSM CoN | 0.81 | 0.57 | 0.73 | 0.73 | 0.72 / 0.86 |
| Heston call | 0.87 | 0.50 | 0.79 | 0.80 | 0.85 / 0.89 |
| Heston DoC | 0.98 | 0.77 | 0.84 | 0.90 | 0.96 / 0.97 |
| Heston CoN | 0.90 | 0.54 | 0.83 | 0.83 | 0.86 / 0.91 |

At 0.1% the hold rates are 0.20–0.51 (BSM CoN 0.1%: 0.00, the gate never holds) with AUCs 0.52–0.75. Reading: except for
the DoC, trade size barely predicts the hold decision while market state does. The inactor has learned *when in the life of the
option to be active*, not *how far from target before moving*; it is a regime gate, not a band in disguise. The two mechanisms
are therefore complementary, which motivates giving the gate the gap as an input.

**Code (`fast_hedging/ppo.py`, `experiment.py`; backward compatible, defaults off).**
- `--inactor-proposal` (`cfg.inactor_proposal`): with `--no-agent-state`, the inactor still receives
  [holding, proposal, proposal − holding, estimated cost of the proposal] (3A+1 inputs). Actor inputs unchanged.
- `--freeze-actor` (`cfg.freeze_actor`): joint phase never updates the actor (inactor only).
- `make_main.sh gate <dyn> <prod> <fee> <seed> <il> <jl> <single_run>` → `<cell>_gate_i<il>_j<jl>_s<seed>`: the main
  from-single recipe (64+64) + `--inactor-proposal`. Isolates the value of the proposal input versus the main Actor–Inactor.
- `make_main.sh xfer <dyn> <prod> <fee> <seed> <il> <single_run_at_other_fee>` → `<cell>_xfer_i<il>_s<seed>`: actor loaded
  from the **0.1% single of the same seed and frozen**, proposal-aware inactor trained alone at 1%. Liquidity transfer: "we
  have a hedger, we only train the thing that enacts it". Compare with the 1% Actor–Inactor (full training) and the band net.
- Smoke-tested locally (CPU, 8 paths): both flags run end to end; actor std logs NaN in the frozen joint phase as expected.
- `make_main.sh` template now `--partition=sae` (was `gpushort,sae`); `report_runs.py` parses gate/xfer names (orders 104/105).

**Submitted 18:38 GMT:** 60 jobs (`gate_jobs.txt`: 6 cells at 1% × 5 seeds × {gate, xfer}), ids up to 36170794, all on sae
(queue now 142 pending / 5 running; gpushort holds two years evals). Rates: each seed's selected (inactor, joint) rates from the
main table; the gate's source single is the seed's selected 1% single, the xfer's source the seed's selected 0.1% single.
Still pending from before: band 19/60, parity 60/60, years 8/12 cells.

### 2026-11-15 09:20 GMT — overnight: 374 done, one failure (xfer), fixed
The watcher ran its 12 hours (18 gpushort feeds, no slot left empty) and exited; restarted. One failure: the `xfer` recipe loads
an actor from the 0.1% single into a 1% run and `_compatible()` compared fees → "incompatible config". Fixed in
`experiment.py`: with `--freeze-actor` the fee keys are skipped (everything else — arch, window, agent state, reward, padding —
must still match). Uploaded; `bsm_call_fee1_xfer_i1e4_s2` resubmitted (36257264), the other 29 xfer jobs had not started.
Queue 09:20: 109 pending, 2 running (sae saturated), gpushort holds the last years eval and a parity run.

### 2026-11-15 09:50 GMT — test years 2021 / 2022 / 2023 (all 12 cells done)
Market `sp500x` (BSM σ and Heston parameters fitted per year to 2017–2023 SPX data; the nets are the main-table nets, trained on 2017–2020 only, so 2022 and 2023 are unseen regimes; 2022 is the high-volatility year, σ=0.239). 2,000 fresh test paths per year, split seed 303. Mean over the 5 selected seeds ± sd, tracking / cost. Model band = delta (call) or delta–gamma (exotics) target with a validation-tuned WW-style band; missing for Heston DoC 0.1% (band job did not produce it, see below).

| Cell | Fee | Year | Single | Actor–Inactor | Δ track | Model band | No hedge |
|---|---|---|---|---|---|---|---|
| bsm call | 0.1% | 2021 | 9.69±0.17 / 14.7 | 9.19±0.14 / 13.3 | -5% | 8.26 / 12.9 | 146.8 |
| bsm call | 0.1% | 2022 | 22.53±1.39 / 33.8 | 20.33±1.40 / 26.4 | -10% | 15.22 / 19.3 | 364.2 |
| bsm call | 0.1% | 2023 | 10.61±0.18 / 15.6 | 10.06±0.17 / 14.0 | -5% | 8.96 / 13.3 | 162.3 |
| bsm call | 1% | 2021 | 37.38±0.61 / 57.5 | 35.11±0.35 / 58.8 | -6% | 34.87 / 54.4 | 146.8 |
| bsm call | 1% | 2022 | 74.94±5.13 / 128.0 | 65.85±1.06 / 108.9 | -12% | 56.23 / 89.3 | 364.2 |
| bsm call | 1% | 2023 | 42.04±0.65 / 60.9 | 40.11±0.45 / 61.7 | -5% | 37.85 / 57.2 | 162.3 |
| bsm doc | 0.1% | 2021 | 3.54±0.14 / 0.7 | 3.57±0.11 / 0.7 | +1% | 3.37 / 0.5 | 151.6 |
| bsm doc | 0.1% | 2022 | 19.16±0.94 / 4.6 | 18.99±0.91 / 3.6 | -1% | 17.73 / 1.6 | 385.2 |
| bsm doc | 0.1% | 2023 | 3.97±0.22 / 0.8 | 3.99±0.20 / 0.7 | +1% | 3.65 / 0.5 | 167.1 |
| bsm doc | 1% | 2021 | 13.37±0.20 / 13.4 | 12.99±0.19 / 11.4 | -3% | 12.65 / 11.8 | 151.6 |
| bsm doc | 1% | 2022 | 56.87±1.52 / 58.2 | 52.35±2.96 / 37.7 | -8% | 47.65 / 39.1 | 385.2 |
| bsm doc | 1% | 2023 | 14.66±0.16 / 14.1 | 14.57±0.28 / 11.8 | -1% | 13.37 / 12.3 | 167.1 |
| bsm con | 0.1% | 2021 | 4.60±0.26 / 3.8 | 4.42±0.31 / 3.5 | -4% | 5.99 / 2.5 | 91.4 |
| bsm con | 0.1% | 2022 | 9.23±0.45 / 7.5 | 9.13±0.53 / 7.0 | -1% | 3.30 / 3.6 | 120.2 |
| bsm con | 0.1% | 2023 | 4.76±0.24 / 4.0 | 4.59±0.31 / 3.7 | -4% | 6.27 / 2.6 | 96.9 |
| bsm con | 1% | 2021 | 20.33±2.22 / 37.9 | 14.59±0.32 / 24.6 | -28% | 15.03 / 23.7 | 91.4 |
| bsm con | 1% | 2022 | 37.73±1.21 / 72.4 | 23.40±2.87 / 39.4 | -38% | 19.90 / 36.0 | 120.2 |
| bsm con | 1% | 2023 | 22.20±2.36 / 40.8 | 15.63±0.20 / 25.9 | -30% | 15.87 / 25.0 | 96.9 |
| heston call | 0.1% | 2021 | 29.50±0.32 / 19.0 | 29.45±0.21 / 16.8 | -0% | 83.43 / 15.2 | 154.6 |
| heston call | 0.1% | 2022 | 41.17±0.47 / 27.2 | 40.96±0.23 / 23.7 | -0% | 115.10 / 20.2 | 224.8 |
| heston call | 0.1% | 2023 | 27.35±0.18 / 18.5 | 27.10±0.16 / 16.3 | -1% | 51.44 / 15.7 | 138.7 |
| heston call | 1% | 2021 | 45.00±0.89 / 57.3 | 44.77±1.65 / 57.4 | -1% | 74.49 / 42.9 | 154.6 |
| heston call | 1% | 2022 | 63.17±1.68 / 81.3 | 63.27±2.66 / 81.9 | +0% | 107.00 / 57.4 | 224.8 |
| heston call | 1% | 2023 | 45.15±0.61 / 59.0 | 43.47±1.14 / 57.1 | -4% | 49.71 / 42.0 | 138.7 |
| heston doc | 0.1% | 2021 | 8.49±0.27 / 2.4 | 8.47±0.24 / 2.0 | -0% | — | 184.9 |
| heston doc | 0.1% | 2022 | 11.99±0.62 / 4.2 | 11.83±0.56 / 3.4 | -1% | — | 265.6 |
| heston doc | 0.1% | 2023 | 5.24±0.10 / 1.3 | 5.23±0.10 / 1.1 | -0% | — | 151.1 |
| heston doc | 1% | 2021 | 33.54±1.14 / 40.0 | 28.69±0.98 / 26.8 | -14% | 87.64 / 106.0 | 184.9 |
| heston doc | 1% | 2022 | 50.33±2.87 / 66.7 | 40.32±1.23 / 38.8 | -20% | 123.86 / 153.4 | 265.6 |
| heston doc | 1% | 2023 | 19.36±0.40 / 23.4 | 17.12±0.32 / 17.4 | -12% | 42.88 / 62.0 | 151.1 |
| heston con | 0.1% | 2021 | 6.23±0.46 / 5.0 | 6.09±0.35 / 4.6 | -2% | 44.42 / 2.9 | 83.6 |
| heston con | 0.1% | 2022 | 9.90±0.92 / 7.8 | 9.82±0.75 / 7.3 | -1% | 52.85 / 3.8 | 114.0 |
| heston con | 0.1% | 2023 | 6.11±0.20 / 4.8 | 5.93±0.18 / 4.5 | -3% | 27.35 / 2.6 | 89.9 |
| heston con | 1% | 2021 | 20.03±1.32 / 32.2 | 16.27±1.40 / 21.1 | -19% | 50.48 / 36.6 | 83.6 |
| heston con | 1% | 2022 | 29.98±2.83 / 50.7 | 23.36±2.50 / 32.7 | -22% | 62.90 / 49.2 | 114.0 |
| heston con | 1% | 2023 | 21.04±0.84 / 32.2 | 16.74±0.46 / 24.3 | -20% | 32.51 / 27.8 | 89.9 |

Reading: the Actor–Inactor gain is stable across years and largest in 2022: BSM CoN 1% −28/−38/−30%, Heston DoC 1% −14/−20/−12%, Heston CoN 1% −19/−22/−20%, BSM call 1% −6/−12/−5%; 0.1% ties everywhere with 5–20% lower cost. Under Heston the model band fails in every year (incompleteness, as in the paper). Under BSM the model band beats both nets in 2022 for the call and the barrier (regime shift hurts the learned nets more than the closed-form hedge); for the digital at 0.1% in 2022 the model band is far better (3.3 vs 9.1). Honest caveat for the paper.

### 2026-11-15 10:40 GMT — why the frozen-actor transfer fails on the call and works on the exotics
Question from the user: the 1% call hedge is "reduced"; why not the exotics'? Checked on the seed-1 test traces
(`scratchpad/gate/shrink_check.py`): OLS of the 1% single net's position on the 0.1% single net's position, same paths.

| Cell | Instr. | slope (intercept) | R² | hedge notional / premium | turnover / premium, 0.1% → 1% |
|---|---|---|---|---|---|
| BSM call | underlying | 0.50 (+0.24) | 0.96 | 9.6× | 67 → 26 |
| Heston call | underlying | 0.59 (+0.14) | 0.90 | 6.4× | 63 → 18 |
| BSM DoC | call K / put H²/K | 0.96 / 0.83 | 0.97 / 0.96 | 1.0× / 0.0× | 2.0 → 1.3 |
| Heston DoC | call K / put H²/K | 0.87 / 1.01 | 0.95 / 0.95 | 1.0× / 0.1× | 4.7 → 3.2 |
| BSM CoN | call K / put K | 0.72 / 0.84 | 0.57 / 0.85 | 0.3× / 0.2× | 9.0 → 3.6 |
| Heston CoN | call K / put K | 1.02 / 0.11 | 0.78 / 0.16 | 0.2× / 0.2× | 10.3 → 3.1 |

Reading. (i) The call's 1% hedge is not scaled down, it is *flattened*: same average level (mean |position| ratio 0.97), half
the sensitivity to the spot (Leland-type). A gate can only pick between the old holding and the frozen actor's proposal, so it
cannot produce a flattened hedge; xfer on the BSM call gives 46–54 against 37.8. (ii) Why the call needs it: a linear instrument
against a convex liability must be re-traded with every spot move; turnover over the life is 60–70× the premium at 0.1%, so at
1% the cost is of the order of the risk and the net halves the sensitivity. (iii) The barrier is hedged with options, whose
convexity moves with the liability's: the hedge is near static (about one call at K, held), turnover is 2–5× the premium and
most of that is the set-up trade. Nothing to flatten; the 0.1% and 1% targets coincide (slope ≈ 0.9–1, R² ≈ 0.95), and what
is left is whether to make small corrections, a timing decision. xfer BSM DoC 13.3 / 12 = full training. (iv) The digital is in
between: its 1% target *does* differ from the 0.1% one (R² 0.16–0.85) and turnover falls by 60–70%, yet the gate on the old
target reaches the same error on BSM (14.5 / 26 against 14.8 / 24, 2 seeds done, 3 more with the same validation level): its
costly trading is concentrated in particular states (near the strike, late), which a state gate can switch off. Heston exotics
xfer still queued. Earlier wording "for the exotics the target does not need to change" is right for the barrier and too strong
for the digital.

### 2026-11-16 01:15 GMT — status after the reconnect; first Heston transfer results; compute accounting
SSH dropped 23:19–01:07; both gpushort slots had emptied, refilled by hand (36194487, 36195729), watcher restarted. Queue: 88 jobs
(5 running on sae, 2 pending on gpushort, 81 pending on sae, all reason `Priority`; another user holds 321 pending sae jobs). Nothing
failed since the fixed xfer job. Progress is slow: about two chain links finished between 22:48 and 01:07.

Runs done: band 56/60, parity 45+45 13/60, gate 10/30, xfer 14/30. `curves/table_now.py` now excludes gate/xfer/parity/band runs from
the main columns (they carry `net=double, table=True` and were being picked into the Actor–Inactor column on validation; the main
table is unchanged once they are excluded).

**Transfer (xfer: 0.1% actor frozen, only the proposal-aware inactor trained at 1%), seed-matched, seeds done only.**
Hours = iterations × s/iteration of the runs, one GPU. Full pipeline at 1% = single + Actor–Inactor joint stage.

| Cell (1%) | seeds | Single | Actor–Inactor | xfer track / cost | val A–I → xfer | full h | xfer h | compute saved |
|---|---|---|---|---|---|---|---|---|
| BSM call | 3 | 40.0 | 37.9 | 49.0 ± 3.7 / 52 | 30.4 → 42.0 | 2.38 | 0.50 | 79% |
| BSM DoC | 4 | 13.6 | 13.6 | 13.3 ± 0.4 / 12 | 22.9 → 25.3 | 1.43 | 0.44 | 69% |
| BSM CoN | 2 | 21.8 | 14.8 | 14.5 ± 0.4 / 26 | 11.2 → 13.9 | 2.53 | 0.63 | 75% |
| Heston call | 1 | 45.5 | 46.7 | 48.8 / 65 | 35.2 → 38.7 | 2.02 | 0.44 | 78% |
| Heston DoC | 2 | 36.4 | 30.5 | **23.6 ± 0.4 / 25** | 16.1 → 13.6 | 1.35 | 0.43 | 68% |
| Heston CoN | 2 | 21.4 | 18.5 | 19.1 ± 0.9 / 25 | 13.3 → 12.8 | 2.00 | 0.69 | 65% |

Reading (interim, 1–4 seeds per cell).
- **Exotics: the frozen low-fee actor plus a new gate matches full training at 1% for a third of the GPU time** (BSM DoC, BSM CoN,
  Heston CoN within one sd of the Actor–Inactor on the same seeds).
- **Heston DoC: transfer beats full training**, 24.1 and 23.2 against 30.4 and 30.6 on seeds 2 and 3 (band net 28.8, 33.9; single
  35.2, 37.5), same cost. The three seeds still running sit at the same validation level (13.6 in all five seeds; the main
  Actor–Inactor's validation is 16.1). Candidate explanation: the actor learned at 0.1% is a better target than the one learned at
  1% (the 1% single is the weakest net in the table for this cell), and the hedge is near static (hold 0.99), so the gate only has to
  enact it. To confirm with 5 seeds before it goes in the paper.
- **Call: transfer fails**, on BSM (49.0 against 37.9) and, more mildly, on Heston (48.8 against 46.7, one seed). Consistent with the
  10:40 entry: the 1% call hedge is flattened and a gate cannot flatten.

**Gate (main recipe + proposal-aware inactor), 10/30 done:** no gain over the main Actor–Inactor so far. BSM call 37.5 (A–I 37.2, same
seed), BSM DoC 13.9 ± 1.4 (4 seeds; seed 4 at 16.4, the others 12.8–13.4; A–I 13.7), BSM CoN 15.2 (15.1), Heston call 42.8 (42.8),
Heston DoC 29.4, 32.8 (30.4, 30.6), Heston CoN 17.0 (17.1). Seeing the proposal does not help the gate when the actor is trained
jointly; what helps on Heston DoC is the *source* of the actor.

**Parity 45+45, 13/60 done:** Heston call 1% 44.3 ± 1.3 (5 seeds) against 44.0 for 64+64; single seeds elsewhere, one worse (Heston DoC
seed 1: 33.6 against 28.7). **Band net 56/60:** unchanged from the 2026-11-14 table. Heston DoC 0.1% years band reference (36278588)
still pending.

### 2026-11-16 01:40 GMT — repository clean-up: engine and experiments separated, legacy removed
Asked by the user: keep the paper and everything used to run its experiments, remove fluff and legacy code, make it modular,
distinguish the hedging engine from the place where experiments are run, update the README.

**New layout.** `hedging_engine/` (library: `calibration.py` + `data/`, `market.py`, `pricing.py`, `features.py`, `networks.py`,
`ppo.py`, `model_band.py`, `NOTES.md`; it never imports `experiments`) and `experiments/` (`train.py`, `baselines.py`, `protocol.py`,
`calibration/`, `slurm/`, `evaluation/`, `analysis/`, `figures/`, ignored `results/`), plus `tests/` and `docs/hpc.md`. The path map is
at the top of this file; `README.md` and `experiments/README.md` describe the layout and which script produced each table and figure.

**What changed in the code.** `ppo.py`, `networks.py`, `pricing.py`, `features.py` are byte-identical to the committed versions (moved
only). `market.py`: the parameter loader now comes from `hedging_engine/calibration.py` (a copy of the old `experiments/utils`
loader, checked equal for sp500 and sp500x under both dynamics) and the cache is `.market_cache` / `$HEDGING_ENGINE_CACHE` (the old
variable still works). `model_band.py` keeps the strategy; its command line moved to `experiments/baselines.py` and `compute_targets`
takes the bank's seed explicitly. `experiment.py` became `experiments/train.py` (imports only). Split seeds and fee regimes live in
`experiments/protocol.py`.

**Brought into the repository** (they existed only on the HPC or in the session scratchpad): `eval_big.py`, `eval_years.py`,
`band_paths.py` and their Slurm jobs, the job lists of the band / parity / gate / transfer / years batches (`experiments/slurm/jobs/`),
the tuning picks, `feed_gpushort.sh`, the run collector, the main-table script, `hold_analysis.py`, `shrink_check.py`, both figure
scripts, `convert_spx.py`, and a `fit_bsm.py` ported from the calibration notebook. The per-path results behind the figures and the
collected run tables were copied to `experiments/results/` (ignored). The converted 2022–2023 option chains are in
`experiments/calibration/option_data/` and are git-ignored (purchased data).

**Removed from the tree** (all in git: tag `pre-cleanup-2026-11-16` = commit fa6b479): the Gym/TorchRL environments and pipeline, the
first vectorised pipeline (`experiments/fast`), hyper-parameter searches and their result dumps, notebooks, `learning_folder`, the
development Slurm batches (old_pipeline, v2_1_5, v2_2_5, v2_3, v2_4), `fast_hedging/analysis.py` and `report.py`, the parity test
against the old environments, superseded calibration files (CMA and first QuantLib fits). Untracked leftovers (old local results,
wandb folders, caches) were moved to `_archive_untracked/`, not deleted.

**Not changed, on purpose.** (i) The engine still carries options the paper does not use (LSTM encoder, GAE critic, CVaR reward,
agent-state inputs, Optuna tuning); pruning them would change code under the paper's results. (ii) `ppo.py` still holds environment,
learning and evaluation in one file. (iii) **The HPC copy is untouched** and keeps running the queued chains from the old layout;
`docs/hpc.md` has the plan for putting the new layout beside it and for the final switch once the queue is empty.
(iv) Nothing was committed.

**Verification (local, tiny CPU runs, 2 threads, one at a time; about 15 minutes in total).** The committed tree (tag) and the new tree
were run on the same tiny problems (24 / 12 / 12 paths, 6 iterations, the paper's flags) and every output compared file by file:
bank files, `summary.json`, trial logs, weights, per-path test arrays, baseline JSON.

| Case | Result |
|---|---|
| BSM call single, 1% and 0.1% | identical |
| BSM call Actor–Inactor from the single (main recipe) | identical |
| BSM call gate (`--inactor-proposal`) | identical |
| BSM call transfer (`--freeze-actor`, actor from the 0.1% single) | identical |
| BSM down-and-out band net | identical |
| BSM cash-or-nothing parity pair (from scratch, `--tune-joint-lr`) | identical |
| Heston call single, Heston cash-or-nothing single | identical |
| Model band: BSM call, BSM down-and-out, Heston call, BSM cash-or-nothing on `sp500x` | identical |
| 25 market banks (BSM and Heston, all three contracts) | identical |

The parity pair first showed inactor weights differing by 3e-6 (validation and test numbers equal); it had run while another local
test was running. Rerun alone it is identical, so that was thread-timing rounding, not code. Not covered locally: the Heston barrier
bank (nested Monte Carlo, too heavy for the laptop; the pricer itself is unchanged byte for byte and `tests/test_pricing.py` runs a
small barrier check) and the long resume tests (`tests/test_resume.py`, the chains in `tests/test_new_modes.py`), which belong on a
GPU node: `experiments/slurm/check_tests.slurm`.

Also checked: `tests/test_pricing.py` passes; the four quick checks of `tests/test_new_modes.py` pass; the README's toy command runs;
`collect_runs.py` run on the HPC returns the same 573 runs as the scratchpad collector (finished runs differ only in the new `followup`
group and `table` flag); `main_table.py`, `shrink_check.py`, `hold_analysis.py` and both figure scripts run from the repository (all 17
figures written to a scratch folder, the paper's figures untouched); `make_main.sh` generates a correct job script in the new layout;
`uv lock` regenerated `uv.lock` for the reduced dependency list (torch, numpy, scipy, pandas, matplotlib pins unchanged).
