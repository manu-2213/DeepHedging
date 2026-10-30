# BSM Call MLP Inactor Search

The [standalone runner](search_inactive_bsm_hedge_call_mlp.py) and
[experimental training helper](../utils/inactor_search.py) do not modify or invoke
the original simulation's training loop. They reuse the existing BSM environment,
SP500 parameter split, actor factory, and shared configuration defaults. No W&B
account or additional packages are needed beyond the project's Python environment.

## Run Locally

From the repository root, run the full search:

```bash
.venv/bin/python -u deep_hedging_v0/experiments/hp_tuning/search_inactive_bsm_hedge_call_mlp.py \
  --output inactor_search_results
```

The default is **98 configurations**, with one actor warm-up at **0.0001** for
training seed 0. With the repository's default configuration, warm-up is 40 epochs
of 400 batches; the screening budgets are 200, 800, 2400, and 8000 joint batches.
One batch is one complete vectorized episode across all paths and options.
This is a substantial research run, not a short smoke test. Warm-up progress prints
an approximate remaining time. Training is sequential to limit memory use.

For a quick installation/plumbing check:

```bash
.venv/bin/python -u deep_hedging_v0/experiments/hp_tuning/search_inactive_bsm_hedge_call_mlp.py \
  --smoke --output inactor_search_smoke
```

Smoke mode uses tiny budgets and shortened schedules. Its rewards must not be used
to choose a configuration. To test every branch through the final stage, add
`--reduction 1 --finalists 14` to the smoke command.

For a smaller exploratory run, with deliberately shortened actor warm-up:

```bash
.venv/bin/python -u deep_hedging_v0/experiments/hp_tuning/search_inactive_bsm_hedge_call_mlp.py \
  --warmup-epochs 2 --episodes-per-epoch 40 \
  --joint-batches 600 --rungs 200 400 600 \
  --output inactor_search_short
```

The shorter warm-up changes the starting actor and is not a reproduction of the
original training budget. Prefer the full warm-up when investigating the original
reward drop. Environment-variable overrides for shared training defaults are
honored; effective settings are recorded in the manifest.

## What Is Searched

Each strategy is crossed with inactor learning rates:

`3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1`.

Rung-1 results from the earlier low-rate grid (3e-6…3e-3) showed reference,
frozen, and slow_actor improving monotonically with inactor LR, so the default
grid was shifted upward. Unguarded strategies (`reference`, `frozen`,
`slow_actor`, `init*`) may fail loudly at the highest rates; failures are
recorded in `failures/history.jsonl` and the search continues. The
trust-region-style strategies (`trust_region`, `critic_warmup`, `contextual_*`,
`ppo3`, `annealed`) stop a PPO pass when approximate KL exceeds 0.02.

| Strategy | Change being tested |
| --- | --- |
| `reference` | Random gate initialization; actor LR 0.0001, updated every third batch within each epoch. |
| `frozen` | Random gate; freeze the entire warmed actor, including its shared critic features. |
| `slow_actor` | Random gate; reduce joint actor LR to 0.00001. |
| `init01` | Initialize hold probability to 1%; freeze actor for 32 batches, then use LR 0.00001. |
| `init05` | Same, with 5% initial hold probability. |
| `init15` | Same, with 15% initial hold probability. |
| `ramp05` | 5% gate; ramp its activation from zero over 64 batches; freeze actor for 64 batches. |
| `no_entropy` | `ramp05` without the inactor entropy bonus. |
| `trust_region` | `ramp05` with normalized gate advantages, PPO clip 0.05, and an approximate-KL stop at 0.02. |
| `critic_warmup` | Fit the gate critic for 32 batches with the gate off, then ramp; actor frozen for 96 batches. |
| `contextual_frozen` | Gate also sees the actor proposal and proposed change in holding; actor stays frozen. |
| `contextual_joint` | Contextual gate with gradual activation and subsequent slow actor updates. |
| `ppo3` | Three gate PPO passes per rollout, with the trust-region safeguards. |
| `annealed` | Trust-region variant with inactor LR decreasing linearly to 10% of its initial value. |

The contextual variants inherit the trust-region settings. Unless specified
otherwise, post-freeze actor updates occur every third batch within an epoch.
The actor warm-up LR is fixed; it is not part of the search.

An untrained gate can immediately suppress many actor trades, before any gradient
update. Controlled initialization isolates this activation shock. Gradual activation
is included in both sampling probabilities and PPO likelihoods, not applied as an
unrecorded action override. Contextual gates address the absence of the current
holding and proposed trade in the original inactor's observation-only input.

The reference is a **reference-style configuration**, not a bit-for-bit rerun of
the old trainer. The new trainer explicitly selects gate action/log-probability
keys, uses the configured critic coefficient, resets holdings at episode boundaries,
and snapshots current and next observations before the environment reuses its
NumPy state buffer. It retains the original actor networks and tanh action mapping,
including the existing [-1, 1] range. Actor PPO uses the sampled Gaussian action
before tanh rather than trying to invert rounded boundary actions. The shared
transform Jacobian cancels in the likelihood ratio; entropy includes its stable
PyTorch log-Jacobian. The non-finite-gradient check remains enabled.

## Efficient And Fair Comparisons

- Warm up once per training seed. Every trial restores identical actor weights
  **and Adam state** from that checkpoint; gate initialization is seeded consistently.
- Keep approximately the best third after each screening stage. Promoted trials
  continue from their last state; they do not restart. The `reference` trial at
  inactor LR 0.001 is retained through all stages when included in the grid.
- Rank by the best stochastic validation mean reward observed so far. Every
  selection evaluation uses the **fully activated gate**, even during a ramp.
  Batch zero is eligible; a selected checkpoint at zero means training never
  exceeded that initial validation score at the evaluation points.
- Validation uses the training parameter configurations but independent simulation
  seeds. The original held-out parameter split is evaluated only after selection.
  Common market paths and policy seeds are used across candidates. A zero-probability
  dummy gate in actor-only controls preserves the actor's random-number stream.
- Report both stochastic and deterministic held-out behavior. Deterministic gates
  can behave differently from sampled gates, especially near probability 0.5.
- Include the unchanged warm-up actor and an actor-only continuation control.
  The latter gets the same rollout budget, but updates the actor every batch;
  it is not an equal-optimizer-step comparison with every-third-batch variants.
- Fail non-finite trials visibly and record tracebacks. Do not silently replace
  invalid gradients or report failed trials as successful.

Successive halving can reject slow learners. Use `--reduction 1` to disable pruning,
or increase the first rung for a more patient search. The default single training
seed is exploratory; `--seeds 0 1 2` repeats warm-up once per seed and ranks by mean
validation reward across those seeds. This multiplies the workload.

Useful controls include `--inactor-lrs`, `--strategies`, `--num-paths`,
`--eval-paths`, `--validation-seeds`, `--test-seeds`, `--eval-every`, `--threads`,
and `--device`. Use `--dry-run` to print the grid without creating output.
The local default is CPU when CUDA is unavailable; MPS is not used.

## Resume

Repeat the same command with `--resume`:

```bash
.venv/bin/python -u deep_hedging_v0/experiments/hp_tuning/search_inactive_bsm_hedge_call_mlp.py \
  --output inactor_search_results --resume
```

Include any custom flags from the original command. Settings and relevant source/data
fingerprints must match. Existing output directories are otherwise refused.
Checkpoints are written atomically every 200 batches by default and at stage ends;
an interruption may require replaying work since the last checkpoint. Paths and
sampling are seeded by batch index, so CPU continuation is reproducible.
Completed stages are skipped. Failed/pruned trials are not automatically reinstated
after a stage has been finalized.

## Changing The Grid Mid-Search

The trial grid (`--inactor-lrs`, `--strategies`) and pruning knobs (`--reduction`,
`--finalists`, `--threads`) may differ on resume. The shared warm-up checkpoint and
every completed per-trial stage are reused; configurations new to the grid start
from the shared warm-up, and dropped configurations are left on disk but skipped.
All other settings (seeds, budgets, paths, PPO and environment parameters) must
match the original command, or resume is refused with the mismatched keys listed.
A signature history in `manifest.json` keeps provenance across such migrations.

Only the new output directory is written during a search. Existing simulation path
caches, W&B runs, and saved weights are not overwritten. Disk path caching is disabled;
the environment still reuses paths in memory within each epoch.

## Warm-Up Crash Recovery

The initial search collector could retain aliases to the environment's reusable
observation buffer. Later steps overwrote earlier observations, corrupting PPO's
action/observation pairs. The corrected collector snapshots both sides of each
transition; latent-action likelihoods also avoid precision loss when tanh saturates.

For a run created before this correction, restart in a **new directory**. Do not
resume the old warm-up: its weights were trained on corrupted rollouts, and its
source fingerprint intentionally no longer matches. Old results are left intact.

```bash
caffeinate -i .venv/bin/python -u deep_hedging_v0/experiments/hp_tuning/search_inactive_bsm_hedge_call_mlp.py \
  --output inactor_search_results_fixed
```

The screen may turn off; keep the lid open and the laptop plugged in. To resume
this corrected run after a later interruption, repeat this command with `--resume`.
Warm-up still uses actor LR 0.0001 and now prints the fraction of saturated actor
proposals. Regression checks can be run with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s deep_hedging_v0/experiments/hp_tuning -p 'test_inactor_search.py' -v
```

## Reports To Return

Send these files from the output directory:

- `summary.csv`: final stochastic/deterministic reward, CVaR, hold rate, turnover,
  and reward changes relative to both actor controls.
- `activation.csv`: reward changes immediately after enabling each untrained gate,
  measured before joint optimization; includes full and scheduled activation.
- `leaderboard.csv`: validation ranking at every screening budget.
- `best_config.json` and `manifest.json`: selected settings, checkpoint locations,
  complete experiment configuration, versions, and source fingerprint.

Per-trial `history.jsonl` contains training rewards and PPO diagnostics;
`activation.json` and `test.json` retain detailed evaluations. `best.pt` stores the
validation-selected model, while `last.pt` stores the continuation state.
`test_results.csv` preserves results separately for each training seed. Failures,
if any, appear in `failures/history.jsonl`.

Higher reward, or a positive reward delta, is better. `tracking_cvar95` is the mean
of the worst 5% of per-option-path average absolute tracking errors, averaged across
evaluation seeds; lower is better. It is not terminal-P&L CVaR. Turnover is average
absolute change in shares per step, not a monetary transaction-cost measure.
Single-seed results and finite validation searches do not establish robustness.