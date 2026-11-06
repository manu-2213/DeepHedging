# Handoff: HPC sync and job submission (DeepHedging v2.1)

You already know the code (`deep_hedging_v0/fast_hedging/NOTES.md`, `paper_notes/findings.tex`).
This file covers the operational side: syncing with the HPC, launching jobs, checking them, and the
traps already hit.

## House rules
- **Never run training, simulations, profiling or tests on the laptop.** All compute runs on the
  QMUL HPC GPUs. Verify code by reading; hand the user slurm jobs instead. Trivial reads (cat, grep, ls)
  are fine. Compiling `paper_notes/findings.tex` with `/Library/TeX/texbin/pdflatex` (twice) is fine.
- **Document every finding or fix** in `paper_notes/findings.tex` (gitignored, recompile the PDF) and in
  memory. Publication quality matters.
- The user cannot run things mid-task for you. Give them exact commands, saying which ones run on the
  **Mac** and which on the **HPC**.

## Current state (as of 2026-11-04)
- **Results caveat:** 23 of 27 v2.1 runs were finished at the last pull, but early stopping censored most of
  the training, so net comparisons are not valid yet. See "Interim results" in `paper_notes/findings.tex`.
- **v2.1 code:** in the working tree, **uncommitted**, and the same code has run on the HPC.
- **v1 results:** still on the HPC in `deep_hedging_v0/experiments/results/liquid_1_5/`. They used the
  old tracking reward, single vs double nets only, and the degenerate $1 digital. Some v1 Heston double
  runs may still be running, and `heston_doc_rnn_single` never started.
- **Heston fix:** the adaptive `phi_max` fix ran on the HPC and all Heston banks show zero warnings. The
  smoke-test `ALL PASSED` line has not been read from the Mac, because `_smoke_*` and logs are not pulled.
  The six BSM DOC validation warnings are false alarms (a 1-cent put that no path reaches).
- **Slurm folder on the Mac:** the v1 folder `slurm/liquid_1_5` was archived to
  `slurm/old_pipeline/liquid_1_5_run1`. The HPC copy still has the old path until the user moves it
  (step 2 below).

## Chained 1-hour jobs (v2.2, slurm folder `v2_2_5`)
Use this for new runs. Each job works for 50 minutes on `gpushort`, saves its state to `results/v2_2_5/<name>/resume/`,
and resubmits itself until the log prints `RUN FINISHED`. The engine keeps the old behaviour without `--time-budget-min`.

**From the QMUL docs (read 2026-11-05):** a 1-hour request gives access to `gpushort`, and any user may submit there.
The docs say queue time depends mostly on cores and RAM, and they recommend checkpointing for long work. They describe
GPU jobs as 8 cores and 11 GB per core, so these jobs ask for 88 GB. The docs also describe `gpushort` as the place for
testing, and they do not give its per-user limits.

**1. On the HPC, before pushing:** check that no job is queued or running. A queued job would start on the new code.
```bash
squeue -u acw796
```
**2. On the Mac:** push the code with the rsync command in step 3 of the launch sequence below.

**3. On the HPC, check that the job is accepted and see the limits:**
```bash
cd /data/home/acw796/DeepHedging/deep_hedging_v0/experiments/simulations/slurm/v2_2_5
mkdir -p logs
sbatch --test-only pilot_a_production.slurm
scontrol show partition gpushort
sacctmgr show assoc user=acw796 format=account,partition,qos,maxjobs,maxsubmit
```
If `sbatch` refuses the default account, regenerate with `ACCOUNT=pilot_sae_gpu bash make_slurm.sh`.

**4. On the HPC, submit the checks and the pilot.** The pilot starts only if the checks pass:
```bash
bash submit_all.sh pilot
```
`logs/check_resume.<id>.out` must end with `ALL PASSED`. If it fails, the pilot jobs stay pending: `scancel` them. Read the pilot before submitting `main`.

**Stopping:** no iteration cap and patience 2000 iterations, which is 5 old epochs.

**Rates and stages:** main runs have no Optuna tuning: actor and joint rate 1e-5, inaction net 1e-3, seed 0 plus final seeds 1 2 3. Submit `first` (BSM call), check it, then `rest`. At 1.63 s per iteration the minimum is about 3.6 GPU hours per single or band run and 7.2 per double run, about 86 for all 18. Add seeds with `FINAL_SEEDS="1 2 3 4 5 6 7 8 9 10" bash make_slurm.sh` and resubmit: finished seeds are skipped.

**Per-user GPU limit:** pending `gpushort` jobs show `QOSMaxGRESPerUser`, and two pilots ran at once (seen 2026-11-05). If the cap is 2 GPUs, the 86 GPU hours take about 2 days.

**Chain traps:**
- **GPU driver:** `gpushort` node `sbg2` has driver 12.9 (seen 2026-11-05), too old for the PyTorch build then in `.venv`. PyTorch fell back to the CPU at about 16 s per iteration, without an error. The job scripts now stop with exit code 3 when no GPU is usable. The fix is `torch==2.11.0+cu128`, the newest CUDA 12.8 build on 2026-11-05. It replaces 2.15.0+cu132, which v2.1 ran on. To go back, install `torch==2.15.0+cu132` from `https://download.pytorch.org/whl/cu132`.
- **GPU type:** `sbg2` holds V100 cards, and `torch 2.11.0+cu128` has no code for them (`no kernel image is available`). The job scripts ask for `ampere|hopper` through `--constraint`. Override with `CONSTRAINT=... bash make_slurm.sh`.
- **V100 cards:** `.venv_v100` (made 2026-11-05) holds `torch 2.11.0+cu126`, which includes V100 code (`sm_70`), plus numpy, scipy, optuna and pandas only. On the HPC, `probe_v100.slurm` is a copy of `probe_pack.slurm` that uses it with `--constraint=v100`, one probe per 16 GB card, at about 1.15 s per iteration. It runs `fast_hedging` only, not the old pipeline.
- **Stale pilots:** rsync never deletes, and `submit_all.sh` globs `pilot_*.slurm`. Remove old pilot files on the HPC whenever the pilot list changes.
- **Exit codes:** 75 means continue, 0 means finished, anything else stops the chain and leaves the state on disk.
- **A broken chain:** a node failure or hard timeout stops the chain. Resubmit the same file: it continues from the last save, at most 100 iterations back.
- **Stopping a run:** `scancel -n <name>`. A job already past its training can still resubmit once, so check `squeue` afterwards.
- **Results are per run:** `job_count` in the results folder counts the jobs, and the chain stops at 1000.
- **Mixed GPUs:** `gpushort` holds several GPU types, so a run may change type between jobs. Exact reproducibility needs the same hardware.
- **Banks:** chained jobs reuse the HPC bank cache. A `BANK_VERSION` change needs `v2_1_5/build_banks.slurm` first, as its 8 hours do not fit one job.
- **Old pipeline:** it has no resume support and stays frozen, so its jobs keep their long walltime.

## HPC facts
| item | value |
|---|---|
| login | `acw796@login.hpc.qmul.ac.uk` (asks for an SSH key passphrase, then a password) |
| repo on HPC | `/data/home/acw796/DeepHedging` |
| repo on Mac | `~/Desktop/PHD/DeepHedging` |
| venv | `DeepHedging/.venv` (every slurm file runs `source ../.venv/bin/activate` from `deep_hedging_v0/`) |
| modules | `module load cuda/12.4.0-gcc-12.2.0` and `module load python/3.11.7-gcc-12.2.0` |
| slurm | account `pilot_sae_gpu`, partition `sae`, 1 GPU, 240 h maximum walltime |
| bank cache | `DeepHedging/.fast_hedging_cache/` (`BANK_VERSION` 4: everything rebuilds) |
| v2 results | `deep_hedging_v0/experiments/results/v2_1_5/<name>/` |

The local `.venv/bin/python` is a dangling symlink, so nothing in the project imports on the Mac.

## Launch sequence
**1. Before syncing:** make sure no v1 job is still running or queued (`squeue -u acw796` on the HPC).
A v1 job that restarts after the sync would run the new code with new defaults. `scancel` them, or wait.

**2. On the HPC, archive the v1 slurm folder** so it matches the Mac:
```bash
cd /data/home/acw796/DeepHedging/deep_hedging_v0/experiments/simulations
mkdir -p slurm/old_pipeline && mv slurm/liquid_1_5 slurm/old_pipeline/liquid_1_5_run1
rm -f *_mlp.py            # MLP scripts were deleted locally; rsync never deletes remotely
```

**3. On the Mac, push the code.** Use one line: pasted backslash continuations have broken before.
```bash
rsync -avz --progress --exclude='.venv/' --exclude='venv/' --exclude='env/' --exclude='__pycache__/' --exclude='.git/' --exclude='wandb/' --exclude='.ipynb_checkpoints/' --exclude='.DS_Store' --exclude='.fast_hedging_cache/' --exclude='.simulation_cache/' --exclude='learning_folder/' --exclude='inactor_search_results*/' --exclude='deep_hedging_v0/experiments/results/' --exclude='paper_notes/' ~/Desktop/PHD/DeepHedging/ acw796@login.hpc.qmul.ac.uk:/data/home/acw796/DeepHedging/
```
Add `-n` for a dry run first. Run it on the **Mac**, not the HPC: `~/Desktop` doesn't exist there.

**4. On the HPC, make sure the venv works.** If it's broken or new packages are missing, rebuild it:
```bash
cd /data/home/acw796/DeepHedging
module load python/3.11.7-gcc-12.2.0
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r deep_hedging_v0/requirements.txt
python -c "import torch, optuna, scipy, jax, gymnasium, pandas; print(torch.__version__, optuna.__version__)"
```

**5. On the HPC, run the smoke test.** It takes up to 6 h:
```bash
cd /data/home/acw796/DeepHedging/deep_hedging_v0/experiments/simulations/slurm/v2_1_5
mkdir -p logs && sbatch smoke_test.slurm
```
Then check `logs/smoke_test.<id>.out` for:
- **`ALL PASSED`:** the parity test against the original environments and the complex128 Heston reference.
- **`SMOKE TEST FINISHED`:** every net and reward, the rescaled digital, a Heston call run and the Heston
  barrier Monte Carlo ran end to end.
- **`WARNING bank[...]` lines:** martingale or one-step drift problems. These matter most for Heston.
- **`s/it` in the `[actor]` and `[joint]` lines:** this gives real GPU speed, so you can estimate whether
  jobs fit 240 h.
- **`logs/smoke_test.<id>.err`:** tracebacks.

If it fails, the most likely places are the new torch code in `pricing.heston_probabilities` (binned
grids, `budget` chunking) and `heston_down_out_call` (pre-drawn `z` tensor).

**6. On the HPC, submit**, from the same folder, only after the smoke test passes:
```bash
bash submit_all.sh            # bank build + 18 main runs (mean_cvar, 3 nets x 6 products)
bash submit_all.sh ablation   # bank build + 6 tracking-reward ablations (BSM call/doc)
bash submit_all.sh lambda     # bank build + 3 GAE-lambda=1 ablations (BSM call)
bash submit_all.sh all        # everything at once (one bank build)
```
Each call submits its own bank build. Prefer `all`, or wait for the first bank job to finish before
starting a second group: two bank builds at once waste GPU time, though cache writes are atomic.
After the bank job finishes, read its log: it greps the `WARNING` lines.

## Traps already hit
- **`sbatch *.slurm` submits only the first file**, and passes the rest as arguments. A `for f in *.slurm`
  loop also submits `smoke_test.slurm` and `build_banks.slurm` together with the runs. Always use
  `submit_all.sh`.
- **The `logs/` folder must exist before `sbatch`**, or the job dies without a log. `submit_all.sh`
  creates it; for the smoke test, run `mkdir -p logs` first.
- **rsync never deletes on the HPC.** Removed or renamed local files linger remotely, and stale slurm
  files can get submitted by mistake. Clean them up with explicit `rm` or `mv` on the HPC. Don't use
  `--delete`: it would wipe HPC-only results and logs.
- **Files keep their original dates after rsync** (`-a`), so check for new files, not timestamps.
- **The Mac's rsync is Apple's openrsync.** If it errors against the HPC, `brew install rsync`.
- **Syncing during runs:** copying only analysis files (`fast_hedging/report.py`, `fast_hedging/analysis.py`)
  is safe, because training never imports them. Pushing engine code while jobs are queued is not.

## Monitoring and results
- **Queue and logs:** `squeue -u acw796`. Per-job logs are in `v2_1_5/logs/<name>.<jobid>.out` and `.err`;
  per-run logs are in `results/v2_1_5/<name>/log.txt`.
- **Resuming:** resubmitting the same slurm file continues its Optuna study (journal file) and skips
  finished final seeds. Use this if a job hits 240 h or dies.
- **Report** (HPC login node; reads files only, takes seconds):
  ```bash
  cd /data/home/acw796/DeepHedging/deep_hedging_v0 && source ../.venv/bin/activate
  python -m fast_hedging.report experiments/results/v2_1_5 --paired
  ```
  Base claims about the method on the **SEED-LEVEL** table. The paired per-seed table only covers
  test-path noise.
- **Pulling results to the Mac** for deeper analysis (`fast_hedging/analysis.py`, pandas), without the
  weights:
  ```bash
  rsync -avz --progress --exclude='*.pt' acw796@login.hpc.qmul.ac.uk:/data/home/acw796/DeepHedging/deep_hedging_v0/experiments/results/ ~/Desktop/PHD/DeepHedging/deep_hedging_v0/experiments/results/
  ```
  Reading JSON and npz files locally is fine. Training is not. rsync only creates the last folder of the
  target path, so run `mkdir -p` on it first if `results/` does not exist on the Mac. Run the report locally with
  `/usr/local/bin/python3 -m fast_hedging.report experiments/results/v2_1_5` from `deep_hedging_v0/`.
- **v1 comparison:** the v1 results in `results/liquid_1_5` use old run names without a reward suffix.
  `analysis._meta` treats those as `tracking`.

## Other notes
- `hpc_notes.txt` is tracked in git but holds no credential (checked 2026-11-01: it only says where the
  HPC password is issued). The slurm files do contain the cluster hostname and the login in a default path.
- Only commit or push when the user asks.
