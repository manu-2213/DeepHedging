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

## Current state (as of 2026-11-01)
- **v2.1 code:** in the working tree, **uncommitted**. Nothing from v2.0 or v2.1 has run on the HPC yet.
  Only `py_compile` was run locally.
- **v1 results:** still on the HPC in `deep_hedging_v0/experiments/results/liquid_1_5/`. They used the
  old tracking reward, single vs double nets only, and the degenerate $1 digital. Some v1 Heston double
  runs may still be running, and `heston_doc_rnn_single` never started.
- **Unverified Heston fix:** the adaptive `phi_max` fix has not run on torch or GPU. Do not trust any
  Heston result until the smoke test prints `ALL PASSED` and the bank logs show no one-step drift
  warnings.
- **Slurm folder on the Mac:** the v1 folder `slurm/liquid_1_5` was archived to
  `slurm/old_pipeline/liquid_1_5_run1`. The HPC copy still has the old path until the user moves it
  (step 2 below).

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
  Reading JSON and npz files locally is fine. Training is not.
- **v1 comparison:** the v1 results in `results/liquid_1_5` use old run names without a reward suffix.
  `analysis._meta` treats those as `tracking`.

## Other notes
- `hpc_notes.txt` is tracked in git but holds no credential (checked 2026-11-01: it only says where the
  HPC password is issued). The slurm files do contain the cluster hostname and the login in a default path.
- Only commit or push when the user asks.
