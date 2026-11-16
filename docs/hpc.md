# Running on the HPC (QMUL Apocrita)

Operational notes: where things are, how runs are launched and watched, and the traps already hit. The experiments
themselves are described in `experiments/README.md`; decisions and results are logged in `TRAINING.md`.

## House rules
- **Training runs on the HPC GPUs, not on the laptop.** Locally: reading results, tables, figures, and tiny CPU checks
  (a few paths, 2 threads) when asked.
- **Never store or type the HPC password in a script or a note.** The user opens the SSH connection; commands go through
  the control socket (`ssh -S ~/.ssh/hpc2.sock hpc2 '<command>'`). In Warp use `/usr/bin/ssh` to open it.
- **Cancel by job id only** (`scancel <id>`), never `scancel -u`.
- **Two gpushort slots, always full.** See "Partitions" below.
- Say which commands run on the **Mac** and which on the **HPC** when handing commands over.

## Facts
| item | value |
|---|---|
| login | `acw796@login.hpc.qmul.ac.uk` (SSH key passphrase, then password) |
| repository on the HPC | `/data/home/acw796/DeepHedging` (`$DEEP_HEDGING_REPO` overrides it in every script) |
| environments | `.venv` (default) and `.venv_v100` (torch build with V100 kernels); the scripts pick by GPU type |
| modules | `module load cuda/12.4.0-gcc-12.2.0` and `module load python/3.11.7-gcc-12.2.0` |
| Slurm | account `pilot_sae_gpu`, partition `sae` (quota 12 GPUs), 1 GPU, 8 cores, 1 hour per job |
| bank cache | `.market_cache/` in the repository (`$HEDGING_ENGINE_CACHE` overrides it) |
| results | `experiments/results/<batch>/<run>/` |

A non-login shell has no Slurm on its path: wrap remote commands as `bash -lc '...'`.

## State of the HPC copy (2026-11-16): still the old layout
The repository was restructured on 2026-11-16 (engine in `hedging_engine/`, experiments in `experiments/`). **The HPC
copy has not been touched**: it still has `deep_hedging_v0/fast_hedging` and `deep_hedging_v0/experiments`, and the jobs
queued there resubmit themselves from those paths. Do not rsync with `--delete`, and do not move anything on the HPC
while those jobs are queued.

The new layout can be put next to the old one at any time, because nothing in it overlaps the old paths:

```bash
# Mac: push the new code (adds hedging_engine/, experiments/, tests/, docs/ beside deep_hedging_v0/)
rsync -avz --progress --exclude='.venv*' --exclude='__pycache__/' --exclude='.git/' --exclude='.DS_Store' --exclude='.market_cache/' --exclude='_archive_untracked/' --exclude='experiments/results/' --exclude='paper_notes/' ~/Desktop/PHD/DeepHedging/ acw796@login.hpc.qmul.ac.uk:/data/home/acw796/DeepHedging/

# HPC: one results folder and one bank cache for both layouts
cd /data/home/acw796/DeepHedging
ln -s ../deep_hedging_v0/experiments/results experiments/results
ln -s .fast_hedging_cache .market_cache
cp deep_hedging_v0/experiments/option_parameter_estimation/option_data/sp500_202[23].json.bz2 experiments/calibration/option_data/
```

Then run the test suite once in the new layout (`cd experiments/slurm && mkdir -p logs && sbatch check_tests.slurm`;
the log must show `ALL PASSED` three times). After that, new jobs are generated in `experiments/slurm` and old jobs keep
running from `deep_hedging_v0`. Bank file
names depend only on the market specification and the seed, so both layouts hit the same cached banks. Once the old queue
is empty: replace the `experiments/results` link by the real folder (`rm experiments/results && mv
deep_hedging_v0/experiments/results experiments/results`), do the same for the cache, and remove `deep_hedging_v0/`.
`experiments/analysis/collect_runs.py` reads either layout.

## How a training run is executed
Every run is a chain of one-hour jobs. `experiments/slurm/make_main.sh <recipe> ...` writes `run_<name>.slurm`; the job
trains for 50 minutes (`--time-budget-min 50`), saves its full state under `experiments/results/main/<name>/resume/` and
exits with code 75; the script then resubmits itself. Exit 0 prints `RUN FINISHED` (and launches the next stage where
the recipe defines one); any other code stops the chain and leaves the state on disk.

```bash
# HPC
cd /data/home/acw796/DeepHedging/experiments/slurm
sbatch $(bash make_main.sh single heston doc 1 2 3e-4)          # one run
while read -r args; do sbatch $(bash make_main.sh band $args); done < jobs/band_jobs.txt    # a listed batch
```

`jobs/*.txt` hold the argument lists of the batches that were submitted for the paper (see `experiments/README.md`).

## Partitions: sae pool, two gpushort slots
- Scripts are submitted to `sae` only. Never submit to `gpushort,sae`: once two of our jobs hold gpushort's per-user cap
  of 2 GPUs, the QOS check blocks every job that lists gpushort.
- gpushort gives this account 2 GPUs at all times. Whenever work is pending, exactly two of our jobs must sit on
  gpushort (pending or running), moved by hand from the sae pool: `scontrol update JobId=<id> Partition=gpushort`.
  `experiments/slurm/feed_gpushort.sh` (kept in the HPC home directory) does this; run it every ten minutes and after
  every reconnect: `ssh -S ~/.ssh/hpc2.sock hpc2 'bash -l ~/feed_gpushort.sh'`. It counts pending and running jobs, so
  it never parks more than two. A resubmitted chain link goes back to `sae`, so the slots empty again and again.
- Short evaluation jobs go to gpushort first: they finish inside the hour.
- sae is often saturated; pending reason `Priority` means waiting behind other users, not a fault.

## Watching
```bash
squeue -u acw796 -o "%i %P %T %j %M %r"                 # queue
tail -f experiments/slurm/logs/mn_<name>.<jobid>.out    # one job
tail experiments/results/main/<name>/log.txt            # one run, across its jobs
sacct -u acw796 -S now-24hours -X -o JobID,JobName%50,State,Elapsed | grep -v COMPLETED    # anything that failed
```
Collect every run into one table (Mac):
```bash
ssh -S ~/.ssh/hpc2.sock hpc2 'bash -lc "cd ~/DeepHedging && python3 -"' < experiments/analysis/collect_runs.py > report.json
```
then split `report.json` into `experiments/results/collected/runs.json` and `curves.json` and run
`python experiments/analysis/main_table.py`.

Pull per-path outputs for the figures (Mac; no weights):
```bash
rsync -avz --exclude='*.pt' --exclude='resume/' --exclude='trials/' acw796@login.hpc.qmul.ac.uk:/data/home/acw796/DeepHedging/experiments/results/main_baselines ~/Desktop/PHD/DeepHedging/experiments/results/
```

## Traps already hit
- **GPU driver.** A node with an old driver made PyTorch fall back to the CPU silently (16 s per iteration). Every job
  script now stops with exit code 3 when no GPU is usable.
- **V100 cards** need `.venv_v100`; the scripts choose it from `nvidia-smi`.
- **Mixed GPUs.** A chain can change GPU type between links, so runs are not bit-reproducible across hardware.
- **A broken chain** (node failure, hard timeout): resubmit the same script; it continues from the last save, at most
  100 iterations back. The chain stops itself after 300 links.
- **Stopping a run:** `scancel <id>`. A job past its training can still resubmit once, so look at `squeue` afterwards.
- **Priority of a chain.** A resubmitted link reads the `#SBATCH` lines of the script on disk, so a nice value must be
  set there (`#SBATCH --nice=N`); `scontrol update nice` lasts for one link only. A released held job keeps its old submit
  time and therefore a high age priority: set its nice before releasing it.
- **The `logs/` folder must exist before `sbatch`**, or the job dies without a log (`make_main.sh` creates it).
- **`sbatch *.slurm` submits only the first file** and passes the rest as arguments. Loop over the files.
- **rsync never deletes on the HPC.** Renamed or removed local files linger there; clean them up by hand. Never use
  `--delete`: it would wipe results and logs that exist only on the HPC.
- **Do not push engine code while jobs are queued**: the next chain link would continue a run with different code.
- **zsh and remote one-liners.** `!` inside an inline remote command is expanded by the local zsh; keep such logic in a
  script on the HPC (that is why `feed_gpushort.sh` exists).
- **Banks.** A `BANK_VERSION` change rebuilds every bank; the Heston barrier bank takes minutes on a GPU
  (`experiments/slurm/bank_heston2000.slurm` pre-builds the Heston banks).
