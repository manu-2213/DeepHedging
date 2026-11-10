#!/bin/bash
# v2.4 fast transformer ('tfpath', one causal pass per path; see networks.Encoder) trained to convergence,
# with and without GRPO, both fees, seeds 1 and 2: the same arguments as the tf_* / tfgrpo_* runs except --arch.
# Their controls are the window-transformer runs of the same name without the 'p' (tf_fee01_s1 -> tfp_fee01_s1).
# Priority: submit_fast.sh holds the pending v24 chain jobs, then submits check_fast (tests) and the runs.
# The 4th seed-1 run releases the held jobs when it starts (or check_fast does, if the tests fail), so the
# release happens on the HPC and does not depend on anyone watching.
set -euo pipefail
cd "$(dirname "$0")"
rm -f run_tfp*.slurm check_fast.slurm submit_fast.sh
python3 - <<'PY'
import re, pathlib
here = pathlib.Path(".")
order = []
for s in (1, 2):
    for src in (f"tf_fee01_s{s}", f"tfgrpo_fee01_s{s}", f"tf_fee1_s{s}", f"tfgrpo_fee1_s{s}"):
        dst = src.replace("tf", "tfp", 1)
        text = (here / f"run_{src}.slurm").read_text()
        assert text.count("--arch transformer") == 1, src
        text = text.replace("--arch transformer", "--arch tfpath").replace(src, dst)
        release = ('\n# priority hand-over: this run releases the jobs submit_fast.sh held\n'
                   'if [ -f "$SLURM_SUBMIT_DIR/held_for_fast.txt" ]; then\n'
                   '  scontrol release $(cat "$SLURM_SUBMIT_DIR/held_for_fast.txt") && rm -f "$SLURM_SUBMIT_DIR/held_for_fast.txt"\n'
                   'fi\n') if dst == "tfpgrpo_fee1_s1" else ""
        text = text.replace("\nOUT=", release + "\nOUT=", 1)
        (here / f"run_{dst}.slurm").write_text(text)
        order.append(dst)
pathlib.Path("fast_order.txt").write_text("\n".join(order) + "\n")
PY
sed -e 's/v24_check/v24_check_fast/' -e 's/^python -u -m fast_hedging.tests.test_new_modes$/python -u -m fast_hedging.tests.test_new_modes || { rc=$?; [ -f "$SLURM_SUBMIT_DIR\/held_for_fast.txt" ] \&\& scontrol release $(cat "$SLURM_SUBMIT_DIR\/held_for_fast.txt") \&\& rm -f "$SLURM_SUBMIT_DIR\/held_for_fast.txt"; exit $rc; }/' check_new_modes.slurm > check_fast.slurm
cat > submit_fast.sh <<'EOS'
#!/bin/bash
# Usage, from this folder: bash submit_fast.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
ids=$(squeue -u "$USER" -h -t PD -o "%i %j %r" | awk '$2 ~ /^v24_/ && $2 !~ /bench/ && $3 !~ /JobHeldUser/ {print $1}' | paste -sd,)
if [ -n "$ids" ]; then scontrol hold "$ids"; echo "$ids" > held_for_fast.txt; echo "held: $ids"; fi
check=$(sbatch --parsable check_fast.slurm); echo "check_fast: $check"
while read -r run; do
  echo "$run: $(sbatch --parsable --dependency=afterok:$check run_$run.slurm)"
done < fast_order.txt
EOS
