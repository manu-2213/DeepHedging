#!/bin/bash
# keep exactly two of our jobs on gpushort (pending or running); feed from the sae pool, years evals first
export PATH=$PATH:/usr/bin:/opt/slurm/bin:/cm/shared/apps/slurm/current/bin
g=$(squeue -u acw796 -h -p gpushort -o %i | wc -l)
n=$((2-g))
if [ "$n" -le 0 ]; then echo "feed: gpushort full ($g)"; exit 0; fi
cands=$( (squeue -u acw796 -h -t PD -p sae -n mn_years -S i -o %i; squeue -u acw796 -h -t PD -p sae -S i -o %i) | awk '!seen[$0]++' | head -n "$n")
for j in $cands; do scontrol update JobId="$j" Partition=gpushort && echo "feed: moved $j to gpushort"; done
squeue -u acw796 -h -p gpushort -o "feed: gpushort %i %T %j %r"
