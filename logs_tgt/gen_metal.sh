#!/bin/bash
# 2026-11-23: targeted Al / Cu top-up for the Lockwood failures: E 0.2-1.5 MeV only (single large cells near the entry
# energy, and the high-E small-cell chains), the full L range (Cu to 3 mm), half near-normal, WITH sub-cell images
# (the SCR is retrained too). $1 = Al | Cu. Training files (seed 44) + one held-out file (seed 8). Resumable.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
M=$1; m=$(echo $M | tr 'A-Z' 'a-z')
LH=0.6; [ $M = Cu ] && LH=0.3
for spec in "8 1" "44 ${NFILES:-16}"; do
  set -- $spec
  .venv/bin/python generate_size_sweep.py --out_dir sweep_${m}_tgt_0.2-1.5MeV_$1 --seed $1 --n_files $2 \
      --L_low 0.005 --L_high $LH --E_low 0.2 --E_high 1.5 --material $M --narrow_frac 0.5 --narrow_sigma_deg 10 \
      >> /home/aubrey/PycharmProjects/GMC/logs_tgt/gen_$m.log 2>&1 || { echo "$M seed $1 FAILED"; exit 1; }
  echo "$M seed $1 done"
done
