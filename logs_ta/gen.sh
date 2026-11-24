#!/bin/bash
# 2026-11-23: size-sweep data for a size-conditioned Ta model (Lockwood SAND79-0414 Ta cases). Square cells 10 um - 1 cm
# (log-uniform, 0.017 - 16.7 g/cm^2; user request: up to 1 cm), E 1 keV - 20 MeV (user request),
# 30 % near-normal sources (--narrow_frac 0.3: the targeted Cu round showed near-normal entries help), sub-cell images.
# Test (2 files, seed 7) + training (N_TRAIN = 40 files, seed 42; user choice for time: ~half the Al/Cu sample density
# in the Lockwood corner, 80 files would match it: 40 / (0.69 x 0.74), the shares of log L in 10 um - 1.2 mm
# and of log E below 1.5 MeV, so the Lockwood corner keeps the
# sample density of the Al/Cu models). The Ta MC references were finished first (they are timed). The earlier 1 keV -
# 1.5 MeV test files (sweep_ta_10um-1cm_1.5MeV_7) are kept as a held-out set for the Lockwood energy range.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
L=/home/aubrey/PycharmProjects/GMC/logs_ta
N_TRAIN=${N_TRAIN:-40}

for spec in "7 2" "42 $N_TRAIN"; do
  set -- $spec
  for k in $(seq 1 $2); do
    .venv/bin/python generate_size_sweep.py --out_dir sweep_ta_10um-1cm_20MeV_$1 --seed $1 --n_files $k \
        --L_low 0.001 --L_high 1.0 --E_low 0.001 --E_high 20 --material Ta --narrow_frac 0.3 >> $L/gen_ta.log 2>&1 || { echo "FAILED"; exit 1; }
  done
  echo "Ta seed $1 done"
done
echo "ta gen done"
