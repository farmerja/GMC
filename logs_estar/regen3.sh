#!/bin/bash
# 2026-11-17: the remaining *_estar 100 um sets (regen.sh mangled its arguments; regen3 = what regen2 had left when it was stopped). Run alone: 10M peaks 19-25.5 GB RAM.
cd /home/aubrey/PycharmProjects/ElectronTransport
L=/home/aubrey/PycharmProjects/GMC/logs_estar
export PYTHONUNBUFFERED=1
run() {  # material tag N seed label extra...
  local mat=$1 tag=$2 N=$3 seed=$4 lab=$5; shift 5
  local t0=$(date +%s)
  .venv/bin/python generate_analog_batches.py --N $N --seed $seed --out analog_${tag}_100um_1keV-10MeV_log_${lab}_${seed}_estar \
    --E_low 1e-3 --E_high 10 --box 1e-2 --log_energy --photons_irrelevant --simulated_only --batch_size 250000 \
    --material $mat "$@" > $L/gen_${tag}_100um_${lab}.log 2>&1
  echo "$mat 100um $lab exit $? $(( $(date +%s) - t0 )) s"
}
run Cu cu 10000000 42 10M --subcell_bins 64
run Al al 2000000 7 2M --subcell_bins 64
run Al al 10000000 42 10M --subcell_bins 64
run Water water 10000000 42 10M
echo ALL DONE
