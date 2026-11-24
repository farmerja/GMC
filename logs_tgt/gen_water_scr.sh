#!/bin/bash
# 2026-11-23: targeted water SCR top-up. uniform:0.5 / uniform:1 Drumm maps show an in-cell pattern (cell edges
# -1.5..-3 %, centre +1.5 %, the same at K32, MC fold noise 0.4 %): a kernel placement error in 0.25-2 mm cells at
# Drumm energies. C: L 0.25-2 mm, E 5-30 MeV, half near-normal, WITH sub-cell images. Seeds 70 (train) / 90
# (held-out), far mod 100 from all other datasets (chunk seeds are seed + 100 x batch + chunk). Resumable.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
for spec in "90 1" "70 ${NC:-8}"; do
  set -- $spec
  .venv/bin/python generate_size_sweep.py --out_dir sweep_water_0.25-2mm_5-30MeV_fwd_$1 --seed $1 --n_files $2 \
      --L_low 0.025 --L_high 0.2 --E_low 5 --E_high 30 --narrow_frac 0.5 --narrow_sigma_deg 10 \
      >> /home/aubrey/PycharmProjects/GMC/logs_tgt/gen_water_C.log 2>&1 || { echo "C seed $1 FAILED"; exit 1; }
  echo "C seed $1 done"
done
echo "water C gen done"
