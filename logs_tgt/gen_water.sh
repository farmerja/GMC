#!/bin/bash
# 2026-11-23: targeted water GMC-only top-up for the Drumm failures, half near-normal (polar |N(0, 10 deg)|):
#   A  L 6-10 mm, E 10-30 MeV: the beam edge in ~1 cm cells (+1-1.6 %, the sym:10-cell probe +11-14 % at 3-4 mm)
#   B  L 1-4 mm,  E 5-30 MeV:  the quad:4 beam edge (-1.8 %) and the 4-5 % narrow exit width at 1.2-2.4 mm
# $1 = A | B (run both side by side). One held-out file (seed 10 / 11) + NA = 6 / NB = 6 training files
# (seed 45 / 46). Resumable.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
run() {  # out seed n L_low L_high E_low
  .venv/bin/python generate_size_sweep.py --out_dir $1 --seed $2 --n_files $3 --L_low $4 --L_high $5 --E_low $6 \
      --E_high 30 --no_subcell --narrow_frac 0.5 --narrow_sigma_deg 10 >> /home/aubrey/PycharmProjects/GMC/logs_tgt/gen_water_$7.log 2>&1 \
      || { echo "$1 FAILED"; exit 1; }
  echo "$1 done"
}
if [ $1 = A ]; then
  run sweep_water_6-10mm_10-30MeV_fwd_10 10 1 0.6 1.0 10 A
  run sweep_water_6-10mm_10-30MeV_fwd_45 45 ${NA:-6} 0.6 1.0 10 A
else
  run sweep_water_1-4mm_5-30MeV_fwd_11 11 1 0.1 0.4 5 B
  run sweep_water_1-4mm_5-30MeV_fwd_46 46 ${NB:-6} 0.1 0.4 5 B
fi
echo "water $1 gen done"
