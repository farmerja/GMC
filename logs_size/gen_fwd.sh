#!/bin/bash
# 2026-11-23: targeted GMC-only top-up for the large-cell regime (Drumm beam through 3-10 mm cells): L 3-10 mm,
# E 5-30 MeV (log-uniform), half the source near-normal (polar |N(0, 10 deg)|), no sub-cell images. Resumable.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
.venv/bin/python generate_size_sweep.py --out_dir sweep_water_3-10mm_5-30MeV_fwd_43 --seed 43 --n_files ${NFILES:-10} \
    --L_low 0.3 --L_high 1.0 --E_low 5 --E_high 30 --no_subcell --narrow_frac 0.5 --narrow_sigma_deg 10 \
    >> /home/aubrey/PycharmProjects/GMC/logs_size/gen_fwd.log 2>&1
echo "gen_fwd exit $?"
