#!/bin/bash
# 2026-11-22: second independent analog-MC Drumm reference (true circle, ESTAR default physics) for the noise floor.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
../ElectronTransport/.venv/bin/python validation/drumm_cylinder.py mc --geom circle --n 8 --N 500000 --seed 4123 --src_seed 2011 \
    --out validation/results/drumm_water_cylinder/mc_circle_n8_500k_s4123_src2011_estar.npz > logs_size/mc_circle2.log 2>&1
echo "mc exit $?"
