#!/bin/bash
# 2026-11-22: size-sweep data, water 1 keV - 30 MeV, square cells 100 um - 1 cm (log-uniform), sub-cell n = max(64, L/10um).
# Resumable: rerun this script and finished files are skipped. Test set first (small), then training files.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
L=/home/aubrey/PycharmProjects/GMC/logs_size
.venv/bin/python generate_size_sweep.py --out_dir sweep_water_100um-1cm_30MeV_7 --seed 7 --n_files 2 >> $L/gen_test.log 2>&1
echo "test exit $?"
.venv/bin/python generate_size_sweep.py --out_dir sweep_water_100um-1cm_30MeV_42 --seed 42 --n_files ${NFILES:-40} >> $L/gen_train.log 2>&1
echo "train exit $?"
