#!/bin/bash
# 2026-11-22: size-conditioned GMC (water, 100 um - 1 cm, 1 keV - 30 MeV), streamed over the sweep files. Resumable:
# rerun this script to continue; `touch ml/models/size_gmc_water_100um-1cm_30MeV/STOP` stops after the current file.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
D=../ElectronTransport/sweep_water_100um-1cm_30MeV
.venv/bin/python ml/size_stream.py gmc --data_dir ${D}_42 --n_files 40 --val_npz ${D}_7/part_0000.npz \
    --out ml/models/size_gmc_water_100um-1cm_30MeV --stop --wait "$@" >> logs_size/train_gmc.log 2>&1
echo "gmc exit $?"
