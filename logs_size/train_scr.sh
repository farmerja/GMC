#!/bin/bash
# 2026-11-22 v2 (lr 5e-4, spread floor 5e-4 cell; v1 at lr 1e-3 spiked): size-conditioned SCR kernel (water, 100 um - 1 cm, 1 keV - 30 MeV), streamed. Resumable as train_gmc.sh.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
D=../ElectronTransport/sweep_water_100um-1cm_30MeV
.venv/bin/python ml/size_stream.py scr --data_dir ${D}_42 --n_files 40 --val_npz ${D}_7/part_0000.npz \
    --out ml/models/size_scr_water_100um-1cm_30MeV_v2 --wait --steps 320000 "$@" >> logs_size/train_scr.log 2>&1
echo "scr exit $?"
