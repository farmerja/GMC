#!/bin/bash
# 2026-11-22 (v2: smooth table on L nodes at the range ends, no clamping at 1 cm): retrain only the scaled boundary exit flow (exit_B) of the size GMC against a SMOOTH sigma(E, L) exit
# scale (spec.sigx_*, from all 40 files); every other model copied from size_gmc_water_100um-1cm_30MeV. Why: the
# 4-file 8x40 table was noisy (+-5-7 % at 20 MeV) and exit_B inherited it (scatter_width_scan.py). Resumable.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
true
D=../ElectronTransport/sweep_water_100um-1cm_30MeV
.venv/bin/python ml/size_stream.py gmc --data_dir ${D}_42 --n_files 40 --val_npz ${D}_7/part_0000.npz \
    --out ml/models/size_gmc_water_100um-1cm_30MeV_sx2 --stop --init_from ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth \
    --train_only exit_B --smooth_exit_sigma --spec_files 40 "$@" >> logs_size/train_gmc_sx2.log 2>&1
echo "gmc_sx2 exit $?"
