#!/bin/bash
# 2026-11-23: size-conditioned Ta GMC + SCR (cells 10 um - 1 cm, E 1 keV - 20 MeV), the Al / Cu recipe
# (logs_lockwood_size/train.sh): GMC and SCR side by side on the GPU, after the Ta data. Resumable: rerun.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
L=logs_ta
until grep -q "ta gen done" $L/gen.out 2>/dev/null; do grep -q FAILED $L/gen.out 2>/dev/null && exit 1; sleep 60; done
D=$E/sweep_ta_10um-1cm_20MeV_42; V=$E/sweep_ta_10um-1cm_20MeV_7/part_0000.npz
NF=$(ls $D/part_*.npz | grep -vc subcell)
( .venv/bin/python ml/size_stream.py gmc --data_dir $D --n_files $NF --val_npz $V --out ml/models/size_gmc_ta_10um-1cm_20MeV \
      --stop --smooth_exit_sigma --spec_files 12 >> $L/train_gmc.log 2>&1; echo "ta gmc exit $?" ) &
( .venv/bin/python ml/size_stream.py scr --data_dir $D --n_files $NF --val_npz $V --out ml/models/size_scr_ta_10um-1cm_20MeV \
      --steps 320000 >> $L/train_scr.log 2>&1; echo "ta scr exit $?" ) &
wait
echo "ta train done"
