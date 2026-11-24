#!/bin/bash
# 2026-11-24: Ta size model on held-out files: the 1 keV - 20 MeV test file and the 1 keV - 1.5 MeV (Lockwood-range) one
cd /home/aubrey/PycharmProjects/GMC
E=../ElectronTransport; R=validation/results/size_gmc/ta
G=ml/models/size_gmc_ta_10um-1cm_20MeV/secondary_gmc.pth; K=ml/models/size_scr_ta_10um-1cm_20MeV
for t in 20MeV:sweep_ta_10um-1cm_20MeV_7/part_0001.npz 1.5MeV:sweep_ta_10um-1cm_1.5MeV_7/part_0001.npz; do
  n=${t%%:*}; f=$E/${t#*:}
  .venv/bin/python validation/size_cell_compare.py gmc --model $G --test_npz $f --out_dir $R/gmc_$n > logs_ta/eval_gmc_$n.log 2>&1; echo "gmc $n exit $?"
  .venv/bin/python validation/size_cell_compare.py scr --model $K --test_npz $f --out_dir $R/scr_$n --n_transits 20000 > logs_ta/eval_scr_$n.log 2>&1; echo "scr $n exit $?"
done
echo "ta eval done"
