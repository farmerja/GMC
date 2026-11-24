#!/bin/bash
# 2026-11-23: size-conditioned Ta GMC + SCR on the Lockwood Ta cases, K8 (the Al / Cu settings, runs_k8.sh): fewest
# cells (one), three random tilings of a 10 um lattice, uniform 20 um (~ the mass thickness of Al 100 um). After training.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
L=logs_ta
until grep -q "ta train done" $L/train.out 2>/dev/null; do sleep 60; done
MOD="--model_ta ml/models/size_gmc_ta_10um-1cm_20MeV/secondary_gmc.pth --kernel_ta ml/models/size_scr_ta_10um-1cm_20MeV"
C="ta_1.0_0 ta_1.0_60 ta_0.5_0 ta_0.5_30 ta_0.5_60 ta_0.3_0"
for mesh in one random:1 random:2 random:3 uniform:20; do
  .venv/bin/python validation/lockwood_quadtree.py run --cases $C --mesh $mesh $MOD --K_min 8 --K_scale 0.25 --tag _k8 >> $L/runs.log 2>&1
  echo "ta $mesh exit $?"
done
echo "ta runs done"
