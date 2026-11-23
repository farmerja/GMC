#!/bin/bash
# 2026-11-23: same meshes as runs.sh with the Drumm SCR sample setting K = max(8, 0.25 x auto) (runs.sh ran with the
# default K = max(32, auto)); outputs tagged _k8.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
L=logs_lockwood_size
MOD="--model_al ml/models/size_gmc_al_50um-6mm_1.5MeV/secondary_gmc.pth --kernel_al ml/models/size_scr_al_50um-6mm_1.5MeV
     --model_cu ml/models/size_gmc_cu_50um-6mm_1.5MeV/secondary_gmc.pth --kernel_cu ml/models/size_scr_cu_50um-6mm_1.5MeV"
for m in one random:1 random:2 random:3 uniform:100; do
  .venv/bin/python validation/lockwood_quadtree.py run --mesh $m $MOD --K_min 8 --K_scale 0.25 --tag _k8 >> $L/runs_k8.log 2>&1
  echo "mesh $m exit $?"
done
echo "lockwood k8 runs done"
