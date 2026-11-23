#!/bin/bash
# 2026-11-23: Lockwood depth-dose with the size-conditioned Al / Cu models on the fewest-cells mesh (one cell), three
# random-size tilings, and the uniform 100 um baseline; all non-missing cases. After all training is done, so the
# timings are clean. Resumable (skips finished outputs).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
L=logs_lockwood_size
until grep -q "lockwood train done" $L/train.out 2>/dev/null; do sleep 60; done
MOD="--model_al ml/models/size_gmc_al_50um-6mm_1.5MeV/secondary_gmc.pth --kernel_al ml/models/size_scr_al_50um-6mm_1.5MeV
     --model_cu ml/models/size_gmc_cu_50um-6mm_1.5MeV/secondary_gmc.pth --kernel_cu ml/models/size_scr_cu_50um-6mm_1.5MeV"
for m in one random:1 random:2 random:3 uniform:100; do
  .venv/bin/python validation/lockwood_quadtree.py run --mesh $m $MOD --K_min 8 --K_scale 0.25 >> $L/runs.log 2>&1
  echo "mesh $m exit $?"
done
echo "lockwood runs done"
