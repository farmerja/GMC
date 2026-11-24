#!/bin/bash
# 2026-11-23: _t1 Lockwood runs again on an idle machine for timing (the _k8_t1 runs of one / random:1 / random:2
# overlapped the water SCR training and data generation). Same settings, tag _k8_t1q.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
MOD="--model_al ml/models/size_gmc_al_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_al ml/models/size_scr_al_50um-6mm_1.5MeV_t1
     --model_cu ml/models/size_gmc_cu_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_cu ml/models/size_scr_cu_50um-6mm_1.5MeV_t1"
for mesh in one random:1 random:2; do
  .venv/bin/python validation/lockwood_quadtree.py run --mesh $mesh $MOD --K_min 8 --K_scale 0.25 --tag _k8_t1q >> logs_tgt/lockwood_t1q.log 2>&1
  .venv/bin/python validation/lockwood_quadtree.py run --cases cu_1.0_0 --mesh $mesh $MOD --K_min 8 --K_scale 0.25 --tag _k8_t1q >> logs_tgt/lockwood_t1q.log 2>&1
  echo "lockwood q $mesh exit $?"
done
echo "lockwood t1q done"
