#!/bin/bash
# 2026-11-23: after lockwood_t1_quiet.sh: redo the one t1q case that overlapped a GPU eval (al_0.3_0 random:1), then
# the water SCR kernels (v2 vs _t1) on the targeted held-out file.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
while kill -0 1392681 2>/dev/null; do sleep 20; done
rm -f validation/results/lockwood_sandia/size/gmc_al_0.3_0_0.314MeV_random_1_200k_k8_t1q.npz
MOD="--model_al ml/models/size_gmc_al_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_al ml/models/size_scr_al_50um-6mm_1.5MeV_t1
     --model_cu ml/models/size_gmc_cu_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_cu ml/models/size_scr_cu_50um-6mm_1.5MeV_t1"
.venv/bin/python validation/lockwood_quadtree.py run --mesh random:1 $MOD --K_min 8 --K_scale 0.25 --tag _k8_t1q >> logs_tgt/lockwood_t1q.log 2>&1
echo "redo exit $?"
E=../ElectronTransport
for v in v2 t1; do
  .venv/bin/python validation/size_cell_compare.py scr --model ml/models/size_scr_water_100um-1cm_30MeV_$v \
      --test_npz $E/sweep_water_0.25-2mm_5-30MeV_fwd_90/part_0000.npz --out_dir validation/results/size_gmc/tgt/water_${v}_tgt_scr \
      --n_transits 20000 > logs_tgt/eval_scr_water_$v.log 2>&1
  echo "water scr $v exit $?"
done
echo "after t1q done"
