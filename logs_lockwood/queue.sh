#!/bin/bash
# GPU sequencing (16 GB): an SCR training holds ~8.6 GB, a GMC training ~3-4.5 GB (peaks in the classifier temperature fit).
cd /home/aubrey/PycharmProjects/GMC
M=ml/models
CU_G=$M/secondary_gmc_cu_100um_elsepa_e40; AL_G=$M/secondary_gmc_al_100um_elsepa_e40
CU_S=$M/subcell_kernel_cu_100um_elsepa_s20k; AL_S=$M/subcell_kernel_al_100um_elsepa_s20k
infer() {  # material, cases...
  m=$1; shift
  echo "=== inference $m $(date)"
  (cd validation && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../.venv/bin/python -u lockwood_slab.py gmc \
     --cases "$@" --model_$m ../$M/secondary_gmc_${m}_100um_elsepa_e40/secondary_gmc.pth \
     --kernel_$m ../$M/subcell_kernel_${m}_100um_elsepa_s20k --N 200000 --n_seeds 2) \
     > logs_lockwood/infer_$m.log 2>&1
  echo "=== inference $m done $(date)"
}
until [ -f $CU_S/done ]; do sleep 15; done
echo "Cu SCR done $(date); starting Cu GMC"
logs_lockwood/train_gmc.sh cu > logs_lockwood/train_gmc_cu.log 2>&1 &
until [ -f $AL_G/done ]; do sleep 15; done
echo "Al GMC done $(date); starting Al SCR"
logs_lockwood/train_scr.sh al > logs_lockwood/train_scr_al.log 2>&1 &
until [ -f $CU_G/done ]; do sleep 15; done
infer cu cu_0.5_0 cu_0.3_0 cu_1.0_0 &
until [ -f $AL_S/done ]; do sleep 15; done
infer al al_1.0_0 al_1.0_60 al_0.5_0 al_0.5_60 al_0.3_0 al_0.3_60 al_0.1_0 al_0.05_0
wait
echo "QUEUE DONE $(date)"
