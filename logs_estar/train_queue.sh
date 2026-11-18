#!/bin/bash
# 2026-11-17: Phase F retrain on the *_estar data (ESTAR stopping + ELSEPA + split ionization deflection).
# Two lanes in parallel on the 16 GB GPU: GMC (~3-4.5 GB) and SCR/charge kernels (~8.6 GB). Each model dir gets
# `done` (seconds) on success.
cd /home/aubrey/PycharmProjects/GMC
ET=../ElectronTransport; M=ml/models; L=logs_estar
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
gmc() {  # data-tag model-tag
  local d=$M/secondary_gmc_$2 s=$(date +%s); mkdir -p $d
  .venv/bin/python -u ml/secondary_gmc.py train --train_npz $ET/analog_$1_estar.npz --out $d --edge_feats --exit_scaled \
    --clf_epochs 30 --flow_epochs 120 --small_epochs 120 --stop > $d/train.log 2>&1 && echo $(( $(date +%s) - s )) > $d/done
  echo "GMC $2 exit $? $(( $(date +%s) - s )) s $(date +%T)"
}
scr() {
  local d=$M/subcell_kernel_$2 s=$(date +%s); mkdir -p $d
  .venv/bin/python -u ml/subcell_kernel.py train --train_npz $ET/analog_$1_estar.npz --out $d --batch 32768 --steps 80000 \
    > $d/train.log 2>&1 && echo $(( $(date +%s) - s )) > $d/done
  echo "SCR $2 exit $? $(( $(date +%s) - s )) s $(date +%T)"
}
ck() {
  local d=$M/charge_kernel_$2 s=$(date +%s); mkdir -p $d
  .venv/bin/python -u ml/charge_kernel.py train --train_npz $ET/analog_$1_estar.npz --out $d > $d/train.log 2>&1 \
    && echo $(( $(date +%s) - s )) > $d/done
  echo "CK $2 exit $? $(( $(date +%s) - s )) s $(date +%T)"
}
W1=water_1mm_1keV-30MeV_log_10M_42
( gmc cu_100um_1keV-10MeV_log_10M_42 cu_100um_estar
  gmc al_100um_1keV-10MeV_log_10M_42 al_100um_estar
  gmc $W1 water_1mm_30MeV_estar
  gmc water_100um_1keV-10MeV_log_10M_42 water_100um_estar ) > $L/lane_gmc.out 2>&1 &
( scr cu_100um_1keV-10MeV_log_10M_42 cu_100um_estar
  scr al_100um_1keV-10MeV_log_10M_42 al_100um_estar
  scr $W1 water_1mm_30MeV_estar
  ck $W1 water_1mm_30MeV_estar ) > $L/lane_scr.out 2>&1 &
wait
echo "TRAIN QUEUE DONE $(date)"
