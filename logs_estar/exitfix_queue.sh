#!/bin/bash
# 2026-11-18: water 1 mm ESTAR exit-flow fix (face pull + photon-transit energy-loss tail).
# Two candidates (seeds 1, 2): exit_B + exit_I retrained with --exit_oversample 4,2,2 on copies of the ESTAR checkpoint.
# Each is then checked on the flat-beam test (GMC seed 11 = selection seed A) and the one-transit diagnostics.
cd /home/aubrey/PycharmProjects/GMC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
M=ml/models; B=$M/secondary_gmc_water_1mm_30MeV_estar; L=logs_estar; ET=../ElectronTransport
R=validation/results/drumm_water_cylinder/flat_beam
TR=$ET/analog_water_1mm_1keV-30MeV_log_10M_42_estar.npz; TE=$ET/analog_water_1mm_1keV-30MeV_log_2M_7_estar.npz
cand() {  # $1 = seed
  d=${B}_os_s$1; rm -rf $d; cp -r $B $d; rm -f $d/done
  t0=$(date +%s)
  .venv/bin/python -u ml/secondary_gmc.py train --train_npz $TR --out $d --only exit_B,exit_I --edge_feats --exit_scaled \
    --flow_epochs 120 --exit_oversample 4,2,2 --seed $1 > $d/train_exitfix.log 2>&1
  echo "train s$1 exit $? $(( $(date +%s) - t0 )) s"; touch $d/done
}
check() {  # $1 = seed
  d=${B}_os_s$1
  .venv/bin/python validation/flat_beam_chain.py gmc --model $d/secondary_gmc.pth --mirror --out $R/flat_gmc_estar_os_s$1_mirror_300k.npz > $L/diag/flat_s$1.log 2>&1
  .venv/bin/python validation/flat_beam_chain.py compare --runs $R/flat_mc_300k_estar.npz $R/flat_gmc_estar_os_s$1_mirror_300k.npz \
    --transit_model $d/secondary_gmc.pth >> $L/diag/flat_s$1.log 2>&1
  .venv/bin/python $L/diag/exit_loss_check.py $d/secondary_gmc.pth $TE > $L/diag/exitloss_s$1.log 2>&1
  .venv/bin/python $L/diag/dau_loss.py $d/secondary_gmc.pth $TE > $L/diag/daulos_s$1.log 2>&1
  echo "check s$1 done"
}
cand 1 &
sleep 600          # stagger the 10M table loads
cand 2 &
wait
check 1; check 2
echo "EXITFIX QUEUE DONE $(date)"
