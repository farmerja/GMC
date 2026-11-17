#!/bin/bash
# GMC sweep variants: boundary exit flow (exit_B) retrained on top of the x2 checkpoint (all other sub-models shared).
cd /home/aubrey/PycharmProjects/GMC
TR=../ElectronTransport/analog_water_1mm_1keV-30MeV_log_10M_42_elsepa.npz
M=ml/models/secondary_gmc_water_1mm_30MeV_elsepa
until [ -f ${M}_x2/train_info.json ] && ! pgrep -f "out ${M}_x2 " >/dev/null; do sleep 30; done
run() {  # name, extra args
  d=${M}_x2_$1; shift
  [ -f $d/done ] && return
  mkdir -p $d && cp ${M}_x2/secondary_gmc.pth ${M}_x2/train_info.json $d/
  echo "=== $d $(date)"; s=$(date +%s)
  .venv/bin/python -u ml/secondary_gmc.py train --train_npz $TR --out $d --edge_feats --exit_scaled "$@" > $d/train.log 2>&1 \
    && echo $(( $(date +%s) - s )) > $d/done
  echo "    finished in $(cat $d/done 2>/dev/null) s"
}
run ep10   --only exit_B --flow_epochs 10
run ep30   --only exit_B --flow_epochs 30
run ep60   --only exit_B --flow_epochs 60
run h128d4 --only exit_B --flow_epochs 30 --exit_hidden 128 --exit_depth 4
run h512d6 --only exit_B --flow_epochs 30 --exit_hidden 512 --exit_depth 6
run data1M --only clf_B,exit_B --max_rows 1000000 --clf_epochs 300 --flow_epochs 300
run data3M --only clf_B,exit_B --max_rows 3000000 --clf_epochs 100 --flow_epochs 100
echo "ALL GMC VARIANTS DONE $(date)"
