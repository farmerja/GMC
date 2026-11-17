#!/bin/bash
# Lockwood benchmark: Cu / Al 100 um SCR (sub-cell) kernels on the ELSEPA data, 20k steps (time budget).
cd /home/aubrey/PycharmProjects/GMC
mat=$1
TR=../ElectronTransport/analog_${mat}_100um_1keV-10MeV_log_10M_42_elsepa.npz
d=ml/models/subcell_kernel_${mat}_100um_elsepa_s20k
mkdir -p $d; s=$(date +%s)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u ml/subcell_kernel.py train --train_npz $TR --out $d --batch 32768 --steps 20000 > $d/train.log 2>&1 \
  && echo $(( $(date +%s) - s )) > $d/done
echo "$mat SCR finished: $(cat $d/done 2>/dev/null) s"
