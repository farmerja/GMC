#!/bin/bash
# 2026-11-22: after SCR v2 training: per-size SCR evaluation, then the Drumm mesh sweep (GPU otherwise idle, so the
# timings are clean). Each step skips outputs that already exist.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
until grep -q "scr exit" logs_size/train_scr.out; do sleep 60; done
K=ml/models/size_scr_water_100um-1cm_30MeV_v2
[ -f validation/results/size_gmc/scr/summary_scr.json ] || .venv/bin/python validation/size_cell_compare.py scr --model $K \
    --test_npz $E/sweep_water_100um-1cm_30MeV_7/part_0001.npz --out_dir validation/results/size_gmc/scr --n_transits 20000 \
    --ref 1mm $E/analog_water_1mm_1keV-30MeV_log_2M_7_estar.npz ml/models/subcell_kernel_water_1mm_30MeV_estar > logs_size/eval_scr.log 2>&1
echo "scr eval exit $?"
for m in uniform:1 quad:4:0.125 quad:2:0.125 quad:4:0.25 uniform:0.5 quad:1:0.125 uniform:2; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --kernel $K --out $out >> logs_size/drumm_sweep.log 2>&1
  echo "drumm $m exit $?"
done
