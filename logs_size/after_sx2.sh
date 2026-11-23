#!/bin/bash
# After the smooth-sigma exit_B retrain: width scan, per-size cell eval, Drumm runs with the new model.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
until grep -q "gmc_sx2 exit" logs_size/train_gmc_sx2.out 2>/dev/null; do sleep 30; done
M=ml/models/size_gmc_water_100um-1cm_30MeV_sx2/secondary_gmc.pth
E=../ElectronTransport
.venv/bin/python validation/scatter_width_scan.py gmc --run validation/results/drumm_size/scatter_scan_100k.npz --model $M > logs_size/scan_sx2.log 2>&1
echo "scan exit $?"
.venv/bin/python validation/size_cell_compare.py gmc --model $M --test_npz $E/sweep_water_100um-1cm_30MeV_7/part_0001.npz \
    --out_dir validation/results/size_gmc/cell_sx2 > logs_size/eval_gmc_sx2.log 2>&1
echo "cell eval exit $?"
for m in quad:4:0.125 uniform:1 centre:10:0.15625:2 centre:10:0.3125:2; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx2.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --model $M --K_min 8 --K_scale 0.25 --out $out >> logs_size/drumm_sx2.log 2>&1
  echo "drumm $m exit $?"
done
echo "after_sx2 done"
