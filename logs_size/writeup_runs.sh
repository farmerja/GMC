#!/bin/bash
# 2026-11-23: quiet-machine runs for the water write-up (Lockwood generation pauses while logs_size/QUIET exists).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
OUT=ml/models/size_gmc_water_100um-1cm_30MeV_sx3
R=validation/results/drumm_size
while pgrep -f generate_size_sweep > /dev/null; do sleep 10; done      # let the current Lockwood file finish
for m in uniform:4 uniform:2 uniform:1 uniform:0.5; do
  out=$R/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx3_chord.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --model $OUT/secondary_gmc.pth --K_min 8 --K_scale 0.25 --out $out >> logs_size/drumm_writeup.log 2>&1
  echo "drumm $m exit $?"
done
.venv/bin/python validation/timing_vs_size.py gmc --out $R/timing_gmc_sx3.json --model $OUT/secondary_gmc.pth > logs_size/timing.log 2>&1
echo "timing gmc exit $?"
../ElectronTransport/.venv/bin/python validation/timing_vs_size.py mc --out $R/timing_mc.json >> logs_size/timing.log 2>&1
echo "timing mc exit $?"
echo "writeup runs done"
