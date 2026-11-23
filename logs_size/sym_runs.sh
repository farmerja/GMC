#!/bin/bash
# Mirror-symmetric fewest-cells meshes (sym:<L_max mm>:<h mm>), GMC _sx2 + SCR K8, after the analog centre chain (clean timings).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
until grep -q "analog chain centre" logs_size/analog_centre.log 2>/dev/null; do sleep 30; done
for m in sym:10:0.125 sym:5:0.125; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx2.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --K_min 8 --K_scale 0.25 --out $out >> logs_size/sym_runs.log 2>&1
  echo "sym $m exit $?"
done
echo "sym runs done"
