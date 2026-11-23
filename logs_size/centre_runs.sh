#!/bin/bash
# GMC + SCR Drumm runs on graded "centre" meshes (1 cm square on the axis), after the analog chains (clean timings).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
until grep -q "analog chain done" logs_size/analog_chain.out 2>/dev/null; do sleep 60; done
for m in centre:10:0.15625:2 centre:10:0.15625:1 centre:10:0.3125:2; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --K_min 8 --K_scale 0.25 --out $out >> logs_size/centre_runs.log 2>&1
  echo "centre $m exit $?"
done
echo "centre runs done"
