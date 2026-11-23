#!/bin/bash
# Local photon energy placed along the entry -> exit chord (drumm_quadtree --photon_place chord), _sx2 + K8, clean timings.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
for m in centre:10:0.15625:2 quad:4:0.125 sym:10:0.125 sym:5:0.125; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx2_chord.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --K_min 8 --K_scale 0.25 --out $out >> logs_size/chord_runs.log 2>&1
  echo "chord $m exit $?"
done
echo "chord runs done"
