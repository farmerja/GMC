#!/bin/bash
# Fewest-cells asymmetric mesh (min:<L_max mm>:<h mm>, best greedy tiling), _sx2 + K8 + chord photons.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
m=min:10:0.125
out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx2_chord.npz
[ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --K_min 8 --K_scale 0.25 --out $out >> logs_size/min_run.log 2>&1
echo "min run exit $?"
