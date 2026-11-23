#!/bin/bash
# Independent repeat (new chain and source seeds) of sym:5 with sx3, for the GMC-vs-GMC null test of the noise estimate.
cd /home/aubrey/PycharmProjects/GMC
until grep -q "writeup runs done" logs_size/writeup_runs.out 2>/dev/null; do sleep 20; done
.venv/bin/python validation/drumm_quadtree.py run --mesh sym:5:0.125 --model ml/models/size_gmc_water_100um-1cm_30MeV_sx3/secondary_gmc.pth \
   --K_min 8 --K_scale 0.25 --seed 31 --src_seed 3001 --out validation/results/drumm_size/gmc_sym_5_0p125_500k_K8_sx3_chord_s31.npz >> logs_size/drumm_writeup.log 2>&1
echo "null run exit $?"
