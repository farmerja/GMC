#!/bin/bash
# Speed knobs for the size-conditioned GMC+SCR on Drumm quadtree meshes (skips existing outputs).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
K=ml/models/size_scr_water_100um-1cm_30MeV_v2
run() {  # mesh tag args...
  m=$1; t=$2; shift 2
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k$t.npz
  [ -f $out ] || { echo "== $m $t $*"; .venv/bin/python validation/drumm_quadtree.py run --mesh $m --kernel $K --out $out "$@"; } >> logs_size/speed_sweep.log 2>&1
  echo "speed $m $t exit $?"
}
run quad:4:0.125 _ode10 --ode_steps 10
run quad:4:0.125 _K16 --K_min 16 --K_scale 0.5
run quad:4:0.125 _K8 --K_min 8 --K_scale 0.25
run quad:4:0.125 _b250k --batch 250000
run quad:5:0.15625 "" 
run quad:4:0.125 _ode10_K16 --ode_steps 10 --K_min 16 --K_scale 0.5
run quad:5:0.15625 _ode10_K16 --ode_steps 10 --K_min 16 --K_scale 0.5
echo "speed sweep done"
