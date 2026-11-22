#!/bin/bash
# Drumm wing diagnostics: daughter calibration and primary-only GMC (skips existing outputs).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
C=ml/models/size_gmc_water_100um-1cm_30MeV/daughter_calib_Bonly.npz
run() {  # mesh tag args...
  m=$1; t=$2; shift 2
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k$t.npz
  [ -f $out ] || { echo "== $m $t $*"; .venv/bin/python validation/drumm_quadtree.py run --mesh $m --out $out "$@"; } >> logs_size/calib_sweep.log 2>&1
  echo "calib $m $t exit $?"
}
run uniform:2 _nosec --no_secondaries --n_seeds 1 --K_min 8 --K_scale 0.25
run uniform:1 _nosec --no_secondaries --n_seeds 1 --K_min 8 --K_scale 0.25
run uniform:2 _calib --calib $C --K_min 8 --K_scale 0.25
run uniform:1 _calib --calib $C --K_min 8 --K_scale 0.25
echo "calib sweep done"
