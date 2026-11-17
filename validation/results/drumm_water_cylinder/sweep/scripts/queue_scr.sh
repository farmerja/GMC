#!/bin/bash
# SCR kernel sweep variants (same data and config as the 40k / 80k kernels unless stated).
cd /home/aubrey/PycharmProjects/GMC
TR=../ElectronTransport/analog_water_1mm_1keV-30MeV_log_10M_42_elsepa.npz
K=ml/models/subcell_kernel_water_1mm_30MeV_elsepa
until [ -f ${K}_x2/train_info.json ] && ! pgrep -f "out ${K}_x2 " >/dev/null; do sleep 30; done
run() {
  d=${K}_$1; shift
  [ -f $d/done ] && return
  mkdir -p $d; echo "=== $d $(date)"; s=$(date +%s)
  .venv/bin/python -u ml/subcell_kernel.py train --train_npz $TR --out $d --batch 32768 "$@" > $d/train.log 2>&1 \
    && echo $(( $(date +%s) - s )) > $d/done
  echo "    finished in $(cat $d/done 2>/dev/null) s"
}
run s10k --steps 10000
run s20k --steps 20000
run s20k_small --steps 20000 --layers 4 --hidden 128 --ctx 128
run s20k_data2M --steps 20000 --max_segments 2000000
echo "ALL SCR VARIANTS DONE $(date)"
