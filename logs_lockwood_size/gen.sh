#!/bin/bash
# 2026-11-23: size-sweep data for size-conditioned Al and Cu models (Lockwood): square cells 50 um - 6 mm (log-uniform),
# E 1 keV - 1.5 MeV, sub-cell n = max(64, L / 10 um). Test (2 files) + training (40 files) per material. Resumable.
# Runs after the targeted water data; pauses between files while the water sx3 checks / Drumm timing runs are active.
cd /home/aubrey/PycharmProjects/ElectronTransport
export PYTHONUNBUFFERED=1
G=/home/aubrey/PycharmProjects/GMC/logs_size/after_fwd.out
L=/home/aubrey/PycharmProjects/GMC/logs_lockwood_size
until grep -q "val gen exit" $G 2>/dev/null; do sleep 60; done
Q=/home/aubrey/PycharmProjects/GMC/logs_size/QUIET   # present while water timing runs need a quiet machine
pause() { while [ -f $Q ] || { grep -q "probes exit" $G 2>/dev/null && ! grep -q "after_fwd done" $G 2>/dev/null; }; do sleep 60; done; }
for M in Al Cu; do
  m=$(echo $M | tr 'A-Z' 'a-z')
  for spec in "7 2" "42 40"; do
    set -- $spec
    for k in $(seq 1 $2); do
      pause
      .venv/bin/python generate_size_sweep.py --out_dir sweep_${m}_50um-6mm_1.5MeV_$1 --seed $1 --n_files $k \
          --L_low 0.005 --L_high 0.6 --E_low 0.001 --E_high 1.5 --material $M >> $L/gen_$m.log 2>&1 || exit 1
    done
    echo "$M seed $1 done"
  done
done
echo "lockwood gen done"
