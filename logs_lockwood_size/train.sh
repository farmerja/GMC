#!/bin/bash
# 2026-11-23: size-conditioned Al / Cu GMC + SCR for Lockwood (cells 50 um - 6 mm, E <= 1.5 MeV). Two lanes (GMC, SCR), each Al then Cu,
# side by side (the 16 GB GPU fits one of each), after that material's data and after the water
# sx3 Drumm timing runs. Smooth exit sigma table from the start (the _sx2 fix). Resumable: rerun.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
L=logs_lockwood_size
# after the water write-up runs (marker written once they are finished) and with no QUIET lock
until [ -f logs_size/water_runs.done ] && [ ! -f logs_size/QUIET ]; do sleep 60; done
lane() {   # $1 = gmc | scr: materials in order, so at most one GMC and one SCR training share the GPU
  for M in Al Cu; do
    m=$(echo $M | tr 'A-Z' 'a-z')
    until grep -q "$M seed 42 done" $L/gen.out 2>/dev/null; do sleep 60; done
    D=$E/sweep_${m}_50um-6mm_1.5MeV_42; V=$E/sweep_${m}_50um-6mm_1.5MeV_7/part_0000.npz
    if [ $1 = gmc ]; then
      .venv/bin/python ml/size_stream.py gmc --data_dir $D --n_files 40 --val_npz $V --out ml/models/size_gmc_${m}_50um-6mm_1.5MeV \
          --stop --smooth_exit_sigma --spec_files 12 >> $L/train_gmc_$m.log 2>&1
    else
      .venv/bin/python ml/size_stream.py scr --data_dir $D --n_files 40 --val_npz $V --out ml/models/size_scr_${m}_50um-6mm_1.5MeV \
          --steps 320000 >> $L/train_scr_$m.log 2>&1
    fi
    echo "$M $1 exit $?"
  done
}
lane gmc &
lane scr &
wait
echo "lockwood train done"
