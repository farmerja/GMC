#!/bin/bash
# Analog-chain (exact) runs on the Drumm meshes for the wing diagnostic (skips existing outputs).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
for spec in uniform:1:200000 uniform:2:200000 quad:4:0.125:200000; do
  m=${spec%:*}; N=${spec##*:}
  out=validation/results/drumm_size/analog_$(echo $m | tr ':.' '_p')_$((N / 1000))k.npz
  [ -f $out ] || ../ElectronTransport/.venv/bin/python validation/drumm_analog_chain.py mc --mesh $m --N $N --out $out >> logs_size/analog_chain.log 2>&1
  echo "analog $m exit $?"
done
echo "analog chain done"
