#!/bin/bash
# Evaluation phase: run alone (timings). Waits for all training + MC references.
cd /home/aubrey/PycharmProjects/GMC
until grep -q "ALL GMC VARIANTS DONE" logs_sweep/queue_gmc.log && grep -q "ALL SCR VARIANTS DONE" logs_sweep/queue_scr.log \
      && ! pgrep -f "drumm_cylinder.py mc" >/dev/null; do sleep 60; done
D=$PWD/validation/results/drumm_water_cylinder
R=$D/sweep/runs; FB=$D/sweep/flat; mkdir -p $R $FB $D/phaseF
M=ml/models/secondary_gmc_water_1mm_30MeV_elsepa; K=ml/models/subcell_kernel_water_1mm_30MeV_elsepa
CK=ml/models/charge_kernel_water_1mm_30MeV_elsepa
PY=.venv/bin/python
echo "=== clean MC timing $(date)"
[ -f $R/mc_timing_50k.npz ] || ../ElectronTransport/.venv/bin/python -u validation/drumm_cylinder.py mc --geom stair --seed 9123 \
   --src_seed 2010 --N 50000 --batch 50000 --out $R/mc_timing_50k.npz
sw() { local lab=$1; shift; [ -f $R/$lab.npz ] && return; echo "=== sweep $lab $(date)"; $PY -u validation/drumm_sweep.py run --ode_steps 50 --label $lab --out $R/$lab.npz "$@" 2>&1 | grep -v Initializing; }
T80="--tally k80k=${K}_x2:32"
sw x2 --model ${M}_x2/secondary_gmc.pth --mirror $T80 --tally k40k=${K}:32 --tally k20k=${K}_s20k:32 \
   --tally k10k=${K}_s10k:32 --tally k20k_small=${K}_s20k_small:32 --tally k20k_data2M=${K}_s20k_data2M:32 \
   --tally K8=${K}_x2:8 --tally K16=${K}_x2:16 --tally K64=${K}_x2:64 --tally K128=${K}_x2:128
sw x2_nomirror --model ${M}_x2/secondary_gmc.pth $T80
for v in ep10 ep30 ep60 h128d4 h512d6 data1M data3M; do sw $v --model ${M}_x2_$v/secondary_gmc.pth --mirror $T80; done
for o in 10 20 100; do sw ode$o --model ${M}_x2/secondary_gmc.pth --mirror --ode_steps $o $T80; done
sw phaseD --model ${M}/secondary_gmc.pth $T80
sw xs60 --model ${M}_xs60/secondary_gmc.pth --mirror $T80
fb() { local lab=$1; shift; [ -f $FB/$lab.npz ] && return; echo "=== flat $lab $(date)"; $PY -u validation/flat_beam_chain.py gmc --ode_steps 50 --out $FB/$lab.npz "$@" 2>&1 | grep -v Initializing; }
fb x2 --model ${M}_x2/secondary_gmc.pth --mirror
fb x2_nomirror --model ${M}_x2/secondary_gmc.pth
for v in ep10 ep30 ep60 h128d4 h512d6 data1M data3M; do fb $v --model ${M}_x2_$v/secondary_gmc.pth --mirror; done
for o in 10 20 100; do fb ode$o --model ${M}_x2/secondary_gmc.pth --mirror --ode_steps $o; done
echo "=== paper runs $(date)"
for conv in "" "--local_photons"; do
  out=$D/phaseF/gmc_n8_500k_x2_mirror${conv:+_localph}.npz
  [ -f $out ] || $PY -u validation/drumm_cylinder.py gmc --model ${M}_x2/secondary_gmc.pth --kernel ${K}_x2 --charge_kernel $CK \
     --N 500000 --n_seeds 2 --mirror --ode_steps 50 $conv --out $out 2>&1 | grep -v Initializing
done
echo "ALL EVAL DONE $(date)"
