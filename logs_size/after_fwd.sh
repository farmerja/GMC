#!/bin/bash
# 2026-11-23: after the targeted data (gen_fwd.sh): held-out targeted file, mixed dir, fine-tune _sx2 -> _sx3, checks.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
until grep -q "gen_fwd exit" logs_size/gen_fwd.out 2>/dev/null; do sleep 60; done
(cd $E && .venv/bin/python generate_size_sweep.py --out_dir sweep_water_3-10mm_5-30MeV_fwd_9 --seed 9 --n_files 1 \
    --L_low 0.3 --L_high 1.0 --E_low 5 --E_high 30 --no_subcell --narrow_frac 0.5 --narrow_sigma_deg 10) >> logs_size/gen_fwd.log 2>&1
echo "val gen exit $?"
.venv/bin/python ml/mix_dirs.py --out $E/mix_water_fwd10_old10 --src $E/sweep_water_3-10mm_5-30MeV_fwd_43:0-9 \
    $E/sweep_water_100um-1cm_30MeV_42:0,4,8,12,16,20,24,28,32,36
M0=ml/models/size_gmc_water_100um-1cm_30MeV_sx2/secondary_gmc.pth
OUT=ml/models/size_gmc_water_100um-1cm_30MeV_sx3
.venv/bin/python ml/size_stream.py gmc --data_dir $E/mix_water_fwd10_old10 --n_files 20 \
    --val_npz $E/sweep_water_100um-1cm_30MeV_7/part_0000.npz --out $OUT --init_from $M0 --finetune --stop \
    --sweeps 3 --clf_passes 6 --flow_passes 24 --small_passes 24 --clf_lr 5e-4 --flow_lr 3e-4 >> logs_size/train_gmc_sx3.log 2>&1
echo "gmc_sx3 exit $?"
for M in $OUT/secondary_gmc.pth $OUT/secondary_gmc_last.pth $M0; do
  tag=$(basename $(dirname $M))_$(basename $M .pth)
  for P in probe_water_centre1cm_logged probe_water_sym10_logged; do
    echo "== $tag $P" >> logs_size/probe_sx3.log
    .venv/bin/python validation/scr_probe.py gmc --data $E/$P/part_0000.npz --model $M 2>&1 | grep -v INFO >> logs_size/probe_sx3.log
  done
done
echo "probes exit $?"
.venv/bin/python validation/scatter_width_scan.py gmc --run validation/results/drumm_size/scatter_scan_100k.npz --model $OUT/secondary_gmc.pth > logs_size/scan_sx3.log 2>&1
.venv/bin/python validation/size_cell_compare.py gmc --model $OUT/secondary_gmc.pth --test_npz $E/sweep_water_100um-1cm_30MeV_7/part_0001.npz \
    --out_dir validation/results/size_gmc/cell_sx3 > logs_size/eval_gmc_sx3.log 2>&1
echo "checks exit $?"
for m in min:10:0.125 sym:10:0.125 centre:10:0.15625:2 sym:5:0.125 quad:4:0.125; do
  out=validation/results/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx3_chord.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --model $OUT/secondary_gmc.pth --K_min 8 --K_scale 0.25 --out $out >> logs_size/drumm_sx3.log 2>&1
  echo "drumm $m exit $?"
done
echo "after_fwd done"
