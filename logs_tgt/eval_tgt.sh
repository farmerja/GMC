#!/bin/bash
# 2026-11-23: checks of the targeted fine-tunes, old vs new on the same inputs. Metal part as soon as the Al / Cu
# fine-tunes are done, water part after _sx4. GPU jobs run one at a time here (training may still share the GPU).
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
L=logs_tgt
R=validation/results
until grep -q "cu gmc exit" $L/lane_gmc.out 2>/dev/null && grep -q "cu scr exit" $L/lane_scr.out 2>/dev/null; do sleep 60; done

# --- Al / Cu: per-size checks on the broad and the targeted held-out files, old vs _t1
for m in al cu; do
  for v in base t1; do
    sfx=$([ $v = t1 ] && echo _t1)
    G=ml/models/size_gmc_${m}_50um-6mm_1.5MeV$sfx/secondary_gmc.pth
    K=ml/models/size_scr_${m}_50um-6mm_1.5MeV$sfx
    for t in broad:sweep_${m}_50um-6mm_1.5MeV_7/part_0001.npz tgt:sweep_${m}_tgt_0.2-1.5MeV_8/part_0000.npz; do
      n=${t%%:*}; f=$E/${t#*:}
      [ -f $R/size_gmc/tgt/${m}_${v}_$n/summary_gmc.json ] || .venv/bin/python validation/size_cell_compare.py gmc --model $G \
          --test_npz $f --out_dir $R/size_gmc/tgt/${m}_${v}_$n > $L/eval_gmc_${m}_${v}_$n.log 2>&1
      [ -f $R/size_gmc/tgt/${m}_${v}_${n}_scr/summary_scr.json ] || .venv/bin/python validation/size_cell_compare.py scr \
          --model $K --test_npz $f --out_dir $R/size_gmc/tgt/${m}_${v}_${n}_scr --n_transits 20000 > $L/eval_scr_${m}_${v}_$n.log 2>&1
    done
  done
done
echo "metal evals exit $?"

# --- Lockwood, all meshes, K8, the _t1 models (Cu 1 MeV has no measurement: explicit --cases)
MOD="--model_al ml/models/size_gmc_al_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_al ml/models/size_scr_al_50um-6mm_1.5MeV_t1
     --model_cu ml/models/size_gmc_cu_50um-6mm_1.5MeV_t1/secondary_gmc.pth --kernel_cu ml/models/size_scr_cu_50um-6mm_1.5MeV_t1"
for mesh in one random:1 random:2 random:3 uniform:100; do
  .venv/bin/python validation/lockwood_quadtree.py run --mesh $mesh $MOD --K_min 8 --K_scale 0.25 --tag _k8_t1 >> $L/lockwood_t1.log 2>&1
  .venv/bin/python validation/lockwood_quadtree.py run --cases cu_1.0_0 --mesh $mesh $MOD --K_min 8 --K_scale 0.25 --tag _k8_t1 >> $L/lockwood_t1.log 2>&1
  echo "lockwood $mesh exit $?"
done
echo "lockwood t1 done"

# --- water _sx4: probes, widths, per-size, Drumm meshes
until grep -q "water gmc exit" $L/lane_gmc.out 2>/dev/null; do sleep 60; done
OUT=ml/models/size_gmc_water_100um-1cm_30MeV_sx4
for M in $OUT/secondary_gmc.pth $OUT/secondary_gmc_last.pth; do
  tag=$(basename $(dirname $M))_$(basename $M .pth)
  for P in probe_water_centre1cm_logged probe_water_sym10_logged; do
    echo "== $tag $P" >> $L/probe_sx4.log
    .venv/bin/python validation/scr_probe.py gmc --data $E/$P/part_0000.npz --model $M 2>&1 | grep -v INFO >> $L/probe_sx4.log
  done
done
.venv/bin/python validation/scatter_width_scan.py gmc --run $R/drumm_size/scatter_scan_100k.npz --model $OUT/secondary_gmc.pth > $L/scan_sx4.log 2>&1
.venv/bin/python validation/size_cell_compare.py gmc --model $OUT/secondary_gmc.pth --test_npz $E/sweep_water_100um-1cm_30MeV_7/part_0001.npz \
    --out_dir $R/size_gmc/cell_sx4 > $L/eval_gmc_sx4.log 2>&1
echo "water checks exit $?"
# attribution: _sx4 GMC with the old SCR kernel (v2) on the two meshes with the in-cell pattern
for m in uniform:0.5 uniform:1; do
  out=$R/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx4_chord.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --model $OUT/secondary_gmc.pth --K_min 8 --K_scale 0.25 --out $out >> $L/drumm_sx4.log 2>&1
  echo "drumm sx4 $m exit $?"
done
# all meshes: _sx4 GMC + the fine-tuned SCR kernel
until grep -q "water scr exit" $L/train_water_scr.out 2>/dev/null; do sleep 60; done
KW=ml/models/size_scr_water_100um-1cm_30MeV_t1
for m in sym:5:0.125 quad:4:0.125 sym:10:0.125 min:10:0.125 centre:10:0.15625:2 uniform:0.5 uniform:1 uniform:2 uniform:4; do
  out=$R/drumm_size/gmc_$(echo $m | tr ':.' '_p')_500k_K8_sx4t1_chord.npz
  [ -f $out ] || .venv/bin/python validation/drumm_quadtree.py run --mesh $m --model $OUT/secondary_gmc.pth --kernel $KW --K_min 8 --K_scale 0.25 --out $out >> $L/drumm_sx4.log 2>&1
  echo "drumm sx4t1 $m exit $?"
done
echo "eval tgt done"
