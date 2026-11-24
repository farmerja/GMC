#!/bin/bash
# 2026-11-23: water SCR fine-tune (size_scr_water_100um-1cm_30MeV_v2 -> _t1) on 8 targeted (0.25-2 mm, 5-30 MeV) +
# 8 broad files, after the Al / Cu SCR fine-tunes (one SCR training on the GPU at a time). Resumable: rerun.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
L=logs_tgt
until grep -q "cu scr exit" $L/lane_scr.out 2>/dev/null && grep -q "water C gen done" $L/gen_water_C.out 2>/dev/null; do
  grep -q FAILED $L/gen_water_C.out 2>/dev/null && exit 1; sleep 60; done
[ -f $E/mix_water_scr_tgt8_old8/manifest.json ] || .venv/bin/python ml/mix_dirs.py --out $E/mix_water_scr_tgt8_old8 \
    --src $E/sweep_water_0.25-2mm_5-30MeV_fwd_70:0-7 $E/sweep_water_100um-1cm_30MeV_42:25,27,29,31,33,35,37,39
.venv/bin/python ml/size_stream.py scr --data_dir $E/mix_water_scr_tgt8_old8 --n_files 16 \
    --val_npz $E/sweep_water_100um-1cm_30MeV_7/part_0000.npz --out ml/models/size_scr_water_100um-1cm_30MeV_t1 \
    --init_from ml/models/size_scr_water_100um-1cm_30MeV_v2 --lr 2e-4 --steps 96000 --sweeps 3 >> $L/train_scr_water.log 2>&1
echo "water scr exit $?"
