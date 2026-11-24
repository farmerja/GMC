#!/bin/bash
# 2026-11-23: targeted fine-tunes (the water _sx3 recipe) for every regime the write-ups found above noise.
# Two lanes on the 16 GB GPU (one GMC + one SCR training at a time), each starting a model as soon as its data exists:
#   GMC lane: Al, Cu (size_gmc_*_50um-6mm_1.5MeV -> _t1), then water (_sx3 -> _sx4)
#   SCR lane: Al, Cu (size_scr_*_50um-6mm_1.5MeV -> _t1); water SCR only if WATER_SCR=1 (decided by the K32 check)
# Mixes (metal 10 + 10 files, water 22 + 22): targeted and broad files 1:1, interleaved, so the rest of the (E, L) range is not forgotten.
# Validation / best-checkpoint selection on the broad held-out file (as _sx3). Resumable: rerun.
cd /home/aubrey/PycharmProjects/GMC
export PYTHONUNBUFFERED=1
E=../ElectronTransport
L=logs_tgt
GMC_FT="--finetune --stop --sweeps 3 --clf_passes 6 --flow_passes 24 --small_passes 24 --clf_lr 5e-4 --flow_lr 3e-4"

metal_mix() {   # $1 = al | cu
  local m=$1
  until grep -q "seed 44 done" $L/gen_$m.out 2>/dev/null; do grep -q FAILED $L/gen_$m.out 2>/dev/null && exit 1; sleep 60; done
  # 10 targeted + 10 broad (every other broad file), alternating; both lanes call this, so under a lock
  (
    flock 9
    [ -f $E/mix_${m}_tgt10_old10/manifest.json ] || .venv/bin/python ml/mix_dirs.py --out $E/mix_${m}_tgt10_old10 \
        --src $E/sweep_${m}_tgt_0.2-1.5MeV_44:0-9 $E/sweep_${m}_50um-6mm_1.5MeV_42:0,2,4,6,8,10,12,14,16,18
  ) 9> $L/mix_$m.lock
}

lane_gmc() {
  for m in al cu; do
    metal_mix $m
    .venv/bin/python ml/size_stream.py gmc --data_dir $E/mix_${m}_tgt10_old10 --n_files 20 \
        --val_npz $E/sweep_${m}_50um-6mm_1.5MeV_7/part_0000.npz --out ml/models/size_gmc_${m}_50um-6mm_1.5MeV_t1 \
        --init_from ml/models/size_gmc_${m}_50um-6mm_1.5MeV/secondary_gmc.pth $GMC_FT >> $L/train_gmc_$m.log 2>&1
    echo "$m gmc exit $?"
  done
  until grep -q "water A gen done" $L/gen_water_A.out 2>/dev/null && grep -q "water B gen done" $L/gen_water_B.out 2>/dev/null; do sleep 60; done
  B=$E/sweep_water_100um-1cm_30MeV_42
  [ -f $E/mix_water_sx4/manifest.json ] || .venv/bin/python ml/mix_dirs.py --out $E/mix_water_sx4 \
      --src $E/sweep_water_6-10mm_10-30MeV_fwd_45:0-5 $E/sweep_water_1-4mm_5-30MeV_fwd_46:0-5 \
            $E/sweep_water_3-10mm_5-30MeV_fwd_43:0-9 $B:1-8 $B:9-15 $B:16-22
  .venv/bin/python ml/size_stream.py gmc --data_dir $E/mix_water_sx4 --n_files 44 \
      --val_npz $E/sweep_water_100um-1cm_30MeV_7/part_0000.npz --out ml/models/size_gmc_water_100um-1cm_30MeV_sx4 \
      --init_from ml/models/size_gmc_water_100um-1cm_30MeV_sx3/secondary_gmc.pth $GMC_FT >> $L/train_gmc_water.log 2>&1
  echo "water gmc exit $?"
}

lane_scr() {
  for m in al cu; do
    metal_mix $m
    .venv/bin/python ml/size_stream.py scr --data_dir $E/mix_${m}_tgt10_old10 --n_files 20 \
        --val_npz $E/sweep_${m}_50um-6mm_1.5MeV_7/part_0000.npz --out ml/models/size_scr_${m}_50um-6mm_1.5MeV_t1 \
        --init_from ml/models/size_scr_${m}_50um-6mm_1.5MeV --lr 2e-4 --steps 120000 --sweeps 3 >> $L/train_scr_$m.log 2>&1
    echo "$m scr exit $?"
  done
}

lane_gmc > $L/lane_gmc.out 2>&1 &
lane_scr > $L/lane_scr.out 2>&1 &
wait
echo "tgt train done"
