#!/bin/bash
# 2026-11-18: after the Drumm ESTAR MC references, run the ESTAR models alone on the GPU (timings are measured).
cd /home/aubrey/PycharmProjects/GMC
D=$PWD/validation/results/drumm_water_cylinder; M=ml/models; L=logs_estar
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
until [ -f $D/mc_stair_n8_500k_s1123_estar.npz ] && ! pgrep -f "drumm_cylinder.py mc" > /dev/null; do sleep 30; done
for c in "" "--local_photons"; do
  t0=$(date +%s)
  .venv/bin/python validation/drumm_cylinder.py gmc --model $M/secondary_gmc_water_1mm_30MeV_estar/secondary_gmc.pth \
    --kernel $M/subcell_kernel_water_1mm_30MeV_estar --charge_kernel $M/charge_kernel_water_1mm_30MeV_estar \
    --N 500000 --n_seeds 2 --mirror $c --out $D/estar/gmc_n8_500k_estar_mirror${c:+_localph}.npz > $L/drumm_gmc${c:+_localph}.log 2>&1
  echo "drumm gmc '$c' exit $? $(( $(date +%s) - t0 )) s"
done
cd validation; MM=../ml/models
t0=$(date +%s)
../.venv/bin/python -u lockwood_slab.py gmc --cases cu_0.5_0 cu_0.3_0 cu_1.0_0 --model_cu $MM/secondary_gmc_cu_100um_estar/secondary_gmc.pth \
  --kernel_cu $MM/subcell_kernel_cu_100um_estar --N 200000 --n_seeds 2 --tag _estar > ../$L/lockwood_gmc_cu_timing.log 2>&1
../.venv/bin/python -u lockwood_slab.py gmc --cases al_1.0_0 al_1.0_60 al_0.5_0 al_0.5_60 al_0.3_0 al_0.3_60 al_0.1_0 al_0.05_0 \
  --model_al $MM/secondary_gmc_al_100um_estar/secondary_gmc.pth --kernel_al $MM/subcell_kernel_al_100um_estar \
  --N 200000 --n_seeds 2 --tag _estar > ../$L/lockwood_gmc_al_timing.log 2>&1
echo "lockwood timing rerun exit $? $(( $(date +%s) - t0 )) s"
echo "EVAL QUEUE DONE $(date)"
