#!/usr/bin/env python3
"""
Peak host RAM and GPU memory of one Drumm chain batch (x2 GMC, 20 ODE steps, MirrorY) + SCR tally (K = 32), to size
the --batch of drumm_cylinder.py / drumm_sweep.py. Run one batch size per process (peak RSS is per process):

    for b in 25000 100000 400000; do python validation/drumm_batch_memory.py $b; done
"""
import os
import resource
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "validation")]
from chain_secondaries import chain_secondaries  # noqa: E402
from drumm_cylinder import L, NC, MirrorY, VacuumKernel, stair_map, triangle_source  # noqa: E402
from ml.secondary_gmc import SecondaryGMC  # noqa: E402
from ml.subcell_kernel import load_kernel, scr_tally  # noqa: E402

B = int(sys.argv[1])
M = os.path.join(ROOT, "ml/models/secondary_gmc_water_1mm_30MeV_elsepa_x2/secondary_gmc.pth")
K = os.path.join(ROOT, "ml/models/subcell_kernel_water_1mm_30MeV_elsepa_x2")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # ru_maxrss in KB on Linux


spec, model, _ = load_kernel(K, "cuda")
base = rss_gb()
torch.cuda.reset_peak_memory_stats()
src = triangle_source(B, 2007)
t0 = time.time()
c = chain_secondaries([MirrorY(SecondaryGMC(M, "cuda", seed=1), 4), VacuumKernel()], src, NC, NC, L, 1e-3,
                      log_every=0, material_map=stair_map(), transit_log=True)
tc = time.time() - t0
after_chain = rss_gb()
gpu_chain = torch.cuda.max_memory_allocated() / 1e9
ntr = len(c["transits"]["E0"])
logbytes = sum(np.asarray(v).nbytes for v in c["transits"].values())
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
f, _ = scr_tally(c, {0: (spec, model)}, NC, NC, 8, n_samples=32, device="cuda", gen=torch.Generator("cuda").manual_seed(0),
                 sampling="sobol")
torch.cuda.synchronize()
tt = time.time() - t0
print(f"B={B}: chain {tc:.0f} s, tally {tt:.0f} s | peak RSS {rss_gb():.2f} GB (after model load {base:.2f}, after chain "
      f"{after_chain:.2f}) | transit log {ntr / 1e6:.2f} M rows, {logbytes / 1e9:.2f} GB | GPU peak allocated: chain "
      f"{gpu_chain:.2f} GB, tally {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, reserved "
      f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB", flush=True)
