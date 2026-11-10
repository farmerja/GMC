#!/usr/bin/env python3
"""
GMC cost per cell transit (GPU), for comparison with the analog-MC cost per primary per cell from
ElectronTransport/sweep_cell_size_energy.py.

The network cost of one GMC call does not depend on the cell size or the material: every model works in the unit
cell, and the networks are the same size for every material. So one measurement per (batch size, ODE steps,
entry energy) is the GMC side of the cost comparison for every cell size. Entry energy only matters through how
often the daughter and photon flows run.

Measures SecondaryGMC.boundary (an electron entering through a face) and SecondaryGMC.interior (a relevant
secondary at its birth point) for batch sizes 1e2..3e5 and ODE steps 50 (the validated default), 20, 10 and 5.
Accuracy at fewer steps is NOT checked here (run secondary_cell_compare.py / secondary_chain_compare.py with
--ode_steps for that).

    python validation/gmc_timing.py --model ml/models/secondary_gmc_cu_100um/secondary_gmc.pth \
        --out validation/results/gmc_timing/gmc_timing.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ml.secondary_gmc import SecondaryGMC  # noqa: E402


def entries(n, E, rng):
    """Boundary entries in the canonical frame (left face), inward hemisphere."""
    mu = rng.uniform(0, 1, n)                  # u = cos(angle from the inward normal)
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - mu * mu)
    return rng.uniform(0, 1, n), mu, s * np.cos(phi), s * np.sin(phi), np.full(n, E)


def timed(fn, reps):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="100,1000,10000,100000,300000")
    ap.add_argument("--steps", default="50,20,10,5")
    ap.add_argument("--energies", default="0.01,0.1,1,10", help="MeV")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    rows = []
    for steps in [int(s) for s in a.steps.split(",")]:
        K = SecondaryGMC(a.model, dev, ode_steps=steps, seed=1)
        y, u, v, w, E = entries(1000, 1.0, rng)
        K.boundary(y, u, v, w, E)                                       # warm-up (CUDA kernels, allocator)
        K.interior(rng.uniform(0, 1, 1000), y, u, v, w, E)
        for n in [int(b) for b in a.batches.split(",")]:
            for E0 in [float(e) for e in a.energies.split(",")]:
                y, u, v, w, E = entries(n, E0, rng)
                x = rng.uniform(0, 1, n)
                reps = int(min(10, max(1, 2e4 // n)))   # small batches are latency-bound: a few repeats suffice
                tb, ob = timed(lambda: K.boundary(y, u, v, w, E), reps)
                ti, oi = timed(lambda: K.interior(x, y, u, v, w, E), reps)
                row = dict(steps=steps, batch=n, E=E0, us_boundary=1e6 * tb / n, us_interior=1e6 * ti / n,
                           daughters_per_boundary=len(ob["daughters"]["E"]) / n,
                           daughters_per_interior=len(oi["daughters"]["E"]) / n, device=dev)
                rows.append(row)
                print(f"steps {steps:3d} batch {n:7d} E {E0:6.3g} MeV: boundary {row['us_boundary']:8.2f} us, "
                      f"interior {row['us_interior']:8.2f} us per transit", flush=True)
    meta = {"model": a.model, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
