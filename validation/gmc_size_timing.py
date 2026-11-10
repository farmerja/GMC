#!/usr/bin/env python3
"""
GMC cost per cell transit vs network size and ODE steps: would a smaller model win back speed in thin cells?

Every network inside a SecondaryGMC kernel (classifiers and the exit / daughter / photon flows) is swapped for an
UNTRAINED one of the given size, with the same inputs and outputs, and a full SecondaryGMC.boundary transit is timed.
Speed does not depend on the weights, so this is the cost a trained model of that size would have. The kernel's
bookkeeping (numpy, CPU <-> GPU copies) is included, so the "floor" row (a tiny network, one ODE step) is the cost no
network size can go below.

Caveat: untrained classifiers draw random relevant-daughter counts, so the daughter flow runs more often than with a
trained model. The swapped sizes are therefore slightly pessimistic; the 256 x 6 row uses the real trained model.
Accuracy of smaller / fewer-step models is NOT tested here.

    python validation/gmc_size_timing.py --out validation/results/gmc_timing/gmc_size_timing.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.exit_classifier import ExitClassifier, ExitClassifierConfig  # noqa: E402
from ml.model import get_model  # noqa: E402
from ml.secondary_gmc import SecondaryGMC  # noqa: E402

TRAINED = (256, 6)   # hidden, depth of the trained secondary GMC flows


def entries(n, E, rng):
    """Boundary entries in the canonical frame (left face), inward hemisphere."""
    mu = rng.uniform(0, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - mu * mu)
    return rng.uniform(0, 1, n), mu, s * np.cos(phi), s * np.sin(phi), np.full(n, E)


def shrink(K, hidden, depth, dev):
    """Replace every network of kernel K by an untrained one of the given size (same input / output dims)."""
    for m in K.m.values():
        if m["kind"] == "flow":
            m["model"] = get_model(m["cond_dim"] + m["target_dim"], m["cond_dim"], hidden_dim=hidden, depth=depth,
                                   expansion_factor=2).to(dev).eval()
        else:
            c = m["model"].cfg
            m["model"] = ExitClassifier(ExitClassifierConfig(cond_dim=c.cond_dim, num_classes=c.num_classes,
                                                             hidden_dim=hidden, depth=max(1, depth - 2),
                                                             expansion=2)).to(dev).eval()


def time_transit(K, n, E, rng, reps):
    y, u, v, w, e = entries(n, E, rng)
    K.boundary(y, u, v, w, e)                      # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        K.boundary(y, u, v, w, e)
    torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t) / reps / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ml/models/secondary_gmc_cu_100um/secondary_gmc.pth",
                    help="trained checkpoint: supplies the I/O dims and the 256 x 6 reference row")
    ap.add_argument("--sizes", default="256x6,128x4,64x3,32x2", help="hidden x depth list")
    ap.add_argument("--steps", default="50,10,4,1")
    ap.add_argument("--batches", default="10000,100000")
    ap.add_argument("--E", type=float, default=1.0, help="entry energy (MeV)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU (timings are for the GPU kernel)")
    dev = "cuda"
    rng = np.random.default_rng(0)
    rows = []
    for size in a.sizes.split(","):
        h, d = (int(x) for x in size.split("x"))
        for steps in (int(s) for s in a.steps.split(",")):
            K = SecondaryGMC(a.model, dev, ode_steps=steps, seed=1)
            if (h, d) != TRAINED:
                shrink(K, h, d, dev)
            n_par = sum(p.numel() for p in K.m["exit_B"]["model"].parameters())
            for n in (int(b) for b in a.batches.split(",")):
                us = time_transit(K, n, a.E, rng, a.reps)
                rows.append(dict(hidden=h, depth=d, steps=steps, batch=n, us_per_transit=us, exit_flow_params=n_par,
                                 trained=(h, d) == TRAINED))
                print(f"{h:3d} x {d}  ({n_par:7d} params)  steps {steps:2d}  batch {n:6d}: {us:7.3f} us/transit",
                      flush=True)
    floor = []
    K = SecondaryGMC(a.model, dev, ode_steps=1, seed=1)
    shrink(K, 8, 1, dev)
    for n in (10000, 100000, 300000):
        us = time_transit(K, n, a.E, rng, a.reps)
        floor.append(dict(batch=n, us_per_transit=us))
        print(f"floor (8 x 1, 1 step)  batch {n:6d}: {us:7.3f} us/transit", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "E_MeV": a.E, "model": a.model,
               "rows": rows, "floor": floor}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
