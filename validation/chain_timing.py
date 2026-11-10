#!/usr/bin/env python3
"""
Measured end-to-end cost, full-grid analog MC vs chained GMC with secondaries, on the same 20 x 20 grid and source
as secondary_mc_refs.py (beam on the 2 central rows of the left face), for several beam energies.

Two steps, two venvs (the MC needs numba + h5py, the GMC needs torch):

    # 1. MC: ElectronTransport venv. Two independent full-grid runs per beam (seed, seed + 1000): time + noise floor
    ../ElectronTransport/.venv/bin/python validation/chain_timing.py mc --material Cu \
        --out validation/results/gmc_timing/chain_timing_cu.npz
    # 2. GMC: GMC venv. Times the chain at each --ode_steps, scores the dose map against MC run 1
    python validation/chain_timing.py gmc --mc validation/results/gmc_timing/chain_timing_cu.npz \
        --model ml/models/secondary_gmc_cu_100um/secondary_gmc.pth

The MC uses all numba threads (16 here); the GMC runs on the GPU. rel-L2 as in secondary_chain_compare.py; the
noise floor is MC run 2 vs MC run 1. One GMC seed per setting: run-to-run spread is ~0.1-0.2 % (see the results
pages), so differences below that are not resolved.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_secondaries import chain_secondaries, log_uniform_source  # noqa: E402

BEAMS = {"100keV": (0.1, 0.1), "1MeV": (1.0, 1.0), "10MeV": (10.0, 10.0), "full": (1e-3, 10.0)}


def source(beam, N, nx, ny, L, seed):
    lo = (ny - 2) // 2
    E_low, E_high = BEAMS[beam]
    return log_uniform_source(N, ny * L, E_low, E_high, beam=(lo * L, (lo + 2) * L), seed=seed)


def rel_l2(d, ref):
    return float(np.sqrt(((d - ref) ** 2).sum() / (ref ** 2).sum()))


def cmd_mc(a):
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    from analog_stepper import VoxelGeometry, run_transport
    from material_data import build_library, named_material
    mat, h5 = named_material(a.material)
    lib = build_library([mat], h5)
    geo = VoxelGeometry.uniform(a.nx * a.cell_size, a.ny * a.cell_size, a.nx, a.ny, material=0)
    run_transport(lib, geo, source("1MeV", 200, a.nx, a.ny, a.cell_size, 0), n_tracks=0)  # numba warm-up
    out = {"meta": json.dumps({**vars(a), "beams": a.beams})}
    for beam in a.beams.split(","):
        src = source(beam, a.N, a.nx, a.ny, a.cell_size, a.seed)
        for k, seed in (("mc", a.seed), ("noise", a.seed + 1000)):
            t = time.time()
            r = run_transport(lib, geo, src, E_cutoff=a.E_cutoff, seed=seed, n_tracks=0)
            out[f"{beam}_{k}_s"] = time.time() - t
            out[f"{beam}_{k}"] = r["energy_tally"]   # MeV per cell, summed over primaries (as the chain dose)
        print(f"{a.material} {beam}: MC {out[f'{beam}_mc_s']:.1f} s / {out[f'{beam}_noise_s']:.1f} s for {a.N} "
              f"primaries, noise floor {rel_l2(out[f'{beam}_noise'], out[f'{beam}_mc']):.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"saved {a.out}")


def cmd_gmc(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    ref = np.load(a.mc)
    m = json.loads(str(ref["meta"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = []
    for beam in m["beams"].split(","):
        src = source(beam, m["N"], m["nx"], m["ny"], m["cell_size"], m["seed"])
        floor = rel_l2(ref[f"{beam}_noise"], ref[f"{beam}_mc"])
        for steps in [int(s) for s in a.ode_steps.split(",")]:
            K = SecondaryGMC(a.model, dev, ode_steps=steps, seed=a.seed)
            if not res:
                chain_secondaries([K], src[:2000], m["nx"], m["ny"], m["cell_size"], m["E_cutoff"], log_every=0)
            t = time.time()
            c = chain_secondaries([K], src, m["nx"], m["ny"], m["cell_size"], m["E_cutoff"], log_every=0)
            dt = time.time() - t
            n_tr = c["transits_boundary"] + c["transits_interior"]
            row = dict(material=m["material"], beam=beam, ode_steps=steps, N=m["N"], gmc_s=dt,
                       mc_s=float(ref[f"{beam}_mc_s"]), speedup=float(ref[f"{beam}_mc_s"]) / dt,
                       transits=int(n_tr), us_per_transit=1e6 * dt / n_tr,
                       mc_us_per_transit=1e6 * float(ref[f"{beam}_mc_s"]) / n_tr,
                       rel_l2=rel_l2(c["dose"], ref[f"{beam}_mc"]), noise_floor=floor,
                       dose_ratio=float(c["dose"].sum() / ref[f"{beam}_mc"].sum()))
            res.append(row)
            print(f"{m['material']} {beam} steps {steps}: GMC {dt:.1f} s vs MC {row['mc_s']:.1f} s "
                  f"(x{row['speedup']:.2f}), {row['us_per_transit']:.1f} us/transit, rel-L2 {row['rel_l2']:.4f} "
                  f"(floor {floor:.4f}), dose ratio {row['dose_ratio']:.4f}", flush=True)
    out = a.out or a.mc.replace(".npz", "_gmc.json")
    json.dump({"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "model": a.model,
               "rows": res}, open(out, "w"), indent=1)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mc")
    p.add_argument("--material", default="Cu")
    p.add_argument("--beams", default="100keV,1MeV,10MeV,full")
    p.add_argument("--N", type=int, default=100000)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=20)
    p.add_argument("--cell_size", type=float, default=1e-2)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", required=True)
    g = sub.add_parser("gmc")
    g.add_argument("--mc", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--ode_steps", default="50,10")
    g.add_argument("--seed", type=int, default=11)
    g.add_argument("--out", default=None)
    a = ap.parse_args()
    cmd_mc(a) if a.cmd == "mc" else cmd_gmc(a)


if __name__ == "__main__":
    main()
