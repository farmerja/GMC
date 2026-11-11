"""
Fine-mesh analog-MC reference for SCR-GMC (sub-cell dose reconstruction). Run in the ElectronTransport venv:

    ../ElectronTransport/.venv/bin/python validation/scr_mc_ref.py \
        --mc_ref validation/results/secondary_chain_cu_100um/mc_ref_mev.npz --n 8

Reuses the source, material map, grid and seed of a secondary_mc_refs.py reference and reruns the full-grid analog MC on
voxels refined n x n per transport cell. Voxel faces between equal materials change nothing physically (flights are
re-sampled at every face, exact since they are memoryless), so this is the same MC with an n-times finer tally.
Writes <mc_ref>_fine<n>.npz with
    dose_fine        (nx * n, ny * n) MeV per fine voxel, seed of the reference
    dose_fine_noise  the same with another seed: MC-vs-MC noise floor on the fine mesh
    n, material_map, meta (of the coarse reference), seconds (wall time of each run, JSON)
and prints the coarse-sum check against the reference's dose_full / dose_full_noise.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def coarsen(d, n):
    return d.reshape(d.shape[0] // n, n, d.shape[1] // n, n).sum((1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc_ref", required=True)
    ap.add_argument("--n", type=int, default=8, help="fine voxels per transport cell, per axis")
    ap.add_argument("--et_dir", default=str(ROOT.parent / "ElectronTransport"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref_path = a.mc_ref if os.path.isabs(a.mc_ref) else os.path.join(str(ROOT), a.mc_ref)
    out = a.out or ref_path[:-4] + f"_fine{a.n}.npz"
    R = np.load(ref_path, allow_pickle=False)
    meta = json.loads(str(R["meta"]))
    src = R["source"]
    if "materials" not in meta:   # single-material references made before --materials existed (Cu, Z = 29)
        meta["materials"] = [{29: "Cu", 13: "Al"}[int(meta["Z"])]]
    mmap = R["material_map"] if "material_map" in R.files else np.zeros((meta["nx"], meta["ny"]), dtype=np.int64)
    sys.path.insert(0, a.et_dir)
    os.chdir(a.et_dir)
    from analog_stepper import VoxelGeometry, run_transport
    from material_data import build_library, named_material

    mats = [named_material(m) for m in meta["materials"]]
    lib = build_library([m for m, _ in mats], [h for _, h in mats])
    nx, ny, L = meta["nx"], meta["ny"], meta["cell_size"]
    geo = VoxelGeometry(nx * L, ny * L, np.repeat(np.repeat(mmap, a.n, 0), a.n, 1))
    res, secs = {}, {}
    for name, seed, coarse_key in (("dose_fine", meta["seed"], "dose_full"),
                                   ("dose_fine_noise", meta["seed"] + 1000, "dose_full_noise")):
        t0 = time.time()
        r = run_transport(lib, geo, src, E_cutoff=meta["E_cutoff"], seed=seed, n_tracks=0)
        res[name] = r["energy_tally"]
        secs[name] = time.time() - t0
        c, ref = coarsen(res[name], a.n), R[coarse_key]
        print(f"{name}: {time.time() - t0:.1f} s; coarse sum vs {coarse_key}: rel-L2 "
              f"{np.sqrt(((c - ref) ** 2).sum() / (ref ** 2).sum()):.3e}, total {c.sum():.6g} vs {ref.sum():.6g}",
              flush=True)
    a_ = coarsen(res["dose_fine"], a.n)
    b_ = coarsen(res["dose_fine_noise"], a.n)
    print(f"coarse noise floor between the two fine runs: rel-L2 {np.sqrt(((a_ - b_) ** 2).sum() / (a_ ** 2).sum()):.3e}")
    np.savez_compressed(out, n=a.n, material_map=mmap, meta=json.dumps(meta), seconds=json.dumps(secs), **res)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
