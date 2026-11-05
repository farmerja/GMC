"""
Analog-MC references for chained GMC with secondaries. Run in the ElectronTransport venv (numba, h5py):

    ../ElectronTransport/.venv/bin/python validation/secondary_mc_refs.py --beam mev \
        --out validation/results/secondary_chain_cu_100um/mc_ref_mev.npz

Writes, for one beam on an nx x ny grid of square cells:
    source              the primaries every chain must reuse
    dose_full           analog MC on the whole grid at once: the true answer
    dose_full_noise     the same with another seed: MC-vs-MC noise floor
    dose_mcchain_sec    the analog MC as the cell kernel, all escaping electrons passed on (exact: must match full)
    dose_mcchain_prim   the analog MC as the cell kernel, primary only (escaping secondaries' energy charged to the
                        cell they leave): the structure of a GMC without secondaries
    material_map        (nx, ny) material index of every cell (indices into meta["materials"])
Beams (entering through the middle --beam_cells cells of the left side, training directions):
    mev   1-10 MeV, log-uniform        full  1 keV-10 MeV, log-uniform (the training spectrum)
Materials and layouts (--materials A[,B] --layout L): material 0 is the background, material 1 the second material.
    uniform   all cells material 0 (default; --materials Cu reproduces the original Cu reference)
    insert    material 1 block in columns 6-13, rows 5-14 of the 20 x 20 grid (scaled with nx, ny): the beam crosses
              0 -> 1 -> 0, and lateral scatter crosses the block's top and bottom faces
    slab      material 0 for x < nx/2, material 1 behind it
    Example (Cu block in Al):
    ../ElectronTransport/.venv/bin/python validation/secondary_mc_refs.py --beam mev --materials Al,Cu \
        --layout insert --out validation/results/secondary_chain_alcu_100um/mc_ref_mev.npz
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

from chain_secondaries import chain_secondaries, log_uniform_source, MCCellKernel  # noqa: E402

BEAMS = {"mev": (1.0, 10.0), "full": (1e-3, 10.0)}
MATERIALS = {"Cu": ("copper_mc_database.h5", 29), "Al": ("aluminum_mc_database.h5", 13)}  # in ElectronTransport/


def material_layout(layout, nx, ny):
    """(nx, ny) int map of material indices for a named layout (see the module docstring)."""
    m = np.zeros((nx, ny), dtype=np.int64)
    if layout == "insert":
        m[round(0.3 * nx):round(0.7 * nx), round(0.25 * ny):round(0.75 * ny)] = 1
    elif layout == "slab":
        m[nx // 2:, :] = 1
    elif layout != "uniform":
        raise ValueError(layout)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--et_dir", default=str(ROOT.parent / "ElectronTransport"))
    ap.add_argument("--materials", default="Cu", help=f"comma list from {sorted(MATERIALS)}; index = map value")
    ap.add_argument("--layout", default="uniform", choices=("uniform", "insert", "slab"))
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--cell_size", type=float, default=1e-2, help="cm (must match the GMC training cell)")
    ap.add_argument("--beam", choices=sorted(BEAMS), default="mev")
    ap.add_argument("--N", type=int, default=200000)
    ap.add_argument("--E_cutoff", type=float, default=1e-3)
    ap.add_argument("--beam_cells", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(str(ROOT), a.out)
    sys.path.insert(0, a.et_dir)
    os.chdir(a.et_dir)
    from analog_stepper import VoxelGeometry, run_transport
    from material_data import Material, build_library

    names = a.materials.split(",")
    mmap = material_layout(a.layout, a.nx, a.ny)
    if mmap.max() >= len(names):
        raise ValueError(f"layout {a.layout} needs {mmap.max() + 1} materials, got {names}")
    h5s = [MATERIALS[n][0] for n in names]
    lib = build_library([Material.from_element(h, MATERIALS[n][1]) for n, h in zip(names, h5s)], h5s)
    E_low, E_high = BEAMS[a.beam]
    y_max = a.ny * a.cell_size
    lo = (a.ny - a.beam_cells) // 2
    beam = (lo * a.cell_size, (lo + a.beam_cells) * a.cell_size)
    src = log_uniform_source(a.N, y_max, E_low, E_high, beam=beam, seed=a.seed)
    geo = VoxelGeometry(a.nx * a.cell_size, y_max, mmap)
    res = {}
    for name, seed in (("full", a.seed), ("full_noise", a.seed + 1000)):
        t0 = time.time()
        r = run_transport(lib, geo, src, E_cutoff=a.E_cutoff, seed=seed, n_tracks=0)
        res[name] = r
        print(f"{name}: {time.time() - t0:.1f} s, balance {json.dumps(r['energy_balance'], default=float)}",
              flush=True)
    chains = {}
    for name, sec in (("mcchain_sec", True), ("mcchain_prim", False)):
        t0 = time.time()
        kern = [MCCellKernel(lib, a.cell_size, a.E_cutoff, seed=a.seed + 1 + (0 if sec else 500) + 100003 * m,
                             secondaries=sec, material=m) for m in range(len(names))]
        c = chain_secondaries(kern, src, a.nx, a.ny, a.cell_size, a.E_cutoff, max_steps=a.max_steps, log_every=100,
                              material_map=mmap)
        chains[name] = c
        print(f"{name}: {time.time() - t0:.1f} s, {c['steps']} steps, {c['transits_boundary']} transits, "
              f"{c['extra_exits']} escaping secondaries passed on, escaped {c['escaped']}, "
              f"in flight {c['in_flight']:.3g}, photons {c['photon_E']:.4g}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, source=src, dose_full=res["full"]["energy_tally"], dose_full_noise=res["full_noise"]["energy_tally"],
        dose_mcchain_sec=chains["mcchain_sec"]["dose"], dose_mcchain_prim=chains["mcchain_prim"]["dose"],
        material_map=mmap, meta=json.dumps({**vars(a), "materials": names, "E_low": E_low, "E_high": E_high}),
        full_balance=json.dumps(res["full"]["energy_balance"], default=float),
        chain_stats=json.dumps({k: {kk: (vv if not isinstance(vv, np.ndarray) else None) for kk, vv in v.items()}
                                for k, v in chains.items()}, default=float))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
