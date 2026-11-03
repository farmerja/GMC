"""
Monte Carlo references for chained GMC electron transport. Run in the ElectronTransport environment (numba, h5py):

    ../ElectronTransport/.venv/bin/python validation/mc_chain.py --out validation/results/chain_analog_cu/mc_ref.npz

Writes one npz with
    source       the primaries (x, y, u, v, E, path, status, w) every chain must reuse
    dose_full    analog MC on the whole grid at once: the true answer (secondaries followed across cells)
    dose_mcchain the same analog MC used as a single-cell kernel inside validation/chain_core.chain: the primary only,
                 its whole energy loss charged to the cell it crosses. This is exactly the GMC model structure, so
                 it is the best any GMC cell model can do; GMC chain vs dose_mcchain is pure model error and
                 dose_mcchain vs dose_full is the cost of the primary-only / local-deposit structure.
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

from chain_core import chain, make_source  # noqa: E402
from ml.cell_topology import classes_from_status  # noqa: E402


class MCKernel:
    """Analog MC (analog_stepper.run_transport) on one cell, as a chain_core kernel."""

    def __init__(self, library, material, cell_size, E_cutoff, seed=1000):
        from analog_stepper import VoxelGeometry, run_transport
        self.run_transport = run_transport
        self.lib, self.cell, self.E_cut, self.seed = library, cell_size, E_cutoff, seed
        self.geo = VoxelGeometry.uniform(cell_size, cell_size, 1, 1, material=library.index(material))

    def __call__(self, y_c, u_c, v_c, w, E, step):
        p = np.zeros((len(E), 8))
        p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 7] = y_c * self.cell, u_c, v_c, E, w
        f = self.run_transport(self.lib, self.geo, p, E_cutoff=self.E_cut, seed=self.seed + 7919 * step,
                               n_tracks=0)["final"]
        status = f[:, 6].astype(int)
        if np.any(status == 6):
            raise RuntimeError("status 6 (lost along z) inside a filled cell")
        return {"cls": classes_from_status(status), "x": f[:, 0] / self.cell, "y": f[:, 1] / self.cell,
                "u": f[:, 2], "v": f[:, 3], "w": f[:, 7], "E_out": f[:, 4]}


def main():
    ap = argparse.ArgumentParser(description="Analog MC references (full grid + MC-chained) for GMC chaining")
    ap.add_argument("--et_dir", default=str(ROOT.parent / "ElectronTransport"))
    ap.add_argument("--h5", default="copper_mc_database.h5")
    ap.add_argument("--Z", type=int, default=29)
    ap.add_argument("--material", default="Copper")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--cell_size", type=float, default=1e-4, help="cm (must match the GMC training cell)")
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--E_low", type=float, default=1e-3)
    ap.add_argument("--E_high", type=float, default=1e-1)
    ap.add_argument("--E_cutoff", type=float, default=1e-3)
    ap.add_argument("--v_spread", type=float, default=1.0)
    ap.add_argument("--w_max_angle", type=float, default=90.0)
    ap.add_argument("--beam_cells", type=int, default=1,
                    help="Beam enters through the middle k cells of the left boundary (0 = full height)")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="validation/results/chain_analog_cu/mc_ref.npz")
    a = ap.parse_args()

    sys.path.insert(0, a.et_dir)
    os.chdir(a.et_dir)  # material tables are opened relative to the ElectronTransport directory
    from analog_stepper import VoxelGeometry, run_transport
    from material_data import Material, build_library

    lib = build_library([Material.from_element(a.h5, a.Z)], a.h5)
    y_max = a.ny * a.cell_size
    beam = None
    if a.beam_cells > 0:
        lo = (a.ny - a.beam_cells) // 2
        beam = (lo * a.cell_size, (lo + a.beam_cells) * a.cell_size)
    src = make_source(a.N, y_max, a.E_low, a.E_high, a.v_spread, a.w_max_angle, beam=beam, seed=a.seed)

    t0 = time.time()
    geo = VoxelGeometry.uniform(a.nx * a.cell_size, y_max, a.nx, a.ny, material=lib.index(a.material))
    full = run_transport(lib, geo, src, E_cutoff=a.E_cutoff, seed=a.seed, n_tracks=0)
    print(f"full-grid MC: {time.time() - t0:.1f} s, balance {json.dumps(full['energy_balance'], default=float)}")

    t0 = time.time()
    kern = MCKernel(lib, a.material, a.cell_size, a.E_cutoff, seed=a.seed + 1)
    mcc = chain(kern, src, a.nx, a.ny, a.cell_size, max_steps=a.max_steps, E_cutoff=a.E_cutoff, log_every=50)
    print(f"MC-chained: {time.time() - t0:.1f} s, {mcc['steps']} steps, {mcc['transits']} transits, "
          f"escaped {mcc['escaped']}, in flight {mcc['in_flight']:.3g}")

    out = os.path.join(str(ROOT), a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, source=src, dose_full=full["energy_tally"], dose_mcchain=mcc["dose"],
        escaped_mcchain=json.dumps(mcc["escaped"]), meta=json.dumps(vars(a)),
        full_balance=json.dumps(full["energy_balance"], default=float),
        escaped_full_electrons=float(full["energy_balance"]["E_escaped_electrons"]),
        photons_full=float(full["energy_balance"]["E_photons"]))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
