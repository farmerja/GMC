# validation/Macro_field.py
"""
GMC macro-chaining dose plot for 2D electrons: chain a single-cell model over an nx x ny block of square cells.

Cell models (validation/gmc_kernels.py), all chained by the same loop (validation/chain_core.py):
    --checkpoint                          2d_electron flow (exit position = nearest face of its 2D point)
    --checkpoint + --exit_classifier      same flow, exit class sampled from the classifier, flow redrawn to agree
    --face_ckpt + --exit_classifier       exit classifier + face-conditioned flow (2d_electron_face); --checkpoint
                                          is then not needed
For a comparison against analog MC on the same grid see validation/electron_chain_compare.py.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ml.utils import setup_environment, norm_params_base_path  # noqa: E402
from ml.transform import resolve_norm_params_base_path  # noqa: E402
from chain_core import chain, make_source, transform_to_canonical, transform_from_canonical  # noqa: E402,F401
from gmc_kernels import BaselineFlowKernel, FaceFlowKernel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def build_source(num_particles, y_max, E_in_low, E_in_high, source, v_spread, w_max_angle_deg,
                 max_incoming_angle_deg, seed):
    """'mc': the ElectronMC.generate_particles (training) source; 'inplane': uniform in-plane angle, w = 0."""
    if source == "mc":
        return make_source(num_particles, y_max, E_in_low, E_in_high, v_spread, w_max_angle_deg, seed=seed)
    if source == "inplane":
        rng = np.random.default_rng(seed)
        p = np.zeros((num_particles, 8))
        p[:, 1] = rng.uniform(0.0, y_max, num_particles)
        ang = rng.uniform(-np.radians(max_incoming_angle_deg), np.radians(max_incoming_angle_deg), num_particles)
        p[:, 2], p[:, 3] = np.cos(ang), np.sin(ang)
        p[:, 4] = rng.uniform(E_in_low, E_in_high, num_particles)
        return p
    raise ValueError(f"Unknown source '{source}'. Expected 'mc' or 'inplane'.")


def simulate_macrofield_block(kernel, E_cutoff, num_particles=50000, E_in_low=0.1, E_in_high=10.0, num_cells_x=10,
                              num_cells_y=10, cell_size=1.0, max_macro_steps=100, max_incoming_angle_deg=85.0,
                              source="mc", v_spread=1.0, w_max_angle_deg=60.0, seed=42):
    """Returns (dose density [MeV / area / primary], nx, ny, cell_dx, cell_dy)."""
    log.info(f"Source: Uniform[{E_in_low:.4g}, {E_in_high:.4g}] MeV, cutoff {E_cutoff:.4g} MeV | "
             f"{num_cells_x}x{num_cells_y} cells of {cell_size:.4g}")
    src = build_source(num_particles, num_cells_y * cell_size, E_in_low, E_in_high, source, v_spread,
                       w_max_angle_deg, max_incoming_angle_deg, seed)
    res = chain(kernel, src, num_cells_x, num_cells_y, cell_size, max_steps=max_macro_steps, E_cutoff=E_cutoff,
                log=log.info)
    E_src = res["E_source"]
    log.info(f"Done after {res['steps']} steps: deposited {res['dose'].sum() / E_src:.4f}, escaped "
             + ", ".join(f"{k} {v / E_src:.4f}" for k, v in res["escaped"].items())
             + f", still in flight {res['in_flight'] / E_src:.4f} of the source energy")
    if getattr(kernel, "sampled", 0):
        log.info(f"Exit classifier: {100 * kernel.unmatched / kernel.sampled:.3f}% of cell transits kept an "
                 f"unmatched flow sample")
    dose_density = res["dose"] / (num_particles * cell_size * cell_size)
    return dose_density, num_cells_x, num_cells_y, cell_size, cell_size


def main():
    parser = argparse.ArgumentParser(description="GMC Macro-Chaining Block Spatial Dose Plot")
    parser.add_argument("--checkpoint", type=str, default=None, help="2d_electron flow best_model.pth")
    parser.add_argument("--norm_params", type=str, default=None,
                        help="Norm params of --checkpoint (.npy base); default ml/data/norm_params_2d_electron_boundary")
    parser.add_argument("--face_ckpt", type=str, default=None, help="2d_electron_face flow (needs --exit_classifier)")
    parser.add_argument("--face_norm", type=str, default=None, help="Norm params of --face_ckpt (.npy base)")
    parser.add_argument("--exit_classifier", type=str, default=None, help="Exit-class checkpoint (ml/exit_classifier.py)")
    parser.add_argument("--num_particles", type=int, default=50000)
    parser.add_argument("--E_in_low", type=float, default=0.1, help="Minimum energy in MeV")
    parser.add_argument("--E_in_high", type=float, default=10.0, help="Maximum energy in MeV")
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--ode_method", type=str, default="euler")
    parser.add_argument("--batch_size", type=int, default=256, help="(unused; kept for old command lines)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="(unused; architecture is read from the checkpoint)")
    parser.add_argument("--depth", type=int, default=8, help="(unused; architecture is read from the checkpoint)")
    parser.add_argument("--expansion_factor", type=int, default=1, help="(unused; read from the checkpoint)")
    parser.add_argument("--num_cells_x", type=int, default=10)
    parser.add_argument("--num_cells_y", type=int, default=10)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--max_macro_steps", type=int, default=100)
    parser.add_argument("--source", type=str, default="mc", choices=["mc", "inplane"],
                        help="Entry beam: 'mc' matches ElectronMC.generate_particles; 'inplane' is uniform-angle, w=0")
    parser.add_argument("--v_spread", type=float, default=1.0, help="Half-range of v for source='mc'")
    parser.add_argument("--w_max_angle", type=float, default=60.0, help="Max out-of-plane tilt (deg) for source='mc'")
    parser.add_argument("--max_incoming_angle", type=float, default=85.0,
                        help="Max in-plane angle (deg) for source='inplane'")
    parser.add_argument("--stop_tol", type=float, default=1e-4,
                        help="Baseline flow: E_out <= E_cutoff + stop_tol (MeV) counts as stopped")
    parser.add_argument("--train_sampling", type=str, default="random")
    parser.add_argument("--scheduler_type", type=str, default="condOT", help="(unused; kept for old command lines)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="spatial_dose_density.png")
    parser.add_argument("--dose_out", type=str, default="code1_dose.npz")
    args = parser.parse_args()

    device = setup_environment(seed=args.seed, allow_mps=False, inference=True)
    kw = dict(device=device, ode_steps=args.ode_steps, ode_method=args.ode_method, seed=args.seed)
    if args.face_ckpt:
        if not (args.exit_classifier and args.face_norm):
            parser.error("--face_ckpt needs --exit_classifier and --face_norm")
        kernel = FaceFlowKernel(args.face_ckpt, args.face_norm, args.exit_classifier, **kw)
        label = "classifier + face flow"
    elif args.checkpoint:
        norm_path = args.norm_params or resolve_norm_params_base_path(
            norm_params_base_path("2d_electron", "boundary"), args.train_sampling)
        kernel = BaselineFlowKernel(args.checkpoint, norm_path, classifier=args.exit_classifier,
                                    stop_tol=args.stop_tol, **kw)
        label = "flow" + (" + classifier" if args.exit_classifier else "")
    else:
        parser.error("give --checkpoint or --face_ckpt")
    log.info(f"Cell model: {label}")

    dose_density, nx, ny, cell_dx, cell_dy = simulate_macrofield_block(
        kernel, kernel.E_cut, num_particles=args.num_particles, E_in_low=args.E_in_low, E_in_high=args.E_in_high,
        num_cells_x=args.num_cells_x, num_cells_y=args.num_cells_y, cell_size=args.cell_size,
        max_macro_steps=args.max_macro_steps, max_incoming_angle_deg=args.max_incoming_angle, source=args.source,
        v_spread=args.v_spread, w_max_angle_deg=args.w_max_angle, seed=args.seed)

    fig, ax_map = plt.subplots(figsize=(8, 7))
    v_max = max(2.2, np.max(dose_density)) if len(dose_density) > 0 else 2.2
    x_edges = np.linspace(0.0, nx * cell_dx, nx + 1)
    y_edges = np.linspace(0.0, ny * cell_dy, ny + 1)
    im = ax_map.imshow(dose_density.T, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       aspect="equal", cmap="hot", vmin=0.0, vmax=v_max)
    ax_map.set_title(f"GMC Spatial Dose Density ({nx} x {ny}, {label})", fontsize=14, pad=12)
    ax_map.set_xlabel("X Position (cm)", fontsize=12)
    ax_map.set_ylabel("Y Position (cm)", fontsize=12)
    ax_map.set_xticks(x_edges)
    ax_map.set_yticks(y_edges)
    ax_map.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    cbar = fig.colorbar(im, ax=ax_map, orientation="vertical", pad=0.03, fraction=0.046)
    cbar.set_label("Spatial Dose Density (MeV / cm² / electron)", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    log.info(f"Plot saved successfully to {args.output}")
    np.savez(args.dose_out, dose_density=dose_density, nx=nx, ny=ny, cell_dx=cell_dx, cell_dy=cell_dy)
    print(f"Exported dose data to {args.dose_out}")


if __name__ == "__main__":
    main()
