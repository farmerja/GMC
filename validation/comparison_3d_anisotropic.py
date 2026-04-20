#!/usr/bin/env python3
"""3D anisotropic scattering validation: GMC vs MC comparison.

This script validates 3D anisotropic scattering models by:
1. Running GMC with trained g=±0.8 models (boundary and internal sources)
2. Running reference MC simulations with same scattering parameters
3. Comparing flux distributions and statistics

The 3D geometry extends the 2D lattice benchmark symmetrically in z-direction
with height = 3.0 and open boundaries on all faces.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.mc3d import run_mc_simulation_3d_njit_wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D anisotropic scattering validation")
    parser.add_argument("--grid-size", type=int, default=7,
                        help="Number of cells per side in x,y (lattice pattern)")
    parser.add_argument("--z-cells", type=int, default=3,
                        help="Number of cells in z-direction")
    parser.add_argument("--cell-width", type=float, default=1.0,
                        help="Width of each cell in x,y")
    parser.add_argument("--cell-height", type=float, default=1.0,
                        help="Height of each cell in z-direction")
    parser.add_argument("--num-particles", type=int, default=50000,
                        help="Number of MC particles to simulate")
    parser.add_argument("--scattering-mode", type=str, default="anisotropic-hg",
                        choices=["isotropic", "anisotropic-hg"],
                        help="Scattering physics mode")
    parser.add_argument("--g-hg", type=float, default=0.8,
                        help="Henyey-Greenstein asymmetry parameter: "
                             "g=0 is isotropic, g=0.8 is forward-peaked, g=-0.8 is backward-peaked")
    parser.add_argument("--source-type", type=str, default="internal",
                        choices=["boundary", "internal"],
                        help="Source type: boundary (x=0 face) or internal (center cell)")
    parser.add_argument("--scatter-ratio", type=float, default=0.9,
                        help="Scattering ratio (sigma_s / sigma_t)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="validation/results/comparison_3d_anisotropic")
    return parser.parse_args()


def create_lattice_3d_geometry(grid_size: int, z_cells: int, cell_width: float, cell_height: float):
    """Create 3D lattice geometry with alternating material pattern.
    
    The 2D lattice pattern (fuel/moderator) is extended uniformly in z-direction.
    Material pattern matches the 2D benchmark:
    - Center 3×3 cells: fuel (high scattering)
    - Outer cells: moderator (lower scattering)
    """
    num_cells_x = grid_size
    num_cells_y = grid_size
    num_cells_z = z_cells
    
    cell_widths = np.full(num_cells_x, cell_width, dtype=np.float64)
    cell_heights = np.full(num_cells_y, cell_width, dtype=np.float64)  # Square in xy
    cell_depths = np.full(num_cells_z, cell_height, dtype=np.float64)
    
    # Material properties (matching 2D lattice benchmark)
    # Fuel: center 3×3 region, Moderator: outer region
    sigma_t_fuel = 1.0
    sigma_s_fuel = 0.9  # High scattering
    sigma_t_mod = 1.0
    sigma_s_mod = 0.5   # Lower scattering
    
    # Create material maps (uniform in z)
    sigma_t_map = np.full((num_cells_z, num_cells_y, num_cells_x), sigma_t_mod, dtype=np.float64)
    sigma_s_map = np.full((num_cells_z, num_cells_y, num_cells_x), sigma_s_mod, dtype=np.float64)
    
    # Set fuel region (center 3×3 in xy, all z)
    center_start = (grid_size - 3) // 2
    center_end = center_start + 3
    sigma_t_map[:, center_start:center_end, center_start:center_end] = sigma_t_fuel
    sigma_s_map[:, center_start:center_end, center_start:center_end] = sigma_s_fuel
    
    sigma_a_map = sigma_t_map - sigma_s_map
    
    # Cell boundaries
    cell_boundaries_x = np.concatenate(([0.0], np.cumsum(cell_widths)))
    cell_boundaries_y = np.concatenate(([0.0], np.cumsum(cell_heights)))
    cell_boundaries_z = np.concatenate(([0.0], np.cumsum(cell_depths)))
    
    # Source location (center cell in 3D)
    source_cell_x = num_cells_x // 2
    source_cell_y = num_cells_y // 2
    source_cell_z = num_cells_z // 2
    
    return {
        "num_cells_x": num_cells_x,
        "num_cells_y": num_cells_y,
        "num_cells_z": num_cells_z,
        "cell_widths": cell_widths,
        "cell_heights": cell_heights,
        "cell_depths": cell_depths,
        "cell_boundaries_x": cell_boundaries_x,
        "cell_boundaries_y": cell_boundaries_y,
        "cell_boundaries_z": cell_boundaries_z,
        "sigma_t_map": sigma_t_map,
        "sigma_a_map": sigma_a_map,
        "sigma_s_map": sigma_s_map,
        "source_cell": (source_cell_x, source_cell_y, source_cell_z),
        "domain_size_x": float(cell_boundaries_x[-1]),
        "domain_size_y": float(cell_boundaries_y[-1]),
        "domain_size_z": float(cell_boundaries_z[-1]),
    }


def run_3d_mc_reference(
    geometry: dict,
    *,
    num_particles: int,
    source_type: str,
    scattering_mode: str,
    g_hg: float,
    seed: int,
) -> dict:
    """Run reference 3D MC simulation."""
    
    print(f"\n{'='*70}")
    print(f"Running 3D MC Reference ({source_type} source)")
    print(f"  Grid: {geometry['num_cells_x']}×{geometry['num_cells_y']}×{geometry['num_cells_z']}")
    print(f"  Particles: {num_particles:,}")
    print(f"  Scattering: {scattering_mode}")
    if scattering_mode == "anisotropic-hg":
        print(f"  HG parameter g: {g_hg:.3f}")
    print(f"{'='*70}")
    
    t0 = time.perf_counter()
    
    if source_type == "boundary":
        # Boundary source at x=0 face (middle of yz plane)
        result = run_mc_simulation_3d_njit_wrapper(
            source_mode="boundary_only",
            num_cells_x=geometry["num_cells_x"],
            num_cells_y=geometry["num_cells_y"],
            num_cells_z=geometry["num_cells_z"],
            cell_widths=geometry["cell_widths"],
            cell_heights=geometry["cell_heights"],
            cell_depths=geometry["cell_depths"],
            cell_boundaries_x=geometry["cell_boundaries_x"],
            cell_boundaries_y=geometry["cell_boundaries_y"],
            cell_boundaries_z=geometry["cell_boundaries_z"],
            total_width=geometry["domain_size_x"],
            total_height=geometry["domain_size_y"],
            total_depth=geometry["domain_size_z"],
            num_particles=num_particles,
            sigma_t_per_cell=geometry["sigma_t_map"],
            sigma_a_per_cell=geometry["sigma_a_map"],
            energy_threshold=1.0e-8,
            seed=seed,
            reflected_boundaries=False,
            angular_sampling=1,
            scattering_mode=scattering_mode,
            g_hg=g_hg,
            verbose=True,
        )
        energy_deposited = result[0]
        track_length = result[1]
        stats = result[2] if len(result) > 2 else {}
    else:  # internal
        # Internal source at center cell
        cx, cy, cz = geometry["source_cell"]
        result = run_mc_simulation_3d_njit_wrapper(
            source_mode="internal_start",
            num_cells_x=geometry["num_cells_x"],
            num_cells_y=geometry["num_cells_y"],
            num_cells_z=geometry["num_cells_z"],
            cell_widths=geometry["cell_widths"],
            cell_heights=geometry["cell_heights"],
            cell_depths=geometry["cell_depths"],
            cell_boundaries_x=geometry["cell_boundaries_x"],
            cell_boundaries_y=geometry["cell_boundaries_y"],
            cell_boundaries_z=geometry["cell_boundaries_z"],
            total_width=geometry["domain_size_x"],
            total_height=geometry["domain_size_y"],
            total_depth=geometry["domain_size_z"],
            num_particles=num_particles,
            start_cell_indices=[cx, cy, cz],
            start_position_relative=[0.5, 0.5, 0.5],
            sigma_t_per_cell=geometry["sigma_t_map"],
            sigma_a_per_cell=geometry["sigma_a_map"],
            energy_threshold=1.0e-8,
            seed=seed,
            reflected_boundaries=False,
            randomize_position_in_cell=False,
            scattering_mode=scattering_mode,
            g_hg=g_hg,
            verbose=True,
        )
        energy_deposited = result[0]
        track_length = result[1]
        stats = result[2] if len(result) > 2 else {}
    
    t_elapsed = time.perf_counter() - t0
    
    # Calculate flux from track length
    cell_volumes = (
        geometry["cell_widths"][:, None, None]
        * geometry["cell_heights"][None, :, None]
        * geometry["cell_depths"][None, None, :]
    )
    cell_volumes = np.transpose(cell_volumes, (2, 1, 0))  # z, y, x order
    
    flux = np.divide(
        track_length,
        cell_volumes,
        out=np.zeros_like(track_length),
        where=cell_volumes > 0.0
    )
    
    print(f"\n3D MC completed in {t_elapsed:.2f}s")
    print(f"  Total energy deposited: {np.sum(energy_deposited):.6e}")
    print(f"  Total track length: {np.sum(track_length):.6e}")
    print(f"  Mean flux: {np.mean(flux):.6e}")
    print(f"  Max flux: {np.max(flux):.6e}")
    
    return {
        "energy_deposited": energy_deposited,
        "track_length": track_length,
        "flux": flux,
        "stats": stats,
        "time": t_elapsed,
    }


def compute_error_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    """Compute error metrics between reference and estimate."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    
    diff = estimate - reference
    
    return {
        "l1": float(np.mean(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff * diff))),
        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1.0e-30)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def plot_3d_slice(data_3d, geometry, z_slice_index, output_path, title, log_scale=True):
    """Plot z-slice of 3D data."""
    X, Y = np.meshgrid(geometry["cell_boundaries_x"], geometry["cell_boundaries_y"])
    
    data_slice = data_3d[z_slice_index, :, :]
    
    plt.figure(figsize=(8, 7))
    if log_scale:
        positive = data_slice[data_slice > 0.0]
        if positive.size > 0:
            vmin = float(np.min(positive))
            vmax = float(np.max(data_slice))
        else:
            vmin, vmax = 1.0e-12, 1.0
        mesh = plt.pcolormesh(X, Y, np.maximum(data_slice, vmin), 
                             shading="auto", cmap="jet", norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        mesh = plt.pcolormesh(X, Y, data_slice, shading="auto", cmap="jet")
    
    plt.colorbar(mesh, label="Flux")
    plt.gca().set_aspect("equal")
    plt.title(f"{title}\n(z-slice {z_slice_index}, z={geometry['cell_boundaries_z'][z_slice_index]:.2f})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create 3D geometry
    print("Creating 3D lattice geometry...")
    geometry = create_lattice_3d_geometry(
        grid_size=args.grid_size,
        z_cells=args.z_cells,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    
    print(f"\n3D Lattice Geometry:")
    print(f"  Grid: {geometry['num_cells_x']}×{geometry['num_cells_y']}×{geometry['num_cells_z']} cells")
    print(f"  Domain: {geometry['domain_size_x']:.1f}×{geometry['domain_size_y']:.1f}×{geometry['domain_size_z']:.1f}")
    print(f"  Cell size: {args.cell_width:.2f}×{args.cell_width:.2f}×{args.cell_height:.2f}")
    print(f"  Total cells: {geometry['num_cells_x'] * geometry['num_cells_y'] * geometry['num_cells_z']}")
    print(f"  Fuel region: center 3×3×{args.z_cells} cells")
    
    # Run MC reference
    mc_result = run_3d_mc_reference(
        geometry,
        num_particles=args.num_particles,
        source_type=args.source_type,
        scattering_mode=args.scattering_mode,
        g_hg=args.g_hg,
        seed=args.seed,
    )
    
    # TODO: Add GMC comparison when 3D GMC loader is available
    # For now, this script validates the 3D MC infrastructure
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n[MC Statistics]")
    print(f"  Source type:     {args.source_type}")
    print(f"  Scattering mode: {args.scattering_mode}")
    if args.scattering_mode == "anisotropic-hg":
        print(f"  HG parameter g:  {args.g_hg:.3f}")
    print(f"  Particles:       {args.num_particles:,}")
    print(f"  Time:            {mc_result['time']:.2f}s")
    
    print(f"\n[Flux Statistics]")
    print(f"  Mean flux:  {np.mean(mc_result['flux']):.6e}")
    print(f"  Max flux:   {np.max(mc_result['flux']):.6e}")
    print(f"  Min flux:   {np.min(mc_result['flux']):.6e}")
    print(f"  Std flux:   {np.std(mc_result['flux']):.6e}")
    
    # Analyze flux distribution by region
    center_start = (args.grid_size - 3) // 2
    center_end = center_start + 3
    fuel_mask = np.zeros_like(mc_result['flux'], dtype=bool)
    fuel_mask[:, center_start:center_end, center_start:center_end] = True
    
    fuel_flux = mc_result['flux'][fuel_mask]
    mod_flux = mc_result['flux'][~fuel_mask]
    
    print(f"\n[Flux by Region]")
    print(f"  Fuel region:")
    print(f"    Mean: {np.mean(fuel_flux):.6e}")
    print(f"    Max:  {np.max(fuel_flux):.6e}")
    print(f"  Moderator region:")
    print(f"    Mean: {np.mean(mod_flux):.6e}")
    print(f"    Max:  {np.max(mod_flux):.6e}")
    print(f"  Fuel/Moderator ratio: {np.mean(fuel_flux) / np.mean(mod_flux):.3f}")
    
    # Save results
    np.savez_compressed(
        outdir / "mc_3d_results.npz",
        scattering_mode=np.asarray(args.scattering_mode),
        g_hg=np.asarray(args.g_hg),
        source_type=np.asarray(args.source_type),
        energy_deposited=mc_result["energy_deposited"],
        track_length=mc_result["track_length"],
        flux=mc_result["flux"],
        geometry_num_cells_x=np.asarray(geometry["num_cells_x"]),
        geometry_num_cells_y=np.asarray(geometry["num_cells_y"]),
        geometry_num_cells_z=np.asarray(geometry["num_cells_z"]),
    )
    
    # Generate plots for middle z-slice
    mid_z = geometry["num_cells_z"] // 2
    plot_3d_slice(
        mc_result["flux"],
        geometry,
        mid_z,
        outdir / f"flux_mc_z{mid_z}.png",
        f"3D MC Flux ({args.scattering_mode}, g={args.g_hg:.2f})",
        log_scale=True,
    )
    
    # Plot all z-slices
    for iz in range(geometry["num_cells_z"]):
        plot_3d_slice(
            mc_result["flux"],
            geometry,
            iz,
            outdir / f"flux_mc_z{iz}.png",
            f"3D MC Flux ({args.scattering_mode}, g={args.g_hg:.2f})",
            log_scale=True,
        )
    
    print(f"\nResults saved to {outdir}")


if __name__ == "__main__":
    main()
