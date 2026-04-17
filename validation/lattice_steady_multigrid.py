#!/usr/bin/env python3
"""Validation driver for the prototype 2D steady-state lattice multigrid workflow.

The geometry and source pattern mirror `validation/lattice_steady.py`, but this
driver stays lightweight and focuses on the new coarse-to-fine refinement path:

1. Run coarse-grid transport with incoming surface-current binning.
2. Refine each coarse cell independently on its embedded fine subgrid with a
   local SN source-iteration solve.
3. Compare the refined fine-grid scalar flux against a direct fine-grid Monte
   Carlo reference built with `physics.mc2d.run_mc_simulation_2d_njit_wrapper`.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "validation" / "results" / ".matplotlib"))

from physics.mc2d import calculate_boundaries_2d, run_mc_simulation_2d_njit_wrapper
from physics.lattice_multigrid_2d import (
    DiscreteOrdinates2D,
    build_hierarchy_from_fine_maps,
    coarsen_fine_scalar_flux,
    run_multigrid_refinement,
)


DOMAIN_SIZE = 7.0
BASE_CELLS = 7
MAT_SCATTERING = 0
MAT_ABSORBING = 1
MAT_SOURCE = 2
ENERGY_THRESHOLD = 1.0e-8
SOURCE_POSITION_RELATIVE = (0.5, 0.5)

MATERIAL_PROPERTIES = {
    MAT_SCATTERING: {"sigma_a": 0.0, "sigma_s": 1.0},
    MAT_ABSORBING: {"sigma_a": 9.5, "sigma_s": 0.5},
    MAT_SOURCE: {"sigma_a": 0.0, "sigma_s": 1.0},
}

LATTICE_PATTERN = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 2, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)


def build_lattice_problem(mesh_refinement: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fine-grid material and cross-section arrays for the 7x7 lattice."""

    num_cells = BASE_CELLS * mesh_refinement
    material_map = np.zeros((num_cells, num_cells), dtype=np.int64)
    sigma_t = np.zeros_like(material_map, dtype=np.float64)
    sigma_s = np.zeros_like(material_map, dtype=np.float64)
    sigma_a = np.zeros_like(material_map, dtype=np.float64)

    base_cell_size = DOMAIN_SIZE / float(BASE_CELLS)

    def coord_to_idx(coord: float) -> int:
        return int(coord * num_cells / DOMAIN_SIZE)

    for base_y in range(BASE_CELLS):
        for base_x in range(BASE_CELLS):
            material_id = int(LATTICE_PATTERN[base_y, base_x])
            y0 = coord_to_idx(base_y * base_cell_size)
            y1 = coord_to_idx((base_y + 1) * base_cell_size)
            x0 = coord_to_idx(base_x * base_cell_size)
            x1 = coord_to_idx((base_x + 1) * base_cell_size)
            material_map[y0:y1, x0:x1] = material_id

    for material_id, props in MATERIAL_PROPERTIES.items():
        mask = material_map == material_id
        sigma_a[mask] = props["sigma_a"]
        sigma_s[mask] = props["sigma_s"]
        sigma_t[mask] = props["sigma_a"] + props["sigma_s"]

    source_mask = material_map == MAT_SOURCE
    return material_map, sigma_t, sigma_s, sigma_a, source_mask


def run_internal_source_reference_mc(
    *,
    sigma_t: np.ndarray,
    sigma_a: np.ndarray,
    source_mask: np.ndarray,
    num_particles: int,
    seed: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run a fine-grid direct MC reference with uniform source-cell sampling."""

    start_time = time.perf_counter()
    num_cells_y, num_cells_x = sigma_t.shape
    cell_widths = np.full(num_cells_x, DOMAIN_SIZE / float(num_cells_x), dtype=np.float64)
    cell_heights = np.full(num_cells_y, DOMAIN_SIZE / float(num_cells_y), dtype=np.float64)
    boundaries_x, boundaries_y = calculate_boundaries_2d(cell_widths, cell_heights)
    source_cells = np.argwhere(source_mask)
    if source_cells.size == 0:
        raise ValueError("Source mask is empty.")

    energy_accum = np.zeros_like(sigma_t, dtype=np.float64)
    track_accum = np.zeros_like(sigma_t, dtype=np.float64)
    rng = np.random.default_rng(seed)
    base_count = num_particles // source_cells.shape[0]
    remainder = num_particles % source_cells.shape[0]

    for idx, (cell_y, cell_x) in enumerate(source_cells):
        particle_count = base_count + (1 if idx < remainder else 0)
        if particle_count == 0:
            continue
        energy, track_length, _stats = run_mc_simulation_2d_njit_wrapper(
            source_mode="internal_start",
            num_cells_x=num_cells_x,
            num_cells_y=num_cells_y,
            cell_widths=cell_widths,
            cell_heights=cell_heights,
            cell_boundaries_x=boundaries_x,
            cell_boundaries_y=boundaries_y,
            total_width=DOMAIN_SIZE,
            total_height=DOMAIN_SIZE,
            num_particles=particle_count,
            start_cell_indices=[int(cell_x), int(cell_y)],
            start_position_relative=list(SOURCE_POSITION_RELATIVE),
            scatter_ratio=None,
            energy_threshold=ENERGY_THRESHOLD,
            seed=int(rng.integers(1, 2**31 - 1)),
            reflected_boundaries=False,
            angular_sampling=1,
            sigma_t_per_cell=sigma_t,
            sigma_a_per_cell=sigma_a,
            randomize_position_in_cell=True,
            verbose=verbose,
        )
        particle_weight = float(particle_count) / float(num_particles)
        energy_accum += particle_weight * energy
        track_accum += particle_weight * track_length

    cell_areas = np.outer(cell_heights, cell_widths)
    flux = np.divide(track_accum, cell_areas, out=np.zeros_like(track_accum), where=cell_areas > 0.0)
    return flux, track_accum, energy_accum, time.perf_counter() - start_time


def area_weighted_relative_l1(estimate: np.ndarray, reference: np.ndarray, areas: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(estimate - reference) * areas))
    denominator = float(np.sum(np.abs(reference) * areas))
    return numerator / max(denominator, 1.0e-12)


def area_weighted_relative_l2(estimate: np.ndarray, reference: np.ndarray, areas: np.ndarray) -> float:
    numerator = float(np.sum(((estimate - reference) ** 2) * areas))
    denominator = float(np.sum((reference ** 2) * areas))
    return math.sqrt(numerator / max(denominator, 1.0e-12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-refinement", type=int, default=2, help="Coarse refinement per base cell. Default: 2 (14x14 coarse mesh).")
    parser.add_argument("--fine-refinement", type=int, default=8, help="Fine refinement per base cell. Default: 8 (56x56 fine mesh).")
    parser.add_argument("--coarse-histories", type=int, default=4000, help="Number of coarse transport histories.")
    parser.add_argument("--reference-histories", type=int, default=12000, help="Number of direct fine-grid MC histories.")
    parser.add_argument("--num-angles", type=int, default=16, help="Number of global discrete ordinates.")
    parser.add_argument("--sn-iterations", type=int, default=18, help="Maximum local source-iteration sweeps.")
    parser.add_argument("--sn-tolerance", type=float, default=1.0e-5, help="Relative convergence tolerance for local SN.")
    parser.add_argument("--max-coarse-events", type=int, default=4000, help="Maximum events tracked per coarse history.")
    parser.add_argument("--seed", type=int, default=420, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz file for saved diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.coarse_refinement <= 0 or args.fine_refinement <= 0:
        raise ValueError("Refinement factors must be positive.")
    if args.fine_refinement % args.coarse_refinement != 0:
        raise ValueError("fine_refinement must be an integer multiple of coarse_refinement.")

    coarse_num_cells = BASE_CELLS * args.coarse_refinement
    fine_num_cells = BASE_CELLS * args.fine_refinement
    fine_material_map, fine_sigma_t, fine_sigma_s, fine_sigma_a, fine_source_mask = build_lattice_problem(args.fine_refinement)

    hierarchy = build_hierarchy_from_fine_maps(
        total_width=DOMAIN_SIZE,
        total_height=DOMAIN_SIZE,
        fine_sigma_t=fine_sigma_t,
        fine_sigma_s=fine_sigma_s,
        fine_sigma_a=fine_sigma_a,
        fine_material_map=fine_material_map,
        fine_source_mask=fine_source_mask,
        coarse_shape=(coarse_num_cells, coarse_num_cells),
    )
    ordinates = DiscreteOrdinates2D.uniform(args.num_angles)

    multigrid_result = run_multigrid_refinement(
        hierarchy,
        num_histories=args.coarse_histories,
        ordinates=ordinates,
        seed=args.seed,
        energy_threshold=ENERGY_THRESHOLD,
        max_coarse_events=args.max_coarse_events,
        sn_max_iterations=args.sn_iterations,
        sn_tolerance=args.sn_tolerance,
        homogeneous_local_blocks=True,
    )
    multigrid_coarse_flux, multigrid_coarse_track = coarsen_fine_scalar_flux(hierarchy, multigrid_result.fine_flux)

    reference_flux, reference_track, reference_energy, reference_elapsed = run_internal_source_reference_mc(
        sigma_t=fine_sigma_t,
        sigma_a=fine_sigma_a,
        source_mask=fine_source_mask,
        num_particles=args.reference_histories,
        seed=args.seed + 17,
    )
    reference_coarse_flux, reference_coarse_track = coarsen_fine_scalar_flux(hierarchy, reference_flux)

    fine_areas = hierarchy.fine_mesh.areas
    coarse_areas = hierarchy.coarse_mesh.areas
    total_multigrid_track = float(np.sum(multigrid_result.fine_track_length))
    total_reference_track = float(np.sum(reference_track))
    total_coarse_absorption = float(np.sum(multigrid_result.coarse_result.energy_deposition))
    total_coarse_leakage = float(np.sum(multigrid_result.coarse_result.leakage_weight))
    coarse_balance_residual = abs(1.0 - total_coarse_absorption - total_coarse_leakage)
    coarse_anchor_max_abs = float(np.max(np.abs(multigrid_coarse_track - multigrid_result.coarse_result.track_length)))

    rel_l1 = area_weighted_relative_l1(multigrid_result.fine_flux, reference_flux, fine_areas)
    rel_l2 = area_weighted_relative_l2(multigrid_result.fine_flux, reference_flux, fine_areas)
    coarse_rel_l1 = area_weighted_relative_l1(multigrid_coarse_flux, reference_coarse_flux, coarse_areas)
    active_surface_bins = multigrid_result.coarse_result.surface_tally.active_bin_count()
    local_iters = multigrid_result.local_iterations[multigrid_result.local_iterations > 0]
    mean_local_iterations = float(local_iters.mean()) if local_iters.size else 0.0
    max_local_iterations = int(local_iters.max()) if local_iters.size else 0
    source_coarse_cells = int(np.count_nonzero(hierarchy.coarse_source_strength > 0.0))

    print(f"Hierarchy: coarse={coarse_num_cells}x{coarse_num_cells}, fine={fine_num_cells}x{fine_num_cells}, local={hierarchy.fine_per_coarse_y}x{hierarchy.fine_per_coarse_x}")
    print(f"Local-material simplification: homogeneous local SN blocks = yes, source-support coarse cells = {source_coarse_cells}")
    print(f"Coarse transport: histories={args.coarse_histories}, mean_events={multigrid_result.coarse_result.mean_events:.1f}, active_surface_bins={active_surface_bins}, elapsed={multigrid_result.coarse_result.elapsed_sec:.2f}s")
    print(f"Coarse balance: absorbed={total_coarse_absorption:.6f}, leaked={total_coarse_leakage:.6f}, residual={coarse_balance_residual:.3e}")
    print(f"Local SN: mean_iterations={mean_local_iterations:.2f}, max_iterations={max_local_iterations}, coarse_track_anchor_max_abs={coarse_anchor_max_abs:.3e}")
    print(f"Reference MC: histories={args.reference_histories}, elapsed={reference_elapsed:.2f}s")
    print(f"Fine-flux comparison: rel_L1={rel_l1:.4f}, rel_L2={rel_l2:.4f}, total_track_ratio={total_multigrid_track / max(total_reference_track, 1.0e-12):.4f}")
    print(f"Coarse-averaged comparison: rel_L1={coarse_rel_l1:.4f}, coarse_track_ratio={float(np.sum(multigrid_coarse_track)) / max(float(np.sum(reference_coarse_track)), 1.0e-12):.4f}")

    output_path = args.output
    if output_path is None:
        output_dir = REPO_ROOT / "validation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"lattice_steady_multigrid_{coarse_num_cells}x{coarse_num_cells}_to_{fine_num_cells}x{fine_num_cells}.npz"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        multigrid_flux=multigrid_result.fine_flux,
        multigrid_track=multigrid_result.fine_track_length,
        multigrid_coarse_track=multigrid_result.coarse_result.track_length,
        multigrid_coarse_flux=multigrid_result.coarse_result.flux,
        reference_flux=reference_flux,
        reference_track=reference_track,
        reference_energy=reference_energy,
        reference_coarse_track=reference_coarse_track,
        reference_coarse_flux=reference_coarse_flux,
        surface_current=multigrid_result.coarse_result.surface_tally.current,
        surface_counts=multigrid_result.coarse_result.surface_tally.counts,
        local_iterations=multigrid_result.local_iterations,
        local_scale_factors=multigrid_result.local_scale_factors,
        fine_sigma_t=fine_sigma_t,
        fine_sigma_s=fine_sigma_s,
        fine_sigma_a=fine_sigma_a,
        fine_material_map=fine_material_map,
        fine_source_mask=fine_source_mask.astype(np.int8),
        coarse_sigma_t=hierarchy.coarse_sigma_t,
        coarse_sigma_s=hierarchy.coarse_sigma_s,
        coarse_sigma_a=hierarchy.coarse_sigma_a,
        coarse_material_map=hierarchy.coarse_material_map,
        coarse_refinement=args.coarse_refinement,
        fine_refinement=args.fine_refinement,
        coarse_histories=args.coarse_histories,
        reference_histories=args.reference_histories,
        num_angles=args.num_angles,
        coarse_elapsed=multigrid_result.coarse_result.elapsed_sec,
        multigrid_elapsed=multigrid_result.elapsed_sec,
        reference_elapsed=reference_elapsed,
        rel_l1=rel_l1,
        rel_l2=rel_l2,
        coarse_rel_l1=coarse_rel_l1,
        total_multigrid_track=total_multigrid_track,
        total_reference_track=total_reference_track,
    )
    print(f"Saved diagnostics to {output_path}")


if __name__ == "__main__":
    main()
