#!/usr/bin/env python3
"""Extended multigrid validation including direct fine-grid GMC comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.multigrid_steady_2d import (
    build_multigrid_hierarchy,
    build_uniform_angular_quadrature,
    compute_flux_error_metrics,
    load_steady_gmc_models,
    reconstruct_fine_flux_from_surface_tally,
    run_coarse_gmc_with_surface_tally,
    run_direct_fine_mc_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D steady multigrid with fine GMC comparison")
    parser.add_argument("--coarse-refinement", type=int, default=2)
    parser.add_argument("--fine-refinement", type=int, default=8)
    parser.add_argument("--coarse-particles", type=int, default=50000)
    parser.add_argument("--fine-gmc-particles", type=int, default=50000, help="Particles for direct fine-grid GMC")
    parser.add_argument("--fine-mc-particles", type=int, default=50000)
    parser.add_argument("--num-angles", type=int, default=16)
    parser.add_argument("--sn-iters", type=int, default=6)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--output-dir", type=str, default="validation/results/lattice_multigrid_comparison")
    return parser.parse_args()


def _plot_map(bx, by, data, output_path, *, title, log_scale=True, cmap="jet"):
    X, Y = np.meshgrid(bx, by)
    plt.figure(figsize=(7, 6))
    if log_scale:
        positive = data[data > 0.0]
        vmin = float(np.min(positive)) if positive.size else 1.0e-12
        vmax = float(np.max(data)) if np.max(data) > 0.0 else 1.0
        mesh = plt.pcolormesh(X, Y, np.maximum(data, vmin), shading="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        mesh = plt.pcolormesh(X, Y, data, shading="auto", cmap=cmap)
    plt.colorbar(mesh)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_error_map(bx, by, reference, estimate, output_path, *, title):
    err = np.abs(np.asarray(reference) - np.asarray(estimate))
    _plot_map(bx, by, err, output_path, title=title, log_scale=True, cmap="magma")


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    hierarchy = build_multigrid_hierarchy(
        coarse_refinement=args.coarse_refinement,
        fine_refinement=args.fine_refinement,
    )
    quadrature = build_uniform_angular_quadrature(args.num_angles)
    models = load_steady_gmc_models(args.device)

    print(f"Running comparison:")
    print(f"  Coarse GMC: {hierarchy.coarse.num_cells_x}x{hierarchy.coarse.num_cells_y} with {args.coarse_particles} particles")
    print(f"  Fine GMC:   {hierarchy.fine.num_cells_x}x{hierarchy.fine.num_cells_y} with {args.fine_gmc_particles} particles")
    print(f"  Fine MC:    {hierarchy.fine.num_cells_x}x{hierarchy.fine.num_cells_y} with {args.fine_mc_particles} particles")
    print(f"  S_n refine: {hierarchy.fine.num_cells_x}x{hierarchy.fine.num_cells_y} from coarse GMC")

    # 1. Coarse GMC with surface tally
    print("\n[1/4] Running coarse GMC...")
    coarse = run_coarse_gmc_with_surface_tally(
        hierarchy.coarse,
        hierarchy,
        models,
        quadrature,
        num_particles=args.coarse_particles,
        seed=args.seed,
    )

    # 2. S_n refinement from coarse
    print("[2/4] Running S_n refinement...")
    refined = reconstruct_fine_flux_from_surface_tally(
        hierarchy,
        coarse["surface_tally"],
        quadrature,
        num_source_iterations=args.sn_iters,
    )

    # 3. Direct fine GMC (NEW!)
    print("[3/4] Running direct fine-grid GMC...")
    fine_gmc = run_coarse_gmc_with_surface_tally(
        hierarchy.fine,
        hierarchy,  # Pass hierarchy but won't use tally for fine grid
        models,
        quadrature,
        num_particles=args.fine_gmc_particles,
        seed=args.seed + 500,
    )

    # 4. Fine MC reference
    print("[4/4] Running fine-grid MC reference...")
    fine_mc = run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=args.fine_mc_particles,
        seed=args.seed + 1000,
    )

    # Extract results
    coarse_flux = coarse["coarse_flux"]
    fine_sn_flux = refined["fine_flux"]
    fine_gmc_flux = fine_gmc["coarse_flux"]  # Actually fine grid
    fine_mc_flux = fine_mc["flux"]
    balance = refined["local_balance"]

    # Compute metrics
    metrics_sn_vs_mc = compute_flux_error_metrics(fine_mc_flux, fine_sn_flux)
    metrics_gmc_vs_mc = compute_flux_error_metrics(fine_mc_flux, fine_gmc_flux)
    metrics_sn_vs_gmc = compute_flux_error_metrics(fine_gmc_flux, fine_sn_flux)

    # Print results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print("\n[Surface tally (coarse GMC)]")
    print(f"  crossing weight sum   : {np.sum(coarse['surface_tally'].crossing_weight):.6e}")
    print(f"  angular flux max      : {np.max(coarse['surface_tally'].angular_flux):.6e}")

    print("\n[S_n Refinement vs MC Reference]")
    for key, value in metrics_sn_vs_mc.items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[Fine GMC vs MC Reference]")
    for key, value in metrics_gmc_vs_mc.items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[S_n Refinement vs Fine GMC]")
    for key, value in metrics_sn_vs_gmc.items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[Local balance residuals (S_n only)]")
    print(f"  max abs residual      : {np.max(np.abs(balance)):.6e}")
    print(f"  mean abs residual     : {np.mean(np.abs(balance)):.6e}")

    print("\n[Flux statistics]")
    print(f"  Coarse GMC flux:  mean={np.mean(coarse_flux):.6e}, max={np.max(coarse_flux):.6e}")
    print(f"  Fine S_n flux:    mean={np.mean(fine_sn_flux):.6e}, max={np.max(fine_sn_flux):.6e}")
    print(f"  Fine GMC flux:    mean={np.mean(fine_gmc_flux):.6e}, max={np.max(fine_gmc_flux):.6e}")
    print(f"  Fine MC flux:     mean={np.mean(fine_mc_flux):.6e}, max={np.max(fine_mc_flux):.6e}")

    # Upsample coarse for visualization
    coarse_flux_upsampled = np.repeat(
        np.repeat(coarse_flux, hierarchy.fine_per_coarse_y, axis=0),
        hierarchy.fine_per_coarse_x,
        axis=1,
    )

    # Save data
    np.savez_compressed(
        outdir / "comparison_results.npz",
        coarse_flux=coarse_flux,
        fine_flux_sn=fine_sn_flux,
        fine_flux_gmc=fine_gmc_flux,
        fine_flux_mc=fine_mc_flux,
        coarse_flux_upsampled=coarse_flux_upsampled,
        local_balance=balance,
        metrics_sn_vs_mc=metrics_sn_vs_mc,
        metrics_gmc_vs_mc=metrics_gmc_vs_mc,
        metrics_sn_vs_gmc=metrics_sn_vs_gmc,
    )

    # Generate plots
    bx_fine = hierarchy.fine.cell_boundaries_x
    by_fine = hierarchy.fine.cell_boundaries_y
    bx_coarse = hierarchy.coarse.cell_boundaries_x
    by_coarse = hierarchy.coarse.cell_boundaries_y

    _plot_map(bx_coarse, by_coarse, coarse_flux, outdir / "coarse_gmc_flux.png", title="Coarse GMC Flux (14×14)")
    _plot_map(bx_fine, by_fine, coarse_flux_upsampled, outdir / "coarse_gmc_upsampled.png", title="Coarse GMC Upsampled")
    _plot_map(bx_fine, by_fine, fine_sn_flux, outdir / "fine_sn_flux.png", title="Fine S_n Refinement")
    _plot_map(bx_fine, by_fine, fine_gmc_flux, outdir / "fine_gmc_flux.png", title="Fine GMC Direct (56×56)")
    _plot_map(bx_fine, by_fine, fine_mc_flux, outdir / "fine_mc_flux.png", title="Fine MC Reference")

    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_sn_flux, outdir / "error_sn_vs_mc.png", title="Error: S_n vs MC")
    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_gmc_flux, outdir / "error_gmc_vs_mc.png", title="Error: Fine GMC vs MC")
    _plot_error_map(bx_fine, by_fine, fine_gmc_flux, fine_sn_flux, outdir / "error_sn_vs_gmc.png", title="Error: S_n vs Fine GMC")

    _plot_map(bx_coarse, by_coarse, np.abs(balance), outdir / "local_balance_abs.png", title="Local Balance Residuals", log_scale=True, cmap="magma")

    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
