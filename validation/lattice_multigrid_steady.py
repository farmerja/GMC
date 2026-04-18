#!/usr/bin/env python3
"""Prototype 2D steady-state multigrid refinement on the lattice benchmark."""

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
    parser = argparse.ArgumentParser(description="2D steady-state lattice multigrid prototype")
    parser.add_argument("--coarse-refinement", type=int, default=2, help="Refinement per base lattice cell for the coarse GMC grid")
    parser.add_argument("--fine-refinement", type=int, default=8, help="Refinement per base lattice cell for the fine reconstruction/reference grid")
    parser.add_argument("--coarse-particles", type=int, default=4000, help="Number of source particles for the coarse GMC pass")
    parser.add_argument("--fine-mc-particles", type=int, default=20000, help="Number of source particles for the direct fine-grid MC reference")
    parser.add_argument("--num-angles", type=int, default=16, help="Number of uniform angular bins/ordinates")
    parser.add_argument("--sn-iters", type=int, default=6, help="Source-iteration count for each local SN solve")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device used for the GMC/CFM pass")
    parser.add_argument("--scattering-mode", type=str, default="isotropic", choices=["isotropic", "anisotropic"])
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--output-dir", type=str, default="validation/results/lattice_multigrid_steady")
    return parser.parse_args()


def _plot_map(
    bx: np.ndarray,
    by: np.ndarray,
    data: np.ndarray,
    output_path: Path,
    *,
    title: str,
    log_scale: bool = True,
    cmap: str = "jet",
) -> None:
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


def _plot_error_map(
    bx: np.ndarray,
    by: np.ndarray,
    reference: np.ndarray,
    estimate: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
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
    models = load_steady_gmc_models(args.device, scattering_mode=args.scattering_mode)
    anisotropic_mode = args.scattering_mode != "isotropic"

    print(
        f"Running coarse GMC on {hierarchy.coarse.num_cells_x}x{hierarchy.coarse.num_cells_y} "
        f"and refining to {hierarchy.fine.num_cells_x}x{hierarchy.fine.num_cells_y}"
    )
    print(f"Scattering mode: {args.scattering_mode}")
    if anisotropic_mode:
        print(
            "WARNING: skipping local SN reconstruction for anisotropic mode because the current "
            "steady 2D SN stage assumes isotropic scatter_source and the surface tally stores "
            "projected angle only, not projected radius r."
        )

    coarse = run_coarse_gmc_with_surface_tally(
        hierarchy.coarse,
        hierarchy,
        models,
        quadrature,
        num_particles=args.coarse_particles,
        seed=args.seed,
    )
    fine_mc = run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=args.fine_mc_particles,
        seed=args.seed + 1000,
        scattering_mode=args.scattering_mode,
    )

    fine_mc_flux = fine_mc["flux"]
    if anisotropic_mode:
        fine_sn_flux = np.full_like(fine_mc_flux, np.nan)
        metrics = None
        balance = np.full_like(coarse["coarse_flux"], np.nan)
        refined = None
    else:
        refined = reconstruct_fine_flux_from_surface_tally(
            hierarchy,
            coarse["surface_tally"],
            quadrature,
            num_source_iterations=args.sn_iters,
        )
        fine_sn_flux = refined["fine_flux"]
        metrics = compute_flux_error_metrics(fine_mc_flux, fine_sn_flux)
        balance = refined["local_balance"]

    print("\n[Surface tally]")
    print(f"  crossing weight sum   : {np.sum(coarse['surface_tally'].crossing_weight):.6e}")
    print(f"  angular flux max      : {np.max(coarse['surface_tally'].angular_flux):.6e}")

    if metrics is not None:
        print("\n[Flux metrics]")
        for key, value in metrics.items():
            print(f"  {key:<12s}: {value:.6e}")

        print("\n[Local balance residuals]")
        print(f"  max abs residual      : {np.max(np.abs(balance)):.6e}")
        print(f"  mean abs residual     : {np.mean(np.abs(balance)):.6e}")

    coarse_flux_upsampled = np.repeat(
        np.repeat(coarse["coarse_flux"], hierarchy.fine_per_coarse_y, axis=0),
        hierarchy.fine_per_coarse_x,
        axis=1,
    )

    np.savez_compressed(
        outdir / "lattice_multigrid_steady_results.npz",
        scattering_mode=np.asarray(args.scattering_mode),
        sn_compatible=np.asarray(not anisotropic_mode),
        coarse_flux=coarse["coarse_flux"],
        fine_flux_sn=fine_sn_flux,
        fine_flux_mc=fine_mc_flux,
        coarse_flux_upsampled=coarse_flux_upsampled,
        local_balance=balance,
        incoming_current=np.zeros_like(balance) if refined is None else refined["incoming_current"],
        outgoing_current=np.zeros_like(balance) if refined is None else refined["outgoing_current"],
        crossing_weight=coarse["surface_tally"].crossing_weight,
        angular_flux=coarse["surface_tally"].angular_flux,
        flux_l1=np.nan if metrics is None else metrics["l1"],
        flux_l2=np.nan if metrics is None else metrics["l2"],
        flux_rel_l2=np.nan if metrics is None else metrics["rel_l2"],
        flux_max_abs=np.nan if metrics is None else metrics["max_abs"],
    )

    _plot_map(
        hierarchy.coarse.cell_boundaries_x,
        hierarchy.coarse.cell_boundaries_y,
        coarse["coarse_flux"],
        outdir / "coarse_gmc_flux.png",
        title="Coarse GMC Scalar Flux",
    )
    _plot_map(
        hierarchy.fine.cell_boundaries_x,
        hierarchy.fine.cell_boundaries_y,
        coarse_flux_upsampled,
        outdir / "coarse_gmc_flux_upsampled.png",
        title="Coarse GMC Flux Upsampled To Fine Grid",
    )
    _plot_map(
        hierarchy.fine.cell_boundaries_x,
        hierarchy.fine.cell_boundaries_y,
        fine_mc_flux,
        outdir / "fine_mc_flux.png",
        title="Fine Direct MC Reference Flux",
    )
    if refined is not None:
        _plot_map(
            hierarchy.fine.cell_boundaries_x,
            hierarchy.fine.cell_boundaries_y,
            fine_sn_flux,
            outdir / "fine_sn_flux.png",
            title="Fine Flux From Local SN Reconstruction",
        )
        _plot_error_map(
            hierarchy.fine.cell_boundaries_x,
            hierarchy.fine.cell_boundaries_y,
            fine_mc_flux,
            fine_sn_flux,
            outdir / "fine_sn_vs_mc_abs_error.png",
            title="|Fine SN - Fine MC|",
        )
        _plot_map(
            hierarchy.coarse.cell_boundaries_x,
            hierarchy.coarse.cell_boundaries_y,
            np.abs(balance),
            outdir / "local_balance_abs.png",
            title="Per-Coarse-Cell |Balance Residual|",
            log_scale=True,
            cmap="magma",
        )
    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
