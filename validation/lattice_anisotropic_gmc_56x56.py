#!/usr/bin/env python3
"""Steady-2D GMC-vs-MC validation on the 56x56 lattice with explicit scattering modes."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
    run_coarse_gmc_with_surface_tally,
    run_direct_fine_mc_reference,
)
from physics.steady_2d_scattering import (
    STEADY_2D_SCATTERING_ANISOTROPIC_HG,
    normalize_steady_2d_scattering_mode,
    resolve_steady_2d_hg_parameter,
)


LATTICE_REFINEMENT = 8
FACE_LABELS = ("left", "right", "top", "bottom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anisotropic steady 2D GMC-vs-MC proof of concept on the 56x56 lattice"
    )
    parser.add_argument("--gmc-particles", type=int, default=50_000, help="Particles for the direct 56x56 GMC solve")
    parser.add_argument("--mc-particles", type=int, default=50_000, help="Particles for the direct 56x56 MC reference")
    parser.add_argument("--num-angles", type=int, default=16, help="Angular bins for GMC face diagnostics")
    parser.add_argument("--sigma-s-scale", type=float, default=1.0, help="Multiply sigma_s by this factor for both GMC and MC")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--train-sampling", type=str, default="random", help="Training sampling tag for the anisotropic checkpoints")
    parser.add_argument(
        "--scattering-mode",
        type=str,
        default="anisotropic-linear",
        choices=["isotropic", "anisotropic", "anisotropic-linear", "anisotropic-hg"],
        help="Steady-2D scattering mode for the base GMC and MC reference.",
    )
    parser.add_argument(
        "--g-hg",
        type=float,
        default=None,
        help="Required Henyey-Greenstein asymmetry parameter when --scattering-mode=anisotropic-hg.",
    )
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation/results/steady_2d_scattering/lattice_56x56_gmc_vs_mc",
    )
    return parser.parse_args()


def _format_hg_tag(g_hg: float) -> str:
    sign = "p" if g_hg >= 0.0 else "m"
    magnitude = f"{abs(float(g_hg)):.6f}".replace(".", "_")
    return f"g_{sign}{magnitude}"


def _resolve_scattering_config(args: argparse.Namespace) -> tuple[str, float, str, str]:
    scattering_mode = normalize_steady_2d_scattering_mode(args.scattering_mode)
    if scattering_mode == STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        if args.g_hg is None:
            raise ValueError("--g-hg is required when --scattering-mode=anisotropic-hg.")
        g_hg = resolve_steady_2d_hg_parameter(scattering_mode, args.g_hg)
        mode_label = f"{scattering_mode} (g={g_hg:.3f})"
        mode_key = f"{scattering_mode.replace('-', '_')}_{_format_hg_tag(g_hg)}"
        return scattering_mode, g_hg, mode_label, mode_key
    if args.g_hg is not None:
        raise ValueError("--g-hg may only be used with --scattering-mode=anisotropic-hg.")
    mode_label = scattering_mode
    mode_key = scattering_mode.replace("-", "_")
    return scattering_mode, 0.0, mode_label, mode_key


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
        mesh = plt.pcolormesh(
            X,
            Y,
            np.maximum(data, vmin),
            shading="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
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


def main() -> None:
    args = parse_args()
    scattering_mode, g_hg, mode_label, mode_key = _resolve_scattering_config(args)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    hierarchy = build_multigrid_hierarchy(
        coarse_refinement=LATTICE_REFINEMENT,
        fine_refinement=LATTICE_REFINEMENT,
    )
    if args.sigma_s_scale <= 0.0:
        raise ValueError(f"sigma_s_scale must be positive, got {args.sigma_s_scale}")
    problem = hierarchy.coarse
    if args.sigma_s_scale != 1.0:
        sigma_s_map = problem.sigma_s_map * float(args.sigma_s_scale)
        problem = replace(problem, sigma_s_map=sigma_s_map, sigma_t_map=problem.sigma_a_map + sigma_s_map)
        hierarchy = replace(hierarchy, coarse=problem, fine=problem)
    quadrature = build_uniform_angular_quadrature(args.num_angles)
    models = load_steady_gmc_models(
        args.device,
        scattering_mode=scattering_mode,
        g_hg=g_hg,
        train_sampling=args.train_sampling,
    )

    print("56x56 steady-2D GMC validation")
    print(f"  Grid:        {problem.num_cells_x}x{problem.num_cells_y}")
    print(f"  GMC:         {args.gmc_particles} particles")
    print(f"  MC:          {args.mc_particles} particles")
    print(f"  Angles:      {args.num_angles}")
    print(f"  Scattering:  {mode_label}")
    print(f"  sigma_s x:   {args.sigma_s_scale}")
    print(f"  Training:    {args.train_sampling}")
    print("  Refinement:  no S_n path in this driver")

    print("\n[1/2] Running direct GMC...")
    gmc_result = run_coarse_gmc_with_surface_tally(
        problem,
        hierarchy,
        models,
        quadrature,
        num_particles=args.gmc_particles,
        seed=args.seed,
    )

    print("[2/2] Running direct MC reference...")
    mc_result = run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=args.mc_particles,
        seed=args.seed + 1000,
        scattering_mode=scattering_mode,
        g_hg=g_hg,
    )

    gmc_flux = gmc_result["coarse_flux"]
    mc_flux = mc_result["flux"]
    surface_current = np.sum(gmc_result["surface_tally"].crossing_weight, axis=(2, 3, 4))
    metrics = compute_flux_error_metrics(mc_flux, gmc_flux)
    gmc_exit_counts = np.asarray([gmc_result["totals"][face] for face in FACE_LABELS], dtype=np.int64)
    mc_exit_counts = np.asarray(mc_result["stats"][:4], dtype=np.int64)

    print("\n[Flux metrics: GMC vs MC]")
    for key, value in metrics.items():
        print(f"  {key:<12s}: {value:.6e}")
    print(f"\n[Surface tally] current sum: {np.sum(gmc_result['surface_tally'].crossing_weight):.6e}")
    print("[Leakage counts]")
    for idx, label in enumerate(FACE_LABELS):
        print(f"  {label:<6s}: GMC={gmc_exit_counts[idx]}, MC={mc_exit_counts[idx]}")

    result_stem = f"lattice_56x56_gmc_vs_mc_{mode_key}"
    np.savez_compressed(
        outdir / f"{result_stem}.npz",
        scattering_mode=np.asarray(scattering_mode),
        scattering_label=np.asarray(mode_label),
        g_hg=np.asarray(g_hg),
        sn_compatible=np.asarray(False),
        refinement=np.asarray(LATTICE_REFINEMENT, dtype=np.int64),
        sigma_s_scale=np.asarray(args.sigma_s_scale),
        train_sampling=np.asarray(args.train_sampling),
        gmc_flux=gmc_flux,
        mc_flux=mc_flux,
        surface_current=surface_current,
        crossing_weight=gmc_result["surface_tally"].crossing_weight,
        angular_flux=gmc_result["surface_tally"].angular_flux,
        gmc_exit_counts=gmc_exit_counts,
        mc_exit_counts=mc_exit_counts,
        flux_l1=np.asarray(metrics["l1"]),
        flux_l2=np.asarray(metrics["l2"]),
        flux_rel_l2=np.asarray(metrics["rel_l2"]),
        flux_max_abs=np.asarray(metrics["max_abs"]),
    )

    _plot_map(
        problem.cell_boundaries_x,
        problem.cell_boundaries_y,
        gmc_flux,
        outdir / f"{result_stem}_gmc_flux.png",
        title=f"GMC Flux (56x56, {mode_label})",
    )
    _plot_map(
        hierarchy.fine.cell_boundaries_x,
        hierarchy.fine.cell_boundaries_y,
        mc_flux,
        outdir / f"{result_stem}_mc_flux.png",
        title=f"MC Flux (56x56, {mode_label})",
    )
    _plot_map(
        problem.cell_boundaries_x,
        problem.cell_boundaries_y,
        np.abs(mc_flux - gmc_flux),
        outdir / f"{result_stem}_abs_error.png",
        title=f"|GMC - MC| (56x56, {mode_label})",
        cmap="magma",
    )
    _plot_map(
        problem.cell_boundaries_x,
        problem.cell_boundaries_y,
        surface_current,
        outdir / f"{result_stem}_surface_current.png",
        title=f"GMC Surface Current (56x56, {mode_label})",
    )

    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
