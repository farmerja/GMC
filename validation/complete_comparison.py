#!/usr/bin/env python3
"""Complete comparison: GMC, MC, and S_n refinement methods."""

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
    reconstruct_fine_flux_with_block_sn,
    run_coarse_gmc_with_surface_tally,
    run_coarse_mc_with_surface_tally,
    run_direct_fine_mc_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete method comparison including MC+S_n")
    parser.add_argument("--coarse-refinement", type=int, default=2)
    parser.add_argument("--fine-refinement", type=int, default=8)
    parser.add_argument("--coarse-particles", type=int, default=50000)
    parser.add_argument("--fine-gmc-particles", type=int, default=50000)
    parser.add_argument("--fine-mc-particles", type=int, default=50000)
    parser.add_argument("--num-angles", type=int, default=16)
    parser.add_argument("--sn-iters", type=int, default=6)
    parser.add_argument("--block-sn-iters", type=int, default=18)
    parser.add_argument("--block-sn-tol", type=float, default=1.0e-5)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--output-dir", type=str, default="validation/results/complete_comparison")
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


def _masked_metrics(reference, estimate, mask=None):
    reference_arr = np.asarray(reference, dtype=np.float64)
    estimate_arr = np.asarray(estimate, dtype=np.float64)
    if mask is None:
        return compute_flux_error_metrics(reference_arr, estimate_arr)
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return {"l1": 0.0, "l2": 0.0, "rel_l2": 0.0, "max_abs": 0.0}
    return compute_flux_error_metrics(reference_arr[active], estimate_arr[active])


def _surface_current_map(surface_tally):
    return np.sum(surface_tally.crossing_weight, axis=(2, 3, 4))


def _surface_current_by_face(surface_tally):
    return np.sum(surface_tally.crossing_weight, axis=(3, 4))


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

    print(f"Complete Method Comparison:")
    print(f"  Coarse grid: {hierarchy.coarse.num_cells_x}x{hierarchy.coarse.num_cells_y}")
    print(f"  Fine grid:   {hierarchy.fine.num_cells_x}x{hierarchy.fine.num_cells_y}")
    print(f"  Coarse histories: {args.coarse_particles}")
    print(f"  Fine histories:   GMC={args.fine_gmc_particles}, MC={args.fine_mc_particles}")
    print()

    print("[1/7] Coarse GMC with fine-face surface tally...")
    coarse_gmc = run_coarse_gmc_with_surface_tally(
        hierarchy.coarse,
        hierarchy,
        models,
        quadrature,
        num_particles=args.coarse_particles,
        seed=args.seed,
    )

    print("[2/7] Coarse MC with true fine-face surface tally...")
    coarse_mc = run_coarse_mc_with_surface_tally(
        hierarchy.coarse,
        hierarchy,
        quadrature,
        num_particles=args.coarse_particles,
        seed=args.seed + 10_000,
    )

    print("[3/7] Old local S_n on GMC and MC surface data...")
    refined_gmc_old = reconstruct_fine_flux_from_surface_tally(
        hierarchy,
        coarse_gmc["surface_tally"],
        quadrature,
        num_source_iterations=args.sn_iters,
    )
    refined_mc_old = reconstruct_fine_flux_from_surface_tally(
        hierarchy,
        coarse_mc["surface_tally"],
        quadrature,
        num_source_iterations=args.sn_iters,
    )

    print("[4/7] Block-S_n A/B on the same GMC and MC surface data...")
    refined_gmc_block = reconstruct_fine_flux_with_block_sn(
        hierarchy,
        coarse_gmc["surface_tally"],
        quadrature,
        coarse_gmc["coarse_track"],
        num_source_iterations=args.block_sn_iters,
        tolerance=args.block_sn_tol,
    )
    refined_mc_block = reconstruct_fine_flux_with_block_sn(
        hierarchy,
        coarse_mc["surface_tally"],
        quadrature,
        coarse_mc["coarse_track"],
        num_source_iterations=args.block_sn_iters,
        tolerance=args.block_sn_tol,
    )

    print("[5/7] Fine GMC direct...")
    fine_gmc = run_coarse_gmc_with_surface_tally(
        hierarchy.fine,
        hierarchy,
        models,
        quadrature,
        num_particles=args.fine_gmc_particles,
        seed=args.seed + 500,
    )

    print("[6/7] Fine MC reference...")
    fine_mc_result = run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=args.fine_mc_particles,
        seed=args.seed + 1000,
    )
    print("[7/7] Aggregating surface and reconstruction diagnostics...")

    # Extract fluxes
    coarse_gmc_flux = coarse_gmc["coarse_flux"]
    coarse_mc_flux = coarse_mc["coarse_flux"]
    fine_sn_from_gmc = refined_gmc_old["fine_flux"]
    fine_sn_from_mc = refined_mc_old["fine_flux"]
    fine_block_from_gmc = refined_gmc_block["fine_flux"]
    fine_block_from_mc = refined_mc_block["fine_flux"]
    fine_gmc_flux = fine_gmc["coarse_flux"]
    fine_mc_flux = fine_mc_result["flux"]
    gmc_surface = coarse_gmc["surface_tally"]
    mc_surface = coarse_mc["surface_tally"]
    gmc_surface_current = _surface_current_map(gmc_surface)
    mc_surface_current = _surface_current_map(mc_surface)
    gmc_surface_by_face = _surface_current_by_face(gmc_surface)
    mc_surface_by_face = _surface_current_by_face(mc_surface)
    active_surface_mask = (gmc_surface.crossing_weight > 0.0) | (mc_surface.crossing_weight > 0.0)

    # Compute all pairwise metrics
    print("\n" + "="*70)
    print("COMPLETE RESULTS COMPARISON")
    print("="*70)

    metrics = {}
    metrics["surface_gmc_vs_mc_all_bins"] = compute_flux_error_metrics(
        mc_surface.angular_flux.reshape(-1),
        gmc_surface.angular_flux.reshape(-1),
    )
    metrics["surface_gmc_vs_mc_active_bins"] = _masked_metrics(
        mc_surface.angular_flux,
        gmc_surface.angular_flux,
        active_surface_mask,
    )
    metrics["surface_current_gmc_vs_mc"] = compute_flux_error_metrics(mc_surface_current, gmc_surface_current)
    metrics["surface_current_facewise_gmc_vs_mc"] = compute_flux_error_metrics(
        mc_surface_by_face.reshape(-1),
        gmc_surface_by_face.reshape(-1),
    )
    metrics["sn_from_gmc_vs_fine_mc"] = compute_flux_error_metrics(fine_mc_flux, fine_sn_from_gmc)
    metrics["sn_from_mc_vs_fine_mc"] = compute_flux_error_metrics(fine_mc_flux, fine_sn_from_mc)
    metrics["block_sn_from_gmc_vs_fine_mc"] = compute_flux_error_metrics(fine_mc_flux, fine_block_from_gmc)
    metrics["block_sn_from_mc_vs_fine_mc"] = compute_flux_error_metrics(fine_mc_flux, fine_block_from_mc)
    metrics["fine_gmc_vs_fine_mc"] = compute_flux_error_metrics(fine_mc_flux, fine_gmc_flux)
    metrics["sn_from_gmc_vs_sn_from_mc"] = compute_flux_error_metrics(fine_sn_from_mc, fine_sn_from_gmc)
    metrics["block_sn_from_gmc_vs_block_sn_from_mc"] = compute_flux_error_metrics(fine_block_from_mc, fine_block_from_gmc)
    metrics["old_vs_block_from_gmc"] = compute_flux_error_metrics(fine_block_from_gmc, fine_sn_from_gmc)
    metrics["old_vs_block_from_mc"] = compute_flux_error_metrics(fine_block_from_mc, fine_sn_from_mc)
    metrics["coarse_gmc_vs_coarse_mc"] = compute_flux_error_metrics(coarse_mc_flux, coarse_gmc_flux)

    print("\n[1. Coarse-surface diagnostics: GMC vs MC]")
    for name in (
        "surface_gmc_vs_mc_all_bins",
        "surface_gmc_vs_mc_active_bins",
        "surface_current_gmc_vs_mc",
        "surface_current_facewise_gmc_vs_mc",
    ):
        print(f"  {name}:")
        for key, value in metrics[name].items():
            print(f"    {key:<12s}: {value:.6e}")
    print(f"  active surface bins      : {int(np.count_nonzero(active_surface_mask))}")
    print(f"  GMC incoming current sum : {np.sum(gmc_surface.crossing_weight):.6e}")
    print(f"  MC incoming current sum  : {np.sum(mc_surface.crossing_weight):.6e}")

    print("\n[2. Old S_n from GMC vs Fine MC Reference]")
    for key, value in metrics["sn_from_gmc_vs_fine_mc"].items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[3. Old S_n from MC vs Fine MC Reference]")
    for key, value in metrics["sn_from_mc_vs_fine_mc"].items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[4. Block S_n from GMC and MC vs Fine MC Reference]")
    print("  From GMC surface:")
    for key, value in metrics["block_sn_from_gmc_vs_fine_mc"].items():
        print(f"    {key:<12s}: {value:.6e}")
    print("  From MC surface:")
    for key, value in metrics["block_sn_from_mc_vs_fine_mc"].items():
        print(f"    {key:<12s}: {value:.6e}")

    print("\n[5. Fine GMC vs Fine MC Reference]")
    for key, value in metrics["fine_gmc_vs_fine_mc"].items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[6. Same-SN comparisons between GMC and MC surfaces]")
    print("  Old S_n:")
    for key, value in metrics["sn_from_gmc_vs_sn_from_mc"].items():
        print(f"    {key:<12s}: {value:.6e}")
    print("  Block S_n:")
    for key, value in metrics["block_sn_from_gmc_vs_block_sn_from_mc"].items():
        print(f"    {key:<12s}: {value:.6e}")

    print("\n[7. Same-surface A/B: old S_n vs block S_n]")
    print("  From GMC surface:")
    for key, value in metrics["old_vs_block_from_gmc"].items():
        print(f"    {key:<12s}: {value:.6e}")
    print("  From MC surface:")
    for key, value in metrics["old_vs_block_from_mc"].items():
        print(f"    {key:<12s}: {value:.6e}")

    print("\n[8. Coarse scalar flux: GMC vs MC]")
    for key, value in metrics["coarse_gmc_vs_coarse_mc"].items():
        print(f"  {key:<12s}: {value:.6e}")

    print("\n[Flux Statistics]")
    print(f"  Coarse GMC:       mean={np.mean(coarse_gmc_flux):.6e}, max={np.max(coarse_gmc_flux):.6e}")
    print(f"  Coarse MC:        mean={np.mean(coarse_mc_flux):.6e}, max={np.max(coarse_mc_flux):.6e}")
    print(f"  Old S_n GMC:      mean={np.mean(fine_sn_from_gmc):.6e}, max={np.max(fine_sn_from_gmc):.6e}")
    print(f"  Old S_n MC:       mean={np.mean(fine_sn_from_mc):.6e}, max={np.max(fine_sn_from_mc):.6e}")
    print(f"  Block S_n GMC:    mean={np.mean(fine_block_from_gmc):.6e}, max={np.max(fine_block_from_gmc):.6e}")
    print(f"  Block S_n MC:     mean={np.mean(fine_block_from_mc):.6e}, max={np.max(fine_block_from_mc):.6e}")
    print(f"  Fine GMC:         mean={np.mean(fine_gmc_flux):.6e}, max={np.max(fine_gmc_flux):.6e}")
    print(f"  Fine MC:          mean={np.mean(fine_mc_flux):.6e}, max={np.max(fine_mc_flux):.6e}")

    # Save results
    np.savez_compressed(
        outdir / "complete_comparison.npz",
        coarse_gmc_track=coarse_gmc["coarse_track"],
        coarse_mc_track=coarse_mc["coarse_track"],
        coarse_gmc_flux=coarse_gmc_flux,
        coarse_mc_flux=coarse_mc_flux,
        gmc_surface_current=gmc_surface_current,
        mc_surface_current=mc_surface_current,
        gmc_surface_current_by_face=gmc_surface_by_face,
        mc_surface_current_by_face=mc_surface_by_face,
        gmc_surface_angular_flux=gmc_surface.angular_flux,
        mc_surface_angular_flux=mc_surface.angular_flux,
        fine_sn_old_from_gmc=fine_sn_from_gmc,
        fine_sn_old_from_mc=fine_sn_from_mc,
        fine_sn_block_from_gmc=fine_block_from_gmc,
        fine_sn_block_from_mc=fine_block_from_mc,
        fine_gmc_flux=fine_gmc_flux,
        fine_mc_flux=fine_mc_flux,
        metrics=metrics,
    )

    # Generate plots
    bx_fine = hierarchy.fine.cell_boundaries_x
    by_fine = hierarchy.fine.cell_boundaries_y
    bx_coarse = hierarchy.coarse.cell_boundaries_x
    by_coarse = hierarchy.coarse.cell_boundaries_y

    _plot_map(bx_coarse, by_coarse, coarse_gmc_flux, outdir / "coarse_gmc.png", title="Coarse GMC")
    _plot_map(bx_coarse, by_coarse, coarse_mc_flux, outdir / "coarse_mc.png", title="Coarse MC")
    _plot_map(bx_coarse, by_coarse, gmc_surface_current, outdir / "surface_current_gmc.png", title="Incoming Surface Current: GMC", log_scale=True, cmap="magma")
    _plot_map(bx_coarse, by_coarse, mc_surface_current, outdir / "surface_current_mc.png", title="Incoming Surface Current: MC", log_scale=True, cmap="magma")
    _plot_error_map(bx_coarse, by_coarse, mc_surface_current, gmc_surface_current, outdir / "surface_current_error_gmc_vs_mc.png", title="Surface Current Error: GMC vs MC")
    _plot_map(bx_fine, by_fine, fine_sn_from_gmc, outdir / "fine_sn_old_from_gmc.png", title="Old S_n from GMC")
    _plot_map(bx_fine, by_fine, fine_sn_from_mc, outdir / "fine_sn_old_from_mc.png", title="Old S_n from MC")
    _plot_map(bx_fine, by_fine, fine_block_from_gmc, outdir / "fine_sn_block_from_gmc.png", title="Block S_n from GMC")
    _plot_map(bx_fine, by_fine, fine_block_from_mc, outdir / "fine_sn_block_from_mc.png", title="Block S_n from MC")
    _plot_map(bx_fine, by_fine, fine_gmc_flux, outdir / "fine_gmc.png", title="Fine GMC")
    _plot_map(bx_fine, by_fine, fine_mc_flux, outdir / "fine_mc.png", title="Fine MC Reference")

    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_sn_from_gmc, outdir / "error_old_sn_gmc_vs_mc.png", title="Error: Old S_n(GMC) vs MC")
    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_sn_from_mc, outdir / "error_old_sn_mc_vs_mc.png", title="Error: Old S_n(MC) vs MC")
    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_block_from_gmc, outdir / "error_block_sn_gmc_vs_mc.png", title="Error: Block S_n(GMC) vs MC")
    _plot_error_map(bx_fine, by_fine, fine_mc_flux, fine_block_from_mc, outdir / "error_block_sn_mc_vs_mc.png", title="Error: Block S_n(MC) vs MC")
    _plot_error_map(bx_fine, by_fine, fine_block_from_gmc, fine_sn_from_gmc, outdir / "error_old_vs_block_from_gmc.png", title="Error: Old vs Block S_n from GMC")
    _plot_error_map(bx_fine, by_fine, fine_block_from_mc, fine_sn_from_mc, outdir / "error_old_vs_block_from_mc.png", title="Error: Old vs Block S_n from MC")

    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
