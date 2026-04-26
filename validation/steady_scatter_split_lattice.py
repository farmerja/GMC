#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from physics.steady_scatter_split_2d import (
    DEFAULT_BOUNDARY_MODEL_SPEC,
    DEFAULT_INTERNAL_MODEL_SPEC,
    Steady2DModelSpec,
    build_rect_lattice_problem,
    compute_flux_error_metrics,
    load_scatter_split_steady_models,
    run_direct_fine_mc_reference,
    run_scatter_split_lattice,
)


def _plot_flux_pair(mc_flux: np.ndarray, model_flux: np.ndarray, *, outpath: Path, refinement: int) -> None:
    import matplotlib.pyplot as plt

    vmax = float(np.percentile(mc_flux[mc_flux > 0.0], 98.0)) if np.any(mc_flux > 0.0) else 1.0
    ref_safe = np.where(mc_flux > 0.0, mc_flux, np.nan)
    rel_err = np.abs(model_flux - mc_flux) / ref_safe

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Scatter-split steady 2D lattice vs MC (refinement={refinement})", fontsize=12)

    im0 = axes[0].imshow(mc_flux, origin="lower", vmin=0.0, vmax=vmax, cmap="hot")
    axes[0].set_title("MC reference")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(model_flux, origin="lower", vmin=0.0, vmax=vmax, cmap="hot")
    axes[1].set_title("Scatter-split GMC")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(rel_err, origin="lower", vmin=0.0, vmax=0.5, cmap="RdYlGn_r")
    axes[2].set_title("|model - MC| / MC")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for axis in axes:
        axis.tick_params(labelsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_model_spec(
    *,
    source_type: str,
    model_dir: str,
    norm_base_path: str,
    fallback: Steady2DModelSpec,
) -> Steady2DModelSpec:
    return Steady2DModelSpec(
        source_type=source_type,
        model_dir=model_dir or fallback.model_dir,
        norm_base_path=norm_base_path or fallback.norm_base_path,
        input_dim=fallback.input_dim,
        cond_dim=fallback.cond_dim,
        train_sampling=fallback.train_sampling,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-multigrid steady 2D scatter-split lattice validator")
    parser.add_argument("--refinement", type=int, default=2, help="Grid refinement factor (default: 2 -> 14x14)")
    parser.add_argument("--n-model", type=int, default=50_000, help="Number of particles for the scatter-split run")
    parser.add_argument("--n-mc", type=int, default=200_000, help="Number of particles for the MC reference")
    parser.add_argument("--seed-model", type=int, default=42)
    parser.add_argument("--seed-mc", type=int, default=123)
    parser.add_argument("--device", default="auto", help="Torch device: cpu, cuda, mps, or auto")
    parser.add_argument("--batch-size", type=int, default=1024, help="Model inference batch size")
    parser.add_argument("--ode-steps", type=int, default=12, help="Number of ODE steps for model inference")
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--energy-threshold", type=float, default=1.0e-8)
    parser.add_argument("--boundary-model-dir", default=DEFAULT_BOUNDARY_MODEL_SPEC.model_dir)
    parser.add_argument("--boundary-norm-file", default=DEFAULT_BOUNDARY_MODEL_SPEC.norm_base_path)
    parser.add_argument("--internal-model-dir", default=DEFAULT_INTERNAL_MODEL_SPEC.model_dir)
    parser.add_argument("--internal-norm-file", default=DEFAULT_INTERNAL_MODEL_SPEC.norm_base_path)
    parser.add_argument("--output-dir", default="validation/results/steady_scatter_split_lattice")
    parser.add_argument("--plot", action="store_true", help="Also save a flux-comparison PNG")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Building lattice problem (refinement={args.refinement}) ...")
    problem = build_rect_lattice_problem(args.refinement)
    print(
        f"  Grid: {problem.num_cells_x}x{problem.num_cells_y}"
        f"  source cells: {problem.source_cells.shape[0]}"
    )

    boundary_spec = _build_model_spec(
        source_type="boundary",
        model_dir=args.boundary_model_dir,
        norm_base_path=args.boundary_norm_file,
        fallback=DEFAULT_BOUNDARY_MODEL_SPEC,
    )
    internal_spec = _build_model_spec(
        source_type="internal",
        model_dir=args.internal_model_dir,
        norm_base_path=args.internal_norm_file,
        fallback=DEFAULT_INTERNAL_MODEL_SPEC,
    )

    print("\nRunning MC reference ...")
    t0 = time.perf_counter()
    mc_result = run_direct_fine_mc_reference(
        problem,
        num_particles=args.n_mc,
        seed=args.seed_mc,
        energy_threshold=args.energy_threshold,
        source_mode="internal_start",
        randomize_start_pos=True,
    )
    mc_time = time.perf_counter() - t0
    mc_flux = mc_result["flux"]
    print(f"  MC done in {mc_time:.2f}s  flux max={mc_flux.max():.4f}")

    print("\nLoading scatter-split steady models ...")
    models = load_scatter_split_steady_models(
        device=args.device,
        boundary_spec=boundary_spec,
        internal_spec=internal_spec,
        batch_size=args.batch_size,
        num_ode_steps=args.ode_steps,
    )
    print(
        f"  boundary checkpoint={models.boundary.checkpoint_path}\n"
        f"  internal checkpoint={models.internal.checkpoint_path}"
    )

    print("\nRunning scatter-split GMC ...")
    t0 = time.perf_counter()
    model_result = run_scatter_split_lattice(
        problem,
        models,
        num_particles=args.n_model,
        seed=args.seed_model,
        energy_threshold=args.energy_threshold,
        max_iterations=args.max_iterations,
        source_mode="internal_start",
        randomize_start_pos=True,
    )
    model_time = time.perf_counter() - t0
    model_flux = model_result.coarse_flux
    print(f"  Model done in {model_time:.2f}s  flux max={model_flux.max():.4f}")
    print(f"  Particle balance: {model_result.totals}")

    metrics = compute_flux_error_metrics(mc_flux, model_flux)
    print("\nScalar-flux error metrics")
    print(f"  L1      : {metrics['l1']:.4e}")
    print(f"  L2      : {metrics['l2']:.4e}")
    print(f"  Rel-L2  : {metrics['rel_l2']:.4f}")
    print(f"  Max abs : {metrics['max_abs']:.4e}")

    tag = f"r{args.refinement}_m{args.n_model}_mc{args.n_mc}"
    arrays_path = outdir / f"flux_arrays_{tag}.npz"
    np.savez(
        arrays_path,
        mc_flux=mc_flux,
        model_flux=model_flux,
        model_track=model_result.coarse_track,
        model_energy=model_result.coarse_energy,
        mc_track=mc_result["track"],
        mc_energy=mc_result["energy"],
    )

    metrics_payload = {
        "model_mode": "steady_scatter_split_non_multigrid",
        "conditioning": models.conditioning,
        "refinement": int(args.refinement),
        "n_model": int(args.n_model),
        "n_mc": int(args.n_mc),
        "time_model_s": round(model_time, 3),
        "time_mc_s": round(mc_time, 3),
        "boundary_model_dir": args.boundary_model_dir,
        "boundary_norm_file": args.boundary_norm_file,
        "internal_model_dir": args.internal_model_dir,
        "internal_norm_file": args.internal_norm_file,
        "particle_balance": {key: int(value) for key, value in model_result.totals.items()},
        "diagnostics": {key: float(value) for key, value in model_result.diagnostics.items()},
        **metrics,
    }
    metrics_path = outdir / f"metrics_{tag}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved arrays:  {arrays_path}")

    if args.plot:
        plot_path = outdir / f"flux_compare_{tag}.png"
        _plot_flux_pair(mc_flux, model_flux, outpath=plot_path, refinement=args.refinement)
        print(f"Saved plot:    {plot_path}")


if __name__ == "__main__":
    main()
