#!/usr/bin/env python3
"""
Split-only hierarchical model vs. MC — scalar-flux comparison on the 7x7 lattice.

This is a standalone validation script for the split-only monolithic scattered
model in ``ml/hierarchical_steady_2d/split_monolithic.py``. It runs the lattice
transport loop with the split sampler and compares the scalar-flux map against a
direct fine-grid MC reference.

Usage
-----
  python validation/split_lattice_comparison.py
  python validation/split_lattice_comparison.py --refinement 2 --n-model 50000 --n-mc 200000
  python validation/split_lattice_comparison.py --outdir validation/results/split_lattice
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.hierarchical_steady_2d.split_monolithic import SplitMonolithicSteady2DSampler
from physics.hierarchical_steady_2d import FACE_BOTTOM, FACE_LEFT, FACE_RIGHT, FACE_TOP
from physics.multigrid_steady_2d import (
    build_rect_lattice_problem,
    compute_flux_error_metrics,
    run_direct_fine_mc_reference,
)


OPPOSITE_FACE = np.array([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=np.int64)


@dataclass(frozen=True)
class SplitModelBundle:
    device: str
    model_mode: str
    boundary_sampler: object
    internal_sampler: object


def _load_split_models(
    *,
    device: str,
    boundary_run: str,
    internal_run: str,
    run_root: str,
) -> SplitModelBundle | object:
    """Load the split-only hierarchical model bundle.

    Prefer the central runtime loader when it already supports
    ``model_mode='hierarchical_split'``. Fall back to direct sampler loading so
    this validator remains usable while the broader runtime wiring is in flight.
    """

    from physics.multigrid_steady_2d import load_steady_gmc_models

    try:
        return load_steady_gmc_models(
            device=device,
            scattering_mode="isotropic",
            model_mode="hierarchical_split",
            split_run_root=run_root,
            split_boundary_run=boundary_run,
            split_internal_run=internal_run,
        )
    except TypeError:
        pass
    except ValueError as exc:
        if "hierarchical_split" not in str(exc):
            raise

    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"
    return SplitModelBundle(
        device=resolved_device,
        model_mode="hierarchical_split",
        boundary_sampler=SplitMonolithicSteady2DSampler.from_run_dir(
            Path(run_root) / boundary_run,
            source_type="boundary",
            device=resolved_device,
        ),
        internal_sampler=SplitMonolithicSteady2DSampler.from_run_dir(
            Path(run_root) / internal_run,
            source_type="internal",
            device=resolved_device,
        ),
    )


def _sample_isotropic_uv_numpy(num: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=int(num))
    return np.cos(phi), np.sin(phi)


def _update_weighted_diagnostics(
    accum: dict[str, float],
    counts: dict[str, int],
    diagnostics: dict[str, float],
    weight: int,
) -> None:
    for key, value in diagnostics.items():
        accum[key] = accum.get(key, 0.0) + float(value) * float(weight)
        counts[key] = counts.get(key, 0) + int(weight)


def _finalize_weighted_diagnostics(accum: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: float(accum[key] / max(1, counts[key]))
        for key in sorted(accum)
    }


def _run_split_lattice_transport(
    problem,
    models,
    *,
    num_particles: int,
    randomize_start_pos: bool,
    seed: int,
    max_iterations: int,
    energy_threshold: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    part = np.zeros((num_particles, 8), dtype=np.float64)
    part[:, 6] = 1.0
    src_cells = problem.source_cells
    src_sel = rng.integers(0, src_cells.shape[0], size=num_particles)
    part[:, 0] = src_cells[src_sel, 1].astype(np.float64)
    part[:, 1] = src_cells[src_sel, 0].astype(np.float64)

    num_cells = problem.num_cells_x * problem.num_cells_y
    track_flat = np.zeros(num_cells, dtype=np.float64)
    energy_flat = np.zeros(num_cells, dtype=np.float64)
    totals = {"left": 0, "right": 0, "top": 0, "bottom": 0, "absorb": 0}
    diag_accum: dict[str, float] = {}
    diag_counts: dict[str, int] = {}

    def advance_active_batch(
        active_idx: np.ndarray,
        sampler: object,
        *,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
    ) -> None:
        if active_idx.size == 0:
            return

        ix = part[active_idx, 0].astype(np.int64)
        iy = part[active_idx, 1].astype(np.int64)
        entry_weight = part[active_idx, 6].astype(np.float64)
        width = problem.cell_widths[ix].astype(np.float64)
        height = problem.cell_heights[iy].astype(np.float64)
        sigma_a = np.maximum(problem.sigma_a_map[iy, ix].astype(np.float64), 0.0)
        sigma_t = np.maximum(problem.sigma_t_map[iy, ix].astype(np.float64), 1.0e-6)

        sampled = sampler.sample_batch(
            width=width,
            height=height,
            initial_x=initial_x,
            initial_y=initial_y,
            initial_u=initial_u,
            initial_v=initial_v,
            sigma_t=sigma_t,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        _update_weighted_diagnostics(diag_accum, diag_counts, sampled.diagnostics, active_idx.size)

        track_length = np.asarray(sampled.track_length, dtype=np.float64)
        exit_face = np.asarray(sampled.exit_face, dtype=np.int64)
        pos_on_face = np.asarray(sampled.pos_on_face, dtype=np.float64)
        u_out = np.asarray(sampled.u_out, dtype=np.float64)
        v_out = np.asarray(sampled.v_out, dtype=np.float64)

        exit_weight = entry_weight * np.exp(-sigma_a * track_length)
        deposit = entry_weight - exit_weight
        lin_idx = iy * problem.num_cells_x + ix
        np.add.at(track_flat, lin_idx, entry_weight * track_length)
        np.add.at(energy_flat, lin_idx, deposit)

        absorbed = exit_weight < energy_threshold
        if np.any(absorbed):
            totals["absorb"] += int(np.count_nonzero(absorbed))
            part[active_idx[absorbed], 7] = 2.0

        survivors = ~absorbed
        if not np.any(survivors):
            return

        surv_idx = active_idx[survivors]
        exit_face = exit_face[survivors]
        pos_on_face = pos_on_face[survivors]
        u_out = u_out[survivors]
        v_out = v_out[survivors]
        ix_s = ix[survivors]
        iy_s = iy[survivors]
        new_ix = ix_s + (exit_face == FACE_RIGHT).astype(np.int64) - (exit_face == FACE_LEFT).astype(np.int64)
        new_iy = iy_s + (exit_face == FACE_TOP).astype(np.int64) - (exit_face == FACE_BOTTOM).astype(np.int64)

        exit_mask = (
            (new_ix < 0)
            | (new_ix >= problem.num_cells_x)
            | (new_iy < 0)
            | (new_iy >= problem.num_cells_y)
        )
        totals["left"] += int(np.count_nonzero((exit_face == FACE_LEFT) & exit_mask))
        totals["right"] += int(np.count_nonzero((exit_face == FACE_RIGHT) & exit_mask))
        totals["top"] += int(np.count_nonzero((exit_face == FACE_TOP) & exit_mask))
        totals["bottom"] += int(np.count_nonzero((exit_face == FACE_BOTTOM) & exit_mask))
        part[surv_idx[exit_mask], 7] = 1.0

        interior = ~exit_mask
        if not np.any(interior):
            return

        cont_idx = surv_idx[interior]
        part[cont_idx, 0] = new_ix[interior]
        part[cont_idx, 1] = new_iy[interior]
        part[cont_idx, 2] = OPPOSITE_FACE[exit_face[interior]].astype(np.float64)
        part[cont_idx, 3] = pos_on_face[interior]
        part[cont_idx, 4] = u_out[interior]
        part[cont_idx, 5] = v_out[interior]
        part[cont_idx, 6] = exit_weight[survivors][interior]

    active = np.flatnonzero(part[:, 7] == 0.0)
    if active.size > 0:
        ix = part[active, 0].astype(np.int64)
        iy = part[active, 1].astype(np.int64)
        width = problem.cell_widths[ix].astype(np.float64)
        height = problem.cell_heights[iy].astype(np.float64)
        if randomize_start_pos:
            initial_x = rng.random(active.size) * width
            initial_y = rng.random(active.size) * height
        else:
            initial_x = 0.5 * width
            initial_y = 0.5 * height
        initial_u, initial_v = _sample_isotropic_uv_numpy(active.size, rng)
        advance_active_batch(
            active,
            models.internal_sampler,
            initial_x=initial_x,
            initial_y=initial_y,
            initial_u=initial_u,
            initial_v=initial_v,
        )

    for _ in range(max_iterations):
        active = np.flatnonzero(part[:, 7] == 0.0)
        if active.size == 0:
            break

        ix = part[active, 0].astype(np.int64)
        iy = part[active, 1].astype(np.int64)
        face = part[active, 2].astype(np.int64)
        pos = part[active, 3].astype(np.float64)
        width = problem.cell_widths[ix].astype(np.float64)
        height = problem.cell_heights[iy].astype(np.float64)
        initial_x = np.where(face == FACE_LEFT, 0.0, np.where(face == FACE_RIGHT, width, pos))
        initial_y = np.where(
            face == FACE_LEFT,
            pos,
            np.where(face == FACE_RIGHT, pos, np.where(face == FACE_TOP, height, 0.0)),
        )
        advance_active_batch(
            active,
            models.boundary_sampler,
            initial_x=initial_x,
            initial_y=initial_y,
            initial_u=part[active, 4].astype(np.float64),
            initial_v=part[active, 5].astype(np.float64),
        )

    flux_grid = np.divide(
        track_flat.reshape(problem.num_cells_y, problem.num_cells_x),
        problem.cell_areas,
        out=np.zeros((problem.num_cells_y, problem.num_cells_x), dtype=np.float64),
        where=problem.cell_areas > 0.0,
    ) / float(num_particles)
    return {
        "coarse_track": track_flat.reshape(problem.num_cells_y, problem.num_cells_x) / float(num_particles),
        "coarse_energy": energy_flat.reshape(problem.num_cells_y, problem.num_cells_x) / float(num_particles),
        "coarse_flux": flux_grid,
        "diagnostics": _finalize_weighted_diagnostics(diag_accum, diag_counts),
        "totals": totals,
    }


def _plot_flux_comparison(
    mc_flux: np.ndarray,
    model_flux: np.ndarray,
    *,
    refinement: int,
    outpath: Path,
) -> None:
    vmax = float(np.percentile(mc_flux[mc_flux > 0], 98)) if np.any(mc_flux > 0) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Scalar flux: split-only monolithic model vs. MC  (refinement={refinement})",
        fontsize=12,
    )

    im0 = axes[0].imshow(mc_flux, origin="lower", vmin=0, vmax=vmax, cmap="hot")
    axes[0].set_title("MC reference")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(model_flux, origin="lower", vmin=0, vmax=vmax, cmap="hot")
    axes[1].set_title("Split-only model")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    ref_safe = np.where(mc_flux > 0, mc_flux, np.nan)
    rel_err = np.abs(model_flux - mc_flux) / ref_safe
    im2 = axes[2].imshow(rel_err, origin="lower", vmin=0, vmax=0.5, cmap="RdYlGn_r")
    axes[2].set_title("|model - MC| / MC")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


def _plot_scatter(mc_flux: np.ndarray, model_flux: np.ndarray, *, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    mask = mc_flux.ravel() > 0
    x = mc_flux.ravel()[mask]
    y = model_flux.ravel()[mask]
    ax.scatter(x, y, s=6, alpha=0.6, edgecolors="none")
    lim = max(float(x.max()), float(y.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("MC flux")
    ax.set_ylabel("Model flux")
    ax.set_title("Cell-by-cell flux: split-only model vs. MC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split-only hierarchical model vs. MC scalar-flux comparison"
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=2,
        help="Grid refinement factor (default 2 -> 14x14 grid)",
    )
    parser.add_argument(
        "--n-model",
        type=int,
        default=50_000,
        help="Number of particles for the split-only model run (default 50000)",
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=200_000,
        help="Number of particles for the MC reference (default 200000)",
    )
    parser.add_argument("--seed-model", type=int, default=42)
    parser.add_argument("--seed-mc", type=int, default=123)
    parser.add_argument("--device", default="auto", help="Torch device: 'cpu', 'cuda', or 'auto' (default)")
    parser.add_argument(
        "--boundary-run",
        default="split_boundary_compact",
        help="Name of the split-only boundary model run directory",
    )
    parser.add_argument(
        "--internal-run",
        default="split_internal_compact",
        help="Name of the split-only internal model run directory",
    )
    parser.add_argument(
        "--run-root",
        default="ml/hierarchical_steady_2d/split_runs",
        help="Root directory containing split-only run subdirectories",
    )
    parser.add_argument(
        "--outdir",
        default="validation/results/split_lattice",
        help="Output directory for plots and metrics JSON",
    )
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--energy-threshold", type=float, default=1.0e-8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Building lattice problem (refinement={args.refinement}) ...")
    problem = build_rect_lattice_problem(args.refinement)
    print(
        f"  Grid: {problem.num_cells_x}x{problem.num_cells_y}  "
        f"source cells: {problem.source_cells.shape[0]}"
    )

    print(f"\nRunning MC reference ({args.n_mc:,} particles) ...")
    t0 = time.perf_counter()
    mc_result = run_direct_fine_mc_reference(
        problem,
        num_particles=args.n_mc,
        randomize_start_pos=True,
        seed=args.seed_mc,
    )
    t_mc = time.perf_counter() - t0
    mc_flux = mc_result["flux"]
    print(f"  Done in {t_mc:.1f}s   flux max={mc_flux.max():.4f}  mean={mc_flux[mc_flux > 0].mean():.4f}")

    print("\nLoading split-only hierarchical model ...")
    models = _load_split_models(
        device=args.device,
        boundary_run=args.boundary_run,
        internal_run=args.internal_run,
        run_root=args.run_root,
    )
    print(f"  Loaded. Running lattice transport ({args.n_model:,} particles) ...")
    t0 = time.perf_counter()
    model_result = _run_split_lattice_transport(
        problem,
        models,
        num_particles=args.n_model,
        randomize_start_pos=True,
        seed=args.seed_model,
        max_iterations=args.max_iterations,
        energy_threshold=args.energy_threshold,
    )
    t_model = time.perf_counter() - t0
    model_flux = model_result["coarse_flux"]
    print(
        f"  Done in {t_model:.1f}s   flux max={model_flux.max():.4f}  "
        f"mean={model_flux[model_flux > 0].mean():.4f}"
    )
    print(f"  Particle balance: {model_result['totals']}")
    if model_result.get("diagnostics"):
        for key, value in sorted(model_result["diagnostics"].items()):
            print(f"    {key}: {value:.4f}")

    metrics = compute_flux_error_metrics(mc_flux, model_flux)
    print("\n-- Scalar-flux error metrics --")
    print(f"  L1  (mean abs)    : {metrics['l1']:.4e}")
    print(f"  L2  (RMS)         : {metrics['l2']:.4e}")
    print(f"  Rel-L2            : {metrics['rel_l2']:.4f}")
    print(f"  Max abs error     : {metrics['max_abs']:.4e}")

    metrics_out = {
        "model_mode": "hierarchical_split",
        "refinement": args.refinement,
        "n_model": args.n_model,
        "n_mc": args.n_mc,
        "time_model_s": round(t_model, 2),
        "time_mc_s": round(t_mc, 2),
        "boundary_run": args.boundary_run,
        "internal_run": args.internal_run,
        **metrics,
    }
    metrics_path = outdir / f"metrics_r{args.refinement}_m{args.n_model}_mc{args.n_mc}.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"\nMetrics saved: {metrics_path}")

    tag = f"r{args.refinement}_m{args.n_model}_mc{args.n_mc}"
    _plot_flux_comparison(
        mc_flux,
        model_flux,
        refinement=args.refinement,
        outpath=outdir / f"flux_comparison_{tag}.png",
    )
    _plot_scatter(
        mc_flux,
        model_flux,
        outpath=outdir / f"flux_scatter_{tag}.png",
    )


if __name__ == "__main__":
    main()
