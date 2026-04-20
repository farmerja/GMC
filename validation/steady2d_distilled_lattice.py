#!/usr/bin/env python3
"""Direct steady-2D lattice comparison for teacher and distilled GMC models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "validation" / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "validation" / "results" / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from ml.distillation.artifacts import save_json
from physics.multigrid_steady_2d import (
    build_rect_lattice_problem,
    build_uniform_angular_quadrature,
    compute_flux_error_metrics,
    load_steady_gmc_models,
    make_single_level_hierarchy,
    run_coarse_gmc_with_surface_tally,
    run_direct_fine_mc_reference,
)


DEFAULT_MODEL_MODES = ("teacher", "student_4", "student_2", "student_1")
FACE_LABELS = ("left", "right", "top", "bottom")


def _normalize_model_modes(values) -> list[str]:
    normalized = [str(value).strip().lower().replace("-", "_") for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate model modes are not allowed: {normalized}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct steady-2D teacher/student lattice comparison")
    parser.add_argument("--refinement", type=int, default=8, help="Lattice refinement; 8 corresponds to the 56x56 case")
    parser.add_argument("--gmc-particles", type=int, default=20_000, help="Particles for each GMC run")
    parser.add_argument("--mc-particles", type=int, default=20_000, help="Particles for the MC reference")
    parser.add_argument("--num-angles", type=int, default=16, help="Angular bins for GMC face diagnostics")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--train-sampling", type=str, default="random", help="Training sampling tag for teacher/student assets")
    parser.add_argument("--scattering-mode", type=str, default="isotropic", choices=["isotropic", "anisotropic"])
    parser.add_argument(
        "--model-modes",
        nargs="+",
        default=list(DEFAULT_MODEL_MODES),
        help="Model variants to run, e.g. teacher student_4 student_2 student_1",
    )
    parser.add_argument("--student-artifact-root", type=str, default="ml/student_artifacts")
    parser.add_argument("--hierarchical-run-root", type=str, default="ml/hierarchical_steady_2d/runs")
    parser.add_argument("--hierarchical-boundary-run", type=str, default="hier_boundary_lhs_stochastic")
    parser.add_argument("--hierarchical-internal-run", type=str, default="hier_internal_lhs_stochastic")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation/results/steady2d_distilled_lattice",
    )
    args = parser.parse_args()
    args.model_modes = _normalize_model_modes(args.model_modes)
    return args


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


def _mode_key(mode: str) -> str:
    return str(mode).strip().lower().replace("-", "_")


def _totals_payload(totals: dict, *, num_particles: int) -> dict:
    leakage = {face: int(totals[face]) for face in FACE_LABELS}
    absorbed = int(totals["absorb"])
    accounted = absorbed + sum(leakage.values())
    return {
        "counts": {**leakage, "absorb": absorbed},
        "fractions": {**{face: float(leakage[face]) / float(num_particles) for face in FACE_LABELS}, "absorb": float(absorbed) / float(num_particles)},
        "accounting_residual": int(num_particles - accounted),
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    problem = build_rect_lattice_problem(refinement=args.refinement)
    hierarchy = make_single_level_hierarchy(problem)
    quadrature = build_uniform_angular_quadrature(args.num_angles)

    print("Steady-2D distilled lattice comparison")
    print(f"  Grid:         {problem.num_cells_x}x{problem.num_cells_y}")
    print(f"  GMC:          {args.gmc_particles} particles / mode")
    print(f"  MC:           {args.mc_particles} particles")
    print(f"  Modes:        {', '.join(args.model_modes)}")
    print(f"  Scattering:   {args.scattering_mode}")
    print(f"  Train tag:    {args.train_sampling}")

    print("\n[1/2] Running MC reference...")
    mc_started = time.perf_counter()
    mc_result = run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=args.mc_particles,
        seed=args.seed + 1000,
        scattering_mode=args.scattering_mode,
    )
    mc_runtime = time.perf_counter() - mc_started
    mc_flux = mc_result["flux"]
    mc_exit_counts = {face: int(mc_result["stats"][idx]) for idx, face in enumerate(FACE_LABELS)}
    mc_absorb = int(mc_result["stats"][4])

    mode_results = {}
    teacher_flux = None
    print("[2/2] Running GMC modes...")
    for mode in args.model_modes:
        print(f"  - {mode}")
        load_kwargs = {
            "device": args.device,
            "scattering_mode": args.scattering_mode,
            "train_sampling": args.train_sampling,
            "model_mode": mode,
            "student_artifact_root": args.student_artifact_root,
        }
        if mode == "hierarchical":
            load_kwargs.update(
                {
                    "hierarchical_run_root": args.hierarchical_run_root,
                    "hierarchical_boundary_run": args.hierarchical_boundary_run,
                    "hierarchical_internal_run": args.hierarchical_internal_run,
                }
            )
        models = load_steady_gmc_models(**load_kwargs)
        started = time.perf_counter()
        result = run_coarse_gmc_with_surface_tally(
            problem,
            hierarchy,
            models,
            quadrature,
            num_particles=args.gmc_particles,
            seed=args.seed,
        )
        runtime_seconds = time.perf_counter() - started
        flux = result["coarse_flux"]
        metrics_vs_mc = compute_flux_error_metrics(mc_flux, flux)
        mode_results[mode] = {
            "flux": flux,
            "runtime_seconds": float(runtime_seconds),
            "metrics_vs_mc": metrics_vs_mc,
            "totals": result["totals"],
        }
        if mode == "teacher":
            teacher_flux = flux

    summary = {
        "refinement": int(args.refinement),
        "grid_shape": [int(problem.num_cells_y), int(problem.num_cells_x)],
        "scattering_mode": args.scattering_mode,
        "train_sampling": args.train_sampling,
        "gmc_particles": int(args.gmc_particles),
        "mc_particles": int(args.mc_particles),
        "mc_runtime_seconds": float(mc_runtime),
        "mc_exit_counts": {**mc_exit_counts, "absorb": mc_absorb},
        "modes": {},
    }
    npz_payload = {
        "mc_flux": mc_flux,
        "mc_runtime_seconds": np.asarray(mc_runtime),
        "mc_exit_counts": np.asarray([mc_exit_counts[face] for face in FACE_LABELS] + [mc_absorb], dtype=np.int64),
    }

    for mode, payload in mode_results.items():
        flux = payload["flux"]
        metrics_vs_teacher = None if teacher_flux is None or mode == "teacher" else compute_flux_error_metrics(teacher_flux, flux)
        totals_payload = _totals_payload(payload["totals"], num_particles=args.gmc_particles)
        speedup = None if mode == "teacher" else (
            mode_results["teacher"]["runtime_seconds"] / payload["runtime_seconds"]
            if "teacher" in mode_results and payload["runtime_seconds"] > 0.0
            else None
        )
        summary["modes"][mode] = {
            "runtime_seconds": float(payload["runtime_seconds"]),
            "speedup_vs_teacher": None if speedup is None else float(speedup),
            "metrics_vs_mc": payload["metrics_vs_mc"],
            "metrics_vs_teacher": metrics_vs_teacher,
            "totals": totals_payload,
        }
        key = _mode_key(mode)
        npz_payload[f"{key}_flux"] = flux
        npz_payload[f"{key}_runtime_seconds"] = np.asarray(payload["runtime_seconds"])
        npz_payload[f"{key}_exit_counts"] = np.asarray(
            [payload["totals"][face] for face in FACE_LABELS] + [payload["totals"]["absorb"]],
            dtype=np.int64,
        )
        _plot_map(
            problem.cell_boundaries_x,
            problem.cell_boundaries_y,
            flux,
            outdir / f"{key}_flux.png",
            title=f"{mode} Flux ({problem.num_cells_x}x{problem.num_cells_y})",
        )
        _plot_map(
            problem.cell_boundaries_x,
            problem.cell_boundaries_y,
            np.abs(mc_flux - flux),
            outdir / f"{key}_abs_error_vs_mc.png",
            title=f"|{mode} - MC|",
            cmap="magma",
        )

    _plot_map(
        hierarchy.fine.cell_boundaries_x,
        hierarchy.fine.cell_boundaries_y,
        mc_flux,
        outdir / "mc_flux.png",
        title=f"MC Flux ({problem.num_cells_x}x{problem.num_cells_y})",
    )

    np.savez_compressed(outdir / "steady2d_distilled_lattice_results.npz", **npz_payload)
    save_json(outdir / "summary.json", summary)

    print("\n[Summary]")
    for mode in args.model_modes:
        payload = summary["modes"][mode]
        metrics = payload["metrics_vs_mc"]
        speedup = payload["speedup_vs_teacher"]
        speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
        print(
            f"  {mode:<10s} runtime={payload['runtime_seconds']:.3f}s  "
            f"speedup={speedup_text:<8s} rel_l2_vs_mc={metrics['rel_l2']:.6e}"
        )

    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
