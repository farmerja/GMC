#!/usr/bin/env python3
"""Single-cell visual comparison for legacy steady-2D vs hierarchical vs MC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "validation" / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / "validation" / "results" / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, qmc, wasserstein_distance

from physics._seed import seeded_numpy_execution
from physics.mc2d import simulate_single_cell_2d_boundary_source, simulate_single_cell_2d_internal_source
from physics.multigrid_steady_2d import (
    _prepare_boundary_conditions,
    _prepare_internal_conditions,
    classify_face_and_snap,
    enforce_outward_angle_from_face,
    load_steady_gmc_models,
)
from ml.hierarchical_steady_2d.runtime import HierarchicalSteady2DSampler
from ml.transform import denorm_angle_phi, denorm_log_feature, denorm_position_relative, denorm_radius_r, polar_to_uv
from ml.utils import compute_ecdf


VARIABLE_SPECS = (
    ("x_out_norm", "x_out / W", False),
    ("y_out_norm", "y_out / H", False),
    ("u", "u", False),
    ("v", "v", False),
    ("time", "track length", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["internal", "boundary"], choices=["internal", "boundary"])
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--sampling", choices=["random", "lhs", "qmc"], default="lhs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-train-sampling", default="random")
    parser.add_argument("--hierarchical-run-root", default="ml/hierarchical_steady_2d/runs")
    parser.add_argument("--hierarchical-boundary-run", default="hier_boundary_lhs_stochastic")
    parser.add_argument("--hierarchical-internal-run", default="hier_internal_lhs_stochastic")
    parser.add_argument("--output-dir", default="validation/results/steady2d_hierarchical_visual_compare")
    return parser.parse_args()


def _generate_samples(n: int, d: int, *, sampling: str, seed: int) -> np.ndarray:
    if sampling == "qmc":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        if (n & (n - 1) == 0) and n != 0:
            return sampler.random_base2(m=int(np.log2(n)))
        return sampler.random(n)
    if sampling == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, d))


def _decode_support(source: str, cond_samples: np.ndarray, *, min_width: float = 0.01, max_width: float = 10.0, min_height: float = 0.01, max_height: float = 10.0) -> dict[str, np.ndarray]:
    log_w = cond_samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width = 10.0 ** log_w
    log_h = cond_samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height = 10.0 ** log_h

    if source == "boundary":
        y_norm = cond_samples[:, 2]
        mu = 2.0 * cond_samples[:, 3] - 1.0
        phi = cond_samples[:, 4] * np.pi - 0.5 * np.pi
        tangent = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        initial_u = tangent * np.cos(phi)
        initial_v = tangent * np.sin(phi)
        initial_x = np.zeros_like(y_norm)
        initial_y = y_norm * height
    else:
        x_norm = cond_samples[:, 2]
        y_norm = cond_samples[:, 3]
        mu = 2.0 * cond_samples[:, 4] - 1.0
        phi = cond_samples[:, 5] * 2.0 * np.pi
        tangent = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        initial_u = tangent * np.cos(phi)
        initial_v = tangent * np.sin(phi)
        initial_x = x_norm * width
        initial_y = y_norm * height

    return {
        "width": width.astype(np.float64),
        "height": height.astype(np.float64),
        "initial_x": initial_x.astype(np.float64),
        "initial_y": initial_y.astype(np.float64),
        "initial_u": initial_u.astype(np.float64),
        "initial_v": initial_v.astype(np.float64),
    }


def _legacy_predictions(source: str, support: dict[str, np.ndarray], *, device: str, train_sampling: str, seed: int) -> dict[str, np.ndarray]:
    models = load_steady_gmc_models(
        device=device,
        scattering_mode="isotropic",
        train_sampling=train_sampling,
        model_mode="teacher",
    )
    width = torch.from_numpy(support["width"].astype(np.float32)).to(device)
    height = torch.from_numpy(support["height"].astype(np.float32)).to(device)
    initial_u = torch.from_numpy(support["initial_u"].astype(np.float32)).to(device)
    initial_v = torch.from_numpy(support["initial_v"].astype(np.float32)).to(device)
    sigma = torch.ones_like(width)

    with torch.no_grad():
        if source == "boundary":
            y_norm = torch.from_numpy((support["initial_y"] / np.maximum(support["height"], 1.0e-12)).astype(np.float32)).to(device)
            cond = _prepare_boundary_conditions(width, height, y_norm, initial_u, initial_v, sigma, models.boundary_norm)
            torch.manual_seed(seed)
            pred = models.boundary_sampler.sample_target(
                cond,
                noise=torch.randn(cond.shape[0], models.boundary_target_dim, device=device),
            )
            norm_params = models.boundary_norm
        else:
            x_norm = torch.from_numpy((support["initial_x"] / np.maximum(support["width"], 1.0e-12)).astype(np.float32)).to(device)
            y_norm = torch.from_numpy((support["initial_y"] / np.maximum(support["height"], 1.0e-12)).astype(np.float32)).to(device)
            cond = _prepare_internal_conditions(width, height, x_norm, y_norm, initial_u, initial_v, sigma, models.internal_norm)
            torch.manual_seed(seed)
            pred = models.internal_sampler.sample_target(
                cond,
                noise=torch.randn(cond.shape[0], models.internal_target_dim, device=device),
            )
            norm_params = models.internal_norm

        x_out = denorm_position_relative(pred[:, 0]) * width
        y_out = denorm_position_relative(pred[:, 1]) * height
        phi_out = denorm_angle_phi(pred[:, 2])
        r_out = denorm_radius_r(pred[:, 3])
        time_val = denorm_log_feature(pred[:, 4], norm_params, "time")
        u_out, v_out = polar_to_uv(phi_out, r_out)
        exit_face, x_out, y_out, _pos_on_face = classify_face_and_snap(x_out, y_out, width, height)
        u_out, v_out = enforce_outward_angle_from_face(exit_face, u_out, v_out)

    x_np = x_out.detach().cpu().numpy().astype(np.float64)
    y_np = y_out.detach().cpu().numpy().astype(np.float64)
    w_np = support["width"]
    h_np = support["height"]
    return {
        "x_out_norm": np.clip(x_np / np.maximum(w_np, 1.0e-12), 0.0, 1.0),
        "y_out_norm": np.clip(y_np / np.maximum(h_np, 1.0e-12), 0.0, 1.0),
        "u": u_out.detach().cpu().numpy().astype(np.float64),
        "v": v_out.detach().cpu().numpy().astype(np.float64),
        "time": np.maximum(time_val.detach().cpu().numpy().astype(np.float64), 0.0),
    }


def _hierarchical_predictions(source: str, support: dict[str, np.ndarray], *, run_root: str, boundary_run: str, internal_run: str, device: str, seed: int) -> dict[str, np.ndarray]:
    run_dir = Path(run_root) / (boundary_run if source == "boundary" else internal_run)
    sampler = HierarchicalSteady2DSampler.from_run_dir(run_dir, source_type=source, device=device)
    sampled = sampler.sample_batch(
        width=support["width"],
        height=support["height"],
        initial_x=support["initial_x"],
        initial_y=support["initial_y"],
        initial_u=support["initial_u"],
        initial_v=support["initial_v"],
        sigma_t=np.ones_like(support["width"]),
        seed=seed,
    )
    return {
        "x_out_norm": np.clip(sampled.x_out.astype(np.float64) / np.maximum(support["width"], 1.0e-12), 0.0, 1.0),
        "y_out_norm": np.clip(sampled.y_out.astype(np.float64) / np.maximum(support["height"], 1.0e-12), 0.0, 1.0),
        "u": sampled.u_out.astype(np.float64),
        "v": sampled.v_out.astype(np.float64),
        "time": np.maximum(sampled.track_length.astype(np.float64), 0.0),
    }


def _mc_ground_truth(source: str, cond_samples: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    with seeded_numpy_execution(seed):
        if source == "boundary":
            data = simulate_single_cell_2d_boundary_source(
                cond_samples.shape[0],
                scatter_ratio=1.0,
                distribution=0,
                cond_samples=cond_samples,
            )
            return {
                "x_out_norm": np.clip(data[:, 5] / np.maximum(data[:, 0], 1.0e-12), 0.0, 1.0),
                "y_out_norm": np.clip(data[:, 6] / np.maximum(data[:, 1], 1.0e-12), 0.0, 1.0),
                "u": data[:, 7],
                "v": data[:, 8],
                "time": np.maximum(data[:, 9], 0.0),
            }

        data = simulate_single_cell_2d_internal_source(
            cond_samples.shape[0],
            scatter_ratio=1.0,
            distribution=0,
            cond_samples=cond_samples,
        )
        return {
            "x_out_norm": np.clip(data[:, 6] / np.maximum(data[:, 0], 1.0e-12), 0.0, 1.0),
            "y_out_norm": np.clip(data[:, 7] / np.maximum(data[:, 1], 1.0e-12), 0.0, 1.0),
            "u": data[:, 8],
            "v": data[:, 9],
            "time": np.maximum(data[:, 10], 0.0),
        }


def _plot_ecdf_grid(output_path: Path, gt: dict[str, np.ndarray], legacy: dict[str, np.ndarray], hierarchical: dict[str, np.ndarray], *, source: str) -> None:
    fig, axes = plt.subplots(1, len(VARIABLE_SPECS), figsize=(4.2 * len(VARIABLE_SPECS), 3.8))
    for ax, (key, xlabel, log10) in zip(axes, VARIABLE_SPECS):
        for values, label, style in (
            (gt[key], "MC", "-"),
            (legacy[key], "Legacy", "--"),
            (hierarchical[key], "Hierarchical", "-."),
        ):
            arr = np.log10(np.maximum(values, 1.0e-12)) if log10 else values
            xs, ys = compute_ecdf(arr)
            ax.plot(xs, ys, style, label=label, linewidth=1.8)
        ax.set_xlabel(f"log10({xlabel})" if log10 else xlabel)
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Steady-2D {source} ECDF comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_hist_grid(output_path: Path, gt: dict[str, np.ndarray], legacy: dict[str, np.ndarray], hierarchical: dict[str, np.ndarray], *, source: str) -> None:
    fig, axes = plt.subplots(1, len(VARIABLE_SPECS), figsize=(4.2 * len(VARIABLE_SPECS), 3.8))
    for ax, (key, xlabel, log10) in zip(axes, VARIABLE_SPECS):
        bins_data = []
        for values in (gt[key], legacy[key], hierarchical[key]):
            arr = np.log10(np.maximum(values, 1.0e-12)) if log10 else values
            bins_data.append(arr)
        all_values = np.concatenate(bins_data, axis=0)
        bins = np.linspace(np.min(all_values), np.max(all_values), 50)
        ax.hist(bins_data[0], bins=bins, density=True, alpha=0.35, label="MC")
        ax.hist(bins_data[1], bins=bins, density=True, alpha=0.35, label="Legacy")
        ax.hist(bins_data[2], bins=bins, density=True, alpha=0.35, label="Hierarchical")
        ax.set_xlabel(f"log10({xlabel})" if log10 else xlabel)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Steady-2D {source} histogram comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _distribution_metrics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    metrics = {}
    for key in reference:
        ref = np.asarray(reference[key], dtype=np.float64).reshape(-1)
        pred = np.asarray(candidate[key], dtype=np.float64).reshape(-1)
        metrics[key] = {
            "wasserstein": float(wasserstein_distance(ref, pred)),
            "ks_stat": float(ks_2samp(ref, pred).statistic),
        }
    return metrics


def _mean_metric(metrics: dict[str, dict[str, float]], key: str) -> float:
    return float(np.mean([float(metrics[name][key]) for name in metrics]))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "num_samples": int(args.num_samples),
        "sampling": args.sampling,
        "seed": int(args.seed),
        "sources": {},
    }

    for source in args.sources:
        cond_dims = 5 if source == "boundary" else 6
        cond_samples = _generate_samples(args.num_samples, cond_dims, sampling=args.sampling, seed=args.seed)
        support = _decode_support(source, cond_samples)
        mc = _mc_ground_truth(source, cond_samples, seed=args.seed + 17)
        legacy = _legacy_predictions(source, support, device=args.device, train_sampling=args.legacy_train_sampling, seed=args.seed + 101)
        hierarchical = _hierarchical_predictions(
            source,
            support,
            run_root=args.hierarchical_run_root,
            boundary_run=args.hierarchical_boundary_run,
            internal_run=args.hierarchical_internal_run,
            device=args.device,
            seed=args.seed + 202,
        )

        source_dir = output_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        _plot_ecdf_grid(source_dir / "ecdf_grid.png", mc, legacy, hierarchical, source=source)
        _plot_hist_grid(source_dir / "hist_grid.png", mc, legacy, hierarchical, source=source)

        legacy_metrics = _distribution_metrics(mc, legacy)
        hierarchical_metrics = _distribution_metrics(mc, hierarchical)
        payload = {
            "legacy_vs_mc": {
                "metrics": legacy_metrics,
                "mean_wasserstein": _mean_metric(legacy_metrics, "wasserstein"),
                "mean_ks_stat": _mean_metric(legacy_metrics, "ks_stat"),
            },
            "hierarchical_vs_mc": {
                "metrics": hierarchical_metrics,
                "mean_wasserstein": _mean_metric(hierarchical_metrics, "wasserstein"),
                "mean_ks_stat": _mean_metric(hierarchical_metrics, "ks_stat"),
            },
        }
        (source_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        summary["sources"][source] = payload
        print(
            f"{source}: legacy mean W={payload['legacy_vs_mc']['mean_wasserstein']:.4e}, "
            f"hierarchical mean W={payload['hierarchical_vs_mc']['mean_wasserstein']:.4e}"
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved visual comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
