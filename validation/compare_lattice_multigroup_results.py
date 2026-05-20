#!/usr/bin/env python3
"""Compare two saved multigroup steady lattice solutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

FLUX_FLOOR = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm", type=Path, required=True, help="Path to the GMC/CFM .npz result file.")
    parser.add_argument("--mc", type=Path, required=True, help="Path to the MC .npz result file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/results/lattice_multigroup_compare"),
        help="Directory for comparison plots and summary text.",
    )
    return parser.parse_args()


def positive_log_norm(*arrays: np.ndarray) -> LogNorm | None:
    positives = []
    for data in arrays:
        arr = np.asarray(data, dtype=np.float64)
        positive = arr[arr > 0.0]
        if positive.size:
            positives.append(positive)
    if not positives:
        return None
    positive = np.concatenate(positives)
    if positive.size == 0:
        return None
    if positive.size == 0:
        return None
    vmin = max(float(np.min(positive)), FLUX_FLOOR)
    vmax = float(np.max(positive))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= 0.0:
        return None
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.0001
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_flux_map(data: np.ndarray, title: str, output_path: Path, norm: LogNorm | None = None) -> None:
    if norm is None:
        norm = positive_log_norm(data)
    arr = np.asarray(data, dtype=np.float64)
    if norm is not None:
        arr = np.maximum(arr, norm.vmin)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(arr, origin="lower", cmap="jet", norm=norm)
    plt.colorbar(label="Flux")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_split_flux_map(
    mc_data: np.ndarray,
    cfm_data: np.ndarray,
    title: str,
    output_path: Path,
    norm: LogNorm | None = None,
) -> None:
    if mc_data.shape != cfm_data.shape:
        raise ValueError(f"Split flux shape mismatch: {mc_data.shape} vs {cfm_data.shape}")
    if norm is None:
        norm = positive_log_norm(mc_data, cfm_data)
    mc_arr = np.asarray(mc_data, dtype=np.float64)
    cfm_arr = np.asarray(cfm_data, dtype=np.float64)
    if norm is not None:
        mc_arr = np.maximum(mc_arr, norm.vmin)
        cfm_arr = np.maximum(cfm_arr, norm.vmin)
    ny, nx = mc_arr.shape
    split_ix = nx // 2
    combined = mc_arr.copy()
    combined[:, split_ix:] = cfm_arr[:, split_ix:]

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(combined, origin="lower", cmap="jet", norm=norm)
    plt.axvline(split_ix - 0.5, color="black", linewidth=2.0)
    plt.colorbar(label="Flux")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scalar_map(data: np.ndarray, title: str, label: str, output_path: Path) -> None:
    arr = np.asarray(data, dtype=np.float64)
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(arr, origin="lower", cmap="viridis")
    plt.colorbar(label=label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_centerline(cfm_flux: np.ndarray, mc_flux: np.ndarray, output_path: Path) -> None:
    iy = cfm_flux.shape[0] // 2
    x = np.arange(cfm_flux.shape[1], dtype=np.int64)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, cfm_flux[iy], label="GMC/CFM", linewidth=2.0)
    plt.plot(x, mc_flux[iy], label="MC", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("Cell x-index")
    plt.ylabel("Centerline total flux")
    plt.title(f"Centerline comparison at y-index {iy}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.cfm) as cfm_npz, np.load(args.mc) as mc_npz:
        cfm_total = np.asarray(cfm_npz["total_flux"], dtype=np.float64)
        mc_total = np.asarray(mc_npz["total_flux"], dtype=np.float64)
        cfm_group = np.asarray(cfm_npz["flux"], dtype=np.float64)
        mc_group = np.asarray(mc_npz["flux"], dtype=np.float64)
        cfm_outer = int(np.asarray(cfm_npz["outer_iterations"]).item())
        mc_outer = int(np.asarray(mc_npz["outer_iterations"]).item())
        cfm_backend = str(np.asarray(cfm_npz["backend"]).item())
        mc_backend = str(np.asarray(mc_npz["backend"]).item())

    if cfm_total.shape != mc_total.shape:
        raise ValueError(f"Total flux shape mismatch: {cfm_total.shape} vs {mc_total.shape}")
    if cfm_group.shape != mc_group.shape:
        raise ValueError(f"Group flux shape mismatch: {cfm_group.shape} vs {mc_group.shape}")

    abs_diff = np.abs(cfm_total - mc_total)
    rel_diff = np.divide(abs_diff, mc_total, out=np.zeros_like(abs_diff), where=mc_total > 0.0)
    total_mask = mc_total > 0.0

    group_abs_l1 = np.sum(np.abs(cfm_group - mc_group), axis=(1, 2))
    group_ref_l1 = np.sum(np.abs(mc_group), axis=(1, 2))
    group_rel_l1 = np.divide(group_abs_l1, group_ref_l1, out=np.zeros_like(group_abs_l1), where=group_ref_l1 > 0.0)

    total_flux_norm = positive_log_norm(cfm_total, mc_total)
    group0_flux_norm = positive_log_norm(cfm_group[0], mc_group[0])

    plot_flux_map(cfm_total, f"Total flux ({cfm_backend})", outdir / "cfm_total_flux.png", norm=total_flux_norm)
    plot_flux_map(mc_total, f"Total flux ({mc_backend})", outdir / "mc_total_flux.png", norm=total_flux_norm)
    plot_split_flux_map(mc_total, cfm_total, "Total flux split view (MC left, CFM right)", outdir / "split_total_flux.png", norm=total_flux_norm)
    plot_flux_map(abs_diff, "|CFM - MC| total flux", outdir / "abs_diff_total_flux.png")
    plot_scalar_map(rel_diff, "Relative total-flux error", "|CFM - MC| / MC", outdir / "rel_diff_total_flux.png")
    plot_flux_map(cfm_group[0], f"Group 0 flux ({cfm_backend})", outdir / "cfm_group0_flux.png", norm=group0_flux_norm)
    plot_flux_map(mc_group[0], f"Group 0 flux ({mc_backend})", outdir / "mc_group0_flux.png", norm=group0_flux_norm)
    for group_idx in range(cfm_group.shape[0]):
        group_norm = positive_log_norm(cfm_group[group_idx], mc_group[group_idx])
        plot_split_flux_map(
            mc_group[group_idx],
            cfm_group[group_idx],
            f"Group {group_idx} flux split view (MC left, CFM right)",
            outdir / f"split_group{group_idx}_flux.png",
            norm=group_norm,
        )
    plot_centerline(cfm_total, mc_total, outdir / "centerline_total_flux.png")

    summary_lines = [
        f"CFM file: {args.cfm}",
        f"MC file: {args.mc}",
        f"CFM backend: {cfm_backend}",
        f"MC backend: {mc_backend}",
        f"CFM outer iterations: {cfm_outer}",
        f"MC outer iterations: {mc_outer}",
        f"Grid shape: {cfm_total.shape}",
        f"CFM total flux mean: {cfm_total.mean():.6e}",
        f"MC total flux mean: {mc_total.mean():.6e}",
        f"CFM total flux max: {cfm_total.max():.6e}",
        f"MC total flux max: {mc_total.max():.6e}",
        f"Abs diff mean: {abs_diff.mean():.6e}",
        f"Abs diff max: {abs_diff.max():.6e}",
        f"Rel diff mean: {rel_diff[total_mask].mean():.6e}" if np.any(total_mask) else "Rel diff mean: n/a",
        f"Rel diff max: {rel_diff[total_mask].max():.6e}" if np.any(total_mask) else "Rel diff max: n/a",
        f"Rel diff L2: {np.linalg.norm(rel_diff):.6e}",
        "",
        "Per-group relative L1 error:",
    ]
    summary_lines.extend(f"  group {g}: {group_rel_l1[g]:.6e}" for g in range(group_rel_l1.shape[0]))
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved comparison plots to {outdir}")
    print((outdir / "summary.txt").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
