#!/usr/bin/env python3
"""Visualize scatter-exit distributions from hierarchical models vs. MC ground truth.

Fixes a single cell configuration (width, height, sigma_t, initial position + direction)
and draws many samples, showing per-exit-face histograms of:
  - exit coordinate (face-local, normalised 0-1)
  - exit direction u, v
  - total track length (log10)

Usage:
    python validation/hier_exit_distribution.py
    python validation/hier_exit_distribution.py --runs hier_boundary_lhs_stochastic hier_boundary_500k
    python validation/hier_exit_distribution.py --source-type internal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.hierarchical_steady_2d.runtime import HierarchicalSteady2DSampler
from physics.hierarchical_steady_2d import (
    FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM,
    simulate_hierarchical_steady_2d_scattered_only,
)

FACE_NAMES = {FACE_LEFT: "Left", FACE_RIGHT: "Right", FACE_TOP: "Top", FACE_BOTTOM: "Bottom"}
FACE_COLORS = {FACE_LEFT: "#4e79a7", FACE_RIGHT: "#f28e2b", FACE_TOP: "#59a14f", FACE_BOTTOM: "#e15759"}
MC_COLOR = "#999999"


def _run_mc(
    N: int,
    source_type: str,
    width: float,
    height: float,
    initial_x: float,
    initial_y: float,
    initial_u: float,
    initial_v: float,
    sigma_t: float,
    seed: int,
) -> dict:
    """Run MC for *scattered-only* histories (matches hierarchical training distribution).

    Uses simulate_hierarchical_steady_2d_scattered_only with fixed cond_samples
    so every particle starts from exactly (width, height, ix, iy, iu, iv).
    Column layout (see physics/hierarchical_steady_2d.py):
      [0]=width [1]=height [2]=ix [3]=iy [4]=iu [5]=iv
      [6]=ballistic_len [7]=ballistic_face [8..9]=ballistic_exit_xy
      [10]=ballistic_face_coord [11]=scatter_prob
      [12]=first_scatter_dist [13..14]=first_scatter_xy [15..16]=first_scatter_dir
      [17]=num_scatters [18]=total_track_length
      [19]=final_x [20]=final_y [21]=exit_face [22]=exit_face_coord [23..24]=final_dir
    """
    from physics.hierarchical_steady_2d import SRC_BOUNDARY, SRC_INTERNAL

    np.random.seed(seed)

    # Build cond_samples: columns 0-5 are fixed values normalised relative to
    # the sampling range. But simulate_hierarchical_steady_2d_scattered_only
    # expects raw uniform samples that it transforms via log10 width/height
    # sampling — we bypass that by passing cond_samples as the raw conditions.
    # The function uses cond_samples[:,0..1] for width/height (as uniform [0,1]
    # mapped via log10), and columns 2..5 for position/direction (also [0,1] for
    # position, but here we use actual physical values through a thin wrapper).
    #
    # Easier: call the Numba function directly with fixed cond_samples that
    # encode our chosen physical values via the expected normalization.
    # Width/height are encoded as: cand_w s.t. 10^(log_min_w + cand_w*(log_max_w-log_min_w)) = W
    # Position encoded as y/H (for boundary) or x/W, y/H (for internal).
    # Direction for boundary: encode (u, v) such that sample_incoming_hemisphere gives (iu, iv).
    # This is complex; instead we use the full-scatter MC directly via mc2d.

    from physics.mc2d import simulate_single_cell_2d_boundary_source, simulate_single_cell_2d_internal_source

    # Encode conditions as normalised inputs that the Numba function expects.
    # For boundary: cond cols [0]=W_norm, [1]=H_norm, [2]=y/H, [3]=u_uniform, [4]=v_uniform
    # We'll use a pre-computed cond_samples array with the *exact* physical values,
    # relying on the fact that the Numba function reads:
    #   W = 10^(log_min_w + cond[0] * (log_max_w - log_min_w))
    # So cond[0] = (log10(W) - log_min_w) / (log_max_w - log_min_w)
    # and similarly for H, then direct y/H and u/v.

    log_min_w, log_max_w = np.log10(0.01), np.log10(10.0)
    log_min_h, log_max_h = np.log10(0.01), np.log10(10.0)

    cand_w = (np.log10(width)  - log_min_w) / (log_max_w - log_min_w)
    cand_h = (np.log10(height) - log_min_h) / (log_max_h - log_min_h)

    if source_type == "boundary":
        # For boundary: cols [0]=W, [1]=H, [2]=y/H, [3]=u_01, [4]=v_01
        # The function calls sample_incoming_hemisphere_plus_x_direction_3d_from_unit(u01, v01)
        # We need the inverse mapping.  That function does:
        #   phi = 2*pi*u01,  cos_theta = v01,  direction = (sin_theta*cos_phi, sin_theta*sin_phi, cos_theta)
        # projected to 2D as (u_2d, v_2d).  We skip the full inversion and just use
        # the mc2d function with direct cond overrides — provide the same u/v by using
        # the raw u,v as cond columns 3,4 and rely on the guard_projected_direction_xy call.
        # Simplest: encode u01, v01 such that the sampler returns iu, iv.
        # sample_incoming_hemisphere_plus_x_direction_3d_from_unit(u01, v01):
        #   phi = 2*pi * u01, ct = v01 -> st = sqrt(1-ct^2)
        #   u_3d = st*cos(phi), v_3d = st*sin(phi), w_3d = ct
        # Then guard_projected_direction_xy normalises (u_3d, v_3d) to 2D unit direction.
        # Given target iu, iv: phi = atan2(iv, iu), st = sqrt(iu^2+iv^2), ct = sqrt(1-st^2 if st<1 else 0)
        # => u01 = phi/(2*pi) mod 1,  v01 = ct.
        r2d = np.sqrt(iu**2 + iv**2)
        r2d = min(r2d, 1.0 - 1e-9)
        ct = np.sqrt(max(1.0 - r2d**2, 0.0))
        phi = np.arctan2(iv, iu)
        u01 = (phi / (2 * np.pi)) % 1.0
        v01 = ct  # == cos_theta; but sample_incoming needs v01 ∈ [0,1] mapped to [-1,1]
        # Actually in the code: ct = v01 directly (not remapped), so v01 = ct ∈ [0,1] for forward hemisphere
        cond_samples = np.tile(
            [cand_w, cand_h, initial_y / height, u01, v01],
            (N, 1),
        )
        data = simulate_single_cell_2d_boundary_source(
            N, scatter_ratio=1.0, distribution=0,
            min_width=0.01, max_width=10.0, min_height=0.01, max_height=10.0,
            cond_samples=cond_samples, scattering_mode_code=0,
        )
        # Filter only scattered (exited=1, absorbed=0)
        mask = (data[:, 10] == 1) & (data[:, 11] == 0)
        data = data[mask]
        # Columns: [5]=final_x [6]=final_y [7]=final_u [8]=final_v [9]=track
        # Need exit face — infer from final position
        exit_face = _infer_exit_face(data[:, 5], data[:, 6], width, height)
        return {
            "exit_face":    exit_face,
            "x_out":        data[:, 5],
            "y_out":        data[:, 6],
            "u_out":        data[:, 7],
            "v_out":        data[:, 8],
            "track_length": data[:, 9],
        }
    else:
        cond_samples = np.tile(
            [cand_w, cand_h, initial_x / width, initial_y / height],
            (N, 1),
        )
        # internal: need to also encode direction - columns [4,5] are u01, v01 for isotropic
        # encode u,v as isotropic sample that gives (iu, iv):
        # sample_isotropic_direction_3d_from_unit(u01, v01):
        #   phi=2pi*u01, ct=2*v01-1, st=sqrt(1-ct^2), u_3d=st*cos(phi), v_3d=st*sin(phi)
        # given target iu, iv: same phi as above, ct free -> just use ct=0 (equatorial)
        phi = np.arctan2(iv, iu)
        u01 = (phi / (2 * np.pi)) % 1.0
        v01 = 0.5  # ct = 0 -> equatorial
        cond_samples = np.hstack([cond_samples, np.full((N, 2), [u01, v01])])
        data = simulate_single_cell_2d_internal_source(
            N, scatter_ratio=1.0, distribution=0,
            min_width=0.01, max_width=10.0, min_height=0.01, max_height=10.0,
            cond_samples=cond_samples, scattering_mode_code=0,
        )
        mask = (data[:, 10] == 1) & (data[:, 11] == 0)
        data = data[mask]
        exit_face = _infer_exit_face(data[:, 5], data[:, 6], width, height)
        return {
            "exit_face":    exit_face,
            "x_out":        data[:, 5],
            "y_out":        data[:, 6],
            "u_out":        data[:, 7],
            "v_out":        data[:, 8],
            "track_length": data[:, 9],
        }


def _infer_exit_face(x: np.ndarray, y: np.ndarray, width: float, height: float) -> np.ndarray:
    """Infer exit face from final position (clamp to nearest boundary)."""
    face = np.full(len(x), FACE_RIGHT, dtype=np.int64)
    tol = 1e-6
    on_left   = x <= tol
    on_right  = x >= width  - tol
    on_bottom = y <= tol
    on_top    = y >= height - tol
    face[on_left]   = FACE_LEFT
    face[on_right]  = FACE_RIGHT
    face[on_top]    = FACE_TOP
    face[on_bottom] = FACE_BOTTOM
    return face


def _run_hierarchical(
    sampler: HierarchicalSteady2DSampler,
    N: int,
    width: float,
    height: float,
    initial_x: float,
    initial_y: float,
    initial_u: float,
    initial_v: float,
    sigma_t: float,
    seed: int,
) -> dict:
    width_arr = np.full(N, width)
    height_arr = np.full(N, height)
    ix_arr = np.full(N, initial_x)
    iy_arr = np.full(N, initial_y)
    iu_arr = np.full(N, initial_u)
    iv_arr = np.full(N, initial_v)
    st_arr = np.full(N, sigma_t)

    batch = sampler.sample_batch(
        width=width_arr, height=height_arr,
        initial_x=ix_arr, initial_y=iy_arr,
        initial_u=iu_arr, initial_v=iv_arr,
        sigma_t=st_arr, seed=seed,
    )
    # Only keep scattered particles (ballistic pass-throughs have track == ballistic length)
    # We sample from the full distribution (scattered + ballistic); filter scattered same way MC does
    # Actually sample_batch already mixes both; keep all for fair comparison.
    return {
        "exit_face": batch.exit_face,
        "x_out": batch.x_out,
        "y_out": batch.y_out,
        "u_out": batch.u_out,
        "v_out": batch.v_out,
        "track_length": batch.track_length,
    }


def _face_coord(face: np.ndarray, x: np.ndarray, y: np.ndarray,
                width: float, height: float) -> np.ndarray:
    """Return normalised face-local coordinate for each particle."""
    coord = np.empty(len(face), dtype=np.float64)
    for f, dim, total in [
        (FACE_LEFT,   y, height),
        (FACE_RIGHT,  y, height),
        (FACE_TOP,    x, width),
        (FACE_BOTTOM, x, width),
    ]:
        mask = face == f
        coord[mask] = np.clip(np.asarray(dim)[mask] / max(total, 1e-12), 0.0, 1.0)
    return coord


def _plot_face(ax, mc_vals, model_vals_dict, title, xlabel, bins=40, log_scale=False):
    """Plot MC (grey) and model histograms (coloured) on one axes."""
    if log_scale:
        safe_mc = np.log10(np.maximum(mc_vals, 1e-12))
        hist_vals = {"MC (truth)": (safe_mc, MC_COLOR)}
        for label, (vals, color) in model_vals_dict.items():
            hist_vals[label] = (np.log10(np.maximum(vals, 1e-12)), color)
        xlabel = f"log₁₀({xlabel})"
    else:
        hist_vals = {"MC (truth)": (mc_vals, MC_COLOR)}
        for label, (vals, color) in model_vals_dict.items():
            hist_vals[label] = (vals, color)

    all_vals = np.concatenate([v for v, _ in hist_vals.values()])
    lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    bin_edges = np.linspace(lo, hi, bins + 1)

    for label, (vals, color) in hist_vals.items():
        clipped = np.clip(vals, lo, hi)
        style = dict(bins=bin_edges, density=True, histtype="step", linewidth=1.5)
        if label == "MC (truth)":
            style.update(color=color, linestyle="--", linewidth=2.0, label=label)
        else:
            style.update(color=color, label=label)
        ax.hist(clipped, **style)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.6)


def plot_exit_distributions(
    mc_data: dict,
    model_data: dict[str, dict],
    width: float,
    height: float,
    output_path: Path,
    title_prefix: str = "",
):
    """Create a grid of histograms: rows = exit faces, cols = [coord, u, v, track]."""
    faces = [FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM]
    col_specs = [
        ("exit coord (norm)", "exit coord", False),
        ("exit u",            "u",          False),
        ("exit v",            "v",          False),
        ("track length",      "track",      True),
    ]
    ncols = len(col_specs)

    fig = plt.figure(figsize=(ncols * 3.5, len(faces) * 3.0 + 1.0))
    fig.suptitle(
        f"{title_prefix}  W={width:.3g}  H={height:.3g}\n"
        f"(rows = exit faces, only scattered histories)",
        fontsize=10, y=0.995,
    )
    gs = gridspec.GridSpec(len(faces), ncols, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.94, bottom=0.05)

    mc_face = mc_data["exit_face"]
    mc_x    = mc_data["x_out"]
    mc_y    = mc_data["y_out"]
    mc_u    = mc_data["u_out"]
    mc_v    = mc_data["v_out"]
    mc_tl   = mc_data["track_length"]
    mc_coord = _face_coord(mc_face, mc_x, mc_y, width, height)

    model_colors = plt.cm.tab10(np.linspace(0, 0.6, max(len(model_data), 1)))

    for row, face in enumerate(faces):
        mc_mask = mc_face == face
        n_mc = mc_mask.sum()
        face_name = FACE_NAMES[face]

        for col, (col_title, col_key, log_s) in enumerate(col_specs):
            ax = fig.add_subplot(gs[row, col])

            if n_mc < 5:
                ax.text(0.5, 0.5, f"No MC data\n(face {face_name})",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{face_name} | {col_title}", fontsize=9)
                continue

            if col_key == "exit coord":
                mc_vals = mc_coord[mc_mask]
            elif col_key == "u":
                mc_vals = mc_u[mc_mask]
            elif col_key == "v":
                mc_vals = mc_v[mc_mask]
            else:
                mc_vals = mc_tl[mc_mask]

            model_vals_dict = {}
            for (run_name, mdata), color in zip(model_data.items(), model_colors):
                m_face = mdata["exit_face"]
                m_mask = m_face == face
                if m_mask.sum() < 5:
                    continue
                m_x    = mdata["x_out"][m_mask]
                m_y    = mdata["y_out"][m_mask]
                m_u    = mdata["u_out"][m_mask]
                m_v    = mdata["v_out"][m_mask]
                m_tl   = mdata["track_length"][m_mask]
                m_coord = _face_coord(m_face[m_mask],
                                      mdata["x_out"][m_mask],
                                      mdata["y_out"][m_mask],
                                      width, height)
                if col_key == "exit coord":
                    model_vals_dict[run_name] = (m_coord, color)
                elif col_key == "u":
                    model_vals_dict[run_name] = (m_u, color)
                elif col_key == "v":
                    model_vals_dict[run_name] = (m_v, color)
                else:
                    model_vals_dict[run_name] = (m_tl, color)

            frac_mc = n_mc / len(mc_face)
            _plot_face(ax, mc_vals, model_vals_dict,
                       title=f"{face_name} [{frac_mc:.1%} of exits] | {col_title}",
                       xlabel=col_key, log_scale=log_s)

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualise hierarchical exit distributions vs MC")
    parser.add_argument("--source-type", default="boundary", choices=["boundary", "internal"])
    parser.add_argument("--runs", nargs="+",
                        default=["hier_boundary_lhs_stochastic", "hier_boundary_500k"],
                        help="Run directory names under ml/hierarchical_steady_2d/runs/")
    parser.add_argument("--run-root", default="ml/hierarchical_steady_2d/runs")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    # Cell parameters
    parser.add_argument("--width", type=float, default=0.125)
    parser.add_argument("--height", type=float, default=0.125)
    parser.add_argument("--sigma-t", type=float, default=1.0)
    # Source position / direction — boundary: fixed entry from left face
    parser.add_argument("--initial-x", type=float, default=None,
                        help="Initial x (physical). Default: 0 for boundary, 0.5*W for internal")
    parser.add_argument("--initial-y", type=float, default=None,
                        help="Initial y (physical). Default: 0.5*H")
    parser.add_argument("--initial-u", type=float, default=0.8,
                        help="Initial direction u component (will be unit-normalised)")
    parser.add_argument("--initial-v", type=float, default=0.2,
                        help="Initial direction v component (will be unit-normalised)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="validation/results/hier_exit_distributions")
    args = parser.parse_args()

    # Adjust for source type
    if args.source_type == "internal":
        # Update default runs to internal if not overridden by user
        if args.runs == ["hier_boundary_lhs_stochastic", "hier_boundary_500k"]:
            args.runs = ["hier_internal_lhs_stochastic", "hier_internal_500k"]

    W, H, st = args.width, args.height, args.sigma_t
    ix = args.initial_x if args.initial_x is not None else (0.0 if args.source_type == "boundary" else 0.5 * W)
    iy = args.initial_y if args.initial_y is not None else 0.5 * H
    # Normalise direction
    norm = max(np.sqrt(args.initial_u**2 + args.initial_v**2), 1e-9)
    iu, iv = args.initial_u / norm, args.initial_v / norm

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cell: W={W:.4g}  H={H:.4g}  sigma_t={st:.4g}")
    print(f"Init: x={ix:.4g}  y={iy:.4g}  u={iu:.4f}  v={iv:.4f}")
    print(f"N={args.n_samples:,}  source_type={args.source_type}")
    print()

    # ── MC ground truth ──────────────────────────────────────────────────────
    print("Running MC ground truth...")
    mc_raw = _run_mc(args.n_samples, args.source_type, W, H, ix, iy, iu, iv, st, args.seed)

    # mc_raw from simulate_hierarchical_steady_2d_scattered_only returns structured array
    mc_data = {
        "exit_face":    np.asarray(mc_raw["exit_face"]).astype(np.int64),
        "x_out":        np.asarray(mc_raw["x_out"]).astype(np.float64),
        "y_out":        np.asarray(mc_raw["y_out"]).astype(np.float64),
        "u_out":        np.asarray(mc_raw["u_out"]).astype(np.float64),
        "v_out":        np.asarray(mc_raw["v_out"]).astype(np.float64),
        "track_length": np.asarray(mc_raw["track_length"]).astype(np.float64),
    }
    face_counts = {FACE_NAMES[f]: int((mc_data["exit_face"] == f).sum()) for f in FACE_NAMES}
    print(f"  MC face counts: {face_counts}")

    # ── Hierarchical models ───────────────────────────────────────────────────
    model_data: dict[str, dict] = {}
    for run_name in args.runs:
        run_dir = Path(args.run_root) / run_name
        if not run_dir.exists():
            print(f"  [SKIP] {run_dir} not found")
            continue
        print(f"Loading {run_name}...")
        sampler = HierarchicalSteady2DSampler.from_run_dir(
            run_dir, source_type=args.source_type, device=args.device
        )
        print(f"  Sampling {args.n_samples:,}...")
        mdata = _run_hierarchical(sampler, args.n_samples, W, H, ix, iy, iu, iv, st, args.seed + 1)
        mdata = {k: np.asarray(v) for k, v in mdata.items()}
        face_counts_m = {FACE_NAMES[f]: int((mdata["exit_face"] == f).sum()) for f in FACE_NAMES}
        print(f"  Face counts: {face_counts_m}")
        tl = mdata["track_length"]
        print(f"  track_length: median={np.median(tl):.4e}  p99={np.percentile(tl,99):.4e}  max={tl.max():.4e}")
        model_data[run_name] = mdata

    # ── Plot ──────────────────────────────────────────────────────────────────
    suffix = f"W{W:.3g}_H{H:.3g}_st{st:.3g}_u{iu:.2f}_v{iv:.2f}_{args.source_type}"
    out_path = out_dir / f"exit_distributions_{suffix}.png"
    title = f"source={args.source_type}  σ_t={st}"
    plot_exit_distributions(mc_data, model_data, W, H, out_path, title_prefix=title)

    # ── Track length tail analysis ────────────────────────────────────────────
    print("\nTrack length tail analysis:")
    ballistic = np.sqrt((ix - mc_data["x_out"])**2 + (iy - mc_data["y_out"])**2)
    print(f"  MC:     p99={np.percentile(mc_data['track_length'],99):.4e}  "
          f"p99.9={np.percentile(mc_data['track_length'],99.9):.4e}  "
          f"max={mc_data['track_length'].max():.4e}")
    for run_name, mdata in model_data.items():
        tl = mdata["track_length"]
        n_extreme = (tl > 10 * W * st).sum()
        print(f"  {run_name}: p99={np.percentile(tl,99):.4e}  "
              f"p99.9={np.percentile(tl,99.9):.4e}  max={tl.max():.4e}  "
              f">10×cell_diag: {n_extreme}/{len(tl)}")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
