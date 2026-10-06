#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.transform import (
    denorm_uv, denorm_u_in, denorm_time,
    denorm_log_feature, denorm_length_target,
    denorm_log_scale_logit, normalize_log_scale_logit, denorm_position_relative,
    denorm_linear_scale_logit, normalize_linear_scale_logit,
    normalize_width, normalize_height, normalize_depth, normalize_position_relative,
    normalize_u_in, normalize_uv, normalize_time, normalize_thickness,
    decode_direction_2d,
    uv_to_polar, polar_to_uv, normalize_angle_phi, normalize_radius_r,
    denorm_angle_phi, denorm_radius_r,
    uvw_to_axis_spherical_x, axis_spherical_x_to_uvw,
)
from ml.utils import (
    DATA_DIM_CHOICES,
    compute_and_print_metrics,
    generate_samples,
    generate_target_noise,
    load_model,
    load_norm_params_or_abort as _load_norm_params_or_abort,
    maybe_set_plot_title,
    plot_uniform_ecdf as _plot_uniform_ecdf,
    plot_uniform_hexbin as _plot_uniform_hexbin,
    resolve_2d_direction_encoding as _resolve_2d_direction_encoding,
    run_uniform_ode_inference as _run_uniform_ode_inference,
    safe_get_sim_data,
    set_plot_titles_enabled,
    setup_test_environment,
)
from physics.mc_data_gen import (
    get_2d_boundary_simulation_data, get_2d_internal_source_simulation_data,
    get_2d_time_dependent_internal_source_data, get_2d_time_dependent_boundary_source_data,
    get_2d_triangle_timedep_internal_source_data,
    get_2d_triangle_timedep_boundary_source_data,
    get_2d_quad_timedep_internal_source_data,
    get_1d_time_dependent_internal_source_data, get_1d_time_dependent_boundary_source_data,
    get_3d_boundary_simulation_data,
    get_3d_internal_source_simulation_data,
    get_3d_time_dependent_internal_source_data,
    get_3d_time_dependent_boundary_source_data,
)


# test.py

def run_uniform_inference_2d_electron(
        model, input_dim, cond_dim, device, test_dir,
        electron_npz=None, train_sampling='random', infer_sampling='random',
        batch_size=20000, ode_method='rk4', num_steps=16,
):
    print("\n--- Running Inference (2D Electron Transport) ---")

    norm_params = _load_norm_params_or_abort(
        '2d_electron', 'boundary', train_sampling, "2D Electron Boundary"
    )
    if norm_params is None:
        return

    # Load MC ground truth dataset
    if electron_npz is None or not os.path.exists(electron_npz):
        raise FileNotFoundError(f"Electron dataset NPZ not found: {electron_npz}")

    from physics.electron_data import load_electron_dataset
    sim_gt = load_electron_dataset(electron_npz)

    # Randomly sample evaluation subset from ground truth or sample conditions
    indices = np.random.choice(len(sim_gt), size=min(batch_size, len(sim_gt)), replace=False)
    gt_sub = sim_gt.iloc[indices].reset_index(drop=True)

    y_in_norm = gt_sub["y_in_norm"].values
    u_in = gt_sub["u_in"].values
    v_in = gt_sub["v_in"].values
    E_in = gt_sub["E_in"].values
    x_in_norm = gt_sub["x_in_norm"].values

    norm_y_in = normalize_position_relative(y_in_norm)
    norm_u_in = normalize_uv(u_in)
    norm_v_in = normalize_uv(v_in)
    # Fixed: replaced normalize_log_scale_probit with normalize_log_scale_logit
    norm_E_in = normalize_log_scale_logit(
        E_in, norm_params["E_in_lower"], norm_params["E_in_upper"]
    )
    norm_x_in = normalize_position_relative(x_in_norm)

    cond_np = np.stack([norm_y_in, norm_u_in, norm_v_in, norm_E_in, norm_x_in], axis=1)
    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)

    target_dim = input_dim - cond_dim
    samples = generate_samples(len(gt_sub), cond_dim + target_dim, sampling=infer_sampling)
    noise = generate_target_noise(samples, cond_dim, target_dim, len(gt_sub), device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model, x_init, device, ode_method, num_steps, batch_size=len(gt_sub)
    )

    # Robust key resolution for path length parameters
    path_lower = norm_params.get("path_lower", norm_params.get("path_length_lower"))
    path_upper = norm_params.get("path_upper", norm_params.get("path_length_upper"))

    # Unpack 7 targets (Fixed: denorm_log_scale_logit + .cpu().numpy() transfers)
    pred_x_out = denorm_position_relative(x_final[:, cond_dim + 0]).cpu().numpy()
    pred_y_out = denorm_position_relative(x_final[:, cond_dim + 1]).cpu().numpy()
    pred_u_out = denorm_uv(x_final[:, cond_dim + 2]).cpu().numpy()
    pred_v_out = denorm_uv(x_final[:, cond_dim + 3]).cpu().numpy()
    pred_E_out = denorm_log_scale_logit(
        x_final[:, cond_dim + 4], norm_params["E_out_lower"], norm_params["E_out_upper"]
    ).cpu().numpy()
    pred_energy_dep = denorm_log_scale_logit(
        x_final[:, cond_dim + 5], norm_params["energy_dep_lower"], norm_params["energy_dep_upper"]
    ).cpu().numpy()
    pred_path_len = denorm_log_scale_logit(
        x_final[:, cond_dim + 6], path_lower, path_upper
    ).cpu().numpy()

    gt_data = {
        "x_out": gt_sub["x_out_norm"].values,
        "y_out": gt_sub["y_out_norm"].values,
        "u_out": gt_sub["u_out"].values,
        "v_out": gt_sub["v_out"].values,
        "E_out": gt_sub["E_out"].values,
        "energy_deposited": gt_sub["energy_deposited"].values,
        "path_length": gt_sub["path_length"].values,
    }

    pred_data = {
        "x_out": np.clip(pred_x_out, 0.0, 1.0),
        "y_out": np.clip(pred_y_out, 0.0, 1.0),
        "u_out": np.clip(pred_u_out, -1.0, 1.0),
        "v_out": np.clip(pred_v_out, -1.0, 1.0),
        "E_out": np.maximum(pred_E_out, 0.0),
        "energy_deposited": np.maximum(pred_energy_dep, 0.0),
        "path_length": np.maximum(pred_path_len, 0.0),
    }

    os.makedirs(test_dir, exist_ok=True)
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag="2d_electron_boundary")
def enforce_exit_physics(
    pred_time_exit,
    budget,
    L_out,
    positions=None,
    *,
    snap_space_to_bounds=True,
):
    """
    Project continuous predictions onto exact physical constraints using exit token.

    Args:
        pred_time_exit: Exit token array (>=0.5 => time exit, <0.5 => space exit).
        budget: Path-length budget array.
        L_out: Predicted path lengths.
        positions: Optional list of position arrays in normalized coordinates.
        snap_space_to_bounds: If True, snap space exits to nearest [0,1] boundary.
    """
    pred_time_exit = np.asarray(pred_time_exit)
    budget = np.asarray(budget)
    L_out = np.asarray(L_out)
    pos_list = [] if positions is None else [np.asarray(p) for p in positions]

    n = min(pred_time_exit.shape[0], budget.shape[0], L_out.shape[0])
    if n == 0:
        return L_out, pos_list

    if pos_list:
        n = min(n, *(p.shape[0] for p in pos_list))
    pred_time_exit = pred_time_exit[:n]
    budget = budget[:n]

    is_time = pred_time_exit >= 0.5
    is_space = ~is_time
    time_idx = np.flatnonzero(is_time)
    space_idx = np.flatnonzero(is_space)

    if time_idx.size > 0:
        L_out[time_idx] = budget[time_idx]
    if space_idx.size > 0:
        L_out[space_idx] = np.minimum(L_out[space_idx], budget[space_idx])

    if not pos_list or not snap_space_to_bounds or space_idx.size == 0:
        return L_out, pos_list

    pos_stacked = np.column_stack([p[space_idx] for p in pos_list])
    n_dims = len(pos_list)

    all_dists = np.empty((space_idx.size, 2 * n_dims), dtype=pos_stacked.dtype)
    all_dists[:, 0::2] = pos_stacked
    all_dists[:, 1::2] = 1.0 - pos_stacked
    min_idx = np.argmin(all_dists, axis=1)

    for d in range(n_dims):
        pos_stacked[min_idx == (2 * d), d] = 0.0
        pos_stacked[min_idx == (2 * d + 1), d] = 1.0
        pos_list[d][space_idx] = pos_stacked[:, d]

    return L_out, pos_list


def plot_uniform_results(gt_x, gt_y, gt_u, gt_v, gt_t, pred_x, pred_y, pred_u, pred_v, pred_t, test_dir, source_type):
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    cdf_vars = [
        ("x_out (norm)", gt_x, pred_x, "x_out", False),
        ("y_out (norm)", gt_y, pred_y, "y_out", False),
        ("Final U", gt_u, pred_u, "u", False),
        ("Final V", gt_v, pred_v, "v", False),
        ("Log10(Time in Cell)", gt_t, pred_t, "time", True),
    ]
    for xlabel, g, p, tag, log10 in cdf_vars:
        _plot_uniform_ecdf(
            g,
            p,
            xlabel=xlabel,
            output_path=os.path.join(test_dir, f"uniform_cdf_{tag}.png"),
            title=f"Uniform Test CDF: {xlabel.replace('Log10(', '').replace(')', '')} (2D {source_type.capitalize()})",
            log10=log10,
        )

    hist_specs = [
        ("x_out (norm)", gt_x, pred_x, "x_out", (0, 1), False),
        ("y_out (norm)", gt_y, pred_y, "y_out", (0, 1), False),
        ("Final U", gt_u, pred_u, "u", (-1,1), False),
        ("Final V", gt_v, pred_v, "v", (-1,1), False),
        ("Time in Cell", gt_t, pred_t, "time", None, True),
    ]
    for label, g, p, tag, lims, use_log_bins in hist_specs:
        plt.figure(figsize=(10,6))
        if use_log_bins:
            eps = 1e-12
            gpos = g[g>eps]; ppos = p[p>eps]
            if gpos.size == 0 or ppos.size == 0:
                bins = 50
            else:
                lo = max(np.min(gpos), np.min(ppos))
                hi = max(np.max(gpos), np.max(ppos))
                if lo <= 0: lo = eps
                bins = np.geomspace(lo, hi, 60)
            plt.hist(g, bins=bins, density=True, alpha=0.6, label="MC", color="tab:blue")
            plt.hist(p, bins=bins, density=True, alpha=0.6, label="GMC", color="tab:red")
            plt.xscale('log')
        else:
            bins = 60 if lims is None else np.linspace(lims[0], lims[1], 60)
            plt.hist(g, bins=bins, density=True, alpha=0.6, label="MC", color="tab:blue")
            plt.hist(p, bins=bins, density=True, alpha=0.6, label="GMC", color="tab:red")
            if lims is not None: plt.xlim(lims)
        plt.xlabel(label); plt.ylabel("Density")
        maybe_set_plot_title(f"Uniform Test Distribution: {label} (2D {source_type.capitalize()})")
        plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
        plt.savefig(os.path.join(test_dir, f"uniform_hist_{tag}.png"), dpi=300)
        plt.close()

    _plot_uniform_hexbin(
        gt_x,
        gt_y,
        pred_x,
        pred_y,
        "x_out (norm)",
        "y_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_xy.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs y_out (norm) (2D {source_type.capitalize()})",
    )
    _plot_uniform_hexbin(
        gt_u,
        gt_v,
        pred_u,
        pred_v,
        "Final U",
        "Final V",
        os.path.join(test_dir, "uniform_hexbin_uv.png"),
        title=f"Uniform Test Hexbin: Final U vs Final V (2D {source_type.capitalize()})",
    )
    eps = 1e-12
    _plot_uniform_hexbin(
        gt_x,
        np.log10(np.maximum(gt_t, eps)),
        pred_x,
        np.log10(np.maximum(pred_t, eps)),
        "x_out (norm)",
        "Log10(Time)",
        os.path.join(test_dir, "uniform_hexbin_x_logtime.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs Log10(Time) (2D {source_type.capitalize()})",
    )

    plt.rcParams.update(plt.rcParamsDefault)


def plot_uniform_results_timedep(gt_x_out, gt_y_out, gt_u, gt_v, gt_L,
                                 pred_x_out, pred_y_out, pred_u, pred_v, pred_L,
                                 test_dir, source_type):
    """Plots for uniform test with (x_out, y_out) target (path-length budget formulation)."""
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

    cdf_vars = [
        ("x_out (norm)", gt_x_out, pred_x_out, "x_out", False),
        ("y_out (norm)", gt_y_out, pred_y_out, "y_out", False),
        ("Final U", gt_u, pred_u, "u", False),
        ("Final V", gt_v, pred_v, "v", False),
        ("Log10(Path Length L)", gt_L, pred_L, "L", True),
    ]
    for xlabel, g, p, tag, log10 in cdf_vars:
        _plot_uniform_ecdf(
            g,
            p,
            xlabel=xlabel,
            output_path=os.path.join(test_dir, f"uniform_cdf_{tag}.png"),
            title=f"Uniform Test CDF: {xlabel.replace('Log10(', '').replace(')', '')} (2D {source_type.capitalize()})",
            log10=log10,
        )

    hist_specs = [
        ("x_out (norm)", gt_x_out, pred_x_out, "x_out", (0, 1), False),
        ("y_out (norm)", gt_y_out, pred_y_out, "y_out", (0, 1), False),
        ("Final U", gt_u, pred_u, "u", (-1, 1), False),
        ("Final V", gt_v, pred_v, "v", (-1, 1), False),
        ("Path Length L", gt_L, pred_L, "L", None, True),
    ]
    for label, g, p, tag, lims, use_log_bins in hist_specs:
        plt.figure(figsize=(10, 6))
        if use_log_bins:
            eps = 1e-12
            gpos = g[g > eps]
            ppos = p[p > eps]
            if gpos.size == 0 or ppos.size == 0:
                bins = 50
            else:
                lo = max(np.min(gpos), np.min(ppos))
                hi = max(np.max(gpos), np.max(ppos))
                if lo <= 0:
                    lo = eps
                bins = np.geomspace(lo, hi, 60)
            plt.hist(g, bins=bins, density=True, alpha=0.6, label="MC", color="tab:blue")
            plt.hist(p, bins=bins, density=True, alpha=0.6, label="GMC", color="tab:red")
            plt.xscale('log')
        else:
            bins = 60 if lims is None else np.linspace(lims[0], lims[1], 60)
            plt.hist(g, bins=bins, density=True, alpha=0.6, label="MC", color="tab:blue")
            plt.hist(p, bins=bins, density=True, alpha=0.6, label="GMC", color="tab:red")
            if lims is not None:
                plt.xlim(lims)
        plt.xlabel(label)
        plt.ylabel("Density")
        maybe_set_plot_title(f"Uniform Test Distribution: {label} (2D {source_type.capitalize()})")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, f"uniform_hist_{tag}.png"), dpi=300)
        plt.close()

    _plot_uniform_hexbin(
        gt_x_out,
        gt_y_out,
        pred_x_out,
        pred_y_out,
        "x_out (norm)",
        "y_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_xy.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs y_out (norm) (2D {source_type.capitalize()})",
    )
    _plot_uniform_hexbin(
        gt_u,
        gt_v,
        pred_u,
        pred_v,
        "Final U",
        "Final V",
        os.path.join(test_dir, "uniform_hexbin_uv.png"),
        title=f"Uniform Test Hexbin: Final U vs Final V (2D {source_type.capitalize()})",
    )
    eps = 1e-12
    _plot_uniform_hexbin(
        gt_x_out,
        np.log10(np.maximum(gt_L, eps)),
        pred_x_out,
        np.log10(np.maximum(pred_L, eps)),
        "x_out (norm)",
        "Log10(L)",
        os.path.join(test_dir, "uniform_hexbin_x_logL.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs Log10(L) (2D {source_type.capitalize()})",
    )
    _plot_uniform_hexbin(
        gt_u,
        np.log10(np.maximum(gt_L, eps)),
        pred_u,
        np.log10(np.maximum(pred_L, eps)),
        "Final U",
        "Log10(L)",
        os.path.join(test_dir, "uniform_hexbin_u_logL.png"),
        title=f"Uniform Test Hexbin: Final U vs Log10(L) (2D {source_type.capitalize()})",
    )

    plt.rcParams.update(plt.rcParamsDefault)


def _report_interior_budget_inconsistency(pred_L, pred_budget, pred_positions, eps=1e-3, tol=1e-12):
    """Report cases where L_out < budget while all predicted positions are interior."""
    pred_L = np.asarray(pred_L)
    pred_budget = np.asarray(pred_budget)
    n = min(pred_L.shape[0], pred_budget.shape[0])
    if n == 0:
        print("Inconsistency check: no samples available.")
        return

    pred_L = pred_L[:n]
    pred_budget = pred_budget[:n]
    interior_mask = np.ones(n, dtype=bool)
    for pos in pred_positions:
        pos = np.asarray(pos)[:n]
        interior_mask &= (pos > eps) & (pos < 1.0 - eps)

    inconsistent = (pred_L < pred_budget - tol) & interior_mask
    n_bad = int(np.sum(inconsistent))
    frac_bad = float(n_bad / n)
    print(
        f"Inconsistent samples (L_out < budget and x_out interior, eps={eps:.1e}): "
        f"{frac_bad:.3%} ({n_bad}/{n})"
    )


def _analyze_space_exits(
    gt_positions,
    gt_space_mask,
    pred_positions,
    pred_space_mask,
    dim_labels=["x", "y"],
    test_dir=None,
    tag="space_exit_boundary_dist",
):
    """
    Quantify spatial exits: check if particles classified as space exits lie
    on a boundary and tally the distribution of boundary faces they hit.
    """
    print("\n--- Space Exit Spatial Distribution ---")

    def get_boundary_stats(positions, mask, labels):
        mask = np.asarray(mask, dtype=bool)
        if np.sum(mask) == 0:
            return None, 0.0, 0, [], 0, np.array([], dtype=float)

        # Shape: (N, D) where N is number of space exits, D is dimensions.
        pos_stacked = np.column_stack([np.asarray(p)[mask] for p in positions])
        n_samples, n_dims = pos_stacked.shape

        # Distances to normalized boundaries (0 and 1).
        dist_lower = pos_stacked
        dist_upper = 1.0 - pos_stacked

        # Order: [x=0, x=1, y=0, y=1, ...].
        all_dists = np.empty((n_samples, 2 * n_dims))
        all_dists[:, 0::2] = dist_lower
        all_dists[:, 1::2] = dist_upper

        boundary_names = []
        for lbl in labels:
            if lbl == "x":
                boundary_names.extend(["Left (x=0)", "Right (x=1)"])
            elif lbl == "y":
                boundary_names.extend(["Bottom (y=0)", "Top (y=1)"])
            elif lbl == "z":
                boundary_names.extend(["Front (z=0)", "Back (z=1)"])
            else:
                boundary_names.extend([f"{lbl}=0", f"{lbl}=1"])

        closest_idx = np.argmin(all_dists, axis=1)
        min_dists = np.min(all_dists, axis=1)

        counts = np.bincount(closest_idx, minlength=2 * n_dims)
        mean_err = np.mean(min_dists)
        exact_hits = np.sum(min_dists == 0.0)
        return counts, mean_err, exact_hits, boundary_names, n_samples, min_dists

    gt_counts, gt_err, gt_exact_hits, b_names, n_gt, gt_min_dists = get_boundary_stats(
        gt_positions, gt_space_mask, dim_labels
    )
    pr_counts, pr_err, pr_exact_hits, _, n_pr, pr_min_dists = get_boundary_stats(
        pred_positions, pred_space_mask, dim_labels
    )

    print(f"Total Space Exits -> GT: {n_gt}, Pred: {n_pr}")
    if n_gt > 0 and n_pr > 0:
        print(f"Mean error dist to nearest boundary -> GT: {gt_err:.2e}, Pred: {pr_err:.2e}")
        print(
            f"Exact boundary hits (dist == 0.0)   -> GT: {gt_exact_hits} ({(gt_exact_hits / n_gt) * 100:.2f}%), "
            f"Pred: {pr_exact_hits} ({(pr_exact_hits / n_pr) * 100:.2f}%)"
        )
        print(f"\n{'Boundary Face':<16} | {'GT Count':<10} | {'GT %':<8} | {'Pred Count':<10} | {'Pred %':<8}")
        print("-" * 65)
        for i, name in enumerate(b_names):
            gt_c = gt_counts[i]
            pr_c = pr_counts[i]
            print(f"{name:<16} | {gt_c:<10} | {gt_c / n_gt:7.1%} | {pr_c:<10} | {pr_c / n_pr:7.1%}")
    elif n_gt > 0:
        print("No predicted space exits to evaluate.")
    elif n_pr > 0:
        print("No ground truth space exits to evaluate.")
    else:
        print("No space exits in either GT or predictions.")

    if test_dir is not None and (n_gt > 0 or n_pr > 0):
        os.makedirs(test_dir, exist_ok=True)
        gt_min_dists = np.asarray(gt_min_dists, dtype=float)
        pr_min_dists = np.asarray(pr_min_dists, dtype=float)
        gt_min_dists = gt_min_dists[np.isfinite(gt_min_dists)]
        pr_min_dists = pr_min_dists[np.isfinite(pr_min_dists)]

        # CDF of distance to nearest boundary.
        plt.figure(figsize=(10, 6))
        if gt_min_dists.size > 0:
            xs = np.sort(gt_min_dists)
            ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
            plt.plot(xs, ys, lw=2, label="GT", color="tab:blue")
        if pr_min_dists.size > 0:
            xs = np.sort(pr_min_dists)
            ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
            plt.plot(xs, ys, lw=2, ls="--", label="Pred", color="tab:red")
        plt.xlabel("Distance to nearest boundary")
        plt.ylabel("ECDF")
        maybe_set_plot_title("Space Exit Distance-To-Boundary CDF")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, f"{tag}_cdf.png"), dpi=300)
        plt.close()

        # PDF histogram of distance to nearest boundary.
        max_dist = 0.0
        if gt_min_dists.size > 0:
            max_dist = max(max_dist, float(np.max(gt_min_dists)))
        if pr_min_dists.size > 0:
            max_dist = max(max_dist, float(np.max(pr_min_dists)))
        if max_dist <= 0.0:
            max_dist = 1e-12
        bins = np.linspace(0.0, max_dist, 60)

        plt.figure(figsize=(10, 6))
        if gt_min_dists.size > 0:
            plt.hist(gt_min_dists, bins=bins, density=True, alpha=0.6, label="GT", color="tab:blue")
        if pr_min_dists.size > 0:
            plt.hist(pr_min_dists, bins=bins, density=True, alpha=0.6, label="Pred", color="tab:red")
        plt.xlabel("Distance to nearest boundary")
        plt.ylabel("PDF")
        maybe_set_plot_title("Space Exit Distance-To-Boundary PDF Histogram")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, f"{tag}_pdf_hist.png"), dpi=300)
        plt.close()


def _analyze_2d_exit_geometry(
    gt_x_out,
    gt_y_out,
    gt_space_exit,
    gt_time_exit,
    pred_x_out,
    pred_y_out,
    pred_space_exit,
    pred_time_exit,
    test_dir,
    tag="2d_exit_geometry",
):
    """Analyze whether space exits land on boundaries and time exits stay interior."""
    gt_x_out = np.asarray(gt_x_out)
    gt_y_out = np.asarray(gt_y_out)
    gt_space_exit = np.asarray(gt_space_exit).astype(bool)
    gt_time_exit = np.asarray(gt_time_exit).astype(bool)
    pred_x_out = np.asarray(pred_x_out)
    pred_y_out = np.asarray(pred_y_out)
    pred_space_exit = np.asarray(pred_space_exit).astype(bool)
    pred_time_exit = np.asarray(pred_time_exit).astype(bool)

    n = min(
        gt_x_out.shape[0], gt_y_out.shape[0], gt_space_exit.shape[0], gt_time_exit.shape[0],
        pred_x_out.shape[0], pred_y_out.shape[0], pred_space_exit.shape[0], pred_time_exit.shape[0],
    )
    if n == 0:
        print("2D exit-geometry analysis: no samples available.")
        return

    gt_x_out = gt_x_out[:n]
    gt_y_out = gt_y_out[:n]
    gt_space_exit = gt_space_exit[:n]
    gt_time_exit = gt_time_exit[:n]
    pred_x_out = pred_x_out[:n]
    pred_y_out = pred_y_out[:n]
    pred_space_exit = pred_space_exit[:n]
    pred_time_exit = pred_time_exit[:n]

    def _boundary_mask(x, y, eps):
        return (x <= eps) | (x >= 1.0 - eps) | (y <= eps) | (y >= 1.0 - eps)

    def _interior_mask(x, y, eps):
        return (x > eps) & (x < 1.0 - eps) & (y > eps) & (y < 1.0 - eps)

    def _safe_ratio(mask_a, mask_b):
        denom = int(np.sum(mask_b))
        if denom == 0:
            return np.nan
        return float(np.sum(mask_a & mask_b) / denom)

    eps_vals = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]

    print("\n--- 2D Exit Geometry Consistency ---")
    print("  space_exit => near boundary,  time_exit => interior")
    print(f"    {'eps':>8}  {'MC space->bnd':>14}  {'Pred space->bnd':>16}  {'MC time->int':>12}  {'Pred time->int':>14}")

    lines = []
    lines.append("2D Exit Geometry Consistency")
    lines.append("space_exit => near boundary, time_exit => interior")
    lines.append(f"{'eps':>8}  {'MC space->bnd':>14}  {'Pred space->bnd':>16}  {'MC time->int':>12}  {'Pred time->int':>14}")

    for eps in eps_vals:
        gt_bnd = _boundary_mask(gt_x_out, gt_y_out, eps)
        pred_bnd = _boundary_mask(pred_x_out, pred_y_out, eps)
        gt_int = _interior_mask(gt_x_out, gt_y_out, eps)
        pred_int = _interior_mask(pred_x_out, pred_y_out, eps)

        mc_space_to_bnd = _safe_ratio(gt_bnd, gt_space_exit)
        pred_space_to_bnd = _safe_ratio(pred_bnd, pred_space_exit)
        mc_time_to_int = _safe_ratio(gt_int, gt_time_exit)
        pred_time_to_int = _safe_ratio(pred_int, pred_time_exit)

        line = (
            f"    {eps:8.1e}  "
            f"{mc_space_to_bnd if np.isfinite(mc_space_to_bnd) else float('nan'):14.3f}  "
            f"{pred_space_to_bnd if np.isfinite(pred_space_to_bnd) else float('nan'):16.3f}  "
            f"{mc_time_to_int if np.isfinite(mc_time_to_int) else float('nan'):12.3f}  "
            f"{pred_time_to_int if np.isfinite(pred_time_to_int) else float('nan'):14.3f}"
        )
        print(line)
        lines.append(line)

    def _face_index(x, y):
        # 0=left, 1=right, 2=bottom, 3=top
        d = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        return np.argmin(d, axis=1)

    face_names = np.array(["left", "right", "bottom", "top"])
    gt_idx = _face_index(gt_x_out[gt_space_exit], gt_y_out[gt_space_exit]) if np.any(gt_space_exit) else np.array([], dtype=int)
    pred_idx = _face_index(pred_x_out[pred_space_exit], pred_y_out[pred_space_exit]) if np.any(pred_space_exit) else np.array([], dtype=int)

    gt_counts = np.bincount(gt_idx, minlength=4).astype(int)
    pred_counts = np.bincount(pred_idx, minlength=4).astype(int)
    gt_total = int(np.sum(gt_counts))
    pred_total = int(np.sum(pred_counts))
    gt_pct = (gt_counts / max(gt_total, 1)) * 100.0
    pred_pct = (pred_counts / max(pred_total, 1)) * 100.0

    print("\n--- 2D Space-Exit Face Distribution ---")
    print(f"  Space exits used: MC={gt_total}  Pred={pred_total}")
    print(f"    {'face':>8}  {'MC count':>10}  {'MC %':>8}  {'Pred count':>12}  {'Pred %':>8}")
    lines.append("")
    lines.append("2D Space-Exit Face Distribution")
    lines.append(f"Space exits used: MC={gt_total}  Pred={pred_total}")
    lines.append(f"{'face':>8}  {'MC count':>10}  {'MC %':>8}  {'Pred count':>12}  {'Pred %':>8}")

    for i, face in enumerate(face_names):
        line = f"    {face:>8}  {gt_counts[i]:10d}  {gt_pct[i]:8.2f}  {pred_counts[i]:12d}  {pred_pct[i]:8.2f}"
        print(line)
        lines.append(line)

    report_path = os.path.join(test_dir, f"{tag}_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved 2D exit-geometry summary to {report_path}")

    xloc = np.arange(4)
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(xloc - width / 2, gt_pct, width=width, label="MC", color="tab:blue", alpha=0.8)
    plt.bar(xloc + width / 2, pred_pct, width=width, label="GMC", color="tab:red", alpha=0.8)
    plt.xticks(xloc, face_names)
    plt.ylabel("Space-exit face percentage (%)")
    maybe_set_plot_title("2D Space-Exit Face Distribution")
    plt.legend()
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, f"{tag}_face_percentages.png"), dpi=300)
    plt.close()


def plot_uniform_results_1d_timedep(gt_data, pred_data, test_dir, source_type):
    """Plots for 1D time-dependent test with path-length budget formulation."""
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

    gt_x_out = gt_data['x_out_norm']
    gt_mu_out = gt_data['mu_out']
    gt_L = gt_data['L']
    gt_space_exit = gt_data['space_exit']
    gt_time_exit = gt_data['time_exit']

    pred_x_out = pred_data['x_out_norm']
    pred_mu_out = pred_data['mu_out']
    pred_L = pred_data['L']

    _plot_uniform_ecdf(
        gt_x_out,
        pred_x_out,
        xlabel='x_out (normalized)',
        output_path=os.path.join(test_dir, 'cdf_x_out.png'),
        figure_size=(6, 5),
    )
    _plot_uniform_ecdf(
        gt_mu_out,
        pred_mu_out,
        xlabel='mu_out',
        output_path=os.path.join(test_dir, 'cdf_mu_out.png'),
        figure_size=(6, 5),
    )
    _plot_uniform_ecdf(
        gt_L,
        pred_L,
        xlabel='log10(L)',
        output_path=os.path.join(test_dir, 'cdf_L.png'),
        figure_size=(6, 5),
        log10=True,
    )

    # 4) x_out histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(gt_x_out, bins=50, alpha=0.6, density=True, label='MC', color='tab:blue')
    ax.hist(pred_x_out, bins=50, alpha=0.6, density=True, label='GMC', color='tab:red')
    ax.set_xlabel('x_out (normalized)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'hist_x_out.png'), dpi=300)
    plt.close()

    # 5) mu_out histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(gt_mu_out, bins=50, alpha=0.6, density=True, label='MC', color='tab:blue')
    ax.hist(pred_mu_out, bins=50, alpha=0.6, density=True, label='GMC', color='tab:red')
    ax.set_xlabel('mu_out')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'hist_mu_out.png'), dpi=300)
    plt.close()

    # 6) L histogram (log scale)
    fig, ax = plt.subplots(figsize=(6, 5))
    eps = 1e-12
    gt_L_pos = gt_L[gt_L > eps]
    pred_L_pos = pred_L[pred_L > eps]
    if len(gt_L_pos) > 0 and len(pred_L_pos) > 0:
        lo = min(gt_L_pos.min(), pred_L_pos.min())
        hi = max(gt_L_pos.max(), pred_L_pos.max())
        bins = np.geomspace(lo, hi, 50)
        ax.hist(gt_L, bins=bins, alpha=0.6, density=True, label='MC', color='tab:blue')
        ax.hist(pred_L, bins=bins, alpha=0.6, density=True, label='GMC', color='tab:red')
        ax.set_xscale('log')
    ax.set_xlabel('L (track length)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'hist_L.png'), dpi=300)
    plt.close()

    # 7) Exit type analysis (MC only)
    fig, ax = plt.subplots(figsize=(6, 5))
    n_space = gt_space_exit.sum()
    n_time = gt_time_exit.sum()
    ax.bar(['Space-like', 'Time-like'], [n_space, n_time], color=['blue', 'orange'])
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'exit_types.png'), dpi=300)
    plt.close()

    # 8) x_out for space-like vs time-like exits (MC)
    fig, ax = plt.subplots(figsize=(6, 5))
    gt_x_space = gt_x_out[gt_space_exit == 1]
    gt_x_time = gt_x_out[gt_time_exit == 1]
    if len(gt_x_space) > 0:
        ax.hist(gt_x_space, bins=50, alpha=0.6, density=True, label='Space-like')
    if len(gt_x_time) > 0:
        ax.hist(gt_x_time, bins=50, alpha=0.6, density=True, label='Time-like')
    ax.set_xlabel('x_out (normalized)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'exit_position_by_type.png'), dpi=300)
    plt.close()

    _plot_uniform_hexbin(
        gt_x_out,
        gt_mu_out,
        pred_x_out,
        pred_mu_out,
        "x_out (normalized)",
        "mu_out",
        os.path.join(test_dir, "uniform_hexbin_x_mu.png"),
        figure_size=(8, 6),
    )
    eps = 1e-12
    _plot_uniform_hexbin(
        gt_mu_out,
        np.log10(np.maximum(gt_L, eps)),
        pred_mu_out,
        np.log10(np.maximum(pred_L, eps)),
        "mu_out",
        "log10(L)",
        os.path.join(test_dir, "uniform_hexbin_mu_logL.png"),
        figure_size=(8, 6),
    )
    _plot_uniform_hexbin(
        gt_x_out,
        np.log10(np.maximum(gt_L, eps)),
        pred_x_out,
        np.log10(np.maximum(pred_L, eps)),
        "x_out (normalized)",
        "log10(L)",
        os.path.join(test_dir, "uniform_hexbin_x_logL.png"),
        figure_size=(8, 6),
    )

    plt.rcParams.update(plt.rcParamsDefault)


def run_uniform_inference_1d_timedep(model, input_dim, cond_dim, device, test_dir,
                                     source_type='internal', train_sampling='random', infer_sampling='random',
                                     batch_size=20000, ode_method='rk4', num_steps=16,
                                     min_thickness=0.01, max_thickness=10.0,
                                     min_path_budget=1e-6, max_path_budget=0.5):
    print("\n--- Running Inference (1D Time-Dependent) ---")

    norm_params = _load_norm_params_or_abort(
        '1d_timedep',
        source_type,
        train_sampling,
        f"1D timedep {source_type}",
    )
    if norm_params is None:
        return

    cond_features = 3 if source_type == 'boundary' else 4
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_thick = samples[:, 0] * (np.log10(max_thickness) - np.log10(min_thickness)) + np.log10(min_thickness)
    thickness_vals = 10**log_thick
    log_budget = samples[:, 1] * (np.log10(max_path_budget) - np.log10(min_path_budget)) + np.log10(min_path_budget)
    path_budget_vals = 10**log_budget

    if source_type == 'boundary':
        # Inverse-CDF map for cosine-weighted mu on [0,1].
        mu_in_vals = np.sqrt(samples[:, 2])
    else:
        x_in_vals = samples[:, 2]
        mu_in_vals = 2.0 * samples[:, 3] - 1.0

    thick_norm = normalize_thickness(thickness_vals, norm_params['thickness_lower'], norm_params['thickness_upper'])
    budget_norm = normalize_log_scale_logit(path_budget_vals, norm_params['budget_lower'], norm_params['budget_upper'])
    if source_type == 'boundary':
        mu_in_norm = normalize_u_in(mu_in_vals)
        cond_np = np.stack([thick_norm, budget_norm, mu_in_norm], axis=1)
    else:
        x_in_norm = normalize_position_relative(x_in_vals)
        mu_in_norm = normalize_uv(mu_in_vals)
        cond_np = np.stack([thick_norm, budget_norm, x_in_norm, mu_in_norm], axis=1)

    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        show_timing=False,
    ).numpy()
    pred_x_out = np.clip(denorm_linear_scale_logit(torch.tensor(x_final[:, cond_dim + 0])).numpy(), 0, 1)
    pred_x_out_raw = pred_x_out.copy()
    pred_mu_out = np.clip(denorm_uv(torch.tensor(x_final[:, cond_dim + 1])).numpy(), -1, 1)
    pred_L = np.maximum(
        denorm_length_target(x_final[:, cond_dim + 2], path_budget_vals[:x_final.shape[0]], norm_params),
        0.0,
    )
    pred_time_exit = None
    if x_final.shape[1] > cond_dim + 3:
        pred_time_exit_prob = denorm_linear_scale_logit(x_final[:, cond_dim + 3])
        pred_time_exit = (pred_time_exit_prob >= 0.5).astype(np.float64)
        print(f"Pred time-like fraction (exit token): {float(pred_time_exit.mean()):.3f}")
        pred_L, [pred_x_out] = enforce_exit_physics(
            pred_time_exit=pred_time_exit,
            budget=path_budget_vals[:pred_L.shape[0]],
            L_out=pred_L,
            positions=[pred_x_out],
        )
    if pred_time_exit is None:
        raise RuntimeError("Timedep inference requires exit token output, but model output has no exit-token dimension.")

    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_1d_time_dependent_boundary_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_thickness=min_thickness, max_thickness=max_thickness,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling
        )
    else:
        sim_gt = safe_get_sim_data(
            get_1d_time_dependent_internal_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_thickness=min_thickness, max_thickness=max_thickness,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling
        )

    sim_gt = sim_gt[(sim_gt["space_exit"] == 1) | (sim_gt["time_exit"] == 1)].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    pred_data = {'x_out_norm': pred_x_out, 'mu_out': pred_mu_out, 'L': pred_L}
    gt_data = {
        'x_out_norm': sim_gt['x_out_norm'].values,
        'mu_out': sim_gt['mu_out'].values,
        'L': sim_gt['L'].values,
        'space_exit': sim_gt['space_exit'].values,
        'time_exit': sim_gt['time_exit'].values,
    }

    _report_interior_budget_inconsistency(
        pred_L=pred_L,
        pred_budget=path_budget_vals[:pred_L.shape[0]],
        pred_positions=[pred_x_out_raw],
    )
    plot_uniform_results_1d_timedep(gt_data, pred_data, test_dir, source_type)
    is_time_pred = pred_time_exit >= 0.5
    _analyze_space_exits(
        gt_positions=[gt_data['x_out_norm']],
        gt_space_mask=(sim_gt['time_exit'].values == 0),
        pred_positions=[pred_x_out_raw],
        pred_space_mask=(~is_time_pred),
        dim_labels=["x"],
        test_dir=test_dir,
        tag="space_exit_boundary_dist_1d_timedep",
    )
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag=f"1d_timedep_{source_type}")


def run_uniform_inference(model, input_dim, cond_dim, device, test_dir,
                          source_type='boundary', scheduler_type='condOT', train_sampling='random', infer_sampling='random',
                          batch_size=20000, ode_method='rk4', num_steps=16,
                          min_width=0.01, max_width=10.0,
                          min_height=0.01, max_height=10.0):
    print("\n--- Running Inference (Uniform Conditions) ---")
    print(f"Source Type: 2D {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"----------------------------------------------")

    norm_params = _load_norm_params_or_abort('2d', source_type, train_sampling, f"2D {source_type}")
    if norm_params is None:
        return

    cond_features = 5 if source_type == 'boundary' else 6
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width_vals = 10**log_w
    log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height_vals = 10**log_h

    if source_type == 'boundary':
        y_in_norm_vals = samples[:, 2]
        mu = 2.0 * samples[:, 3] - 1.0
        # Analytic mapping for u>0 hemisphere; avoids rejection loops.
        phi = samples[:, 4] * np.pi - np.pi / 2.0
        t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        u_in_vals = t * np.cos(phi)
        v_in_vals = t * np.sin(phi)
    else:
        x_in_norm_vals = samples[:, 2]
        y_in_norm_vals = samples[:, 3]
        mu = 2.0 * samples[:, 4] - 1.0
        phi = samples[:, 5] * 2.0 * np.pi
        t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        u_in_vals = t * np.cos(phi)
        v_in_vals = t * np.sin(phi)

    # Normalize conditions
    norm_w_vals = normalize_width(width_vals, norm_params['width_lower'], norm_params['width_upper'])
    norm_h_vals = normalize_height(height_vals, norm_params['height_lower'], norm_params['height_upper'])
    norm_y_in_vals = normalize_position_relative(y_in_norm_vals)

    if source_type == 'boundary':
        # Convert (u_in, v_in) to polar (phi_in, r_in)
        phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)
        norm_phi_in_vals = normalize_angle_phi(phi_in_vals)
        norm_r_in_vals = normalize_radius_r(r_in_vals)
        cond_np = np.stack([norm_w_vals, norm_h_vals, norm_y_in_vals, norm_phi_in_vals, norm_r_in_vals], axis=1)
    else:
        norm_x_in_vals = normalize_position_relative(x_in_norm_vals)
        # Convert (u_in, v_in) to polar (phi_in, r_in)
        phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)
        norm_phi_in_vals = normalize_angle_phi(phi_in_vals)
        norm_r_in_vals = normalize_radius_r(r_in_vals)
        cond_np = np.stack([norm_w_vals, norm_h_vals, norm_x_in_vals, norm_y_in_vals, norm_phi_in_vals, norm_r_in_vals], axis=1)

    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"

    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=batch_size,
    )
    pred_logit_x    = x_final[:, cond_dim + 0]
    pred_logit_y    = x_final[:, cond_dim + 1]
    pred_logit_phi  = x_final[:, cond_dim + 2]
    pred_logit_r    = x_final[:, cond_dim + 3]
    pred_logit_time = x_final[:, cond_dim + 4]

    pred_x    = denorm_position_relative(pred_logit_x).numpy()
    pred_y    = denorm_position_relative(pred_logit_y).numpy()
    pred_phi  = denorm_angle_phi(pred_logit_phi).numpy()
    pred_r    = denorm_radius_r(pred_logit_r).numpy()
    pred_time = denorm_log_feature(pred_logit_time.numpy(), norm_params, 'time')

    pred_u, pred_v = polar_to_uv(pred_phi, pred_r)

    print("Generating ground-truth MC samples under the same uniform ranges...")
    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_2d_boundary_simulation_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            sampling=infer_sampling,
        )
    else:
        sim_gt = safe_get_sim_data(
            get_2d_internal_source_simulation_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            sampling=infer_sampling,
        )
    sim_gt = sim_gt[sim_gt["exited"] == 1].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_u = sim_gt['final_dir_u'].values
    gt_v = sim_gt['final_dir_v'].values
    gt_t = sim_gt['time_in_cell'].values
    gt_x = sim_gt['final_pos_x_norm'].values
    gt_y = sim_gt['final_pos_y_norm'].values

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results(
        gt_x,
        gt_y,
        gt_u,
        gt_v,
        gt_t,
        np.clip(pred_x, 0.0, 1.0),
        np.clip(pred_y, 0.0, 1.0),
        pred_u,
        pred_v,
        pred_time,
        test_dir,
        source_type,
    )
    pred_data = {
        'x_out_norm': np.clip(pred_x, 0.0, 1.0),
        'y_out_norm': np.clip(pred_y, 0.0, 1.0),
        'u': pred_u,
        'v': pred_v,
        'time': pred_time,
    }
    gt_data = {'x_out_norm': gt_x, 'y_out_norm': gt_y, 'u': gt_u, 'v': gt_v, 'time': gt_t}
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag=f"2d_{source_type}")


def plot_uniform_results_3d(gt, pred, test_dir, source_type):
    """Basic distribution plots for 3D single-cell models."""
    gt_x = gt["x_out_norm"]
    gt_y = gt["y_out_norm"]
    gt_z = gt["z_out_norm"]
    gt_mu = gt["mu_out"]
    gt_phi01 = (gt["phi_out"] / (2.0 * np.pi)) % 1.0
    gt_t = gt["time_in_cell"]

    pr_x = pred["x_out_norm"]
    pr_y = pred["y_out_norm"]
    pr_z = pred["z_out_norm"]
    pr_mu = pred["mu_out"]
    pr_phi01 = (pred["phi_out"] / (2.0 * np.pi)) % 1.0
    pr_t = pred["time_in_cell"]

    cdf_vars = [
        ("x_out (norm)", gt_x, pr_x, "x_out", False),
        ("y_out (norm)", gt_y, pr_y, "y_out", False),
        ("z_out (norm)", gt_z, pr_z, "z_out", False),
        ("mu_out (=u_out)", gt_mu, pr_mu, "mu_out", False),
        ("phi_out / 2π (mod 1)", gt_phi01, pr_phi01, "phi_out", False),
        ("log10(Time in Cell)", gt_t, pr_t, "time", True),
    ]

    for xlabel, g, p, tag, log10 in cdf_vars:
        _plot_uniform_ecdf(
            g,
            p,
            xlabel=xlabel,
            output_path=os.path.join(test_dir, f"uniform_cdf_{tag}.png"),
            title=f"Uniform Test CDF: {xlabel.replace('log10(', '').replace(')', '')} (3D {source_type.capitalize()})",
            log10=log10,
        )

    eps = 1e-12
    _plot_uniform_hexbin(
        gt_x,
        gt_y,
        pr_x,
        pr_y,
        "x_out (norm)",
        "y_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_xy.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs y_out (norm) (3D {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_y,
        gt_z,
        pr_y,
        pr_z,
        "y_out (norm)",
        "z_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_yz.png"),
        title=f"Uniform Test Hexbin: y_out (norm) vs z_out (norm) (3D {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_x,
        gt_z,
        pr_x,
        pr_z,
        "x_out (norm)",
        "z_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_xz.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs z_out (norm) (3D {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_mu,
        np.log10(np.maximum(gt_t, eps)),
        pr_mu,
        np.log10(np.maximum(pr_t, eps)),
        "mu_out (=u_out)",
        "log10(Time)",
        os.path.join(test_dir, "uniform_hexbin_mu_logtime.png"),
        title=f"Uniform Test Hexbin: mu_out (=u_out) vs log10(Time) (3D {source_type.capitalize()})",
        figure_size=(10, 8),
    )


def run_uniform_inference_3d(model, input_dim, cond_dim, device, test_dir,
                             source_type='boundary', train_sampling='random', infer_sampling='random',
                             batch_size=20000, ode_method='rk4', num_steps=16,
                             min_width=0.01, max_width=10.0,
                             min_height=0.01, max_height=10.0,
                             min_depth=0.01, max_depth=10.0):
    print("\n--- Running Inference (Uniform Conditions) ---")
    print(f"Source Type: 3D {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"------------------------------------------------")

    norm_params = _load_norm_params_or_abort('3d', source_type, train_sampling, f"3D {source_type}")
    if norm_params is None:
        return

    cond_features = 7 if source_type == 'boundary' else 8
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width_vals = 10 ** log_w
    log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height_vals = 10 ** log_h
    log_d = samples[:, 2] * (np.log10(max_depth) - np.log10(min_depth)) + np.log10(min_depth)
    depth_vals = 10 ** log_d

    if source_type == 'boundary':
        y_in_norm_vals = samples[:, 3]
        z_in_norm_vals = samples[:, 4]
        mu_in_vals = samples[:, 5]
        phi_in_vals = samples[:, 6] * 2.0 * np.pi
    else:
        x_in_norm_vals = samples[:, 3]
        y_in_norm_vals = samples[:, 4]
        z_in_norm_vals = samples[:, 5]
        mu_in_vals = 2.0 * samples[:, 6] - 1.0
        phi_in_vals = samples[:, 7] * 2.0 * np.pi

    norm_w_vals = normalize_width(width_vals, norm_params['width_lower'], norm_params['width_upper'])
    norm_h_vals = normalize_height(height_vals, norm_params['height_lower'], norm_params['height_upper'])
    norm_d_vals = normalize_depth(depth_vals, norm_params['depth_lower'], norm_params['depth_upper'])

    if source_type == 'boundary':
        norm_y_in = normalize_position_relative(y_in_norm_vals)
        norm_z_in = normalize_position_relative(z_in_norm_vals)
        norm_phi_in = normalize_angle_phi(phi_in_vals)
        norm_mu_in = normalize_u_in(mu_in_vals)
        cond_np = np.stack([norm_w_vals, norm_h_vals, norm_d_vals, norm_y_in, norm_z_in, norm_phi_in, norm_mu_in], axis=1)
    else:
        norm_x_in = normalize_position_relative(x_in_norm_vals)
        norm_y_in = normalize_position_relative(y_in_norm_vals)
        norm_z_in = normalize_position_relative(z_in_norm_vals)
        norm_phi_in = normalize_angle_phi(phi_in_vals)
        norm_mu_in = normalize_uv(mu_in_vals)
        cond_np = np.stack([norm_w_vals, norm_h_vals, norm_d_vals, norm_x_in, norm_y_in, norm_z_in, norm_phi_in, norm_mu_in], axis=1)

    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"

    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=batch_size,
    )
    pred_logit_x = x_final[:, cond_dim + 0]
    pred_logit_y = x_final[:, cond_dim + 1]
    pred_logit_z = x_final[:, cond_dim + 2]
    pred_logit_phi = x_final[:, cond_dim + 3]
    pred_logit_mu = x_final[:, cond_dim + 4]
    pred_logit_t = x_final[:, cond_dim + 5]

    pred_x = denorm_linear_scale_logit(pred_logit_x).numpy()
    pred_y = denorm_linear_scale_logit(pred_logit_y).numpy()
    pred_z = denorm_linear_scale_logit(pred_logit_z).numpy()
    pred_phi = denorm_angle_phi(pred_logit_phi).numpy()
    pred_mu = denorm_uv(pred_logit_mu).numpy()
    pred_t = denorm_log_feature(pred_logit_t.numpy(), norm_params, 'time')

    pred = {
        "x_out_norm": np.clip(pred_x, 0.0, 1.0),
        "y_out_norm": np.clip(pred_y, 0.0, 1.0),
        "z_out_norm": np.clip(pred_z, 0.0, 1.0),
        "phi_out": (pred_phi % (2.0 * np.pi)),
        "mu_out": np.clip(pred_mu, -1.0, 1.0),
        "time_in_cell": np.maximum(pred_t, 0.0),
    }

    print("Generating ground-truth MC samples under the same uniform ranges...")
    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_3d_boundary_simulation_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_depth=min_depth, max_depth=max_depth,
            sampling=infer_sampling,
        )
    else:
        sim_gt = safe_get_sim_data(
            get_3d_internal_source_simulation_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_depth=min_depth, max_depth=max_depth,
            sampling=infer_sampling,
        )
    sim_gt = sim_gt[sim_gt["exited"] == 1].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_phi, gt_mu = uvw_to_axis_spherical_x(sim_gt["final_dir_u"].values,
                                           sim_gt["final_dir_v"].values,
                                           sim_gt["final_dir_w"].values)
    gt = {
        "x_out_norm": sim_gt["final_pos_x_norm"].values,
        "y_out_norm": sim_gt["final_pos_y_norm"].values,
        "z_out_norm": sim_gt["final_pos_z_norm"].values,
        "phi_out": gt_phi,
        "mu_out": gt_mu,
        "time_in_cell": sim_gt["time_in_cell"].values,
    }

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results_3d(gt, pred, test_dir, source_type)
    compute_and_print_metrics(gt, pred, test_dir, tag=f"3d_{source_type}")

    print("\n--- Summary Statistics ---")
    print(f"MC samples:  {len(gt['time_in_cell'])}")
    print(f"CFM samples: {len(pred['time_in_cell'])}")
    for k in ("x_out_norm", "y_out_norm", "z_out_norm", "mu_out", "time_in_cell"):
        print(f"MC  {k}: mean={np.mean(gt[k]):.4f}, std={np.std(gt[k]):.4f}")
        print(f"CFM {k}: mean={np.mean(pred[k]):.4f}, std={np.std(pred[k]):.4f}")


def plot_uniform_results_3d_timedep(gt, pred, test_dir, source_type):
    """Basic distribution plots for 3D time-dependent (path budget) models."""
    gt_x = gt["x_out_norm"]
    gt_y = gt["y_out_norm"]
    gt_z = gt["z_out_norm"]
    gt_mu = gt["mu_out"]
    gt_phi01 = (gt["phi_out"] / (2.0 * np.pi)) % 1.0
    gt_L = gt["L"]

    pr_x = pred["x_out_norm"]
    pr_y = pred["y_out_norm"]
    pr_z = pred["z_out_norm"]
    pr_mu = pred["mu_out"]
    pr_phi01 = (pred["phi_out"] / (2.0 * np.pi)) % 1.0
    pr_L = pred["L"]

    cdf_vars = [
        ("x_out (norm)", gt_x, pr_x, "x_out", False),
        ("y_out (norm)", gt_y, pr_y, "y_out", False),
        ("z_out (norm)", gt_z, pr_z, "z_out", False),
        ("mu_out (=u_out)", gt_mu, pr_mu, "mu_out", False),
        ("phi_out / 2π (mod 1)", gt_phi01, pr_phi01, "phi_out", False),
        ("log10(L)", gt_L, pr_L, "L", True),
    ]
    for xlabel, g, p, tag, log10 in cdf_vars:
        _plot_uniform_ecdf(
            g,
            p,
            xlabel=xlabel,
            output_path=os.path.join(test_dir, f"uniform_cdf_{tag}.png"),
            title=f"Uniform Test CDF: {xlabel.replace('log10(', '').replace(')', '')} (3D timedep {source_type.capitalize()})",
            log10=log10,
        )

    eps = 1e-12
    _plot_uniform_hexbin(
        gt_x,
        gt_y,
        pr_x,
        pr_y,
        "x_out (norm)",
        "y_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_xy.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs y_out (norm) (3D timedep {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_y,
        gt_z,
        pr_y,
        pr_z,
        "y_out (norm)",
        "z_out (norm)",
        os.path.join(test_dir, "uniform_hexbin_yz.png"),
        title=f"Uniform Test Hexbin: y_out (norm) vs z_out (norm) (3D timedep {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_x,
        np.log10(np.maximum(gt_L, eps)),
        pr_x,
        np.log10(np.maximum(pr_L, eps)),
        "x_out (norm)",
        "log10(L)",
        os.path.join(test_dir, "uniform_hexbin_x_logL.png"),
        title=f"Uniform Test Hexbin: x_out (norm) vs log10(L) (3D timedep {source_type.capitalize()})",
        figure_size=(10, 8),
    )
    _plot_uniform_hexbin(
        gt_mu,
        np.log10(np.maximum(gt_L, eps)),
        pr_mu,
        np.log10(np.maximum(pr_L, eps)),
        "mu_out (=u_out)",
        "log10(L)",
        os.path.join(test_dir, "uniform_hexbin_mu_logL.png"),
        title=f"Uniform Test Hexbin: mu_out (=u_out) vs log10(L) (3D timedep {source_type.capitalize()})",
        figure_size=(10, 8),
    )


def run_uniform_inference_3d_timedep(model, input_dim, cond_dim, device, test_dir,
                                     source_type='boundary', train_sampling='random', infer_sampling='random',
                                     batch_size=20000, ode_method='rk4', num_steps=16,
                                     min_width=0.01, max_width=10.0,
                                     min_height=0.01, max_height=10.0,
                                     min_depth=0.01, max_depth=10.0,
                                     min_path_budget=1e-5, max_path_budget=0.1):
    print("\n--- Running Inference (Uniform Conditions, 3D Time-Dependent) ---")
    print(f"Source Type: 3D {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Path budget range: [{min_path_budget}, {max_path_budget}]")
    print(f"---------------------------------------------------------------")

    norm_params = _load_norm_params_or_abort(
        '3d_timedep',
        source_type,
        train_sampling,
        f"3D timedep {source_type}",
    )
    if norm_params is None:
        return

    cond_features = 8 if source_type == 'boundary' else 9
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width_vals = 10 ** log_w
    log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height_vals = 10 ** log_h
    log_d = samples[:, 2] * (np.log10(max_depth) - np.log10(min_depth)) + np.log10(min_depth)
    depth_vals = 10 ** log_d
    log_budget = samples[:, 3] * (np.log10(max_path_budget) - np.log10(min_path_budget)) + np.log10(min_path_budget)
    budget_vals = 10 ** log_budget

    if source_type == 'boundary':
        y_in_norm_vals = samples[:, 4]
        z_in_norm_vals = samples[:, 5]
        mu_in_vals = samples[:, 6]
        phi_in_vals = samples[:, 7] * 2.0 * np.pi
    else:
        x_in_norm_vals = samples[:, 4]
        y_in_norm_vals = samples[:, 5]
        z_in_norm_vals = samples[:, 6]
        mu_in_vals = 2.0 * samples[:, 7] - 1.0
        phi_in_vals = samples[:, 8] * 2.0 * np.pi

    norm_w = normalize_width(width_vals, norm_params['width_lower'], norm_params['width_upper'])
    norm_h = normalize_height(height_vals, norm_params['height_lower'], norm_params['height_upper'])
    norm_d = normalize_depth(depth_vals, norm_params['depth_lower'], norm_params['depth_upper'])
    norm_budget = normalize_log_scale_logit(budget_vals, norm_params['budget_lower'], norm_params['budget_upper'])

    if source_type == 'boundary':
        norm_y = normalize_position_relative(y_in_norm_vals)
        norm_z = normalize_position_relative(z_in_norm_vals)
        sin_phi = np.sin(phi_in_vals)
        cos_phi = np.cos(phi_in_vals)
        norm_mu = normalize_u_in(mu_in_vals)
        cond_np = np.stack([norm_w, norm_h, norm_d, norm_budget, norm_y, norm_z, sin_phi, cos_phi, norm_mu], axis=1)
    else:
        norm_x = normalize_position_relative(x_in_norm_vals)
        norm_y = normalize_position_relative(y_in_norm_vals)
        norm_z = normalize_position_relative(z_in_norm_vals)
        sin_phi = np.sin(phi_in_vals)
        cos_phi = np.cos(phi_in_vals)
        norm_mu = normalize_uv(mu_in_vals)
        cond_np = np.stack([norm_w, norm_h, norm_d, norm_budget, norm_x, norm_y, norm_z, sin_phi, cos_phi, norm_mu], axis=1)

    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"

    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=batch_size,
    )
    pred_logit_x = x_final[:, cond_dim + 0]
    pred_logit_y = x_final[:, cond_dim + 1]
    pred_logit_z = x_final[:, cond_dim + 2]
    pred_sin_phi = x_final[:, cond_dim + 3].cpu().numpy()
    pred_cos_phi = x_final[:, cond_dim + 4].cpu().numpy()
    pred_logit_mu = x_final[:, cond_dim + 5]
    pred_logit_L_frac = x_final[:, cond_dim + 6]

    pred_x = denorm_linear_scale_logit(pred_logit_x).numpy()
    pred_y = denorm_linear_scale_logit(pred_logit_y).numpy()
    pred_z = denorm_linear_scale_logit(pred_logit_z).numpy()
    pred_x_raw = pred_x.copy()
    pred_y_raw = pred_y.copy()
    pred_z_raw = pred_z.copy()
    pred_phi = (np.arctan2(pred_sin_phi, pred_cos_phi) + 2.0 * np.pi) % (2.0 * np.pi)
    pred_mu = denorm_uv(pred_logit_mu).numpy()
    pred_L = np.maximum(
        denorm_length_target(pred_logit_L_frac, budget_vals[:pred_logit_L_frac.shape[0]], norm_params).cpu().numpy(),
        0.0,
    )
    pred_time_exit = None
    if x_final.shape[1] > cond_dim + 7:
        pred_time_exit_prob = denorm_linear_scale_logit(x_final[:, cond_dim + 7]).cpu().numpy()
        pred_time_exit = (pred_time_exit_prob >= 0.5).astype(np.float64)
        pred_L, [pred_x, pred_y, pred_z] = enforce_exit_physics(
            pred_time_exit=pred_time_exit,
            budget=budget_vals[:pred_L.shape[0]],
            L_out=pred_L,
            positions=[pred_x, pred_y, pred_z],
        )
    if pred_time_exit is None:
        raise RuntimeError("Timedep inference requires exit token output, but model output has no exit-token dimension.")

    pred = {
        "x_out_norm": np.clip(pred_x, 0.0, 1.0),
        "y_out_norm": np.clip(pred_y, 0.0, 1.0),
        "z_out_norm": np.clip(pred_z, 0.0, 1.0),
        "phi_out": (pred_phi % (2.0 * np.pi)),
        "mu_out": np.clip(pred_mu, -1.0, 1.0),
        "L": np.maximum(pred_L, 0.0),
    }

    print("Generating MC ground truth...")
    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_3d_time_dependent_boundary_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_depth=min_depth, max_depth=max_depth,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling,
        )
    else:
        sim_gt = safe_get_sim_data(
            get_3d_time_dependent_internal_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_depth=min_depth, max_depth=max_depth,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling,
        )
    sim_gt = sim_gt[(sim_gt["space_exit"] == 1) | (sim_gt["time_exit"] == 1)].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_phi, gt_mu = uvw_to_axis_spherical_x(sim_gt["final_dir_u"].values,
                                           sim_gt["final_dir_v"].values,
                                           sim_gt["final_dir_w"].values)
    gt = {
        "x_out_norm": sim_gt["final_pos_x_norm"].values,
        "y_out_norm": sim_gt["final_pos_y_norm"].values,
        "z_out_norm": sim_gt["final_pos_z_norm"].values,
        "phi_out": gt_phi,
        "mu_out": gt_mu,
        "L": sim_gt["L"].values,
        "space_exit": sim_gt["space_exit"].values,
        "time_exit": sim_gt["time_exit"].values,
    }

    gt_time_like_frac = float(sim_gt["time_exit"].mean())
    pred_time_like_frac = float(np.mean(pred_time_exit))
    print(f"GT time-like fraction:   {gt_time_like_frac:.3f}")
    print(f"Pred time-like fraction: {pred_time_like_frac:.3f}")
    _report_interior_budget_inconsistency(
        pred_L=pred["L"],
        pred_budget=budget_vals[:pred["L"].shape[0]],
        pred_positions=[pred_x_raw, pred_y_raw, pred_z_raw],
    )

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results_3d_timedep(gt, pred, test_dir, source_type)
    is_time_pred = pred_time_exit >= 0.5
    _analyze_space_exits(
        gt_positions=[gt["x_out_norm"], gt["y_out_norm"], gt["z_out_norm"]],
        gt_space_mask=(sim_gt["time_exit"].values == 0),
        pred_positions=[pred_x_raw, pred_y_raw, pred_z_raw],
        pred_space_mask=(~is_time_pred),
        dim_labels=["x", "y", "z"],
        test_dir=test_dir,
        tag="space_exit_boundary_dist_3d_timedep",
    )
    compute_and_print_metrics(gt, pred, test_dir, tag=f"3d_timedep_{source_type}")

    # Exit type counts (MC)
    n_space = np.sum(gt["space_exit"])
    n_time = np.sum(gt["time_exit"])
    plt.figure(figsize=(6, 5))
    plt.bar(["Space-like", "Time-like"], [n_space, n_time], color=["blue", "orange"])
    plt.ylabel("Count")
    maybe_set_plot_title(f"Exit Types (MC) - 3D timedep {source_type.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, "exit_types.png"), dpi=300)
    plt.close()

    print("\n--- Summary Statistics ---")
    print(f"MC samples:  {len(gt['L'])}")
    print(f"CFM samples: {len(pred['L'])}")
    for k in ("x_out_norm", "y_out_norm", "z_out_norm", "mu_out", "L"):
        print(f"MC  {k}: mean={np.mean(gt[k]):.4f}, std={np.std(gt[k]):.4f}")
        print(f"CFM {k}: mean={np.mean(pred[k]):.4f}, std={np.std(pred[k]):.4f}")


def run_uniform_inference_timedep(model, input_dim, cond_dim, device, test_dir,
                                  source_type='boundary', scheduler_type='condOT', train_sampling='random', infer_sampling='random',
                                  batch_size=20000, ode_method='rk4', num_steps=16,
                                  min_width=0.01, max_width=10.0,
                                  min_height=0.01, max_height=10.0,
                                  min_path_budget=1e-5, max_path_budget=0.1,
                                  angular_sampling=0):
    print("\n--- Running Inference (Uniform Conditions, Time-Dependent) ---")
    print(f"Source Type: 2D {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Path budget range: [{min_path_budget}, {max_path_budget}]")
    print(f"---------------------------------------------------------------")

    norm_params = _load_norm_params_or_abort(
        '2d_timedep',
        source_type,
        train_sampling,
        f"2D timedep {source_type}",
    )
    if norm_params is None:
        return
    direction_encoding_2d = _resolve_2d_direction_encoding(norm_params)

    cond_features = 6 if source_type == 'boundary' else 7
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width_vals = 10**log_w
    log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height_vals = 10**log_h
    log_budget = samples[:, 2] * (np.log10(max_path_budget) - np.log10(min_path_budget)) + np.log10(min_path_budget)
    budget_vals = 10**log_budget

    if source_type == 'boundary':
        y_in_norm_vals = samples[:, 3]
        # Match physics/mc2dtime.py boundary injection:
        # angular_sampling=1 -> cosine-law in u, angular_sampling=0 -> uniform-u.
        if angular_sampling == 1:
            u_in_vals = np.sqrt(samples[:, 4])
        else:
            u_in_vals = samples[:, 4]
        azim = samples[:, 5] * 2.0 * np.pi
        tangential = np.sqrt(np.clip(1.0 - u_in_vals * u_in_vals, 0.0, 1.0))
        v_in_vals = tangential * np.cos(azim)
    else:
        x_in_norm_vals = samples[:, 3]
        y_in_norm_vals = samples[:, 4]
        mu = 2.0 * samples[:, 5] - 1.0
        phi = samples[:, 6] * 2.0 * np.pi
        t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        u_in_vals = t * np.cos(phi)
        v_in_vals = t * np.sin(phi)

    norm_w_vals = normalize_width(width_vals, norm_params['width_lower'], norm_params['width_upper'])
    norm_h_vals = normalize_height(height_vals, norm_params['height_lower'], norm_params['height_upper'])
    norm_budget_vals = normalize_log_scale_logit(budget_vals, norm_params['budget_lower'], norm_params['budget_upper'])
    norm_y_in_vals = normalize_position_relative(y_in_norm_vals)
    phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)

    if source_type == 'boundary':
        cond_np = np.stack(
            [
                norm_w_vals, norm_h_vals, norm_budget_vals, norm_y_in_vals,
                normalize_angle_phi(phi_in_vals), normalize_radius_r(r_in_vals),
            ],
            axis=1,
        )
    else:
        norm_x_in_vals = normalize_position_relative(x_in_norm_vals)
        cond_np = np.stack(
            [
                norm_w_vals, norm_h_vals, norm_budget_vals,
                norm_x_in_vals, norm_y_in_vals,
                normalize_angle_phi(phi_in_vals), normalize_radius_r(r_in_vals),
            ],
            axis=1,
        )

    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"

    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=batch_size,
    )
    pred_logit_x_out = x_final[:, cond_dim + 0]
    pred_logit_y_out = x_final[:, cond_dim + 1]

    pred_x_out = denorm_position_relative(pred_logit_x_out).numpy()
    pred_y_out = denorm_position_relative(pred_logit_y_out).numpy()
    pred_x_out_raw = pred_x_out.copy()
    pred_y_out_raw = pred_y_out.copy()
    pred_u_t, pred_v_t = decode_direction_2d(
        [x_final[:, cond_dim + 2], x_final[:, cond_dim + 3]],
        direction_encoding_2d,
        norm_params=norm_params,
    )
    pred_u = pred_u_t.cpu().numpy()
    pred_v = pred_v_t.cpu().numpy()
    pred_logit_L_frac = x_final[:, cond_dim + 4]
    pred_L = np.maximum(
        denorm_length_target(pred_logit_L_frac, budget_vals[:pred_logit_L_frac.shape[0]], norm_params).cpu().numpy(),
        0.0,
    )
    if x_final.shape[1] > cond_dim + 5:
        pred_time_exit_prob = denorm_linear_scale_logit(x_final[:, cond_dim + 5]).cpu().numpy()
        pred_time_exit = (pred_time_exit_prob >= 0.5).astype(np.float64)
    else:
        raise RuntimeError("Timedep inference requires exit token output, but model output has no exit-token dimension.")

    pred_L, [pred_x_out, pred_y_out] = enforce_exit_physics(
        pred_time_exit=pred_time_exit,
        budget=budget_vals[:pred_L.shape[0]],
        L_out=pred_L,
        positions=[pred_x_out, pred_y_out],
    )

    print("Generating ground-truth MC samples under the same uniform ranges...")
    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_2d_time_dependent_boundary_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            angular_sampling=angular_sampling,
            sampling=infer_sampling,
        )
    else:
        sim_gt = safe_get_sim_data(
            get_2d_time_dependent_internal_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling,
        )
    sim_gt = sim_gt[(sim_gt["space_exit"] == 1) | (sim_gt["time_exit"] == 1)].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_x_out = sim_gt['final_pos_x_norm'].values
    gt_y_out = sim_gt['final_pos_y_norm'].values
    gt_u = sim_gt['final_dir_u'].values
    gt_v = sim_gt['final_dir_v'].values
    gt_L = sim_gt['L'].values

    gt_time_like_frac = float(sim_gt['time_exit'].mean())
    pred_time_exit_mask = pred_time_exit >= 0.5
    pred_space_exit_mask = ~pred_time_exit_mask
    pred_time_like_frac = float(np.mean(pred_time_exit_mask))
    print(f"GT time-like fraction:   {gt_time_like_frac:.3f}")
    print(f"Pred time-like fraction: {pred_time_like_frac:.3f}")
    _report_interior_budget_inconsistency(
        pred_L=pred_L,
        pred_budget=budget_vals[:pred_L.shape[0]],
        pred_positions=[pred_x_out_raw, pred_y_out_raw],
    )

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results_timedep(gt_x_out, gt_y_out, gt_u, gt_v, gt_L,
                                 pred_x_out, pred_y_out, pred_u, pred_v, pred_L,
                                 test_dir, source_type)
    _analyze_space_exits(
        gt_positions=[gt_x_out, gt_y_out],
        gt_space_mask=(sim_gt['time_exit'].values == 0),
        pred_positions=[pred_x_out_raw, pred_y_out_raw],
        pred_space_mask=(~pred_time_exit_mask),
        dim_labels=["x", "y"],
        test_dir=test_dir,
        tag="space_exit_boundary_dist_2d_timedep",
    )
    _analyze_2d_exit_geometry(
        gt_x_out=gt_x_out,
        gt_y_out=gt_y_out,
        gt_space_exit=sim_gt['space_exit'].values,
        gt_time_exit=sim_gt['time_exit'].values,
        pred_x_out=pred_x_out_raw,
        pred_y_out=pred_y_out_raw,
        pred_space_exit=pred_space_exit_mask,
        pred_time_exit=pred_time_exit_mask,
        test_dir=test_dir,
        tag=f"uniform_2d_timedep_{source_type}_exit_geometry",
    )
    pred_data = {'x_out_norm': pred_x_out, 'y_out_norm': pred_y_out, 'u_out': pred_u, 'v_out': pred_v, 'L': pred_L}
    gt_data = {'x_out_norm': gt_x_out, 'y_out_norm': gt_y_out, 'u_out': gt_u, 'v_out': gt_v, 'L': gt_L}
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag=f"2d_timedep_{source_type}")


def run_uniform_inference_2d_triangle_timedep(
    model, input_dim, cond_dim, device, test_dir,
    source_type='internal', scheduler_type='condOT', train_sampling='random', infer_sampling='random',
    batch_size=20000, ode_method='rk4', num_steps=16,
    min_width=0.01, max_width=10.0,
    min_height=0.01, max_height=10.0,
    min_path_budget=1e-5, max_path_budget=0.1,
    angular_sampling=0,
):
    print("\n--- Running Inference (Uniform Conditions, 2D Triangle Time-Dependent) ---")
    print(f"Source Type: 2D Triangle {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Path budget range: [{min_path_budget}, {max_path_budget}]")
    if source_type == 'boundary':
        print(f"Angular sampling: {angular_sampling} (0=uniform-normal, 1=cosine-law)")
    print(f"--------------------------------------------------------------------------")

    norm_params = _load_norm_params_or_abort(
        '2d_triangle_timedep',
        source_type,
        train_sampling,
        f"2D triangle timedep {source_type}",
    )
    if norm_params is None:
        return
    direction_encoding_2d = _resolve_2d_direction_encoding(norm_params)

    cond_features = 7 if source_type == 'boundary' else 8
    target_dim = input_dim - cond_dim
    samples = generate_samples(batch_size, cond_features + target_dim, sampling=infer_sampling)

    log_L = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    L_tri_vals = 10**log_L
    log_r = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    r_tri_vals = 10**log_r
    theta_vals = 0.01 + samples[:, 2] * (np.pi - 0.02)
    log_budget = samples[:, 3] * (np.log10(max_path_budget) - np.log10(min_path_budget)) + np.log10(min_path_budget)
    budget_vals = 10**log_budget

    if source_type == 'boundary':
        xi1_in_vals = samples[:, 4]
        xi2_in_vals = np.zeros_like(xi1_in_vals)
        normal_unit = samples[:, 5]
        if angular_sampling == 1:
            v_in_vals = np.sqrt(np.clip(normal_unit, 0.0, 1.0))
        else:
            v_in_vals = np.clip(normal_unit, 0.0, 1.0)
        v_in_vals = np.maximum(v_in_vals, 1e-9)
        phi = samples[:, 6] * 2.0 * np.pi
        tangent_mag = np.sqrt(np.clip(1.0 - v_in_vals * v_in_vals, 0.0, 1.0))
        u_in_vals = tangent_mag * np.cos(phi)
    else:
        # Folded simplex mapping preserves low-discrepancy sequence structure.
        u1 = samples[:, 4]
        u2 = samples[:, 5]
        mask = (u1 + u2) > 1.0
        u1 = np.where(mask, 1.0 - u1, u1)
        u2 = np.where(mask, 1.0 - u2, u2)
        xi1_in_vals = u1
        xi2_in_vals = u2

        mu = 2.0 * samples[:, 6] - 1.0
        phi = samples[:, 7] * 2.0 * np.pi
        t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        u_in_vals = t * np.cos(phi)
        v_in_vals = t * np.sin(phi)

    norm_L_tri = normalize_log_scale_logit(L_tri_vals, norm_params['L_tri_lower'], norm_params['L_tri_upper'])
    norm_r_tri = normalize_log_scale_logit(r_tri_vals, norm_params['r_tri_lower'], norm_params['r_tri_upper'])
    norm_theta = normalize_linear_scale_logit(theta_vals / np.pi)
    norm_budget = normalize_log_scale_logit(budget_vals, norm_params['budget_lower'], norm_params['budget_upper'])
    norm_xi1_in = normalize_position_relative(xi1_in_vals)
    if source_type != 'boundary':
        norm_xi2_in = normalize_position_relative(xi2_in_vals)
    phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)
    if source_type == 'boundary':
        cond_np = np.stack(
            [
                norm_L_tri, norm_r_tri, norm_theta, norm_budget, norm_xi1_in,
                normalize_angle_phi(phi_in_vals), normalize_radius_r(r_in_vals),
            ],
            axis=1,
        )
    else:
        cond_np = np.stack(
            [
                norm_L_tri, norm_r_tri, norm_theta, norm_budget, norm_xi1_in, norm_xi2_in,
                normalize_angle_phi(phi_in_vals), normalize_radius_r(r_in_vals),
            ],
            axis=1,
        )
    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"

    noise = generate_target_noise(samples, cond_features, target_dim, batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=batch_size,
    )
    pred_logit_xi1_out = x_final[:, cond_dim + 0]
    pred_logit_xi2_out = x_final[:, cond_dim + 1]

    pred_xi1_out = denorm_position_relative(pred_logit_xi1_out).numpy()
    pred_xi2_out = denorm_position_relative(pred_logit_xi2_out).numpy()
    pred_u_t, pred_v_t = decode_direction_2d(
        [x_final[:, cond_dim + 2], x_final[:, cond_dim + 3]],
        direction_encoding_2d,
        norm_params=norm_params,
    )
    pred_u = pred_u_t.cpu().numpy()
    pred_v = pred_v_t.cpu().numpy()
    pred_L = np.maximum(
        denorm_length_target(x_final[:, cond_dim + 4], budget_vals[:x_final.shape[0]], norm_params).cpu().numpy(),
        0.0,
    )
    if x_final.shape[1] > cond_dim + 5:
        pred_time_exit_prob = denorm_linear_scale_logit(x_final[:, cond_dim + 5]).cpu().numpy()
        pred_time_exit = (pred_time_exit_prob >= 0.5).astype(np.float64)
    else:
        raise RuntimeError("Timedep inference requires exit token output, but model output has no exit-token dimension.")

    pred_L, _ = enforce_exit_physics(
        pred_time_exit=pred_time_exit,
        budget=budget_vals[:pred_L.shape[0]],
        L_out=pred_L,
        positions=None,
        snap_space_to_bounds=False,
    )

    print("Generating ground-truth MC samples under the same uniform ranges...")
    if source_type == 'boundary':
        sim_gt = safe_get_sim_data(
            get_2d_triangle_timedep_boundary_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_L=min_width, max_L=max_width,
            min_r=min_height, max_r=max_height,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            angular_sampling=angular_sampling,
            sampling=infer_sampling,
        )
    else:
        sim_gt = safe_get_sim_data(
            get_2d_triangle_timedep_internal_source_data,
            num_neutrons=batch_size, scatter_ratio=1.0, distribution=0,
            min_L=min_width, max_L=max_width,
            min_r=min_height, max_r=max_height,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=infer_sampling,
        )
    sim_gt = sim_gt[(sim_gt["space_exit"] == 1) | (sim_gt["time_exit"] == 1)].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_xi1_out = sim_gt['xi1_out'].values
    gt_xi2_out = sim_gt['xi2_out'].values
    gt_u = sim_gt['u_out'].values
    gt_v = sim_gt['v_out'].values
    gt_L = sim_gt['L'].values

    gt_time_like_frac = float(sim_gt['time_exit'].mean())
    pred_time_like_frac = float(np.mean(pred_time_exit))
    print(f"GT time-like fraction:   {gt_time_like_frac:.3f}")
    print(f"Pred time-like fraction: {pred_time_like_frac:.3f}")
    _report_interior_budget_inconsistency(
        pred_L=pred_L,
        pred_budget=budget_vals[:pred_L.shape[0]],
        pred_positions=[pred_xi1_out, pred_xi2_out],
    )

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results_timedep(
        gt_xi1_out, gt_xi2_out, gt_u, gt_v, gt_L,
        pred_xi1_out, pred_xi2_out, pred_u, pred_v, pred_L,
        test_dir, source_type,
    )
    pred_data = {'xi1_out': pred_xi1_out, 'xi2_out': pred_xi2_out, 'u_out': pred_u, 'v_out': pred_v, 'L': pred_L}
    gt_data = {'xi1_out': gt_xi1_out, 'xi2_out': gt_xi2_out, 'u_out': gt_u, 'v_out': gt_v, 'L': gt_L}
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag=f"2d_triangle_timedep_{source_type}")


def run_uniform_inference_2d_quad_timedep(
    model, input_dim, cond_dim, device, test_dir,
    source_type='internal', scheduler_type='condOT', train_sampling='random', infer_sampling='random',
    batch_size=20000, ode_method='rk4', num_steps=16,
    min_width=0.01, max_width=10.0,
    min_height=0.01, max_height=10.0,
    min_path_budget=1e-5, max_path_budget=0.1,
):
    print("\n--- Running Inference (Uniform Conditions, 2D Quadrilateral Time-Dependent) ---")
    print(f"Source Type: 2D Quadrilateral {source_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Path budget range: [{min_path_budget}, {max_path_budget}]")
    print(f"-------------------------------------------------------------------------------")

    _ = min_height
    _ = max_height

    if source_type != 'internal':
        raise NotImplementedError("Boundary source for 2D quad timedep is not supported.")

    norm_params = _load_norm_params_or_abort(
        '2d_quad_timedep',
        source_type,
        train_sampling,
        "2D quad timedep internal",
    )
    if norm_params is None:
        return
    direction_encoding_2d = _resolve_2d_direction_encoding(norm_params)

    cond_features = 10
    target_dim = input_dim - cond_dim
    pool_size = batch_size * 20
    samples_pool = generate_samples(pool_size, cond_features + target_dim, sampling=infer_sampling)

    log_L_pool = np.log10(min_width) + samples_pool[:, 0] * (np.log10(max_width) - np.log10(min_width))
    L_pool = 10**log_L_pool
    x3_pool = L_pool * (samples_pool[:, 1] * 4.0 - 2.0)
    y3_pool = L_pool * (samples_pool[:, 2] * 1.9 + 0.1)
    x4_pool = L_pool * (samples_pool[:, 3] * 4.0 - 2.0)
    y4_pool = L_pool * (samples_pool[:, 4] * 1.9 + 0.1)

    c1 = (x3_pool - L_pool) * (y4_pool - y3_pool) - y3_pool * (x4_pool - x3_pool) > 1e-4 * L_pool**2
    c2 = x3_pool * y4_pool - x4_pool * y3_pool > 1e-4 * L_pool**2
    valid_indices = np.where(c1 & c2)[0][:batch_size]
    if len(valid_indices) == 0:
        print("ERROR: Failed to generate valid quadrilateral samples.")
        return
    if len(valid_indices) < batch_size:
        print(f"Warning: only {len(valid_indices)} valid quads generated for requested batch_size={batch_size}.")

    samples = samples_pool[valid_indices]
    actual_batch_size = len(samples)
    L_quad_vals = L_pool[valid_indices]
    x3_vals = x3_pool[valid_indices]
    y3_vals = y3_pool[valid_indices]
    x4_vals = x4_pool[valid_indices]
    y4_vals = y4_pool[valid_indices]

    budget_vals = 10**(np.log10(min_path_budget) + samples[:, 5] * (np.log10(max_path_budget) - np.log10(min_path_budget)))
    u_in_vals = samples[:, 6]
    v_in_vals = samples[:, 7]
    mu = 2.0 * samples[:, 8] - 1.0
    phi = samples[:, 9] * 2.0 * np.pi
    t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
    dir_u_in_vals = t * np.cos(phi)
    dir_v_in_vals = t * np.sin(phi)

    norm_L = normalize_log_scale_logit(L_quad_vals, norm_params['L_lower'], norm_params['L_upper'])
    x3_01 = (x3_vals - norm_params['x3_lower']) / (norm_params['x3_upper'] - norm_params['x3_lower'] + 1e-12)
    norm_x3 = normalize_linear_scale_logit(np.clip(x3_01, 0.0, 1.0))
    norm_y3 = normalize_log_scale_logit(y3_vals, norm_params['y3_lower'], norm_params['y3_upper'])
    x4_01 = (x4_vals - norm_params['x4_lower']) / (norm_params['x4_upper'] - norm_params['x4_lower'] + 1e-12)
    norm_x4 = normalize_linear_scale_logit(np.clip(x4_01, 0.0, 1.0))
    norm_y4 = normalize_log_scale_logit(y4_vals, norm_params['y4_lower'], norm_params['y4_upper'])
    norm_budget = normalize_log_scale_logit(budget_vals, norm_params['budget_lower'], norm_params['budget_upper'])
    norm_u_in = normalize_position_relative(u_in_vals)
    norm_v_in = normalize_position_relative(v_in_vals)
    phi_in_vals, r_in_vals = uv_to_polar(dir_u_in_vals, dir_v_in_vals)
    cond_np = np.stack(
        [
            norm_L, norm_x3, norm_y3, norm_x4, norm_y4, norm_budget, norm_u_in, norm_v_in,
            normalize_angle_phi(phi_in_vals), normalize_radius_r(r_in_vals),
        ],
        axis=1,
    )
    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
    assert cond.shape[1] == cond_dim, f"Cond tensor dim mismatch: {cond.shape[1]} vs {cond_dim}"
    noise = generate_target_noise(samples, cond_features, target_dim, actual_batch_size, device)
    x_init = torch.cat([cond, noise], dim=1)

    x_final = _run_uniform_ode_inference(
        model,
        x_init,
        device,
        ode_method,
        num_steps,
        batch_size=actual_batch_size,
    )
    pred_logit_u_out = x_final[:, cond_dim + 0]
    pred_logit_v_out = x_final[:, cond_dim + 1]

    pred_u_out = denorm_position_relative(pred_logit_u_out).numpy()
    pred_v_out = denorm_position_relative(pred_logit_v_out).numpy()
    pred_dir_u_t, pred_dir_v_t = decode_direction_2d(
        [x_final[:, cond_dim + 2], x_final[:, cond_dim + 3]],
        direction_encoding_2d,
        norm_params=norm_params,
    )
    pred_dir_u = pred_dir_u_t.cpu().numpy()
    pred_dir_v = pred_dir_v_t.cpu().numpy()
    pred_L = np.maximum(
        denorm_length_target(x_final[:, cond_dim + 4], budget_vals[:x_final.shape[0]], norm_params, key='L_track').cpu().numpy(),
        0.0,
    )
    if x_final.shape[1] > cond_dim + 5:
        pred_time_exit_prob = denorm_linear_scale_logit(x_final[:, cond_dim + 5]).cpu().numpy()
        pred_time_exit = (pred_time_exit_prob >= 0.5).astype(np.float64)
    else:
        raise RuntimeError("Timedep inference requires exit token output, but model output has no exit-token dimension.")

    pred_L, _ = enforce_exit_physics(
        pred_time_exit=pred_time_exit,
        budget=budget_vals[:pred_L.shape[0]],
        L_out=pred_L,
        positions=None,
        snap_space_to_bounds=False,
    )

    print("Generating ground-truth MC samples under the same uniform ranges...")
    sim_gt = safe_get_sim_data(
        get_2d_quad_timedep_internal_source_data,
        num_neutrons=actual_batch_size, scatter_ratio=1.0, distribution=0,
        min_width=min_width, max_width=max_width,
        min_path_budget=min_path_budget, max_path_budget=max_path_budget,
        sampling=infer_sampling,
    )
    sim_gt = sim_gt[(sim_gt["space_exit"] == 1) | (sim_gt["time_exit"] == 1)].copy()
    if len(sim_gt) == 0:
        print("ERROR: No exiting particles in MC GT for comparison.")
        return

    gt_u_out = sim_gt['u_out'].values
    gt_v_out = sim_gt['v_out'].values
    gt_dir_u = sim_gt['dir_u_out'].values
    gt_dir_v = sim_gt['dir_v_out'].values
    gt_L = sim_gt['L_track'].values

    gt_time_like_frac = float(sim_gt['time_exit'].mean())
    pred_time_like_frac = float(np.mean(pred_time_exit))
    print(f"GT time-like fraction:   {gt_time_like_frac:.3f}")
    print(f"Pred time-like fraction: {pred_time_like_frac:.3f}")
    _report_interior_budget_inconsistency(
        pred_L=pred_L,
        pred_budget=budget_vals[:pred_L.shape[0]],
        pred_positions=[pred_u_out, pred_v_out],
    )

    os.makedirs(test_dir, exist_ok=True)
    plot_uniform_results_timedep(
        gt_u_out, gt_v_out, gt_dir_u, gt_dir_v, gt_L,
        pred_u_out, pred_v_out, pred_dir_u, pred_dir_v, pred_L,
        test_dir, source_type,
    )
    pred_data = {'u_out': pred_u_out, 'v_out': pred_v_out, 'dir_u_out': pred_dir_u, 'dir_v_out': pred_dir_v, 'L': pred_L}
    gt_data = {'u_out': gt_u_out, 'v_out': gt_v_out, 'dir_u_out': gt_dir_u, 'dir_v_out': gt_dir_v, 'L': gt_L}
    compute_and_print_metrics(gt_data, pred_data, test_dir, tag=f"2d_quad_timedep_{source_type}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Conditional Flow Matching model (Uniform only)')
    parser.add_argument('--data_dim', type=str, default='2d_timedep',
                        choices=[d for d in DATA_DIM_CHOICES if d != '1d'],
                        help='Dimension of the problem (2d, 2d_timedep, 2d_triangle_timedep, 2d_quad_timedep, 1d_timedep, 3d, or 3d_timedep)')
    parser.add_argument('--source_type', type=str, default='internal', choices=['boundary', 'internal'],
                        help='Source type (boundary or internal)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimensions in the model')
    parser.add_argument('--depth', type=int, default=4, help='Number of flow blocks in the model')
    parser.add_argument('--expansion_factor', type=int, default=1, help='FFN expansion factor inside each flow block')
    parser.add_argument('--scheduler', type=str, default='condOT',
                        choices=['condOT', 'polynomial', 'linearVP', 'cosine'],
                        help='Scheduler used during training')
    parser.add_argument('--batch_size', type=int, default=20000, help='Batch size for testing')
    parser.add_argument('--ode_method', type=str, default='rk4', 
                        choices=['dopri5', 'rk4', 'euler', 'midpoint'], help='ODE solver method')
    parser.add_argument('--num_steps', type=int, default=16, help='Nominal steps for ODE solver')
    parser.add_argument('--min_width', type=float, default=0.01, help='Min width for GT data generation')
    parser.add_argument('--max_width', type=float, default=10.0, help='Max width for GT data generation')
    parser.add_argument('--min_height', type=float, default=0.01, help='Min height for GT data generation')
    parser.add_argument('--max_height', type=float, default=10.0, help='Max height for GT data generation')
    parser.add_argument('--min_depth', type=float, default=0.01, help='Min depth for 3D GT data generation')
    parser.add_argument('--max_depth', type=float, default=10.0, help='Max depth for 3D GT data generation')
    parser.add_argument('--min_path_budget', type=float, default=1e-6,
                        help='Minimum path-length budget ℓ_t (for timedep models)')
    parser.add_argument('--max_path_budget', type=float, default=10.0,
                        help='Maximum path-length budget ℓ_t (for timedep models)')
    parser.add_argument('--angular_sampling', type=int, default=0, choices=[0, 1],
                        help='2D timedep boundary angular sampling: 0=uniform-normal (ML), 1=cosine-law (physical)')
    parser.add_argument('--min_thickness', type=float, default=0.001,
                        help='Min thickness for 1D timedep GT data generation')
    parser.add_argument('--max_thickness', type=float, default=100.0,
                        help='Max thickness for 1D timedep GT data generation')
    parser.add_argument('--train_sampling', type=str, default='random', choices=['random', 'qmc', 'lhs'],
                        help='Sampling strategy used at training time (model + norm file suffix).')
    parser.add_argument('--infer_sampling', type=str, default='random', choices=['random', 'qmc', 'lhs'],
                        help='Sampling strategy used for inference condition/noise generation.')
    parser.add_argument('--no_plot_titles', action='store_true',
                        help='Disable titles on all generated plots.')
    parser.add_argument('--electron_npz', type=str, default=None,
                        help='Path to electron NPZ dataset file for evaluation.')
    return parser.parse_args()

def main():
    args = parse_args()
    set_plot_titles_enabled(not args.no_plot_titles)
    device, test_dir = setup_test_environment(
        args.source_type, args.scheduler, args.data_dim,
        args.infer_sampling, args.train_sampling
    )
    model, input_dim, cond_dim = load_model(
        device, args.hidden_dim, args.depth, args.expansion_factor,
        args.source_type, args.data_dim, args.train_sampling, args.scheduler,
    )

    if args.data_dim == '1d_timedep':
        run_uniform_inference_1d_timedep(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_thickness=args.min_thickness, max_thickness=args.max_thickness,
            min_path_budget=args.min_path_budget, max_path_budget=args.max_path_budget
        )
    elif args.data_dim == '2d_electron':
        run_uniform_inference_2d_electron(
            model, input_dim, cond_dim, device, test_dir,
            electron_npz=args.electron_npz,
            train_sampling=args.train_sampling, infer_sampling=args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
        )
    elif args.data_dim == '2d_timedep':
        run_uniform_inference_timedep(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.scheduler, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height,
            min_path_budget=args.min_path_budget, max_path_budget=args.max_path_budget,
            angular_sampling=args.angular_sampling,
        )
    elif args.data_dim == '2d_triangle_timedep':
        run_uniform_inference_2d_triangle_timedep(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.scheduler, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height,
            min_path_budget=args.min_path_budget, max_path_budget=args.max_path_budget,
            angular_sampling=args.angular_sampling,
        )
    elif args.data_dim == '2d_quad_timedep':
        run_uniform_inference_2d_quad_timedep(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.scheduler, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height,
            min_path_budget=args.min_path_budget, max_path_budget=args.max_path_budget,
        )
    elif args.data_dim == '3d':
        run_uniform_inference_3d(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height,
            min_depth=args.min_depth, max_depth=args.max_depth
        )
    elif args.data_dim == '3d_timedep':
        run_uniform_inference_3d_timedep(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height,
            min_depth=args.min_depth, max_depth=args.max_depth,
            min_path_budget=args.min_path_budget, max_path_budget=args.max_path_budget,
        )
    else:
        run_uniform_inference(
            model, input_dim, cond_dim, device, test_dir,
            args.source_type, args.scheduler, args.train_sampling, args.infer_sampling,
            batch_size=args.batch_size, ode_method=args.ode_method, num_steps=args.num_steps,
            min_width=args.min_width, max_width=args.max_width,
            min_height=args.min_height, max_height=args.max_height
        )

if __name__ == "__main__":
    main()
