# validation/Macro_field_diagnostic.py

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.utils import (
    setup_environment,
    load_model,
    run_uniform_ode_inference,
    load_norm_params_for_data_dim,
)
from ml.transform import (
    normalize_position_relative,
    normalize_log_scale_probit,
    normalize_angle_phi,
    normalize_radius_r,
    denorm_position_relative,
    denorm_linear_scale_probit,
    denorm_angle_phi,
    denorm_radius_r,
    uv_to_polar,
    polar_to_uv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("DiagnosticSuite")


# ==============================================================================
# Helper Functions for Canonical Transformations (Used in Test 3)
# ==============================================================================
def transform_to_canonical(x_local, y_local, u, v, face_idx):
    y_c, u_c, v_c = np.zeros_like(y_local), np.zeros_like(u), np.zeros_like(v)
    m0, m1, m2, m3 = (face_idx == 0), (face_idx == 1), (face_idx == 2), (face_idx == 3)

    y_c[m0], u_c[m0], v_c[m0] = y_local[m0], u[m0], v[m0]
    y_c[m1], u_c[m1], v_c[m1] = y_local[m1], -u[m1], v[m1]
    y_c[m2], u_c[m2], v_c[m2] = x_local[m2], v[m2], u[m2]
    y_c[m3], u_c[m3], v_c[m3] = x_local[m3], -v[m3], u[m3]

    return np.zeros_like(x_local), y_c, u_c, v_c


def transform_from_canonical(x_out_c, y_out_c, u_out_c, v_out_c, face_idx):
    x_out, y_out, u_out, v_out = (
        np.zeros_like(x_out_c),
        np.zeros_like(y_out_c),
        np.zeros_like(u_out_c),
        np.zeros_like(v_out_c),
    )
    m0, m1, m2, m3 = (face_idx == 0), (face_idx == 1), (face_idx == 2), (face_idx == 3)

    x_out[m0], y_out[m0], u_out[m0], v_out[m0] = x_out_c[m0], y_out_c[m0], u_out_c[m0], v_out_c[m0]
    x_out[m1], y_out[m1], u_out[m1], v_out[m1] = 1.0 - x_out_c[m1], y_out_c[m1], -u_out_c[m1], v_out_c[m1]
    x_out[m2], y_out[m2], u_out[m2], v_out[m2] = y_out_c[m2], x_out_c[m2], v_out_c[m2], u_out_c[m2]
    x_out[m3], y_out[m3], u_out[m3], v_out[m3] = y_out_c[m3], 1.0 - x_out_c[m3], v_out_c[m3], -u_out_c[m3]

    return x_out, y_out, u_out, v_out


# ==============================================================================
# TEST 1: Single-Step Entrance Position Sweep
# ==============================================================================
def run_test1_ysweep(model, norm_params, E_fixed=5.0, num_points=200, ode_steps=10, device="cpu", output_file="diag_test1_ysweep.png"):
    print("\n" + "=" * 70)
    print(" [TEST 1] Single-Step Entrance Position y_in Sweep ")
    print("=" * 70)
    print(f" -> Grid resolution: {num_points} evaluation points")
    print(f" -> Fixed Incident Energy: {E_fixed} MeV")
    print(f" -> ODE steps: {ode_steps}")

    model.eval()

    E_in_lower_norm = float(norm_params["E_in_lower"])
    E_in_upper_norm = float(norm_params["E_in_upper"])
    E_cutoff = float(norm_params.get("E_cutoff", 0.1))

    cond_dim = getattr(model, "cond_dim", 5)
    input_dim = getattr(model, "input_dim", 12)
    target_dim = input_dim - cond_dim

    print(" -> Constructing uniform y_in sweep batch...")
    y_sweep = np.linspace(0.001, 0.999, num_points, dtype=np.float32)

    u_fixed = np.ones(num_points, dtype=np.float32)
    v_fixed = np.zeros(num_points, dtype=np.float32)
    x_in_fixed = np.zeros(num_points, dtype=np.float32)

    phi_in, r_in = uv_to_polar(u_fixed, v_fixed)

    norm_y_in = normalize_position_relative(y_sweep)
    norm_phi_in = normalize_angle_phi(phi_in)
    norm_r_in = normalize_radius_r(r_in)
    norm_E_in = normalize_log_scale_probit(np.full(num_points, E_fixed, dtype=np.float32), E_in_lower_norm, E_in_upper_norm)
    norm_x_in = normalize_position_relative(x_in_fixed)

    cond_np = np.stack([norm_y_in, norm_phi_in, norm_r_in, norm_E_in, norm_x_in], axis=1)
    cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=device)

    torch.manual_seed(42)
    noise_tensor = torch.randn((num_points, target_dim), dtype=torch.float32, device=device)
    x_init = torch.cat([cond_tensor, noise_tensor], dim=1)

    print(" -> Running model ODE inference batch...")
    x_final = run_uniform_ode_inference(
        model, x_init, device, ode_method="euler", num_steps=ode_steps, batch_size=num_points
    )

    pred_f_out = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 4]).cpu().numpy(), 0.0, 1.0)
    pred_E_out = np.maximum(pred_f_out * (E_fixed - E_cutoff) + E_cutoff, E_cutoff)
    pred_E_dep = np.maximum(0.0, E_fixed - pred_E_out)

    print(f" -> Mean predicted energy loss ΔE: {np.mean(pred_E_dep):.4f} MeV")
    print(f" -> Max ΔE (center/peak): {np.max(pred_E_dep):.4f} MeV | Min ΔE (edges): {np.min(pred_E_dep):.4f} MeV")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(y_sweep, pred_E_dep, color="crimson", linewidth=2)
    ax1.set_xlabel("Entrance Position $y_{\\text{in}}$ (local cell coordinate)", fontsize=11)
    ax1.set_ylabel("Predicted Energy Deposition $\\Delta E$ (MeV)", fontsize=11)
    ax1.set_title(f"Predicted $\\Delta E$ vs $y_{{\\text{{in}}}}$ ($E_{{\\text{{in}}}} = {E_fixed}\\text{{ MeV}}$)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(y_sweep, pred_f_out, color="navy", linewidth=2)
    ax2.set_xlabel("Entrance Position $y_{\\text{in}}$ (local cell coordinate)", fontsize=11)
    ax2.set_ylabel("Predicted Residual Energy Fraction $f_{\\text{out}}$", fontsize=11)
    ax2.set_title("Predicted $f_{\\text{out}}$ vs $y_{\\text{in}}$", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f" [TEST 1 COMPLETED] Saved output plot to: {output_file}\n")


# ==============================================================================
# TEST 2: Latent Coordinate Probit Gradient Audit
# ==============================================================================
def run_test2_probit_audit(num_points=1000, output_file="diag_test2_probit_audit.png"):
    print("\n" + "=" * 70)
    print(" [TEST 2] Probit Coordinate Transformation & Slope Audit ")
    print("=" * 70)
    print(f" -> Evaluating grid resolution: {num_points} points across y ∈ [0.0001, 0.9999]")

    y_grid = np.linspace(0.0001, 0.9999, num_points)
    dy = y_grid[1] - y_grid[0]

    print(" -> Computing probit inverse CDF mappings and numerical gradients...")
    y_probit = normalize_position_relative(y_grid)
    d_probit_dy = np.gradient(y_probit, dy)

    y_linear = 2.0 * y_grid - 1.0
    d_linear_dy = np.full_like(y_grid, 2.0)

    center_idx = num_points // 2
    edge_idx = 10

    print(f" -> Slope at center (y = 0.5): {d_probit_dy[center_idx]:.3f}")
    print(f" -> Slope near boundary tail (y ≈ 0.01): {d_probit_dy[edge_idx]:.3f} (Slope expansion ratio: {d_probit_dy[edge_idx]/d_probit_dy[center_idx]:.1f}x)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(y_grid, y_probit, label="Probit Mapping $\\Phi^{-1}(y)$", color="darkred", linewidth=2)
    ax1.plot(y_grid, y_linear, label="Linear Mapping $2y - 1$", color="teal", linestyle="--", linewidth=2)
    ax1.axvspan(0.00, 0.05, color="gray", alpha=0.2, label="Boundary Latent Tails")
    ax1.axvspan(0.95, 1.00, color="gray", alpha=0.2)
    ax1.set_xlabel("Physical Local Coordinate $y \\in [0, 1]$", fontsize=11)
    ax1.set_ylabel("Normalized Latent Coordinate $y_{\\text{norm}}$", fontsize=11)
    ax1.set_title("Coordinate Transformation Comparison", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(y_grid, d_probit_dy, label="$d(\\text{Probit})/dy$", color="darkred", linewidth=2)
    ax2.plot(y_grid, d_linear_dy, label="$d(\\text{Linear})/dy$", color="teal", linestyle="--", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Physical Local Coordinate $y \\in [0, 1]$", fontsize=11)
    ax2.set_ylabel("Slope / Transformation Gradient (log scale)", fontsize=11)
    ax2.set_title("Transformation Gradient $d(y_{\\text{norm}})/dy$", fontsize=12)
    ax2.legend(loc="upper center")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f" [TEST 2 COMPLETED] Saved output plot to: {output_file}\n")


# ==============================================================================
# TEST 3: Boundary Step Energy Accumulation Isolation
# ==============================================================================
def _simulate_chained_macro(
    model, norm_params, suppress_escaping_dose=False, num_particles=20000, ode_steps=10, max_macro_steps=50, device="cpu"
):
    mode_str = "Boundary-Suppressed" if suppress_escaping_dose else "Standard Full-Step"
    print(f"   --> Launching Macro-Chaining Simulation [{mode_str}] | Particles: {num_particles}")

    model.eval()
    num_cells_x, num_cells_y = 10, 10
    cell_dx, cell_dy = 1.0, 1.0

    E_in_lower_norm = float(norm_params["E_in_lower"])
    E_in_upper_norm = float(norm_params["E_in_upper"])
    E_cutoff = float(norm_params.get("E_cutoff", 0.1))

    cond_dim = getattr(model, "cond_dim", 5)
    input_dim = getattr(model, "input_dim", 12)
    target_dim = input_dim - cond_dim

    pos_cell_x = np.zeros(num_particles, dtype=int)
    pos_y_phys = np.random.uniform(0.0, num_cells_y * cell_dy, size=num_particles).astype(np.float32)
    pos_cell_y = np.clip(np.floor(pos_y_phys / cell_dy).astype(int), 0, num_cells_y - 1)

    x_local = np.zeros(num_particles, dtype=np.float32)
    y_local = np.clip((pos_y_phys - pos_cell_y * cell_dy) / cell_dy, 0.0, 1.0)
    face_idx = np.zeros(num_particles, dtype=int)

    angles_rad = np.random.uniform(-np.radians(85.0), np.radians(85.0), size=num_particles).astype(np.float32)
    u_curr, v_curr = np.cos(angles_rad), np.sin(angles_rad)
    E_curr = np.random.uniform(0.1, 10.0, size=num_particles).astype(np.float32)

    active = np.ones(num_particles, dtype=bool)
    dose_grid = np.zeros((num_cells_x, num_cells_y), dtype=np.float64)

    for step_idx in range(max_macro_steps):
        act_indices = np.where(active)[0]
        num_act = len(act_indices)
        if num_act == 0:
            print(f"      Step {step_idx:02d}: All particles extinguished/escaped.")
            break

        if step_idx % 10 == 0 or step_idx == max_macro_steps - 1:
            print(f"      Step {step_idx:02d}: Active particles = {num_act} / {num_particles}")

        x_c, y_c, u_c, v_c = transform_to_canonical(
            x_local[act_indices], y_local[act_indices], u_curr[act_indices], v_curr[act_indices], face_idx[act_indices]
        )
        phi_in_c, r_in_c = uv_to_polar(u_c, v_c)

        cond_np = np.stack(
            [
                normalize_position_relative(y_c),
                normalize_angle_phi(phi_in_c),
                normalize_radius_r(r_in_c),
                normalize_log_scale_probit(E_curr[act_indices], E_in_lower_norm, E_in_upper_norm),
                normalize_position_relative(x_c),
            ],
            axis=1,
        )
        cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=device)

        torch.manual_seed(42 + step_idx)
        noise_tensor = torch.randn((len(act_indices), target_dim), dtype=torch.float32, device=device)
        x_init = torch.cat([cond_tensor, noise_tensor], dim=1)

        x_final = run_uniform_ode_inference(model, x_init, device, ode_method="euler", num_steps=ode_steps, batch_size=256)

        pred_x_out_c = denorm_position_relative(x_final[:, cond_dim + 0]).cpu().numpy()
        pred_y_out_c = denorm_position_relative(x_final[:, cond_dim + 1]).cpu().numpy()

        pred_phi_out_c = denorm_angle_phi(x_final[:, cond_dim + 2]).cpu().numpy()
        pred_r_out_c = denorm_radius_r(x_final[:, cond_dim + 3]).cpu().numpy()
        pred_u_out_c, pred_v_out_c = polar_to_uv(pred_phi_out_c, pred_r_out_c)

        pred_f_out = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 4]).cpu().numpy(), 0.0, 1.0)
        pred_E_out = np.maximum(pred_f_out * (E_curr[act_indices] - E_cutoff) + E_cutoff, E_cutoff)
        pred_energy_dep = np.maximum(0.0, E_curr[act_indices] - pred_E_out)

        x_out_phys, y_out_phys, u_out_phys, v_out_phys = transform_from_canonical(
            pred_x_out_c, pred_y_out_c, pred_u_out_c, pred_v_out_c, face_idx[act_indices]
        )

        d_left, d_right = np.abs(x_out_phys), np.abs(1.0 - x_out_phys)
        d_bottom, d_top = np.abs(y_out_phys), np.abs(1.0 - y_out_phys)
        min_face = np.argmin(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1)

        next_x, next_y = pos_cell_x[act_indices].copy(), pos_cell_y[act_indices].copy()
        next_x[min_face == 0] -= 1
        next_x[min_face == 1] += 1
        next_y[min_face == 2] -= 1
        next_y[min_face == 3] += 1

        out_of_bounds = (next_x < 0) | (next_x >= num_cells_x) | (next_y < 0) | (next_y >= num_cells_y)

        if suppress_escaping_dose:
            valid = ~out_of_bounds
            np.add.at(dose_grid, (pos_cell_x[act_indices[valid]], pos_cell_y[act_indices[valid]]), pred_energy_dep[valid])
        else:
            np.add.at(dose_grid, (pos_cell_x[act_indices], pos_cell_y[act_indices]), pred_energy_dep)

        pos_cell_x[act_indices], pos_cell_y[act_indices] = next_x, next_y
        u_curr[act_indices], v_curr[act_indices] = u_out_phys, v_out_phys
        E_curr[act_indices] = pred_E_out

        depleted = E_curr[act_indices] <= (E_cutoff + 1e-4)
        active[act_indices[out_of_bounds | depleted]] = False

    return dose_grid / (num_particles * cell_dx * cell_dy)


def run_test3_boundary_tally(model, norm_params, num_particles=30000, ode_steps=10, device="cpu", output_file="diag_test3_boundary_tally.png"):
    print("\n" + "=" * 70)
    print(" [TEST 3] Macro-Step Boundary Energy Accumulation Isolation ")
    print("=" * 70)

    print(" -> Phase 1/2: Simulating with Standard Full-Step Dose Tallying...")
    dose_std = _simulate_chained_macro(
        model, norm_params, suppress_escaping_dose=False, num_particles=num_particles, ode_steps=ode_steps, device=device
    )

    print("\n -> Phase 2/2: Simulating with Boundary-Suppressed Dose Tallying...")
    dose_supp = _simulate_chained_macro(
        model, norm_params, suppress_escaping_dose=True, num_particles=num_particles, ode_steps=ode_steps, device=device
    )

    print(" -> Generating diagnostic spatial dose heatmaps & line profiles...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmax = max(np.max(dose_std), np.max(dose_supp))

    im0 = axes[0].imshow(dose_std.T, origin="lower", extent=[0, 0.1, 0, 0.1], cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Standard Tally (Full Step ΔE)", fontsize=11)
    axes[0].set_xlabel("X Position (cm)")
    axes[0].set_ylabel("Y Position (cm)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(dose_supp.T, origin="lower", extent=[0, 0.1, 0, 0.1], cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("Boundary-Suppressed Tally", fontsize=11)
    axes[1].set_xlabel("X Position (cm)")
    fig.colorbar(im1, ax=axes[1])

    y_centers = np.linspace(0.005, 0.095, 10)
    axes[2].plot(y_centers, dose_std[0, :], label="Standard Tally", color="crimson", marker="o")
    axes[2].plot(y_centers, dose_supp[0, :], label="Boundary-Suppressed", color="navy", marker="s", linestyle="--")
    axes[2].set_xlabel("Y Position at Entrance Column X=0 (cm)")
    axes[2].set_ylabel("Dose Density (MeV / cm² / e⁻)")
    axes[2].set_title("Entrance Column Profile Comparison", fontsize=11)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f" [TEST 3 COMPLETED] Saved output plot to: {output_file}\n")


# ==============================================================================
# MAIN DRIVER & TOGGLE SWITCH
# ==============================================================================
def main():
    TEST_MODE = "all"

    parser = argparse.ArgumentParser(description="Unified Diagnostic Suite for Central Peak Discrepancy")
    parser.add_argument("--test", type=str, default=str(TEST_MODE), choices=["1", "2", "3", "all"], help="Test mode to execute")
    parser.add_argument("--checkpoint", type=str, default="ml/models/2d_electron_boundary/best_model.pth", help="Path to checkpoint file")
    parser.add_argument("--train_sampling", type=str, default="random", help="Sampling strategy (e.g., 'random', 'uniform')")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--depth", type=int, default=8, help="Model depth / number of blocks")
    parser.add_argument("--E_fixed", type=float, default=5.0, help="Fixed incident energy for Test 1 (MeV)")
    parser.add_argument("--num_particles", type=int, default=20000, help="Particle count for Test 3")
    parser.add_argument("--ode_steps", type=int, default=10, help="ODE solver integration steps")
    args = parser.parse_args()

    mode = args.test.lower()

    print("=" * 70)
    print(" GMC MACRO-FIELD DIAGNOSTIC SUITE INITIALIZED ")
    print("=" * 70)
    print(f" Running Mode        : {mode.upper()}")
    print(f" Checkpoint Path     : {args.checkpoint}")
    print(f" Training Sampling   : {args.train_sampling}")

    device = setup_environment(seed=42, allow_mps=False, inference=True)
    print(f" Compute Device      : {device}")

    print(" Loading normalization parameters...")
    norm_params, _ = load_norm_params_for_data_dim("2d_electron", "boundary", args.train_sampling)

    def load_model_instance():
        print(" Loading neural network architecture & weights...")
        model, _, _ = load_model(
            device,
            args.hidden_dim,
            args.depth,
            source_type="boundary",
            data_dim="2d_electron",
            train_sampling=args.train_sampling,
            scheduler_type="condOT",
        )
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f" Successfully loaded weights from: {args.checkpoint}")
        else:
            print(f" WARNING: Checkpoint '{args.checkpoint}' not found! Running with uninitialized weights.")
        model.to(device)
        return model

    if mode in ["1", "all"]:
        model = load_model_instance()
        run_test1_ysweep(
            model,
            norm_params,
            E_fixed=args.E_fixed,
            ode_steps=args.ode_steps,
            device=device,
            output_file="diag_test1_ysweep.png",
        )

    if mode in ["2", "all"]:
        run_test2_probit_audit(
            output_file="diag_test2_probit_audit.png"
        )

    if mode in ["3", "all"]:
        model = load_model_instance()
        run_test3_boundary_tally(
            model,
            norm_params,
            num_particles=args.num_particles,
            ode_steps=args.ode_steps,
            device=device,
            output_file="diag_test3_boundary_tally.png",
        )

    print("=" * 70)
    print(" ALL REQUESTED DIAGNOSTIC TESTS COMPLETED SUCCESSFULLY ")
    print("=" * 70)


if __name__ == "__main__":
    main()