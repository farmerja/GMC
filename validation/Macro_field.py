# validation/macro_chaining_plot.py

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def transform_to_canonical(x_local, y_local, u, v, face_idx):
    """
    Map cell entry states on any face to canonical Left-Face frame (x'_in = 0).
    face_idx: 0=Left (x=0), 1=Right (x=1), 2=Bottom (y=0), 3=Top (y=1)
    """
    y_c = np.zeros_like(y_local)
    u_c = np.zeros_like(u)
    v_c = np.zeros_like(v)

    # Face 0: Left face
    m0 = (face_idx == 0)
    y_c[m0] = y_local[m0]
    u_c[m0] = u[m0]
    v_c[m0] = v[m0]

    # Face 1: Right face
    m1 = (face_idx == 1)
    y_c[m1] = y_local[m1]
    u_c[m1] = -u[m1]
    v_c[m1] = v[m1]

    # Face 2: Bottom face
    m2 = (face_idx == 2)
    y_c[m2] = x_local[m2]
    u_c[m2] = v[m2]
    v_c[m2] = u[m2]

    # Face 3: Top face
    m3 = (face_idx == 3)
    y_c[m3] = x_local[m3]
    u_c[m3] = -v[m3]
    v_c[m3] = u[m3]

    x_c = np.zeros_like(x_local)
    return x_c, y_c, u_c, v_c


def transform_from_canonical(x_out_c, y_out_c, u_out_c, v_out_c, face_idx):
    """
    Invert predictions from canonical Left-Face frame back to physical cell space.
    """
    x_out = np.zeros_like(x_out_c)
    y_out = np.zeros_like(y_out_c)
    u_out = np.zeros_like(u_out_c)
    v_out = np.zeros_like(v_out_c)

    m0 = (face_idx == 0)
    x_out[m0] = x_out_c[m0]
    y_out[m0] = y_out_c[m0]
    u_out[m0] = u_out_c[m0]
    v_out[m0] = v_out_c[m0]

    m1 = (face_idx == 1)
    x_out[m1] = 1.0 - x_out_c[m1]
    y_out[m1] = y_out_c[m1]
    u_out[m1] = -u_out_c[m1]
    v_out[m1] = v_out_c[m1]

    m2 = (face_idx == 2)
    x_out[m2] = y_out_c[m2]
    y_out[m2] = x_out_c[m2]
    u_out[m2] = v_out_c[m2]
    v_out[m2] = u_out_c[m2]

    m3 = (face_idx == 3)
    x_out[m3] = y_out_c[m3]
    y_out[m3] = 1.0 - x_out_c[m3]
    u_out[m3] = v_out_c[m3]
    v_out[m3] = -u_out_c[m3]

    return x_out, y_out, u_out, v_out


def simulate_macrofield_block(
        model,
        norm_params,
        num_particles=50000,
        E_in_low=0.1,
        E_in_high=10.0,
        ode_steps=10,
        ode_method="euler",
        batch_size=256,
        num_cells_x=10,
        num_cells_y=10,
        cell_size=1.0,
        max_macro_steps=100,
        max_incoming_angle_deg=85.0,
        device="cpu",
):
    model.eval()

    cell_dx = cell_size
    cell_dy = cell_size

    x_min, x_max = 0.0, num_cells_x * cell_dx
    y_min, y_max = 0.0, num_cells_y * cell_dy

    E_in_lower_norm = float(norm_params["E_in_lower"])
    E_in_upper_norm = float(norm_params["E_in_upper"])
    E_cutoff = float(norm_params.get("E_cutoff", 0.1))

    log.info(
        f"Simulation energy spectrum: Uniform[{E_in_low:.2f}, {E_in_high:.2f}] MeV | "
        f"Energy cutoff: {E_cutoff:.4f} MeV"
    )
    log.info(f"Target Geometry: {num_cells_x}x{num_cells_y} grid of {cell_size:.1f} cm x {cell_size:.1f} cm cells")

    cond_dim = getattr(model, "cond_dim", 5)
    input_dim = getattr(model, "input_dim", 12)
    target_dim = input_dim - cond_dim

    # Discrete integer grid indices
    pos_cell_x = np.zeros(num_particles, dtype=int)
    pos_y_phys = np.random.uniform(y_min, y_max, size=num_particles).astype(np.float32)
    pos_cell_y = np.clip(np.floor(pos_y_phys / cell_dy).astype(int), 0, num_cells_y - 1)

    # Entry local coordinates on Left face (x=0)
    x_local = np.zeros(num_particles, dtype=np.float32)
    y_local = np.clip((pos_y_phys - pos_cell_y * cell_dy) / cell_dy, 0.0, 1.0)
    face_idx = np.zeros(num_particles, dtype=int)  # 0 = Left face

    # Sample incoming angle theta in range [-max_angle, +max_angle]
    angles_rad = np.random.uniform(
        -np.radians(max_incoming_angle_deg),
        np.radians(max_incoming_angle_deg),
        size=num_particles,
    ).astype(np.float32)

    u_curr = np.cos(angles_rad)
    v_curr = np.sin(angles_rad)

    # Sample primary particle energies uniformly across [E_in_low, E_in_high]
    E_curr = np.random.uniform(E_in_low, E_in_high, size=num_particles).astype(np.float32)

    active = np.ones(num_particles, dtype=bool)
    dose_grid = np.zeros((num_cells_x, num_cells_y), dtype=np.float64)

    for step_idx in range(max_macro_steps):
        act_indices = np.where(active)[0]
        num_active = len(act_indices)
        if num_active == 0:
            log.info(f"All particles terminated at macro step {step_idx}.")
            break

        # 1. Transform entry state to canonical Left-Face frame (x'_in = 0)
        x_c, y_c, u_c, v_c = transform_to_canonical(
            x_local[act_indices], y_local[act_indices], u_curr[act_indices], v_curr[act_indices], face_idx[act_indices]
        )

        phi_in_c, r_in_c = uv_to_polar(u_c, v_c)

        norm_y_in = normalize_position_relative(y_c)
        norm_phi_in = normalize_angle_phi(phi_in_c)
        norm_r_in = normalize_radius_r(r_in_c)
        norm_E_in = normalize_log_scale_probit(E_curr[act_indices], E_in_lower_norm, E_in_upper_norm)
        norm_x_in = normalize_position_relative(x_c)

        cond_np = np.stack([norm_y_in, norm_phi_in, norm_r_in, norm_E_in, norm_x_in], axis=1)
        cond_tensor = torch.tensor(cond_np, dtype=torch.float32, device=device)

        torch.manual_seed(42 + step_idx)
        noise_tensor = torch.randn((num_active, target_dim), dtype=torch.float32, device=device)
        x_init = torch.cat([cond_tensor, noise_tensor], dim=1)

        # 2. ODE Model Inference in canonical frame
        x_final = run_uniform_ode_inference(
            model, x_init, device, ode_method=ode_method, num_steps=ode_steps, batch_size=batch_size
        )

        pred_x_out_c = denorm_position_relative(x_final[:, cond_dim + 0]).cpu().numpy()
        pred_y_out_c = denorm_position_relative(x_final[:, cond_dim + 1]).cpu().numpy()

        pred_phi_out_c = denorm_angle_phi(x_final[:, cond_dim + 2]).cpu().numpy()
        pred_r_out_c = denorm_radius_r(x_final[:, cond_dim + 3]).cpu().numpy()
        pred_u_out_c, pred_v_out_c = polar_to_uv(pred_phi_out_c, pred_r_out_c)

        pred_f_out = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 4]).cpu().numpy(), 0.0, 1.0)
        pred_E_out = np.maximum(pred_f_out * (E_curr[act_indices] - E_cutoff) + E_cutoff, E_cutoff)
        pred_energy_dep = np.maximum(0.0, E_curr[act_indices] - pred_E_out)

        # 3. Transform predictions back from Canonical frame to physical cell space
        x_out_phys, y_out_phys, u_out_phys, v_out_phys = transform_from_canonical(
            pred_x_out_c, pred_y_out_c, pred_u_out_c, pred_v_out_c, face_idx[act_indices]
        )

        # Direct cell dose accumulation
        np.add.at(dose_grid, (pos_cell_x[act_indices], pos_cell_y[act_indices]), pred_energy_dep)

        # 4. Determine exit face boundary of current cell [0, 1] x [0, 1]
        d_left = np.abs(x_out_phys)
        d_right = np.abs(1.0 - x_out_phys)
        d_bottom = np.abs(y_out_phys)
        d_top = np.abs(1.0 - y_out_phys)

        min_face = np.argmin(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1)

        m_left = (min_face == 0)
        m_right = (min_face == 1)
        m_bottom = (min_face == 2)
        m_top = (min_face == 3)

        pos_cell_x[act_indices[m_left]] -= 1
        face_idx[act_indices[m_left]] = 1
        x_local[act_indices[m_left]] = 1.0
        y_local[act_indices[m_left]] = np.clip(y_out_phys[m_left], 0.0, 1.0)

        pos_cell_x[act_indices[m_right]] += 1
        face_idx[act_indices[m_right]] = 0
        x_local[act_indices[m_right]] = 0.0
        y_local[act_indices[m_right]] = np.clip(y_out_phys[m_right], 0.0, 1.0)

        pos_cell_y[act_indices[m_bottom]] -= 1
        face_idx[act_indices[m_bottom]] = 3
        x_local[act_indices[m_bottom]] = np.clip(x_out_phys[m_bottom], 0.0, 1.0)
        y_local[act_indices[m_bottom]] = 1.0

        pos_cell_y[act_indices[m_top]] += 1
        face_idx[act_indices[m_top]] = 2
        x_local[act_indices[m_top]] = np.clip(x_out_phys[m_top], 0.0, 1.0)
        y_local[act_indices[m_top]] = 0.0

        u_curr[act_indices] = u_out_phys
        v_curr[act_indices] = v_out_phys
        E_curr[act_indices] = pred_E_out

        out_of_bounds = (
            (pos_cell_x[act_indices] < 0) | (pos_cell_x[act_indices] >= num_cells_x) |
            (pos_cell_y[act_indices] < 0) | (pos_cell_y[act_indices] >= num_cells_y)
        )
        depleted = E_curr[act_indices] <= (E_cutoff + 1e-4)

        active[act_indices[out_of_bounds | depleted]] = False

        log.info(
            f"Step {step_idx:2d}: {num_active:5d} active | "
            f"Mean E_out: {np.mean(pred_E_out):.2f} MeV | "
            f"Mean ΔE: {np.mean(pred_energy_dep):.4f} MeV"
        )

    micro_cell_area = cell_dx * cell_dy
    dose_density = dose_grid / (num_particles * micro_cell_area)

    return dose_density, num_cells_x, num_cells_y, cell_dx, cell_dy


def main():
    parser = argparse.ArgumentParser(description="GMC Macro-Chaining 10x10 Block Spatial Dose Plot")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--num_particles", type=int, default=50000)
    parser.add_argument("--E_in_low", type=float, default=0.1, help="Minimum energy in MeV")
    parser.add_argument("--E_in_high", type=float, default=10.0, help="Maximum energy in MeV")
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--ode_method", type=str, default="euler")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--expansion_factor", type=int, default=1)
    parser.add_argument("--num_cells_x", type=int, default=10)
    parser.add_argument("--num_cells_y", type=int, default=10)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--max_macro_steps", type=int, default=100)
    parser.add_argument("--max_incoming_angle", type=float, default=85.0)
    parser.add_argument("--train_sampling", type=str, default="random")
    parser.add_argument("--scheduler_type", type=str, default="condOT")
    parser.add_argument("--output", type=str, default="spatial_dose_density.png")
    args = parser.parse_args()

    device = setup_environment(seed=42, allow_mps=False, inference=True)

    norm_params, norm_path = load_norm_params_for_data_dim("2d_electron", "boundary", train_sampling=args.train_sampling)
    log.info(f"Loaded normalization parameters from {norm_path}")

    model, input_dim, cond_dim = load_model(
        device,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        expansion_factor=args.expansion_factor,
        source_type="boundary",
        data_dim="2d_electron",
        train_sampling=args.train_sampling,
        scheduler_type=args.scheduler_type,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        log.info(f"Loading checkpoint weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.to(device)

    dose_density, nx, ny, cell_dx, cell_dy = simulate_macrofield_block(
        model=model,
        norm_params=norm_params,
        num_particles=args.num_particles,
        E_in_low=args.E_in_low,
        E_in_high=args.E_in_high,
        ode_steps=args.ode_steps,
        ode_method=args.ode_method,
        batch_size=args.batch_size,
        num_cells_x=args.num_cells_x,
        num_cells_y=args.num_cells_y,
        cell_size=args.cell_size,
        max_macro_steps=args.max_macro_steps,
        max_incoming_angle_deg=args.max_incoming_angle,
        device=device,
    )

    fig, ax_map = plt.subplots(figsize=(8, 7))

    v_max = max(2.2, np.max(dose_density)) if len(dose_density) > 0 else 2.2
    x_edges = np.linspace(0.0, nx * cell_dx, nx + 1)
    y_edges = np.linspace(0.0, ny * cell_dy, ny + 1)

    im = ax_map.imshow(
        dose_density.T,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        cmap="hot",
        vmin=0.0,
        vmax=300,
    )

    ax_map.set_title("GMC Spatial Dose Density (10 x 10)", fontsize=14, pad=12)
    ax_map.set_xlabel("X Position (cm)", fontsize=12)
    ax_map.set_ylabel("Y Position (cm)", fontsize=12)
    ax_map.set_xticks(x_edges)
    ax_map.set_yticks(y_edges)
    ax_map.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax_map, orientation="vertical", pad=0.03, fraction=0.046)
    cbar.set_label("Spatial Dose Density (MeV / cm² / electron)", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    log.info(f"Plot saved successfully to {args.output}")
    # Export variables needed for combined plotting
    np.savez(
        "code1_dose.npz",  # Change to "code2_dose.npz" in your second script
        dose_density=dose_density,
        nx=nx,
        ny=ny,
        cell_dx=cell_dx,
        cell_dy=cell_dy,
    )
    print("Exported dose data successfully.")


if __name__ == "__main__":
    main()