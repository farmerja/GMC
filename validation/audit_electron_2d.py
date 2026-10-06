# validation/audit_electron_2d.py

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.utils import (
    setup_environment,
    load_model,
    run_uniform_ode_inference,
    load_norm_params_for_data_dim,
)
# Surrounding lines in audit_electron_2d_2.py (Imports)
from ml.transform import (
    normalize_position_relative,
    normalize_angle_phi,
    normalize_radius_r,
    normalize_log_scale_probit,
    denorm_linear_scale_probit,
    denorm_log_scale_probit,
    denorm_position_relative,
    uv_to_polar,
)
from physics.electron_data import load_electron_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_conditional_diversity_check(model, norm_params, device, target_dim, cond_dim, out_dir):
    """Verifies model generates stochastic spread for a single fixed beam condition."""
    log.info("Running Conditional Diversity / Mode Collapse Check...")

    # Define 1 single fixed input beam: E_in = 30 MeV, centered entry, forward direction
    E_in_lower, E_in_upper = norm_params["E_in_lower"], norm_params["E_in_upper"]

    phi_in, r_in = uv_to_polar(1.0, 0.0)
    norm_y_in = normalize_position_relative(0.5)
    norm_phi_in = normalize_angle_phi(phi_in)
    norm_r_in = normalize_radius_r(r_in)
    norm_E_in = normalize_log_scale_probit(30.0, E_in_lower, E_in_upper)
    norm_x_in = normalize_position_relative(0.0)
    single_cond = torch.tensor([[norm_y_in, norm_u_in, norm_v_in, norm_E_in, norm_x_in]],
                               dtype=torch.float32, device=device).repeat(1000, 1)

    noise = torch.randn((1000, target_dim), dtype=torch.float32, device=device)
    x_init = torch.cat([single_cond, noise], dim=1)

    x_final = run_uniform_ode_inference(model, x_init, device, "euler", 100, show_timing=False)

    pred_y_out = denorm_position_relative(x_final[:, cond_dim + 1]).numpy()

    # --- CHANGE: Decode 7-target fractional energy (f_out) ---
    if target_dim >= 7:
        E_cutoff = float(norm_params.get("E_cutoff", 0.0))
        pred_f_out = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 4]).numpy(), 0.0, 1.0)
        pred_E_out = pred_f_out * (30.0 - E_cutoff) + E_cutoff
    else:
        E_out_lower, E_out_upper = norm_params["E_out_lower"], norm_params["E_out_upper"]
        pred_E_out = denorm_log_scale_probit(x_final[:, cond_dim + 4], E_out_lower, E_out_upper).numpy()
    print(f"Fixed Condition Spread (1000 noise samples for 30 MeV beam):")
    print(f"  y_out std: {np.std(pred_y_out):.4f} (Non-zero proves stochastic scattering works)")
    print(f"  E_out std: {np.std(pred_E_out):.4f} MeV")

    plt.figure(figsize=(6, 4))
    plt.hist(pred_E_out, bins=30, density=True, color="tab:red", alpha=0.7)
    plt.title("Conditional E_out Spread for Fixed 30 MeV Input")
    plt.xlabel("E_out (MeV)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audit_conditional_diversity.png"), dpi=300)
    plt.close()


def run_ode_step_convergence(model, x_init, cond_dim, device, out_dir):
    """Evaluates stability of model predictions across different ODE step counts."""
    log.info("Running ODE Step Convergence Sweep...")
    step_counts = [10, 25, 50, 100, 200]
    mean_E_outs = []

    for steps in step_counts:
        x_final = run_uniform_ode_inference(model, x_init, device, "euler", steps, show_timing=False)
        # --- CHANGE: Dynamically access fractional energy target index ---
        mean_E_outs.append(torch.mean(x_final[:, cond_dim + 4]).item())
    plt.figure(figsize=(6, 4))
    plt.plot(step_counts, mean_E_outs, "o-", color="tab:blue")
    plt.title("ODE Solver Convergence")
    plt.xlabel("Number of Steps")
    plt.ylabel("Mean Normalized E_out")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audit_ode_convergence.png"), dpi=300)
    plt.close()


def main():
    device = setup_environment(seed=42, allow_mps=False, inference=True)
    out_dir = "ml/validation_results/electron_2d_audit"
    os.makedirs(out_dir, exist_ok=True)

    norm_params, _ = load_norm_params_for_data_dim("2d_electron", "boundary")
    model, input_dim, cond_dim = load_model(
        device, hidden_dim=256, depth=8, expansion_factor=2,
        source_type="boundary", data_dim="2d_electron"
    )
    target_dim = input_dim - cond_dim

    run_conditional_diversity_check(model, norm_params, device, target_dim, cond_dim, out_dir)

    # Dummy noise setup for ODE sweep
    dummy_cond = torch.zeros((500, cond_dim), device=device)
    dummy_noise = torch.randn((500, target_dim), device=device)
    dummy_x_init = torch.cat([dummy_cond, dummy_noise], dim=1)

    run_ode_step_convergence(model, dummy_x_init, device, out_dir)
    log.info(f"✓ Audit complete. Diagnostic plots saved to {out_dir}")


if __name__ == "__main__":
    main()