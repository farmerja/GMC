# validation/electron_2d.py

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.utils import (
    setup_environment,
    load_model,
    run_uniform_ode_inference,
    compute_and_print_metrics,
    plot_uniform_ecdf,
    plot_uniform_hexbin,
    load_norm_params_for_data_dim,
)
from ml.transform import (
    normalize_position_relative,
    normalize_uv,
    normalize_log_scale_probit,
    normalize_angle_phi,
    normalize_radius_r,
    denorm_position_relative,
    denorm_uv,
    denorm_log_scale_probit,
    denorm_linear_scale_probit,
    denorm_angle_phi,
    denorm_radius_r,
    uv_to_polar,
    polar_to_uv,
)
from physics.electron_data import load_electron_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def validate_electron_2d_boundary(
        electron_npz_train=None,
        electron_npz_test=None,
        hidden_dim=256,
        depth=8,
        expansion_factor=2,
        num_samples=10000,
        num_steps=100,
        ode_method="euler",
        batch_size=256,
        train_sampling="random",
        infer_sampling="random",
        scheduler_type="condOT",
):
    """
    Validate a trained 2D electron boundary transport model against test data.
    """
    device = setup_environment(seed=42, allow_mps=False, inference=True)

    # Determine test dataset path
    data_path = electron_npz_test or electron_npz_train
    if data_path is None:
        log.error("Either --test_npz or --train_npz must be provided.")
        return

    log.info(f"Loading test dataset from: {data_path}")
    sim_test = load_electron_dataset(data_path)

    # Load normalization parameters via standard utility helper
    try:
        norm_params, norm_path = load_norm_params_for_data_dim(
            "2d_electron", "boundary", train_sampling=train_sampling
        )
        log.info(f"Loaded normalization parameters from: {norm_path}")
    except FileNotFoundError:
        log.error("Normalization parameters not found in ml/data/. Train model first.")
        return

    # Load trained model
    model, input_dim, cond_dim = load_model(
        device,
        hidden_dim=hidden_dim,
        depth=depth,
        expansion_factor=expansion_factor,
        source_type="boundary",
        data_dim="2d_electron",
        train_sampling=train_sampling,
        scheduler_type=scheduler_type,
    )

    target_dim = input_dim - cond_dim
    log.info(f"Model dimensions: input_dim={input_dim}, cond_dim={cond_dim}, target_dim={target_dim}")

    # Subset test dataset
    num_eval = min(num_samples, len(sim_test))
    test_sub = sim_test.iloc[:num_eval].reset_index(drop=True)

    # Extract normalization bounds and dynamic cutoff
    E_in_lower = norm_params["E_in_lower"]
    E_in_upper = norm_params["E_in_upper"]
    E_out_lower = norm_params["E_out_lower"]
    E_out_upper = norm_params["E_out_upper"]
    energy_dep_lower = norm_params["energy_dep_lower"]
    energy_dep_upper = norm_params["energy_dep_upper"]
    path_lower = norm_params.get("path_lower", -12.0)
    path_upper = norm_params.get("path_upper", 1.0)
    E_cutoff = float(
        norm_params.get(
            "E_cutoff",
            getattr(sim_test, "attrs", {}).get(
                "energy_cutoff",
                getattr(sim_test, "attrs", {}).get("E_cutoff", sim_test["E_out"].min())
            )
        )
    )
    log.info(f"Using energy cutoff threshold: {E_cutoff:.4f} MeV")

    # Build normalized condition tensor directly from test dataset
    log.info(f"Preparing {num_eval} input conditions from test dataset")
    phi_in, r_in = uv_to_polar(test_sub["u_in"].values, test_sub["v_in"].values)
    norm_y_in = normalize_position_relative(test_sub["y_in_norm"].values)
    norm_phi_in = normalize_angle_phi(phi_in)
    norm_r_in = normalize_radius_r(r_in)
    norm_E_in = normalize_log_scale_probit(test_sub["E_in"].values, E_in_lower, E_in_upper)
    norm_x_in = normalize_position_relative(test_sub["x_in_norm"].values)

    cond_np = np.stack([norm_y_in, norm_phi_in, norm_r_in, norm_E_in, norm_x_in], axis=1)
    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)

    # Generate target noise
    torch.manual_seed(42)
    noise = torch.randn((num_eval, target_dim), dtype=torch.float32, device=device)
    x_init = torch.cat([cond, noise], dim=1)

    log.info(f"Running ODE inference (method={ode_method}, steps={num_steps})")
    x_final = run_uniform_ode_inference(
        model, x_init, device, ode_method, num_steps, batch_size=batch_size
    )

    # Ground truth from test data
    gt_E_in = test_sub["E_in"].values
    gt_x_out_norm = test_sub["x_out_norm"].values
    gt_y_out_norm = test_sub["y_out_norm"].values
    gt_u_out = test_sub["u_out"].values
    gt_v_out = test_sub["v_out"].values
    gt_E_out = test_sub["E_out"].values

    log.info(f"Denormalizing predictions with target_dim={target_dim}")

    # Expected target order: [x_out, y_out, phi_out, r_out, f_out, f_dep, path_length]
    if target_dim >= 7:
        pred_x_out_norm = denorm_position_relative(x_final[:, cond_dim + 0]).numpy()
        pred_y_out_norm = denorm_position_relative(x_final[:, cond_dim + 1]).numpy()

        # Decode polar directions -> Cartesian (u, v)
        pred_phi_out = denorm_angle_phi(x_final[:, cond_dim + 2]).numpy()
        pred_r_out = denorm_radius_r(x_final[:, cond_dim + 3]).numpy()  # Clamped to [0, 1]
        pred_u_out, pred_v_out = polar_to_uv(pred_phi_out, pred_r_out)

        # Decode fractional energy partitioning relative to E_cutoff
        pred_f_out = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 4]).numpy(), 0.0, 1.0)
        pred_E_out = pred_f_out * (gt_E_in - E_cutoff) + E_cutoff
        pred_E_out = np.maximum(pred_E_out, E_cutoff)

        pred_f_dep = np.clip(denorm_linear_scale_probit(x_final[:, cond_dim + 5]).numpy(), 0.0, 1.0)
        remaining_energy = np.maximum(0.0, gt_E_in - pred_E_out)
        pred_energy_dep = pred_f_dep * remaining_energy

        pred_path = denorm_log_scale_probit(x_final[:, cond_dim + 6], path_lower, path_upper).numpy()
    elif target_dim == 5:
        log.warning(f"Model has target_dim={target_dim}. Expected 7. Denormalizing spatial & energy features only.")
        pred_x_out_norm = denorm_position_relative(x_final[:, cond_dim + 0]).numpy()
        pred_y_out_norm = denorm_position_relative(x_final[:, cond_dim + 1]).numpy()
        pred_u_out = denorm_uv(x_final[:, cond_dim + 2]).numpy()
        pred_v_out = denorm_uv(x_final[:, cond_dim + 3]).numpy()
        pred_E_out = denorm_log_scale_probit(x_final[:, cond_dim + 4], E_out_lower, E_out_upper).numpy()
        pred_energy_dep = None
        pred_path = None
    else:
        raise ValueError(f"Unexpected target_dim={target_dim}. Expected 5 or 7.")

    # Output directory setup
    test_dir = (
        f"ml/validation_results/electron_2d_boundary"
        f"_train-{train_sampling}_infer-{infer_sampling}"
    )
    os.makedirs(test_dir, exist_ok=True)
    log.info(f"Saving results to {test_dir}")

    # =========================================================================
    # INSTANTANEOUS PHYSICS & CONSERVATION LAW AUDIT
    # =========================================================================
    log.info("Auditing physical conservation laws...")

    # 1. Spatial bounds check
    out_of_bounds = (pred_x_out_norm < 0.0) | (pred_x_out_norm > 1.0) | \
                    (pred_y_out_norm < 0.0) | (pred_y_out_norm > 1.0)
    bounds_violation_rate = np.mean(out_of_bounds) * 100.0

    # 2. Direction cosine norm check (u^2 + v^2 <= 1.0)
    dir_norm_sq = pred_u_out ** 2 + pred_v_out ** 2
    dir_violation_rate = np.mean(dir_norm_sq > 1.0001) * 100.0

    # 3. Energy balance check (E_out + E_deposited <= E_in)
    if target_dim >= 7:
        e_sum = pred_E_out + pred_energy_dep
        unphysical_energy = e_sum > (gt_E_in + 1e-3)
        energy_violation_rate = np.mean(unphysical_energy) * 100.0
        mean_energy_imbalance = np.mean(gt_E_in - e_sum)
    else:
        energy_violation_rate = 0.0
        mean_energy_imbalance = 0.0

    print(f"\n==================== PHYSICS CONSERVATION AUDIT ====================")
    print(f"  Spatial Out-of-Bounds Exit Rate (x,y ∉ [0,1]):  {bounds_violation_rate:.3f}%")
    print(f"  Direction Cosine Violation Rate (u²+v² > 1.0): {dir_violation_rate:.3f}%")
    if target_dim >= 7:
        print(f"  Unphysical Energy Creation Rate (> E_in):       {energy_violation_rate:.3f}%")
        print(f"  Mean Energy Imbalance (E_in - [E_out + ΔE]):    {mean_energy_imbalance:.4f} MeV")
    print(f"===================================================================\n")

    # =========================================================================
    # PLOTTING & VISUALIZATION
    # =========================================================================
    log.info("Generating plots...")

    # 1D ECDF comparisons
    plot_uniform_ecdf(
        gt_E_out,
        np.maximum(pred_E_out, 0.0),
        r"$\log_{10}(E_{\mathrm{out}}\;/\;\mathrm{MeV})$",
        os.path.join(test_dir, "ecdf_E_out.png"),
        log10=True,
        title="Output Energy Distribution"
    )

    if target_dim >= 7:
        gt_energy_dep = test_sub["energy_deposited"].values
        gt_path = test_sub["path_length"].values

        plot_uniform_ecdf(gt_energy_dep, np.maximum(pred_energy_dep, 0.0), "Energy Deposited (MeV)",
                          os.path.join(test_dir, "ecdf_energy_dep.png"), log10=True,
                          title="Energy Deposited Distribution")
        plot_uniform_ecdf(gt_path, np.maximum(pred_path, 0.0), "Path Length (cm)",
                          os.path.join(test_dir, "ecdf_path_length.png"), log10=True,
                          title="Path Length Distribution")

    # 2D Marginal Hexbin comparisons
    plot_uniform_hexbin(
        gt_x_out_norm, gt_y_out_norm,
        np.clip(pred_x_out_norm, 0.0, 1.0), np.clip(pred_y_out_norm, 0.0, 1.0),
        "x_out (normalized)", "y_out (normalized)",
        os.path.join(test_dir, "hexbin_exit_position.png"),
        title="Exit Position Distribution"
    )

    plot_uniform_hexbin(
        gt_u_out, gt_v_out,
        np.clip(pred_u_out, -1.0, 1.0), np.clip(pred_v_out, -1.0, 1.0),
        "u_out", "v_out",
        os.path.join(test_dir, "hexbin_exit_direction.png"),
        title="Exit Direction Distribution"
    )

    # =========================================================================
    # JOINT FEATURE CORRELATION PLOTS
    # =========================================================================
    if target_dim >= 7:
        log.info("Generating joint feature correlation plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # MC Ground Truth
        axes[0].hexbin(gt_path, gt_E_out, gridsize=50, cmap="Blues", mincnt=1)
        axes[0].set_title("MC: Path Length vs Output Energy")
        axes[0].set_xlabel("Path Length (cm)")
        axes[0].set_ylabel("E_out (MeV)")
        axes[0].grid(alpha=0.2)

        # GMC Model Prediction
        axes[1].hexbin(pred_path, np.maximum(pred_E_out, 0.0), gridsize=50, cmap="Reds", mincnt=1)
        axes[1].set_title("GMC: Path Length vs Output Energy")
        axes[1].set_xlabel("Path Length (cm)")
        axes[1].set_ylabel("E_out (MeV)")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, "joint_path_vs_Eout.png"), dpi=300)
        plt.close()

    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================
    log.info("Computing error metrics...")
    gt_dict = {
        "E_out": gt_E_out,
        "x_out": gt_x_out_norm,
        "y_out": gt_y_out_norm,
        "u_out": gt_u_out,
        "v_out": gt_v_out,
    }

    pred_dict = {
        "E_out": np.maximum(pred_E_out, 0.0),
        "x_out": np.clip(pred_x_out_norm, 0.0, 1.0),
        "y_out": np.clip(pred_y_out_norm, 0.0, 1.0),
        "u_out": np.clip(pred_u_out, -1.0, 1.0),
        "v_out": np.clip(pred_v_out, -1.0, 1.0),
    }

    if target_dim >= 7:
        gt_dict["energy_deposited"] = test_sub["energy_deposited"].values
        gt_dict["path_length"] = test_sub["path_length"].values
        pred_dict["energy_deposited"] = np.maximum(pred_energy_dep, 0.0)
        pred_dict["path_length"] = np.maximum(pred_path, 0.0)

    compute_and_print_metrics(gt_dict, pred_dict, test_dir, "2D Electron Boundary")

    log.info(f"✓ Validation complete. Results and diagnostic plots saved to {test_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate 2D electron boundary transport model"
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default=None,
        help="Path to test NPZ dataset file",
    )
    parser.add_argument(
        "--train_npz",
        type=str,
        default=None,
        help="Path to training NPZ dataset file (fallback if --test_npz is not provided)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--expansion_factor", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--ode_method", type=str, default="euler")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_sampling", type=str, default="random")
    parser.add_argument("--infer_sampling", type=str, default="random")
    parser.add_argument("--scheduler_type", type=str, default="condOT")

    args = parser.parse_args()

    validate_electron_2d_boundary(
        electron_npz_train=args.train_npz,
        electron_npz_test=args.test_npz,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        expansion_factor=args.expansion_factor,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        ode_method=args.ode_method,
        batch_size=args.batch_size,
        train_sampling=args.train_sampling,
        infer_sampling=args.infer_sampling,
        scheduler_type=args.scheduler_type,
    )