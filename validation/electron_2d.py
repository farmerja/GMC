# validation/electron_2d.py (corrected)

import os
import numpy as np
import torch
import logging
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.utils import (
    setup_environment,
    load_model,
    run_uniform_ode_inference,
    generate_samples,
    compute_and_print_metrics,
    plot_uniform_ecdf,
    plot_uniform_hexbin,
)
from ml.transform import (
    denorm_position_relative,
    denorm_uv,
    denorm_log_scale_probit,
    load_normalization_params,
)
from physics.electron_data import load_electron_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def validate_electron_2d_boundary(
        electron_npz_train,
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
    Validate a trained 2D electron boundary model against test data.
    """
    device = setup_environment(seed=42, allow_mps=False, inference=True)

    # Load test data
    if electron_npz_test is None:
        log.info("Using training data for validation (no separate test set)")
        sim_test = load_electron_dataset(electron_npz_train)
    else:
        sim_test = load_electron_dataset(electron_npz_test)

    # Load normalization parameters
    try:
        norm_params, _ = load_normalization_params(
            "ml/data/norm_params_2d_electron_boundary.npy",
            train_sampling=train_sampling
        )
    except FileNotFoundError:
        log.error("Normalization parameters not found. Train model first.")
        return

    # Load model (this will tell us the actual input_dim and cond_dim used in training)
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

    log.info(f"Generating {num_samples} samples via inference")
    samples = generate_samples(num_samples, input_dim, sampling=infer_sampling, seed=42)

    # Build condition and noise
    cond_np = samples[:, :cond_dim]
    cond = torch.tensor(cond_np, dtype=torch.float32, device=device)

    noise_np = np.random.randn(num_samples, target_dim).astype(np.float32)
    x_init = torch.cat([
        cond,
        torch.tensor(noise_np, dtype=torch.float32, device=device)
    ], dim=1)

    log.info(f"Running ODE inference (method={ode_method}, steps={num_steps})")
    x_final = run_uniform_ode_inference(
        model, x_init, device, ode_method, num_steps, batch_size=batch_size
    )

    # Extract normalization bounds
    E_in_lower = norm_params["E_in_lower"]
    E_in_upper = norm_params["E_in_upper"]
    E_out_lower = norm_params["E_out_lower"]
    E_out_upper = norm_params["E_out_upper"]
    energy_dep_lower = norm_params["energy_dep_lower"]
    energy_dep_upper = norm_params["energy_dep_upper"]

    log.info(f"Denormalizing predictions with target_dim={target_dim}")

    # Denormalize predictions based on actual model output dimension
    # Conditions: [y_in, u_in, v_in, E_in, x_in]
    pred_y_in_norm = denorm_position_relative(x_final[:, 0]).numpy()
    pred_u_in = denorm_uv(x_final[:, 1]).numpy()
    pred_v_in = denorm_uv(x_final[:, 2]).numpy()
    pred_E_in = denorm_log_scale_probit(x_final[:, 3], E_in_lower, E_in_upper).numpy()
    pred_x_in_norm = denorm_position_relative(x_final[:, 4]).numpy()

    # Targets start at cond_dim
    # Expected order: [x_out, y_out, u_out, v_out, E_out, energy_dep, path_length]
    if target_dim >= 7:
        pred_x_out_norm = denorm_position_relative(x_final[:, cond_dim + 0]).numpy()
        pred_y_out_norm = denorm_position_relative(x_final[:, cond_dim + 1]).numpy()
        pred_u_out = denorm_uv(x_final[:, cond_dim + 2]).numpy()
        pred_v_out = denorm_uv(x_final[:, cond_dim + 3]).numpy()
        pred_E_out = np.clip(
            denorm_log_scale_probit(x_final[:, cond_dim + 4], E_out_lower, E_out_upper).numpy(),
            0.01,  # Minimum 10 keV
            None
        )
        pred_energy_dep = denorm_log_scale_probit(x_final[:, cond_dim + 5], energy_dep_lower, energy_dep_upper).numpy()
        pred_path = denorm_linear_scale_probit(x_final[:, cond_dim + 6]).numpy() * 2.0
        pred_path = np.clip(pred_path, 0.0, None)
    elif target_dim == 5:
        # Older model format with only 5 targets
        log.warning(f"Model has target_dim={target_dim}. Expected 7. Using subset of predictions.")
        pred_x_out_norm = denorm_position_relative(x_final[:, cond_dim + 0]).numpy()
        pred_y_out_norm = denorm_position_relative(x_final[:, cond_dim + 1]).numpy()
        pred_u_out = denorm_uv(x_final[:, cond_dim + 2]).numpy()
        pred_v_out = denorm_uv(x_final[:, cond_dim + 3]).numpy()
        pred_E_out = denorm_log_scale_probit(x_final[:, cond_dim + 4], E_out_lower, E_out_upper).numpy()
        pred_energy_dep = np.zeros_like(pred_E_out)  # Placeholder
        pred_path = np.zeros_like(pred_E_out)  # Placeholder
    else:
        raise ValueError(f"Unexpected target_dim={target_dim}. Expected 5 or 7.")

    # Ground truth from test data
    num_test = min(num_samples, len(sim_test))
    gt_y_in_norm = sim_test["y_in_norm"].values[:num_test]
    gt_u_in = sim_test["u_in"].values[:num_test]
    gt_v_in = sim_test["v_in"].values[:num_test]
    gt_E_in = sim_test["E_in"].values[:num_test]

    gt_x_out_norm = sim_test["x_out_norm"].values[:num_test]
    gt_y_out_norm = sim_test["y_out_norm"].values[:num_test]
    gt_u_out = sim_test["u_out"].values[:num_test]
    gt_v_out = sim_test["v_out"].values[:num_test]
    gt_E_out = sim_test["E_out"].values[:num_test]
    gt_energy_dep = sim_test["energy_deposited"].values[:num_test]
    gt_path = sim_test["path_length"].values[:num_test]

    # Create output directory
    test_dir = (
        f"ml/validation_results/electron_2d_boundary"
        f"_train-{train_sampling}_infer-{infer_sampling}"
    )
    os.makedirs(test_dir, exist_ok=True)
    log.info(f"Saving results to {test_dir}")

    # Plot comparisons
    log.info("Generating plots...")
    plot_uniform_ecdf(gt_E_in, pred_E_in[:num_test], "E_in (MeV)",
                      os.path.join(test_dir, "ecdf_E_in.png"), log10=True,
                      title="Input Energy Distribution")
    plot_uniform_ecdf(gt_E_out, pred_E_out[:num_test], "E_out (MeV)",
                      os.path.join(test_dir, "ecdf_E_out.png"), log10=True,
                      title="Output Energy Distribution")

    if target_dim >= 7:
        plot_uniform_ecdf(gt_energy_dep, pred_energy_dep[:num_test], "Energy Deposited (MeV)",
                          os.path.join(test_dir, "ecdf_energy_dep.png"), log10=True,
                          title="Energy Deposited Distribution")
        plot_uniform_ecdf(gt_path, pred_path[:num_test], "Path Length (cm)",
                          os.path.join(test_dir, "ecdf_path_length.png"), log10=True,
                          title="Path Length Distribution")

    plot_uniform_hexbin(
        gt_x_out_norm, gt_y_out_norm,
        np.clip(pred_x_out_norm[:num_test], 0.0, 1.0), np.clip(pred_y_out_norm[:num_test], 0.0, 1.0),
        "x_out (normalized)", "y_out (normalized)",
        os.path.join(test_dir, "hexbin_exit_position.png"),
        title="Exit Position Distribution"
    )

    plot_uniform_hexbin(
        gt_u_out, gt_v_out,
        pred_u_out[:num_test], pred_v_out[:num_test],
        "u_out", "v_out",
        os.path.join(test_dir, "hexbin_exit_direction.png"),
        title="Exit Direction Distribution"
    )

    # Compute metrics
    log.info("Computing error metrics...")
    gt_dict = {
        "E_in": gt_E_in,
        "E_out": gt_E_out,
        "energy_deposited": gt_energy_dep,
        "path_length": gt_path,
        "x_out": gt_x_out_norm,
        "y_out": gt_y_out_norm,
        "u_out": gt_u_out,
        "v_out": gt_v_out,
    }

    pred_dict = {
        "E_in": pred_E_in[:num_test],
        "E_out": np.clip(pred_E_out[:num_test], 0.0, None),
        "energy_deposited": np.clip(pred_energy_dep[:num_test], 0.0, None),
        "path_length": np.clip(pred_path[:num_test], 1e-12, None),
        "x_out": np.clip(pred_x_out_norm[:num_test], 0.0, 1.0),
        "y_out": np.clip(pred_y_out_norm[:num_test], 0.0, 1.0),
        "u_out": pred_u_out[:num_test],
        "v_out": pred_v_out[:num_test],
    }

    compute_and_print_metrics(gt_dict, pred_dict, test_dir, "2D Electron Boundary")

    log.info(f"✓ Validation complete. Results saved to {test_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate 2D electron boundary transport model"
    )
    parser.add_argument(
        "--train_npz",
        type=str,
        required=True,
        help="Path to training NPZ file (for normalization)",
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default=None,
        help="Path to test NPZ file (if None, uses train_npz)",
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