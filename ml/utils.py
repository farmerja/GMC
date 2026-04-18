#!/usr/bin/env python3
import inspect
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_matching.solver import ODESolver
from ml.model import WrappedModel, get_model
from ml.transform import (
    direction_feature_count_2d,
    load_normalization_params,
    normalize_direction_encoding_2d,
    resolve_norm_params_base_path,
    save_normalization_params,
)
from physics.steady_2d_scattering import (
    STEADY_2D_SCATTERING_TAG,
    normalize_steady_2d_scattering_mode,
    steady_2d_model_dir as resolve_steady_2d_model_dir,
    steady_2d_model_dir_candidates as resolve_steady_2d_model_dir_candidates,
    steady_2d_norm_params_base_path as resolve_steady_2d_norm_params_base_path,
    steady_2d_norm_params_base_path_candidates as resolve_steady_2d_norm_params_base_path_candidates,
    steady_2d_training_dir as resolve_steady_2d_training_dir,
)

try:
    from scipy.stats import ks_2samp, norm, qmc, wasserstein_distance
except ImportError:
    qmc = norm = wasserstein_distance = ks_2samp = None


DATA_DIM_CHOICES = (
    "1d",
    "2d",
    "3d",
    "1d_timedep",
    "2d_timedep",
    "3d_timedep",
    "2d_triangle_timedep",
    "2d_quad_timedep",
)
TIME_DEPENDENT_DATA_DIMS = {
    "1d_timedep",
    "2d_timedep",
    "2d_triangle_timedep",
    "2d_quad_timedep",
    "3d_timedep",
}
TWO_D_TIME_DEPENDENT_DATA_DIMS = {
    "2d_timedep",
    "2d_triangle_timedep",
    "2d_quad_timedep",
}
PLOT_TITLES_ENABLED = True


def set_plot_titles_enabled(enabled):
    global PLOT_TITLES_ENABLED
    PLOT_TITLES_ENABLED = bool(enabled)


def setup_environment(seed=41, *, allow_mps=True, inference=False):
    if allow_mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(not inference)
    return device


def _safe_int_env(var_name):
    value = os.environ.get(var_name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def available_cpu_count():
    candidates = []
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except Exception:
            pass
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 0:
        candidates.append(cpu_count)
    for env_var in ("NSLOTS", "SLURM_CPUS_PER_TASK", "PBS_NP"):
        env_count = _safe_int_env(env_var)
        if env_count is not None:
            candidates.append(env_count)
    return min(candidates) if candidates else 1


def safe_get_sim_data(func, **kwargs):
    params = inspect.signature(func).parameters
    if "sampling" not in inspect.signature(func).parameters:
        kwargs.pop("sampling", None)
    if "scattering_mode" not in params:
        kwargs.pop("scattering_mode", None)
    return func(**kwargs)


def analytic_log_bounds(min_val, max_val, name):
    min_val = float(min_val)
    max_val = float(max_val)
    if min_val <= 0.0 or max_val <= 0.0:
        raise ValueError(f"{name} bounds must be > 0 for log scaling, got [{min_val}, {max_val}]")
    if max_val <= min_val:
        raise ValueError(f"{name} bounds must satisfy max > min, got [{min_val}, {max_val}]")
    return np.log10(min_val), np.log10(max_val)


def save_norm_params_artifacts(norm_params_path, norm_params, label):
    directory = os.path.dirname(norm_params_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    meta_path = save_normalization_params(norm_params_path, norm_params)
    print(f"Saved {label} normalization parameters to {meta_path}")
    return meta_path


def setup_test_environment(
    source_type="boundary",
    scheduler_type="condOT",
    data_dim="2d",
    infer_sampling="random",
    train_sampling="random",
):
    test_dir = (
        f"ml/testing_results/cfm_test_{data_dim}_{source_type}_{scheduler_type}"
        f"_train-{train_sampling}_infer-{infer_sampling}"
    )
    os.makedirs(test_dir, exist_ok=True)
    device = setup_environment(seed=42, allow_mps=False, inference=True)
    print(f"Testing Conditional Flow Matching (CFM) model - {infer_sampling.upper()} CONDITIONS")
    print(f"  Data Dim: {data_dim}")
    print(f"  Source Type: {data_dim.upper()} {source_type}")
    print(f"  Scheduler Type: {scheduler_type}")
    return device, test_dir


def generate_samples(n, d, sampling="random", seed=42):
    if sampling == "qmc":
        if qmc is None:
            raise ImportError("scipy.stats.qmc is required for QMC sampling.")
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (n & (n - 1) == 0) and n != 0:
                return sampler.random_base2(m=int(np.log2(n)))
            return sampler.random(n)
    if sampling == "lhs":
        if qmc is None:
            raise ImportError("scipy.stats.qmc is required for LHS sampling.")
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, d))


def generate_target_noise(samples, cond_features, target_dim, batch_size, device):
    if target_dim <= 0:
        return torch.empty((batch_size, 0), device=device)
    if norm is None:
        return torch.randn((batch_size, target_dim), device=device)
    noise_unif = np.clip(samples[:, cond_features : cond_features + target_dim], 1e-7, 1.0 - 1e-7)
    noise_np = norm.ppf(noise_unif)
    return torch.tensor(noise_np, dtype=torch.float32, device=device)


def steady_2d_model_dir(data_dim, source_type, scattering_mode="isotropic"):
    if data_dim == "2d":
        return resolve_steady_2d_model_dir(source_type, scattering_mode=scattering_mode)
    return f"ml/models/{data_dim}_{source_type}"


def steady_2d_training_dir(data_dim, source_type, scattering_mode="isotropic"):
    if data_dim == "2d":
        return resolve_steady_2d_training_dir(source_type, scattering_mode=scattering_mode)
    return f"ml/training_results/{data_dim}_{source_type}"


def norm_params_base_path(data_dim, source_type, scattering_mode="isotropic"):
    if data_dim == "1d_timedep":
        return f"ml/data/norm_params_1d_timedep_{source_type}.npy"
    if data_dim == "2d":
        return resolve_steady_2d_norm_params_base_path(source_type, scattering_mode=scattering_mode)
    if data_dim == "2d_timedep":
        return f"ml/data/norm_params_2d_timedep_{source_type}.npy"
    if data_dim == "2d_triangle_timedep":
        return f"ml/data/norm_params_2d_triangle_timedep_{source_type}.npy"
    if data_dim == "2d_quad_timedep":
        return "ml/data/norm_params_2d_quad_timedep_internal.npy"
    if data_dim == "3d":
        return f"ml/data/norm_params_3d_{source_type}.npy"
    if data_dim == "3d_timedep":
        return f"ml/data/norm_params_3d_timedep_{source_type}.npy"
    raise ValueError(f"Unsupported data_dim='{data_dim}'")


def load_norm_params_for_data_dim(data_dim, source_type, train_sampling, scattering_mode="isotropic"):
    candidates = norm_params_base_path_candidates(data_dim, source_type, scattering_mode=scattering_mode)
    last_error = None
    for base_path in candidates:
        try:
            return load_normalization_params(base_path, train_sampling=train_sampling)
        except FileNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"Normalization metadata not found for data_dim='{data_dim}', source_type='{source_type}'."
    )


def load_norm_params_only(data_dim, source_type, train_sampling, scattering_mode="isotropic"):
    norm_params, _ = load_norm_params_for_data_dim(
        data_dim,
        source_type,
        train_sampling,
        scattering_mode=scattering_mode,
    )
    return norm_params


def load_norm_params_or_abort(data_dim, source_type, train_sampling, description, scattering_mode="isotropic"):
    try:
        return load_norm_params_only(
            data_dim,
            source_type,
            train_sampling,
            scattering_mode=scattering_mode,
        )
    except FileNotFoundError:
        print(f"FATAL: normalization parameters for {description} not found. Cannot proceed.")
        return None


def resolve_2d_direction_encoding(norm_params):
    return normalize_direction_encoding_2d(norm_params=norm_params)


def resolve_norm_params_file(base_path, train_sampling="random"):
    return resolve_norm_params_base_path(base_path, train_sampling=train_sampling)


def steady_2d_model_dir_candidates(data_dim, source_type, scattering_mode="isotropic"):
    if data_dim == "2d":
        return resolve_steady_2d_model_dir_candidates(source_type, scattering_mode=scattering_mode)
    return (steady_2d_model_dir(data_dim, source_type, scattering_mode=scattering_mode),)


def norm_params_base_path_candidates(data_dim, source_type, scattering_mode="isotropic"):
    if data_dim == "2d":
        return resolve_steady_2d_norm_params_base_path_candidates(source_type, scattering_mode=scattering_mode)
    return (norm_params_base_path(data_dim, source_type, scattering_mode=scattering_mode),)


def compute_model_dims(data_dim, source_type, *, train_sampling="random", scattering_mode="isotropic"):
    dir_feats_2d = None
    if data_dim in TWO_D_TIME_DEPENDENT_DATA_DIMS:
        norm_params = load_norm_params_only(data_dim, source_type, train_sampling)
        dir_feats_2d = direction_feature_count_2d(resolve_2d_direction_encoding(norm_params))
    if data_dim == "1d_timedep":
        return (7, 3) if source_type == "boundary" else (8, 4)
    if data_dim == "2d_timedep":
        cond_dim = (4 if source_type == "boundary" else 5) + dir_feats_2d
        return cond_dim + 4 + dir_feats_2d, cond_dim
    if data_dim == "2d_triangle_timedep":
        cond_dim = (5 if source_type == "boundary" else 6) + dir_feats_2d
        return cond_dim + 4 + dir_feats_2d, cond_dim
    if data_dim == "2d_quad_timedep":
        if source_type == "boundary":
            raise NotImplementedError("Boundary source for 2D quad timedep is not supported.")
        cond_dim = 8 + dir_feats_2d
        return cond_dim + 4 + dir_feats_2d, cond_dim
    if data_dim == "3d":
        cond_dim = 7 if source_type == "boundary" else 8
        return cond_dim + 6, cond_dim
    if data_dim == "3d_timedep":
        cond_dim = 9 if source_type == "boundary" else 10
        return cond_dim + 8, cond_dim
    if source_type == "boundary":
        return 10, 5
    return 11, 6


def _looks_like_state_dict(obj):
    if not isinstance(obj, dict):
        return False
    tensor_keys = [k for k, v in obj.items() if isinstance(k, str) and torch.is_tensor(v)]
    return bool(tensor_keys) and any(
        k.startswith(("input_proj.", "cond_proj.", "time_embed.", "blocks.", "output."))
        for k in tensor_keys
    )


def _extract_state_dict(checkpoint):
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                return candidate
    raise ValueError("checkpoint does not contain a recognizable model state_dict")


def _normalize_state_dict_keys(state_dict):
    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _infer_architecture_from_state_dict(state_dict):
    inferred = {}
    input_proj_w = state_dict.get("input_proj.weight")
    if torch.is_tensor(input_proj_w) and input_proj_w.ndim == 2:
        inferred["hidden_dim"] = int(input_proj_w.shape[0])
        inferred["input_dim"] = int(input_proj_w.shape[1])
    cond_proj_w = state_dict.get("cond_proj.0.weight")
    if torch.is_tensor(cond_proj_w) and cond_proj_w.ndim == 2:
        inferred["cond_dim"] = int(cond_proj_w.shape[1])
    block_ids = []
    for key in state_dict.keys():
        match = re.match(r"^blocks\.(\d+)\.", key)
        if match:
            block_ids.append(int(match.group(1)))
    if block_ids:
        inferred["depth"] = max(block_ids) + 1
    ff0_w = state_dict.get("blocks.0.ff.0.weight")
    hidden_dim = inferred.get("hidden_dim")
    if torch.is_tensor(ff0_w) and ff0_w.ndim == 2 and hidden_dim:
        ff_dim = int(ff0_w.shape[0])
        if ff_dim % hidden_dim == 0:
            inferred["expansion_factor"] = ff_dim // hidden_dim
    return inferred


def load_model(
    device,
    hidden_dim,
    depth,
    expansion_factor=2,
    source_type="boundary",
    data_dim="2d",
    train_sampling="random",
    scheduler_type="condOT",
    scattering_mode="isotropic",
):
    if data_dim == "2d":
        scattering_mode = normalize_steady_2d_scattering_mode(scattering_mode)
    input_dim, cond_dim = compute_model_dims(
        data_dim,
        source_type,
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
    )
    mode_suffix = (
        f", scattering_mode={scattering_mode}, asset_tag={STEADY_2D_SCATTERING_TAG}"
        if data_dim == "2d"
        else ""
    )
    print(f"Loading model for {data_dim} {source_type} source (input_dim={input_dim}, cond_dim={cond_dim}{mode_suffix})")
    scheduler_suffix = f"_{scheduler_type}" if scheduler_type != "condOT" else ""
    sampling_suffix = f"_{train_sampling}" if train_sampling != "random" else ""
    model_dirs = steady_2d_model_dir_candidates(data_dim, source_type, scattering_mode=scattering_mode)
    model_paths = list(
        dict.fromkeys(
            path
            for model_dir in model_dirs
            for path in (
                os.path.join(model_dir, f"best_model{scheduler_suffix}{sampling_suffix}.pth"),
                os.path.join(model_dir, f"trained_model_final{scheduler_suffix}{sampling_suffix}.pth"),
                os.path.join(model_dir, f"best_model{scheduler_suffix}.pth"),
                os.path.join(model_dir, f"trained_model_final{scheduler_suffix}.pth"),
                os.path.join(model_dir, "best_model.pth"),
                os.path.join(model_dir, "trained_model_final.pth"),
            )
        )
    )

    model = None
    loaded_model_path = None
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
            inferred = _infer_architecture_from_state_dict(state_dict)
            ckpt_input_dim = inferred.get("input_dim")
            ckpt_cond_dim = inferred.get("cond_dim")
            if ckpt_input_dim is not None and ckpt_input_dim != input_dim:
                print(
                    f"Warning: Skipping model at {model_path} due to input_dim mismatch "
                    f"(checkpoint={ckpt_input_dim}, expected={input_dim})."
                )
                continue
            if ckpt_cond_dim is not None and ckpt_cond_dim != cond_dim:
                print(
                    f"Warning: Skipping model at {model_path} due to cond_dim mismatch "
                    f"(checkpoint={ckpt_cond_dim}, expected={cond_dim})."
                )
                continue
            model_hidden_dim = inferred.get("hidden_dim", hidden_dim)
            model_depth = inferred.get("depth", depth)
            model_expansion = inferred.get("expansion_factor", expansion_factor)
            if (
                model_hidden_dim != hidden_dim
                or model_depth != depth
                or model_expansion != expansion_factor
            ):
                print(
                    "Checkpoint architecture differs from CLI args; using checkpoint values: "
                    f"hidden_dim={model_hidden_dim}, depth={model_depth}, expansion_factor={model_expansion}"
                )
            model = get_model(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=model_hidden_dim,
                depth=model_depth,
                expansion_factor=model_expansion,
            ).to(device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                preview = ", ".join(unexpected_keys[:5])
                suffix = " ..." if len(unexpected_keys) > 5 else ""
                print(f"Note: ignoring unexpected checkpoint keys: {preview}{suffix}")
            if missing_keys:
                preview = ", ".join(missing_keys[:5])
                suffix = " ..." if len(missing_keys) > 5 else ""
                print(f"Note: missing model keys when loading checkpoint: {preview}{suffix}")
            print(f"Successfully loaded model from {model_path}")
            loaded_model_path = model_path
            break
        except Exception as exc:
            print(f"Warning: Failed to load model from {model_path}: {exc}")

    if model is None:
        model = get_model(
            input_dim=input_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            expansion_factor=expansion_factor,
        ).to(device)
        print("Warning: No saved model found. Using untrained model.")
    elif train_sampling != "random" and not loaded_model_path.endswith(f"_{train_sampling}.pth"):
        print(
            f"Warning: requested train_sampling='{train_sampling}' but loaded non-suffixed model: "
            f"{loaded_model_path}"
        )

    model.eval()
    return model, input_dim, cond_dim


def build_uniform_time_grid(num_steps, device):
    if num_steps <= 1:
        return torch.tensor([0.0, 1.0], device=device), None
    base_steps = max(1, num_steps // 2)
    fine_steps = num_steps - base_steps
    base_grid = torch.linspace(0, 0.8, base_steps + 1, device=device)[:-1]
    fine_grid = torch.linspace(0.8, 1.0, fine_steps + 1, device=device)
    time_grid = torch.cat([base_grid, fine_grid])
    return time_grid, time_grid[1] - time_grid[0]


def run_uniform_ode_inference(model, x_init, device, ode_method, num_steps, batch_size=None, show_timing=True):
    wrapped_model = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_model)
    time_grid, step_size = build_uniform_time_grid(num_steps, device)
    if show_timing:
        print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        model.eval()
        wrapped_model.eval()
        sol = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=ode_method,
            step_size=step_size,
            return_intermediates=False,
        )
    dt = time.time() - t0
    if show_timing:
        if batch_size is None:
            print(f"Inference completed in {dt:.4f} s")
        else:
            print(f"Inference completed in {dt:.4f} s ({dt / batch_size * 1000:.4f} ms/sample)")
    return sol.cpu().detach()


def maybe_set_plot_title(title):
    if PLOT_TITLES_ENABLED:
        plt.title(title)


def compute_ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def plot_uniform_ecdf(gt_values, pred_values, xlabel, output_path, *, title=None, figure_size=(10, 6), log10=False):
    plt.figure(figsize=figure_size)
    if log10:
        eps = 1e-12
        gt_plot = np.log10(np.maximum(gt_values, eps))
        pred_plot = np.log10(np.maximum(pred_values, eps))
    else:
        gt_plot = gt_values
        pred_plot = pred_values
    xg, yg = compute_ecdf(gt_plot)
    xp, yp = compute_ecdf(pred_plot)
    if xg.size:
        plt.plot(xg, yg, label="MC", lw=2, color="tab:blue")
    if xp.size:
        plt.plot(xp, yp, "--", label="GMC", lw=2, color="tab:red")
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    if title:
        maybe_set_plot_title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_uniform_hexbin(
    gt_x,
    gt_y,
    pred_x,
    pred_y,
    xlabel,
    ylabel,
    output_path,
    *,
    title=None,
    figure_size=(12, 9),
):
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    plt.figure(figsize=figure_size)
    hb_mc = plt.hexbin(gt_x, gt_y, gridsize=60, cmap="Blues", alpha=0.7, norm=LogNorm())
    plt.hexbin(pred_x, pred_y, gridsize=60, cmap="Reds", alpha=0.5, norm=LogNorm())
    mc_p = Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.7)
    ml_p = Rectangle((0, 0), 1, 1, color="tab:red", alpha=0.5)
    plt.legend([mc_p, ml_p], ["MC", "GMC"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        maybe_set_plot_title(title)
    cbar = plt.colorbar(hb_mc)
    cbar.set_label("Density (log scale)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_and_print_metrics(gt_dict, pred_dict, test_dir, tag):
    print(f"\n--- Error Metrics ({tag}) ---")
    metrics_file = os.path.join(test_dir, f"error_metrics_{tag}.txt")
    if wasserstein_distance is None or ks_2samp is None:
        print("scipy is required to compute Wasserstein/KS metrics.")
        return
    with open(metrics_file, "w") as handle:
        handle.write(f"Error Metrics ({tag})\n")
        header = f"{'Metric':<20} | {'Wasserstein':<15} | {'KS Stat':<15}"
        print(header)
        print("-" * 55)
        handle.write(header + "\n")
        handle.write("-" * 55 + "\n")
        for key in gt_dict:
            if key not in pred_dict or key in ("space_exit", "time_exit"):
                continue
            g = np.asarray(gt_dict[key]).flatten()
            p = np.asarray(pred_dict[key]).flatten()
            g = g[np.isfinite(g)]
            p = p[np.isfinite(p)]
            if g.size == 0 or p.size == 0:
                continue
            wd = wasserstein_distance(g, p)
            ks = ks_2samp(g, p).statistic
            line = f"{key:<20} | {wd:<15.4e} | {ks:<15.4e}"
            print(line)
            handle.write(line + "\n")
