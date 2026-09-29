#!/usr/bin/env python3
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from scipy import special

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Normalization Parameters ---
_NORM_SCHEMA_VERSION = 2
_DIRECTION_ENCODINGS_2D = {"polar"}


def _coerce_loaded_scalar(value):
    if isinstance(value, np.ndarray):
        if value.shape == () or value.size == 1:
            return value.reshape(()).item()
    return value


def _serialize_norm_value(value):
    if isinstance(value, (str, bytes)):
        return np.asarray(value)
    if isinstance(value, (bool, np.bool_)):
        return np.asarray(bool(value))
    return np.asarray(value)


def norm_metadata_path(base_path):
    return str(Path(base_path).with_suffix(".npz"))


def resolve_norm_params_base_path(base_path, train_sampling="random"):
    base = Path(base_path)
    if base.suffix == ".npz":
        base = base.with_suffix(".npy")
    elif base.suffix != ".npy":
        base = base.with_suffix(".npy")

    suffix = f"_{train_sampling}" if train_sampling != "random" else ""
    return str(base.with_name(f"{base.stem}{suffix}{base.suffix}"))


def normalize_direction_encoding_2d(direction_encoding_2d=None, *, norm_params=None):
    requested = "" if direction_encoding_2d is None else str(direction_encoding_2d).strip().lower()
    if requested in _DIRECTION_ENCODINGS_2D:
        return requested
    if requested not in {"", "auto"}:
        raise ValueError(
            f"Unsupported direction_encoding_2d='{direction_encoding_2d}'. "
            "Expected one of: auto, polar."
        )

    if norm_params is None:
        return "polar"

    saved = str(norm_params.get("direction_encoding_2d", "")).strip().lower()
    if saved in {"", "auto"}:
        return "polar"
    if saved == "polar":
        return saved
    raise ValueError(
        "Only polar direction encoding is supported for 2D problems. "
        f"Found saved direction_encoding_2d='{saved}'. Regenerate normalization metadata and retrain the 2D model."
    )


def direction_feature_count_2d(direction_encoding_2d=None, *, norm_params=None):
    normalize_direction_encoding_2d(direction_encoding_2d, norm_params=norm_params)
    return 2


def save_normalization_params(base_path, norm_params):
    base = Path(base_path)
    if base.suffix != ".npy":
        raise ValueError(f"Expected a .npy base path, got: {base_path}")
    payload = {"_schema_version": np.asarray(_NORM_SCHEMA_VERSION, dtype=np.int64)}
    for key, value in norm_params.items():
        payload[key] = _serialize_norm_value(value)
    meta_path = base.with_suffix(".npz")
    np.savez(meta_path, **payload)
    return str(meta_path)


def load_normalization_params(base_path, train_sampling="random"):
    resolved_base = resolve_norm_params_base_path(base_path, train_sampling=train_sampling)
    meta_path = Path(norm_metadata_path(resolved_base))
    if meta_path.exists():
        with np.load(meta_path, allow_pickle=False) as payload:
            norm_params = {}
            for key in payload.files:
                if key == "_schema_version":
                    continue
                norm_params[key] = _coerce_loaded_scalar(payload[key])
        return norm_params, str(meta_path)

    raise FileNotFoundError(
        f"Normalization metadata not found for base '{base_path}' "
        f"(expected '{meta_path}')."
    )


def compute_normalization_params_1d(sim_data):
    log_times, log_thickness = np.log10(sim_data["time_in_cell"].values + 1e-12), np.log10(sim_data["thickness"].values + 1e-12)
    return np.percentile(log_times, 0.000001), np.percentile(log_times, 99.999999), np.percentile(log_thickness, 0.000001), np.percentile(log_thickness, 99.999999)

def compute_normalization_params_2d(sim_data):
    log_times = np.log10(sim_data["time_in_cell"].values + 1e-12)
    log_width = np.log10(sim_data["width"].values + 1e-12)
    log_height = np.log10(sim_data["height"].values + 1e-12)
    return np.percentile(log_times, 0.000001), np.percentile(log_times, 99.999999), np.percentile(log_width, 0.000001), np.percentile(log_width, 99.999999), np.percentile(log_height, 0.000001), np.percentile(log_height, 99.999999)

def compute_normalization_params_2d_internal(sim_data): return compute_normalization_params_2d(sim_data)

def compute_normalization_params_3d(sim_data):
    log_times, log_width = np.log10(sim_data["time_in_cell"].values + 1e-12), np.log10(sim_data["width"].values + 1e-12)
    log_height, log_depth = np.log10(sim_data["height"].values + 1e-12), np.log10(sim_data["depth"].values + 1e-12)
    return np.percentile(log_times, 0.000001), np.percentile(log_times, 99.999999), np.percentile(log_width, 0.000001), np.percentile(log_width, 99.999999), np.percentile(log_height, 0.000001), np.percentile(log_height, 99.999999), np.percentile(log_depth, 0.000001), np.percentile(log_depth, 99.999999)

def compute_normalization_params_2d_timedep(sim_data, source_type='internal'):
    log_width, log_height = np.log10(sim_data["width"].values + 1e-12), np.log10(sim_data["height"].values + 1e-12)
    log_budget, log_L = np.log10(sim_data["path_budget"].values + 1e-12), np.log10(sim_data["L"].values + 1e-12)
    return np.percentile(log_width, 0.000001), np.percentile(log_width, 99.999999), np.percentile(log_height, 0.000001), np.percentile(log_height, 99.999999), np.percentile(log_budget, 0.000001), np.percentile(log_budget, 99.999999), np.percentile(log_L, 0.000001), np.percentile(log_L, 99.999999)

def compute_normalization_params_2d_triangle_timedep(sim_data, source_type='internal'):
    log_L, log_r = np.log10(sim_data["L_tri"].values + 1e-12), np.log10(sim_data["r_tri"].values + 1e-12)
    log_budget, log_L_track = np.log10(sim_data["path_budget"].values + 1e-12), np.log10(sim_data["L"].values + 1e-12)
    return np.percentile(log_L, 0.000001), np.percentile(log_L, 99.999999), np.percentile(log_r, 0.000001), np.percentile(log_r, 99.999999), np.percentile(log_budget, 0.000001), np.percentile(log_budget, 99.999999), np.percentile(log_L_track, 0.000001), np.percentile(log_L_track, 99.999999)

def compute_normalization_params_2d_quad_timedep(sim_data, source_type='internal'):
    log_L, log_y3, log_y4 = np.log10(sim_data["L_quad"].values + 1e-12), np.log10(sim_data["y3"].values + 1e-12), np.log10(sim_data["y4"].values + 1e-12)
    x3_vals, x4_vals = sim_data["x3"].values, sim_data["x4"].values
    log_budget, log_L_track = np.log10(sim_data["path_budget"].values + 1e-12), np.log10(sim_data["L_track"].values + 1e-12)
    return np.percentile(log_L, 0.000001), np.percentile(log_L, 99.999999), np.percentile(x3_vals, 0.000001), np.percentile(x3_vals, 99.999999), np.percentile(log_y3, 0.000001), np.percentile(log_y3, 99.999999), np.percentile(x4_vals, 0.000001), np.percentile(x4_vals, 99.999999), np.percentile(log_y4, 0.000001), np.percentile(log_y4, 99.999999), np.percentile(log_budget, 0.000001), np.percentile(log_budget, 99.999999), np.percentile(log_L_track, 0.000001), np.percentile(log_L_track, 99.999999)

def compute_normalization_params_3d_timedep(sim_data, source_type='internal'):
    log_width, log_height = np.log10(sim_data["width"].values + 1e-12), np.log10(sim_data["height"].values + 1e-12)
    log_depth, log_budget = np.log10(sim_data["depth"].values + 1e-12), np.log10(sim_data["path_budget"].values + 1e-12)
    log_L = np.log10(sim_data["L"].values + 1e-12)
    return np.percentile(log_width, 0.000001), np.percentile(log_width, 99.999999), np.percentile(log_height, 0.000001), np.percentile(log_height, 99.999999), np.percentile(log_depth, 0.000001), np.percentile(log_depth, 99.999999), np.percentile(log_budget, 0.000001), np.percentile(log_budget, 99.999999), np.percentile(log_L, 0.000001), np.percentile(log_L, 99.999999)

def compute_normalization_params_1d_timedep(sim_data, source_type='internal'):
    log_thickness, log_budget = np.log10(sim_data["thickness"].values + 1e-12), np.log10(sim_data["path_budget"].values + 1e-12)
    log_L = np.log10(sim_data["L"].values + 1e-12)
    return np.percentile(log_thickness, 0.000001), np.percentile(log_thickness, 99.999999), np.percentile(log_budget, 0.000001), np.percentile(log_budget, 99.999999), np.percentile(log_L, 0.000001), np.percentile(log_L, 99.999999)


def compute_normalization_params_2d_electron(sim_data):
    """Compute normalization parameters for 2D electron boundary transport."""
    log_E_in = np.log10(sim_data["E_in"].values + 1e-12)
    log_E_out = np.log10(np.maximum(sim_data["E_out"].values, 1e-12) + 1e-12)
    log_energy_dep = np.log10(np.maximum(sim_data["energy_deposited"].values, 1e-12) + 1e-12)
    log_path = np.log10(np.maximum(sim_data["path_length"].values, 1e-12) + 1e-12)

    E_in_lower = np.percentile(log_E_in, 0.01)
    E_in_upper = np.percentile(log_E_in, 99.99)
    E_out_lower = np.percentile(log_E_out, 0.01)
    E_out_upper = np.percentile(log_E_out, 99.99)
    energy_dep_lower = np.percentile(log_energy_dep, 0.01)
    energy_dep_upper = np.percentile(log_energy_dep, 99.99)
    path_lower = np.percentile(log_path, 0.01)
    path_upper = np.percentile(log_path, 99.99)

    return (
        E_in_lower, E_in_upper,
        E_out_lower, E_out_upper,
        energy_dep_lower, energy_dep_upper,
        path_lower, path_upper,
    )
# --- Generic, Type-Aware Normalization/Denormalization Functions ---
def _get_lib(x): return torch if isinstance(x, torch.Tensor) else np

_SQRT2 = math.sqrt(2.0)


def _probit_transform(x, eps=1e-6):
    if isinstance(x, torch.Tensor):
        x_clipped = torch.clamp(x, eps, 1.0 - eps)
        return _SQRT2 * torch.erfinv(2.0 * x_clipped - 1.0)
    x_arr = np.clip(x, eps, 1.0 - eps)
    return _SQRT2 * special.erfinv(2.0 * x_arr - 1.0)


def _inv_probit_transform(z):
    if isinstance(z, torch.Tensor):
        return 0.5 * (1.0 + torch.erf(z / _SQRT2))
    return 0.5 * (1.0 + special.erf(z / _SQRT2))


def normalize_log_scale_probit(value, lower, upper):
    return _probit_transform((_get_lib(value).log10(value + 1e-12) - lower) / (upper - lower))


def normalize_linear_scale_probit(value_01):
    return _probit_transform(value_01)


def denorm_log_scale_probit(probit_value, lower, upper):
    log_val = _inv_probit_transform(probit_value) * (upper - lower) + lower
    return torch.pow(10.0, log_val) if isinstance(probit_value, torch.Tensor) else 10**log_val


def denorm_linear_scale_probit(probit_value):
    return _inv_probit_transform(probit_value)


def robust_log_stats(values, eps=1e-12):
    y = np.log10(np.asarray(values, dtype=np.float64) + eps)
    med = float(np.median(y))
    iqr = float(np.percentile(y, 75.0) - np.percentile(y, 25.0))
    scale = max(iqr / 1.349, 1e-6)
    return med, scale


def normalize_log_scale_robust(value, center, scale):
    safe_scale = max(float(scale), 1e-12)
    return (_get_lib(value).log10(value + 1e-12) - center) / safe_scale


def denorm_log_scale_robust(z_value, center, scale):
    log_val = z_value * max(float(scale), 1e-12) + center
    return torch.pow(10.0, log_val) if isinstance(z_value, torch.Tensor) else 10**log_val


def normalize_log_feature(value, norm_params, key):
    center_key = f"{key}_center"
    scale_key = f"{key}_scale"
    if center_key in norm_params and scale_key in norm_params:
        return normalize_log_scale_robust(value, norm_params[center_key], norm_params[scale_key])
    lower_key = f"{key}_lower"
    upper_key = f"{key}_upper"
    return normalize_log_scale_logit(value, norm_params[lower_key], norm_params[upper_key])


def denorm_log_feature(z_value, norm_params, key):
    center_key = f"{key}_center"
    scale_key = f"{key}_scale"
    if center_key in norm_params and scale_key in norm_params:
        return denorm_log_scale_robust(z_value, norm_params[center_key], norm_params[scale_key])
    lower_key = f"{key}_lower"
    upper_key = f"{key}_upper"
    return denorm_log_scale_logit(z_value, norm_params[lower_key], norm_params[upper_key])


# Backwards-compatible names kept for older scripts.
def normalize_log_scale_logit(value, lower, upper): return normalize_log_scale_probit(value, lower, upper)
def normalize_linear_scale_logit(value_01): return normalize_linear_scale_probit(value_01)
def denorm_log_scale_logit(logit_value, lower, upper): return denorm_log_scale_probit(logit_value, lower, upper)
def denorm_linear_scale_logit(logit_value): return denorm_linear_scale_probit(logit_value)


def denorm_length_target(z_value, budget_value, norm_params, key="L"):
    if bool(norm_params.get("use_L_over_budget", False)):
        frac = denorm_linear_scale_logit(z_value)
        if isinstance(frac, torch.Tensor):
            return torch.clamp(frac, 0.0, 1.0) * budget_value
        return np.clip(frac, 0.0, 1.0) * budget_value
    return denorm_log_feature(z_value, norm_params, key)

# Format Helpers
def normalize_time(time_val, lower, upper): return normalize_log_scale_logit(time_val, lower, upper)
def denorm_time(logit_time, lower, upper): return denorm_log_scale_logit(logit_time, lower, upper)
def normalize_thickness(thickness, t_lower, t_upper): return normalize_log_scale_logit(thickness, t_lower, t_upper)
def denorm_thickness(logit_thickness, t_lower, t_upper): return denorm_log_scale_logit(logit_thickness, t_lower, t_upper)
def normalize_width(width, w_lower, w_upper): return normalize_log_scale_logit(width, w_lower, w_upper)
def denorm_width(logit_width, w_lower, w_upper): return denorm_log_scale_logit(logit_width, w_lower, w_upper)
def normalize_height(height, h_lower, h_upper): return normalize_log_scale_logit(height, h_lower, h_upper)
def denorm_height(logit_height, h_lower, h_upper): return denorm_log_scale_logit(logit_height, h_lower, h_upper)
def normalize_depth(depth, d_lower, d_upper): return normalize_log_scale_logit(depth, d_lower, d_upper)
def denorm_depth(logit_depth, d_lower, d_upper): return denorm_log_scale_logit(logit_depth, d_lower, d_upper)
def normalize_position_relative(pos_relative_01): return normalize_linear_scale_logit(pos_relative_01)
def denorm_position_relative(logit_pos_relative): return denorm_linear_scale_logit(logit_pos_relative)
def normalize_uv(uv_val): return normalize_linear_scale_logit((uv_val + 1.0) / 2.0)
def normalize_u_in(u_in_val): return normalize_linear_scale_logit(u_in_val)
def denorm_uv(logit_uv): return 2.0 * denorm_linear_scale_logit(logit_uv) - 1.0
def denorm_u_in(logit_u_in): return denorm_linear_scale_logit(logit_u_in)


def encode_direction_2d(u, v, direction_encoding_2d=None, *, norm_params=None):
    normalize_direction_encoding_2d(direction_encoding_2d, norm_params=norm_params)
    phi, r = uv_to_polar(u, v)
    return [normalize_angle_phi(phi), normalize_radius_r(r)]


def decode_direction_2d(features, direction_encoding_2d=None, *, norm_params=None):
    normalize_direction_encoding_2d(direction_encoding_2d, norm_params=norm_params)
    return polar_to_uv(denorm_angle_phi(features[0]), denorm_radius_r(features[1]))


def _angle_to_sincos(phi):
    return np.sin(phi), np.cos(phi)

# Polar & 3D Helpers
def uv_to_polar(u, v):
    lib = _get_lib(u)
    r = lib.sqrt(u**2 + v**2)
    phi = lib.arctan2(v, u)
    phi = torch.where(phi < 0, phi + 2.0 * np.pi, phi) if isinstance(phi, torch.Tensor) else np.where(phi < 0, phi + 2.0 * np.pi, phi)
    return phi, r

def polar_to_uv(phi, r):
    lib = _get_lib(phi)
    return r * lib.cos(phi), r * lib.sin(phi)

def normalize_angle_phi(phi): return normalize_linear_scale_logit(phi / (2.0 * np.pi))
def normalize_radius_r(r): return normalize_linear_scale_logit(r)
def denorm_angle_phi(logit_phi): return denorm_linear_scale_logit(logit_phi) * (2.0 * np.pi)
def denorm_radius_r(logit_r): 
    r = denorm_linear_scale_logit(logit_r)
    return torch.clamp(r, 0.0, 1.0) if isinstance(logit_r, torch.Tensor) else np.clip(r, 0.0, 1.0)

def uvw_to_axis_spherical_x(u, v, w, eps=1e-12):
    lib = _get_lib(u)
    mu = lib.clip(u, -1.0, 1.0)
    t = lib.sqrt(lib.clip(1.0 - mu * mu, 0.0, 1.0))
    phi = lib.arctan2(w, v)
    if isinstance(phi, torch.Tensor):
        phi = torch.where(phi < 0, phi + 2.0 * np.pi, phi)
        phi = torch.where(t < eps, torch.zeros_like(phi), phi)
    else:
        phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)
        phi = np.where(t < eps, 0.0, phi)
    return phi, mu

def axis_spherical_x_to_uvw(phi, mu):
    lib = _get_lib(phi)
    mu = lib.clip(mu, -1.0, 1.0)
    t = lib.sqrt(lib.clip(1.0 - mu * mu, 0.0, 1.0))
    return mu, t * lib.cos(phi), t * lib.sin(phi)

# --- Dataset ---
class SimDataset(Dataset):
    def __init__(self, sim_data, norm_params, data_type='1d_boundary'):
        self.sim_data = sim_data.reset_index(drop=True)
        self.norm_params = norm_params
        self.data_type = data_type.lower()
        self.t_lower, self.t_upper = norm_params.get('time_lower', 0), norm_params.get('time_upper', 1)
        self.use_sincos_circular = bool(norm_params.get('use_sincos_circular', False))
        self.direction_encoding_2d = normalize_direction_encoding_2d(norm_params=norm_params)
        self.direction_feature_dim_2d = direction_feature_count_2d(norm_params=norm_params)
        self.use_L_over_budget = bool(norm_params.get('use_L_over_budget', False))

        if self.data_type == '1d_boundary':
            self.cond_dim, self.target_dim = 2, 2
            self.thickness_lower, self.thickness_upper = norm_params['thickness_lower'], norm_params['thickness_upper']
        elif self.data_type == '1d_internal':
            self.cond_dim, self.target_dim = 3, 2
            self.thickness_lower, self.thickness_upper = norm_params['thickness_lower'], norm_params['thickness_upper']
        elif self.data_type == '2d_boundary':
            self.cond_dim, self.target_dim = 5, 5
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
        elif self.data_type == '2d_internal':
            self.cond_dim, self.target_dim = 6, 5
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']

        # -------------------------------------------------------------
        # TIME DEPENDENT DATASETS (target_dim += 1 for Explicit Token)
        # -------------------------------------------------------------
        elif self.data_type == '1d_timedep_internal':
            self.cond_dim, self.target_dim = 4, 4 # 3 physical + 1 exit token
            self.thickness_lower, self.thickness_upper = norm_params['thickness_lower'], norm_params['thickness_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '1d_timedep_boundary':
            self.cond_dim, self.target_dim = 3, 4 # 3 physical + 1 exit token
            self.thickness_lower, self.thickness_upper = norm_params['thickness_lower'], norm_params['thickness_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '2d_timedep_internal':
            dir_feats = self.direction_feature_dim_2d
            self.cond_dim = 5 + dir_feats
            self.target_dim = 4 + dir_feats
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '2d_timedep_boundary':
            dir_feats = self.direction_feature_dim_2d
            self.cond_dim = 4 + dir_feats
            self.target_dim = 4 + dir_feats
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '2d_triangle_timedep_internal':
            dir_feats = self.direction_feature_dim_2d
            self.cond_dim = 6 + dir_feats
            self.target_dim = 4 + dir_feats
            self.L_tri_lower, self.L_tri_upper = norm_params['L_tri_lower'], norm_params['L_tri_upper']
            self.r_tri_lower, self.r_tri_upper = norm_params['r_tri_lower'], norm_params['r_tri_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '2d_triangle_timedep_boundary':
            dir_feats = self.direction_feature_dim_2d
            self.cond_dim = 5 + dir_feats
            self.target_dim = 4 + dir_feats
            self.L_tri_lower, self.L_tri_upper = norm_params['L_tri_lower'], norm_params['L_tri_upper']
            self.r_tri_lower, self.r_tri_upper = norm_params['r_tri_lower'], norm_params['r_tri_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '2d_quad_timedep_internal':
            dir_feats = self.direction_feature_dim_2d
            self.cond_dim = 8 + dir_feats
            self.target_dim = 4 + dir_feats
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
            self.x3_lower, self.x3_upper = norm_params['x3_lower'], norm_params['x3_upper']
            self.y3_lower, self.y3_upper = norm_params['y3_lower'], norm_params['y3_upper']
            self.x4_lower, self.x4_upper = norm_params['x4_lower'], norm_params['x4_upper']
            self.y4_lower, self.y4_upper = norm_params['y4_lower'], norm_params['y4_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_track_lower, self.L_track_upper = norm_params['L_track_lower'], norm_params['L_track_upper']
        elif self.data_type == '3d_boundary':
            self.cond_dim, self.target_dim = 7, 6
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.depth_lower, self.depth_upper = norm_params['depth_lower'], norm_params['depth_upper']
        elif self.data_type == '3d_internal':
            self.cond_dim, self.target_dim = 8, 6
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.depth_lower, self.depth_upper = norm_params['depth_lower'], norm_params['depth_upper']
        elif self.data_type == '3d_timedep_internal':
            self.cond_dim = 10 if self.use_sincos_circular else 9
            self.target_dim = 8 if self.use_sincos_circular else 7
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.depth_lower, self.depth_upper = norm_params['depth_lower'], norm_params['depth_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == '3d_timedep_boundary':
            self.cond_dim = 9 if self.use_sincos_circular else 8
            self.target_dim = 8 if self.use_sincos_circular else 7
            self.width_lower, self.width_upper = norm_params['width_lower'], norm_params['width_upper']
            self.height_lower, self.height_upper = norm_params['height_lower'], norm_params['height_upper']
            self.depth_lower, self.depth_upper = norm_params['depth_lower'], norm_params['depth_upper']
            self.budget_lower, self.budget_upper = norm_params['budget_lower'], norm_params['budget_upper']
            self.L_lower, self.L_upper = norm_params['L_lower'], norm_params['L_upper']
        elif self.data_type == "2d_electron_boundary":
            self.cond_dim = 5
            self.target_dim = 7
            self.E_in_lower = norm_params.get("E_in_lower", 0.0)
            self.E_in_upper = norm_params.get("E_in_upper", 1.0)
            self.E_out_lower = norm_params.get("E_out_lower", 0.0)
            self.E_out_upper = norm_params.get("E_out_upper", 1.0)
            self.energy_dep_lower = norm_params.get("energy_dep_lower", -12.0)
            self.energy_dep_upper = norm_params.get("energy_dep_upper", 1.0)
            self.path_lower = norm_params.get("path_lower", 0.0)
            self.path_upper = norm_params.get("path_upper", 1.0)
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")

        self.input_dim = self.cond_dim + self.target_dim
        self._cached_x1 = None
        self._precompute_tensors()

    def __len__(self):
        return len(self.sim_data)

    def _get_exit_logit(self, row):
        """Helper to generate continuous target proxy for discrete Space vs Time exit."""
        return normalize_linear_scale_logit(torch.tensor(float(row["time_exit"]), dtype=torch.float32))

    def _encode_2d_direction(self, u_val, v_val):
        return encode_direction_2d(u_val, v_val, self.direction_encoding_2d, norm_params=self.norm_params)

    def _compute_item(self, idx_or_row):
        if isinstance(idx_or_row, int) or isinstance(idx_or_row, np.integer):
            row = self.sim_data.iloc[idx_or_row]
        else:
            row = idx_or_row

        if self.data_type == '2d_boundary':
            phi_in, r_in = uv_to_polar(row["initial_dir_u"], row["initial_dir_v"])
            phi_out, r_out = uv_to_polar(row["final_dir_u"], row["final_dir_v"])
            cond = torch.tensor([
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
                normalize_angle_phi(phi_out),
                normalize_radius_r(r_out),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time"),
            ], dtype=torch.float32)

        elif self.data_type == '2d_internal':
            phi_in, r_in = uv_to_polar(row["initial_dir_u"], row["initial_dir_v"])
            phi_out, r_out = uv_to_polar(row["final_dir_u"], row["final_dir_v"])
            cond = torch.tensor([
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_position_relative(row["initial_pos_x_norm"]),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
                normalize_angle_phi(phi_out),
                normalize_radius_r(r_out),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time"),
            ], dtype=torch.float32)

        elif self.data_type == '3d_boundary':
            cond = torch.tensor([
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_depth(row["depth"], self.depth_lower, self.depth_upper),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_position_relative(row["initial_pos_z_norm"]),
                normalize_angle_phi(uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])[0]),
                normalize_u_in(float(np.clip(uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])[1], 0.0, 1.0)))
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
                normalize_position_relative(row["final_pos_z_norm"]),
                normalize_angle_phi(uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])[0]),
                normalize_uv(float(np.clip(uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])[1], -1.0, 1.0))),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time")
            ], dtype=torch.float32)

        elif self.data_type == '3d_internal':
            cond = torch.tensor([
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_depth(row["depth"], self.depth_lower, self.depth_upper),
                normalize_position_relative(row["initial_pos_x_norm"]),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_position_relative(row["initial_pos_z_norm"]),
                normalize_angle_phi(uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])[0]),
                normalize_uv(float(np.clip(uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])[1], -1.0, 1.0)))
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
                normalize_position_relative(row["final_pos_z_norm"]),
                normalize_angle_phi(uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])[0]),
                normalize_uv(float(np.clip(uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])[1], -1.0, 1.0))),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time")
            ], dtype=torch.float32)

        elif self.data_type == '1d_internal':
            cond = torch.tensor([
                normalize_thickness(row["thickness"], self.thickness_lower, self.thickness_upper),
                normalize_position_relative(row["initial_pos_x_norm"]),
                normalize_uv(row["initial_dir_u"])
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_uv(row["final_dir_u"]),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time")
            ], dtype=torch.float32)

        elif self.data_type == '1d_boundary':
            cond = torch.tensor([
                normalize_thickness(row["thickness"], self.thickness_lower, self.thickness_upper),
                normalize_u_in(row["initial_dir_u"])
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_uv(row["final_dir_u"]),
                normalize_log_feature(row["time_in_cell"], self.norm_params, "time")
            ], dtype=torch.float32)

        # -------------------------------------------------------------
        # TIME-DEPENDENT DATASETS (Target Dim + 1 for Explicit Indicator)
        # -------------------------------------------------------------
        elif self.data_type == '1d_timedep_internal':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            cond = torch.tensor([
                normalize_thickness(row["thickness"], self.thickness_lower, self.thickness_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["x_in_norm"]),
                normalize_uv(row["mu_in"])
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["x_out_norm"]),
                normalize_uv(row["mu_out"]),
                L_norm,
                self._get_exit_logit(row) # NEW INDICATOR
            ], dtype=torch.float32)

        elif self.data_type == '1d_timedep_boundary':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            cond = torch.tensor([
                normalize_thickness(row["thickness"], self.thickness_lower, self.thickness_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_u_in(row["mu_in"])
            ], dtype=torch.float32)
            target = torch.tensor([
                normalize_position_relative(row["x_out_norm"]),
                normalize_uv(row["mu_out"]),
                L_norm,
                self._get_exit_logit(row) # NEW INDICATOR
            ], dtype=torch.float32)

        elif self.data_type == '2d_timedep_internal':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            cond_vals = [
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["initial_pos_x_norm"]),
                normalize_position_relative(row["initial_pos_y_norm"]),
            ] + self._encode_2d_direction(row["initial_dir_u"], row["initial_dir_v"])
            target_vals = [
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
            ] + self._encode_2d_direction(row["final_dir_u"], row["final_dir_v"]) + [
                L_norm,
                self._get_exit_logit(row),
            ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        elif self.data_type == '2d_timedep_boundary':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            cond_vals = [
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["initial_pos_y_norm"]),
            ] + self._encode_2d_direction(row["initial_dir_u"], row["initial_dir_v"])
            target_vals = [
                normalize_position_relative(row["final_pos_x_norm"]),
                normalize_position_relative(row["final_pos_y_norm"]),
            ] + self._encode_2d_direction(row["final_dir_u"], row["final_dir_v"]) + [
                L_norm,
                self._get_exit_logit(row),
            ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        elif self.data_type == '2d_triangle_timedep_internal':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            base_cond = [
                normalize_log_scale_logit(row["L_tri"], self.L_tri_lower, self.L_tri_upper),
                normalize_log_scale_logit(row["r_tri"], self.r_tri_lower, self.r_tri_upper),
                normalize_linear_scale_logit(row["theta"] / np.pi),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["xi1_in"]),
                normalize_position_relative(row["xi2_in"]),
            ]
            cond_vals = base_cond + self._encode_2d_direction(row["u_in"], row["v_in"])
            target_vals = [
                normalize_position_relative(row["xi1_out"]),
                normalize_position_relative(row["xi2_out"]),
            ] + self._encode_2d_direction(row["u_out"], row["v_out"]) + [
                L_norm,
                self._get_exit_logit(row),
            ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        elif self.data_type == '2d_triangle_timedep_boundary':
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            base_cond = [
                normalize_log_scale_logit(row["L_tri"], self.L_tri_lower, self.L_tri_upper),
                normalize_log_scale_logit(row["r_tri"], self.r_tri_lower, self.r_tri_upper),
                normalize_linear_scale_logit(row["theta"] / np.pi),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["xi1_in"]),
            ]
            cond_vals = base_cond + self._encode_2d_direction(row["u_in"], row["v_in"])
            target_vals = [
                normalize_position_relative(row["xi1_out"]),
                normalize_position_relative(row["xi2_out"]),
            ] + self._encode_2d_direction(row["u_out"], row["v_out"]) + [
                L_norm,
                self._get_exit_logit(row),
            ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        elif self.data_type == '2d_quad_timedep_internal':
            L_ratio = min(max(float(row["L_track"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L_track"], self.norm_params, "L_track")
            base_cond = [
                normalize_log_scale_logit(row["L_quad"], self.L_lower, self.L_upper),
                normalize_linear_scale_logit(min(max((row["x3"] - self.x3_lower) / (self.x3_upper - self.x3_lower + 1e-12), 0.0), 1.0)),
                normalize_log_scale_logit(row["y3"], self.y3_lower, self.y3_upper),
                normalize_linear_scale_logit(min(max((row["x4"] - self.x4_lower) / (self.x4_upper - self.x4_lower + 1e-12), 0.0), 1.0)),
                normalize_log_scale_logit(row["y4"], self.y4_lower, self.y4_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["u_in"]),
                normalize_position_relative(row["v_in"]),
            ]
            cond_vals = base_cond + self._encode_2d_direction(row["dir_u_in"], row["dir_v_in"])
            target_vals = [
                normalize_position_relative(row["u_out"]),
                normalize_position_relative(row["v_out"]),
            ] + self._encode_2d_direction(row["dir_u_out"], row["dir_v_out"]) + [
                L_norm,
                self._get_exit_logit(row),
            ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        elif self.data_type == '3d_timedep_internal':
            phi_in, mu_in = uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])
            phi_out, mu_out = uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            base_cond = [
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_depth(row["depth"], self.depth_lower, self.depth_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["initial_pos_x_norm"]),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_position_relative(row["initial_pos_z_norm"]),
            ]
            if self.use_sincos_circular:
                sin_phi_in, cos_phi_in = _angle_to_sincos(float(phi_in))
                sin_phi_out, cos_phi_out = _angle_to_sincos(float(phi_out))
                cond_vals = base_cond + [sin_phi_in, cos_phi_in, normalize_uv(float(np.clip(mu_in, -1.0, 1.0)))]
                target_vals = [
                    normalize_position_relative(row["final_pos_x_norm"]),
                    normalize_position_relative(row["final_pos_y_norm"]),
                    normalize_position_relative(row["final_pos_z_norm"]),
                    sin_phi_out,
                    cos_phi_out,
                    normalize_uv(float(np.clip(mu_out, -1.0, 1.0))),
                    L_norm,
                    self._get_exit_logit(row),
                ]
            else:
                cond_vals = base_cond + [normalize_angle_phi(phi_in), normalize_uv(float(np.clip(mu_in, -1.0, 1.0)))]
                target_vals = [
                    normalize_position_relative(row["final_pos_x_norm"]),
                    normalize_position_relative(row["final_pos_y_norm"]),
                    normalize_position_relative(row["final_pos_z_norm"]),
                    normalize_angle_phi(phi_out),
                    normalize_uv(float(np.clip(mu_out, -1.0, 1.0))),
                    L_norm,
                    self._get_exit_logit(row),
                ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)
        elif self.data_type == "2d_electron_boundary":
            cond = torch.tensor([
                normalize_position_relative(row["y_in_norm"]),
                normalize_uv(row["u_in"]),
                normalize_uv(row["v_in"]),
                normalize_log_scale_probit(row["E_in"], self.E_in_lower, self.E_in_upper),
                normalize_position_relative(row["x_in_norm"]),
            ], dtype=torch.float32)

            target = torch.tensor([
                normalize_position_relative(row["x_out_norm"]),
                normalize_position_relative(row["y_out_norm"]),
                normalize_uv(row["u_out"]),
                normalize_uv(row["v_out"]),
                normalize_log_scale_probit(row["E_out"], self.E_out_lower, self.E_out_upper),
                normalize_log_scale_probit(row["energy_deposited"] + 1e-12, self.energy_dep_lower,
                                           self.energy_dep_upper),
                normalize_log_scale_probit(row["path_length"], self.path_lower, self.path_upper),
            ], dtype=torch.float32)
        elif self.data_type == '3d_timedep_boundary':
            phi_in, mu_in = uvw_to_axis_spherical_x(row["initial_dir_u"], row["initial_dir_v"], row["initial_dir_w"])
            phi_out, mu_out = uvw_to_axis_spherical_x(row["final_dir_u"], row["final_dir_v"], row["final_dir_w"])
            L_ratio = min(max(float(row["L"]) / max(float(row["path_budget"]), 1e-12), 0.0), 1.0)
            L_norm = normalize_linear_scale_logit(L_ratio) if self.use_L_over_budget else normalize_log_feature(row["L"], self.norm_params, "L")
            base_cond = [
                normalize_width(row["width"], self.width_lower, self.width_upper),
                normalize_height(row["height"], self.height_lower, self.height_upper),
                normalize_depth(row["depth"], self.depth_lower, self.depth_upper),
                normalize_log_scale_logit(row["path_budget"], self.budget_lower, self.budget_upper),
                normalize_position_relative(row["initial_pos_y_norm"]),
                normalize_position_relative(row["initial_pos_z_norm"]),
            ]
            if self.use_sincos_circular:
                sin_phi_in, cos_phi_in = _angle_to_sincos(float(phi_in))
                sin_phi_out, cos_phi_out = _angle_to_sincos(float(phi_out))
                cond_vals = base_cond + [sin_phi_in, cos_phi_in, normalize_u_in(float(np.clip(mu_in, 0.0, 1.0)))]
                target_vals = [
                    normalize_position_relative(row["final_pos_x_norm"]),
                    normalize_position_relative(row["final_pos_y_norm"]),
                    normalize_position_relative(row["final_pos_z_norm"]),
                    sin_phi_out,
                    cos_phi_out,
                    normalize_uv(float(np.clip(mu_out, -1.0, 1.0))),
                    L_norm,
                    self._get_exit_logit(row),
                ]
            else:
                cond_vals = base_cond + [normalize_angle_phi(phi_in), normalize_u_in(float(np.clip(mu_in, 0.0, 1.0)))]
                target_vals = [
                    normalize_position_relative(row["final_pos_x_norm"]),
                    normalize_position_relative(row["final_pos_y_norm"]),
                    normalize_position_relative(row["final_pos_z_norm"]),
                    normalize_angle_phi(phi_out),
                    normalize_uv(float(np.clip(mu_out, -1.0, 1.0))),
                    L_norm,
                    self._get_exit_logit(row),
                ]
            cond = torch.tensor(cond_vals, dtype=torch.float32)
            target = torch.tensor(target_vals, dtype=torch.float32)

        else:
            raise NotImplementedError(f"__getitem__ not implemented for {self.data_type}")

        if cond.numel() != self.cond_dim: raise ValueError(f"cond dim mismatch: expected {self.cond_dim}, got {cond.numel()}")
        if target.numel() != self.target_dim: raise ValueError(f"target dim mismatch: expected {self.target_dim}, got {target.numel()}")

        return torch.cat([cond, target]), cond, target

    def _precompute_tensors(self):
        """Precompute normalized model inputs once to remove per-sample Python overhead."""
        total_samples = len(self.sim_data)
        if total_samples == 0:
            self._cached_x1 = torch.empty((0, self.input_dim), dtype=torch.float32)
            return

        print(f"Precomputing dataset tensors for {total_samples} samples (vectorized)...")
        df = self.sim_data

        def col(name: str) -> np.ndarray:
            return df[name].to_numpy(dtype=np.float64, copy=False)

        def as_tensor(*cols: np.ndarray) -> torch.Tensor:
            arr = np.stack(cols, axis=1).astype(np.float32, copy=False)
            return torch.from_numpy(np.ascontiguousarray(arr))

        def stack_ct(cond_cols, target_cols):
            cond_t = as_tensor(*cond_cols)
            target_t = as_tensor(*target_cols)
            return torch.cat([cond_t, target_t], dim=1).contiguous()

        if self.data_type == '1d_boundary':
            cond_cols = [
                normalize_thickness(col("thickness"), self.thickness_lower, self.thickness_upper),
                normalize_u_in(col("initial_dir_u")),
            ]
            target_cols = [
                normalize_uv(col("final_dir_u")),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '1d_internal':
            cond_cols = [
                normalize_thickness(col("thickness"), self.thickness_lower, self.thickness_upper),
                normalize_position_relative(col("initial_pos_x_norm")),
                normalize_uv(col("initial_dir_u")),
            ]
            target_cols = [
                normalize_uv(col("final_dir_u")),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_boundary':
            phi_in, r_in = uv_to_polar(col("initial_dir_u"), col("initial_dir_v"))
            phi_out, r_out = uv_to_polar(col("final_dir_u"), col("final_dir_v"))
            cond_cols = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ]
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
                normalize_angle_phi(phi_out),
                normalize_radius_r(r_out),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_internal':
            phi_in, r_in = uv_to_polar(col("initial_dir_u"), col("initial_dir_v"))
            phi_out, r_out = uv_to_polar(col("final_dir_u"), col("final_dir_v"))
            cond_cols = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_position_relative(col("initial_pos_x_norm")),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ]
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
                normalize_angle_phi(phi_out),
                normalize_radius_r(r_out),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '3d_boundary':
            phi_in, mu_in = uvw_to_axis_spherical_x(col("initial_dir_u"), col("initial_dir_v"), col("initial_dir_w"))
            phi_out, mu_out = uvw_to_axis_spherical_x(col("final_dir_u"), col("final_dir_v"), col("final_dir_w"))
            cond_cols = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_depth(col("depth"), self.depth_lower, self.depth_upper),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_position_relative(col("initial_pos_z_norm")),
                normalize_angle_phi(phi_in),
                normalize_u_in(np.clip(mu_in, 0.0, 1.0)),
            ]
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
                normalize_position_relative(col("final_pos_z_norm")),
                normalize_angle_phi(phi_out),
                normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '3d_internal':
            phi_in, mu_in = uvw_to_axis_spherical_x(col("initial_dir_u"), col("initial_dir_v"), col("initial_dir_w"))
            phi_out, mu_out = uvw_to_axis_spherical_x(col("final_dir_u"), col("final_dir_v"), col("final_dir_w"))
            cond_cols = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_depth(col("depth"), self.depth_lower, self.depth_upper),
                normalize_position_relative(col("initial_pos_x_norm")),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_position_relative(col("initial_pos_z_norm")),
                normalize_angle_phi(phi_in),
                normalize_uv(np.clip(mu_in, -1.0, 1.0)),
            ]
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
                normalize_position_relative(col("final_pos_z_norm")),
                normalize_angle_phi(phi_out),
                normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                normalize_log_feature(col("time_in_cell"), self.norm_params, "time"),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '1d_timedep_internal':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")
            cond_cols = [
                normalize_thickness(col("thickness"), self.thickness_lower, self.thickness_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("x_in_norm")),
                normalize_uv(col("mu_in")),
            ]
            target_cols = [
                normalize_position_relative(col("x_out_norm")),
                normalize_uv(col("mu_out")),
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '1d_timedep_boundary':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")
            cond_cols = [
                normalize_thickness(col("thickness"), self.thickness_lower, self.thickness_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_u_in(col("mu_in")),
            ]
            target_cols = [
                normalize_position_relative(col("x_out_norm")),
                normalize_uv(col("mu_out")),
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_timedep_internal':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("initial_pos_x_norm")),
                normalize_position_relative(col("initial_pos_y_norm")),
            ]
            cond_cols = base_cond + self._encode_2d_direction(col("initial_dir_u"), col("initial_dir_v"))
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
            ] + self._encode_2d_direction(col("final_dir_u"), col("final_dir_v")) + [
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_timedep_boundary':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("initial_pos_y_norm")),
            ]
            cond_cols = base_cond + self._encode_2d_direction(col("initial_dir_u"), col("initial_dir_v"))
            target_cols = [
                normalize_position_relative(col("final_pos_x_norm")),
                normalize_position_relative(col("final_pos_y_norm")),
            ] + self._encode_2d_direction(col("final_dir_u"), col("final_dir_v")) + [
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_triangle_timedep_internal':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_log_scale_logit(col("L_tri"), self.L_tri_lower, self.L_tri_upper),
                normalize_log_scale_logit(col("r_tri"), self.r_tri_lower, self.r_tri_upper),
                normalize_linear_scale_logit(col("theta") / np.pi),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("xi1_in")),
                normalize_position_relative(col("xi2_in")),
            ]
            cond_cols = base_cond + self._encode_2d_direction(col("u_in"), col("v_in"))
            target_cols = [
                normalize_position_relative(col("xi1_out")),
                normalize_position_relative(col("xi2_out")),
            ] + self._encode_2d_direction(col("u_out"), col("v_out")) + [
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_triangle_timedep_boundary':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_log_scale_logit(col("L_tri"), self.L_tri_lower, self.L_tri_upper),
                normalize_log_scale_logit(col("r_tri"), self.r_tri_lower, self.r_tri_upper),
                normalize_linear_scale_logit(col("theta") / np.pi),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("xi1_in")),
            ]
            cond_cols = base_cond + self._encode_2d_direction(col("u_in"), col("v_in"))
            target_cols = [
                normalize_position_relative(col("xi1_out")),
                normalize_position_relative(col("xi2_out")),
            ] + self._encode_2d_direction(col("u_out"), col("v_out")) + [
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '2d_quad_timedep_internal':
            path_budget = col("path_budget")
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L_track") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L_track"), self.norm_params, "L_track")

            x3_01 = np.clip((col("x3") - self.x3_lower) / (self.x3_upper - self.x3_lower + 1e-12), 0.0, 1.0)
            x4_01 = np.clip((col("x4") - self.x4_lower) / (self.x4_upper - self.x4_lower + 1e-12), 0.0, 1.0)

            base_cond = [
                normalize_log_scale_logit(col("L_quad"), self.L_lower, self.L_upper),
                normalize_linear_scale_logit(x3_01),
                normalize_log_scale_logit(col("y3"), self.y3_lower, self.y3_upper),
                normalize_linear_scale_logit(x4_01),
                normalize_log_scale_logit(col("y4"), self.y4_lower, self.y4_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("u_in")),
                normalize_position_relative(col("v_in")),
            ]
            cond_cols = base_cond + self._encode_2d_direction(col("dir_u_in"), col("dir_v_in"))
            target_cols = [
                normalize_position_relative(col("u_out")),
                normalize_position_relative(col("v_out")),
            ] + self._encode_2d_direction(col("dir_u_out"), col("dir_v_out")) + [
                L_norm,
                normalize_linear_scale_logit(col("time_exit")),
            ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == '3d_timedep_internal':
            path_budget = col("path_budget")
            phi_in, mu_in = uvw_to_axis_spherical_x(col("initial_dir_u"), col("initial_dir_v"), col("initial_dir_w"))
            phi_out, mu_out = uvw_to_axis_spherical_x(col("final_dir_u"), col("final_dir_v"), col("final_dir_w"))
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_depth(col("depth"), self.depth_lower, self.depth_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("initial_pos_x_norm")),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_position_relative(col("initial_pos_z_norm")),
            ]
            if self.use_sincos_circular:
                cond_cols = base_cond + [np.sin(phi_in), np.cos(phi_in), normalize_uv(np.clip(mu_in, -1.0, 1.0))]
                target_cols = [
                    normalize_position_relative(col("final_pos_x_norm")),
                    normalize_position_relative(col("final_pos_y_norm")),
                    normalize_position_relative(col("final_pos_z_norm")),
                    np.sin(phi_out),
                    np.cos(phi_out),
                    normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                    L_norm,
                    normalize_linear_scale_logit(col("time_exit")),
                ]
            else:
                cond_cols = base_cond + [normalize_angle_phi(phi_in), normalize_uv(np.clip(mu_in, -1.0, 1.0))]
                target_cols = [
                    normalize_position_relative(col("final_pos_x_norm")),
                    normalize_position_relative(col("final_pos_y_norm")),
                    normalize_position_relative(col("final_pos_z_norm")),
                    normalize_angle_phi(phi_out),
                    normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                    L_norm,
                    normalize_linear_scale_logit(col("time_exit")),
                ]
            x1 = stack_ct(cond_cols, target_cols)

        elif self.data_type == "2d_electron_boundary":
            cond_cols = [
                normalize_position_relative(col("y_in_norm")),
                normalize_uv(col("u_in")),
                normalize_uv(col("v_in")),
                normalize_log_scale_probit(col("E_in"), self.E_in_lower, self.E_in_upper),
                normalize_position_relative(col("x_in_norm")),
            ]
            target_cols = [
                normalize_position_relative(col("x_out_norm")),
                normalize_position_relative(col("y_out_norm")),
                normalize_uv(col("u_out")),
                normalize_uv(col("v_out")),
                normalize_log_scale_probit(col("E_out"), self.E_out_lower, self.E_out_upper),
                normalize_log_scale_probit(col("energy_deposited") + 1e-12, self.energy_dep_lower, self.energy_dep_upper),
                normalize_log_scale_probit(col("path_length"), self.path_lower, self.path_upper),
            ]
            x1 = stack_ct(cond_cols, target_cols)
        elif self.data_type == '3d_timedep_boundary':
            path_budget = col("path_budget")
            phi_in, mu_in = uvw_to_axis_spherical_x(col("initial_dir_u"), col("initial_dir_v"), col("initial_dir_w"))
            phi_out, mu_out = uvw_to_axis_spherical_x(col("final_dir_u"), col("final_dir_v"), col("final_dir_w"))
            if self.use_L_over_budget:
                L_ratio = np.clip(col("L") / np.maximum(path_budget, 1e-12), 0.0, 1.0)
                L_norm = normalize_linear_scale_logit(L_ratio)
            else:
                L_norm = normalize_log_feature(col("L"), self.norm_params, "L")

            base_cond = [
                normalize_width(col("width"), self.width_lower, self.width_upper),
                normalize_height(col("height"), self.height_lower, self.height_upper),
                normalize_depth(col("depth"), self.depth_lower, self.depth_upper),
                normalize_log_scale_logit(path_budget, self.budget_lower, self.budget_upper),
                normalize_position_relative(col("initial_pos_y_norm")),
                normalize_position_relative(col("initial_pos_z_norm")),
            ]
            if self.use_sincos_circular:
                cond_cols = base_cond + [np.sin(phi_in), np.cos(phi_in), normalize_u_in(np.clip(mu_in, 0.0, 1.0))]
                target_cols = [
                    normalize_position_relative(col("final_pos_x_norm")),
                    normalize_position_relative(col("final_pos_y_norm")),
                    normalize_position_relative(col("final_pos_z_norm")),
                    np.sin(phi_out),
                    np.cos(phi_out),
                    normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                    L_norm,
                    normalize_linear_scale_logit(col("time_exit")),
                ]
            else:
                cond_cols = base_cond + [normalize_angle_phi(phi_in), normalize_u_in(np.clip(mu_in, 0.0, 1.0))]
                target_cols = [
                    normalize_position_relative(col("final_pos_x_norm")),
                    normalize_position_relative(col("final_pos_y_norm")),
                    normalize_position_relative(col("final_pos_z_norm")),
                    normalize_angle_phi(phi_out),
                    normalize_uv(np.clip(mu_out, -1.0, 1.0)),
                    L_norm,
                    normalize_linear_scale_logit(col("time_exit")),
                ]
            x1 = stack_ct(cond_cols, target_cols)

        else:
            raise NotImplementedError(f"Precompute not implemented for {self.data_type}")

        if x1.shape[1] != self.input_dim:
            raise ValueError(f"input dim mismatch: expected {self.input_dim}, got {x1.shape[1]}")
        self._cached_x1 = x1

    def __getitem__(self, idx):
        if self._cached_x1 is None:
            return self._compute_item(idx)

        x1 = self._cached_x1[idx]
        return x1, x1[:self.cond_dim], x1[self.cond_dim:]
