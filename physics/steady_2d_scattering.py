"""Shared 2D steady-state scattering helpers.

The transport is conceptually 3D: particles carry a unit 3D direction, while
the 2D workflows use only the projected x-y components for geometry and ML
features. This module centralizes the direction reorientation rules and the
2D steady scattering-mode tagging used by model/norm assets.
"""

from __future__ import annotations

import math

from numba import njit


STEADY_2D_SCATTERING_ISOTROPIC = "isotropic"
STEADY_2D_SCATTERING_ANISOTROPIC = "anisotropic"
STEADY_2D_SCATTERING_TAG = "steady2d-v2"
STEADY_2D_ANISOTROPIC_ASSET_ROOT = "ml/steady_2d_anisotropic"


def normalize_steady_2d_scattering_mode(scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = str(scattering_mode).strip().lower()
    if mode in {STEADY_2D_SCATTERING_ISOTROPIC, STEADY_2D_SCATTERING_ANISOTROPIC}:
        return mode
    raise ValueError(
        f"Unsupported scattering_mode='{scattering_mode}'. "
        f"Expected '{STEADY_2D_SCATTERING_ISOTROPIC}' or '{STEADY_2D_SCATTERING_ANISOTROPIC}'."
    )


def steady_2d_scattering_mode_code(scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> int:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    return 0 if mode == STEADY_2D_SCATTERING_ISOTROPIC else 1


def steady_2d_scattering_asset_tag(scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return ""
    return f"_{STEADY_2D_SCATTERING_TAG}_{mode}"


def _steady_2d_legacy_model_dir(source_type: str, scattering_mode: str) -> str:
    return f"ml/models/2d_{source_type}{steady_2d_scattering_asset_tag(scattering_mode)}"


def _steady_2d_legacy_training_dir(source_type: str, scattering_mode: str) -> str:
    return f"ml/training_results/2d_{source_type}{steady_2d_scattering_asset_tag(scattering_mode)}"


def _steady_2d_legacy_norm_params_base_path(source_type: str, scattering_mode: str) -> str:
    return f"ml/data/norm_params_2d_{source_type}{steady_2d_scattering_asset_tag(scattering_mode)}.npy"


def steady_2d_model_dir(source_type: str, scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_model_dir(source_type, mode)
    return f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/models/2d_{source_type}"


def steady_2d_training_dir(source_type: str, scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_training_dir(source_type, mode)
    return f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/training_results/2d_{source_type}"


def steady_2d_norm_params_base_path(source_type: str, scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_norm_params_base_path(source_type, mode)
    return f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/data/norm_params_2d_{source_type}.npy"


def steady_2d_model_dir_candidates(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
) -> tuple[str, ...]:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    primary = steady_2d_model_dir(source_type, scattering_mode=mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return (primary,)
    return (primary, _steady_2d_legacy_model_dir(source_type, mode))


def steady_2d_norm_params_base_path_candidates(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
) -> tuple[str, ...]:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    primary = steady_2d_norm_params_base_path(source_type, scattering_mode=mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return (primary,)
    return (primary, _steady_2d_legacy_norm_params_base_path(source_type, mode))


@njit(inline="always")
def guard_projected_direction_xy(u: float, v: float, eps: float):
    if abs(u) + abs(v) < eps:
        return eps, 0.0
    return u, v


@njit(inline="always")
def normalize_direction_3d(u: float, v: float, w: float, eps: float):
    norm = math.sqrt(u * u + v * v + w * w)
    if norm < eps:
        return 1.0, 0.0, 0.0
    inv = 1.0 / norm
    return u * inv, v * inv, w * inv


@njit(inline="always")
def sample_isotropic_direction_3d_from_unit(unit_mu_z: float, unit_phi: float, eps: float):
    mu_z = 2.0 * unit_mu_z - 1.0
    phi = 2.0 * math.pi * unit_phi
    t = math.sqrt(max(0.0, 1.0 - mu_z * mu_z))
    u = t * math.cos(phi)
    v = t * math.sin(phi)
    w = mu_z
    return normalize_direction_3d(u, v, w, eps)


@njit(inline="always")
def sample_incoming_hemisphere_plus_x_direction_3d_from_unit(unit_mu_x: float, unit_phi: float, eps: float):
    u = unit_mu_x
    if u < eps:
        u = eps
    phi = 2.0 * math.pi * unit_phi
    t = math.sqrt(max(0.0, 1.0 - u * u))
    v = t * math.cos(phi)
    w = t * math.sin(phi)
    return normalize_direction_3d(u, v, w, eps)


@njit(inline="always")
def sample_anisotropic_mu_from_unit(unit_x: float):
    return 2.0 * math.sqrt(unit_x) - 1.0


@njit(inline="always")
def sample_scattered_direction_3d_from_unit(
    current_u: float,
    current_v: float,
    current_w: float,
    unit_mu: float,
    unit_phi: float,
    scattering_mode_code: int,
    eps: float,
):
    if scattering_mode_code == 0:
        return sample_isotropic_direction_3d_from_unit(unit_mu, unit_phi, eps)

    du, dv, dw = normalize_direction_3d(current_u, current_v, current_w, eps)
    mu = sample_anisotropic_mu_from_unit(unit_mu)
    phi = 2.0 * math.pi * unit_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

    if abs(dw) < 0.999:
        ref_x, ref_y, ref_z = 0.0, 0.0, 1.0
    else:
        ref_x, ref_y, ref_z = 0.0, 1.0, 0.0

    e1_x = ref_y * dw - ref_z * dv
    e1_y = ref_z * du - ref_x * dw
    e1_z = ref_x * dv - ref_y * du
    e1_x, e1_y, e1_z = normalize_direction_3d(e1_x, e1_y, e1_z, eps)

    e2_x = dv * e1_z - dw * e1_y
    e2_y = dw * e1_x - du * e1_z
    e2_z = du * e1_y - dv * e1_x

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    out_u = mu * du + sin_theta * (cos_phi * e1_x + sin_phi * e2_x)
    out_v = mu * dv + sin_theta * (cos_phi * e1_y + sin_phi * e2_y)
    out_w = mu * dw + sin_theta * (cos_phi * e1_z + sin_phi * e2_z)
    return normalize_direction_3d(out_u, out_v, out_w, eps)
