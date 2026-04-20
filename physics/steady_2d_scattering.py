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
STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR = "anisotropic-linear"
STEADY_2D_SCATTERING_ANISOTROPIC_HG = "anisotropic-hg"
STEADY_2D_SCATTERING_TAG = "steady2d-v2"
STEADY_2D_SCATTERING_ASSET_ROOT = "ml/steady_2d_scattering"
STEADY_2D_ANISOTROPIC_ASSET_ROOT = f"{STEADY_2D_SCATTERING_ASSET_ROOT}/anisotropic_linear"
STEADY_2D_HG_ASSET_ROOT = f"{STEADY_2D_SCATTERING_ASSET_ROOT}/anisotropic_hg"
STEADY_2D_HG_G_LIMIT = 0.999


def normalize_steady_2d_scattering_mode(scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> str:
    mode = str(scattering_mode).strip().lower()
    # Legacy compatibility: "anisotropic" maps to "anisotropic-linear"
    if mode == "anisotropic" or mode == STEADY_2D_SCATTERING_ANISOTROPIC:
        return STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR
    if mode in {STEADY_2D_SCATTERING_ISOTROPIC, 
                STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR, STEADY_2D_SCATTERING_ANISOTROPIC_HG}:
        return mode
    raise ValueError(
        f"Unsupported scattering_mode='{scattering_mode}'. "
        f"Expected '{STEADY_2D_SCATTERING_ISOTROPIC}', '{STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR}', "
        f"or '{STEADY_2D_SCATTERING_ANISOTROPIC_HG}'."
    )


def steady_2d_scattering_mode_code(scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC) -> int:
    """Convert scattering mode to integer code for JIT kernels.
    
    Returns:
        0 = isotropic
        1 = anisotropic-linear (forward-peaked with mu = 2*sqrt(u) - 1)
        2 = anisotropic-hg (Henyey-Greenstein with parameter g)
    """
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return 0
    elif mode == STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR:
        return 1
    elif mode == STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        return 2
    else:
        return 0  # Fallback to isotropic


def _steady_2d_hg_storage_component(g_hg: float) -> str:
    sign = "p" if g_hg >= 0.0 else "m"
    magnitude = f"{abs(float(g_hg)):.6f}".replace(".", "_")
    return f"g_{sign}{magnitude}"


def steady_2d_scattering_asset_tag(
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return ""
    if mode == STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        resolved_g = resolve_steady_2d_hg_parameter(mode, g_hg)
        return f"_{STEADY_2D_SCATTERING_TAG}_{mode}_{_steady_2d_hg_storage_component(resolved_g)}"
    return f"_{STEADY_2D_SCATTERING_TAG}_{mode}"


def _steady_2d_scattering_legacy_asset_tag(
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return ""
    if mode == STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR:
        return f"_{STEADY_2D_SCATTERING_TAG}_anisotropic"
    return steady_2d_scattering_asset_tag(mode)


def steady_2d_scattering_asset_root(
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR:
        return STEADY_2D_ANISOTROPIC_ASSET_ROOT
    if mode == STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        resolved_g = resolve_steady_2d_hg_parameter(mode, g_hg)
        return f"{STEADY_2D_HG_ASSET_ROOT}/{_steady_2d_hg_storage_component(resolved_g)}"
    raise ValueError(f"No dedicated asset root for scattering_mode='{mode}'.")


def _steady_2d_legacy_model_dir(source_type: str, scattering_mode: str) -> str:
    return f"ml/models/2d_{source_type}{_steady_2d_scattering_legacy_asset_tag(scattering_mode)}"


def _steady_2d_legacy_training_dir(source_type: str, scattering_mode: str) -> str:
    return f"ml/training_results/2d_{source_type}{_steady_2d_scattering_legacy_asset_tag(scattering_mode)}"


def _steady_2d_legacy_norm_params_base_path(source_type: str, scattering_mode: str) -> str:
    return f"ml/data/norm_params_2d_{source_type}{_steady_2d_scattering_legacy_asset_tag(scattering_mode)}.npy"


def steady_2d_model_dir(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_model_dir(source_type, mode)
    return f"{steady_2d_scattering_asset_root(mode, g_hg=g_hg)}/models/2d_{source_type}"


def steady_2d_training_dir(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_training_dir(source_type, mode)
    return f"{steady_2d_scattering_asset_root(mode, g_hg=g_hg)}/training_results/2d_{source_type}"


def steady_2d_norm_params_base_path(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> str:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode == STEADY_2D_SCATTERING_ISOTROPIC:
        return _steady_2d_legacy_norm_params_base_path(source_type, mode)
    return f"{steady_2d_scattering_asset_root(mode, g_hg=g_hg)}/data/norm_params_2d_{source_type}.npy"


def steady_2d_model_dir_candidates(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> tuple[str, ...]:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    primary = steady_2d_model_dir(source_type, scattering_mode=mode, g_hg=g_hg)
    if mode != STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR:
        return (primary,)
    return (primary, _steady_2d_legacy_model_dir(source_type, mode))


def steady_2d_norm_params_base_path_candidates(
    source_type: str,
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
) -> tuple[str, ...]:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    primary = steady_2d_norm_params_base_path(source_type, scattering_mode=mode, g_hg=g_hg)
    if mode != STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR:
        return (primary,)
    return (primary, _steady_2d_legacy_norm_params_base_path(source_type, mode))


def resolve_steady_2d_hg_parameter(
    scattering_mode: str = STEADY_2D_SCATTERING_ISOTROPIC,
    g_hg: float | None = None,
    *,
    stored_g_hg: float | None = None,
) -> float:
    """Resolve the effective HG asymmetry parameter for steady 2D workflows.

    Explicit HG runs use the caller-provided value when available, otherwise the
    stored training metadata value. Values must lie in the open interval
    (-1, 1) for numerical stability.
    """
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    if mode != STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        return 0.0

    candidate = g_hg if g_hg is not None else stored_g_hg
    if candidate is None:
        raise ValueError("anisotropic-hg requires an explicit g_hg value.")

    g_value = float(candidate)
    if not math.isfinite(g_value):
        raise ValueError(f"Invalid HG asymmetry parameter g_hg={candidate!r}.")
    if not (-1.0 < g_value < 1.0):
        raise ValueError(
            f"g_hg must satisfy -1 < g_hg < 1 for anisotropic-hg, got {g_value}."
        )
    if g_value > STEADY_2D_HG_G_LIMIT:
        return STEADY_2D_HG_G_LIMIT
    if g_value < -STEADY_2D_HG_G_LIMIT:
        return -STEADY_2D_HG_G_LIMIT
    return g_value


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
    """Sample mu from linear forward-peaked distribution: mu = 2*sqrt(u) - 1."""
    return 2.0 * math.sqrt(unit_x) - 1.0


@njit(inline="always")
def sample_henyey_greenstein_mu_from_unit(unit_x: float, g: float, eps: float = 1.0e-10):
    """Sample mu from Henyey-Greenstein phase function.
    
    The HG phase function is: p(mu) = (1 - g^2) / (2 * (1 + g^2 - 2*g*mu)^(3/2))
    
    For g = 0: reduces to isotropic (mu uniform in [-1, 1])
    For g → 1: strongly forward-peaked
    For g → -1: strongly backward-peaked
    
    Inversion formula:
    mu = 1/(2g) * [1 + g^2 - ((1 - g^2) / (1 - g + 2*g*u))^2]
    """
    if g >= 1.0 - eps:
        g = 1.0 - eps
    elif g <= -1.0 + eps:
        g = -1.0 + eps

    if abs(g) < eps:
        # Isotropic case: mu uniform in [-1, 1]
        return 2.0 * unit_x - 1.0
    
    # General HG case
    term = (1.0 - g * g) / (1.0 - g + 2.0 * g * unit_x)
    mu = (1.0 + g * g - term * term) / (2.0 * g)
    
    # Clamp to valid range [-1, 1] for numerical stability
    if mu > 1.0:
        return 1.0
    if mu < -1.0:
        return -1.0
    return mu


@njit(inline="always")
def sample_scattered_direction_3d_from_unit(
    current_u: float,
    current_v: float,
    current_w: float,
    unit_mu: float,
    unit_phi: float,
    scattering_mode_code: int,
    g_hg: float,
    eps: float,
):
    """Sample scattered direction in 3D based on scattering mode.
    
    Args:
        current_u, current_v, current_w: Current 3D direction (normalized)
        unit_mu: Random variable in [0,1] for polar angle sampling
        unit_phi: Random variable in [0,1] for azimuthal angle sampling
        scattering_mode_code: 0=isotropic, 1=anisotropic-linear, 2=anisotropic-hg
        g_hg: Henyey-Greenstein asymmetry parameter (used only if scattering_mode_code==2)
        eps: Small epsilon for numerical stability
        
    Returns:
        (out_u, out_v, out_w): Scattered 3D direction (normalized)
    """
    if scattering_mode_code == 0:
        # Isotropic scattering
        return sample_isotropic_direction_3d_from_unit(unit_mu, unit_phi, eps)

    # Anisotropic scattering (linear or HG)
    du, dv, dw = normalize_direction_3d(current_u, current_v, current_w, eps)
    
    # Sample mu based on mode
    if scattering_mode_code == 1:
        # Linear forward-peaked: mu = 2*sqrt(u) - 1
        mu = sample_anisotropic_mu_from_unit(unit_mu)
    elif scattering_mode_code == 2:
        # Henyey-Greenstein with parameter g_hg
        mu = sample_henyey_greenstein_mu_from_unit(unit_mu, g_hg, eps)
    else:
        # Fallback to isotropic
        mu = 2.0 * unit_mu - 1.0
    
    phi = 2.0 * math.pi * unit_phi
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

    # Construct local coordinate system with incident direction as z-axis
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

    # Rotate scattered direction into global frame
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    out_u = mu * du + sin_theta * (cos_phi * e1_x + sin_phi * e2_x)
    out_v = mu * dv + sin_theta * (cos_phi * e1_y + sin_phi * e2_y)
    out_w = mu * dw + sin_theta * (cos_phi * e1_z + sin_phi * e2_z)
    return normalize_direction_3d(out_u, out_v, out_w, eps)
