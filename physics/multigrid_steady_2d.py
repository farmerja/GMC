"""Prototype 2D steady-state multigrid refinement for the lattice benchmark.

This module implements a small but end-to-end coarse-to-fine workflow:

1. Run the existing steady 2D GMC/CFM transport on a coarse Cartesian lattice.
2. Bin inter-cell surface crossings on nested fine face patches and angular bins.
3. Convert crossing tallies into incoming angular-flux boundary conditions.
4. Solve independent local SN problems inside each coarse cell on its embedded
   fine subgrid and assemble a refined fine-grid scalar-flux estimate.
5. Compare the refined estimate against a direct fine-grid Monte Carlo reference.

The first prototype intentionally targets aligned nested grids on the existing
7x7 lattice pattern, where a coarse cell remains material-homogeneous for
refinement pairs such as 14x14 -> 56x56.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from physics.mc2d import calculate_boundaries_2d, run_mc_simulation_2d_njit_wrapper
from physics.steady_2d_scattering import (
    STEADY_2D_SCATTERING_ANISOTROPIC_HG,
    STEADY_2D_SCATTERING_ANISOTROPIC_LINEAR,
    STEADY_2D_SCATTERING_TAG,
    guard_projected_direction_xy,
    normalize_steady_2d_scattering_mode,
    resolve_steady_2d_hg_parameter,
    steady_2d_scattering_mode_code,
    sample_isotropic_direction_3d_from_unit,
    sample_scattered_direction_3d_from_unit,
)


FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3
FACE_TO_FACE = torch.tensor([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=torch.int64)

REGION_SCATTERING = 0
REGION_ABSORBING = 1
REGION_SOURCE = 2

LATTICE_PATTERN = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 2, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)

MATERIAL_PROPERTIES = {
    REGION_SCATTERING: {"sigma_a": 0.0, "sigma_s": 1.0},
    REGION_ABSORBING: {"sigma_a": 9.5, "sigma_s": 0.5},
    REGION_SOURCE: {"sigma_a": 0.0, "sigma_s": 1.0},
}


@dataclass(frozen=True)
class RectLatticeProblem:
    material_map: np.ndarray
    sigma_a_map: np.ndarray
    sigma_s_map: np.ndarray
    sigma_t_map: np.ndarray
    cell_widths: np.ndarray
    cell_heights: np.ndarray
    cell_boundaries_x: np.ndarray
    cell_boundaries_y: np.ndarray
    cell_areas: np.ndarray
    source_cells: np.ndarray
    num_cells_x: int
    num_cells_y: int
    refinement: int
    domain_size: float


@dataclass(frozen=True)
class MultigridHierarchy:
    coarse: RectLatticeProblem
    fine: RectLatticeProblem
    fine_per_coarse_x: int
    fine_per_coarse_y: int


@dataclass(frozen=True)
class AngularQuadrature2D:
    theta_centers: np.ndarray
    theta_edges: np.ndarray
    weights: np.ndarray
    face_bin_measure: np.ndarray
    face_incoming_mask: np.ndarray
    mu_x: np.ndarray
    mu_y: np.ndarray


@dataclass
class SurfaceTallyCollection:
    """Incoming surface data per coarse cell.

    Arrays are indexed as [iy, ix, face, segment, angle_bin].
    `crossing_weight` stores the sum of surviving particle weights at crossing.
    """

    crossing_weight: np.ndarray
    angular_flux: np.ndarray
    segment_lengths: np.ndarray


@dataclass
class LocalSNSolution:
    scalar_flux: np.ndarray
    outgoing_angular_flux: np.ndarray
    incoming_current: float
    outgoing_current: float
    absorbed_total: float
    source_total: float
    balance_residual: float


@dataclass
class SteadyModelBundle:
    device: str
    scattering_mode: str
    g_hg: float
    boundary_solver: object
    internal_solver: object
    boundary_norm: Dict[str, float]
    internal_norm: Dict[str, float]
    boundary_cond_dim: int
    internal_cond_dim: int
    boundary_target_dim: int
    internal_target_dim: int


def _coord_to_idx(coord: float, total_cells: int, domain_size: float) -> int:
    idx = int(coord * total_cells / domain_size)
    return min(max(idx, 0), total_cells)


def create_rect_lattice_material_map(
    refinement: int,
    *,
    domain_size: float = 7.0,
    base_cells: int = 7,
    lattice_pattern: np.ndarray = LATTICE_PATTERN,
) -> np.ndarray:
    num_cells_x = base_cells * refinement
    num_cells_y = base_cells * refinement
    material_map = np.zeros((num_cells_y, num_cells_x), dtype=np.int64)
    base_cell_size = float(domain_size) / float(base_cells)
    for base_i in range(base_cells):
        for base_j in range(base_cells):
            mat_type = int(lattice_pattern[base_i, base_j])
            iy_start = _coord_to_idx(base_i * base_cell_size, num_cells_y, domain_size)
            iy_end = _coord_to_idx((base_i + 1) * base_cell_size, num_cells_y, domain_size)
            ix_start = _coord_to_idx(base_j * base_cell_size, num_cells_x, domain_size)
            ix_end = _coord_to_idx((base_j + 1) * base_cell_size, num_cells_x, domain_size)
            material_map[iy_start:iy_end, ix_start:ix_end] = mat_type
    return material_map


def build_rect_lattice_problem(refinement: int, *, domain_size: float = 7.0) -> RectLatticeProblem:
    if refinement <= 0:
        raise ValueError(f"refinement must be positive; got {refinement}")
    material_map = create_rect_lattice_material_map(refinement=refinement, domain_size=domain_size)
    num_cells_y, num_cells_x = material_map.shape
    cell_widths = np.full(num_cells_x, domain_size / num_cells_x, dtype=np.float64)
    cell_heights = np.full(num_cells_y, domain_size / num_cells_y, dtype=np.float64)
    bx, by = calculate_boundaries_2d(cell_widths, cell_heights)
    cell_areas = np.outer(cell_heights, cell_widths)
    sigma_a_map = np.zeros_like(material_map, dtype=np.float64)
    sigma_s_map = np.zeros_like(material_map, dtype=np.float64)
    for region_id, props in MATERIAL_PROPERTIES.items():
        mask = material_map == int(region_id)
        sigma_a_map[mask] = float(props["sigma_a"])
        sigma_s_map[mask] = float(props["sigma_s"])
    sigma_t_map = sigma_a_map + sigma_s_map
    source_cells = np.argwhere(material_map == REGION_SOURCE).astype(np.int64)
    if source_cells.size == 0:
        raise RuntimeError("The lattice problem contains no source cells.")
    return RectLatticeProblem(
        material_map=material_map,
        sigma_a_map=sigma_a_map,
        sigma_s_map=sigma_s_map,
        sigma_t_map=sigma_t_map,
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=bx,
        cell_boundaries_y=by,
        cell_areas=cell_areas,
        source_cells=source_cells,
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        refinement=refinement,
        domain_size=float(domain_size),
    )


def build_multigrid_hierarchy(
    coarse_refinement: int = 2,
    fine_refinement: int = 8,
    *,
    domain_size: float = 7.0,
) -> MultigridHierarchy:
    coarse = build_rect_lattice_problem(coarse_refinement, domain_size=domain_size)
    fine = build_rect_lattice_problem(fine_refinement, domain_size=domain_size)
    if fine.num_cells_x % coarse.num_cells_x != 0 or fine.num_cells_y % coarse.num_cells_y != 0:
        raise ValueError(
            "Fine and coarse grids must be nested. "
            f"Got coarse=({coarse.num_cells_x},{coarse.num_cells_y}) and "
            f"fine=({fine.num_cells_x},{fine.num_cells_y})."
        )
    return MultigridHierarchy(
        coarse=coarse,
        fine=fine,
        fine_per_coarse_x=fine.num_cells_x // coarse.num_cells_x,
        fine_per_coarse_y=fine.num_cells_y // coarse.num_cells_y,
    )


def make_single_level_hierarchy(problem: RectLatticeProblem) -> MultigridHierarchy:
    return MultigridHierarchy(
        coarse=problem,
        fine=problem,
        fine_per_coarse_x=1,
        fine_per_coarse_y=1,
    )


def _face_outward_normal(face: int) -> Tuple[float, float]:
    if face == FACE_LEFT:
        return (-1.0, 0.0)
    if face == FACE_RIGHT:
        return (1.0, 0.0)
    if face == FACE_TOP:
        return (0.0, 1.0)
    if face == FACE_BOTTOM:
        return (0.0, -1.0)
    raise ValueError(f"Unknown face: {face}")


def build_uniform_angular_quadrature(num_angles: int, *, projection_samples: int = 64) -> AngularQuadrature2D:
    if num_angles < 4:
        raise ValueError("num_angles must be >= 4.")
    theta_edges = np.linspace(0.0, 2.0 * np.pi, num_angles + 1, dtype=np.float64)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    weights = np.diff(theta_edges)
    mu_x = np.cos(theta_centers)
    mu_y = np.sin(theta_centers)

    face_bin_measure = np.zeros((4, num_angles), dtype=np.float64)
    face_incoming_mask = np.zeros((4, num_angles), dtype=bool)
    for face in range(4):
        nx, ny = _face_outward_normal(face)
        for a in range(num_angles):
            lo = theta_edges[a]
            hi = theta_edges[a + 1]
            samples = np.linspace(lo, hi, projection_samples, endpoint=False, dtype=np.float64) + (
                (hi - lo) / max(1, projection_samples) * 0.5
            )
            proj = nx * np.cos(samples) + ny * np.sin(samples)
            incoming = proj < 0.0
            if np.any(incoming):
                face_bin_measure[face, a] = float(np.mean(np.abs(proj[incoming])) * (hi - lo) * np.mean(incoming))
                face_incoming_mask[face, a] = True
    return AngularQuadrature2D(
        theta_centers=theta_centers,
        theta_edges=theta_edges,
        weights=weights,
        face_bin_measure=face_bin_measure,
        face_incoming_mask=face_incoming_mask,
        mu_x=mu_x,
        mu_y=mu_y,
    )


def _projection_to_angle_index(theta: np.ndarray, quadrature: AngularQuadrature2D) -> np.ndarray:
    angle = np.mod(theta, 2.0 * np.pi)
    idx = np.searchsorted(quadrature.theta_edges, angle, side="right") - 1
    return np.clip(idx, 0, quadrature.theta_centers.size - 1)


def initialize_surface_tallies(
    hierarchy: MultigridHierarchy,
    quadrature: AngularQuadrature2D,
) -> SurfaceTallyCollection:
    coarse = hierarchy.coarse
    segments = max(hierarchy.fine_per_coarse_x, hierarchy.fine_per_coarse_y)
    crossing_weight = np.zeros(
        (coarse.num_cells_y, coarse.num_cells_x, 4, segments, quadrature.theta_centers.size),
        dtype=np.float64,
    )
    angular_flux = np.zeros_like(crossing_weight)
    segment_lengths = np.zeros((coarse.num_cells_y, coarse.num_cells_x, 4), dtype=np.float64)
    for iy in range(coarse.num_cells_y):
        for ix in range(coarse.num_cells_x):
            segment_lengths[iy, ix, FACE_LEFT] = coarse.cell_heights[iy] / hierarchy.fine_per_coarse_y
            segment_lengths[iy, ix, FACE_RIGHT] = coarse.cell_heights[iy] / hierarchy.fine_per_coarse_y
            segment_lengths[iy, ix, FACE_TOP] = coarse.cell_widths[ix] / hierarchy.fine_per_coarse_x
            segment_lengths[iy, ix, FACE_BOTTOM] = coarse.cell_widths[ix] / hierarchy.fine_per_coarse_x
    return SurfaceTallyCollection(
        crossing_weight=crossing_weight,
        angular_flux=angular_flux,
        segment_lengths=segment_lengths,
    )


def _segment_index_for_face(
    face: np.ndarray,
    pos_on_face: np.ndarray,
    hierarchy: MultigridHierarchy,
) -> np.ndarray:
    coarse = hierarchy.coarse
    pos_clamped = np.maximum(pos_on_face, 0.0)
    seg_idx = np.zeros_like(face, dtype=np.int64)
    is_lr = (face == FACE_LEFT) | (face == FACE_RIGHT)
    if np.any(is_lr):
        seg_len = coarse.cell_heights[0] / hierarchy.fine_per_coarse_y
        seg_idx[is_lr] = np.floor(pos_clamped[is_lr] / max(seg_len, 1.0e-12)).astype(np.int64)
        seg_idx[is_lr] = np.clip(seg_idx[is_lr], 0, hierarchy.fine_per_coarse_y - 1)
    if np.any(~is_lr):
        seg_len = coarse.cell_widths[0] / hierarchy.fine_per_coarse_x
        seg_idx[~is_lr] = np.floor(pos_clamped[~is_lr] / max(seg_len, 1.0e-12)).astype(np.int64)
        seg_idx[~is_lr] = np.clip(seg_idx[~is_lr], 0, hierarchy.fine_per_coarse_x - 1)
    return seg_idx


def record_surface_crossings(
    tally: SurfaceTallyCollection,
    hierarchy: MultigridHierarchy,
    quadrature: AngularQuadrature2D,
    target_ix: np.ndarray,
    target_iy: np.ndarray,
    target_face: np.ndarray,
    pos_on_face: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    weight: np.ndarray,
    *,
    num_histories: int,
) -> None:
    if weight.size == 0:
        return
    theta = np.arctan2(v, u)
    angle_idx = _projection_to_angle_index(theta, quadrature)
    segment_idx = _segment_index_for_face(target_face, pos_on_face, hierarchy)
    valid = (
        (target_ix >= 0)
        & (target_ix < hierarchy.coarse.num_cells_x)
        & (target_iy >= 0)
        & (target_iy < hierarchy.coarse.num_cells_y)
        & (weight > 0.0)
    )
    for ix, iy, face, seg, ang, w in zip(
        target_ix[valid],
        target_iy[valid],
        target_face[valid],
        segment_idx[valid],
        angle_idx[valid],
        weight[valid],
    ):
        tally.crossing_weight[int(iy), int(ix), int(face), int(seg), int(ang)] += float(w) / float(num_histories)


def finalize_surface_angular_flux(
    tally: SurfaceTallyCollection,
    quadrature: AngularQuadrature2D,
    *,
    grazing_floor: float = 1.0e-10,
) -> None:
    angular_flux = np.zeros_like(tally.crossing_weight)
    num_faces = tally.crossing_weight.shape[2]
    for face in range(num_faces):
        measure = quadrature.face_bin_measure[face]
        for iy in range(tally.crossing_weight.shape[0]):
            for ix in range(tally.crossing_weight.shape[1]):
                seg_len = tally.segment_lengths[iy, ix, face]
                denom = seg_len * np.maximum(measure, grazing_floor)
                angular_flux[iy, ix, face, :, :] = tally.crossing_weight[iy, ix, face, :, :] / denom[None, :]
                mask = ~quadrature.face_incoming_mask[face]
                if np.any(mask):
                    angular_flux[iy, ix, face, :, mask] = 0.0
    tally.angular_flux = angular_flux


def classify_face_and_snap(
    x: torch.Tensor,
    y: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    tol: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_left = torch.abs(x)
    d_right = torch.abs(W - x)
    d_top = torch.abs(H - y)
    d_bottom = torch.abs(y)
    d_stack = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
    d_min = d_stack.min(dim=1, keepdim=True).values
    tol_floor = 8.0 * torch.finfo(d_stack.dtype).eps * torch.maximum(W, H)
    tol_eff = torch.maximum(torch.full_like(W, tol), tol_floor)
    tie_mask = torch.abs(d_stack - d_min) <= tol_eff.unsqueeze(1)
    face = torch.argmin(d_stack, dim=1)
    tie_rows = tie_mask.sum(dim=1) > 1
    if tie_rows.any():
        # Randomize ties to avoid systematic corner bias while preserving mass.
        tie_weights = tie_mask[tie_rows].to(torch.float32)
        face = face.clone()
        face[tie_rows] = torch.multinomial(tie_weights, num_samples=1).squeeze(1)
    zeros = torch.zeros_like(x)
    x_snap = torch.where(face == FACE_LEFT, zeros, torch.where(face == FACE_RIGHT, W, x))
    y_snap = torch.where(face == FACE_BOTTOM, zeros, torch.where(face == FACE_TOP, H, y))
    x_snap = torch.minimum(torch.maximum(x_snap, zeros), W)
    y_snap = torch.minimum(torch.maximum(y_snap, zeros), H)
    pos_on_face = torch.where((face == FACE_LEFT) | (face == FACE_RIGHT), y_snap, x_snap)
    return face, x_snap, y_snap, pos_on_face


def enforce_outward_angle_from_face(
    face: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normals = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
        device=u.device,
    )
    n = normals[face]
    direction = torch.stack([u, v], dim=1)
    dot = (direction * n).sum(dim=1)
    needs_flip = dot <= 0.0
    reflected = direction - 2.0 * dot.unsqueeze(1) * n
    corrected = torch.where(needs_flip.unsqueeze(1), reflected, direction)
    return corrected[:, 0], corrected[:, 1]


def _build_time_grid(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    t_grid = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], device=device, dtype=torch.float32)
    step_size = t_grid[1] - t_grid[0]
    return t_grid, step_size


def _assert_steady_2d_norm_mode(
    norm_params: Dict[str, float],
    scattering_mode: str,
    label: str,
    *,
    g_hg: float | None = None,
) -> None:
    saved_mode = str(norm_params.get("steady_2d_scattering_mode", "isotropic")).strip().lower() or "isotropic"
    saved_tag = str(norm_params.get("steady_2d_scattering_tag", "")).strip()
    saved_normalized = normalize_steady_2d_scattering_mode(saved_mode)
    requested_normalized = normalize_steady_2d_scattering_mode(scattering_mode)
    if saved_normalized != requested_normalized:
        raise ValueError(
            f"{label} normalization metadata uses scattering_mode='{saved_mode}', "
            f"but '{scattering_mode}' was requested."
        )

    if requested_normalized != "isotropic" and saved_tag != STEADY_2D_SCATTERING_TAG:
        raise ValueError(
            f"{label} normalization metadata tag mismatch: expected '{STEADY_2D_SCATTERING_TAG}', got '{saved_tag}'."
        )

    if requested_normalized == STEADY_2D_SCATTERING_ANISOTROPIC_HG:
        resolved_g_hg = resolve_steady_2d_hg_parameter(requested_normalized, g_hg)
        saved_g_hg = norm_params.get("steady_2d_scattering_hg_g")
        if saved_g_hg is None:
            raise ValueError(
                f"{label} normalization metadata is missing steady_2d_scattering_hg_g for anisotropic-hg."
            )
        if abs(float(saved_g_hg) - resolved_g_hg) > 1.0e-9:
            raise ValueError(
                f"{label} normalization metadata uses g_hg={float(saved_g_hg):.6f}, "
                f"but g_hg={resolved_g_hg:.6f} was requested."
            )


def load_steady_gmc_models(
    device: Optional[str] = None,
    *,
    scattering_mode: str = "isotropic",
    g_hg: float | None = None,
    train_sampling: str = "random",
) -> SteadyModelBundle:
    from flow_matching.solver import ODESolver
    from ml.model import WrappedModel
    from ml.utils import load_model, load_norm_params_for_data_dim

    scattering_mode = normalize_steady_2d_scattering_mode(scattering_mode)
    resolved_g_hg = resolve_steady_2d_hg_parameter(scattering_mode, g_hg)
    resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    boundary_model, boundary_input_dim, boundary_cond_dim = load_model(
        resolved_device,
        hidden_dim=160,
        depth=5,
        expansion_factor=1,
        source_type="boundary",
        data_dim="2d",
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
        g_hg=resolved_g_hg,
    )
    internal_model, internal_input_dim, internal_cond_dim = load_model(
        resolved_device,
        hidden_dim=160,
        depth=5,
        expansion_factor=1,
        source_type="internal",
        data_dim="2d",
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
        g_hg=resolved_g_hg,
    )
    boundary_norm, _ = load_norm_params_for_data_dim(
        "2d",
        "boundary",
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
        g_hg=resolved_g_hg,
    )
    internal_norm, _ = load_norm_params_for_data_dim(
        "2d",
        "internal",
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
        g_hg=resolved_g_hg,
    )
    _assert_steady_2d_norm_mode(boundary_norm, scattering_mode, "Boundary", g_hg=resolved_g_hg)
    _assert_steady_2d_norm_mode(internal_norm, scattering_mode, "Internal", g_hg=resolved_g_hg)
    return SteadyModelBundle(
        device=resolved_device,
        scattering_mode=scattering_mode,
        g_hg=resolved_g_hg,
        boundary_solver=ODESolver(velocity_model=WrappedModel(boundary_model).to(resolved_device)),
        internal_solver=ODESolver(velocity_model=WrappedModel(internal_model).to(resolved_device)),
        boundary_norm=boundary_norm,
        internal_norm=internal_norm,
        boundary_cond_dim=boundary_cond_dim,
        internal_cond_dim=internal_cond_dim,
        boundary_target_dim=int(boundary_input_dim - boundary_cond_dim),
        internal_target_dim=int(internal_input_dim - internal_cond_dim),
    )


def _sample_isotropic_uv(num: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = torch.rand(num, device=device, dtype=torch.float32) * 2.0 - 1.0
    phi = torch.rand(num, device=device, dtype=torch.float32) * (2.0 * math.pi)
    t = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
    u = t * torch.cos(phi)
    v = t * torch.sin(phi)
    tiny = (u.abs() + v.abs()) < 1.0e-12
    if tiny.any():
        u = u.clone()
        u[tiny] = 1.0e-6
    return u, v


def _prepare_boundary_conditions(
    W: torch.Tensor,
    H: torch.Tensor,
    yn: torch.Tensor,
    ux_m: torch.Tensor,
    uy_m: torch.Tensor,
    sigma_s: torch.Tensor,
    norm_params: Dict[str, float],
) -> torch.Tensor:
    from ml.transform import (
        normalize_angle_phi,
        normalize_height,
        normalize_position_relative,
        normalize_radius_r,
        normalize_width,
        uv_to_polar,
    )

    phi_in, r_in = uv_to_polar(ux_m.detach().cpu().numpy(), uy_m.detach().cpu().numpy())
    phi_in_t = torch.from_numpy(phi_in).to(W.device)
    r_in_t = torch.from_numpy(r_in).to(W.device)
    W_opt = W * sigma_s
    H_opt = H * sigma_s
    return torch.stack(
        [
            normalize_width(W_opt, norm_params["width_lower"], norm_params["width_upper"]),
            normalize_height(H_opt, norm_params["height_lower"], norm_params["height_upper"]),
            normalize_position_relative(yn),
            normalize_angle_phi(phi_in_t),
            normalize_radius_r(r_in_t),
        ],
        dim=1,
    )


def _prepare_internal_conditions(
    W: torch.Tensor,
    H: torch.Tensor,
    x_rel: torch.Tensor,
    y_rel: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    sigma_s: torch.Tensor,
    norm_params: Dict[str, float],
) -> torch.Tensor:
    from ml.transform import (
        normalize_angle_phi,
        normalize_height,
        normalize_position_relative,
        normalize_radius_r,
        normalize_width,
        uv_to_polar,
    )

    phi_in, r_in = uv_to_polar(u.detach().cpu().numpy(), v.detach().cpu().numpy())
    phi_in_t = torch.from_numpy(phi_in).to(W.device)
    r_in_t = torch.from_numpy(r_in).to(W.device)
    W_opt = W * sigma_s
    H_opt = H * sigma_s
    return torch.stack(
        [
            normalize_width(W_opt, norm_params["width_lower"], norm_params["width_upper"]),
            normalize_height(H_opt, norm_params["height_lower"], norm_params["height_upper"]),
            normalize_position_relative(x_rel),
            normalize_position_relative(y_rel),
            normalize_angle_phi(phi_in_t),
            normalize_radius_r(r_in_t),
        ],
        dim=1,
    )


def run_coarse_gmc_with_surface_tally(
    problem: RectLatticeProblem,
    hierarchy: MultigridHierarchy,
    models: SteadyModelBundle,
    quadrature: AngularQuadrature2D,
    *,
    num_particles: int = 2000,
    randomize_start_pos: bool = True,
    seed: int = 52,
    max_iterations: int = 10_000,
    energy_threshold: float = 1.0e-8,
) -> Dict[str, object]:
    """Run coarse-grid GMC and record inter-cell incoming surface data."""

    from ml.transform import denorm_angle_phi, denorm_log_feature, denorm_position_relative, denorm_radius_r, polar_to_uv

    device = models.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_grid, step_size = _build_time_grid(device)
    tally = initialize_surface_tallies(hierarchy, quadrature)

    cell_widths_t = torch.from_numpy(problem.cell_widths.astype(np.float32)).to(device)
    cell_heights_t = torch.from_numpy(problem.cell_heights.astype(np.float32)).to(device)
    sigma_a_t = torch.from_numpy(problem.sigma_a_map.astype(np.float32)).to(device)
    sigma_s_t = torch.from_numpy(problem.sigma_s_map.astype(np.float32)).to(device).clamp_min(1.0e-6)
    sigma_t_t = torch.from_numpy(problem.sigma_t_map.astype(np.float32)).to(device).clamp_min(1.0e-6)

    part = torch.zeros((num_particles, 8), dtype=torch.float32, device=device)
    part[:, 6] = 1.0

    src_cells = problem.source_cells
    sel = np.random.randint(0, src_cells.shape[0], size=num_particles)
    part[:, 0] = torch.from_numpy(src_cells[sel, 1].astype(np.float32)).to(device)
    part[:, 1] = torch.from_numpy(src_cells[sel, 0].astype(np.float32)).to(device)

    energy_flat = torch.zeros(problem.num_cells_x * problem.num_cells_y, dtype=torch.float32, device=device)
    track_flat = torch.zeros_like(energy_flat)
    totals = torch.zeros(5, dtype=torch.int64, device=device)

    active = (part[:, 7] == 0).nonzero(as_tuple=True)[0]
    if active.numel() > 0:
        ix = part[active, 0].long()
        iy = part[active, 1].long()
        W = cell_widths_t[ix]
        H = cell_heights_t[iy]
        sigma_s = sigma_s_t[iy, ix]
        sigma_a = sigma_a_t[iy, ix]
        sigma_t = sigma_t_t[iy, ix]
        if randomize_start_pos:
            x_in = torch.rand(active.numel(), device=device) * W
            y_in = torch.rand(active.numel(), device=device) * H
        else:
            x_in = 0.5 * W
            y_in = 0.5 * H
        u0, v0 = _sample_isotropic_uv(active.numel(), device)
        cond = _prepare_internal_conditions(
            W,
            H,
            x_in / W.clamp_min(1.0e-12),
            y_in / H.clamp_min(1.0e-12),
            u0,
            v0,
            sigma_s,
            models.internal_norm,
        )
        noise = torch.randn(active.numel(), models.internal_target_dim, device=device)
        x0 = torch.cat([cond, noise], dim=1)
        with torch.no_grad():
            out = models.internal_solver.sample(
                time_grid=t_grid,
                x_init=x0,
                method="rk4",
                step_size=step_size,
                return_intermediates=False,
            )
        pred = out[:, models.internal_cond_dim:]
        x_out = denorm_position_relative(pred[:, 0]) * W
        y_out = denorm_position_relative(pred[:, 1]) * H
        phi_out = denorm_angle_phi(pred[:, 2])
        r_out = denorm_radius_r(pred[:, 3])
        dt = denorm_log_feature(pred[:, 4], models.internal_norm, "time").clamp_min(0.0)
        u_out, v_out = polar_to_uv(phi_out, r_out)
        exit_face, x_out, y_out, pos_on_face = classify_face_and_snap(x_out, y_out, W, H)
        u_out, v_out = enforce_outward_angle_from_face(exit_face, u_out, v_out)
        dt_phys = dt / sigma_s
        E_before = part[active, 6]
        optical_depth = sigma_a * dt_phys
        track_add = torch.where(
            optical_depth > 0.01,
            E_before * (1.0 - torch.exp(-optical_depth)) / sigma_a.clamp_min(1.0e-12),
            E_before * dt_phys,
        )
        deposit = E_before * (1.0 - torch.exp(-optical_depth))
        lin_idx = iy * problem.num_cells_x + ix
        track_flat.index_add_(0, lin_idx, track_add)
        energy_flat.index_add_(0, lin_idx, deposit)
        part[active, 6] = E_before - deposit
        absorbed = part[active, 6] < energy_threshold
        if absorbed.any():
            totals[4] += absorbed.sum()
            part[active[absorbed], 7] = 2
        survivors = ~absorbed
        if survivors.any():
            surv_idx = active[survivors]
            exit_face = exit_face[survivors]
            pos_on_face = pos_on_face[survivors]
            u_out = u_out[survivors]
            v_out = v_out[survivors]
            ix_s = ix[survivors]
            iy_s = iy[survivors]
            new_ix = ix_s + (exit_face == FACE_RIGHT).long() - (exit_face == FACE_LEFT).long()
            new_iy = iy_s + (exit_face == FACE_TOP).long() - (exit_face == FACE_BOTTOM).long()
            exit_msk = (
                (new_ix < 0)
                | (new_ix >= problem.num_cells_x)
                | (new_iy < 0)
                | (new_iy >= problem.num_cells_y)
            )
            totals[0] += torch.count_nonzero((exit_face == FACE_LEFT) & exit_msk)
            totals[1] += torch.count_nonzero((exit_face == FACE_RIGHT) & exit_msk)
            totals[2] += torch.count_nonzero((exit_face == FACE_TOP) & exit_msk)
            totals[3] += torch.count_nonzero((exit_face == FACE_BOTTOM) & exit_msk)
            part[surv_idx[exit_msk], 7] = 1
            interior = ~exit_msk
            if interior.any():
                target_face = FACE_TO_FACE[exit_face[interior]].cpu().numpy()
                record_surface_crossings(
                    tally,
                    hierarchy,
                    quadrature,
                    new_ix[interior].cpu().numpy(),
                    new_iy[interior].cpu().numpy(),
                    target_face,
                    pos_on_face[interior].cpu().numpy(),
                    u_out[interior].cpu().numpy(),
                    v_out[interior].cpu().numpy(),
                    part[surv_idx[interior], 6].cpu().numpy(),
                    num_histories=num_particles,
                )
                cont_idx = surv_idx[interior]
                part[cont_idx, 0] = new_ix[interior].float()
                part[cont_idx, 1] = new_iy[interior].float()
                part[cont_idx, 2] = FACE_TO_FACE[exit_face[interior]].float()
                part[cont_idx, 3] = pos_on_face[interior]
                part[cont_idx, 4] = u_out[interior]
                part[cont_idx, 5] = v_out[interior]

    orient_r = torch.tensor(
        np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[-1.0, 0.0], [0.0, -1.0]],
                [[0.0, -1.0], [1.0, 0.0]],
                [[0.0, 1.0], [-1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        device=device,
    )

    for _ in range(max_iterations):
        active = (part[:, 7] == 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        p = part[active]
        ix = p[:, 0].long()
        iy = p[:, 1].long()
        face = p[:, 2].long()
        pos = p[:, 3]
        u = p[:, 4]
        v = p[:, 5]
        E = p[:, 6]
        W = cell_widths_t[ix]
        H = cell_heights_t[iy]
        sigma_s = sigma_s_t[iy, ix]
        sigma_a = sigma_a_t[iy, ix]
        sigma_t = sigma_t_t[iy, ix]

        zeros_pos = torch.zeros_like(pos)
        px = torch.where(face == FACE_LEFT, zeros_pos, torch.where(face == FACE_RIGHT, W, pos))
        py = torch.where(face == FACE_LEFT, pos, torch.where(face == FACE_RIGHT, pos, torch.where(face == FACE_TOP, H, zeros_pos)))
        pos_phys = torch.stack([px, py], dim=1)
        tx = torch.where(face == FACE_LEFT, torch.zeros_like(W), torch.where(face == FACE_RIGHT, W, torch.where(face == FACE_TOP, H, torch.zeros_like(W))))
        ty = torch.where(face == FACE_RIGHT, H, torch.where(face == FACE_BOTTOM, W, torch.zeros_like(W)))
        t_vecs = torch.stack([tx, ty], dim=1)
        Rs = orient_r[face]
        dir_phys = torch.stack([u, v], dim=1).unsqueeze(-1)
        pos_model = torch.bmm(Rs, pos_phys.unsqueeze(-1)).squeeze(-1) + t_vecs
        dir_model = torch.bmm(Rs, dir_phys).squeeze(-1)
        ux_m = dir_model[:, 0]
        uy_m = dir_model[:, 1]
        is_lr = face < 2
        Wm = torch.where(is_lr, W, H)
        Hm = torch.where(is_lr, H, W)
        yn = torch.clamp(pos_model[:, 1] / Hm.clamp_min(1.0e-12), 0.0, 1.0)

        cond = _prepare_boundary_conditions(Wm, Hm, yn, ux_m, uy_m, sigma_s, models.boundary_norm)
        noise = torch.randn(active.numel(), models.boundary_target_dim, device=device)
        x0 = torch.cat([cond, noise], dim=1)
        with torch.no_grad():
            out = models.boundary_solver.sample(
                time_grid=t_grid,
                x_init=x0,
                method="rk4",
                step_size=step_size,
                return_intermediates=False,
            )
        from ml.transform import denorm_angle_phi, denorm_log_feature, denorm_position_relative, denorm_radius_r, polar_to_uv

        pred = out[:, models.boundary_cond_dim:]
        x_rel = denorm_position_relative(pred[:, 0])
        y_rel = denorm_position_relative(pred[:, 1])
        phi_out = denorm_angle_phi(pred[:, 2])
        r_out = denorm_radius_r(pred[:, 3])
        dt = denorm_log_feature(pred[:, 4], models.boundary_norm, "time").clamp_min(1.0e-9)
        x_m = x_rel * Wm
        y_m = y_rel * Hm
        face_model, x_m, y_m, _ = classify_face_and_snap(x_m, y_m, Wm, Hm)
        ux_n, uy_n = polar_to_uv(phi_out, r_out)
        ux_n, uy_n = enforce_outward_angle_from_face(face_model, ux_n, uy_n)
        dt_phys = dt / sigma_s

        optical_depth = sigma_a * dt_phys
        track_add = torch.where(
            optical_depth > 0.01,
            E * (1.0 - torch.exp(-optical_depth)) / sigma_a.clamp_min(1.0e-12),
            E * dt_phys,
        )
        deposit = E * (1.0 - torch.exp(-optical_depth))
        lin_idx = iy * problem.num_cells_x + ix
        track_flat.index_add_(0, lin_idx, track_add)
        energy_flat.index_add_(0, lin_idx, deposit)
        part[active, 6] = E - deposit

        Rs_T = Rs.transpose(1, 2)
        pos_phys_exit = torch.bmm(Rs_T, (torch.stack([x_m, y_m], dim=1) - t_vecs).unsqueeze(-1)).squeeze(-1)
        dir_phys_exit = torch.bmm(Rs_T, torch.stack([ux_n, uy_n], dim=1).unsqueeze(-1)).squeeze(-1)
        x_p = pos_phys_exit[:, 0]
        y_p = pos_phys_exit[:, 1]
        u_p = dir_phys_exit[:, 0]
        v_p = dir_phys_exit[:, 1]
        x_pc = torch.minimum(torch.maximum(x_p, torch.zeros_like(W)), W)
        y_pc = torch.minimum(torch.maximum(y_p, torch.zeros_like(H)), H)
        b, x_pc, y_pc, s = classify_face_and_snap(x_pc, y_pc, W, H)

        new_ix = ix + (b == FACE_RIGHT).long() - (b == FACE_LEFT).long()
        new_iy = iy + (b == FACE_TOP).long() - (b == FACE_BOTTOM).long()
        exit_msk = (
            (new_ix < 0)
            | (new_ix >= problem.num_cells_x)
            | (new_iy < 0)
            | (new_iy >= problem.num_cells_y)
        )
        totals[0] += torch.count_nonzero((b == FACE_LEFT) & exit_msk)
        totals[1] += torch.count_nonzero((b == FACE_RIGHT) & exit_msk)
        totals[2] += torch.count_nonzero((b == FACE_TOP) & exit_msk)
        totals[3] += torch.count_nonzero((b == FACE_BOTTOM) & exit_msk)
        part[active[exit_msk], 7] = 1

        survivors = ~exit_msk
        if survivors.any():
            record_surface_crossings(
                tally,
                hierarchy,
                quadrature,
                new_ix[survivors].cpu().numpy(),
                new_iy[survivors].cpu().numpy(),
                FACE_TO_FACE[b[survivors]].cpu().numpy(),
                s[survivors].cpu().numpy(),
                u_p[survivors].cpu().numpy(),
                v_p[survivors].cpu().numpy(),
                part[active[survivors], 6].cpu().numpy(),
                num_histories=num_particles,
            )
            surv_idx = active[survivors]
            part[surv_idx, 0] = new_ix[survivors].float()
            part[surv_idx, 1] = new_iy[survivors].float()
            part[surv_idx, 2] = FACE_TO_FACE[b[survivors]].float()
            part[surv_idx, 3] = s[survivors]
            part[surv_idx, 4] = u_p[survivors]
            part[surv_idx, 5] = v_p[survivors]

        alive = (part[:, 7] == 0) & (part[:, 6] >= energy_threshold)
        absorbed = (part[:, 7] == 0) & ~alive
        if absorbed.any():
            totals[4] += torch.count_nonzero(absorbed)
            part[absorbed, 7] = 2

    finalize_surface_angular_flux(tally, quadrature)
    energy_grid = energy_flat.view(problem.num_cells_y, problem.num_cells_x).detach().cpu().numpy() / float(num_particles)
    track_grid = track_flat.view(problem.num_cells_y, problem.num_cells_x).detach().cpu().numpy() / float(num_particles)
    flux_grid = np.divide(track_grid, problem.cell_areas, out=np.zeros_like(track_grid), where=problem.cell_areas > 0.0)
    return {
        "surface_tally": tally,
        "coarse_energy": energy_grid,
        "coarse_track": track_grid,
        "coarse_flux": flux_grid,
        "totals": {
            "left": int(totals[0].item()),
            "right": int(totals[1].item()),
            "top": int(totals[2].item()),
            "bottom": int(totals[3].item()),
            "absorb": int(totals[4].item()),
        },
    }


def _sample_isotropic_direction_2d(rng: np.random.Generator) -> Tuple[float, float]:
    u, v, _ = sample_isotropic_direction_3d_from_unit(float(rng.random()), float(rng.random()), 1.0e-12)
    return guard_projected_direction_xy(u, v, 1.0e-12)


def _sample_scattered_direction_3d(
    rng: np.random.Generator,
    current_direction: Tuple[float, float, float],
    *,
    scattering_mode: str = "isotropic",
    g_hg: float = 0.0,
) -> Tuple[float, float, float]:
    mode = normalize_steady_2d_scattering_mode(scattering_mode)
    mode_code = steady_2d_scattering_mode_code(mode)
    return sample_scattered_direction_3d_from_unit(
        float(current_direction[0]),
        float(current_direction[1]),
        float(current_direction[2]),
        float(rng.random()),
        float(rng.random()),
        mode_code,
        1.0e-12,
        g_hg,
    )


def _distance_to_rect_boundary(
    problem: RectLatticeProblem,
    cell_x: int,
    cell_y: int,
    x: float,
    y: float,
    u: float,
    v: float,
) -> Tuple[float, int]:
    distance = math.inf
    face = -1
    eps = 1.0e-12
    if u > eps:
        candidate = (problem.cell_boundaries_x[cell_x + 1] - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_RIGHT
    elif u < -eps:
        candidate = (problem.cell_boundaries_x[cell_x] - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_LEFT
    if v > eps:
        candidate = (problem.cell_boundaries_y[cell_y + 1] - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_TOP
    elif v < -eps:
        candidate = (problem.cell_boundaries_y[cell_y] - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_BOTTOM
    return max(distance, 0.0), face


def _absorption_weighted_track_length(weight: float, sigma_a: float, distance: float) -> float:
    if distance <= 0.0 or weight <= 0.0:
        return 0.0
    if sigma_a <= 1.0e-12:
        return weight * distance
    optical_depth = sigma_a * distance
    if optical_depth < 1.0e-5:
        return weight * distance
    return weight * (1.0 - math.exp(-optical_depth)) / sigma_a


def run_coarse_mc_with_surface_tally(
    problem: RectLatticeProblem,
    hierarchy: MultigridHierarchy,
    quadrature: AngularQuadrature2D,
    *,
    num_particles: int = 20_000,
    randomize_start_pos: bool = True,
    seed: int = 52,
    max_events: int = 20_000,
    energy_threshold: float = 1.0e-8,
    scattering_mode: str = "isotropic",
    g_hg: float = 0.0,
) -> Dict[str, object]:
    """Run analog MC on the coarse mesh and tally true inter-cell surface crossings."""

    scattering_mode = normalize_steady_2d_scattering_mode(scattering_mode)
    rng = np.random.default_rng(seed)
    tally = initialize_surface_tallies(hierarchy, quadrature)
    track_grid = np.zeros_like(problem.sigma_t_map, dtype=np.float64)
    energy_grid = np.zeros_like(problem.sigma_t_map, dtype=np.float64)
    totals = {"left": 0, "right": 0, "top": 0, "bottom": 0, "absorb": 0}
    epsilon = 1.0e-10 * float(problem.domain_size)

    for _ in range(num_particles):
        src_index = int(rng.integers(0, problem.source_cells.shape[0]))
        cell_y, cell_x = map(int, problem.source_cells[src_index])
        if randomize_start_pos:
            x = problem.cell_boundaries_x[cell_x] + problem.cell_widths[cell_x] * float(rng.random())
            y = problem.cell_boundaries_y[cell_y] + problem.cell_heights[cell_y] * float(rng.random())
        else:
            x = problem.cell_boundaries_x[cell_x] + 0.5 * problem.cell_widths[cell_x]
            y = problem.cell_boundaries_y[cell_y] + 0.5 * problem.cell_heights[cell_y]
        u, v = _sample_isotropic_direction_2d(rng)
        w_mag = math.sqrt(max(0.0, 1.0 - u * u - v * v))
        w = w_mag if float(rng.random()) < 0.5 else -w_mag
        weight = 1.0

        for _event in range(max_events):
            sigma_s = max(0.0, float(problem.sigma_s_map[cell_y, cell_x]))
            sigma_a = max(0.0, float(problem.sigma_a_map[cell_y, cell_x]))
            if sigma_s <= 1.0e-12:
                free_path = math.inf
            else:
                free_path = -math.log(max(1.0e-15, 1.0 - float(rng.random()))) / sigma_s
            boundary_distance, face = _distance_to_rect_boundary(problem, cell_x, cell_y, x, y, u, v)
            travel_distance = min(free_path, boundary_distance)
            travel_distance = max(travel_distance, 0.0)

            if travel_distance <= 0.0 and face < 0:
                break

            weight_before = weight
            track_grid[cell_y, cell_x] += _absorption_weighted_track_length(weight_before, sigma_a, travel_distance)
            weight *= math.exp(-sigma_a * travel_distance)
            energy_grid[cell_y, cell_x] += weight_before - weight
            x += travel_distance * u
            y += travel_distance * v

            if weight <= energy_threshold:
                totals["absorb"] += 1
                break

            at_boundary = face >= 0 and abs(travel_distance - boundary_distance) <= 1.0e-10 * max(1.0, boundary_distance)
            if at_boundary:
                next_cell_x = cell_x + (1 if face == FACE_RIGHT else -1 if face == FACE_LEFT else 0)
                next_cell_y = cell_y + (1 if face == FACE_TOP else -1 if face == FACE_BOTTOM else 0)
                if not (0 <= next_cell_x < problem.num_cells_x and 0 <= next_cell_y < problem.num_cells_y):
                    if face == FACE_LEFT:
                        totals["left"] += 1
                    elif face == FACE_RIGHT:
                        totals["right"] += 1
                    elif face == FACE_TOP:
                        totals["top"] += 1
                    else:
                        totals["bottom"] += 1
                    break

                incoming_face = int(FACE_TO_FACE[face].item())
                if face in (FACE_LEFT, FACE_RIGHT):
                    pos_on_face = y - problem.cell_boundaries_y[next_cell_y]
                else:
                    pos_on_face = x - problem.cell_boundaries_x[next_cell_x]
                record_surface_crossings(
                    tally,
                    hierarchy,
                    quadrature,
                    np.asarray([next_cell_x], dtype=np.int64),
                    np.asarray([next_cell_y], dtype=np.int64),
                    np.asarray([incoming_face], dtype=np.int64),
                    np.asarray([pos_on_face], dtype=np.float64),
                    np.asarray([u], dtype=np.float64),
                    np.asarray([v], dtype=np.float64),
                    np.asarray([weight], dtype=np.float64),
                    num_histories=num_particles,
                )
                cell_x = next_cell_x
                cell_y = next_cell_y
                x = min(
                    max(x + epsilon * u, problem.cell_boundaries_x[cell_x] + epsilon),
                    problem.cell_boundaries_x[cell_x + 1] - epsilon,
                )
                y = min(
                    max(y + epsilon * v, problem.cell_boundaries_y[cell_y] + epsilon),
                    problem.cell_boundaries_y[cell_y + 1] - epsilon,
                )
            else:
                u, v, w = _sample_scattered_direction_3d(
                    rng,
                    (u, v, w),
                    scattering_mode=scattering_mode,
                    g_hg=g_hg,
                )
                u, v = guard_projected_direction_xy(u, v, 1.0e-12)

    track_grid /= float(num_particles)
    energy_grid /= float(num_particles)
    finalize_surface_angular_flux(tally, quadrature)
    flux_grid = np.divide(track_grid, problem.cell_areas, out=np.zeros_like(track_grid), where=problem.cell_areas > 0.0)
    return {
        "surface_tally": tally,
        "coarse_energy": energy_grid,
        "coarse_track": track_grid,
        "coarse_flux": flux_grid,
        "totals": totals,
    }


def solve_local_sn(
    incoming_angular_flux: np.ndarray,
    quadrature: AngularQuadrature2D,
    *,
    width: float,
    height: float,
    nx: int,
    ny: int,
    sigma_t: float,
    sigma_s: float,
    fixed_source_strength: float = 0.0,
    num_source_iterations: int = 6,
    positivity_clamp: bool = True,
) -> LocalSNSolution:
    """Solve a local 2D SN problem on a rectangular subgrid with diamond-difference sweeps."""

    dx = float(width) / float(nx)
    dy = float(height) / float(ny)
    cell_area = dx * dy
    scalar_flux = np.zeros((ny, nx), dtype=np.float64)
    outgoing = np.zeros_like(incoming_angular_flux)
    face_segment_length = {
        FACE_LEFT: dy,
        FACE_RIGHT: dy,
        FACE_TOP: dx,
        FACE_BOTTOM: dx,
    }

    fixed_source = np.full((ny, nx), float(fixed_source_strength), dtype=np.float64)
    incoming_current = 0.0
    for face in range(4):
        incoming_current += float(
            np.sum(
                incoming_angular_flux[face]
                * face_segment_length[face]
                * quadrature.face_bin_measure[face][None, :]
            )
        )

    for _ in range(max(1, num_source_iterations)):
        new_flux = np.zeros_like(scalar_flux)
        outgoing.fill(0.0)
        scatter_source = sigma_s * scalar_flux / (2.0 * math.pi)
        total_source = fixed_source / (2.0 * math.pi) + scatter_source

        for ang_idx, theta in enumerate(quadrature.theta_centers):
            mu = math.cos(theta)
            eta = math.sin(theta)
            if abs(mu) < 1.0e-10 and abs(eta) < 1.0e-10:
                continue
            ax = 2.0 * abs(mu) / dx
            ay = 2.0 * abs(eta) / dy
            denom = sigma_t + ax + ay
            if denom <= 0.0:
                continue

            if mu > 0.0:
                x_indices = range(nx)
                incoming_x_face = incoming_angular_flux[FACE_LEFT, :, ang_idx].copy()
                exit_face_x = FACE_RIGHT
            else:
                x_indices = range(nx - 1, -1, -1)
                incoming_x_face = incoming_angular_flux[FACE_RIGHT, :, ang_idx].copy()
                exit_face_x = FACE_LEFT

            if eta > 0.0:
                y_indices = range(ny)
                incoming_y = incoming_angular_flux[FACE_BOTTOM, :, ang_idx].copy()
                exit_face_y = FACE_TOP
            else:
                y_indices = range(ny - 1, -1, -1)
                incoming_y = incoming_angular_flux[FACE_TOP, :, ang_idx].copy()
                exit_face_y = FACE_BOTTOM

            for jy in y_indices:
                psi_x = incoming_x_face[jy]
                next_incoming_y = np.zeros(nx, dtype=np.float64)
                for ix in x_indices:
                    psi_y = incoming_y[ix]
                    psi_center = (total_source[jy, ix] + ax * psi_x + ay * psi_y) / denom
                    psi_out_x = 2.0 * psi_center - psi_x
                    psi_out_y = 2.0 * psi_center - psi_y
                    if positivity_clamp:
                        psi_center = max(psi_center, 0.0)
                        psi_out_x = max(psi_out_x, 0.0)
                        psi_out_y = max(psi_out_y, 0.0)
                    psi_x = psi_out_x
                    next_incoming_y[ix] = psi_out_y
                    new_flux[jy, ix] += quadrature.weights[ang_idx] * psi_center
                outgoing[exit_face_x, jy, ang_idx] = psi_x
                incoming_y = next_incoming_y
            outgoing[exit_face_y, :, ang_idx] = incoming_y

        scalar_flux = new_flux

    absorbed_total = float(np.sum((sigma_t - sigma_s) * scalar_flux) * cell_area)
    outgoing_current = 0.0
    for face in range(4):
        outgoing_current += float(
            np.sum(
                outgoing[face]
                * face_segment_length[face]
                * quadrature.face_bin_measure[face][None, :]
            )
        )
    source_total = float(np.sum(fixed_source) * cell_area)
    balance_residual = source_total + incoming_current - absorbed_total - outgoing_current
    return LocalSNSolution(
        scalar_flux=scalar_flux,
        outgoing_angular_flux=outgoing,
        incoming_current=incoming_current,
        outgoing_current=outgoing_current,
        absorbed_total=absorbed_total,
        source_total=source_total,
        balance_residual=balance_residual,
    )


def reconstruct_fine_flux_from_surface_tally(
    hierarchy: MultigridHierarchy,
    tally: SurfaceTallyCollection,
    quadrature: AngularQuadrature2D,
    *,
    num_source_iterations: int = 6,
) -> Dict[str, object]:
    coarse = hierarchy.coarse
    fine = hierarchy.fine
    local_nx = hierarchy.fine_per_coarse_x
    local_ny = hierarchy.fine_per_coarse_y
    fine_flux = np.zeros((fine.num_cells_y, fine.num_cells_x), dtype=np.float64)
    local_balance = np.zeros((coarse.num_cells_y, coarse.num_cells_x), dtype=np.float64)
    local_outgoing_current = np.zeros_like(local_balance)
    local_incoming_current = np.zeros_like(local_balance)
    total_source_area = float(np.sum(coarse.material_map == REGION_SOURCE) * coarse.cell_areas[0, 0])

    for iy in range(coarse.num_cells_y):
        for ix in range(coarse.num_cells_x):
            fixed_source = 0.0
            if coarse.material_map[iy, ix] == REGION_SOURCE and total_source_area > 0.0:
                fixed_source = 1.0 / total_source_area
            solution = solve_local_sn(
                tally.angular_flux[iy, ix],
                quadrature,
                width=float(coarse.cell_widths[ix]),
                height=float(coarse.cell_heights[iy]),
                nx=local_nx,
                ny=local_ny,
                sigma_t=float(coarse.sigma_t_map[iy, ix]),
                sigma_s=float(coarse.sigma_s_map[iy, ix]),
                fixed_source_strength=fixed_source,
                num_source_iterations=num_source_iterations,
            )
            ys = iy * local_ny
            xs = ix * local_nx
            fine_flux[ys : ys + local_ny, xs : xs + local_nx] = solution.scalar_flux
            local_balance[iy, ix] = solution.balance_residual
            local_outgoing_current[iy, ix] = solution.outgoing_current
            local_incoming_current[iy, ix] = solution.incoming_current

    return {
        "fine_flux": fine_flux,
        "local_balance": local_balance,
        "incoming_current": local_incoming_current,
        "outgoing_current": local_outgoing_current,
    }


def reconstruct_fine_flux_with_block_sn(
    hierarchy: MultigridHierarchy,
    tally: SurfaceTallyCollection,
    quadrature: AngularQuadrature2D,
    coarse_track_length: np.ndarray,
    *,
    num_source_iterations: int = 18,
    tolerance: float = 1.0e-5,
) -> Dict[str, object]:
    """Reconstruct the fine flux with the lattice block-SN solver on the same boundary data."""

    from physics.lattice_multigrid_2d import DiscreteOrdinates2D, solve_local_sn_block

    coarse = hierarchy.coarse
    fine = hierarchy.fine
    local_nx = hierarchy.fine_per_coarse_x
    local_ny = hierarchy.fine_per_coarse_y
    fine_flux = np.zeros((fine.num_cells_y, fine.num_cells_x), dtype=np.float64)
    local_iterations = np.zeros((coarse.num_cells_y, coarse.num_cells_x), dtype=np.int64)
    local_scale = np.zeros_like(local_iterations, dtype=np.float64)

    num_angles = int(quadrature.theta_centers.size)
    ordinates = DiscreteOrdinates2D.uniform(num_angles)
    if not np.allclose(ordinates.angles, quadrature.theta_centers):
        raise ValueError("Quadrature and discrete ordinates must share the same angular abscissae.")

    fixed_source_density = np.zeros_like(fine.sigma_t_map, dtype=np.float64)
    source_mask = fine.material_map == REGION_SOURCE
    total_source_area = float(np.sum(fine.cell_areas[source_mask]))
    if total_source_area > 0.0:
        fixed_source_density[source_mask] = 1.0 / total_source_area

    for iy in range(coarse.num_cells_y):
        for ix in range(coarse.num_cells_x):
            ys = slice(iy * local_ny, (iy + 1) * local_ny)
            xs = slice(ix * local_nx, (ix + 1) * local_nx)
            sigma_t_block = np.full((local_ny, local_nx), float(coarse.sigma_t_map[iy, ix]), dtype=np.float64)
            sigma_s_block = np.full((local_ny, local_nx), float(coarse.sigma_s_map[iy, ix]), dtype=np.float64)
            block_flux, iteration_count, scale = solve_local_sn_block(
                sigma_t=sigma_t_block,
                sigma_s=sigma_s_block,
                fixed_source=fixed_source_density[ys, xs],
                cell_widths=fine.cell_widths[xs],
                cell_heights=fine.cell_heights[ys],
                boundary_angular_flux=np.array(tally.angular_flux[iy, ix], dtype=np.float64, copy=True),
                ordinates=ordinates,
                target_track_length=float(coarse_track_length[iy, ix]),
                max_iterations=num_source_iterations,
                tolerance=tolerance,
            )
            fine_flux[ys, xs] = block_flux
            local_iterations[iy, ix] = iteration_count
            local_scale[iy, ix] = scale

    return {
        "fine_flux": fine_flux,
        "local_iterations": local_iterations,
        "local_scale_factors": local_scale,
    }


def run_direct_fine_mc_reference(
    problem: RectLatticeProblem,
    *,
    num_particles: int = 20_000,
    randomize_start_pos: bool = True,
    seed: int = 450,
    energy_threshold: float = 1.0e-8,
    scattering_mode: str = "isotropic",
    g_hg: float = 0.0,
) -> Dict[str, object]:
    """Run direct fine-grid MC by distributing particles uniformly over source cells."""

    scattering_mode = normalize_steady_2d_scattering_mode(scattering_mode)
    num_src = max(1, problem.source_cells.shape[0])
    base_count = num_particles // num_src
    remainder = num_particles % num_src
    energy_accum = np.zeros_like(problem.sigma_t_map, dtype=np.float64)
    track_accum = np.zeros_like(problem.sigma_t_map, dtype=np.float64)
    stats_accum = np.zeros(8, dtype=np.int64)
    for k in range(num_src):
        n_k = base_count + (1 if k < remainder else 0)
        if n_k == 0:
            continue
        iy_k, ix_k = int(problem.source_cells[k, 0]), int(problem.source_cells[k, 1])
        e_k, tl_k, stats_k = run_mc_simulation_2d_njit_wrapper(
            source_mode="internal_start",
            num_cells_x=problem.num_cells_x,
            num_cells_y=problem.num_cells_y,
            cell_widths=problem.cell_widths.astype(np.float64),
            cell_heights=problem.cell_heights.astype(np.float64),
            cell_boundaries_x=problem.cell_boundaries_x.astype(np.float64),
            cell_boundaries_y=problem.cell_boundaries_y.astype(np.float64),
            total_width=float(problem.domain_size),
            total_height=float(problem.domain_size),
            num_particles=n_k,
            start_cell_indices=[ix_k, iy_k],
            start_position_relative=[0.5, 0.5],
            scatter_ratio=None,
            energy_threshold=energy_threshold,
            seed=seed + k,
            reflected_boundaries=False,
            angular_sampling=1,
            sigma_t_per_cell=problem.sigma_t_map.astype(np.float64),
            sigma_a_per_cell=problem.sigma_a_map.astype(np.float64),
            randomize_position_in_cell=randomize_start_pos,
            verbose=False,
            scattering_mode=scattering_mode,
            g_hg=g_hg,
        )
        weight = float(n_k) / float(num_particles)
        energy_accum += weight * e_k
        track_accum += weight * tl_k
        stats_accum += np.asarray(stats_k, dtype=np.int64)
    flux = np.divide(track_accum, problem.cell_areas, out=np.zeros_like(track_accum), where=problem.cell_areas > 0.0)
    return {
        "energy": energy_accum,
        "track": track_accum,
        "flux": flux,
        "stats": stats_accum,
    }


def compute_flux_error_metrics(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return {
        "l1": float(np.mean(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff * diff))),
        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-30)),
        "max_abs": float(np.max(np.abs(diff))),
    }
