from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch

from flow_matching.solver import ODESolver

from ml.model import WrappedModel, get_model
from ml.transform import (
    denorm_angle_phi,
    denorm_log_feature,
    denorm_position_relative,
    denorm_radius_r,
    load_normalization_params,
    normalize_angle_phi,
    normalize_height,
    normalize_position_relative,
    normalize_radius_r,
    normalize_width,
    polar_to_uv,
    uv_to_polar,
)
from physics.hierarchical_steady_2d import (
    EPSILON,
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    ballistic_exit_2d,
    opposite_face_2d,
    scatter_before_exit_probability,
)


MAT_SCATTERING = 0
MAT_ABSORBING = 1
MAT_SOURCE = 2

ACTIVE = 0
EXITED = 1
ABSORBED = 2

START_INTERNAL = 0
START_BOUNDARY = 1

DEFAULT_LATTICE_PATTERN = np.array(
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

DEFAULT_MATERIAL_PROPERTIES: dict[int, dict[str, float]] = {
    MAT_SCATTERING: {"sigma_a": 0.0, "sigma_s": 1.0},
    MAT_ABSORBING: {"sigma_a": 9.5, "sigma_s": 0.5},
    MAT_SOURCE: {"sigma_a": 0.0, "sigma_s": 1.0},
}

OPPOSITE_FACE = np.array([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=np.int64)


@dataclass(frozen=True)
class RectLatticeProblem:
    refinement: int
    base_cells: int
    domain_size: float
    lattice_pattern: np.ndarray
    material_map: np.ndarray
    sigma_t_map: np.ndarray
    sigma_s_map: np.ndarray
    sigma_a_map: np.ndarray
    cell_widths: np.ndarray
    cell_heights: np.ndarray
    cell_boundaries_x: np.ndarray
    cell_boundaries_y: np.ndarray
    cell_areas: np.ndarray
    source_cells: np.ndarray
    num_cells_x: int
    num_cells_y: int
    total_width: float
    total_height: float


@dataclass(frozen=True)
class Steady2DModelSpec:
    source_type: str
    model_dir: str
    norm_base_path: str
    input_dim: int
    cond_dim: int
    train_sampling: str = "random"


@dataclass(frozen=True)
class ScatteredSampleBatch:
    x_out: np.ndarray
    y_out: np.ndarray
    u_out: np.ndarray
    v_out: np.ndarray
    track_length: np.ndarray
    exit_face: np.ndarray
    pos_on_face: np.ndarray
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class ScatterSplitTransportResult:
    coarse_track: np.ndarray
    coarse_energy: np.ndarray
    coarse_flux: np.ndarray
    totals: dict[str, int]
    diagnostics: dict[str, float]


DEFAULT_BOUNDARY_MODEL_SPEC = Steady2DModelSpec(
    source_type="boundary",
    model_dir="ml/models/2d_boundary_scatter_split",
    norm_base_path="ml/data/norm_params_2d_boundary_scatter_split.npz",
    input_dim=10,
    cond_dim=5,
)
DEFAULT_INTERNAL_MODEL_SPEC = Steady2DModelSpec(
    source_type="internal",
    model_dir="ml/models/2d_internal_scatter_split",
    norm_base_path="ml/data/norm_params_2d_internal_scatter_split.npz",
    input_dim=11,
    cond_dim=6,
)


def _resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _calculate_boundaries_2d(cell_widths: np.ndarray, cell_heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(([0.0], np.cumsum(cell_widths, dtype=np.float64))),
        np.concatenate(([0.0], np.cumsum(cell_heights, dtype=np.float64))),
    )


def build_rect_lattice_problem(
    refinement: int,
    *,
    domain_size: float = 7.0,
    base_cells: int = 7,
    lattice_pattern: np.ndarray | None = None,
    material_properties: dict[int, dict[str, float]] | None = None,
) -> RectLatticeProblem:
    if refinement <= 0:
        raise ValueError(f"refinement must be >= 1, got {refinement}")

    pattern = np.asarray(DEFAULT_LATTICE_PATTERN if lattice_pattern is None else lattice_pattern, dtype=np.int64)
    if pattern.shape != (base_cells, base_cells):
        raise ValueError(f"Expected lattice pattern shape {(base_cells, base_cells)}, got {pattern.shape}")
    props = DEFAULT_MATERIAL_PROPERTIES if material_properties is None else material_properties

    num_cells_x = int(base_cells * refinement)
    num_cells_y = int(base_cells * refinement)
    material_map = np.zeros((num_cells_y, num_cells_x), dtype=np.int64)

    for base_iy in range(base_cells):
        for base_ix in range(base_cells):
            material_map[
                base_iy * refinement : (base_iy + 1) * refinement,
                base_ix * refinement : (base_ix + 1) * refinement,
            ] = int(pattern[base_iy, base_ix])

    sigma_s_map = np.zeros_like(material_map, dtype=np.float64)
    sigma_a_map = np.zeros_like(material_map, dtype=np.float64)
    for mat_id, values in props.items():
        mask = material_map == int(mat_id)
        sigma_s_map[mask] = float(values["sigma_s"])
        sigma_a_map[mask] = float(values["sigma_a"])
    sigma_t_map = sigma_s_map + sigma_a_map

    cell_widths = np.full(num_cells_x, float(domain_size) / num_cells_x, dtype=np.float64)
    cell_heights = np.full(num_cells_y, float(domain_size) / num_cells_y, dtype=np.float64)
    cell_boundaries_x, cell_boundaries_y = _calculate_boundaries_2d(cell_widths, cell_heights)
    cell_areas = np.outer(cell_heights, cell_widths)
    source_cells = np.argwhere(material_map == MAT_SOURCE).astype(np.int64, copy=False)
    if source_cells.size == 0:
        raise RuntimeError("Lattice pattern contains no source cells (material id 2).")

    return RectLatticeProblem(
        refinement=int(refinement),
        base_cells=int(base_cells),
        domain_size=float(domain_size),
        lattice_pattern=pattern.copy(),
        material_map=material_map,
        sigma_t_map=sigma_t_map,
        sigma_s_map=sigma_s_map,
        sigma_a_map=sigma_a_map,
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=cell_boundaries_x,
        cell_boundaries_y=cell_boundaries_y,
        cell_areas=cell_areas,
        source_cells=source_cells,
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        total_width=float(domain_size),
        total_height=float(domain_size),
    )


def _is_state_dict(candidate: object) -> bool:
    return isinstance(candidate, dict) and any(isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items())


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if _is_state_dict(checkpoint):
        return dict(checkpoint)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if _is_state_dict(candidate):
                return dict(candidate)
    raise TypeError("Checkpoint does not contain a recognized state_dict payload.")


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        key_norm = str(key)
        if key_norm.startswith("module."):
            key_norm = key_norm[len("module.") :]
        normalized[key_norm] = value
    return normalized


def _resolve_model_checkpoint(spec: Steady2DModelSpec) -> Path:
    model_dir = Path(spec.model_dir)
    suffix = f"_{spec.train_sampling}" if spec.train_sampling != "random" else ""
    candidates = [
        model_dir / f"best_model{suffix}.pth",
        model_dir / f"trained_model_final{suffix}.pth",
        model_dir / "best_model.pth",
        model_dir / "trained_model_final.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model checkpoint found under {model_dir}")


def _infer_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    input_proj = state_dict["input_proj.weight"]
    hidden_dim = int(input_proj.shape[0])
    block_indices = [
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.endswith(".norm.weight")
    ]
    if not block_indices:
        raise ValueError("Could not infer model depth from checkpoint state_dict.")
    depth = max(block_indices) + 1
    ff_shape = state_dict["blocks.0.ff.0.weight"].shape
    expansion_factor = max(1, int(ff_shape[0] // max(ff_shape[1], 1)))
    return hidden_dim, depth, expansion_factor


def _build_time_grid(num_steps: int, device: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    if num_steps <= 1:
        return torch.tensor([0.0, 1.0], device=device), None
    base_steps = max(1, num_steps // 2)
    fine_steps = max(1, num_steps - base_steps)
    base_grid = torch.linspace(0.0, 0.8, base_steps + 1, device=device)[:-1]
    fine_grid = torch.linspace(0.8, 1.0, fine_steps + 1, device=device)
    time_grid = torch.cat([base_grid, fine_grid])
    return time_grid, time_grid[1] - time_grid[0]


def _continuous_track_length(weight: np.ndarray, sigma_a: np.ndarray, distance: np.ndarray) -> np.ndarray:
    weight_arr = np.asarray(weight, dtype=np.float64)
    sigma_a_arr = np.maximum(np.asarray(sigma_a, dtype=np.float64), 0.0)
    distance_arr = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    optical_depth = sigma_a_arr * distance_arr
    return np.where(
        optical_depth > 1.0e-2,
        weight_arr * (1.0 - np.exp(-optical_depth)) / np.maximum(sigma_a_arr, EPSILON),
        weight_arr * distance_arr,
    )


def _sample_isotropic_uv_numpy(num_particles: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.uniform(0.0, 2.0 * np.pi, size=num_particles)
    mu = rng.uniform(-1.0, 1.0, size=num_particles)
    t = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    u = t * np.cos(phi)
    v = t * np.sin(phi)
    zero = (np.abs(u) < EPSILON) & (np.abs(v) < EPSILON)
    u[zero] = EPSILON
    return u.astype(np.float64), v.astype(np.float64)


def _sample_incoming_plus_x_uv_numpy(num_particles: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if num_particles <= 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    u = np.empty(num_particles, dtype=np.float64)
    v = np.empty(num_particles, dtype=np.float64)
    filled = 0
    while filled < num_particles:
        sample_count = max(8, (num_particles - filled) * 2)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=sample_count)
        mu = rng.uniform(-1.0, 1.0, size=sample_count)
        t = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
        cand_u = t * np.cos(phi)
        cand_v = t * np.sin(phi)
        keep = cand_u > EPSILON
        take = min(int(np.count_nonzero(keep)), num_particles - filled)
        if take <= 0:
            continue
        u[filled : filled + take] = cand_u[keep][:take]
        v[filled : filled + take] = cand_v[keep][:take]
        filled += take
    return u, v


def _ballistic_exit_batch(
    x0: np.ndarray,
    y0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist_left = np.where(u < -EPSILON, (0.0 - x0) / u, np.inf)
    dist_right = np.where(u > EPSILON, (width - x0) / u, np.inf)
    dist_top = np.where(v > EPSILON, (height - y0) / v, np.inf)
    dist_bottom = np.where(v < -EPSILON, (0.0 - y0) / v, np.inf)
    distances = np.stack([dist_left, dist_right, dist_top, dist_bottom], axis=1)
    distances = np.where(distances >= 0.0, distances, np.inf)
    face = np.argmin(distances, axis=1).astype(np.int64)
    distance = distances[np.arange(distances.shape[0]), face].astype(np.float64)
    distance = np.where(np.isfinite(distance), distance, 0.0)
    x_exit = np.clip(x0 + distance * u, 0.0, width)
    y_exit = np.clip(y0 + distance * v, 0.0, height)
    x_exit = np.where(face == FACE_LEFT, 0.0, np.where(face == FACE_RIGHT, width, x_exit))
    y_exit = np.where(face == FACE_BOTTOM, 0.0, np.where(face == FACE_TOP, height, y_exit))
    coord = np.where((face == FACE_LEFT) | (face == FACE_RIGHT), y_exit, x_exit)
    return distance, face, x_exit, y_exit, coord


def _classify_face_and_snap_numpy(
    x: np.ndarray,
    y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    width_arr = np.asarray(width, dtype=np.float64)
    height_arr = np.asarray(height, dtype=np.float64)
    distances = np.stack(
        [
            np.abs(x_arr),
            np.abs(width_arr - x_arr),
            np.abs(height_arr - y_arr),
            np.abs(y_arr),
        ],
        axis=1,
    )
    face = np.argmin(distances, axis=1).astype(np.int64)
    x_snap = np.clip(x_arr, 0.0, width_arr)
    y_snap = np.clip(y_arr, 0.0, height_arr)
    x_snap = np.where(face == FACE_LEFT, 0.0, np.where(face == FACE_RIGHT, width_arr, x_snap))
    y_snap = np.where(face == FACE_BOTTOM, 0.0, np.where(face == FACE_TOP, height_arr, y_snap))
    pos_on_face = np.where((face == FACE_LEFT) | (face == FACE_RIGHT), y_snap, x_snap)
    return face, x_snap, y_snap, pos_on_face


def _enforce_outward_numpy(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out_u = np.asarray(u, dtype=np.float64).copy()
    out_v = np.asarray(v, dtype=np.float64).copy()

    radius = np.sqrt(np.maximum(out_u * out_u + out_v * out_v, 0.0))
    large = radius > 1.0
    if np.any(large):
        out_u[large] /= radius[large]
        out_v[large] /= radius[large]

    normals = np.zeros((face.shape[0], 2), dtype=np.float64)
    normals[face == FACE_LEFT] = (-1.0, 0.0)
    normals[face == FACE_RIGHT] = (1.0, 0.0)
    normals[face == FACE_TOP] = (0.0, 1.0)
    normals[face == FACE_BOTTOM] = (0.0, -1.0)

    direction = np.stack([out_u, out_v], axis=1)
    dot = np.sum(direction * normals, axis=1)
    needs_reflect = dot <= 0.0
    if np.any(needs_reflect):
        direction[needs_reflect] = direction[needs_reflect] - 2.0 * dot[needs_reflect, None] * normals[needs_reflect]

    degenerate = np.linalg.norm(direction, axis=1) < EPSILON
    if np.any(degenerate):
        direction[degenerate] = normals[degenerate]

    return (
        direction[:, 0],
        direction[:, 1],
        {
            "direction_unit_disk_clip_fraction": float(np.mean(large)) if large.size else 0.0,
            "direction_outward_reflect_fraction": float(np.mean(needs_reflect)) if needs_reflect.size else 0.0,
            "direction_degenerate_fix_fraction": float(np.mean(degenerate)) if degenerate.size else 0.0,
        },
    )


def _to_boundary_model_frame(
    entry_face: np.ndarray,
    entry_coord: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face = np.asarray(entry_face, dtype=np.int64)
    coord = np.asarray(entry_coord, dtype=np.float64)
    width_arr = np.asarray(width, dtype=np.float64)
    height_arr = np.asarray(height, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    width_m = np.where((face == FACE_LEFT) | (face == FACE_RIGHT), width_arr, height_arr)
    height_m = np.where((face == FACE_LEFT) | (face == FACE_RIGHT), height_arr, width_arr)

    y_entry = np.empty_like(coord)
    u_m = np.empty_like(u_arr)
    v_m = np.empty_like(v_arr)

    left = face == FACE_LEFT
    right = face == FACE_RIGHT
    top = face == FACE_TOP
    bottom = face == FACE_BOTTOM

    y_entry[left] = coord[left]
    u_m[left] = u_arr[left]
    v_m[left] = v_arr[left]

    y_entry[right] = height_arr[right] - coord[right]
    u_m[right] = -u_arr[right]
    v_m[right] = -v_arr[right]

    y_entry[top] = coord[top]
    u_m[top] = -v_arr[top]
    v_m[top] = u_arr[top]

    y_entry[bottom] = width_arr[bottom] - coord[bottom]
    u_m[bottom] = v_arr[bottom]
    v_m[bottom] = -u_arr[bottom]

    return width_m, height_m, y_entry, u_m, v_m


def _from_boundary_model_frame(
    entry_face: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    x_model: np.ndarray,
    y_model: np.ndarray,
    u_model: np.ndarray,
    v_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face = np.asarray(entry_face, dtype=np.int64)
    width_arr = np.asarray(width, dtype=np.float64)
    height_arr = np.asarray(height, dtype=np.float64)
    x_m = np.asarray(x_model, dtype=np.float64)
    y_m = np.asarray(y_model, dtype=np.float64)
    u_m = np.asarray(u_model, dtype=np.float64)
    v_m = np.asarray(v_model, dtype=np.float64)

    x_phys = np.empty_like(x_m)
    y_phys = np.empty_like(y_m)
    u_phys = np.empty_like(u_m)
    v_phys = np.empty_like(v_m)

    left = face == FACE_LEFT
    right = face == FACE_RIGHT
    top = face == FACE_TOP
    bottom = face == FACE_BOTTOM

    x_phys[left] = x_m[left]
    y_phys[left] = y_m[left]
    u_phys[left] = u_m[left]
    v_phys[left] = v_m[left]

    x_phys[right] = width_arr[right] - x_m[right]
    y_phys[right] = height_arr[right] - y_m[right]
    u_phys[right] = -u_m[right]
    v_phys[right] = -v_m[right]

    x_phys[top] = y_m[top]
    y_phys[top] = height_arr[top] - x_m[top]
    u_phys[top] = v_m[top]
    v_phys[top] = -u_m[top]

    x_phys[bottom] = width_arr[bottom] - y_m[bottom]
    y_phys[bottom] = x_m[bottom]
    u_phys[bottom] = -v_m[bottom]
    v_phys[bottom] = u_m[bottom]

    return x_phys, y_phys, u_phys, v_phys


class SteadyScatterModel:
    def __init__(
        self,
        *,
        spec: Steady2DModelSpec,
        device: str,
        batch_size: int,
        num_ode_steps: int,
    ) -> None:
        self.spec = spec
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)

        checkpoint_path = _resolve_model_checkpoint(spec)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
        hidden_dim, depth, expansion_factor = _infer_architecture(state_dict)
        model = get_model(
            input_dim=spec.input_dim,
            cond_dim=spec.cond_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            expansion_factor=expansion_factor,
        ).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.solver = ODESolver(velocity_model=WrappedModel(model).to(self.device))
        self.norm_params, self.norm_path = load_normalization_params(spec.norm_base_path, train_sampling=spec.train_sampling)
        self.target_dim = spec.input_dim - spec.cond_dim
        self.time_grid, self.step_size = _build_time_grid(num_ode_steps, self.device)
        self.checkpoint_path = str(checkpoint_path)

    def _predict(self, cond_values: np.ndarray, *, seed: int | None) -> np.ndarray:
        if cond_values.shape[0] == 0:
            return np.zeros((0, self.target_dim), dtype=np.float32)

        generator = None
        if self.device != "mps":
            generator = torch.Generator(device=self.device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(int(seed))

        cond_np = np.asarray(cond_values, dtype=np.float32)
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, cond_np.shape[0], self.batch_size):
                stop = min(start + self.batch_size, cond_np.shape[0])
                batch_cond = torch.from_numpy(cond_np[start:stop]).to(self.device)
                if generator is None:
                    noise = torch.randn(batch_cond.shape[0], self.target_dim, device=self.device)
                else:
                    noise = torch.randn(batch_cond.shape[0], self.target_dim, device=self.device, generator=generator)
                x_init = torch.cat([batch_cond, noise], dim=1)
                out = self.solver.sample(
                    time_grid=self.time_grid,
                    x_init=x_init,
                    method="rk4",
                    step_size=self.step_size,
                    return_intermediates=False,
                )
                outputs.append(out[:, self.spec.cond_dim :].detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def sample_scattered_internal(
        self,
        *,
        width: np.ndarray,
        height: np.ndarray,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        sigma_s: np.ndarray,
        seed: int | None = None,
    ) -> ScatteredSampleBatch:
        if self.spec.source_type != "internal":
            raise ValueError("sample_scattered_internal requires an internal-source model.")

        width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
        height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
        x0 = np.asarray(initial_x, dtype=np.float64).reshape(-1)
        y0 = np.asarray(initial_y, dtype=np.float64).reshape(-1)
        u0 = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        v0 = np.asarray(initial_v, dtype=np.float64).reshape(-1)
        sigma_s_arr = np.maximum(np.asarray(sigma_s, dtype=np.float64).reshape(-1), EPSILON)

        phi_in, r_in = uv_to_polar(u0, v0)
        cond = np.stack(
            [
                normalize_width(width_arr * sigma_s_arr, self.norm_params["width_lower"], self.norm_params["width_upper"]),
                normalize_height(height_arr * sigma_s_arr, self.norm_params["height_lower"], self.norm_params["height_upper"]),
                normalize_position_relative(np.clip(x0 / np.maximum(width_arr, EPSILON), 0.0, 1.0)),
                normalize_position_relative(np.clip(y0 / np.maximum(height_arr, EPSILON), 0.0, 1.0)),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        pred = self._predict(cond, seed=seed)
        x_pred = denorm_position_relative(pred[:, 0]) * width_arr
        y_pred = denorm_position_relative(pred[:, 1]) * height_arr
        face, x_snap, y_snap, pos_on_face = _classify_face_and_snap_numpy(x_pred, y_pred, width_arr, height_arr)
        phi_out = denorm_angle_phi(pred[:, 2])
        r_out = denorm_radius_r(pred[:, 3])
        u_pred, v_pred = polar_to_uv(phi_out, r_out)
        u_out, v_out, direction_diag = _enforce_outward_numpy(face, u_pred, v_pred)

        track_length = denorm_log_feature(pred[:, 4], self.norm_params, "time") / sigma_s_arr
        straight_line = np.sqrt(np.maximum((x_snap - x0) ** 2 + (y_snap - y0) ** 2, 0.0))
        track_clip = track_length < straight_line
        track_length = np.maximum(track_length, straight_line)

        diagnostics = {
            **direction_diag,
            "track_lower_bound_clip_fraction": float(np.mean(track_clip)) if track_clip.size else 0.0,
        }
        return ScatteredSampleBatch(
            x_out=x_snap.astype(np.float32),
            y_out=y_snap.astype(np.float32),
            u_out=u_out.astype(np.float32),
            v_out=v_out.astype(np.float32),
            track_length=np.maximum(track_length, 0.0).astype(np.float32),
            exit_face=face.astype(np.int64),
            pos_on_face=pos_on_face.astype(np.float32),
            diagnostics=diagnostics,
        )

    def sample_scattered_boundary(
        self,
        *,
        width: np.ndarray,
        height: np.ndarray,
        entry_face: np.ndarray,
        entry_coord: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        sigma_s: np.ndarray,
        seed: int | None = None,
    ) -> ScatteredSampleBatch:
        if self.spec.source_type != "boundary":
            raise ValueError("sample_scattered_boundary requires a boundary-source model.")

        width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
        height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
        entry_face_arr = np.asarray(entry_face, dtype=np.int64).reshape(-1)
        entry_coord_arr = np.asarray(entry_coord, dtype=np.float64).reshape(-1)
        u0 = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        v0 = np.asarray(initial_v, dtype=np.float64).reshape(-1)
        sigma_s_arr = np.maximum(np.asarray(sigma_s, dtype=np.float64).reshape(-1), EPSILON)

        width_m, height_m, y_entry, u_m, v_m = _to_boundary_model_frame(
            entry_face_arr,
            entry_coord_arr,
            width_arr,
            height_arr,
            u0,
            v0,
        )
        phi_in, r_in = uv_to_polar(u_m, v_m)
        cond = np.stack(
            [
                normalize_width(width_m * sigma_s_arr, self.norm_params["width_lower"], self.norm_params["width_upper"]),
                normalize_height(height_m * sigma_s_arr, self.norm_params["height_lower"], self.norm_params["height_upper"]),
                normalize_position_relative(np.clip(y_entry / np.maximum(height_m, EPSILON), 0.0, 1.0)),
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        pred = self._predict(cond, seed=seed)
        x_model = denorm_position_relative(pred[:, 0]) * width_m
        y_model = denorm_position_relative(pred[:, 1]) * height_m
        face_model, x_model, y_model, _coord_model = _classify_face_and_snap_numpy(x_model, y_model, width_m, height_m)
        phi_out = denorm_angle_phi(pred[:, 2])
        r_out = denorm_radius_r(pred[:, 3])
        u_model, v_model = polar_to_uv(phi_out, r_out)
        u_model, v_model, direction_diag = _enforce_outward_numpy(face_model, u_model, v_model)

        track_length = denorm_log_feature(pred[:, 4], self.norm_params, "time") / sigma_s_arr
        straight_line = np.sqrt(np.maximum(x_model * x_model + (y_model - y_entry) ** 2, 0.0))
        track_clip = track_length < straight_line
        track_length = np.maximum(track_length, straight_line)

        x_phys, y_phys, u_phys, v_phys = _from_boundary_model_frame(
            entry_face_arr,
            width_arr,
            height_arr,
            x_model,
            y_model,
            u_model,
            v_model,
        )
        face_phys, x_snap, y_snap, pos_on_face = _classify_face_and_snap_numpy(x_phys, y_phys, width_arr, height_arr)
        u_out, v_out, post_diag = _enforce_outward_numpy(face_phys, u_phys, v_phys)

        diagnostics = {
            **direction_diag,
            **{f"physical_{key}": value for key, value in post_diag.items()},
            "track_lower_bound_clip_fraction": float(np.mean(track_clip)) if track_clip.size else 0.0,
        }
        return ScatteredSampleBatch(
            x_out=x_snap.astype(np.float32),
            y_out=y_snap.astype(np.float32),
            u_out=u_out.astype(np.float32),
            v_out=v_out.astype(np.float32),
            track_length=np.maximum(track_length, 0.0).astype(np.float32),
            exit_face=face_phys.astype(np.int64),
            pos_on_face=pos_on_face.astype(np.float32),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class ScatterSplitModelSet:
    device: str
    boundary: SteadyScatterModel
    internal: SteadyScatterModel
    conditioning: str = "at_least_one_scatter_in_cell"


def load_scatter_split_steady_models(
    *,
    device: str = "auto",
    boundary_spec: Steady2DModelSpec = DEFAULT_BOUNDARY_MODEL_SPEC,
    internal_spec: Steady2DModelSpec = DEFAULT_INTERNAL_MODEL_SPEC,
    batch_size: int = 1024,
    num_ode_steps: int = 12,
) -> ScatterSplitModelSet:
    resolved_device = _resolve_device(device)
    return ScatterSplitModelSet(
        device=resolved_device,
        boundary=SteadyScatterModel(
            spec=boundary_spec,
            device=resolved_device,
            batch_size=batch_size,
            num_ode_steps=num_ode_steps,
        ),
        internal=SteadyScatterModel(
            spec=internal_spec,
            device=resolved_device,
            batch_size=batch_size,
            num_ode_steps=num_ode_steps,
        ),
    )


def _accumulate_model_diagnostics(
    accum: dict[str, float],
    counts: dict[str, int],
    diagnostics: dict[str, float],
    weight: int,
    prefix: str,
) -> None:
    for key, value in diagnostics.items():
        named = f"{prefix}_{key}"
        accum[named] = accum.get(named, 0.0) + float(value) * weight
        counts[named] = counts.get(named, 0) + int(weight)


def _finalize_weighted_diagnostics(accum: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {key: float(accum[key] / max(counts.get(key, 1), 1)) for key in sorted(accum)}


def run_scatter_split_lattice(
    problem: RectLatticeProblem,
    models: ScatterSplitModelSet,
    *,
    num_particles: int,
    source_mode: str = "internal_start",
    randomize_start_pos: bool = True,
    start_position_relative: tuple[float, float] = (0.5, 0.5),
    energy_threshold: float = 1.0e-8,
    max_iterations: int = 10_000,
    seed: int = 42,
) -> ScatterSplitTransportResult:
    if num_particles <= 0:
        raise ValueError(f"num_particles must be positive, got {num_particles}")
    if source_mode not in {"internal_start", "boundary_only"}:
        raise ValueError(f"Unsupported source_mode={source_mode!r}")

    rng = np.random.default_rng(seed)
    num_cells = problem.num_cells_x * problem.num_cells_y

    cell_x = np.zeros(num_particles, dtype=np.int64)
    cell_y = np.zeros(num_particles, dtype=np.int64)
    start_kind = np.zeros(num_particles, dtype=np.int8)
    face = np.full(num_particles, -1, dtype=np.int64)
    coord = np.zeros(num_particles, dtype=np.float64)
    x = np.zeros(num_particles, dtype=np.float64)
    y = np.zeros(num_particles, dtype=np.float64)
    u = np.zeros(num_particles, dtype=np.float64)
    v = np.zeros(num_particles, dtype=np.float64)
    weight = np.ones(num_particles, dtype=np.float64)
    status = np.zeros(num_particles, dtype=np.int8)

    if source_mode == "internal_start":
        selection = rng.integers(0, problem.source_cells.shape[0], size=num_particles)
        cell_y[:] = problem.source_cells[selection, 0]
        cell_x[:] = problem.source_cells[selection, 1]
        start_kind[:] = START_INTERNAL
        width0 = problem.cell_widths[cell_x]
        height0 = problem.cell_heights[cell_y]
        if randomize_start_pos:
            x[:] = rng.random(num_particles) * width0
            y[:] = rng.random(num_particles) * height0
        else:
            x[:] = float(start_position_relative[0]) * width0
            y[:] = float(start_position_relative[1]) * height0
        u[:], v[:] = _sample_isotropic_uv_numpy(num_particles, rng)
    else:
        cell_x[:] = 0
        cell_y[:] = rng.integers(0, problem.num_cells_y, size=num_particles)
        start_kind[:] = START_BOUNDARY
        face[:] = FACE_LEFT
        coord[:] = rng.random(num_particles) * problem.cell_heights[cell_y]
        u[:], v[:] = _sample_incoming_plus_x_uv_numpy(num_particles, rng)

    track_flat = np.zeros(num_cells, dtype=np.float64)
    energy_flat = np.zeros(num_cells, dtype=np.float64)
    totals = {"left": 0, "right": 0, "top": 0, "bottom": 0, "absorb": 0}
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}

    def tally_and_advance(
        active_idx: np.ndarray,
        *,
        cell_ix: np.ndarray,
        cell_iy: np.ndarray,
        path_length: np.ndarray,
        sigma_a_local: np.ndarray,
        exit_face: np.ndarray,
        pos_on_face: np.ndarray,
        u_out: np.ndarray,
        v_out: np.ndarray,
    ) -> None:
        if active_idx.size == 0:
            return

        entry_weight = weight[active_idx].astype(np.float64, copy=True)
        track_add = _continuous_track_length(entry_weight, sigma_a_local, path_length)
        exit_weight = entry_weight * np.exp(-sigma_a_local * path_length)
        deposit = entry_weight - exit_weight
        lin_idx = cell_iy * problem.num_cells_x + cell_ix
        np.add.at(track_flat, lin_idx, track_add)
        np.add.at(energy_flat, lin_idx, deposit)

        absorbed = exit_weight < energy_threshold
        if np.any(absorbed):
            totals["absorb"] += int(np.count_nonzero(absorbed))
            status[active_idx[absorbed]] = ABSORBED

        survivors = ~absorbed
        if not np.any(survivors):
            return

        surv_idx = active_idx[survivors]
        surv_face = exit_face[survivors]
        surv_coord = pos_on_face[survivors]
        surv_u = u_out[survivors]
        surv_v = v_out[survivors]
        surv_weight = exit_weight[survivors]
        surv_ix = cell_ix[survivors]
        surv_iy = cell_iy[survivors]

        new_ix = surv_ix + (surv_face == FACE_RIGHT).astype(np.int64) - (surv_face == FACE_LEFT).astype(np.int64)
        new_iy = surv_iy + (surv_face == FACE_TOP).astype(np.int64) - (surv_face == FACE_BOTTOM).astype(np.int64)
        mesh_exit = (
            (new_ix < 0)
            | (new_ix >= problem.num_cells_x)
            | (new_iy < 0)
            | (new_iy >= problem.num_cells_y)
        )

        totals["left"] += int(np.count_nonzero((surv_face == FACE_LEFT) & mesh_exit))
        totals["right"] += int(np.count_nonzero((surv_face == FACE_RIGHT) & mesh_exit))
        totals["top"] += int(np.count_nonzero((surv_face == FACE_TOP) & mesh_exit))
        totals["bottom"] += int(np.count_nonzero((surv_face == FACE_BOTTOM) & mesh_exit))
        if np.any(mesh_exit):
            status[surv_idx[mesh_exit]] = EXITED

        interior = ~mesh_exit
        if not np.any(interior):
            return

        cont_idx = surv_idx[interior]
        cell_x[cont_idx] = new_ix[interior]
        cell_y[cont_idx] = new_iy[interior]
        start_kind[cont_idx] = START_BOUNDARY
        face[cont_idx] = OPPOSITE_FACE[surv_face[interior]]
        coord[cont_idx] = surv_coord[interior]
        u[cont_idx] = surv_u[interior]
        v[cont_idx] = surv_v[interior]
        weight[cont_idx] = surv_weight[interior]

    def process_internal(active_idx: np.ndarray) -> None:
        cell_ix = cell_x[active_idx]
        cell_iy = cell_y[active_idx]
        width_local = problem.cell_widths[cell_ix]
        height_local = problem.cell_heights[cell_iy]
        sigma_s_local = np.maximum(problem.sigma_s_map[cell_iy, cell_ix], 0.0)
        sigma_a_local = np.maximum(problem.sigma_a_map[cell_iy, cell_ix], 0.0)
        x0 = x[active_idx]
        y0 = y[active_idx]
        u0 = u[active_idx]
        v0 = v[active_idx]

        ballistic_length, exit_face, _x_exit, _y_exit, exit_coord = _ballistic_exit_batch(
            x0, y0, u0, v0, width_local, height_local
        )
        scatter_prob = -np.expm1(-sigma_s_local * ballistic_length)
        scatter_prob = np.clip(scatter_prob, 0.0, 1.0)
        scatter_mask = rng.random(active_idx.size) < scatter_prob

        diagnostic_sums["scatter_probability_sum"] = diagnostic_sums.get("scatter_probability_sum", 0.0) + float(np.sum(scatter_prob))
        diagnostic_counts["scatter_probability_sum"] = diagnostic_counts.get("scatter_probability_sum", 0) + int(active_idx.size)
        diagnostic_sums["sampled_scatter_count"] = diagnostic_sums.get("sampled_scatter_count", 0.0) + float(np.count_nonzero(scatter_mask))
        diagnostic_counts["sampled_scatter_count"] = diagnostic_counts.get("sampled_scatter_count", 0) + 1
        diagnostic_sums["analytic_no_scatter_count"] = diagnostic_sums.get("analytic_no_scatter_count", 0.0) + float(np.count_nonzero(~scatter_mask))
        diagnostic_counts["analytic_no_scatter_count"] = diagnostic_counts.get("analytic_no_scatter_count", 0) + 1

        if np.any(~scatter_mask):
            idx = active_idx[~scatter_mask]
            tally_and_advance(
                idx,
                cell_ix=cell_ix[~scatter_mask],
                cell_iy=cell_iy[~scatter_mask],
                path_length=ballistic_length[~scatter_mask],
                sigma_a_local=sigma_a_local[~scatter_mask],
                exit_face=exit_face[~scatter_mask],
                pos_on_face=exit_coord[~scatter_mask],
                u_out=u0[~scatter_mask],
                v_out=v0[~scatter_mask],
            )

        if np.any(scatter_mask):
            idx = active_idx[scatter_mask]
            diagnostic_sums["internal_model_calls"] = diagnostic_sums.get("internal_model_calls", 0.0) + float(idx.size)
            diagnostic_counts["internal_model_calls"] = diagnostic_counts.get("internal_model_calls", 0) + 1
            sampled = models.internal.sample_scattered_internal(
                width=width_local[scatter_mask],
                height=height_local[scatter_mask],
                initial_x=x0[scatter_mask],
                initial_y=y0[scatter_mask],
                initial_u=u0[scatter_mask],
                initial_v=v0[scatter_mask],
                sigma_s=np.maximum(sigma_s_local[scatter_mask], EPSILON),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            _accumulate_model_diagnostics(
                diagnostic_sums,
                diagnostic_counts,
                sampled.diagnostics,
                idx.size,
                prefix="internal_model",
            )
            tally_and_advance(
                idx,
                cell_ix=cell_ix[scatter_mask],
                cell_iy=cell_iy[scatter_mask],
                path_length=np.asarray(sampled.track_length, dtype=np.float64),
                sigma_a_local=sigma_a_local[scatter_mask],
                exit_face=np.asarray(sampled.exit_face, dtype=np.int64),
                pos_on_face=np.asarray(sampled.pos_on_face, dtype=np.float64),
                u_out=np.asarray(sampled.u_out, dtype=np.float64),
                v_out=np.asarray(sampled.v_out, dtype=np.float64),
            )

    def process_boundary(active_idx: np.ndarray) -> None:
        cell_ix = cell_x[active_idx]
        cell_iy = cell_y[active_idx]
        width_local = problem.cell_widths[cell_ix]
        height_local = problem.cell_heights[cell_iy]
        sigma_s_local = np.maximum(problem.sigma_s_map[cell_iy, cell_ix], 0.0)
        sigma_a_local = np.maximum(problem.sigma_a_map[cell_iy, cell_ix], 0.0)
        face_local = face[active_idx]
        coord_local = coord[active_idx]
        u0 = u[active_idx]
        v0 = v[active_idx]

        x0 = np.where(
            face_local == FACE_LEFT, 0.0,
            np.where(face_local == FACE_RIGHT, width_local,
            coord_local))
        y0 = np.where(
            face_local == FACE_BOTTOM, 0.0,
            np.where(face_local == FACE_TOP, height_local,
            coord_local))
        x0 = np.where(face_local == FACE_LEFT, x0 + 1.0e-9, np.where(face_local == FACE_RIGHT, x0 - 1.0e-9, x0))
        y0 = np.where(face_local == FACE_BOTTOM, y0 + 1.0e-9, np.where(face_local == FACE_TOP, y0 - 1.0e-9, y0))

        ballistic_length, exit_face, _x_exit, _y_exit, exit_coord = _ballistic_exit_batch(
            x0, y0, u0, v0, width_local, height_local
        )
        scatter_prob = -np.expm1(-sigma_s_local * ballistic_length)
        scatter_prob = np.clip(scatter_prob, 0.0, 1.0)
        scatter_mask = rng.random(active_idx.size) < scatter_prob

        diagnostic_sums["scatter_probability_sum"] = diagnostic_sums.get("scatter_probability_sum", 0.0) + float(np.sum(scatter_prob))
        diagnostic_counts["scatter_probability_sum"] = diagnostic_counts.get("scatter_probability_sum", 0) + int(active_idx.size)
        diagnostic_sums["sampled_scatter_count"] = diagnostic_sums.get("sampled_scatter_count", 0.0) + float(np.count_nonzero(scatter_mask))
        diagnostic_counts["sampled_scatter_count"] = diagnostic_counts.get("sampled_scatter_count", 0) + 1
        diagnostic_sums["analytic_no_scatter_count"] = diagnostic_sums.get("analytic_no_scatter_count", 0.0) + float(np.count_nonzero(~scatter_mask))
        diagnostic_counts["analytic_no_scatter_count"] = diagnostic_counts.get("analytic_no_scatter_count", 0) + 1

        if np.any(~scatter_mask):
            idx = active_idx[~scatter_mask]
            tally_and_advance(
                idx,
                cell_ix=cell_ix[~scatter_mask],
                cell_iy=cell_iy[~scatter_mask],
                path_length=ballistic_length[~scatter_mask],
                sigma_a_local=sigma_a_local[~scatter_mask],
                exit_face=exit_face[~scatter_mask],
                pos_on_face=exit_coord[~scatter_mask],
                u_out=u0[~scatter_mask],
                v_out=v0[~scatter_mask],
            )

        if np.any(scatter_mask):
            idx = active_idx[scatter_mask]
            diagnostic_sums["boundary_model_calls"] = diagnostic_sums.get("boundary_model_calls", 0.0) + float(idx.size)
            diagnostic_counts["boundary_model_calls"] = diagnostic_counts.get("boundary_model_calls", 0) + 1
            sampled = models.boundary.sample_scattered_boundary(
                width=width_local[scatter_mask],
                height=height_local[scatter_mask],
                entry_face=face_local[scatter_mask],
                entry_coord=coord_local[scatter_mask],
                initial_u=u0[scatter_mask],
                initial_v=v0[scatter_mask],
                sigma_s=np.maximum(sigma_s_local[scatter_mask], EPSILON),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            _accumulate_model_diagnostics(
                diagnostic_sums,
                diagnostic_counts,
                sampled.diagnostics,
                idx.size,
                prefix="boundary_model",
            )
            tally_and_advance(
                idx,
                cell_ix=cell_ix[scatter_mask],
                cell_iy=cell_iy[scatter_mask],
                path_length=np.asarray(sampled.track_length, dtype=np.float64),
                sigma_a_local=sigma_a_local[scatter_mask],
                exit_face=np.asarray(sampled.exit_face, dtype=np.int64),
                pos_on_face=np.asarray(sampled.pos_on_face, dtype=np.float64),
                u_out=np.asarray(sampled.u_out, dtype=np.float64),
                v_out=np.asarray(sampled.v_out, dtype=np.float64),
            )

    for _iteration in range(int(max_iterations)):
        active_internal = np.flatnonzero((status == ACTIVE) & (start_kind == START_INTERNAL))
        active_boundary = np.flatnonzero((status == ACTIVE) & (start_kind == START_BOUNDARY))
        if active_internal.size == 0 and active_boundary.size == 0:
            break
        if active_internal.size:
            process_internal(active_internal)
        if active_boundary.size:
            process_boundary(active_boundary)
    else:
        stuck = np.flatnonzero(status == ACTIVE)
        if stuck.size:
            totals["absorb"] += int(stuck.size)
            status[stuck] = ABSORBED
            diagnostic_sums["max_iterations_terminated_count"] = diagnostic_sums.get("max_iterations_terminated_count", 0.0) + float(stuck.size)
            diagnostic_counts["max_iterations_terminated_count"] = diagnostic_counts.get("max_iterations_terminated_count", 0) + 1

    coarse_track = track_flat.reshape(problem.num_cells_y, problem.num_cells_x) / float(num_particles)
    coarse_energy = energy_flat.reshape(problem.num_cells_y, problem.num_cells_x) / float(num_particles)
    coarse_flux = np.divide(
        coarse_track,
        problem.cell_areas,
        out=np.zeros_like(coarse_track),
        where=problem.cell_areas > 0.0,
    )
    diagnostics = _finalize_weighted_diagnostics(diagnostic_sums, diagnostic_counts)
    diagnostics["active_residual_count"] = float(np.count_nonzero(status == ACTIVE))
    diagnostics["conditioned_on_scatter"] = 1.0

    return ScatterSplitTransportResult(
        coarse_track=coarse_track,
        coarse_energy=coarse_energy,
        coarse_flux=coarse_flux,
        totals=totals,
        diagnostics=diagnostics,
    )


def run_direct_fine_mc_reference(
    problem: RectLatticeProblem,
    *,
    num_particles: int,
    source_mode: str = "internal_start",
    randomize_start_pos: bool = True,
    start_position_relative: tuple[float, float] = (0.5, 0.5),
    energy_threshold: float = 1.0e-8,
    seed: int = 123,
) -> dict[str, object]:
    from physics.mc2d import run_mc_simulation_2d_njit_wrapper

    if num_particles <= 0:
        raise ValueError(f"num_particles must be positive, got {num_particles}")
    if source_mode not in {"internal_start", "boundary_only"}:
        raise ValueError(f"Unsupported source_mode={source_mode!r}")

    if source_mode == "boundary_only":
        energy, track, stats = run_mc_simulation_2d_njit_wrapper(
            source_mode="boundary_only",
            num_cells_x=problem.num_cells_x,
            num_cells_y=problem.num_cells_y,
            cell_widths=problem.cell_widths,
            cell_heights=problem.cell_heights,
            cell_boundaries_x=problem.cell_boundaries_x,
            cell_boundaries_y=problem.cell_boundaries_y,
            total_width=problem.total_width,
            total_height=problem.total_height,
            num_particles=num_particles,
            start_cell_indices=(0, 0),
            start_position_relative=start_position_relative,
            scatter_ratio=1.0,
            energy_threshold=energy_threshold,
            seed=seed,
            reflected_boundaries=False,
            angular_sampling=1,
            sigma_t_per_cell=problem.sigma_t_map,
            sigma_a_per_cell=problem.sigma_a_map,
            randomize_position_in_cell=randomize_start_pos,
            verbose=False,
        )
    else:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(num_particles, np.full(problem.source_cells.shape[0], 1.0 / problem.source_cells.shape[0]))
        energy_accum = np.zeros((problem.num_cells_y, problem.num_cells_x), dtype=np.float64)
        track_accum = np.zeros((problem.num_cells_y, problem.num_cells_x), dtype=np.float64)
        stats_accum = np.zeros(8, dtype=np.int64)
        total_launched = 0

        for cell_idx, launched in enumerate(counts):
            if launched <= 0:
                continue
            start_iy = int(problem.source_cells[cell_idx, 0])
            start_ix = int(problem.source_cells[cell_idx, 1])
            energy_i, track_i, stats_i = run_mc_simulation_2d_njit_wrapper(
                source_mode="internal_start",
                num_cells_x=problem.num_cells_x,
                num_cells_y=problem.num_cells_y,
                cell_widths=problem.cell_widths,
                cell_heights=problem.cell_heights,
                cell_boundaries_x=problem.cell_boundaries_x,
                cell_boundaries_y=problem.cell_boundaries_y,
                total_width=problem.total_width,
                total_height=problem.total_height,
                num_particles=int(launched),
                start_cell_indices=(start_ix, start_iy),
                start_position_relative=start_position_relative,
                scatter_ratio=1.0,
                energy_threshold=energy_threshold,
                seed=seed + cell_idx + 1,
                reflected_boundaries=False,
                sigma_t_per_cell=problem.sigma_t_map,
                sigma_a_per_cell=problem.sigma_a_map,
                randomize_position_in_cell=randomize_start_pos,
                verbose=False,
            )
            energy_accum += energy_i * float(launched)
            track_accum += track_i * float(launched)
            stats_accum += np.asarray(stats_i, dtype=np.int64)
            total_launched += int(launched)

        if total_launched <= 0:
            raise RuntimeError("MC reference launched zero particles.")
        energy = energy_accum / float(total_launched)
        track = track_accum / float(total_launched)
        stats = tuple(int(value) for value in stats_accum.tolist())

    flux = np.divide(track, problem.cell_areas, out=np.zeros_like(track), where=problem.cell_areas > 0.0)
    return {"energy": energy, "track": track, "flux": flux, "stats": stats}


def compute_flux_error_metrics(reference_flux: np.ndarray, model_flux: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference_flux, dtype=np.float64)
    pred = np.asarray(model_flux, dtype=np.float64)
    mask = np.isfinite(ref) & np.isfinite(pred)
    if not np.any(mask):
        return {"l1": math.nan, "l2": math.nan, "rel_l2": math.nan, "max_abs": math.nan}
    diff = pred[mask] - ref[mask]
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(ref[mask] * ref[mask])))
    rel_l2 = float(l2 / denom) if denom > 0.0 else math.nan
    max_abs = float(np.max(np.abs(diff)))
    return {"l1": l1, "l2": l2, "rel_l2": rel_l2, "max_abs": max_abs}


__all__ = [
    "DEFAULT_BOUNDARY_MODEL_SPEC",
    "DEFAULT_INTERNAL_MODEL_SPEC",
    "DEFAULT_LATTICE_PATTERN",
    "DEFAULT_MATERIAL_PROPERTIES",
    "MAT_ABSORBING",
    "MAT_SCATTERING",
    "MAT_SOURCE",
    "RectLatticeProblem",
    "ScatterSplitModelSet",
    "ScatterSplitTransportResult",
    "ScatteredSampleBatch",
    "Steady2DModelSpec",
    "SteadyScatterModel",
    "build_rect_lattice_problem",
    "compute_flux_error_metrics",
    "load_scatter_split_steady_models",
    "run_direct_fine_mc_reference",
    "run_scatter_split_lattice",
]
