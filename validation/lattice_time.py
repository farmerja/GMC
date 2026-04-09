#!/usr/bin/env python3
"""
Rectangle-mesh single-group CFM lattice solver.

"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

# Allow running from a repo checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flow_matching.solver import ODESolver
    from ml.model import WrappedModel, get_model
    from ml.transform import (
        decode_direction_2d,
        denorm_angle_phi,
        denorm_length_target,
        denorm_linear_scale_logit,
        denorm_position_relative,
        denorm_radius_r,
        load_normalization_params,
        normalize_angle_phi,
        normalize_direction_encoding_2d as normalize_direction_encoding_2d_shared,
        normalize_height,
        normalize_log_scale_logit,
        normalize_position_relative,
        normalize_radius_r,
        normalize_width,
        polar_to_uv,
        uv_to_polar,
    )
    HAS_ML = True
except Exception as exc:  # pragma: no cover - import environment dependent
    print(f"Warning: ML dependencies not found ({exc}).")
    HAS_ML = False


# -----------------------------------------------------------------------------
# Logging / general constants
# -----------------------------------------------------------------------------
log = logging.getLogger("rect_lattice_single_group_cfm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EPSILON = 1.0e-12
ENERGY_THRESHOLD = 1.0e-8
DEFAULT_SEED = 52
DEFAULT_MC_REFERENCE_SEED = 450
DEFAULT_MC_REPLICATES = 5
MAX_ITERATIONS = 100000
CFM_MAX_PARTICLES_PER_PASS = 100000
PROGRESS_HEARTBEAT_SEC = 2.0
ENABLE_TF32 = True
USE_TORCH_COMPILE = False
USE_AMP = False
AMP_DTYPE = torch.bfloat16
TIME_GRID_STEPS = 12
MC_REFERENCE_MAX_STEPS = 200000

# -----------------------------------------------------------------------------
# Lattice defaults reused from the original rectangular benchmark
# -----------------------------------------------------------------------------
DOMAIN_SIZE = 7.0
BASE_CELLS = 7
MESH_REFINEMENT = 16
TOTAL_WIDTH = DOMAIN_SIZE
TOTAL_HEIGHT = DOMAIN_SIZE

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
    dtype=int,
)

REGION_SCATTERING = 0
REGION_ABSORBING = 1
REGION_SOURCE = 2

MATERIAL_PROPERTIES = {
    REGION_SCATTERING: {"sigma_a": 0.0, "sigma_s": 1.0},
    REGION_ABSORBING: {"sigma_a": 9.5, "sigma_s": 0.5},
    REGION_SOURCE: {"sigma_a": 0.0, "sigma_s": 1.0},
}

PARTICLE_VELOCITY = 1.0
MAX_TIME = 3.3
NUM_TIME_BINS = 30
DT = MAX_TIME / NUM_TIME_BINS
NUM_PARTICLES = 10000

SOURCE_TIME_PROFILE = "pulse"
PULSE_TIME = 0.0
GAUSSIAN_MEAN = 1.0
GAUSSIAN_STD = 0.2
STEP_ON_TIME = 0.0
STEP_OFF_TIME = 5.0

RANDOMIZE_START_POS_IN_CELL = True
START_POS_X_REL = 0.5
START_POS_Y_REL = 0.5

TRAIN_SAMPLING = "random"
DIRECTION_ENCODING_2D = "polar"

DEFAULT_BOUNDARY_MODEL_DIR = "ml/models/2d_timedep_boundary"
DEFAULT_BOUNDARY_NORM_FILE = "ml/data/norm_params_2d_timedep_boundary.npy"
DEFAULT_INTERNAL_MODEL_DIR = "ml/models/2d_timedep_internal"
DEFAULT_INTERNAL_NORM_FILE = "ml/data/norm_params_2d_timedep_internal.npy"

DEFAULT_BOUNDARY_INPUT_DIM = 12
DEFAULT_BOUNDARY_COND_DIM = 6
DEFAULT_BOUNDARY_TARGET_DIM = 6
DEFAULT_INTERNAL_INPUT_DIM = 13
DEFAULT_INTERNAL_COND_DIM = 7
DEFAULT_INTERNAL_TARGET_DIM = 6

FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3
FACE_TO_FACE = np.array([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=np.int64)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RectangularLatticeProblem:
    material_map: np.ndarray
    source_cells: np.ndarray  # (num_source, 2) -> [iy, ix]
    cell_widths: np.ndarray
    cell_heights: np.ndarray
    cell_boundaries_x: np.ndarray
    cell_boundaries_y: np.ndarray
    cell_areas: np.ndarray
    sigma_a_map: np.ndarray
    sigma_s_map: np.ndarray
    sigma_a_flat: np.ndarray
    sigma_s_flat: np.ndarray
    num_cells_x: int
    num_cells_y: int
    mesh_refinement: int
    source_mode: str = "internal_start"


@dataclass(frozen=True)
class ModelBundle:
    solver: object
    norm_params: dict
    cond_dim: int
    target_dim: int


@dataclass(frozen=True)
class RuntimeConfig:
    num_particles: int
    mesh_refinement: int
    output_dir: str
    seed: int
    mc_seed: int
    mc_ref_seed: int
    mc_replicates: int
    mc_ref_particles: int
    boundary_model_dir: str
    boundary_norm_file: str
    internal_model_dir: str
    internal_norm_file: str
    train_sampling: str
    device: str
    randomize_start_pos: bool
    run_mc_reference: bool


# -----------------------------------------------------------------------------
# Lightweight fallbacks / generic helpers
# -----------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"



def maybe_enable_tf32(device: str) -> None:
    if device == "cuda" and ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


@contextmanager
def global_random_seed(seed: int):
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)



def build_time_grid(
    num_steps: int = TIME_GRID_STEPS,
    split: float = 0.8,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    dev = device or get_device()
    num_steps = max(1, int(num_steps))

    if num_steps == 1:
        grid = torch.tensor([0.0], device=dev)
        return grid, None
    if num_steps == 2:
        grid = torch.tensor([0.0, 1.0], device=dev)
        return grid, grid[1] - grid[0]

    base_count = max(2, num_steps // 2)
    fine_count = max(2, num_steps - base_count + 1)

    base_grid = torch.linspace(0.0, split, base_count, device=dev)
    fine_grid = torch.linspace(split, 1.0, fine_count, device=dev)[1:]
    t_grid = torch.cat([base_grid, fine_grid])

    if t_grid.numel() != num_steps:
        t_grid = torch.linspace(0.0, 1.0, num_steps, device=dev)

    step_size = t_grid[1] - t_grid[0] if t_grid.numel() > 1 else None
    return t_grid, step_size



def _maybe_compile_model(model: torch.nn.Module) -> torch.nn.Module:
    if not USE_TORCH_COMPILE or not hasattr(torch, "compile"):
        return model
    try:  # pragma: no cover - compile path depends on runtime
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        return model


@contextmanager
def amp_context(device: str):
    if USE_AMP and device == "cuda":
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            yield
    else:
        yield


# -----------------------------------------------------------------------------
# Checkpoint/model loading
# -----------------------------------------------------------------------------
def _resolve_norm_params_file(base_path: str, train_sampling: str) -> str:
    if train_sampling == "random":
        return base_path
    stem, suffix = os.path.splitext(base_path)
    return f"{stem}_{train_sampling}{suffix}"



def _candidate_model_paths(model_dir: str, train_sampling: str) -> list[str]:
    sampling_suffix = f"_{train_sampling}" if train_sampling != "random" else ""
    candidates = [
        os.path.join(model_dir, f"best_model{sampling_suffix}.pth"),
        os.path.join(model_dir, f"trained_model_final{sampling_suffix}.pth"),
        os.path.join(model_dir, "best_model.pth"),
        os.path.join(model_dir, "trained_model_final.pth"),
    ]
    return list(dict.fromkeys(candidates))



def _extract_state_dict(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return None
    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        return ckpt_obj["model_state_dict"]
    return ckpt_obj



def _strip_common_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict



def _infer_arch_from_state_dict(
    state_dict: dict,
    default_input_dim: int,
    default_cond_dim: int,
    default_target_dim: int,
) -> Tuple[int, int, int, int, int, int]:
    input_dim = int(default_input_dim)
    cond_dim = int(default_cond_dim)
    target_dim = int(default_target_dim)
    hidden_dim = 128
    depth = 8
    expansion_factor = 1

    w_in = state_dict.get("input_proj.weight")
    w_out = state_dict.get("output.weight")
    if isinstance(w_in, torch.Tensor) and w_in.ndim == 2:
        hidden_dim = int(w_in.shape[0])
        input_dim = int(w_in.shape[1])
    if isinstance(w_out, torch.Tensor) and w_out.ndim == 2:
        target_dim = int(w_out.shape[0])
    if input_dim > target_dim:
        cond_dim = int(input_dim - target_dim)

    block_indices = []
    for key in state_dict.keys():
        match = re.match(r"^blocks\.(\d+)\.", key)
        if match:
            block_indices.append(int(match.group(1)))
    if block_indices:
        depth = max(block_indices) + 1

    ff0 = state_dict.get("blocks.0.ff.0.weight")
    if isinstance(ff0, torch.Tensor) and ff0.ndim == 2 and hidden_dim > 0:
        expansion_factor = max(1, int(round(float(ff0.shape[0]) / float(hidden_dim))))

    return input_dim, cond_dim, target_dim, hidden_dim, depth, expansion_factor



def _load_solver_from_checkpoint(
    model_dir: str,
    norm_file: str,
    default_input_dim: int,
    default_cond_dim: int,
    default_target_dim: int,
    train_sampling: str,
    device: str,
) -> ModelBundle:
    if not HAS_ML:
        raise RuntimeError("CFM models requested but the ML stack could not be imported.")

    model_paths = _candidate_model_paths(model_dir, train_sampling)
    last_error: Optional[Exception] = None
    loaded_solver = None
    loaded_path = None

    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
        try:
            ckpt_obj = torch.load(model_path, map_location=device)
            state_dict = _extract_state_dict(ckpt_obj)
            if state_dict is None:
                raise RuntimeError(f"Unsupported checkpoint payload type: {type(ckpt_obj)}")
            state_dict = _strip_common_prefix(state_dict)
            (
                input_dim,
                cond_dim,
                target_dim,
                hidden_dim,
                depth,
                expansion_factor,
            ) = _infer_arch_from_state_dict(
                state_dict,
                default_input_dim=default_input_dim,
                default_cond_dim=default_cond_dim,
                default_target_dim=default_target_dim,
            )
            model = get_model(
                input_dim=input_dim,
                cond_dim=cond_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                expansion_factor=expansion_factor,
            ).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            solver = ODESolver(velocity_model=WrappedModel(_maybe_compile_model(model)))
            solver._input_dim = int(input_dim)
            solver._cond_dim = int(cond_dim)
            solver._target_dim = int(target_dim)
            loaded_solver = solver
            loaded_path = model_path
            break
        except Exception as exc:  # pragma: no cover - depends on user checkpoint
            last_error = exc
            log.warning("Failed to load %s (%s)", model_path, exc)

    if loaded_solver is None:
        raise FileNotFoundError(
            f"Could not load a checkpoint from {model_dir}. Tried: {model_paths}. "
            f"Last error: {last_error}"
        )

    norm_params, _ = load_normalization_params(_resolve_norm_params_file(norm_file, train_sampling))
    norm_params["cond_dim"] = int(getattr(loaded_solver, "_cond_dim", default_cond_dim))
    norm_params["target_dim"] = int(getattr(loaded_solver, "_target_dim", default_target_dim))

    log.info(
        "Loaded model %s (cond_dim=%d target_dim=%d)",
        loaded_path,
        int(norm_params["cond_dim"]),
        int(norm_params["target_dim"]),
    )
    return ModelBundle(
        solver=loaded_solver,
        norm_params=norm_params,
        cond_dim=int(norm_params["cond_dim"]),
        target_dim=int(norm_params["target_dim"]),
    )



def load_rect_models(config: RuntimeConfig) -> Tuple[ModelBundle, ModelBundle]:
    boundary = _load_solver_from_checkpoint(
        model_dir=config.boundary_model_dir,
        norm_file=config.boundary_norm_file,
        default_input_dim=DEFAULT_BOUNDARY_INPUT_DIM,
        default_cond_dim=DEFAULT_BOUNDARY_COND_DIM,
        default_target_dim=DEFAULT_BOUNDARY_TARGET_DIM,
        train_sampling=config.train_sampling,
        device=config.device,
    )
    internal = _load_solver_from_checkpoint(
        model_dir=config.internal_model_dir,
        norm_file=config.internal_norm_file,
        default_input_dim=DEFAULT_INTERNAL_INPUT_DIM,
        default_cond_dim=DEFAULT_INTERNAL_COND_DIM,
        default_target_dim=DEFAULT_INTERNAL_TARGET_DIM,
        train_sampling=config.train_sampling,
        device=config.device,
    )
    return boundary, internal


# -----------------------------------------------------------------------------
# Geometry / lattice helpers
# -----------------------------------------------------------------------------
def _coord_to_idx(coord: float, total_cells: int, total_length: float) -> int:
    return int(coord * total_cells / total_length)



def calculate_boundaries_2d(cell_widths: np.ndarray, cell_heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bx = np.concatenate(([0.0], np.cumsum(np.asarray(cell_widths, dtype=np.float64))))
    by = np.concatenate(([0.0], np.cumsum(np.asarray(cell_heights, dtype=np.float64))))
    return bx, by



def create_rect_lattice_material_map(
    base_cells: int = BASE_CELLS,
    refinement: int = MESH_REFINEMENT,
    domain_size: float = DOMAIN_SIZE,
) -> np.ndarray:
    num_cells_x = base_cells * refinement
    num_cells_y = base_cells * refinement
    material_map = np.zeros((num_cells_y, num_cells_x), dtype=np.int64)
    base_cell_size = domain_size / base_cells
    for base_i in range(base_cells):
        for base_j in range(base_cells):
            mat_type = int(LATTICE_PATTERN[base_i, base_j])
            iy_start = _coord_to_idx(base_i * base_cell_size, num_cells_y, domain_size)
            iy_end = _coord_to_idx((base_i + 1) * base_cell_size, num_cells_y, domain_size)
            ix_start = _coord_to_idx(base_j * base_cell_size, num_cells_x, domain_size)
            ix_end = _coord_to_idx((base_j + 1) * base_cell_size, num_cells_x, domain_size)
            material_map[iy_start:iy_end, ix_start:ix_end] = mat_type
    return material_map



def build_structured_rect_problem(refinement: int = MESH_REFINEMENT) -> RectangularLatticeProblem:
    if refinement <= 0:
        raise ValueError(f"mesh refinement must be positive; got {refinement}.")

    num_cells_x = BASE_CELLS * refinement
    num_cells_y = BASE_CELLS * refinement
    material_map = create_rect_lattice_material_map(refinement=refinement)
    cell_widths = np.full(num_cells_x, TOTAL_WIDTH / num_cells_x, dtype=np.float64)
    cell_heights = np.full(num_cells_y, TOTAL_HEIGHT / num_cells_y, dtype=np.float64)
    cell_boundaries_x, cell_boundaries_y = calculate_boundaries_2d(cell_widths, cell_heights)
    cell_areas = np.outer(cell_heights, cell_widths)

    sigma_a_map = np.zeros_like(material_map, dtype=np.float64)
    sigma_s_map = np.zeros_like(material_map, dtype=np.float64)
    for region_id, props in MATERIAL_PROPERTIES.items():
        mask = material_map == int(region_id)
        sigma_a_map[mask] = float(props["sigma_a"])
        sigma_s_map[mask] = float(props["sigma_s"])

    source_cells = np.argwhere(material_map == REGION_SOURCE).astype(np.int64)
    if source_cells.size == 0:
        raise RuntimeError("The structured lattice does not contain any source cells.")

    return RectangularLatticeProblem(
        material_map=material_map,
        source_cells=source_cells,
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=cell_boundaries_x,
        cell_boundaries_y=cell_boundaries_y,
        cell_areas=cell_areas,
        sigma_a_map=sigma_a_map,
        sigma_s_map=sigma_s_map,
        sigma_a_flat=sigma_a_map.ravel(order="C"),
        sigma_s_flat=sigma_s_map.ravel(order="C"),
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        mesh_refinement=refinement,
        source_mode="internal_start",
    )


# -----------------------------------------------------------------------------
# Source / sampling helpers
# -----------------------------------------------------------------------------
def sample_source_times(num_particles: int) -> np.ndarray:
    if SOURCE_TIME_PROFILE == "pulse":
        return np.full(num_particles, PULSE_TIME, dtype=np.float64)
    if SOURCE_TIME_PROFILE == "gaussian":
        return np.clip(np.random.normal(GAUSSIAN_MEAN, GAUSSIAN_STD, num_particles), 0.0, MAX_TIME)
    if SOURCE_TIME_PROFILE == "continuous":
        return np.random.uniform(0.0, MAX_TIME, num_particles)
    if SOURCE_TIME_PROFILE == "step":
        return np.random.uniform(STEP_ON_TIME, STEP_OFF_TIME if STEP_OFF_TIME is not None else MAX_TIME, num_particles)
    return np.zeros(num_particles, dtype=np.float64)



def sample_isotropic_uv(num: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = torch.rand(num, device=device) * 2.0 - 1.0
    phi = torch.rand(num, device=device) * (2.0 * math.pi)
    t = torch.sqrt(torch.clamp(1.0 - mu * mu, min=0.0))
    u = t * torch.cos(phi)
    v = t * torch.sin(phi)
    tiny = (u.abs() + v.abs()) < 1.0e-12
    if tiny.any():
        u = u.clone()
        u[tiny] = 1.0e-6
    return u, v


# -----------------------------------------------------------------------------
# Rectangle transform helpers (torch, batched)
# -----------------------------------------------------------------------------
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



def entry_position_from_face(
    face: torch.Tensor,
    pos_on_face: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros_like(pos_on_face)
    x = torch.where(
        face == FACE_LEFT,
        zeros,
        torch.where(
            face == FACE_RIGHT,
            W,
            torch.where(face == FACE_TOP, pos_on_face, pos_on_face),
        ),
    )
    y = torch.where(
        face == FACE_LEFT,
        pos_on_face,
        torch.where(
            face == FACE_RIGHT,
            pos_on_face,
            torch.where(face == FACE_TOP, H, zeros),
        ),
    )
    return x, y



def transform_face_to_model(
    face: torch.Tensor,
    pos_on_face: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    W_safe = torch.clamp(W, min=1.0e-12)
    H_safe = torch.clamp(H, min=1.0e-12)
    Wm = torch.where((face == FACE_LEFT) | (face == FACE_RIGHT), W, H)
    Hm = torch.where((face == FACE_LEFT) | (face == FACE_RIGHT), H, W)
    y_in = torch.where(
        face == FACE_LEFT,
        pos_on_face / H_safe,
        torch.where(
            face == FACE_RIGHT,
            1.0 - pos_on_face / H_safe,
            torch.where(face == FACE_TOP, pos_on_face / W_safe, 1.0 - pos_on_face / W_safe),
        ),
    ).clamp(0.0, 1.0)
    u_m = torch.where(
        face == FACE_LEFT,
        u,
        torch.where(face == FACE_RIGHT, -u, torch.where(face == FACE_TOP, -v, v)),
    )
    v_m = torch.where(
        face == FACE_LEFT,
        v,
        torch.where(face == FACE_RIGHT, -v, torch.where(face == FACE_TOP, u, -u)),
    )
    return Wm, Hm, y_in, u_m, v_m



def transform_model_to_physical(
    face: torch.Tensor,
    x_rel: torch.Tensor,
    y_rel: torch.Tensor,
    u_m: torch.Tensor,
    v_m: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_m = x_rel * torch.where((face == FACE_LEFT) | (face == FACE_RIGHT), W, H)
    y_m = y_rel * torch.where((face == FACE_LEFT) | (face == FACE_RIGHT), H, W)

    x = torch.where(
        face == FACE_LEFT,
        x_m,
        torch.where(face == FACE_RIGHT, W - x_m, torch.where(face == FACE_TOP, y_m, W - y_m)),
    )
    y = torch.where(
        face == FACE_LEFT,
        y_m,
        torch.where(face == FACE_RIGHT, H - y_m, torch.where(face == FACE_TOP, H - x_m, x_m)),
    )

    u = torch.where(
        face == FACE_LEFT,
        u_m,
        torch.where(face == FACE_RIGHT, -u_m, torch.where(face == FACE_TOP, v_m, -v_m)),
    )
    v = torch.where(
        face == FACE_LEFT,
        v_m,
        torch.where(face == FACE_RIGHT, -v_m, torch.where(face == FACE_TOP, -u_m, u_m)),
    )
    return x, y, u, v



def nudge_inside_rect(
    x: torch.Tensor,
    y: torch.Tensor,
    face: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    frac: float = 5.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = torch.clamp(frac * W, min=1.0e-12)
    dy = torch.clamp(frac * H, min=1.0e-12)
    x_new = torch.where(face == FACE_LEFT, dx, torch.where(face == FACE_RIGHT, W - dx, x))
    y_new = torch.where(face == FACE_BOTTOM, dy, torch.where(face == FACE_TOP, H - dy, y))
    x_new = torch.minimum(torch.maximum(x_new, torch.zeros_like(W)), W)
    y_new = torch.minimum(torch.maximum(y_new, torch.zeros_like(H)), H)
    return x_new, y_new



def face_outward_normals(face: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    normals = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    return normals[face]


# -----------------------------------------------------------------------------
# Tally helpers
# -----------------------------------------------------------------------------
def scatter_add_cell_time(
    cell_idx: torch.Tensor,
    time_bin: torch.Tensor,
    values: torch.Tensor,
    target: torch.Tensor,
) -> None:
    num_time_bins, num_cells = target.shape
    valid = (
        (cell_idx >= 0)
        & (cell_idx < num_cells)
        & (time_bin >= 0)
        & (time_bin < num_time_bins)
    )
    if not valid.any():
        return
    flat_idx = time_bin[valid].to(torch.long) * num_cells + cell_idx[valid].to(torch.long)
    target.view(-1).index_add_(0, flat_idx, values[valid].to(target.dtype))



def scatter_add_time(
    time_bin: torch.Tensor,
    values: torch.Tensor,
    target: torch.Tensor,
) -> None:
    valid = (time_bin >= 0) & (time_bin < target.numel())
    if not valid.any():
        return
    target.index_add_(0, time_bin[valid].to(torch.long), values[valid].to(target.dtype))



def tally_flux(
    cell_idx: torch.Tensor,
    time_bin: torch.Tensor,
    weight_start: torch.Tensor,
    path_length: torch.Tensor,
    sigma_a: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optical_depth = sigma_a * path_length
    contrib = torch.where(
        sigma_a > 1.0e-12,
        weight_start * (-torch.expm1(-optical_depth)) / sigma_a,
        weight_start * path_length,
    )
    scatter_add_cell_time(cell_idx, time_bin, contrib, target)



def tally_energy(
    cell_idx: torch.Tensor,
    time_bin: torch.Tensor,
    energy: torch.Tensor,
    target: torch.Tensor,
) -> None:
    scatter_add_cell_time(cell_idx, time_bin, energy, target)


# -----------------------------------------------------------------------------
# Conditioning and model output decoding
# -----------------------------------------------------------------------------
def build_rect_cond_from_uv(
    *,
    width_feat: torch.Tensor,
    height_feat: torch.Tensor,
    budget_feat: torch.Tensor,
    direction_u: torch.Tensor,
    direction_v: torch.Tensor,
    cond_dim: int,
    direction_encoding_2d: str,
    y_in_feat: Optional[torch.Tensor] = None,
    x_in_feat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    normalize_direction_encoding_2d_shared(direction_encoding_2d)
    phi_in, r_in = uv_to_polar(direction_u, direction_v)

    if (x_in_feat is None) and (y_in_feat is not None):
        base = [width_feat, height_feat, budget_feat, y_in_feat]
    elif (x_in_feat is not None) and (y_in_feat is not None):
        base = [width_feat, height_feat, budget_feat, x_in_feat, y_in_feat]
    else:
        raise RuntimeError("Invalid 2D timedep conditioning features: expected boundary(y) or internal(x,y).")

    cond_polar = torch.stack(base + [normalize_angle_phi(phi_in), normalize_radius_r(r_in)], dim=1)
    if cond_polar.shape[1] == cond_dim:
        return cond_polar
    raise RuntimeError(
        f"Unsupported 2D timedep cond_dim={cond_dim} for polar direction encoding "
        f"(polar_dim={cond_polar.shape[1]})."
    )



def draw_rect_solver_outputs(
    solver: object,
    cond: torch.Tensor,
    cond_dim: int,
    target_dim: int,
    t_grid: torch.Tensor,
    step_size: torch.Tensor,
    budget_opt_actual: torch.Tensor,
    norm_params: dict,
    direction_encoding_2d: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn(cond.shape[0], target_dim, device=cond.device)
    with amp_context(device):
        out = solver.sample(
            time_grid=t_grid,
            x_init=torch.cat([cond, noise], dim=1),
            method="rk4",
            step_size=step_size,
            return_intermediates=False,
        )
    pred = out[:, cond_dim:]
    x_out_norm = denorm_position_relative(pred[:, 0])
    y_out_norm = denorm_position_relative(pred[:, 1])
    normalize_direction_encoding_2d_shared(direction_encoding_2d)
    if pred.shape[1] != 6:
        raise RuntimeError(
            f"Timedep inference now expects 6 targets [x, y, phi, r, L, time_exit]; got {pred.shape[1]}. "
            "This checkpoint is from an older 2D encoding and should be regenerated."
        )
    u_out, v_out = decode_direction_2d([pred[:, 2], pred[:, 3]], "polar", norm_params=norm_params)
    L_out = denorm_length_target(pred[:, 4], budget_opt_actual, norm_params).clamp(min=1.0e-9)
    time_exit_prob = denorm_linear_scale_logit(pred[:, 5]).clamp(0.0, 1.0)

    return x_out_norm, y_out_norm, u_out, v_out, L_out, time_exit_prob


# -----------------------------------------------------------------------------
# Training-regime diagnostic
# -----------------------------------------------------------------------------
def check_rect_training_regime(problem: RectangularLatticeProblem, norm_params: dict) -> bool:
    required = ["width_lower", "width_upper", "height_lower", "height_upper", "budget_lower", "budget_upper"]
    missing = [k for k in required if k not in norm_params]
    if missing:
        log.warning("Skipping rectangular regime check; norm file is missing keys: %s", ", ".join(missing))
        return False

    width_opt = np.repeat(problem.cell_widths[None, :], problem.num_cells_y, axis=0) * problem.sigma_s_map
    height_opt = np.repeat(problem.cell_heights[:, None], problem.num_cells_x, axis=1) * problem.sigma_s_map
    budget_opt = np.full_like(problem.sigma_s_map, DT * PARTICLE_VELOCITY, dtype=np.float64) * problem.sigma_s_map

    def log_rng(arr: np.ndarray) -> Tuple[float, float]:
        arr = np.maximum(arr, 1.0e-30)
        return float(np.log10(arr.min())), float(np.log10(arr.max()))

    w_rng = log_rng(width_opt)
    h_rng = log_rng(height_opt)
    b_rng = log_rng(budget_opt)

    ok = (
        w_rng[0] >= float(norm_params["width_lower"]) - 1.0e-5
        and w_rng[1] <= float(norm_params["width_upper"]) + 1.0e-5
        and h_rng[0] >= float(norm_params["height_lower"]) - 1.0e-5
        and h_rng[1] <= float(norm_params["height_upper"]) + 1.0e-5
        and b_rng[0] >= float(norm_params["budget_lower"]) - 1.0e-5
        and b_rng[1] <= float(norm_params["budget_upper"]) + 1.0e-5
    )

    print("\n[Rectangular CFM Training Regime Diagnostics]")
    print(f"  width*mfp range  (log10): [{w_rng[0]:.6f}, {w_rng[1]:.6f}] vs [{norm_params['width_lower']:.6f}, {norm_params['width_upper']:.6f}]")
    print(f"  height*mfp range (log10): [{h_rng[0]:.6f}, {h_rng[1]:.6f}] vs [{norm_params['height_lower']:.6f}, {norm_params['height_upper']:.6f}]")
    print(f"  budget range     (log10): [{b_rng[0]:.6f}, {b_rng[1]:.6f}] vs [{norm_params['budget_lower']:.6f}, {norm_params['budget_upper']:.6f}]")
    print(f"  Overall regime status: {'OK' if ok else 'OUT-OF-BOUNDS'}")
    return ok


# -----------------------------------------------------------------------------
# Main pure-CFM rectangle solver
# -----------------------------------------------------------------------------
@torch.inference_mode()
def run_rect_single_group_cfm(
    problem: RectangularLatticeProblem,
    boundary_model: ModelBundle,
    internal_model: ModelBundle,
    num_particles: int,
    device: str,
    randomize_start_pos: bool = True,
) -> dict:
    if not HAS_ML:
        raise RuntimeError("Cannot run the rectangular CFM solver because the ML stack is unavailable.")

    t_grid, step_size = build_time_grid(TIME_GRID_STEPS, device=device)
    bnd_norm = boundary_model.norm_params
    int_norm = internal_model.norm_params
    dir_encoding = normalize_direction_encoding_2d_shared(int_norm.get("direction_encoding_2d", DIRECTION_ENCODING_2D))

    check_rect_training_regime(problem, bnd_norm)
    check_rect_training_regime(problem, int_norm)

    compute_dtype = torch.float32 if device == "mps" else torch.float64
    tally_dtype = torch.float32 if device == "mps" else torch.float64

    cell_widths_t = torch.from_numpy(problem.cell_widths).to(device=device, dtype=torch.float32)
    cell_heights_t = torch.from_numpy(problem.cell_heights).to(device=device, dtype=torch.float32)
    sigma_a_t = torch.from_numpy(problem.sigma_a_flat).to(device=device, dtype=torch.float32)
    sigma_s_t = torch.from_numpy(problem.sigma_s_flat).to(device=device, dtype=torch.float32)
    face_to_face_t = torch.from_numpy(FACE_TO_FACE).to(device=device, dtype=torch.long)

    num_cells = problem.num_cells_x * problem.num_cells_y

    source_cells_t = torch.from_numpy(problem.source_cells).to(device=device, dtype=torch.long)
    rand_sel = torch.randint(0, source_cells_t.shape[0], (num_particles,), device=device)
    cell_y = source_cells_t[rand_sel, 0].clone()
    cell_x = source_cells_t[rand_sel, 1].clone()

    W_emit = cell_widths_t[cell_x]
    H_emit = cell_heights_t[cell_y]
    if randomize_start_pos:
        x = torch.rand(num_particles, device=device) * W_emit
        y = torch.rand(num_particles, device=device) * H_emit
    else:
        x = torch.full((num_particles,), float(START_POS_X_REL), device=device) * W_emit
        y = torch.full((num_particles,), float(START_POS_Y_REL), device=device) * H_emit
    u, v = sample_isotropic_uv(num_particles, device)

    weight = torch.ones(num_particles, dtype=torch.float32, device=device)
    current_time = torch.from_numpy(sample_source_times(num_particles)).to(device=device, dtype=compute_dtype)
    current_tbin = (current_time / DT).to(torch.int32)
    status = torch.zeros(num_particles, dtype=torch.int8, device=device)  # 0 active, 1 leaked, 2 terminated-in-domain
    terminal_code = torch.zeros(num_particles, dtype=torch.int8, device=device)  # 1-4 side leak, 7 absorb, 8 max-time
    is_on_boundary = torch.zeros(num_particles, dtype=torch.bool, device=device)
    entry_face = torch.zeros(num_particles, dtype=torch.long, device=device)
    entry_pos = torch.zeros(num_particles, dtype=torch.float32, device=device)

    track_length_t = torch.zeros((NUM_TIME_BINS, num_cells), dtype=tally_dtype, device=device)
    energy_dep_t = torch.zeros((NUM_TIME_BINS, num_cells), dtype=tally_dtype, device=device)
    leak_left_t = torch.zeros(NUM_TIME_BINS, dtype=tally_dtype, device=device)
    leak_right_t = torch.zeros(NUM_TIME_BINS, dtype=tally_dtype, device=device)
    leak_top_t = torch.zeros(NUM_TIME_BINS, dtype=tally_dtype, device=device)
    leak_bottom_t = torch.zeros(NUM_TIME_BINS, dtype=tally_dtype, device=device)

    active_count = int((status == 0).sum().item())
    pbar = tqdm(total=num_particles, desc="Rect single-group CFM")
    completed = 0
    last_heartbeat = time.time()
    wall_start = last_heartbeat
    iterations = 0

    while active_count > 0 and iterations < MAX_ITERATIONS:
        iterations += 1
        pending = (status == 0).clone()
        for branch in ("boundary", "internal"):
            if branch == "boundary":
                branch_mask = pending & is_on_boundary
                model = boundary_model
                norm = bnd_norm
            else:
                branch_mask = pending & (~is_on_boundary)
                model = internal_model
                norm = int_norm

            if not branch_mask.any():
                continue

            all_idx = branch_mask.nonzero(as_tuple=True)[0]
            for start in range(0, all_idx.numel(), CFM_MAX_PARTICLES_PER_PASS):
                idx = all_idx[start : start + CFM_MAX_PARTICLES_PER_PASS]
                if idx.numel() == 0:
                    continue
                pending[idx] = False

                ix = cell_x[idx]
                iy = cell_y[idx]
                W = cell_widths_t[ix]
                H = cell_heights_t[iy]
                cell_flat = iy * problem.num_cells_x + ix
                sigma_a = sigma_a_t[cell_flat]
                sigma_s = torch.clamp(sigma_s_t[cell_flat], min=EPSILON)

                time_bin = current_tbin[idx].to(torch.int64).clamp(0, NUM_TIME_BINS - 1)
                next_t = (time_bin + 1).to(compute_dtype) * DT
                remaining_bin = (next_t - current_time[idx]).clamp_min(0.0)
                remaining_global = (MAX_TIME - current_time[idx]).clamp_min(0.0)
                budget_phys = torch.minimum(remaining_bin, remaining_global) * PARTICLE_VELOCITY

                tiny = budget_phys < 1.0e-7
                if tiny.any():
                    t_idx = idx[tiny]
                    tiny_active = status[t_idx] == 0
                    tiny_low_weight = (weight[t_idx] < ENERGY_THRESHOLD) & tiny_active
                    if tiny_low_weight.any():
                        lw_idx = t_idx[tiny_low_weight]
                        status[lw_idx] = 2
                        terminal_code[lw_idx] = 7

                    tiny_alive = status[t_idx] == 0
                    if tiny_alive.any():
                        t_act = t_idx[tiny_alive]
                        tb = current_tbin[t_act].to(torch.int64).clamp(0, NUM_TIME_BINS - 1)
                        t_next = (tb + 1).to(compute_dtype) * DT
                        current_time[t_act] = torch.minimum(t_next, torch.tensor(MAX_TIME, dtype=compute_dtype, device=device))
                        current_tbin[t_act] += 1
                        expired = current_tbin[t_act] >= NUM_TIME_BINS
                        if expired.any():
                            e_idx = t_act[expired]
                            status[e_idx] = 2
                            terminal_code[e_idx] = 8
                    keep = ~tiny
                    idx = idx[keep]
                    if idx.numel() == 0:
                        continue
                    ix = ix[keep]
                    iy = iy[keep]
                    W = W[keep]
                    H = H[keep]
                    cell_flat = cell_flat[keep]
                    sigma_a = sigma_a[keep]
                    sigma_s = sigma_s[keep]
                    time_bin = time_bin[keep]
                    next_t = next_t[keep]
                    remaining_bin = remaining_bin[keep]
                    remaining_global = remaining_global[keep]
                    budget_phys = budget_phys[keep]

                budget_opt = torch.clamp(budget_phys.to(torch.float32) * sigma_s, min=1.0e-6)
                width_feat = normalize_width(torch.clamp(W * sigma_s, min=1.0e-12), norm["width_lower"], norm["width_upper"])
                height_feat = normalize_height(torch.clamp(H * sigma_s, min=1.0e-12), norm["height_lower"], norm["height_upper"])
                budget_feat = normalize_log_scale_logit(budget_opt, norm["budget_lower"], norm["budget_upper"])

                if branch == "boundary":
                    face_in = entry_face[idx]
                    pos_in = entry_pos[idx]
                    Wm, Hm, y_in, u_m_in, v_m_in = transform_face_to_model(face_in, pos_in, u[idx], v[idx], W, H)
                    bnd_width_feat = normalize_width(torch.clamp(Wm * sigma_s, min=1.0e-12), norm["width_lower"], norm["width_upper"])
                    bnd_height_feat = normalize_height(torch.clamp(Hm * sigma_s, min=1.0e-12), norm["height_lower"], norm["height_upper"])
                    cond = build_rect_cond_from_uv(
                        width_feat=bnd_width_feat,
                        height_feat=bnd_height_feat,
                        budget_feat=budget_feat,
                        direction_u=u_m_in,
                        direction_v=v_m_in,
                        cond_dim=model.cond_dim,
                        direction_encoding_2d=dir_encoding,
                        y_in_feat=normalize_position_relative(y_in),
                    )
                    x_rel_pred, y_rel_pred, u_m_out, v_m_out, L_out, time_exit_prob = draw_rect_solver_outputs(
                        model.solver,
                        cond,
                        model.cond_dim,
                        model.target_dim,
                        t_grid,
                        step_size,
                        budget_opt,
                        norm,
                        dir_encoding,
                        device,
                    )
                    x_m = x_rel_pred * Wm
                    y_m = y_rel_pred * Hm
                    space_exit_m = time_exit_prob < 0.5
                    if space_exit_m.any():
                        m_idx = space_exit_m.nonzero(as_tuple=True)[0]
                        _, x_m_snap, y_m_snap, _ = classify_face_and_snap(
                            x_m[m_idx],
                            y_m[m_idx],
                            Wm[m_idx],
                            Hm[m_idx],
                        )
                        x_m = x_m.clone()
                        y_m = y_m.clone()
                        x_m[m_idx] = x_m_snap
                        y_m[m_idx] = y_m_snap
                    x_raw, y_raw, u_out, v_out = transform_model_to_physical(
                        face_in,
                        x_m / torch.clamp(Wm, min=1.0e-12),
                        y_m / torch.clamp(Hm, min=1.0e-12),
                        u_m_out,
                        v_m_out,
                        W,
                        H,
                    )
                else:
                    x_in_rel = torch.clamp(x[idx] / torch.clamp(W, min=1.0e-12), 1.0e-4, 0.9999)
                    y_in_rel = torch.clamp(y[idx] / torch.clamp(H, min=1.0e-12), 1.0e-4, 0.9999)
                    cond = build_rect_cond_from_uv(
                        width_feat=width_feat,
                        height_feat=height_feat,
                        budget_feat=budget_feat,
                        direction_u=u[idx],
                        direction_v=v[idx],
                        cond_dim=model.cond_dim,
                        direction_encoding_2d=dir_encoding,
                        x_in_feat=normalize_position_relative(x_in_rel),
                        y_in_feat=normalize_position_relative(y_in_rel),
                    )
                    x_rel_pred, y_rel_pred, u_out, v_out, L_out, time_exit_prob = draw_rect_solver_outputs(
                        model.solver,
                        cond,
                        model.cond_dim,
                        model.target_dim,
                        t_grid,
                        step_size,
                        budget_opt,
                        norm,
                        dir_encoding,
                        device,
                    )
                    x_raw = x_rel_pred * W
                    y_raw = y_rel_pred * H

                x_prev = x[idx]
                y_prev = y[idx]
                budget_phys_f32 = budget_phys.to(torch.float32)
                dx_raw = x_raw - x_prev
                dy_raw = y_raw - y_prev
                dist_raw = torch.sqrt(dx_raw * dx_raw + dy_raw * dy_raw).clamp_min(1.0e-12)
                scale = torch.where(
                    dist_raw > budget_phys_f32,
                    budget_phys_f32 / dist_raw,
                    torch.ones_like(dist_raw),
                )
                x_budget_raw = x_prev + dx_raw * scale
                y_budget_raw = y_prev + dy_raw * scale

                x_raw_phys = x_budget_raw.clone()
                y_raw_phys = y_budget_raw.clone()
                x_work = torch.minimum(torch.maximum(x_budget_raw, torch.zeros_like(W)), W)
                y_work = torch.minimum(torch.maximum(y_budget_raw, torch.zeros_like(H)), H)
                time_exit = time_exit_prob >= 0.5
                space_exit = ~time_exit

                if space_exit.any():
                    s_idx = space_exit.nonzero(as_tuple=True)[0]
                    face_exit, x_snap, y_snap, pos_snap = classify_face_and_snap(
                        x_budget_raw[s_idx],
                        y_budget_raw[s_idx],
                        W[s_idx],
                        H[s_idx],
                    )
                    normals = face_outward_normals(face_exit, device=device, dtype=u_out.dtype)
                    outward = (u_out[s_idx] * normals[:, 0] + v_out[s_idx] * normals[:, 1]) > 0.0

                    if outward.any():
                        out_idx = s_idx[outward]
                        x_work[out_idx] = x_snap[outward]
                        y_work[out_idx] = y_snap[outward]
                        x_raw_phys[out_idx] = x_snap[outward]
                        y_raw_phys[out_idx] = y_snap[outward]

                    inward = ~outward
                    if inward.any():
                        bad_idx = s_idx[inward]
                        inward_W = W[bad_idx]
                        inward_H = H[bad_idx]
                        inward_x = x_work[bad_idx]
                        inward_y = y_work[bad_idx]
                        d_left = torch.abs(inward_x)
                        d_right = torch.abs(inward_W - inward_x)
                        d_top = torch.abs(inward_H - inward_y)
                        d_bottom = torch.abs(inward_y)
                        d_stack = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
                        inward_face = torch.argmin(d_stack, dim=1)
                        dmin = d_stack.min(dim=1).values
                        eps_bnd = torch.clamp(
                            8.0 * torch.finfo(inward_x.dtype).eps * torch.maximum(inward_W, inward_H),
                            min=1.0e-12,
                        )
                        near_bnd = dmin <= eps_bnd
                        x_inside = inward_x
                        y_inside = inward_y
                        if near_bnd.any():
                            x_inside = x_inside.clone()
                            y_inside = y_inside.clone()
                            dx_in = torch.clamp(8.0 * torch.finfo(inward_x.dtype).eps * inward_W, min=1.0e-12)
                            dy_in = torch.clamp(8.0 * torch.finfo(inward_y.dtype).eps * inward_H, min=1.0e-12)
                            x_inside[near_bnd] = torch.where(
                                inward_face[near_bnd] == FACE_LEFT,
                                dx_in[near_bnd],
                                torch.where(
                                    inward_face[near_bnd] == FACE_RIGHT,
                                    inward_W[near_bnd] - dx_in[near_bnd],
                                    inward_x[near_bnd],
                                ),
                            )
                            y_inside[near_bnd] = torch.where(
                                inward_face[near_bnd] == FACE_BOTTOM,
                                dy_in[near_bnd],
                                torch.where(
                                    inward_face[near_bnd] == FACE_TOP,
                                    inward_H[near_bnd] - dy_in[near_bnd],
                                    inward_y[near_bnd],
                                ),
                            )
                        x_work[bad_idx] = x_inside
                        y_work[bad_idx] = y_inside
                        x_raw_phys[bad_idx] = x_inside
                        y_raw_phys[bad_idx] = y_inside
                        space_exit = space_exit.clone()
                        space_exit[bad_idx] = False

                if time_exit.any():
                    t_local = time_exit.nonzero(as_tuple=True)[0]
                    x_t_all = x_work[t_local]
                    y_t_all = y_work[t_local]
                    W_t_all = W[t_local]
                    H_t_all = H[t_local]
                    d_left_all = torch.abs(x_t_all)
                    d_right_all = torch.abs(W_t_all - x_t_all)
                    d_top_all = torch.abs(H_t_all - y_t_all)
                    d_bottom_all = torch.abs(y_t_all)
                    dmin_all = torch.stack([d_left_all, d_right_all, d_top_all, d_bottom_all], dim=1).min(dim=1).values
                    eps_bnd = torch.clamp(
                        8.0 * torch.finfo(x_t_all.dtype).eps * torch.maximum(W_t_all, H_t_all),
                        min=1.0e-12,
                    )
                    near_bnd = dmin_all <= eps_bnd
                    if near_bnd.any():
                        tt = t_local[near_bnd]
                        x_t = x_work[tt]
                        y_t = y_work[tt]
                        W_t = W[tt]
                        H_t = H[tt]
                        d_left = torch.abs(x_t)
                        d_right = torch.abs(W_t - x_t)
                        d_top = torch.abs(H_t - y_t)
                        d_bottom = torch.abs(y_t)
                        d_stack = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
                        face_min = torch.argmin(d_stack, dim=1)
                        dx_in = torch.clamp(8.0 * torch.finfo(W_t.dtype).eps * W_t, min=1.0e-12)
                        dy_in = torch.clamp(8.0 * torch.finfo(H_t.dtype).eps * H_t, min=1.0e-12)
                        x_new = torch.where(face_min == FACE_LEFT, dx_in, torch.where(face_min == FACE_RIGHT, W_t - dx_in, x_t))
                        y_new = torch.where(face_min == FACE_BOTTOM, dy_in, torch.where(face_min == FACE_TOP, H_t - dy_in, y_t))
                        x_work = x_work.clone()
                        y_work = y_work.clone()
                        x_raw_phys = x_raw_phys.clone()
                        y_raw_phys = y_raw_phys.clone()
                        x_work[tt] = x_new
                        y_work[tt] = y_new
                        x_raw_phys[tt] = x_new
                        y_raw_phys[tt] = y_new

                dx = x_work - x_prev
                dy = y_work - y_prev
                dist_geom = torch.sqrt(dx * dx + dy * dy).clamp_min(1.0e-12)

                L_phys_pred = (L_out / sigma_s).to(torch.float32)
                L_phys = torch.minimum(L_phys_pred, budget_phys_f32)
                L_phys = torch.where(time_exit, budget_phys_f32, L_phys)
                L_phys = torch.maximum(L_phys, dist_geom)

                weight_start = weight[idx].clone()
                tally_flux(cell_flat, time_bin, weight_start, L_phys, sigma_a, track_length_t)
                loss_frac = -torch.expm1(-sigma_a * L_phys)
                weight[idx] = weight[idx] * torch.clamp_min(1.0 - loss_frac, 0.0)
                tally_energy(cell_flat, time_bin, weight_start - weight[idx], energy_dep_t)

                current_time[idx] = current_time[idx] + L_phys.to(compute_dtype) / PARTICLE_VELOCITY
                hit_full_bin = time_exit & (remaining_bin <= (remaining_global + 1.0e-12))
                current_time[idx] = torch.where(hit_full_bin, next_t, current_time[idx])
                current_tbin[idx] += hit_full_bin.to(torch.int32)

                x[idx] = x_work
                y[idx] = y_work
                u[idx] = u_out
                v[idx] = v_out
                is_on_boundary[idx] = False

                low_weight = (weight[idx] < ENERGY_THRESHOLD) & (status[idx] == 0)
                if low_weight.any():
                    lw = idx[low_weight]
                    status[lw] = 2
                    terminal_code[lw] = 7

                expired = ((current_tbin[idx] >= NUM_TIME_BINS) | (current_time[idx] >= (MAX_TIME - 1.0e-12))) & (status[idx] == 0)
                if expired.any():
                    ee_idx = idx[expired]
                    status[ee_idx] = 2
                    terminal_code[ee_idx] = 8

                active_local = status[idx] == 0
                space_exit_active = space_exit & active_local
                if space_exit_active.any():
                    s_local = space_exit_active.nonzero(as_tuple=True)[0]
                    p_idx = idx[s_local]
                    ix_exit = ix[s_local]
                    iy_exit = iy[s_local]
                    face_exit, _, _, pos_exit = classify_face_and_snap(
                        x_raw_phys[s_local],
                        y_raw_phys[s_local],
                        W[s_local],
                        H[s_local],
                    )
                    tbin_exit = current_tbin[p_idx].to(torch.int64).clamp(0, NUM_TIME_BINS - 1)

                    new_ix = ix_exit + (face_exit == FACE_RIGHT).to(torch.long) - (face_exit == FACE_LEFT).to(torch.long)
                    new_iy = iy_exit + (face_exit == FACE_TOP).to(torch.long) - (face_exit == FACE_BOTTOM).to(torch.long)
                    leaked = (new_ix < 0) | (new_ix >= problem.num_cells_x) | (new_iy < 0) | (new_iy >= problem.num_cells_y)

                    if leaked.any():
                        leak_idx = p_idx[leaked]
                        leak_face = face_exit[leaked]
                        leak_w = weight[leak_idx]
                        leak_tbin = tbin_exit[leaked]
                        scatter_add_time(leak_tbin[leak_face == FACE_LEFT], leak_w[leak_face == FACE_LEFT], leak_left_t)
                        scatter_add_time(leak_tbin[leak_face == FACE_RIGHT], leak_w[leak_face == FACE_RIGHT], leak_right_t)
                        scatter_add_time(leak_tbin[leak_face == FACE_TOP], leak_w[leak_face == FACE_TOP], leak_top_t)
                        scatter_add_time(leak_tbin[leak_face == FACE_BOTTOM], leak_w[leak_face == FACE_BOTTOM], leak_bottom_t)
                        status[leak_idx] = 1
                        terminal_code[leak_idx] = (leak_face + 1).to(torch.int8)

                    survived = ~leaked
                    if survived.any():
                        surv_idx = p_idx[survived]
                        cell_x[surv_idx] = new_ix[survived]
                        cell_y[surv_idx] = new_iy[survived]
                        entry_face[surv_idx] = face_to_face_t[face_exit[survived]]
                        entry_pos[surv_idx] = pos_exit[survived]
                        W_new = cell_widths_t[cell_x[surv_idx]]
                        H_new = cell_heights_t[cell_y[surv_idx]]
                        x_entry, y_entry = entry_position_from_face(entry_face[surv_idx], pos_exit[survived], W_new, H_new)
                        x[surv_idx] = x_entry
                        y[surv_idx] = y_entry
                        is_on_boundary[surv_idx] = True

        active_count = int((status == 0).sum().item())
        newly_completed = num_particles - active_count - completed
        if newly_completed > 0:
            pbar.update(newly_completed)
            completed += newly_completed

        now = time.time()
        if (now - last_heartbeat) >= PROGRESS_HEARTBEAT_SEC:
            on_bnd = int(((status == 0) & is_on_boundary).sum().item())
            in_cell = int(((status == 0) & (~is_on_boundary)).sum().item())
            pbar.set_postfix_str(f"iter={iterations} active={active_count} boundary={on_bnd} internal={in_cell}")
            pbar.refresh()
            last_heartbeat = now

    if (status == 0).any():
        stale = status == 0
        status[stale] = 2
        terminal_code[stale] = 8

    pbar.close()
    elapsed = time.time() - wall_start

    particle_norm = max(float(num_particles), 1.0)
    tracklen_vs_time_flat = track_length_t.detach().cpu().numpy().astype(np.float64) / particle_norm
    energy_vs_time_flat = energy_dep_t.detach().cpu().numpy().astype(np.float64) / particle_norm
    tracklen_vs_time = tracklen_vs_time_flat.reshape(NUM_TIME_BINS, problem.num_cells_y, problem.num_cells_x)
    energy_vs_time = energy_vs_time_flat.reshape(NUM_TIME_BINS, problem.num_cells_y, problem.num_cells_x)
    flux_vs_time = np.where(problem.cell_areas[None, :, :] > 0.0, tracklen_vs_time / (problem.cell_areas[None, :, :] * DT), 0.0)
    flux_total = np.where(problem.cell_areas > 0.0, tracklen_vs_time.sum(axis=0) / problem.cell_areas, 0.0)
    energy_total = energy_vs_time.sum(axis=0)

    totals = {
        "left": int((terminal_code == 1).sum().item()),
        "right": int((terminal_code == 2).sum().item()),
        "top": int((terminal_code == 3).sum().item()),
        "bottom": int((terminal_code == 4).sum().item()),
        "absorb": int((terminal_code == 7).sum().item()),
        "time_like": int((terminal_code == 8).sum().item()),
    }

    time_edges = np.linspace(0.0, MAX_TIME, NUM_TIME_BINS + 1)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

    return {
        "energy_vs_time": energy_vs_time,
        "tracklen_vs_time": tracklen_vs_time,
        "flux_vs_time": flux_vs_time,
        "energy_total": energy_total,
        "flux_total": flux_total,
        "flux_left_vs_time": leak_left_t.detach().cpu().numpy().astype(np.float64) / particle_norm,
        "flux_right_vs_time": leak_right_t.detach().cpu().numpy().astype(np.float64) / particle_norm,
        "flux_top_vs_time": leak_top_t.detach().cpu().numpy().astype(np.float64) / particle_norm,
        "flux_bottom_vs_time": leak_bottom_t.detach().cpu().numpy().astype(np.float64) / particle_norm,
        "time_edges": time_edges,
        "time_centers": time_centers,
        "elapsed": elapsed,
        "totals": totals,
        "num_tracked": int(sum(totals.values())),
        "iterations": iterations,
        "status": status.detach().cpu().numpy(),
        "terminal_code": terminal_code.detach().cpu().numpy(),
    }


# -----------------------------------------------------------------------------
# MC reference
# -----------------------------------------------------------------------------
def maybe_warn_mc_alignment(problem: RectangularLatticeProblem, mc_problem) -> None:
    try:
        mismatches: list[str] = []
        if hasattr(mc_problem, "material_map") and not np.array_equal(np.asarray(mc_problem.material_map), problem.material_map):
            mismatches.append("material_map")
        if hasattr(mc_problem, "source_cells") and not np.array_equal(np.asarray(mc_problem.source_cells), problem.source_cells):
            mismatches.append("source_cells")
        if hasattr(mc_problem, "num_cells_x") and int(mc_problem.num_cells_x) != problem.num_cells_x:
            mismatches.append("num_cells_x")
        if hasattr(mc_problem, "num_cells_y") and int(mc_problem.num_cells_y) != problem.num_cells_y:
            mismatches.append("num_cells_y")
        if hasattr(mc_problem, "cell_widths") and not np.allclose(np.asarray(mc_problem.cell_widths), problem.cell_widths):
            mismatches.append("cell_widths")
        if hasattr(mc_problem, "cell_heights") and not np.allclose(np.asarray(mc_problem.cell_heights), problem.cell_heights):
            mismatches.append("cell_heights")
        if mismatches:
            log.warning("Single-group MC reference problem differs from this driver in: %s", ", ".join(mismatches))
    except Exception as exc:  # pragma: no cover - alignment is optional
        log.warning("Could not validate MC reference alignment (%s)", exc)



def run_rect_mc_reference(
    problem: RectangularLatticeProblem,
    num_particles: int,
    seed: int,
) -> Optional[dict]:
    try:
        from physics import mc2dtimemg as module
    except Exception as exc:
        log.warning("MC reference unavailable: could not import physics.mc2dtimemg (%s)", exc)
        return None

    try:
        mc_problem = module.build_default_lattice_problem(
            group_mode="single",
            mesh_refinement=int(problem.mesh_refinement),
            max_time=float(MAX_TIME),
            num_time_bins=int(NUM_TIME_BINS),
            synthetic_materials=False,
        )
        maybe_warn_mc_alignment(problem, mc_problem)

        if hasattr(module, "run_original_mc"):
            result = module.run_original_mc(
                mc_problem,
                n_particles=int(num_particles),
                seed=int(seed),
                max_steps=int(MC_REFERENCE_MAX_STEPS),
            )
        elif hasattr(module, "run_iterative_mc"):
            result = module.run_iterative_mc(
                mc_problem,
                n_particles=int(num_particles),
                seed=int(seed),
                max_steps=int(MC_REFERENCE_MAX_STEPS),
            )
        else:
            raise RuntimeError("physics.mc2dtimemg exposes neither run_original_mc nor run_iterative_mc")

        out = dict(result)
        if "num_tracked" not in out:
            totals = out.get("totals", {})
            out["num_tracked"] = int(
                sum(int(totals.get(k, 0)) for k in ("left", "right", "top", "bottom", "absorb", "time_like"))
            )
        return out
    except Exception as exc:
        log.warning("MC reference failed (%s)", exc)
        return None


def get_empty_mc_results(problem: RectangularLatticeProblem) -> dict:
    shape = (NUM_TIME_BINS, problem.num_cells_y, problem.num_cells_x)
    zeros_map = np.zeros((problem.num_cells_y, problem.num_cells_x), dtype=np.float64)
    zeros_time = np.zeros(NUM_TIME_BINS, dtype=np.float64)
    time_edges = np.linspace(0.0, MAX_TIME, NUM_TIME_BINS + 1, dtype=np.float64)
    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    return {
        "energy_vs_time": np.zeros(shape, dtype=np.float64),
        "tracklen_vs_time": np.zeros(shape, dtype=np.float64),
        "flux_vs_time": np.zeros(shape, dtype=np.float64),
        "energy_total": zeros_map.copy(),
        "flux_total": zeros_map.copy(),
        "flux_left_vs_time": zeros_time.copy(),
        "flux_right_vs_time": zeros_time.copy(),
        "flux_top_vs_time": zeros_time.copy(),
        "flux_bottom_vs_time": zeros_time.copy(),
        "time_edges": time_edges,
        "time_centers": time_centers,
        "elapsed": 0.0,
        "totals": {"left": 0, "right": 0, "top": 0, "bottom": 0, "absorb": 0, "time_like": 0},
        "num_tracked": 0,
    }


def run_rect_mc_replicate_mean(
    problem: RectangularLatticeProblem,
    num_particles: int,
    num_replicates: int,
    seed: int,
) -> Tuple[dict, dict]:
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be positive; got {num_replicates}.")

    base = int(num_particles) // int(num_replicates)
    rem = int(num_particles) % int(num_replicates)
    run_counts = [base + (1 if i < rem else 0) for i in range(int(num_replicates))]
    rng = np.random.default_rng(int(seed))

    runs: list[dict] = []
    elapsed_total = 0.0
    totals_acc = np.zeros(6, dtype=np.int64)

    for i, n_i in enumerate(run_counts):
        if n_i <= 0:
            continue
        seed_i = int(rng.integers(0, 2**31 - 1))
        res_i = run_rect_mc_reference(problem, n_i, seed_i)
        if res_i is None:
            empty = get_empty_mc_results(problem)
            return empty, empty

        runs.append(res_i)
        elapsed_total += float(res_i.get("elapsed", 0.0))
        totals_i = res_i.get("totals", {})
        totals_acc += np.array(
            [
                int(totals_i.get("left", 0)),
                int(totals_i.get("right", 0)),
                int(totals_i.get("top", 0)),
                int(totals_i.get("bottom", 0)),
                int(totals_i.get("absorb", 0)),
                int(totals_i.get("time_like", 0)),
            ],
            dtype=np.int64,
        )
        log.info(
            "MC replicate %d/%d complete via physics.mc2dtimemg (original, N=%d, %.2fs)",
            i + 1,
            int(num_replicates),
            int(n_i),
            float(res_i.get("elapsed", 0.0)),
        )

    if not runs:
        empty = get_empty_mc_results(problem)
        return empty, empty

    stack_keys = [
        "energy_vs_time",
        "tracklen_vs_time",
        "flux_vs_time",
        "energy_total",
        "flux_total",
        "flux_left_vs_time",
        "flux_right_vs_time",
        "flux_bottom_vs_time",
        "flux_top_vs_time",
    ]
    ddof = 1 if len(runs) > 1 else 0

    mean: dict = {}
    std: dict = {}
    for key in stack_keys:
        vals = np.stack([np.asarray(r[key], dtype=np.float64) for r in runs], axis=0)
        mean[key] = np.mean(vals, axis=0)
        std[key] = np.std(vals, axis=0, ddof=ddof)

    mean["time_edges"] = np.asarray(runs[0]["time_edges"], dtype=np.float64)
    mean["time_centers"] = np.asarray(runs[0]["time_centers"], dtype=np.float64)
    mean["elapsed"] = elapsed_total
    mean["totals"] = {
        "left": int(totals_acc[0]),
        "right": int(totals_acc[1]),
        "top": int(totals_acc[2]),
        "bottom": int(totals_acc[3]),
        "absorb": int(totals_acc[4]),
        "time_like": int(totals_acc[5]),
    }
    mean["num_tracked"] = int(np.sum(totals_acc))

    std["time_edges"] = mean["time_edges"]
    std["time_centers"] = mean["time_centers"]
    std["elapsed"] = 0.0
    std["totals"] = mean["totals"].copy()
    std["num_tracked"] = mean["num_tracked"]
    return mean, std


# -----------------------------------------------------------------------------
# Plotting / output
# -----------------------------------------------------------------------------
def plot_rect_map(
    problem: RectangularLatticeProblem,
    data: np.ndarray,
    title: str,
    filename: Path,
    log_scale: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "jet",
) -> None:
    values = np.asarray(data, dtype=np.float64)
    positive = values
    positive = positive[positive > 0.0]
    if log_scale and positive.size == 0:
        return
    X, Y = np.meshgrid(problem.cell_boundaries_x, problem.cell_boundaries_y)
    plt.figure(figsize=(8, 7))
    if log_scale:
        vmin_eff = max(float(positive.min()) if vmin is None else float(vmin), 1.0e-12)
        vmax_eff = max(float(np.max(values)) if vmax is None else float(vmax), vmin_eff * 1.01)
        mesh = plt.pcolormesh(X, Y, np.maximum(values, vmin_eff), cmap=cmap, norm=LogNorm(vmin=vmin_eff, vmax=vmax_eff), shading="auto")
    else:
        mesh = plt.pcolormesh(X, Y, values, cmap=cmap, shading="auto")
    plt.colorbar(mesh, label=title)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=240)
    plt.close()



def positive_floor(data: np.ndarray, default: float = 1.0e-12) -> float:
    positive = np.asarray(data, dtype=np.float64)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return default
    return float(positive.min())



def positive_floor_many(*arrays: np.ndarray, default: float = 1.0e-12) -> float:
    mins = [positive_floor(arr, default=np.inf) for arr in arrays]
    mins = [val for val in mins if np.isfinite(val)]
    if not mins:
        return default
    return float(min(mins))



def plot_rect_map_pair(
    problem: RectangularLatticeProblem,
    left: np.ndarray,
    right: np.ndarray,
    titles: Tuple[str, str],
    filename: Path,
    log_scale: bool = True,
) -> None:
    X, Y = np.meshgrid(problem.cell_boundaries_x, problem.cell_boundaries_y)
    plt.figure(figsize=(14, 6))
    axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]

    norm = None
    if log_scale:
        vmin = positive_floor_many(left, right, default=1.0e-12)
        vmax = max(float(np.max(left)), float(np.max(right)), vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, values, title in zip(axes, (left, right), titles):
        if norm is not None:
            mesh = ax.pcolormesh(X, Y, np.maximum(values, norm.vmin), cmap="jet", norm=norm, shading="auto")
        else:
            mesh = ax.pcolormesh(X, Y, values, cmap="jet", shading="auto")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(title)

    plt.colorbar(mesh, ax=axes)
    plt.tight_layout()
    plt.savefig(filename, dpi=240)
    plt.close()



def plot_error_map_auto(
    problem: RectangularLatticeProblem,
    ref: np.ndarray,
    test: np.ndarray,
    title: str,
    filename: Path,
    use_log: bool = True,
    cmap: str = "magma",
) -> None:
    err = np.abs(np.asarray(test, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    if not np.any(err > 0.0):
        return
    X, Y = np.meshgrid(problem.cell_boundaries_x, problem.cell_boundaries_y)
    norm = None
    if use_log:
        valid = err[err > 0.0]
        if valid.size == 0:
            return
        vmin = max(float(valid.min()), 1.0e-12)
        vmax = max(float(err.max()), vmin * 1.01)
        err = np.maximum(err, vmin)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    plt.figure(figsize=(8, 7))
    mesh = plt.pcolormesh(X, Y, err, cmap=cmap, norm=norm, shading="auto")
    plt.colorbar(mesh, label="Absolute Error")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=240)
    plt.close()


def source_row_index(problem: RectangularLatticeProblem) -> int:
    if problem.source_cells.size == 0:
        return int(problem.num_cells_y // 2)
    return int(np.clip(np.round(np.mean(problem.source_cells[:, 0])), 0, problem.num_cells_y - 1))


def plot_flux_lineout_comparison(
    problem: RectangularLatticeProblem,
    cfm_grid: np.ndarray,
    mc_grid: np.ndarray,
    mc_std_grid: np.ndarray,
    mc_ref_grid: np.ndarray,
    output_dir: Path,
) -> None:
    x_centers = 0.5 * (problem.cell_boundaries_x[:-1] + problem.cell_boundaries_x[1:])
    y_centers = 0.5 * (problem.cell_boundaries_y[:-1] + problem.cell_boundaries_y[1:])
    row_idx = source_row_index(problem)
    col_idx = int(np.argmin(np.abs(x_centers - 0.5 * (problem.cell_boundaries_x[0] + problem.cell_boundaries_x[-1]))))

    line_specs = [
        ("x", x_centers, cfm_grid[row_idx, :], mc_grid[row_idx, :], mc_std_grid[row_idx, :], mc_ref_grid[row_idx, :], "x (cm)", f"y-row={row_idx}", "source_row"),
        ("y", y_centers, cfm_grid[:, col_idx], mc_grid[:, col_idx], mc_std_grid[:, col_idx], mc_ref_grid[:, col_idx], "y (cm)", f"x-col={col_idx}", "vertical_center"),
    ]

    for _, centers, cfm_data, mc_data, mc_std_data, mc_ref_data, x_label, loc_text, suffix in line_specs:
        for use_log, scale_label in ((True, "log"), (False, "linear")):
            plt.figure(figsize=(10, 6))
            floor = 1.0e-12 if use_log else -np.inf
            plt.plot(centers, np.clip(mc_ref_data, floor, None), "k-", linewidth=2.3, label="MC ref")
            plt.plot(centers, np.clip(cfm_data, floor, None), "r-o", linewidth=1.7, markersize=3.5, label="CFM")
            plt.errorbar(
                centers,
                np.clip(mc_data, floor, None),
                yerr=np.clip(mc_std_data, 0.0, None),
                fmt="b--x",
                linewidth=1.3,
                markersize=3.5,
                capsize=2,
                elinewidth=1.0,
                errorevery=max(1, len(centers) // 20),
                label="MC mean ±1σ",
            )
            if use_log:
                plt.yscale("log")
            plt.xlabel(x_label)
            plt.ylabel("Scalar Flux")
            plt.title(f"Scalar Flux Lineout ({loc_text}) - {scale_label}")
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"scalar_flux_lineout_{suffix}_{scale_label}.png", dpi=240)
            plt.close()


def plot_total_flux_vs_time_comparison(
    cfm_results: dict,
    mc_results: dict,
    mc_std_results: dict,
    mc_ref_results: dict,
    filename: Path,
) -> None:
    t = np.asarray(cfm_results["time_centers"], dtype=np.float64)
    cfm_total_flux_t = np.asarray(cfm_results["flux_vs_time"], dtype=np.float64).sum(axis=(1, 2))
    mc_total_flux_t = np.asarray(mc_results["flux_vs_time"], dtype=np.float64).sum(axis=(1, 2))
    mc_ref_total_flux_t = np.asarray(mc_ref_results["flux_vs_time"], dtype=np.float64).sum(axis=(1, 2))
    mc_total_flux_std_t = np.sqrt(np.maximum(0.0, np.sum(np.asarray(mc_std_results["flux_vs_time"], dtype=np.float64) ** 2, axis=(1, 2))))

    plt.figure(figsize=(9, 5))
    plt.plot(t, mc_ref_total_flux_t, "k-", linewidth=2.2, label="MC ref")
    plt.plot(t, cfm_total_flux_t, "r--", linewidth=2.0, label="CFM")
    plt.errorbar(t, mc_total_flux_t, yerr=mc_total_flux_std_t, fmt="b-.", linewidth=1.4, capsize=2, label="MC mean ±1σ")
    plt.xlabel("Time")
    plt.ylabel("Total Flux")
    plt.title("Total Flux vs Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=240)
    plt.close()


def plot_comparison_suite(
    problem: RectangularLatticeProblem,
    cfm_results: dict,
    mc_results: dict,
    mc_std_results: dict,
    mc_ref_results: dict,
    output_dir: Path,
) -> None:
    flux_vmin, flux_vmax = 1.0e-6, 1.0e0
    energy_ref = np.asarray(mc_ref_results["energy_total"], dtype=np.float64)
    energy_positive = energy_ref[energy_ref > 0.0]
    energy_vmin = max(float(energy_positive.min()), 1.0e-12) if energy_positive.size > 0 else 1.0e-12
    energy_vmax = max(float(energy_ref.max()), energy_vmin * 1.01)

    plot_rect_map(problem, cfm_results["flux_total"], "CFM Time-Integrated Flux", output_dir / "cfm_flux_total.png", log_scale=True, vmin=flux_vmin, vmax=flux_vmax)
    plot_rect_map(problem, mc_results["flux_total"], "MC Mean Time-Integrated Flux", output_dir / "mc_flux_total.png", log_scale=True, vmin=flux_vmin, vmax=flux_vmax)
    plot_rect_map(problem, mc_ref_results["flux_total"], "MC Reference Time-Integrated Flux", output_dir / "mc_ref_flux_total.png", log_scale=True, vmin=flux_vmin, vmax=flux_vmax)

    plot_rect_map(problem, cfm_results["energy_total"], "CFM Time-Integrated Energy Deposition", output_dir / "cfm_energy_total.png", log_scale=True, vmin=energy_vmin, vmax=energy_vmax)
    plot_rect_map(problem, mc_results["energy_total"], "MC Mean Time-Integrated Energy Deposition", output_dir / "mc_energy_total.png", log_scale=True, vmin=energy_vmin, vmax=energy_vmax)
    plot_rect_map(problem, mc_ref_results["energy_total"], "MC Reference Time-Integrated Energy Deposition", output_dir / "mc_ref_energy_total.png", log_scale=True, vmin=energy_vmin, vmax=energy_vmax)

    plot_error_map_auto(problem, mc_ref_results["flux_total"], cfm_results["flux_total"], "CFM vs MC Reference Flux Absolute Error", output_dir / "cfm_vs_ref_flux_abs_error.png", use_log=True, cmap="jet")
    plot_error_map_auto(problem, mc_ref_results["flux_total"], mc_results["flux_total"], "MC Mean vs MC Reference Flux Absolute Error", output_dir / "mc_vs_ref_flux_abs_error.png", use_log=True, cmap="jet")
    plot_error_map_auto(problem, mc_ref_results["energy_total"], cfm_results["energy_total"], "CFM vs MC Reference Energy Absolute Error", output_dir / "cfm_vs_ref_energy_abs_error.png", use_log=True, cmap="magma")
    plot_error_map_auto(problem, mc_ref_results["energy_total"], mc_results["energy_total"], "MC Mean vs MC Reference Energy Absolute Error", output_dir / "mc_vs_ref_energy_abs_error.png", use_log=True, cmap="magma")

    plot_flux_lineout_comparison(
        problem,
        np.asarray(cfm_results["flux_total"], dtype=np.float64),
        np.asarray(mc_results["flux_total"], dtype=np.float64),
        np.asarray(mc_std_results["flux_total"], dtype=np.float64),
        np.asarray(mc_ref_results["flux_total"], dtype=np.float64),
        output_dir,
    )
    plot_total_flux_vs_time_comparison(cfm_results, mc_results, mc_std_results, mc_ref_results, output_dir / "total_flux_vs_time.png")



def total_boundary_leakage(results: dict) -> np.ndarray:
    return (
        np.asarray(results["flux_left_vs_time"], dtype=np.float64)
        + np.asarray(results["flux_right_vs_time"], dtype=np.float64)
        + np.asarray(results["flux_top_vs_time"], dtype=np.float64)
        + np.asarray(results["flux_bottom_vs_time"], dtype=np.float64)
    )



def plot_boundary_leakage(results: dict, filename: Path) -> None:
    t = results["time_centers"]
    total_leak = total_boundary_leakage(results)
    plt.figure(figsize=(9, 4))
    plt.plot(t, total_leak, linewidth=2.0)
    plt.xlabel("Time")
    plt.ylabel("Boundary leakage")
    plt.title("Total boundary leakage vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()



def plot_boundary_leakage_pair(cfm_results: dict, mc_results: dict, filename: Path) -> None:
    t = cfm_results["time_centers"]
    cfm_total = total_boundary_leakage(cfm_results)
    mc_total = total_boundary_leakage(mc_results)
    plt.figure(figsize=(9, 4))
    plt.plot(t, cfm_total, linewidth=2.0, label="CFM")
    plt.plot(t, mc_total, linewidth=2.0, label="MC")
    plt.xlabel("Time")
    plt.ylabel("Boundary leakage / particle")
    plt.title("Boundary leakage vs time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()



def save_results_npz(
    problem: RectangularLatticeProblem,
    results: dict,
    output_dir: Path,
    mc_results: Optional[dict] = None,
    mc_std_results: Optional[dict] = None,
    mc_ref_results: Optional[dict] = None,
) -> Path:
    out_path = output_dir / "rect_lattice_single_group_cfm_results.npz"
    payload = dict(
        material_map=problem.material_map,
        source_cells=problem.source_cells,
        cell_widths=problem.cell_widths,
        cell_heights=problem.cell_heights,
        cell_boundaries_x=problem.cell_boundaries_x,
        cell_boundaries_y=problem.cell_boundaries_y,
        cell_areas=problem.cell_areas,
        sigma_a_map=problem.sigma_a_map,
        sigma_s_map=problem.sigma_s_map,
        energy_vs_time=results["energy_vs_time"],
        tracklen_vs_time=results["tracklen_vs_time"],
        flux_vs_time=results["flux_vs_time"],
        energy_total=results["energy_total"],
        flux_total=results["flux_total"],
        flux_left_vs_time=results["flux_left_vs_time"],
        flux_right_vs_time=results["flux_right_vs_time"],
        flux_top_vs_time=results["flux_top_vs_time"],
        flux_bottom_vs_time=results["flux_bottom_vs_time"],
        time_edges=results["time_edges"],
        time_centers=results["time_centers"],
        elapsed=np.float64(results["elapsed"]),
        iterations=np.int64(results.get("iterations", -1)),
        totals=np.array(
            [
                int(results["totals"].get("left", 0)),
                int(results["totals"].get("right", 0)),
                int(results["totals"].get("top", 0)),
                int(results["totals"].get("bottom", 0)),
                int(results["totals"].get("absorb", 0)),
                int(results["totals"].get("time_like", 0)),
            ],
            dtype=np.int64,
        ),
        num_tracked=np.int64(results["num_tracked"]),
        status=np.asarray(results.get("status", np.empty(0, dtype=np.int8)), dtype=np.int8),
        terminal_code=np.asarray(results.get("terminal_code", np.empty(0, dtype=np.int8)), dtype=np.int8),
    )
    if mc_results is not None:
        payload.update(
            mc_energy_vs_time=mc_results["energy_vs_time"],
            mc_tracklen_vs_time=mc_results["tracklen_vs_time"],
            mc_flux_vs_time=mc_results["flux_vs_time"],
            mc_energy_total=mc_results["energy_total"],
            mc_flux_total=mc_results["flux_total"],
            mc_flux_left_vs_time=mc_results["flux_left_vs_time"],
            mc_flux_right_vs_time=mc_results["flux_right_vs_time"],
            mc_flux_top_vs_time=mc_results["flux_top_vs_time"],
            mc_flux_bottom_vs_time=mc_results["flux_bottom_vs_time"],
            mc_time_edges=mc_results["time_edges"],
            mc_time_centers=mc_results["time_centers"],
            mc_elapsed=np.float64(mc_results["elapsed"]),
            mc_totals=np.array(
                [
                    int(mc_results["totals"].get("left", 0)),
                    int(mc_results["totals"].get("right", 0)),
                    int(mc_results["totals"].get("top", 0)),
                    int(mc_results["totals"].get("bottom", 0)),
                    int(mc_results["totals"].get("absorb", 0)),
                    int(mc_results["totals"].get("time_like", 0)),
                ],
                dtype=np.int64,
            ),
            mc_num_tracked=np.int64(mc_results["num_tracked"]),
            flux_total_abs_error=np.abs(results["flux_total"] - mc_results["flux_total"]),
            energy_total_abs_error=np.abs(results["energy_total"] - mc_results["energy_total"]),
        )
    if mc_std_results is not None:
        payload.update(
            mc_flux_total_std=np.asarray(mc_std_results["flux_total"], dtype=np.float64),
            mc_energy_total_std=np.asarray(mc_std_results["energy_total"], dtype=np.float64),
            mc_flux_vs_time_std=np.asarray(mc_std_results["flux_vs_time"], dtype=np.float64),
            mc_energy_vs_time_std=np.asarray(mc_std_results["energy_vs_time"], dtype=np.float64),
        )
    if mc_ref_results is not None:
        payload.update(
            mc_ref_energy_vs_time=np.asarray(mc_ref_results["energy_vs_time"], dtype=np.float64),
            mc_ref_tracklen_vs_time=np.asarray(mc_ref_results["tracklen_vs_time"], dtype=np.float64),
            mc_ref_flux_vs_time=np.asarray(mc_ref_results["flux_vs_time"], dtype=np.float64),
            mc_ref_energy_total=np.asarray(mc_ref_results["energy_total"], dtype=np.float64),
            mc_ref_flux_total=np.asarray(mc_ref_results["flux_total"], dtype=np.float64),
            mc_ref_flux_left_vs_time=np.asarray(mc_ref_results["flux_left_vs_time"], dtype=np.float64),
            mc_ref_flux_right_vs_time=np.asarray(mc_ref_results["flux_right_vs_time"], dtype=np.float64),
            mc_ref_flux_top_vs_time=np.asarray(mc_ref_results["flux_top_vs_time"], dtype=np.float64),
            mc_ref_flux_bottom_vs_time=np.asarray(mc_ref_results["flux_bottom_vs_time"], dtype=np.float64),
            mc_ref_time_edges=np.asarray(mc_ref_results["time_edges"], dtype=np.float64),
            mc_ref_time_centers=np.asarray(mc_ref_results["time_centers"], dtype=np.float64),
            mc_ref_elapsed=np.float64(mc_ref_results["elapsed"]),
            mc_ref_totals=np.array(
                [
                    int(mc_ref_results["totals"].get("left", 0)),
                    int(mc_ref_results["totals"].get("right", 0)),
                    int(mc_ref_results["totals"].get("top", 0)),
                    int(mc_ref_results["totals"].get("bottom", 0)),
                    int(mc_ref_results["totals"].get("absorb", 0)),
                    int(mc_ref_results["totals"].get("time_like", 0)),
                ],
                dtype=np.int64,
            ),
            mc_ref_num_tracked=np.int64(mc_ref_results["num_tracked"]),
        )
    np.savez(out_path, **payload)
    return out_path



def print_terminal_stats(label: str, results: dict) -> None:
    totals = results["totals"]
    tracked = max(1, int(results["num_tracked"]))
    print(f"\n[{label} Exit State Statistics]")
    for key in ("left", "right", "top", "bottom", "absorb", "time_like"):
        val = int(totals.get(key, 0))
        print(f"  {key:<10s}: {val:>8d} ({100.0 * val / tracked:6.2f}%)")
    print(f"  {'─' * 40}\n  Total tracked: {tracked}")



def print_flux_energy_stats(problem: RectangularLatticeProblem, results: dict, mc_results: Optional[dict] = None) -> None:
    flux = np.asarray(results["flux_total"], dtype=np.float64)
    energy = np.asarray(results["energy_total"], dtype=np.float64)
    flux_integral = float(np.sum(flux * problem.cell_areas))
    energy_integral = float(np.sum(energy))

    print("\n[Flux And Energy Totals]")
    print(f"  integrated_flux     : CFM={flux_integral:.6e}")
    print(f"  total_energy_dep    : CFM={energy_integral:.6e}")

    if mc_results is not None:
        mc_flux = np.asarray(mc_results["flux_total"], dtype=np.float64)
        mc_energy = np.asarray(mc_results["energy_total"], dtype=np.float64)
        mc_flux_integral = float(np.sum(mc_flux * problem.cell_areas))
        mc_energy_integral = float(np.sum(mc_energy))
        flux_rel_l2 = float(np.linalg.norm(flux - mc_flux) / max(np.linalg.norm(mc_flux), 1.0e-30))
        energy_rel_l2 = float(np.linalg.norm(energy - mc_energy) / max(np.linalg.norm(mc_energy), 1.0e-30))
        print(f"  integrated_flux     : MC ={mc_flux_integral:.6e}")
        print(f"  total_energy_dep    : MC ={mc_energy_integral:.6e}")
        print(f"  rel_l2              : flux={flux_rel_l2:.6e} | energy={energy_rel_l2:.6e}")


# -----------------------------------------------------------------------------
# CLI / runtime configuration
# -----------------------------------------------------------------------------
def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    device = str(args.device or get_device())
    mc_seed = int(args.mc_reference_seed)
    mc_ref_seed = int(args.mc_ref_seed if args.mc_ref_seed is not None else (mc_seed + 1))
    mc_ref_particles = int(args.mc_ref_particles if args.mc_ref_particles is not None else (10 * int(args.num_particles)))
    return RuntimeConfig(
        num_particles=int(args.num_particles),
        mesh_refinement=int(args.mesh_refinement),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        mc_seed=mc_seed,
        mc_ref_seed=mc_ref_seed,
        mc_replicates=int(args.mc_replicates),
        mc_ref_particles=mc_ref_particles,
        boundary_model_dir=str(args.boundary_model_dir),
        boundary_norm_file=str(args.boundary_norm_file),
        internal_model_dir=str(args.internal_model_dir),
        internal_norm_file=str(args.internal_norm_file),
        train_sampling=str(args.train_sampling),
        device=device,
        randomize_start_pos=bool(args.randomize_start_pos),
        run_mc_reference=bool(args.run_mc_reference),
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rectangular single-group CFM lattice solver"
    )
    parser.add_argument("--num-particles", type=int, default=NUM_PARTICLES)
    parser.add_argument("--mesh-refinement", type=int, default=MESH_REFINEMENT)
    parser.add_argument("--output-dir", type=str, default="validation/results/lattice_2d_rect_single_group_cfm")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mc-reference-seed", type=int, default=DEFAULT_MC_REFERENCE_SEED)
    parser.add_argument("--mc-ref-seed", type=int, default=None)
    parser.add_argument("--mc-replicates", type=int, default=DEFAULT_MC_REPLICATES)
    parser.add_argument("--mc-ref-particles", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--boundary-model-dir", type=str, default=DEFAULT_BOUNDARY_MODEL_DIR)
    parser.add_argument("--boundary-norm-file", type=str, default=DEFAULT_BOUNDARY_NORM_FILE)
    parser.add_argument("--internal-model-dir", type=str, default=DEFAULT_INTERNAL_MODEL_DIR)
    parser.add_argument("--internal-norm-file", type=str, default=DEFAULT_INTERNAL_NORM_FILE)
    parser.add_argument("--train-sampling", type=str, default=TRAIN_SAMPLING, choices=["random", "qmc", "lhs"])
    parser.add_argument("--randomize-start-pos", dest="randomize_start_pos", action="store_true", default=True)
    parser.add_argument("--fixed-start-pos", dest="randomize_start_pos", action="store_false")
    parser.add_argument("--run-mc-reference", dest="run_mc_reference", action="store_true", default=True)
    parser.add_argument("--no-mc-reference", dest="run_mc_reference", action="store_false")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    maybe_enable_tf32(config.device)
    torch.set_grad_enabled(False)

    os.makedirs(config.output_dir, exist_ok=True)
    outdir = Path(config.output_dir)

    problem = build_structured_rect_problem(refinement=config.mesh_refinement)
    print(f"Built structured rectangular lattice (refinement={config.mesh_refinement})")
    print(f"Mesh cells: {problem.num_cells_x * problem.num_cells_y}")
    print(f"Source cells: {problem.source_cells.shape[0]}")
    print(f"Device: {config.device}")
    print(f"Seeds: CFM={config.seed} | MC={config.mc_seed} | MC_REF={config.mc_ref_seed}")
    print(f"MC replicates: {config.mc_replicates} | MC ref particles: {config.mc_ref_particles}")
    print(f"Rectangular models:\n  boundary={config.boundary_model_dir}\n  internal={config.internal_model_dir}")
    print(f"MC comparison stack: {'enabled' if config.run_mc_reference else 'disabled'}")

    with global_random_seed(config.seed):
        boundary_model, internal_model = load_rect_models(config)
        results = run_rect_single_group_cfm(
            problem,
            boundary_model,
            internal_model,
            num_particles=config.num_particles,
            device=config.device,
            randomize_start_pos=config.randomize_start_pos,
        )

    mc_results = None
    mc_std_results = None
    mc_ref_results = None
    if config.run_mc_reference:
        print(
            f"Running matched single-group rectangular MC replicate mean "
            f"(replicates={config.mc_replicates}, seed={config.mc_seed})"
        )
        mc_results, mc_std_results = run_rect_mc_replicate_mean(
            problem,
            num_particles=config.num_particles,
            num_replicates=config.mc_replicates,
            seed=config.mc_seed,
        )
        print(
            f"Running matched single-group rectangular MC reference "
            f"(particles={config.mc_ref_particles}, seed={config.mc_ref_seed})"
        )
        mc_ref_results = run_rect_mc_reference(problem, config.mc_ref_particles, config.mc_ref_seed)
        if mc_results is None or mc_ref_results is None:
            print("MC comparison unavailable; continuing with CFM-only outputs.")
            mc_results = None
            mc_std_results = None
            mc_ref_results = None

    print_terminal_stats("Rectangular CFM", results)
    if mc_results is not None:
        print_terminal_stats("Rectangular MC Mean", mc_results)
    if mc_ref_results is not None:
        print_terminal_stats("Rectangular MC Reference", mc_ref_results)
    print_flux_energy_stats(problem, results, mc_results)

    results_path = save_results_npz(
        problem,
        results,
        outdir,
        mc_results=mc_results,
        mc_std_results=mc_std_results,
        mc_ref_results=mc_ref_results,
    )
    print(f"\nSaved results to {results_path}")

    plot_rect_map(problem, results["flux_total"], "Rectangular CFM time-integrated flux", outdir / "rect_flux_total.png", log_scale=True)
    plot_rect_map(problem, results["energy_total"], "Rectangular CFM time-integrated energy deposition", outdir / "rect_energy_total.png", log_scale=True)
    plot_boundary_leakage(results, outdir / "rect_boundary_leakage_vs_time.png")
    if mc_results is not None and mc_ref_results is not None and mc_std_results is not None:
        plot_comparison_suite(problem, results, mc_results, mc_std_results, mc_ref_results, outdir)
        plot_boundary_leakage_pair(results, mc_results, outdir / "rect_boundary_leakage_vs_time_compare.png")
    elif mc_results is not None:
        plot_rect_map(problem, mc_results["flux_total"], "Rectangular MC time-integrated flux", outdir / "rect_mc_flux_total.png", log_scale=True)
        plot_rect_map(problem, mc_results["energy_total"], "Rectangular MC time-integrated energy deposition", outdir / "rect_mc_energy_total.png", log_scale=True)
        plot_rect_map_pair(
            problem,
            results["flux_total"],
            mc_results["flux_total"],
            ("Rectangular CFM flux", "Rectangular MC flux"),
            outdir / "rect_cfm_vs_mc_flux.png",
            log_scale=True,
        )
        plot_rect_map_pair(
            problem,
            results["energy_total"],
            mc_results["energy_total"],
            ("Rectangular CFM energy", "Rectangular MC energy"),
            outdir / "rect_cfm_vs_mc_energy.png",
            log_scale=True,
        )
        plot_error_map_auto(
            problem,
            mc_results["flux_total"],
            results["flux_total"],
            "Rectangular |CFM - MC| flux absolute error",
            outdir / "rect_cfm_vs_mc_flux_abs_error.png",
            use_log=True,
            cmap="jet",
        )
        plot_error_map_auto(
            problem,
            mc_results["energy_total"],
            results["energy_total"],
            "Rectangular |CFM - MC| energy absolute error",
            outdir / "rect_cfm_vs_mc_energy_abs_error.png",
            use_log=True,
            cmap="jet",
        )
        plot_boundary_leakage_pair(results, mc_results, outdir / "rect_cfm_vs_mc_boundary_leakage_vs_time.png")
    print(f"Plots saved to {outdir}")


if __name__ == "__main__":
    main()
