"""validation.utils – Shared utilities for validation scripts (2D CFM/MC helpers)

This module centralizes common helpers used across validation scripts:
- Device + seeding
- Geometry helpers (boundaries, orientation matrices)
- Model discovery + normalization loading
- ODE time grid construction
- GPU tally buffering utilities
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ml.transform import denorm_length_target, denorm_log_feature, load_normalization_params


# ───────────────────────────────────────────────────────────────────────
# Device, Seeds, Atomic I/O
# ───────────────────────────────────────────────────────────────────────

DEFAULT_DETERMINISTIC_SEED = 42
_GLOBAL_RANDOM_LOCK = threading.RLock()

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_seed(seed: Optional[int], default: int = DEFAULT_DETERMINISTIC_SEED) -> int:
    resolved = int(default if seed is None else seed)
    resolved %= (2**31 - 1)
    return resolved if resolved != 0 else 1


def set_random_seed(seed: int = DEFAULT_DETERMINISTIC_SEED) -> int:
    seed_i = resolve_seed(seed)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_i)
        torch.cuda.manual_seed_all(seed_i)
    return seed_i


@contextmanager
def global_random_seed(seed: Optional[int] = None) -> Iterator[int]:
    seed_i = resolve_seed(seed)
    with _GLOBAL_RANDOM_LOCK:
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        set_random_seed(seed_i)
        try:
            yield seed_i
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def exclusive_path_lock(
    target_path: Path | str,
    *,
    timeout_sec: float = 300.0,
    poll_interval_sec: float = 0.05,
) -> Iterator[Path]:
    target = Path(target_path)
    lock_path = target.with_name(f"{target.name}.lock")
    start = time.monotonic()
    lock_fd: Optional[int] = None

    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            if (time.monotonic() - start) >= timeout_sec:
                raise TimeoutError(f"Timed out waiting for lock on {target}")
            time.sleep(poll_interval_sec)

    try:
        yield target
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def atomic_savez(path: Path | str, *, compressed: bool = False, **payload: Any) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.stem}.",
        suffix=out_path.suffix or ".npz",
        dir=str(out_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **payload)
        else:
            np.savez(tmp_path, **payload)
        os.replace(tmp_path, out_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    return out_path


# ───────────────────────────────────────────────────────────────────────
# Geometry helpers
# ───────────────────────────────────────────────────────────────────────

def calculate_boundaries_2d(widths: np.ndarray, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative cell boundary coordinates along X and Y.

    Parameters
    - widths: 1D array of cell widths along X
    - heights: 1D array of cell heights along Y
    """
    bx = np.concatenate([[0.0], np.cumsum(widths)])
    by = np.concatenate([[0.0], np.cumsum(heights)])
    return bx, by


# Orientation matrices (face-local → model frame) and face-to-face mapping
# NOTE: Face indices (physical): 0=left, 1=right, 2=top, 3=bottom
# Pure rotations (no reflections), det = +1
# Map physical → canonical coordinates where the entry edge is the left edge (x=0).
_ORIENT_R_NP = [
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),    # Face 0: left (0°)
    np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),  # Face 1: right (180°)
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),   # Face 2: top (+90°)
    np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),   # Face 3: bottom (-90°)
]

def get_orientation_tensor(device: Optional[str] = None) -> torch.Tensor:
    dev = device or get_device()
    return torch.tensor(np.array(_ORIENT_R_NP), dtype=torch.float32, device=dev)

# Face index mapping when entering next cell (flip/invert)
FACE_TO_FACE = torch.tensor([1, 0, 3, 2], dtype=torch.int64)

# 3D face index convention (used by mc3dmesh + 3D CFM scripts):
#   0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmin, 5=zmax
FACE_TO_FACE_3D = torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.int64)

# 3D orientation matrices (physical cell frame -> canonical model frame where entry face is x=0).
# Each matrix is a proper rotation (det=+1). Translation is handled separately in
# `transform_face_to_model_3d` so that the entry plane lands on x=0 and the remaining
# coordinates lie in [0,Hm]x[0,Dm].
_ORIENT_R3D_NP = [
    # entry xmin (face 0): identity
    np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]], dtype=np.float32),
    # entry xmax (face 1): 180° about +z (x,y flip)
    np.array([[-1.0,  0.0, 0.0],
              [ 0.0, -1.0, 0.0],
              [ 0.0,  0.0, 1.0]], dtype=np.float32),
    # entry ymin (face 2): +90° about +z (x<-y), with x-range flipped into y-model
    np.array([[ 0.0,  1.0, 0.0],
              [-1.0,  0.0, 0.0],
              [ 0.0,  0.0, 1.0]], dtype=np.float32),
    # entry ymax (face 3): -90° about +z (x<- -y)
    np.array([[ 0.0, -1.0, 0.0],
              [ 1.0,  0.0, 0.0],
              [ 0.0,  0.0, 1.0]], dtype=np.float32),
    # entry zmin (face 4): cyclic permute (x<-z, y<-x, z<-y)
    np.array([[0.0, 0.0,  1.0],
              [1.0, 0.0,  0.0],
              [0.0, 1.0,  0.0]], dtype=np.float32),
    # entry zmax (face 5): like face 4 with z and y flipped (keeps det=+1)
    np.array([[ 0.0, 0.0, -1.0],
              [ 1.0, 0.0,  0.0],
              [ 0.0,-1.0,  0.0]], dtype=np.float32),
]


def get_orientation_tensor_3d(device: Optional[str] = None) -> torch.Tensor:
    dev = device or get_device()
    return torch.tensor(np.array(_ORIENT_R3D_NP), dtype=torch.float32, device=dev)

# Canonical-to-physical face mapping based on entry face
# Maps canonical_exit_face to physical_exit_face for each entry_face
# canonical_to_physical[entry_face][canonical_exit_face] = physical_exit_face
def get_canonical_to_physical_tensor(device: Optional[str] = None) -> torch.Tensor:
    """Get the canonical-to-physical face lookup table.

    This eliminates bias from coordinate-based face determination by using
    rotation-aware lookup based on the entry face orientation.

    Mapping:
    - Entry from left (face 0): identity rotation
    - Entry from right (face 1): 180° rotation
    - Entry from top (face 2): +90° rotation
    - Entry from bottom (face 3): -90° rotation
    """
    dev = device or get_device()
    return torch.tensor([
        [3, 1, 2, 0],  # Entry from left (face 0)
        [2, 0, 3, 1],  # Entry from right (face 1)
        [0, 3, 1, 2],  # Entry from top (face 2)
        [1, 2, 0, 3],  # Entry from bottom (face 3)
    ], dtype=torch.int64, device=dev)

# Perimeter-to-physical face conversion
# perimeter_to_face_and_pos returns: 0=bottom, 1=right, 2=top, 3=left (in canonical frame)
# Physical faces used by boundary model: 0=left, 1=right, 2=top, 3=bottom
_PERIMETER_TO_PHYSICAL_NP = np.array([3, 1, 2, 0], dtype=np.int64)

def get_perimeter_to_physical_tensor(device: Optional[str] = None) -> torch.Tensor:
    """Get the perimeter-to-physical face conversion mapping."""
    dev = device or get_device()
    return torch.tensor(_PERIMETER_TO_PHYSICAL_NP, dtype=torch.int64, device=dev)


# ───────────────────────────────────────────────────────────────────────
# Model discovery & normalization
# ───────────────────────────────────────────────────────────────────────

def find_model_file(model_dir: str, patterns: Optional[Sequence[str]] = None) -> str:
    """Find a model checkpoint file inside `model_dir` or current directory.

    Returns last lexicographic match to prefer latest, mirroring previous logic.
    """
    patterns = patterns or ("best_model*.pth", "trained_model_final*.pth")
    search_dirs = [model_dir, "."]
    for d in search_dirs:
        files: list[Path] = []
        for p in patterns:
            files.extend(sorted(Path(d).glob(p)))
        if files:
            return str(files[-1])
    raise FileNotFoundError(f"Model file not found in {model_dir}")


def load_norm_params_file(path: os.PathLike | str, *, train_sampling: str = "random") -> Tuple[str, Dict[str, Any]]:
    """Load normalization metadata from a single configured path."""
    norm_params, source_path = load_normalization_params(str(path), train_sampling=train_sampling)
    return source_path, norm_params


def decode_budgeted_length(z_value, budget_value, norm_params: Dict[str, Any], key: str = "L"):
    """Decode timedep path-length targets using saved normalization metadata."""
    return denorm_length_target(z_value, budget_value, norm_params, key=key)


# ───────────────────────────────────────────────────────────────────────
# ODE time grid
# ───────────────────────────────────────────────────────────────────────

def build_time_grid(num_steps: int = 12, split: float = 0.8, device: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct the piecewise time grid used in the multi-cell solver.

    Mirrors `validation.multi_cell` by using a coarse segment up to `split`
    followed by a refined segment to 1.0. Returns (T_grid, step_size); the
    latter matches the first coarse step and is `None` when no step exists.
    """
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
    T_grid = torch.cat([base_grid, fine_grid])

    if T_grid.numel() != num_steps:
        T_grid = torch.linspace(0.0, 1.0, num_steps, device=dev)

    step_size = T_grid[1] - T_grid[0] if T_grid.numel() > 1 else None
    return T_grid, step_size


# ───────────────────────────────────────────────────────────────────────
# GPU tally buffer
# ───────────────────────────────────────────────────────────────────────

class TallyBuffer:
    """Accumulate (indices, values) pairs on GPU and flush with a single op.

    Use for energy deposition or track-length tallies with flat index buffers.
    """

    def __init__(self, capacity_hint: int = 8192):
        self._idx: list[torch.Tensor] = []
        self._val: list[torch.Tensor] = []
        self._n = 0
        self.capacity_hint = int(capacity_hint)

    @property
    def size(self) -> int:
        return self._n

    def add(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        # Expect 1D tensors on correct device; caller responsibility
        self._idx.append(indices)
        self._val.append(values)
        self._n += int(indices.numel())

    def flush_index_add(self, target: torch.Tensor) -> int:
        if self._n == 0:
            return 0
        idx = torch.cat(self._idx)
        val = torch.cat(self._val)
        target.index_add_(0, idx, val)
        self._idx.clear(); self._val.clear(); n = self._n; self._n = 0
        return n

    def flush_scatter_add(self, target: torch.Tensor) -> int:
        if self._n == 0:
            return 0
        idx = torch.cat(self._idx)
        val = torch.cat(self._val)
        target.scatter_add_(0, idx, val)
        self._idx.clear(); self._val.clear(); n = self._n; self._n = 0
        return n


# ───────────────────────────────────────────────────────────────────────
# Model loader and CFM helpers
# ───────────────────────────────────────────────────────────────────────

def load_model_and_norms(
    model_dir: str,
    norm_path: os.PathLike | str,
    input_dim: int,
    cond_dim: int,
    hidden_dim: int,
    depth: int,
    device: Optional[str] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """Load a CFM model, wrap, solver, and normalization params.

    Returns dict with keys: model, wrapper, solver, model_path, norm, norm_path.
    """
    dev = device or get_device()
    # Import locally to avoid global dependency when unused
    from ml.funcs import get_model, WrappedModel
    from flow_matching.solver import ODESolver

    model = get_model(input_dim, cond_dim, hidden_dim, depth).to(dev)
    model_path = find_model_file(model_dir)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    wrapper = WrappedModel(model).to(dev)
    solver = ODESolver(velocity_model=wrapper)
    norm_path, norm = load_norm_params_file(norm_path)

    # Warmup pass to trigger CUDA initialization (cuDNN, kernel compilation, etc.)
    if warmup and dev == "cuda":
        with torch.no_grad():
            dummy_input = torch.randn(8, input_dim, device=dev)
            dummy_t = torch.rand(8, device=dev)
            _ = model(dummy_input, dummy_t)
            torch.cuda.synchronize()

    return {
        "model": model,
        "wrapper": wrapper,
        "solver": solver,
        "model_path": model_path,
        "norm": norm,
        "norm_path": norm_path,
    }


def build_boundary_cond_tensor(
    Wm: torch.Tensor,
    Hm: torch.Tensor,
    yn: torch.Tensor,
    ux_m: torch.Tensor,
    uy_m: torch.Tensor,
    norm_params: Dict[str, Any],
) -> torch.Tensor:
    """Build boundary condition tensor [N,5] using ml.funcs normalizers.

    norm_params: steady-state normalization metadata dictionary
    """
    from ml.funcs import (
        normalize_width, normalize_height, normalize_position_relative,
        normalize_angle_phi, normalize_radius_r, uv_to_polar,
    )
    width_l, width_u = norm_params["width_lower"], norm_params["width_upper"]
    height_l, height_u = norm_params["height_lower"], norm_params["height_upper"]
    phi_in, r_in = uv_to_polar(ux_m, uy_m)
    cond = torch.stack([
        normalize_width(Wm, width_l, width_u),
        normalize_height(Hm, height_l, height_u),
        normalize_position_relative(yn),
        normalize_angle_phi(phi_in),
        normalize_radius_r(r_in)
    ], dim=1)
    return cond


def run_boundary_model(
    solver: Any,
    cond: torch.Tensor,
    target_dim: int,
    norm_params: Dict[str, Any],
    T_grid: torch.Tensor,
    step_size: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run boundary solver and return denormalized predictions in perimeter/polar form.

    Returns (perimeter_p, u_model, v_model, dt)
    """
    from ml.funcs import (
        denorm_perimeter_p, denorm_angle_phi, denorm_radius_r, polar_to_uv,
    )
    noise = torch.randn(cond.shape[0], target_dim, device=cond.device)
    x0 = torch.cat([cond, noise], dim=1)
    with torch.no_grad():
        out = solver.sample(T_grid=T_grid, x_init=x0, method='rk4', step_size=step_size,
                            return_intermediates=False)
    pred = out[:, cond.shape[1]:]
    perimeter_p = denorm_perimeter_p(pred[:, 0])
    phi_out = denorm_angle_phi(pred[:, 1])
    r_out = denorm_radius_r(pred[:, 2])
    dt = denorm_log_feature(pred[:, 3], norm_params, "time").clamp_min(1e-9)
    u_model, v_model = polar_to_uv(phi_out, r_out)
    return perimeter_p, u_model, v_model, dt


def build_internal_cond_tensor(
    W: torch.Tensor,
    H: torch.Tensor,
    x_rel: torch.Tensor,
    y_rel: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    norm_params: Dict[str, Any],
) -> torch.Tensor:
    """Build internal condition tensor [N,6] using ml.funcs normalizers.

    norm_params: steady-state normalization metadata dictionary
    """
    from ml.funcs import (
        normalize_width, normalize_height, normalize_position_relative,
        normalize_angle_phi, normalize_radius_r, uv_to_polar,
    )
    width_l, width_u = norm_params["width_lower"], norm_params["width_upper"]
    height_l, height_u = norm_params["height_lower"], norm_params["height_upper"]
    phi_in, r_in = uv_to_polar(u, v)
    cond = torch.stack([
        normalize_width(W, width_l, width_u),
        normalize_height(H, height_l, height_u),
        normalize_position_relative(x_rel),
        normalize_position_relative(y_rel),
        normalize_angle_phi(phi_in),
        normalize_radius_r(r_in),
    ], dim=1)
    return cond


def run_internal_model(
    solver: Any,
    cond: torch.Tensor,
    target_dim: int,
    norm_params: Dict[str, Any],
    T_grid: torch.Tensor,
    step_size: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run internal solver and return denormalized predictions in perimeter/polar form.

    Returns (perimeter_p, u_out, v_out, dt)
    """
    from ml.funcs import (
        denorm_perimeter_p, denorm_angle_phi, denorm_radius_r, polar_to_uv,
    )
    noise = torch.randn(cond.shape[0], target_dim, device=cond.device)
    x0 = torch.cat([cond, noise], dim=1)
    with torch.no_grad():
        out = solver.sample(T_grid=T_grid, x_init=x0, method='rk4', step_size=step_size,
                            return_intermediates=False)
    pred = out[:, cond.shape[1]:]
    perimeter_p = denorm_perimeter_p(pred[:, 0])
    phi_out = denorm_angle_phi(pred[:, 1])
    r_out = denorm_radius_r(pred[:, 2])
    dt = denorm_log_feature(pred[:, 3], norm_params, "time").clamp_min(0.)
    u_out, v_out = polar_to_uv(phi_out, r_out)
    return perimeter_p, u_out, v_out, dt


# ───────────────────────────────────────────────────────────────────────
# Face-local transform helpers
# ───────────────────────────────────────────────────────────────────────

def enforce_outward_angle(
    p: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enforce that predicted angles point outward from the exit face.

    Given a perimeter position p and angle (u, v), this function:
    1. Determines which face the particle exits from based on p
    2. Checks if the angle points outward from that face
    3. If the angle points inward, reflects it to point outward

    This ensures physical consistency: particles exiting a face must have
    velocities pointing away from that face into the next cell.

    Args:
        p: Perimeter coordinate in [0, 1) (tensor)
        u, v: Exit direction components (tensors)
        W, H: Cell dimensions (tensors or scalars)
        eps: Small epsilon for numerical stability

    Returns:
        (u_corrected, v_corrected): Angle components guaranteed to point outward
    """
    # Determine exit face from perimeter using canonical convention
    # Face IDs: 0=bottom, 1=right, 2=top, 3=left
    face_canonical, _ = perimeter_to_face_and_pos(p, W, H, eps)

    # Define outward normals for each canonical face
    # Shape: [4, 2] for [face_id, (nx, ny)]
    device = p.device if isinstance(p, torch.Tensor) else u.device
    normals = torch.tensor([
        [0.0, -1.0],   # Face 0 (bottom): points down (negative y)
        [1.0, 0.0],    # Face 1 (right): points right (positive x)
        [0.0, 1.0],    # Face 2 (top): points up (positive y)
        [-1.0, 0.0],   # Face 3 (left): points left (negative x)
    ], dtype=torch.float32, device=device)

    # Get the outward normal for each particle's exit face
    # Shape: [N, 2]
    n = normals[face_canonical]

    # Compute dot product: (u, v) · n
    # If dot product > 0, angle points outward (valid)
    # If dot product <= 0, angle points inward (invalid, needs flip)
    direction = torch.stack([u, v], dim=1)  # Shape: [N, 2]
    dot_product = (direction * n).sum(dim=1)  # Shape: [N]

    # Reflect angles that point inward (dot product <= 0)
    # Reflection formula: v_reflected = v - 2 * (v · n) * n
    # Only apply reflection when dot product <= 0
    needs_flip = dot_product <= 0

    if needs_flip.any():
        # Compute reflection for all particles (will only use where needed)
        # reflection = direction - 2 * (direction · n) * n
        reflection_factor = 2.0 * dot_product.unsqueeze(1)  # Shape: [N, 1]
        reflected = direction - reflection_factor * n  # Shape: [N, 2]

        # Apply reflection only where needed
        corrected = torch.where(
            needs_flip.unsqueeze(1),  # Shape: [N, 1]
            reflected,
            direction
        )
        u_out, v_out = corrected[:, 0], corrected[:, 1]
    else:
        # No corrections needed
        u_out, v_out = u, v

    return u_out, v_out


def transform_face_to_model(
    face: torch.Tensor,
    pos_on_face: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    R_tensor: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform physical face-local state to model frame and compute dims.

    Returns (Wm, Hm, yn, ux_m, uy_m, Rs, t_vecs)
    """
    zeros_pos = torch.zeros_like(pos_on_face)
    px = torch.where(face == 0, zeros_pos,
         torch.where(face == 1, W,
         torch.where(face == 2, pos_on_face, pos_on_face)))
    py = torch.where(face == 0, pos_on_face,
         torch.where(face == 1, pos_on_face,
         torch.where(face == 2, H, zeros_pos)))
    pos_phys = torch.stack([px, py], dim=1)

    # Translation so that the rotated entry edge lands on x=0 and y ∈ [0, Hm]
    tx = torch.where(face == 0, torch.zeros_like(W),
         torch.where(face == 1, W,
         torch.where(face == 2, H, torch.zeros_like(W))))
    ty = torch.where(face == 1, H,          # right: y' = -y + H
         torch.where(face == 3, W,          # bottom: y' = -x + W
         torch.zeros_like(W)))              # left/top: no y shift
    t_vecs = torch.stack([tx, ty], dim=1)

    Rs = R_tensor[face]
    dir_phys = torch.stack([u, v], dim=1).unsqueeze(-1)
    pos_model = torch.bmm(Rs, pos_phys.unsqueeze(-1)).squeeze(-1) + t_vecs
    dir_model = torch.bmm(Rs, dir_phys).squeeze(-1)
    ux_m, uy_m = dir_model[:, 0], dir_model[:, 1]

    is_lr = face < 2
    Wm, Hm = torch.where(is_lr, W, H), torch.where(is_lr, H, W)
    yn = torch.clamp(pos_model[:, 1] / (Hm + eps), 0.0, 1.0)
    return Wm, Hm, yn, ux_m, uy_m, Rs, t_vecs


def entry_position_from_face_3d(
    face: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x,y,z) on the given 3D face using two face-local coordinates.

    Face convention: 0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmin, 5=zmax.
    pos_a/pos_b meaning depends on face:
      - x faces: (pos_a,pos_b)=(y,z)
      - y faces: (pos_a,pos_b)=(x,z)
      - z faces: (pos_a,pos_b)=(x,y)
    """
    zeros = torch.zeros_like(pos_a)
    x = torch.where(face == 0, zeros,
        torch.where(face == 1, W,
        torch.where((face == 2) | (face == 3), pos_a, pos_a)))
    y = torch.where((face == 0) | (face == 1), pos_a,
        torch.where(face == 2, zeros,
        torch.where(face == 3, H, pos_b)))
    z = torch.where(face == 4, zeros,
        torch.where(face == 5, D,
        pos_b))
    return x, y, z


def classify_face_and_snap_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify nearest 3D face and snap coordinates onto it (batched).

    Uses distance-based argmin for all particles to avoid face-priority bias.
    """
    d_stack = torch.stack([x, W - x, y, H - y, z, D - z], dim=1)
    face = torch.argmin(d_stack, dim=1)

    zeros = torch.zeros_like(x)
    x_snap = torch.where(face == 0, zeros, torch.where(face == 1, W, x))
    y_snap = torch.where(face == 2, zeros, torch.where(face == 3, H, y))
    z_snap = torch.where(face == 4, zeros, torch.where(face == 5, D, z))

    x_snap = torch.minimum(torch.maximum(x_snap, zeros), W)
    y_snap = torch.minimum(torch.maximum(y_snap, zeros), H)
    z_snap = torch.minimum(torch.maximum(z_snap, zeros), D)

    # Face-local coordinates for boundary model entry.
    is_xface = face < 2
    is_yface = (face >= 2) & (face < 4)
    pos_a = torch.where(is_xface, y_snap, x_snap)
    pos_b = torch.where(is_yface, z_snap, torch.where(is_xface, z_snap, y_snap))

    return face, x_snap, y_snap, z_snap, pos_a, pos_b


def enforce_outward_by_face_3d(
    face: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reflect directions so that they point outward through the given face."""
    normals = torch.tensor([
        [-1.0,  0.0,  0.0],  # xmin
        [ 1.0,  0.0,  0.0],  # xmax
        [ 0.0, -1.0,  0.0],  # ymin
        [ 0.0,  1.0,  0.0],  # ymax
        [ 0.0,  0.0, -1.0],  # zmin
        [ 0.0,  0.0,  1.0],  # zmax
    ], dtype=torch.float32, device=u.device)
    n = normals[face]
    dot = u * n[:, 0] + v * n[:, 1] + w * n[:, 2]
    needs_flip = dot <= 0
    u_ref = u - 2.0 * dot * n[:, 0]
    v_ref = v - 2.0 * dot * n[:, 1]
    w_ref = w - 2.0 * dot * n[:, 2]
    u = torch.where(needs_flip, u_ref, u)
    v = torch.where(needs_flip, v_ref, v)
    w = torch.where(needs_flip, w_ref, w)
    return u, v, w


def transform_face_to_model_3d(
    face: torch.Tensor,
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    D: torch.Tensor,
    R_tensor_3d: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
]:
    """Transform 3D physical face-local state into the canonical model frame.

    Returns:
      (Wm, Hm, Dm, y_norm, z_norm, u_m, v_m, w_m, Rs, t_vecs)

    Canonical model frame:
      - entry face is always x=0
      - (y,z) parameterize the entry face in [0,Hm]x[0,Dm]
      - direction is expressed in the canonical axes, with u_m being the +x component
    """
    x0, y0, z0 = entry_position_from_face_3d(face, pos_a, pos_b, W, H, D)
    pos_phys = torch.stack([x0, y0, z0], dim=1)

    # Translation chosen per face so that entry plane maps to x=0 and (y,z) are non-negative.
    tx = torch.where(face == 1, W,  # xmax: x'=-x+W
         torch.where(face == 3, H,  # ymax: x'=-y+H
         torch.where(face == 5, D,  # zmax: x'=-z+D
         torch.zeros_like(W))))
    ty = torch.where(face == 1, H,  # xmax: y'=-y+H
         torch.where(face == 2, W,  # ymin: y'=-x+W
         torch.zeros_like(W)))
    tz = torch.where(face == 5, H,  # zmax: z'=-y+H
         torch.zeros_like(W))
    t_vecs = torch.stack([tx, ty, tz], dim=1)

    Rs = R_tensor_3d[face]
    dir_phys = torch.stack([u, v, w], dim=1).unsqueeze(-1)
    pos_model = torch.bmm(Rs, pos_phys.unsqueeze(-1)).squeeze(-1) + t_vecs
    dir_model = torch.bmm(Rs, dir_phys).squeeze(-1)
    u_m, v_m, w_m = dir_model[:, 0], dir_model[:, 1], dir_model[:, 2]

    is_xface = face < 2
    is_yface = (face >= 2) & (face < 4)
    Wm = torch.where(is_xface, W, torch.where(is_yface, H, D))
    Hm = torch.where(is_xface, H, torch.where(is_yface, W, W))
    Dm = torch.where(is_xface, D, torch.where(is_yface, D, H))

    y_norm = torch.clamp(pos_model[:, 1] / (Hm + eps), 0.0, 1.0)
    z_norm = torch.clamp(pos_model[:, 2] / (Dm + eps), 0.0, 1.0)
    return Wm, Hm, Dm, y_norm, z_norm, u_m, v_m, w_m, Rs, t_vecs


def perimeter_to_face_and_pos(p: torch.Tensor, Wm: torch.Tensor, Hm: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert perimeter coordinate p ∈ [0,1) to face and position using geometric perimeter.
    Uses actual geometric perimeter based on Wm, Hm (not uniform quarters).
    Convention: left → top → right → bottom (counterclockwise from bottom-left).

    Args:
        p: perimeter coordinate (tensor)
        Wm, Hm: canonical box dimensions (tensors matching p shape)

    Returns:
        face: 0=bottom, 1=right, 2=top, 3=left (tensor)
        position: position along that face (tensor)
    """
    # Batched operation
    P = 2.0 * (Wm + Hm)  # Total perimeter
    s = p * P  # Arc length from start

    # Determine face and position for each particle
    face = torch.zeros_like(s, dtype=torch.int64)
    position = torch.zeros_like(s)

    # Left face: s < Hm
    mask_left = s < Hm
    face[mask_left] = 3
    position[mask_left] = s[mask_left]

    # Top face: Hm <= s < Hm + Wm
    mask_top = (s >= Hm) & (s < Hm + Wm)
    face[mask_top] = 2
    position[mask_top] = s[mask_top] - (Hm[mask_top] if isinstance(Hm, torch.Tensor) and Hm.dim() > 0 else Hm)

    # Right face: Hm + Wm <= s < 2*Hm + Wm
    mask_right = (s >= Hm + Wm) & (s < 2*Hm + Wm)
    face[mask_right] = 1
    if isinstance(Hm, torch.Tensor) and Hm.dim() > 0:
        position[mask_right] = (2*Hm + Wm)[mask_right] - s[mask_right]
    else:
        position[mask_right] = (2*Hm + Wm) - s[mask_right]

    # Bottom face: s >= 2*Hm + Wm
    mask_bottom = s >= 2*Hm + Wm
    face[mask_bottom] = 0
    if isinstance(P, torch.Tensor) and P.dim() > 0:
        position[mask_bottom] = P[mask_bottom] - s[mask_bottom]
    else:
        position[mask_bottom] = P - s[mask_bottom]

    return face, position


def _perimeter_to_model_xy(
    p: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map perimeter coordinate p ∈ [0,1) to (face_idx, x, y) using geometric perimeter.

    Uses actual geometric perimeter (not uniform quarters).
    Convention: left → top → right → bottom (counterclockwise from bottom-left).
    Physical face IDs: 0=left, 1=right, 2=top, 3=bottom
    """
    p_unit = torch.remainder(p, 1.0)

    # Get face and position along that face
    face_canonical, pos_on_face = perimeter_to_face_and_pos(p_unit, W, H, eps)

    # Convert canonical face (3=left, 2=top, 1=right, 0=bottom) to (x, y)
    x = torch.zeros_like(pos_on_face)
    y = torch.zeros_like(pos_on_face)

    # Left face (canonical=3): x=0, y=pos
    mask_left = face_canonical == 3
    x[mask_left] = 0.0
    y[mask_left] = pos_on_face[mask_left]

    # Top face (canonical=2): x=pos, y=H
    mask_top = face_canonical == 2
    x[mask_top] = pos_on_face[mask_top]
    y[mask_top] = H[mask_top] if isinstance(H, torch.Tensor) and H.dim() > 0 else H

    # Right face (canonical=1): x=W, y=pos
    mask_right = face_canonical == 1
    x[mask_right] = W[mask_right] if isinstance(W, torch.Tensor) and W.dim() > 0 else W
    y[mask_right] = pos_on_face[mask_right]

    # Bottom face (canonical=0): x=pos, y=0
    mask_bottom = face_canonical == 0
    x[mask_bottom] = pos_on_face[mask_bottom]
    y[mask_bottom] = 0.0

    # Clamp to bounds
    if isinstance(W, torch.Tensor):
        W_clamped = torch.clamp(W, min=eps)
        x = torch.clamp(x, torch.zeros_like(W), W_clamped)
    else:
        x = torch.clamp(x, min=0.0, max=max(W, eps))

    if isinstance(H, torch.Tensor):
        H_clamped = torch.clamp(H, min=eps)
        y = torch.clamp(y, torch.zeros_like(H), H_clamped)
    else:
        y = torch.clamp(y, min=0.0, max=max(H, eps))

    # Map canonical face to physical face convention (0=left, 1=right, 2=top, 3=bottom)
    face_physical = torch.where(face_canonical == 3, torch.zeros_like(face_canonical),  # left -> 0
                    torch.where(face_canonical == 1, torch.ones_like(face_canonical),   # right -> 1
                    torch.where(face_canonical == 2, torch.full_like(face_canonical, 2), # top -> 2
                                                      torch.full_like(face_canonical, 3))))  # bottom -> 3

    return face_physical, x, y


def project_to_exit_face(
    face: torch.Tensor,
    perimeter_p: torch.Tensor,
    ux_m: torch.Tensor,
    uy_m: torch.Tensor,
    Wm_model: torch.Tensor,
    Hm_model: torch.Tensor,
    Wm: torch.Tensor,
    Hm: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    Rs: torch.Tensor,
    t_vecs: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project model-frame prediction to physical exit using rotation-aware lookup table.

    This eliminates bias from coordinate-based face determination by using the
    canonical-to-physical lookup table based on entry face orientation.

    IMPORTANT: Wm_model and Hm_model MUST be GEOMETRIC dimensions (not optical).
    Perimeter decoding is a purely geometric operation. Optical dimensions are
    ONLY for network conditioning, never for geometry.

    Args:
        face: Entry face index for each particle
        perimeter_p: Perimeter coordinate from model (in [0,1))
        ux_m, uy_m: Exit direction in model frame
        Wm_model, Hm_model: Canonical box dimensions (GEOMETRIC, not optical)
        Wm, Hm: Canonical box dimensions (should equal Wm_model, Hm_model)
        W, H: Physical cell dimensions (before rotation)
        Rs, t_vecs: Rotation matrices and translation vectors
        eps: Small epsilon for numerical stability

    Returns:
        (exit_face, pos_on_face, u_exit, v_exit, x_exit, y_exit)
    """
    # Get device from input tensors
    device = perimeter_p.device

    # Get lookup table on correct device
    _CANONICAL_TO_PHYSICAL = get_canonical_to_physical_tensor(device)

    # Convert perimeter coordinate to canonical face and position using GEOMETRIC dimensions
    # IMPORTANT: Wm_model, Hm_model must be GEOMETRIC (not optical) for correct perimeter decoding
    p_scaled = perimeter_p % 1.0  # Ensure [0, 1)
    exit_face_canonical, pos_on_exit_face_canonical = perimeter_to_face_and_pos(p_scaled, Wm_model, Hm_model, eps=eps)

    # Scale position from model space to physical canonical space
    # When Wm_model=Wm and Hm_model=Hm (geometric), scale_factor=1.0
    # This code path exists for compatibility but should always use geometric dimensions
    Wm_model_clamped = torch.clamp(Wm_model, min=eps)
    Hm_model_clamped = torch.clamp(Hm_model, min=eps)

    # Position is along one dimension depending on face
    # For left/right faces (canonical 3/1): position is along y
    # For top/bottom faces (canonical 2/0): position is along x
    # Scale from model to physical dimensions (should be 1.0 when using geometric)
    is_vertical = (exit_face_canonical == 3) | (exit_face_canonical == 1)  # left or right
    scale_factor = torch.where(is_vertical, Hm / Hm_model_clamped, Wm / Wm_model_clamped)
    pos_on_exit_face_physical = pos_on_exit_face_canonical * scale_factor

    # Transform canonical exit face to physical exit face using rotation-aware lookup
    # This eliminates bias from coordinate-based face determination
    b = _CANONICAL_TO_PHYSICAL[face, exit_face_canonical]

    # Construct canonical exit position (xm, ym) from face and position in physical space
    xm = torch.where(
        exit_face_canonical == 3, torch.zeros_like(Wm),           # left: x=0
        torch.where(exit_face_canonical == 1, Wm,                 # right: x=Wm
        torch.where(exit_face_canonical == 2, pos_on_exit_face_physical,  # top: x=pos
                                              pos_on_exit_face_physical))  # bottom: x=pos
    )
    ym = torch.where(
        exit_face_canonical == 3, pos_on_exit_face_physical,    # left: y=pos
        torch.where(exit_face_canonical == 1, pos_on_exit_face_physical,  # right: y=pos
        torch.where(exit_face_canonical == 2, Hm,                 # top: y=Hm
                                              torch.zeros_like(Hm)))  # bottom: y=0
    )

    # Inverse transform back to physical coordinates
    Rs_T = Rs.transpose(1, 2)
    pos_phys_exit = torch.bmm(Rs_T, (torch.stack([xm, ym], dim=1) - t_vecs).unsqueeze(-1)).squeeze(-1)
    dir_phys_exit = torch.bmm(Rs_T, torch.stack([ux_m, uy_m], dim=1).unsqueeze(-1)).squeeze(-1)
    x_p, y_p = pos_phys_exit[:, 0], pos_phys_exit[:, 1]
    u_p, v_p = dir_phys_exit[:, 0], dir_phys_exit[:, 1]

    # Compute position on exit face in physical coordinates
    # Clamp to handle small numerical errors
    x_pc = x_p.clamp(min=torch.zeros_like(W), max=W)
    y_pc = y_p.clamp(min=torch.zeros_like(H), max=H)
    s = torch.where(b < 2, y_pc, x_pc)  # if left/right use y; if top/bottom use x

    return b, s, u_p, v_p, x_p, y_p


# ───────────────────────────────────────────────────────────────────────
# Transport helpers for 2D particle simulations
# ───────────────────────────────────────────────────────────────────────

def determine_exit_face_for_survivors(
    perimeter_p: torch.Tensor,
    W_model: torch.Tensor,
    H_model: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Determine exit face and position on face for surviving particles using geometric perimeter.

    IMPORTANT: W_model and H_model MUST be GEOMETRIC dimensions (not optical).
    Perimeter decoding is a purely geometric operation.

    Args:
        perimeter_p: Perimeter coordinate from model (in [0,1))
        W_model, H_model: Cell dimensions for perimeter decoding (GEOMETRIC, not optical)
        W, H: Physical cell dimensions (should equal W_model, H_model for internal source)
        u, v: Exit direction components

    Returns:
        (exit_face, pos_on_face, u_exit, v_exit)
    """
    # Use corrected perimeter conversion which returns proper face ID and position
    face_physical, x_model, y_model = _perimeter_to_model_xy(perimeter_p, W_model, H_model)

    # Scale from model frame to physical frame
    W_model_clamped = torch.clamp(W_model, min=1e-9)
    H_model_clamped = torch.clamp(H_model, min=1e-9)
    x_rel = torch.clamp(x_model / W_model_clamped, 0.0, 1.0)
    y_rel = torch.clamp(y_model / H_model_clamped, 0.0, 1.0)
    x_exit = x_rel * W
    y_exit = y_rel * H

    # Position on face based on which face
    # Face convention: 0=left, 1=right, 2=top, 3=bottom
    pos_on_face = torch.where(face_physical < 2, y_exit, x_exit)

    return face_physical, pos_on_face, u, v


def get_cell_material_properties(
    cell_x: torch.Tensor,
    cell_y: torch.Tensor,
    num_cells_x: int,
    sigma_t: torch.Tensor,
    sigma_a: torch.Tensor,
    scatter_ratio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get material properties for given cells.
    
    Returns (lin_idx, sigma_t_values, sigma_a_values, scatter_ratio_values)
    """
    lin_idx = cell_y * num_cells_x + cell_x
    return lin_idx, sigma_t[lin_idx], sigma_a[lin_idx], scatter_ratio[lin_idx]


def process_energy_deposition_and_absorption(
    particles: torch.Tensor,
    active: torch.Tensor,
    energy_col: int,
    status_col: int,
    sigma_a: torch.Tensor,
    dt: torch.Tensor,
    energy_buffer: TallyBuffer,
    lin_idx: torch.Tensor,
    threshold: float,
    totals_tensor: torch.Tensor,
    absorb_idx: int = 4,
) -> torch.Tensor:
    """Handle energy deposition and check for absorption.
    
    Returns mask of absorbed particles.
    """
    # Energy deposition
    E = particles[active, energy_col]
    deposit = E * (1 - torch.exp(-sigma_a * dt))
    energy_buffer.add(lin_idx, deposit)
    particles[active, energy_col] -= deposit
    
    # Check absorption
    absorbed = particles[active, energy_col] < threshold
    if absorbed.any():
        totals_tensor[absorb_idx] += absorbed.sum()
        particles[active[absorbed], status_col] = 2
    
    return absorbed


def update_particles_after_projection(
    particles: torch.Tensor,
    active: torch.Tensor,
    exit_faces: torch.Tensor,
    positions: torch.Tensor,
    velocities_u: torch.Tensor,
    velocities_v: torch.Tensor,
    current_ix: torch.Tensor,
    current_iy: torch.Tensor,
    num_cells_x: int,
    num_cells_y: int,
    totals_tensor: torch.Tensor,
    face_to_face: torch.Tensor,
) -> torch.Tensor:
    """Update particle states after projection and handle mesh exits.
    
    Returns mask of particles that exit the mesh.
    """
    # Calculate new cell indices
    new_ix = current_ix + (exit_faces == 1).long() - (exit_faces == 0).long()
    new_iy = current_iy + (exit_faces == 2).long() - (exit_faces == 3).long()
    
    # Check mesh boundaries
    exit_msk = (new_ix < 0) | (new_ix >= num_cells_x) | \
               (new_iy < 0) | (new_iy >= num_cells_y)
    
    # Update exit tallies
    totals_tensor[0] += ((exit_faces == 0) & exit_msk).sum()
    totals_tensor[1] += ((exit_faces == 1) & exit_msk).sum()
    totals_tensor[2] += ((exit_faces == 2) & exit_msk).sum()
    totals_tensor[3] += ((exit_faces == 3) & exit_msk).sum()
    
    # Mark exited particles
    particles[active[exit_msk], 7] = 1
    
    # Update continuing particles
    survivors = ~exit_msk
    if (surv_idx := active[survivors]).numel() > 0:
        particles[surv_idx, 0:2] = torch.stack([new_ix[survivors], new_iy[survivors]], dim=1).float()
        particles[surv_idx, 2] = face_to_face[exit_faces[survivors]].float()
        particles[surv_idx, 3:6] = torch.stack([positions[survivors], velocities_u[survivors], velocities_v[survivors]], dim=1)
    
    return exit_msk


def flush_buffers_if_needed(
    energy_buffer: TallyBuffer,
    track_buffer: TallyBuffer,
    energy_flat: torch.Tensor,
    tracklen_flat: torch.Tensor,
    threshold: int = 8192,
) -> None:
    """Flush tally buffers if they exceed threshold."""
    if energy_buffer.size >= threshold:
        energy_buffer.flush_index_add(energy_flat)
    if track_buffer.size >= threshold:
        track_buffer.flush_index_add(tracklen_flat)


# ───────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────

def plot_map_log(
    data: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    title: str,
    out_dir: os.PathLike | str,
    filename: Optional[str] = None,
    cmap: str = 'jet',
    label: str = 'Value',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay_lines_x: Optional[Sequence[float]] = None,
    overlay_lines_y: Optional[Sequence[float]] = None,
    dpi: int = 300,
    add_title: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    file_format: Optional[str] = None,
) -> Optional[str]:
    X, Y = np.meshgrid(bx, by)
    valid = data[data > 0]
    if valid.size == 0:
        return None
    vmin = vmin if vmin is not None else valid.min()
    vmax = vmax if vmax is not None else float(np.max(data))
    norm = LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, vmin * 1.01))
    plot_data = np.maximum(data, vmin * 0.1)
    plt.figure(figsize=figsize) if figsize else plt.figure()
    im = plt.pcolormesh(X, Y, plot_data, cmap=cmap, norm=norm)
    plt.colorbar(im, label=label)
    plt.xlabel('x (cm)'); plt.ylabel('y (cm)')
    if add_title:
        plt.title(title)
    plt.gca().set_aspect('equal')
    if overlay_lines_x is not None:
        for x in overlay_lines_x:
            plt.axvline(x, color='white', linewidth=0.3, alpha=0.5)
    if overlay_lines_y is not None:
        for y in overlay_lines_y:
            plt.axhline(y, color='white', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # Determine filename and extension
    if filename is None:
        default_ext = f".{file_format}" if file_format else ".png"
        fname = f"{title.lower().replace(' ', '_')}{default_ext}"
    else:
        fname = filename
    fpath = out_dir / fname
    plt.savefig(fpath, dpi=dpi)
    plt.close()
    return str(fpath)


def plot_difference_map(
    a: np.ndarray,
    b: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    title: str,
    out_dir: os.PathLike | str,
    filename: Optional[str] = None,
    dpi: int = 300,
    add_title: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    file_format: Optional[str] = None,
) -> Optional[str]:
    X, Y = np.meshgrid(bx, by)
    max_val = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if max_val <= 0:
        return None
    diff = (a - b) / max_val * 100.0
    vmax = float(np.max(np.abs(diff)))
    plt.figure(figsize=figsize) if figsize else plt.figure()
    im = plt.pcolormesh(X, Y, diff, cmap='jet', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Relative Difference (%)')
    plt.xlabel('x (cm)'); plt.ylabel('y (cm)')
    if add_title:
        plt.title(title)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        default_ext = f".{file_format}" if file_format else ".png"
        fname = f"{title.lower().replace(' ', '_')}_difference{default_ext}"
    else:
        fname = filename
    fpath = out_dir / fname
    plt.savefig(fpath, dpi=dpi)
    plt.close()
    return str(fpath)


def centers_from_boundaries(b: np.ndarray) -> np.ndarray:
    return (b[:-1] + b[1:]) * 0.5


def plot_line_profile(
    arr: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    axis: str,
    idx: int,
    title: str,
    out_dir: os.PathLike | str,
    filename: Optional[str] = None,
    ylog: bool = True,
    dpi: int = 300,
    add_title: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    file_format: Optional[str] = None,
    ylabel: str = 'Value',
) -> Optional[str]:
    cx, cy = centers_from_boundaries(bx), centers_from_boundaries(by)
    plt.figure(figsize=figsize) if figsize else plt.figure()
    if axis == 'x':
        y = arr[idx, :]
        x = cx
        plt.plot(x, y, 'o-', markersize=3)
        plt.xlabel('x (cm)')
    elif axis == 'y':
        y = arr[:, idx]
        x = cy
        plt.plot(x, y, 'o-', markersize=3)
        plt.xlabel('y (cm)')
    else:
        return None
    plt.ylabel(ylabel)
    if ylog:
        plt.semilogy()
    if add_title:
        plt.title(title)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        default_ext = f".{file_format}" if file_format else ".png"
        fname = f"{title.lower().replace(' ', '_')}{default_ext}"
    else:
        fname = filename
    fpath = out_dir / fname
    plt.savefig(fpath, dpi=dpi)
    plt.close()
    return str(fpath)


def plot_line_overlay(
    a: np.ndarray,
    b: Optional[np.ndarray],
    bx: np.ndarray,
    by: np.ndarray,
    axis: str,
    idx: int,
    labels: Tuple[str, str] | Tuple[str, str, str] = ("A", "B"),
    title: str = "Line Profile",
    out_dir: os.PathLike | str = ".",
    filename: Optional[str] = None,
    ylog: bool = True,
    dpi: int = 300,
    c: Optional[np.ndarray] = None,  # Third dataset (e.g., MC reference)
    add_title: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    file_format: Optional[str] = None,
    ylabel: str = 'Value',
) -> Optional[str]:
    """Overlay two or three line profiles on the same plot along axis at index.

    If `b` is None or empty, only plots `a`.
    If `c` is provided, plots it as reference in black.
    """
    cx, cy = centers_from_boundaries(bx), centers_from_boundaries(by)
    if axis == 'x':
        x = cx
        y1 = a[idx, :]
        y2 = b[idx, :] if (b is not None and b.size) else None
        y3 = c[idx, :] if (c is not None and c.size) else None
        xlabel = 'x (cm)'
    elif axis == 'y':
        x = cy
        y1 = a[:, idx]
        y2 = b[:, idx] if (b is not None and b.size) else None
        y3 = c[:, idx] if (c is not None and c.size) else None
        xlabel = 'y (cm)'
    else:
        return None

    plt.figure(figsize=figsize) if figsize else plt.figure()
    if ylog:
        # CFM - blue solid with markers
        plt.semilogy(x, y1, 'b-o', label=labels[0], linewidth=2, markersize=4, markevery=max(1, len(x)//20))
        # MC same count - red dashed with markers
        if y2 is not None:
            plt.semilogy(x, y2, 'r--s', label=labels[1], linewidth=2, markersize=4, markevery=max(1, len(x)//20))
        # MC reference - black solid with markers
        if y3 is not None:
            label3 = labels[2] if len(labels) > 2 else "MC Ref"
            plt.semilogy(x, y3, 'k-^', label=label3, linewidth=1.5, markersize=3, markevery=max(1, len(x)//20))
    else:
        # CFM - blue solid with markers
        plt.plot(x, y1, 'b-o', label=labels[0], linewidth=2, markersize=4, markevery=max(1, len(x)//20))
        # MC same count - red dashed with markers
        if y2 is not None:
            plt.plot(x, y2, 'r--s', label=labels[1], linewidth=2, markersize=4, markevery=max(1, len(x)//20))
        # MC reference - black solid with markers
        if y3 is not None:
            label3 = labels[2] if len(labels) > 2 else "MC Ref"
            plt.plot(x, y3, 'k-^', label=label3, linewidth=1.5, markersize=3, markevery=max(1, len(x)//20))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if add_title:
        plt.title(title)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        default_ext = f".{file_format}" if file_format else ".png"
        fname = f"{title.lower().replace(' ', '_')}{default_ext}"
    else:
        fname = filename
    fpath = out_dir / fname
    plt.savefig(fpath, dpi=dpi)
    plt.close()
    return str(fpath)
