#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CFM multi-cell transport for 2D lattice problem with heterogeneous materials."""

from __future__ import annotations
import os, time, random, logging, sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ───────────────────────────────────────────────────────────────────────
# 0.  External modules
# ───────────────────────────────────────────────────────────────────────

try:
    from flow_matching.solver import ODESolver
except ImportError:
    raise ImportError("flow_matching.solver not found – install package or add to PYTHONPATH")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.model import WrappedModel
    from ml.transform import (
        normalize_width, normalize_height, normalize_position_relative,
        normalize_angle_phi, normalize_radius_r,
        denorm_position_relative, denorm_angle_phi, denorm_radius_r,
        uv_to_polar, polar_to_uv
    )
    from ml.utils import load_model as load_saved_model
except ImportError as e:
    raise ImportError(f"ML utilities missing or outdated: {e}")
from ml.transform import denorm_log_feature, load_normalization_params

MC_AVAILABLE = True
try:
    from physics.mc2d import run_mc_simulation_2d_njit_wrapper, calculate_boundaries_2d
except Exception:
    MC_AVAILABLE = False
    def run_mc_simulation_2d_njit_wrapper(*a, **k): return (np.zeros((1,1)), np.zeros((1,1)), (0,0,0,0,0,0,0,0))
    def calculate_boundaries_2d(*a, **k): return np.array([0.0, 1.0]), np.array([0.0, 1.0])

# ───────────────────────────────────────────────────────────────────────
# 1.  Global parameters & lattice (heterogeneous) config
# ───────────────────────────────────────────────────────────────────────

# Switch between homogeneous and heterogeneous
USE_HETEROGENEOUS = True

# SOURCE CONFIGURATION
SOURCE_MODE = 'internal_start'  # 'boundary_only' or 'internal_start'
RANDOMIZE_START_POS_IN_CELL = True
START_POS_X_REL = 0.5
START_POS_Y_REL = 0.5

# If SOURCE_MODE == 'internal_start' and USE_HETEROGENEOUS is False,
# set these start cell indices (0-based):
START_CELL_X = 30
START_CELL_Y = 30

# Common parameters
HIDDEN_DIM, DEPTH = 256, 6
EPSILON = 1e-9
ENERGY_THRESHOLD = 1e-8
MAX_ITERATIONS = 10000
SEED = 420

# Model configurations
BOUNDARY_MODEL_DIR = "ml/models/2d_boundary"
BOUNDARY_NORM_FILE = "ml/data/norm_params_2d_boundary.npy"
BOUNDARY_INPUT_DIM, BOUNDARY_COND_DIM, BOUNDARY_TARGET_DIM = 10, 5, 5

INTERNAL_MODEL_DIR = "ml/models/2d_internal"
INTERNAL_NORM_FILE = "ml/data/norm_params_2d_internal.npy"
INTERNAL_INPUT_DIM, INTERNAL_COND_DIM, INTERNAL_TARGET_DIM = 11, 6, 5

# Particle counts
NUM_PARTICLES =100000
NUM_PARTICLES_MC = 100000
NUM_PARTICLES_MC_HIGH = 1000000

# ───────────────────────────────────────────────────────────────────────
# 1a. Lattice Configuration (from your snippet)
# ───────────────────────────────────────────────────────────────────────

DOMAIN_SIZE = 7.0  # cm
BASE_CELLS = 7
MESH_REFINEMENT = 16
NUM_CELLS_X = BASE_CELLS * MESH_REFINEMENT
NUM_CELLS_Y = BASE_CELLS * MESH_REFINEMENT
TOTAL_WIDTH = DOMAIN_SIZE
TOTAL_HEIGHT = DOMAIN_SIZE

# Results directory (defined here after MESH_REFINEMENT)
RESULTS_DIR = f"validation/results/lattice_2d_{MESH_REFINEMENT}"

MAT_SCATTERING = 0
MAT_ABSORBING  = 1
MAT_SOURCE     = 2

MATERIAL_PROPERTIES = {
    MAT_SCATTERING: {"sigma_a": 0.0,  "sigma_s": 1.0},   # σ_t = 1.0
    MAT_ABSORBING:  {"sigma_a": 9.5,  "sigma_s": 0.5},   # σ_t = 10.0 
    MAT_SOURCE:     {"sigma_a": 0.0,  "sigma_s": 1.0},   # Treat source region as scattering
}

LATTICE_PATTERN = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 2, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=int)

SOURCE_BASE_CELL_X = 3
SOURCE_BASE_CELL_Y = 3

def create_lattice_geometry():
    material_map = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=int)
    base_cell_size = DOMAIN_SIZE / BASE_CELLS

    def coord_to_idx(coord, total_cells):
        return int(coord * total_cells / DOMAIN_SIZE)

    for base_i in range(BASE_CELLS):
        for base_j in range(BASE_CELLS):
            mat_type = LATTICE_PATTERN[base_i, base_j]
            y_start = base_i * base_cell_size
            y_end   = (base_i + 1) * base_cell_size
            x_start = base_j * base_cell_size
            x_end   = (base_j + 1) * base_cell_size
            iy_start = coord_to_idx(y_start, NUM_CELLS_Y)
            iy_end   = coord_to_idx(y_end,   NUM_CELLS_Y)
            ix_start = coord_to_idx(x_start, NUM_CELLS_X)
            ix_end   = coord_to_idx(x_end,   NUM_CELLS_X)
            for i in range(iy_start, iy_end):
                for j in range(ix_start, ix_end):
                    material_map[i, j] = mat_type
    return material_map

def create_material_arrays():
    material_map = create_lattice_geometry()
    sigma_t_map = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=np.float32)
    sigma_s_map = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=np.float32)
    sigma_a_map = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=np.float32)
    for mat_id, props in MATERIAL_PROPERTIES.items():
        mask = (material_map == mat_id)
        sigma_a_map[mask] = props["sigma_a"]
        sigma_s_map[mask] = props["sigma_s"]
        sigma_t_map[mask] = props["sigma_a"] + props["sigma_s"]
    return material_map, sigma_t_map, sigma_s_map, sigma_a_map

# Build maps
if USE_HETEROGENEOUS:
    MATERIAL_MAP, SIGMA_T_MAP, SIGMA_S_MAP, SIGMA_A_MAP = create_material_arrays()
    SCATTER_RATIO_MAP = np.where(SIGMA_T_MAP > 0, SIGMA_S_MAP / SIGMA_T_MAP, 0.0).astype(np.float32)
    CELL_WIDTHS  = np.full(NUM_CELLS_X, TOTAL_WIDTH  / NUM_CELLS_X, dtype=np.float32)
    CELL_HEIGHTS = np.full(NUM_CELLS_Y, TOTAL_HEIGHT / NUM_CELLS_Y, dtype=np.float32)
    # Source (refined grid center of base source cell)
    START_CELL_X = SOURCE_BASE_CELL_X * MESH_REFINEMENT + MESH_REFINEMENT // 2
    START_CELL_Y = SOURCE_BASE_CELL_Y * MESH_REFINEMENT + MESH_REFINEMENT // 2
    # All refined cells that belong to the source zone (material id == 2)
    SOURCE_REFINED_CELLS = np.argwhere(MATERIAL_MAP == MAT_SOURCE)
else:
    # Homogeneous configuration
    CELL_WIDTHS  = np.full(NUM_CELLS_X, TOTAL_WIDTH  / NUM_CELLS_X, dtype=np.float32)
    CELL_HEIGHTS = np.full(NUM_CELLS_Y, TOTAL_HEIGHT / NUM_CELLS_Y, dtype=np.float32)
    SIGMA_T_MAP  = np.ones((NUM_CELLS_Y, NUM_CELLS_X), dtype=np.float32)
    SIGMA_S_MAP  = np.full((NUM_CELLS_Y, NUM_CELLS_X), 0.578, dtype=np.float32)
    SIGMA_A_MAP  = SIGMA_T_MAP - SIGMA_S_MAP
    MATERIAL_MAP = np.zeros((NUM_CELLS_Y, NUM_CELLS_X), dtype=int)
    SCATTER_RATIO_MAP = SIGMA_S_MAP / SIGMA_T_MAP
    # For homogeneous case, use single source cell at START_CELL_X, START_CELL_Y
    SOURCE_REFINED_CELLS = np.array([[START_CELL_Y, START_CELL_X]], dtype=int)

# Boundaries
def calculate_boundaries_local(widths: np.ndarray, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bx = np.concatenate([[0.], np.cumsum(widths)])
    by = np.concatenate([[0.], np.cumsum(heights)])
    return bx, by

CELL_BOUNDARIES_X, CELL_BOUNDARIES_Y = calculate_boundaries_local(CELL_WIDTHS, CELL_HEIGHTS)

# Cell areas for scalar flux normalization (track-length / area)
CELL_AREAS = np.outer(CELL_HEIGHTS, CELL_WIDTHS)  # shape (Ny,Nx)

# ───────────────────────────────────────────────────────────────────────
# 2.  Device & logging
# ───────────────────────────────────────────────────────────────────────

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
if device == "cuda":
    torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lattice_hetero")
print(f"Using device: {device}")
print(f"Lattice problem - Source mode: {SOURCE_MODE}; Heterogeneous: {USE_HETEROGENEOUS}")
if SOURCE_MODE == 'internal_start':
    num_source_cells = SOURCE_REFINED_CELLS.shape[0]
    print(f"Source region: {num_source_cells} refined cells (base cell: {SOURCE_BASE_CELL_X}, {SOURCE_BASE_CELL_Y})")

# ───────────────────────────────────────────────────────────────────────
# 3.  Face transforms and exit-face helpers
# ───────────────────────────────────────────────────────────────────────

# Physical faces: 0=left, 1=right, 2=top, 3=bottom
_ORIENT_R = np.asarray([
    [[ 1., 0.], [ 0., 1.]],  # left
    [[-1., 0.], [ 0.,-1.]],  # right
    [[ 0.,-1.], [ 1., 0.]],  # top
    [[ 0., 1.], [-1., 0.]],  # bottom
], dtype=np.float32)
_ORIENT_R_TENSOR = torch.from_numpy(_ORIENT_R)
_F2F = torch.tensor([1, 0, 3, 2], dtype=torch.int64)  # face flip map


def classify_face_and_snap(x: Tensor, y: Tensor, W: Tensor, H: Tensor, tol: float = 1.0e-12):
    d_left = torch.abs(x)
    d_right = torch.abs(W - x)
    d_top = torch.abs(H - y)
    d_bottom = torch.abs(y)
    d_stack = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
    face = torch.argmin(d_stack, dim=1)
    zeros = torch.zeros_like(x)
    x_snap = torch.where(face == 0, zeros, torch.where(face == 1, W, x))
    y_snap = torch.where(face == 3, zeros, torch.where(face == 2, H, y))
    x_snap = torch.minimum(torch.maximum(x_snap, zeros), W)
    y_snap = torch.minimum(torch.maximum(y_snap, zeros), H)
    pos_on_face = torch.where((face == 0) | (face == 1), y_snap, x_snap)
    return face, x_snap, y_snap, pos_on_face


def enforce_outward_angle_from_face(face: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    normals = torch.tensor(
        [
            [-1.0, 0.0],  # left
            [1.0, 0.0],   # right
            [0.0, 1.0],   # top
            [0.0, -1.0],  # bottom
        ],
        dtype=torch.float32,
        device=u.device,
    )
    n = normals[face]
    direction = torch.stack([u, v], dim=1)
    dot_product = (direction * n).sum(dim=1)
    needs_flip = dot_product <= 0
    if needs_flip.any():
        reflected = direction - 2.0 * dot_product.unsqueeze(1) * n
        direction = torch.where(needs_flip.unsqueeze(1), reflected, direction)
    return direction[:, 0], direction[:, 1]

# ───────────────────────────────────────────────────────────────────────
# 4.  Load models & normalization parameters
# ───────────────────────────────────────────────────────────────────────

def _load_norm_params_with_fallback(*candidate_paths: str) -> Tuple[dict, str]:
    last_error: Optional[Exception] = None
    for path in candidate_paths:
        try:
            norm_params, resolved_path = load_normalization_params(path)
            return norm_params, resolved_path
        except FileNotFoundError as exc:
            last_error = exc
    raise FileNotFoundError(
        f"Normalization parameters not found. Tried: {candidate_paths}. Last error: {last_error}"
    )

log.info("Loading models...")

boundary_model, BOUNDARY_INPUT_DIM, BOUNDARY_COND_DIM = load_saved_model(
    device=device,
    hidden_dim=HIDDEN_DIM,
    depth=DEPTH,
    expansion_factor=1,
    source_type="boundary",
    data_dim="2d",
)
BOUNDARY_TARGET_DIM = BOUNDARY_INPUT_DIM - BOUNDARY_COND_DIM
boundary_model.eval()
boundary_wrapper = WrappedModel(boundary_model).to(device)
boundary_solver = ODESolver(velocity_model=boundary_wrapper)
log.info("Loaded boundary steady model (input_dim=%d, cond_dim=%d, target_dim=%d)", BOUNDARY_INPUT_DIM, BOUNDARY_COND_DIM, BOUNDARY_TARGET_DIM)

# boundary norm params
bnd_norm, bnd_norm_path = _load_norm_params_with_fallback(
    BOUNDARY_NORM_FILE,
    str(Path(BOUNDARY_MODEL_DIR) / "norm_params_2d_boundary.npy"),
    "data/norm_params_2d_boundary.npy",
)
log.info("Loaded boundary norm params from %s", bnd_norm_path)
width_l_bnd, width_u_bnd = bnd_norm["width_lower"], bnd_norm["width_upper"]
height_l_bnd, height_u_bnd = bnd_norm["height_lower"], bnd_norm["height_upper"]

# internal model if needed
internal_solver = None
if SOURCE_MODE == 'internal_start':
    internal_model, INTERNAL_INPUT_DIM, INTERNAL_COND_DIM = load_saved_model(
        device=device,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        expansion_factor=1,
        source_type="internal",
        data_dim="2d",
    )
    INTERNAL_TARGET_DIM = INTERNAL_INPUT_DIM - INTERNAL_COND_DIM
    internal_model.eval()
    internal_wrapper = WrappedModel(internal_model).to(device)
    internal_solver = ODESolver(velocity_model=internal_wrapper)
    log.info("Loaded internal steady model (input_dim=%d, cond_dim=%d, target_dim=%d)", INTERNAL_INPUT_DIM, INTERNAL_COND_DIM, INTERNAL_TARGET_DIM)
    int_norm, int_norm_path = _load_norm_params_with_fallback(
        INTERNAL_NORM_FILE,
        str(Path(INTERNAL_MODEL_DIR) / "norm_params_2d_internal.npy"),
        "data/norm_params_2d_internal.npy",
    )
    log.info("Loaded internal norm params from %s", int_norm_path)
    width_l_int, width_u_int = int_norm["width_lower"], int_norm["width_upper"]
    height_l_int, height_u_int = int_norm["height_lower"], int_norm["height_upper"]
else:
    int_norm = None
    width_l_int = width_u_int = height_l_int = height_u_int = None

# ───────────────────────────────────────────────────────────────────────
# 4a. MFP Diagnostics - Check if material thicknesses are within training regime
# ───────────────────────────────────────────────────────────────────────

try:
    # Calculate MFP (mean free path) thicknesses for each cell
    # MFP = physical_dimension * sigma_s (collision cross-section)
    W_map = np.tile(CELL_WIDTHS, (NUM_CELLS_Y, 1)).astype(np.float64)
    H_map = np.tile(CELL_HEIGHTS.reshape(NUM_CELLS_Y, 1), (1, NUM_CELLS_X)).astype(np.float64)

    # Use sigma_s for MFP calculation (scattering/collision cross-section)
    W_mfp = W_map * SIGMA_S_MAP.astype(np.float64)
    H_mfp = H_map * SIGMA_S_MAP.astype(np.float64)

    # Get min/max values
    w_mfp_min = float(W_mfp.min())
    w_mfp_max = float(W_mfp.max())
    h_mfp_min = float(H_mfp.min())
    h_mfp_max = float(H_mfp.max())

    # Log10 values for comparison with training bounds
    w_mfp_min_log10 = float(np.log10(max(w_mfp_min, 1e-30)))
    w_mfp_max_log10 = float(np.log10(max(w_mfp_max, 1e-30)))
    h_mfp_min_log10 = float(np.log10(max(h_mfp_min, 1e-30)))
    h_mfp_max_log10 = float(np.log10(max(h_mfp_max, 1e-30)))

    print("\n[Lattice MFP Analysis] Optical thickness (MFP) bounds vs training regime:")
    print(f"  Width_MFP range (linear):   [{w_mfp_min:.6g}, {w_mfp_max:.6g}]")
    print(f"  Width_MFP range (log10):    [{w_mfp_min_log10:.6f}, {w_mfp_max_log10:.6f}]")
    print(f"  Height_MFP range (linear):  [{h_mfp_min:.6g}, {h_mfp_max:.6g}]")
    print(f"  Height_MFP range (log10):   [{h_mfp_min_log10:.6f}, {h_mfp_max_log10:.6f}]")

    # Training bounds from normalization parameters
    print(f"\n  Training regime (boundary model):")
    print(f"  Width bounds (log10):  [{width_l_bnd:.6f}, {width_u_bnd:.6f}] → linear [{10**width_l_bnd:.6g}, {10**width_u_bnd:.6g}]")
    print(f"  Height bounds (log10): [{height_l_bnd:.6f}, {height_u_bnd:.6f}] → linear [{10**height_l_bnd:.6g}, {10**height_u_bnd:.6g}]")

    if SOURCE_MODE == 'internal_start' and width_l_int is not None:
        print(f"\n  Training regime (internal model):")
        print(f"  Width bounds (log10):  [{width_l_int:.6f}, {width_u_int:.6f}] → linear [{10**width_l_int:.6g}, {10**width_u_int:.6g}]")
        print(f"  Height bounds (log10): [{height_l_int:.6f}, {height_u_int:.6f}] → linear [{10**height_l_int:.6g}, {10**height_u_int:.6g}]")

    # Check if values are within training regime
    in_width_regime = (w_mfp_min_log10 >= width_l_bnd and w_mfp_max_log10 <= width_u_bnd)
    in_height_regime = (h_mfp_min_log10 >= height_l_bnd and h_mfp_max_log10 <= height_u_bnd)

    print(f"\n  Status:")
    print(f"  Width MFP within training regime: {'✓ YES' if in_width_regime else '✗ NO'}")
    print(f"  Height MFP within training regime: {'✓ YES' if in_height_regime else '✗ NO'}")

    # Show which materials contribute to min/max
    min_w_idx = np.unravel_index(W_mfp.argmin(), W_mfp.shape)
    max_w_idx = np.unravel_index(W_mfp.argmax(), W_mfp.shape)
    min_h_idx = np.unravel_index(H_mfp.argmin(), H_mfp.shape)
    max_h_idx = np.unravel_index(H_mfp.argmax(), H_mfp.shape)

    mat_names = {0: 'Scattering', 1: 'Absorbing', 2: 'Source'}
    print(f"\n  Material contributions:")
    print(f"  Min width MFP:  {w_mfp_min:.6g} from {mat_names.get(MATERIAL_MAP[min_w_idx], 'Unknown')}")
    print(f"  Max width MFP:  {w_mfp_max:.6g} from {mat_names.get(MATERIAL_MAP[max_w_idx], 'Unknown')}")
    print(f"  Min height MFP: {h_mfp_min:.6g} from {mat_names.get(MATERIAL_MAP[min_h_idx], 'Unknown')}")
    print(f"  Max height MFP: {h_mfp_max:.6g} from {mat_names.get(MATERIAL_MAP[max_h_idx], 'Unknown')}")

except Exception as e:
    print(f"\n[Warning] MFP diagnostics failed: {e}")

# ───────────────────────────────────────────────────────────────────────
# 5.  Core CFM transport simulation (heterogeneous-aware)
# ───────────────────────────────────────────────────────────────────────

_ORIENT_R_TENSOR = _ORIENT_R_TENSOR.to(device)
_F2F = _F2F.to(device)

CELL_WIDTHS_T  = torch.from_numpy(CELL_WIDTHS).to(device)
CELL_HEIGHTS_T = torch.from_numpy(CELL_HEIGHTS).to(device)
SIGMA_T_MAP_T  = torch.from_numpy(SIGMA_T_MAP).to(device)
SIGMA_S_MAP_T  = torch.from_numpy(SIGMA_S_MAP).to(device)
SIGMA_A_MAP_T  = torch.from_numpy(SIGMA_A_MAP).to(device)

def run_cfm_mesh(num_particles: int):
    """Particle transport via Conditional Flow Matching on the lattice problem with heterogeneous materials.
    Returns:
        energy_grid: (Ny,Nx) energy deposition per source particle
        flux_grid:   (Ny,Nx) scalar flux (track-length/area) per source particle
        totals:      dict tallies
        elapsed:     float seconds
    """
    W_cell = CELL_WIDTHS[0]; H_cell = CELL_HEIGHTS[0]
    log.info("CFM lattice sim start (%d particles, %s)", num_particles, SOURCE_MODE)
    wall_start = time.time()

    # Particle state: [cell_x, cell_y, face, pos_on_face, u, v, energy, status]
    part = torch.zeros((num_particles, 8), dtype=torch.float32, device=device)
    part[:, 6] = 1.0  # energy

    # Initialization
    if SOURCE_MODE == 'boundary_only':
        part[:, 0] = 0
        part[:, 1] = torch.from_numpy(np.random.randint(0, NUM_CELLS_Y, num_particles).astype(np.float32)).to(device)
        part[:, 2] = 0
        part[:, 3] = torch.from_numpy(np.random.uniform(0, CELL_HEIGHTS[0], num_particles).astype(np.float32)).to(device)
        # 3D-projected half-space (u>0)
        u_samples = np.zeros(num_particles, dtype=np.float32)
        v_samples = np.zeros(num_particles, dtype=np.float32)
        for i in range(num_particles):
            while True:
                mu = np.random.uniform(-1.0, 1.0)
                phi = np.random.uniform(0.0, 2.0*np.pi)
                t = np.sqrt(1.0 - mu*mu)
                u = t*np.cos(phi); v = t*np.sin(phi)
                if u > 0.0:
                    u_samples[i] = u; v_samples[i] = v; break
        part[:, 4] = torch.from_numpy(u_samples).to(device)
        part[:, 5] = torch.from_numpy(v_samples).to(device)
    else:
        # Internal start: distribute uniformly across all refined source cells
        if SOURCE_REFINED_CELLS.size == 0:
            raise RuntimeError("No source cells found in MATERIAL_MAP (material id == 2)")
        src_cells_np = SOURCE_REFINED_CELLS  # shape (Ns, 2) with (iy, ix)
        src_ix_t = torch.from_numpy(src_cells_np[:, 1].astype(np.int64)).to(device)
        src_iy_t = torch.from_numpy(src_cells_np[:, 0].astype(np.int64)).to(device)
        rand_sel = torch.randint(0, src_cells_np.shape[0], (num_particles,), device=device)
        part[:, 0] = src_ix_t[rand_sel].to(torch.float32)
        part[:, 1] = src_iy_t[rand_sel].to(torch.float32)

    # Accumulators
    totals_tensor = torch.zeros(5, dtype=torch.int64, device=device)
    energy_flat = torch.zeros(NUM_CELLS_X * NUM_CELLS_Y, dtype=torch.float32, device=device)
    track_flat  = torch.zeros(NUM_CELLS_X * NUM_CELLS_Y, dtype=torch.float32, device=device)  # for scalar flux

    # Energy/track buffers
    BUFFER_SIZE = 8192
    buf_indices_E, buf_deposits_E = [], []
    buf_indices_T, buf_tracks_T = [], []
    num_buffered_items_E = 0
    num_buffered_items_T = 0
    def flush_buffers():
        nonlocal num_buffered_items_E, num_buffered_items_T
        if buf_indices_E:
            energy_flat.index_add_(0, torch.cat(buf_indices_E), torch.cat(buf_deposits_E))
            buf_indices_E.clear(); buf_deposits_E.clear(); num_buffered_items_E = 0
        if buf_indices_T:
            track_flat.index_add_(0, torch.cat(buf_indices_T), torch.cat(buf_tracks_T))
            buf_indices_T.clear(); buf_tracks_T.clear(); num_buffered_items_T = 0

    # ODE time grid
    num_ode_steps = 12
    base_grid = torch.linspace(0, 0.8, max(2, num_ode_steps // 2), device=device)
    fine_grid = torch.linspace(0.8, 1.0, max(2, num_ode_steps // 2 + 1), device=device)[1:]
    T_grid = torch.cat([base_grid, fine_grid])
    step_size = T_grid[1] - T_grid[0]

    if device == "cuda":
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)

    inf_time_total = 0.0
    steps_taken = 0
    start_iteration = 0

    # Step 0: internal source propagation inside source cells using INTERNAL model
    if SOURCE_MODE == 'internal_start':
        log.info("Step 0: Internal source within source region (%d source cells)", SOURCE_REFINED_CELLS.shape[0])
        active = (part[:, 7] == 0).nonzero(as_tuple=True)[0]
        batch = active.numel()
        if batch > 0:
            # Get per-particle cell coordinates
            cell_x = part[active, 0].long()
            cell_y = part[active, 1].long()
            # Get per-particle cell dimensions
            W = CELL_WIDTHS_T[cell_x]
            H = CELL_HEIGHTS_T[cell_y]
            # Get per-particle material properties
            sigma_t_source = SIGMA_T_MAP_T[cell_y, cell_x].clamp_min(EPSILON)
            sigma_s_source = SIGMA_S_MAP_T[cell_y, cell_x].clamp_min(EPSILON)
            sigma_a_source = SIGMA_A_MAP_T[cell_y, cell_x]

            if RANDOMIZE_START_POS_IN_CELL:
                x_in_cell = torch.rand(batch, device=device) * W
                y_in_cell = torch.rand(batch, device=device) * H
            else:
                x_in_cell = START_POS_X_REL * W
                y_in_cell = START_POS_Y_REL * H

            # 3D isotropic projected to 2D
            u_samples = np.zeros(batch, dtype=np.float32)
            v_samples = np.zeros(batch, dtype=np.float32)
            for i in range(batch):
                mu = np.random.uniform(-1.0, 1.0)
                phi_angle = np.random.uniform(0.0, 2.0*np.pi)
                t = np.sqrt(1.0 - mu*mu)
                u_samples[i] = t * np.cos(phi_angle)
                v_samples[i] = t * np.sin(phi_angle)
            phi_in, r_in = uv_to_polar(u_samples, v_samples)
            phi_in_t = torch.from_numpy(phi_in).to(device)
            r_in_t   = torch.from_numpy(r_in).to(device)

            # Normalize (optical units use local sigma_s)
            W_opt = W * sigma_s_source
            H_opt = H * sigma_s_source
            cond = torch.stack([
                normalize_width(W_opt.expand(batch),  width_l_int,   width_u_int),
                normalize_height(H_opt.expand(batch), height_l_int,  height_u_int),
                normalize_position_relative(x_in_cell / W),
                normalize_position_relative(y_in_cell / H),
                normalize_angle_phi(phi_in_t),
                normalize_radius_r(r_in_t)
            ], dim=1)

            noise = torch.randn(batch, INTERNAL_TARGET_DIM, device=device)
            x0 = torch.cat([cond, noise], dim=1)

            # Inference
            if device == "cuda":
                evt_start.record()
            else:
                wall_inf_start = time.time()
            with torch.no_grad():
                out = internal_solver.sample(time_grid=T_grid, x_init=x0,
                                             method='rk4', step_size=step_size,
                                             return_intermediates=False)
            if device == "cuda":
                evt_end.record(); torch.cuda.synchronize()
                inf_time = evt_start.elapsed_time(evt_end) / 1_000.0
            else:
                inf_time = time.time() - wall_inf_start
            inf_time_total += inf_time

            # Outputs: [x_out, y_out, phi_out, r_out, time]
            pred = out[:, INTERNAL_COND_DIM:]
            x_out = denorm_position_relative(pred[:, 0]) * W
            y_out = denorm_position_relative(pred[:, 1]) * H
            pred_phi = denorm_angle_phi(pred[:, 2])
            pred_r = denorm_radius_r(pred[:, 3])
            dt = denorm_log_feature(pred[:, 4], int_norm, "time").clamp_min(0.)

            u_out, v_out = polar_to_uv(pred_phi, pred_r)
            exit_face, x_out, y_out, pos_on_face = classify_face_and_snap(x_out, y_out, W, H)
            u_out, v_out = enforce_outward_angle_from_face(exit_face, u_out, v_out)

            # Physical time uses local sigma_s
            dt_phys = dt / sigma_s_source

            # Track-length (scalar flux numerator): correct for absorption along path
            E_before = part[active, 6]
            # Integrate weight along path: ∫ E₀·exp(-σₐ·s) ds = E₀·(1-exp(-σₐ·L))/σₐ
            # Use hybrid approach: simple formula when optical depth is small (numerical stability)
            optical_depth = sigma_a_source * dt_phys
            track_add = torch.where(
                optical_depth > 0.01,
                E_before * (1.0 - torch.exp(-optical_depth)) / sigma_a_source,
                E_before * dt_phys  # optically thin: use simple formula
            )
            # Per-particle linear indices
            lin_idx = cell_y * NUM_CELLS_X + cell_x
            buf_indices_T.append(lin_idx)
            buf_tracks_T.append(track_add)
            num_buffered_items_T += track_add.numel()

            # Absorption with local sigma_a
            deposit = E_before * (1 - torch.exp(-sigma_a_source * dt_phys))
            # Energy deposition
            buf_indices_E.append(lin_idx)
            buf_deposits_E.append(deposit)
            num_buffered_items_E += deposit.numel()

            part[active, 6] = E_before - deposit

            absorbed = (part[active, 6] < ENERGY_THRESHOLD)
            if absorbed.any():
                totals_tensor[4] += absorbed.sum()
                part[active[absorbed], 7] = 2

            # Determine exit
            survivors = ~absorbed
            if survivors.any():
                surv_idx = active[survivors]
                surv_cell_x = cell_x[survivors]
                surv_cell_y = cell_y[survivors]
                exit_face = exit_face[survivors]
                pos_on_face = pos_on_face[survivors]

                part[surv_idx, 2] = exit_face.float()
                part[surv_idx, 3] = pos_on_face
                part[surv_idx, 4] = u_out[survivors]
                part[surv_idx, 5] = v_out[survivors]

                new_cell_x = surv_cell_x + (exit_face == 1).long() - (exit_face == 0).long()
                new_cell_y = surv_cell_y + (exit_face == 2).long() - (exit_face == 3).long()
                mesh_exit = (new_cell_x < 0) | (new_cell_x >= NUM_CELLS_X) | (new_cell_y < 0) | (new_cell_y >= NUM_CELLS_Y)

                totals_tensor[0] += ((exit_face == 0) & mesh_exit).sum()
                totals_tensor[1] += ((exit_face == 1) & mesh_exit).sum()
                totals_tensor[2] += ((exit_face == 2) & mesh_exit).sum()
                totals_tensor[3] += ((exit_face == 3) & mesh_exit).sum()
                part[surv_idx[mesh_exit], 7] = 1

                continuing = surv_idx[~mesh_exit]
                if continuing.numel() > 0:
                    part[continuing, 0] = new_cell_x[~mesh_exit].float()
                    part[continuing, 1] = new_cell_y[~mesh_exit].float()
                    part[continuing, 2] = _F2F[exit_face[~mesh_exit]].float()

        steps_taken += 1
        start_iteration = 1
        log.info("Step 0 complete. Active particles: %d", int((part[:,7]==0).sum().item()))

    # Main loop (boundary model for cell-to-cell propagation)
    for it in range(start_iteration, MAX_ITERATIONS):
        iter_start_time = time.time()
        active = (part[:, 7] == 0).nonzero(as_tuple=True)[0]
        batch = active.numel()
        if batch == 0: break

        p = part[active]
        ix, iy = p[:, 0].long(), p[:, 1].long()
        face = p[:, 2].long()
        pos  = p[:, 3]
        u, v, E = p[:, 4], p[:, 5], p[:, 6]
        W = CELL_WIDTHS_T[ix]; H = CELL_HEIGHTS_T[iy]

        # Rotate into canonical frame
        zeros_pos = torch.zeros_like(pos)
        px = torch.where(face == 0, zeros_pos,
             torch.where(face == 1, W,
             torch.where(face == 2, pos, pos)))
        py = torch.where(face == 0, pos,
             torch.where(face == 1, pos,
             torch.where(face == 2, H, zeros_pos)))
        pos_phys = torch.stack([px, py], dim=1)

        tx = torch.where(face == 0, torch.zeros_like(W),
             torch.where(face == 1, W,
             torch.where(face == 2, H, torch.zeros_like(W))))
        ty = torch.where(face == 1, H,
             torch.where(face == 3, W,
             torch.zeros_like(W)))
        t_vecs = torch.stack([tx, ty], dim=1)

        Rs = _ORIENT_R_TENSOR[face]
        dir_phys = torch.stack([u, v], dim=1).unsqueeze(-1)
        pos_model = torch.bmm(Rs, pos_phys.unsqueeze(-1)).squeeze(-1) + t_vecs
        dir_model = torch.bmm(Rs, dir_phys).squeeze(-1)
        ux_m, uy_m = dir_model[:, 0], dir_model[:, 1]

        is_lr = face < 2
        Wm, Hm = torch.where(is_lr, W, H), torch.where(is_lr, H, W)

        yn = torch.clamp(pos_model[:, 1] / (Hm + EPSILON), 0.0, 1.0)

        # Local (heterogeneous) optical scaling
        sigma_t_local = SIGMA_T_MAP_T[iy, ix].clamp_min(EPSILON)
        sigma_s_local = SIGMA_S_MAP_T[iy, ix].clamp_min(EPSILON)
        sigma_a_local = SIGMA_A_MAP_T[iy, ix].clamp_min(0.0)
        Wm_optical = Wm * sigma_s_local
        Hm_optical = Hm * sigma_s_local

        # Direction to polar
        ux_m_np = ux_m.detach().cpu().numpy()
        uy_m_np = uy_m.detach().cpu().numpy()
        phi_in, r_in = uv_to_polar(ux_m_np, uy_m_np)
        phi_in_t = torch.from_numpy(phi_in).to(device)
        r_in_t   = torch.from_numpy(r_in).to(device)

        cond = torch.stack([
            normalize_width(Wm_optical,  width_l_bnd,  width_u_bnd),
            normalize_height(Hm_optical, height_l_bnd, height_u_bnd),
            normalize_position_relative(yn),
            normalize_angle_phi(phi_in_t),
            normalize_radius_r(r_in_t)
        ], dim=1)

        noise = torch.randn(batch, BOUNDARY_TARGET_DIM, device=device)
        x0 = torch.cat([cond, noise], dim=1)

        # Inference
        if device == "cuda": evt_start.record()
        else: wall_inf_start = time.time()
        with torch.no_grad():
            out = boundary_solver.sample(time_grid=T_grid, x_init=x0,
                                         method='rk4', step_size=step_size,
                                         return_intermediates=False)
        if device == "cuda":
            evt_end.record(); torch.cuda.synchronize()
            inf_time = evt_start.elapsed_time(evt_end) / 1_000.0
        else:
            inf_time = time.time() - wall_inf_start
        inf_time_total += inf_time

        pred = out[:, BOUNDARY_COND_DIM:]
        x_rel_pred = denorm_position_relative(pred[:, 0])
        y_rel_pred = denorm_position_relative(pred[:, 1])
        pred_phi = denorm_angle_phi(pred[:, 2])
        pred_r   = denorm_radius_r(pred[:, 3])
        dt       = denorm_log_feature(pred[:, 4], bnd_norm, "time").clamp_min(1e-9)

        x_m = x_rel_pred * Wm
        y_m = y_rel_pred * Hm
        exit_face_model, x_m, y_m, _ = classify_face_and_snap(x_m, y_m, Wm, Hm)

        ux_n, uy_n = polar_to_uv(pred_phi, pred_r)
        ux_n, uy_n = enforce_outward_angle_from_face(exit_face_model, ux_n, uy_n)

        # Convert time back to physical units with local sigma_s
        dt_phys = dt / sigma_s_local

        # Track-length (scalar flux numerator): correct for absorption along path
        lin_idx = iy * NUM_CELLS_X + ix
        # Integrate weight along path: ∫ E₀·exp(-σₐ·s) ds = E₀·(1-exp(-σₐ·L))/σₐ
        # Use hybrid approach: simple formula when optical depth is small (numerical stability)
        optical_depth = sigma_a_local * dt_phys
        track_add = torch.where(
            optical_depth > 0.01,
            E * (1.0 - torch.exp(-optical_depth)) / sigma_a_local,
            E * dt_phys  # optically thin: use simple formula
        )
        buf_indices_T.append(lin_idx)
        buf_tracks_T.append(track_add)
        num_buffered_items_T += track_add.numel()

        # Energy deposition with local sigma_a
        deposit = E * (1 - torch.exp(-sigma_a_local * dt_phys))
        buf_indices_E.append(lin_idx)
        buf_deposits_E.append(deposit)
        num_buffered_items_E += deposit.numel()
        part[active, 6] -= deposit

        Rs_T = Rs.transpose(1, 2)
        pos_phys_exit = torch.bmm(Rs_T, (torch.stack([x_m, y_m], dim=1) - t_vecs).unsqueeze(-1)).squeeze(-1)
        dir_phys_exit = torch.bmm(Rs_T, torch.stack([ux_n, uy_n], dim=1).unsqueeze(-1)).squeeze(-1)
        x_p, y_p = pos_phys_exit[:, 0], pos_phys_exit[:, 1]
        u_p, v_p = dir_phys_exit[:, 0], dir_phys_exit[:, 1]

        x_pc = torch.minimum(torch.maximum(x_p, torch.zeros_like(W)), W)
        y_pc = torch.minimum(torch.maximum(y_p, torch.zeros_like(H)), H)
        b, x_pc, y_pc, s = classify_face_and_snap(x_pc, y_pc, W, H)

        new_ix = ix + (b == 1).long() - (b == 0).long()
        new_iy = iy + (b == 2).long() - (b == 3).long()
        exit_msk = (new_ix < 0) | (new_ix >= NUM_CELLS_X) | (new_iy < 0) | (new_iy >= NUM_CELLS_Y)

        totals_tensor[0] += torch.count_nonzero((b == 0) & exit_msk)
        totals_tensor[1] += torch.count_nonzero((b == 1) & exit_msk)
        totals_tensor[2] += torch.count_nonzero((b == 2) & exit_msk)
        totals_tensor[3] += torch.count_nonzero((b == 3) & exit_msk)
        part[active[exit_msk], 7] = 1

        survivors = ~exit_msk
        if (surv_idx := active[survivors]).numel() > 0:
            part[surv_idx, 0:2] = torch.stack([new_ix[survivors], new_iy[survivors]], dim=1).float()
            part[surv_idx, 2] = _F2F[b[survivors]].float()
            part[surv_idx, 3:6] = torch.stack([s[survivors], u_p[survivors], v_p[survivors]], dim=1)

        # Kill if under threshold
        if (abs_msk := (part[:, 6] < ENERGY_THRESHOLD) & (part[:, 7] == 0)).any():
            totals_tensor[4] += torch.count_nonzero(abs_msk)
            part[abs_msk, 7] = 2

        # Flush if needed
        if (num_buffered_items_E + num_buffered_items_T) >= BUFFER_SIZE:
            flush_buffers()

        steps_taken += 1
        iter_wall = time.time() - iter_start_time
        if it % 10 == 0 or batch == 0:
            log.info("iter %d – active %d – time %.4fs – inference %.4fs",
                     it, batch, iter_wall, inf_time)

    flush_buffers()

    elapsed = time.time() - wall_start
    energy_grid = energy_flat.view(NUM_CELLS_Y, NUM_CELLS_X).detach().cpu().numpy() / num_particles
    track_grid  = track_flat.view(NUM_CELLS_Y, NUM_CELLS_X).detach().cpu().numpy() / num_particles
    # Scalar flux = track-length / cell area
    flux_grid   = np.divide(track_grid, CELL_AREAS, out=np.zeros_like(track_grid), where=(CELL_AREAS>0))

    totals = {
        "left": int(totals_tensor[0].item()),
        "right": int(totals_tensor[1].item()),
        "top": int(totals_tensor[2].item()),
        "bottom": int(totals_tensor[3].item()),
        "absorb": int(totals_tensor[4].item()),
    }

    log.info("CFM time %.2fs – inference avg %.4fs (%.1f%% of total)",
             elapsed,
             inf_time_total / max(1, steps_taken),
             100.0 * inf_time_total / max(1e-9, elapsed))
    return energy_grid, flux_grid, totals, elapsed

# ───────────────────────────────────────────────────────────────────────
# 6.  Plotting helpers
# ───────────────────────────────────────────────────────────────────────

def _plot_maps(grids_to_plot: Dict[str, np.ndarray], bx: np.ndarray, by: np.ndarray,
               vmin: Optional[float] = None, vmax: Optional[float] = None, label: str = 'Energy'):
    X, Y = np.meshgrid(bx, by)
    for name, arr in grids_to_plot.items():
        if arr.size == 0 or np.all(arr <= 0):
            log.warning(f"Grid '{name}' contains no positive values. Skipping map plot."); continue
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "p")
        if vmin is None or vmax is None:
            valid = arr[arr > 0]
            data_vmin, data_vmax = (valid.min(), arr.max()) if valid.size > 0 else (1e-12, 1.0)
        else:
            data_vmin, data_vmax = vmin, vmax
        norm = LogNorm(vmin=data_vmin, vmax=data_vmax)
        plot_data = np.maximum(arr, data_vmin)
        plt.figure()
        plt.pcolormesh(X, Y, plot_data, cmap='jet', norm=norm)
        plt.colorbar(label=label)
        plt.gca().set_aspect('equal')
        plt.title(f"{name} - Lattice ({SOURCE_MODE}, hetero={USE_HETEROGENEOUS})")
        plt.xlabel('x (cm)'); plt.ylabel('y (cm)')
        plt.tight_layout()
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(RESULTS_DIR) / f"{safe_name}.png", dpi=300); plt.close()

def _plot_absolute_errors(ref_grid, grid_A, grid_B, bx, by, labelA: str, labelB: str, basenameA: str, basenameB: str):
    """Absolute error maps of A and B against reference with consistent scale"""
    X, Y = np.meshgrid(bx, by)

    # Calculate absolute errors
    err_A = np.abs(grid_A - ref_grid) if grid_A.size else None
    err_B = np.abs(grid_B - ref_grid) if grid_B.size else None

    # Determine consistent scale: vmin = 1e-8, vmax = max MC error
    vmin = 1e-8
    vmax = None
    if err_B is not None and np.any(err_B > 0):
        vmax = float(np.max(err_B))
    elif err_A is not None and np.any(err_A > 0):
        vmax = float(np.max(err_A))

    if vmax is None or vmax <= vmin:
        log.warning("No valid errors to plot"); return

    def _plot_err(label, err, fname):
        if err is None:
            return
        valid = err[err > 0]
        if valid.size == 0:
            log.warning(f"No positive errors for {label}"); return
        norm = LogNorm(vmin=vmin, vmax=vmax)
        plot_data = np.maximum(err, vmin)
        plt.figure()
        plt.pcolormesh(X, Y, plot_data, cmap='jet', norm=norm)
        plt.colorbar(label='Absolute Error (log scale)')
        plt.gca().set_aspect('equal')
        plt.title(f'{label} - Lattice ({SOURCE_MODE})')
        plt.xlabel('x (cm)'); plt.ylabel('y (cm)')
        plt.tight_layout()
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(RESULTS_DIR) / fname, dpi=300); plt.close()

    if ref_grid.size and np.any(ref_grid > 0):
        _plot_err(f"{labelA} Abs Error vs Ref", err_A,
                  f'{basenameA}_absolute_error_{SOURCE_MODE}.png')
        _plot_err(f"{labelB} Abs Error vs Ref", err_B,
                  f'{basenameB}_absolute_error_{SOURCE_MODE}.png')

def _plot_line_comparison(cfm_grid, mc_low_grid, mc_ref_grid, bx, quantity_label="Energy"):
    if cfm_grid.shape[0] < 2: return
    if SOURCE_MODE == 'internal_start':
        slice_y = min(START_CELL_Y, cfm_grid.shape[0] - 1)
        slice_label = f"Y={slice_y} (source row)"
    else:
        slice_y = cfm_grid.shape[0] // 2
        slice_label = f"Y={slice_y} (middle)"
    cell_centers_x = (bx[:-1] + bx[1:]) / 2
    plt.figure(figsize=(12,6))
    all_data = []
    cfm_data = cfm_grid[slice_y, :]
    mc_low_data = mc_low_grid[slice_y, :] if mc_low_grid.size else None
    mc_ref_data = mc_ref_grid[slice_y, :] if mc_ref_grid.size else None
    if mc_ref_data is not None:
        plt.semilogy(cell_centers_x, mc_ref_data, 'k-', label=f'MC Ref ({NUM_PARTICLES_MC_HIGH/1e6:.1f}M)', linewidth=2.5, alpha=0.9)
        if np.any(mc_ref_data > 0): all_data.append(mc_ref_data[mc_ref_data > 0])
    plt.semilogy(cell_centers_x, cfm_data, 'r-o', label=f'CFM ({NUM_PARTICLES/1e3:.0f}k)', markersize=4, alpha=0.8)
    if np.any(cfm_data > 0): all_data.append(cfm_data[cfm_data > 0])
    if mc_low_data is not None:
        plt.semilogy(cell_centers_x, mc_low_data, 'b--x', label=f'MC ({NUM_PARTICLES_MC/1e3:.0f}k)', markersize=5, alpha=0.8)
        if np.any(mc_low_data > 0): all_data.append(mc_low_data[mc_low_data > 0])
    plt.title(f'{quantity_label} Comparison - Lattice {slice_label} ({SOURCE_MODE}, hetero={USE_HETEROGENEOUS})')
    plt.xlabel('x (cm)'); plt.ylabel(quantity_label)
    plt.legend(fontsize=10); plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlim(bx[0], bx[-1])
    if all_data:
        all_valid = np.concatenate(all_data)
        plt.ylim(bottom=all_valid.min() * 0.5, top=all_valid.max() * 2)
    plt.tight_layout()
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS_DIR) / f'{quantity_label.lower()}_comparison_{NUM_PARTICLES}.png', dpi=300)
    plt.close()

# ───────────────────────────────────────────────────────────────────────
# 7.  Main
# ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Run CFM
    cfm_energy, cfm_flux, cfm_totals, cfm_t = run_cfm_mesh(NUM_PARTICLES)
    cfm_total_energy = float(np.sum(cfm_energy))
    print("\n" + "="*60)
    print(f"DIAGNOSTIC: CFM Total Energy Deposition = {cfm_total_energy:.8f}")
    print("="*60)

    print(f"\nCFM Boundary Statistics ({SOURCE_MODE}):")
    total_tracked_cfm = sum(cfm_totals.values())
    for k, v in cfm_totals.items():
        pct = (v / max(1, total_tracked_cfm)) * 100.0
        print(f"  - {k:<10s}: {v:>8d} ({pct:6.2f}%)")
    print(f"  {'-'*45}")
    print(f"  Total tracked: {total_tracked_cfm} / {NUM_PARTICLES}")

    # 2) MC runs
    mc_low_energy = np.zeros_like(cfm_energy)
    mc_low_flux = np.zeros_like(cfm_flux)
    mc_ref_energy = np.zeros_like(cfm_energy)
    mc_ref_flux = np.zeros_like(cfm_flux)
    mc_low_t = 0.0
    mc_ref_t = 0.0

    if MC_AVAILABLE:
        mc_cell_widths = CELL_WIDTHS.astype(np.float64)
        mc_cell_heights = CELL_HEIGHTS.astype(np.float64)
        mc_bx, mc_by = calculate_boundaries_2d(mc_cell_widths, mc_cell_heights)

        # Low-fidelity MC (heterogeneous): get both energy and track-length
        print(f"\nRunning low-quality MC comparison for lattice ({NUM_PARTICLES_MC:,} particles, hetero={USE_HETEROGENEOUS})…")
        mc_low_start = time.time()

        if SOURCE_MODE == 'internal_start':
            # Distribute MC particles uniformly across ALL refined source cells
            # The MC wrapper accepts a single start cell per call, so we run it multiple times
            src_cells = SOURCE_REFINED_CELLS  # (Ns, 2) with (iy, ix)
            num_src = max(1, src_cells.shape[0])
            base_count = NUM_PARTICLES_MC // num_src
            remainder = NUM_PARTICLES_MC % num_src

            energy_accum = np.zeros_like(SIGMA_T_MAP, dtype=np.float64)
            tracklen_accum = np.zeros_like(SIGMA_T_MAP, dtype=np.float64)
            stats_accum = [0, 0, 0, 0, 0, 0, 0, 0]  # [left, right, bottom, top, absorbed, steps, reflections, warnings]

            for k in range(num_src):
                n_k = base_count + (1 if k < remainder else 0)
                if n_k == 0:
                    continue
                iy_k, ix_k = int(src_cells[k, 0]), int(src_cells[k, 1])

                res = run_mc_simulation_2d_njit_wrapper(
                    source_mode='internal_start',
                    num_cells_x=NUM_CELLS_X, num_cells_y=NUM_CELLS_Y,
                    cell_widths=mc_cell_widths, cell_heights=mc_cell_heights,
                    cell_boundaries_x=mc_bx, cell_boundaries_y=mc_by,
                    total_width=TOTAL_WIDTH, total_height=TOTAL_HEIGHT,
                    num_particles=n_k, seed=np.random.randint(0, 2**31 - 1),
                    reflected_boundaries=False, energy_threshold=ENERGY_THRESHOLD,
                    scatter_ratio=None, angular_sampling=1,
                    sigma_t_per_cell=SIGMA_T_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                    sigma_a_per_cell=SIGMA_A_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                    start_cell_indices=[ix_k, iy_k],
                    start_position_relative=[START_POS_X_REL, START_POS_Y_REL],
                    randomize_position_in_cell=RANDOMIZE_START_POS_IN_CELL,
                    verbose=False
                )

                if isinstance(res, tuple) and len(res) >= 3:
                    e_k, tl_k, stats_k = res[0], res[1], res[2]
                    # Accumulate statistics
                    for i in range(8):
                        stats_accum[i] += stats_k[i]
                elif isinstance(res, tuple) and len(res) >= 2:
                    e_k, tl_k = res[0], res[1]
                else:
                    e_k = res
                    tl_k = np.zeros_like(e_k)

                # Weight by particle fraction
                w = float(n_k) / float(NUM_PARTICLES_MC)
                energy_accum += w * e_k
                tracklen_accum += w * tl_k

            mc_low_energy = energy_accum
            mc_low_track = tracklen_accum
        else:
            # Boundary source mode
            mc_low_energy, mc_low_track, _ = run_mc_simulation_2d_njit_wrapper(
                source_mode='boundary_only',
                num_cells_x=NUM_CELLS_X, num_cells_y=NUM_CELLS_Y,
                cell_widths=mc_cell_widths, cell_heights=mc_cell_heights,
                cell_boundaries_x=mc_bx, cell_boundaries_y=mc_by,
                total_width=TOTAL_WIDTH, total_height=TOTAL_HEIGHT,
                num_particles=NUM_PARTICLES_MC, seed=np.random.randint(0, 2**31 - 1),
                reflected_boundaries=False, energy_threshold=ENERGY_THRESHOLD,
                scatter_ratio=None, angular_sampling=1,
                sigma_t_per_cell=SIGMA_T_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                sigma_a_per_cell=SIGMA_A_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                start_cell_indices=None,
                start_position_relative=None,
                randomize_position_in_cell=False,
                verbose=True
            )

        mc_low_t = time.time() - mc_low_start
        print(f"MC (low) time {mc_low_t:.2f}s")

        # Print MC (low) boundary statistics for internal_start mode
        if SOURCE_MODE == 'internal_start':
            print(f"\nMC (low) Boundary Statistics ({SOURCE_MODE}):")
            exited_left, exited_right, exited_bottom, exited_top, absorbed, travel_steps, boundary_reflections, warning_flags = stats_accum
            total = exited_left + exited_right + exited_bottom + exited_top + absorbed
            print(f"  - left      : {exited_left:8d} ({exited_left/NUM_PARTICLES_MC:6.2%})")
            print(f"  - right     : {exited_right:8d} ({exited_right/NUM_PARTICLES_MC:6.2%})")
            print(f"  - top       : {exited_top:8d} ({exited_top/NUM_PARTICLES_MC:6.2%})")
            print(f"  - bottom    : {exited_bottom:8d} ({exited_bottom/NUM_PARTICLES_MC:6.2%})")
            print(f"  - absorb    : {absorbed:8d} ({absorbed/NUM_PARTICLES_MC:6.2%})")
            print(f"  ---------------------------------------------")
            print(f"  Total tracked: {total} / {NUM_PARTICLES_MC}")

        # Scalar flux: track-length per particle divided by cell area
        mc_low_flux = np.divide(mc_low_track, CELL_AREAS, out=np.zeros_like(mc_low_track), where=(CELL_AREAS>0))

        # High-fidelity MC ref - Check if saved reference exists
        mc_ref_file = Path(RESULTS_DIR) / f'mc_ref_data_{NUM_PARTICLES_MC_HIGH}.npz'
        mc_ref_loaded = False
        if mc_ref_file.exists():
            print(f"\nLoading existing MC reference from {mc_ref_file}...")
            mc_ref_data = np.load(mc_ref_file)
            # Verify that the saved reference matches current configuration
            if (mc_ref_data['num_particles'] == NUM_PARTICLES_MC_HIGH and
                mc_ref_data['num_cells_x'] == NUM_CELLS_X and
                mc_ref_data['num_cells_y'] == NUM_CELLS_Y and
                mc_ref_data['mesh_refinement'] == MESH_REFINEMENT and
                str(mc_ref_data['source_mode']) == SOURCE_MODE):
                mc_ref_energy = mc_ref_data['mc_ref_energy']
                mc_ref_track = mc_ref_data['mc_ref_track']
                mc_ref_flux = mc_ref_data['mc_ref_flux']
                mc_ref_t = float(mc_ref_data['mc_ref_time'])
                mc_ref_loaded = True
                print(f"MC reference loaded successfully (computed with {NUM_PARTICLES_MC_HIGH:,} particles)")
            else:
                print("WARNING: Saved MC reference has different configuration. Recomputing...")

        # DISABLED: Reference solution generation
        if not mc_ref_loaded:
            print(f"\nRunning high-quality MC reference for lattice ({NUM_PARTICLES_MC_HIGH:,} particles)…")
            mc_ref_start = time.time()

            if SOURCE_MODE == 'internal_start':
                # Distribute MC particles uniformly across ALL refined source cells
                src_cells = SOURCE_REFINED_CELLS  # (Ns, 2) with (iy, ix)
                num_src = max(1, src_cells.shape[0])
                base_count = NUM_PARTICLES_MC_HIGH // num_src
                remainder = NUM_PARTICLES_MC_HIGH % num_src

                energy_accum = np.zeros_like(SIGMA_T_MAP, dtype=np.float64)
                tracklen_accum = np.zeros_like(SIGMA_T_MAP, dtype=np.float64)
                stats_accum = [0, 0, 0, 0, 0, 0, 0, 0]  # [left, right, bottom, top, absorbed, steps, reflections, warnings]

                for k in range(num_src):
                    n_k = base_count + (1 if k < remainder else 0)
                    if n_k == 0:
                        continue
                    iy_k, ix_k = int(src_cells[k, 0]), int(src_cells[k, 1])

                    res = run_mc_simulation_2d_njit_wrapper(
                        source_mode='internal_start',
                        num_cells_x=NUM_CELLS_X, num_cells_y=NUM_CELLS_Y,
                        cell_widths=mc_cell_widths, cell_heights=mc_cell_heights,
                        cell_boundaries_x=mc_bx, cell_boundaries_y=mc_by,
                        total_width=TOTAL_WIDTH, total_height=TOTAL_HEIGHT,
                        num_particles=n_k, seed=np.random.randint(0, 2**31 - 1),
                        reflected_boundaries=False, energy_threshold=ENERGY_THRESHOLD,
                        scatter_ratio=None, angular_sampling=1,
                        sigma_t_per_cell=SIGMA_T_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                        sigma_a_per_cell=SIGMA_A_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                        start_cell_indices=[ix_k, iy_k],
                        start_position_relative=[START_POS_X_REL, START_POS_Y_REL],
                        randomize_position_in_cell=RANDOMIZE_START_POS_IN_CELL,
                        verbose=False
                    )

                    if isinstance(res, tuple) and len(res) >= 3:
                        e_k, tl_k, stats_k = res[0], res[1], res[2]
                        # Accumulate statistics
                        for i in range(8):
                            stats_accum[i] += stats_k[i]
                    elif isinstance(res, tuple) and len(res) >= 2:
                        e_k, tl_k = res[0], res[1]
                    else:
                        e_k = res
                        tl_k = np.zeros_like(e_k)

                    # Weight by particle fraction
                    w = float(n_k) / float(NUM_PARTICLES_MC_HIGH)
                    energy_accum += w * e_k
                    tracklen_accum += w * tl_k

                mc_ref_energy = energy_accum
                mc_ref_track = tracklen_accum
            else:
                # Boundary source mode
                mc_ref_energy, mc_ref_track, _ = run_mc_simulation_2d_njit_wrapper(
                    source_mode='boundary_only',
                    num_cells_x=NUM_CELLS_X, num_cells_y=NUM_CELLS_Y,
                    cell_widths=mc_cell_widths, cell_heights=mc_cell_heights,
                    cell_boundaries_x=mc_bx, cell_boundaries_y=mc_by,
                    total_width=TOTAL_WIDTH, total_height=TOTAL_HEIGHT,
                    num_particles=NUM_PARTICLES_MC_HIGH, seed=np.random.randint(0, 2**31 - 1),
                    reflected_boundaries=False, energy_threshold=ENERGY_THRESHOLD,
                    scatter_ratio=None, angular_sampling=1,
                    sigma_t_per_cell=SIGMA_T_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                    sigma_a_per_cell=SIGMA_A_MAP.astype(np.float64) if USE_HETEROGENEOUS else None,
                    start_cell_indices=None,
                    start_position_relative=None,
                    randomize_position_in_cell=False,
                    verbose=True
                )

            mc_ref_t = time.time() - mc_ref_start
            print(f"MC (ref) time {mc_ref_t:.2f}s vs CFM time {cfm_t:.2f}s → Speed‑up: {mc_ref_t / max(1e-9, cfm_t):.1f}×")

            # Print MC (ref) boundary statistics for internal_start mode
            if SOURCE_MODE == 'internal_start':
                print(f"\nMC (ref) Boundary Statistics ({SOURCE_MODE}):")
                exited_left, exited_right, exited_bottom, exited_top, absorbed, travel_steps, boundary_reflections, warning_flags = stats_accum
                total = exited_left + exited_right + exited_bottom + exited_top + absorbed
                print(f"  - left      : {exited_left:8d} ({exited_left/NUM_PARTICLES_MC_HIGH:6.2%})")
                print(f"  - right     : {exited_right:8d} ({exited_right/NUM_PARTICLES_MC_HIGH:6.2%})")
                print(f"  - top       : {exited_top:8d} ({exited_top/NUM_PARTICLES_MC_HIGH:6.2%})")
                print(f"  - bottom    : {exited_bottom:8d} ({exited_bottom/NUM_PARTICLES_MC_HIGH:6.2%})")
                print(f"  - absorb    : {absorbed:8d} ({absorbed/NUM_PARTICLES_MC_HIGH:6.2%})")
                print(f"  ---------------------------------------------")
                print(f"  Total tracked: {total} / {NUM_PARTICLES_MC_HIGH}")

            # Compute flux before saving
            mc_ref_flux = np.divide(mc_ref_track, CELL_AREAS, out=np.zeros_like(mc_ref_track), where=(CELL_AREAS>0))

            # Save MC reference for future runs
            mc_ref_save_data = {
                'mc_ref_energy': mc_ref_energy,
                'mc_ref_track': mc_ref_track,
                'mc_ref_flux': mc_ref_flux,
                'mc_ref_time': mc_ref_t,
                'num_particles': NUM_PARTICLES_MC_HIGH,
                'num_cells_x': NUM_CELLS_X,
                'num_cells_y': NUM_CELLS_Y,
                'mesh_refinement': MESH_REFINEMENT,
                'domain_size': DOMAIN_SIZE,
                'source_mode': SOURCE_MODE
            }
            np.savez(mc_ref_file, **mc_ref_save_data)
            print(f"MC reference saved to {mc_ref_file}")

        # Energy totals
        mc_low_total_energy = float(np.sum(mc_low_energy))
        mc_ref_total_energy = float(np.sum(mc_ref_energy))
        print("\n" + "="*60)
        print(f"LATTICE - TOTAL ENERGY DEPOSITION COMPARISON ({NUM_CELLS_X}x{NUM_CELLS_Y} cells)")
        print(f"Total domain: W={TOTAL_WIDTH} cm, H={TOTAL_HEIGHT} cm")
        print("="*60)
        print(f"  CFM:        {cfm_total_energy:.8f}")
        print(f"  MC (low):   {mc_low_total_energy:.8f}")
        print(f"  MC (ref):   {mc_ref_total_energy:.8f}")
        print(f"  ---")
        if mc_ref_total_energy > 0:
            print(f"  CFM/MC(ref):    {cfm_total_energy/mc_ref_total_energy:.6f}")
            print(f"  MC(low)/MC(ref):{mc_low_total_energy/mc_ref_total_energy:.6f}")
        print("="*60)

    # 3) Plotting — ENERGY MAPS (existing behavior)
    energy_maps = { f"CFM Energy ({NUM_PARTICLES/1e3:.0f}k)": cfm_energy }
    if MC_AVAILABLE:
        energy_maps[f"MC Energy ({NUM_PARTICLES_MC/1e3:.0f}k)"] = mc_low_energy
        energy_maps[f"MC Ref Energy ({NUM_PARTICLES_MC_HIGH/1e6:.1f}M)"] = mc_ref_energy
    _plot_maps(energy_maps, CELL_BOUNDARIES_X, CELL_BOUNDARIES_Y, label="Energy")

    # ENERGY absolute errors
    if MC_AVAILABLE:
        _plot_absolute_errors(mc_ref_energy, cfm_energy, mc_low_energy,
                              CELL_BOUNDARIES_X, CELL_BOUNDARIES_Y,
                              labelA="CFM Energy", labelB="MC(low) Energy",
                              basenameA="cfm_energy", basenameB="mc_low_energy")

    # 4) NEW — FLUX MAPS
    flux_maps = { f"CFM Flux ({NUM_PARTICLES/1e3:.0f}k)": cfm_flux }
    if MC_AVAILABLE:
        flux_maps[f"MC Flux ({NUM_PARTICLES_MC/1e3:.0f}k)"] = mc_low_flux
        flux_maps[f"MC Ref Flux ({NUM_PARTICLES_MC_HIGH/1e6:.1f}M)"] = mc_ref_flux
    _plot_maps(flux_maps, CELL_BOUNDARIES_X, CELL_BOUNDARIES_Y, vmin=1e-6, vmax=1e0, label="Scalar Flux (track/area)")

    # FLUX absolute errors vs reference
    if MC_AVAILABLE:
        _plot_absolute_errors(mc_ref_flux, cfm_flux, mc_low_flux,
                              CELL_BOUNDARIES_X, CELL_BOUNDARIES_Y,
                              labelA="CFM Flux", labelB="MC(low) Flux",
                              basenameA="cfm_flux", basenameB="mc_low_flux")

    # Optional line comparisons (energy & flux)
    _plot_line_comparison(cfm_energy, mc_low_energy if MC_AVAILABLE else np.zeros_like(cfm_energy),
                          mc_ref_energy if MC_AVAILABLE else np.zeros_like(cfm_energy),
                          CELL_BOUNDARIES_X, quantity_label="Energy")
    _plot_line_comparison(cfm_flux, mc_low_flux if MC_AVAILABLE else np.zeros_like(cfm_flux),
                          mc_ref_flux if MC_AVAILABLE else np.zeros_like(cfm_flux),
                          CELL_BOUNDARIES_X, quantity_label="Scalar Flux")

    # Save simulation data to numpy archive
    save_data = {
        'cfm_energy': cfm_energy,
        'cfm_flux': cfm_flux,
        'mc_energy': mc_low_energy,
        'mc_flux': mc_low_flux,
        'mc_energy_ref': mc_ref_energy,
        'mc_flux_ref': mc_ref_flux,
        'num_particles_cfm': NUM_PARTICLES,
        'num_particles_mc': NUM_PARTICLES_MC,
        'num_particles_mc_high': NUM_PARTICLES_MC_HIGH,
        'mesh_refinement': MESH_REFINEMENT,
        'num_cells_x': NUM_CELLS_X,
        'num_cells_y': NUM_CELLS_Y,
        'total_width': TOTAL_WIDTH,
        'total_height': TOTAL_HEIGHT,
        'cell_boundaries_x': CELL_BOUNDARIES_X,
        'cell_boundaries_y': CELL_BOUNDARIES_Y,
        'material_map': MATERIAL_MAP,
        'sigma_t_map': SIGMA_T_MAP,
        'sigma_a_map': SIGMA_A_MAP,
        'sigma_s_map': SIGMA_S_MAP,
        'cfm_time': cfm_t,
        'mc_time': mc_low_t,
        'mc_ref_time': mc_ref_t
    }

    # Save as numpy archive with particle count in filename
    save_path = Path(RESULTS_DIR) / f'lattice_simulation_data_{NUM_PARTICLES}.npz'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **save_data)
    print(f"\nSimulation data saved to {save_path}")

    print(f"\nResults saved to {RESULTS_DIR}")

# Export handles
multi_cell_cfm = run_cfm_mesh

def set_random_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    main()
