#!/usr/bin/env python3
"""Iterative multigroup steady lattice solve using the existing steady 2D CFM models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MC_AVAILABLE = True
try:
    from physics.mc2d import calculate_boundaries_2d, run_mc_particle_bank_2d_njit_wrapper
except Exception:
    MC_AVAILABLE = False

from physics.multigroup_steady_lattice import (
    MultigroupSteadyLatticeProblem,
    allocate_group_particle_counts,
    build_mixed_source_particle_bank,
    build_multigroup_steady_lattice_problem,
    group_effective_xs,
    group_outscatter_source,
    iterate_multigroup_steady_lattice,
    mixed_source_particle_weights,
)


EPSILON = 1.0e-9
DEFAULT_NUM_PARTICLES = 20_000
DEFAULT_MAX_OUTER = 25
DEFAULT_TOL = 1.0e-5
DEFAULT_SEED = 420
HIDDEN_DIM = 256
DEPTH = 6

_ORIENT_R = np.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0], [0.0, -1.0]],
        [[0.0, -1.0], [1.0, 0.0]],
        [[0.0, 1.0], [-1.0, 0.0]],
    ],
    dtype=np.float32,
)
_F2F = np.asarray([1, 0, 3, 2], dtype=np.int64)
_U64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SPLITMIX64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX64_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX64_MUL2 = np.uint64(0x94D049BB133111EB)
_U53_SCALE = 1.0 / float(1 << 53)

log = logging.getLogger("lattice_multigroup_steady")


@dataclass(frozen=True)
class SteadyModelBundle:
    device: str
    boundary_solver: object
    internal_solver: object
    boundary_norm: dict
    internal_norm: dict
    boundary_cond_dim: int
    boundary_target_dim: int
    internal_cond_dim: int
    internal_target_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=DEFAULT_NUM_PARTICLES,
                        help="Total particles per outer iteration across all groups.")
    parser.add_argument("--mesh-refinement", type=int, default=8)
    parser.add_argument("--num-groups", type=int, default=7)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-outer", type=int, default=DEFAULT_MAX_OUTER)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL)
    parser.add_argument("--sigma-s-scale", type=float, default=1.0,
                        help="Multiplier applied to the base material scattering cross sections.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", type=str, default="cfm", choices=["cfm", "mc"])
    parser.add_argument("--cfm-batch-size", type=int, default=2048,
                        help="Max particles per model inference chunk in the CFM backend.")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/results/lattice_multigroup_steady"))
    parser.add_argument("--mc-comparison", action="store_true", default=False,
                        help="Run a traditional MC reference solve for comparison.")
    parser.add_argument("--num-particles-mc", type=int, default=50_000,
                        help="Total particles per outer iteration across all groups for the MC comparison.")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(device: str) -> SteadyModelBundle:
    from flow_matching.solver import ODESolver
    from ml.model import WrappedModel
    from ml.transform import load_normalization_params
    from ml.utils import load_model as load_saved_model

    boundary_model, boundary_input_dim, boundary_cond_dim = load_saved_model(
        device=device,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        expansion_factor=1,
        source_type="boundary",
        data_dim="2d",
    )
    boundary_model.eval()
    boundary_solver = ODESolver(velocity_model=WrappedModel(boundary_model).to(device))
    boundary_norm, _ = load_normalization_params("ml/data/norm_params_2d_boundary.npy")

    internal_model, internal_input_dim, internal_cond_dim = load_saved_model(
        device=device,
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        expansion_factor=1,
        source_type="internal",
        data_dim="2d",
    )
    internal_model.eval()
    internal_solver = ODESolver(velocity_model=WrappedModel(internal_model).to(device))
    internal_norm, _ = load_normalization_params("ml/data/norm_params_2d_internal.npy")

    return SteadyModelBundle(
        device=device,
        boundary_solver=boundary_solver,
        internal_solver=internal_solver,
        boundary_norm=boundary_norm,
        internal_norm=internal_norm,
        boundary_cond_dim=boundary_cond_dim,
        boundary_target_dim=boundary_input_dim - boundary_cond_dim,
        internal_cond_dim=internal_cond_dim,
        internal_target_dim=internal_input_dim - internal_cond_dim,
    )


def classify_face_and_snap(x: torch.Tensor, y: torch.Tensor, width: torch.Tensor, height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.stack([torch.abs(x), torch.abs(width - x), torch.abs(height - y), torch.abs(y)], dim=1)
    face = torch.argmin(distances, dim=1)
    zeros = torch.zeros_like(x)
    x_snap = torch.where(face == 0, zeros, torch.where(face == 1, width, x))
    y_snap = torch.where(face == 3, zeros, torch.where(face == 2, height, y))
    x_snap = torch.minimum(torch.maximum(x_snap, zeros), width)
    y_snap = torch.minimum(torch.maximum(y_snap, zeros), height)
    pos_on_face = torch.where((face == 0) | (face == 1), y_snap, x_snap)
    return face, x_snap, y_snap, pos_on_face


def enforce_outward_angle_from_face(face: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normals = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
        device=u.device,
    )
    direction = torch.stack([u, v], dim=1)
    normal = normals[face]
    dot = torch.sum(direction * normal, dim=1)
    reflected = direction - 2.0 * dot.unsqueeze(1) * normal
    direction = torch.where((dot <= 0.0).unsqueeze(1), reflected, direction)
    return direction[:, 0], direction[:, 1]


def plot_map(boundaries_x: np.ndarray, boundaries_y: np.ndarray, data: np.ndarray, output_path: Path, title: str) -> None:
    if not np.any(data > 0.0):
        return
    x_mesh, y_mesh = np.meshgrid(boundaries_x, boundaries_y)
    positive = data[data > 0.0]
    vmin = float(np.min(positive))
    vmax = float(np.max(data))
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(
        x_mesh,
        y_mesh,
        np.maximum(data, vmin),
        shading="auto",
        cmap="jet",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.colorbar()
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def sample_solver_batched(
    solver: object,
    cond: torch.Tensor,
    noise: torch.Tensor,
    target_dim: int,
    time_grid: torch.Tensor,
    step_size: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    batch = int(cond.shape[0])
    if batch == 0:
        return torch.empty((0, target_dim), dtype=cond.dtype, device=cond.device)
    if noise.shape != (batch, target_dim):
        raise ValueError(f"noise must have shape {(batch, target_dim)}; got {tuple(noise.shape)}.")

    outputs = []
    for start in range(0, batch, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), batch)
        cond_chunk = cond[start:stop]
        noise_chunk = noise[start:stop]
        out = solver.sample(
            time_grid=time_grid,
            x_init=torch.cat([cond_chunk, noise_chunk], dim=1),
            method="rk4",
            step_size=step_size,
            return_intermediates=False,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=0)


def _splitmix64_step(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    next_state = (np.asarray(state, dtype=np.uint64) + _SPLITMIX64_GAMMA) & _U64_MASK
    mixed = next_state.copy()
    mixed ^= mixed >> np.uint64(30)
    mixed = (mixed * _SPLITMIX64_MUL1) & _U64_MASK
    mixed ^= mixed >> np.uint64(27)
    mixed = (mixed * _SPLITMIX64_MUL2) & _U64_MASK
    mixed ^= mixed >> np.uint64(31)
    return next_state, mixed


def _advance_particle_noise_states(
    noise_state: np.ndarray,
    particle_indices: np.ndarray,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(noise_state[particle_indices], dtype=np.uint64).copy()
    if state.size == 0:
        return np.empty((0, target_dim), dtype=np.float32), state

    pair_count = (int(target_dim) + 1) // 2
    normals: list[np.ndarray] = []
    tiny = np.finfo(np.float64).tiny
    for _ in range(pair_count):
        state, raw_u1 = _splitmix64_step(state)
        state, raw_u2 = _splitmix64_step(state)
        u1 = np.maximum(((raw_u1 >> np.uint64(11)).astype(np.float64)) * _U53_SCALE, tiny)
        u2 = ((raw_u2 >> np.uint64(11)).astype(np.float64)) * _U53_SCALE
        radius = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        normals.append(radius * np.cos(theta))
        normals.append(radius * np.sin(theta))

    noise = np.stack(normals, axis=1)[:, : int(target_dim)].astype(np.float32, copy=False)
    return noise, state


def build_particle_model_noise(
    noise_state: np.ndarray,
    particle_indices: np.ndarray,
    target_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    noise_np, next_state = _advance_particle_noise_states(noise_state, particle_indices, target_dim)
    noise_state[particle_indices] = next_state
    return torch.from_numpy(noise_np).to(device=device, dtype=dtype)


def run_cfm_fixed_source(
    problem: MultigroupSteadyLatticeProblem,
    models: SteadyModelBundle,
    *,
    source_map: np.ndarray,
    sigma_a_map: np.ndarray,
    sigma_s_map: np.ndarray,
    num_particles: int,
    seed: int,
    batch_size: int,
    max_events: int = 10_000,
) -> dict:
    from ml.transform import (
        denorm_angle_phi,
        denorm_log_feature,
        denorm_position_relative,
        denorm_radius_r,
        normalize_angle_phi,
        normalize_height,
        normalize_position_relative,
        normalize_radius_r,
        normalize_width,
        polar_to_uv,
        uv_to_polar,
    )

    geometry = problem.geometry
    if int(num_particles) <= 0:
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": 0.0}
    total_source = float(np.sum(source_map * geometry.cell_areas))
    if total_source <= 0.0:
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": 0.0}

    bank = build_mixed_source_particle_bank(problem, num_particles, seed)
    particle_weights = mixed_source_particle_weights(problem, source_map, bank)
    active_bank = np.flatnonzero(particle_weights > EPSILON)
    if active_bank.size == 0:
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": total_source}

    set_all_seeds(seed)
    torch.set_grad_enabled(False)
    device = models.device
    width_t = torch.from_numpy(geometry.cell_widths.astype(np.float32)).to(device)
    height_t = torch.from_numpy(geometry.cell_heights.astype(np.float32)).to(device)
    sigma_a_t = torch.from_numpy(np.asarray(sigma_a_map, dtype=np.float32)).to(device)
    sigma_s_t = torch.from_numpy(np.asarray(sigma_s_map, dtype=np.float32)).to(device)
    orient_t = torch.from_numpy(_ORIENT_R).to(device)
    f2f_t = torch.from_numpy(_F2F).to(device)

    particle = torch.zeros((active_bank.size, 8), dtype=torch.float32, device=device)
    particle[:, 0] = torch.from_numpy(bank.cell_x[active_bank].astype(np.float32)).to(device)
    particle[:, 1] = torch.from_numpy(bank.cell_y[active_bank].astype(np.float32)).to(device)
    particle_weight = torch.from_numpy(particle_weights[active_bank].astype(np.float64)).to(device)
    noise_state = bank.trajectory_seeds[active_bank].astype(np.uint64, copy=True)

    track_flat = torch.zeros(geometry.num_cells_x * geometry.num_cells_y, dtype=torch.float64, device=device)
    base_grid = torch.linspace(0.0, 0.8, 6, device=device)
    fine_grid = torch.linspace(0.8, 1.0, 7, device=device)[1:]
    time_grid = torch.cat([base_grid, fine_grid])
    step_size = time_grid[1] - time_grid[0]
    wall_start = time.time()

    def accumulate(active: torch.Tensor, length: torch.Tensor, sigma_a_local: torch.Tensor) -> None:
        safe_sigma_a = sigma_a_local.to(dtype=torch.float64).clamp_min(EPSILON)
        energy_before = particle_weight[active]
        optical_depth = safe_sigma_a * length.to(dtype=torch.float64)
        lin_idx = particle[active, 1].long() * geometry.num_cells_x + particle[active, 0].long()
        track_add = torch.where(
            optical_depth > 0.01,
            energy_before * (1.0 - torch.exp(-optical_depth)) / safe_sigma_a,
            energy_before * length,
        )
        track_flat.index_add_(0, lin_idx, track_add)
        deposit = energy_before * (1.0 - torch.exp(-optical_depth))
        particle_weight[active] = energy_before - deposit

    active = (particle[:, 7] == 0).nonzero(as_tuple=True)[0]
    if active.numel() > 0:
        batch = active.numel()
        active_idx_np = active.detach().cpu().numpy().astype(np.int64, copy=False)
        cell_x = particle[active, 0].long()
        cell_y = particle[active, 1].long()
        width = width_t[cell_x]
        height = height_t[cell_y]
        sigma_s_local = sigma_s_t[cell_y, cell_x].clamp_min(EPSILON)
        sigma_a_local = sigma_a_t[cell_y, cell_x].clamp_min(0.0)
        bank_mu = 2.0 * bank.dir_unit_mu[active_bank[active_idx_np]] - 1.0
        bank_phi = 2.0 * np.pi * bank.dir_unit_phi[active_bank[active_idx_np]]
        mu = bank_mu
        phi = bank_phi
        tangent = np.sqrt(1.0 - mu * mu)
        phi_in, r_in = uv_to_polar(tangent * np.cos(phi), tangent * np.sin(phi))
        phi_in_t = torch.from_numpy(np.asarray(phi_in, dtype=np.float32)).to(device)
        r_in_t = torch.from_numpy(np.asarray(r_in, dtype=np.float32)).to(device)
        start_rx_t = torch.from_numpy(bank.pos_rx[active_bank[active_idx_np]].astype(np.float32)).to(device)
        start_ry_t = torch.from_numpy(bank.pos_ry[active_bank[active_idx_np]].astype(np.float32)).to(device)
        cond = torch.stack(
            [
                normalize_width(width * sigma_s_local, models.internal_norm["width_lower"], models.internal_norm["width_upper"]),
                normalize_height(height * sigma_s_local, models.internal_norm["height_lower"], models.internal_norm["height_upper"]),
                normalize_position_relative(start_rx_t),
                normalize_position_relative(start_ry_t),
                normalize_angle_phi(phi_in_t),
                normalize_radius_r(r_in_t),
            ],
            dim=1,
        )
        noise = build_particle_model_noise(
            noise_state,
            active_idx_np,
            models.internal_target_dim,
            device=device,
            dtype=cond.dtype,
        )
        out = sample_solver_batched(
            models.internal_solver,
            cond,
            noise,
            models.internal_target_dim,
            time_grid,
            step_size,
            chunk_size=batch_size,
        )
        pred = out[:, models.internal_cond_dim :]
        x_out = denorm_position_relative(pred[:, 0]) * width
        y_out = denorm_position_relative(pred[:, 1]) * height
        dt = denorm_log_feature(pred[:, 4], models.internal_norm, "time").clamp_min(0.0) / sigma_s_local
        pred_phi = denorm_angle_phi(pred[:, 2])
        pred_r = denorm_radius_r(pred[:, 3])
        u_out, v_out = polar_to_uv(pred_phi, pred_r)
        face, x_out, y_out, pos_on_face = classify_face_and_snap(x_out, y_out, width, height)
        u_out, v_out = enforce_outward_angle_from_face(face, u_out, v_out)
        accumulate(active, dt, sigma_a_local)

        alive = particle_weight[active] >= EPSILON
        if alive.any():
            next_idx = active[alive]
            next_face = face[alive]
            next_x = cell_x[alive] + (next_face == 1).long() - (next_face == 0).long()
            next_y = cell_y[alive] + (next_face == 2).long() - (next_face == 3).long()
            interior = (
                (next_x >= 0)
                & (next_x < geometry.num_cells_x)
                & (next_y >= 0)
                & (next_y < geometry.num_cells_y)
            )
            particle[next_idx[interior], 0] = next_x[interior].float()
            particle[next_idx[interior], 1] = next_y[interior].float()
            particle[next_idx[interior], 2] = f2f_t[next_face[interior]].float()
            particle[next_idx[interior], 3] = pos_on_face[alive][interior]
            particle[next_idx[interior], 4] = u_out[alive][interior]
            particle[next_idx[interior], 5] = v_out[alive][interior]
            particle[next_idx[~interior], 7] = 1.0
        particle[active[~alive], 7] = 2.0

    for _ in range(max_events):
        active = (particle[:, 7] == 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            break
        batch = active.numel()
        active_idx_np = active.detach().cpu().numpy().astype(np.int64, copy=False)
        ix = particle[active, 0].long()
        iy = particle[active, 1].long()
        face = particle[active, 2].long()
        pos = particle[active, 3]
        u = particle[active, 4]
        v = particle[active, 5]
        width = width_t[ix]
        height = height_t[iy]
        sigma_s_local = sigma_s_t[iy, ix].clamp_min(EPSILON)
        sigma_a_local = sigma_a_t[iy, ix].clamp_min(0.0)

        pos_x = torch.where(face == 0, torch.zeros_like(pos), torch.where(face == 1, width, pos))
        pos_y = torch.where(face == 2, height, torch.where(face == 3, torch.zeros_like(pos), pos))
        pos_phys = torch.stack([pos_x, pos_y], dim=1)
        shift_x = torch.where(face == 1, width, torch.where(face == 2, height, torch.zeros_like(width)))
        shift_y = torch.where(face == 1, height, torch.where(face == 3, width, torch.zeros_like(width)))
        shift = torch.stack([shift_x, shift_y], dim=1)
        rotation = orient_t[face]
        pos_model = torch.bmm(rotation, pos_phys.unsqueeze(-1)).squeeze(-1) + shift
        dir_model = torch.bmm(rotation, torch.stack([u, v], dim=1).unsqueeze(-1)).squeeze(-1)
        width_model = torch.where(face < 2, width, height)
        height_model = torch.where(face < 2, height, width)
        phi_in, r_in = uv_to_polar(dir_model[:, 0].detach().cpu().numpy(), dir_model[:, 1].detach().cpu().numpy())
        phi_in_t = torch.from_numpy(np.asarray(phi_in, dtype=np.float32)).to(device)
        r_in_t = torch.from_numpy(np.asarray(r_in, dtype=np.float32)).to(device)
        cond = torch.stack(
            [
                normalize_width(width_model * sigma_s_local, models.boundary_norm["width_lower"], models.boundary_norm["width_upper"]),
                normalize_height(height_model * sigma_s_local, models.boundary_norm["height_lower"], models.boundary_norm["height_upper"]),
                normalize_position_relative(torch.clamp(pos_model[:, 1] / (height_model + EPSILON), 0.0, 1.0)),
                normalize_angle_phi(phi_in_t),
                normalize_radius_r(r_in_t),
            ],
            dim=1,
        )
        noise = build_particle_model_noise(
            noise_state,
            active_idx_np,
            models.boundary_target_dim,
            device=device,
            dtype=cond.dtype,
        )
        out = sample_solver_batched(
            models.boundary_solver,
            cond,
            noise,
            models.boundary_target_dim,
            time_grid,
            step_size,
            chunk_size=batch_size,
        )
        pred = out[:, models.boundary_cond_dim :]
        x_model = denorm_position_relative(pred[:, 0]) * width_model
        y_model = denorm_position_relative(pred[:, 1]) * height_model
        dt = denorm_log_feature(pred[:, 4], models.boundary_norm, "time").clamp_min(EPSILON) / sigma_s_local
        pred_phi = denorm_angle_phi(pred[:, 2])
        pred_r = denorm_radius_r(pred[:, 3])
        u_new, v_new = polar_to_uv(pred_phi, pred_r)
        face_model, x_model, y_model, _ = classify_face_and_snap(x_model, y_model, width_model, height_model)
        u_new, v_new = enforce_outward_angle_from_face(face_model, u_new, v_new)
        accumulate(active, dt, sigma_a_local)

        pos_exit = torch.bmm(rotation.transpose(1, 2), (torch.stack([x_model, y_model], dim=1) - shift).unsqueeze(-1)).squeeze(-1)
        dir_exit = torch.bmm(rotation.transpose(1, 2), torch.stack([u_new, v_new], dim=1).unsqueeze(-1)).squeeze(-1)
        clamped_x = torch.minimum(torch.maximum(pos_exit[:, 0], torch.zeros_like(width)), width)
        clamped_y = torch.minimum(torch.maximum(pos_exit[:, 1], torch.zeros_like(height)), height)
        face_phys, _, _, pos_on_face = classify_face_and_snap(clamped_x, clamped_y, width, height)
        next_ix = ix + (face_phys == 1).long() - (face_phys == 0).long()
        next_iy = iy + (face_phys == 2).long() - (face_phys == 3).long()
        interior = (
            (next_ix >= 0)
            & (next_ix < geometry.num_cells_x)
            & (next_iy >= 0)
            & (next_iy < geometry.num_cells_y)
        )
        alive = particle_weight[active] >= EPSILON
        keep = interior & alive
        particle[active[~keep], 7] = torch.where(alive[~keep], torch.ones_like(particle[active[~keep], 7]), torch.full_like(particle[active[~keep], 7], 2.0))
        if keep.any():
            kept = active[keep]
            particle[kept, 0] = next_ix[keep].float()
            particle[kept, 1] = next_iy[keep].float()
            particle[kept, 2] = f2f_t[face_phys[keep]].float()
            particle[kept, 3] = pos_on_face[keep]
            particle[kept, 4] = dir_exit[:, 0][keep]
            particle[kept, 5] = dir_exit[:, 1][keep]

    elapsed = time.time() - wall_start
    track_grid = track_flat.view(geometry.num_cells_y, geometry.num_cells_x).detach().cpu().numpy()
    flux_grid = np.divide(track_grid, geometry.cell_areas, out=np.zeros_like(track_grid), where=geometry.cell_areas > 0.0)
    return {"flux": flux_grid, "elapsed": elapsed, "source_strength": total_source}


def run_mc_fixed_source(
    problem: MultigroupSteadyLatticeProblem,
    *,
    source_map: np.ndarray,
    sigma_a_map: np.ndarray,
    sigma_s_map: np.ndarray,
    num_particles: int,
    seed: int,
) -> dict:
    """Run the MC reference solve from a fixed mixed source proposal bank."""
    geometry = problem.geometry
    total_source = float(np.sum(source_map * geometry.cell_areas))
    if int(num_particles) <= 0:
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": 0.0}
    if total_source <= 0.0:
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": 0.0}

    bank = build_mixed_source_particle_bank(problem, num_particles, seed)
    start_weights = mixed_source_particle_weights(problem, source_map, bank)
    if not np.any(start_weights > EPSILON):
        zeros = np.zeros((geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
        return {"flux": zeros, "elapsed": 0.0, "source_strength": total_source}

    # Use sigma_s as the transport cross-section (survival biasing), while
    # sigma_a drives continuous weight reduction along each particle path.
    sigma_t_map = np.asarray(sigma_s_map, dtype=np.float64).clip(min=float(EPSILON))
    sigma_a_arr = np.asarray(sigma_a_map, dtype=np.float64).clip(min=0.0)

    cell_widths = geometry.cell_widths.astype(np.float64)
    cell_heights = geometry.cell_heights.astype(np.float64)
    bx, by = calculate_boundaries_2d(cell_widths, cell_heights)
    total_width = float(bx[-1])
    total_height = float(by[-1])

    wall_start = time.time()
    _, track_accum, _ = run_mc_particle_bank_2d_njit_wrapper(
        num_cells_x=geometry.num_cells_x,
        num_cells_y=geometry.num_cells_y,
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=bx,
        cell_boundaries_y=by,
        total_width=total_width,
        total_height=total_height,
        start_cell_indices_x=bank.cell_x,
        start_cell_indices_y=bank.cell_y,
        start_position_relative_x=bank.pos_rx,
        start_position_relative_y=bank.pos_ry,
        start_dir_unit_mu=bank.dir_unit_mu,
        start_dir_unit_phi=bank.dir_unit_phi,
        start_weights=start_weights,
        particle_seeds=bank.trajectory_seeds,
        energy_threshold=EPSILON,
        reflected_boundaries=False,
        sigma_t_per_cell=sigma_t_map,
        sigma_a_per_cell=sigma_a_arr,
        verbose=False,
    )
    elapsed = time.time() - wall_start
    flux_grid = np.divide(track_accum, geometry.cell_areas, out=np.zeros_like(track_accum), where=geometry.cell_areas > 0.0)
    return {"flux": flux_grid, "elapsed": elapsed, "source_strength": total_source}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = resolve_device(args.device)
    models = load_models(device) if args.backend == "cfm" else None
    problem = build_multigroup_steady_lattice_problem(
        refinement=args.mesh_refinement,
        num_groups=args.num_groups,
        sigma_s_multiplier=args.sigma_s_scale,
    )
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    def solve_group(
        group: int,
        source_map: np.ndarray,
        sigma_a_eff: np.ndarray,
        sigma_s_within: np.ndarray,
        group_particles: int,
        seed: int,
    ) -> dict:
        if args.backend == "cfm":
            assert models is not None
            result = run_cfm_fixed_source(
                problem,
                models,
                source_map=source_map,
                sigma_a_map=sigma_a_eff,
                sigma_s_map=sigma_s_within,
                num_particles=group_particles,
                seed=seed,
                batch_size=args.cfm_batch_size,
            )
        else:
            result = run_mc_fixed_source(
                problem,
                source_map=source_map,
                sigma_a_map=sigma_a_eff,
                sigma_s_map=sigma_s_within,
                num_particles=group_particles,
                seed=seed,
            )
        log.info(
            "backend=%s group=%d seed=%d particles=%d source=%.6e elapsed=%.2fs",
            args.backend,
            group,
            seed,
            group_particles,
            result["source_strength"],
            result["elapsed"],
        )
        return result

    result = iterate_multigroup_steady_lattice(
        problem,
        solve_group,
        num_particles=args.num_particles,
        max_outer_iterations=args.max_outer,
        rel_tol=args.tolerance,
        seed=args.seed,
    )

    total_flux = np.sum(result.flux, axis=0)
    print(f"Backend: {args.backend}")
    print(f"Device: {device}")
    print(f"Grid: {problem.geometry.num_cells_x}x{problem.geometry.num_cells_y}")
    print(f"Groups: {problem.num_groups}")
    print(f"Sigma_s scale: {args.sigma_s_scale:g}")
    print(f"Converged: {result.converged} in {result.outer_iterations} outer iterations")
    print("Scatter matrix S[g_in, g_out] (row-normalized probabilities):")
    print(np.array2string(problem.scattering_matrix, precision=4, suppress_small=True))
    print("Absorption multipliers:")
    print(np.array2string(problem.absorption_multipliers, precision=4, suppress_small=True))
    print("Outer-iteration history:")
    for idx in range(result.outer_iterations):
        print(
            f"  iter={idx + 1:2d} "
            f"max_rel={result.convergence_history[idx]:.6e} "
            f"source_l1={float(np.sum(result.source_strength_history[idx])):.6e} "
            f"particles={result.particle_counts_history[idx].tolist()}"
        )

    # ── Optional MC comparison ──────────────────────────────────────────
    mc_flux = None
    mc_group_flux = None
    if args.mc_comparison and args.backend != "cfm":
        print("\nWARNING: --mc-comparison currently compares MC against a CFM main solve. Skipping because backend=mc.")
    elif args.mc_comparison and args.backend == "cfm":
        if not MC_AVAILABLE:
            print("\nWARNING: MC physics module not available – skipping MC comparison.")
        else:
            print(f"\nRunning MC reference comparison ({args.num_particles_mc:,} total particles/outer-equivalent)...")
            mc_group_flux = np.zeros_like(result.flux)
            mc_t_start = time.time()
            final_source = problem.external_source + np.stack(
                [group_outscatter_source(problem, result.flux, g) for g in range(problem.num_groups)],
                axis=0,
            )
            mc_counts = allocate_group_particle_counts(problem, final_source, args.num_particles_mc)

            for g in range(problem.num_groups):
                n_g = int(mc_counts[g])
                if n_g == 0:
                    continue
                sigma_a_eff, sigma_s_within = group_effective_xs(problem, g)
                source_map_g = final_source[g]
                mc_result = run_mc_fixed_source(
                    problem,
                    source_map=source_map_g,
                    sigma_a_map=sigma_a_eff,
                    sigma_s_map=sigma_s_within,
                    num_particles=n_g,
                    seed=args.seed + g,
                )
                mc_group_flux[g] = mc_result["flux"]
                log.info("MC compare group=%d particles=%d source=%.4e elapsed=%.2fs", g, n_g, mc_result["source_strength"], mc_result["elapsed"])

            mc_total_elapsed = time.time() - mc_t_start
            mc_flux = np.sum(mc_group_flux, axis=0)

            # ── Error metrics ───────────────────────────────────────────
            mask = mc_flux > 0.0
            if mask.any():
                abs_err = np.abs(total_flux - mc_flux)
                rel_err = np.divide(abs_err, mc_flux, out=np.zeros_like(abs_err), where=mask)
                print(f"\nMC comparison complete in {mc_total_elapsed:.1f}s")
                print(f"  Main total flux — mean={total_flux.mean():.4e}  max={total_flux.max():.4e}")
                print(f"  MC  total flux  — mean={mc_flux.mean():.4e}  max={mc_flux.max():.4e}")
                print(f"  Abs error       — mean={abs_err.mean():.4e}  max={abs_err.max():.4e}")
                print(f"  Rel error (%)   — mean={rel_err[mask].mean()*100:.2f}%  max={rel_err[mask].max()*100:.2f}%")
            else:
                print("\nMC comparison: MC flux was all-zero, cannot compute relative error.")

    # ── Save & plot ─────────────────────────────────────────────────────
    save_dict = dict(
        flux=result.flux,
        total_flux=total_flux,
        source=result.source,
        convergence_history=result.convergence_history,
        source_strength_history=result.source_strength_history,
        particle_counts_history=result.particle_counts_history,
        scatter_matrix=problem.scattering_matrix,
        absorption_multipliers=problem.absorption_multipliers,
        external_source=problem.external_source,
        converged=np.asarray(result.converged),
        outer_iterations=np.asarray(result.outer_iterations),
        backend=np.asarray(args.backend),
        seed=np.asarray(args.seed),
        num_particles=np.asarray(args.num_particles),
        sigma_s_scale=np.asarray(args.sigma_s_scale),
    )
    if mc_flux is not None:
        save_dict["mc_total_flux"] = mc_flux
        save_dict["mc_group_flux"] = mc_group_flux

    np.savez_compressed(outdir / "lattice_multigroup_steady.npz", **save_dict)
    plot_map(problem.geometry.cell_boundaries_x, problem.geometry.cell_boundaries_y, total_flux, outdir / "total_flux.png", "Total Multigroup Flux (CFM)")
    plot_map(problem.geometry.cell_boundaries_x, problem.geometry.cell_boundaries_y, result.flux[0], outdir / "group_0_flux.png", "Group 0 Flux (CFM)")

    if mc_flux is not None:
        plot_map(problem.geometry.cell_boundaries_x, problem.geometry.cell_boundaries_y, mc_flux, outdir / "mc_total_flux.png", "Total Multigroup Flux (MC Reference)")
        plot_map(problem.geometry.cell_boundaries_x, problem.geometry.cell_boundaries_y, mc_group_flux[0], outdir / "mc_group_0_flux.png", "Group 0 Flux (MC Reference)")
        # Relative-error map
        mask = mc_flux > 0.0
        rel_err_map = np.divide(np.abs(total_flux - mc_flux), mc_flux, out=np.zeros_like(mc_flux), where=mask)
        if np.any(rel_err_map > 0.0):
            plot_map(problem.geometry.cell_boundaries_x, problem.geometry.cell_boundaries_y, rel_err_map, outdir / "rel_error_cfm_vs_mc.png", "|CFM − MC| / MC (relative error)")

    print(f"Saved results to {outdir}")


if __name__ == "__main__":
    main()
