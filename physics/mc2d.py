"""mc2d.py - 2D steady-state Monte Carlo transport (single-cell and multi-cell).

Single-cell functions treat the cell as a rectangle and track every particle from
entry to exit/absorption. Multi-cell functions operate on a Cartesian mesh with
per-cell cross-section arrays (always used; the Python wrapper fills homogeneous
arrays from scatter_ratio when no per-cell data is provided).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from matplotlib.colors import LogNorm

from .mc1d import sample_unit
from ._seed import seeded_numpy_execution
from .steady_2d_scattering import (
    guard_projected_direction_xy,
    sample_incoming_hemisphere_plus_x_direction_3d_from_unit,
    sample_isotropic_direction_3d_from_unit,
    sample_scattered_direction_3d_from_unit,
    steady_2d_scattering_mode_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@njit
def find_cell_indices_2d_njit(pos, cell_boundaries_x, cell_boundaries_y,
                               num_cells_x, num_cells_y):
    """Find (ix, iy) cell indices for a given position."""
    x, y = pos[0], pos[1]
    total_width = cell_boundaries_x[-1]
    total_height = cell_boundaries_y[-1]

    clip_eps = 1e-12
    x = np.minimum(np.maximum(x, clip_eps), total_width - clip_eps)
    y = np.minimum(np.maximum(y, clip_eps), total_height - clip_eps)

    ix = np.searchsorted(cell_boundaries_x, x, side='right') - 1
    iy = np.searchsorted(cell_boundaries_y, y, side='right') - 1

    ix = min(max(ix, 0), num_cells_x - 1)
    iy = min(max(iy, 0), num_cells_y - 1)
    return ix, iy


def calculate_boundaries_2d(cell_widths, cell_heights):
    """Return (boundaries_x, boundaries_y) from per-axis cell sizes."""
    cell_boundaries_x = np.concatenate(([0.0], np.cumsum(cell_widths)))
    cell_boundaries_y = np.concatenate(([0.0], np.cumsum(cell_heights)))
    return cell_boundaries_x, cell_boundaries_y


# ---------------------------------------------------------------------------
# Single-cell 2D steady-state
# ---------------------------------------------------------------------------

@njit(parallel=True)
def simulate_single_cell_2d_boundary_source(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    cond_samples=None,
    scattering_mode_code: int = 0,
):
    """2D single-cell boundary source. Particles enter from x=0.

    Returns array with columns:
      [0] width, [1] height, [2] initial_pos_y,
      [3] initial_dir_u, [4] initial_dir_v,
      [5] final_pos_x, [6] final_pos_y,
      [7] final_dir_u, [8] final_dir_v,
      [9] time_in_cell, [10] exited, [11] absorbed
    """
    data = np.zeros((num_neutrons, 12), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    log_min_w = np.log10(min_width)
    log_max_w = np.log10(max_width)
    log_min_h = np.log10(min_height)
    log_max_h = np.log10(max_height)

    for i in prange(num_neutrons):
        cand_w = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        W = 10.0 ** (log_min_w + cand_w * (log_max_w - log_min_w))
        cand_h = cond_samples[i, 1] if use_precond else sample_unit(distribution)
        H = 10.0 ** (log_min_h + cand_h * (log_max_h - log_min_h))

        data[i, 0] = W
        data[i, 1] = H

        pos_x = 0.0
        pos_y = H * (cond_samples[i, 2] if use_precond else np.random.random())
        data[i, 2] = pos_y

        if use_precond:
            u, v, w = sample_incoming_hemisphere_plus_x_direction_3d_from_unit(cond_samples[i, 3], cond_samples[i, 4], eps)
        else:
            u, v, w = sample_incoming_hemisphere_plus_x_direction_3d_from_unit(np.random.random(), np.random.random(), eps)
        u, v = guard_projected_direction_xy(u, v, eps)
        data[i, 3] = u
        data[i, 4] = v

        total_path = 0.0
        exited_cell = 0
        absorbed_in_cell = 0
        alive = True

        while alive:
            free_path = -np.log(np.random.rand()) / sigma_t

            dist_x_right = (W - pos_x) / u if u > eps else np.inf
            dist_x_left = (0.0 - pos_x) / u if u < -eps else np.inf
            dist_y_top = (H - pos_y) / v if v > eps else np.inf
            dist_y_bot = (0.0 - pos_y) / v if v < -eps else np.inf

            dist_boundary = min(dist_x_right, dist_x_left, dist_y_top, dist_y_bot)
            step_dist = max(0.0, min(free_path, dist_boundary))

            pos_x += step_dist * u
            pos_y += step_dist * v
            total_path += step_dist

            boundary_hit = abs(step_dist - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(pos_x - W) < eps * W + eps:
                    pos_x = W
                elif abs(pos_x - 0.0) < eps * W + eps:
                    pos_x = 0.0
                if abs(pos_y - H) < eps * H + eps:
                    pos_y = H
                elif abs(pos_y - 0.0) < eps * H + eps:
                    pos_y = 0.0
                pos_x = min(max(pos_x, 0.0), W)
                pos_y = min(max(pos_y, 0.0), H)

                exited_cell = 1
                data[i, 5] = pos_x
                data[i, 6] = pos_y
                data[i, 7] = u
                data[i, 8] = v
                data[i, 9] = total_path
                data[i, 10] = exited_cell
                data[i, 11] = absorbed_in_cell
                alive = False
            else:
                if np.random.random() < sigma_a_prob:
                    absorbed_in_cell = 1
                    data[i, 5] = pos_x
                    data[i, 6] = pos_y
                    data[i, 7] = u
                    data[i, 8] = v
                    data[i, 9] = total_path
                    data[i, 10] = exited_cell
                    data[i, 11] = absorbed_in_cell
                    alive = False
                else:
                    u, v, w = sample_scattered_direction_3d_from_unit(
                        u,
                        v,
                        w,
                        np.random.random(),
                        np.random.random(),
                        scattering_mode_code,
                        eps,
                    )
                    u, v = guard_projected_direction_xy(u, v, eps)

    return data


@njit(parallel=True)
def simulate_single_cell_2d_internal_source(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    cond_samples=None,
    scattering_mode_code: int = 0,
):
    """2D single-cell internal source. Particles start uniformly within cell.

    Returns array with columns:
      [0] width, [1] height,
      [2] initial_pos_x, [3] initial_pos_y,
      [4] initial_dir_u, [5] initial_dir_v,
      [6] final_pos_x, [7] final_pos_y,
      [8] final_dir_u, [9] final_dir_v,
      [10] time_in_cell, [11] exited, [12] absorbed
    """
    data = np.zeros((num_neutrons, 13), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    log_min_w = np.log10(min_width)
    log_max_w = np.log10(max_width)
    log_min_h = np.log10(min_height)
    log_max_h = np.log10(max_height)

    for i in prange(num_neutrons):
        cand_w = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        W = 10.0 ** (log_min_w + cand_w * (log_max_w - log_min_w))
        cand_h = cond_samples[i, 1] if use_precond else sample_unit(distribution)
        H = 10.0 ** (log_min_h + cand_h * (log_max_h - log_min_h))

        data[i, 0] = W
        data[i, 1] = H

        if use_precond:
            pos_x = W * cond_samples[i, 2]
            pos_y = H * cond_samples[i, 3]
        else:
            pos_x = np.random.uniform(0.0, W)
            pos_y = np.random.uniform(0.0, H)
        data[i, 2] = pos_x
        data[i, 3] = pos_y

        if use_precond:
            u, v, w = sample_isotropic_direction_3d_from_unit(cond_samples[i, 4], cond_samples[i, 5], eps)
        else:
            u, v, w = sample_isotropic_direction_3d_from_unit(np.random.random(), np.random.random(), eps)
        u, v = guard_projected_direction_xy(u, v, eps)
        data[i, 4] = u
        data[i, 5] = v

        total_path = 0.0
        exited_cell = 0
        absorbed_in_cell = 0
        alive = True

        while alive:
            free_path = -np.log(np.random.rand()) / sigma_t

            dist_x_right = (W - pos_x) / u if u > eps else np.inf
            dist_x_left = (0.0 - pos_x) / u if u < -eps else np.inf
            dist_y_top = (H - pos_y) / v if v > eps else np.inf
            dist_y_bot = (0.0 - pos_y) / v if v < -eps else np.inf

            dist_boundary = min(dist_x_right, dist_x_left, dist_y_top, dist_y_bot)
            step_dist = max(0.0, min(free_path, dist_boundary))

            pos_x += step_dist * u
            pos_y += step_dist * v
            total_path += step_dist

            boundary_hit = abs(step_dist - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(pos_x - W) < eps * W + eps:
                    pos_x = W
                elif abs(pos_x - 0.0) < eps * W + eps:
                    pos_x = 0.0
                if abs(pos_y - H) < eps * H + eps:
                    pos_y = H
                elif abs(pos_y - 0.0) < eps * H + eps:
                    pos_y = 0.0
                pos_x = min(max(pos_x, 0.0), W)
                pos_y = min(max(pos_y, 0.0), H)

                exited_cell = 1
                data[i, 6] = pos_x
                data[i, 7] = pos_y
                data[i, 8] = u
                data[i, 9] = v
                data[i, 10] = total_path
                data[i, 11] = exited_cell
                data[i, 12] = absorbed_in_cell
                alive = False
            else:
                if np.random.random() < sigma_a_prob:
                    absorbed_in_cell = 1
                    data[i, 6] = pos_x
                    data[i, 7] = pos_y
                    data[i, 8] = u
                    data[i, 9] = v
                    data[i, 10] = total_path
                    data[i, 11] = exited_cell
                    data[i, 12] = absorbed_in_cell
                    alive = False
                else:
                    u, v, w = sample_scattered_direction_3d_from_unit(
                        u,
                        v,
                        w,
                        np.random.random(),
                        np.random.random(),
                        scattering_mode_code,
                        eps,
                    )
                    u, v = guard_projected_direction_xy(u, v, eps)

    return data


# ---------------------------------------------------------------------------
# Multi-cell 2D steady-state (consolidated: always uses per-cell XS arrays)
# ---------------------------------------------------------------------------

@njit
def multi_cell_mc_boundary_source_2d_njit(
    num_cells_x, num_cells_y, cell_widths, cell_heights,
    cell_boundaries_x, cell_boundaries_y, total_width, total_height,
    num_particles,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflected_boundaries, angular_sampling, scattering_mode_code
):
    """2D multi-cell MC, boundary source at x=0. Uses per-cell XS arrays.

    For homogeneous materials, pass constant arrays built from scatter_ratio.
    """
    energy_deposited = np.zeros((num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_cells_y, num_cells_x), dtype=np.float64)
    exited_left, exited_right, exited_bottom, exited_top = 0, 0, 0, 0
    absorbed = 0
    travel_steps = 0
    boundary_reflections = 0
    epsilon = 1e-9
    boundary_epsilon = epsilon * 10.0
    warning_flags = 0

    for _ in range(num_particles):
        energy = 1.0
        pos_y = np.random.uniform(0.0, total_height)
        pos = np.array([boundary_epsilon, pos_y], dtype=np.float64)

        # Sample incoming direction (+x hemisphere)
        while True:
            if angular_sampling == 0:
                mu = np.sqrt(np.random.uniform(0.0, 1.0))
            else:
                mu = np.random.uniform(-1.0, 1.0)
            phi = np.random.uniform(0.0, 2.0 * np.pi)
            t = np.sqrt(1.0 - mu * mu)
            u = t * np.cos(phi)
            v = t * np.sin(phi)
            u, v = guard_projected_direction_xy(u, v, epsilon)
            if u > epsilon:
                w_mag = np.sqrt(max(0.0, 1.0 - u * u - v * v))
                w = w_mag if np.random.random() < 0.5 else -w_mag
                break
        direction = np.array([u, v, w], dtype=np.float64)

        finished = False
        max_steps_per_particle = 1000000
        steps_this_particle = 0

        while not finished and steps_this_particle < max_steps_per_particle:
            ix, iy = find_cell_indices_2d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, num_cells_x, num_cells_y
            )

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y:
                sigma_t = sigma_t_per_cell[iy, ix]
                sigma_a = sigma_a_per_cell[iy, ix]
                sigma_s = sigma_t - sigma_a
                if sigma_s < 0.0:
                    sigma_s = 0.0
            else:
                sigma_t = 1.0
                sigma_a = 0.0
                sigma_s = 1.0
                warning_flags += 2

            dist_boundary = np.inf
            u, v = direction[0], direction[1]

            if abs(u) > epsilon:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]) / u
                if dist_x > epsilon:
                    dist_boundary = min(dist_boundary, dist_x)

            if abs(v) > epsilon:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]) / v
                if dist_y > epsilon:
                    dist_boundary = min(dist_boundary, dist_y)

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > epsilon else np.inf
            travel_dist = min(free_path, dist_boundary)
            travel_dist = max(travel_dist, epsilon)

            travel_steps += 1

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y:
                energy_deposited[iy, ix] += energy_before - energy
                optical_depth = sigma_a * travel_dist
                if optical_depth > 0.01:
                    track_length[iy, ix] += energy_before * (1.0 - np.exp(-optical_depth)) / sigma_a
                else:
                    track_length[iy, ix] += energy_before * travel_dist
            else:
                warning_flags += 2

            pos += travel_dist * direction
            steps_this_particle += 1

            if energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < boundary_epsilon

            if is_boundary_event:
                is_left = pos[0] <= boundary_epsilon
                is_right = pos[0] >= total_width - boundary_epsilon
                is_bottom = pos[1] <= boundary_epsilon
                is_top = pos[1] >= total_height - boundary_epsilon

                if is_left or is_right or is_bottom or is_top:
                    if not reflected_boundaries:
                        if is_left:
                            exited_left += 1
                        elif is_right:
                            exited_right += 1
                        elif is_bottom:
                            exited_bottom += 1
                        elif is_top:
                            exited_top += 1
                        finished = True
                    else:
                        boundary_reflections += 1
                        if is_left or is_right:
                            direction[0] = -direction[0]
                            pos[0] = boundary_epsilon if is_left else total_width - boundary_epsilon
                        if is_bottom or is_top:
                            direction[1] = -direction[1]
                            pos[1] = boundary_epsilon if is_bottom else total_height - boundary_epsilon
                else:
                    pos[0] += direction[0] * epsilon
                    pos[1] += direction[1] * epsilon
            else:
                u_new, v_new, w_new = sample_scattered_direction_3d_from_unit(
                    direction[0],
                    direction[1],
                    direction[2],
                    np.random.random(),
                    np.random.random(),
                    scattering_mode_code,
                    epsilon,
                )
                u_new, v_new = guard_projected_direction_xy(u_new, v_new, epsilon)
                direction[0] = u_new
                direction[1] = v_new
                direction[2] = w_new

            if steps_this_particle >= max_steps_per_particle:
                warning_flags += 4
                absorbed += 1
                finished = True

        if not finished:
            absorbed += 1
            warning_flags += 8

    stats = (exited_left, exited_right, exited_bottom, exited_top,
             absorbed, travel_steps, boundary_reflections, warning_flags)
    return energy_deposited, track_length, stats


@njit
def multi_cell_mc_internal_source_2d_njit(
    num_cells_x, num_cells_y, cell_widths, cell_heights,
    cell_boundaries_x, cell_boundaries_y, total_width, total_height,
    num_particles,
    start_cell_indices_tuple, start_position_relative_tuple,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflected_boundaries, scattering_mode_code, randomize_start_pos=False
):
    """2D multi-cell MC, internal point-in-cell source. Uses per-cell XS arrays.

    For homogeneous materials, pass constant arrays built from scatter_ratio.
    """
    start_ix, start_iy = start_cell_indices_tuple
    start_rx, start_ry = start_position_relative_tuple

    energy_deposited = np.zeros((num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_cells_y, num_cells_x), dtype=np.float64)
    exited_left, exited_right, exited_bottom, exited_top = 0, 0, 0, 0
    absorbed = 0
    travel_steps = 0
    boundary_reflections = 0
    epsilon = 1e-9
    boundary_epsilon = epsilon * 10.0
    warning_flags = 0

    for _ in range(num_particles):
        energy = 1.0

        if randomize_start_pos:
            particle_rx = np.random.rand()
            particle_ry = np.random.rand()
        else:
            particle_rx = start_rx
            particle_ry = start_ry

        start_pos_x = cell_boundaries_x[start_ix] + particle_rx * cell_widths[start_ix]
        start_pos_y = cell_boundaries_y[start_iy] + particle_ry * cell_heights[start_iy]
        pos = np.array([start_pos_x, start_pos_y], dtype=np.float64)

        while True:
            u0, v0, w0 = sample_isotropic_direction_3d_from_unit(np.random.random(), np.random.random(), epsilon)
            u0, v0 = guard_projected_direction_xy(u0, v0, epsilon)
            if abs(u0) > epsilon or abs(v0) > epsilon:
                break
        direction = np.array([u0, v0, w0], dtype=np.float64)

        finished = False
        max_steps_per_particle = 1000000
        steps_this_particle = 0

        while not finished and steps_this_particle < max_steps_per_particle:
            ix, iy = find_cell_indices_2d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, num_cells_x, num_cells_y
            )

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y:
                sigma_t = sigma_t_per_cell[iy, ix]
                sigma_a = sigma_a_per_cell[iy, ix]
                sigma_s = sigma_t - sigma_a
                if sigma_s < 0.0:
                    sigma_s = 0.0
            else:
                sigma_t = 1.0
                sigma_a = 0.0
                sigma_s = 1.0
                warning_flags += 2

            dist_boundary = np.inf
            u, v = direction[0], direction[1]

            if abs(u) > epsilon:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]) / u
                if dist_x > epsilon:
                    dist_boundary = min(dist_boundary, dist_x)

            if abs(v) > epsilon:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]) / v
                if dist_y > epsilon:
                    dist_boundary = min(dist_boundary, dist_y)

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > epsilon else np.inf
            travel_dist = min(free_path, dist_boundary)
            travel_dist = max(travel_dist, epsilon)

            travel_steps += 1

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y:
                energy_deposited[iy, ix] += energy_before - energy
                optical_depth = sigma_a * travel_dist
                if optical_depth > 0.01:
                    track_length[iy, ix] += energy_before * (1.0 - np.exp(-optical_depth)) / sigma_a
                else:
                    track_length[iy, ix] += energy_before * travel_dist
            else:
                warning_flags += 2

            pos += travel_dist * direction
            steps_this_particle += 1

            if energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < boundary_epsilon

            if is_boundary_event:
                is_left = pos[0] <= boundary_epsilon
                is_right = pos[0] >= total_width - boundary_epsilon
                is_bottom = pos[1] <= boundary_epsilon
                is_top = pos[1] >= total_height - boundary_epsilon

                if is_left or is_right or is_bottom or is_top:
                    if not reflected_boundaries:
                        if is_left:
                            exited_left += 1
                        elif is_right:
                            exited_right += 1
                        elif is_bottom:
                            exited_bottom += 1
                        elif is_top:
                            exited_top += 1
                        finished = True
                    else:
                        boundary_reflections += 1
                        if is_left or is_right:
                            direction[0] = -direction[0]
                            pos[0] = boundary_epsilon if is_left else total_width - boundary_epsilon
                        if is_bottom or is_top:
                            direction[1] = -direction[1]
                            pos[1] = boundary_epsilon if is_bottom else total_height - boundary_epsilon
                else:
                    pos[0] += direction[0] * epsilon
                    pos[1] += direction[1] * epsilon
            else:
                u_new, v_new, w_new = sample_scattered_direction_3d_from_unit(
                    direction[0],
                    direction[1],
                    direction[2],
                    np.random.random(),
                    np.random.random(),
                    scattering_mode_code,
                    epsilon,
                )
                u_new, v_new = guard_projected_direction_xy(u_new, v_new, epsilon)
                direction[0] = u_new
                direction[1] = v_new
                direction[2] = w_new

            if steps_this_particle >= max_steps_per_particle:
                warning_flags += 4
                absorbed += 1
                finished = True

        if not finished:
            absorbed += 1
            warning_flags += 8

    stats = (exited_left, exited_right, exited_bottom, exited_top,
             absorbed, travel_steps, boundary_reflections, warning_flags)
    return energy_deposited, track_length, stats


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def run_mc_simulation_2d_njit_wrapper(
    source_mode,
    num_cells_x, num_cells_y, cell_widths, cell_heights,
    cell_boundaries_x, cell_boundaries_y, total_width, total_height,
    num_particles,
    start_cell_indices, start_position_relative,
    scatter_ratio, energy_threshold, seed,
    reflected_boundaries, angular_sampling=1,
    sigma_t_per_cell=None, sigma_a_per_cell=None,
    randomize_position_in_cell=False, verbose=True,
    scattering_mode="isotropic",
):
    """Run 2D multi-cell MC simulation.

    Pass sigma_t_per_cell / sigma_a_per_cell for heterogeneous materials;
    otherwise they are constructed from scatter_ratio (homogeneous).
    """
    # Keep global extents consistent with mesh boundaries to avoid boundary
    # misclassification from scalar-total roundoff mismatches.
    mesh_total_width = float(cell_boundaries_x[-1])
    mesh_total_height = float(cell_boundaries_y[-1])

    # Build per-cell arrays if not provided (homogeneous case).
    if sigma_t_per_cell is None or sigma_a_per_cell is None:
        sigma_t_per_cell = np.ones((num_cells_y, num_cells_x), dtype=np.float64)
        sigma_a_per_cell = np.full((num_cells_y, num_cells_x),
                                   1.0 - scatter_ratio, dtype=np.float64)
    else:
        if sigma_t_per_cell.ndim == 1:
            sigma_t_per_cell = sigma_t_per_cell.reshape(num_cells_y, num_cells_x)
        if sigma_a_per_cell.ndim == 1:
            sigma_a_per_cell = sigma_a_per_cell.reshape(num_cells_y, num_cells_x)
        sigma_t_per_cell = sigma_t_per_cell.astype(np.float64)
        sigma_a_per_cell = sigma_a_per_cell.astype(np.float64)

    start_time = time.time()
    scattering_mode_code = steady_2d_scattering_mode_code(scattering_mode)

    with seeded_numpy_execution(seed):
        if source_mode == 'internal_start':
            if verbose:
                print("\nRunning 2D MC (Internal Source)...")
            if not (0 <= start_cell_indices[0] < num_cells_x and
                    0 <= start_cell_indices[1] < num_cells_y):
                raise ValueError(f"Start cell indices {start_cell_indices} out of bounds")
            if not (0.0 <= start_position_relative[0] <= 1.0 and
                    0.0 <= start_position_relative[1] <= 1.0):
                raise ValueError(f"Start position relative {start_position_relative} must be in [0,1]")

            energy_deposited_raw, track_length_raw, stats = multi_cell_mc_internal_source_2d_njit(
                num_cells_x, num_cells_y, cell_widths, cell_heights,
                cell_boundaries_x, cell_boundaries_y, mesh_total_width, mesh_total_height,
                num_particles,
                tuple(start_cell_indices), tuple(start_position_relative),
                sigma_t_per_cell, sigma_a_per_cell,
                energy_threshold,
                reflected_boundaries, scattering_mode_code, randomize_position_in_cell
            )

        elif source_mode == 'boundary_only':
            if verbose:
                print("\nRunning 2D MC (Boundary Source)...")
            energy_deposited_raw, track_length_raw, stats = multi_cell_mc_boundary_source_2d_njit(
                num_cells_x, num_cells_y, cell_widths, cell_heights,
                cell_boundaries_x, cell_boundaries_y, mesh_total_width, mesh_total_height,
                num_particles,
                sigma_t_per_cell, sigma_a_per_cell,
                energy_threshold,
                reflected_boundaries, angular_sampling, scattering_mode_code
            )
        else:
            raise ValueError(f"Unsupported source_mode for 2D MC: {source_mode}")

    end_time = time.time()
    if verbose:
        print(f"2D MC took {end_time - start_time:.3f} seconds.")

    exited_left, exited_right, exited_bottom, exited_top, absorbed, travel_steps, boundary_reflections, warning_flags = stats

    if verbose:
        bc_type_str = "Reflected BC" if reflected_boundaries else "Exiting BC"
        print(f"\n2D MC ({source_mode}, {bc_type_str}) statistics:")
        print(f"  Exited left:    {exited_left} ({exited_left/num_particles:.2%})")
        print(f"  Exited right:   {exited_right} ({exited_right/num_particles:.2%})")
        print(f"  Exited bottom:  {exited_bottom} ({exited_bottom/num_particles:.2%})")
        print(f"  Exited top:     {exited_top} ({exited_top/num_particles:.2%})")
        print(f"  Absorbed:       {absorbed} ({absorbed/num_particles:.2%})")
        print(f"  Avg steps:      {travel_steps/num_particles:.2f}")
        total_accounted = exited_left + exited_right + exited_bottom + exited_top + absorbed
        if total_accounted != num_particles:
            print(f"  WARNING: {num_particles - total_accounted} particles unaccounted for!")
        if warning_flags > 0:
            print(f"  WARNING: warning_flags={warning_flags}")

    return energy_deposited_raw / num_particles, track_length_raw / num_particles, stats


def plot_2d_energy_deposition(energy_deposited, cell_boundaries_x, cell_boundaries_y,
                               title="2D Energy Deposition"):
    """Plot 2D energy deposition using pcolormesh with log colour scale."""
    X, Y = np.meshgrid(cell_boundaries_x, cell_boundaries_y)

    valid_energy = energy_deposited[energy_deposited > 0]
    if valid_energy.size == 0:
        print("Warning: No positive energy deposition found.")
        vmin, vmax = 1e-9, 1e-8
    else:
        vmin = valid_energy.min()
        vmax = energy_deposited.max()

    if vmin >= vmax:
        vmin = vmax * 0.1

    plot_data = np.maximum(energy_deposited, vmin * 0.1)

    plt.figure()
    plt.pcolormesh(X, Y, plot_data, cmap='jet', shading='flat',
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(label='Average Energy Deposited per Source Particle')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(title, fontsize=10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(cell_boundaries_x[0], cell_boundaries_x[-1])
    plt.ylim(cell_boundaries_y[0], cell_boundaries_y[-1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
    plt.tight_layout()
    plt.savefig("2dmc_energy_deposition_plot.png", dpi=300)
    plt.show()
