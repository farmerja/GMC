"""mc2dtime.py - 2D time-dependent Monte Carlo transport.

This module provides:
  * Single-cell 2D, time-dependent (path-length budget) MC
    - Internal source (uniform in cell)
    - Boundary source (x=0 incoming) with correct cosine-law injection
  * Multi-cell 2D time-dependent MC (heterogeneous via per-cell cross sections)

Key refactors / fixes vs prior version:
  * One single-cell kernel handles both internal and boundary sources via `source_type`.
  * Unified output schema for internal/boundary (boundary uses initial_pos_x=0).
  * Boundary angular sampling is selectable for ML-vs-physics use cases:
      - angular_sampling=1 -> cosine-weighted in normal component (physical flux)
      - angular_sampling=0 -> uniform in normal component (better feature coverage)
  * Minor robustness: clamp remaining budget to avoid negative due to roundoff.

Notes on "2D" angular sampling:
  This implementation treats directions as the (u,v) components of a 3D unit vector.
  That is the standard reduction for 2D geometries that are invariant in the third
  dimension: particles can still have an out-of-plane component, which manifests as
  |(u,v)| <= 1. If you want a strictly 2D-in-plane physics model (|(u,v)|==1),
  you can change `_sample_isotropic_dir_uv()` accordingly.
"""

from __future__ import annotations

import time
import numpy as np
from numba import njit, prange

from .mc1d import sample_unit
from .mc2d import find_cell_indices_2d_njit

# Source-type flags (Numba-friendly ints)
SRC_INTERNAL = 0          # uniform in cell
SRC_BOUNDARY_X0 = 1       # boundary at x=0, incoming (+x)


# ---------------------------------------------------------------------------
# Small helpers (Numba-friendly)
# ---------------------------------------------------------------------------

@njit(inline='always')
def _log_uniform_from_unit(u: float, log_min: float, log_max: float) -> float:
    return 10.0 ** (log_min + u * (log_max - log_min))


@njit(inline='always')
def _sample_isotropic_dir_uv(eps: float) -> tuple[float, float]:
    """Sample (u,v) as the in-plane components of an isotropic 3D direction.

    Implementation: sample mu_z ~ Uniform[-1,1], phi ~ Uniform[0,2pi), then
    u = sin(theta)*cos(phi), v = sin(theta)*sin(phi) where sin(theta)=sqrt(1-mu_z^2).
    """
    mu_z = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    t = np.sqrt(max(0.0, 1.0 - mu_z * mu_z))
    u = t * np.cos(phi)
    v = t * np.sin(phi)
    # Avoid both exactly zero (rare, but can cause dist_boundary==inf everywhere)
    if abs(u) < eps and abs(v) < eps:
        u = eps
    return u, v


@njit(inline='always')
def _sample_isotropic_dir_uv_from_unit(unit_mu_z: float, unit_phi: float, eps: float) -> tuple[float, float]:
    mu_z = 2.0 * unit_mu_z - 1.0
    phi = 2.0 * np.pi * unit_phi
    t = np.sqrt(max(0.0, 1.0 - mu_z * mu_z))
    u = t * np.cos(phi)
    v = t * np.sin(phi)
    if abs(u) < eps and abs(v) < eps:
        u = eps
    return u, v


@njit(inline='always')
def _sample_incoming_flux_plus_x_uv(angular_sampling: int, eps: float) -> tuple[float, float]:
    """Sample incoming flux directions to x=0 (normal +x), returning (u,v).

    angular_sampling:
      1 -> cosine-weighted in u (physical incoming flux)
      0 -> uniform in u (better ML feature-space coverage)
    """
    if angular_sampling == 1:
        u = np.sqrt(np.random.random())  # cosine-law in normal component
    else:
        u = np.random.random()           # uniform normal component
    if u < eps:
        u = eps
    phi = np.random.uniform(0.0, 2.0 * np.pi)  # azimuth about +x
    s = np.sqrt(max(0.0, 1.0 - u * u))
    v = s * np.cos(phi)  # y-component
    return u, v


@njit(inline='always')
def _sample_incoming_flux_plus_x_uv_from_unit(
    unit_u: float,
    unit_phi: float,
    angular_sampling: int,
    eps: float,
) -> tuple[float, float]:
    if angular_sampling == 1:
        u = np.sqrt(unit_u)
    else:
        u = unit_u
    if u < eps:
        u = eps
    phi = 2.0 * np.pi * unit_phi
    s = np.sqrt(max(0.0, 1.0 - u * u))
    v = s * np.cos(phi)
    return u, v


# ---------------------------------------------------------------------------
# Single-cell 2D time-dependent (path-length budget formulation)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def simulate_single_cell_2d_time_dependent(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
    source_type: int = SRC_INTERNAL,
    angular_sampling: int = 1,
):
    """2D single-cell time-dependent MC (path-length budget) with selectable source.

    Parameters
    ----------
    source_type:
        SRC_INTERNAL (0): (x,y) uniform in [0,W]x[0,H], direction isotropic (3D projected)
        SRC_BOUNDARY_X0 (1): x=0, y uniform in [0,H], direction sampled as incoming flux
    angular_sampling:
        Boundary-source sampling only:
          1 -> cosine-weighted in u (physical)
          0 -> uniform in u (ML coverage)

    Returns
    -------
    data : (N, 16) float64
        Columns:
          [0] width, [1] height, [2] path_budget,
          [3] initial_pos_x, [4] initial_pos_y,
          [5] initial_dir_u, [6] initial_dir_v,
          [7] final_pos_x, [8] final_pos_y,
          [9] final_dir_u, [10] final_dir_v,
          [11] L, [12] num_interactions,
          [13] space_exit, [14] time_exit, [15] absorbed
    """
    data = np.zeros((num_neutrons, 16), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9

    use_precond = cond_samples is not None

    log_min_w = np.log10(min_width)
    log_max_w = np.log10(max_width)
    log_min_h = np.log10(min_height)
    log_max_h = np.log10(max_height)
    log_min_budget = np.log10(min_path_budget)
    log_max_budget = np.log10(max_path_budget)

    for i in prange(num_neutrons):
        # --- Conditioning variables ---
        cand_w = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        W = _log_uniform_from_unit(cand_w, log_min_w, log_max_w)
        cand_h = cond_samples[i, 1] if use_precond else sample_unit(distribution)
        H = _log_uniform_from_unit(cand_h, log_min_h, log_max_h)
        cand_b = cond_samples[i, 2] if use_precond else sample_unit(distribution)
        path_budget = _log_uniform_from_unit(cand_b, log_min_budget, log_max_budget)

        data[i, 0] = W
        data[i, 1] = H
        data[i, 2] = path_budget

        # --- Initial state ---
        if source_type == SRC_INTERNAL:
            if use_precond:
                pos_x = W * cond_samples[i, 3]
                pos_y = H * cond_samples[i, 4]
                u, v = _sample_isotropic_dir_uv_from_unit(cond_samples[i, 5], cond_samples[i, 6], eps)
            else:
                pos_x = np.random.uniform(0.0, W)
                pos_y = np.random.uniform(0.0, H)
                u, v = _sample_isotropic_dir_uv(eps)
        else:
            pos_x = 0.0
            if use_precond:
                pos_y = H * cond_samples[i, 3]
                u, v = _sample_incoming_flux_plus_x_uv_from_unit(cond_samples[i, 4], cond_samples[i, 5], angular_sampling, eps)
            else:
                pos_y = np.random.uniform(0.0, H)
                u, v = _sample_incoming_flux_plus_x_uv(angular_sampling, eps)

        data[i, 3] = pos_x
        data[i, 4] = pos_y
        data[i, 5] = u
        data[i, 6] = v

        L = 0.0
        num_interactions = 0.0
        space_exit = 0.0
        time_exit = 0.0
        absorbed = 0.0

        # --- Random walk ---
        while True:
            remaining_budget = path_budget - L
            if remaining_budget <= 0.0:
                time_exit = 1.0
                break

            free_path = -np.log(np.random.random()) / sigma_t

            dist_x_right = (W - pos_x) / u if u > eps else np.inf
            dist_x_left = (0.0 - pos_x) / u if u < -eps else np.inf
            dist_y_top = (H - pos_y) / v if v > eps else np.inf
            dist_y_bot = (0.0 - pos_y) / v if v < -eps else np.inf

            dist_boundary = dist_x_right
            if dist_x_left < dist_boundary:
                dist_boundary = dist_x_left
            if dist_y_top < dist_boundary:
                dist_boundary = dist_y_top
            if dist_y_bot < dist_boundary:
                dist_boundary = dist_y_bot

            step = free_path
            if dist_boundary < step:
                step = dist_boundary
            if remaining_budget < step:
                step = remaining_budget
            if step < 0.0:
                step = 0.0

            pos_x += step * u
            pos_y += step * v
            L += step

            boundary_hit = abs(step - dist_boundary) < eps * dist_boundary + eps
            budget_hit = abs(step - remaining_budget) < eps * remaining_budget + eps

            # Boundary precedence if both occur simultaneously
            if boundary_hit:
                # Snap / clamp
                if abs(pos_x - W) < eps * W + eps:
                    pos_x = W
                elif abs(pos_x) < eps * W + eps:
                    pos_x = 0.0
                if abs(pos_y - H) < eps * H + eps:
                    pos_y = H
                elif abs(pos_y) < eps * H + eps:
                    pos_y = 0.0

                pos_x = min(max(pos_x, 0.0), W)
                pos_y = min(max(pos_y, 0.0), H)
                space_exit = 1.0
                break

            if budget_hit:
                pos_x = min(max(pos_x, 0.0), W)
                pos_y = min(max(pos_y, 0.0), H)
                time_exit = 1.0
                break

            # Collision
            num_interactions += 1.0
            if np.random.random() < sigma_a_prob:
                pos_x = min(max(pos_x, 0.0), W)
                pos_y = min(max(pos_y, 0.0), H)
                absorbed = 1.0
                break

            # Scatter: isotropic (3D projected)
            u, v = _sample_isotropic_dir_uv(eps)

        # --- Final state ---
        data[i, 7] = pos_x
        data[i, 8] = pos_y
        data[i, 9] = u
        data[i, 10] = v
        data[i, 11] = L
        data[i, 12] = num_interactions
        data[i, 13] = space_exit
        data[i, 14] = time_exit
        data[i, 15] = absorbed

    return data


# ---------------------------------------------------------------------------
# Backwards-compatible convenience wrapper (boundary mode)
# ---------------------------------------------------------------------------

def simulate_single_cell_2d_time_dependent_boundary(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
    angular_sampling: int = 1,
):
    """Deprecated: kept for API compatibility.

    Use:
        simulate_single_cell_2d_time_dependent(..., source_type=SRC_BOUNDARY_X0)

    Returns the *same* 16-column schema as simulate_single_cell_2d_time_dependent.
    """
    return simulate_single_cell_2d_time_dependent(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
        source_type=SRC_BOUNDARY_X0,
        angular_sampling=angular_sampling,
    )


# ---------------------------------------------------------------------------
# Multi-cell 2D time-dependent (kept; hetero/homo handled via sigma arrays)
# ---------------------------------------------------------------------------

@njit
def multi_cell_mc_time_dependent_2d_njit(
    num_cells_x, num_cells_y, cell_widths, cell_heights,
    cell_boundaries_x, cell_boundaries_y, total_width, total_height,
    num_particles,
    start_cell_indices_tuple, start_position_relative_tuple,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflected_boundaries,
    particle_velocity,
    max_time,
    num_time_bins,
    start_times,
    randomize_start_pos=False,
    source_type=SRC_INTERNAL,
    source_y_min=-1.0,
    source_y_max=-1.0,
    detector_right_y_min=-1.0,
    detector_right_y_max=-1.0,
):
    """Time-dependent 2D multi-cell MC with per-cell materials.

    Notes
    -----
    Homogeneous media are a special case of heterogeneous:
      - set sigma_t_per_cell[:,:] = constant
      - set sigma_a_per_cell[:,:] = constant

    Returns
    -------
    energy_deposited : (t,y,x)
    track_length     : (t,y,x)
    flux_left/right/bottom/top : (t,)
    stats : tuple of counters
    """
    start_ix, start_iy = start_cell_indices_tuple
    start_rx, start_ry = start_position_relative_tuple

    dt = max_time / num_time_bins

    energy_deposited = np.zeros((num_time_bins, num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_time_bins, num_cells_y, num_cells_x), dtype=np.float64)

    flux_left = np.zeros(num_time_bins, dtype=np.float64)
    flux_right = np.zeros(num_time_bins, dtype=np.float64)
    flux_bottom = np.zeros(num_time_bins, dtype=np.float64)
    flux_top = np.zeros(num_time_bins, dtype=np.float64)
    detector_right = np.zeros(num_time_bins, dtype=np.float64)

    exited_left = 0
    exited_right = 0
    exited_bottom = 0
    exited_top = 0
    absorbed = 0
    timed_out = 0
    travel_steps = 0
    boundary_reflections = 0
    epsilon = 1e-9
    boundary_epsilon = epsilon * 10.0
    warning_flags = 0

    for n_idx in range(num_particles):
        energy = 1.0
        current_time = start_times[n_idx]

        if current_time >= max_time:
            continue

        if randomize_start_pos:
            particle_ry = np.random.rand()
            if source_type == SRC_BOUNDARY_X0:
                particle_rx = 0.0
            else:
                particle_rx = np.random.rand()
        else:
            particle_rx = start_rx
            particle_ry = start_ry

        if source_type == SRC_BOUNDARY_X0 and source_y_max > source_y_min:
            start_pos_x = cell_boundaries_x[0]
            start_pos_y = source_y_min + np.random.rand() * (source_y_max - source_y_min)
            start_pos_y = min(max(start_pos_y, 0.0), total_height - epsilon)
        elif source_type == SRC_BOUNDARY_X0:
            start_pos_x = cell_boundaries_x[start_ix]
            start_pos_y = cell_boundaries_y[start_iy] + particle_ry * cell_heights[start_iy]
        else:
            start_pos_x = cell_boundaries_x[start_ix] + particle_rx * cell_widths[start_ix]
            start_pos_y = cell_boundaries_y[start_iy] + particle_ry * cell_heights[start_iy]
        pos = np.array([start_pos_x, start_pos_y], dtype=np.float64)

        # Random starting direction depends on source type.
        while True:
            if source_type == SRC_BOUNDARY_X0:
                u0, v0 = _sample_incoming_flux_plus_x_uv(1, epsilon)
            else:
                u0, v0 = _sample_isotropic_dir_uv(epsilon)
            if abs(u0) > epsilon or abs(v0) > epsilon:
                break
        direction = np.array([u0, v0], dtype=np.float64)

        finished = False
        max_steps_per_particle = 1_000_000
        steps_this_particle = 0

        while (not finished) and (steps_this_particle < max_steps_per_particle):
            if current_time >= max_time:
                timed_out += 1
                finished = True
                continue

            ix, iy = find_cell_indices_2d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, num_cells_x, num_cells_y
            )

            # If indices are invalid, terminate defensively (prevents OOB in boundaries arrays)
            if ix < 0 or ix >= num_cells_x or iy < 0 or iy >= num_cells_y:
                warning_flags += 16
                absorbed += 1
                finished = True
                continue

            sigma_t = sigma_t_per_cell[iy, ix]
            sigma_a = sigma_a_per_cell[iy, ix]
            sigma_s = sigma_t - sigma_a
            if sigma_s < 0.0:
                sigma_s = 0.0

            dist_boundary = np.inf
            u = direction[0]
            v = direction[1]

            # Distance to the next cell boundary in x/y
            if abs(u) > epsilon:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]
                ) / u
                if dist_x > epsilon:
                    dist_boundary = min(dist_boundary, dist_x)

            if abs(v) > epsilon:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]
                ) / v
                if dist_y > epsilon:
                    dist_boundary = min(dist_boundary, dist_y)

            # Time-bin boundary in distance units
            current_time_bin = int(current_time / dt)
            if current_time_bin >= num_time_bins:
                current_time_bin = num_time_bins - 1
            next_time_bin_edge = (current_time_bin + 1) * dt
            time_to_next_bin = next_time_bin_edge - current_time
            dist_to_time_boundary = time_to_next_bin * particle_velocity

            # Scatter-only collision distance (implicit capture style)
            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > epsilon else np.inf

            travel_dist = min(free_path, dist_boundary, dist_to_time_boundary)
            if travel_dist < epsilon:
                travel_dist = epsilon

            travel_steps += 1
            travel_time = travel_dist / particle_velocity

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)
            deposited = energy_before - energy
            if deposited < 0.0:
                deposited = 0.0

            time_bin_idx = int(current_time / dt)
            if time_bin_idx < 0:
                time_bin_idx = 0
            if time_bin_idx >= num_time_bins:
                time_bin_idx = num_time_bins - 1

            energy_deposited[time_bin_idx, iy, ix] += deposited

            optical_depth = sigma_a * travel_dist
            if optical_depth > 0.01 and sigma_a > 0.0:
                tl = energy_before * (1.0 - np.exp(-optical_depth)) / sigma_a
            else:
                tl = energy_before * travel_dist
            track_length[time_bin_idx, iy, ix] += tl

            pos += travel_dist * direction
            current_time += travel_time
            steps_this_particle += 1

            if energy < energy_threshold:
                absorbed += 1
                finished = True
                continue

            if current_time >= max_time:
                timed_out += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < boundary_epsilon
            is_time_bin_crossing = abs(travel_dist - dist_to_time_boundary) < boundary_epsilon

            if is_boundary_event:
                x = pos[0]
                y = pos[1]
                exit_time_bin = int(current_time / dt)
                if exit_time_bin < 0:
                    exit_time_bin = 0
                if exit_time_bin >= num_time_bins:
                    exit_time_bin = num_time_bins - 1

                if x <= boundary_epsilon:
                    if reflected_boundaries:
                        pos[0] = boundary_epsilon
                        direction[0] = abs(direction[0])
                        boundary_reflections += 1
                    else:
                        flux_left[exit_time_bin] += energy
                        exited_left += 1
                        finished = True
                elif x >= total_width - boundary_epsilon:
                    if reflected_boundaries:
                        pos[0] = total_width - boundary_epsilon
                        direction[0] = -abs(direction[0])
                        boundary_reflections += 1
                    else:
                        flux_right[exit_time_bin] += energy
                        if detector_right_y_max > detector_right_y_min and y >= detector_right_y_min and y <= detector_right_y_max:
                            detector_right[exit_time_bin] += energy
                        exited_right += 1
                        finished = True
                elif y <= boundary_epsilon:
                    if reflected_boundaries:
                        pos[1] = boundary_epsilon
                        direction[1] = abs(direction[1])
                        boundary_reflections += 1
                    else:
                        flux_bottom[exit_time_bin] += energy
                        exited_bottom += 1
                        finished = True
                elif y >= total_height - boundary_epsilon:
                    if reflected_boundaries:
                        pos[1] = total_height - boundary_epsilon
                        direction[1] = -abs(direction[1])
                        boundary_reflections += 1
                    else:
                        flux_top[exit_time_bin] += energy
                        exited_top += 1
                        finished = True
                else:
                    # Internal cell boundary: nudge
                    pos += direction * epsilon
            elif is_time_bin_crossing:
                # Just crossed a time boundary; no physics event required.
                pass
            else:
                # Scatter: new isotropic direction (3D projected)
                u_new, v_new = _sample_isotropic_dir_uv(epsilon)
                direction[0] = u_new
                direction[1] = v_new

            if steps_this_particle >= max_steps_per_particle:
                warning_flags += 4
                absorbed += 1
                finished = True

        if not finished:
            absorbed += 1
            warning_flags += 8

    stats = (
        exited_left, exited_right, exited_bottom, exited_top,
        absorbed, timed_out, travel_steps, boundary_reflections, warning_flags
    )

    return (
        energy_deposited, track_length,
        flux_left, flux_right, flux_bottom, flux_top, detector_right, stats
    )


def run_mc_time_dependent_2d_wrapper(
    num_cells_x, num_cells_y, cell_widths, cell_heights,
    cell_boundaries_x, cell_boundaries_y, total_width, total_height,
    num_particles,
    start_cell_indices, start_position_relative,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflected_boundaries,
    particle_velocity, max_time, num_time_bins,
    source_time_profile='pulse',
    pulse_time=0.0,
    gaussian_mean=1.0, gaussian_std=0.2,
    step_on_time=0.0, step_off_time=None,
    seed=None,
    randomize_position_in_cell=False,
    source_type='internal',
    source_y_min=None,
    source_y_max=None,
    detector_right_y_min=None,
    detector_right_y_max=None,
    return_detector_right=False,
    verbose=True
):
    """Run time-dependent 2D multi-cell MC simulation."""
    if seed is not None:
        np.random.seed(seed)

    if source_time_profile == 'pulse':
        start_times = np.full(num_particles, pulse_time, dtype=np.float64)
    elif source_time_profile == 'gaussian':
        start_times = np.random.normal(gaussian_mean, gaussian_std, num_particles)
        start_times = np.clip(start_times, 0.0, max_time)
    elif source_time_profile == 'continuous':
        start_times = np.random.uniform(0.0, max_time, num_particles)
    elif source_time_profile == 'step':
        off_time = step_off_time if step_off_time is not None else max_time
        start_times = np.random.uniform(step_on_time, off_time, num_particles)
    else:
        start_times = np.zeros(num_particles, dtype=np.float64)

    start_times = start_times.astype(np.float64)

    # Keep global extents consistent with the mesh boundaries passed to the kernel.
    mesh_total_width = float(cell_boundaries_x[-1])
    mesh_total_height = float(cell_boundaries_y[-1])

    if verbose:
        print(
            f"Running 2D time-dependent multi-cell MC: N={num_particles}, "
            f"max_time={max_time}, bins={num_time_bins}, reflected={reflected_boundaries}, source_type={source_type}"
        )

    if source_type == 'internal':
        source_type_flag = SRC_INTERNAL
    elif source_type in ('boundary_x0', 'boundary'):
        source_type_flag = SRC_BOUNDARY_X0
    else:
        raise ValueError(f"Unsupported 2D time-dependent MC source_type: {source_type}")

    source_y_min_f = -1.0 if source_y_min is None else float(source_y_min)
    source_y_max_f = -1.0 if source_y_max is None else float(source_y_max)
    detector_right_y_min_f = -1.0 if detector_right_y_min is None else float(detector_right_y_min)
    detector_right_y_max_f = -1.0 if detector_right_y_max is None else float(detector_right_y_max)

    t0 = time.time()
    out = multi_cell_mc_time_dependent_2d_njit(
        int(num_cells_x), int(num_cells_y),
        cell_widths.astype(np.float64),
        cell_heights.astype(np.float64),
        cell_boundaries_x.astype(np.float64),
        cell_boundaries_y.astype(np.float64),
        mesh_total_width, mesh_total_height,
        int(num_particles),
        (int(start_cell_indices[0]), int(start_cell_indices[1])),
        (float(start_position_relative[0]), float(start_position_relative[1])),
        sigma_t_per_cell.astype(np.float64),
        sigma_a_per_cell.astype(np.float64),
        float(energy_threshold),
        bool(reflected_boundaries),
        float(particle_velocity),
        float(max_time),
        int(num_time_bins),
        start_times,
        randomize_start_pos=bool(randomize_position_in_cell),
        source_type=int(source_type_flag),
        source_y_min=float(source_y_min_f),
        source_y_max=float(source_y_max_f),
        detector_right_y_min=float(detector_right_y_min_f),
        detector_right_y_max=float(detector_right_y_max_f),
    )
    t1 = time.time()

    if verbose:
        print(f"2D time-dependent multi-cell MC took {t1 - t0:.3f} s.")

    energy_deposited, track_length, flux_left, flux_right, flux_bottom, flux_top, detector_right, stats = out
    if return_detector_right:
        return energy_deposited, track_length, flux_left, flux_right, flux_bottom, flux_top, detector_right, stats
    return energy_deposited, track_length, flux_left, flux_right, flux_bottom, flux_top, stats
