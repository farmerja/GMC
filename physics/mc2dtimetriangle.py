"""Triangle-specific 2D Monte Carlo transport helpers.

This module keeps the existing single-cell, time-dependent triangle samplers used
for ML data generation and adds a multi-cell triangle-mesh transport path that
matches the style of the other `physics/` solvers:

  * explicit Numba kernel
  * Python wrapper with time-profile handling
  * optional homogeneous material setup via `scatter_ratio`
  * helper utilities for triangle-mesh adjacency and structured mesh creation
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit


_TRIANGLE_EDGE_VERTEX_IDS = ((0, 2), (0, 1), (1, 2))


@njit(inline="always")
def _log_uniform_from_unit(u: float, log_min: float, log_max: float) -> float:
    return 10.0 ** (log_min + u * (log_max - log_min))


@njit(inline="always")
def _sample_triangle_barycentric() -> tuple[float, float]:
    xi1 = np.random.rand()
    xi2 = np.random.rand()
    if xi1 + xi2 > 1.0:
        xi1 = 1.0 - xi1
        xi2 = 1.0 - xi2
    return xi1, xi2


@njit(inline="always")
def _sample_isotropic_dir_uv(eps: float) -> tuple[float, float]:
    mu_z = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    t_val = np.sqrt(max(0.0, 1.0 - mu_z * mu_z))
    u = t_val * np.cos(phi)
    v = t_val * np.sin(phi)
    if abs(u) < eps and abs(v) < eps:
        u = eps
    return u, v


@njit(inline="always")
def _sample_incoming_flux_edge_dir_from_unit(
    unit_normal: float,
    unit_phi: float,
    nx: float,
    ny: float,
    tx: float,
    ty: float,
    angular_sampling: int,
    eps: float,
) -> tuple[float, float]:
    if angular_sampling == 1:
        mu_n = np.sqrt(unit_normal)
    else:
        mu_n = unit_normal
    if mu_n < eps:
        mu_n = eps
    phi = 2.0 * np.pi * unit_phi
    tangential = np.sqrt(max(0.0, 1.0 - mu_n * mu_n)) * np.cos(phi)
    u = mu_n * nx + tangential * tx
    v = mu_n * ny + tangential * ty
    if abs(u) < eps and abs(v) < eps:
        u = nx * eps
        v = ny * eps
    return u, v


@njit(inline="always")
def _sample_incoming_flux_edge_dir(
    nx: float,
    ny: float,
    tx: float,
    ty: float,
    angular_sampling: int,
    eps: float,
) -> tuple[float, float]:
    return _sample_incoming_flux_edge_dir_from_unit(
        np.random.rand(),
        np.random.rand(),
        nx,
        ny,
        tx,
        ty,
        angular_sampling,
        eps,
    )


@njit(inline="always")
def _triangle_contains_reference(xi1: float, xi2: float, tol: float) -> bool:
    return xi1 >= -tol and xi2 >= -tol and (xi1 + xi2) <= 1.0 + tol


@njit(inline="always")
def _clamp_triangle_reference(xi1: float, xi2: float) -> tuple[float, float]:
    xi1 = min(max(xi1, 0.0), 1.0)
    xi2 = min(max(xi2, 0.0), 1.0)
    xi_sum = xi1 + xi2
    if xi_sum > 1.0:
        if xi_sum > 0.0:
            inv_sum = 1.0 / xi_sum
            xi1 *= inv_sum
            xi2 *= inv_sum
        else:
            xi1 = 0.0
            xi2 = 0.0
    return xi1, xi2


@njit(inline="always")
def _snap_triangle_reference_to_edge(
    xi1: float,
    xi2: float,
    edge_local: int,
) -> tuple[float, float]:
    if edge_local == 0:
        xi1 = 0.0
    elif edge_local == 1:
        xi2 = 0.0
    else:
        diff = 0.5 * (xi1 + xi2 - 1.0)
        xi1 -= diff
        xi2 -= diff
    return _clamp_triangle_reference(xi1, xi2)


@njit(inline="always")
def _reference_to_physical(
    cell_idx: int,
    xi1: float,
    xi2: float,
    cell_origins: np.ndarray,
    cell_edge_vectors: np.ndarray,
) -> tuple[float, float]:
    origin_x = cell_origins[cell_idx, 0]
    origin_y = cell_origins[cell_idx, 1]
    e1x = cell_edge_vectors[cell_idx, 0, 0]
    e1y = cell_edge_vectors[cell_idx, 0, 1]
    e2x = cell_edge_vectors[cell_idx, 1, 0]
    e2y = cell_edge_vectors[cell_idx, 1, 1]
    return origin_x + xi1 * e1x + xi2 * e2x, origin_y + xi1 * e1y + xi2 * e2y


@njit(inline="always")
def _physical_to_reference(
    cell_idx: int,
    pos_x: float,
    pos_y: float,
    cell_origins: np.ndarray,
    cell_inv_basis: np.ndarray,
) -> tuple[float, float]:
    dx = pos_x - cell_origins[cell_idx, 0]
    dy = pos_y - cell_origins[cell_idx, 1]
    xi1 = cell_inv_basis[cell_idx, 0, 0] * dx + cell_inv_basis[cell_idx, 0, 1] * dy
    xi2 = cell_inv_basis[cell_idx, 1, 0] * dx + cell_inv_basis[cell_idx, 1, 1] * dy
    return xi1, xi2


@njit(inline="always")
def _reference_velocity(
    cell_idx: int,
    u: float,
    v: float,
    cell_inv_basis: np.ndarray,
) -> tuple[float, float]:
    return (
        cell_inv_basis[cell_idx, 0, 0] * u + cell_inv_basis[cell_idx, 0, 1] * v,
        cell_inv_basis[cell_idx, 1, 0] * u + cell_inv_basis[cell_idx, 1, 1] * v,
    )


@njit(inline="always")
def _triangle_boundary_distance(
    xi1: float,
    xi2: float,
    v_ref_1: float,
    v_ref_2: float,
    eps: float,
) -> tuple[float, int, float, float, float]:
    s0 = np.inf
    s1 = np.inf
    s2 = np.inf

    if v_ref_1 < -eps:
        s0 = -xi1 / v_ref_1
    if v_ref_2 < -eps:
        s1 = -xi2 / v_ref_2

    v_ref_sum = v_ref_1 + v_ref_2
    if v_ref_sum > eps:
        s2 = (1.0 - xi1 - xi2) / v_ref_sum

    edge_local = 0
    dist_boundary = s0
    if s1 < dist_boundary:
        dist_boundary = s1
        edge_local = 1
    if s2 < dist_boundary:
        dist_boundary = s2
        edge_local = 2

    return dist_boundary, edge_local, s0, s1, s2


@njit(inline="always")
def _reflect_direction_about_edge(
    u: float,
    v: float,
    cell_idx: int,
    edge_local: int,
    cell_edge_vectors: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    e1x = cell_edge_vectors[cell_idx, 0, 0]
    e1y = cell_edge_vectors[cell_idx, 0, 1]
    e2x = cell_edge_vectors[cell_idx, 1, 0]
    e2y = cell_edge_vectors[cell_idx, 1, 1]

    if edge_local == 0:
        tx = e2x
        ty = e2y
    elif edge_local == 1:
        tx = e1x
        ty = e1y
    else:
        tx = e2x - e1x
        ty = e2y - e1y

    t_norm = np.sqrt(tx * tx + ty * ty)
    if t_norm < eps:
        return -u, -v

    tx /= t_norm
    ty /= t_norm
    dot = u * tx + v * ty
    return 2.0 * dot * tx - u, 2.0 * dot * ty - v


@njit
def _find_triangle_cell_for_position(
    pos_x: float,
    pos_y: float,
    cell_origins: np.ndarray,
    cell_inv_basis: np.ndarray,
    tol: float,
) -> int:
    num_cells = cell_origins.shape[0]
    for cell_idx in range(num_cells):
        xi1, xi2 = _physical_to_reference(cell_idx, pos_x, pos_y, cell_origins, cell_inv_basis)
        if _triangle_contains_reference(xi1, xi2, tol):
            return cell_idx
    return -1


@njit
def simulate_single_cell_2d_triangle_timedep_internal(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    min_L: float = 0.001,
    max_L: float = 10.0,
    min_r: float = 0.001,
    max_r: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
):
    """2D single-cell time-dependent internal source in the reference triangle."""
    data = np.zeros((num_neutrons, 16), dtype=np.float64)
    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9

    log_min_L = np.log10(min_L)
    log_max_L = np.log10(max_L)
    log_min_r = np.log10(min_r)
    log_max_r = np.log10(max_r)
    log_min_budget = np.log10(min_path_budget)
    log_max_budget = np.log10(max_path_budget)

    use_precond = cond_samples is not None

    for i in range(num_neutrons):
        cand_L = cond_samples[i, 0] if use_precond else np.random.rand()
        cand_r = cond_samples[i, 1] if use_precond else np.random.rand()
        cand_theta = cond_samples[i, 2] if use_precond else np.random.rand()
        cand_b = cond_samples[i, 3] if use_precond else np.random.rand()
        L_tri = _log_uniform_from_unit(cand_L, log_min_L, log_max_L)
        r_tri = _log_uniform_from_unit(cand_r, log_min_r, log_max_r)
        theta = 0.01 + cand_theta * (np.pi - 0.02)
        path_budget = _log_uniform_from_unit(cand_b, log_min_budget, log_max_budget)

        if use_precond:
            xi1_in = cond_samples[i, 4]
            xi2_in = cond_samples[i, 5]
            if xi1_in + xi2_in > 1.0:
                xi1_in = 1.0 - xi1_in
                xi2_in = 1.0 - xi2_in
        else:
            xi1_in, xi2_in = _sample_triangle_barycentric()

        if use_precond:
            mu_z = 2.0 * cond_samples[i, 6] - 1.0
            phi = 2.0 * np.pi * cond_samples[i, 7]
            t_val = np.sqrt(max(0.0, 1.0 - mu_z * mu_z))
            u_in = t_val * np.cos(phi)
            v_in = t_val * np.sin(phi)
            if abs(u_in) < eps and abs(v_in) < eps:
                u_in = eps
        else:
            u_in, v_in = _sample_isotropic_dir_uv(eps)

        data[i, 0:8] = (L_tri, r_tri, theta, path_budget, xi1_in, xi2_in, u_in, v_in)

        L_track = 0.0
        space_exit = 0
        time_exit = 0
        absorbed = 0
        alive = True

        xi1 = xi1_in
        xi2 = xi2_in
        cot_theta = np.cos(theta) / np.sin(theta)
        sin_theta = np.sin(theta)
        u = u_in
        v = v_in

        while alive:
            remaining_budget = path_budget - L_track
            if remaining_budget <= 0.0:
                time_exit = 1
                alive = False
                break

            free_path = -np.log(np.random.rand()) / sigma_t

            v_ref_1 = u / L_tri - v * cot_theta / L_tri
            v_ref_2 = v / (r_tri * sin_theta)
            s_exit, edge_local, _, _, _ = _triangle_boundary_distance(xi1, xi2, v_ref_1, v_ref_2, eps)

            step_dist = max(0.0, min(free_path, s_exit, remaining_budget))
            xi1 += step_dist * v_ref_1
            xi2 += step_dist * v_ref_2
            L_track += step_dist

            boundary_hit = abs(step_dist - s_exit) < eps * s_exit + eps
            budget_hit = abs(step_dist - remaining_budget) < eps * remaining_budget + eps

            if boundary_hit:
                xi1, xi2 = _snap_triangle_reference_to_edge(xi1, xi2, edge_local)
                space_exit = 1
                alive = False
            elif budget_hit:
                xi1, xi2 = _clamp_triangle_reference(xi1, xi2)
                time_exit = 1
                alive = False
            else:
                if np.random.random() < sigma_a_prob:
                    xi1, xi2 = _clamp_triangle_reference(xi1, xi2)
                    absorbed = 1
                    alive = False
                else:
                    u, v = _sample_isotropic_dir_uv(eps)

        data[i, 8] = xi1
        data[i, 9] = xi2
        data[i, 10] = u
        data[i, 11] = v
        data[i, 12] = L_track
        data[i, 13] = float(space_exit)
        data[i, 14] = float(time_exit)
        data[i, 15] = float(absorbed)

    return data


@njit
def simulate_single_cell_2d_triangle_timedep_boundary(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    min_L: float = 0.001,
    max_L: float = 10.0,
    min_r: float = 0.001,
    max_r: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
    angular_sampling: int = 1,
):
    """2D single-cell time-dependent boundary source in the reference triangle.

    Boundary injection uses the base edge `xi2 = 0` with inward normal `+y`.
    """
    data = np.zeros((num_neutrons, 16), dtype=np.float64)
    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9

    log_min_L = np.log10(min_L)
    log_max_L = np.log10(max_L)
    log_min_r = np.log10(min_r)
    log_max_r = np.log10(max_r)
    log_min_budget = np.log10(min_path_budget)
    log_max_budget = np.log10(max_path_budget)

    use_precond = cond_samples is not None

    for i in range(num_neutrons):
        cand_L = cond_samples[i, 0] if use_precond else np.random.rand()
        cand_r = cond_samples[i, 1] if use_precond else np.random.rand()
        cand_theta = cond_samples[i, 2] if use_precond else np.random.rand()
        cand_b = cond_samples[i, 3] if use_precond else np.random.rand()
        L_tri = _log_uniform_from_unit(cand_L, log_min_L, log_max_L)
        r_tri = _log_uniform_from_unit(cand_r, log_min_r, log_max_r)
        theta = 0.01 + cand_theta * (np.pi - 0.02)
        path_budget = _log_uniform_from_unit(cand_b, log_min_budget, log_max_budget)

        if use_precond:
            xi1_in = cond_samples[i, 4]
        else:
            xi1_in = np.random.rand()
        xi2_in = 0.0

        if use_precond:
            u_in, v_in = _sample_incoming_flux_edge_dir_from_unit(
                cond_samples[i, 5],
                cond_samples[i, 6],
                0.0,
                1.0,
                1.0,
                0.0,
                angular_sampling,
                eps,
            )
        else:
            u_in, v_in = _sample_incoming_flux_edge_dir(
                0.0,
                1.0,
                1.0,
                0.0,
                angular_sampling,
                eps,
            )

        data[i, 0:8] = (L_tri, r_tri, theta, path_budget, xi1_in, xi2_in, u_in, v_in)

        L_track = 0.0
        space_exit = 0
        time_exit = 0
        absorbed = 0
        alive = True

        xi1 = xi1_in
        xi2 = xi2_in
        sin_theta = np.sin(theta)
        if sin_theta < eps:
            sin_theta = eps
        cot_theta = np.cos(theta) / sin_theta
        u = u_in
        v = v_in

        while alive:
            remaining_budget = path_budget - L_track
            if remaining_budget <= 0.0:
                time_exit = 1
                alive = False
                break

            free_path = -np.log(np.random.rand()) / sigma_t

            v_ref_1 = u / L_tri - v * cot_theta / L_tri
            v_ref_2 = v / (r_tri * sin_theta)
            s_exit, edge_local, _, _, _ = _triangle_boundary_distance(xi1, xi2, v_ref_1, v_ref_2, eps)

            step_dist = max(0.0, min(free_path, s_exit, remaining_budget))
            xi1 += step_dist * v_ref_1
            xi2 += step_dist * v_ref_2
            L_track += step_dist

            boundary_hit = abs(step_dist - s_exit) < eps * s_exit + eps
            budget_hit = abs(step_dist - remaining_budget) < eps * remaining_budget + eps

            if boundary_hit:
                xi1, xi2 = _snap_triangle_reference_to_edge(xi1, xi2, edge_local)
                space_exit = 1
                alive = False
            elif budget_hit:
                xi1, xi2 = _clamp_triangle_reference(xi1, xi2)
                time_exit = 1
                alive = False
            else:
                if np.random.random() < sigma_a_prob:
                    xi1, xi2 = _clamp_triangle_reference(xi1, xi2)
                    absorbed = 1
                    alive = False
                else:
                    u, v = _sample_isotropic_dir_uv(eps)

        data[i, 8] = xi1
        data[i, 9] = xi2
        data[i, 10] = u
        data[i, 11] = v
        data[i, 12] = L_track
        data[i, 13] = float(space_exit)
        data[i, 14] = float(time_exit)
        data[i, 15] = float(absorbed)

    return data


@njit
def multi_cell_mc_time_dependent_2d_triangle_njit(
    cell_origins,
    cell_edge_vectors,
    cell_inv_basis,
    neighbor_cells,
    boundary_edge_ids,
    boundary_edge_side,
    num_boundary_edges,
    num_particles,
    start_cell_indices,
    start_barycentric_tuple,
    sigma_t_per_cell,
    sigma_a_per_cell,
    energy_threshold,
    reflected_boundaries,
    particle_velocity,
    max_time,
    num_time_bins,
    start_times,
    randomize_start_pos=False,
):
    """Time-dependent 2D Monte Carlo transport on a triangle mesh."""
    num_cells = cell_origins.shape[0]
    dt = max_time / num_time_bins

    energy_deposited = np.zeros((num_time_bins, num_cells), dtype=np.float64)
    track_length = np.zeros((num_time_bins, num_cells), dtype=np.float64)
    boundary_flux = np.zeros((num_time_bins, num_boundary_edges), dtype=np.float64)

    exited_boundary = 0
    absorbed = 0
    timed_out = 0
    travel_steps = 0
    internal_edge_crossings = 0
    boundary_reflections = 0
    warning_flags = 0
    exit_side_counts = np.zeros(5, dtype=np.int64)

    eps = 1e-9
    boundary_eps = eps * 10.0
    containment_tol = 1e-7

    start_xi1 = start_barycentric_tuple[0]
    start_xi2 = start_barycentric_tuple[1]

    for n_idx in range(num_particles):
        current_time = start_times[n_idx]
        if current_time >= max_time:
            continue

        current_cell = start_cell_indices[n_idx]
        if randomize_start_pos:
            xi1, xi2 = _sample_triangle_barycentric()
        else:
            xi1 = start_xi1
            xi2 = start_xi2

        pos_x, pos_y = _reference_to_physical(
            current_cell, xi1, xi2, cell_origins, cell_edge_vectors
        )
        u, v = _sample_isotropic_dir_uv(eps)
        energy = 1.0

        finished = False
        steps_this_particle = 0
        max_steps_per_particle = 1_000_000

        while (not finished) and (steps_this_particle < max_steps_per_particle):
            if current_time >= max_time:
                timed_out += 1
                finished = True
                continue

            if current_cell < 0 or current_cell >= num_cells:
                warning_flags += 1
                absorbed += 1
                finished = True
                continue

            sigma_t = sigma_t_per_cell[current_cell]
            sigma_a = sigma_a_per_cell[current_cell]
            sigma_s = sigma_t - sigma_a
            if sigma_s < 0.0:
                sigma_s = 0.0

            v_ref_1, v_ref_2 = _reference_velocity(current_cell, u, v, cell_inv_basis)
            dist_boundary, edge_local, _, _, _ = _triangle_boundary_distance(
                xi1, xi2, v_ref_1, v_ref_2, eps
            )

            current_time_bin = int(current_time / dt)
            if current_time_bin < 0:
                current_time_bin = 0
            if current_time_bin >= num_time_bins:
                current_time_bin = num_time_bins - 1
            next_time_bin_edge = (current_time_bin + 1) * dt
            time_to_next_bin = next_time_bin_edge - current_time
            dist_to_time_boundary = time_to_next_bin * particle_velocity

            if sigma_s > eps:
                free_path = -np.log(1.0 - np.random.rand()) / sigma_s
            else:
                free_path = np.inf

            travel_dist = min(free_path, dist_boundary, dist_to_time_boundary)
            if travel_dist < eps:
                travel_dist = eps

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

            energy_deposited[time_bin_idx, current_cell] += deposited

            optical_depth = sigma_a * travel_dist
            if optical_depth > 0.01 and sigma_a > 0.0:
                tl = energy_before * (1.0 - np.exp(-optical_depth)) / sigma_a
            else:
                tl = energy_before * travel_dist
            track_length[time_bin_idx, current_cell] += tl

            pos_x += travel_dist * u
            pos_y += travel_dist * v
            xi1 += travel_dist * v_ref_1
            xi2 += travel_dist * v_ref_2
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

            is_boundary_event = abs(travel_dist - dist_boundary) < boundary_eps
            is_time_bin_crossing = abs(travel_dist - dist_to_time_boundary) < boundary_eps

            if is_boundary_event:
                xi1, xi2 = _snap_triangle_reference_to_edge(xi1, xi2, edge_local)
                neighbor = neighbor_cells[current_cell, edge_local]

                if neighbor >= 0:
                    internal_edge_crossings += 1
                    pos_x += u * eps
                    pos_y += v * eps
                    next_cell = neighbor
                    xi1_new, xi2_new = _physical_to_reference(
                        next_cell, pos_x, pos_y, cell_origins, cell_inv_basis
                    )
                    if not _triangle_contains_reference(xi1_new, xi2_new, containment_tol):
                        fallback_cell = _find_triangle_cell_for_position(
                            pos_x, pos_y, cell_origins, cell_inv_basis, containment_tol
                        )
                        if fallback_cell < 0:
                            warning_flags += 16
                            absorbed += 1
                            finished = True
                            continue
                        next_cell = fallback_cell
                        xi1_new, xi2_new = _physical_to_reference(
                            next_cell, pos_x, pos_y, cell_origins, cell_inv_basis
                        )
                    current_cell = next_cell
                    xi1, xi2 = _clamp_triangle_reference(xi1_new, xi2_new)
                else:
                    boundary_id = boundary_edge_ids[current_cell, edge_local]
                    exit_time_bin = int(current_time / dt)
                    if exit_time_bin < 0:
                        exit_time_bin = 0
                    if exit_time_bin >= num_time_bins:
                        exit_time_bin = num_time_bins - 1

                    if reflected_boundaries:
                        u, v = _reflect_direction_about_edge(
                            u, v, current_cell, edge_local, cell_edge_vectors, eps
                        )
                        boundary_reflections += 1
                        pos_x += u * eps
                        pos_y += v * eps
                        xi1, xi2 = _physical_to_reference(
                            current_cell, pos_x, pos_y, cell_origins, cell_inv_basis
                        )
                        xi1, xi2 = _clamp_triangle_reference(xi1, xi2)
                    else:
                        if boundary_id >= 0:
                            boundary_flux[exit_time_bin, boundary_id] += energy
                            side = int(boundary_edge_side[boundary_id])
                            if side < 0 or side >= exit_side_counts.shape[0]:
                                side = 4
                            exit_side_counts[side] += 1
                        else:
                            exit_side_counts[4] += 1
                        exited_boundary += 1
                        finished = True
            elif is_time_bin_crossing:
                pass
            else:
                u, v = _sample_isotropic_dir_uv(eps)

            if steps_this_particle >= max_steps_per_particle:
                warning_flags += 4
                absorbed += 1
                finished = True

        if not finished:
            absorbed += 1
            warning_flags += 8

    stats = (
        exited_boundary,
        absorbed,
        timed_out,
        travel_steps,
        internal_edge_crossings,
        boundary_reflections,
        warning_flags,
        exit_side_counts[0],
        exit_side_counts[1],
        exit_side_counts[2],
        exit_side_counts[3],
        exit_side_counts[4],
    )
    return energy_deposited, track_length, boundary_flux, stats


def _normalize_triangle_start_cell_indices(start_cell_indices, num_particles: int) -> np.ndarray:
    particle_count = max(int(num_particles), 0)
    if np.isscalar(start_cell_indices):
        return np.full(particle_count, int(start_cell_indices), dtype=np.int64)

    arr = np.asarray(start_cell_indices, dtype=np.int64).reshape(-1)
    if arr.size == 1:
        return np.full(particle_count, int(arr[0]), dtype=np.int64)
    if arr.size != particle_count:
        raise ValueError(
            "Triangle mesh transport expects start_cell_indices to be either a scalar "
            f"or length num_particles={particle_count}; received shape {arr.shape}."
        )
    return arr


def _prepare_triangle_cell_geometry(
    vertices: np.ndarray,
    cell_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_cells = cell_vertices.shape[0]
    cell_origins = vertices[cell_vertices[:, 0]].astype(np.float64, copy=True)
    cell_edge_vectors = np.zeros((num_cells, 2, 2), dtype=np.float64)
    cell_inv_basis = np.zeros((num_cells, 2, 2), dtype=np.float64)

    for cell_idx in range(num_cells):
        v0 = vertices[cell_vertices[cell_idx, 0]]
        v1 = vertices[cell_vertices[cell_idx, 1]]
        v2 = vertices[cell_vertices[cell_idx, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) <= 1e-12:
            raise ValueError(f"Triangle cell {cell_idx} is degenerate.")

        cell_edge_vectors[cell_idx, 0] = e1
        cell_edge_vectors[cell_idx, 1] = e2
        inv_det = 1.0 / det
        cell_inv_basis[cell_idx, 0, 0] = inv_det * e2[1]
        cell_inv_basis[cell_idx, 0, 1] = -inv_det * e2[0]
        cell_inv_basis[cell_idx, 1, 0] = -inv_det * e1[1]
        cell_inv_basis[cell_idx, 1, 1] = inv_det * e1[0]

    return cell_origins, cell_edge_vectors, cell_inv_basis


def calculate_triangle_mesh_adjacency(
    cell_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build triangle-triangle adjacency and boundary-edge ids.

    Parameters
    ----------
    cell_vertices:
        `(num_cells, 3)` integer array of vertex ids. The local edge ordering is
        tied to the single-cell reference triangle:
          `edge 0 -> xi1 = 0`,
          `edge 1 -> xi2 = 0`,
          `edge 2 -> xi1 + xi2 = 1`.

    Returns
    -------
    neighbor_cells:
        `(num_cells, 3)` with `-1` for domain-boundary edges.
    boundary_edge_ids:
        `(num_cells, 3)` with `-1` for internal edges and a dense boundary-edge id
        for external edges.
    boundary_edges:
        `(num_boundary_edges, 4)` rows of
        `[cell_idx, local_edge, vertex_a, vertex_b]`.
    """
    cell_vertices = np.asarray(cell_vertices, dtype=np.int64)
    if cell_vertices.ndim != 2 or cell_vertices.shape[1] != 3:
        raise ValueError(
            f"Expected cell_vertices with shape (num_cells, 3); got {cell_vertices.shape}."
        )

    num_cells = cell_vertices.shape[0]
    neighbor_cells = np.full((num_cells, 3), -1, dtype=np.int64)
    boundary_edge_ids = np.full((num_cells, 3), -1, dtype=np.int64)

    edge_to_owner: dict[tuple[int, int], tuple[int, int]] = {}
    for cell_idx in range(num_cells):
        tri = cell_vertices[cell_idx]
        if len({int(tri[0]), int(tri[1]), int(tri[2])}) != 3:
            raise ValueError(f"Triangle cell {cell_idx} repeats a vertex: {tri.tolist()}.")
        for local_edge, (ia, ib) in enumerate(_TRIANGLE_EDGE_VERTEX_IDS):
            va = int(tri[ia])
            vb = int(tri[ib])
            edge_key = (va, vb) if va < vb else (vb, va)
            owner = edge_to_owner.pop(edge_key, None)
            if owner is None:
                edge_to_owner[edge_key] = (cell_idx, local_edge)
            else:
                other_cell, other_local_edge = owner
                neighbor_cells[cell_idx, local_edge] = other_cell
                neighbor_cells[other_cell, other_local_edge] = cell_idx

    boundary_rows: list[list[int]] = []
    for cell_idx in range(num_cells):
        tri = cell_vertices[cell_idx]
        for local_edge, (ia, ib) in enumerate(_TRIANGLE_EDGE_VERTEX_IDS):
            if neighbor_cells[cell_idx, local_edge] >= 0:
                continue
            boundary_edge_ids[cell_idx, local_edge] = len(boundary_rows)
            boundary_rows.append(
                [cell_idx, local_edge, int(tri[ia]), int(tri[ib])]
            )

    if boundary_rows:
        boundary_edges = np.asarray(boundary_rows, dtype=np.int64)
    else:
        boundary_edges = np.zeros((0, 4), dtype=np.int64)

    return neighbor_cells, boundary_edge_ids, boundary_edges


def _build_boundary_metadata_from_neighbors(
    cell_vertices: np.ndarray,
    neighbor_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_cells = cell_vertices.shape[0]
    boundary_edge_ids = np.full((num_cells, 3), -1, dtype=np.int64)
    boundary_rows: list[list[int]] = []

    for cell_idx in range(num_cells):
        tri = cell_vertices[cell_idx]
        for local_edge, (ia, ib) in enumerate(_TRIANGLE_EDGE_VERTEX_IDS):
            if neighbor_cells[cell_idx, local_edge] >= 0:
                continue
            boundary_edge_ids[cell_idx, local_edge] = len(boundary_rows)
            boundary_rows.append(
                [cell_idx, local_edge, int(tri[ia]), int(tri[ib])]
            )

    if boundary_rows:
        boundary_edges = np.asarray(boundary_rows, dtype=np.int64)
    else:
        boundary_edges = np.zeros((0, 4), dtype=np.int64)

    return boundary_edge_ids, boundary_edges


def _classify_triangle_boundary_edge_sides(
    vertices: np.ndarray,
    boundary_edges: np.ndarray,
    tol: float = 1.0e-9,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    if boundary_edges.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    out = np.full(boundary_edges.shape[0], 4, dtype=np.int64)
    for bid, row in enumerate(boundary_edges):
        pa = vertices[int(row[2])]
        pb = vertices[int(row[3])]
        mx = 0.5 * (pa[0] + pb[0])
        my = 0.5 * (pa[1] + pb[1])
        if abs(mx - x_min) <= tol:
            out[bid] = 0
        elif abs(mx - x_max) <= tol:
            out[bid] = 1
        elif abs(my - y_max) <= tol:
            out[bid] = 2
        elif abs(my - y_min) <= tol:
            out[bid] = 3
    return out


def build_structured_triangle_mesh_2d(
    cell_widths,
    cell_heights,
    diagonal: str = "backslash",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a structured triangle mesh by splitting each Cartesian cell in two."""
    cell_widths = np.asarray(cell_widths, dtype=np.float64).reshape(-1)
    cell_heights = np.asarray(cell_heights, dtype=np.float64).reshape(-1)
    if cell_widths.size == 0 or cell_heights.size == 0:
        raise ValueError("cell_widths and cell_heights must be non-empty.")

    diagonal_key = diagonal.strip().lower()
    if diagonal_key not in {"backslash", "slash"}:
        raise ValueError("diagonal must be either 'backslash' or 'slash'.")

    boundaries_x = np.concatenate(([0.0], np.cumsum(cell_widths)))
    boundaries_y = np.concatenate(([0.0], np.cumsum(cell_heights)))
    num_cells_x = cell_widths.size
    num_cells_y = cell_heights.size

    vertices = np.zeros(((num_cells_x + 1) * (num_cells_y + 1), 2), dtype=np.float64)

    def vertex_id(ix: int, iy: int) -> int:
        return iy * (num_cells_x + 1) + ix

    for iy in range(num_cells_y + 1):
        for ix in range(num_cells_x + 1):
            vid = vertex_id(ix, iy)
            vertices[vid, 0] = boundaries_x[ix]
            vertices[vid, 1] = boundaries_y[iy]

    cell_rows: list[list[int]] = []
    for iy in range(num_cells_y):
        for ix in range(num_cells_x):
            bl = vertex_id(ix, iy)
            br = vertex_id(ix + 1, iy)
            tl = vertex_id(ix, iy + 1)
            tr = vertex_id(ix + 1, iy + 1)

            if diagonal_key == "backslash":
                cell_rows.append([bl, br, tl])
                cell_rows.append([br, tr, tl])
            else:
                cell_rows.append([bl, br, tr])
                cell_rows.append([bl, tr, tl])

    cell_vertices = np.asarray(cell_rows, dtype=np.int64)
    neighbor_cells, boundary_edge_ids, boundary_edges = calculate_triangle_mesh_adjacency(
        cell_vertices
    )
    return vertices, cell_vertices, neighbor_cells, boundary_edge_ids, boundary_edges


def run_mc_time_dependent_2d_triangle_wrapper(
    vertices,
    cell_vertices,
    num_particles,
    start_cell_indices=0,
    start_position_relative=(0.25, 0.25),
    scatter_ratio: float = 1.0,
    sigma_t_per_cell=None,
    sigma_a_per_cell=None,
    energy_threshold: float = 1e-12,
    reflected_boundaries: bool = False,
    particle_velocity: float = 1.0,
    max_time: float = 1.0,
    num_time_bins: int = 100,
    source_time_profile: str = "pulse",
    pulse_time: float = 0.0,
    gaussian_mean: float = 1.0,
    gaussian_std: float = 0.2,
    step_on_time: float = 0.0,
    step_off_time=None,
    neighbor_cells=None,
    boundary_edge_ids=None,
    seed=None,
    randomize_position_in_cell: bool = False,
    verbose: bool = True,
):
    """Run time-dependent MC transport on a triangle mesh.

    Parameters mirror the rectangular time-dependent wrapper where practical.
    `start_position_relative` is interpreted as reference-triangle barycentric
    coordinates `(xi1, xi2)` inside `start_cell_indices`, which may be either a
    single start cell id or one start cell id per particle.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    cell_vertices = np.asarray(cell_vertices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"Expected vertices with shape (num_vertices, 2); got {vertices.shape}.")
    if cell_vertices.ndim != 2 or cell_vertices.shape[1] != 3:
        raise ValueError(
            f"Expected cell_vertices with shape (num_cells, 3); got {cell_vertices.shape}."
        )

    num_cells = cell_vertices.shape[0]
    start_cell_indices = _normalize_triangle_start_cell_indices(start_cell_indices, num_particles)
    if start_cell_indices.size > 0:
        start_cell_min = int(start_cell_indices.min())
        start_cell_max = int(start_cell_indices.max())
        if start_cell_min < 0 or start_cell_max >= num_cells:
            raise ValueError(
                f"Start cell indices out of bounds for {num_cells} cells: "
                f"min={start_cell_min}, max={start_cell_max}."
            )

    bary = np.asarray(start_position_relative, dtype=np.float64).reshape(-1)
    if bary.size != 2:
        raise ValueError(
            "Triangle start_position_relative must contain exactly two barycentric "
            f"coordinates; got {start_position_relative!r}."
        )
    if bary[0] < 0.0 or bary[1] < 0.0 or bary[0] + bary[1] > 1.0:
        raise ValueError(
            "Triangle start_position_relative must satisfy xi1 >= 0, xi2 >= 0, "
            f"xi1 + xi2 <= 1; got {tuple(map(float, bary))}."
        )

    cell_origins, cell_edge_vectors, cell_inv_basis = _prepare_triangle_cell_geometry(
        vertices, cell_vertices
    )

    if neighbor_cells is None:
        neighbor_cells, auto_boundary_edge_ids, boundary_edges = calculate_triangle_mesh_adjacency(
            cell_vertices
        )
        if boundary_edge_ids is None:
            boundary_edge_ids = auto_boundary_edge_ids
    else:
        neighbor_cells = np.asarray(neighbor_cells, dtype=np.int64)
        if neighbor_cells.shape != (num_cells, 3):
            raise ValueError(
                f"neighbor_cells must have shape {(num_cells, 3)}; got {neighbor_cells.shape}."
            )
        auto_boundary_edge_ids, boundary_edges = _build_boundary_metadata_from_neighbors(
            cell_vertices, neighbor_cells
        )
        if boundary_edge_ids is None:
            boundary_edge_ids = auto_boundary_edge_ids

    boundary_edge_ids = np.asarray(boundary_edge_ids, dtype=np.int64)
    if boundary_edge_ids.shape != (num_cells, 3):
        raise ValueError(
            f"boundary_edge_ids must have shape {(num_cells, 3)}; got {boundary_edge_ids.shape}."
        )
    valid_boundary_ids = boundary_edge_ids[boundary_edge_ids >= 0]
    if valid_boundary_ids.size == 0:
        num_boundary_edges = 0
    else:
        num_boundary_edges = int(valid_boundary_ids.max()) + 1
    boundary_edge_side = _classify_triangle_boundary_edge_sides(vertices, boundary_edges)

    if sigma_t_per_cell is None or sigma_a_per_cell is None:
        sigma_t_per_cell = np.ones(num_cells, dtype=np.float64)
        sigma_a_per_cell = np.full(num_cells, 1.0 - scatter_ratio, dtype=np.float64)
    else:
        sigma_t_per_cell = np.asarray(sigma_t_per_cell, dtype=np.float64).reshape(-1)
        sigma_a_per_cell = np.asarray(sigma_a_per_cell, dtype=np.float64).reshape(-1)
        if sigma_t_per_cell.size != num_cells or sigma_a_per_cell.size != num_cells:
            raise ValueError(
                "sigma_t_per_cell and sigma_a_per_cell must each have one value per "
                f"triangle cell ({num_cells}); got {sigma_t_per_cell.size} and "
                f"{sigma_a_per_cell.size}."
            )

    if seed is not None:
        np.random.seed(seed)

    if source_time_profile == "pulse":
        start_times = np.full(num_particles, pulse_time, dtype=np.float64)
    elif source_time_profile == "gaussian":
        start_times = np.random.normal(gaussian_mean, gaussian_std, num_particles)
        start_times = np.clip(start_times, 0.0, max_time)
    elif source_time_profile == "continuous":
        start_times = np.random.uniform(0.0, max_time, num_particles)
    elif source_time_profile == "step":
        off_time = max_time if step_off_time is None else step_off_time
        start_times = np.random.uniform(step_on_time, off_time, num_particles)
    else:
        start_times = np.zeros(num_particles, dtype=np.float64)

    if verbose:
        print(
            f"Running 2D triangle-mesh time-dependent MC: N={num_particles}, "
            f"cells={num_cells}, boundary_edges={num_boundary_edges}, "
            f"max_time={max_time}, bins={num_time_bins}, reflected={reflected_boundaries}"
        )

    t0 = time.time()
    out = multi_cell_mc_time_dependent_2d_triangle_njit(
        cell_origins.astype(np.float64),
        cell_edge_vectors.astype(np.float64),
        cell_inv_basis.astype(np.float64),
        neighbor_cells.astype(np.int64),
        boundary_edge_ids.astype(np.int64),
        boundary_edge_side.astype(np.int64),
        int(num_boundary_edges),
        int(num_particles),
        start_cell_indices.astype(np.int64),
        (float(bary[0]), float(bary[1])),
        sigma_t_per_cell.astype(np.float64),
        sigma_a_per_cell.astype(np.float64),
        float(energy_threshold),
        bool(reflected_boundaries),
        float(particle_velocity),
        float(max_time),
        int(num_time_bins),
        start_times.astype(np.float64),
        randomize_start_pos=bool(randomize_position_in_cell),
    )
    t1 = time.time()

    if verbose:
        stats = np.asarray(out[3], dtype=np.int64)
        exited_boundary, absorbed, timed_out, travel_steps, internal_crossings, boundary_reflections, warning_flags = stats[:7]
        print(f"2D triangle-mesh MC took {t1 - t0:.3f} s.")
        print("Triangle-mesh particle statistics:")
        print(f"  Exited boundary:        {exited_boundary} ({exited_boundary / max(num_particles, 1):.2%})")
        print(f"  Absorbed:               {absorbed} ({absorbed / max(num_particles, 1):.2%})")
        print(f"  Timed out:              {timed_out} ({timed_out / max(num_particles, 1):.2%})")
        if stats.size >= 12:
            print(
                "  Exit sides:             "
                f"L={stats[7]} R={stats[8]} T={stats[9]} B={stats[10]} O={stats[11]}"
            )
        print(f"  Internal edge crossings:{internal_crossings}")
        print(f"  Boundary reflections:   {boundary_reflections}")
        print(f"  Avg steps:              {travel_steps / max(num_particles, 1):.2f}")
        total_accounted = exited_boundary + absorbed + timed_out
        if total_accounted != num_particles:
            print(f"  WARNING: {num_particles - total_accounted} particles unaccounted for!")
        if warning_flags > 0:
            print(f"  WARNING: warning_flags={warning_flags}")

    return out


__all__ = [
    "simulate_single_cell_2d_triangle_timedep_internal",
    "simulate_single_cell_2d_triangle_timedep_boundary",
    "calculate_triangle_mesh_adjacency",
    "build_structured_triangle_mesh_2d",
    "multi_cell_mc_time_dependent_2d_triangle_njit",
    "run_mc_time_dependent_2d_triangle_wrapper",
]
