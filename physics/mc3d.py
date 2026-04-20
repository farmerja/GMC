"""mc3d.py - 3D steady-state Monte Carlo transport (single-cell and multi-cell mesh)."""

from __future__ import annotations

import time
import numpy as np
from numba import njit, prange

from ._seed import seeded_numpy_execution
from .mc1d import sample_unit
from .steady_2d_scattering import (
    resolve_steady_2d_hg_parameter,
    sample_scattered_direction_3d_from_unit,
    steady_2d_scattering_mode_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@njit
def _sample_isotropic_dir_3d(eps: float = 1e-12):
    """Return (u,v,w) isotropic on the unit sphere."""
    mu = np.random.uniform(-1.0, 1.0)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    t = np.sqrt(max(0.0, 1.0 - mu * mu))
    u = t * np.cos(phi)
    v = t * np.sin(phi)
    w = mu
    if abs(u) < eps and abs(v) < eps and abs(w) < eps:
        w = eps
    return u, v, w


@njit(inline="always")
def _sample_scattered_dir_3d(
    current_u: float,
    current_v: float,
    current_w: float,
    scattering_mode_code: int = 0,
    g_hg: float = 0.0,
    eps: float = 1.0e-12,
):
    return sample_scattered_direction_3d_from_unit(
        current_u,
        current_v,
        current_w,
        np.random.random(),
        np.random.random(),
        scattering_mode_code,
        g_hg,
        eps,
    )


def resolve_3d_scattering_parameters(
    scattering_mode: str = "isotropic",
    g_hg: float | None = None,
) -> tuple[str, int, float]:
    """Resolve user-facing 3D scattering configuration into kernel-friendly values."""
    mode = str(scattering_mode).strip().lower()
    mode_code = steady_2d_scattering_mode_code(mode)
    resolved_g_hg = resolve_steady_2d_hg_parameter(mode, g_hg)
    return mode, mode_code, resolved_g_hg


@njit
def _sample_isotropic_dir_3d_from_unit(unit_mu: float, unit_phi: float, eps: float = 1e-12):
    mu = 2.0 * unit_mu - 1.0
    phi = 2.0 * np.pi * unit_phi
    t = np.sqrt(max(0.0, 1.0 - mu * mu))
    u = t * np.cos(phi)
    v = t * np.sin(phi)
    w = mu
    if abs(u) < eps and abs(v) < eps and abs(w) < eps:
        w = eps
    return u, v, w


@njit
def _sample_incoming_hemisphere_plus_x(angular_sampling, eps):
    """Sample direction in +x hemisphere (u>0).

    angular_sampling:
      0 -> cosine-weighted (Lambert) about +x: mu_x = sqrt(xi)
      1 -> uniform hemisphere: mu_x = xi
    """
    xi = np.random.random()
    mu_x = np.sqrt(xi) if angular_sampling == 0 else xi
    mu_x = max(mu_x, eps)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    t = np.sqrt(max(0.0, 1.0 - mu_x * mu_x))
    u = mu_x
    v = t * np.cos(phi)
    w = t * np.sin(phi)
    return u, v, w


@njit
def _sample_incoming_hemisphere_plus_x_from_unit(unit_mu_x, unit_phi, angular_sampling, eps):
    mu_x = np.sqrt(unit_mu_x) if angular_sampling == 0 else unit_mu_x
    mu_x = max(mu_x, eps)
    phi = 2.0 * np.pi * unit_phi
    t = np.sqrt(max(0.0, 1.0 - mu_x * mu_x))
    u = mu_x
    v = t * np.cos(phi)
    w = t * np.sin(phi)
    return u, v, w


@njit
def _track_length_contribution(energy_before, sigma_a, travel_dist):
    """Track-length estimator consistent with continuous attenuation."""
    tau = sigma_a * travel_dist
    if tau > 0.01:
        return energy_before * (1.0 - np.exp(-tau)) / sigma_a
    return energy_before * travel_dist


def calculate_boundaries_3d(cell_widths, cell_heights, cell_depths):
    """Return (boundaries_x, boundaries_y, boundaries_z) from per-axis cell sizes."""
    bx = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(cell_widths)))
    by = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(cell_heights)))
    bz = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(cell_depths)))
    return bx, by, bz


@njit
def find_cell_indices_3d_njit(pos, cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                               num_cells_x, num_cells_y, num_cells_z):
    """Find (ix,iy,iz) indices for a position; clamps slightly inside bounds for robustness."""
    x = pos[0]
    y = pos[1]
    z = pos[2]

    total_x = cell_boundaries_x[-1]
    total_y = cell_boundaries_y[-1]
    total_z = cell_boundaries_z[-1]

    clip_eps = 1e-12
    x = np.minimum(np.maximum(x, clip_eps), total_x - clip_eps)
    y = np.minimum(np.maximum(y, clip_eps), total_y - clip_eps)
    z = np.minimum(np.maximum(z, clip_eps), total_z - clip_eps)

    ix = np.searchsorted(cell_boundaries_x, x, side="right") - 1
    iy = np.searchsorted(cell_boundaries_y, y, side="right") - 1
    iz = np.searchsorted(cell_boundaries_z, z, side="right") - 1

    if ix < 0:
        ix = 0
    elif ix >= num_cells_x:
        ix = num_cells_x - 1
    if iy < 0:
        iy = 0
    elif iy >= num_cells_y:
        iy = num_cells_y - 1
    if iz < 0:
        iz = 0
    elif iz >= num_cells_z:
        iz = num_cells_z - 1

    return ix, iy, iz


# ---------------------------------------------------------------------------
# Single-cell steady-state (boundary source and internal source)
# ---------------------------------------------------------------------------

@njit
def simulate_single_cell_3d_boundary_source(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    angular_sampling: int = 1,
    cond_samples=None,
    scattering_mode_code: int = 0,
    g_hg: float = 0.0,
):
    """3D boundary source. Particles enter from the x=0 plane.

    Returns array with columns:
      [0] width, [1] height, [2] depth,
      [3] y_in, [4] z_in,
      [5] u_in, [6] v_in, [7] w_in,
      [8] x_out, [9] y_out, [10] z_out,
      [11] u_out, [12] v_out, [13] w_out,
      [14] track_length, [15] exited, [16] absorbed
    """
    data = np.zeros((num_neutrons, 17), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    log_min_w = np.log10(min_width)
    log_max_w = np.log10(max_width)
    log_min_h = np.log10(min_height)
    log_max_h = np.log10(max_height)
    log_min_d = np.log10(min_depth)
    log_max_d = np.log10(max_depth)

    for i in prange(num_neutrons):
        W = 10.0 ** (log_min_w + (cond_samples[i, 0] if use_precond else sample_unit(distribution)) * (log_max_w - log_min_w))
        H = 10.0 ** (log_min_h + (cond_samples[i, 1] if use_precond else sample_unit(distribution)) * (log_max_h - log_min_h))
        D = 10.0 ** (log_min_d + (cond_samples[i, 2] if use_precond else sample_unit(distribution)) * (log_max_d - log_min_d))

        data[i, 0] = W
        data[i, 1] = H
        data[i, 2] = D

        x = 0.0
        if use_precond:
            y = H * cond_samples[i, 3]
            z = D * cond_samples[i, 4]
        else:
            y = np.random.uniform(0.0, H)
            z = np.random.uniform(0.0, D)
        data[i, 3] = y
        data[i, 4] = z

        if use_precond:
            u, v, w = _sample_incoming_hemisphere_plus_x_from_unit(cond_samples[i, 5], cond_samples[i, 6], angular_sampling, eps)
        else:
            u, v, w = _sample_incoming_hemisphere_plus_x(angular_sampling, eps)
        data[i, 5] = u
        data[i, 6] = v
        data[i, 7] = w

        total_path = 0.0
        exited = 0.0
        absorbed = 0.0

        while True:
            free_path = -np.log(np.random.random()) / sigma_t

            dist_x_right = (W - x) / u if u > eps else np.inf
            dist_x_left = (0.0 - x) / u if u < -eps else np.inf
            dist_y_top = (H - y) / v if v > eps else np.inf
            dist_y_bottom = (0.0 - y) / v if v < -eps else np.inf
            dist_z_front = (D - z) / w if w > eps else np.inf
            dist_z_back = (0.0 - z) / w if w < -eps else np.inf

            dist_boundary = min(
                dist_x_right, dist_x_left, dist_y_top, dist_y_bottom, dist_z_front, dist_z_back
            )
            step = max(0.0, min(free_path, dist_boundary))

            x += step * u
            y += step * v
            z += step * w
            total_path += step

            boundary_hit = abs(step - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(x - W) < eps * W + eps:
                    x = W
                elif abs(x - 0.0) < eps * W + eps:
                    x = 0.0
                if abs(y - H) < eps * H + eps:
                    y = H
                elif abs(y - 0.0) < eps * H + eps:
                    y = 0.0
                if abs(z - D) < eps * D + eps:
                    z = D
                elif abs(z - 0.0) < eps * D + eps:
                    z = 0.0

                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                z = min(max(z, 0.0), D)

                exited = 1.0
                data[i, 8] = x
                data[i, 9] = y
                data[i, 10] = z
                data[i, 11] = u
                data[i, 12] = v
                data[i, 13] = w
                data[i, 14] = total_path
                data[i, 15] = exited
                data[i, 16] = absorbed
                break

            if np.random.random() < sigma_a_prob:
                absorbed = 1.0
                data[i, 8] = x
                data[i, 9] = y
                data[i, 10] = z
                data[i, 11] = u
                data[i, 12] = v
                data[i, 13] = w
                data[i, 14] = total_path
                data[i, 15] = exited
                data[i, 16] = absorbed
                break

            u, v, w = sample_scattered_direction_3d_from_unit(
                u,
                v,
                w,
                np.random.random(),
                np.random.random(),
                scattering_mode_code,
                g_hg,
                eps,
            )

    return data


@njit
def simulate_single_cell_3d_internal_source(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    cond_samples=None,
    scattering_mode_code: int = 0,
    g_hg: float = 0.0,
):
    """3D internal source. Particles start uniformly within the volume.

    Returns array with columns:
      [0] width, [1] height, [2] depth,
      [3] x_in, [4] y_in, [5] z_in,
      [6] u_in, [7] v_in, [8] w_in,
      [9] x_out, [10] y_out, [11] z_out,
      [12] u_out, [13] v_out, [14] w_out,
      [15] track_length, [16] exited, [17] absorbed
    """
    data = np.zeros((num_neutrons, 18), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    log_min_w = np.log10(min_width)
    log_max_w = np.log10(max_width)
    log_min_h = np.log10(min_height)
    log_max_h = np.log10(max_height)
    log_min_d = np.log10(min_depth)
    log_max_d = np.log10(max_depth)

    for i in prange(num_neutrons):
        W = 10.0 ** (log_min_w + (cond_samples[i, 0] if use_precond else sample_unit(distribution)) * (log_max_w - log_min_w))
        H = 10.0 ** (log_min_h + (cond_samples[i, 1] if use_precond else sample_unit(distribution)) * (log_max_h - log_min_h))
        D = 10.0 ** (log_min_d + (cond_samples[i, 2] if use_precond else sample_unit(distribution)) * (log_max_d - log_min_d))

        data[i, 0] = W
        data[i, 1] = H
        data[i, 2] = D

        if use_precond:
            x = W * cond_samples[i, 3]
            y = H * cond_samples[i, 4]
            z = D * cond_samples[i, 5]
        else:
            x = np.random.uniform(0.0, W)
            y = np.random.uniform(0.0, H)
            z = np.random.uniform(0.0, D)
        data[i, 3] = x
        data[i, 4] = y
        data[i, 5] = z

        if use_precond:
            u, v, w = _sample_isotropic_dir_3d_from_unit(cond_samples[i, 6], cond_samples[i, 7], eps)
        else:
            u, v, w = _sample_isotropic_dir_3d(eps)
        data[i, 6] = u
        data[i, 7] = v
        data[i, 8] = w

        total_path = 0.0
        exited = 0.0
        absorbed = 0.0

        while True:
            free_path = -np.log(np.random.random()) / sigma_t

            dist_x_right = (W - x) / u if u > eps else np.inf
            dist_x_left = (0.0 - x) / u if u < -eps else np.inf
            dist_y_top = (H - y) / v if v > eps else np.inf
            dist_y_bottom = (0.0 - y) / v if v < -eps else np.inf
            dist_z_front = (D - z) / w if w > eps else np.inf
            dist_z_back = (0.0 - z) / w if w < -eps else np.inf

            dist_boundary = min(
                dist_x_right, dist_x_left, dist_y_top, dist_y_bottom, dist_z_front, dist_z_back
            )
            step = max(0.0, min(free_path, dist_boundary))

            x += step * u
            y += step * v
            z += step * w
            total_path += step

            boundary_hit = abs(step - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(x - W) < eps * W + eps:
                    x = W
                elif abs(x - 0.0) < eps * W + eps:
                    x = 0.0
                if abs(y - H) < eps * H + eps:
                    y = H
                elif abs(y - 0.0) < eps * H + eps:
                    y = 0.0
                if abs(z - D) < eps * D + eps:
                    z = D
                elif abs(z - 0.0) < eps * D + eps:
                    z = 0.0

                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                z = min(max(z, 0.0), D)

                exited = 1.0
                data[i, 9] = x
                data[i, 10] = y
                data[i, 11] = z
                data[i, 12] = u
                data[i, 13] = v
                data[i, 14] = w
                data[i, 15] = total_path
                data[i, 16] = exited
                data[i, 17] = absorbed
                break

            if np.random.random() < sigma_a_prob:
                absorbed = 1.0
                data[i, 9] = x
                data[i, 10] = y
                data[i, 11] = z
                data[i, 12] = u
                data[i, 13] = v
                data[i, 14] = w
                data[i, 15] = total_path
                data[i, 16] = exited
                data[i, 17] = absorbed
                break

            u, v, w = _sample_isotropic_dir_3d(eps)

    return data


# ---------------------------------------------------------------------------
# Multi-cell steady-state Numba kernels (consolidated: always per-cell XS arrays)
# ---------------------------------------------------------------------------

@njit
def multi_cell_mc_boundary_source_3d_njit(
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
    angular_sampling,
    scattering_mode_code=0,
    g_hg=0.0,
):
    """3D MC on a Cartesian mesh, boundary source at x=0.

    Materials given as per-cell arrays sigma_t_per_cell[iz,iy,ix] and
    sigma_a_per_cell[iz,iy,ix]. For homogeneous media, pass constant arrays
    constructed in the Python wrapper.
    """
    energy_deposited = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)

    exited_xmin = exited_xmax = exited_ymin = exited_ymax = exited_zmin = exited_zmax = 0
    absorbed = 0
    travel_steps = 0
    boundary_reflections = 0
    warning_flags = 0

    eps = 1e-9
    boundary_eps = eps * 10.0

    for _ in range(num_particles):
        energy = 1.0

        y0 = np.random.uniform(0.0, total_height)
        z0 = np.random.uniform(0.0, total_depth)
        pos = np.array([boundary_eps, y0, z0], dtype=np.float64)

        u, v, w = _sample_incoming_hemisphere_plus_x(angular_sampling, eps)
        direction = np.array([u, v, w], dtype=np.float64)

        finished = False
        steps_this_particle = 0
        max_steps_per_particle = 1_000_000

        while (not finished) and (steps_this_particle < max_steps_per_particle):
            ix, iy, iz = find_cell_indices_3d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                num_cells_x, num_cells_y, num_cells_z
            )

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                sigma_t = sigma_t_per_cell[iz, iy, ix]
                sigma_a = sigma_a_per_cell[iz, iy, ix]
                sigma_s = sigma_t - sigma_a
                if sigma_s < 0.0:
                    sigma_s = 0.0
            else:
                sigma_t = 1.0
                sigma_a = 0.0
                sigma_s = 1.0
                warning_flags += 2

            dist_boundary = np.inf
            u = direction[0]
            v = direction[1]
            w = direction[2]

            dist_x = np.inf
            if abs(u) > eps:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]
                ) / u
                if dist_x > eps:
                    dist_boundary = min(dist_boundary, dist_x)

            dist_y = np.inf
            if abs(v) > eps:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]
                ) / v
                if dist_y > eps:
                    dist_boundary = min(dist_boundary, dist_y)

            dist_z = np.inf
            if abs(w) > eps:
                dist_z = (cell_boundaries_z[iz + 1] - pos[2]) / w if w > 0.0 else (
                    cell_boundaries_z[iz] - pos[2]
                ) / w
                if dist_z > eps:
                    dist_boundary = min(dist_boundary, dist_z)

            if dist_boundary == np.inf:
                warning_flags += 1
                absorbed += 1
                finished = True
                continue

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > eps else np.inf
            travel_dist = min(free_path, dist_boundary)
            if travel_dist < eps:
                travel_dist = eps

            travel_steps += 1
            steps_this_particle += 1

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                energy_deposited[iz, iy, ix] += (energy_before - energy)
                if sigma_a > 1e-16:
                    track_length[iz, iy, ix] += _track_length_contribution(
                        energy_before, sigma_a, travel_dist
                    )
                else:
                    track_length[iz, iy, ix] += energy_before * travel_dist

            pos += travel_dist * direction

            if energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < eps * 1.1

            if is_boundary_event:
                pos += direction * eps

                hit_x = abs(dist_x - dist_boundary) < boundary_eps
                hit_y = abs(dist_y - dist_boundary) < boundary_eps
                hit_z = abs(dist_z - dist_boundary) < boundary_eps

                is_xmin = hit_x and (pos[0] <= boundary_eps * 2.0)
                is_xmax = hit_x and (pos[0] >= total_width - boundary_eps * 2.0)
                is_ymin = hit_y and (pos[1] <= boundary_eps * 2.0)
                is_ymax = hit_y and (pos[1] >= total_height - boundary_eps * 2.0)
                is_zmin = hit_z and (pos[2] <= boundary_eps * 2.0)
                is_zmax = hit_z and (pos[2] >= total_depth - boundary_eps * 2.0)

                if is_xmin or is_xmax or is_ymin or is_ymax or is_zmin or is_zmax:
                    vacuum_hit = (
                        (is_xmin and (not reflect_xmin))
                        or (is_xmax and (not reflect_xmax))
                        or (is_ymin and (not reflect_ymin))
                        or (is_ymax and (not reflect_ymax))
                        or (is_zmin and (not reflect_zmin))
                        or (is_zmax and (not reflect_zmax))
                    )
                    if vacuum_hit:
                        if is_xmin and (not reflect_xmin):
                            exited_xmin += 1
                        elif is_xmax and (not reflect_xmax):
                            exited_xmax += 1
                        elif is_ymin and (not reflect_ymin):
                            exited_ymin += 1
                        elif is_ymax and (not reflect_ymax):
                            exited_ymax += 1
                        elif is_zmin and (not reflect_zmin):
                            exited_zmin += 1
                        elif is_zmax and (not reflect_zmax):
                            exited_zmax += 1
                        finished = True
                    else:
                        boundary_reflections += 1
                        if is_xmin or is_xmax:
                            direction[0] = -direction[0]
                            pos[0] = boundary_eps if is_xmin else total_width - boundary_eps
                        if is_ymin or is_ymax:
                            direction[1] = -direction[1]
                            pos[1] = boundary_eps if is_ymin else total_height - boundary_eps
                        if is_zmin or is_zmax:
                            direction[2] = -direction[2]
                            pos[2] = boundary_eps if is_zmin else total_depth - boundary_eps
            else:
                u, v, w = _sample_scattered_dir_3d(direction[0], direction[1], direction[2], scattering_mode_code, g_hg, eps)
                direction[0] = u
                direction[1] = v
                direction[2] = w

        if steps_this_particle >= max_steps_per_particle:
            warning_flags += 4
            absorbed += 1

    stats = (
        exited_xmin, exited_xmax, exited_ymin, exited_ymax, exited_zmin, exited_zmax,
        absorbed, travel_steps, boundary_reflections, warning_flags
    )
    return energy_deposited, track_length, stats


@njit
def multi_cell_mc_internal_source_3d_njit(
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    start_cell_indices_tuple, start_position_relative_tuple,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
    randomize_start_pos=False,
    scattering_mode_code=0,
    g_hg=0.0,
):
    """3D MC on a Cartesian mesh, internal point-in-cell source.

    Materials given as per-cell arrays. For homogeneous media, pass constant arrays
    constructed in the Python wrapper.
    """
    start_ix, start_iy, start_iz = start_cell_indices_tuple
    start_rx, start_ry, start_rz = start_position_relative_tuple

    energy_deposited = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)

    exited_xmin = exited_xmax = exited_ymin = exited_ymax = exited_zmin = exited_zmax = 0
    absorbed = 0
    travel_steps = 0
    boundary_reflections = 0
    warning_flags = 0

    eps = 1e-9
    boundary_eps = eps * 10.0

    for _ in range(num_particles):
        energy = 1.0

        if randomize_start_pos:
            rx = np.random.rand()
            ry = np.random.rand()
            rz = np.random.rand()
        else:
            rx = start_rx
            ry = start_ry
            rz = start_rz

        x0 = cell_boundaries_x[start_ix] + rx * cell_widths[start_ix]
        y0 = cell_boundaries_y[start_iy] + ry * cell_heights[start_iy]
        z0 = cell_boundaries_z[start_iz] + rz * cell_depths[start_iz]
        pos = np.array([x0, y0, z0], dtype=np.float64)

        u, v, w = _sample_isotropic_dir_3d(eps)
        direction = np.array([u, v, w], dtype=np.float64)

        finished = False
        steps_this_particle = 0
        max_steps_per_particle = 1_000_000

        while (not finished) and (steps_this_particle < max_steps_per_particle):
            ix, iy, iz = find_cell_indices_3d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                num_cells_x, num_cells_y, num_cells_z
            )

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                sigma_t = sigma_t_per_cell[iz, iy, ix]
                sigma_a = sigma_a_per_cell[iz, iy, ix]
                sigma_s = sigma_t - sigma_a
                if sigma_s < 0.0:
                    sigma_s = 0.0
            else:
                sigma_t = 1.0
                sigma_a = 0.0
                sigma_s = 1.0
                warning_flags += 2

            dist_boundary = np.inf
            u = direction[0]
            v = direction[1]
            w = direction[2]

            dist_x = np.inf
            if abs(u) > eps:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]
                ) / u
                if dist_x > eps:
                    dist_boundary = min(dist_boundary, dist_x)

            dist_y = np.inf
            if abs(v) > eps:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]
                ) / v
                if dist_y > eps:
                    dist_boundary = min(dist_boundary, dist_y)

            dist_z = np.inf
            if abs(w) > eps:
                dist_z = (cell_boundaries_z[iz + 1] - pos[2]) / w if w > 0.0 else (
                    cell_boundaries_z[iz] - pos[2]
                ) / w
                if dist_z > eps:
                    dist_boundary = min(dist_boundary, dist_z)

            if dist_boundary == np.inf:
                warning_flags += 1
                absorbed += 1
                finished = True
                continue

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > eps else np.inf
            travel_dist = min(free_path, dist_boundary)
            if travel_dist < eps:
                travel_dist = eps

            travel_steps += 1
            steps_this_particle += 1

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                energy_deposited[iz, iy, ix] += (energy_before - energy)
                if sigma_a > 1e-16:
                    track_length[iz, iy, ix] += _track_length_contribution(
                        energy_before, sigma_a, travel_dist
                    )
                else:
                    track_length[iz, iy, ix] += energy_before * travel_dist

            pos += travel_dist * direction

            if energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < eps * 1.1
            if is_boundary_event:
                pos += direction * eps

                hit_x = abs(dist_x - dist_boundary) < boundary_eps
                hit_y = abs(dist_y - dist_boundary) < boundary_eps
                hit_z = abs(dist_z - dist_boundary) < boundary_eps

                is_xmin = hit_x and (pos[0] <= boundary_eps * 2.0)
                is_xmax = hit_x and (pos[0] >= total_width - boundary_eps * 2.0)
                is_ymin = hit_y and (pos[1] <= boundary_eps * 2.0)
                is_ymax = hit_y and (pos[1] >= total_height - boundary_eps * 2.0)
                is_zmin = hit_z and (pos[2] <= boundary_eps * 2.0)
                is_zmax = hit_z and (pos[2] >= total_depth - boundary_eps * 2.0)

                if is_xmin or is_xmax or is_ymin or is_ymax or is_zmin or is_zmax:
                    vacuum_hit = (
                        (is_xmin and (not reflect_xmin))
                        or (is_xmax and (not reflect_xmax))
                        or (is_ymin and (not reflect_ymin))
                        or (is_ymax and (not reflect_ymax))
                        or (is_zmin and (not reflect_zmin))
                        or (is_zmax and (not reflect_zmax))
                    )
                    if vacuum_hit:
                        if is_xmin and (not reflect_xmin):
                            exited_xmin += 1
                        elif is_xmax and (not reflect_xmax):
                            exited_xmax += 1
                        elif is_ymin and (not reflect_ymin):
                            exited_ymin += 1
                        elif is_ymax and (not reflect_ymax):
                            exited_ymax += 1
                        elif is_zmin and (not reflect_zmin):
                            exited_zmin += 1
                        elif is_zmax and (not reflect_zmax):
                            exited_zmax += 1
                        finished = True
                    else:
                        boundary_reflections += 1
                        if is_xmin or is_xmax:
                            direction[0] = -direction[0]
                            pos[0] = boundary_eps if is_xmin else total_width - boundary_eps
                        if is_ymin or is_ymax:
                            direction[1] = -direction[1]
                            pos[1] = boundary_eps if is_ymin else total_height - boundary_eps
                        if is_zmin or is_zmax:
                            direction[2] = -direction[2]
                            pos[2] = boundary_eps if is_zmin else total_depth - boundary_eps
            else:
                u, v, w = _sample_scattered_dir_3d(direction[0], direction[1], direction[2], scattering_mode_code, g_hg, eps)
                direction[0] = u
                direction[1] = v
                direction[2] = w

        if steps_this_particle >= max_steps_per_particle:
            warning_flags += 4
            absorbed += 1

    stats = (
        exited_xmin, exited_xmax, exited_ymin, exited_ymax, exited_zmin, exited_zmax,
        absorbed, travel_steps, boundary_reflections, warning_flags
    )
    return energy_deposited, track_length, stats


@njit
def multi_cell_mc_volume_box_source_3d_njit(
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    source_x0, source_x1, source_y0, source_y1, source_z0, source_z1,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
    scattering_mode_code=0,
    g_hg=0.0,
):
    """3D MC on a Cartesian mesh, uniform isotropic source in an axis-aligned box.

    Materials given as per-cell arrays. For homogeneous media, pass constant arrays
    constructed in the Python wrapper.
    """
    energy_deposited = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)
    track_length = np.zeros((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)

    exited_xmin = exited_xmax = exited_ymin = exited_ymax = exited_zmin = exited_zmax = 0
    absorbed = 0
    travel_steps = 0
    boundary_reflections = 0
    warning_flags = 0

    eps = 1e-9
    boundary_eps = eps * 10.0

    for _ in range(num_particles):
        energy = 1.0

        x0 = np.random.uniform(source_x0, source_x1)
        y0 = np.random.uniform(source_y0, source_y1)
        z0 = np.random.uniform(source_z0, source_z1)
        pos = np.array([x0, y0, z0], dtype=np.float64)

        u, v, w = _sample_isotropic_dir_3d(eps)
        direction = np.array([u, v, w], dtype=np.float64)

        finished = False
        steps_this_particle = 0
        max_steps_per_particle = 1_000_000

        while (not finished) and (steps_this_particle < max_steps_per_particle):
            ix, iy, iz = find_cell_indices_3d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                num_cells_x, num_cells_y, num_cells_z
            )

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                sigma_t = sigma_t_per_cell[iz, iy, ix]
                sigma_a = sigma_a_per_cell[iz, iy, ix]
                sigma_s = sigma_t - sigma_a
                if sigma_s < 0.0:
                    sigma_s = 0.0
            else:
                sigma_t = 1.0
                sigma_a = 0.0
                sigma_s = 1.0
                warning_flags += 2

            dist_boundary = np.inf
            u = direction[0]
            v = direction[1]
            w = direction[2]

            dist_x = np.inf
            if abs(u) > eps:
                dist_x = (cell_boundaries_x[ix + 1] - pos[0]) / u if u > 0.0 else (
                    cell_boundaries_x[ix] - pos[0]
                ) / u
                if dist_x > eps:
                    dist_boundary = min(dist_boundary, dist_x)

            dist_y = np.inf
            if abs(v) > eps:
                dist_y = (cell_boundaries_y[iy + 1] - pos[1]) / v if v > 0.0 else (
                    cell_boundaries_y[iy] - pos[1]
                ) / v
                if dist_y > eps:
                    dist_boundary = min(dist_boundary, dist_y)

            dist_z = np.inf
            if abs(w) > eps:
                dist_z = (cell_boundaries_z[iz + 1] - pos[2]) / w if w > 0.0 else (
                    cell_boundaries_z[iz] - pos[2]
                ) / w
                if dist_z > eps:
                    dist_boundary = min(dist_boundary, dist_z)

            if dist_boundary == np.inf:
                warning_flags += 1
                absorbed += 1
                finished = True
                continue

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > eps else np.inf
            travel_dist = min(free_path, dist_boundary)
            if travel_dist < eps:
                travel_dist = eps

            travel_steps += 1
            steps_this_particle += 1

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)

            if 0 <= ix < num_cells_x and 0 <= iy < num_cells_y and 0 <= iz < num_cells_z:
                energy_deposited[iz, iy, ix] += (energy_before - energy)
                if sigma_a > 1e-16:
                    track_length[iz, iy, ix] += _track_length_contribution(
                        energy_before, sigma_a, travel_dist
                    )
                else:
                    track_length[iz, iy, ix] += energy_before * travel_dist

            pos += travel_dist * direction

            if energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < eps * 1.1
            if is_boundary_event:
                pos += direction * eps

                hit_x = abs(dist_x - dist_boundary) < boundary_eps
                hit_y = abs(dist_y - dist_boundary) < boundary_eps
                hit_z = abs(dist_z - dist_boundary) < boundary_eps

                is_xmin = hit_x and (pos[0] <= boundary_eps * 2.0)
                is_xmax = hit_x and (pos[0] >= total_width - boundary_eps * 2.0)
                is_ymin = hit_y and (pos[1] <= boundary_eps * 2.0)
                is_ymax = hit_y and (pos[1] >= total_height - boundary_eps * 2.0)
                is_zmin = hit_z and (pos[2] <= boundary_eps * 2.0)
                is_zmax = hit_z and (pos[2] >= total_depth - boundary_eps * 2.0)

                if is_xmin or is_xmax or is_ymin or is_ymax or is_zmin or is_zmax:
                    vacuum_hit = (
                        (is_xmin and (not reflect_xmin))
                        or (is_xmax and (not reflect_xmax))
                        or (is_ymin and (not reflect_ymin))
                        or (is_ymax and (not reflect_ymax))
                        or (is_zmin and (not reflect_zmin))
                        or (is_zmax and (not reflect_zmax))
                    )
                    if vacuum_hit:
                        if is_xmin and (not reflect_xmin):
                            exited_xmin += 1
                        elif is_xmax and (not reflect_xmax):
                            exited_xmax += 1
                        elif is_ymin and (not reflect_ymin):
                            exited_ymin += 1
                        elif is_ymax and (not reflect_ymax):
                            exited_ymax += 1
                        elif is_zmin and (not reflect_zmin):
                            exited_zmin += 1
                        elif is_zmax and (not reflect_zmax):
                            exited_zmax += 1
                        finished = True
                    else:
                        boundary_reflections += 1
                        if is_xmin or is_xmax:
                            direction[0] = -direction[0]
                            pos[0] = boundary_eps if is_xmin else total_width - boundary_eps
                        if is_ymin or is_ymax:
                            direction[1] = -direction[1]
                            pos[1] = boundary_eps if is_ymin else total_height - boundary_eps
                        if is_zmin or is_zmax:
                            direction[2] = -direction[2]
                            pos[2] = boundary_eps if is_zmin else total_depth - boundary_eps
            else:
                u, v, w = _sample_scattered_dir_3d(direction[0], direction[1], direction[2], scattering_mode_code, g_hg, eps)
                direction[0] = u
                direction[1] = v
                direction[2] = w

        if steps_this_particle >= max_steps_per_particle:
            warning_flags += 4
            absorbed += 1

    stats = (
        exited_xmin, exited_xmax, exited_ymin, exited_ymax, exited_zmin, exited_zmax,
        absorbed, travel_steps, boundary_reflections, warning_flags
    )
    return energy_deposited, track_length, stats


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def run_mc_simulation_3d_njit_wrapper(
    source_mode,
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    start_cell_indices=None, start_position_relative=None,
    source_box=None,
    scatter_ratio=1.0, energy_threshold=1e-8, seed=None,
    reflected_boundaries=False, angular_sampling=1,
    boundary_reflective_faces=None,
    sigma_t_per_cell=None, sigma_a_per_cell=None,
    randomize_position_in_cell=False, verbose=True,
    scattering_mode="isotropic",
    g_hg=None,
):
    """Python wrapper for 3D mesh MC Numba kernels.

    For homogeneous media, pass scatter_ratio and leave sigma_t_per_cell /
    sigma_a_per_cell as None; the wrapper builds constant arrays automatically.
    For heterogeneous media, pass sigma_t_per_cell and sigma_a_per_cell
    as 3D arrays with index order [z, y, x].
    """
    start_time = time.time()

    # Build per-cell arrays if not provided (homogeneous case).
    if sigma_t_per_cell is None or sigma_a_per_cell is None:
        sigma_t_per_cell = np.ones((num_cells_z, num_cells_y, num_cells_x), dtype=np.float64)
        sigma_a_per_cell = np.full(
            (num_cells_z, num_cells_y, num_cells_x), 1.0 - scatter_ratio, dtype=np.float64
        )
    else:
        if sigma_t_per_cell.ndim == 1:
            sigma_t_per_cell = sigma_t_per_cell.reshape(num_cells_z, num_cells_y, num_cells_x)
        if sigma_a_per_cell.ndim == 1:
            sigma_a_per_cell = sigma_a_per_cell.reshape(num_cells_z, num_cells_y, num_cells_x)

    scattering_mode, scattering_mode_code, resolved_g_hg = resolve_3d_scattering_parameters(
        scattering_mode=scattering_mode,
        g_hg=g_hg,
    )

    if boundary_reflective_faces is None:
        reflect_xmin = reflected_boundaries
        reflect_xmax = reflected_boundaries
        reflect_ymin = reflected_boundaries
        reflect_ymax = reflected_boundaries
        reflect_zmin = reflected_boundaries
        reflect_zmax = reflected_boundaries
    else:
        if len(boundary_reflective_faces) != 6:
            raise ValueError("boundary_reflective_faces must be length 6: (xmin,xmax,ymin,ymax,zmin,zmax)")
        reflect_xmin = bool(boundary_reflective_faces[0])
        reflect_xmax = bool(boundary_reflective_faces[1])
        reflect_ymin = bool(boundary_reflective_faces[2])
        reflect_ymax = bool(boundary_reflective_faces[3])
        reflect_zmin = bool(boundary_reflective_faces[4])
        reflect_zmax = bool(boundary_reflective_faces[5])

    with seeded_numpy_execution(seed):
        if source_mode == "boundary_only":
            if verbose:
                bc = "CustomFaces" if boundary_reflective_faces is not None else (
                    "Reflected" if reflected_boundaries else "Exiting"
                )
                print(f"\nRunning Numba 3D mesh MC (boundary source @ x=0, {bc} BC)...")

            energy_raw, track_raw, stats = multi_cell_mc_boundary_source_3d_njit(
                num_cells_x, num_cells_y, num_cells_z,
                cell_widths, cell_heights, cell_depths,
                cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                total_width, total_height, total_depth,
                num_particles,
                sigma_t_per_cell, sigma_a_per_cell,
                energy_threshold,
                reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
                angular_sampling,
                scattering_mode_code,
                resolved_g_hg,
            )

        elif source_mode == "internal_start":
            if start_cell_indices is None or start_position_relative is None:
                raise ValueError("internal_start requires start_cell_indices and start_position_relative")
            if not (
                0 <= start_cell_indices[0] < num_cells_x
                and 0 <= start_cell_indices[1] < num_cells_y
                and 0 <= start_cell_indices[2] < num_cells_z
            ):
                raise ValueError(f"Start cell indices out of bounds: {start_cell_indices}")
            if not (
                0.0 <= start_position_relative[0] <= 1.0
                and 0.0 <= start_position_relative[1] <= 1.0
                and 0.0 <= start_position_relative[2] <= 1.0
            ):
                raise ValueError(f"Start position relative must be in [0,1]: {start_position_relative}")

            start_cell_indices_tuple = tuple(start_cell_indices)
            start_position_relative_tuple = tuple(start_position_relative)

            if verbose:
                bc = "CustomFaces" if boundary_reflective_faces is not None else (
                    "Reflected" if reflected_boundaries else "Exiting"
                )
                rnd = "randomized" if randomize_position_in_cell else "fixed"
                print(f"\nRunning Numba 3D mesh MC (internal source, {bc} BC, {rnd} in-cell pos)...")

            energy_raw, track_raw, stats = multi_cell_mc_internal_source_3d_njit(
                num_cells_x, num_cells_y, num_cells_z,
                cell_widths, cell_heights, cell_depths,
                cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                total_width, total_height, total_depth,
                num_particles,
                start_cell_indices_tuple, start_position_relative_tuple,
                sigma_t_per_cell, sigma_a_per_cell,
                energy_threshold,
                reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
                randomize_position_in_cell,
                scattering_mode_code,
                resolved_g_hg,
            )

        elif source_mode == "volume_box":
            if source_box is None:
                raise ValueError("volume_box requires source_box=((x0,x1),(y0,y1),(z0,z1))")
            sx0, sx1 = float(source_box[0][0]), float(source_box[0][1])
            sy0, sy1 = float(source_box[1][0]), float(source_box[1][1])
            sz0, sz1 = float(source_box[2][0]), float(source_box[2][1])
            if not (
                0.0 <= sx0 <= sx1 <= total_width
                and 0.0 <= sy0 <= sy1 <= total_height
                and 0.0 <= sz0 <= sz1 <= total_depth
            ):
                raise ValueError("source_box must lie within the domain bounds")

            if verbose:
                bc = "CustomFaces" if boundary_reflective_faces is not None else (
                    "Reflected" if reflected_boundaries else "Exiting"
                )
                print(f"\nRunning Numba 3D mesh MC (volume box source, {bc} BC)...")

            energy_raw, track_raw, stats = multi_cell_mc_volume_box_source_3d_njit(
                num_cells_x, num_cells_y, num_cells_z,
                cell_widths, cell_heights, cell_depths,
                cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                total_width, total_height, total_depth,
                num_particles,
                sx0, sx1, sy0, sy1, sz0, sz1,
                sigma_t_per_cell, sigma_a_per_cell,
                energy_threshold,
                reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
                scattering_mode_code,
                resolved_g_hg,
            )

        else:
            raise ValueError(f"Unsupported source_mode for 3D mesh MC: {source_mode}")

    end_time = time.time()
    if verbose:
        print(f"Numba 3D mesh MC took {end_time - start_time:.3f} seconds.")
        if scattering_mode_code != 0:
            print(f"  Scattering mode:        {scattering_mode}")
        if scattering_mode_code == 2:
            print(f"  HG asymmetry parameter: {resolved_g_hg:.3f}")

    (
        exited_xmin, exited_xmax, exited_ymin, exited_ymax, exited_zmin, exited_zmax,
        absorbed, travel_steps, boundary_reflections, warning_flags
    ) = stats

    if verbose:
        if boundary_reflective_faces is None:
            bc_type = "Reflected BC" if reflected_boundaries else "Exiting BC"
        else:
            bc_type = (
                f"Faces refl=(xmin={reflect_xmin}, xmax={reflect_xmax}, "
                f"ymin={reflect_ymin}, ymax={reflect_ymax}, "
                f"zmin={reflect_zmin}, zmax={reflect_zmax})"
            )
        print(f"\nMonte Carlo (3D mesh {source_mode}, Numba, {bc_type}) particle statistics:")
        print(f"  Exited x-min (x=0):      {exited_xmin} ({exited_xmin/num_particles:.2%})")
        print(f"  Exited x-max (x=Lx):     {exited_xmax} ({exited_xmax/num_particles:.2%})")
        print(f"  Exited y-min (y=0):      {exited_ymin} ({exited_ymin/num_particles:.2%})")
        print(f"  Exited y-max (y=Ly):     {exited_ymax} ({exited_ymax/num_particles:.2%})")
        print(f"  Exited z-min (z=0):      {exited_zmin} ({exited_zmin/num_particles:.2%})")
        print(f"  Exited z-max (z=Lz):     {exited_zmax} ({exited_zmax/num_particles:.2%})")
        print(f"  Absorbed in medium:      {absorbed} ({absorbed/num_particles:.2%})")
        if boundary_reflections > 0:
            print(f"  Boundary reflections:    {boundary_reflections}")
        print(f"  Average travel steps:    {travel_steps/num_particles:.2f}")
        total_accounted = (
            exited_xmin + exited_xmax + exited_ymin + exited_ymax
            + exited_zmin + exited_zmax + absorbed
        )
        if total_accounted != num_particles:
            print(f"  WARNING: {num_particles - total_accounted} particles unaccounted for in MC!")
        if warning_flags > 0:
            print(f"  WARNING: Numba function reported warning_flags={warning_flags}")

    return energy_raw / num_particles, track_raw / num_particles, stats
