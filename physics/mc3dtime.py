"""mc3dtime.py - 3D time-dependent Monte Carlo transport (single-cell and multi-cell).

Key refactors / fixes vs prior version:
  * One single-cell kernel handles internal vs boundary source via `source_type`.
  * Unified output schema for internal/boundary (boundary uses x_in=0).
  * Minor robustness: clamp remaining budget to avoid negative due to roundoff.
"""

from __future__ import annotations

import time
import numpy as np
from numba import njit, prange

from .mc1d import sample_unit
from .mc3d import (
    _sample_isotropic_dir_3d,
    _sample_isotropic_dir_3d_from_unit,
    _sample_incoming_hemisphere_plus_x,
    _sample_incoming_hemisphere_plus_x_from_unit,
    _sample_scattered_dir_3d,
    _track_length_contribution,
    find_cell_indices_3d_njit,
    resolve_3d_scattering_parameters,
)
from .steady_2d_scattering import sample_scattered_direction_3d_from_unit

# Source-type flags (Numba-friendly ints)
SRC_INTERNAL = 0          # uniform in volume
SRC_BOUNDARY_X0 = 1       # boundary at x=0, incoming (+x)


@njit(inline='always')
def _log_uniform_from_unit(u: float, log_min: float, log_max: float) -> float:
    return 10.0 ** (log_min + u * (log_max - log_min))


# ---------------------------------------------------------------------------
# Single-cell time-dependent (path-length budget)
# ---------------------------------------------------------------------------

@njit
def simulate_single_cell_3d_time_dependent(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
    source_type: int = SRC_INTERNAL,
    angular_sampling: int = 1,
    scattering_mode_code: int = 0,
    g_hg: float = 0.0,
):
    """3D time-dependent (path-length budget) single-cell MC.

    Parameters
    ----------
    source_type:
        SRC_INTERNAL (0): (x,y,z) uniform in [0,W]x[0,H]x[0,D], isotropic direction
        SRC_BOUNDARY_X0 (1): x=0, (y,z) uniform, incoming hemisphere directions (+x)
                             via _sample_incoming_hemisphere_plus_x(angular_sampling)

    Returns
    -------
    data : (N, 21) float64
        Columns:
          [0] width, [1] height, [2] depth, [3] path_budget,
          [4] x_in, [5] y_in, [6] z_in,
          [7] u_in, [8] v_in, [9] w_in,
          [10] x_f, [11] y_f, [12] z_f,
          [13] u_f, [14] v_f, [15] w_f,
          [16] L, [17] num_interactions,
          [18] space_exit, [19] time_exit, [20] absorbed
    """
    data = np.zeros((num_neutrons, 21), dtype=np.float64)

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
    log_min_budget = np.log10(min_path_budget)
    log_max_budget = np.log10(max_path_budget)

    for i in prange(num_neutrons):
        W = _log_uniform_from_unit(
            cond_samples[i, 0] if use_precond else sample_unit(distribution), log_min_w, log_max_w
        )
        H = _log_uniform_from_unit(
            cond_samples[i, 1] if use_precond else sample_unit(distribution), log_min_h, log_max_h
        )
        D = _log_uniform_from_unit(
            cond_samples[i, 2] if use_precond else sample_unit(distribution), log_min_d, log_max_d
        )
        path_budget = _log_uniform_from_unit(
            cond_samples[i, 3] if use_precond else sample_unit(distribution), log_min_budget, log_max_budget
        )

        data[i, 0] = W
        data[i, 1] = H
        data[i, 2] = D
        data[i, 3] = path_budget

        if source_type == SRC_INTERNAL:
            if use_precond:
                x = W * cond_samples[i, 4]
                y = H * cond_samples[i, 5]
                z = D * cond_samples[i, 6]
                u, v, w = _sample_isotropic_dir_3d_from_unit(cond_samples[i, 7], cond_samples[i, 8], eps)
            else:
                x = np.random.uniform(0.0, W)
                y = np.random.uniform(0.0, H)
                z = np.random.uniform(0.0, D)
                u, v, w = _sample_isotropic_dir_3d(eps)
        else:
            x = 0.0
            if use_precond:
                y = H * cond_samples[i, 4]
                z = D * cond_samples[i, 5]
                u, v, w = _sample_incoming_hemisphere_plus_x_from_unit(cond_samples[i, 6], cond_samples[i, 7], angular_sampling, eps)
            else:
                y = np.random.uniform(0.0, H)
                z = np.random.uniform(0.0, D)
                u, v, w = _sample_incoming_hemisphere_plus_x(angular_sampling, eps)

        data[i, 4] = x
        data[i, 5] = y
        data[i, 6] = z
        data[i, 7] = u
        data[i, 8] = v
        data[i, 9] = w

        L = 0.0
        num_interactions = 0.0
        space_exit = 0.0
        time_exit = 0.0
        absorbed = 0.0

        while True:
            remaining_budget = path_budget - L
            if remaining_budget <= 0.0:
                time_exit = 1.0
                break

            free_path = -np.log(np.random.random()) / sigma_t

            dist_x_right = (W - x) / u if u > eps else np.inf
            dist_x_left = -x / u if u < -eps else np.inf
            dist_y_top = (H - y) / v if v > eps else np.inf
            dist_y_bottom = -y / v if v < -eps else np.inf
            dist_z_front = (D - z) / w if w > eps else np.inf
            dist_z_back = -z / w if w < -eps else np.inf

            dist_boundary = dist_x_right
            if dist_x_left < dist_boundary:
                dist_boundary = dist_x_left
            if dist_y_top < dist_boundary:
                dist_boundary = dist_y_top
            if dist_y_bottom < dist_boundary:
                dist_boundary = dist_y_bottom
            if dist_z_front < dist_boundary:
                dist_boundary = dist_z_front
            if dist_z_back < dist_boundary:
                dist_boundary = dist_z_back

            step = free_path
            if dist_boundary < step:
                step = dist_boundary
            if remaining_budget < step:
                step = remaining_budget
            if step < 0.0:
                step = 0.0

            x += step * u
            y += step * v
            z += step * w
            L += step

            boundary_hit = abs(step - dist_boundary) < eps * dist_boundary + eps
            budget_hit = abs(step - remaining_budget) < eps * remaining_budget + eps

            if boundary_hit:
                # Snap/clamp
                if abs(x - W) < eps * W + eps:
                    x = W
                elif abs(x) < eps * W + eps:
                    x = 0.0
                if abs(y - H) < eps * H + eps:
                    y = H
                elif abs(y) < eps * H + eps:
                    y = 0.0
                if abs(z - D) < eps * D + eps:
                    z = D
                elif abs(z) < eps * D + eps:
                    z = 0.0

                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                z = min(max(z, 0.0), D)

                space_exit = 1.0
                break

            if budget_hit:
                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                z = min(max(z, 0.0), D)
                time_exit = 1.0
                break

            num_interactions += 1.0
            if np.random.random() < sigma_a_prob:
                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                z = min(max(z, 0.0), D)
                absorbed = 1.0
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

        data[i, 10] = x
        data[i, 11] = y
        data[i, 12] = z
        data[i, 13] = u
        data[i, 14] = v
        data[i, 15] = w
        data[i, 16] = L
        data[i, 17] = num_interactions
        data[i, 18] = space_exit
        data[i, 19] = time_exit
        data[i, 20] = absorbed

    return data


def simulate_single_cell_3d_time_dependent_boundary(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    angular_sampling: int = 1,
    cond_samples=None,
    scattering_mode_code: int = 0,
    g_hg: float = 0.0,
):
    """Deprecated: kept for API compatibility.

    Use:
        simulate_single_cell_3d_time_dependent(..., source_type=SRC_BOUNDARY_X0)
    """
    return simulate_single_cell_3d_time_dependent(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_depth=min_depth,
        max_depth=max_depth,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
        source_type=SRC_BOUNDARY_X0,
        angular_sampling=angular_sampling,
        scattering_mode_code=scattering_mode_code,
        g_hg=g_hg,
    )


# ---------------------------------------------------------------------------
# Multi-cell time-dependent Numba kernel (unchanged)
# ---------------------------------------------------------------------------

@njit
def multi_cell_mc_time_dependent_3d_njit(
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    source_x0, source_x1, source_y0, source_y1, source_z0, source_z1,
    start_times,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
    particle_velocity,
    max_time, num_time_bins,
    scattering_mode_code=0,
    g_hg=0.0,
):
    """Time-dependent 3D MC with per-cell materials and (x,y)-only mesh tallies.

    Tallies are accumulated in:
      track_length[time_bin, y, x]

    Transport is fully 3D (x,y,z), but the tally collapses z (equivalent to z having 1 bin).
    For homogeneous media, pass constant arrays from the Python wrapper.
    """
    dt = max_time / num_time_bins

    track_length = np.zeros((num_time_bins, num_cells_y, num_cells_x), dtype=np.float64)
    energy_deposition = np.zeros((num_time_bins, num_cells_y, num_cells_x), dtype=np.float64)

    exited_xmin = exited_xmax = exited_ymin = exited_ymax = exited_zmin = exited_zmax = 0
    absorbed = 0
    timed_out = 0
    travel_steps = 0
    boundary_reflections = 0
    warning_flags = 0

    eps = 1e-9
    boundary_eps = eps * 10.0

    for n_idx in range(num_particles):
        energy = 1.0
        current_time = start_times[n_idx]

        if current_time >= max_time:
            continue

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
            if current_time >= max_time:
                timed_out += 1
                finished = True
                continue

            ix, iy, iz = find_cell_indices_3d_njit(
                pos, cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
                num_cells_x, num_cells_y, num_cells_z
            )

            sigma_t = sigma_t_per_cell[iz, iy, ix]
            sigma_a = sigma_a_per_cell[iz, iy, ix]
            sigma_s = sigma_t - sigma_a
            if sigma_s < 0.0:
                sigma_s = 0.0

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

            tbin = int(current_time / dt)
            if tbin < 0:
                tbin = 0
            elif tbin >= num_time_bins:
                tbin = num_time_bins - 1
            next_time_edge = (tbin + 1) * dt
            time_to_next = next_time_edge - current_time
            if time_to_next < 0.0:
                time_to_next = 0.0
            dist_time = time_to_next * particle_velocity

            free_path = -np.log(1.0 - np.random.rand()) / sigma_s if sigma_s > eps else np.inf

            travel_dist = min(free_path, dist_boundary, dist_time)
            if travel_dist < eps:
                travel_dist = eps

            travel_steps += 1
            steps_this_particle += 1

            travel_time = travel_dist / particle_velocity

            energy_before = energy
            energy *= np.exp(-sigma_a * travel_dist)
            deposited = energy_before - energy
            if deposited < 0.0:
                deposited = 0.0

            track_length[tbin, iy, ix] += _track_length_contribution(energy_before, sigma_a, travel_dist)
            energy_deposition[tbin, iy, ix] += deposited

            pos += travel_dist * direction
            current_time += travel_time

            if energy_threshold > 0.0 and energy <= energy_threshold:
                absorbed += 1
                finished = True
                continue

            is_boundary_event = abs(travel_dist - dist_boundary) < boundary_eps
            is_time_crossing = abs(travel_dist - dist_time) < boundary_eps

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
            elif is_time_crossing:
                pass
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
        absorbed, timed_out, travel_steps, boundary_reflections, warning_flags
    )
    return track_length, energy_deposition, stats


# ---------------------------------------------------------------------------
# Python wrapper (unchanged)
# ---------------------------------------------------------------------------

def run_mc_time_dependent_3d_xy_wrapper(
    num_cells_x, num_cells_y, num_cells_z,
    cell_widths, cell_heights, cell_depths,
    cell_boundaries_x, cell_boundaries_y, cell_boundaries_z,
    total_width, total_height, total_depth,
    num_particles,
    source_box,
    sigma_t_per_cell, sigma_a_per_cell,
    energy_threshold,
    particle_velocity,
    max_time, num_time_bins,
    source_time_range=(0.0, 0.0),
    seed=None,
    boundary_reflective_faces=None,
    return_energy=False,
    verbose=True,
    scattering_mode="isotropic",
    g_hg=None,
):
    """Time-dependent 3D transport with per-cell materials and (x,y)-mesh tallies."""
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(time.time()))

    if boundary_reflective_faces is None:
        reflect_xmin = reflect_xmax = False
        reflect_ymin = reflect_ymax = False
        reflect_zmin = reflect_zmax = False
    else:
        if len(boundary_reflective_faces) != 6:
            raise ValueError("boundary_reflective_faces must be length 6: (xmin,xmax,ymin,ymax,zmin,zmax)")
        reflect_xmin = bool(boundary_reflective_faces[0])
        reflect_xmax = bool(boundary_reflective_faces[1])
        reflect_ymin = bool(boundary_reflective_faces[2])
        reflect_ymax = bool(boundary_reflective_faces[3])
        reflect_zmin = bool(boundary_reflective_faces[4])
        reflect_zmax = bool(boundary_reflective_faces[5])

    sx0, sx1 = float(source_box[0][0]), float(source_box[0][1])
    sy0, sy1 = float(source_box[1][0]), float(source_box[1][1])
    sz0, sz1 = float(source_box[2][0]), float(source_box[2][1])
    if not (
        0.0 <= sx0 <= sx1 <= total_width
        and 0.0 <= sy0 <= sy1 <= total_height
        and 0.0 <= sz0 <= sz1 <= total_depth
    ):
        raise ValueError("source_box must lie within the domain bounds")

    t0, t1 = float(source_time_range[0]), float(source_time_range[1])
    if t1 < t0:
        raise ValueError("source_time_range must be (t0,t1) with t1>=t0")
    start_times = np.random.uniform(t0, t1, num_particles).astype(np.float64)
    start_times = np.clip(start_times, 0.0, float(max_time))

    if sigma_t_per_cell.ndim == 1:
        sigma_t_per_cell = sigma_t_per_cell.reshape(num_cells_z, num_cells_y, num_cells_x)
    if sigma_a_per_cell.ndim == 1:
        sigma_a_per_cell = sigma_a_per_cell.reshape(num_cells_z, num_cells_y, num_cells_x)

    scattering_mode, scattering_mode_code, resolved_g_hg = resolve_3d_scattering_parameters(
        scattering_mode=scattering_mode,
        g_hg=g_hg,
    )

    if verbose:
        print(
            f"\nRunning Numba 3D time-dependent MC (xy-tally) with {num_particles} particles, "
            f"t in [{t0},{t1}], max_time={max_time}, bins={num_time_bins}..."
        )

    start = time.time()
    tl_raw, edep_raw, stats = multi_cell_mc_time_dependent_3d_njit(
        int(num_cells_x), int(num_cells_y), int(num_cells_z),
        cell_widths.astype(np.float64),
        cell_heights.astype(np.float64),
        cell_depths.astype(np.float64),
        cell_boundaries_x.astype(np.float64),
        cell_boundaries_y.astype(np.float64),
        cell_boundaries_z.astype(np.float64),
        float(total_width), float(total_height), float(total_depth),
        int(num_particles),
        sx0, sx1, sy0, sy1, sz0, sz1,
        start_times,
        sigma_t_per_cell.astype(np.float64),
        sigma_a_per_cell.astype(np.float64),
        float(energy_threshold),
            reflect_xmin, reflect_xmax, reflect_ymin, reflect_ymax, reflect_zmin, reflect_zmax,
            float(particle_velocity),
            float(max_time), int(num_time_bins),
            scattering_mode_code,
            resolved_g_hg,
        )
    end = time.time()

    if verbose:
        print(f"Numba 3D time-dependent MC took {end - start:.3f} seconds.")
        if scattering_mode_code != 0:
            print(f"  Scattering mode:        {scattering_mode}")
        if scattering_mode_code == 2:
            print(f"  HG asymmetry parameter: {resolved_g_hg:.3f}")

    if return_energy:
        return tl_raw / num_particles, edep_raw / num_particles, stats
    return tl_raw / num_particles, stats
