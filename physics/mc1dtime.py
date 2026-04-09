"""mc1dtime.py - 1D time-dependent Monte Carlo transport.

This module provides:
  * Single-cell, path-length-budget time-dependent MC (internal or boundary source)
  * 1D slab time-dependent reference solvers

Key refactors / fixes vs prior version:
  * One single-cell kernel handles both internal and boundary sources via `source_type`.
  * Consistent output schema for internal/boundary (boundary simply has x_in_norm=0).
  * Boundary angular sampling is cosine-law (mu ~ 2*mu on [0,1]) for incoming flux.
  * Minor robustness: clamp remaining budget to avoid negative due to roundoff.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .mc1d import sample_unit

# Source-type flags (Numba-friendly ints)
SRC_INTERNAL = 0          # uniform-in-cell source
SRC_BOUNDARY_X0 = 1       # boundary source at x=0, incoming (+x)


@njit(inline='always')
def _log_uniform_from_unit(u: float, log_min: float, log_max: float) -> float:
    """Map u in [0,1] to 10**(log_min + u*(log_max-log_min))."""
    return 10.0 ** (log_min + u * (log_max - log_min))


@njit(inline='always')
def _sample_mu_isotropic_1d(eps: float) -> float:
    """Sample mu ~ Uniform[-1,1], avoiding exactly 0."""
    mu = 2.0 * np.random.random() - 1.0
    if abs(mu) < eps:
        mu = eps if np.random.random() > 0.5 else -eps
    return mu


@njit(inline='always')
def _sample_mu_incoming_cosine_1d(eps: float) -> float:
    """Sample incoming mu>0 with cosine-law p(mu)=2*mu, mu in [0,1]."""
    mu = np.sqrt(np.random.random())
    if mu < eps:
        mu = eps
    return mu


@njit(inline='always')
def _sample_mu_isotropic_1d_from_unit(unit_mu: float, eps: float) -> float:
    mu = 2.0 * unit_mu - 1.0
    if abs(mu) < eps:
        mu = eps if unit_mu >= 0.5 else -eps
    return mu


@njit(inline='always')
def _sample_mu_incoming_cosine_1d_from_unit(unit_mu: float, eps: float) -> float:
    mu = np.sqrt(unit_mu)
    if mu < eps:
        mu = eps
    return mu


@njit(parallel=True)
def simulate_cell_1d_time_dependent(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
    source_type: int = SRC_INTERNAL,
):
    """1D time-dependent MC in a single cell using a path-length budget ("time").

    Parameters
    ----------
    source_type:
        SRC_INTERNAL (0): x_in ~ Uniform(0, thickness), mu_in ~ Uniform(-1,1)
        SRC_BOUNDARY_X0 (1): x_in=0, mu_in sampled from cosine-law incoming flux (mu>0)

    Returns
    -------
    data : (N, 11) float64
        Columns:
          [0] thickness,
          [1] path_budget,
          [2] x_in_norm,
          [3] mu_in,
          [4] x_out_norm,
          [5] mu_out,
          [6] L,                 (total path length traveled)
          [7] num_interactions,  (collisions)
          [8] space_exit,        (1 if exited through x=0 or x=thickness)
          [9] time_exit,         (1 if budget depleted before exit)
          [10] absorbed          (1 if absorbed)
    """
    data = np.zeros((num_neutrons, 11), dtype=np.float64)

    sigma_a_prob = 1.0 - scatter_ratio  # absorption probability at collision (analog)
    sigma_t = 1.0
    eps = 1e-9

    use_precond = cond_samples is not None

    log_min_thick = np.log10(min_thickness)
    log_max_thick = np.log10(max_thickness)
    log_min_budget = np.log10(min_path_budget)
    log_max_budget = np.log10(max_path_budget)

    for i in prange(num_neutrons):
        # --- Conditioning variables (geometry + budget) ---
        u_thick = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        thickness = _log_uniform_from_unit(u_thick, log_min_thick, log_max_thick)
        data[i, 0] = thickness

        u_budget = cond_samples[i, 1] if use_precond else sample_unit(distribution)
        path_budget = _log_uniform_from_unit(u_budget, log_min_budget, log_max_budget)
        data[i, 1] = path_budget

        # --- Initial state ---
        if source_type == SRC_INTERNAL:
            if use_precond:
                x = thickness * cond_samples[i, 2]
                mu = _sample_mu_isotropic_1d_from_unit(cond_samples[i, 3], eps)
            else:
                x = np.random.uniform(0.0, thickness)
                mu = _sample_mu_isotropic_1d(eps)
        else:
            # Boundary at x=0, incoming (+x)
            x = 0.0
            if use_precond:
                mu = _sample_mu_incoming_cosine_1d_from_unit(cond_samples[i, 2], eps)
            else:
                mu = _sample_mu_incoming_cosine_1d(eps)

        data[i, 2] = x / thickness
        data[i, 3] = mu

        L = 0.0
        num_interactions = 0.0
        space_exit = 0.0
        time_exit = 0.0
        absorbed = 0.0

        # --- Random walk ---
        while True:
            # Remaining "time" (path budget) — clamp for numerical robustness
            remaining_budget = path_budget - L
            if remaining_budget <= 0.0:
                time_exit = 1.0
                break

            free_path = -np.log(np.random.random()) / sigma_t

            # Distance to nearest boundary along current direction
            dist_right = (thickness - x) / mu if mu > eps else np.inf
            dist_left = -x / mu if mu < -eps else np.inf
            dist_boundary = dist_right if dist_right < dist_left else dist_left

            # Step distance is limited by collision, boundary, and budget
            step = free_path
            if dist_boundary < step:
                step = dist_boundary
            if remaining_budget < step:
                step = remaining_budget
            if step < 0.0:
                step = 0.0

            # Advance
            x += step * mu
            L += step

            # Event classification (boundary precedence if both hit simultaneously)
            boundary_hit = abs(step - dist_boundary) < eps * dist_boundary + eps
            budget_hit = abs(step - remaining_budget) < eps * remaining_budget + eps

            if boundary_hit:
                # Snap/clamp to boundary
                if abs(x - thickness) < eps * thickness + eps:
                    x = thickness
                elif abs(x) < eps * thickness + eps:
                    x = 0.0
                x = min(max(x, 0.0), thickness)
                space_exit = 1.0
                break

            if budget_hit:
                time_exit = 1.0
                # x is already the location at budget exhaustion
                x = min(max(x, 0.0), thickness)
                break

            # Collision
            num_interactions += 1.0
            if np.random.random() < sigma_a_prob:
                absorbed = 1.0
                x = min(max(x, 0.0), thickness)
                break

            # Scatter: isotropic in 1D angular cosine
            mu = _sample_mu_isotropic_1d(eps)

        # --- Final state ---
        data[i, 4] = x / thickness
        data[i, 5] = mu
        data[i, 6] = L
        data[i, 7] = num_interactions
        data[i, 8] = space_exit
        data[i, 9] = time_exit
        data[i, 10] = absorbed

    return data


# ---------------------------------------------------------------------------
# Backwards-compatible convenience wrapper (same kernel, boundary mode)
# ---------------------------------------------------------------------------

def simulate_cell_1d_time_dependent_boundary(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
):
    """Deprecated: kept for API compatibility.

    Use:
        simulate_cell_1d_time_dependent(..., source_type=SRC_BOUNDARY_X0)

    Returns the *same* 11-column schema as simulate_cell_1d_time_dependent.
    """
    return simulate_cell_1d_time_dependent(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
        source_type=SRC_BOUNDARY_X0,
    )


# ---------------------------------------------------------------------------
# Reference 1D slab solvers (kept as-is)
# ---------------------------------------------------------------------------

@njit
def simulate_1d_slab_time_dependent(
    num_particles: int,
    slab_thickness: float,
    sigma_t: float,
    sigma_s: float,
    particle_velocity: float,
    max_time: float,
    num_time_bins: int,
    source_type: int = 0,
    source_time: float = 0.0
):
    """1D time-dependent MC for a homogeneous slab geometry.

    source_type:
        0 = left boundary (cosine-law incoming),
        1 = uniform internal,
        2 = delta at center

    Returns:
        leakage_left, leakage_right, absorption, flux_center (all vs time-bin)
    """
    sigma_a = sigma_t - sigma_s
    dt = max_time / num_time_bins
    eps = 1e-9

    leakage_left = np.zeros(num_time_bins, dtype=np.float64)
    leakage_right = np.zeros(num_time_bins, dtype=np.float64)
    absorption = np.zeros(num_time_bins, dtype=np.float64)
    flux_center = np.zeros(num_time_bins, dtype=np.float64)

    center_left = 0.45 * slab_thickness
    center_right = 0.55 * slab_thickness
    center_width = center_right - center_left

    for _ in range(num_particles):
        if source_type == 0:
            x = 0.0
            mu = np.sqrt(np.random.random())
            if mu < eps:
                mu = eps
        elif source_type == 1:
            x = np.random.uniform(0.0, slab_thickness)
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps
        else:
            x = 0.5 * slab_thickness
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps

        current_time = source_time
        weight = 1.0
        alive = True

        while alive and current_time < max_time:
            free_path = -np.log(np.random.random()) / sigma_t

            dist_right = (slab_thickness - x) / mu if mu > eps else np.inf
            dist_left = -x / mu if mu < -eps else np.inf
            dist_boundary = dist_right if dist_right < dist_left else dist_left

            time_remaining = max_time - current_time
            dist_to_max_time = time_remaining * particle_velocity

            step_dist = free_path
            if dist_boundary < step_dist:
                step_dist = dist_boundary
            if dist_to_max_time < step_dist:
                step_dist = dist_to_max_time
            if step_dist < 0.0:
                step_dist = 0.0

            travel_time = step_dist / particle_velocity

            x_old = x
            x_new = x + step_dist * mu

            # Track-length flux in a small "center" region
            if mu > eps:
                x_enter = max(x_old, center_left)
                x_exit = min(x_new, center_right)
            else:
                x_enter = min(x_old, center_right)
                x_exit = max(x_new, center_left)

            if (x_old <= center_right and x_new >= center_left) or \
               (x_new <= center_right and x_old >= center_left):
                track_in_center = abs(x_exit - x_enter)
                if track_in_center > 0.0 and center_width > 0.0:
                    t_mid = current_time + 0.5 * travel_time
                    if t_mid < max_time:
                        tbin = int(t_mid / dt)
                        if 0 <= tbin < num_time_bins:
                            flux_center[tbin] += weight * track_in_center / center_width

            x = x_new
            current_time += travel_time

            boundary_hit = abs(step_dist - dist_boundary) < eps * max(1.0, dist_boundary)
            time_hit = abs(step_dist - dist_to_max_time) < eps * max(1.0, dist_to_max_time)

            if time_hit and not boundary_hit:
                alive = False
            elif boundary_hit:
                tbin = int(current_time / dt)
                if tbin >= num_time_bins:
                    tbin = num_time_bins - 1
                if tbin < 0:
                    tbin = 0

                if abs(x) < eps or x < 0.0:
                    leakage_left[tbin] += weight
                else:
                    leakage_right[tbin] += weight
                alive = False
            else:
                if np.random.random() < sigma_a / sigma_t:
                    tbin = int(current_time / dt)
                    if tbin >= num_time_bins:
                        tbin = num_time_bins - 1
                    if tbin < 0:
                        tbin = 0
                    absorption[tbin] += weight
                    alive = False
                else:
                    mu = 2.0 * np.random.random() - 1.0
                    if abs(mu) < eps:
                        mu = eps if np.random.random() > 0.5 else -eps

    return leakage_left, leakage_right, absorption, flux_center


@njit
def simulate_1d_slab_time_dependent_full(
    num_particles: int,
    slab_thickness: float,
    sigma_t: float,
    sigma_s: float,
    particle_velocity: float,
    max_time: float,
    num_time_bins: int,
    num_space_bins: int,
    source_type: int = 0,
    source_time: float = 0.0,
    source_half_width: float = 0.5,
    source_sigma: float = 0.5
):
    """1D time-dependent MC with full spatial flux profile using implicit capture.

    source_type:
        0 = left boundary,
        1 = uniform,
        2 = delta at center,
        3 = square IC at center,
        4 = gaussian IC at center

    Returns:
        flux (num_space_bins, num_time_bins),
        leakage_left, leakage_right, absorption,
        total_weighted_path_length, total_scatters
    """
    sigma_a = sigma_t - sigma_s
    dt = max_time / num_time_bins
    dx = slab_thickness / num_space_bins
    eps = 1e-9
    weight_cutoff = 1e-6
    center_x = 0.5 * slab_thickness

    flux = np.zeros((num_space_bins, num_time_bins), dtype=np.float64)
    leakage_left = np.zeros(num_time_bins, dtype=np.float64)
    leakage_right = np.zeros(num_time_bins, dtype=np.float64)
    absorption = np.zeros(num_time_bins, dtype=np.float64)

    total_weighted_path_length = 0.0
    total_scatters = 0

    for _ in range(num_particles):
        if source_type == 0:
            x = 0.0
            mu = np.sqrt(np.random.random())
            if mu < eps:
                mu = eps
        elif source_type == 1:
            x = np.random.uniform(0.0, slab_thickness)
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps
        elif source_type == 2:
            x = center_x
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps
        elif source_type == 3:
            a = center_x - source_half_width
            b = center_x + source_half_width
            if a < 0.0:
                a = 0.0
            if b > slab_thickness:
                b = slab_thickness
            if b <= a + eps:
                x = center_x
            else:
                x = np.random.uniform(a, b)
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps
        else:
            if source_sigma <= eps:
                x = center_x
            else:
                while True:
                    x_cand = center_x + np.random.normal(0.0, source_sigma)
                    if 0.0 <= x_cand <= slab_thickness:
                        x = x_cand
                        break
            mu = 2.0 * np.random.random() - 1.0
            if abs(mu) < eps:
                mu = eps if np.random.random() > 0.5 else -eps

        current_time = source_time
        weight = 1.0
        alive = True

        while alive and current_time < max_time:
            if sigma_s > eps:
                free_path = -np.log(np.random.random()) / sigma_s
            else:
                free_path = np.inf

            dist_right = (slab_thickness - x) / mu if mu > eps else np.inf
            dist_left = -x / mu if mu < -eps else np.inf
            dist_boundary = dist_right if dist_right < dist_left else dist_left

            time_remaining = max_time - current_time
            dist_to_max_time = time_remaining * particle_velocity

            step_dist = free_path
            if dist_boundary < step_dist:
                step_dist = dist_boundary
            if dist_to_max_time < step_dist:
                step_dist = dist_to_max_time
            if step_dist < 0.0:
                step_dist = 0.0

            if sigma_a > 1e-12:
                total_weighted_path_length += weight * (1.0 - np.exp(-sigma_a * step_dist)) / sigma_a
            else:
                total_weighted_path_length += weight * step_dist

            travel_time = step_dist / particle_velocity

            x_old = x
            x_new = x + step_dist * mu

            # Segment the track to accumulate space/time-binned flux and absorption
            s_traveled = 0.0
            abs_mu = abs(mu) if abs(mu) > 1e-12 else 1e-12

            while s_traveled < step_dist - 1e-12:
                x_seg = x_old + mu * s_traveled
                t_seg = current_time + s_traveled / particle_velocity
                x_for_bin = x_seg + (1e-12 if mu > 0.0 else (-1e-12 if mu < 0.0 else 0.0))
                sbin = int(x_for_bin / dx)
                if sbin < 0:
                    sbin = 0
                if sbin >= num_space_bins:
                    sbin = num_space_bins - 1

                tbin = int(t_seg / dt)
                if tbin < 0:
                    tbin = 0
                if tbin >= num_time_bins:
                    tbin = num_time_bins - 1

                if mu > 0.0:
                    boundary_x = (sbin + 1) * dx
                    dist_space = (boundary_x - x_seg) / abs_mu
                else:
                    boundary_x = sbin * dx
                    dist_space = (x_seg - boundary_x) / abs_mu

                time_boundary = (tbin + 1) * dt
                dist_time = (time_boundary - t_seg) * particle_velocity

                if dist_space < 0.0:
                    dist_space = 0.0
                if dist_time < 0.0:
                    dist_time = 0.0

                remaining = step_dist - s_traveled
                seg = remaining
                if dist_space < seg:
                    seg = dist_space
                if dist_time < seg:
                    seg = dist_time

                if seg <= 0.0:
                    s_traveled += 1e-14
                    continue

                if sigma_a > 1e-12:
                    w_seg_start = weight * np.exp(-sigma_a * s_traveled)
                    w_seg_end = w_seg_start * np.exp(-sigma_a * seg)
                    flux_val = (w_seg_start - w_seg_end) / sigma_a
                    absorption[tbin] += (w_seg_start - w_seg_end)
                else:
                    flux_val = weight * seg

                flux[sbin, tbin] += flux_val / dx
                s_traveled += seg

            weight = weight * np.exp(-sigma_a * step_dist)

            x = x_new
            current_time += travel_time

            if weight < weight_cutoff:
                alive = False
                continue

            boundary_hit = abs(step_dist - dist_boundary) < eps * 1.1
            time_hit = abs(step_dist - dist_to_max_time) < eps * 1.1

            if time_hit and not boundary_hit:
                alive = False
            elif boundary_hit:
                tbin = int(current_time / dt)
                if tbin >= num_time_bins:
                    tbin = num_time_bins - 1
                if tbin < 0:
                    tbin = 0

                if abs(x) < eps or x < 0.0:
                    leakage_left[tbin] += weight
                else:
                    leakage_right[tbin] += weight
                alive = False
            else:
                total_scatters += 1
                mu = 2.0 * np.random.random() - 1.0
                if abs(mu) < eps:
                    mu = eps if np.random.random() > 0.5 else -eps

    return flux, leakage_left, leakage_right, absorption, total_weighted_path_length, total_scatters
