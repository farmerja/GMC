"""mc1d.py - 1D single-cell steady-state Monte Carlo transport."""

import numpy as np
from numba import njit, prange


@njit()
def sample_unit(distribution: int) -> float:
    """Sample a value in [0,1] using specified distribution."""
    if distribution == 0:
        return np.random.uniform(0, 1)
    elif distribution == 1:
        while True:
            candidate = np.random.normal(loc=0.5, scale=0.15)
            if 0 <= candidate <= 1:
                return candidate
    elif distribution == 2:
        return np.random.beta(a=2, b=5)
    elif distribution == 3:
        while True:
            candidate = np.random.poisson(lam=0.5)
            if 0 <= candidate <= 1:
                return float(candidate)
    elif distribution == 4:
        while True:
            candidate = np.random.exponential(scale=0.5)
            if 0 <= candidate <= 1:
                return candidate
    else:
        return np.random.uniform(0, 1)


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
def simulate_cell_with_random_thickness(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    cutoff: float = 1.0e-14,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    cond_samples=None,
):
    """1D boundary source simulation with random thickness.

    Returns array with columns:
      [0] initial_mu, [1] final_mu, [2] total_path, [3] num_interactions,
      [4] exited_cell, [5] absorbed_in_cell, [6] thickness
    """
    data = np.zeros((num_neutrons, 7), dtype=np.float64)
    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    for i in prange(num_neutrons):
        candidate_thickness = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        log_min = np.log10(min_thickness)
        log_max = np.log10(max_thickness)
        log_thickness = log_min + candidate_thickness * (log_max - log_min)
        thickness_i = 10 ** log_thickness
        data[i, 6] = thickness_i

        if use_precond:
            mu = _sample_mu_incoming_cosine_1d_from_unit(cond_samples[i, 1], eps)
        else:
            mu = _sample_mu_incoming_cosine_1d_from_unit(np.random.random(), eps)
        data[i, 0] = mu

        x_position = 0.0
        total_path = 0.0
        num_interactions = 0
        exited_cell = 0
        absorbed_in_cell = 0
        alive = True

        while alive:
            free_path = -np.log(np.random.random()) / sigma_t
            distance_to_right = (thickness_i - x_position) / mu if mu > eps else np.inf
            distance_to_left = -x_position / mu if mu < -eps else np.inf
            dist_boundary = min(distance_to_right, distance_to_left)

            step_dist = min(free_path, dist_boundary)
            x_position += step_dist * mu
            total_path += abs(step_dist)

            boundary_hit = abs(step_dist - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(x_position - thickness_i) < eps:
                    x_position = thickness_i
                elif abs(x_position - 0.0) < eps:
                    x_position = 0.0
                exited_cell = 1
                data[i, 1] = mu
                data[i, 2] = total_path
                data[i, 3] = num_interactions
                data[i, 4] = exited_cell
                data[i, 5] = absorbed_in_cell
                alive = False
            else:
                num_interactions += 1
                if np.random.random() < sigma_a_prob:
                    absorbed_in_cell = 1
                    data[i, 1] = mu
                    data[i, 2] = total_path
                    data[i, 3] = num_interactions
                    data[i, 4] = exited_cell
                    data[i, 5] = absorbed_in_cell
                    alive = False
                else:
                    mu = 2.0 * np.random.random() - 1.0
                    if abs(mu) < eps:
                        mu = eps if np.random.random() > 0.5 else -eps
    return data


@njit(parallel=True)
def simulate_internal_source_cell(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    cond_samples=None,
):
    """1D internal source simulation with random thickness.

    Returns array with columns:
      [0] initial_mu, [1] final_mu, [2] total_path, [3] num_interactions,
      [4] exited_cell, [5] absorbed_in_cell, [6] thickness, [7] initial_position_x
    """
    data = np.zeros((num_neutrons, 8), dtype=np.float64)
    sigma_a_prob = 1.0 - scatter_ratio
    sigma_t = 1.0
    eps = 1e-9
    use_precond = cond_samples is not None

    for i in prange(num_neutrons):
        candidate_thickness = cond_samples[i, 0] if use_precond else sample_unit(distribution)
        log_min = np.log10(min_thickness)
        log_max = np.log10(max_thickness)
        log_thickness = log_min + candidate_thickness * (log_max - log_min)
        thickness_i = 10 ** log_thickness
        data[i, 6] = thickness_i

        x_position = thickness_i * (cond_samples[i, 1] if use_precond else np.random.random())
        data[i, 7] = x_position

        if use_precond:
            mu = _sample_mu_isotropic_1d_from_unit(cond_samples[i, 2], eps)
        else:
            mu = _sample_mu_isotropic_1d_from_unit(np.random.random(), eps)
        data[i, 0] = mu

        total_path = 0.0
        num_interactions = 0
        exited_cell = 0
        absorbed_in_cell = 0
        alive = True

        while alive:
            free_path = -np.log(np.random.random()) / sigma_t
            distance_to_right = (thickness_i - x_position) / mu if mu > eps else np.inf
            distance_to_left = -x_position / mu if mu < -eps else np.inf
            dist_boundary = min(distance_to_right, distance_to_left)

            step_dist = min(free_path, dist_boundary)
            x_position += step_dist * mu
            total_path += abs(step_dist)

            boundary_hit = abs(step_dist - dist_boundary) < eps * dist_boundary + eps

            if boundary_hit:
                if abs(x_position - thickness_i) < eps:
                    x_position = thickness_i
                elif abs(x_position - 0.0) < eps:
                    x_position = 0.0
                exited_cell = 1
                data[i, 1] = mu
                data[i, 2] = total_path
                data[i, 3] = num_interactions
                data[i, 4] = exited_cell
                data[i, 5] = absorbed_in_cell
                alive = False
            else:
                num_interactions += 1
                if np.random.random() < sigma_a_prob:
                    absorbed_in_cell = 1
                    data[i, 1] = mu
                    data[i, 2] = total_path
                    data[i, 3] = num_interactions
                    data[i, 4] = exited_cell
                    data[i, 5] = absorbed_in_cell
                    alive = False
                else:
                    mu = 2.0 * np.random.random() - 1.0
                    if abs(mu) < eps:
                        mu = eps if np.random.random() > 0.5 else -eps
    return data
