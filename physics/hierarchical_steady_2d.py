"""Optional hierarchical steady-state 2D single-cell add-on.

This module keeps the existing monolithic steady-2D workflow untouched and
provides the first slice of an alternative hierarchical decomposition for
rectangular cells in isotropic media:

1. Compute exact ballistic exit geometry from the initial state.
2. Evaluate/sample the exact gate for whether a first collision occurs before exit.
3. Generate scattered-only histories without rejection by sampling the first
   collision distance from the exponential truncated to ``[0, L_ballistic]``.
4. Return face-first outputs suited for future staged models.

The transport geometry is 2D, while directions follow the repository's current
steady-2D convention: directions are sampled in 3D and projected onto x-y for
transport and ML features. The projected pair ``(u, v)`` is never renormalized
to unit length.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .mc1d import sample_unit
from .steady_2d_scattering import (
    guard_projected_direction_xy,
    sample_incoming_hemisphere_plus_x_direction_3d_from_unit,
    sample_isotropic_direction_3d_from_unit,
)


FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3

SRC_BOUNDARY_X0 = 0
SRC_INTERNAL = 1

FACE_NAMES = ("left", "right", "top", "bottom")

EPSILON = 1.0e-12


@njit(inline="always")
def _clip_open_interval(value: float, eps: float):
    return min(max(value, eps), 1.0 - eps)


@njit(inline="always")
def face_length_2d(face: int, width: float, height: float) -> float:
    if face == FACE_LEFT or face == FACE_RIGHT:
        return height
    return width


@njit(inline="always")
def face_coordinate_2d(face: int, x: float, y: float) -> float:
    """Return the local coordinate along a face: y on left/right, x on top/bottom."""
    if face == FACE_LEFT or face == FACE_RIGHT:
        return y
    return x


@njit(inline="always")
def direction_is_outward_for_face_2d(face: int, u: float, v: float, eps: float = EPSILON) -> bool:
    if face == FACE_LEFT:
        return u < -eps
    if face == FACE_RIGHT:
        return u > eps
    if face == FACE_TOP:
        return v > eps
    if face == FACE_BOTTOM:
        return v < -eps
    return False


@njit(inline="always")
def _snap_to_face(face: int, x: float, y: float, width: float, height: float):
    x_out = min(max(x, 0.0), width)
    y_out = min(max(y, 0.0), height)
    if face == FACE_LEFT:
        x_out = 0.0
    elif face == FACE_RIGHT:
        x_out = width
    elif face == FACE_TOP:
        y_out = height
    elif face == FACE_BOTTOM:
        y_out = 0.0
    return x_out, y_out


@njit
def ballistic_exit_2d(x: float, y: float, u: float, v: float, width: float, height: float, eps: float = EPSILON):
    """Return exact ballistic exit distance/face/point for a rectangle."""
    distance = math.inf
    face = -1

    if u > eps:
        candidate = (width - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_RIGHT
    elif u < -eps:
        candidate = (0.0 - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_LEFT

    if v > eps:
        candidate = (height - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_TOP
    elif v < -eps:
        candidate = (0.0 - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_BOTTOM

    if face < 0 or not np.isfinite(distance):
        return 0.0, face, x, y, 0.0

    distance = max(distance, 0.0)
    x_out, y_out = _snap_to_face(face, x + distance * u, y + distance * v, width, height)
    pos_on_face = face_coordinate_2d(face, x_out, y_out)
    return distance, face, x_out, y_out, pos_on_face


@njit(inline="always")
def scatter_before_exit_probability(sigma_t: float, ballistic_length: float) -> float:
    """Exact gate probability for a first collision before ballistic exit."""
    if sigma_t <= 0.0 or ballistic_length <= 0.0:
        return 0.0
    return 1.0 - math.exp(-sigma_t * ballistic_length)


@njit(inline="always")
def sample_truncated_first_collision_distance(unit: float, sigma_t: float, ballistic_length: float):
    """Sample Exp(sigma_t) conditioned on collision before ballistic exit."""
    if sigma_t <= 0.0 or ballistic_length <= 0.0:
        return 0.0

    unit_clipped = _clip_open_interval(unit, EPSILON)
    no_collision_prob = math.exp(-sigma_t * ballistic_length)
    mass = 1.0 - no_collision_prob
    return -math.log(1.0 - unit_clipped * mass) / sigma_t


@njit(inline="always")
def _sample_width_height(distribution: int, min_width: float, max_width: float, min_height: float, max_height: float, use_precond: bool, cond_samples, row: int):
    log_min_w = math.log10(min_width)
    log_max_w = math.log10(max_width)
    log_min_h = math.log10(min_height)
    log_max_h = math.log10(max_height)

    cand_w = cond_samples[row, 0] if use_precond else sample_unit(distribution)
    cand_h = cond_samples[row, 1] if use_precond else sample_unit(distribution)
    width = 10.0 ** (log_min_w + cand_w * (log_max_w - log_min_w))
    height = 10.0 ** (log_min_h + cand_h * (log_max_h - log_min_h))
    return width, height


@njit(inline="always")
def _sample_initial_state(source_type: int, width: float, height: float, use_precond: bool, cond_samples, row: int, eps: float):
    if source_type == SRC_BOUNDARY_X0:
        unit_y = cond_samples[row, 2] if use_precond else np.random.random()
        unit_y = _clip_open_interval(unit_y, eps)
        x = 0.0
        y = height * unit_y
        unit_a = cond_samples[row, 3] if use_precond else np.random.random()
        unit_b = cond_samples[row, 4] if use_precond else np.random.random()
        u, v, _ = sample_incoming_hemisphere_plus_x_direction_3d_from_unit(unit_a, unit_b, eps)
    else:
        unit_x = cond_samples[row, 2] if use_precond else np.random.random()
        unit_y = cond_samples[row, 3] if use_precond else np.random.random()
        unit_x = _clip_open_interval(unit_x, eps)
        unit_y = _clip_open_interval(unit_y, eps)
        x = width * unit_x
        y = height * unit_y
        unit_a = cond_samples[row, 4] if use_precond else np.random.random()
        unit_b = cond_samples[row, 5] if use_precond else np.random.random()
        u, v, _ = sample_isotropic_direction_3d_from_unit(unit_a, unit_b, eps)

    u, v = guard_projected_direction_xy(u, v, eps)
    return x, y, u, v


@njit(inline="always")
def _sample_isotropic_projected_direction(eps: float):
    u, v, _ = sample_isotropic_direction_3d_from_unit(np.random.random(), np.random.random(), eps)
    return guard_projected_direction_xy(u, v, eps)


@njit
def _transport_after_first_scatter(width: float, height: float, x: float, y: float, sigma_t: float, eps: float):
    """Continue isotropic pure-scatter transport from a collision site to exit."""
    total_track = 0.0
    num_scatters = 1
    first_u, first_v = _sample_isotropic_projected_direction(eps)
    u = first_u
    v = first_v

    while True:
        boundary_distance, face, x_exit, y_exit, pos_on_face = ballistic_exit_2d(x, y, u, v, width, height, eps)
        free_path = -math.log(max(np.random.random(), eps)) / sigma_t
        travel = min(boundary_distance, free_path)

        x += travel * u
        y += travel * v
        total_track += travel

        boundary_hit = free_path >= boundary_distance - eps * max(1.0, boundary_distance)
        if boundary_hit:
            x_exit, y_exit = _snap_to_face(face, x, y, width, height)
            pos_on_face = face_coordinate_2d(face, x_exit, y_exit)
            return num_scatters, total_track, x_exit, y_exit, face, pos_on_face, u, v, first_u, first_v

        num_scatters += 1
        u, v = _sample_isotropic_projected_direction(eps)


@njit(parallel=True)
def simulate_hierarchical_steady_2d_gate(
    num_neutrons: int,
    sigma_t: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    cond_samples=None,
    source_type: int = SRC_INTERNAL,
):
    """Generate exact ballistic/gate metadata for the hierarchical add-on.

    Returns columns:
      [0] width, [1] height,
      [2] initial_pos_x, [3] initial_pos_y,
      [4] initial_dir_u, [5] initial_dir_v,
      [6] ballistic_length, [7] ballistic_face,
      [8] ballistic_exit_x, [9] ballistic_exit_y,
      [10] ballistic_face_coord_local, [11] scatter_before_exit_prob,
      [12] scatter_before_exit
    """
    data = np.zeros((num_neutrons, 13), dtype=np.float64)
    use_precond = cond_samples is not None
    eps = EPSILON

    for i in prange(num_neutrons):
        width, height = _sample_width_height(
            distribution,
            min_width,
            max_width,
            min_height,
            max_height,
            use_precond,
            cond_samples,
            i,
        )
        x0, y0, u0, v0 = _sample_initial_state(source_type, width, height, use_precond, cond_samples, i, eps)
        ballistic_length, ballistic_face, x_ball, y_ball, coord_ball = ballistic_exit_2d(x0, y0, u0, v0, width, height, eps)
        gate_prob = scatter_before_exit_probability(sigma_t, ballistic_length)
        scatter_before_exit = 1.0 if np.random.random() < gate_prob else 0.0

        data[i, 0] = width
        data[i, 1] = height
        data[i, 2] = x0
        data[i, 3] = y0
        data[i, 4] = u0
        data[i, 5] = v0
        data[i, 6] = ballistic_length
        data[i, 7] = ballistic_face
        data[i, 8] = x_ball
        data[i, 9] = y_ball
        data[i, 10] = coord_ball
        data[i, 11] = gate_prob
        data[i, 12] = scatter_before_exit

    return data


@njit(parallel=True)
def simulate_hierarchical_steady_2d_scattered_only(
    num_neutrons: int,
    sigma_t: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    cond_samples=None,
    source_type: int = SRC_INTERNAL,
):
    """Generate histories conditioned on at least one scatter before exit.

    The first collision distance is sampled exactly from the exponential
    distribution truncated to the ballistic path length, so no rejection
    sampling is used for the conditioning event.

    Returns columns:
      [0] width, [1] height,
      [2] initial_pos_x, [3] initial_pos_y,
      [4] initial_dir_u, [5] initial_dir_v,
      [6] ballistic_length, [7] ballistic_face,
      [8] ballistic_exit_x, [9] ballistic_exit_y,
      [10] ballistic_face_coord_local, [11] scatter_before_exit_prob,
      [12] first_scatter_distance,
      [13] first_scatter_pos_x, [14] first_scatter_pos_y,
      [15] first_scatter_out_dir_u, [16] first_scatter_out_dir_v,
      [17] num_scatters, [18] total_track_length,
      [19] final_pos_x, [20] final_pos_y,
      [21] exit_face, [22] exit_face_coord_local,
      [23] final_dir_u, [24] final_dir_v
    """
    data = np.zeros((num_neutrons, 25), dtype=np.float64)
    use_precond = cond_samples is not None
    eps = EPSILON

    for i in prange(num_neutrons):
        width, height = _sample_width_height(
            distribution,
            min_width,
            max_width,
            min_height,
            max_height,
            use_precond,
            cond_samples,
            i,
        )
        x0, y0, u0, v0 = _sample_initial_state(source_type, width, height, use_precond, cond_samples, i, eps)
        ballistic_length, ballistic_face, x_ball, y_ball, coord_ball = ballistic_exit_2d(x0, y0, u0, v0, width, height, eps)
        gate_prob = scatter_before_exit_probability(sigma_t, ballistic_length)

        if sigma_t <= 0.0 or gate_prob <= 0.0:
            data[i, 0] = width
            data[i, 1] = height
            data[i, 2] = x0
            data[i, 3] = y0
            data[i, 4] = u0
            data[i, 5] = v0
            data[i, 6] = ballistic_length
            data[i, 7] = ballistic_face
            data[i, 8] = x_ball
            data[i, 9] = y_ball
            data[i, 10] = coord_ball
            data[i, 11] = gate_prob
            data[i, 12] = np.nan
            data[i, 13] = np.nan
            data[i, 14] = np.nan
            data[i, 15] = np.nan
            data[i, 16] = np.nan
            data[i, 17] = 0.0
            data[i, 18] = np.nan
            data[i, 19] = np.nan
            data[i, 20] = np.nan
            data[i, 21] = np.nan
            data[i, 22] = np.nan
            data[i, 23] = np.nan
            data[i, 24] = np.nan
            continue

        first_scatter_distance = sample_truncated_first_collision_distance(np.random.random(), sigma_t, ballistic_length)

        x_scatter = x0 + first_scatter_distance * u0
        y_scatter = y0 + first_scatter_distance * v0
        x_scatter = min(max(x_scatter, 0.0), width)
        y_scatter = min(max(y_scatter, 0.0), height)

        num_scatters, post_scatter_track, x_exit, y_exit, exit_face, exit_coord, u_exit, v_exit, first_u, first_v = _transport_after_first_scatter(
            width,
            height,
            x_scatter,
            y_scatter,
            sigma_t,
            eps,
        )

        data[i, 0] = width
        data[i, 1] = height
        data[i, 2] = x0
        data[i, 3] = y0
        data[i, 4] = u0
        data[i, 5] = v0
        data[i, 6] = ballistic_length
        data[i, 7] = ballistic_face
        data[i, 8] = x_ball
        data[i, 9] = y_ball
        data[i, 10] = coord_ball
        data[i, 11] = gate_prob
        data[i, 12] = first_scatter_distance
        data[i, 13] = x_scatter
        data[i, 14] = y_scatter
        data[i, 15] = first_u
        data[i, 16] = first_v
        data[i, 17] = num_scatters
        data[i, 18] = first_scatter_distance + post_scatter_track
        data[i, 19] = x_exit
        data[i, 20] = y_exit
        data[i, 21] = exit_face
        data[i, 22] = exit_coord
        data[i, 23] = u_exit
        data[i, 24] = v_exit

    return data
