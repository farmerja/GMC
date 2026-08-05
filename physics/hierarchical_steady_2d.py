from __future__ import annotations

import math

import numpy as np


FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3
FACE_NAMES = ("left", "right", "top", "bottom")

EPSILON = 1.0e-12
LEGENDRE_BASIS_ORDER: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (2, 0),
    (1, 1),
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 2),
)


def face_local_coordinate_2d(face: int, x: float, y: float) -> float:
    return float(y if int(face) in (FACE_LEFT, FACE_RIGHT) else x)


def opposite_face_2d(face: int) -> int:
    face_int = int(face)
    if face_int == FACE_LEFT:
        return FACE_RIGHT
    if face_int == FACE_RIGHT:
        return FACE_LEFT
    if face_int == FACE_TOP:
        return FACE_BOTTOM
    if face_int == FACE_BOTTOM:
        return FACE_TOP
    raise ValueError(f"Unknown face id: {face}")


def boundary_start_position_2d(face: int, coord: float, width: float, height: float) -> tuple[float, float]:
    face_int = int(face)
    coord_f = float(coord)
    width_f = float(width)
    height_f = float(height)
    if face_int == FACE_LEFT:
        return 0.0, coord_f
    if face_int == FACE_RIGHT:
        return width_f, coord_f
    if face_int == FACE_TOP:
        return coord_f, height_f
    if face_int == FACE_BOTTOM:
        return coord_f, 0.0
    raise ValueError(f"Unknown face id: {face}")


def direction_is_outward_for_face_2d(face: int, u: float, v: float, eps: float = EPSILON) -> bool:
    face_int = int(face)
    if face_int == FACE_LEFT:
        return float(u) <= eps
    if face_int == FACE_RIGHT:
        return float(u) >= -eps
    if face_int == FACE_TOP:
        return float(v) >= -eps
    if face_int == FACE_BOTTOM:
        return float(v) <= eps
    raise ValueError(f"Unknown face id: {face}")


def ballistic_exit_2d(
    x0: float,
    y0: float,
    u: float,
    v: float,
    width: float,
    height: float,
    eps: float = EPSILON,
) -> tuple[float, int, float, float, float]:
    x = float(x0)
    y = float(y0)
    u_f = float(u)
    v_f = float(v)
    width_f = float(width)
    height_f = float(height)

    distances = np.array(
        [
            (0.0 - x) / u_f if u_f < -eps else np.inf,
            (width_f - x) / u_f if u_f > eps else np.inf,
            (height_f - y) / v_f if v_f > eps else np.inf,
            (0.0 - y) / v_f if v_f < -eps else np.inf,
        ],
        dtype=np.float64,
    )
    distances = np.where(distances >= 0.0, distances, np.inf)
    face = int(np.argmin(distances))
    distance = float(distances[face])
    if not np.isfinite(distance):
        return 0.0, FACE_LEFT, float(np.clip(x, 0.0, width_f)), float(np.clip(y, 0.0, height_f)), face_local_coordinate_2d(FACE_LEFT, x, y)

    x_exit = float(np.clip(x + distance * u_f, 0.0, width_f))
    y_exit = float(np.clip(y + distance * v_f, 0.0, height_f))
    if face == FACE_LEFT:
        x_exit = 0.0
    elif face == FACE_RIGHT:
        x_exit = width_f
    elif face == FACE_TOP:
        y_exit = height_f
    else:
        y_exit = 0.0
    coord = face_local_coordinate_2d(face, x_exit, y_exit)
    return distance, face, x_exit, y_exit, coord


def scatter_before_exit_probability(sigma_s: float, distance: float) -> float:
    optical_distance = max(float(sigma_s), 0.0) * max(float(distance), 0.0)
    if optical_distance <= 0.0:
        return 0.0
    if optical_distance >= 64.0:
        return 1.0
    return float(-math.expm1(-optical_distance))


def _legendre(order: int, value: float) -> float:
    x = float(value)
    if order == 0:
        return 1.0
    if order == 1:
        return x
    if order == 2:
        return 0.5 * (3.0 * x * x - 1.0)
    raise ValueError(f"Unsupported Legendre order: {order}")


def compute_straight_track_integrals_2d(
    *,
    width: float,
    height: float,
    x0: float,
    y0: float,
    u: float,
    v: float,
    length: float,
    sigma_a: float,
) -> tuple[np.float64, np.ndarray, np.ndarray]:
    ballistic_length, _face, _x_exit, _y_exit, _coord = ballistic_exit_2d(
        x0=x0,
        y0=y0,
        u=u,
        v=v,
        width=width,
        height=height,
    )
    path_length = max(0.0, min(float(length), ballistic_length))
    sigma_a_f = max(float(sigma_a), 0.0)

    if path_length <= 0.0:
        return np.float64(0.0), np.zeros(5, dtype=np.float64), np.zeros(len(LEGENDRE_BASIS_ORDER), dtype=np.float64)

    if sigma_a_f > EPSILON:
        weighted_track = (1.0 - math.exp(-sigma_a_f * path_length)) / sigma_a_f
    else:
        weighted_track = path_length

    u_f = float(u)
    v_f = float(v)
    angular = np.asarray(
        [
            weighted_track * u_f,
            weighted_track * v_f,
            weighted_track * u_f * u_f,
            weighted_track * u_f * v_f,
            weighted_track * v_f * v_f,
        ],
        dtype=np.float64,
    )
    legendre = np.asarray(
        [weighted_track * _legendre(px, u_f) * _legendre(py, v_f) for px, py in LEGENDRE_BASIS_ORDER],
        dtype=np.float64,
    )
    return np.float64(weighted_track), angular, legendre


__all__ = [
    "EPSILON",
    "FACE_BOTTOM",
    "FACE_LEFT",
    "FACE_NAMES",
    "FACE_RIGHT",
    "FACE_TOP",
    "LEGENDRE_BASIS_ORDER",
    "ballistic_exit_2d",
    "boundary_start_position_2d",
    "compute_straight_track_integrals_2d",
    "direction_is_outward_for_face_2d",
    "face_local_coordinate_2d",
    "opposite_face_2d",
    "scatter_before_exit_probability",
]
