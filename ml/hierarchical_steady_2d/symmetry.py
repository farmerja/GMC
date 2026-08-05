from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
)


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

_LEGENDRE_INDEX = {basis: idx for idx, basis in enumerate(LEGENDRE_BASIS_ORDER)}

_D4_TRANSFORMS = (
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64),
    np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
    np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
    np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
    np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64),
)

_FACE_TO_LEFT_TRANSFORM = {
    FACE_LEFT: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    FACE_RIGHT: np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
    FACE_TOP: np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64),
    FACE_BOTTOM: np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64),
}


@dataclass(frozen=True)
class CanonicalizedInternalState2D:
    width: float
    height: float
    x: float
    y: float
    u: float
    v: float
    transform: np.ndarray


@dataclass(frozen=True)
class CanonicalizedBoundaryState2D:
    width: float
    height: float
    y_on_entry: float
    u: float
    v: float
    transform: np.ndarray


def _dims_after_transform(matrix: np.ndarray, width: float, height: float) -> tuple[float, float]:
    dims = np.abs(matrix) @ np.array([width, height], dtype=np.float64)
    return float(dims[0]), float(dims[1])


def _transform_centered_point(
    matrix: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    centered = np.array([x - 0.5 * width, y - 0.5 * height], dtype=np.float64)
    transformed = matrix @ centered
    width_t, height_t = _dims_after_transform(matrix, width, height)
    return (
        float(transformed[0] + 0.5 * width_t),
        float(transformed[1] + 0.5 * height_t),
        width_t,
        height_t,
    )


def _transform_direction(matrix: np.ndarray, u: float, v: float) -> tuple[float, float]:
    out = matrix @ np.array([u, v], dtype=np.float64)
    return float(out[0]), float(out[1])


def classify_boundary_face(x: float, y: float, width: float, height: float, eps: float = EPSILON) -> int:
    distances = np.array([abs(x), abs(width - x), abs(height - y), abs(y)], dtype=np.float64)
    face = int(np.argmin(distances))
    if face == 0:
        return FACE_LEFT
    if face == 1:
        return FACE_RIGHT
    if face == 2:
        return FACE_TOP
    return FACE_BOTTOM


def face_local_coordinate(face: int, x: float, y: float) -> float:
    return float(y if face in (FACE_LEFT, FACE_RIGHT) else x)


def point_from_boundary_face(face: int, coord: float, width: float, height: float) -> tuple[float, float]:
    if face == FACE_LEFT:
        return 0.0, float(coord)
    if face == FACE_RIGHT:
        return float(width), float(coord)
    if face == FACE_TOP:
        return float(coord), float(height)
    if face == FACE_BOTTOM:
        return float(coord), 0.0
    raise ValueError(f"Unknown boundary face: {face}")


def encode_boundary_unit_vector(x: float, y: float, width: float, height: float, eps: float = EPSILON) -> tuple[float, float]:
    dx = float(x) - 0.5 * float(width)
    dy = float(y) - 0.5 * float(height)
    norm = math.hypot(dx, dy)
    if norm < eps:
        return 1.0, 0.0
    return dx / norm, dy / norm


def decode_boundary_unit_vector(
    unit_x: float,
    unit_y: float,
    width: float,
    height: float,
    eps: float = EPSILON,
) -> tuple[float, float, int, float]:
    dx = float(unit_x)
    dy = float(unit_y)
    norm = math.hypot(dx, dy)
    if norm < eps:
        dx, dy = 1.0, 0.0
        norm = 1.0
    dx /= norm
    dy /= norm
    tx = math.inf if abs(dx) < eps else (0.5 * float(width)) / abs(dx)
    ty = math.inf if abs(dy) < eps else (0.5 * float(height)) / abs(dy)
    distance = min(tx, ty)
    x = 0.5 * float(width) + distance * dx
    y = 0.5 * float(height) + distance * dy
    x = min(max(x, 0.0), float(width))
    y = min(max(y, 0.0), float(height))
    face = classify_boundary_face(x, y, width, height, eps=eps)
    if face == FACE_LEFT:
        x = 0.0
    elif face == FACE_RIGHT:
        x = float(width)
    elif face == FACE_TOP:
        y = float(height)
    else:
        y = 0.0
    coord = face_local_coordinate(face, x, y)
    return x, y, face, coord


def _internal_score(width: float, height: float, x: float, y: float, u: float, v: float, eps: float) -> tuple[float, ...]:
    x_rel = x / max(width, eps)
    y_rel = y / max(height, eps)
    return (
        0.0 if width >= height - eps else 1.0,
        0.0 if u >= -eps else 1.0,
        0.0 if v >= -eps else 1.0,
        0.0 if u + eps >= v else 1.0,
        max(-u, 0.0),
        max(-v, 0.0),
        max(v - u, 0.0),
        -u,
        v,
        x_rel,
        y_rel,
    )


def canonicalize_internal_state(
    width: float,
    height: float,
    x: float,
    y: float,
    u: float,
    v: float,
    eps: float = EPSILON,
) -> CanonicalizedInternalState2D:
    best: CanonicalizedInternalState2D | None = None
    best_score: tuple[float, ...] | None = None
    for matrix in _D4_TRANSFORMS:
        x_t, y_t, width_t, height_t = _transform_centered_point(matrix, x, y, width, height)
        u_t, v_t = _transform_direction(matrix, u, v)
        score = _internal_score(width_t, height_t, x_t, y_t, u_t, v_t, eps)
        if best_score is None or score < best_score:
            best = CanonicalizedInternalState2D(
                width=width_t,
                height=height_t,
                x=x_t,
                y=y_t,
                u=u_t,
                v=v_t,
                transform=matrix.copy(),
            )
            best_score = score
    if best is None:
        raise RuntimeError("Failed to canonicalize internal state.")
    return best


def _boundary_score(width: float, height: float, y_on_entry: float, u: float, v: float, eps: float) -> tuple[float, ...]:
    y_rel = y_on_entry / max(height, eps)
    return (
        0.0 if width >= height - eps else 1.0,
        0.0 if u > eps else 1.0,
        0.0 if v >= -eps else 1.0,
        max(-u, 0.0),
        max(-v, 0.0),
        -u,
        v,
        y_rel,
    )


def _map_boundary_state_to_left_entry(
    face: int,
    coord: float,
    u: float,
    v: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float, np.ndarray]:
    matrix = _FACE_TO_LEFT_TRANSFORM[int(face)]
    x_face, y_face = point_from_boundary_face(int(face), coord, width, height)
    x_t, y_t, width_t, height_t = _transform_centered_point(matrix, x_face, y_face, width, height)
    u_t, v_t = _transform_direction(matrix, u, v)
    return width_t, height_t, y_t, u_t, v_t, matrix.copy()


def canonicalize_boundary_state(
    entry_face: int,
    entry_coord: float,
    u: float,
    v: float,
    width: float,
    height: float,
    eps: float = EPSILON,
) -> CanonicalizedBoundaryState2D:
    x0, y0 = point_from_boundary_face(int(entry_face), float(entry_coord), width, height)
    best: CanonicalizedBoundaryState2D | None = None
    best_score: tuple[float, ...] | None = None
    for matrix in _D4_TRANSFORMS:
        x_t, y_t, width_t, height_t = _transform_centered_point(matrix, x0, y0, width, height)
        u_t, v_t = _transform_direction(matrix, u, v)
        face_t = classify_boundary_face(x_t, y_t, width_t, height_t, eps=eps)
        coord_t = face_local_coordinate(face_t, x_t, y_t)
        width_m, height_m, y_entry, u_m, v_m, face_matrix = _map_boundary_state_to_left_entry(
            face_t,
            coord_t,
            u_t,
            v_t,
            width_t,
            height_t,
        )
        total_matrix = face_matrix @ matrix
        score = _boundary_score(width_m, height_m, y_entry, u_m, v_m, eps)
        if best_score is None or score < best_score:
            best = CanonicalizedBoundaryState2D(
                width=width_m,
                height=height_m,
                y_on_entry=y_entry,
                u=u_m,
                v=v_m,
                transform=total_matrix.copy(),
            )
            best_score = score
    if best is None:
        raise RuntimeError("Failed to canonicalize boundary state.")
    return best


def inverse_transform_point(
    transform: np.ndarray,
    width_physical: float,
    height_physical: float,
    width_canonical: float,
    height_canonical: float,
    x_canonical: float,
    y_canonical: float,
) -> tuple[float, float]:
    centered = np.array(
        [x_canonical - 0.5 * width_canonical, y_canonical - 0.5 * height_canonical],
        dtype=np.float64,
    )
    physical = transform.T @ centered
    return (
        float(physical[0] + 0.5 * width_physical),
        float(physical[1] + 0.5 * height_physical),
    )


def forward_transform_point(
    transform: np.ndarray,
    width_physical: float,
    height_physical: float,
    width_canonical: float,
    height_canonical: float,
    x_physical: float,
    y_physical: float,
) -> tuple[float, float]:
    centered = np.array(
        [x_physical - 0.5 * width_physical, y_physical - 0.5 * height_physical],
        dtype=np.float64,
    )
    canonical = transform @ centered
    return (
        float(canonical[0] + 0.5 * width_canonical),
        float(canonical[1] + 0.5 * height_canonical),
    )


def forward_transform_direction(transform: np.ndarray, u: float, v: float) -> tuple[float, float]:
    direction = transform @ np.array([u, v], dtype=np.float64)
    return float(direction[0]), float(direction[1])


def inverse_transform_direction(transform: np.ndarray, u: float, v: float) -> tuple[float, float]:
    direction = transform.T @ np.array([u, v], dtype=np.float64)
    return float(direction[0]), float(direction[1])


def inverse_transform_boundary_unit_vector(
    transform: np.ndarray,
    unit_x: float,
    unit_y: float,
) -> tuple[float, float]:
    return inverse_transform_direction(transform, unit_x, unit_y)


def forward_transform_angular_moments(transform: np.ndarray, moments: Iterable[float]) -> np.ndarray:
    moment_arr = np.asarray(list(moments), dtype=np.float64)
    if moment_arr.shape != (5,):
        raise ValueError(f"Angular moments must have shape (5,), got {moment_arr.shape}")
    first = transform @ moment_arr[:2]
    second_phys = np.array(
        [
            [moment_arr[2], moment_arr[3]],
            [moment_arr[3], moment_arr[4]],
        ],
        dtype=np.float64,
    )
    second_can = transform @ second_phys @ transform.T
    return np.array(
        [first[0], first[1], second_can[0, 0], second_can[0, 1], second_can[1, 1]],
        dtype=np.float64,
    )


def inverse_transform_angular_moments(transform: np.ndarray, moments: Iterable[float]) -> np.ndarray:
    moment_arr = np.asarray(list(moments), dtype=np.float64)
    if moment_arr.shape != (5,):
        raise ValueError(f"Angular moments must have shape (5,), got {moment_arr.shape}")
    first = transform.T @ moment_arr[:2]
    second_can = np.array(
        [
            [moment_arr[2], moment_arr[3]],
            [moment_arr[3], moment_arr[4]],
        ],
        dtype=np.float64,
    )
    second_phys = transform.T @ second_can @ transform
    return np.array(
        [first[0], first[1], second_phys[0, 0], second_phys[0, 1], second_phys[1, 1]],
        dtype=np.float64,
    )


def forward_transform_legendre_coeffs(transform: np.ndarray, coeffs: Iterable[float]) -> np.ndarray:
    coeff_arr = np.asarray(list(coeffs), dtype=np.float64)
    if coeff_arr.shape != (len(LEGENDRE_BASIS_ORDER),):
        raise ValueError(
            f"Legendre coefficient array must have shape ({len(LEGENDRE_BASIS_ORDER)},), "
            f"got {coeff_arr.shape}"
        )
    axis_for_x = int(np.argmax(np.abs(transform[0])))
    axis_for_y = int(np.argmax(np.abs(transform[1])))
    sign_x = float(transform[0, axis_for_x])
    sign_y = float(transform[1, axis_for_y])
    out = np.zeros_like(coeff_arr)
    for idx, (deg_x, deg_y) in enumerate(LEGENDRE_BASIS_ORDER):
        if axis_for_x == 0 and axis_for_y == 1:
            phys_basis = (deg_x, deg_y)
        elif axis_for_x == 1 and axis_for_y == 0:
            phys_basis = (deg_y, deg_x)
        else:
            raise ValueError(f"Unsupported symmetry transform:\n{transform}")
        parity = (sign_x if (deg_x % 2 == 1) else 1.0) * (sign_y if (deg_y % 2 == 1) else 1.0)
        out[idx] = parity * coeff_arr[_LEGENDRE_INDEX[phys_basis]]
    return out


def inverse_transform_legendre_coeffs(transform: np.ndarray, coeffs: Iterable[float]) -> np.ndarray:
    coeff_arr = np.asarray(list(coeffs), dtype=np.float64)
    if coeff_arr.shape != (len(LEGENDRE_BASIS_ORDER),):
        raise ValueError(
            f"Legendre coefficient array must have shape ({len(LEGENDRE_BASIS_ORDER)},), "
            f"got {coeff_arr.shape}"
        )
    axis_for_x = int(np.argmax(np.abs(transform[0])))
    axis_for_y = int(np.argmax(np.abs(transform[1])))
    sign_x = float(transform[0, axis_for_x])
    sign_y = float(transform[1, axis_for_y])
    out = np.zeros_like(coeff_arr)
    for idx, (deg_x, deg_y) in enumerate(LEGENDRE_BASIS_ORDER):
        if axis_for_x == 0 and axis_for_y == 1:
            phys_basis = (deg_x, deg_y)
        elif axis_for_x == 1 and axis_for_y == 0:
            phys_basis = (deg_y, deg_x)
        else:
            raise ValueError(f"Unsupported symmetry transform:\n{transform}")
        parity = (sign_x if (deg_x % 2 == 1) else 1.0) * (sign_y if (deg_y % 2 == 1) else 1.0)
        out[_LEGENDRE_INDEX[phys_basis]] += parity * coeff_arr[idx]
    return out
