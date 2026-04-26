from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import qmc


EPSILON = 1.0e-12
FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3
FACE_TO_FACE = np.array([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=np.int64)


@dataclass(frozen=True)
class BallisticExitBatch2D:
    distance: np.ndarray
    exit_face: np.ndarray
    x_exit: np.ndarray
    y_exit: np.ndarray
    pos_on_face: np.ndarray
    scatter_probability: np.ndarray


def scatter_before_exit_probability(sigma_s: float | np.ndarray, distance: float | np.ndarray) -> np.ndarray:
    sigma_s_arr = np.maximum(np.asarray(sigma_s, dtype=np.float64), 0.0)
    distance_arr = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    return 1.0 - np.exp(-sigma_s_arr * distance_arr)


def exit_weight_after_absorption(weight: np.ndarray, sigma_a: np.ndarray, distance: np.ndarray) -> np.ndarray:
    weight_arr = np.asarray(weight, dtype=np.float64)
    sigma_a_arr = np.maximum(np.asarray(sigma_a, dtype=np.float64), 0.0)
    distance_arr = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    return weight_arr * np.exp(-sigma_a_arr * distance_arr)


def weighted_track_integral(weight: np.ndarray, sigma_a: np.ndarray, distance: np.ndarray) -> np.ndarray:
    weight_arr = np.asarray(weight, dtype=np.float64)
    sigma_a_arr = np.maximum(np.asarray(sigma_a, dtype=np.float64), 0.0)
    distance_arr = np.maximum(np.asarray(distance, dtype=np.float64), 0.0)
    optical_depth = sigma_a_arr * distance_arr
    return np.where(
        optical_depth > 1.0e-2,
        weight_arr * (1.0 - np.exp(-optical_depth)) / np.maximum(sigma_a_arr, EPSILON),
        weight_arr * distance_arr,
    )


def classify_face_and_snap(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    tol: float = 1.0e-10,
) -> tuple[int, float, float, float]:
    distances = np.array(
        [abs(x), abs(width - x), abs(height - y), abs(y)],
        dtype=np.float64,
    )
    face = int(np.argmin(distances))
    x_snap = float(np.clip(x, 0.0, width))
    y_snap = float(np.clip(y, 0.0, height))
    if face == FACE_LEFT or abs(x_snap) <= tol:
        face = FACE_LEFT
        x_snap = 0.0
    elif face == FACE_RIGHT or abs(width - x_snap) <= tol:
        face = FACE_RIGHT
        x_snap = width
    elif face == FACE_TOP or abs(height - y_snap) <= tol:
        face = FACE_TOP
        y_snap = height
    else:
        face = FACE_BOTTOM
        y_snap = 0.0
    pos_on_face = y_snap if face in (FACE_LEFT, FACE_RIGHT) else x_snap
    return face, x_snap, y_snap, pos_on_face


def ballistic_exit_2d(
    x0: float,
    y0: float,
    u: float,
    v: float,
    width: float,
    height: float,
) -> tuple[float, int, float, float, float]:
    if abs(u) < EPSILON and abs(v) < EPSILON:
        raise ValueError("Direction vector must be non-zero.")

    dist_x_right = (width - x0) / u if u > EPSILON else np.inf
    dist_x_left = (0.0 - x0) / u if u < -EPSILON else np.inf
    dist_y_top = (height - y0) / v if v > EPSILON else np.inf
    dist_y_bottom = (0.0 - y0) / v if v < -EPSILON else np.inf
    distance = float(max(0.0, min(dist_x_right, dist_x_left, dist_y_top, dist_y_bottom)))
    x_exit = float(x0 + distance * u)
    y_exit = float(y0 + distance * v)
    face, x_snap, y_snap, pos_on_face = classify_face_and_snap(x_exit, y_exit, width, height)
    return distance, face, x_snap, y_snap, pos_on_face


def ballistic_exit_batch_2d(
    *,
    x0: np.ndarray,
    y0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    sigma_s: np.ndarray | float,
) -> BallisticExitBatch2D:
    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    y0_arr = np.asarray(y0, dtype=np.float64).reshape(-1)
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
    height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
    sigma_s_arr = np.broadcast_to(np.asarray(sigma_s, dtype=np.float64), x0_arr.shape)

    num_rows = x0_arr.shape[0]
    distance = np.zeros(num_rows, dtype=np.float64)
    exit_face = np.zeros(num_rows, dtype=np.int64)
    x_exit = np.zeros(num_rows, dtype=np.float64)
    y_exit = np.zeros(num_rows, dtype=np.float64)
    pos_on_face = np.zeros(num_rows, dtype=np.float64)
    for idx in range(num_rows):
        dist, face, x_out, y_out, face_pos = ballistic_exit_2d(
            float(x0_arr[idx]),
            float(y0_arr[idx]),
            float(u_arr[idx]),
            float(v_arr[idx]),
            float(width_arr[idx]),
            float(height_arr[idx]),
        )
        distance[idx] = dist
        exit_face[idx] = face
        x_exit[idx] = x_out
        y_exit[idx] = y_out
        pos_on_face[idx] = face_pos
    return BallisticExitBatch2D(
        distance=distance,
        exit_face=exit_face,
        x_exit=x_exit,
        y_exit=y_exit,
        pos_on_face=pos_on_face,
        scatter_probability=scatter_before_exit_probability(sigma_s_arr, distance),
    )


def sample_scatter_mask(
    sigma_s: np.ndarray,
    ballistic_distance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    scatter_probability = scatter_before_exit_probability(sigma_s, ballistic_distance)
    return rng.random(scatter_probability.shape[0]) < scatter_probability


def sample_first_scatter_distance_conditional(
    sigma_s: float,
    ballistic_distance: float,
    rng: np.random.Generator,
) -> float:
    sigma_s = float(max(sigma_s, EPSILON))
    ballistic_distance = float(max(ballistic_distance, 0.0))
    if ballistic_distance <= 0.0:
        return 0.0
    cdf_max = 1.0 - math.exp(-sigma_s * ballistic_distance)
    u = float(rng.random())
    return -math.log(max(1.0 - u * cdf_max, EPSILON)) / sigma_s


def sample_internal_direction_uv(rng: np.random.Generator) -> tuple[float, float]:
    mu_z = 2.0 * float(rng.random()) - 1.0
    phi = 2.0 * math.pi * float(rng.random())
    transverse = math.sqrt(max(0.0, 1.0 - mu_z * mu_z))
    u = transverse * math.cos(phi)
    v = transverse * math.sin(phi)
    if abs(u) < EPSILON and abs(v) < EPSILON:
        return EPSILON, 0.0
    return u, v


def sample_boundary_direction_uv(rng: np.random.Generator) -> tuple[float, float]:
    u = max(float(rng.random()), EPSILON)
    phi = 2.0 * math.pi * float(rng.random())
    transverse = math.sqrt(max(0.0, 1.0 - u * u))
    return u, transverse * math.cos(phi)


def _generate_condition_samples(num_samples: int, num_dims: int, sampling: str) -> np.ndarray | None:
    if sampling == "random":
        return None
    if sampling == "qmc":
        sampler = qmc.Sobol(d=num_dims, scramble=True)
        power = int(np.ceil(np.log2(max(num_samples, 1))))
        return np.ascontiguousarray(sampler.random_base2(m=power)[:num_samples])
    if sampling == "lhs":
        sampler = qmc.LatinHypercube(d=num_dims)
        return np.ascontiguousarray(sampler.random(n=num_samples))
    raise ValueError(f"Unsupported sampling method: {sampling!r}")


def _log_uniform(sample: float, minimum: float, maximum: float) -> float:
    lower = math.log10(minimum)
    upper = math.log10(maximum)
    return 10.0 ** (lower + sample * (upper - lower))


def _continue_scattering_until_exit(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    sigma_s: float,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, int]:
    total_path = 0.0
    num_scatters = 1
    u, v = sample_internal_direction_uv(rng)
    while True:
        free_path = -math.log(max(float(rng.random()), EPSILON)) / max(sigma_s, EPSILON)
        distance, _face, x_exit, y_exit, _pos = ballistic_exit_2d(x, y, u, v, width, height)
        step = min(free_path, distance)
        x = float(x + step * u)
        y = float(y + step * v)
        total_path += step
        if free_path >= distance - 1.0e-10:
            face, x_snap, y_snap, _ = classify_face_and_snap(x_exit, y_exit, width, height)
            return x_snap, y_snap, u, v, total_path, num_scatters
        num_scatters += 1
        u, v = sample_internal_direction_uv(rng)


def _scattered_only_row(
    *,
    source: Literal["boundary", "internal"],
    width: float,
    height: float,
    x0: float,
    y0: float,
    u0: float,
    v0: float,
    sigma_s: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    ballistic_distance, ballistic_face, ballistic_x, ballistic_y, ballistic_pos = ballistic_exit_2d(
        x0,
        y0,
        u0,
        v0,
        width,
        height,
    )
    first_scatter_distance = sample_first_scatter_distance_conditional(sigma_s, ballistic_distance, rng)
    x_scatter = float(x0 + first_scatter_distance * u0)
    y_scatter = float(y0 + first_scatter_distance * v0)
    x_out, y_out, u_out, v_out, extra_path, num_scatters = _continue_scattering_until_exit(
        x=x_scatter,
        y=y_scatter,
        width=width,
        height=height,
        sigma_s=sigma_s,
        rng=rng,
    )
    total_path = first_scatter_distance + extra_path
    row = {
        "width": width,
        "height": height,
        "initial_dir_u": u0,
        "initial_dir_v": v0,
        "final_pos_x": x_out,
        "final_pos_y": y_out,
        "final_dir_u": u_out,
        "final_dir_v": v_out,
        "time_in_cell": total_path,
        "exited": 1.0,
        "absorbed": 0.0,
        "num_scatters": float(num_scatters),
        "ballistic_length": ballistic_distance,
        "ballistic_exit_x": ballistic_x,
        "ballistic_exit_y": ballistic_y,
        "ballistic_face": float(ballistic_face),
        "ballistic_pos_on_face": ballistic_pos,
        "scatter_before_exit_prob": float(scatter_before_exit_probability(sigma_s, ballistic_distance)),
    }
    if source == "boundary":
        row["initial_pos_y"] = y0
    else:
        row["initial_pos_x"] = x0
        row["initial_pos_y"] = y0
    return row


def generate_scattered_only_2d_dataframe(
    *,
    source: Literal["boundary", "internal"],
    num_neutrons: int = 100000,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    sigma_s: float = 1.0,
    sampling: str = "random",
    seed: int | None = None,
) -> pd.DataFrame:
    if source not in {"boundary", "internal"}:
        raise ValueError(f"Unsupported source: {source!r}")
    if sigma_s <= 0.0:
        raise ValueError("sigma_s must be positive for scattered-only conditioning.")

    cond_dims = 5 if source == "boundary" else 6
    cond = _generate_condition_samples(num_neutrons, cond_dims, sampling)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for idx in range(num_neutrons):
        if cond is None:
            width = _log_uniform(float(rng.random()), min_width, max_width)
            height = _log_uniform(float(rng.random()), min_height, max_height)
            if source == "boundary":
                y0 = height * float(rng.random())
                u0, v0 = sample_boundary_direction_uv(rng)
                x0 = 0.0
            else:
                x0 = width * float(rng.random())
                y0 = height * float(rng.random())
                u0, v0 = sample_internal_direction_uv(rng)
        else:
            width = _log_uniform(float(cond[idx, 0]), min_width, max_width)
            height = _log_uniform(float(cond[idx, 1]), min_height, max_height)
            if source == "boundary":
                x0 = 0.0
                y0 = height * float(cond[idx, 2])
                local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1, dtype=np.uint64))
                u0 = max(float(cond[idx, 3]), EPSILON)
                phi = 2.0 * math.pi * float(cond[idx, 4])
                transverse = math.sqrt(max(0.0, 1.0 - u0 * u0))
                v0 = transverse * math.cos(phi)
                rng_for_row = local_rng
                rows.append(
                    _scattered_only_row(
                        source=source,
                        width=width,
                        height=height,
                        x0=x0,
                        y0=y0,
                        u0=u0,
                        v0=v0,
                        sigma_s=sigma_s,
                        rng=rng_for_row,
                    )
                )
                continue
            x0 = width * float(cond[idx, 2])
            y0 = height * float(cond[idx, 3])
            mu_z = 2.0 * float(cond[idx, 4]) - 1.0
            phi = 2.0 * math.pi * float(cond[idx, 5])
            transverse = math.sqrt(max(0.0, 1.0 - mu_z * mu_z))
            u0 = transverse * math.cos(phi)
            v0 = transverse * math.sin(phi)
        rows.append(
            _scattered_only_row(
                source=source,
                width=width,
                height=height,
                x0=x0,
                y0=y0,
                u0=u0,
                v0=v0,
                sigma_s=sigma_s,
                rng=rng,
            )
        )

    frame = pd.DataFrame(rows)
    frame["initial_pos_y_norm"] = (frame["initial_pos_y"] / np.maximum(frame["height"], EPSILON)).clip(0.0, 1.0)
    if source == "internal":
        frame["initial_pos_x_norm"] = (frame["initial_pos_x"] / np.maximum(frame["width"], EPSILON)).clip(0.0, 1.0)
    frame["final_pos_x_norm"] = (frame["final_pos_x"] / np.maximum(frame["width"], EPSILON)).clip(0.0, 1.0)
    frame["final_pos_y_norm"] = (frame["final_pos_y"] / np.maximum(frame["height"], EPSILON)).clip(0.0, 1.0)
    return frame


def get_2d_scatter_split_boundary_data(**kwargs) -> pd.DataFrame:
    return generate_scattered_only_2d_dataframe(source="boundary", **kwargs)


def get_2d_scatter_split_internal_data(**kwargs) -> pd.DataFrame:
    return generate_scattered_only_2d_dataframe(source="internal", **kwargs)
