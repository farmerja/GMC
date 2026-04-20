"""mc_data_gen.py - DataFrame-returning wrappers for Monte Carlo simulations.

This file is intentionally a "user-facing" API layer: it calls Numba kernels and
returns pandas DataFrames with consistent column names and helpful derived features.

Key refactors / fixes vs prior version:
  * Time-dependent (path-budget) generators are consolidated:
      - One function per dimension handles internal vs boundary via a `source` argument.
  * 2D time-dependent boundary-source angular sampling is configurable in mc2dtime.py
    for physical (cosine-weighted) vs ML-coverage (uniform-u) generation.
  * Homogeneous vs heterogeneous is not duplicated: pass constant per-cell arrays for
    homogeneous media.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import qmc

from .mc1d import simulate_cell_with_random_thickness, simulate_internal_source_cell
from .mc1dtime import simulate_cell_1d_time_dependent, SRC_INTERNAL as SRC1D_INTERNAL, SRC_BOUNDARY_X0 as SRC1D_BOUNDARY
from .hierarchical_steady_2d import (
    FACE_BOTTOM as HIER2D_FACE_BOTTOM,
    FACE_LEFT as HIER2D_FACE_LEFT,
    FACE_NAMES as HIER2D_FACE_NAMES,
    FACE_RIGHT as HIER2D_FACE_RIGHT,
    FACE_TOP as HIER2D_FACE_TOP,
    SRC_BOUNDARY_X0 as HIER2D_BOUNDARY,
    SRC_INTERNAL as HIER2D_INTERNAL,
    simulate_hierarchical_steady_2d_gate,
    simulate_hierarchical_steady_2d_scattered_only,
)
from .mc2d import simulate_single_cell_2d_boundary_source, simulate_single_cell_2d_internal_source
from .mc2dtimetriangle import (
    simulate_single_cell_2d_triangle_timedep_internal,
    simulate_single_cell_2d_triangle_timedep_boundary,
)
from .mc2dtime import simulate_single_cell_2d_time_dependent, SRC_INTERNAL as SRC2D_INTERNAL, SRC_BOUNDARY_X0 as SRC2D_BOUNDARY
from .mc3d import simulate_single_cell_3d_boundary_source, simulate_single_cell_3d_internal_source
from .mc3dtime import simulate_single_cell_3d_time_dependent, SRC_INTERNAL as SRC3D_INTERNAL, SRC_BOUNDARY_X0 as SRC3D_BOUNDARY
from .steady_2d_scattering import steady_2d_scattering_mode_code


def _generate_cond_samples(num_neutrons: int, num_dims: int, sampling: str = 'random'):
    """Generate conditioning variable samples in [0, 1]^d.

    Returns:
      - None for 'random' (kernels use their own RNG), OR
      - (N,d) array for 'qmc' (Sobol) or 'lhs' (Latin hypercube).

    When QMC/LHS is enabled, these unit samples are also used for the initial
    position and initial direction variables, not just geometry / budget terms.
    """
    if sampling == 'random':
        return None
    if sampling == 'qmc':
        sampler = qmc.Sobol(d=num_dims, scramble=True)
        m = int(np.ceil(np.log2(max(num_neutrons, 1))))
        n_sobol = 2 ** m
        samples = sampler.random(n_sobol)
        return np.ascontiguousarray(samples[:num_neutrons])
    if sampling == 'lhs':
        sampler = qmc.LatinHypercube(d=num_dims)
        return np.ascontiguousarray(sampler.random(n=num_neutrons))
    raise ValueError(f"Unknown sampling method: {sampling!r}. Use 'random', 'qmc', or 'lhs'.")


def _generate_valid_quad_precond_samples(num_neutrons: int, sampling: str):
    """Generate QMC/LHS samples for quadrilateral cases, keeping only valid shapes."""
    if sampling == 'random':
        return None

    # Columns:
    #   0: L scale
    #   1-4: x3, y3, x4, y4 shape controls
    #   5: path budget
    #   6: triangle selector for area-weighted physical start
    #   7-8: barycentric coordinates in the selected triangle
    #   9-10: direction controls
    num_dims = 11
    accepted = []
    remaining = int(num_neutrons)

    while remaining > 0:
        pool_size = max(remaining * 4, 64)
        pool = _generate_cond_samples(pool_size, num_dims, sampling)

        x3 = pool[:, 1] * 4.0 - 2.0
        y3 = pool[:, 2] * 1.9 + 0.1
        x4 = pool[:, 3] * 4.0 - 2.0
        y4 = pool[:, 4] * 1.9 + 0.1
        valid = ((x3 - 1.0) * (y4 - y3) - y3 * (x4 - x3) > 1e-4) & ((x3 * y4 - x4 * y3) > 1e-4)

        if np.any(valid):
            batch = pool[valid][:remaining]
            accepted.append(batch)
            remaining -= batch.shape[0]

    return np.ascontiguousarray(np.vstack(accepted)[:num_neutrons])


def _print_budget_exit_stats(df: pd.DataFrame, num_neutrons: int) -> None:
    """Prints standardized exit statistics for time-dependent single-cell datasets."""
    n_space = int(df["space_exit"].sum())
    n_time = int(df["time_exit"].sum())
    n_abs = int(df["absorbed"].sum())
    n_exited = n_space + n_time
    print(f"  Space-like exits: {n_space} ({100*n_space/num_neutrons:.1f}%)")
    print(f"  Time-like exits: {n_time} ({100*n_time/num_neutrons:.1f}%)")
    print(f"  Absorbed: {n_abs} ({100*n_abs/num_neutrons:.1f}%)")
    print(f"  Total exited (space+time): {n_exited} ({100*n_exited/num_neutrons:.1f}%)")


def _append_face_local_features(
    df: pd.DataFrame,
    *,
    face_col: str,
    coord_col: str,
    prefix: str,
) -> None:
    eps = 1e-12
    face = df[face_col].to_numpy(dtype=np.int64, copy=False)
    face_length = np.where(
        (face == HIER2D_FACE_LEFT) | (face == HIER2D_FACE_RIGHT),
        df["height"].to_numpy(dtype=np.float64, copy=False),
        df["width"].to_numpy(dtype=np.float64, copy=False),
    )
    face_name_map = dict(enumerate(HIER2D_FACE_NAMES))
    df[f"{prefix}_face_name"] = df[face_col].astype(np.int64).map(face_name_map)
    df[f"{prefix}_face_length"] = face_length
    df[f"{prefix}_face_coord_local_norm"] = (df[coord_col] / (face_length + eps)).clip(0.0, 1.0)


def _append_exit_direction_compatibility(df: pd.DataFrame, *, face_col: str, u_col: str, v_col: str, out_col: str) -> None:
    face = df[face_col].to_numpy(dtype=np.int64, copy=False)
    u = df[u_col].to_numpy(dtype=np.float64, copy=False)
    v = df[v_col].to_numpy(dtype=np.float64, copy=False)
    compatible = (
        ((face == HIER2D_FACE_LEFT) & (u < 0.0))
        | ((face == HIER2D_FACE_RIGHT) & (u > 0.0))
        | ((face == HIER2D_FACE_TOP) & (v > 0.0))
        | ((face == HIER2D_FACE_BOTTOM) & (v < 0.0))
    )
    df[out_col] = compatible.astype(np.int64)


# ---------------------------------------------------------------------------
# 1D steady-state
# ---------------------------------------------------------------------------

def get_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 1D boundary source steady-state simulation data."""
    print(f"Generating 1D boundary source data ({num_neutrons} neutrons)...")
    cond_samples = _generate_cond_samples(num_neutrons, 2, sampling)
    start_time = time.time()
    results = simulate_cell_with_random_thickness(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        cond_samples=cond_samples,
    )
    end_time = time.time()
    print(f"1D Boundary data generation took {end_time - start_time:.2f} seconds.")
    df = pd.DataFrame(
        data=results,
        columns=[
            "initial_angle_x",
            "final_angle_x",
            "time_in_cell",
            "num_interactions",
            "exited_cell",
            "absorbed_in_cell",
            "thickness",
        ],
    )
    df["initial_angle_x"] = df["initial_angle_x"].abs()
    return df


def get_internal_source_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 1D internal source steady-state simulation data."""
    print(f"Generating 1D internal source data ({num_neutrons} neutrons)...")
    cond_samples = _generate_cond_samples(num_neutrons, 3, sampling)
    start_time = time.time()
    results = simulate_internal_source_cell(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        cond_samples=cond_samples,
    )
    end_time = time.time()
    print(f"1D Internal data generation took {end_time - start_time:.2f} seconds.")
    df = pd.DataFrame(
        data=results,
        columns=[
            "initial_angle_x",
            "final_angle_x",
            "time_in_cell",
            "num_interactions",
            "exited_cell",
            "absorbed_in_cell",
            "thickness",
            "initial_position_x",
        ],
    )
    return df


# ---------------------------------------------------------------------------
# 1D time-dependent (path-length budget)
# ---------------------------------------------------------------------------

def get_1d_time_dependent_data(
    *,
    source: Literal["internal", "boundary"] = "internal",
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_thickness: float = 0.01,
    max_thickness: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 1D time-dependent single-cell data (internal or boundary source)."""
    src_flag = SRC1D_INTERNAL if source == "internal" else SRC1D_BOUNDARY

    print(f"Generating 1D time-dependent BUDGET {source} source data ({num_neutrons} neutrons)...")
    print(f"  Thickness range: [{min_thickness}, {max_thickness}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")

    cond_dims = 4 if source == "internal" else 3
    cond_samples = _generate_cond_samples(num_neutrons, cond_dims, sampling)
    start_time = time.time()
    results = simulate_cell_1d_time_dependent(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
        source_type=src_flag,
    )
    end_time = time.time()
    print(f"Data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "thickness",
            "path_budget",
            "x_in_norm",
            "mu_in",
            "x_out_norm",
            "mu_out",
            "L",
            "num_interactions",
            "space_exit",
            "time_exit",
            "absorbed",
        ],
    )

    _print_budget_exit_stats(df, num_neutrons)
    return df


# Backwards-compatible names (thin wrappers)

def get_1d_time_dependent_internal_source_data(**kwargs) -> pd.DataFrame:
    return get_1d_time_dependent_data(source="internal", **kwargs)


def get_1d_time_dependent_boundary_source_data(**kwargs) -> pd.DataFrame:
    return get_1d_time_dependent_data(source="boundary", **kwargs)


# ---------------------------------------------------------------------------
# 2D steady-state
# ---------------------------------------------------------------------------

def get_2d_boundary_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    sampling: str = 'random',
    scattering_mode: str = "isotropic",
) -> pd.DataFrame:
    """Get 2D boundary source steady-state simulation data."""
    print(f"Generating 2D single-cell boundary source data ({num_neutrons} neutrons, scattering_mode={scattering_mode})...")
    cond_samples = _generate_cond_samples(num_neutrons, 5, sampling)
    start_time = time.time()
    results = simulate_single_cell_2d_boundary_source(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        cond_samples=cond_samples,
        scattering_mode_code=steady_2d_scattering_mode_code(scattering_mode),
    )
    end_time = time.time()
    print(f"2D Boundary data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "initial_pos_y",
            "initial_dir_u",
            "initial_dir_v",
            "final_pos_x",
            "final_pos_y",
            "final_dir_u",
            "final_dir_v",
            "time_in_cell",
            "exited",
            "absorbed",
        ],
    )
    eps = 1e-12
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    return df


def get_2d_internal_source_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    sampling: str = 'random',
    scattering_mode: str = "isotropic",
) -> pd.DataFrame:
    """Get 2D internal source steady-state simulation data."""
    print(f"Generating 2D single-cell internal source data ({num_neutrons} neutrons, scattering_mode={scattering_mode})...")
    cond_samples = _generate_cond_samples(num_neutrons, 6, sampling)
    start_time = time.time()
    results = simulate_single_cell_2d_internal_source(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        cond_samples=cond_samples,
        scattering_mode_code=steady_2d_scattering_mode_code(scattering_mode),
    )
    end_time = time.time()
    print(f"2D Internal data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "initial_pos_x",
            "initial_pos_y",
            "initial_dir_u",
            "initial_dir_v",
            "final_pos_x",
            "final_pos_y",
            "final_dir_u",
            "final_dir_v",
            "time_in_cell",
            "exited",
            "absorbed",
        ],
    )
    eps = 1e-12
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    return df


# ---------------------------------------------------------------------------
# 2D steady-state hierarchical add-on
# ---------------------------------------------------------------------------

def get_2d_hierarchical_steady_gate_data(
    *,
    source: Literal["internal", "boundary"] = "internal",
    num_neutrons: int = 500000,
    sigma_t: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Hierarchical steady-2D add-on: exact ballistic metadata and scatter gate."""
    if source not in {"internal", "boundary"}:
        raise ValueError(f"Unsupported source={source!r}. Use 'internal' or 'boundary'.")
    src_flag = HIER2D_INTERNAL if source == "internal" else HIER2D_BOUNDARY

    print(f"Generating 2D hierarchical steady gate data ({num_neutrons} neutrons, source={source})...")
    cond_dims = 6 if source == "internal" else 5
    cond_samples = _generate_cond_samples(num_neutrons, cond_dims, sampling)
    start_time = time.time()
    results = simulate_hierarchical_steady_2d_gate(
        num_neutrons=num_neutrons,
        sigma_t=sigma_t,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        cond_samples=cond_samples,
        source_type=src_flag,
    )
    end_time = time.time()
    print(f"2D Hierarchical steady gate data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "initial_pos_x",
            "initial_pos_y",
            "initial_dir_u",
            "initial_dir_v",
            "ballistic_length",
            "ballistic_face",
            "ballistic_exit_x",
            "ballistic_exit_y",
            "ballistic_face_coord_local",
            "scatter_before_exit_prob",
            "scatter_before_exit",
        ],
    )

    eps = 1e-12
    df["source_type"] = source
    df["sigma_t"] = float(sigma_t)
    df["ballistic_optical_length"] = df["sigma_t"] * df["ballistic_length"]
    df["ballistic_face"] = df["ballistic_face"].astype(np.int64)
    df["scatter_before_exit"] = df["scatter_before_exit"].astype(np.int64)
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["ballistic_exit_x_norm"] = (df["ballistic_exit_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["ballistic_exit_y_norm"] = (df["ballistic_exit_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    _append_face_local_features(
        df,
        face_col="ballistic_face",
        coord_col="ballistic_face_coord_local",
        prefix="ballistic",
    )
    return df


def get_2d_hierarchical_steady_scattered_only_data(
    *,
    source: Literal["internal", "boundary"] = "internal",
    num_neutrons: int = 500000,
    sigma_t: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Hierarchical steady-2D add-on: scattered-only data conditioned on first scatter."""
    if sigma_t <= 0.0:
        raise ValueError("sigma_t must be positive for scattered-only conditioned data.")
    if source not in {"internal", "boundary"}:
        raise ValueError(f"Unsupported source={source!r}. Use 'internal' or 'boundary'.")

    src_flag = HIER2D_INTERNAL if source == "internal" else HIER2D_BOUNDARY

    print(f"Generating 2D hierarchical steady scattered-only data ({num_neutrons} neutrons, source={source})...")
    cond_dims = 6 if source == "internal" else 5
    cond_samples = _generate_cond_samples(num_neutrons, cond_dims, sampling)
    start_time = time.time()
    results = simulate_hierarchical_steady_2d_scattered_only(
        num_neutrons=num_neutrons,
        sigma_t=sigma_t,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        cond_samples=cond_samples,
        source_type=src_flag,
    )
    end_time = time.time()
    print(f"2D Hierarchical steady scattered-only data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "initial_pos_x",
            "initial_pos_y",
            "initial_dir_u",
            "initial_dir_v",
            "ballistic_length",
            "ballistic_face",
            "ballistic_exit_x",
            "ballistic_exit_y",
            "ballistic_face_coord_local",
            "scatter_before_exit_prob",
            "first_scatter_distance",
            "first_scatter_pos_x",
            "first_scatter_pos_y",
            "first_scatter_out_dir_u",
            "first_scatter_out_dir_v",
            "num_scatters",
            "total_track_length",
            "final_pos_x",
            "final_pos_y",
            "exit_face",
            "exit_face_coord_local",
            "final_dir_u",
            "final_dir_v",
        ],
    )

    eps = 1e-12
    df["source_type"] = source
    df["sigma_t"] = float(sigma_t)
    df["ballistic_optical_length"] = df["sigma_t"] * df["ballistic_length"]
    df["ballistic_face"] = df["ballistic_face"].astype(np.int64)
    df["exit_face"] = df["exit_face"].astype(np.int64)
    df["num_scatters"] = df["num_scatters"].astype(np.int64)
    df["scatter_before_exit"] = 1
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["first_scatter_pos_x_norm"] = (df["first_scatter_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["first_scatter_pos_y_norm"] = (df["first_scatter_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["ballistic_exit_x_norm"] = (df["ballistic_exit_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["ballistic_exit_y_norm"] = (df["ballistic_exit_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["first_scatter_distance_frac_ballistic"] = (
        df["first_scatter_distance"] / (df["ballistic_length"] + eps)
    ).clip(0.0, 1.0)
    _append_face_local_features(
        df,
        face_col="ballistic_face",
        coord_col="ballistic_face_coord_local",
        prefix="ballistic",
    )
    _append_face_local_features(
        df,
        face_col="exit_face",
        coord_col="exit_face_coord_local",
        prefix="exit",
    )
    _append_exit_direction_compatibility(
        df,
        face_col="exit_face",
        u_col="final_dir_u",
        v_col="final_dir_v",
        out_col="final_dir_outward_compatible",
    )
    return df


# ---------------------------------------------------------------------------
# 2D time-dependent (path-length budget)
# ---------------------------------------------------------------------------

def get_2d_time_dependent_data(
    *,
    source: Literal["internal", "boundary"] = "internal",
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    angular_sampling: int = 1,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 2D time-dependent single-cell data (internal or boundary source)."""
    src_flag = SRC2D_INTERNAL if source == "internal" else SRC2D_BOUNDARY

    print(f"Generating 2D time-dependent BUDGET {source} source data ({num_neutrons} neutrons)...")
    print(f"  Width range: [{min_width}, {max_width}]")
    print(f"  Height range: [{min_height}, {max_height}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")

    cond_dims = 7 if source == "internal" else 6
    cond_samples = _generate_cond_samples(num_neutrons, cond_dims, sampling)
    start_time = time.time()
    results = simulate_single_cell_2d_time_dependent(
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
        source_type=src_flag,
        angular_sampling=angular_sampling,
    )
    end_time = time.time()
    print(f"2D Time-dependent {source} data generation took {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "path_budget",
            "initial_pos_x",
            "initial_pos_y",
            "initial_dir_u",
            "initial_dir_v",
            "final_pos_x",
            "final_pos_y",
            "final_dir_u",
            "final_dir_v",
            "L",
            "num_interactions",
            "space_exit",
            "time_exit",
            "absorbed",
        ],
    )

    eps = 1e-12
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)

    _print_budget_exit_stats(df, num_neutrons)
    return df


def get_2d_time_dependent_internal_source_data(**kwargs) -> pd.DataFrame:
    return get_2d_time_dependent_data(source="internal", **kwargs)


def get_2d_time_dependent_boundary_source_data(**kwargs) -> pd.DataFrame:
    return get_2d_time_dependent_data(source="boundary", **kwargs)


# ---------------------------------------------------------------------------
# 3D steady-state
# ---------------------------------------------------------------------------

def get_3d_boundary_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    angular_sampling: int = 1,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 3D boundary-source single-cell steady-state simulation data."""
    print(f"Generating 3D single-cell boundary source data ({num_neutrons} neutrons)...")
    cond_samples = _generate_cond_samples(num_neutrons, 7, sampling)
    start = time.time()
    results = simulate_single_cell_3d_boundary_source(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_depth=min_depth,
        max_depth=max_depth,
        angular_sampling=angular_sampling,
        cond_samples=cond_samples,
    )
    end = time.time()
    print(f"3D Boundary data generation took {end - start:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "depth",
            "initial_pos_y",
            "initial_pos_z",
            "initial_dir_u",
            "initial_dir_v",
            "initial_dir_w",
            "final_pos_x",
            "final_pos_y",
            "final_pos_z",
            "final_dir_u",
            "final_dir_v",
            "final_dir_w",
            "time_in_cell",
            "exited",
            "absorbed",
        ],
    )

    eps = 1e-12
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["initial_pos_z_norm"] = (df["initial_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_z_norm"] = (df["final_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)
    return df


def get_3d_internal_source_simulation_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 3D internal-source single-cell steady-state simulation data."""
    print(f"Generating 3D single-cell internal source data ({num_neutrons} neutrons)...")
    cond_samples = _generate_cond_samples(num_neutrons, 8, sampling)
    start = time.time()
    results = simulate_single_cell_3d_internal_source(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        distribution=distribution,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        min_depth=min_depth,
        max_depth=max_depth,
        cond_samples=cond_samples,
    )
    end = time.time()
    print(f"3D Internal data generation took {end - start:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "depth",
            "initial_pos_x",
            "initial_pos_y",
            "initial_pos_z",
            "initial_dir_u",
            "initial_dir_v",
            "initial_dir_w",
            "final_pos_x",
            "final_pos_y",
            "final_pos_z",
            "final_dir_u",
            "final_dir_v",
            "final_dir_w",
            "time_in_cell",
            "exited",
            "absorbed",
        ],
    )

    eps = 1e-12
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["initial_pos_z_norm"] = (df["initial_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_z_norm"] = (df["final_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)
    return df


# ---------------------------------------------------------------------------
# 3D time-dependent (path-length budget)
# ---------------------------------------------------------------------------

def get_3d_time_dependent_data(
    *,
    source: Literal["internal", "boundary"] = "internal",
    num_neutrons: int = 500000,
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
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 3D time-dependent single-cell data (internal or boundary source)."""
    src_flag = SRC3D_INTERNAL if source == "internal" else SRC3D_BOUNDARY

    print(f"Generating 3D time-dependent BUDGET {source} source data ({num_neutrons} neutrons)...")
    print(f"  Width/Height/Depth ranges: [{min_width},{max_width}] / [{min_height},{max_height}] / [{min_depth},{max_depth}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")

    cond_dims = 9 if source == "internal" else 8
    cond_samples = _generate_cond_samples(num_neutrons, cond_dims, sampling)
    start = time.time()
    results = simulate_single_cell_3d_time_dependent(
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
        source_type=src_flag,
        angular_sampling=angular_sampling,
    )
    end = time.time()
    print(f"3D Time-dependent {source} data generation took {end - start:.2f} seconds.")

    df = pd.DataFrame(
        data=results,
        columns=[
            "width",
            "height",
            "depth",
            "path_budget",
            "initial_pos_x",
            "initial_pos_y",
            "initial_pos_z",
            "initial_dir_u",
            "initial_dir_v",
            "initial_dir_w",
            "final_pos_x",
            "final_pos_y",
            "final_pos_z",
            "final_dir_u",
            "final_dir_v",
            "final_dir_w",
            "L",
            "num_interactions",
            "space_exit",
            "time_exit",
            "absorbed",
        ],
    )

    eps = 1e-12
    df["initial_pos_x_norm"] = (df["initial_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["initial_pos_y_norm"] = (df["initial_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["initial_pos_z_norm"] = (df["initial_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)
    df["final_pos_x_norm"] = (df["final_pos_x"] / (df["width"] + eps)).clip(0.0, 1.0)
    df["final_pos_y_norm"] = (df["final_pos_y"] / (df["height"] + eps)).clip(0.0, 1.0)
    df["final_pos_z_norm"] = (df["final_pos_z"] / (df["depth"] + eps)).clip(0.0, 1.0)

    _print_budget_exit_stats(df, num_neutrons)
    return df


def get_3d_time_dependent_internal_source_data(**kwargs) -> pd.DataFrame:
    return get_3d_time_dependent_data(source="internal", **kwargs)


def get_3d_time_dependent_boundary_source_data(**kwargs) -> pd.DataFrame:
    return get_3d_time_dependent_data(source="boundary", **kwargs)


def get_2d_triangle_timedep_internal_source_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_L: float = 0.001,
    max_L: float = 10.0,
    min_r: float = 0.001,
    max_r: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    min_width: float | None = None,
    max_width: float | None = None,
    min_height: float | None = None,
    max_height: float | None = None,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 2D triangle time-dependent internal source simulation data."""
    _ = distribution

    if min_width is not None:
        min_L = min_width
    if max_width is not None:
        max_L = max_width
    if min_height is not None:
        min_r = min_height
    if max_height is not None:
        max_r = max_height

    print(f"Generating 2D triangle time-dependent internal source data ({num_neutrons} neutrons)...")
    print(f"  L range: [{min_L}, {max_L}]")
    print(f"  r range: [{min_r}, {max_r}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")
    cond_samples = _generate_cond_samples(num_neutrons, 8, sampling)
    start = time.time()
    data = simulate_single_cell_2d_triangle_timedep_internal(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        min_L=min_L,
        max_L=max_L,
        min_r=min_r,
        max_r=max_r,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
    )
    end = time.time()
    print(f"2D Triangle time-dependent internal data generation took {end - start:.2f} seconds.")

    cols = [
        "L_tri",
        "r_tri",
        "theta",
        "path_budget",
        "xi1_in",
        "xi2_in",
        "u_in",
        "v_in",
        "xi1_out",
        "xi2_out",
        "u_out",
        "v_out",
        "L",
        "space_exit",
        "time_exit",
        "absorbed",
    ]
    df = pd.DataFrame(data, columns=cols)
    _print_budget_exit_stats(df, num_neutrons)
    return df


def get_2d_triangle_timedep_boundary_source_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_L: float = 0.001,
    max_L: float = 10.0,
    min_r: float = 0.001,
    max_r: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    min_width: float | None = None,
    max_width: float | None = None,
    min_height: float | None = None,
    max_height: float | None = None,
    angular_sampling: int = 1,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 2D triangle time-dependent boundary source simulation data."""
    _ = distribution

    if min_width is not None:
        min_L = min_width
    if max_width is not None:
        max_L = max_width
    if min_height is not None:
        min_r = min_height
    if max_height is not None:
        max_r = max_height

    print(f"Generating 2D triangle time-dependent boundary source data ({num_neutrons} neutrons)...")
    print(f"  L range: [{min_L}, {max_L}]")
    print(f"  r range: [{min_r}, {max_r}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")
    print(f"  Angular sampling: {angular_sampling} (0=uniform-normal, 1=cosine-law)")
    cond_samples = _generate_cond_samples(num_neutrons, 7, sampling)
    start = time.time()
    data = simulate_single_cell_2d_triangle_timedep_boundary(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        min_L=min_L,
        max_L=max_L,
        min_r=min_r,
        max_r=max_r,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
        angular_sampling=angular_sampling,
    )
    end = time.time()
    print(f"2D Triangle time-dependent boundary data generation took {end - start:.2f} seconds.")

    cols = [
        "L_tri",
        "r_tri",
        "theta",
        "path_budget",
        "xi1_in",
        "xi2_in",
        "u_in",
        "v_in",
        "xi1_out",
        "xi2_out",
        "u_out",
        "v_out",
        "L",
        "space_exit",
        "time_exit",
        "absorbed",
    ]
    df = pd.DataFrame(data, columns=cols)
    _print_budget_exit_stats(df, num_neutrons)
    return df


# ---------------------------------------------------------------------------
# 2D quadrilateral time-dependent (internal)
# Triangle transport solvers live in physics.mc2dtimetriangle, while the
# DataFrame-returning wrapper stays here for API consistency across problems.
# ---------------------------------------------------------------------------

@njit
def compute_homography(L, x3, y3, x4, y4):
    """Compute homography H and inverse G for unit-square to quadrilateral mapping."""
    dx1, dx2 = L - x3, x4 - x3
    dy1, dy2 = -y3, y4 - y3
    sx, sy = x3 - L - x4, y3 - y4

    D = dx1 * dy2 - dy1 * dx2
    if abs(D) < 1e-12:
        return np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)

    g = (sx * dy2 - sy * dx2) / D
    h = (dx1 * sy - dy1 * sx) / D

    H = np.zeros((3, 3), dtype=np.float64)
    H[0, 0] = L * (1.0 + g)
    H[0, 1] = x4 * (1.0 + h)
    H[1, 1] = y4 * (1.0 + h)
    H[2, 0] = g
    H[2, 1] = h
    H[2, 2] = 1.0

    G = np.linalg.inv(H)
    return H, G


@njit
def simulate_single_cell_2d_quad_timedep_internal(
    num_neutrons: int,
    scatter_ratio: float = 1.0,
    min_width: float = 0.001,
    max_width: float = 10.0,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    cond_samples=None,
):
    """2D single-cell time-dependent internal source in a convex quadrilateral."""
    data = np.zeros((num_neutrons, 18), dtype=np.float64)
    sigma_a_prob, sigma_t, eps = 1.0 - scatter_ratio, 1.0, 1e-9

    log_min_L, log_max_L = np.log10(min_width), np.log10(max_width)
    log_min_budget, log_max_budget = np.log10(min_path_budget), np.log10(max_path_budget)

    use_precond = cond_samples is not None

    for i in range(num_neutrons):
        cand_L = cond_samples[i, 0] if use_precond else np.random.rand()
        L_quad = 10.0 ** (log_min_L + cand_L * (log_max_L - log_min_L))
        if use_precond:
            x3 = L_quad * (cond_samples[i, 1] * 4.0 - 2.0)
            y3 = L_quad * (cond_samples[i, 2] * 1.9 + 0.1)
            x4 = L_quad * (cond_samples[i, 3] * 4.0 - 2.0)
            y4 = L_quad * (cond_samples[i, 4] * 1.9 + 0.1)
        else:
            convex = False
            while not convex:
                x3 = L_quad * np.random.uniform(-2.0, 2.0)
                y3 = L_quad * np.random.uniform(0.1, 2.0)
                x4 = L_quad * np.random.uniform(-2.0, 2.0)
                y4 = L_quad * np.random.uniform(0.1, 2.0)

                c1 = (x3 - L_quad) * (y4 - y3) - y3 * (x4 - x3) > 1e-4 * L_quad**2
                c2 = x3 * y4 - x4 * y3 > 1e-4 * L_quad**2
                if c1 and c2:
                    convex = True

        cand_b = cond_samples[i, 5] if use_precond else np.random.rand()
        path_budget = 10.0 ** (log_min_budget + cand_b * (log_max_budget - log_min_budget))
        _, G = compute_homography(L_quad, x3, y3, x4, y4)

        # Area-weighted start sampling for uniform physical position.
        A1 = 0.5 * L_quad * y3
        A2 = 0.5 * (x3 * y4 - x4 * y3)
        tri_selector = cond_samples[i, 6] if use_precond else np.random.rand()
        if tri_selector < A1 / (A1 + A2):
            if use_precond:
                r1, r2 = cond_samples[i, 7], cond_samples[i, 8]
            else:
                r1, r2 = np.random.rand(), np.random.rand()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            x_in = r1 * L_quad + r2 * x3
            y_in = r2 * y3
        else:
            if use_precond:
                r1, r2 = cond_samples[i, 7], cond_samples[i, 8]
            else:
                r1, r2 = np.random.rand(), np.random.rand()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            x_in = r1 * x3 + r2 * x4
            y_in = r1 * y3 + r2 * y4

        p0 = G[0, 0] * x_in + G[0, 1] * y_in + G[0, 2]
        p1 = G[1, 0] * x_in + G[1, 1] * y_in + G[1, 2]
        p2 = G[2, 0] * x_in + G[2, 1] * y_in + G[2, 2]

        p2_safe = max(p2, eps)
        u_in = max(0.0, min(p0 / p2_safe, 1.0))
        v_in = max(0.0, min(p1 / p2_safe, 1.0))

        # Isotropic 3D direction projected to 2D.
        if use_precond:
            mu_z = 2.0 * cond_samples[i, 9] - 1.0
            phi = 2.0 * np.pi * cond_samples[i, 10]
        else:
            mu_z = np.random.uniform(-1.0, 1.0)
            phi = np.random.uniform(0.0, 2.0 * np.pi)
        t_val = np.sqrt(1.0 - mu_z * mu_z)
        dir_u_in, dir_v_in = t_val * np.cos(phi), t_val * np.sin(phi)

        data[i, 0:10] = (L_quad, x3, y3, x4, y4, path_budget, u_in, v_in, dir_u_in, dir_v_in)

        L_track = 0.0
        space_exit, time_exit, absorbed = 0, 0, 0
        alive = True
        dir_u, dir_v = dir_u_in, dir_v_in

        while alive:
            remaining_budget = path_budget - L_track
            if remaining_budget <= 0.0:
                time_exit, alive = 1, False
                break

            free_path = -np.log(np.random.rand()) / sigma_t

            q0 = G[0, 0] * dir_u + G[0, 1] * dir_v
            q1 = G[1, 0] * dir_u + G[1, 1] * dir_v
            q2 = G[2, 0] * dir_u + G[2, 1] * dir_v

            s1 = -p0 / q0 if q0 < -eps else np.inf
            s2 = (p2 - p0) / (q0 - q2) if (q0 - q2) > eps else np.inf
            s3 = -p1 / q1 if q1 < -eps else np.inf
            s4 = (p2 - p1) / (q1 - q2) if (q1 - q2) > eps else np.inf

            s_exit = min(s1, s2, s3, s4)
            step_dist = max(0.0, min(free_path, s_exit, remaining_budget))

            p0 = p0 + step_dist * q0
            p1 = p1 + step_dist * q1
            p2 = p2 + step_dist * q2
            L_track += step_dist

            boundary_hit = abs(step_dist - s_exit) < eps * s_exit + eps
            budget_hit = abs(step_dist - remaining_budget) < eps * remaining_budget + eps

            if boundary_hit:
                if abs(step_dist - s1) < eps * s1 + eps:
                    p0 = 0.0
                if abs(step_dist - s2) < eps * s2 + eps:
                    p0 = p2
                if abs(step_dist - s3) < eps * s3 + eps:
                    p1 = 0.0
                if abs(step_dist - s4) < eps * s4 + eps:
                    p1 = p2
                space_exit, alive = 1, False
            elif budget_hit:
                time_exit, alive = 1, False
            else:
                if np.random.random() < sigma_a_prob:
                    absorbed, alive = 1, False
                else:
                    mu_z = np.random.uniform(-1.0, 1.0)
                    phi = np.random.uniform(0.0, 2.0 * np.pi)
                    t_val = np.sqrt(1.0 - mu_z * mu_z)
                    dir_u, dir_v = t_val * np.cos(phi), t_val * np.sin(phi)

        u_out = min(max(p0 / max(p2, eps), 0.0), 1.0)
        v_out = min(max(p1 / max(p2, eps), 0.0), 1.0)
        data[i, 10:18] = (
            u_out,
            v_out,
            dir_u,
            dir_v,
            L_track,
            float(space_exit),
            float(time_exit),
            float(absorbed),
        )

    return data


def get_2d_quad_timedep_internal_source_data(
    num_neutrons: int = 500000,
    scatter_ratio: float = 1.0,
    distribution: int = 0,
    min_width: float = 0.001,
    max_width: float = 10.0,
    min_height: float | None = None,
    max_height: float | None = None,
    min_path_budget: float = 1e-5,
    max_path_budget: float = 0.1,
    sampling: str = 'random',
) -> pd.DataFrame:
    """Get 2D quadrilateral time-dependent internal source simulation data."""
    _ = distribution
    _ = min_height
    _ = max_height

    print(f"Generating 2D quadrilateral time-dependent internal source data ({num_neutrons} neutrons)...")
    print(f"  Width range: [{min_width}, {max_width}]")
    print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")
    cond_samples = _generate_valid_quad_precond_samples(num_neutrons, sampling)
    start = time.time()
    data = simulate_single_cell_2d_quad_timedep_internal(
        num_neutrons=num_neutrons,
        scatter_ratio=scatter_ratio,
        min_width=min_width,
        max_width=max_width,
        min_path_budget=min_path_budget,
        max_path_budget=max_path_budget,
        cond_samples=cond_samples,
    )
    end = time.time()
    print(f"2D Quadrilateral time-dependent internal data generation took {end - start:.2f} seconds.")

    cols = [
        "L_quad",
        "x3",
        "y3",
        "x4",
        "y4",
        "path_budget",
        "u_in",
        "v_in",
        "dir_u_in",
        "dir_v_in",
        "u_out",
        "v_out",
        "dir_u_out",
        "dir_v_out",
        "L_track",
        "space_exit",
        "time_exit",
        "absorbed",
    ]
    df = pd.DataFrame(data, columns=cols)
    _print_budget_exit_stats(df, num_neutrons)
    return df
