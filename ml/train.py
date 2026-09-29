#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import sys
import logging
log = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import TrainingTracker, get_flow_matching_path, get_model
from ml.transform import (
    SimDataset,
    compute_normalization_params_1d,
    compute_normalization_params_1d_timedep,
    compute_normalization_params_2d,
    compute_normalization_params_2d_internal,
    compute_normalization_params_2d_quad_timedep,
    compute_normalization_params_2d_timedep,
    compute_normalization_params_2d_triangle_timedep,
    compute_normalization_params_3d,
    compute_normalization_params_3d_timedep,
    robust_log_stats,
    compute_normalization_params_2d_electron,
)
from ml.utils import (
    DATA_DIM_CHOICES,
    TIME_DEPENDENT_DATA_DIMS,
    TWO_D_TIME_DEPENDENT_DATA_DIMS,
    analytic_log_bounds as _analytic_log_bounds,
    available_cpu_count as _available_cpu_count,
    safe_get_sim_data,
    save_norm_params_artifacts as _save_norm_params_artifacts,
    setup_environment,
)
from physics.mc_data_gen import (
    get_simulation_data, get_internal_source_simulation_data,
    get_2d_boundary_simulation_data, get_2d_internal_source_simulation_data,
    get_2d_time_dependent_internal_source_data, get_2d_time_dependent_boundary_source_data,
    get_2d_triangle_timedep_internal_source_data,
    get_2d_triangle_timedep_boundary_source_data,
    get_2d_quad_timedep_internal_source_data,
    get_1d_time_dependent_internal_source_data, get_1d_time_dependent_boundary_source_data,
    get_3d_boundary_simulation_data,
    get_3d_internal_source_simulation_data,
    get_3d_time_dependent_internal_source_data,
    get_3d_time_dependent_boundary_source_data,
)
from physics.electron_data import load_electron_dataset


def prepare_data(device, data_dim='2d', source_type='boundary', batch_size=2048, val_ratio=0.1,
                 min_thick_or_width=0.001, max_thick_or_width=10.0,
                 min_height=0.001, max_height=10.0,
                 min_depth=0.001, max_depth=10.0,
                 num_neutrons=100000,
                 particle_velocity=1.0, max_time=10.0,
                 min_path_budget=0.00001, max_path_budget=0.1,
                 direction_encoding_2d='polar',
                 angular_sampling=0,
                 sampling='random',
                 train_workers=None, val_workers=None,electron_npz=None):
    print(f"Loading simulation data for {data_dim} {source_type} source...")
    print(f"  Number of neutrons: {num_neutrons}")
    print(f"  Sampling method: {sampling.upper()}")
    direction_encoding_2d = str(direction_encoding_2d).strip().lower()
    if direction_encoding_2d != 'polar':
        raise ValueError(
            f"Unsupported direction_encoding_2d='{direction_encoding_2d}'. "
            "Only 'polar' is supported for 2D problems."
        )
    if data_dim in TIME_DEPENDENT_DATA_DIMS:
        print(f"  Path budget range: [{min_path_budget}, {max_path_budget}]")
    if data_dim == '2d' or data_dim in TWO_D_TIME_DEPENDENT_DATA_DIMS:
        print(f"  2D direction encoding: {direction_encoding_2d}")

    os.makedirs("ml/data", exist_ok=True)
    sampling_suffix = f"_{sampling}" if sampling != 'random' else ""

    if data_dim == '1d':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_thickness=min_thick_or_width, max_thickness=max_thick_or_width,
                sampling=sampling
            )
            data_type_str = '1d_boundary'
            sim_data = sim_data[sim_data["exited_cell"] == 1].copy()
            print(f"Total samples after filtering for exit: {len(sim_data)}")
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _ = compute_normalization_params_1d(sim_data)
            thick_lower, thick_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "thickness")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'thickness_lower': thick_lower, 'thickness_upper': thick_upper
            }
            norm_params_path = f"ml/data/norm_params_1d_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "1D",
            )
        else:
            sim_data = safe_get_sim_data(
                get_internal_source_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_thickness=min_thick_or_width, max_thickness=max_thick_or_width,
                sampling=sampling
            )
            data_type_str = '1d_internal'
            sim_data = sim_data[sim_data["exited_cell"] == 1].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _ = compute_normalization_params_1d(sim_data)
            thick_lower, thick_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "thickness")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'thickness_lower': thick_lower, 'thickness_upper': thick_upper
            }
            norm_params_path = f"ml/data/norm_params_1d_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "1D internal",
            )

    elif data_dim == '2d':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_2d_boundary_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                sampling=sampling
            )
            data_type_str = '2d_boundary'
            sim_data = sim_data[sim_data["exited"] == 1].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _, _, _ = compute_normalization_params_2d(sim_data)
            width_lower, width_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'width_lower': width_lower, 'width_upper': width_upper,
                'height_lower': height_lower, 'height_upper': height_upper,
                'direction_encoding_2d': direction_encoding_2d,
            }
            norm_params_path = f"ml/data/norm_params_2d_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "2D boundary",
            )

        else: 
            sim_data = safe_get_sim_data(
                get_2d_internal_source_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                sampling=sampling
            )
            data_type_str = '2d_internal'
            sim_data = sim_data[sim_data["exited"] == 1].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _, _, _ = compute_normalization_params_2d_internal(sim_data)
            width_lower, width_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'width_lower': width_lower, 'width_upper': width_upper,
                'height_lower': height_lower, 'height_upper': height_upper,
                'direction_encoding_2d': direction_encoding_2d,
            }
            norm_params_path = f"ml/data/norm_params_2d_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "2D internal",
            )

    elif data_dim == '3d':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_3d_boundary_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_depth=min_depth, max_depth=max_depth,
                sampling=sampling
            )
            data_type_str = '3d_boundary'
            sim_data = sim_data[sim_data["exited"] == 1].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _, _, _, _, _ = compute_normalization_params_3d(sim_data)
            w_lower, w_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            h_lower, h_upper = _analytic_log_bounds(min_height, max_height, "height")
            d_lower, d_upper = _analytic_log_bounds(min_depth, max_depth, "depth")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'width_lower': w_lower, 'width_upper': w_upper,
                'height_lower': h_lower, 'height_upper': h_upper,
                'depth_lower': d_lower, 'depth_upper': d_upper,
            }
            norm_params_path = f"ml/data/norm_params_3d_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "3D boundary",
            )

        else:
            sim_data = safe_get_sim_data(
                get_3d_internal_source_simulation_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_depth=min_depth, max_depth=max_depth,
                sampling=sampling
            )
            data_type_str = '3d_internal'
            sim_data = sim_data[sim_data["exited"] == 1].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            time_lower, time_upper, _, _, _, _, _, _ = compute_normalization_params_3d(sim_data)
            w_lower, w_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            h_lower, h_upper = _analytic_log_bounds(min_height, max_height, "height")
            d_lower, d_upper = _analytic_log_bounds(min_depth, max_depth, "depth")
            time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
            norm_params = {
                'time_lower': time_lower, 'time_upper': time_upper,
                'time_center': time_center, 'time_scale': time_scale,
                'width_lower': w_lower, 'width_upper': w_upper,
                'height_lower': h_lower, 'height_upper': h_upper,
                'depth_lower': d_lower, 'depth_upper': d_upper,
            }
            norm_params_path = f"ml/data/norm_params_3d_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "3D internal",
            )

    elif data_dim == '3d_timedep':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_3d_time_dependent_boundary_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_depth=min_depth, max_depth=max_depth,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '3d_timedep_boundary'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            _, _, _, _, _, _, _, _, LL, LU = compute_normalization_params_3d_timedep(sim_data, 'boundary')
            wL, wU = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            hL, hU = _analytic_log_bounds(min_height, max_height, "height")
            dL, dU = _analytic_log_bounds(min_depth, max_depth, "depth")
            bL, bU = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'width_lower': wL, 'width_upper': wU,
                'height_lower': hL, 'height_upper': hU,
                'depth_lower': dL, 'depth_upper': dU,
                'budget_lower': bL, 'budget_upper': bU,
                'L_lower': LL, 'L_upper': LU,
                'L_center': L_center, 'L_scale': L_scale,
                'use_sincos_circular': True,
                'use_L_over_budget': True,
                'time_lower': LL, 'time_upper': LU, 
            }
            norm_params_path = f"ml/data/norm_params_3d_timedep_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "3D timedep boundary",
            )
        else:
            sim_data = safe_get_sim_data(
                get_3d_time_dependent_internal_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_depth=min_depth, max_depth=max_depth,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '3d_timedep_internal'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            _, _, _, _, _, _, _, _, LL, LU = compute_normalization_params_3d_timedep(sim_data, 'internal')
            wL, wU = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            hL, hU = _analytic_log_bounds(min_height, max_height, "height")
            dL, dU = _analytic_log_bounds(min_depth, max_depth, "depth")
            bL, bU = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'width_lower': wL, 'width_upper': wU,
                'height_lower': hL, 'height_upper': hU,
                'depth_lower': dL, 'depth_upper': dU,
                'budget_lower': bL, 'budget_upper': bU,
                'L_lower': LL, 'L_upper': LU,
                'L_center': L_center, 'L_scale': L_scale,
                'use_sincos_circular': True,
                'use_L_over_budget': True,
                'time_lower': LL, 'time_upper': LU,
            }
            norm_params_path = f"ml/data/norm_params_3d_timedep_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "3D timedep internal",
            )

    elif data_dim == '2d_timedep':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_2d_time_dependent_boundary_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                angular_sampling=angular_sampling,
                sampling=sampling
            )
            data_type_str = '2d_timedep_boundary'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            _, _, _, _, _, _, L_lower, L_upper = compute_normalization_params_2d_timedep(sim_data, 'boundary')
            width_lower, width_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
            budget_lower, budget_upper = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'width_lower': width_lower, 'width_upper': width_upper,
                'height_lower': height_lower, 'height_upper': height_upper,
                'budget_lower': budget_lower, 'budget_upper': budget_upper,
                'L_lower': L_lower, 'L_upper': L_upper,
                'L_center': L_center, 'L_scale': L_scale,
                'direction_encoding_2d': direction_encoding_2d,
                'use_L_over_budget': True,
                'time_lower': L_lower, 'time_upper': L_upper 
            }
            norm_params_path = f"ml/data/norm_params_2d_timedep_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "2D timedep boundary",
            )
        else:
            sim_data = safe_get_sim_data(
                get_2d_time_dependent_internal_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_width=min_thick_or_width, max_width=max_thick_or_width,
                min_height=min_height, max_height=max_height,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '2d_timedep_internal'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

            _, _, _, _, _, _, L_lower, L_upper = compute_normalization_params_2d_timedep(sim_data, 'internal')
            width_lower, width_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "width")
            height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
            budget_lower, budget_upper = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'width_lower': width_lower, 'width_upper': width_upper,
                'height_lower': height_lower, 'height_upper': height_upper,
                'budget_lower': budget_lower, 'budget_upper': budget_upper,
                'L_lower': L_lower, 'L_upper': L_upper,
                'L_center': L_center, 'L_scale': L_scale,
                'direction_encoding_2d': direction_encoding_2d,
                'use_L_over_budget': True,
                'time_lower': L_lower, 'time_upper': L_upper 
            }
            norm_params_path = f"ml/data/norm_params_2d_timedep_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "2D timedep internal",
            )

    elif data_dim == '2d_triangle_timedep':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_2d_triangle_timedep_boundary_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_L=min_thick_or_width, max_L=max_thick_or_width,
                min_r=min_height, max_r=max_height,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                angular_sampling=angular_sampling,
                sampling=sampling
            )
            data_type_str = '2d_triangle_timedep_boundary'
        else:
            sim_data = safe_get_sim_data(
                get_2d_triangle_timedep_internal_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_L=min_thick_or_width, max_L=max_thick_or_width,
                min_r=min_height, max_r=max_height,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '2d_triangle_timedep_internal'
        sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
        if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

        _, _, _, _, _, _, LL, LU = compute_normalization_params_2d_triangle_timedep(sim_data, source_type)
        L_tri_L, L_tri_U = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "L_tri")
        r_tri_L, r_tri_U = _analytic_log_bounds(min_height, max_height, "r_tri")
        budget_L, budget_U = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
        L_center, L_scale = robust_log_stats(sim_data["L"].values)

        norm_params = {
            'L_tri_lower': L_tri_L, 'L_tri_upper': L_tri_U,
            'r_tri_lower': r_tri_L, 'r_tri_upper': r_tri_U,
            'budget_lower': budget_L, 'budget_upper': budget_U,
            'L_lower': LL, 'L_upper': LU,
            'L_center': L_center, 'L_scale': L_scale,
            'direction_encoding_2d': direction_encoding_2d,
            'use_L_over_budget': True,
            'time_lower': LL, 'time_upper': LU,  
        }
        norm_params_path = f"ml/data/norm_params_{data_type_str}{sampling_suffix}.npy"
        _save_norm_params_artifacts(
            norm_params_path, norm_params,
            data_type_str,
        )

    elif data_dim == '2d_quad_timedep':
        if source_type == 'boundary':
            raise NotImplementedError("Boundary source for Quads not yet supported.")

        sim_data = safe_get_sim_data(
            get_2d_quad_timedep_internal_source_data,
            num_neutrons=num_neutrons,
            scatter_ratio=1.0, distribution=0,
            min_width=min_thick_or_width, max_width=max_thick_or_width,
            min_path_budget=min_path_budget, max_path_budget=max_path_budget,
            sampling=sampling
        )
        data_type_str = '2d_quad_timedep_internal'
        sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
        if len(sim_data) == 0: raise ValueError("No exiting neutrons found.")

        _, _, x3_L, x3_U, y3_L, y3_U, x4_L, x4_U, y4_L, y4_U, _, _, tL, tU = \
            compute_normalization_params_2d_quad_timedep(sim_data, 'internal')
        L_L, L_U = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "L_quad")
        budget_L, budget_U = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
        L_track_center, L_track_scale = robust_log_stats(sim_data["L_track"].values)

        norm_params = {
            'L_lower': L_L, 'L_upper': L_U,
            'x3_lower': x3_L, 'x3_upper': x3_U,
            'y3_lower': y3_L, 'y3_upper': y3_U,
            'x4_lower': x4_L, 'x4_upper': x4_U,
            'y4_lower': y4_L, 'y4_upper': y4_U,
            'budget_lower': budget_L, 'budget_upper': budget_U,
            'L_track_lower': tL, 'L_track_upper': tU,
            'L_track_center': L_track_center, 'L_track_scale': L_track_scale,
            'direction_encoding_2d': direction_encoding_2d,
            'use_L_over_budget': True,
            'time_lower': tL, 'time_upper': tU,  
        }

        norm_params_path = f"ml/data/norm_params_{data_type_str}{sampling_suffix}.npy"
        _save_norm_params_artifacts(
            norm_params_path, norm_params,
            data_type_str,
        )

    elif data_dim == '1d_timedep':
        if source_type == 'boundary':
            sim_data = safe_get_sim_data(
                get_1d_time_dependent_boundary_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_thickness=min_thick_or_width, max_thickness=max_thick_or_width,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '1d_timedep_boundary'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting particles found.")

            _, _, _, _, L_lower, L_upper = compute_normalization_params_1d_timedep(sim_data, 'boundary')
            thick_lower, thick_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "thickness")
            budget_lower, budget_upper = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'thickness_lower': thick_lower, 'thickness_upper': thick_upper,
                'budget_lower': budget_lower, 'budget_upper': budget_upper,
                'L_lower': L_lower, 'L_upper': L_upper,
                'L_center': L_center, 'L_scale': L_scale,
                'use_L_over_budget': True,
                'time_lower': L_lower, 'time_upper': L_upper 
            }
            norm_params_path = f"ml/data/norm_params_1d_timedep_boundary{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "1D timedep boundary",
            )

        else: 
            sim_data = safe_get_sim_data(
                get_1d_time_dependent_internal_source_data,
                num_neutrons=num_neutrons,
                scatter_ratio=1.0, distribution=0,
                min_thickness=min_thick_or_width, max_thickness=max_thick_or_width,
                min_path_budget=min_path_budget, max_path_budget=max_path_budget,
                sampling=sampling
            )
            data_type_str = '1d_timedep_internal'
            sim_data = sim_data[(sim_data["space_exit"] == 1) | (sim_data["time_exit"] == 1)].copy()
            if len(sim_data) == 0: raise ValueError("No exiting particles found.")

            _, _, _, _, L_lower, L_upper = compute_normalization_params_1d_timedep(sim_data, 'internal')
            thick_lower, thick_upper = _analytic_log_bounds(min_thick_or_width, max_thick_or_width, "thickness")
            budget_lower, budget_upper = _analytic_log_bounds(min_path_budget, max_path_budget, "path_budget")
            L_center, L_scale = robust_log_stats(sim_data["L"].values)
            norm_params = {
                'thickness_lower': thick_lower, 'thickness_upper': thick_upper,
                'budget_lower': budget_lower, 'budget_upper': budget_upper,
                'L_lower': L_lower, 'L_upper': L_upper,
                'L_center': L_center, 'L_scale': L_scale,
                'use_L_over_budget': True,
                'time_lower': L_lower, 'time_upper': L_upper 
            }
            norm_params_path = f"ml/data/norm_params_1d_timedep_internal{sampling_suffix}.npy"
            _save_norm_params_artifacts(
                norm_params_path, norm_params,
                "1D timedep internal",
            )
    elif data_dim == "2d_electron":
        log.info(f"Loading electron data from: {electron_npz}")
        sim_data = load_electron_dataset(electron_npz)
        data_type_str = "2d_electron_boundary"

        log.info(f"Using all {len(sim_data)} particles for training")

        if len(sim_data) == 0:
            raise ValueError("No electrons loaded from NPZ file")

        E_in_lower, E_in_upper, E_out_lower, E_out_upper, \
            energy_dep_lower, energy_dep_upper, path_lower, path_upper = \
            compute_normalization_params_2d_electron(sim_data)

        E_in_center, E_in_scale = robust_log_stats(sim_data["E_in"].values)
        E_out_center, E_out_scale = robust_log_stats(sim_data["E_out"].values)
        energy_dep_center, energy_dep_scale = robust_log_stats(
            np.maximum(sim_data["energy_deposited"].values, 1e-12)
        )
        path_center, path_scale = robust_log_stats(sim_data["path_length"].values)

        norm_params = {
            "E_in_lower": E_in_lower,
            "E_in_upper": E_in_upper,
            "E_in_center": E_in_center,
            "E_in_scale": E_in_scale,
            "E_out_lower": E_out_lower,
            "E_out_upper": E_out_upper,
            "E_out_center": E_out_center,
            "E_out_scale": E_out_scale,
            "energy_dep_lower": energy_dep_lower,
            "energy_dep_upper": energy_dep_upper,
            "energy_dep_center": energy_dep_center,
            "energy_dep_scale": energy_dep_scale,
            "path_lower": path_lower,
            "path_upper": path_upper,
            "path_center": path_center,
            "path_scale": path_scale,
            "direction_encoding_2d": "polar",
        }

        norm_params_path = f"ml/data/norm_params_2d_electron_boundary{sampling_suffix}.npy"
        _save_norm_params_artifacts(
            norm_params_path,
            norm_params,
            "2D Electron Boundary",
        )
    else:
        raise ValueError(f"Unknown data_dim: {data_dim}")

    # Create Dataset
    full_dataset = SimDataset(sim_data, norm_params, data_type=data_type_str)
    input_dim = full_dataset.input_dim
    cond_dim = full_dataset.cond_dim

    # Split dataset
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
         raise ValueError(f"Dataset split resulted in zero samples. Increase dataset size or adjust val_ratio.")
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoaders
    device_str = str(device)
    use_pin_memory = device_str.startswith("cuda")
    available_cpus = _available_cpu_count()
    if train_workers is None:
        if available_cpus <= 1:
            train_workers = 0
        else:
            train_workers = min(4, available_cpus - 1)
    else:
        train_workers = max(0, int(train_workers))

    if val_workers is None:
        if available_cpus <= 1:
            val_workers = 0
        else:
            val_workers = min(2, train_workers)
    else:
        val_workers = max(0, int(val_workers))
    print(f"DataLoader workers: train={train_workers}, val={val_workers}, available_cpus={available_cpus}")
    
    drop_last_train = len(train_dataset) >= batch_size
    if not drop_last_train:
        print(
            f"Warning: train dataset size ({len(train_dataset)}) < batch_size ({batch_size}); "
            "setting drop_last=False to avoid zero-batch epochs."
        )
    train_loader_kwargs = dict(
        batch_size=batch_size, shuffle=True, drop_last=drop_last_train,
        num_workers=train_workers, pin_memory=use_pin_memory, persistent_workers=train_workers > 0,
    )
    if train_workers > 0: train_loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader_kwargs = dict(
        batch_size=batch_size * 2, shuffle=False, drop_last=False,
        num_workers=val_workers, pin_memory=use_pin_memory, persistent_workers=val_workers > 0,
    )
    if val_workers > 0: val_loader_kwargs["prefetch_factor"] = 2
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    return train_loader, val_loader, norm_params, input_dim, cond_dim

def validate(model, val_loader, path, device, epoch=0):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    non_blocking = str(device).startswith("cuda")
    cond_dim = path.cond_dim

    with torch.no_grad():
        for x1_val, _, _ in val_loader:
            x1_val = x1_val.to(device, non_blocking=non_blocking)
            cond_val = x1_val[:, :cond_dim]
            target_val = x1_val[:, cond_dim:]

            # Noise should have same shape as target_val
            noise = torch.randn_like(target_val)
            # x0 = conditions + noise
            x0_val = torch.cat([cond_val, noise], dim=1)

            if x0_val.shape[1] != x1_val.shape[1]:
                 print(f"Warning: Dim mismatch validation. x1: {x1_val.shape}, x0: {x0_val.shape}")
                 continue

            t_sample_val = torch.rand(x1_val.shape[0], device=device)

            path_sample_val = path.sample(t=t_sample_val, x_0=x0_val, x_1=x1_val)
            pred_velocity = model(path_sample_val.x_t, path_sample_val.t)

            if pred_velocity.shape != path_sample_val.dx_t.shape:
                 print(f"Warning: Velocity shape mismatch validation. Pred: {pred_velocity.shape}, Target: {path_sample_val.dx_t.shape}")
                 continue
            velocity_loss = torch.pow(pred_velocity - path_sample_val.dx_t, 2).mean()

            v = velocity_loss.item()
            if not np.isnan(v):
                val_loss += v
                num_batches += 1
            else:
                print(f"Warning: NaN loss during validation epoch {epoch}")

    avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
    model.train()
    return avg_val_loss

def train(num_epochs=50, batch_size=2048, hidden_dim=128, depth=4, expansion_factor=2, lr=4e-4, weight_decay=1e-5,
          min_thick_or_width=0.001, max_thick_or_width=10.0,
          min_height=0.001, max_height=10.0,
          min_depth=0.001, max_depth=10.0,
          scheduler_type='condOT', save_prefix='',
          gradient_accumulation_steps=1, data_dim='2d', source_type='boundary', num_neutrons=100000,
          particle_velocity=1.0, max_time=10.0,
          min_path_budget=1e-5, max_path_budget=0.1,
          direction_encoding_2d='polar',
          angular_sampling=0,
          sampling='random',
          train_workers=None, val_workers=None,electron_npz=None):

    device = setup_environment()
    non_blocking = str(device).startswith("cuda")
    train_loader, val_loader, norm_params, input_dim, cond_dim = prepare_data(
        device, data_dim, source_type, batch_size,
        min_thick_or_width=min_thick_or_width, max_thick_or_width=max_thick_or_width,
        min_height=min_height, max_height=max_height,
        min_depth=min_depth, max_depth=max_depth,
        num_neutrons=num_neutrons,
        particle_velocity=particle_velocity, max_time=max_time,
        min_path_budget=min_path_budget, max_path_budget=max_path_budget,
        direction_encoding_2d=direction_encoding_2d,
        angular_sampling=angular_sampling,
        sampling=sampling,
        train_workers=train_workers,
        val_workers=val_workers,
        electron_npz=electron_npz,
    )

    # Define results directory based on data_dim and source_type
    model_dir = f"ml/models/{data_dim}_{source_type}"
    os.makedirs(model_dir, exist_ok=True)

    training_dir = f"ml/training_results/{data_dim}_{source_type}"
    os.makedirs(training_dir, exist_ok=True)

    print(f"Models will be saved in: {model_dir}")
    print(f"Training results will be saved in: {training_dir}")

    print(f"Initializing ConditionalFlow model for {data_dim} {source_type} source:")
    print(f"  Input Dim: {input_dim}, Cond Dim: {cond_dim}, Hidden Dim: {hidden_dim}, Depth: {depth}, Expansion Factor: {expansion_factor}")

    model = get_model(
        input_dim=input_dim, cond_dim=cond_dim,
        hidden_dim=hidden_dim, depth=depth, expansion_factor=expansion_factor
    ).to(device)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)\n")

    if str(device).startswith("cuda"):
        # torch.compile with reduce-overhead can cause severe slowdowns 
        # due to recompilation, especially with dynamic validation batch sizes
        compiled_model = model # torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    # Use fused=True on CUDA to coalesce hundreds of small optimizer kernels into one, massively reducing CPU overhead.
    optimizer_kwargs = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)
    if str(device).startswith("cuda"):
        optimizer_kwargs["fused"] = True
        
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    total_batches_per_epoch = len(train_loader)
    total_optimization_steps = (
        num_epochs * total_batches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_optimization_steps = max(1, total_optimization_steps)
    print(f"Total optimization steps: {total_optimization_steps}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_optimization_steps,
        pct_start=0.1, div_factor=25, final_div_factor=1000, anneal_strategy='cos'
    )

    # Get path with correct cond_dim
    path = get_flow_matching_path(cond_dim=cond_dim, scheduler_type=scheduler_type)
    # Use the passed save_prefix for filenames
    filename_prefix = save_prefix
    print(f"Using flow matching path with {scheduler_type} scheduler and cond_dim={cond_dim}")
    print(f"Filename prefix for saving models: '{filename_prefix}' (in '{model_dir}')")

    tracker = TrainingTracker(save_dir=training_dir)
    start_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")
    print(f"  Batch size: {batch_size}, Grad Accum Steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    global_step = 0
    epoch_times = []
    epoch_train_losses = []
    epoch_val_losses = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches_processed = 0
        compiled_model.train()
        print(f"Epoch {epoch+1}: processing {len(train_loader)} training batches...")

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x1, _, _) in enumerate(train_loader):
            x1 = x1.to(device, non_blocking=non_blocking)
            cond = x1[:, :cond_dim]
            target = x1[:, cond_dim:]

            # Create random noise for input x0, matching target shape
            noise = torch.randn_like(target)
            x0 = torch.cat([cond, noise], dim=1)

            if x0.shape[1] != x1.shape[1]:
                 print(f"Warning: Dim mismatch train batch {batch_idx}. x1: {x1.shape}, x0: {x0.shape}")
                 continue

            t_sample = torch.rand(x1.shape[0], device=device)
            path_sample = path.sample(t=t_sample, x_0=x0, x_1=x1)

            pred_velocity = compiled_model(path_sample.x_t, path_sample.t)
            if pred_velocity.shape != path_sample.dx_t.shape:
                print(f"Warning: Velocity shape mismatch train batch {batch_idx}. Pred: {pred_velocity.shape}, Target: {path_sample.dx_t.shape}")
                continue
            loss = torch.pow(pred_velocity - path_sample.dx_t, 2).mean()
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # Single sync per batch — GPU runs forward→backward→step uninterrupted
            loss_value = loss.item() * gradient_accumulation_steps
            if not np.isnan(loss_value):
                 epoch_loss += loss_value
                 num_batches_processed += 1
                 tracker.add_train_loss(loss_value)
            else:
                print(f"Warning: NaN loss detected epoch {epoch+1}, batch {batch_idx+1}.")
                optimizer.zero_grad(set_to_none=True)

        avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else 0
        val_loss = validate(compiled_model, val_loader, path, device, epoch)
        epoch_train_losses.append(float(avg_epoch_loss))
        epoch_val_losses.append(float(val_loss))
        is_best = tracker.add_val_loss(val_loss)
        tracker.plot_losses(filename=f"training_loss{filename_prefix}.png")

        # Calculate timing information
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_hours, eta_rem = divmod(eta_seconds, 3600)
        eta_minutes, eta_secs = divmod(eta_rem, 60)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_epoch_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}")
        print(f"  Epoch time: {epoch_time:.2f}s - ETA: {int(eta_hours)}h {int(eta_minutes)}m {int(eta_secs)}s")

        if is_best:
            print(f"New best model found! Val loss: {val_loss:.6f}")
            save_path = os.path.join(model_dir, f"best_model{filename_prefix}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Saving final model...")
    save_path = os.path.join(model_dir, f"trained_model_final{filename_prefix}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")

    # Save loss histories for experiment reporting.
    history_csv_path = os.path.join(training_dir, f"loss_history{filename_prefix}.csv")
    with open(history_csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for idx, (train_l, val_l) in enumerate(zip(epoch_train_losses, epoch_val_losses), start=1):
            f.write(f"{idx},{train_l:.10e},{val_l:.10e}\n")
    np.save(os.path.join(training_dir, f"train_loss_batch_history{filename_prefix}.npy"), np.asarray(tracker.train_losses, dtype=np.float64))
    np.save(os.path.join(training_dir, f"val_loss_epoch_history{filename_prefix}.npy"), np.asarray(epoch_val_losses, dtype=np.float64))
    print(f"Saved loss history CSV to {history_csv_path}")

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600); minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best validation loss achieved: {tracker.best_val_loss:.6f}")
    print("Training complete!")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Conditional Flow Matching model')
    parser.add_argument('--data_dim', type=str, default='2d',
                        choices=DATA_DIM_CHOICES,
                        help='Dimension of the problem (1d, 2d, 3d, 1d_timedep, 2d_timedep, 3d_timedep, 2d_triangle_timedep, 2d_quad_timedep, or 2d_electron)')
    parser.add_argument('--source_type', type=str, default='boundary', choices=['boundary', 'internal'],
                        help='Source type (boundary or internal)')
    parser.add_argument('--num_neutrons', type=int, default=100000,
                        help='Number of neutrons for simulation data generation')
    parser.add_argument('--min_width', type=float, default=0.01, help='Minimum cell width')
    parser.add_argument('--max_width', type=float, default=10.0, help='Maximum cell width')
    parser.add_argument('--min_height', type=float, default=0.01, help='Minimum cell height (2D/3D)')
    parser.add_argument('--max_height', type=float, default=10.0, help='Maximum cell height (2D/3D)')
    parser.add_argument('--min_depth', type=float, default=0.01, help='Minimum cell depth (3D only)')
    parser.add_argument('--max_depth', type=float, default=1.0, help='Maximum cell depth (3D only)')
    parser.add_argument('--particle_velocity', type=float, default=1.0)
    parser.add_argument('--max_time', type=float, default=10.0)
    parser.add_argument('--min_path_budget', type=float, default=1e-6,
                        help='Minimum path-length budget ℓ_t (for 1d_timedep/2d_timedep/2d_triangle_timedep/2d_quad_timedep)')
    parser.add_argument('--max_path_budget', type=float, default=10.0,
                        help='Maximum path-length budget ℓ_t (for 1d_timedep/2d_timedep/2d_triangle_timedep/2d_quad_timedep)')
    parser.add_argument('--angular_sampling', type=int, default=0, choices=[0, 1],
                        help='Boundary angular sampling for timedep datasets: 0=uniform-u (ML), 1=cosine-law (physical)')
    parser.add_argument('--direction_encoding_2d', type=str, default='polar', choices=['polar'],
                        help='Encoding for 2D direction channels. Fixed to polar.')

    parser.add_argument('--num_epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=4e-4, help='Maximum learning rate for OneCycleLR')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')

    parser.add_argument('--hidden_dim', type=int, default=160, help='Hidden dimensions in the model')
    parser.add_argument('--depth', type=int, default=5, help='Number of flow blocks in the model')
    parser.add_argument('--expansion_factor', type=int, default=1, help='FFN expansion factor inside each flow block')

    parser.add_argument('--scheduler', type=str, default='condOT',
                        choices=['condOT', 'polynomial', 'linearVP', 'cosine'],
                        help='Type of scheduler for the flow matching path')
    parser.add_argument('--sampling', type=str, default='random', choices=['random', 'qmc', 'lhs'],
                        help='Sampling method for condition generation during data simulation')
    parser.add_argument('--train_workers', type=int, default=None,
                        help='Number of DataLoader workers for train loader (default: auto)')
    parser.add_argument('--val_workers', type=int, default=None,
                        help='Number of DataLoader workers for val loader (default: auto)')
    parser.add_argument('--electron_npz', type=str, default=None,
                        help='Path to electron NPZ file (required when data_dim=2d_electron)')

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()

    print(f"Starting training for {args.data_dim} {args.source_type} source with {args.scheduler} scheduler")
    if args.data_dim in TIME_DEPENDENT_DATA_DIMS:
        print(f"Path-length budget parameters: min={args.min_path_budget}, max={args.max_path_budget}")
    if args.data_dim == "2d_electron" and args.electron_npz is None:
        print("ERROR: data_dim='2d_electron' requires --electron_npz argument")
        return

    save_prefix = f"_{args.scheduler}" if args.scheduler != 'condOT' else ''
    if args.sampling != 'random':
        save_prefix += f"_{args.sampling}"

    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        expansion_factor=args.expansion_factor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_thick_or_width=args.min_width,
        max_thick_or_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        scheduler_type=args.scheduler,
        save_prefix=save_prefix,
        gradient_accumulation_steps=args.grad_accum,
        data_dim=args.data_dim,
        source_type=args.source_type,
        num_neutrons=args.num_neutrons,
        particle_velocity=args.particle_velocity,
        max_time=args.max_time,
        min_path_budget=args.min_path_budget,
        max_path_budget=args.max_path_budget,
        direction_encoding_2d=args.direction_encoding_2d,
        angular_sampling=args.angular_sampling,
        sampling=args.sampling,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        electron_npz=args.electron_npz,  # Use args.electron_npz here
    )

if __name__ == '__main__':
    main()
