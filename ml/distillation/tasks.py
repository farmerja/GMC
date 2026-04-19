from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch

from ml.transform import (
    normalize_angle_phi,
    normalize_height,
    normalize_position_relative,
    normalize_radius_r,
    normalize_width,
    uv_to_polar,
)
from ml.utils import compute_model_dims, generate_samples, load_norm_params_for_data_dim
from physics.steady_2d_scattering import normalize_steady_2d_scattering_mode


@dataclass(frozen=True)
class DistillationTaskSpec:
    task_name: str
    data_dim: str
    source_type: str
    scattering_mode: str
    cond_dim: int
    target_dim: int
    norm_params: Dict[str, float]
    teacher_hidden_dim: int
    teacher_depth: int
    teacher_expansion_factor: int
    teacher_time_grid_kind: str
    teacher_time_grid_split: float
    teacher_time_grid_schedule: str | None
    teacher_default_num_steps: int
    sample_conditions: Callable[..., torch.Tensor]


def _steady_2d_support_sampler(
    source_type: str,
    norm_params: Dict[str, float],
) -> Callable[..., torch.Tensor]:
    min_width = float(10 ** float(norm_params["width_lower"]))
    max_width = float(10 ** float(norm_params["width_upper"]))
    min_height = float(10 ** float(norm_params["height_lower"]))
    max_height = float(10 ** float(norm_params["height_upper"]))

    def sample_conditions(
        num_samples: int,
        *,
        sampling: str = "lhs",
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        samples = generate_samples(int(num_samples), 5 if source_type == "boundary" else 6, sampling=sampling, seed=seed)

        log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
        width_vals = 10**log_w
        log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
        height_vals = 10**log_h

        if source_type == "boundary":
            y_in_norm_vals = samples[:, 2]
            mu = 2.0 * samples[:, 3] - 1.0
            phi = samples[:, 4] * np.pi - np.pi / 2.0
            t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
            u_in_vals = t * np.cos(phi)
            v_in_vals = t * np.sin(phi)
            phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)
            cond_np = np.stack(
                [
                    normalize_width(width_vals, norm_params["width_lower"], norm_params["width_upper"]),
                    normalize_height(height_vals, norm_params["height_lower"], norm_params["height_upper"]),
                    normalize_position_relative(y_in_norm_vals),
                    normalize_angle_phi(phi_in_vals),
                    normalize_radius_r(r_in_vals),
                ],
                axis=1,
            )
        else:
            x_in_norm_vals = samples[:, 2]
            y_in_norm_vals = samples[:, 3]
            mu = 2.0 * samples[:, 4] - 1.0
            phi = samples[:, 5] * 2.0 * np.pi
            t = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
            u_in_vals = t * np.cos(phi)
            v_in_vals = t * np.sin(phi)
            phi_in_vals, r_in_vals = uv_to_polar(u_in_vals, v_in_vals)
            cond_np = np.stack(
                [
                    normalize_width(width_vals, norm_params["width_lower"], norm_params["width_upper"]),
                    normalize_height(height_vals, norm_params["height_lower"], norm_params["height_upper"]),
                    normalize_position_relative(x_in_norm_vals),
                    normalize_position_relative(y_in_norm_vals),
                    normalize_angle_phi(phi_in_vals),
                    normalize_radius_r(r_in_vals),
                ],
                axis=1,
            )

        return torch.tensor(cond_np, dtype=torch.float32, device=device)

    return sample_conditions


def resolve_distillation_task(
    task_name: str,
    *,
    source_type: str,
    teacher_train_sampling: str = "random",
    scattering_mode: str = "isotropic",
) -> DistillationTaskSpec:
    normalized_task = str(task_name).strip().lower()
    if normalized_task != "steady2d":
        raise ValueError(f"Unsupported distillation task '{task_name}'. Only 'steady2d' is currently implemented.")

    normalized_source = str(source_type).strip().lower()
    if normalized_source not in {"boundary", "internal"}:
        raise ValueError(f"Unsupported steady2d source_type '{source_type}'")

    normalized_mode = normalize_steady_2d_scattering_mode(scattering_mode)
    norm_params, _ = load_norm_params_for_data_dim(
        "2d",
        normalized_source,
        train_sampling=teacher_train_sampling,
        scattering_mode=normalized_mode,
    )
    input_dim, cond_dim = compute_model_dims(
        "2d",
        normalized_source,
        train_sampling=teacher_train_sampling,
        scattering_mode=normalized_mode,
    )
    return DistillationTaskSpec(
        task_name=normalized_task,
        data_dim="2d",
        source_type=normalized_source,
        scattering_mode=normalized_mode,
        cond_dim=int(cond_dim),
        target_dim=int(input_dim - cond_dim),
        norm_params=norm_params,
        teacher_hidden_dim=160,
        teacher_depth=5,
        teacher_expansion_factor=1,
        teacher_time_grid_kind="schedule",
        teacher_time_grid_split=0.8,
        teacher_time_grid_schedule="steady2d_default",
        teacher_default_num_steps=6,
        sample_conditions=_steady_2d_support_sampler(normalized_source, norm_params),
    )
