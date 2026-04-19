#!/usr/bin/env python3
"""Shared ODE inference schedules used by GMC teachers and student comparisons."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


def _resolve_device(device: Optional[str]) -> str:
    return str(device or "cpu")


def _grid_step_size(time_grid: torch.Tensor) -> Optional[torch.Tensor]:
    if time_grid.numel() <= 1:
        return None
    return time_grid[1] - time_grid[0]


def build_explicit_time_grid(
    points: Iterable[float],
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    time_grid = torch.tensor(list(points), device=_resolve_device(device), dtype=torch.float32)
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError(f"Expected at least two time-grid points, got {tuple(time_grid.shape)}")
    return time_grid, _grid_step_size(time_grid)


def build_teacher_time_grid(
    device: Optional[str] = None,
    *,
    schedule: str = "steady2d_default",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return an explicit teacher schedule used by existing GMC transport code."""

    if schedule != "steady2d_default":
        raise ValueError(f"Unsupported teacher schedule: {schedule}")
    return build_explicit_time_grid([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], device=device)


def build_uniform_time_grid(
    num_steps: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_steps = int(num_steps)
    if num_steps <= 1:
        return build_explicit_time_grid([0.0, 1.0], device=device)
    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=_resolve_device(device), dtype=torch.float32)
    return time_grid, _grid_step_size(time_grid)


def build_piecewise_time_grid(
    num_steps: int,
    *,
    split: float = 0.8,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct a coarse-then-refined grid over [0, 1].

    `num_steps` is interpreted as the nominal number of solver steps, so the
    returned grid contains `num_steps + 1` points.
    """

    num_steps = int(num_steps)
    if not 0.0 < float(split) < 1.0:
        raise ValueError(f"split must lie in (0, 1), got {split}")
    if num_steps <= 1:
        return build_explicit_time_grid([0.0, 1.0], device=device)

    dev = _resolve_device(device)
    coarse_steps = max(1, num_steps // 2)
    fine_steps = max(1, num_steps - coarse_steps)
    coarse_grid = torch.linspace(0.0, float(split), coarse_steps + 1, device=dev, dtype=torch.float32)[:-1]
    fine_grid = torch.linspace(float(split), 1.0, fine_steps + 1, device=dev, dtype=torch.float32)
    time_grid = torch.cat([coarse_grid, fine_grid])
    return time_grid, _grid_step_size(time_grid)
