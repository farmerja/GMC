from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from flow_matching.solver import ODESolver

from ml.model import WrappedModel, get_model
from ml.ode_grid import build_piecewise_time_grid, build_teacher_time_grid, build_uniform_time_grid
from ml.utils import (
    _extract_state_dict,
    _infer_architecture_from_state_dict,
    _normalize_state_dict_keys,
    steady_2d_model_dir_candidates,
)


@dataclass(frozen=True)
class FlowTeacherConfig:
    data_dim: str
    source_type: str
    checkpoint_path: str
    hidden_dim: int
    depth: int
    expansion_factor: int
    train_sampling: str = "random"
    scheduler_type: str = "condOT"
    scattering_mode: str = "isotropic"
    ode_method: str = "rk4"
    ode_num_steps: int = 12
    time_grid_kind: str = "piecewise"
    time_grid_split: float = 0.8
    time_grid_schedule: str | None = None


def _resolve_teacher_grid(
    *,
    device: str,
    ode_num_steps: int,
    time_grid_kind: str,
    time_grid_split: float,
    time_grid_schedule: str | None,
):
    normalized_kind = str(time_grid_kind).strip().lower()
    if normalized_kind == "uniform":
        return build_uniform_time_grid(ode_num_steps, device=device)
    if normalized_kind == "piecewise":
        return build_piecewise_time_grid(ode_num_steps, split=time_grid_split, device=device)
    if normalized_kind == "schedule":
        return build_teacher_time_grid(device=device, schedule=time_grid_schedule or "steady2d_default")
    raise ValueError(f"Unsupported teacher time_grid_kind='{time_grid_kind}'")


class FlowTeacherSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        cond_dim: int,
        target_dim: int,
        ode_method: str = "rk4",
        ode_num_steps: int = 12,
        time_grid_kind: str = "piecewise",
        time_grid_split: float = 0.8,
        time_grid_schedule: str | None = None,
    ):
        self.model = model.to(device)
        self.device = str(device)
        self.cond_dim = int(cond_dim)
        self.target_dim = int(target_dim)
        self.ode_method = str(ode_method)
        self.ode_num_steps = int(ode_num_steps)
        self.time_grid_kind = str(time_grid_kind)
        self.time_grid_split = float(time_grid_split)
        self.time_grid_schedule = time_grid_schedule
        self.wrapped_model = WrappedModel(self.model).to(self.device)
        self.solver = ODESolver(velocity_model=self.wrapped_model)
        self.time_grid, self.step_size = _resolve_teacher_grid(
            device=self.device,
            ode_num_steps=self.ode_num_steps,
            time_grid_kind=self.time_grid_kind,
            time_grid_split=self.time_grid_split,
            time_grid_schedule=self.time_grid_schedule,
        )

    @classmethod
    def from_config(
        cls,
        config: FlowTeacherConfig,
        *,
        input_dim: int,
        cond_dim: int,
        device: Optional[str] = None,
    ) -> "FlowTeacherSampler":
        resolved_device = str(
            device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = load_teacher_model_checkpoint(
            config.checkpoint_path,
            device=resolved_device,
            input_dim=input_dim,
            cond_dim=cond_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            expansion_factor=config.expansion_factor,
            data_dim=config.data_dim,
            source_type=config.source_type,
            train_sampling=config.train_sampling,
            scattering_mode=config.scattering_mode,
        )
        return cls(
            model,
            device=resolved_device,
            cond_dim=cond_dim,
            target_dim=int(input_dim - cond_dim),
            ode_method=config.ode_method,
            ode_num_steps=config.ode_num_steps,
            time_grid_kind=config.time_grid_kind,
            time_grid_split=config.time_grid_split,
            time_grid_schedule=config.time_grid_schedule,
        )

    @classmethod
    def from_task(
        cls,
        task_spec,
        *,
        checkpoint_path: str,
        device: Optional[str] = None,
        ode_method: str = "rk4",
        ode_num_steps: int | None = None,
        teacher_train_sampling: str = "random",
        time_grid_kind: str | None = None,
        time_grid_split: float | None = None,
        time_grid_schedule: str | None = None,
    ) -> "FlowTeacherSampler":
        return cls.from_config(
            FlowTeacherConfig(
                data_dim=task_spec.data_dim,
                source_type=task_spec.source_type,
                checkpoint_path=checkpoint_path,
                hidden_dim=task_spec.teacher_hidden_dim,
                depth=task_spec.teacher_depth,
                expansion_factor=task_spec.teacher_expansion_factor,
                train_sampling=teacher_train_sampling,
                scattering_mode=task_spec.scattering_mode,
                ode_method=ode_method,
                ode_num_steps=int(ode_num_steps or task_spec.teacher_default_num_steps),
                time_grid_kind=time_grid_kind or task_spec.teacher_time_grid_kind,
                time_grid_split=float(
                    task_spec.teacher_time_grid_split if time_grid_split is None else time_grid_split
                ),
                time_grid_schedule=time_grid_schedule or task_spec.teacher_time_grid_schedule,
            ),
            input_dim=task_spec.cond_dim + task_spec.target_dim,
            cond_dim=task_spec.cond_dim,
            device=device,
        )

    @torch.inference_mode()
    def sample_target(
        self,
        conditions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        if conditions.ndim != 2 or conditions.shape[1] != self.cond_dim:
            raise ValueError(
                f"Expected conditions shape [batch, {self.cond_dim}], got {tuple(conditions.shape)}"
            )
        num_conditions = int(conditions.shape[0])
        if num_conditions == 0:
            return torch.empty((0, self.target_dim), dtype=torch.float32)

        cond = conditions.to(self.device, dtype=torch.float32)
        if noise is None:
            generator = torch.Generator(device=self.device)
            if noise_seed is not None:
                generator.manual_seed(int(noise_seed))
            latent = torch.randn((num_conditions, self.target_dim), device=self.device, generator=generator)
        else:
            if noise.shape != (num_conditions, self.target_dim):
                raise ValueError(
                    f"Expected noise shape [{num_conditions}, {self.target_dim}], got {tuple(noise.shape)}"
                )
            latent = noise.to(self.device, dtype=torch.float32)
        x_init = torch.cat([cond, latent], dim=1)
        self.model.eval()
        self.wrapped_model.eval()
        sol = self.solver.sample(
            time_grid=self.time_grid,
            x_init=x_init,
            method=self.ode_method,
            step_size=self.step_size,
            return_intermediates=False,
        )
        return sol[:, self.cond_dim :].detach().cpu()

    @torch.inference_mode()
    def sample_targets(
        self,
        conditions: torch.Tensor,
        *,
        num_shots: int,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        if int(num_shots) <= 0:
            raise ValueError(f"num_shots must be positive, got {num_shots}")
        generator = torch.Generator(device="cpu")
        if noise_seed is not None:
            generator.manual_seed(int(noise_seed))
        noise = torch.randn(
            (conditions.shape[0], int(num_shots), self.target_dim),
            generator=generator,
            dtype=torch.float32,
        )
        outputs = [
            self.sample_target(conditions, noise=noise[:, shot_idx, :])
            for shot_idx in range(int(num_shots))
        ]
        return torch.stack(outputs, dim=1)


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _verify_checkpoint_metadata(
    checkpoint,
    *,
    checkpoint_path: Path,
    input_dim: int,
    cond_dim: int,
    data_dim: str,
    source_type: str,
    train_sampling: str,
    scattering_mode: str,
) -> None:
    if not isinstance(checkpoint, dict):
        metadata = {}
    else:
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            metadata = checkpoint

    has_rich_metadata = any(
        key in metadata for key in ("data_dim", "source_type", "train_sampling", "scattering_mode")
    )
    if not has_rich_metadata and data_dim == "2d":
        allowed_dirs = [
            Path(path)
            for path in steady_2d_model_dir_candidates(data_dim, source_type, scattering_mode=scattering_mode)
        ]
        if allowed_dirs and not any(_path_is_under(checkpoint_path, model_dir) for model_dir in allowed_dirs):
            raise ValueError(
                f"Teacher checkpoint {checkpoint_path} does not live under the expected model directories for "
                f"{data_dim}/{source_type}/{scattering_mode}, and it does not expose matching metadata."
            )

    checkpoint_name = checkpoint_path.name
    if metadata.get("train_sampling") is None:
        if train_sampling != "random" and f"_{train_sampling}.pth" not in checkpoint_name:
            raise ValueError(
                f"Teacher checkpoint {checkpoint_path} does not match requested train_sampling='{train_sampling}'."
            )
        if train_sampling == "random" and any(suffix in checkpoint_name for suffix in ("_lhs.pth", "_qmc.pth")):
            raise ValueError(
                f"Teacher checkpoint {checkpoint_path} appears to be sampling-specific, "
                "but train_sampling='random' was requested."
            )

    expected_values = {
        "input_dim": int(input_dim),
        "cond_dim": int(cond_dim),
        "data_dim": str(data_dim),
        "source_type": str(source_type),
        "train_sampling": str(train_sampling),
        "scattering_mode": str(scattering_mode),
    }
    for key, expected in expected_values.items():
        actual = metadata.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"Teacher checkpoint metadata mismatch for '{key}': expected {expected!r}, got {actual!r}"
            )


def load_teacher_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str,
    input_dim: int,
    cond_dim: int,
    hidden_dim: int,
    depth: int,
    expansion_factor: int,
    data_dim: str,
    source_type: str,
    train_sampling: str,
    scattering_mode: str,
) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    _verify_checkpoint_metadata(
        checkpoint,
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        cond_dim=cond_dim,
        data_dim=data_dim,
        source_type=source_type,
        train_sampling=train_sampling,
        scattering_mode=scattering_mode,
    )

    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    inferred = _infer_architecture_from_state_dict(state_dict)
    ckpt_input_dim = inferred.get("input_dim")
    ckpt_cond_dim = inferred.get("cond_dim")
    if ckpt_input_dim is not None and ckpt_input_dim != int(input_dim):
        raise ValueError(
            f"Teacher checkpoint input_dim mismatch: expected {input_dim}, got {ckpt_input_dim}"
        )
    if ckpt_cond_dim is not None and ckpt_cond_dim != int(cond_dim):
        raise ValueError(
            f"Teacher checkpoint cond_dim mismatch: expected {cond_dim}, got {ckpt_cond_dim}"
        )

    model = get_model(
        input_dim=int(input_dim),
        cond_dim=int(cond_dim),
        hidden_dim=int(inferred.get("hidden_dim", hidden_dim)),
        depth=int(inferred.get("depth", depth)),
        expansion_factor=int(inferred.get("expansion_factor", expansion_factor)),
    ).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Teacher checkpoint state_dict is incompatible with the inferred architecture. "
            f"missing_keys={missing_keys[:5]}, unexpected_keys={unexpected_keys[:5]}"
        )
    model.eval()
    return model
