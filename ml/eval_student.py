#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.distillation.artifacts import resolve_student_checkpoint_path, resolve_student_run_dir, save_json
from ml.distillation.student import load_student_checkpoint
from ml.distillation.tasks import resolve_distillation_task
from ml.distillation.teacher import FlowTeacherSampler
from ml.distillation.trainer import (
    benchmark_sampler,
    condition_slice_metrics,
    prediction_metrics,
    sample_standard_noise,
    sample_student_predictions,
)
from ml.utils import setup_environment


def _normalize_steps(step_values) -> list[int]:
    normalized = [int(value) for value in step_values]
    allowed = {1, 2, 4}
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported student step counts {invalid}; only [4, 2, 1] are supported.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate student step counts are not allowed: {normalized}")
    return normalized


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-step GMC students against teacher supervision.")
    parser.add_argument("--task", default="steady2d", choices=["steady2d"])
    parser.add_argument("--source-type", required=True, choices=["boundary", "internal"])
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--checkpoint", nargs="+", default=None)
    parser.add_argument("--scattering-mode", default="isotropic")
    parser.add_argument("--teacher-train-sampling", default="random")
    parser.add_argument("--teacher-ode-method", default="rk4")
    parser.add_argument("--teacher-num-steps", type=int, default=None)
    parser.add_argument("--teacher-time-grid", choices=["uniform", "piecewise", "schedule"], default=None)
    parser.add_argument("--teacher-time-split", type=float, default=None)
    parser.add_argument("--teacher-time-schedule", default=None)
    parser.add_argument("--support-samples", type=int, default=1024)
    parser.add_argument("--support-sampling", default="lhs", choices=["random", "lhs", "qmc"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--artifact-root", default="ml/student_artifacts")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    args.steps = _normalize_steps(args.steps)
    return args


def _resolve_checkpoints(args: argparse.Namespace, task_spec):
    if args.checkpoint:
        return [Path(value) for value in args.checkpoint]
    return [
        resolve_student_checkpoint_path(
            task_spec.task_name,
            task_spec.source_type,
            num_steps,
            scattering_mode=task_spec.scattering_mode,
            artifact_root=args.artifact_root,
        )
        for num_steps in args.steps
    ]


def _validate_teacher_metadata(metadata, teacher: FlowTeacherSampler, task_spec, args: argparse.Namespace) -> None:
    if not isinstance(metadata, dict):
        return
    expected = {
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_train_sampling": args.teacher_train_sampling,
        "teacher_ode_method": args.teacher_ode_method,
        "teacher_num_steps": int(getattr(teacher, "ode_num_steps", task_spec.teacher_default_num_steps)),
        "teacher_time_grid_kind": str(getattr(teacher, "time_grid_kind", task_spec.teacher_time_grid_kind)),
        "teacher_time_grid_schedule": getattr(teacher, "time_grid_schedule", task_spec.teacher_time_grid_schedule),
        "teacher_time_grid_split": float(getattr(teacher, "time_grid_split", task_spec.teacher_time_grid_split)),
    }
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value is None:
            continue
        if key == "teacher_time_grid_split":
            if abs(float(actual_value) - float(expected_value)) > 1.0e-8:
                raise ValueError(
                    f"Student checkpoint metadata mismatch for '{key}': expected {expected_value}, got {actual_value}"
                )
            continue
        if actual_value != expected_value:
            raise ValueError(
                f"Student checkpoint metadata mismatch for '{key}': expected {expected_value!r}, got {actual_value!r}"
            )


def main(argv=None) -> None:
    args = parse_args(argv)
    device = str(args.device or setup_environment(seed=args.seed, inference=True))
    task_spec = resolve_distillation_task(
        args.task,
        source_type=args.source_type,
        teacher_train_sampling=args.teacher_train_sampling,
        scattering_mode=args.scattering_mode,
    )
    teacher = FlowTeacherSampler.from_task(
        task_spec,
        checkpoint_path=args.teacher_checkpoint,
        device=device,
        ode_method=args.teacher_ode_method,
        ode_num_steps=args.teacher_num_steps,
        teacher_train_sampling=args.teacher_train_sampling,
        time_grid_kind=args.teacher_time_grid,
        time_grid_split=args.teacher_time_split,
        time_grid_schedule=args.teacher_time_schedule,
    )
    conditions = task_spec.sample_conditions(
        args.support_samples,
        sampling=args.support_sampling,
        seed=args.seed,
        device="cpu",
    )
    evaluation_noise = sample_standard_noise(
        conditions.shape[0],
        task_spec.target_dim,
        seed=args.seed + 77,
        device="cpu",
    )
    teacher_targets = teacher.sample_target(conditions, noise=evaluation_noise)

    for checkpoint in _resolve_checkpoints(args, task_spec):
        model, config, metadata = load_student_checkpoint(checkpoint, map_location=device)
        _validate_teacher_metadata(metadata, teacher, task_spec, args)
        model = model.to(device)
        model.eval()
        student_targets = sample_student_predictions(
            model,
            conditions,
            evaluation_noise,
            device=device,
            batch_size=args.batch_size,
        )
        metrics = prediction_metrics(student_targets, teacher_targets)
        teacher_runtime = benchmark_sampler(
            lambda cond_batch, noise_batch: teacher.sample_target(cond_batch, noise=noise_batch),
            conditions,
            evaluation_noise,
            batch_size=args.batch_size,
            repeats=args.benchmark_repeats,
            device=device,
        )
        student_runtime = benchmark_sampler(
            lambda cond_batch, noise_batch: model(
                cond_batch.to(device),
                noise_batch.to(device),
            ).detach(),
            conditions,
            evaluation_noise,
            batch_size=args.batch_size,
            repeats=args.benchmark_repeats,
            device=device,
        )
        speedup = (
            teacher_runtime["runtime_seconds"] / student_runtime["runtime_seconds"]
            if student_runtime["runtime_seconds"] > 0.0
            else float("inf")
        )
        slices = condition_slice_metrics(conditions, student_targets, teacher_targets)
        run_dir = checkpoint.parent if checkpoint.parent.name.endswith("_step") else resolve_student_run_dir(
            task_spec.task_name,
            task_spec.source_type,
            config.num_steps,
            scattering_mode=task_spec.scattering_mode,
            artifact_root=args.artifact_root,
        )
        save_json(
            run_dir / "eval_metrics.json",
            {
                "checkpoint": str(checkpoint),
                "loss": float(metrics["loss"]),
                "mae": float(metrics["mae"]),
                "mean_bias_by_dim": metrics["mean_bias_by_dim"],
                "std_ratio_by_dim": metrics["std_ratio_by_dim"],
                "support_samples": int(args.support_samples),
                "support_sampling": args.support_sampling,
                "seed": int(args.seed),
                "teacher_checkpoint": args.teacher_checkpoint,
                "task_name": task_spec.task_name,
                "source_type": task_spec.source_type,
                "scattering_mode": task_spec.scattering_mode,
                "student_num_steps": int(config.num_steps),
                "teacher_runtime": teacher_runtime,
                "student_runtime": student_runtime,
                "student_speedup_vs_teacher": float(speedup),
                "condition_slice_metrics": slices,
                "metadata": metadata,
            },
        )


if __name__ == "__main__":
    main()
