#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.distillation.artifacts import resolve_student_run_dir, save_json
from ml.distillation.student import FixedStepStudent, save_student_checkpoint
from ml.distillation.tasks import resolve_distillation_task
from ml.distillation.teacher import FlowTeacherSampler
from ml.distillation.trainer import sample_standard_noise, split_condition_tensor, train_student_model
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
    parser = argparse.ArgumentParser(description="Train fixed-step GMC student models from a saved teacher.")
    parser.add_argument("--task", default="steady2d", choices=["steady2d"])
    parser.add_argument("--source-type", required=True, choices=["boundary", "internal"])
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--scattering-mode", default="isotropic")
    parser.add_argument("--teacher-train-sampling", default="random")
    parser.add_argument("--teacher-ode-method", default="rk4")
    parser.add_argument("--teacher-num-steps", type=int, default=None)
    parser.add_argument("--teacher-time-grid", choices=["uniform", "piecewise", "schedule"], default=None)
    parser.add_argument("--teacher-time-split", type=float, default=None)
    parser.add_argument("--teacher-time-schedule", default=None)
    parser.add_argument("--support-samples", type=int, default=4096)
    parser.add_argument("--support-sampling", default="lhs", choices=["random", "lhs", "qmc"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--student-hidden-dim", type=int, default=128)
    parser.add_argument("--student-depth", type=int, default=3)
    parser.add_argument("--student-expansion-factor", type=int, default=2)
    parser.add_argument("--artifact-root", default="ml/student_artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    args.steps = _normalize_steps(args.steps)
    return args


def train_step_count(args: argparse.Namespace, task_spec, teacher: FlowTeacherSampler, num_steps: int, device: str) -> None:
    run_dir = resolve_student_run_dir(
        task_spec.task_name,
        task_spec.source_type,
        num_steps,
        scattering_mode=task_spec.scattering_mode,
        artifact_root=args.artifact_root,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    all_conditions = task_spec.sample_conditions(
        args.support_samples,
        sampling=args.support_sampling,
        seed=args.seed,
        device="cpu",
    )
    train_conditions, val_conditions = split_condition_tensor(
        all_conditions,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_noise = sample_standard_noise(
        train_conditions.shape[0],
        task_spec.target_dim,
        seed=args.seed + num_steps * 1000,
        device="cpu",
    )
    val_noise = sample_standard_noise(
        val_conditions.shape[0],
        task_spec.target_dim,
        seed=args.seed + num_steps * 1000 + 1,
        device="cpu",
    )
    train_targets = teacher.sample_target(train_conditions, noise=train_noise)
    val_targets = teacher.sample_target(val_conditions, noise=val_noise)

    model = FixedStepStudent(
        cond_dim=task_spec.cond_dim,
        target_dim=task_spec.target_dim,
        num_steps=num_steps,
        hidden_dim=args.student_hidden_dim,
        depth=args.student_depth,
        expansion_factor=args.student_expansion_factor,
    )
    started_at = time.time()
    result = train_student_model(
        model,
        train_conditions,
        train_noise,
        train_targets,
        val_conditions,
        val_noise,
        val_targets,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    finished_at = time.time()

    model.load_state_dict(result.best_state_dict)
    metadata = {
        "task_name": task_spec.task_name,
        "source_type": task_spec.source_type,
        "scattering_mode": task_spec.scattering_mode,
        "student_num_steps": int(num_steps),
        "distillation_mode": "direct_teacher_supervision",
        "teacher_train_sampling": args.teacher_train_sampling,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_ode_method": args.teacher_ode_method,
        "teacher_num_steps": int(getattr(teacher, "ode_num_steps", task_spec.teacher_default_num_steps)),
        "teacher_time_grid_kind": str(getattr(teacher, "time_grid_kind", task_spec.teacher_time_grid_kind)),
        "teacher_time_grid_split": float(
            getattr(teacher, "time_grid_split", task_spec.teacher_time_grid_split)
        ),
        "teacher_time_grid_schedule": getattr(teacher, "time_grid_schedule", task_spec.teacher_time_grid_schedule),
        "support_samples": int(args.support_samples),
        "support_sampling": args.support_sampling,
        "val_ratio": float(args.val_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "runtime_seconds": float(finished_at - started_at),
    }
    save_student_checkpoint(run_dir / "best_student.pth", model, metadata=metadata)
    final_model = FixedStepStudent(
        cond_dim=task_spec.cond_dim,
        target_dim=task_spec.target_dim,
        num_steps=num_steps,
        hidden_dim=args.student_hidden_dim,
        depth=args.student_depth,
        expansion_factor=args.student_expansion_factor,
    )
    final_model.load_state_dict(result.final_state_dict)
    save_student_checkpoint(run_dir / "final_student.pth", final_model, metadata={**metadata, "checkpoint": "final"})
    np.savez(
        run_dir / "history.npz",
        train_loss=np.asarray(result.train_history, dtype=np.float64),
        val_loss=np.asarray(result.val_history, dtype=np.float64),
    )
    save_json(
        run_dir / "metrics.json",
        {
            "student_config": asdict(model.config()),
            "best_val_loss": float(result.best_val_loss),
            "final_val_loss": float(result.final_val_loss),
            **metadata,
        },
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    device = str(args.device or setup_environment(seed=args.seed))
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

    for num_steps in args.steps:
        train_step_count(args, task_spec, teacher, num_steps, device)


if __name__ == "__main__":
    main()
