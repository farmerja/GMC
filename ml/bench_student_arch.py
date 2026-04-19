#!/usr/bin/env python3
"""Benchmark student architecture variants: accuracy vs inference speed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.distillation.artifacts import resolve_student_run_dir
from ml.distillation.student import load_student_checkpoint
from ml.distillation.tasks import resolve_distillation_task
from ml.distillation.trainer import sample_standard_noise, split_condition_tensor
from ml.distillation.teacher import FlowTeacherSampler
from ml.utils import setup_environment

WARMUP_ITERS = 20
BENCH_ITERS = 200
BATCH_SIZE = 4096


def bench_model(model, cond, noise, device: str) -> float:
    model.eval()
    cond = cond.to(device)
    noise = noise.to(device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(cond, noise)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            model(cond, noise)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / BENCH_ITERS * 1000  # ms per call


def main():
    parser = argparse.ArgumentParser(description="Benchmark student arch variants")
    parser.add_argument("--task", default="steady2d")
    parser.add_argument("--source-type", default="boundary", choices=["boundary", "internal"])
    parser.add_argument("--artifact-roots", nargs="+",
                        default=["ml/student_artifacts", "ml/student_artifacts_sweep"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--scattering-mode", default="isotropic")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-time-schedule", default="steady2d_default")
    parser.add_argument("--support-samples", type=int, default=8192)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or setup_environment()
    print(f"Device: {device}\n")

    task_spec = resolve_distillation_task(
        args.task, source_type=args.source_type, scattering_mode=args.scattering_mode
    )

    # Build validation set from teacher
    teacher_ckpt = args.teacher_checkpoint or f"ml/models/2d_{args.source_type}/best_model.pth"
    teacher = FlowTeacherSampler.from_task(
        task_spec,
        checkpoint_path=teacher_ckpt,
        device=device,
        time_grid_schedule=args.teacher_time_schedule,
    )
    all_cond = task_spec.sample_conditions(args.support_samples, sampling="lhs", seed=99, device="cpu")
    _, val_cond = split_condition_tensor(all_cond, val_ratio=0.5, seed=99)
    val_noise = sample_standard_noise(val_cond.shape[0], task_spec.target_dim, seed=100, device="cpu")
    with torch.no_grad():
        val_targets = teacher.sample_target(val_cond, noise=val_noise)

    bench_cond = val_cond[:BATCH_SIZE].to(device)
    bench_noise = val_noise[:BATCH_SIZE].to(device)
    val_cond_d = val_cond.to(device)
    val_noise_d = val_noise.to(device)
    val_targets_d = val_targets.to(device)

    # Collect all checkpoints: from artifact_roots (by hidden_dim subdirs in sweep)
    checkpoints = []
    for root in args.artifact_roots:
        root_path = Path(root)
        # Check for hidden_dim subdirs (sweep layout: <root>/h<hidden>/<task>/...)
        sweep_dirs = sorted(root_path.glob("h*"))
        if sweep_dirs:
            for sd in sweep_dirs:
                run_dir = resolve_student_run_dir(
                    args.task, args.source_type, args.steps,
                    scattering_mode=args.scattering_mode, artifact_root=str(sd),
                )
                ckpt = run_dir / "best_student.pth"
                if ckpt.exists():
                    checkpoints.append(ckpt)
        else:
            run_dir = resolve_student_run_dir(
                args.task, args.source_type, args.steps,
                scattering_mode=args.scattering_mode, artifact_root=root,
            )
            ckpt = run_dir / "best_student.pth"
            if ckpt.exists():
                checkpoints.append(ckpt)

    if not checkpoints:
        print("No checkpoints found.")
        return

    results = []
    for ckpt in checkpoints:
        metrics_path = ckpt.parent / "metrics.json"
        model, cfg, _ = load_student_checkpoint(ckpt, map_location=device)
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        train_val_loss = json.loads(metrics_path.read_text())["best_val_loss"] if metrics_path.exists() else float("nan")

        with torch.no_grad():
            pred = model(val_cond_d, val_noise_d)
        rel_l2 = (pred - val_targets_d).norm() / val_targets_d.norm()

        ms_per_call = bench_model(model, bench_cond, bench_noise, device)

        results.append({
            "hidden_dim": cfg.hidden_dim,
            "depth": cfg.depth,
            "num_steps": cfg.num_steps,
            "params": num_params,
            "train_val_loss": train_val_loss,
            "rel_l2_vs_teacher": rel_l2.item(),
            "ms_per_batch": ms_per_call,
        })
        print(
            f"  hidden={cfg.hidden_dim:3d} depth={cfg.depth} params={num_params:>9,} "
            f"train_val_loss={train_val_loss:.4f}  rel_l2={rel_l2.item():.4f}  "
            f"{ms_per_call:.3f} ms/{BATCH_SIZE} samples"
        )

    # Teacher timing via sampler (wraps ODE solve + model)
    def _teacher_call(cond, noise):
        with torch.no_grad():
            return teacher.sample_target(cond, noise=noise)

    class _TeacherWrapper(torch.nn.Module):
        def forward(self, cond, noise):
            return _teacher_call(cond, noise)

    ms_teacher = bench_model(_TeacherWrapper(), bench_cond, bench_noise, device)
    teacher_params = sum(p.numel() for p in teacher.model.parameters())
    print(f"\n  teacher  params={teacher_params:>9,}  "
          f"{ms_teacher:.3f} ms/{BATCH_SIZE} samples  (6 ODE steps)")
    if results:
        baseline = next((r for r in results if r["hidden_dim"] == 128), results[0])
        print(f"  speedup hidden=128 vs teacher: {ms_teacher / baseline['ms_per_batch']:.1f}x")


if __name__ == "__main__":
    main()
