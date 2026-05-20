#!/usr/bin/env python3
"""Retrain the steady 2D boundary/internal CFM models with a shared configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    Path("ml/models/2d_boundary/best_model.pth"),
    Path("ml/models/2d_boundary/trained_model_final.pth"),
    Path("ml/models/2d_internal/best_model.pth"),
    Path("ml/models/2d_internal/trained_model_final.pth"),
)
NORM_FILES = (
    Path("ml/data/norm_params_2d_boundary.npz"),
    Path("ml/data/norm_params_2d_internal.npz"),
)
TRAINING_DIRS = (
    Path("ml/training_results/2d_boundary"),
    Path("ml/training_results/2d_internal"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-neutrons", type=int, default=500_000, help="MC samples per model.")
    parser.add_argument("--num-epochs", type=int, default=50, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--min-width", type=float, default=0.01, help="Minimum cell width.")
    parser.add_argument("--max-width", type=float, default=2.0, help="Maximum cell width.")
    parser.add_argument(
        "--min-height",
        type=float,
        default=None,
        help="Minimum cell height. Defaults to --min-width.",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="Maximum cell height. Defaults to --max-width.",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak OneCycle learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=160, help="Hidden dimension.")
    parser.add_argument("--depth", type=int, default=5, help="Flow block depth.")
    parser.add_argument("--expansion-factor", type=int, default=1, help="FFN expansion factor.")
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "qmc", "lhs"])
    parser.add_argument(
        "--train-workers",
        type=int,
        default=0,
        help="DataLoader workers for training. Defaults to 0 for safer local retrains.",
    )
    parser.add_argument(
        "--val-workers",
        type=int,
        default=0,
        help="DataLoader workers for validation. Defaults to 0 for safer local retrains.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("ml/backups"),
        help="Root directory for pre-retrain backups.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Do not snapshot current models/norm params/training histories before retraining.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and backup plan without launching training.",
    )
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def restore_current_artifacts(backup_dir: Path | None) -> None:
    if backup_dir is None or not backup_dir.exists():
        return
    for rel_path in MODEL_FILES + NORM_FILES:
        copy_if_exists(backup_dir / rel_path, REPO_ROOT / rel_path)
    for rel_path in TRAINING_DIRS:
        src = backup_dir / rel_path
        dst = REPO_ROOT / rel_path
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def planned_backup_dir(args: argparse.Namespace) -> Path | None:
    if args.skip_backup:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / args.backup_root / f"steady_2d_pre_retrain_{timestamp}").resolve()


def backup_current_artifacts(args: argparse.Namespace, backup_dir: Path | None) -> Path | None:
    if backup_dir is None:
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in MODEL_FILES + NORM_FILES + TRAINING_DIRS:
        copy_if_exists(REPO_ROOT / rel_path, backup_dir / rel_path)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "models": [str(path) for path in MODEL_FILES],
        "norm_files": [str(path) for path in NORM_FILES],
        "training_dirs": [str(path) for path in TRAINING_DIRS],
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return backup_dir


def training_command(args: argparse.Namespace, source_type: str) -> list[str]:
    min_height = args.min_width if args.min_height is None else args.min_height
    max_height = args.max_width if args.max_height is None else args.max_height
    cmd = [
        sys.executable,
        "ml/train.py",
        "--data_dim",
        "2d",
        "--source_type",
        source_type,
        "--num_neutrons",
        str(args.num_neutrons),
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--min_width",
        str(args.min_width),
        "--max_width",
        str(args.max_width),
        "--min_height",
        str(min_height),
        "--max_height",
        str(max_height),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--hidden_dim",
        str(args.hidden_dim),
        "--depth",
        str(args.depth),
        "--expansion_factor",
        str(args.expansion_factor),
        "--sampling",
        args.sampling,
    ]
    if args.train_workers is not None:
        cmd.extend(["--train_workers", str(args.train_workers)])
    if args.val_workers is not None:
        cmd.extend(["--val_workers", str(args.val_workers)])
    return cmd


def main() -> int:
    args = parse_args()
    min_height = args.min_width if args.min_height is None else args.min_height
    max_height = args.max_width if args.max_height is None else args.max_height
    if min_height <= 0.0 or max_height <= 0.0 or args.min_width <= 0.0 or args.max_width <= 0.0:
        raise ValueError("Width and height bounds must be strictly positive.")
    if args.min_width >= args.max_width:
        raise ValueError("--min-width must be less than --max-width.")
    if min_height >= max_height:
        raise ValueError("The effective min-height must be less than max-height.")

    backup_dir = planned_backup_dir(args)
    commands = {
        "boundary": training_command(args, "boundary"),
        "internal": training_command(args, "internal"),
    }
    plan = {
        "num_neutrons": args.num_neutrons,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "width_range": [args.min_width, args.max_width],
        "height_range": [min_height, max_height],
        "sampling": args.sampling,
        "backup_dir": None if backup_dir is None else str(backup_dir),
        "commands": commands,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    backup_current_artifacts(args, backup_dir)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str((REPO_ROOT / "validation/results/.matplotlib").resolve()))
    for source_type in ("boundary", "internal"):
        print(f"\n=== Retraining {source_type} model ===")
        cmd = commands[source_type]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
        except subprocess.CalledProcessError:
            restore_current_artifacts(backup_dir)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
