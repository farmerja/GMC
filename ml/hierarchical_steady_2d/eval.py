#!/usr/bin/env python3
"""Evaluate standalone hierarchical steady-2D checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from physics.mc_data_gen import get_2d_hierarchical_steady_scattered_only_data

from .artifacts import save_json
from .data import (
    FACE_TASK,
    JOINT_TASK,
    TRACK_TASK,
    build_face_task_arrays,
    build_joint_task_arrays,
    build_track_task_arrays,
    prepare_hierarchical_scattered_only_dataframe,
)
from .metrics import classification_metrics
from .models import FaceClassifier, GaussianRegressor, HierarchicalMLP, load_checkpoint
from .normalization import HierarchicalNormalizationBundle
from .schema import TASK_NAME, normalize_stage_name


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone hierarchical steady-2D models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization", default=None)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--num-samples", "--num-neutrons", dest="num_samples", type=int, default=8000)
    parser.add_argument("--sampling", default="lhs", choices=["random", "lhs", "qmc"])
    parser.add_argument("--source", "--source-type", dest="source_type", default="internal", choices=["internal", "boundary"])
    parser.add_argument("--sigma-t", type=float, default=1.0)
    parser.add_argument("--min-width", type=float, default=0.01)
    parser.add_argument("--max-width", type=float, default=10.0)
    parser.add_argument("--min-height", type=float, default=0.01)
    parser.add_argument("--max-height", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _load_eval_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset_csv is not None:
        raw_df = pd.read_csv(args.dataset_csv)
    else:
        raw_df = get_2d_hierarchical_steady_scattered_only_data(
            source=args.source_type,
            num_neutrons=int(args.num_samples),
            sigma_t=float(args.sigma_t),
            min_width=float(args.min_width),
            max_width=float(args.max_width),
            min_height=float(args.min_height),
            max_height=float(args.max_height),
            sampling=str(args.sampling),
        )
    return prepare_hierarchical_scattered_only_dataframe(raw_df)


def _task_arrays_for_stage(stage: str, frame: pd.DataFrame, normalization: HierarchicalNormalizationBundle):
    if stage == FACE_TASK:
        return build_face_task_arrays(frame, normalization)
    if stage == JOINT_TASK:
        return build_joint_task_arrays(frame, normalization)
    if stage == TRACK_TASK:
        return build_track_task_arrays(frame, normalization)
    raise ValueError(f"Unsupported hierarchical stage {stage!r}")


def _regression_metrics(
    stage: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    normalization: HierarchicalNormalizationBundle,
) -> dict[str, object]:
    mean_space_mse = float(np.mean((predictions - targets) ** 2))
    if stage == JOINT_TASK:
        pred_raw = normalization.denormalize_joint_targets(predictions)
        target_raw = normalization.denormalize_joint_targets(targets)
        abs_err = np.abs(pred_raw - target_raw)
        return {
            "mean_space_mse": mean_space_mse,
            "mae": float(np.mean(abs_err)),
            "mae_by_dim": [float(v) for v in np.mean(abs_err, axis=0).tolist()],
        }

    pred_log = normalization.denormalize_track_targets(predictions)
    target_log = normalization.denormalize_track_targets(targets)
    pred_raw = np.power(10.0, pred_log)
    target_raw = np.power(10.0, target_log)
    return {
        "mean_space_mse": mean_space_mse,
        "mae_total_track_length": float(np.mean(np.abs(pred_raw - target_raw))),
        "mae_total_track_length_log10": float(np.mean(np.abs(pred_log - target_log))),
    }


def _resolve_device(device: str | None) -> str:
    if device:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_normalization_path(checkpoint_path: Path, explicit_path: str | None, metadata: dict[str, object]) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    metadata_path = metadata.get("normalization_path")
    if metadata_path:
        candidates.append(Path(str(metadata_path)))
    candidates.append(checkpoint_path.parents[2] / "metadata" / "normalization.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError("Could not resolve normalization.json for hierarchical steady-2D evaluation.")


def evaluate_hierarchical_checkpoint(
    checkpoint: str | Path,
    *,
    normalization: str | None = None,
    dataframe: pd.DataFrame | None = None,
    dataset_csv: str | None = None,
    num_samples: int = 8000,
    sampling: str = "lhs",
    source_type: str = "internal",
    sigma_t: float = 1.0,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    batch_size: int = 512,
    device: str | None = None,
    output: str | Path | None = None,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint)
    resolved_device = _resolve_device(device)
    model, _config, metadata = load_checkpoint(checkpoint_path, map_location=resolved_device)
    if not isinstance(model, (FaceClassifier, GaussianRegressor, HierarchicalMLP)):
        raise ValueError(
            "This evaluator expects standalone hierarchical steady-2D checkpoints "
            f"saved by the package trainer. Got {type(model).__name__}."
        )

    stage = normalize_stage_name(metadata.get("task", metadata.get("stage", "")))
    training_config = metadata.get("training_config", {})
    trained_source = None
    if isinstance(training_config, dict):
        trained_source = training_config.get("source")
    if dataframe is None and dataset_csv is None and trained_source is not None and str(trained_source) != str(source_type):
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained for source={trained_source!r}, "
            f"not source_type={source_type!r}."
        )
    norm_path = _resolve_normalization_path(checkpoint_path, normalization, metadata)
    normalization_bundle = HierarchicalNormalizationBundle.load(norm_path)

    if dataframe is not None:
        frame = prepare_hierarchical_scattered_only_dataframe(dataframe.copy())
    else:
        eval_args = argparse.Namespace(
            dataset_csv=dataset_csv,
            source_type=source_type,
            num_samples=num_samples,
            sigma_t=sigma_t,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            sampling=sampling,
        )
        frame = _load_eval_frame(eval_args)

    task_data = _task_arrays_for_stage(stage, frame, normalization_bundle)
    target_kind = str(metadata.get("target_kind", task_data.target_kind))

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(task_data.inputs, dtype=np.float32, copy=True)),
            torch.from_numpy(
                np.array(
                    task_data.targets,
                    dtype=np.int64 if target_kind == "classification" else np.float32,
                    copy=True,
                )
            ),
        ),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )

    model = model.to(resolved_device)
    model.eval()

    if target_kind == "classification":
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits_all.append(model(batch_x.to(resolved_device)).cpu())
                labels_all.append(batch_y.cpu())
        metrics = classification_metrics(torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0))
    else:
        pred_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        std_batches: list[np.ndarray] = []
        nll_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(resolved_device)
                batch_y = batch_y.to(resolved_device)
                if isinstance(model, GaussianRegressor):
                    loss, mean, log_std = model.nll(batch_x, batch_y)
                    pred_batches.append(mean.cpu().numpy())
                    std_batches.append(torch.exp(log_std).cpu().numpy())
                    nll_losses.append(float(loss.item()) * int(batch_x.shape[0]))
                else:
                    pred_batches.append(model(batch_x).cpu().numpy())
                target_batches.append(batch_y.cpu().numpy())
        metrics = _regression_metrics(
            stage,
            np.concatenate(pred_batches, axis=0),
            np.concatenate(target_batches, axis=0),
            normalization_bundle,
        )
        if nll_losses:
            gaussian_nll = float(np.sum(nll_losses) / max(1, len(frame)))
            metrics["loss"] = gaussian_nll
            metrics["gaussian_nll"] = gaussian_nll
        else:
            metrics["loss"] = float(metrics["mean_space_mse"])
        if std_batches:
            avg_pred_std = np.mean(np.concatenate(std_batches, axis=0), axis=0)
            names = ("exit_face_coord_local_norm", "final_dir_u", "final_dir_v") if stage == JOINT_TASK else ("total_track_length_log10",)
            metrics["avg_pred_std_by_dim"] = {name: float(value) for name, value in zip(names, avg_pred_std)}

    output_path = Path(output) if output else checkpoint_path.with_name("eval_metrics.json")
    payload: dict[str, object] = {
        "task_name": TASK_NAME,
        "stage": stage,
        "checkpoint": str(checkpoint_path),
        "normalization_path": str(norm_path),
        "metrics": metrics,
        "num_samples": int(len(frame)),
        "source_type": source_type,
        "sampling": sampling,
        "sigma_t": float(sigma_t),
        "conditioning_mode": metadata.get("conditioning_mode", "unknown"),
        "target_kind": target_kind,
        "model_type": type(model).__name__,
        "trained_source": trained_source,
        "stochastic_output": bool(metadata.get("stochastic_output", False)),
    }
    save_json(output_path, payload)
    return payload


def main(argv=None) -> None:
    args = parse_args(argv)
    payload = evaluate_hierarchical_checkpoint(
        args.checkpoint,
        normalization=args.normalization,
        dataset_csv=args.dataset_csv,
        num_samples=args.num_samples,
        sampling=args.sampling,
        source_type=args.source_type,
        sigma_t=args.sigma_t,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        batch_size=args.batch_size,
        device=args.device,
        output=args.output,
    )
    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_name("eval_metrics.json")
    print(f"Saved evaluation metrics to {output_path} for stage {payload['stage']}")


if __name__ == "__main__":
    main()
