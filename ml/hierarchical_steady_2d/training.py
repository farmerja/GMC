"""Standalone training workflow for hierarchical steady-2D tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from physics.mc_data_gen import get_2d_hierarchical_steady_scattered_only_data

from .data import (
    FACE_TASK,
    JOINT_TARGET_COLUMNS,
    JOINT_TASK,
    TRACK_TASK,
    HierarchicalTaskArrays,
    build_face_task_arrays,
    build_joint_task_arrays,
    build_track_task_arrays,
    prepare_hierarchical_scattered_only_dataframe,
)
from .models import FaceClassifier, GaussianRegressor, save_checkpoint
from .normalization import HierarchicalNormalizationBundle, fit_hierarchical_normalization
from .schema import normalize_stage_name


DEFAULT_OUTPUT_ROOT = Path("ml/hierarchical_steady_2d/runs")
DEFAULT_TASKS = (FACE_TASK, JOINT_TASK, TRACK_TASK)


@dataclass(frozen=True)
class HierarchicalTrainingConfig:
    source: str = "internal"
    num_neutrons: int = 100000
    sigma_t: float = 1.0
    distribution: int = 0
    min_width: float = 0.01
    max_width: float = 10.0
    min_height: float = 0.01
    max_height: float = 10.0
    sampling: str = "random"
    tasks: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TASKS)
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    val_ratio: float = 0.1
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0
    seed: int = 0
    device: str = "auto"
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str | None = None


def _normalize_task_list(tasks: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not tasks:
        raise ValueError("At least one hierarchical task must be requested.")

    normalized: list[str] = []
    for task in tasks:
        task_name = normalize_stage_name(task)
        if task_name == "all":
            for item in DEFAULT_TASKS:
                if item not in normalized:
                    normalized.append(item)
            continue
        if task_name not in DEFAULT_TASKS:
            raise ValueError(f"Unsupported hierarchical task {task!r}. Use one of {DEFAULT_TASKS} or 'all'.")
        if task_name not in normalized:
            normalized.append(task_name)
    return tuple(normalized)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_dataframe(df: pd.DataFrame, *, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1); got {val_ratio}")
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to create train/validation splits.")

    rng = np.random.default_rng(seed)
    if "exit_face" in df.columns:
        val_indices: list[int] = []
        for _, group in df.groupby("exit_face", sort=True):
            group_indices = group.index.to_numpy(dtype=np.int64, copy=True)
            if group_indices.size <= 1:
                continue
            rng.shuffle(group_indices)
            val_count = max(1, int(round(group_indices.size * val_ratio)))
            val_count = min(val_count, group_indices.size - 1)
            val_indices.extend(group_indices[:val_count].tolist())

        if val_indices:
            val_idx = np.array(sorted(set(val_indices)), dtype=np.int64)
            train_df = df.drop(index=val_idx).reset_index(drop=True)
            val_df = df.loc[val_idx].reset_index(drop=True)
            if not train_df.empty and not val_df.empty:
                return train_df, val_df

    indices = rng.permutation(len(df))
    val_count = int(round(len(df) * val_ratio))
    val_count = min(max(val_count, 1), len(df) - 1)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _task_builder(
    task: str,
    df: pd.DataFrame,
    normalization: HierarchicalNormalizationBundle,
) -> HierarchicalTaskArrays:
    if task == FACE_TASK:
        return build_face_task_arrays(df, normalization)
    if task == JOINT_TASK:
        return build_joint_task_arrays(df, normalization)
    if task == TRACK_TASK:
        return build_track_task_arrays(df, normalization)
    raise ValueError(f"Unknown hierarchical task: {task}")


def _make_loader(task_data: HierarchicalTaskArrays, *, batch_size: int, shuffle: bool) -> DataLoader:
    inputs = torch.from_numpy(np.array(task_data.inputs, dtype=np.float32, copy=True))
    if task_data.target_kind == "classification":
        targets = torch.from_numpy(np.array(task_data.targets, dtype=np.int64, copy=True))
    else:
        targets = torch.from_numpy(np.array(task_data.targets, dtype=np.float32, copy=True))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    pred = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred == targets))
    metrics: dict[str, float] = {"accuracy": accuracy}
    for face_id in range(4):
        mask = targets == face_id
        if np.any(mask):
            metrics[f"accuracy_face_{face_id}"] = float(np.mean(pred[mask] == targets[mask]))
    return metrics


def _regression_metrics(
    task: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    normalization: HierarchicalNormalizationBundle,
) -> dict[str, float]:
    if task == JOINT_TASK:
        pred_raw = normalization.denormalize_joint_targets(predictions)
        target_raw = normalization.denormalize_joint_targets(targets)
        mae = np.mean(np.abs(pred_raw - target_raw), axis=0)
        return {
            f"mae_{name}": float(value)
            for name, value in zip(JOINT_TARGET_COLUMNS, mae)
        }

    pred_log = normalization.denormalize_track_targets(predictions)
    target_log = normalization.denormalize_track_targets(targets)
    pred_raw = np.power(10.0, pred_log)
    target_raw = np.power(10.0, target_log)
    mae_raw = float(np.mean(np.abs(pred_raw - target_raw)))
    mae_log = float(np.mean(np.abs(pred_log - target_log)))
    return {
        "mae_total_track_length": mae_raw,
        "mae_total_track_length_log10": mae_log,
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    task: str,
    target_kind: str,
    loss_fn: nn.Module | None,
    device: torch.device,
    normalization: HierarchicalNormalizationBundle,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    num_samples = 0
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    std_batches: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if target_kind == "classification":
                outputs = model(inputs)
                if loss_fn is None:
                    raise ValueError("Classification evaluation requires a loss function.")
                loss = loss_fn(outputs, targets)
                pred_batches.append(outputs.detach().cpu().numpy())
            else:
                if isinstance(model, GaussianRegressor):
                    loss, mean, log_std = model.nll(inputs, targets)
                    pred_batches.append(mean.detach().cpu().numpy())
                    std_batches.append(torch.exp(log_std).detach().cpu().numpy())
                else:
                    outputs = model(inputs)
                    if loss_fn is None:
                        raise ValueError("Deterministic regression evaluation requires a loss function.")
                    loss = loss_fn(outputs, targets)
                    pred_batches.append(outputs.detach().cpu().numpy())
            batch_size = inputs.shape[0]
            total_loss += float(loss.item()) * batch_size
            num_samples += batch_size
            target_batches.append(targets.detach().cpu().numpy())

    mean_loss = total_loss / max(1, num_samples)
    predictions = np.concatenate(pred_batches, axis=0)
    targets_np = np.concatenate(target_batches, axis=0)
    if target_kind == "classification":
        metrics = _classification_metrics(predictions, targets_np)
    else:
        metrics = _regression_metrics(task, predictions, targets_np, normalization)
        if isinstance(model, GaussianRegressor):
            metrics["gaussian_nll"] = float(mean_loss)
        if std_batches:
            avg_pred_std = np.mean(np.concatenate(std_batches, axis=0), axis=0)
            metrics.update(
                {
                    f"avg_pred_std_{name}": float(value)
                    for name, value in zip(
                        JOINT_TARGET_COLUMNS if task == JOINT_TASK else ("total_track_length_log10",),
                        avg_pred_std,
                    )
                }
            )
    metrics["loss"] = float(mean_loss)
    return float(mean_loss), metrics


def _create_run_dir(config: HierarchicalTrainingConfig) -> Path:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if config.run_name:
        run_name = config.run_name
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}_{config.source}_n{config.num_neutrons}_seed{config.seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_dataset_metadata(run_dir: Path, prepared_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    metadata = {
        "num_rows_total": int(len(prepared_df)),
        "num_rows_train": int(len(train_df)),
        "num_rows_val": int(len(val_df)),
        "columns": list(prepared_df.columns),
        "exit_face_counts_total": {
            str(face): int(count)
            for face, count in prepared_df["exit_face"].value_counts().sort_index().items()
        },
        "exit_face_counts_train": {
            str(face): int(count)
            for face, count in train_df["exit_face"].value_counts().sort_index().items()
        },
        "exit_face_counts_val": {
            str(face): int(count)
            for face, count in val_df["exit_face"].value_counts().sort_index().items()
        },
    }
    _write_json(run_dir / "metadata" / "dataset_summary.json", metadata)


def _conditioning_mode(task: str) -> str:
    if task == FACE_TASK:
        return "base_only"
    if task == JOINT_TASK:
        return "teacher_forced_face"
    if task == TRACK_TASK:
        return "teacher_forced_face_and_joint"
    return "unknown"


def _train_single_task(
    *,
    task: str,
    config: HierarchicalTrainingConfig,
    train_data: HierarchicalTaskArrays,
    val_data: HierarchicalTaskArrays,
    normalization: HierarchicalNormalizationBundle,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    task_dir = run_dir / task
    models_dir = task_dir / "models"
    metrics_dir = task_dir / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if train_data.target_kind == "classification":
        model = FaceClassifier(
            input_dim=int(train_data.inputs.shape[1]),
            num_classes=4,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            dropout=config.dropout,
        ).to(device)
        loss_fn: nn.Module | None = nn.CrossEntropyLoss()
    else:
        model = GaussianRegressor(
            input_dim=int(train_data.inputs.shape[1]),
            target_dim=int(train_data.targets.shape[1]),
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            dropout=config.dropout,
        ).to(device)
        loss_fn = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = _make_loader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = _make_loader(val_data, batch_size=config.batch_size, shuffle=False)
    normalization_json_path = run_dir / "metadata" / "normalization.json"

    best_val_loss = float("inf")
    best_metrics: dict[str, Any] | None = None
    train_epoch_losses: list[float] = []
    val_epoch_losses: list[float] = []

    for epoch in range(config.epochs):
        model.train()
        total_epoch_loss = 0.0
        num_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if train_data.target_kind == "classification":
                outputs = model(inputs)
                if loss_fn is None:
                    raise ValueError("Classification training requires a loss function.")
                loss = loss_fn(outputs, targets)
            else:
                if not isinstance(model, GaussianRegressor):
                    raise ValueError("Regression stages must use GaussianRegressor.")
                loss, _mean, _log_std = model.nll(inputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            batch_loss = float(loss.item())
            total_epoch_loss += batch_loss * batch_size
            num_samples += batch_size

        train_epoch_loss = total_epoch_loss / max(1, num_samples)
        train_epoch_losses.append(float(train_epoch_loss))

        val_loss, val_metrics = _evaluate_model(
            model,
            val_loader,
            task=task,
            target_kind=train_data.target_kind,
            loss_fn=loss_fn,
            device=device,
            normalization=normalization,
        )
        val_epoch_losses.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_metrics = {"epoch": epoch + 1, **val_metrics}
            save_checkpoint(
                models_dir / "best_model.pt",
                model,
                metadata={
                    "task": task,
                    "stage": task,
                    "input_names": list(train_data.input_names),
                    "target_names": list(train_data.target_names),
                    "target_kind": train_data.target_kind,
                    "conditioning_mode": _conditioning_mode(task),
                    "stochastic_output": bool(train_data.target_kind != "classification"),
                    "training_config": asdict(config),
                    "normalization_path": str(normalization_json_path),
                },
            )

    save_checkpoint(
        models_dir / "final_model.pt",
        model,
        metadata={
            "task": task,
            "stage": task,
            "input_names": list(train_data.input_names),
            "target_names": list(train_data.target_names),
            "target_kind": train_data.target_kind,
            "conditioning_mode": _conditioning_mode(task),
            "stochastic_output": bool(train_data.target_kind != "classification"),
            "training_config": asdict(config),
            "normalization_path": str(normalization_json_path),
        },
    )

    final_val_loss, final_metrics = _evaluate_model(
        model,
        val_loader,
        task=task,
        target_kind=train_data.target_kind,
        loss_fn=loss_fn,
        device=device,
        normalization=normalization,
    )

    task_metadata = {
        "task": task,
        "input_dim": int(train_data.inputs.shape[1]),
        "output_dim": int(4 if train_data.target_kind == "classification" else train_data.targets.shape[1]),
        "input_names": list(train_data.input_names),
        "target_names": list(train_data.target_names),
        "target_kind": train_data.target_kind,
        "model_family": "categorical_classifier" if train_data.target_kind == "classification" else "diagonal_gaussian",
        "num_train": int(train_data.inputs.shape[0]),
        "num_val": int(val_data.inputs.shape[0]),
        "conditioning_mode": _conditioning_mode(task),
    }
    _write_json(metrics_dir / "task_metadata.json", task_metadata)

    np.savez(
        metrics_dir / "history.npz",
        train_epoch_losses=np.asarray(train_epoch_losses, dtype=np.float32),
        val_epoch_losses=np.asarray(val_epoch_losses, dtype=np.float32),
    )

    metrics_payload = {
        "task": task,
        "best": best_metrics or {},
        "final": final_metrics,
        "final_val_loss": float(final_val_loss),
        "best_val_loss": float(best_val_loss),
    }
    _write_json(metrics_dir / "metrics.json", metrics_payload)
    return metrics_payload


def run_hierarchical_training(
    config: HierarchicalTrainingConfig,
    *,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run the standalone hierarchical steady-2D training workflow.

    The exact no-scatter gate remains analytic in the physics layer. These
    models are trained only on the scattered branch conditioned on at least one
    collision before the ballistic exit.
    """
    tasks = _normalize_task_list(config.tasks)
    device = _resolve_device(config.device)
    _set_seed(config.seed)

    if dataframe is None:
        raw_df = get_2d_hierarchical_steady_scattered_only_data(
            source=config.source,
            num_neutrons=config.num_neutrons,
            sigma_t=config.sigma_t,
            distribution=config.distribution,
            min_width=config.min_width,
            max_width=config.max_width,
            min_height=config.min_height,
            max_height=config.max_height,
            sampling=config.sampling,
        )
    else:
        raw_df = dataframe.copy()

    prepared_df = prepare_hierarchical_scattered_only_dataframe(raw_df)
    train_df, val_df = _split_dataframe(prepared_df, val_ratio=config.val_ratio, seed=config.seed)
    normalization = fit_hierarchical_normalization(train_df, source=config.source, sigma_t=config.sigma_t)

    run_dir = _create_run_dir(config)
    _write_json(run_dir / "metadata" / "run_config.json", {**asdict(config), "tasks": list(tasks), "device": str(device)})
    normalization.save(run_dir / "metadata")
    _save_dataset_metadata(run_dir, prepared_df, train_df, val_df)

    results: dict[str, Any] = {"run_dir": str(run_dir), "tasks": {}}
    for task in tasks:
        train_task = _task_builder(task, train_df, normalization)
        val_task = _task_builder(task, val_df, normalization)
        results["tasks"][task] = _train_single_task(
            task=task,
            config=config,
            train_data=train_task,
            val_data=val_task,
            normalization=normalization,
            run_dir=run_dir,
            device=device,
        )

    _write_json(run_dir / "metadata" / "results.json", results)
    return results
