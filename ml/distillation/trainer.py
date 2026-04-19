from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def split_condition_tensor(
    conditions: torch.Tensor,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if conditions.ndim != 2:
        raise ValueError(f"Expected [num_samples, cond_dim], got {tuple(conditions.shape)}")
    num_samples = int(conditions.shape[0])
    val_count = max(1, int(round(num_samples * float(val_ratio))))
    if val_count >= num_samples:
        val_count = max(1, num_samples - 1)
    generator = torch.Generator(device=conditions.device if conditions.is_cuda else "cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator, device=conditions.device)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return conditions[train_idx], conditions[val_idx]


def sample_standard_noise(
    num_samples: int,
    target_dim: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((int(num_samples), int(target_dim)), generator=generator, dtype=torch.float32, device=device)


def _as_loader(
    conditions: torch.Tensor,
    noise: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = (conditions, noise) if targets is None else (conditions, noise, targets)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
    )


def prediction_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, object]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    error = pred - target
    target_std = target.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    pred_std = pred.std(dim=0, unbiased=False)
    return {
        "loss": float(error.pow(2).mean().item()),
        "mae": float(error.abs().mean().item()),
        "mean_bias_by_dim": [float(v) for v in error.mean(dim=0).tolist()],
        "std_ratio_by_dim": [float(v) for v in (pred_std / target_std).tolist()],
    }


@torch.inference_mode()
def sample_student_predictions(
    model: torch.nn.Module,
    conditions: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    model = model.to(device)
    model.eval()
    outputs: List[torch.Tensor] = []
    loader = _as_loader(conditions, noise, batch_size=batch_size, shuffle=False)
    for cond_batch, noise_batch in loader:
        outputs.append(model(cond_batch.to(device), noise_batch.to(device)).detach().cpu())
    if not outputs:
        target_dim = getattr(model, "target_dim", noise.shape[1] if noise.ndim == 2 else 0)
        return torch.empty((0, int(target_dim)), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class TrainStudentResult:
    best_val_loss: float
    final_val_loss: float
    train_history: List[float]
    val_history: List[float]
    best_state_dict: Dict[str, torch.Tensor]
    final_state_dict: Dict[str, torch.Tensor]


def train_student_model(
    model: torch.nn.Module,
    train_conditions: torch.Tensor,
    train_noise: torch.Tensor,
    train_targets: torch.Tensor,
    val_conditions: torch.Tensor,
    val_noise: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    device: str,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
) -> TrainStudentResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    train_loader = _as_loader(train_conditions, train_noise, train_targets, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_history: List[float] = []
    val_history: List[float] = []

    for _epoch in range(int(epochs)):
        model.train()
        epoch_losses = []
        for cond_batch, noise_batch, target_batch in train_loader:
            cond_batch = cond_batch.to(device)
            noise_batch = noise_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(cond_batch, noise_batch)
            loss = (pred - target_batch).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        train_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        metrics = evaluate_student(
            model,
            val_conditions,
            val_noise,
            val_targets,
            device=device,
            batch_size=batch_size,
        )
        val_history.append(float(metrics["loss"]))
        if metrics["loss"] < best_val:
            best_val = float(metrics["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_metrics = evaluate_student(
        model,
        val_conditions,
        val_noise,
        val_targets,
        device=device,
        batch_size=batch_size,
    )
    return TrainStudentResult(
        best_val_loss=float(best_val),
        final_val_loss=float(final_metrics["loss"]),
        train_history=train_history,
        val_history=val_history,
        best_state_dict=best_state,
        final_state_dict=final_state,
    )


@torch.inference_mode()
def evaluate_student(
    model: torch.nn.Module,
    conditions: torch.Tensor,
    noise: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    batch_size: int = 256,
) -> Dict[str, object]:
    pred = sample_student_predictions(
        model,
        conditions,
        noise,
        device=device,
        batch_size=batch_size,
    )
    return prediction_metrics(pred, targets)


def _synchronize_device(device: str) -> None:
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_str == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_sampler(
    sample_fn,
    conditions: torch.Tensor,
    noise: torch.Tensor,
    *,
    batch_size: int,
    repeats: int = 3,
    device: str = "cpu",
) -> Dict[str, float]:
    loader = list(_as_loader(conditions, noise, batch_size=batch_size, shuffle=False))
    if not loader:
        return {"runtime_seconds": 0.0, "samples_per_second": 0.0}

    sample_fn(loader[0][0], loader[0][1])
    _synchronize_device(device)

    started_at = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        for cond_batch, noise_batch in loader:
            sample_fn(cond_batch, noise_batch)
        _synchronize_device(device)
    elapsed = time.perf_counter() - started_at
    avg_runtime = elapsed / float(max(1, int(repeats)))
    total_samples = float(sum(int(batch[0].shape[0]) for batch in loader))
    samples_per_second = total_samples / avg_runtime if avg_runtime > 0.0 else float("inf")
    return {
        "runtime_seconds": float(avg_runtime),
        "samples_per_second": float(samples_per_second),
    }


def condition_slice_metrics(
    conditions: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_features: int = 2,
    num_bins: int = 4,
) -> Dict[str, List[Dict[str, float]]]:
    if conditions.shape[0] == 0:
        return {}
    cond_cpu = conditions.detach().cpu()
    err_cpu = (pred - target).abs().mean(dim=1).detach().cpu()
    slice_metrics: Dict[str, List[Dict[str, float]]] = {}
    for feat_idx in range(min(int(max_features), int(cond_cpu.shape[1]))):
        values = cond_cpu[:, feat_idx].numpy()
        edges = np.quantile(values, np.linspace(0.0, 1.0, int(num_bins) + 1))
        feat_slices: List[Dict[str, float]] = []
        for bin_idx in range(int(num_bins)):
            lo = float(edges[bin_idx])
            hi = float(edges[bin_idx + 1])
            if bin_idx == int(num_bins) - 1:
                mask = (values >= lo) & (values <= hi)
            else:
                mask = (values >= lo) & (values < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            feat_slices.append(
                {
                    "lower": lo,
                    "upper": hi,
                    "count": count,
                    "mae": float(err_cpu[mask].mean().item()),
                }
            )
        slice_metrics[f"cond_{feat_idx}"] = feat_slices
    return slice_metrics
