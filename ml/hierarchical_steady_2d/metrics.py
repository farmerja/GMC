from __future__ import annotations

import numpy as np
import torch

from .schema import FACE_ORDER, FACE_LABELS


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    pred = logits.argmax(dim=1)
    correct = pred.eq(labels)
    per_class = {}
    for face in FACE_ORDER:
        mask = labels == int(face)
        count = int(mask.sum().item())
        if count == 0:
            per_class[FACE_LABELS[int(face)]] = None
        else:
            per_class[FACE_LABELS[int(face)]] = float(correct[mask].float().mean().item())
    return {
        "loss": float(torch.nn.functional.cross_entropy(logits, labels).item()),
        "accuracy": float(correct.float().mean().item()),
        "per_class_accuracy": per_class,
        "class_counts": {
            FACE_LABELS[int(face)]: int((labels == int(face)).sum().item())
            for face in FACE_ORDER
        },
    }


def regression_metrics(
    mean: torch.Tensor,
    target: torch.Tensor,
    *,
    faces: np.ndarray | None = None,
) -> dict[str, object]:
    error = mean - target
    metrics: dict[str, object] = {
        "mae": float(error.abs().mean().item()),
        "rmse": float(torch.sqrt((error ** 2).mean()).item()),
        "mean_bias_by_dim": [float(v) for v in error.mean(dim=0).tolist()],
        "mae_by_dim": [float(v) for v in error.abs().mean(dim=0).tolist()],
    }
    if faces is not None:
        faces_np = np.asarray(faces, dtype=np.int64)
        err_np = error.abs().mean(dim=1).detach().cpu().numpy()
        per_face = {}
        for face in FACE_ORDER:
            mask = faces_np == int(face)
            if np.any(mask):
                per_face[FACE_LABELS[int(face)]] = float(np.mean(err_np[mask]))
            else:
                per_face[FACE_LABELS[int(face)]] = None
        metrics["per_face_mae"] = per_face
    return metrics
