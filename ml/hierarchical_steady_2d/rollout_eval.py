#!/usr/bin/env python3
"""End-to-end rollout evaluation for hierarchical steady-2D models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, qmc, wasserstein_distance
import torch
from torch.utils.data import DataLoader, TensorDataset

from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    ballistic_exit_2d,
    scatter_before_exit_probability,
)
from physics.mc_data_gen import (
    get_2d_boundary_simulation_data,
    get_2d_internal_source_simulation_data,
)

from .artifacts import save_json
from .data import (
    build_face_inputs,
    build_joint_inputs,
    build_track_inputs,
    prepare_hierarchical_condition_dataframe,
)
from .eval import _resolve_device
from .models import FaceClassifier, GaussianRegressor, HierarchicalMLP, load_checkpoint
from .normalization import HierarchicalNormalizationBundle


NUM_FACES = 4


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out hierarchical steady-2D models and compare against MC GT.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source", "--source-type", dest="source_type", default="internal", choices=["internal", "boundary"])
    parser.add_argument("--num-samples", "--support-samples", dest="num_samples", type=int, default=8192)
    parser.add_argument("--sampling", default="lhs", choices=["random", "lhs", "qmc"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-batch-size", type=int, default=512)
    parser.add_argument("--min-width", type=float, default=0.01)
    parser.add_argument("--max-width", type=float, default=10.0)
    parser.add_argument("--min-height", type=float, default=0.01)
    parser.add_argument("--max-height", type=float, default=10.0)
    parser.add_argument("--face-sampling", choices=["categorical", "argmax"], default="categorical")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _generate_samples(n: int, d: int, *, sampling: str, seed: int) -> np.ndarray:
    if sampling == "qmc":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        if (n & (n - 1) == 0) and n != 0:
            return sampler.random_base2(m=int(np.log2(n)))
        return sampler.random(n)
    if sampling == "lhs":
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, d))


def _resolve_run_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    face_ckpt = run_dir / "face" / "models" / "best_model.pt"
    joint_ckpt = run_dir / "joint" / "models" / "best_model.pt"
    track_ckpt = run_dir / "track" / "models" / "best_model.pt"
    normalization_path = run_dir / "metadata" / "normalization.json"
    for path in (face_ckpt, joint_ckpt, track_ckpt, normalization_path):
        if not path.exists():
            raise FileNotFoundError(f"Required hierarchical rollout artifact not found: {path}")
    return face_ckpt, joint_ckpt, track_ckpt, normalization_path


def _load_stage_model(checkpoint_path: Path, *, device: str, expected_stage: str) -> torch.nn.Module:
    model, _config, metadata = load_checkpoint(checkpoint_path, map_location=device)
    if not isinstance(model, (FaceClassifier, GaussianRegressor, HierarchicalMLP)):
        raise ValueError(
            "Hierarchical rollout expects standalone hierarchical checkpoints, "
            f"got {type(model).__name__} from {checkpoint_path}"
        )
    actual_stage = str(metadata.get("stage", metadata.get("task", "")))
    if actual_stage and actual_stage != expected_stage:
        raise ValueError(
            f"Checkpoint {checkpoint_path} reports stage={actual_stage!r}, "
            f"but rollout expected {expected_stage!r}."
        )
    if expected_stage == "face":
        if not isinstance(model, FaceClassifier):
            raise ValueError(f"Face rollout stage requires FaceClassifier, got {type(model).__name__} from {checkpoint_path}")
    elif expected_stage in {"joint", "track"}:
        if not isinstance(model, GaussianRegressor):
            raise ValueError(
                f"{expected_stage} rollout stage requires GaussianRegressor for stochastic sampling, "
                f"got {type(model).__name__} from {checkpoint_path}"
            )
    return model.to(device).eval()


def _decode_uniform_conditions(
    *,
    source_type: str,
    num_samples: int,
    sampling: str,
    seed: int,
    min_width: float,
    max_width: float,
    min_height: float,
    max_height: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cond_features = 5 if source_type == "boundary" else 6
    samples = _generate_samples(num_samples, cond_features + 2, sampling=sampling, seed=seed)

    log_w = samples[:, 0] * (np.log10(max_width) - np.log10(min_width)) + np.log10(min_width)
    width = 10.0 ** log_w
    log_h = samples[:, 1] * (np.log10(max_height) - np.log10(min_height)) + np.log10(min_height)
    height = 10.0 ** log_h

    if source_type == "boundary":
        y_norm = samples[:, 2]
        mu = 2.0 * samples[:, 3] - 1.0
        phi = samples[:, 4] * np.pi - 0.5 * np.pi
        tangent = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        initial_u = tangent * np.cos(phi)
        initial_v = tangent * np.sin(phi)
        initial_x = np.zeros_like(y_norm)
        initial_y = y_norm * height
    else:
        x_norm = samples[:, 2]
        y_norm = samples[:, 3]
        mu = 2.0 * samples[:, 4] - 1.0
        phi = samples[:, 5] * 2.0 * np.pi
        tangent = np.sqrt(np.clip(1.0 - mu * mu, 0.0, 1.0))
        initial_u = tangent * np.cos(phi)
        initial_v = tangent * np.sin(phi)
        initial_x = x_norm * width
        initial_y = y_norm * height

    frame = pd.DataFrame(
        {
            "width": width,
            "height": height,
            "initial_pos_x": initial_x,
            "initial_pos_y": initial_y,
            "initial_dir_u": initial_u,
            "initial_dir_v": initial_v,
        }
    )
    gate_uniform = samples[:, cond_features]
    face_uniform = samples[:, cond_features + 1]
    return frame, gate_uniform.astype(np.float64), face_uniform.astype(np.float64)


def _append_ballistic_metadata(frame: pd.DataFrame, *, sigma_t: float) -> pd.DataFrame:
    ballistic_length = np.zeros(len(frame), dtype=np.float64)
    ballistic_face = np.zeros(len(frame), dtype=np.int64)
    ballistic_exit_x = np.zeros(len(frame), dtype=np.float64)
    ballistic_exit_y = np.zeros(len(frame), dtype=np.float64)
    ballistic_face_coord_local = np.zeros(len(frame), dtype=np.float64)
    scatter_prob = np.zeros(len(frame), dtype=np.float64)

    width = frame["width"].to_numpy(dtype=np.float64, copy=False)
    height = frame["height"].to_numpy(dtype=np.float64, copy=False)
    x = frame["initial_pos_x"].to_numpy(dtype=np.float64, copy=False)
    y = frame["initial_pos_y"].to_numpy(dtype=np.float64, copy=False)
    u = frame["initial_dir_u"].to_numpy(dtype=np.float64, copy=False)
    v = frame["initial_dir_v"].to_numpy(dtype=np.float64, copy=False)

    for idx in range(len(frame)):
        distance, face, x_exit, y_exit, coord = ballistic_exit_2d(
            float(x[idx]),
            float(y[idx]),
            float(u[idx]),
            float(v[idx]),
            float(width[idx]),
            float(height[idx]),
        )
        ballistic_length[idx] = float(distance)
        ballistic_face[idx] = int(face)
        ballistic_exit_x[idx] = float(x_exit)
        ballistic_exit_y[idx] = float(y_exit)
        ballistic_face_coord_local[idx] = float(coord)
        scatter_prob[idx] = float(scatter_before_exit_probability(float(sigma_t), float(distance)))

    out = frame.copy()
    out["ballistic_length"] = ballistic_length
    out["ballistic_face"] = ballistic_face
    out["ballistic_exit_x"] = ballistic_exit_x
    out["ballistic_exit_y"] = ballistic_exit_y
    out["ballistic_face_coord_local"] = ballistic_face_coord_local
    out["scatter_before_exit_prob"] = scatter_prob
    return out


def _predict_batched(model: torch.nn.Module, inputs: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_inputs,) in loader:
            batch_inputs = batch_inputs.to(device)
            if isinstance(model, GaussianRegressor):
                mean, _log_std = model(batch_inputs)
                outputs.append(mean.cpu().numpy())
            else:
                outputs.append(model(batch_inputs).cpu().numpy())
    if not outputs:
        output_dim = getattr(model, "output_dim", getattr(model, "target_dim", getattr(model, "num_classes", 0)))
        return np.zeros((0, int(output_dim)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _sample_regression_batched(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    device: str,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_inputs,) in loader:
            batch_inputs = batch_inputs.to(device)
            if isinstance(model, GaussianRegressor):
                outputs.append(model.sample(batch_inputs, generator=generator).cpu().numpy())
            else:
                outputs.append(model(batch_inputs).cpu().numpy())
    if not outputs:
        output_dim = getattr(model, "target_dim", getattr(model, "output_dim", 0))
        return np.zeros((0, int(output_dim)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _sample_faces(logits: np.ndarray, uniforms: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "argmax":
        return np.argmax(logits, axis=1).astype(np.int64)

    uniforms = np.clip(np.asarray(uniforms, dtype=np.float64), 0.0, 1.0 - 1.0e-12)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    return np.sum(uniforms[:, None] > cdf, axis=1, dtype=np.int64)


def _enforce_outward_numpy(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out_u = np.asarray(u, dtype=np.float64).copy()
    out_v = np.asarray(v, dtype=np.float64).copy()

    radius = np.sqrt(np.maximum(out_u * out_u + out_v * out_v, 0.0))
    large = radius > 1.0
    if np.any(large):
        out_u[large] /= radius[large]
        out_v[large] /= radius[large]

    normals = np.zeros((face.shape[0], 2), dtype=np.float64)
    normals[face == FACE_LEFT] = (-1.0, 0.0)
    normals[face == FACE_RIGHT] = (1.0, 0.0)
    normals[face == FACE_TOP] = (0.0, 1.0)
    normals[face == FACE_BOTTOM] = (0.0, -1.0)

    direction = np.stack([out_u, out_v], axis=1)
    dot = np.sum(direction * normals, axis=1)
    needs_flip = dot <= 0.0
    if np.any(needs_flip):
        direction[needs_flip] = direction[needs_flip] - 2.0 * dot[needs_flip, None] * normals[needs_flip]

    tiny = np.linalg.norm(direction, axis=1) < 1.0e-8
    if np.any(tiny):
        direction[tiny] = 1.0e-8 * normals[tiny]

    out_u = direction[:, 0]
    out_v = direction[:, 1]
    diagnostics = {
        "direction_unit_disk_clip_fraction": float(np.mean(large)) if large.size else 0.0,
        "direction_outward_reflect_fraction": float(np.mean(needs_flip)) if needs_flip.size else 0.0,
        "direction_tiny_reset_fraction": float(np.mean(tiny)) if tiny.size else 0.0,
    }
    return out_u.astype(np.float32), out_v.astype(np.float32), diagnostics


def _face_coord_to_global_norm(face: np.ndarray, coord_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coord = np.clip(np.asarray(coord_norm, dtype=np.float64), 0.0, 1.0)
    x_norm = np.zeros_like(coord)
    y_norm = np.zeros_like(coord)

    left = face == FACE_LEFT
    right = face == FACE_RIGHT
    top = face == FACE_TOP
    bottom = face == FACE_BOTTOM

    x_norm[left] = 0.0
    y_norm[left] = coord[left]
    x_norm[right] = 1.0
    y_norm[right] = coord[right]
    x_norm[top] = coord[top]
    y_norm[top] = 1.0
    x_norm[bottom] = coord[bottom]
    y_norm[bottom] = 0.0
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def _distribution_metrics(gt_data: dict[str, np.ndarray], pred_data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for key, gt in gt_data.items():
        pred = pred_data[key]
        gt_arr = np.asarray(gt, dtype=np.float64).reshape(-1)
        pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
        metrics[key] = {
            "wasserstein": float(wasserstein_distance(gt_arr, pred_arr)),
            "ks_stat": float(ks_2samp(gt_arr, pred_arr).statistic),
        }
    return metrics


def _mean_metric(metrics: dict[str, dict[str, float]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics.values()]
    return float(np.mean(values)) if values else float("nan")


def _load_mc_ground_truth(
    *,
    source_type: str,
    num_samples: int,
    sampling: str,
    min_width: float,
    max_width: float,
    min_height: float,
    max_height: float,
) -> dict[str, np.ndarray]:
    if source_type == "boundary":
        sim_gt = get_2d_boundary_simulation_data(
            num_neutrons=int(num_samples),
            scatter_ratio=1.0,
            distribution=0,
            min_width=float(min_width),
            max_width=float(max_width),
            min_height=float(min_height),
            max_height=float(max_height),
            sampling=str(sampling),
        )
    else:
        sim_gt = get_2d_internal_source_simulation_data(
            num_neutrons=int(num_samples),
            scatter_ratio=1.0,
            distribution=0,
            min_width=float(min_width),
            max_width=float(max_width),
            min_height=float(min_height),
            max_height=float(max_height),
            sampling=str(sampling),
        )
    sim_gt = sim_gt[sim_gt["exited"] == 1].copy()
    if sim_gt.empty:
        raise RuntimeError("No exiting Monte Carlo ground-truth particles were generated.")
    return {
        "x_out_norm": sim_gt["final_pos_x_norm"].to_numpy(dtype=np.float64, copy=True),
        "y_out_norm": sim_gt["final_pos_y_norm"].to_numpy(dtype=np.float64, copy=True),
        "u": sim_gt["final_dir_u"].to_numpy(dtype=np.float64, copy=True),
        "v": sim_gt["final_dir_v"].to_numpy(dtype=np.float64, copy=True),
        "time": sim_gt["time_in_cell"].to_numpy(dtype=np.float64, copy=True),
    }


def _hierarchical_rollout_predictions(
    *,
    frame: pd.DataFrame,
    gate_uniform: np.ndarray,
    face_uniform: np.ndarray,
    face_model: torch.nn.Module,
    joint_model: torch.nn.Module,
    track_model: torch.nn.Module,
    normalization: HierarchicalNormalizationBundle,
    device: str,
    batch_size: int,
    face_sampling: str,
    sample_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    prepared = prepare_hierarchical_condition_dataframe(frame)
    scatter_prob = prepared["scatter_before_exit_prob"].to_numpy(dtype=np.float64, copy=False)
    scatter_mask = gate_uniform < scatter_prob
    device_str = str(device)
    if device_str == "cpu" or device_str.startswith("cuda"):
        generator = torch.Generator(device=device_str if device_str.startswith("cuda") else "cpu")
        generator.manual_seed(int(sample_seed))
    else:
        generator = None
        torch.manual_seed(int(sample_seed))

    num_rows = len(prepared)
    final_x_norm = np.clip(
        prepared["ballistic_exit_x"].to_numpy(dtype=np.float64, copy=False)
        / np.maximum(prepared["width"].to_numpy(dtype=np.float64, copy=False), 1.0e-12),
        0.0,
        1.0,
    )
    final_y_norm = np.clip(
        prepared["ballistic_exit_y"].to_numpy(dtype=np.float64, copy=False)
        / np.maximum(prepared["height"].to_numpy(dtype=np.float64, copy=False), 1.0e-12),
        0.0,
        1.0,
    )
    final_u = prepared["initial_dir_u"].to_numpy(dtype=np.float64, copy=True)
    final_v = prepared["initial_dir_v"].to_numpy(dtype=np.float64, copy=True)
    final_time = prepared["ballistic_length"].to_numpy(dtype=np.float64, copy=True)
    exit_face = prepared["ballistic_face"].to_numpy(dtype=np.int64, copy=True)

    if np.any(scatter_mask):
        scatter_df = prepared.loc[scatter_mask].reset_index(drop=True)

        face_inputs, _ = build_face_inputs(scatter_df, normalization)
        face_logits = _predict_batched(face_model, face_inputs, device=device, batch_size=batch_size)
        face_ids = _sample_faces(face_logits, face_uniform[scatter_mask], mode=face_sampling)

        joint_inputs, _ = build_joint_inputs(scatter_df, normalization, conditioning_face_ids=face_ids)
        joint_pred_norm = _sample_regression_batched(
            joint_model,
            joint_inputs,
            device=device,
            batch_size=batch_size,
            generator=generator,
        )
        joint_pred_raw = normalization.denormalize_joint_targets(joint_pred_norm)
        joint_coord_raw = joint_pred_raw[:, 0].copy()
        joint_pred_raw[:, 0] = np.clip(joint_pred_raw[:, 0], 0.0, 1.0)
        joint_coord_clip_fraction = float(np.mean(np.abs(joint_pred_raw[:, 0] - joint_coord_raw) > 1.0e-12))
        joint_u, joint_v, joint_projection_diag = _enforce_outward_numpy(face_ids, joint_pred_raw[:, 1], joint_pred_raw[:, 2])
        joint_pred_raw[:, 1] = joint_u
        joint_pred_raw[:, 2] = joint_v

        track_inputs, _ = build_track_inputs(
            scatter_df,
            normalization,
            conditioning_face_ids=face_ids,
            conditioning_joint_targets=joint_pred_raw,
        )
        track_pred_norm = _sample_regression_batched(
            track_model,
            track_inputs,
            device=device,
            batch_size=batch_size,
            generator=generator,
        )
        track_log10 = normalization.denormalize_track_targets(track_pred_norm).reshape(-1)
        track_raw = np.power(10.0, track_log10)

        scatter_x_norm, scatter_y_norm = _face_coord_to_global_norm(face_ids, joint_pred_raw[:, 0])
        width = scatter_df["width"].to_numpy(dtype=np.float64, copy=False)
        height = scatter_df["height"].to_numpy(dtype=np.float64, copy=False)
        x0 = scatter_df["initial_pos_x"].to_numpy(dtype=np.float64, copy=False)
        y0 = scatter_df["initial_pos_y"].to_numpy(dtype=np.float64, copy=False)
        x_exit = scatter_x_norm.astype(np.float64) * width
        y_exit = scatter_y_norm.astype(np.float64) * height
        straight_line_min = np.sqrt(np.maximum((x_exit - x0) ** 2 + (y_exit - y0) ** 2, 0.0))
        track_lower_bound_clip_fraction = float(np.mean(track_raw < straight_line_min))
        track_raw = np.maximum(track_raw, straight_line_min)

        scatter_indices = np.flatnonzero(scatter_mask)
        final_x_norm[scatter_indices] = scatter_x_norm
        final_y_norm[scatter_indices] = scatter_y_norm
        final_u[scatter_indices] = joint_pred_raw[:, 1]
        final_v[scatter_indices] = joint_pred_raw[:, 2]
        final_time[scatter_indices] = track_raw
        exit_face[scatter_indices] = face_ids

    pred_data = {
        "x_out_norm": np.clip(final_x_norm, 0.0, 1.0),
        "y_out_norm": np.clip(final_y_norm, 0.0, 1.0),
        "u": final_u,
        "v": final_v,
        "time": np.maximum(final_time, 0.0),
    }
    diagnostics = {
        "scatter_fraction_sampled": float(np.mean(scatter_mask)),
        "scatter_probability_mean": float(np.mean(scatter_prob)),
        "face_left_fraction": float(np.mean(exit_face == FACE_LEFT)),
        "face_right_fraction": float(np.mean(exit_face == FACE_RIGHT)),
        "face_top_fraction": float(np.mean(exit_face == FACE_TOP)),
        "face_bottom_fraction": float(np.mean(exit_face == FACE_BOTTOM)),
        "joint_coord_clip_fraction_given_scatter": float(joint_coord_clip_fraction) if np.any(scatter_mask) else 0.0,
        "track_lower_bound_clip_fraction_given_scatter": float(track_lower_bound_clip_fraction) if np.any(scatter_mask) else 0.0,
    }
    if np.any(scatter_mask):
        diagnostics.update(
            {
                f"{key}_given_scatter": float(value)
                for key, value in joint_projection_diag.items()
            }
        )
    return pred_data, diagnostics


def evaluate_hierarchical_rollout(
    run_dir: str | Path,
    *,
    source_type: str,
    num_samples: int = 8192,
    sampling: str = "lhs",
    seed: int = 123,
    model_batch_size: int = 512,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    face_sampling: str = "categorical",
    device: str | None = None,
    output: str | Path | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    face_ckpt, joint_ckpt, track_ckpt, normalization_path = _resolve_run_paths(run_dir)
    run_config_path = run_dir / "metadata" / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config at {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    trained_source = str(run_config.get("source", source_type))
    if trained_source != source_type:
        raise ValueError(f"Run dir {run_dir} was trained for source={trained_source!r}, not {source_type!r}.")
    sigma_t = float(run_config.get("sigma_t", 1.0))
    if abs(sigma_t - 1.0) > 1.0e-8:
        raise ValueError(
            "Legacy steady-2D comparison currently assumes sigma_t=1.0 to match the existing monolithic path. "
            f"Found sigma_t={sigma_t} in {run_config_path}."
        )

    resolved_device = _resolve_device(device)
    normalization = HierarchicalNormalizationBundle.load(normalization_path)
    face_model = _load_stage_model(face_ckpt, device=resolved_device, expected_stage="face")
    joint_model = _load_stage_model(joint_ckpt, device=resolved_device, expected_stage="joint")
    track_model = _load_stage_model(track_ckpt, device=resolved_device, expected_stage="track")

    condition_frame, gate_uniform, face_uniform = _decode_uniform_conditions(
        source_type=source_type,
        num_samples=int(num_samples),
        sampling=sampling,
        seed=int(seed),
        min_width=float(min_width),
        max_width=float(max_width),
        min_height=float(min_height),
        max_height=float(max_height),
    )
    condition_frame = _append_ballistic_metadata(condition_frame, sigma_t=sigma_t)

    started_at = time.perf_counter()
    pred_data, diagnostics = _hierarchical_rollout_predictions(
        frame=condition_frame,
        gate_uniform=gate_uniform,
        face_uniform=face_uniform,
        face_model=face_model,
        joint_model=joint_model,
        track_model=track_model,
        normalization=normalization,
        device=resolved_device,
        batch_size=int(model_batch_size),
        face_sampling=face_sampling,
        sample_seed=int(seed) + 1000,
    )
    runtime_seconds = time.perf_counter() - started_at

    gt_data = _load_mc_ground_truth(
        source_type=source_type,
        num_samples=int(num_samples),
        sampling=sampling,
        min_width=float(min_width),
        max_width=float(max_width),
        min_height=float(min_height),
        max_height=float(max_height),
    )
    metrics = _distribution_metrics(gt_data, pred_data)

    output_path = Path(output) if output else run_dir / f"rollout_eval_{source_type}.json"
    payload: dict[str, object] = {
        "run_dir": str(run_dir),
        "source_type": source_type,
        "num_samples": int(num_samples),
        "sampling": sampling,
        "seed": int(seed),
        "model_batch_size": int(model_batch_size),
        "device": resolved_device,
        "sigma_t": sigma_t,
        "face_sampling": face_sampling,
        "runtime_seconds": float(runtime_seconds),
        "samples_per_second": float(num_samples / runtime_seconds) if runtime_seconds > 0.0 else float("inf"),
        "metrics": metrics,
        "mean_wasserstein": _mean_metric(metrics, "wasserstein"),
        "mean_ks_stat": _mean_metric(metrics, "ks_stat"),
        "diagnostics": diagnostics,
    }
    save_json(output_path, payload)
    return payload


def main(argv=None) -> None:
    args = parse_args(argv)
    payload = evaluate_hierarchical_rollout(
        args.run_dir,
        source_type=args.source_type,
        num_samples=args.num_samples,
        sampling=args.sampling,
        seed=args.seed,
        model_batch_size=args.model_batch_size,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        face_sampling=args.face_sampling,
        device=args.device,
        output=args.output,
    )
    output_path = Path(args.output) if args.output else Path(args.run_dir) / f"rollout_eval_{args.source_type}.json"
    print(
        "Saved hierarchical rollout evaluation to "
        f"{output_path} with mean Wasserstein={payload['mean_wasserstein']:.4e} "
        f"and mean KS={payload['mean_ks_stat']:.4e}"
    )


if __name__ == "__main__":
    main()
