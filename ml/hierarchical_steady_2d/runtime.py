from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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

from .data import build_face_inputs, build_joint_inputs, build_track_inputs, prepare_hierarchical_condition_dataframe
from .models import FaceClassifier, GaussianRegressor, load_checkpoint
from .normalization import HierarchicalNormalizationBundle


@dataclass(frozen=True)
class HierarchicalSampleBatch:
    x_out: np.ndarray
    y_out: np.ndarray
    u_out: np.ndarray
    v_out: np.ndarray
    track_length: np.ndarray
    exit_face: np.ndarray
    pos_on_face: np.ndarray
    diagnostics: dict[str, float]


def _sample_faces(logits: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(uniforms, dtype=np.float64), 0.0, 1.0 - 1.0e-12)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    return np.sum(clipped[:, None] > cdf, axis=1, dtype=np.int64)


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

    diagnostics = {
        "direction_unit_disk_clip_fraction": float(np.mean(large)) if large.size else 0.0,
        "direction_outward_reflect_fraction": float(np.mean(needs_flip)) if needs_flip.size else 0.0,
        "direction_tiny_reset_fraction": float(np.mean(tiny)) if tiny.size else 0.0,
    }
    return direction[:, 0].astype(np.float32), direction[:, 1].astype(np.float32), diagnostics


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


def _predict_batched(model: FaceClassifier, inputs: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_inputs,) in loader:
            outputs.append(model(batch_inputs.to(device)).cpu().numpy())
    if not outputs:
        return np.zeros((0, int(model.num_classes)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _sample_regression_batched(
    model: GaussianRegressor,
    inputs: np.ndarray,
    *,
    device: str,
    batch_size: int,
    generator: torch.Generator | None,
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
            outputs.append(model.sample(batch_inputs.to(device), generator=generator).cpu().numpy())
    if not outputs:
        return np.zeros((0, int(model.target_dim)), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


class HierarchicalSteady2DSampler:
    def __init__(
        self,
        *,
        source_type: str,
        face_model: FaceClassifier,
        joint_model: GaussianRegressor,
        track_model: GaussianRegressor,
        normalization: HierarchicalNormalizationBundle,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.source_type = str(source_type)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.face_model = face_model.to(self.device).eval()
        self.joint_model = joint_model.to(self.device).eval()
        self.track_model = track_model.to(self.device).eval()
        self.normalization = normalization

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        source_type: str,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> "HierarchicalSteady2DSampler":
        resolved = Path(run_dir)
        normalization = HierarchicalNormalizationBundle.load(resolved / "metadata" / "normalization.json")
        if normalization.source != str(source_type):
            raise ValueError(
                f"Hierarchical run {resolved} was trained for source={normalization.source!r}, not {source_type!r}."
            )

        face_model, _face_cfg, face_meta = load_checkpoint(resolved / "face" / "models" / "best_model.pt", map_location=device)
        joint_model, _joint_cfg, joint_meta = load_checkpoint(resolved / "joint" / "models" / "best_model.pt", map_location=device)
        track_model, _track_cfg, track_meta = load_checkpoint(resolved / "track" / "models" / "best_model.pt", map_location=device)

        if not isinstance(face_model, FaceClassifier):
            raise TypeError(f"Expected FaceClassifier in {resolved / 'face' / 'models' / 'best_model.pt'}")
        if not isinstance(joint_model, GaussianRegressor):
            raise TypeError(f"Expected GaussianRegressor in {resolved / 'joint' / 'models' / 'best_model.pt'}")
        if not isinstance(track_model, GaussianRegressor):
            raise TypeError(f"Expected GaussianRegressor in {resolved / 'track' / 'models' / 'best_model.pt'}")

        if str(face_meta.get("stage", face_meta.get("task", ""))) != "face":
            raise ValueError(f"Face checkpoint in {resolved} does not report stage='face'.")
        if str(joint_meta.get("stage", joint_meta.get("task", ""))) != "joint":
            raise ValueError(f"Joint checkpoint in {resolved} does not report stage='joint'.")
        if str(track_meta.get("stage", track_meta.get("task", ""))) != "track":
            raise ValueError(f"Track checkpoint in {resolved} does not report stage='track'.")

        return cls(
            source_type=source_type,
            face_model=face_model,
            joint_model=joint_model,
            track_model=track_model,
            normalization=normalization,
            device=device,
            batch_size=batch_size,
        )

    def sample_batch(
        self,
        *,
        width: np.ndarray,
        height: np.ndarray,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        sigma_t: np.ndarray,
        seed: int | None = None,
    ) -> HierarchicalSampleBatch:
        width = np.asarray(width, dtype=np.float64).reshape(-1)
        height = np.asarray(height, dtype=np.float64).reshape(-1)
        initial_x = np.asarray(initial_x, dtype=np.float64).reshape(-1)
        initial_y = np.asarray(initial_y, dtype=np.float64).reshape(-1)
        initial_u = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        initial_v = np.asarray(initial_v, dtype=np.float64).reshape(-1)
        sigma_t = np.maximum(np.asarray(sigma_t, dtype=np.float64).reshape(-1), 1.0e-6)

        num_rows = width.shape[0]
        if not all(arr.shape[0] == num_rows for arr in (height, initial_x, initial_y, initial_u, initial_v, sigma_t)):
            raise ValueError("All hierarchical sampler input arrays must have matching first dimension.")

        rng = np.random.default_rng(seed)
        torch_generator = torch.Generator(device=self.device if self.device.startswith("cuda") else "cpu")
        if seed is not None:
            torch_generator.manual_seed(int(seed))
        else:
            torch_generator.seed()

        width_scaled = width * sigma_t
        height_scaled = height * sigma_t
        x_scaled = initial_x * sigma_t
        y_scaled = initial_y * sigma_t

        ballistic_length = np.zeros(num_rows, dtype=np.float64)
        ballistic_face = np.zeros(num_rows, dtype=np.int64)
        ballistic_exit_x = np.zeros(num_rows, dtype=np.float64)
        ballistic_exit_y = np.zeros(num_rows, dtype=np.float64)
        ballistic_face_coord = np.zeros(num_rows, dtype=np.float64)
        scatter_prob = np.zeros(num_rows, dtype=np.float64)

        for idx in range(num_rows):
            distance, face, x_exit, y_exit, coord = ballistic_exit_2d(
                float(x_scaled[idx]),
                float(y_scaled[idx]),
                float(initial_u[idx]),
                float(initial_v[idx]),
                float(width_scaled[idx]),
                float(height_scaled[idx]),
            )
            ballistic_length[idx] = float(distance)
            ballistic_face[idx] = int(face)
            ballistic_exit_x[idx] = float(x_exit)
            ballistic_exit_y[idx] = float(y_exit)
            ballistic_face_coord[idx] = float(coord)
            scatter_prob[idx] = float(scatter_before_exit_probability(1.0, float(distance)))

        scatter_mask = rng.random(num_rows) < scatter_prob
        final_x = ballistic_exit_x / sigma_t
        final_y = ballistic_exit_y / sigma_t
        final_u = initial_u.astype(np.float64, copy=True)
        final_v = initial_v.astype(np.float64, copy=True)
        final_track = ballistic_length / sigma_t
        final_face = ballistic_face.copy()
        final_pos_on_face = ballistic_face_coord / sigma_t

        diagnostics = {
            "scatter_fraction_sampled": float(np.mean(scatter_mask)) if num_rows else 0.0,
            "scatter_probability_mean": float(np.mean(scatter_prob)) if num_rows else 0.0,
        }

        if np.any(scatter_mask):
            scatter_idx = np.flatnonzero(scatter_mask)
            scatter_df = pd.DataFrame(
                {
                    "width": width_scaled[scatter_idx],
                    "height": height_scaled[scatter_idx],
                    "initial_pos_x": x_scaled[scatter_idx],
                    "initial_pos_y": y_scaled[scatter_idx],
                    "initial_dir_u": initial_u[scatter_idx],
                    "initial_dir_v": initial_v[scatter_idx],
                    "ballistic_length": ballistic_length[scatter_idx],
                    "ballistic_face": ballistic_face[scatter_idx],
                    "ballistic_face_coord_local": ballistic_face_coord[scatter_idx],
                    "scatter_before_exit_prob": scatter_prob[scatter_idx],
                }
            )
            prepared = prepare_hierarchical_condition_dataframe(scatter_df)

            face_inputs, _ = build_face_inputs(prepared, self.normalization)
            face_logits = _predict_batched(self.face_model, face_inputs, device=self.device, batch_size=self.batch_size)
            face_ids = _sample_faces(face_logits, rng.random(scatter_idx.shape[0]))

            joint_inputs, _ = build_joint_inputs(prepared, self.normalization, conditioning_face_ids=face_ids)
            joint_norm = _sample_regression_batched(
                self.joint_model,
                joint_inputs,
                device=self.device,
                batch_size=self.batch_size,
                generator=torch_generator,
            )
            joint_raw = self.normalization.denormalize_joint_targets(joint_norm)
            joint_coord_raw = joint_raw[:, 0].copy()
            joint_raw[:, 0] = np.clip(joint_raw[:, 0], 0.0, 1.0)
            diagnostics["joint_coord_clip_fraction_given_scatter"] = float(
                np.mean(np.abs(joint_raw[:, 0] - joint_coord_raw) > 1.0e-12)
            )
            joint_u, joint_v, joint_diag = _enforce_outward_numpy(face_ids, joint_raw[:, 1], joint_raw[:, 2])
            joint_raw[:, 1] = joint_u
            joint_raw[:, 2] = joint_v
            diagnostics.update({f"{key}_given_scatter": float(value) for key, value in joint_diag.items()})

            track_inputs, _ = build_track_inputs(
                prepared,
                self.normalization,
                conditioning_face_ids=face_ids,
                conditioning_joint_targets=joint_raw,
            )
            track_norm = _sample_regression_batched(
                self.track_model,
                track_inputs,
                device=self.device,
                batch_size=self.batch_size,
                generator=torch_generator,
            )
            track_log10 = self.normalization.denormalize_track_targets(track_norm).reshape(-1)
            track_optical = np.power(10.0, track_log10)
            track_physical = track_optical / sigma_t[scatter_idx]

            x_norm, y_norm = _face_coord_to_global_norm(face_ids, joint_raw[:, 0])
            x_phys = x_norm.astype(np.float64) * width[scatter_idx]
            y_phys = y_norm.astype(np.float64) * height[scatter_idx]
            straight_line_min = np.sqrt(
                np.maximum((x_phys - initial_x[scatter_idx]) ** 2 + (y_phys - initial_y[scatter_idx]) ** 2, 0.0)
            )
            diagnostics["track_lower_bound_clip_fraction_given_scatter"] = float(np.mean(track_physical < straight_line_min))
            track_physical = np.maximum(track_physical, straight_line_min)

            final_x[scatter_idx] = x_phys
            final_y[scatter_idx] = y_phys
            final_u[scatter_idx] = joint_raw[:, 1]
            final_v[scatter_idx] = joint_raw[:, 2]
            final_track[scatter_idx] = track_physical
            final_face[scatter_idx] = face_ids
            final_pos_on_face[scatter_idx] = np.where(
                (face_ids == FACE_LEFT) | (face_ids == FACE_RIGHT),
                y_phys,
                x_phys,
            )
        else:
            diagnostics["joint_coord_clip_fraction_given_scatter"] = 0.0
            diagnostics["track_lower_bound_clip_fraction_given_scatter"] = 0.0
            diagnostics["direction_unit_disk_clip_fraction_given_scatter"] = 0.0
            diagnostics["direction_outward_reflect_fraction_given_scatter"] = 0.0
            diagnostics["direction_tiny_reset_fraction_given_scatter"] = 0.0

        diagnostics["face_left_fraction"] = float(np.mean(final_face == FACE_LEFT)) if num_rows else 0.0
        diagnostics["face_right_fraction"] = float(np.mean(final_face == FACE_RIGHT)) if num_rows else 0.0
        diagnostics["face_top_fraction"] = float(np.mean(final_face == FACE_TOP)) if num_rows else 0.0
        diagnostics["face_bottom_fraction"] = float(np.mean(final_face == FACE_BOTTOM)) if num_rows else 0.0

        return HierarchicalSampleBatch(
            x_out=final_x.astype(np.float32),
            y_out=final_y.astype(np.float32),
            u_out=final_u.astype(np.float32),
            v_out=final_v.astype(np.float32),
            track_length=np.maximum(final_track, 0.0).astype(np.float32),
            exit_face=final_face.astype(np.int64),
            pos_on_face=final_pos_on_face.astype(np.float32),
            diagnostics=diagnostics,
        )
