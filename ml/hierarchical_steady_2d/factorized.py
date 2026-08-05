from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
)

from .models import ConditionalFlowModel, GaussianRegressor, load_checkpoint, save_checkpoint
from .symmetry import (
    LEGENDRE_BASIS_ORDER,
    CanonicalizedBoundaryState2D,
    CanonicalizedInternalState2D,
    canonicalize_boundary_state,
    canonicalize_internal_state,
    classify_boundary_face,
    decode_boundary_unit_vector,
    encode_boundary_unit_vector,
    face_local_coordinate,
    forward_transform_angular_moments,
    forward_transform_direction,
    forward_transform_point,
    inverse_transform_angular_moments,
    inverse_transform_direction,
    inverse_transform_legendre_coeffs,
    inverse_transform_point,
)


EPSILON = 1.0e-12
LOCATION_STAGE = "location"
KINEMATICS_STAGE = "kinematics"

ANGULAR_MOMENT_NAMES = (
    "moment_u",
    "moment_v",
    "moment_uu",
    "moment_uv",
    "moment_vv",
)
LEGENDRE_COEFF_NAMES = tuple(f"legendre_p{i}_p{j}" for i, j in LEGENDRE_BASIS_ORDER)

_BOUNDARY_BASE_COLUMNS = (
    "width_log10",
    "height_log10",
    "entry_y_norm",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length_log10",
    "ballistic_exit_unit_x",
    "ballistic_exit_unit_y",
    "scatter_before_exit_prob",
    "sigma_s",
    "sigma_a",
)
_INTERNAL_BASE_COLUMNS = (
    "width_log10",
    "height_log10",
    "initial_pos_x_norm",
    "initial_pos_y_norm",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length_log10",
    "ballistic_exit_unit_x",
    "ballistic_exit_unit_y",
    "scatter_before_exit_prob",
    "sigma_s",
    "sigma_a",
)
_LOCATION_TARGET_COLUMNS = ("exit_unit_x", "exit_unit_y")
_KINEMATICS_TARGET_COLUMNS = (
    "final_dir_u",
    "final_dir_v",
    "total_track_length_log10",
    "weighted_track_length_log10",
) + ANGULAR_MOMENT_NAMES


@dataclass(frozen=True)
class FactorizedTrainingConfig:
    source: str = "internal"
    num_neutrons: int = 100000
    sigma_s: float = 1.0
    sigma_a: float = 0.0
    distribution: int = 0
    min_width: float = 0.01
    max_width: float = 10.0
    min_height: float = 0.01
    max_height: float = 10.0
    sampling: str = "random"
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    val_ratio: float = 0.1
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.0
    flow_steps: int = 10
    seed: int = 0
    device: str = "auto"
    output_root: str = "ml/hierarchical_steady_2d/factorized_runs"
    run_name: str | None = None


@dataclass(frozen=True)
class FactorizedNormalizationBundle:
    source: str
    base_feature_names: tuple[str, ...]
    base_feature_mean: np.ndarray
    base_feature_std: np.ndarray
    location_target_names: tuple[str, ...]
    location_target_mean: np.ndarray
    location_target_std: np.ndarray
    kinematics_feature_names: tuple[str, ...]
    kinematics_feature_mean: np.ndarray
    kinematics_feature_std: np.ndarray
    kinematics_target_names: tuple[str, ...]
    kinematics_target_mean: np.ndarray
    kinematics_target_std: np.ndarray

    def normalize_base_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.base_feature_mean) / self.base_feature_std).astype(np.float32, copy=False)

    def normalize_location_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.location_target_mean) / self.location_target_std).astype(np.float32, copy=False)

    def denormalize_location_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.location_target_std + self.location_target_mean).astype(np.float32, copy=False)

    def normalize_kinematics_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.kinematics_feature_mean) / self.kinematics_feature_std).astype(np.float32, copy=False)

    def normalize_kinematics_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.kinematics_target_mean) / self.kinematics_target_std).astype(np.float32, copy=False)

    def denormalize_kinematics_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.kinematics_target_std + self.kinematics_target_mean).astype(np.float32, copy=False)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "base_feature_names": list(self.base_feature_names),
            "location_target_names": list(self.location_target_names),
            "kinematics_feature_names": list(self.kinematics_feature_names),
            "kinematics_target_names": list(self.kinematics_target_names),
        }

    def save(self, output_dir: str | Path) -> tuple[Path, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "normalization.json"
        npz_path = output_path / "normalization.npz"
        json_path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True))
        np.savez(
            npz_path,
            base_feature_mean=self.base_feature_mean,
            base_feature_std=self.base_feature_std,
            location_target_mean=self.location_target_mean,
            location_target_std=self.location_target_std,
            kinematics_feature_mean=self.kinematics_feature_mean,
            kinematics_feature_std=self.kinematics_feature_std,
            kinematics_target_mean=self.kinematics_target_mean,
            kinematics_target_std=self.kinematics_target_std,
        )
        return json_path, npz_path

    @classmethod
    def load(cls, json_path: str | Path, npz_path: str | Path | None = None) -> "FactorizedNormalizationBundle":
        json_file = Path(json_path)
        payload = json.loads(json_file.read_text())
        npz_file = Path(npz_path) if npz_path is not None else json_file.with_suffix(".npz")
        arrays = np.load(npz_file)
        return cls(
            source=str(payload["source"]),
            base_feature_names=tuple(payload["base_feature_names"]),
            base_feature_mean=np.asarray(arrays["base_feature_mean"], dtype=np.float32),
            base_feature_std=np.asarray(arrays["base_feature_std"], dtype=np.float32),
            location_target_names=tuple(payload["location_target_names"]),
            location_target_mean=np.asarray(arrays["location_target_mean"], dtype=np.float32),
            location_target_std=np.asarray(arrays["location_target_std"], dtype=np.float32),
            kinematics_feature_names=tuple(payload["kinematics_feature_names"]),
            kinematics_feature_mean=np.asarray(arrays["kinematics_feature_mean"], dtype=np.float32),
            kinematics_feature_std=np.asarray(arrays["kinematics_feature_std"], dtype=np.float32),
            kinematics_target_names=tuple(payload["kinematics_target_names"]),
            kinematics_target_mean=np.asarray(arrays["kinematics_target_mean"], dtype=np.float32),
            kinematics_target_std=np.asarray(arrays["kinematics_target_std"], dtype=np.float32),
        )


@dataclass(frozen=True)
class FactorizedPreparedData:
    source: str
    dataframe: pd.DataFrame
    base_columns: tuple[str, ...]
    kinematics_feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class FactorizedSampleBatch:
    x_out: np.ndarray
    y_out: np.ndarray
    u_out: np.ndarray
    v_out: np.ndarray
    track_length: np.ndarray
    exit_face: np.ndarray
    pos_on_face: np.ndarray
    weighted_track_length: np.ndarray
    angular_moments: np.ndarray
    legendre_coeffs: np.ndarray
    diagnostics: dict[str, float]


def _fit_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _face_one_hot(face_values: np.ndarray, *, prefix: str) -> tuple[np.ndarray, tuple[str, ...]]:
    face_ids = np.asarray(face_values, dtype=np.int64)
    one_hot = np.eye(4, dtype=np.float32)[face_ids]
    names = tuple(f"{prefix}_{name}" for name in ("left", "right", "top", "bottom"))
    return one_hot, names


def _encode_exit_direction_angle(face: int, u: float, v: float) -> float:
    """
    Encode exit direction (u, v) as a single angle ∈ (-π/2, π/2) in the
    outward half-space of the given face.  The angle is measured from the
    face outward normal toward the face tangent.

    Faces and their (normal, tangent) pairs:
        LEFT  → normal=(-1,0), tangent=(0,+1)
        RIGHT → normal=(+1,0), tangent=(0,+1)
        TOP   → normal=(0,+1), tangent=(+1,0)
        BOTTOM→ normal=(0,-1), tangent=(+1,0)
    """
    if face == FACE_LEFT:
        n, t = -u, v
    elif face == FACE_RIGHT:
        n, t = u, v
    elif face == FACE_TOP:
        n, t = v, u
    else:  # FACE_BOTTOM
        n, t = -v, u
    # n should be > 0 for outward; clamp to avoid log-domain issues
    n = max(n, 1e-8)
    return math.atan2(t, n)


def _decode_exit_direction_angle(face: int, angle: float) -> tuple[float, float]:
    """Inverse of _encode_exit_direction_angle — always returns an outward unit vector."""
    n = math.cos(angle)
    t = math.sin(angle)
    if face == FACE_LEFT:
        return -n, t
    elif face == FACE_RIGHT:
        return n, t
    elif face == FACE_TOP:
        return t, n
    else:  # FACE_BOTTOM
        return t, -n


def _normalize_exit_direction(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out_u = np.asarray(u, dtype=np.float64).copy()
    out_v = np.asarray(v, dtype=np.float64).copy()
    radius = np.sqrt(np.maximum(out_u * out_u + out_v * out_v, 0.0))
    oversized = radius > 1.0
    if np.any(oversized):
        out_u[oversized] /= radius[oversized]
        out_v[oversized] /= radius[oversized]

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
    diagnostics = {
        "direction_unit_disk_clip_fraction": float(np.mean(oversized)) if oversized.size else 0.0,
        "direction_outward_reflect_fraction": float(np.mean(needs_flip)) if needs_flip.size else 0.0,
    }
    return direction[:, 0].astype(np.float32), direction[:, 1].astype(np.float32), diagnostics


def _split_dataframe(df: pd.DataFrame, *, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least two rows for factorized training.")
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for _, group in df.groupby("exit_face", sort=True):
        group_idx = group.index.to_numpy(dtype=np.int64, copy=True)
        if group_idx.size <= 1:
            continue
        rng.shuffle(group_idx)
        val_count = max(1, int(round(group_idx.size * val_ratio)))
        val_count = min(val_count, group_idx.size - 1)
        val_indices.extend(group_idx[:val_count].tolist())
    if val_indices:
        val_idx = np.array(sorted(set(val_indices)), dtype=np.int64)
        return (
            df.drop(index=val_idx).reset_index(drop=True),
            df.loc[val_idx].reset_index(drop=True),
        )
    perm = rng.permutation(len(df))
    val_count = min(max(int(round(len(df) * val_ratio)), 1), len(df) - 1)
    return df.iloc[perm[val_count:]].reset_index(drop=True), df.iloc[perm[:val_count]].reset_index(drop=True)


def _base_columns_for_source(source: str) -> tuple[str, ...]:
    return _BOUNDARY_BASE_COLUMNS if source == "boundary" else _INTERNAL_BASE_COLUMNS


def _canonicalize_row(source: str, row: pd.Series) -> dict[str, object]:
    width = float(row["width"])
    height = float(row["height"])
    u0 = float(row["initial_dir_u"])
    v0 = float(row["initial_dir_v"])
    sigma_s = float(row["sigma_s"])
    sigma_a = float(row["sigma_a"])
    ballistic_prob = float(row["scatter_before_exit_prob"])
    if source == "boundary":
        ballistic_face = int(row["ballistic_face"])
        ballistic_coord = float(row["ballistic_face_coord_local"])
        canonical: CanonicalizedBoundaryState2D = canonicalize_boundary_state(
            ballistic_face,
            ballistic_coord,
            u0,
            v0,
            width,
            height,
        )
        x0 = 0.0
        y0 = canonical.y_on_entry
        base = {
            "width": canonical.width,
            "height": canonical.height,
            "entry_y_norm": canonical.y_on_entry / max(canonical.height, EPSILON),
            "initial_dir_u": canonical.u,
            "initial_dir_v": canonical.v,
        }
    else:
        canonical = canonicalize_internal_state(
            width,
            height,
            float(row["initial_pos_x"]),
            float(row["initial_pos_y"]),
            u0,
            v0,
        )
        x0 = canonical.x
        y0 = canonical.y
        base = {
            "width": canonical.width,
            "height": canonical.height,
            "initial_pos_x_norm": canonical.x / max(canonical.width, EPSILON),
            "initial_pos_y_norm": canonical.y / max(canonical.height, EPSILON),
            "initial_dir_u": canonical.u,
            "initial_dir_v": canonical.v,
        }

    ballistic_x_phys = float(row["ballistic_exit_x"])
    ballistic_y_phys = float(row["ballistic_exit_y"])
    ballistic_x_can, ballistic_y_can = forward_transform_point(
        canonical.transform,
        width,
        height,
        canonical.width,
        canonical.height,
        ballistic_x_phys,
        ballistic_y_phys,
    )
    ballistic_unit_x, ballistic_unit_y = encode_boundary_unit_vector(
        ballistic_x_can,
        ballistic_y_can,
        canonical.width,
        canonical.height,
    )
    exit_x_can, exit_y_can = forward_transform_point(
        canonical.transform,
        width,
        height,
        canonical.width,
        canonical.height,
        float(row["final_pos_x"]),
        float(row["final_pos_y"]),
    )
    exit_unit_x, exit_unit_y = encode_boundary_unit_vector(
        exit_x_can,
        exit_y_can,
        canonical.width,
        canonical.height,
    )
    final_u_can, final_v_can = forward_transform_direction(
        canonical.transform,
        float(row["final_dir_u"]),
        float(row["final_dir_v"]),
    )
    exit_face_can = classify_boundary_face(exit_x_can, exit_y_can, canonical.width, canonical.height)
    exit_coord_can = face_local_coordinate(exit_face_can, exit_x_can, exit_y_can)
    exit_coord_norm = exit_coord_can / max(
        canonical.height if exit_face_can in (FACE_LEFT, FACE_RIGHT) else canonical.width,
        EPSILON,
    )
    angular_raw = np.array([float(row[name]) for name in ANGULAR_MOMENT_NAMES], dtype=np.float64)
    angular_can = forward_transform_angular_moments(canonical.transform, angular_raw)
    return {
        "width": canonical.width,
        "height": canonical.height,
        "width_log10": math.log10(max(canonical.width, EPSILON)),
        "height_log10": math.log10(max(canonical.height, EPSILON)),
        "ballistic_length_log10": math.log10(max(float(row["ballistic_length"]), EPSILON)),
        "ballistic_exit_unit_x": ballistic_unit_x,
        "ballistic_exit_unit_y": ballistic_unit_y,
        "scatter_before_exit_prob": ballistic_prob,
        "sigma_s": sigma_s,
        "sigma_a": sigma_a,
        "exit_unit_x": exit_unit_x,
        "exit_unit_y": exit_unit_y,
        "exit_face": exit_face_can,
        "exit_face_coord_norm": exit_coord_norm,
        "final_dir_u": final_u_can,
        "final_dir_v": final_v_can,
        "total_track_length_log10": math.log10(max(float(row["total_track_length"]), EPSILON)),
        "weighted_track_length_log10": math.log10(max(float(row["weighted_track_length"]), EPSILON)),
        **base,
        **{name: float(value) for name, value in zip(ANGULAR_MOMENT_NAMES, angular_can)},
    }


def _canonicalize_runtime_state(
    source: str,
    *,
    width: float,
    height: float,
    initial_x: float,
    initial_y: float,
    initial_u: float,
    initial_v: float,
    ballistic_length: float,
    ballistic_face: int,
    ballistic_coord: float,
    ballistic_exit_x: float,
    ballistic_exit_y: float,
    scatter_before_exit_prob: float,
    sigma_s: float,
    sigma_a: float,
) -> tuple[dict[str, object], CanonicalizedBoundaryState2D | CanonicalizedInternalState2D]:
    if source == "boundary":
        canonical = canonicalize_boundary_state(
            int(ballistic_face),
            float(ballistic_coord),
            initial_u,
            initial_v,
            width,
            height,
        )
        base = {
            "width": canonical.width,
            "height": canonical.height,
            "width_log10": math.log10(max(canonical.width, EPSILON)),
            "height_log10": math.log10(max(canonical.height, EPSILON)),
            "entry_y_norm": canonical.y_on_entry / max(canonical.height, EPSILON),
            "initial_dir_u": canonical.u,
            "initial_dir_v": canonical.v,
        }
    else:
        canonical = canonicalize_internal_state(width, height, initial_x, initial_y, initial_u, initial_v)
        base = {
            "width": canonical.width,
            "height": canonical.height,
            "width_log10": math.log10(max(canonical.width, EPSILON)),
            "height_log10": math.log10(max(canonical.height, EPSILON)),
            "initial_pos_x_norm": canonical.x / max(canonical.width, EPSILON),
            "initial_pos_y_norm": canonical.y / max(canonical.height, EPSILON),
            "initial_dir_u": canonical.u,
            "initial_dir_v": canonical.v,
        }
    ballistic_x_can, ballistic_y_can = forward_transform_point(
        canonical.transform,
        width,
        height,
        canonical.width,
        canonical.height,
        ballistic_exit_x,
        ballistic_exit_y,
    )
    ballistic_unit_x, ballistic_unit_y = encode_boundary_unit_vector(
        ballistic_x_can,
        ballistic_y_can,
        canonical.width,
        canonical.height,
    )
    return (
        {
            **base,
            "ballistic_length_log10": math.log10(max(ballistic_length, EPSILON)),
            "ballistic_exit_unit_x": ballistic_unit_x,
            "ballistic_exit_unit_y": ballistic_unit_y,
            "scatter_before_exit_prob": scatter_before_exit_prob,
            "sigma_s": sigma_s,
            "sigma_a": sigma_a,
        },
        canonical,
    )


def prepare_factorized_dataframe(df: pd.DataFrame, *, source: str) -> FactorizedPreparedData:
    required = {
        "width",
        "height",
        "initial_dir_u",
        "initial_dir_v",
        "ballistic_length",
        "ballistic_face",
        "ballistic_face_coord_local",
        "ballistic_exit_x",
        "ballistic_exit_y",
        "scatter_before_exit_prob",
        "sigma_s",
        "sigma_a",
        "final_pos_x",
        "final_pos_y",
        "final_dir_u",
        "final_dir_v",
        "total_track_length",
        "weighted_track_length",
        *ANGULAR_MOMENT_NAMES,
    }
    if source == "internal":
        required |= {"initial_pos_x", "initial_pos_y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Factorized hierarchical dataframe is missing columns: {missing}")
    rows = [_canonicalize_row(source, row) for _, row in df.iterrows()]
    prepared = pd.DataFrame(rows)
    face_one_hot, face_names = _face_one_hot(prepared["exit_face"].to_numpy(dtype=np.int64), prefix="exit_face_is")
    face_frame = pd.DataFrame(face_one_hot, columns=face_names)
    prepared = pd.concat([prepared.reset_index(drop=True), face_frame], axis=1)
    kinematics_feature_columns = _base_columns_for_source(source) + _LOCATION_TARGET_COLUMNS + ("exit_face_coord_norm",) + face_names
    return FactorizedPreparedData(
        source=source,
        dataframe=prepared,
        base_columns=_base_columns_for_source(source),
        kinematics_feature_columns=kinematics_feature_columns,
    )


def fit_factorized_normalization(prepared: FactorizedPreparedData) -> FactorizedNormalizationBundle:
    base_values = prepared.dataframe.loc[:, prepared.base_columns].to_numpy(dtype=np.float32, copy=True)
    location_values = prepared.dataframe.loc[:, _LOCATION_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    kinematics_feature_values = prepared.dataframe.loc[:, prepared.kinematics_feature_columns].to_numpy(dtype=np.float32, copy=True)
    kinematics_target_values = prepared.dataframe.loc[:, _KINEMATICS_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    base_mean, base_std = _fit_stats(base_values)
    location_mean, location_std = _fit_stats(location_values)
    kin_feat_mean, kin_feat_std = _fit_stats(kinematics_feature_values)
    kin_target_mean, kin_target_std = _fit_stats(kinematics_target_values)
    return FactorizedNormalizationBundle(
        source=prepared.source,
        base_feature_names=prepared.base_columns,
        base_feature_mean=base_mean,
        base_feature_std=base_std,
        location_target_names=_LOCATION_TARGET_COLUMNS,
        location_target_mean=location_mean,
        location_target_std=location_std,
        kinematics_feature_names=prepared.kinematics_feature_columns,
        kinematics_feature_mean=kin_feat_mean,
        kinematics_feature_std=kin_feat_std,
        kinematics_target_names=_KINEMATICS_TARGET_COLUMNS,
        kinematics_target_mean=kin_target_mean,
        kinematics_target_std=kin_target_std,
    )


def _make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    aux: np.ndarray | None = None,
) -> DataLoader:
    aux_values = aux if aux is not None else np.zeros((inputs.shape[0], 0), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(targets, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(aux_values, dtype=np.float32, copy=True)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _eval_gaussian_model(
    model: GaussianRegressor,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    count = 0
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets, _aux in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss, mean, _ = model.nll(inputs, targets)
            total_loss += float(loss.item()) * inputs.shape[0]
            count += inputs.shape[0]
            pred_batches.append(mean.cpu().numpy())
            target_batches.append(targets.cpu().numpy())
    return total_loss / max(1, count), np.concatenate(pred_batches, axis=0), np.concatenate(target_batches, axis=0)


# ---------------------------------------------------------------------------
# Physics-informed source distributions for flow matching
# ---------------------------------------------------------------------------

def _source_location(
    base_features_raw: torch.Tensor,
    normalization: "FactorizedNormalizationBundle",
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Source distribution for exit location in the same normalized space used by the flow target.

    We start from the per-sample ballistic exit encoding, add isotropic noise in the
    raw boundary-vector space, project back to the boundary encoding manifold, then
    normalize with the location target statistics before passing the source to the flow.
    """
    raw_features = base_features_raw.to(device=device, dtype=torch.float32)
    N = raw_features.shape[0]
    names = normalization.base_feature_names
    ux_idx = names.index("ballistic_exit_unit_x") if "ballistic_exit_unit_x" in names else None
    uy_idx = names.index("ballistic_exit_unit_y") if "ballistic_exit_unit_y" in names else None
    if ux_idx is not None and uy_idx is not None:
        ux = raw_features[:, ux_idx]
        uy = raw_features[:, uy_idx]
        mean = torch.stack([ux, uy], dim=1)
    else:
        mean = torch.zeros(N, 2, device=device)
    _gen = generator if (generator is not None and generator.device == torch.device(device)) else None
    noise = torch.randn(N, 2, device=device, generator=_gen) * 0.5
    raw_source = mean + noise
    norm = torch.clamp(torch.norm(raw_source, dim=1, keepdim=True), min=1e-6)
    raw_source = raw_source / norm
    target_mean = torch.as_tensor(normalization.location_target_mean, dtype=torch.float32, device=device)
    target_std = torch.as_tensor(normalization.location_target_std, dtype=torch.float32, device=device)
    return (raw_source - target_mean) / target_std


def _source_kinematics(
    n: int,
    target_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Source distribution for kinematics in normalized target space."""
    _gen = generator if (generator is not None and generator.device == torch.device(device)) else None
    return torch.randn(n, target_dim, device=device, generator=_gen)


def _sample_stage_source(
    stage: str,
    batch_size: int,
    target_dim: int,
    normalization: "FactorizedNormalizationBundle",
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    base_features_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    if stage == LOCATION_STAGE:
        if base_features_raw is None:
            raise ValueError("Location-stage source sampling requires per-sample raw base features.")
        return _source_location(
            base_features_raw,
            normalization,
            device=device,
            generator=generator,
        )
    return _source_kinematics(batch_size, target_dim, device=device, generator=generator)


def _decode_kinematics_outputs(
    kin_raw: np.ndarray,
    target_names: tuple[str, ...],
    exit_face_can: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    target_index = {name: idx for idx, name in enumerate(target_names)}
    num_rows = kin_raw.shape[0]

    total_track = np.power(10.0, kin_raw[:, target_index["total_track_length_log10"]].astype(np.float64))
    weighted_track = np.power(10.0, kin_raw[:, target_index["weighted_track_length_log10"]].astype(np.float64))

    angular_can = np.zeros((num_rows, len(ANGULAR_MOMENT_NAMES)), dtype=np.float64)
    for idx, name in enumerate(ANGULAR_MOMENT_NAMES):
        if name in target_index:
            angular_can[:, idx] = kin_raw[:, target_index[name]].astype(np.float64)

    legendre_can = np.zeros((num_rows, len(LEGENDRE_COEFF_NAMES)), dtype=np.float64)
    for idx, name in enumerate(LEGENDRE_COEFF_NAMES):
        if name in target_index:
            legendre_can[:, idx] = kin_raw[:, target_index[name]].astype(np.float64)

    if "final_dir_u" in target_index and "final_dir_v" in target_index:
        final_u_can, final_v_can, direction_diag = _normalize_exit_direction(
            exit_face_can,
            kin_raw[:, target_index["final_dir_u"]].astype(np.float64),
            kin_raw[:, target_index["final_dir_v"]].astype(np.float64),
        )
        return (
            final_u_can.astype(np.float64),
            final_v_can.astype(np.float64),
            total_track,
            weighted_track,
            angular_can,
            legendre_can,
            direction_diag,
        )

    if "final_dir_angle" in target_index:
        final_u_can = np.zeros(num_rows, dtype=np.float64)
        final_v_can = np.zeros(num_rows, dtype=np.float64)
        final_dir_angles = kin_raw[:, target_index["final_dir_angle"]].astype(np.float64)
        for idx in range(num_rows):
            final_u_can[idx], final_v_can[idx] = _decode_exit_direction_angle(
                int(exit_face_can[idx]),
                float(final_dir_angles[idx]),
            )
        return final_u_can, final_v_can, total_track, weighted_track, angular_can, legendre_can, {}

    raise ValueError(
        "Factorized kinematics targets must include either ('final_dir_u', 'final_dir_v') "
        "or the legacy 'final_dir_angle'."
    )


# ---------------------------------------------------------------------------
# Flow training / evaluation
# ---------------------------------------------------------------------------

def _train_flow_model(
    stage: str,
    inputs_train: np.ndarray,
    targets_train: np.ndarray,
    inputs_val: np.ndarray,
    targets_val: np.ndarray,
    *,
    config: FactorizedTrainingConfig,
    device: torch.device,
    output_dir: Path,
    normalization: "FactorizedNormalizationBundle",
    base_raw_train: np.ndarray | None = None,
    base_raw_val: np.ndarray | None = None,
) -> dict[str, object]:
    """Train a ConditionalFlowModel with CFM loss and physics-informed sources."""
    condition_dim = inputs_train.shape[1]
    target_dim = targets_train.shape[1]
    model = ConditionalFlowModel(
        condition_dim=condition_dim,
        target_dim=target_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        ode_steps=config.flow_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _make_loader(
        inputs_train,
        targets_train,
        batch_size=config.batch_size,
        shuffle=True,
        aux=base_raw_train if stage == LOCATION_STAGE else None,
    )
    val_loader = _make_loader(
        inputs_val,
        targets_val,
        batch_size=max(config.batch_size, 256),
        shuffle=False,
        aux=base_raw_val if stage == LOCATION_STAGE else None,
    )

    best_val = math.inf
    history_train: list[float] = []
    history_val: list[float] = []

    for _epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch_cond, batch_x1, batch_aux in train_loader:
            batch_cond = batch_cond.to(device)
            batch_x1 = batch_x1.to(device)
            bs = batch_cond.shape[0]
            x0 = _sample_stage_source(
                stage,
                bs,
                target_dim,
                normalization,
                device=device,
                base_features_raw=batch_aux if batch_aux.shape[1] > 0 else None,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.flow_matching_loss(x0, batch_x1, batch_cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.item()) * bs
            seen += bs
        train_loss = running / max(1, seen)

        # Validation loss
        model.eval()
        val_running = 0.0
        val_seen = 0
        with torch.no_grad():
            for vbatch_cond, vbatch_x1, vbatch_aux in val_loader:
                vbatch_cond = vbatch_cond.to(device)
                vbatch_x1 = vbatch_x1.to(device)
                vbs = vbatch_cond.shape[0]
                vx0 = _sample_stage_source(
                    stage,
                    vbs,
                    target_dim,
                    normalization,
                    device=device,
                    base_features_raw=vbatch_aux if vbatch_aux.shape[1] > 0 else None,
                )
                vloss = model.flow_matching_loss(vx0, vbatch_x1, vbatch_cond)
                val_running += float(vloss.item()) * vbs
                val_seen += vbs
        val_loss = val_running / max(1, val_seen)
        history_train.append(train_loss)
        history_val.append(val_loss)
        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / stage / "models" / "best_model.pt",
                model,
                metadata={"stage": stage, "variant": "factorized_flow"},
            )
    save_checkpoint(
        output_dir / stage / "models" / "final_model.pt",
        model,
        metadata={"stage": stage, "variant": "factorized_flow"},
    )
    best_model, _, _meta = load_checkpoint(output_dir / stage / "models" / "best_model.pt", map_location=device)
    best_model = best_model.to(device)

    # Compute validation metrics via sampling (use source → integrate)
    best_model.eval()
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for vbatch_cond, vbatch_x1, vbatch_aux in val_loader:
            vbatch_cond = vbatch_cond.to(device)
            vbs = vbatch_cond.shape[0]
            vx0 = _sample_stage_source(
                stage,
                vbs,
                target_dim,
                normalization,
                device=device,
                base_features_raw=vbatch_aux if vbatch_aux.shape[1] > 0 else None,
            )
            pred = best_model.sample(vx0, vbatch_cond)
            pred_batches.append(pred.cpu().numpy())
            target_batches.append(vbatch_x1.numpy())
    pred_val = np.concatenate(pred_batches, axis=0)
    target_val = np.concatenate(target_batches, axis=0)

    metrics: dict[str, object] = {"best_val_cfm_loss": float(best_val)}
    if stage == LOCATION_STAGE:
        pred_raw = normalization.denormalize_location_targets(pred_val)
        target_raw = normalization.denormalize_location_targets(target_val)
        pred_norm = np.linalg.norm(pred_raw, axis=1)
        target_norm = np.linalg.norm(target_raw, axis=1)
        metrics["mae_exit_unit_x"] = float(np.mean(np.abs(pred_raw[:, 0] - target_raw[:, 0])))
        metrics["mae_exit_unit_y"] = float(np.mean(np.abs(pred_raw[:, 1] - target_raw[:, 1])))
        metrics["mean_pred_unit_norm"] = float(np.mean(pred_norm))
        metrics["mean_target_unit_norm"] = float(np.mean(target_norm))
    else:
        pred_raw = normalization.denormalize_kinematics_targets(pred_val)
        target_raw = normalization.denormalize_kinematics_targets(target_val)
        abs_err = np.mean(np.abs(pred_raw - target_raw), axis=0)
        metrics.update({f"mae_{name}": float(value) for name, value in zip(_KINEMATICS_TARGET_COLUMNS, abs_err)})
    metrics_dir = output_dir / stage / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    np.savez(metrics_dir / "history.npz", train=np.asarray(history_train), val=np.asarray(history_val))
    return {"metrics": metrics, "best_val_loss": float(best_val)}


def run_factorized_training(
    config: FactorizedTrainingConfig,
    *,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, object]:
    from physics.mc_data_gen import get_2d_factorized_hierarchical_steady_scattered_only_data

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    if dataframe is None:
        dataframe = get_2d_factorized_hierarchical_steady_scattered_only_data(
            source=config.source,
            num_neutrons=config.num_neutrons,
            sigma_s=config.sigma_s,
            sigma_a=config.sigma_a,
            distribution=config.distribution,
            min_width=config.min_width,
            max_width=config.max_width,
            min_height=config.min_height,
            max_height=config.max_height,
            sampling=config.sampling,
        )
    prepared = prepare_factorized_dataframe(dataframe, source=config.source)
    train_df, val_df = _split_dataframe(prepared.dataframe, val_ratio=config.val_ratio, seed=config.seed)
    normalization = fit_factorized_normalization(
        FactorizedPreparedData(
            source=prepared.source,
            dataframe=train_df,
            base_columns=prepared.base_columns,
            kinematics_feature_columns=prepared.kinematics_feature_columns,
        )
    )
    run_name = config.run_name or f"factorized_{config.source}_{config.num_neutrons}"
    run_dir = Path(config.output_root) / run_name
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    normalization.save(run_dir / "metadata")
    (run_dir / "metadata" / "run_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))

    location_train_inputs = normalization.normalize_base_features(train_df.loc[:, prepared.base_columns].to_numpy(dtype=np.float32, copy=True))
    location_val_inputs = normalization.normalize_base_features(val_df.loc[:, prepared.base_columns].to_numpy(dtype=np.float32, copy=True))
    location_train_targets = normalization.normalize_location_targets(train_df.loc[:, _LOCATION_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    location_val_targets = normalization.normalize_location_targets(val_df.loc[:, _LOCATION_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    # Raw base features (un-normalized) for physics-informed source sampling
    base_raw_train = train_df.loc[:, prepared.base_columns].to_numpy(dtype=np.float32, copy=True)
    base_raw_val = val_df.loc[:, prepared.base_columns].to_numpy(dtype=np.float32, copy=True)
    location_result = _train_flow_model(
        LOCATION_STAGE,
        location_train_inputs,
        location_train_targets,
        location_val_inputs,
        location_val_targets,
        config=config,
        device=device,
        output_dir=run_dir,
        normalization=normalization,
        base_raw_train=base_raw_train,
        base_raw_val=base_raw_val,
    )

    kin_train_inputs = normalization.normalize_kinematics_features(train_df.loc[:, prepared.kinematics_feature_columns].to_numpy(dtype=np.float32, copy=True))
    kin_val_inputs = normalization.normalize_kinematics_features(val_df.loc[:, prepared.kinematics_feature_columns].to_numpy(dtype=np.float32, copy=True))
    kin_train_targets = normalization.normalize_kinematics_targets(train_df.loc[:, _KINEMATICS_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    kin_val_targets = normalization.normalize_kinematics_targets(val_df.loc[:, _KINEMATICS_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    kinematics_result = _train_flow_model(
        KINEMATICS_STAGE,
        kin_train_inputs,
        kin_train_targets,
        kin_val_inputs,
        kin_val_targets,
        config=config,
        device=device,
        output_dir=run_dir,
        normalization=normalization,
    )
    result = {
        "run_dir": str(run_dir),
        "location": location_result,
        "kinematics": kinematics_result,
    }
    (run_dir / "metadata" / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


class FactorizedSteady2DSampler:
    def __init__(
        self,
        *,
        source_type: str,
        location_model: ConditionalFlowModel,
        kinematics_model: ConditionalFlowModel,
        normalization: FactorizedNormalizationBundle,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.source_type = source_type
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.location_model = location_model.to(self.device).eval()
        self.kinematics_model = kinematics_model.to(self.device).eval()
        self.normalization = normalization

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        source_type: str,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> "FactorizedSteady2DSampler":
        import torch
        # Resolve "auto" to a concrete device string before torch.load
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        resolved = Path(run_dir)
        normalization = FactorizedNormalizationBundle.load(resolved / "metadata" / "normalization.json")
        if normalization.source != str(source_type):
            raise ValueError(
                f"Factorized run {resolved} was trained for source={normalization.source!r}, not {source_type!r}."
            )
        location_model, _cfg_loc, meta_loc = load_checkpoint(resolved / LOCATION_STAGE / "models" / "best_model.pt", map_location=device)
        kinematics_model, _cfg_kin, meta_kin = load_checkpoint(resolved / KINEMATICS_STAGE / "models" / "best_model.pt", map_location=device)
        if not isinstance(location_model, ConditionalFlowModel) or not isinstance(kinematics_model, ConditionalFlowModel):
            raise TypeError("Factorized checkpoints must contain ConditionalFlowModel models.")
        if str(meta_loc.get("stage", "")) != LOCATION_STAGE:
            raise ValueError(f"Expected stage={LOCATION_STAGE!r} in {resolved / LOCATION_STAGE / 'models' / 'best_model.pt'}")
        if str(meta_kin.get("stage", "")) != KINEMATICS_STAGE:
            raise ValueError(f"Expected stage={KINEMATICS_STAGE!r} in {resolved / KINEMATICS_STAGE / 'models' / 'best_model.pt'}")
        return cls(
            source_type=source_type,
            location_model=location_model,
            kinematics_model=kinematics_model,
            normalization=normalization,
            device=device,
            batch_size=batch_size,
        )

    def _predict_batched(
        self,
        model: ConditionalFlowModel,
        cond_values: np.ndarray,
        *,
        generator: torch.Generator,
        stage: str,
        base_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample from the flow model using physics-informed source distributions."""
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.array(cond_values, dtype=np.float32, copy=True))),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        device_obj = torch.device(self.device)
        batches: list[np.ndarray] = []
        offset = 0
        with torch.no_grad():
            for (batch_cond,) in loader:
                bs = batch_cond.shape[0]
                batch_cond = batch_cond.to(device_obj)
                raw_slice = None if base_raw is None else torch.from_numpy(base_raw[offset : offset + bs])
                x0 = _sample_stage_source(
                    stage,
                    bs,
                    model.target_dim,
                    self.normalization,
                    device=device_obj,
                    generator=generator,
                    base_features_raw=raw_slice,
                )
                _gen = generator if (generator is not None and generator.device == device_obj) else None
                pred = model.sample(x0, batch_cond, generator=_gen)
                batches.append(pred.cpu().numpy())
                offset += bs
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, model.target_dim), dtype=np.float32)

    def sample_scattered_batch(
        self,
        *,
        width: np.ndarray,
        height: np.ndarray,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        ballistic_length: np.ndarray,
        ballistic_face: np.ndarray,
        ballistic_coord: np.ndarray,
        ballistic_exit_x: np.ndarray,
        ballistic_exit_y: np.ndarray,
        scatter_before_exit_prob: np.ndarray,
        sigma_s: np.ndarray,
        sigma_a: np.ndarray,
        seed: int | None = None,
    ) -> FactorizedSampleBatch:
        width = np.asarray(width, dtype=np.float64).reshape(-1)
        height = np.asarray(height, dtype=np.float64).reshape(-1)
        initial_x = np.asarray(initial_x, dtype=np.float64).reshape(-1)
        initial_y = np.asarray(initial_y, dtype=np.float64).reshape(-1)
        initial_u = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        initial_v = np.asarray(initial_v, dtype=np.float64).reshape(-1)
        ballistic_length = np.asarray(ballistic_length, dtype=np.float64).reshape(-1)
        ballistic_face = np.asarray(ballistic_face, dtype=np.int64).reshape(-1)
        ballistic_coord = np.asarray(ballistic_coord, dtype=np.float64).reshape(-1)
        ballistic_exit_x = np.asarray(ballistic_exit_x, dtype=np.float64).reshape(-1)
        ballistic_exit_y = np.asarray(ballistic_exit_y, dtype=np.float64).reshape(-1)
        scatter_before_exit_prob = np.asarray(scatter_before_exit_prob, dtype=np.float64).reshape(-1)
        sigma_s = np.asarray(sigma_s, dtype=np.float64).reshape(-1)
        sigma_a = np.asarray(sigma_a, dtype=np.float64).reshape(-1)
        num_rows = width.shape[0]
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device if self.device.startswith("cuda") else "cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()

        base_rows: list[dict[str, object]] = []
        canonical_states: list[CanonicalizedBoundaryState2D | CanonicalizedInternalState2D] = []
        for idx in range(num_rows):
            base_row, canonical_state = _canonicalize_runtime_state(
                self.source_type,
                width=float(width[idx]),
                height=float(height[idx]),
                initial_x=float(initial_x[idx]),
                initial_y=float(initial_y[idx]),
                initial_u=float(initial_u[idx]),
                initial_v=float(initial_v[idx]),
                ballistic_length=float(ballistic_length[idx]),
                ballistic_face=int(ballistic_face[idx]),
                ballistic_coord=float(ballistic_coord[idx]),
                ballistic_exit_x=float(ballistic_exit_x[idx]),
                ballistic_exit_y=float(ballistic_exit_y[idx]),
                scatter_before_exit_prob=float(scatter_before_exit_prob[idx]),
                sigma_s=float(sigma_s[idx]),
                sigma_a=float(sigma_a[idx]),
            )
            base_rows.append(base_row)
            canonical_states.append(canonical_state)
        prepared = pd.DataFrame(base_rows)
        base_raw = prepared.loc[:, self.normalization.base_feature_names].to_numpy(dtype=np.float32, copy=True)
        base_inputs = self.normalization.normalize_base_features(base_raw)
        location_norm = self._predict_batched(
            self.location_model, base_inputs, generator=generator,
            stage=LOCATION_STAGE, base_raw=base_raw,
        )
        location_raw = self.normalization.denormalize_location_targets(location_norm)
        location_vec = np.asarray(location_raw, dtype=np.float64)
        location_norms = np.linalg.norm(location_vec, axis=1, keepdims=True)
        location_norms = np.where(location_norms < EPSILON, 1.0, location_norms)
        location_vec /= location_norms

        exit_x_can = np.zeros(num_rows, dtype=np.float64)
        exit_y_can = np.zeros(num_rows, dtype=np.float64)
        exit_face_can = np.zeros(num_rows, dtype=np.int64)
        exit_coord_can = np.zeros(num_rows, dtype=np.float64)
        exit_coord_norm = np.zeros(num_rows, dtype=np.float64)
        for idx in range(num_rows):
            x_can, y_can, face_can, coord_can = decode_boundary_unit_vector(
                float(location_vec[idx, 0]),
                float(location_vec[idx, 1]),
                float(prepared.iloc[idx]["width"]),
                float(prepared.iloc[idx]["height"]),
            )
            exit_x_can[idx] = x_can
            exit_y_can[idx] = y_can
            exit_face_can[idx] = face_can
            exit_coord_can[idx] = coord_can
            denom = float(prepared.iloc[idx]["height"]) if face_can in (FACE_LEFT, FACE_RIGHT) else float(prepared.iloc[idx]["width"])
            exit_coord_norm[idx] = coord_can / max(denom, EPSILON)
        face_one_hot, face_names = _face_one_hot(exit_face_can, prefix="exit_face_is")
        kin_frame = prepared.copy()
        kin_frame["exit_unit_x"] = location_vec[:, 0]
        kin_frame["exit_unit_y"] = location_vec[:, 1]
        kin_frame["exit_face_coord_norm"] = exit_coord_norm
        for col_name, col_values in zip(face_names, face_one_hot.T):
            kin_frame[col_name] = col_values
        kin_inputs = self.normalization.normalize_kinematics_features(
            kin_frame.loc[:, self.normalization.kinematics_feature_names].to_numpy(dtype=np.float32, copy=True)
        )
        kin_norm = self._predict_batched(
            self.kinematics_model, kin_inputs, generator=generator,
            stage=KINEMATICS_STAGE,
        )
        kin_raw = self.normalization.denormalize_kinematics_targets(kin_norm)
        final_u_can, final_v_can, total_track, weighted_track, angular_can, legendre_can, direction_diag = _decode_kinematics_outputs(
            kin_raw,
            self.normalization.kinematics_target_names,
            exit_face_can,
        )

        x_out = np.zeros(num_rows, dtype=np.float32)
        y_out = np.zeros(num_rows, dtype=np.float32)
        u_out = np.zeros(num_rows, dtype=np.float32)
        v_out = np.zeros(num_rows, dtype=np.float32)
        exit_face = np.zeros(num_rows, dtype=np.int64)
        pos_on_face = np.zeros(num_rows, dtype=np.float32)
        angular_out = np.zeros((num_rows, len(ANGULAR_MOMENT_NAMES)), dtype=np.float32)
        legendre_out = np.zeros((num_rows, len(LEGENDRE_COEFF_NAMES)), dtype=np.float32)
        for idx in range(num_rows):
            row = prepared.iloc[idx]
            canonical_state = canonical_states[idx]
            x_phys, y_phys = inverse_transform_point(
                canonical_state.transform,
                float(width[idx]),
                float(height[idx]),
                float(row["width"]),
                float(row["height"]),
                float(exit_x_can[idx]),
                float(exit_y_can[idx]),
            )
            u_phys, v_phys = inverse_transform_direction(canonical_state.transform, float(final_u_can[idx]), float(final_v_can[idx]))
            face_phys = classify_boundary_face(x_phys, y_phys, float(width[idx]), float(height[idx]))
            x_phys = 0.0 if face_phys == FACE_LEFT else float(width[idx]) if face_phys == FACE_RIGHT else x_phys
            y_phys = float(height[idx]) if face_phys == FACE_TOP else 0.0 if face_phys == FACE_BOTTOM else y_phys
            x_out[idx] = float(np.clip(x_phys, 0.0, float(width[idx])))
            y_out[idx] = float(np.clip(y_phys, 0.0, float(height[idx])))
            u_out[idx] = float(u_phys)
            v_out[idx] = float(v_phys)
            exit_face[idx] = face_phys
            pos_on_face[idx] = float(face_local_coordinate(face_phys, x_out[idx], y_out[idx]))
            angular_out[idx] = inverse_transform_angular_moments(canonical_state.transform, angular_can[idx]).astype(np.float32)
            legendre_out[idx] = inverse_transform_legendre_coeffs(canonical_state.transform, legendre_can[idx]).astype(np.float32)
        diagnostics = {
            "location_unit_norm_mean": float(np.mean(np.linalg.norm(location_vec, axis=1))) if num_rows else 0.0,
            **{f"{key}_given_scatter": float(value) for key, value in direction_diag.items()},
        }
        return FactorizedSampleBatch(
            x_out=x_out,
            y_out=y_out,
            u_out=u_out,
            v_out=v_out,
            track_length=np.maximum(total_track, 0.0).astype(np.float32),
            exit_face=exit_face,
            pos_on_face=pos_on_face,
            weighted_track_length=np.maximum(weighted_track, 0.0).astype(np.float32),
            angular_moments=angular_out,
            legendre_coeffs=legendre_out,
            diagnostics=diagnostics,
        )
