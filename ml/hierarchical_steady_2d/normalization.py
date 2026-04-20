from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BASE_CONTINUOUS_COLUMNS, JOINT_TARGET_COLUMNS, TRACK_TARGET_COLUMNS
from .schema import SCHEMA_VERSION, STAGE_TRACK


EPSILON = 1.0e-12
MIN_STD = 1.0e-6


def _fit_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.where(std < MIN_STD, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def robust_log_stats(values, eps: float = EPSILON) -> tuple[float, float]:
    data = np.log10(np.asarray(values, dtype=np.float64) + eps)
    median = float(np.median(data))
    iqr = float(np.percentile(data, 75.0) - np.percentile(data, 25.0))
    scale = max(iqr / 1.349, 1.0e-6)
    return median, scale


def fit_stage_normalization(frame, *, stage: str) -> dict[str, float | int | str]:
    norm = {
        "schema_version": int(SCHEMA_VERSION),
        "stage": str(stage),
    }
    for key, column in (
        ("width", "width"),
        ("height", "height"),
        ("ballistic_optical_length", "ballistic_optical_length"),
    ):
        center, scale = robust_log_stats(frame[column].to_numpy(dtype=np.float64, copy=False))
        norm[f"{key}_center"] = center
        norm[f"{key}_scale"] = scale

    if str(stage) == STAGE_TRACK:
        center, scale = robust_log_stats(frame["total_track_length"].to_numpy(dtype=np.float64, copy=False))
        norm["total_track_length_center"] = center
        norm["total_track_length_scale"] = scale
    return norm


def encode_log_feature(values, norm: dict[str, float | int | str], key: str, eps: float = EPSILON):
    center = float(norm[f"{key}_center"])
    scale = max(float(norm[f"{key}_scale"]), 1.0e-12)
    return (np.log10(np.asarray(values, dtype=np.float64) + eps) - center) / scale


def decode_log_feature(values, norm: dict[str, float | int | str], key: str):
    center = float(norm[f"{key}_center"])
    scale = max(float(norm[f"{key}_scale"]), 1.0e-12)
    return np.power(10.0, np.asarray(values, dtype=np.float64) * scale + center)


def save_normalization(path: str | Path, norm: dict[str, float | int | str]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(norm, indent=2, sort_keys=True))


def load_normalization(path: str | Path) -> dict[str, float | int | str]:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class HierarchicalNormalizationBundle:
    source: str
    sigma_t: float
    base_feature_names: tuple[str, ...]
    base_feature_mean: np.ndarray
    base_feature_std: np.ndarray
    joint_target_names: tuple[str, ...]
    joint_target_mean: np.ndarray
    joint_target_std: np.ndarray
    track_target_names: tuple[str, ...]
    track_target_mean: np.ndarray
    track_target_std: np.ndarray
    version: int = SCHEMA_VERSION

    def normalize_base_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.base_feature_mean) / self.base_feature_std).astype(np.float32, copy=False)

    def denormalize_base_features(self, values: np.ndarray) -> np.ndarray:
        return (values * self.base_feature_std + self.base_feature_mean).astype(np.float32, copy=False)

    def normalize_joint_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.joint_target_mean) / self.joint_target_std).astype(np.float32, copy=False)

    def denormalize_joint_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.joint_target_std + self.joint_target_mean).astype(np.float32, copy=False)

    def normalize_track_targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.track_target_mean) / self.track_target_std).astype(np.float32, copy=False)

    def denormalize_track_targets(self, values: np.ndarray) -> np.ndarray:
        return (values * self.track_target_std + self.track_target_mean).astype(np.float32, copy=False)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "source": self.source,
            "sigma_t": float(self.sigma_t),
            "base_feature_names": list(self.base_feature_names),
            "joint_target_names": list(self.joint_target_names),
            "track_target_names": list(self.track_target_names),
            "base_feature_mean": self.base_feature_mean.tolist(),
            "base_feature_std": self.base_feature_std.tolist(),
            "joint_target_mean": self.joint_target_mean.tolist(),
            "joint_target_std": self.joint_target_std.tolist(),
            "track_target_mean": self.track_target_mean.tolist(),
            "track_target_std": self.track_target_std.tolist(),
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
            joint_target_mean=self.joint_target_mean,
            joint_target_std=self.joint_target_std,
            track_target_mean=self.track_target_mean,
            track_target_std=self.track_target_std,
        )
        return json_path, npz_path

    @classmethod
    def load(cls, json_path: str | Path, npz_path: str | Path | None = None) -> "HierarchicalNormalizationBundle":
        json_path = Path(json_path)
        payload = json.loads(json_path.read_text())
        npz_file = Path(npz_path) if npz_path is not None else json_path.with_suffix(".npz")
        arrays = np.load(npz_file)
        return cls(
            source=str(payload["source"]),
            sigma_t=float(payload["sigma_t"]),
            base_feature_names=tuple(payload["base_feature_names"]),
            base_feature_mean=np.asarray(arrays["base_feature_mean"], dtype=np.float32),
            base_feature_std=np.asarray(arrays["base_feature_std"], dtype=np.float32),
            joint_target_names=tuple(payload["joint_target_names"]),
            joint_target_mean=np.asarray(arrays["joint_target_mean"], dtype=np.float32),
            joint_target_std=np.asarray(arrays["joint_target_std"], dtype=np.float32),
            track_target_names=tuple(payload["track_target_names"]),
            track_target_mean=np.asarray(arrays["track_target_mean"], dtype=np.float32),
            track_target_std=np.asarray(arrays["track_target_std"], dtype=np.float32),
            version=int(payload["version"]),
        )


def fit_hierarchical_normalization(
    prepared_train_df: pd.DataFrame,
    *,
    source: str,
    sigma_t: float,
) -> HierarchicalNormalizationBundle:
    base_values = prepared_train_df.loc[:, BASE_CONTINUOUS_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    joint_values = prepared_train_df.loc[:, JOINT_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    track_values = prepared_train_df.loc[:, TRACK_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)

    base_mean, base_std = _fit_stats(base_values)
    joint_mean, joint_std = _fit_stats(joint_values)
    track_mean, track_std = _fit_stats(track_values)

    return HierarchicalNormalizationBundle(
        source=source,
        sigma_t=float(sigma_t),
        base_feature_names=tuple(BASE_CONTINUOUS_COLUMNS),
        base_feature_mean=base_mean,
        base_feature_std=base_std,
        joint_target_names=tuple(JOINT_TARGET_COLUMNS),
        joint_target_mean=joint_mean,
        joint_target_std=joint_std,
        track_target_names=tuple(TRACK_TARGET_COLUMNS),
        track_target_mean=track_mean,
        track_target_std=track_std,
    )
