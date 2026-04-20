"""Dataset and schema helpers for the standalone hierarchical steady-2D workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

from .schema import STAGE_FACE, STAGE_JOINT, STAGE_TRACK
from physics.hierarchical_steady_2d import FACE_LEFT, FACE_NAMES, FACE_RIGHT, direction_is_outward_for_face_2d

if TYPE_CHECKING:
    from .normalization import HierarchicalNormalizationBundle


FACE_TASK = STAGE_FACE
JOINT_TASK = STAGE_JOINT
TRACK_TASK = STAGE_TRACK
TASK_SEQUENCE = (FACE_TASK, JOINT_TASK, TRACK_TASK)

NUM_FACES = 4
FACE_NAME_TO_ID = {name: idx for idx, name in enumerate(FACE_NAMES)}

BASE_CONTINUOUS_COLUMNS = (
    "width_log10",
    "height_log10",
    "initial_pos_x_norm",
    "initial_pos_y_norm",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length_log10",
    "ballistic_face_coord_local_norm",
    "scatter_before_exit_prob",
)
JOINT_TARGET_COLUMNS = (
    "exit_face_coord_local_norm",
    "final_dir_u",
    "final_dir_v",
)
TRACK_TARGET_COLUMNS = ("total_track_length_log10",)

REQUIRED_CONDITION_COLUMNS = (
    "width",
    "height",
    "initial_pos_x",
    "initial_pos_y",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length",
    "ballistic_face",
    "ballistic_face_coord_local",
    "scatter_before_exit_prob",
)

REQUIRED_SCATTERED_ONLY_COLUMNS = (
    "width",
    "height",
    "initial_pos_x",
    "initial_pos_y",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length",
    "ballistic_face",
    "scatter_before_exit_prob",
    "exit_face",
    "final_dir_u",
    "final_dir_v",
    "total_track_length",
)


@dataclass(frozen=True)
class HierarchicalTaskArrays:
    task: str
    inputs: np.ndarray
    targets: np.ndarray
    input_names: tuple[str, ...]
    target_names: tuple[str, ...]
    target_kind: str


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for hierarchical steady-2D workflow: {missing}")


def _face_lengths(face_values: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    return np.where(
        (face_values == FACE_LEFT) | (face_values == FACE_RIGHT),
        height,
        width,
    )


def _maybe_add_norm_column(df: pd.DataFrame, raw_col: str, norm_col: str, face_col: str | None = None) -> None:
    eps = 1e-12
    if norm_col in df.columns:
        return
    if raw_col not in df.columns:
        raise ValueError(f"Missing {raw_col!r}; cannot derive {norm_col!r}.")

    if face_col is None:
        if raw_col.endswith("_x") or raw_col.endswith("_x_local"):
            denom = df["width"].to_numpy(dtype=np.float64, copy=False)
        elif raw_col.endswith("_y") or raw_col.endswith("_y_local"):
            denom = df["height"].to_numpy(dtype=np.float64, copy=False)
        else:
            raise ValueError(f"Cannot infer normalization denominator for column {raw_col!r}.")
    else:
        face = df[face_col].to_numpy(dtype=np.int64, copy=False)
        denom = _face_lengths(
            face,
            df["width"].to_numpy(dtype=np.float64, copy=False),
            df["height"].to_numpy(dtype=np.float64, copy=False),
        )
    df[norm_col] = (df[raw_col].to_numpy(dtype=np.float64, copy=False) / (denom + eps)).clip(0.0, 1.0)


def _compute_outward_compatibility(exit_face: np.ndarray, final_dir_u: np.ndarray, final_dir_v: np.ndarray) -> np.ndarray:
    compatible = np.zeros(exit_face.shape[0], dtype=np.int64)
    for idx, (face, u_val, v_val) in enumerate(zip(exit_face, final_dir_u, final_dir_v)):
        compatible[idx] = int(direction_is_outward_for_face_2d(int(face), float(u_val), float(v_val)))
    return compatible


def _validate_face_ids(face_values: np.ndarray, *, label: str) -> np.ndarray:
    face_ids = np.asarray(face_values, dtype=np.int64)
    if face_ids.ndim != 1:
        raise ValueError(f"{label} must be a 1D array of face ids; got shape {face_ids.shape}.")
    if np.any((face_ids < 0) | (face_ids >= NUM_FACES)):
        raise ValueError(f"{label} must only contain ids in [0, {NUM_FACES - 1}].")
    return face_ids


def _face_one_hot(face_values: np.ndarray, *, prefix: str) -> tuple[np.ndarray, tuple[str, ...]]:
    face_ids = _validate_face_ids(face_values, label=prefix)
    one_hot = np.eye(NUM_FACES, dtype=np.float32)[face_ids]
    names = tuple(f"{prefix}_{name}" for name in FACE_NAMES)
    return one_hot, names


def prepare_hierarchical_scattered_only_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich scattered-only hierarchical data for downstream tasks.

    Model inputs intentionally use only the branch-start information available
    under ``Law(Y | X0, E)``: geometry, initial state, and ballistic metadata.
    First-scatter fields remain available as metadata/debug information but are
    not part of the learned task conditions.
    """
    _require_columns(df, REQUIRED_SCATTERED_ONLY_COLUMNS)
    prepared = df.copy()

    if "scatter_before_exit" in prepared.columns:
        if not bool((prepared["scatter_before_exit"] == 1).all()):
            raise ValueError("Hierarchical scattered-only datasets must satisfy scatter_before_exit == 1 for all rows.")
    else:
        prepared["scatter_before_exit"] = 1

    _maybe_add_norm_column(prepared, "initial_pos_x", "initial_pos_x_norm")
    _maybe_add_norm_column(prepared, "initial_pos_y", "initial_pos_y_norm")
    _maybe_add_norm_column(prepared, "ballistic_face_coord_local", "ballistic_face_coord_local_norm", face_col="ballistic_face")
    _maybe_add_norm_column(prepared, "exit_face_coord_local", "exit_face_coord_local_norm", face_col="exit_face")

    prepared["width_log10"] = np.log10(np.maximum(prepared["width"].to_numpy(dtype=np.float64, copy=False), 1e-12))
    prepared["height_log10"] = np.log10(np.maximum(prepared["height"].to_numpy(dtype=np.float64, copy=False), 1e-12))
    prepared["ballistic_length_log10"] = np.log10(
        np.maximum(prepared["ballistic_length"].to_numpy(dtype=np.float64, copy=False), 1e-12)
    )
    prepared["total_track_length_log10"] = np.log10(
        np.maximum(prepared["total_track_length"].to_numpy(dtype=np.float64, copy=False), 1e-12)
    )

    prepared["ballistic_face"] = prepared["ballistic_face"].astype(np.int64)
    prepared["exit_face"] = prepared["exit_face"].astype(np.int64)
    if np.any((prepared["ballistic_face"] < 0) | (prepared["ballistic_face"] >= NUM_FACES)):
        raise ValueError("ballistic_face must be in [0, 3].")
    if np.any((prepared["exit_face"] < 0) | (prepared["exit_face"] >= NUM_FACES)):
        raise ValueError("exit_face must be in [0, 3].")

    if "final_dir_outward_compatible" not in prepared.columns:
        prepared["final_dir_outward_compatible"] = _compute_outward_compatibility(
            prepared["exit_face"].to_numpy(dtype=np.int64, copy=False),
            prepared["final_dir_u"].to_numpy(dtype=np.float64, copy=False),
            prepared["final_dir_v"].to_numpy(dtype=np.float64, copy=False),
        )

    if not bool((prepared["final_dir_outward_compatible"] == 1).all()):
        raise ValueError("final_dir_u/final_dir_v must be outward-compatible with exit_face.")

    prepared["ballistic_face_name"] = prepared["ballistic_face"].map(dict(enumerate(FACE_NAMES)))
    prepared["exit_face_name"] = prepared["exit_face"].map(dict(enumerate(FACE_NAMES)))
    return prepared


def prepare_hierarchical_condition_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich branch-start conditioning data for rollout/inference."""
    _require_columns(df, REQUIRED_CONDITION_COLUMNS)
    prepared = df.copy()

    _maybe_add_norm_column(prepared, "initial_pos_x", "initial_pos_x_norm")
    _maybe_add_norm_column(prepared, "initial_pos_y", "initial_pos_y_norm")
    _maybe_add_norm_column(prepared, "ballistic_face_coord_local", "ballistic_face_coord_local_norm", face_col="ballistic_face")

    prepared["width_log10"] = np.log10(np.maximum(prepared["width"].to_numpy(dtype=np.float64, copy=False), 1e-12))
    prepared["height_log10"] = np.log10(np.maximum(prepared["height"].to_numpy(dtype=np.float64, copy=False), 1e-12))
    prepared["ballistic_length_log10"] = np.log10(
        np.maximum(prepared["ballistic_length"].to_numpy(dtype=np.float64, copy=False), 1e-12)
    )

    prepared["ballistic_face"] = prepared["ballistic_face"].astype(np.int64)
    if np.any((prepared["ballistic_face"] < 0) | (prepared["ballistic_face"] >= NUM_FACES)):
        raise ValueError("ballistic_face must be in [0, 3].")

    prepared["ballistic_face_name"] = prepared["ballistic_face"].map(dict(enumerate(FACE_NAMES)))
    return prepared


def base_condition_column_names() -> tuple[str, ...]:
    _, face_names = _face_one_hot(np.zeros(1, dtype=np.int64), prefix="ballistic_face_is")
    return BASE_CONTINUOUS_COLUMNS + face_names


def _build_base_conditions(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
) -> tuple[np.ndarray, tuple[str, ...]]:
    base_cont = prepared_df.loc[:, BASE_CONTINUOUS_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    base_cont = normalization.normalize_base_features(base_cont)
    ballistic_face = prepared_df["ballistic_face"].to_numpy(dtype=np.int64, copy=False)
    ballistic_one_hot, ballistic_names = _face_one_hot(ballistic_face, prefix="ballistic_face_is")
    inputs = np.concatenate([base_cont, ballistic_one_hot], axis=1).astype(np.float32, copy=False)
    return inputs.astype(np.float32, copy=False), BASE_CONTINUOUS_COLUMNS + ballistic_names


def build_face_task_arrays(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
) -> HierarchicalTaskArrays:
    inputs, input_names = _build_base_conditions(prepared_df, normalization)
    targets = prepared_df["exit_face"].to_numpy(dtype=np.int64, copy=False)
    return HierarchicalTaskArrays(
        task=FACE_TASK,
        inputs=inputs,
        targets=targets,
        input_names=input_names,
        target_names=("exit_face",),
        target_kind="classification",
    )


def build_face_inputs(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
) -> tuple[np.ndarray, tuple[str, ...]]:
    return _build_base_conditions(prepared_df, normalization)


def build_joint_task_arrays(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
    *,
    conditioning_face_ids: Sequence[int] | np.ndarray | None = None,
) -> HierarchicalTaskArrays:
    base_inputs, base_names = _build_base_conditions(prepared_df, normalization)
    if conditioning_face_ids is None:
        conditioning_face = prepared_df["exit_face"].to_numpy(dtype=np.int64, copy=False)
    else:
        conditioning_face = _validate_face_ids(np.asarray(conditioning_face_ids), label="conditioning_face_ids")
        if conditioning_face.shape[0] != len(prepared_df):
            raise ValueError("conditioning_face_ids length must match the number of rows.")

    face_one_hot, face_names = _face_one_hot(conditioning_face, prefix="condition_face_is")
    inputs = np.concatenate([base_inputs, face_one_hot], axis=1).astype(np.float32, copy=False)
    raw_targets = prepared_df.loc[:, JOINT_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    targets = normalization.normalize_joint_targets(raw_targets)
    return HierarchicalTaskArrays(
        task=JOINT_TASK,
        inputs=inputs.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        input_names=base_names + face_names,
        target_names=JOINT_TARGET_COLUMNS,
        target_kind="regression",
    )


def build_joint_inputs(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
    *,
    conditioning_face_ids: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    base_inputs, base_names = _build_base_conditions(prepared_df, normalization)
    conditioning_face = _validate_face_ids(np.asarray(conditioning_face_ids), label="conditioning_face_ids")
    if conditioning_face.shape[0] != len(prepared_df):
        raise ValueError("conditioning_face_ids length must match the number of rows.")

    face_one_hot, face_names = _face_one_hot(conditioning_face, prefix="condition_face_is")
    inputs = np.concatenate([base_inputs, face_one_hot], axis=1).astype(np.float32, copy=False)
    return inputs, base_names + face_names


def build_track_task_arrays(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
    *,
    conditioning_face_ids: Sequence[int] | np.ndarray | None = None,
    conditioning_joint_targets: np.ndarray | None = None,
) -> HierarchicalTaskArrays:
    base_inputs, base_names = _build_base_conditions(prepared_df, normalization)

    if conditioning_face_ids is None:
        conditioning_face = prepared_df["exit_face"].to_numpy(dtype=np.int64, copy=False)
    else:
        conditioning_face = _validate_face_ids(np.asarray(conditioning_face_ids), label="conditioning_face_ids")
        if conditioning_face.shape[0] != len(prepared_df):
            raise ValueError("conditioning_face_ids length must match the number of rows.")

    if conditioning_joint_targets is None:
        joint_context = prepared_df.loc[:, JOINT_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    else:
        joint_context = np.asarray(conditioning_joint_targets, dtype=np.float32)
        if joint_context.shape != (len(prepared_df), len(JOINT_TARGET_COLUMNS)):
            raise ValueError(
                "conditioning_joint_targets must have shape "
                f"({len(prepared_df)}, {len(JOINT_TARGET_COLUMNS)}); got {joint_context.shape}."
            )
    joint_context = normalization.normalize_joint_targets(joint_context)

    face_one_hot, face_names = _face_one_hot(conditioning_face, prefix="condition_face_is")
    joint_names = (
        "condition_exit_face_coord_local_norm",
        "condition_final_dir_u",
        "condition_final_dir_v",
    )
    inputs = np.concatenate([base_inputs, face_one_hot, joint_context], axis=1).astype(np.float32, copy=False)

    raw_targets = prepared_df.loc[:, TRACK_TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    targets = normalization.normalize_track_targets(raw_targets)
    return HierarchicalTaskArrays(
        task=TRACK_TASK,
        inputs=inputs.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        input_names=base_names + face_names + joint_names,
        target_names=TRACK_TARGET_COLUMNS,
        target_kind="regression",
    )


def build_track_inputs(
    prepared_df: pd.DataFrame,
    normalization: "HierarchicalNormalizationBundle",
    *,
    conditioning_face_ids: Sequence[int] | np.ndarray,
    conditioning_joint_targets: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    base_inputs, base_names = _build_base_conditions(prepared_df, normalization)

    conditioning_face = _validate_face_ids(np.asarray(conditioning_face_ids), label="conditioning_face_ids")
    if conditioning_face.shape[0] != len(prepared_df):
        raise ValueError("conditioning_face_ids length must match the number of rows.")

    joint_context = np.asarray(conditioning_joint_targets, dtype=np.float32)
    if joint_context.shape != (len(prepared_df), len(JOINT_TARGET_COLUMNS)):
        raise ValueError(
            "conditioning_joint_targets must have shape "
            f"({len(prepared_df)}, {len(JOINT_TARGET_COLUMNS)}); got {joint_context.shape}."
        )
    joint_context = normalization.normalize_joint_targets(joint_context)

    face_one_hot, face_names = _face_one_hot(conditioning_face, prefix="condition_face_is")
    joint_names = (
        "condition_exit_face_coord_local_norm",
        "condition_final_dir_u",
        "condition_final_dir_v",
    )
    inputs = np.concatenate([base_inputs, face_one_hot, joint_context], axis=1).astype(np.float32, copy=False)
    return inputs, base_names + face_names + joint_names
