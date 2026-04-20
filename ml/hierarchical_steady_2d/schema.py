from __future__ import annotations

from dataclasses import dataclass

from physics.hierarchical_steady_2d import FACE_NAMES


TASK_NAME = "hierarchical_steady_2d"
SCHEMA_VERSION = 1

STAGE_FACE = "face"
STAGE_JOINT = "joint"
STAGE_EXIT = "exit"
STAGE_TRACK = "track"
STAGES = (STAGE_FACE, STAGE_JOINT, STAGE_TRACK)
STAGE_ALIASES = {
    STAGE_EXIT: STAGE_JOINT,
}

FACE_LABELS = dict(enumerate(FACE_NAMES))
FACE_ORDER = tuple(FACE_LABELS.keys())

BASE_FEATURE_COLUMNS = (
    "width_log10",
    "height_log10",
    "initial_pos_x_norm",
    "initial_pos_y_norm",
    "initial_dir_u",
    "initial_dir_v",
    "ballistic_length_log10",
    "ballistic_face_coord_local_norm",
    "scatter_before_exit_prob",
    "ballistic_face_is_left",
    "ballistic_face_is_right",
    "ballistic_face_is_top",
    "ballistic_face_is_bottom",
)

EXIT_TARGET_COLUMNS = (
    "exit_face_coord_local_norm",
    "final_dir_u",
    "final_dir_v",
)

TRACK_TARGET_COLUMNS = ("total_track_length_log10",)


@dataclass(frozen=True)
class StageSpec:
    name: str
    task_type: str
    target_dim: int
    stratify_column: str
    condition_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


def normalize_stage_name(stage: str) -> str:
    normalized = str(stage).strip().lower()
    return STAGE_ALIASES.get(normalized, normalized)


def resolve_stage(stage: str) -> StageSpec:
    normalized = normalize_stage_name(stage)
    if normalized == STAGE_FACE:
        return StageSpec(
            name=normalized,
            task_type="classification",
            target_dim=4,
            stratify_column="exit_face",
            condition_columns=BASE_FEATURE_COLUMNS,
            target_columns=("exit_face",),
        )
    if normalized == STAGE_JOINT:
        return StageSpec(
            name=normalized,
            task_type="regression",
            target_dim=3,
            stratify_column="exit_face",
            condition_columns=BASE_FEATURE_COLUMNS + (
                "condition_face_is_left",
                "condition_face_is_right",
                "condition_face_is_top",
                "condition_face_is_bottom",
            ),
            target_columns=EXIT_TARGET_COLUMNS,
        )
    if normalized == STAGE_TRACK:
        return StageSpec(
            name=normalized,
            task_type="regression",
            target_dim=1,
            stratify_column="exit_face",
            condition_columns=BASE_FEATURE_COLUMNS + (
                "condition_face_is_left",
                "condition_face_is_right",
                "condition_face_is_top",
                "condition_face_is_bottom",
            ) + EXIT_TARGET_COLUMNS,
            target_columns=TRACK_TARGET_COLUMNS,
        )
    raise ValueError(
        f"Unsupported hierarchical stage '{stage}'. "
        f"Expected one of {STAGES} or aliases {tuple(STAGE_ALIASES)}."
    )
