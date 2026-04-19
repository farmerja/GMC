from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _normalize_mode(mode: str) -> str:
    return str(mode).strip().lower().replace(" ", "_")


def resolve_student_run_dir(
    task_name: str,
    source_type: str,
    num_steps: int,
    *,
    scattering_mode: str = "isotropic",
    artifact_root: str = "ml/student_artifacts",
) -> Path:
    return (
        Path(artifact_root)
        / _normalize_mode(task_name)
        / _normalize_mode(source_type)
        / _normalize_mode(scattering_mode)
        / f"{int(num_steps)}_step"
    )


def resolve_student_checkpoint_path(
    task_name: str,
    source_type: str,
    num_steps: int,
    *,
    checkpoint_name: str = "best_student.pth",
    scattering_mode: str = "isotropic",
    artifact_root: str = "ml/student_artifacts",
) -> Path:
    return resolve_student_run_dir(
        task_name,
        source_type,
        num_steps,
        scattering_mode=scattering_mode,
        artifact_root=artifact_root,
    ) / checkpoint_name


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
