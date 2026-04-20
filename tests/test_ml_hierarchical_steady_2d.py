from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from physics.hierarchical_steady_2d import FACE_BOTTOM, FACE_LEFT, FACE_RIGHT, FACE_TOP

from ml.hierarchical_steady_2d.data import (
    JOINT_TASK,
    TRACK_TASK,
    build_face_task_arrays,
    build_face_inputs,
    build_joint_task_arrays,
    build_joint_inputs,
    build_track_task_arrays,
    build_track_inputs,
    prepare_hierarchical_condition_dataframe,
    prepare_hierarchical_scattered_only_dataframe,
)
from ml.hierarchical_steady_2d.models import FaceClassifier, GaussianRegressor, load_checkpoint
from ml.hierarchical_steady_2d.normalization import HierarchicalNormalizationBundle, fit_hierarchical_normalization


def _sample_scattered_only_df() -> pd.DataFrame:
    rows = []
    exit_faces = [FACE_RIGHT, FACE_TOP, FACE_LEFT, FACE_BOTTOM] * 3
    ballistic_faces = [FACE_TOP, FACE_RIGHT, FACE_BOTTOM, FACE_LEFT] * 3
    for idx in range(12):
        width = 1.0 + 0.1 * idx
        height = 1.5 + 0.05 * idx
        exit_face = exit_faces[idx]
        ballistic_face = ballistic_faces[idx]

        if exit_face == FACE_LEFT:
            exit_coord = 0.20 * height
            final_u, final_v = -0.8, 0.1
        elif exit_face == FACE_RIGHT:
            exit_coord = 0.30 * height
            final_u, final_v = 0.9, -0.2
        elif exit_face == FACE_TOP:
            exit_coord = 0.60 * width
            final_u, final_v = 0.1, 0.7
        else:
            exit_coord = 0.40 * width
            final_u, final_v = -0.2, -0.9

        if ballistic_face in (FACE_LEFT, FACE_RIGHT):
            ballistic_coord = 0.5 * height
        else:
            ballistic_coord = 0.5 * width

        rows.append(
            {
                "width": width,
                "height": height,
                "initial_pos_x": 0.25 * width,
                "initial_pos_y": 0.35 * height,
                "initial_dir_u": 0.3 - 0.02 * idx,
                "initial_dir_v": -0.4 + 0.03 * idx,
                "ballistic_length": 0.8 + 0.05 * idx,
                "ballistic_face": ballistic_face,
                "ballistic_face_coord_local": ballistic_coord,
                "scatter_before_exit_prob": 0.6 + 0.01 * idx,
                "scatter_before_exit": 1,
                "exit_face": exit_face,
                "exit_face_coord_local": exit_coord,
                "final_dir_u": final_u,
                "final_dir_v": final_v,
                "total_track_length": 1.4 + 0.08 * idx,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_hierarchical_scattered_only_dataframe_adds_derived_columns():
    prepared = prepare_hierarchical_scattered_only_dataframe(_sample_scattered_only_df())

    expected = {
        "initial_pos_x_norm",
        "initial_pos_y_norm",
        "ballistic_face_coord_local_norm",
        "exit_face_coord_local_norm",
        "width_log10",
        "height_log10",
        "ballistic_length_log10",
        "total_track_length_log10",
        "ballistic_face_name",
        "exit_face_name",
        "final_dir_outward_compatible",
    }
    assert expected <= set(prepared.columns)
    assert (prepared["scatter_before_exit"] == 1).all()
    assert prepared["ballistic_face_coord_local_norm"].between(0.0, 1.0).all()
    assert prepared["exit_face_coord_local_norm"].between(0.0, 1.0).all()
    assert (prepared["final_dir_outward_compatible"] == 1).all()


def test_task_builders_produce_expected_dimensions():
    prepared = prepare_hierarchical_scattered_only_dataframe(_sample_scattered_only_df())
    normalization = fit_hierarchical_normalization(prepared, source="internal", sigma_t=1.0)

    face_task = build_face_task_arrays(prepared, normalization)
    joint_task = build_joint_task_arrays(prepared, normalization)
    track_task = build_track_task_arrays(prepared, normalization)

    assert face_task.inputs.shape == (len(prepared), len(face_task.input_names))
    assert face_task.targets.shape == (len(prepared),)

    assert joint_task.task == JOINT_TASK
    assert joint_task.inputs.shape == (len(prepared), len(joint_task.input_names))
    assert joint_task.targets.shape == (len(prepared), 3)

    assert track_task.task == TRACK_TASK
    assert track_task.inputs.shape == (len(prepared), len(track_task.input_names))
    assert track_task.targets.shape == (len(prepared), 1)

    shifted_faces = np.roll(prepared["exit_face"].to_numpy(dtype=np.int64), 1)
    shifted_joint = prepared[["exit_face_coord_local_norm", "final_dir_u", "final_dir_v"]].to_numpy(dtype=np.float32)
    shifted_joint[:, 0] = np.clip(shifted_joint[:, 0] * 0.9, 0.0, 1.0)

    conditioned_joint = build_joint_task_arrays(prepared, normalization, conditioning_face_ids=shifted_faces)
    conditioned_track = build_track_task_arrays(
        prepared,
        normalization,
        conditioning_face_ids=shifted_faces,
        conditioning_joint_targets=shifted_joint,
    )
    assert conditioned_joint.inputs.shape == joint_task.inputs.shape
    assert conditioned_track.inputs.shape == track_task.inputs.shape

    condition_frame = prepare_hierarchical_condition_dataframe(
        prepared[
            [
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
            ]
        ]
    )
    face_inputs, face_names = build_face_inputs(condition_frame, normalization)
    joint_inputs, joint_names = build_joint_inputs(condition_frame, normalization, conditioning_face_ids=shifted_faces)
    track_inputs, track_names = build_track_inputs(
        condition_frame,
        normalization,
        conditioning_face_ids=shifted_faces,
        conditioning_joint_targets=shifted_joint,
    )
    assert face_inputs.shape == (len(prepared), len(face_names))
    assert joint_inputs.shape == (len(prepared), len(joint_names))
    assert track_inputs.shape == (len(prepared), len(track_names))


def test_normalization_bundle_round_trip(tmp_path):
    prepared = prepare_hierarchical_scattered_only_dataframe(_sample_scattered_only_df())
    normalization = fit_hierarchical_normalization(prepared, source="boundary", sigma_t=1.0)
    json_path, npz_path = normalization.save(tmp_path)
    loaded = HierarchicalNormalizationBundle.load(json_path, npz_path)

    assert loaded.source == normalization.source
    assert loaded.sigma_t == normalization.sigma_t
    assert loaded.base_feature_names == normalization.base_feature_names
    assert np.allclose(loaded.base_feature_mean, normalization.base_feature_mean)
    assert np.allclose(loaded.joint_target_std, normalization.joint_target_std)
    assert np.allclose(loaded.track_target_mean, normalization.track_target_mean)


def test_training_smoke_writes_task_artifacts(tmp_path):
    pytest.importorskip("torch")

    from ml.hierarchical_steady_2d.eval import evaluate_hierarchical_checkpoint
    from ml.hierarchical_steady_2d.rollout_eval import (
        _append_ballistic_metadata,
        _decode_uniform_conditions,
        _hierarchical_rollout_predictions,
        _load_stage_model,
        evaluate_hierarchical_rollout,
    )
    from ml.hierarchical_steady_2d.training import HierarchicalTrainingConfig, run_hierarchical_training

    config = HierarchicalTrainingConfig(
        source="internal",
        tasks=("face", "joint", "track"),
        epochs=1,
        batch_size=4,
        hidden_dim=16,
        depth=1,
        val_ratio=0.25,
        output_root=str(tmp_path),
        run_name="smoke_run",
        device="cpu",
        seed=7,
    )
    results = run_hierarchical_training(config, dataframe=_sample_scattered_only_df())

    assert set(results["tasks"]) == {"face", "joint", "track"}
    run_dir = tmp_path / "smoke_run"
    assert (run_dir / "metadata" / "run_config.json").exists()
    assert (run_dir / "metadata" / "normalization.json").exists()
    assert (run_dir / "metadata" / "normalization.npz").exists()
    for task in ("face", "joint", "track"):
        assert (run_dir / task / "models" / "best_model.pt").exists()
        assert (run_dir / task / "models" / "final_model.pt").exists()
        assert (run_dir / task / "metrics" / "metrics.json").exists()
        assert (run_dir / task / "metrics" / "history.npz").exists()

    face_model, _, _ = load_checkpoint(run_dir / "face" / "models" / "best_model.pt")
    joint_model, _, _ = load_checkpoint(run_dir / "joint" / "models" / "best_model.pt")
    track_model, _, _ = load_checkpoint(run_dir / "track" / "models" / "best_model.pt")
    assert isinstance(face_model, FaceClassifier)
    assert isinstance(joint_model, GaussianRegressor)
    assert isinstance(track_model, GaussianRegressor)

    face_eval = evaluate_hierarchical_checkpoint(
        run_dir / "face" / "models" / "best_model.pt",
        dataframe=_sample_scattered_only_df(),
        device="cpu",
        output=run_dir / "face" / "metrics" / "eval_metrics.json",
    )
    joint_eval = evaluate_hierarchical_checkpoint(
        run_dir / "joint" / "models" / "best_model.pt",
        dataframe=_sample_scattered_only_df(),
        device="cpu",
        output=run_dir / "joint" / "metrics" / "eval_metrics.json",
    )
    track_eval = evaluate_hierarchical_checkpoint(
        run_dir / "track" / "models" / "best_model.pt",
        dataframe=_sample_scattered_only_df(),
        device="cpu",
        output=run_dir / "track" / "metrics" / "eval_metrics.json",
    )

    assert face_eval["stage"] == "face"
    assert face_eval["target_kind"] == "classification"
    assert "accuracy" in face_eval["metrics"]
    assert joint_eval["stage"] == "joint"
    assert joint_eval["target_kind"] == "regression"
    assert "mae" in joint_eval["metrics"]
    assert track_eval["stage"] == "track"
    assert "mae_total_track_length" in track_eval["metrics"]

    rollout_eval = evaluate_hierarchical_rollout(
        run_dir,
        source_type="internal",
        num_samples=16,
        sampling="random",
        seed=11,
        model_batch_size=8,
        device="cpu",
        output=run_dir / "rollout_eval_internal.json",
    )
    assert "metrics" in rollout_eval
    assert "x_out_norm" in rollout_eval["metrics"]
    assert "mean_wasserstein" in rollout_eval

    normalization = HierarchicalNormalizationBundle.load(run_dir / "metadata" / "normalization.json")
    condition_frame, gate_uniform, face_uniform = _decode_uniform_conditions(
        source_type="internal",
        num_samples=16,
        sampling="random",
        seed=13,
        min_width=0.01,
        max_width=10.0,
        min_height=0.01,
        max_height=10.0,
    )
    condition_frame = _append_ballistic_metadata(condition_frame, sigma_t=1.0)
    face_stage_model = _load_stage_model(run_dir / "face" / "models" / "best_model.pt", device="cpu", expected_stage="face")
    joint_stage_model = _load_stage_model(run_dir / "joint" / "models" / "best_model.pt", device="cpu", expected_stage="joint")
    track_stage_model = _load_stage_model(run_dir / "track" / "models" / "best_model.pt", device="cpu", expected_stage="track")

    pred_a, diag_a = _hierarchical_rollout_predictions(
        frame=condition_frame,
        gate_uniform=gate_uniform,
        face_uniform=face_uniform,
        face_model=face_stage_model,
        joint_model=joint_stage_model,
        track_model=track_stage_model,
        normalization=normalization,
        device="cpu",
        batch_size=8,
        face_sampling="categorical",
        sample_seed=101,
    )
    pred_b, diag_b = _hierarchical_rollout_predictions(
        frame=condition_frame,
        gate_uniform=gate_uniform,
        face_uniform=face_uniform,
        face_model=face_stage_model,
        joint_model=joint_stage_model,
        track_model=track_stage_model,
        normalization=normalization,
        device="cpu",
        batch_size=8,
        face_sampling="categorical",
        sample_seed=101,
    )
    pred_c, _diag_c = _hierarchical_rollout_predictions(
        frame=condition_frame,
        gate_uniform=gate_uniform,
        face_uniform=face_uniform,
        face_model=face_stage_model,
        joint_model=joint_stage_model,
        track_model=track_stage_model,
        normalization=normalization,
        device="cpu",
        batch_size=8,
        face_sampling="categorical",
        sample_seed=202,
    )

    assert diag_a["scatter_fraction_sampled"] > 0.0
    assert diag_a == diag_b
    for key in pred_a:
        assert np.allclose(pred_a[key], pred_b[key])
    assert any(not np.allclose(pred_a[key], pred_c[key]) for key in pred_a)
    x_exit = pred_a["x_out_norm"] * condition_frame["width"].to_numpy(dtype=np.float64)
    y_exit = pred_a["y_out_norm"] * condition_frame["height"].to_numpy(dtype=np.float64)
    x0 = condition_frame["initial_pos_x"].to_numpy(dtype=np.float64)
    y0 = condition_frame["initial_pos_y"].to_numpy(dtype=np.float64)
    straight_line_min = np.sqrt(np.maximum((x_exit - x0) ** 2 + (y_exit - y0) ** 2, 0.0))
    assert np.all(pred_a["time"] + 1.0e-9 >= straight_line_min)

    with pytest.raises(ValueError, match="trained for source"):
        evaluate_hierarchical_checkpoint(
            run_dir / "face" / "models" / "best_model.pt",
            num_samples=8,
            sampling="random",
            source_type="boundary",
            device="cpu",
        )


def test_wrapper_scripts_smoke(tmp_path):
    pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "wrapper_runs"

    subprocess.run(
        [
            sys.executable,
            "ml/train_hierarchical_steady_2d.py",
            "--source",
            "internal",
            "--num-neutrons",
            "32",
            "--tasks",
            "face",
            "joint",
            "track",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--output-root",
            str(output_root),
            "--run-name",
            "wrapper_run",
        ],
        cwd=repo_root,
        check=True,
    )

    checkpoint = output_root / "wrapper_run" / "face" / "models" / "best_model.pt"
    eval_output = output_root / "wrapper_run" / "face" / "metrics" / "wrapper_eval.json"
    subprocess.run(
        [
            sys.executable,
            "ml/eval_hierarchical_steady_2d.py",
            "--checkpoint",
            str(checkpoint),
            "--num-samples",
            "16",
            "--source",
            "internal",
            "--sampling",
            "random",
            "--device",
            "cpu",
            "--output",
            str(eval_output),
        ],
        cwd=repo_root,
        check=True,
    )

    assert checkpoint.exists()
    assert eval_output.exists()

    rollout_output = output_root / "wrapper_run" / "rollout_eval_internal.json"
    subprocess.run(
        [
            sys.executable,
            "ml/rollout_eval_hierarchical_steady_2d.py",
            "--run-dir",
            str(output_root / "wrapper_run"),
            "--num-samples",
            "16",
            "--source",
            "internal",
            "--sampling",
            "random",
            "--device",
            "cpu",
            "--output",
            str(rollout_output),
        ],
        cwd=repo_root,
        check=True,
    )

    assert rollout_output.exists()
