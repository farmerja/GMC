import numpy as np
import pytest
import torch

from ml.distillation import tasks as task_module
from ml.distillation.artifacts import resolve_student_checkpoint_path, resolve_student_run_dir
from ml.distillation.student import FixedStepStudent, load_student_checkpoint
from ml.distillation.teacher import FlowTeacherConfig, FlowTeacherSampler
from ml.distillation.trainer import (
    condition_slice_metrics,
    prediction_metrics,
    sample_standard_noise,
)
from ml.eval_student import main as eval_student_main
from ml.model import get_model
from ml.ode_grid import build_piecewise_time_grid
from ml.train_student import main as train_student_main
from ml.transform import (
    denorm_angle_phi,
    denorm_height,
    denorm_position_relative,
    denorm_radius_r,
    denorm_width,
    polar_to_uv,
)


class ToyTeacher:
    def sample_target(self, conditions, *, noise=None, noise_seed=None):
        cond = conditions.to(torch.float32)
        if noise is None:
            noise = sample_standard_noise(cond.shape[0], 2, seed=int(noise_seed or 0), device=cond.device)
        else:
            noise = noise.to(torch.float32)
        return torch.stack(
            [
                cond[:, 0] + 0.5 * cond[:, 1] + 0.25 * noise[:, 0],
                cond[:, 1] - cond[:, 0] + 0.25 * noise[:, 1],
            ],
            dim=-1,
        ).cpu()


class ToyTask:
    task_name = "steady2d"
    data_dim = "2d"
    source_type = "boundary"
    scattering_mode = "isotropic"
    cond_dim = 2
    target_dim = 2
    norm_params = {}
    teacher_hidden_dim = 1
    teacher_depth = 1
    teacher_expansion_factor = 1
    teacher_time_grid_kind = "piecewise"
    teacher_time_grid_split = 0.8
    teacher_time_grid_schedule = None
    teacher_default_num_steps = 9

    @staticmethod
    def sample_conditions(num_samples, *, sampling="lhs", seed=42, device="cpu"):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        return torch.rand((int(num_samples), 2), generator=gen, dtype=torch.float32, device=device)


def test_steady2d_boundary_support_sampler_stays_inside_support(monkeypatch):
    fake_norm = {
        "width_lower": -2.0,
        "width_upper": 1.0,
        "height_lower": -1.0,
        "height_upper": 0.5,
    }

    def fake_load_norm_params(*_args, **_kwargs):
        return fake_norm, "unused"

    monkeypatch.setattr(task_module, "load_norm_params_for_data_dim", fake_load_norm_params)
    spec = task_module.resolve_distillation_task(
        "steady2d",
        source_type="boundary",
        teacher_train_sampling="random",
        scattering_mode="isotropic",
    )
    cond = spec.sample_conditions(128, sampling="random", seed=5, device="cpu")

    width = denorm_width(cond[:, 0], fake_norm["width_lower"], fake_norm["width_upper"]).numpy()
    height = denorm_height(cond[:, 1], fake_norm["height_lower"], fake_norm["height_upper"]).numpy()
    y = denorm_position_relative(cond[:, 2]).numpy()
    phi = denorm_angle_phi(cond[:, 3]).numpy()
    r = denorm_radius_r(cond[:, 4]).numpy()
    u, _v = polar_to_uv(phi, r)

    assert cond.shape == (128, 5)
    assert np.all(np.isfinite(cond.numpy()))
    assert np.all(width >= 10 ** fake_norm["width_lower"])
    assert np.all(width <= 10 ** fake_norm["width_upper"])
    assert np.all(height >= 10 ** fake_norm["height_lower"])
    assert np.all(height <= 10 ** fake_norm["height_upper"])
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert np.all(r <= 1.0 + 1.0e-6)
    assert np.all(u >= -1.0e-6)


def test_flow_teacher_sampler_uses_configured_piecewise_grid(tmp_path):
    teacher_model = get_model(input_dim=4, cond_dim=2, hidden_dim=12, depth=2, expansion_factor=2)
    checkpoint = tmp_path / "toy_teacher.pth"
    torch.save(
        {
            "state_dict": teacher_model.state_dict(),
            "metadata": {
                "input_dim": 4,
                "cond_dim": 2,
                "data_dim": "toy",
                "source_type": "boundary",
                "train_sampling": "random",
                "scattering_mode": "isotropic",
            },
        },
        checkpoint,
    )

    config = FlowTeacherConfig(
        data_dim="toy",
        source_type="boundary",
        checkpoint_path=str(checkpoint),
        hidden_dim=12,
        depth=2,
        expansion_factor=2,
        ode_num_steps=9,
        time_grid_kind="piecewise",
        time_grid_split=0.7,
    )
    sampler = FlowTeacherSampler.from_config(config, input_dim=4, cond_dim=2, device="cpu")
    expected_grid, expected_step = build_piecewise_time_grid(9, split=0.7, device="cpu")

    assert torch.allclose(sampler.time_grid.cpu(), expected_grid.cpu())
    assert sampler.step_size.item() == pytest.approx(expected_step.item())


def test_fixed_step_student_predicts_single_target():
    model = FixedStepStudent(cond_dim=3, target_dim=2, num_steps=4, hidden_dim=16, depth=2, expansion_factor=1)
    cond = torch.randn(7, 3)
    noise = torch.randn(7, 2)
    pred = model(cond, noise)

    assert pred.shape == (7, 2)


def test_prediction_and_slice_metrics_have_expected_shape():
    pred = torch.tensor([[1.0, 3.0], [2.0, 5.0], [4.0, 7.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 2.0], [2.0, 4.0], [5.0, 8.0]], dtype=torch.float32)
    conditions = torch.tensor([[0.1, 0.5], [0.4, 0.2], [0.9, 0.1]], dtype=torch.float32)

    metrics = prediction_metrics(pred, target)
    slices = condition_slice_metrics(conditions, pred, target, max_features=2, num_bins=2)

    assert metrics["loss"] > 0.0
    assert len(metrics["mean_bias_by_dim"]) == 2
    assert len(metrics["std_ratio_by_dim"]) == 2
    assert "cond_0" in slices
    assert "cond_1" in slices


def test_train_and_eval_student_clis_with_toy_teacher(tmp_path, monkeypatch):
    artifact_root = tmp_path / "student_artifacts"

    monkeypatch.setattr("ml.train_student.resolve_distillation_task", lambda *args, **kwargs: ToyTask())
    monkeypatch.setattr("ml.eval_student.resolve_distillation_task", lambda *args, **kwargs: ToyTask())
    monkeypatch.setattr("ml.train_student.FlowTeacherSampler.from_task", lambda *args, **kwargs: ToyTeacher())
    monkeypatch.setattr("ml.eval_student.FlowTeacherSampler.from_task", lambda *args, **kwargs: ToyTeacher())

    train_student_main(
        [
            "--source-type",
            "boundary",
            "--teacher-checkpoint",
            str(tmp_path / "teacher.pth"),
            "--steps",
            "2",
            "--support-samples",
            "128",
            "--epochs",
            "20",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.01",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    run_dir = resolve_student_run_dir("steady2d", "boundary", 2, artifact_root=str(artifact_root))
    best_ckpt = resolve_student_checkpoint_path("steady2d", "boundary", 2, artifact_root=str(artifact_root))

    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.npz").exists()
    assert best_ckpt.exists()

    model, config, metadata = load_student_checkpoint(best_ckpt, map_location="cpu")
    assert isinstance(model, FixedStepStudent)
    assert config.num_steps == 2
    assert metadata["support_samples"] == 128

    eval_student_main(
        [
            "--source-type",
            "boundary",
            "--teacher-checkpoint",
            str(tmp_path / "teacher.pth"),
            "--checkpoint",
            str(best_ckpt),
            "--support-samples",
            "64",
            "--batch-size",
            "32",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    eval_metrics = (run_dir / "eval_metrics.json").read_text()
    assert (run_dir / "eval_metrics.json").exists()
    assert "student_speedup_vs_teacher" in eval_metrics
    assert "mean_bias_by_dim" in eval_metrics
