import torch

from ml import utils as ml_utils
from physics.mc_data_gen import (
    get_3d_internal_source_simulation_data,
    get_3d_time_dependent_data,
)


class _DummyModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state_dict, strict=False):
        return [], []

    def eval(self):
        return self


def test_3d_hg_asset_paths_are_g_specific():
    assert ml_utils.steady_2d_model_dir(
        "3d",
        "boundary",
        scattering_mode="anisotropic-hg",
        g_hg=0.9,
    ) == "ml/models/3d_boundary_scatter3d-v1_anisotropic_hg_g_p0_900000"
    assert ml_utils.norm_params_base_path(
        "3d_timedep",
        "internal",
        scattering_mode="anisotropic-hg",
        g_hg=-0.9,
    ) == "ml/data/norm_params_3d_timedep_internal_scatter3d-v1_anisotropic_hg_g_m0_900000.npy"


def test_3d_test_environment_separates_hg_output_dirs(monkeypatch):
    created = []

    monkeypatch.setattr(ml_utils.os, "makedirs", lambda path, exist_ok=True: created.append((path, exist_ok)))
    monkeypatch.setattr(ml_utils, "setup_environment", lambda **_kwargs: "cpu")

    device, test_dir = ml_utils.setup_test_environment(
        source_type="internal",
        scheduler_type="condOT",
        data_dim="3d_timedep",
        infer_sampling="random",
        train_sampling="random",
        scattering_mode="anisotropic-hg",
        g_hg=0.9,
    )

    assert device == "cpu"
    assert test_dir.endswith("cfm_test_3d_timedep_internal_condOT_anisotropic_hg_g_p0_900_train-random_infer-random")
    assert created == [(test_dir, True)]


def test_load_model_uses_3d_hg_directory(monkeypatch):
    expected_path = "ml/models/3d_boundary_scatter3d-v1_anisotropic_hg_g_p0_900000/best_model.pth"
    seen_exists = []
    seen_loads = []

    def fake_exists(path):
        seen_exists.append(path)
        return path == expected_path

    def fake_torch_load(path, map_location=None):
        seen_loads.append((path, map_location))
        return {
            "input_proj.weight": torch.zeros((8, 13)),
            "cond_proj.0.weight": torch.zeros((8, 7)),
            "blocks.0.ff.0.weight": torch.zeros((16, 8)),
            "output.weight": torch.zeros((13, 8)),
        }

    monkeypatch.setattr(ml_utils.os.path, "exists", fake_exists)
    monkeypatch.setattr(ml_utils.torch, "load", fake_torch_load)
    monkeypatch.setattr(ml_utils, "get_model", lambda **_kwargs: _DummyModel())

    model, input_dim, cond_dim = ml_utils.load_model(
        device="cpu",
        hidden_dim=8,
        depth=1,
        expansion_factor=2,
        source_type="boundary",
        data_dim="3d",
        scattering_mode="anisotropic-hg",
        g_hg=0.9,
    )

    assert isinstance(model, _DummyModel)
    assert input_dim == 13
    assert cond_dim == 7
    assert seen_loads == [(expected_path, "cpu")]
    assert expected_path in seen_exists


def test_3d_internal_source_data_generation_smoke_supports_anisotropic_hg():
    df = get_3d_internal_source_simulation_data(
        num_neutrons=8,
        sampling="random",
        scatter_ratio=0.7,
        min_width=0.2,
        max_width=0.2,
        min_height=0.3,
        max_height=0.3,
        min_depth=0.4,
        max_depth=0.4,
        scattering_mode="anisotropic-hg",
        g_hg=0.9,
    )

    assert len(df) == 8
    assert {"initial_dir_u", "initial_dir_v", "initial_dir_w", "final_dir_w"}.issubset(df.columns)
    assert ((df["initial_dir_u"] ** 2 + df["initial_dir_v"] ** 2 + df["initial_dir_w"] ** 2) > 0.99).all()


def test_3d_timedep_boundary_data_generation_smoke_supports_anisotropic_linear():
    df = get_3d_time_dependent_data(
        source="boundary",
        num_neutrons=8,
        sampling="random",
        scatter_ratio=0.7,
        min_width=0.2,
        max_width=0.2,
        min_height=0.3,
        max_height=0.3,
        min_depth=0.4,
        max_depth=0.4,
        min_path_budget=0.05,
        max_path_budget=0.05,
        scattering_mode="anisotropic-linear",
    )

    assert len(df) == 8
    assert {"space_exit", "time_exit", "absorbed", "final_dir_w"}.issubset(df.columns)
    assert ((df["space_exit"] + df["time_exit"] + df["absorbed"]) > 0.5).all()


def test_safe_get_sim_data_preserves_scattering_kwargs_for_wrapper_functions():
    captured = {}

    def wrapper(**kwargs):
        captured.update(kwargs)
        return kwargs

    ml_utils.safe_get_sim_data(
        wrapper,
        num_neutrons=4,
        sampling="random",
        scattering_mode="anisotropic-hg",
        g_hg=0.8,
    )

    assert captured["sampling"] == "random"
    assert captured["scattering_mode"] == "anisotropic-hg"
    assert captured["g_hg"] == 0.8
