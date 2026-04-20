import numpy as np
import pytest
import torch

from ml import utils as ml_utils
from physics import mc2d, mc_data_gen, multigrid_steady_2d as mg
from physics.steady_2d_scattering import (
    STEADY_2D_HG_ASSET_ROOT,
    STEADY_2D_HG_G_LIMIT,
    steady_2d_scattering_mode_code,
)


class _DummyModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state_dict, strict=False):
        return [], []

    def eval(self):
        return self


def test_get_2d_boundary_simulation_data_forwards_explicit_hg_parameter(monkeypatch):
    seen = {}

    def fake_simulate(**kwargs):
        seen.update(kwargs)
        return np.zeros((1, 12), dtype=np.float64)

    monkeypatch.setattr(mc_data_gen, "simulate_single_cell_2d_boundary_source", fake_simulate)

    df = mc_data_gen.get_2d_boundary_simulation_data(
        num_neutrons=1,
        sampling="random",
        scattering_mode="anisotropic-hg",
        g_hg=0.61,
    )

    assert seen["scattering_mode_code"] == steady_2d_scattering_mode_code("anisotropic-hg")
    assert seen["g_hg"] == 0.61
    assert len(df) == 1


def test_get_2d_internal_source_simulation_data_requires_explicit_hg_parameter():
    with pytest.raises(ValueError):
        mc_data_gen.get_2d_internal_source_simulation_data(
            num_neutrons=1,
            sampling="random",
            scattering_mode="anisotropic-hg",
        )


def test_load_model_uses_mode_specific_hg_directory(monkeypatch):
    expected_path = f"{STEADY_2D_HG_ASSET_ROOT}/g_p0_610000/models/2d_boundary/best_model.pth"
    seen_exists = []
    seen_loads = []

    def fake_exists(path):
        seen_exists.append(path)
        return path == expected_path

    def fake_torch_load(path, map_location=None):
        seen_loads.append((path, map_location))
        return {
            "input_proj.weight": torch.zeros((8, 10)),
            "cond_proj.0.weight": torch.zeros((8, 5)),
            "blocks.0.ff.0.weight": torch.zeros((16, 8)),
            "output.weight": torch.zeros((10, 8)),
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
        data_dim="2d",
        scattering_mode="anisotropic-hg",
        g_hg=0.61,
    )

    assert isinstance(model, _DummyModel)
    assert input_dim == 10
    assert cond_dim == 5
    assert seen_loads == [(expected_path, "cpu")]
    assert all(STEADY_2D_HG_ASSET_ROOT in path for path in seen_exists[:2])


def test_run_mc_simulation_2d_wrapper_forwards_resolved_hg_parameter(monkeypatch):
    seen = {}

    def fake_boundary_kernel(
        _num_cells_x,
        _num_cells_y,
        _cell_widths,
        _cell_heights,
        _cell_boundaries_x,
        _cell_boundaries_y,
        _total_width,
        _total_height,
        _num_particles,
        _sigma_t_per_cell,
        _sigma_a_per_cell,
        _energy_threshold,
        _reflected_boundaries,
        _angular_sampling,
        scattering_mode_code,
        g_hg=0.0,
    ):
        seen["scattering_mode_code"] = scattering_mode_code
        seen["g_hg"] = g_hg
        return np.zeros((1, 1)), np.zeros((1, 1)), (0, 0, 0, 0, 1, 0, 0, 0)

    monkeypatch.setattr(mc2d, "multi_cell_mc_boundary_source_2d_njit", fake_boundary_kernel)

    energy, track, stats = mc2d.run_mc_simulation_2d_njit_wrapper(
        source_mode="boundary_only",
        num_cells_x=1,
        num_cells_y=1,
        cell_widths=np.array([1.0]),
        cell_heights=np.array([1.0]),
        cell_boundaries_x=np.array([0.0, 1.0]),
        cell_boundaries_y=np.array([0.0, 1.0]),
        total_width=1.0,
        total_height=1.0,
        num_particles=1,
        start_cell_indices=(0, 0),
        start_position_relative=(0.5, 0.5),
        scatter_ratio=1.0,
        energy_threshold=1.0e-4,
        seed=7,
        reflected_boundaries=False,
        scattering_mode="anisotropic-hg",
        g_hg=0.9995,
        verbose=False,
    )

    assert seen["scattering_mode_code"] == steady_2d_scattering_mode_code("anisotropic-hg")
    assert seen["g_hg"] == STEADY_2D_HG_G_LIMIT
    assert energy.shape == (1, 1)
    assert track.shape == (1, 1)
    assert stats[-1] == 0


def test_run_mc_simulation_2d_wrapper_rejects_invalid_hg_parameter():
    with pytest.raises(ValueError):
        mc2d.run_mc_simulation_2d_njit_wrapper(
            source_mode="boundary_only",
            num_cells_x=1,
            num_cells_y=1,
            cell_widths=np.array([1.0]),
            cell_heights=np.array([1.0]),
            cell_boundaries_x=np.array([0.0, 1.0]),
            cell_boundaries_y=np.array([0.0, 1.0]),
            total_width=1.0,
            total_height=1.0,
            num_particles=1,
            start_cell_indices=(0, 0),
            start_position_relative=(0.5, 0.5),
            scatter_ratio=1.0,
            energy_threshold=1.0e-4,
            seed=7,
            reflected_boundaries=False,
            scattering_mode="anisotropic-hg",
            g_hg=5.0,
            verbose=False,
        )


def test_multigrid_norm_mode_accepts_matching_hg_metadata():
    norm_params = {
        "steady_2d_scattering_mode": "anisotropic-hg",
        "steady_2d_scattering_tag": "steady2d-v2",
        "steady_2d_scattering_hg_g": 0.9,
    }

    mg._assert_steady_2d_norm_mode(
        norm_params,
        "anisotropic-hg",
        "Boundary",
        g_hg=0.9,
    )


def test_multigrid_norm_mode_rejects_hg_g_mismatch():
    norm_params = {
        "steady_2d_scattering_mode": "anisotropic-hg",
        "steady_2d_scattering_tag": "steady2d-v2",
        "steady_2d_scattering_hg_g": 0.9,
    }

    with pytest.raises(ValueError, match="g_hg=0.900000"):
        mg._assert_steady_2d_norm_mode(
            norm_params,
            "anisotropic-hg",
            "Boundary",
            g_hg=-0.9,
        )


def test_multigrid_norm_mode_rejects_linear_for_hg_request():
    norm_params = {
        "steady_2d_scattering_mode": "anisotropic-linear",
        "steady_2d_scattering_tag": "steady2d-v2",
        "steady_2d_scattering_hg_g": 0.0,
    }

    with pytest.raises(ValueError, match="scattering_mode='anisotropic-linear'"):
        mg._assert_steady_2d_norm_mode(
            norm_params,
            "anisotropic-hg",
            "Boundary",
            g_hg=0.9,
        )


def test_setup_test_environment_separates_hg_output_dirs(monkeypatch):
    created = []

    monkeypatch.setattr(ml_utils.os, "makedirs", lambda path, exist_ok=True: created.append((path, exist_ok)))
    monkeypatch.setattr(ml_utils, "setup_environment", lambda **_kwargs: "cpu")

    device, test_dir = ml_utils.setup_test_environment(
        source_type="boundary",
        scheduler_type="condOT",
        data_dim="2d",
        infer_sampling="random",
        train_sampling="random",
        scattering_mode="anisotropic-hg",
        g_hg=-0.9,
    )

    assert device == "cpu"
    assert test_dir.endswith("cfm_test_2d_boundary_condOT_anisotropic_hg_g_m0_900_train-random_infer-random")
    assert created == [(test_dir, True)]
