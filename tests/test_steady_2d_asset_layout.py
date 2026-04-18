from ml import utils as ml_utils
from physics.steady_2d_scattering import (
    STEADY_2D_ANISOTROPIC_ASSET_ROOT,
    steady_2d_model_dir,
    steady_2d_model_dir_candidates,
    steady_2d_norm_params_base_path,
    steady_2d_norm_params_base_path_candidates,
    steady_2d_training_dir,
)


def test_anisotropic_steady_2d_assets_use_isolated_layout():
    assert steady_2d_model_dir("boundary", scattering_mode="anisotropic") == (
        f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/models/2d_boundary"
    )
    assert steady_2d_training_dir("internal", scattering_mode="anisotropic") == (
        f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/training_results/2d_internal"
    )
    assert steady_2d_norm_params_base_path("boundary", scattering_mode="anisotropic") == (
        f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/data/norm_params_2d_boundary.npy"
    )
    assert steady_2d_model_dir_candidates("boundary", scattering_mode="anisotropic") == (
        f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/models/2d_boundary",
        "ml/models/2d_boundary_steady2d-v2_anisotropic",
    )
    assert steady_2d_norm_params_base_path_candidates("internal", scattering_mode="anisotropic") == (
        f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/data/norm_params_2d_internal.npy",
        "ml/data/norm_params_2d_internal_steady2d-v2_anisotropic.npy",
    )


def test_load_norm_params_for_data_dim_falls_back_to_legacy_anisotropic_path(monkeypatch):
    attempts = []
    primary = f"{STEADY_2D_ANISOTROPIC_ASSET_ROOT}/data/norm_params_2d_boundary.npy"
    legacy = "ml/data/norm_params_2d_boundary_steady2d-v2_anisotropic.npy"

    def fake_load_normalization_params(base_path, train_sampling="random"):
        attempts.append((base_path, train_sampling))
        if base_path == primary:
            raise FileNotFoundError("new anisotropic asset missing")
        if base_path == legacy:
            return {"steady_2d_scattering_mode": "anisotropic"}, legacy.replace(".npy", ".npz")
        raise AssertionError(f"unexpected path: {base_path}")

    monkeypatch.setattr(ml_utils, "load_normalization_params", fake_load_normalization_params)

    norm_params, loaded_path = ml_utils.load_norm_params_for_data_dim(
        "2d",
        "boundary",
        train_sampling="random",
        scattering_mode="anisotropic",
    )

    assert norm_params["steady_2d_scattering_mode"] == "anisotropic"
    assert loaded_path == legacy.replace(".npy", ".npz")
    assert attempts == [(primary, "random"), (legacy, "random")]
