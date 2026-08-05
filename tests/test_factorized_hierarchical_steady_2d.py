import math

import numpy as np
import pytest

from ml.hierarchical_steady_2d.factorized import (
    FactorizedSteady2DSampler,
    FactorizedTrainingConfig,
    prepare_factorized_dataframe,
    run_factorized_training,
)
from ml.hierarchical_steady_2d.symmetry import (
    canonicalize_boundary_state,
    canonicalize_internal_state,
    forward_transform_angular_moments,
    forward_transform_direction,
    forward_transform_legendre_coeffs,
    forward_transform_point,
    inverse_transform_angular_moments,
    inverse_transform_direction,
    inverse_transform_legendre_coeffs,
    inverse_transform_point,
)
from physics.hierarchical_steady_2d import compute_straight_track_integrals_2d, direction_is_outward_for_face_2d
from physics.mc_data_gen import get_2d_factorized_hierarchical_steady_scattered_only_data


def test_factorized_symmetry_round_trip_for_internal_state_outputs():
    width = 2.0
    height = 1.0
    x = 1.6
    y = 0.2
    u = -0.3
    v = 0.8
    angular = np.array([0.2, -0.4, 0.7, -0.1, 0.3], dtype=np.float64)
    legendre = np.array([0.5, -0.25, 0.1, 0.3, -0.2, 0.05, -0.07, 0.09], dtype=np.float64)

    canonical = canonicalize_internal_state(width, height, x, y, u, v)
    x_can, y_can = forward_transform_point(
        canonical.transform,
        width,
        height,
        canonical.width,
        canonical.height,
        x,
        y,
    )
    u_can, v_can = forward_transform_direction(canonical.transform, u, v)
    angular_can = forward_transform_angular_moments(canonical.transform, angular)
    legendre_can = forward_transform_legendre_coeffs(canonical.transform, legendre)

    x_back, y_back = inverse_transform_point(
        canonical.transform,
        width,
        height,
        canonical.width,
        canonical.height,
        x_can,
        y_can,
    )
    u_back, v_back = inverse_transform_direction(canonical.transform, u_can, v_can)
    angular_back = inverse_transform_angular_moments(canonical.transform, angular_can)
    legendre_back = inverse_transform_legendre_coeffs(canonical.transform, legendre_can)

    assert np.allclose([x_can, y_can], [canonical.x, canonical.y])
    assert np.allclose([u_can, v_can], [canonical.u, canonical.v])
    assert np.allclose([x_back, y_back], [x, y])
    assert np.allclose([u_back, v_back], [u, v])
    assert np.allclose(angular_back, angular)
    assert np.allclose(legendre_back, legendre)


def test_factorized_boundary_canonicalization_keeps_left_inward_entry():
    canonical = canonicalize_boundary_state(
        entry_face=2,
        entry_coord=0.4,
        u=-0.5,
        v=-0.4,
        width=1.0,
        height=2.0,
    )

    assert canonical.width >= canonical.height
    assert canonical.u > 0.0
    assert canonical.y_on_entry >= 0.0
    assert canonical.y_on_entry <= canonical.height


def test_compute_straight_track_integrals_matches_simple_closed_form():
    weighted_track, angular, legendre = compute_straight_track_integrals_2d(
        width=1.0,
        height=1.0,
        x0=0.0,
        y0=0.5,
        u=1.0,
        v=0.0,
        length=2.0,
        sigma_a=0.5,
    )

    expected_weighted_track = (1.0 - math.exp(-1.0)) / 0.5
    assert weighted_track == np.float64(expected_weighted_track)
    assert angular[0] == np.float64(expected_weighted_track)
    assert angular[1] == 0.0
    assert angular[2] == np.float64(expected_weighted_track)
    assert angular[3] == 0.0
    assert angular[4] == 0.0
    assert legendre.shape == (8,)
    assert np.isfinite(legendre).all()


def test_factorized_scattered_only_wrapper_emits_richer_targets():
    df = get_2d_factorized_hierarchical_steady_scattered_only_data(
        source="internal",
        num_neutrons=24,
        sigma_s=1.0,
        sigma_a=0.5,
        sampling="lhs",
    )

    expected = {
        "weighted_track_length",
        "moment_u",
        "moment_v",
        "moment_uu",
        "moment_uv",
        "moment_vv",
        "legendre_p1_p0",
        "legendre_p0_p1",
        "legendre_p2_p0",
        "legendre_p1_p1",
        "legendre_p0_p2",
        "legendre_p2_p1",
        "legendre_p1_p2",
        "legendre_p2_p2",
    }
    assert expected <= set(df.columns)
    assert len(df) == 24
    assert (df["num_scatters"] >= 1).all()
    assert (df["weighted_track_length"] > 0.0).all()
    assert df["scatter_before_exit_prob"].between(0.0, 1.0).all()

    for face, u_val, v_val in zip(df["exit_face"], df["final_dir_u"], df["final_dir_v"]):
        assert direction_is_outward_for_face_2d(int(face), float(u_val), float(v_val))


def test_prepare_factorized_dataframe_adds_exit_unit_targets_and_face_one_hot():
    raw = get_2d_factorized_hierarchical_steady_scattered_only_data(
        source="boundary",
        num_neutrons=16,
        sigma_s=1.0,
        sigma_a=0.2,
        sampling="lhs",
    )
    prepared = prepare_factorized_dataframe(raw, source="boundary")

    frame = prepared.dataframe
    assert {
        "exit_unit_x",
        "exit_unit_y",
        "exit_face_is_left",
        "exit_face_is_right",
        "exit_face_is_top",
        "exit_face_is_bottom",
        "final_dir_u",
        "final_dir_v",
    } <= set(frame.columns)
    norms = np.sqrt(frame["exit_unit_x"].to_numpy() ** 2 + frame["exit_unit_y"].to_numpy() ** 2)
    assert np.allclose(norms, 1.0, atol=1.0e-6)
    assert frame["exit_face_coord_norm"].between(0.0, 1.0).all()
    raw_dir_norms = np.sqrt(raw["final_dir_u"].to_numpy() ** 2 + raw["final_dir_v"].to_numpy() ** 2)
    prepared_dir_norms = np.sqrt(frame["final_dir_u"].to_numpy() ** 2 + frame["final_dir_v"].to_numpy() ** 2)
    assert np.allclose(prepared_dir_norms, raw_dir_norms, atol=1.0e-6)
    assert np.max(np.abs(prepared_dir_norms - 1.0)) > 1.0e-3


def test_factorized_training_smoke_writes_artifacts_and_runtime_loader(tmp_path):
    pytest.importorskip("torch")

    dataframe = get_2d_factorized_hierarchical_steady_scattered_only_data(
        source="internal",
        num_neutrons=48,
        sigma_s=1.0,
        sigma_a=0.3,
        sampling="lhs",
    )
    config = FactorizedTrainingConfig(
        source="internal",
        epochs=1,
        batch_size=8,
        hidden_dim=16,
        depth=1,
        val_ratio=0.25,
        output_root=str(tmp_path),
        run_name="factorized_smoke",
        device="cpu",
        seed=9,
    )
    results = run_factorized_training(config, dataframe=dataframe)

    run_dir = tmp_path / "factorized_smoke"
    assert results["run_dir"] == str(run_dir)
    assert (run_dir / "metadata" / "run_config.json").exists()
    assert (run_dir / "metadata" / "normalization.json").exists()
    assert (run_dir / "metadata" / "normalization.npz").exists()
    for stage in ("location", "kinematics"):
        assert (run_dir / stage / "models" / "best_model.pt").exists()
        assert (run_dir / stage / "models" / "final_model.pt").exists()
        assert (run_dir / stage / "metrics" / "metrics.json").exists()
        assert (run_dir / stage / "metrics" / "history.npz").exists()

    sampler = FactorizedSteady2DSampler.from_run_dir(run_dir, source_type="internal", device="cpu", batch_size=4)
    assert sampler.normalization.kinematics_target_names[:2] == ("final_dir_u", "final_dir_v")
    assert not any(name.startswith("legendre_") for name in sampler.normalization.kinematics_target_names)
    sample = sampler.sample_scattered_batch(
        width=dataframe["width"].to_numpy()[:4],
        height=dataframe["height"].to_numpy()[:4],
        initial_x=dataframe["initial_pos_x"].to_numpy()[:4],
        initial_y=dataframe["initial_pos_y"].to_numpy()[:4],
        initial_u=dataframe["initial_dir_u"].to_numpy()[:4],
        initial_v=dataframe["initial_dir_v"].to_numpy()[:4],
        ballistic_length=dataframe["ballistic_length"].to_numpy()[:4],
        ballistic_face=dataframe["ballistic_face"].to_numpy()[:4],
        ballistic_coord=dataframe["ballistic_face_coord_local"].to_numpy()[:4],
        ballistic_exit_x=dataframe["ballistic_exit_x"].to_numpy()[:4],
        ballistic_exit_y=dataframe["ballistic_exit_y"].to_numpy()[:4],
        scatter_before_exit_prob=dataframe["scatter_before_exit_prob"].to_numpy()[:4],
        sigma_s=dataframe["sigma_s"].to_numpy()[:4],
        sigma_a=dataframe["sigma_a"].to_numpy()[:4],
        seed=5,
    )

    assert sample.x_out.shape == (4,)
    assert sample.y_out.shape == (4,)
    assert sample.track_length.shape == (4,)
    assert sample.angular_moments.shape == (4, 5)
    assert sample.legendre_coeffs.shape == (4, 8)
    assert np.allclose(sample.legendre_coeffs, 0.0)
    dir_norms = np.sqrt(sample.u_out ** 2 + sample.v_out ** 2)
    assert np.all(dir_norms <= 1.0 + 1.0e-6)
    for face, u_val, v_val in zip(sample.exit_face, sample.u_out, sample.v_out):
        assert direction_is_outward_for_face_2d(int(face), float(u_val), float(v_val))
