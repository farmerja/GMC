import math

import numpy as np

from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    ballistic_exit_2d,
    direction_is_outward_for_face_2d,
    sample_truncated_first_collision_distance,
    scatter_before_exit_probability,
)
from physics.mc_data_gen import (
    get_2d_hierarchical_steady_gate_data,
    get_2d_hierarchical_steady_scattered_only_data,
)


def test_ballistic_exit_2d_returns_expected_face_distance_and_local_coord():
    distance, face, x_out, y_out, coord = ballistic_exit_2d(0.25, 0.5, 1.0, 0.0, 1.0, 2.0)
    assert distance == 0.75
    assert face == FACE_RIGHT
    assert x_out == 1.0
    assert y_out == 0.5
    assert coord == 0.5

    distance, face, x_out, y_out, coord = ballistic_exit_2d(0.25, 0.5, 0.0, 1.0, 1.0, 2.0)
    assert distance == 1.5
    assert face == FACE_TOP
    assert x_out == 0.25
    assert y_out == 2.0
    assert coord == 0.25

    distance, face, x_out, y_out, coord = ballistic_exit_2d(0.25, 0.5, -1.0, 0.0, 1.0, 2.0)
    assert distance == 0.25
    assert face == FACE_LEFT
    assert x_out == 0.0
    assert y_out == 0.5
    assert coord == 0.5

    distance, face, x_out, y_out, coord = ballistic_exit_2d(0.25, 0.5, 0.0, -1.0, 1.0, 2.0)
    assert distance == 0.5
    assert face == FACE_BOTTOM
    assert x_out == 0.25
    assert y_out == 0.0
    assert coord == 0.25


def test_gate_probability_and_truncated_first_collision_are_well_behaved():
    assert scatter_before_exit_probability(1.0, 0.0) == 0.0
    assert scatter_before_exit_probability(0.0, 1.0) == 0.0
    assert scatter_before_exit_probability(1.0, 0.75) == np.float64(1.0 - math.exp(-0.75))

    ballistic_length = 0.75
    samples = np.array(
        [
            sample_truncated_first_collision_distance(unit, 1.0, ballistic_length)
            for unit in np.linspace(0.01, 0.99, 9)
        ],
        dtype=np.float64,
    )
    assert np.all(samples > 0.0)
    assert np.all(samples < ballistic_length)


def test_gate_wrapper_adds_face_local_features():
    df = get_2d_hierarchical_steady_gate_data(
        source="boundary",
        num_neutrons=32,
        sampling="lhs",
    )

    assert len(df) == 32
    assert {"ballistic_face_name", "ballistic_face_length", "ballistic_face_coord_local_norm"} <= set(df.columns)
    assert set(df["ballistic_face"].unique()).issubset({FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM})
    assert df["ballistic_face_coord_local_norm"].between(0.0, 1.0).all()
    assert df["scatter_before_exit_prob"].between(0.0, 1.0).all()
    assert set(df["scatter_before_exit"].unique()).issubset({0, 1})


def test_scattered_only_wrapper_returns_consistent_face_first_targets():
    df = get_2d_hierarchical_steady_scattered_only_data(
        source="internal",
        num_neutrons=32,
        sampling="lhs",
    )

    assert len(df) == 32
    assert (df["scatter_before_exit"] == 1).all()
    assert (df["num_scatters"] >= 1).all()
    assert (df["first_scatter_distance"] > 0.0).all()
    assert (df["first_scatter_distance"] < df["ballistic_length"]).all()
    assert (df["total_track_length"] >= df["first_scatter_distance"]).all()
    assert df["exit_face_coord_local_norm"].between(0.0, 1.0).all()
    assert (df["first_scatter_distance_frac_ballistic"] > 0.0).all()
    assert (df["first_scatter_distance_frac_ballistic"] < 1.0).all()
    assert (df["final_dir_outward_compatible"] == 1).all()

    for face, u_val, v_val in zip(df["exit_face"], df["final_dir_u"], df["final_dir_v"]):
        assert direction_is_outward_for_face_2d(int(face), float(u_val), float(v_val))
