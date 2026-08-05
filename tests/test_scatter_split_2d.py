import math

import numpy as np

from physics.scatter_split_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    ballistic_exit_2d,
    exit_weight_after_absorption,
    generate_scattered_only_2d_dataframe,
    scatter_before_exit_probability,
    weighted_track_integral,
)


def test_ballistic_exit_2d_hits_right_face_for_axis_aligned_track():
    distance, face, x_exit, y_exit, pos_on_face = ballistic_exit_2d(
        0.25,
        0.5,
        1.0,
        0.0,
        1.0,
        2.0,
    )

    assert distance == 0.75
    assert face == FACE_RIGHT
    assert x_exit == 1.0
    assert y_exit == 0.5
    assert pos_on_face == 0.5


def test_scatter_probability_matches_closed_form():
    sigma_s = np.array([0.0, 0.5, 2.0], dtype=np.float64)
    distance = np.array([1.0, 3.0, 0.25], dtype=np.float64)
    expected = 1.0 - np.exp(-sigma_s * distance)
    actual = scatter_before_exit_probability(sigma_s, distance)
    assert np.allclose(actual, expected)


def test_absorption_helpers_match_exact_integrals():
    weight = np.array([2.0], dtype=np.float64)
    sigma_a = np.array([0.5], dtype=np.float64)
    distance = np.array([3.0], dtype=np.float64)

    exit_weight = exit_weight_after_absorption(weight, sigma_a, distance)[0]
    track = weighted_track_integral(weight, sigma_a, distance)[0]

    assert exit_weight == 2.0 * math.exp(-1.5)
    assert track == 2.0 * (1.0 - math.exp(-1.5)) / 0.5


def test_scattered_only_dataframe_has_conditioned_rows():
    frame = generate_scattered_only_2d_dataframe(
        source="internal",
        num_neutrons=32,
        min_width=0.5,
        max_width=1.0,
        min_height=0.5,
        max_height=1.0,
        sigma_s=1.0,
        sampling="lhs",
        seed=7,
    )

    assert len(frame) == 32
    assert (frame["num_scatters"] >= 1.0).all()
    assert (frame["time_in_cell"] > 0.0).all()
    assert (frame["d_entry_exit"] >= 0.0).all()
    assert (frame["time_prime"] >= 0.0).all()
    assert np.allclose(
        frame["time_prime"].values + frame["d_entry_exit"].values,
        frame["time_in_cell"].values,
    )
    assert frame["scatter_before_exit_prob"].between(0.0, 1.0).all()
    assert frame["exit_face"].isin([FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM]).all()
    assert frame["final_pos_on_face_norm"].between(0.0, 1.0).all()
    assert frame["initial_pos_x_norm"].between(0.0, 1.0).all()
    assert frame["initial_pos_y_norm"].between(0.0, 1.0).all()
    assert frame["final_pos_x_norm"].between(0.0, 1.0).all()
    assert frame["final_pos_y_norm"].between(0.0, 1.0).all()

    face = frame["exit_face"].to_numpy(dtype=np.int64, copy=False)
    x_out = frame["final_pos_x"].to_numpy(dtype=np.float64, copy=False)
    y_out = frame["final_pos_y"].to_numpy(dtype=np.float64, copy=False)
    width = frame["width"].to_numpy(dtype=np.float64, copy=False)
    height = frame["height"].to_numpy(dtype=np.float64, copy=False)
    pos_norm = frame["final_pos_on_face_norm"].to_numpy(dtype=np.float64, copy=False)
    pos = frame["final_pos_on_face"].to_numpy(dtype=np.float64, copy=False)

    assert np.allclose(pos, pos_norm * np.where((face == FACE_LEFT) | (face == FACE_RIGHT), height, width))
    assert np.allclose(x_out[face == FACE_LEFT], 0.0)
    assert np.allclose(x_out[face == FACE_RIGHT], width[face == FACE_RIGHT])
    assert np.allclose(y_out[face == FACE_BOTTOM], 0.0)
    assert np.allclose(y_out[face == FACE_TOP], height[face == FACE_TOP])
