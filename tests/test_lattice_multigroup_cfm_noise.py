from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.lattice_multigroup_steady import build_particle_model_noise


def test_particle_model_noise_is_independent_of_batch_companions():
    target_dim = 5
    state_joint = np.array([11, 22, 33], dtype=np.uint64)
    state_solo = np.array([11, 22, 33], dtype=np.uint64)

    noise_joint_first = build_particle_model_noise(
        state_joint,
        np.array([0, 2], dtype=np.int64),
        target_dim,
        device="cpu",
        dtype=torch.float32,
    )
    noise_joint_second = build_particle_model_noise(
        state_joint,
        np.array([2], dtype=np.int64),
        target_dim,
        device="cpu",
        dtype=torch.float32,
    )

    noise_solo_first = build_particle_model_noise(
        state_solo,
        np.array([2], dtype=np.int64),
        target_dim,
        device="cpu",
        dtype=torch.float32,
    )
    noise_solo_second = build_particle_model_noise(
        state_solo,
        np.array([2], dtype=np.int64),
        target_dim,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.allclose(noise_joint_first[1], noise_solo_first[0])
    assert torch.allclose(noise_joint_second[0], noise_solo_second[0])


def test_particle_model_noise_leaves_inactive_states_unchanged():
    state = np.array([101, 202, 303], dtype=np.uint64)
    original = state.copy()

    _ = build_particle_model_noise(
        state,
        np.array([1], dtype=np.int64),
        4,
        device="cpu",
        dtype=torch.float32,
    )

    assert state[0] == original[0]
    assert state[2] == original[2]
    assert state[1] != original[1]
