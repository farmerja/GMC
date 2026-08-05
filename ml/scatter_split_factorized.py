from __future__ import annotations

import numpy as np
import scipy.special
import torch
from torch import nn

from ml.transform import (
    denorm_position_relative,
    denorm_width,
    denorm_height,
    normalize_angle_phi,
    normalize_height,
    normalize_log_feature,
    normalize_position_relative,
    normalize_radius_r,
    normalize_width,
    polar_to_uv,
    uv_to_polar,
)
from physics.scatter_split_2d import (
    EPSILON,
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
)


FACE_NAMES = ("left", "right", "top", "bottom")
NUM_FACES = 4
SCATTER_SPLIT_VARIANT = "face_factored_v1"
SCATTER_SPLIT_TARGET_DIM = 5


def face_one_hot(face_values: np.ndarray) -> np.ndarray:
    face_ids = np.asarray(face_values, dtype=np.int64).reshape(-1)
    return np.eye(NUM_FACES, dtype=np.float32)[face_ids]


def _face_lengths(face: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    face_arr = np.asarray(face, dtype=np.int64)
    width_arr = np.asarray(width, dtype=np.float64)
    height_arr = np.asarray(height, dtype=np.float64)
    return np.where(
        (face_arr == FACE_LEFT) | (face_arr == FACE_RIGHT),
        height_arr,
        width_arr,
    )


def normalize_face_position(face: np.ndarray, pos_on_face: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    lengths = _face_lengths(face, width, height)
    coord = np.asarray(pos_on_face, dtype=np.float64) / np.maximum(lengths, EPSILON)
    return np.clip(coord, 0.0, 1.0)


def decode_exit_position(face: np.ndarray, coord_norm: np.ndarray, width: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_arr = np.asarray(face, dtype=np.int64).reshape(-1)
    width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
    height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
    coord_01 = np.clip(denorm_position_relative(coord_norm), 0.0, 1.0)

    x_out = np.zeros_like(coord_01, dtype=np.float64)
    y_out = np.zeros_like(coord_01, dtype=np.float64)
    pos_on_face = np.zeros_like(coord_01, dtype=np.float64)

    left = face_arr == FACE_LEFT
    right = face_arr == FACE_RIGHT
    top = face_arr == FACE_TOP
    bottom = face_arr == FACE_BOTTOM

    y_out[left] = coord_01[left] * height_arr[left]
    pos_on_face[left] = y_out[left]

    x_out[right] = width_arr[right]
    y_out[right] = coord_01[right] * height_arr[right]
    pos_on_face[right] = y_out[right]

    x_out[top] = coord_01[top] * width_arr[top]
    y_out[top] = height_arr[top]
    pos_on_face[top] = x_out[top]

    x_out[bottom] = coord_01[bottom] * width_arr[bottom]
    pos_on_face[bottom] = x_out[bottom]

    return (
        x_out.astype(np.float32, copy=False),
        y_out.astype(np.float32, copy=False),
        pos_on_face.astype(np.float32, copy=False),
    )


def encode_boundary_base_condition(
    *,
    width: np.ndarray,
    height: np.ndarray,
    initial_y: np.ndarray,
    initial_u: np.ndarray,
    initial_v: np.ndarray,
    sigma_s: np.ndarray,
    norm_params: dict[str, float],
) -> np.ndarray:
    width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
    height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
    initial_y_arr = np.asarray(initial_y, dtype=np.float64).reshape(-1)
    sigma_s_arr = np.maximum(np.asarray(sigma_s, dtype=np.float64).reshape(-1), EPSILON)
    phi_in, r_in = uv_to_polar(
        np.asarray(initial_u, dtype=np.float64).reshape(-1),
        np.asarray(initial_v, dtype=np.float64).reshape(-1),
    )
    return np.stack(
        [
            normalize_width(width_arr * sigma_s_arr, norm_params["width_lower"], norm_params["width_upper"]),
            normalize_height(height_arr * sigma_s_arr, norm_params["height_lower"], norm_params["height_upper"]),
            normalize_position_relative(np.clip(initial_y_arr / np.maximum(height_arr, EPSILON), 0.0, 1.0)),
            normalize_angle_phi(phi_in),
            normalize_radius_r(r_in),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def encode_internal_base_condition(
    *,
    width: np.ndarray,
    height: np.ndarray,
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    initial_u: np.ndarray,
    initial_v: np.ndarray,
    sigma_s: np.ndarray,
    norm_params: dict[str, float],
) -> np.ndarray:
    width_arr = np.asarray(width, dtype=np.float64).reshape(-1)
    height_arr = np.asarray(height, dtype=np.float64).reshape(-1)
    initial_x_arr = np.asarray(initial_x, dtype=np.float64).reshape(-1)
    initial_y_arr = np.asarray(initial_y, dtype=np.float64).reshape(-1)
    sigma_s_arr = np.maximum(np.asarray(sigma_s, dtype=np.float64).reshape(-1), EPSILON)
    phi_in, r_in = uv_to_polar(
        np.asarray(initial_u, dtype=np.float64).reshape(-1),
        np.asarray(initial_v, dtype=np.float64).reshape(-1),
    )
    return np.stack(
        [
            normalize_width(width_arr * sigma_s_arr, norm_params["width_lower"], norm_params["width_upper"]),
            normalize_height(height_arr * sigma_s_arr, norm_params["height_lower"], norm_params["height_upper"]),
            normalize_position_relative(np.clip(initial_x_arr / np.maximum(width_arr, EPSILON), 0.0, 1.0)),
            normalize_position_relative(np.clip(initial_y_arr / np.maximum(height_arr, EPSILON), 0.0, 1.0)),
            normalize_angle_phi(phi_in),
            normalize_radius_r(r_in),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def encode_flow_targets(frame, norm_params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    exit_face = frame["exit_face"].to_numpy(dtype=np.int64, copy=False)
    phi_out, r_out = uv_to_polar(
        frame["final_dir_u"].to_numpy(dtype=np.float64, copy=False),
        frame["final_dir_v"].to_numpy(dtype=np.float64, copy=False),
    )
    target = np.stack(
        [
            normalize_position_relative(frame["final_pos_on_face_norm"].to_numpy(dtype=np.float64, copy=False)),
            np.sin(phi_out),
            np.cos(phi_out),
            normalize_radius_r(r_out),
            normalize_log_feature(frame["time_prime"].to_numpy(dtype=np.float64, copy=False), norm_params, "time_prime"),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return target, exit_face.astype(np.int64, copy=False)


def sample_scatter_split_source_noise(
    cond: torch.Tensor,
    norm_params: dict[str, float],
    *,
    generator: torch.Generator | None = None,
    np_rng: np.random.Generator | None = None,
) -> torch.Tensor:
    batch_size = cond.shape[0]
    device = cond.device
    rand_kwargs = {"device": device}
    if generator is not None:
        rand_kwargs["generator"] = generator

    noise_coord = torch.randn(batch_size, 1, **rand_kwargs)
    noise_phi = torch.randn(batch_size, 2, **rand_kwargs)

    u_r = torch.rand(batch_size, **rand_kwargs).clamp(1.0e-6, 1.0 - 1.0e-6)
    r0 = u_r.sqrt().clamp(1.0e-6, 1.0 - 1.0e-6)
    noise_r = torch.from_numpy(
        (np.sqrt(2.0) * scipy.special.erfinv(2.0 * r0.detach().cpu().numpy() - 1.0)).astype(np.float32)
    ).to(device).unsqueeze(1)

    width_mfp = denorm_width(
        cond[:, 0],
        norm_params["width_lower"],
        norm_params["width_upper"],
    ).clamp(min=1.0e-6)
    height_mfp = denorm_height(
        cond[:, 1],
        norm_params["height_lower"],
        norm_params["height_upper"],
    ).clamp(min=1.0e-6)
    alpha_np = torch.minimum(width_mfp, height_mfp).clamp(min=2.0).detach().cpu().numpy().astype(np.float64)
    if np_rng is None:
        np_rng = np.random.default_rng()
    gamma_samples = np_rng.gamma(shape=alpha_np, scale=1.0).astype(np.float64, copy=False)
    gamma_samples = np.clip(gamma_samples, 1.0e-12, None)
    noise_l = torch.from_numpy(
        normalize_log_feature(gamma_samples, norm_params, "time_prime").astype(np.float32)
    ).to(device).unsqueeze(1)

    return torch.cat([noise_coord, noise_phi, noise_r, noise_l], dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int = 2) -> None:
        super().__init__()
        ff_dim = max(hidden_dim, hidden_dim * max(int(expansion_factor), 1))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ScatterSplitFaceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 3, expansion_factor: int = 2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.expansion_factor = int(expansion_factor)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(self.hidden_dim, expansion_factor=self.expansion_factor) for _ in range(self.depth)]
        )
        self.output = nn.Linear(self.hidden_dim, NUM_FACES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


def get_face_classifier(
    input_dim: int,
    hidden_dim: int = 128,
    depth: int = 3,
    expansion_factor: int = 2,
) -> ScatterSplitFaceClassifier:
    return ScatterSplitFaceClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        expansion_factor=expansion_factor,
    )
