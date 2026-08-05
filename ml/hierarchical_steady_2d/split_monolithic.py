from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.transform import (
    denorm_angle_phi,
    denorm_log_scale_robust,
    denorm_position_relative,
    denorm_radius_r,
    normalize_angle_phi,
    normalize_log_scale_robust,
    normalize_position_relative,
    normalize_radius_r,
    normalize_width,
    normalize_height,
    polar_to_uv,
    uv_to_polar,
)
from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    ballistic_exit_2d,
    scatter_before_exit_probability,
)
from physics.mc_data_gen import get_2d_factorized_hierarchical_steady_scattered_only_data

from .models import ConditionalFlowModel, load_checkpoint, save_checkpoint


EPSILON = 1.0e-12
STAGE_NAME = "monolithic"
FACE_NAMES = ("left", "right", "top", "bottom")
TARGET_NAMES = (
    "final_pos_x_norm",
    "final_pos_y_norm",
    "final_dir_phi",
    "final_dir_r",
    "total_track_length_log10",
)


def _analytic_log_bounds(min_value: float, max_value: float, label: str) -> tuple[float, float]:
    if min_value <= 0.0 or max_value <= 0.0:
        raise ValueError(f"{label} bounds must be > 0, got [{min_value}, {max_value}]")
    if max_value <= min_value:
        raise ValueError(f"{label} bounds must satisfy max > min, got [{min_value}, {max_value}]")
    return float(np.log10(min_value)), float(np.log10(max_value))


def _robust_log_stats(values: np.ndarray) -> tuple[float, float]:
    data = np.log10(np.asarray(values, dtype=np.float64) + EPSILON)
    median = float(np.median(data))
    iqr = float(np.percentile(data, 75.0) - np.percentile(data, 25.0))
    scale = max(iqr / 1.349, 1.0e-6)
    return median, scale


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _face_one_hot(face_values: np.ndarray, *, prefix: str) -> tuple[np.ndarray, tuple[str, ...]]:
    face_ids = np.asarray(face_values, dtype=np.int64)
    one_hot = np.eye(4, dtype=np.float32)[face_ids]
    names = tuple(f"{prefix}_{name}" for name in FACE_NAMES)
    return one_hot, names


def _face_coord_norm(face: np.ndarray, coord: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    lengths = np.where(
        (face == FACE_LEFT) | (face == FACE_RIGHT),
        height,
        width,
    )
    return np.clip(coord / (lengths + EPSILON), 0.0, 1.0)


def _classify_face_and_snap_numpy(
    x: np.ndarray,
    y: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    distances = np.stack(
        [
            np.abs(x),
            np.abs(width - x),
            np.abs(height - y),
            np.abs(y),
        ],
        axis=1,
    )
    face = np.argmin(distances, axis=1).astype(np.int64)
    x_snap = np.clip(x, 0.0, width)
    y_snap = np.clip(y, 0.0, height)
    x_snap = np.where(face == FACE_LEFT, 0.0, np.where(face == FACE_RIGHT, width, x_snap))
    y_snap = np.where(face == FACE_BOTTOM, 0.0, np.where(face == FACE_TOP, height, y_snap))
    pos_on_face = np.where((face == FACE_LEFT) | (face == FACE_RIGHT), y_snap, x_snap)
    return face, x_snap.astype(np.float32), y_snap.astype(np.float32), pos_on_face.astype(np.float32)


def _enforce_outward_numpy(face: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    out_u = np.asarray(u, dtype=np.float64).copy()
    out_v = np.asarray(v, dtype=np.float64).copy()

    radius = np.sqrt(np.maximum(out_u * out_u + out_v * out_v, 0.0))
    large = radius > 1.0
    if np.any(large):
        out_u[large] /= radius[large]
        out_v[large] /= radius[large]

    normals = np.zeros((face.shape[0], 2), dtype=np.float64)
    normals[face == FACE_LEFT] = (-1.0, 0.0)
    normals[face == FACE_RIGHT] = (1.0, 0.0)
    normals[face == FACE_TOP] = (0.0, 1.0)
    normals[face == FACE_BOTTOM] = (0.0, -1.0)

    direction = np.stack([out_u, out_v], axis=1)
    dot = np.sum(direction * normals, axis=1)
    needs_flip = dot <= 0.0
    if np.any(needs_flip):
        direction[needs_flip] = direction[needs_flip] - 2.0 * dot[needs_flip, None] * normals[needs_flip]

    tiny = np.linalg.norm(direction, axis=1) < 1.0e-8
    if np.any(tiny):
        direction[tiny] = 1.0e-8 * normals[tiny]

    diagnostics = {
        "direction_unit_disk_clip_fraction": float(np.mean(large)) if large.size else 0.0,
        "direction_outward_reflect_fraction": float(np.mean(needs_flip)) if needs_flip.size else 0.0,
        "direction_tiny_reset_fraction": float(np.mean(tiny)) if tiny.size else 0.0,
    }
    return direction[:, 0].astype(np.float32), direction[:, 1].astype(np.float32), diagnostics


@dataclass(frozen=True)
class SplitMonolithicTrainingConfig:
    source: str = "internal"
    num_neutrons: int = 100000
    sigma_s: float = 1.0
    sigma_a: float = 0.0
    distribution: int = 0
    min_width: float = 0.01
    max_width: float = 10.0
    min_height: float = 0.01
    max_height: float = 10.0
    sampling: str = "random"
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    val_ratio: float = 0.1
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.0
    flow_steps: int = 10
    seed: int = 0
    device: str = "auto"
    output_root: str = "ml/hierarchical_steady_2d/split_runs"
    run_name: str | None = None


@dataclass(frozen=True)
class SplitMonolithicNormalizationBundle:
    source: str
    width_lower: float
    width_upper: float
    height_lower: float
    height_upper: float
    ballistic_length_center: float
    ballistic_length_scale: float
    total_track_length_center: float
    total_track_length_scale: float
    base_feature_names: tuple[str, ...]
    target_names: tuple[str, ...] = TARGET_NAMES

    def encode_inputs(self, frame: pd.DataFrame) -> np.ndarray:
        width = frame["width"].to_numpy(dtype=np.float64, copy=False)
        height = frame["height"].to_numpy(dtype=np.float64, copy=False)
        phi_in, r_in = uv_to_polar(
            frame["initial_dir_u"].to_numpy(dtype=np.float64, copy=False),
            frame["initial_dir_v"].to_numpy(dtype=np.float64, copy=False),
        )
        ballistic_face = frame["ballistic_face"].to_numpy(dtype=np.int64, copy=False)
        ballistic_coord_norm = _face_coord_norm(
            ballistic_face,
            frame["ballistic_face_coord_local"].to_numpy(dtype=np.float64, copy=False),
            width,
            height,
        )
        base_parts = [
            normalize_width(width, self.width_lower, self.width_upper),
            normalize_height(height, self.height_lower, self.height_upper),
        ]
        if self.source == "boundary":
            base_parts.append(
                normalize_position_relative(
                    frame["initial_pos_y"].to_numpy(dtype=np.float64, copy=False) / (height + EPSILON)
                )
            )
        else:
            base_parts.extend(
                [
                    normalize_position_relative(
                        frame["initial_pos_x"].to_numpy(dtype=np.float64, copy=False) / (width + EPSILON)
                    ),
                    normalize_position_relative(
                        frame["initial_pos_y"].to_numpy(dtype=np.float64, copy=False) / (height + EPSILON)
                    ),
                ]
            )
        base_parts.extend(
            [
                normalize_angle_phi(phi_in),
                normalize_radius_r(r_in),
                normalize_log_scale_robust(
                    frame["ballistic_length"].to_numpy(dtype=np.float64, copy=False),
                    self.ballistic_length_center,
                    self.ballistic_length_scale,
                ),
                normalize_position_relative(ballistic_coord_norm),
                normalize_position_relative(
                    np.clip(frame["scatter_before_exit_prob"].to_numpy(dtype=np.float64, copy=False), 0.0, 1.0)
                ),
            ]
        )
        base = np.stack(base_parts, axis=1).astype(np.float32, copy=False)
        face_one_hot, _ = _face_one_hot(ballistic_face, prefix="ballistic_face_is")
        return np.concatenate([base, face_one_hot], axis=1).astype(np.float32, copy=False)

    def encode_targets(self, frame: pd.DataFrame) -> np.ndarray:
        phi_out, r_out = uv_to_polar(
            frame["final_dir_u"].to_numpy(dtype=np.float64, copy=False),
            frame["final_dir_v"].to_numpy(dtype=np.float64, copy=False),
        )
        width = frame["width"].to_numpy(dtype=np.float64, copy=False)
        height = frame["height"].to_numpy(dtype=np.float64, copy=False)
        targets = np.stack(
            [
                normalize_position_relative(
                    frame["final_pos_x"].to_numpy(dtype=np.float64, copy=False) / (width + EPSILON)
                ),
                normalize_position_relative(
                    frame["final_pos_y"].to_numpy(dtype=np.float64, copy=False) / (height + EPSILON)
                ),
                normalize_angle_phi(phi_out),
                normalize_radius_r(r_out),
                normalize_log_scale_robust(
                    frame["total_track_length"].to_numpy(dtype=np.float64, copy=False),
                    self.total_track_length_center,
                    self.total_track_length_scale,
                ),
            ],
            axis=1,
        )
        return np.asarray(targets, dtype=np.float32)

    def decode_targets(self, values: np.ndarray) -> dict[str, np.ndarray]:
        raw = np.asarray(values, dtype=np.float64)
        x_rel = denorm_position_relative(raw[:, 0])
        y_rel = denorm_position_relative(raw[:, 1])
        phi = denorm_angle_phi(raw[:, 2])
        radius = denorm_radius_r(raw[:, 3])
        track = denorm_log_scale_robust(
            raw[:, 4],
            self.total_track_length_center,
            self.total_track_length_scale,
        )
        return {
            "x_rel": np.asarray(x_rel, dtype=np.float64),
            "y_rel": np.asarray(y_rel, dtype=np.float64),
            "phi": np.asarray(phi, dtype=np.float64),
            "radius": np.asarray(radius, dtype=np.float64),
            "track_length": np.asarray(track, dtype=np.float64),
        }

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "width_lower": self.width_lower,
            "width_upper": self.width_upper,
            "height_lower": self.height_lower,
            "height_upper": self.height_upper,
            "ballistic_length_center": self.ballistic_length_center,
            "ballistic_length_scale": self.ballistic_length_scale,
            "total_track_length_center": self.total_track_length_center,
            "total_track_length_scale": self.total_track_length_scale,
            "base_feature_names": list(self.base_feature_names),
            "target_names": list(self.target_names),
        }

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "normalization.json"
        json_path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True))
        return json_path

    @classmethod
    def load(cls, path: str | Path) -> "SplitMonolithicNormalizationBundle":
        payload = json.loads(Path(path).read_text())
        return cls(
            source=str(payload["source"]),
            width_lower=float(payload["width_lower"]),
            width_upper=float(payload["width_upper"]),
            height_lower=float(payload["height_lower"]),
            height_upper=float(payload["height_upper"]),
            ballistic_length_center=float(payload["ballistic_length_center"]),
            ballistic_length_scale=float(payload["ballistic_length_scale"]),
            total_track_length_center=float(payload["total_track_length_center"]),
            total_track_length_scale=float(payload["total_track_length_scale"]),
            base_feature_names=tuple(payload["base_feature_names"]),
            target_names=tuple(payload["target_names"]),
        )


@dataclass(frozen=True)
class SplitMonolithicSampleBatch:
    x_out: np.ndarray
    y_out: np.ndarray
    u_out: np.ndarray
    v_out: np.ndarray
    track_length: np.ndarray
    exit_face: np.ndarray
    pos_on_face: np.ndarray
    diagnostics: dict[str, float]


def _base_feature_names(source: str) -> tuple[str, ...]:
    common = (
        "initial_dir_phi",
        "initial_dir_r",
        "ballistic_length",
        "ballistic_face_coord_local_norm",
        "scatter_before_exit_prob",
        "ballistic_face_is_left",
        "ballistic_face_is_right",
        "ballistic_face_is_top",
        "ballistic_face_is_bottom",
    )
    if source == "boundary":
        return ("width", "height", "entry_y_norm") + common
    return ("width", "height", "initial_pos_x_norm", "initial_pos_y_norm") + common


def fit_split_monolithic_normalization(
    frame: pd.DataFrame,
    *,
    source: str,
    min_width: float,
    max_width: float,
    min_height: float,
    max_height: float,
) -> SplitMonolithicNormalizationBundle:
    width_lower, width_upper = _analytic_log_bounds(min_width, max_width, "width")
    height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
    ballistic_center, ballistic_scale = _robust_log_stats(frame["ballistic_length"].to_numpy(dtype=np.float64, copy=False))
    track_center, track_scale = _robust_log_stats(frame["total_track_length"].to_numpy(dtype=np.float64, copy=False))
    return SplitMonolithicNormalizationBundle(
        source=source,
        width_lower=width_lower,
        width_upper=width_upper,
        height_lower=height_lower,
        height_upper=height_upper,
        ballistic_length_center=ballistic_center,
        ballistic_length_scale=ballistic_scale,
        total_track_length_center=track_center,
        total_track_length_scale=track_scale,
        base_feature_names=_base_feature_names(source),
    )


def _split_dataframe(df: pd.DataFrame, *, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least two rows for split-only training.")
    rng = np.random.default_rng(seed)
    val_indices: list[int] = []
    for _, group in df.groupby("exit_face", sort=True):
        group_idx = group.index.to_numpy(dtype=np.int64, copy=True)
        if group_idx.size <= 1:
            continue
        rng.shuffle(group_idx)
        val_count = max(1, int(round(group_idx.size * val_ratio)))
        val_count = min(val_count, group_idx.size - 1)
        val_indices.extend(group_idx[:val_count].tolist())
    if val_indices:
        val_idx = np.array(sorted(set(val_indices)), dtype=np.int64)
        return df.drop(index=val_idx).reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)
    perm = rng.permutation(len(df))
    val_count = min(max(int(round(len(df) * val_ratio)), 1), len(df) - 1)
    return df.iloc[perm[val_count:]].reset_index(drop=True), df.iloc[perm[:val_count]].reset_index(drop=True)


def _make_loader(inputs: np.ndarray, targets: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(targets, dtype=np.float32, copy=True)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate_model(
    model: ConditionalFlowModel,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: SplitMonolithicNormalizationBundle,
    seed: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        for cond, target in loader:
            cond = cond.to(device)
            target = target.to(device)
            x0 = torch.randn(target.shape[0], target.shape[1], device=device, generator=generator)
            loss = model.flow_matching_loss(x0, target, cond)
            pred = model.sample(x0, cond, generator=generator)
            batch_size = cond.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            pred_batches.append(pred.cpu().numpy())
            target_batches.append(target.cpu().numpy())
    pred = np.concatenate(pred_batches, axis=0)
    target = np.concatenate(target_batches, axis=0)
    pred_dec = normalization.decode_targets(pred)
    target_dec = normalization.decode_targets(target)
    pred_u, pred_v = polar_to_uv(pred_dec["phi"], pred_dec["radius"])
    target_u, target_v = polar_to_uv(target_dec["phi"], target_dec["radius"])
        metrics = {
            "mae_x_out_norm": float(np.mean(np.abs(pred_dec["x_rel"] - target_dec["x_rel"]))),
            "mae_y_out_norm": float(np.mean(np.abs(pred_dec["y_rel"] - target_dec["y_rel"]))),
            "mae_final_dir_u": float(np.mean(np.abs(pred_u - target_u))),
            "mae_final_dir_v": float(np.mean(np.abs(pred_v - target_v))),
            "mae_total_track_length_log10": float(
                np.mean(
                    np.abs(
                        np.log10(np.maximum(pred_dec["track_length"], EPSILON))
                        - np.log10(np.maximum(target_dec["track_length"], EPSILON))
                    )
                )
            ),
            "loss": float(total_loss / max(1, total_count)),
        }
    return metrics["loss"], metrics


def run_split_monolithic_training(
    config: SplitMonolithicTrainingConfig,
    *,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, object]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    if dataframe is None:
        dataframe = get_2d_factorized_hierarchical_steady_scattered_only_data(
            source=config.source,
            num_neutrons=config.num_neutrons,
            sigma_s=config.sigma_s,
            sigma_a=config.sigma_a,
            distribution=config.distribution,
            min_width=config.min_width,
            max_width=config.max_width,
            min_height=config.min_height,
            max_height=config.max_height,
            sampling=config.sampling,
        )
    train_df, val_df = _split_dataframe(dataframe, val_ratio=config.val_ratio, seed=config.seed)
    normalization = fit_split_monolithic_normalization(
        train_df,
        source=config.source,
        min_width=config.min_width,
        max_width=config.max_width,
        min_height=config.min_height,
        max_height=config.max_height,
    )

    train_inputs = normalization.encode_inputs(train_df)
    train_targets = normalization.encode_targets(train_df)
    val_inputs = normalization.encode_inputs(val_df)
    val_targets = normalization.encode_targets(val_df)

    run_name = config.run_name or f"split_{config.source}_{config.num_neutrons}"
    run_dir = Path(config.output_root) / run_name
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    normalization_path = normalization.save(run_dir / "metadata")
    (run_dir / "metadata" / "run_config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))

    model = ConditionalFlowModel(
        condition_dim=train_inputs.shape[1],
        target_dim=train_targets.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        ode_steps=config.flow_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = _make_loader(train_inputs, train_targets, batch_size=config.batch_size, shuffle=True)
    val_loader = _make_loader(val_inputs, val_targets, batch_size=max(config.batch_size, 256), shuffle=False)

    best_val_loss = math.inf
    train_history: list[float] = []
    val_history: list[float] = []
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(config.seed + epoch))
        for cond, target in train_loader:
            cond = cond.to(device)
            target = target.to(device)
            x0 = torch.randn(target.shape[0], target.shape[1], device=device, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = model.flow_matching_loss(x0, target, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = cond.shape[0]
            running += float(loss.item()) * batch_size
            seen += batch_size
        train_history.append(running / max(1, seen))
        val_loss, _ = _evaluate_model(
            model,
            val_loader,
            device=device,
            normalization=normalization,
            seed=config.seed + 1000 + epoch,
        )
        val_history.append(float(val_loss))
        if val_loss <= best_val_loss:
            best_val_loss = float(val_loss)
            save_checkpoint(
                run_dir / STAGE_NAME / "models" / "best_model.pt",
                model,
                metadata={
                    "stage": STAGE_NAME,
                    "variant": "split_monolithic_flow",
                    "source": config.source,
                    "normalization_path": str(normalization_path),
                    "input_names": list(normalization.base_feature_names),
                    "target_names": list(normalization.target_names),
                },
            )

    save_checkpoint(
        run_dir / STAGE_NAME / "models" / "final_model.pt",
        model,
        metadata={
            "stage": STAGE_NAME,
            "variant": "split_monolithic_flow",
            "source": config.source,
            "normalization_path": str(normalization_path),
            "input_names": list(normalization.base_feature_names),
            "target_names": list(normalization.target_names),
        },
    )

    best_model, _cfg, _meta = load_checkpoint(run_dir / STAGE_NAME / "models" / "best_model.pt", map_location=device)
    if not isinstance(best_model, ConditionalFlowModel):
        raise TypeError("Split-only monolithic checkpoint must contain a ConditionalFlowModel.")
    best_model = best_model.to(device)
    final_val_loss, final_metrics = _evaluate_model(
        best_model,
        val_loader,
        device=device,
        normalization=normalization,
        seed=config.seed + 2000,
    )
    metrics_dir = run_dir / STAGE_NAME / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_val_cfm_loss": float(best_val_loss),
        **final_metrics,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    np.savez(
        metrics_dir / "history.npz",
        train=np.asarray(train_history, dtype=np.float32),
        val=np.asarray(val_history, dtype=np.float32),
    )
    result = {
        "run_dir": str(run_dir),
        STAGE_NAME: {
            "best_val_loss": float(best_val_loss),
            "metrics": metrics_payload,
            "final_val_loss": float(final_val_loss),
        },
    }
    (run_dir / "metadata" / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


class SplitMonolithicSteady2DSampler:
    def __init__(
        self,
        *,
        source_type: str,
        model: ConditionalFlowModel,
        normalization: SplitMonolithicNormalizationBundle,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.source_type = str(source_type)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.model = model.to(self.device).eval()
        self.normalization = normalization

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str | Path,
        *,
        source_type: str,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> "SplitMonolithicSteady2DSampler":
        resolved = Path(run_dir)
        normalization = SplitMonolithicNormalizationBundle.load(resolved / "metadata" / "normalization.json")
        if normalization.source != str(source_type):
            raise ValueError(
                f"Split-only run {resolved} was trained for source={normalization.source!r}, not {source_type!r}."
            )
        model, _cfg, meta = load_checkpoint(resolved / STAGE_NAME / "models" / "best_model.pt", map_location=device)
        if not isinstance(model, ConditionalFlowModel):
            raise TypeError("Split-only monolithic checkpoint must contain a ConditionalFlowModel.")
        if str(meta.get("stage", "")) != STAGE_NAME:
            raise ValueError(f"Expected stage={STAGE_NAME!r} in {resolved / STAGE_NAME / 'models' / 'best_model.pt'}")
        return cls(
            source_type=source_type,
            model=model,
            normalization=normalization,
            device=device,
            batch_size=batch_size,
        )

    def _predict_batched(self, inputs: np.ndarray, *, generator: torch.Generator) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.array(inputs, dtype=np.float32, copy=True))),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        device_obj = torch.device(self.device)
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for (cond,) in loader:
                cond = cond.to(device_obj)
                noise = torch.randn(cond.shape[0], self.model.target_dim, device=device_obj, generator=generator)
                pred = self.model.sample(noise, cond, generator=generator)
                batches.append(pred.cpu().numpy())
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, self.model.target_dim), dtype=np.float32)

    def sample_batch(
        self,
        *,
        width: np.ndarray,
        height: np.ndarray,
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        initial_u: np.ndarray,
        initial_v: np.ndarray,
        sigma_t: np.ndarray,
        seed: int | None = None,
    ) -> SplitMonolithicSampleBatch:
        width = np.asarray(width, dtype=np.float64).reshape(-1)
        height = np.asarray(height, dtype=np.float64).reshape(-1)
        initial_x = np.asarray(initial_x, dtype=np.float64).reshape(-1)
        initial_y = np.asarray(initial_y, dtype=np.float64).reshape(-1)
        initial_u = np.asarray(initial_u, dtype=np.float64).reshape(-1)
        initial_v = np.asarray(initial_v, dtype=np.float64).reshape(-1)
        sigma_t = np.maximum(np.asarray(sigma_t, dtype=np.float64).reshape(-1), 1.0e-6)

        num_rows = width.shape[0]
        if not all(arr.shape[0] == num_rows for arr in (height, initial_x, initial_y, initial_u, initial_v, sigma_t)):
            raise ValueError("All split-only sampler input arrays must have matching first dimension.")
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device if self.device.startswith("cuda") else "cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        else:
            generator.seed()

        ballistic_length = np.zeros(num_rows, dtype=np.float64)
        ballistic_face = np.zeros(num_rows, dtype=np.int64)
        ballistic_exit_x = np.zeros(num_rows, dtype=np.float64)
        ballistic_exit_y = np.zeros(num_rows, dtype=np.float64)
        ballistic_coord = np.zeros(num_rows, dtype=np.float64)
        scatter_prob = np.zeros(num_rows, dtype=np.float64)
        for idx in range(num_rows):
            distance, face, x_exit, y_exit, coord = ballistic_exit_2d(
                float(initial_x[idx]),
                float(initial_y[idx]),
                float(initial_u[idx]),
                float(initial_v[idx]),
                float(width[idx]),
                float(height[idx]),
            )
            ballistic_length[idx] = float(distance)
            ballistic_face[idx] = int(face)
            ballistic_exit_x[idx] = float(x_exit)
            ballistic_exit_y[idx] = float(y_exit)
            ballistic_coord[idx] = float(coord)
            scatter_prob[idx] = float(scatter_before_exit_probability(float(sigma_t[idx]), float(distance)))

        scatter_mask = rng.random(num_rows) < scatter_prob
        final_x = ballistic_exit_x.copy()
        final_y = ballistic_exit_y.copy()
        final_u = initial_u.astype(np.float64, copy=True)
        final_v = initial_v.astype(np.float64, copy=True)
        final_track = ballistic_length.copy()
        final_face = ballistic_face.copy()
        final_pos_on_face = ballistic_coord.copy()

        diagnostics = {
            "scatter_fraction_sampled": float(np.mean(scatter_mask)) if num_rows else 0.0,
            "scatter_probability_mean": float(np.mean(scatter_prob)) if num_rows else 0.0,
        }
        if np.any(scatter_mask):
            scatter_idx = np.flatnonzero(scatter_mask)
            scatter_frame = pd.DataFrame(
                {
                    "width": width[scatter_idx],
                    "height": height[scatter_idx],
                    "initial_pos_x": initial_x[scatter_idx],
                    "initial_pos_y": initial_y[scatter_idx],
                    "initial_dir_u": initial_u[scatter_idx],
                    "initial_dir_v": initial_v[scatter_idx],
                    "ballistic_length": ballistic_length[scatter_idx],
                    "ballistic_face": ballistic_face[scatter_idx],
                    "ballistic_face_coord_local": ballistic_coord[scatter_idx],
                    "scatter_before_exit_prob": scatter_prob[scatter_idx],
                }
            )
            cond = self.normalization.encode_inputs(scatter_frame)
            pred_norm = self._predict_batched(cond, generator=generator)
            decoded = self.normalization.decode_targets(pred_norm)
            x_phys = decoded["x_rel"] * width[scatter_idx]
            y_phys = decoded["y_rel"] * height[scatter_idx]
            face, x_snap, y_snap, pos_on_face = _classify_face_and_snap_numpy(
                x_phys,
                y_phys,
                width[scatter_idx],
                height[scatter_idx],
            )
            u_pred, v_pred = polar_to_uv(decoded["phi"], decoded["radius"])
            u_out, v_out, direction_diag = _enforce_outward_numpy(face, u_pred, v_pred)
            diagnostics.update({f"{key}_given_scatter": float(value) for key, value in direction_diag.items()})
            track_physical = decoded["track_length"]
            straight_line_min = np.sqrt(
                np.maximum((x_snap.astype(np.float64) - initial_x[scatter_idx]) ** 2 + (y_snap.astype(np.float64) - initial_y[scatter_idx]) ** 2, 0.0)
            )
            diagnostics["track_lower_bound_clip_fraction_given_scatter"] = float(np.mean(track_physical < straight_line_min))
            track_physical = np.maximum(track_physical, straight_line_min)

            final_x[scatter_idx] = x_snap.astype(np.float64)
            final_y[scatter_idx] = y_snap.astype(np.float64)
            final_u[scatter_idx] = u_out.astype(np.float64)
            final_v[scatter_idx] = v_out.astype(np.float64)
            final_track[scatter_idx] = track_physical.astype(np.float64)
            final_face[scatter_idx] = face.astype(np.int64)
            final_pos_on_face[scatter_idx] = pos_on_face.astype(np.float64)
        else:
            diagnostics["direction_unit_disk_clip_fraction_given_scatter"] = 0.0
            diagnostics["direction_outward_reflect_fraction_given_scatter"] = 0.0
            diagnostics["track_lower_bound_clip_fraction_given_scatter"] = 0.0

        return SplitMonolithicSampleBatch(
            x_out=final_x.astype(np.float32),
            y_out=final_y.astype(np.float32),
            u_out=final_u.astype(np.float32),
            v_out=final_v.astype(np.float32),
            track_length=np.maximum(final_track, 0.0).astype(np.float32),
            exit_face=final_face.astype(np.int64),
            pos_on_face=final_pos_on_face.astype(np.float32),
            diagnostics=diagnostics,
        )
