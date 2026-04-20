from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import math

import torch
from torch import nn


@dataclass(frozen=True)
class ClassifierConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0
    model_type: str = "classifier"


@dataclass(frozen=True)
class GaussianRegressorConfig:
    input_dim: int
    target_dim: int
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0
    model_type: str = "gaussian_regressor"


@dataclass(frozen=True)
class HierarchicalMLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.0
    model_type: str = "hierarchical_mlp"


class _MlpBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        self.network = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FaceClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, *, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.backbone = _MlpBackbone(input_dim, hidden_dim, depth, dropout)
        self.head = nn.Linear(self.backbone.output_dim, int(num_classes))
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def config(self) -> ClassifierConfig:
        return ClassifierConfig(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )


class GaussianRegressor(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, *, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.backbone = _MlpBackbone(input_dim, hidden_dim, depth, dropout)
        self.mean_head = nn.Linear(self.backbone.output_dim, int(target_dim))
        self.log_std_head = nn.Linear(self.backbone.output_dim, int(target_dim))
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(min=-6.0, max=2.0)
        return mean, log_std

    def nll(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        inv_var = torch.exp(-2.0 * log_std)
        loss = 0.5 * (((target - mean) ** 2) * inv_var + 2.0 * log_std + math.log(2.0 * math.pi))
        return loss.mean(), mean, log_std

    @torch.inference_mode()
    def sample(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        mean, log_std = self.forward(x)
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean + torch.exp(log_std) * noise

    def config(self) -> GaussianRegressorConfig:
        return GaussianRegressorConfig(
            input_dim=self.input_dim,
            target_dim=self.target_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )


class HierarchicalMLP(nn.Module):
    """Lightweight shared MLP used by the standalone training loop."""

    def __init__(self, input_dim: int, output_dim: int, *, hidden_dim: int = 128, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.backbone = _MlpBackbone(input_dim, hidden_dim, depth, dropout)
        self.head = nn.Linear(self.backbone.output_dim, int(output_dim))
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def config(self) -> HierarchicalMLPConfig:
        return HierarchicalMLPConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )


def save_checkpoint(path: str | Path, model: nn.Module, *, metadata: dict[str, Any] | None = None) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, FaceClassifier):
        config = asdict(model.config())
        model_type = "classifier"
    elif isinstance(model, GaussianRegressor):
        config = asdict(model.config())
        model_type = "gaussian_regressor"
    elif isinstance(model, HierarchicalMLP):
        config = asdict(model.config())
        model_type = "hierarchical_mlp"
    else:
        raise TypeError(f"Unsupported model type for checkpoint save: {type(model)!r}")

    torch.save(
        {
            "model_type": model_type,
            "config": config,
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        resolved,
    )


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=map_location)
    model_type = str(payload["model_type"])
    config = dict(payload["config"])
    if model_type == "classifier":
        model = FaceClassifier(
            config["input_dim"],
            config["num_classes"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            dropout=config["dropout"],
        )
    elif model_type == "gaussian_regressor":
        model = GaussianRegressor(
            config["input_dim"],
            config["target_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            dropout=config["dropout"],
        )
    elif model_type == "hierarchical_mlp":
        model = HierarchicalMLP(
            config["input_dim"],
            config["output_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unsupported hierarchical checkpoint model_type '{model_type}'")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, dict(payload.get("metadata", {}))
