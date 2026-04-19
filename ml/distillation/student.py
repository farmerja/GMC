from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class StudentConfig:
    cond_dim: int
    target_dim: int
    num_steps: int
    hidden_dim: int = 128
    depth: int = 3
    expansion_factor: int = 2


class StudentBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int = 2):
        super().__init__()
        inner_dim = hidden_dim * max(1, int(expansion_factor))
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.conditioning = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        nn.init.zeros_(self.conditioning.weight)
        nn.init.zeros_(self.conditioning.bias)

    def forward(self, x: torch.Tensor, combined_cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x_ff = self.ff(self.norm(x))
        scale, shift = self.conditioning(combined_cond).chunk(2, dim=-1)
        return residual + x_ff * (1.0 + scale) + shift


class FixedStepStudent(nn.Module):
    """Predict one GMC target from conditioning features and latent noise."""

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        num_steps: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        expansion_factor: int = 2,
    ):
        super().__init__()
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}")
        if target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.cond_dim = int(cond_dim)
        self.target_dim = int(target_dim)
        self.num_steps = int(num_steps)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.expansion_factor = int(expansion_factor)

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.state_proj = nn.Linear(self.target_dim, self.hidden_dim)
        self.step_embed = nn.Embedding(self.num_steps, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [StudentBlock(self.hidden_dim, expansion_factor=self.expansion_factor) for _ in range(self.depth)]
        )
        self.output = nn.Linear(self.hidden_dim, self.target_dim)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output.bias)

    def _single_step(self, cond_emb: torch.Tensor, state: torch.Tensor, step_idx: int) -> torch.Tensor:
        step_ids = torch.full((state.shape[0],), int(step_idx), device=state.device, dtype=torch.long)
        step_emb = self.step_embed(step_ids)
        combined_cond = torch.cat([cond_emb, step_emb], dim=-1)
        h = self.state_proj(state)
        for block in self.blocks:
            h = block(h, combined_cond)
        return state + self.output(h)

    def forward(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"Expected cond shape [batch, {self.cond_dim}], got {tuple(cond.shape)}"
            )
        if noise.ndim != 2 or noise.shape != (cond.shape[0], self.target_dim):
            raise ValueError(
                f"Expected noise shape [{cond.shape[0]}, {self.target_dim}], got {tuple(noise.shape)}"
            )
        cond_emb = self.cond_proj(cond)
        state = noise
        for step_idx in range(self.num_steps):
            state = self._single_step(cond_emb, state, step_idx)
        return state

    @torch.inference_mode()
    def sample_target(
        self,
        conditions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        noise_seed: int | None = None,
    ) -> torch.Tensor:
        if noise is None:
            generator = torch.Generator(device=conditions.device if conditions.is_cuda else "cpu")
            if noise_seed is not None:
                generator.manual_seed(int(noise_seed))
            noise = torch.randn(
                (conditions.shape[0], self.target_dim),
                generator=generator,
                device=conditions.device,
                dtype=conditions.dtype,
            )
        return self.forward(conditions, noise)

    def config(self) -> StudentConfig:
        return StudentConfig(
            cond_dim=self.cond_dim,
            target_dim=self.target_dim,
            num_steps=self.num_steps,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            expansion_factor=self.expansion_factor,
        )


def save_student_checkpoint(
    path: str | Path,
    model: FixedStepStudent,
    *,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config()),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_student_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[FixedStepStudent, StudentConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    config = StudentConfig(**payload["config"])
    model = FixedStepStudent(**asdict(config))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, dict(payload.get("metadata", {}))
