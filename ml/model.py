#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from flow_matching.utils import ModelWrapper
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler, PolynomialConvexScheduler, LinearVPScheduler, CosineScheduler
import logging
import os
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Model Components ---
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        if half_dim < 1:
            raise ValueError(f"Embedding dimension 'dim' ({self.dim}) must be at least 2.")

        if half_dim == 1:
            inv_freq = torch.ones(1, dtype=torch.float32)
        else:
            scale = np.log(10000.0) / (half_dim - 1)
            inv_freq = torch.exp(-scale * torch.arange(half_dim, dtype=torch.float32))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t):
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        else:
            t = t.reshape(-1)

        t_ = t.float()
        emb = t_.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        if emb.dtype != t.dtype:
            emb = emb.to(dtype=t.dtype)
        return emb

class FlowBlock(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=2):
        super().__init__()
        if expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {expansion_factor}")
        ff_dim = hidden_dim * expansion_factor
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nn.SiLU(), nn.Linear(ff_dim, hidden_dim))
        self.conditioning = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        nn.init.zeros_(self.conditioning.weight)
        nn.init.zeros_(self.conditioning.bias)

    def forward(self, x, combined_cond):
        residual = x
        x_norm = self.norm(x)
        x_ff = self.ff(x_norm)
        scale, shift = self.conditioning(combined_cond).chunk(2, dim=-1)
        return residual + (x_ff * (1 + scale) + shift)

class ConditionalFlow(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=128, depth=4, time_embed_dim=64, expansion_factor=2):
        super().__init__()
        self.input_dim, self.cond_dim = input_dim, cond_dim
        self.target_dim = input_dim - cond_dim
        if self.target_dim <= 0: raise ValueError(f"input_dim ({input_dim}) must be > cond_dim ({cond_dim})")
        if self.cond_dim + self.target_dim != self.input_dim:
            raise ValueError(f"cond_dim + target_dim must equal input_dim ({self.cond_dim} + {self.target_dim} != {self.input_dim})")
        log.info(f"Initializing ConditionalFlow: input_dim={input_dim}, cond_dim={cond_dim}, target_dim={self.target_dim}, hidden_dim={hidden_dim}, depth={depth}, expansion_factor={expansion_factor}")
        self.time_embed = nn.Sequential(SinusoidalEmbedding(time_embed_dim), nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cond_proj = nn.Sequential(nn.Linear(self.cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([FlowBlock(hidden_dim, expansion_factor=expansion_factor) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, self.target_dim)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t):
        batch_size = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)

        if t.dim() == 0:
            t = t.expand(batch_size)
        else:
            t = t.reshape(-1)
            if t.numel() == 1:
                t = t.expand(batch_size)
            elif t.numel() != batch_size:
                raise ValueError(f"t must be scalar or have batch size {batch_size}, got shape {tuple(t.shape)}")

        cond = x[:, :self.cond_dim]
        time_emb = self.time_embed(t)
        cond_emb = self.cond_proj(cond)
        combined_cond = torch.cat([time_emb, cond_emb], dim=-1)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, combined_cond)
        out = self.output(h)
        zeros = out.new_zeros((out.size(0), self.cond_dim))
        return torch.cat([zeros, out], dim=-1)

# --- Flow Matching Path & Factories ---
class ConditionalFlowPath:
    def __init__(self, cond_dim, scheduler_type='condOT'):
        self.cond_dim = cond_dim
        log.info(f"Initializing ConditionalFlowPath: cond_dim={cond_dim}, scheduler={scheduler_type}")
        scheduler_map = {'condot': CondOTScheduler(), 'polynomial': PolynomialConvexScheduler(n=1.0),
                         'linearvp': LinearVPScheduler(), 'cosine': CosineScheduler()}
        self.scheduler = scheduler_map.get(scheduler_type.lower(), CondOTScheduler())
        self.affine_path = AffineProbPath(scheduler=self.scheduler)

    def sample(self, t, x_0, x_1):
        target_0, target_1 = x_0[:, self.cond_dim:], x_1[:, self.cond_dim:]
        path_sample = self.affine_path.sample(t=t, x_0=target_0, x_1=target_1)
        x_t = torch.cat([x_1[:, :self.cond_dim], path_sample.x_t], dim=1)
        dx_t = torch.cat([torch.zeros_like(x_1[:, :self.cond_dim]), path_sample.dx_t], dim=1)
        return SimpleNamespace(x_t=x_t, dx_t=dx_t, t=t)

def get_flow_matching_path(cond_dim, scheduler_type='condOT', **kwargs):
    return ConditionalFlowPath(cond_dim=cond_dim, scheduler_type=scheduler_type)

def get_model(input_dim, cond_dim, hidden_dim=128, depth=4, expansion_factor=2, **kwargs):
    return ConditionalFlow(
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        expansion_factor=expansion_factor,
    )

# --- Training Utilities ---
class TrainingTracker:
    def __init__(self, save_dir='.'):
        self.train_losses, self.val_losses = [], []
        self.save_dir = save_dir; self.best_val_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)
    def add_train_loss(self, loss): self.train_losses.append(loss)
    def add_val_loss(self, loss):
        self.val_losses.append(loss); is_best = loss < self.best_val_loss
        if is_best: self.best_val_loss = loss
        return is_best
    def plot_losses(self, smoothing=0.05, filename="training_loss.png"):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))
        if self.train_losses:
            batches = np.arange(len(self.train_losses))
            ema = 0.0; smooth_losses = [ema := smoothing * l + (1-smoothing) * ema for l in self.train_losses]
            plt.plot(batches, smooth_losses, linewidth=2, label='Train Loss (EMA)')
        if self.val_losses:
            val_indices = np.linspace(0, len(self.train_losses)-1, len(self.val_losses), dtype=int)
            plt.plot(val_indices, self.val_losses, 'o-', label='Validation Loss')
        plt.xlabel('Training Batches'); plt.ylabel('Loss'); plt.title('Training and Validation Losses')
        plt.legend(); plt.grid(True, which='both', linestyle='--'); plt.yscale('log')
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, filename), dpi=300); plt.close()
