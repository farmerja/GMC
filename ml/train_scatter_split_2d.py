#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import TrainingTracker, get_flow_matching_path, get_model
from ml.train import validate
from ml.transform import (
    SimDataset,
    compute_normalization_params_2d,
    compute_normalization_params_2d_internal,
    robust_log_stats,
)
from ml.utils import (
    analytic_log_bounds as _analytic_log_bounds,
    available_cpu_count as _available_cpu_count,
    save_norm_params_artifacts as _save_norm_params_artifacts,
)
from physics.scatter_split_2d import (
    get_2d_scatter_split_boundary_data,
    get_2d_scatter_split_internal_data,
)


def _prepare_scatter_split_data(
    *,
    device: str,
    source_type: str,
    batch_size: int,
    val_ratio: float,
    min_width: float,
    max_width: float,
    min_height: float,
    max_height: float,
    num_neutrons: int,
    sampling: str,
    seed: int,
    train_workers: int | None,
    val_workers: int | None,
) -> tuple[DataLoader, DataLoader, dict[str, float], int, int]:
    os.makedirs("ml/data", exist_ok=True)
    generator_fn = (
        get_2d_scatter_split_boundary_data
        if source_type == "boundary"
        else get_2d_scatter_split_internal_data
    )
    sim_data = generator_fn(
        num_neutrons=num_neutrons,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        sampling=sampling,
        seed=seed,
    )
    if len(sim_data) == 0:
        raise ValueError("Scattered-only generator returned zero rows.")

    data_type_str = f"2d_{source_type}"
    if source_type == "boundary":
        time_lower, time_upper, _, _, _, _ = compute_normalization_params_2d(sim_data)
    else:
        time_lower, time_upper, _, _, _, _ = compute_normalization_params_2d_internal(sim_data)
    width_lower, width_upper = _analytic_log_bounds(min_width, max_width, "width")
    height_lower, height_upper = _analytic_log_bounds(min_height, max_height, "height")
    time_center, time_scale = robust_log_stats(sim_data["time_in_cell"].values)
    norm_params = {
        "time_lower": time_lower,
        "time_upper": time_upper,
        "time_center": time_center,
        "time_scale": time_scale,
        "width_lower": width_lower,
        "width_upper": width_upper,
        "height_lower": height_lower,
        "height_upper": height_upper,
        "direction_encoding_2d": "polar",
        "training_condition": "scatter_before_exit",
    }
    norm_params_path = f"ml/data/norm_params_2d_{source_type}_scatter_split.npy"
    _save_norm_params_artifacts(norm_params_path, norm_params, f"2D {source_type} scatter-split")

    full_dataset = SimDataset(sim_data, norm_params, data_type=data_type_str)
    input_dim = full_dataset.input_dim
    cond_dim = full_dataset.cond_dim
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"Dataset split resulted in zero samples (train={train_size}, val={val_size})."
        )
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    device_str = str(device)
    use_pin_memory = device_str.startswith("cuda")
    available_cpus = _available_cpu_count()
    if train_workers is None:
        train_workers = 0 if available_cpus <= 1 else min(4, available_cpus - 1)
    if val_workers is None:
        val_workers = 0 if available_cpus <= 1 else min(2, train_workers)

    drop_last_train = len(train_dataset) >= batch_size
    train_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=train_workers,
        pin_memory=use_pin_memory,
        persistent_workers=train_workers > 0,
    )
    if train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
    val_loader_kwargs = dict(
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
        num_workers=val_workers,
        pin_memory=use_pin_memory,
        persistent_workers=val_workers > 0,
    )
    if val_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    return train_loader, val_loader, norm_params, input_dim, cond_dim


def _resolve_device(device: str, *, seed: int) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        resolved = "cuda"
    elif requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS requested but not available.")
        resolved = "mps"
    elif requested == "cpu":
        resolved = "cpu"
    else:
        raise ValueError(f"Unsupported device option: {device!r}")

    if resolved == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(True)
    print(f"Using device: {resolved}")
    return resolved


def train_scatter_split_2d(
    *,
    source_type: str,
    num_epochs: int = 50,
    batch_size: int = 2048,
    hidden_dim: int = 128,
    depth: int = 4,
    expansion_factor: int = 2,
    lr: float = 4.0e-4,
    weight_decay: float = 1.0e-5,
    min_width: float = 0.01,
    max_width: float = 10.0,
    min_height: float = 0.01,
    max_height: float = 10.0,
    val_ratio: float = 0.1,
    scheduler_type: str = "condOT",
    gradient_accumulation_steps: int = 1,
    num_neutrons: int = 100000,
    sampling: str = "random",
    seed: int = 42,
    device: str = "auto",
    plot_losses: bool = True,
    train_workers: int | None = None,
    val_workers: int | None = None,
) -> None:
    device = _resolve_device(device, seed=seed)
    non_blocking = str(device).startswith("cuda")
    train_loader, val_loader, _norm_params, input_dim, cond_dim = _prepare_scatter_split_data(
        device=device,
        source_type=source_type,
        batch_size=batch_size,
        val_ratio=val_ratio,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        num_neutrons=num_neutrons,
        sampling=sampling,
        seed=seed,
        train_workers=train_workers,
        val_workers=val_workers,
    )

    model_dir = f"ml/models/2d_{source_type}_scatter_split"
    training_dir = f"ml/training_results/2d_{source_type}_scatter_split"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)

    model = get_model(
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        expansion_factor=expansion_factor,
    ).to(device)
    compiled_model = model

    optimizer_kwargs = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)
    if str(device).startswith("cuda"):
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    total_batches_per_epoch = len(train_loader)
    total_optimization_steps = (
        num_epochs * total_batches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_optimization_steps = max(1, total_optimization_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_optimization_steps,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy="cos",
    )

    path = get_flow_matching_path(cond_dim=cond_dim, scheduler_type=scheduler_type)
    tracker = TrainingTracker(save_dir=training_dir)
    start_time = time.time()
    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        compiled_model.train()
        epoch_loss = 0.0
        num_batches_processed = 0
        for batch_idx, (x1, _, _) in enumerate(train_loader):
            x1 = x1.to(device, non_blocking=non_blocking)
            cond = x1[:, :cond_dim]
            target = x1[:, cond_dim:]
            noise = torch.randn_like(target)
            x0 = torch.cat([cond, noise], dim=1)
            t_sample = torch.rand(x1.shape[0], device=device)
            path_sample = path.sample(t=t_sample, x_0=x0, x_1=x1)
            pred_velocity = compiled_model(path_sample.x_t, path_sample.t)
            loss = torch.pow(pred_velocity - path_sample.dx_t, 2).mean()
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            loss_value = loss.item() * gradient_accumulation_steps
            if not np.isnan(loss_value):
                epoch_loss += loss_value
                num_batches_processed += 1
                tracker.add_train_loss(loss_value)

        avg_epoch_loss = epoch_loss / max(1, num_batches_processed)
        val_loss = validate(compiled_model, val_loader, path, device, epoch)
        epoch_train_losses.append(float(avg_epoch_loss))
        epoch_val_losses.append(float(val_loss))
        is_best = tracker.add_val_loss(val_loss)
        if plot_losses:
            tracker.plot_losses(filename="training_loss.png")
        if is_best:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_epoch_loss:.6f} "
            f"- Val Loss: {val_loss:.6f} - LR: {current_lr:.2e}"
        )

    torch.save(model.state_dict(), os.path.join(model_dir, "trained_model_final.pth"))
    history_csv_path = os.path.join(training_dir, "loss_history.csv")
    with open(history_csv_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss\n")
        for index, (train_loss, val_loss) in enumerate(zip(epoch_train_losses, epoch_val_losses), start=1):
            handle.write(f"{index},{train_loss:.10e},{val_loss:.10e}\n")
    np.save(os.path.join(training_dir, "train_loss_batch_history.npy"), np.asarray(tracker.train_losses, dtype=np.float64))
    np.save(os.path.join(training_dir, "val_loss_epoch_history.npy"), np.asarray(epoch_val_losses, dtype=np.float64))
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f}s. Best val loss: {tracker.best_val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D steady scatter-split GMC models.")
    parser.add_argument("--source_type", choices=["boundary", "internal"], default="boundary")
    parser.add_argument("--num_neutrons", type=int, default=100000)
    parser.add_argument("--min_width", type=float, default=0.01)
    parser.add_argument("--max_width", type=float, default=10.0)
    parser.add_argument("--min_height", type=float, default=0.01)
    parser.add_argument("--max_height", type=float, default=10.0)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--expansion_factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--scheduler_type", type=str, default="condOT")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sampling", choices=["random", "qmc", "lhs"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable_plots", action="store_true")
    parser.add_argument("--train_workers", type=int, default=None)
    parser.add_argument("--val_workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_scatter_split_2d(
        source_type=args.source_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        expansion_factor=args.expansion_factor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        val_ratio=args.val_ratio,
        scheduler_type=args.scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_neutrons=args.num_neutrons,
        sampling=args.sampling,
        seed=args.seed,
        device=args.device,
        plot_losses=not args.disable_plots,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
    )


if __name__ == "__main__":
    main()
