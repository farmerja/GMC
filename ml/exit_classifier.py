#!/usr/bin/env python3
"""
Exit-class classifier for GMC cell models.

The flow-matching cell model predicts the continuous part of a particle's exit state. Which face the particle leaves
through (or whether it stops inside the cell) is discrete, and a continuous flow represents it poorly: the exit
positions live on the cell boundary (a set of lower dimension, made of disconnected faces), so the flow leaks mass
into the interior, blurs corners and mislabels faces when the nearest face is guessed afterwards. This module learns
the discrete part directly:

    p(exit | entry) = P(class | entry) * p(continuous exit state | class, entry)
                      ^^^^^^^^^^^^^^^^^ this module

Design choices that keep it usable across the planned extensions:
    * Classes follow ml/cell_topology.CellTopology: faces first (2D polygons: CCW edges, 3D: face table), then
      interior outcomes ("absorbed", "census", ...). Square, quad, triangle, box and tetrahedron are built in.
    * The classifier's input is exactly the conditioning vector of the flow (``SimDataset(...).cond``), so the same
      normalized tensor feeds both models at inference, and any data_type (2D/3D, time-dependent with a time
      budget, quads/triangles with shape parameters, electrons or photons) works without changes here.
    * An optional per-sample valid-class mask lets one classifier cover cells with different face counts (e.g.
      triangles and quads in one mesh with 4 face slots).
    * Trained with plain cross-entropy (no class reweighting), so the softmax is the Monte Carlo branching
      probability, then temperature-scaled on held-out data. Inference SAMPLES from it; never take the argmax.
    * One model per (particle species, material, cell topology), like the flows.

Usage (from the GMC root):
    python ml/exit_classifier.py --data_dim 2d_electron --electron_npz path/to/data.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.cell_topology import CellTopology, get_topology  # noqa: E402


# --------------------------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * expansion), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(dim * expansion, dim))

    def forward(self, x):
        return x + self.net(x)


@dataclass
class ExitClassifierConfig:
    cond_dim: int
    num_classes: int
    hidden_dim: int = 256
    depth: int = 4
    expansion: int = 2
    dropout: float = 0.0


class ExitClassifier(nn.Module):
    """MLP: flow conditioning vector -> logits over exit classes (faces + interior outcomes)."""

    def __init__(self, cfg: ExitClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.cond_dim, cfg.hidden_dim)
        self.blocks = nn.Sequential(*[_ResBlock(cfg.hidden_dim, cfg.expansion, cfg.dropout) for _ in range(cfg.depth)])
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.num_classes))
        # Post-hoc temperature (calibration); 1.0 until fit_temperature() runs.
        self.register_buffer("log_temperature", torch.zeros(()))

    def raw_logits(self, cond):
        return self.out(self.blocks(self.inp(cond)))

    def forward(self, cond, valid_mask=None):
        """Calibrated logits; classes where ``valid_mask`` is False get -inf."""
        logits = self.raw_logits(cond) / self.log_temperature.exp()
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.to(torch.bool), float("-inf"))
        return logits

    @torch.no_grad()
    def predict_proba(self, cond, valid_mask=None, batch_size=65536):
        outs = []
        for i in range(0, cond.shape[0], batch_size):
            m = None if valid_mask is None else valid_mask[i:i + batch_size]
            outs.append(torch.softmax(self(cond[i:i + batch_size], m), dim=-1))
        return torch.cat(outs) if outs else cond.new_zeros((0, self.cfg.num_classes))

    @torch.no_grad()
    def sample(self, cond, valid_mask=None, generator=None):
        """Draw one exit class per row from the predicted branching probabilities."""
        p = self.predict_proba(cond, valid_mask)
        return torch.multinomial(p, 1, generator=generator).squeeze(1)


# --------------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------------
def conditions_from_sim_data(sim_data, norm_params, data_type):
    """Normalized flow-conditioning tensor [N, cond_dim] for any SimDataset data_type."""
    from ml.transform import SimDataset

    ds = SimDataset(sim_data, norm_params, data_type=data_type)
    return ds._cached_x1[:, :ds.cond_dim].clone(), ds.cond_dim


def split_indices(n, val_ratio=0.1, seed=42):
    """Same split as torch.utils.data.random_split(generator=manual_seed(seed)) in ml/train.py, so the classifier's
    validation rows are also the flow's validation rows."""
    val_size = int(n * val_ratio)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[: n - val_size], perm[n - val_size:]


# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def classification_metrics(probs, labels, n_bins=15):
    """NLL, accuracy, top-label expected calibration error and predicted-vs-empirical class fractions."""
    probs = probs.double().cpu()
    labels = labels.cpu()
    n, k = probs.shape
    nll = -torch.log(probs[torch.arange(n), labels].clamp_min(1e-12)).mean().item()
    conf, pred = probs.max(1)
    acc = (pred == labels).double()
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.double().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    emp = torch.bincount(labels, minlength=k).double() / n
    return {"nll": nll, "accuracy": acc.mean().item(), "ece": ece,
            "pred_fraction": probs.mean(0).tolist(), "mc_fraction": emp.tolist()}


@torch.no_grad()
def binned_fraction_check(probs, labels, feature, n_bins=10):
    """Predicted vs MC class fractions in quantile bins of one conditioning feature.

    Returns per-bin predicted fractions, MC fractions, binomial standard errors and z-scores. A calibrated
    classifier has |z| ~ N(0, 1): this is the check that the branching ratios are right as a function of energy,
    angle, entry position, cell shape, ...; aggregate accuracy says little about that.
    """
    probs, labels, feature = probs.double().cpu(), labels.cpu(), feature.double().cpu()
    k = probs.shape[1]
    q = torch.quantile(feature, torch.linspace(0, 1, n_bins + 1, dtype=torch.float64))
    idx = torch.bucketize(feature, q[1:-1])
    centers, pred, mc, se = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        p_mc = torch.bincount(labels[m], minlength=k).double() / nb
        p_pr = probs[m].mean(0)
        centers.append(feature[m].mean().item())
        pred.append(p_pr)
        mc.append(p_mc)
        # Floor p at 0.5/nb: near p = 0 the binomial s.e. collapses and a negligible deviation would read as many sigma
        p_se = p_pr.clamp(0.5 / nb, 1 - 0.5 / nb)
        se.append(torch.sqrt(p_se * (1 - p_se) / nb))
    pred, mc, se = torch.stack(pred), torch.stack(mc), torch.stack(se)
    z = (mc - pred) / se
    return {"centers": np.array(centers), "pred": pred.numpy(), "mc": mc.numpy(), "se": se.numpy(),
            "z": z.numpy(), "max_abs_z": float(z.abs().max())}


# --------------------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------------------
def fit_temperature(model, cond, labels, valid_mask=None, max_iter=100):
    """Fit a scalar temperature on held-out data by minimizing NLL (Guo et al. 2017). Leaves the argmax unchanged."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model.raw_logits(cond[i:i + 65536]) for i in range(0, cond.shape[0], 65536)])
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
    log_t = torch.zeros((), device=cond.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        model.log_temperature.copy_(log_t.detach())
    return float(log_t.detach().exp())


def train_exit_classifier(cond, labels, topology: CellTopology, *, valid_mask=None, val_ratio=0.1, split_seed=42,
                          hidden_dim=256, depth=4, expansion=2, dropout=0.0, epochs=60, batch_size=4096, lr=2e-3,
                          weight_decay=1e-4, patience=None, device="cpu", log_every=1, prior_bias=False):
    """Train on (cond, labels); returns (model, history, (train_idx, val_idx)).

    cond: [N, cond_dim] float tensor (normalized flow conditioning). labels: [N] long tensor of class indices.
    valid_mask: optional [N, num_classes] bool tensor (mixed cell types sharing one model).
    prior_bias: start the output bias at log(training class frequency), with a floor of 0.1 / N_train for classes that
        never occur. Without it, classes with no examples keep a small but non-negligible probability, which
        matters for many-class layouts (e.g. the joint face x count classifier). Off by default (old behaviour).
    """
    n = cond.shape[0]
    if labels.min() < 0 or labels.max() >= topology.num_classes:
        raise ValueError(f"labels must be in [0, {topology.num_classes}); got [{labels.min()}, {labels.max()}]")
    train_idx, val_idx = split_indices(n, val_ratio, split_seed)
    cond, labels = cond.to(device).float(), labels.to(device).long()
    valid_mask = None if valid_mask is None else valid_mask.to(device).bool()
    tr_idx, va_idx = train_idx.to(device), val_idx.to(device)

    cfg = ExitClassifierConfig(cond.shape[1], topology.num_classes, hidden_dim, depth, expansion, dropout)
    model = ExitClassifier(cfg).to(device)
    if prior_bias:
        freq = torch.bincount(labels[tr_idx], minlength=cfg.num_classes).double() / tr_idx.numel()
        with torch.no_grad():
            model.out[-1].bias.copy_(torch.log(freq.clamp_min(0.1 / tr_idx.numel())).float())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(tr_idx.numel() / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)

    def mask_of(idx):
        return None if valid_mask is None else valid_mask[idx]

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot = 0.0
        for i in range(0, va_idx.numel(), 65536):
            b = va_idx[i:i + 65536]
            tot += F.cross_entropy(model(cond[b], mask_of(b)), labels[b], reduction="sum").item()
        model.train()
        return tot / max(1, va_idx.numel())

    history = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        perm = tr_idx[torch.randperm(tr_idx.numel(), device=device)]
        tot = 0.0
        for i in range(0, perm.numel(), batch_size):
            b = perm[i:i + batch_size]
            loss = F.cross_entropy(model(cond[b], mask_of(b)), labels[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * b.numel()
        history["train"].append(tot / tr_idx.numel())
        history["val"].append(val_loss())
        if history["val"][-1] < best - 1e-6:
            best, bad = history["val"][-1], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}/{epochs}: train NLL {history['train'][-1]:.5f} | "
                  f"val NLL {history['val'][-1]:.5f} | {time.time() - t0:.0f}s")
        if patience is not None and bad >= patience:
            print(f"  early stop at epoch {ep + 1} (best val NLL {best:.5f})")
            break
    model.load_state_dict(best_state)
    T = fit_temperature(model, cond[va_idx], labels[va_idx], mask_of(va_idx))
    print(f"  fitted temperature T = {T:.4f}")
    model.eval()
    return model, history, (train_idx, val_idx)


# --------------------------------------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------------------------------------
def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def save_exit_classifier(path, model: ExitClassifier, topology: CellTopology, *, data_type, norm_params,
                         metrics=None, extra=None):
    """Checkpoint with everything inference needs: weights, config, class topology, flow data_type and the norm
    params the conditioning was built with (so a mismatch with the flow's norm params can be detected)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(model.cfg),
        "topology": topology.to_dict(),
        "data_type": data_type,
        "norm_params": _jsonable(norm_params),
        "metrics": metrics or {},
        "extra": extra or {},
    }, path)


def load_exit_classifier(path, device="cpu"):
    """Returns (model, topology, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ExitClassifier(ExitClassifierConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, CellTopology.from_dict(ckpt["topology"]), ckpt


def check_norm_compatibility(ckpt, norm_params, keys: Sequence[str], rtol=1e-6):
    """Raise if the classifier was trained with different normalization of the shared conditioning features."""
    bad = []
    for k in keys:
        a, b = ckpt["norm_params"].get(k), norm_params.get(k)
        if a is None or b is None or not np.isclose(float(a), float(b), rtol=rtol, atol=1e-12):
            bad.append(f"{k}: classifier={a} flow={b}")
    if bad:
        raise ValueError("Exit classifier / flow normalization mismatch:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------------------------------------------------
# Coupling with a flow that was NOT trained face-conditioned (interim use)
# --------------------------------------------------------------------------------------------------------------------
def rejection_match(target_class: np.ndarray, sample_fn: Callable[[np.ndarray], dict],
                    implied_class_fn: Callable[[dict], np.ndarray], max_tries: int = 16):
    """Make an unconditioned flow agree with the classifier by rejection.

    For every row, redraw the flow (``sample_fn(rows) -> dict of arrays``) until the class implied by its sample
    (``implied_class_fn``: nearest face, or interior if stopped) equals ``target_class``. Accepted rows are then
    distributed as p_flow(state | class, entry) with class ~ classifier, i.e. the classifier fixes the face/absorption
    branching while the flow supplies the within-class state. Rows still unmatched after ``max_tries`` keep their
    last flow sample; their fraction is returned and measures how badly the flow mis-allocates mass across classes.
    A face-conditioned flow removes the need for this.

    Returns (samples dict, unmatched bool mask).
    """
    n = target_class.shape[0]
    out = sample_fn(np.arange(n))
    pending = implied_class_fn(out) != target_class
    for _ in range(max_tries):
        rows = np.flatnonzero(pending)
        if rows.size == 0:
            break
        new = sample_fn(rows)
        ok = implied_class_fn(new) == target_class[rows]
        acc = rows[ok]
        for k in out:
            out[k][acc] = new[k][ok]
        pending[acc] = False
    return out, pending


# --------------------------------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------------------------------
def plot_binned_check(check, class_names, xlabel, path, log_x=False):
    import matplotlib.pyplot as plt

    k = len(class_names)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.2), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        x = check["centers"]
        ax.errorbar(x, check["mc"][:, j], yerr=check["se"][:, j], fmt="o", ms=3, label="MC", color="k")
        ax.plot(x, check["pred"][:, j], "-", label="classifier", color="C1")
        ax.set_title(f"{class_names[j]}  (max|z|={np.abs(check['z'][:, j]).max():.1f})", fontsize=9)
        ax.set_xlabel(xlabel)
        if log_x:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("fraction")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def _load_problem(args):
    """(sim_data, data_type, topology, labels, norm_params, diagnostic feature columns) for a CLI data_dim.

    Add new data_dims here: anything with a SimDataset data_type and an integer exit-class column works.
    """
    from ml.transform import build_norm_params_2d_electron, load_normalization_params

    if args.data_dim == "2d_electron":
        from physics.electron_data import load_electron_dataset

        sim_data = load_electron_dataset(args.electron_npz)
        data_type = "2d_electron_boundary"
        topology = get_topology("square", interior=("absorbed",))
        if args.norm_params:
            norm_params, _ = load_normalization_params(args.norm_params)
        else:
            norm_params = build_norm_params_2d_electron(sim_data)
        labels = sim_data["exit_class"].to_numpy()
        features = {"E_in [MeV]": (sim_data["E_in"].to_numpy(), True),
                    "y_in / H": (sim_data["y_in_norm"].to_numpy(), False),
                    "u_in": (sim_data["u_in"].to_numpy(), False)}
        return sim_data, data_type, topology, labels, norm_params, features
    raise ValueError(f"data_dim '{args.data_dim}' is not wired into the exit-classifier CLI yet")


def main():
    p = argparse.ArgumentParser(description="Train a GMC exit-class (face / absorbed / census) classifier")
    p.add_argument("--data_dim", default="2d_electron")
    p.add_argument("--electron_npz", type=str, default=None)
    p.add_argument("--norm_params", type=str, default=None,
                   help="Existing flow norm params (.npy base path) to share; default: compute from the data")
    p.add_argument("--out", type=str, default=None, help="Checkpoint path (default ml/models/<type>/exit_classifier.pth)")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=None,
                   help="Early-stop patience in epochs (default: off; the one-cycle LR needs the full run to anneal)")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sim_data, data_type, topology, labels, norm_params, features = _load_problem(args)
    cond, cond_dim = conditions_from_sim_data(sim_data, norm_params, data_type)
    labels_t = torch.as_tensor(np.array(labels), dtype=torch.long)
    print(f"Exit classifier: data_type={data_type}, topology={topology.name}, classes={topology.class_names}, "
          f"cond_dim={cond_dim}, N={len(labels)}, device={device}")

    model, history, (_, val_idx) = train_exit_classifier(
        cond, labels_t, topology, val_ratio=args.val_ratio, hidden_dim=args.hidden_dim, depth=args.depth,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, device=device)

    probs = model.predict_proba(cond[val_idx].to(device)).cpu()
    y_val = labels_t[val_idx]
    metrics = classification_metrics(probs, y_val)
    # Entropy of the MC branching itself: the NLL of a perfect model is the conditional entropy, which is NOT 0
    # (exit faces are random), so compare val NLL against the class-prior entropy (an input-blind model) instead.
    prior = torch.bincount(labels_t, minlength=topology.num_classes).double() / len(labels_t)
    metrics["prior_entropy"] = float(-(prior[prior > 0] * prior[prior > 0].log()).sum())

    out_path = args.out or f"ml/models/{data_type}/exit_classifier.pth"
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    checks = {}
    for name, (values, log_x) in features.items():
        chk = binned_fraction_check(probs, y_val, torch.as_tensor(values)[val_idx])
        checks[name] = chk["max_abs_z"]
        slug = name.split()[0].replace("/", "").replace(" ", "")
        plot_binned_check(chk, topology.class_names, name, os.path.join(out_dir, f"exit_classifier_check_{slug}.png"),
                          log_x=log_x)
    metrics["binned_max_abs_z"] = checks
    metrics["temperature"] = float(model.log_temperature.exp())
    save_exit_classifier(out_path, model, topology, data_type=data_type, norm_params=norm_params, metrics=metrics,
                         extra={"history": history, "source": args.electron_npz})

    print(f"\nSaved {out_path}")
    print(json.dumps({k: v for k, v in metrics.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
