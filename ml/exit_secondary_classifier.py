#!/usr/bin/env python3
"""
Joint exit-class x secondary-electron-count classifier for GMC cell models.

Extends the face-only exit classifier (ml/exit_classifier.py, unchanged) to the discrete part of a cell transit with
secondaries:

    P(primary exit class, n_sec | entry)      class in {faces..., interior outcomes},  n_sec in {0, 1, ..., N_max}

n_sec counts the secondary ELECTRONS that leave the cell (every generation: knock-ons, Auger electrons and their
descendants). Secondaries that stop inside the cell are part of the cell's dose and are not counted. The top count
class means "N_max or more" (default N_max = 10); training reports how many samples were merged into it.

Sampling the face and the count jointly keeps their correlation (e.g. backscattered primaries or large energy losses
come with more secondaries) without imposing an order. Downstream, the primary's continuous state can be conditioned
on n_sec, and each of the n_sec secondaries gets its own exit face / state from separate secondary models.

Photons are not counted yet: a combined face x electron x photon classifier follows once photons are transported in
the analog MC.

Everything that does not depend on the class layout (network, training loop, temperature scaling) is imported from
ml/exit_classifier.py. Checkpoints are written to a different file (exit_secondary_classifier.pth) and carry
kind = "exit_secondary", so the two classifiers never overwrite or get mistaken for each other.

Usage (from the GMC root):
    python ml/exit_secondary_classifier.py --data_dim 2d_electron --electron_npz train.npz [--test_npz test.npz]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.cell_topology import CellTopology, get_topology  # noqa: E402
from ml.exit_classifier import (ExitClassifier, ExitClassifierConfig, train_exit_classifier,  # noqa: E402
                                conditions_from_sim_data, _jsonable)

CHECKPOINT_KIND = "exit_secondary"
DEFAULT_MAX_SECONDARIES = 10


# --------------------------------------------------------------------------------------------------------------------
# Joint class layout
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class JointExitSpec:
    """Joint classes: index = exit_class * (max_secondaries + 1) + min(n_sec, max_secondaries)."""
    topology: CellTopology
    max_secondaries: int = DEFAULT_MAX_SECONDARIES
    secondary_species: str = "electron"

    @property
    def n_count_classes(self) -> int:
        return self.max_secondaries + 1

    @property
    def num_classes(self) -> int:
        return self.topology.num_classes * self.n_count_classes

    def count_label(self, n: int) -> str:
        return f">={n}" if n == self.max_secondaries else str(n)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(f"{c}|n={self.count_label(n)}" for c in self.topology.class_names
                     for n in range(self.n_count_classes))

    def encode(self, exit_class, n_sec):
        exit_class = np.asarray(exit_class, dtype=np.int64)
        n = np.minimum(np.asarray(n_sec, dtype=np.int64), self.max_secondaries)
        if np.any(n < 0) or np.any(exit_class < 0) or np.any(exit_class >= self.topology.num_classes):
            raise ValueError("exit_class / n_sec out of range")
        return exit_class * self.n_count_classes + n

    def decode(self, joint):
        """joint class index (numpy or torch) -> (exit_class, n_sec)."""
        return joint // self.n_count_classes, joint % self.n_count_classes

    def marginals(self, probs):
        """[N, num_classes] joint probabilities -> ([N, n_exit_classes], [N, n_count_classes])."""
        p = probs.reshape(probs.shape[0], self.topology.num_classes, self.n_count_classes)
        return p.sum(2), p.sum(1)

    def valid_mask(self, exit_class_mask):
        """Expand a per-sample exit-class mask (cell_topology.valid_class_mask) to the joint classes."""
        m = np.asarray(exit_class_mask, dtype=bool)
        return np.repeat(m, self.n_count_classes, axis=1)

    def to_dict(self):
        return {"topology": self.topology.to_dict(), "max_secondaries": self.max_secondaries,
                "secondary_species": self.secondary_species}

    @classmethod
    def from_dict(cls, d):
        return cls(CellTopology.from_dict(d["topology"]), int(d["max_secondaries"]),
                   d.get("secondary_species", "electron"))


# --------------------------------------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_exit_and_count(model: ExitClassifier, spec: JointExitSpec, cond, valid_mask=None, generator=None):
    """One (exit_class, n_sec) draw per row. A drawn n_sec == max_secondaries means 'max_secondaries or more'."""
    joint = model.sample(cond, valid_mask=valid_mask, generator=generator)
    return spec.decode(joint)


def save_exit_secondary_classifier(path, model, spec: JointExitSpec, *, data_type, norm_params, metrics=None,
                                   extra=None):
    if os.path.basename(path) == "exit_classifier.pth":
        raise ValueError("refusing to write the joint classifier over the face-only exit_classifier.pth")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "kind": CHECKPOINT_KIND,
        "model_state_dict": model.state_dict(),
        "config": vars(model.cfg).copy(),
        "spec": spec.to_dict(),
        "data_type": data_type,
        "norm_params": _jsonable(norm_params),
        "metrics": metrics or {},
        "extra": extra or {},
    }, path)


def load_exit_secondary_classifier(path, device="cpu"):
    """Returns (model, spec, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"{path} is not an exit_secondary checkpoint (kind={ckpt.get('kind')}); the face-only "
                         "classifier loads with ml.exit_classifier.load_exit_classifier")
    model = ExitClassifier(ExitClassifierConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, JointExitSpec.from_dict(ckpt["spec"]), ckpt


# --------------------------------------------------------------------------------------------------------------------
# Data checks
# --------------------------------------------------------------------------------------------------------------------
def check_bank_complete(npz_path, rtol=1e-6):
    """Secondary counts come from the escaping-electron bank, so a bank overflow in the MC would silently undercount
    them. The MC's energy balance (deposited + escaping electrons + photons = source) only closes when nothing
    overflowed; returns the relative residual and raises if it does not close."""
    with np.load(npz_path, allow_pickle=False) as d:
        if "electron_bank" not in d.files:
            raise ValueError(f"{npz_path} has no electron_bank (needed for secondary counts)")
        N = d["inputs"].shape[0]
        E_in = float(d["inputs"][:, 4].sum())
        area = float(np.prod(d["box_dims"])) / float(np.prod(d["grid_dims"]))
        deposited = float(d["energy_tally"].sum()) * N * area  # stored as MeV / cm^2 / primary per voxel
        escaped = float(d["electron_bank"][:, 5].sum()) if d["electron_bank"].size else 0.0
        photons = float(d["photon_bank"][:, 5].sum()) if "photon_bank" in d.files and d["photon_bank"].size else 0.0
    resid = (E_in - deposited - escaped - photons) / E_in
    if abs(resid) > rtol:
        raise ValueError(f"{npz_path}: energy balance residual {resid:.3e}; the MC banks probably overflowed "
                         "(raise bank_factor in analog_stepper.run_transport) and secondary counts are incomplete")
    return resid


# --------------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------------
def _two_sample_z(p, q, n_p, n_q):
    se = np.sqrt(np.maximum(p * (1 - p) / n_p + q * (1 - q) / n_q, 1e-300))
    return (p - q) / se


@torch.no_grad()
def joint_metrics(probs, exit_class, n_sec, spec: JointExitSpec, E_in=None, n_ebins=8):
    """Calibration of the joint classifier on held-out rows (probs: [N, K] numpy, labels numpy).

    Compares predicted probabilities (averaged over the rows, i.e. what sampling would produce) with MC frequencies:
        * exit-class marginal, count marginal P(n = k), tail P(n >= k) for k = 1..N_max
        * joint face x count table
        * mean secondary count per exit class and per E_in bin
    z-scores use the binomial error of the MC frequency (the predicted mean has negligible noise).
    """
    n = probs.shape[0]
    K1 = spec.n_count_classes
    nc = np.minimum(n_sec, spec.max_secondaries)
    joint_lab = spec.encode(exit_class, nc)
    nll = float(-np.log(np.clip(probs[np.arange(n), joint_lab], 1e-12, None)).mean())
    pf, pn = spec.marginals(probs)
    out = {"nll": nll, "N": int(n)}

    def zfreq(pred, emp):
        # Binomial error of the MC frequency, with p floored at 0.5 / n: for a predicted p ~ 0 the error would
        # otherwise collapse and a single MC event would read as a huge deviation.
        p = np.clip(pred, 0.5 / n, 1 - 0.5 / n)
        return (emp - pred) / np.sqrt(p * (1 - p) / n)

    mc_n = np.bincount(nc, minlength=K1) / n
    pr_n = pn.mean(0)
    out["count_pmf"] = {spec.count_label(k): {"mc": float(mc_n[k]), "pred": float(pr_n[k]),
                                              "z": float(zfreq(pr_n[k], mc_n[k]))} for k in range(K1)}
    out["count_tail"] = {}
    for k in range(1, K1):
        mc_t, pr_t = float(mc_n[k:].sum()), float(pr_n[k:].sum())
        out["count_tail"][f">={k}"] = {"mc": mc_t, "pred": pr_t, "z": float(zfreq(pr_t, mc_t)),
                                       "mc_events": int(round(mc_t * n))}
    mc_f = np.bincount(exit_class, minlength=spec.topology.num_classes) / n
    pr_f = pf.mean(0)
    out["exit_class"] = {c: {"mc": float(mc_f[i]), "pred": float(pr_f[i]), "z": float(zfreq(pr_f[i], mc_f[i]))}
                         for i, c in enumerate(spec.topology.class_names)}
    # Joint table, mean count per exit class
    mc_j = np.bincount(joint_lab, minlength=spec.num_classes).reshape(-1, K1) / n
    pr_j = probs.mean(0).reshape(-1, K1)
    zj = zfreq(pr_j, mc_j)
    seen = (mc_j > 0) | (pr_j > 5.0 / n)
    out["joint_max_abs_z"] = float(np.abs(zj[seen]).max()) if seen.any() else 0.0
    ks = np.arange(K1)
    out["mean_count_by_exit_class"] = {}
    for i, c in enumerate(spec.topology.class_names):
        m = exit_class == i
        if m.sum() == 0:
            continue
        # E[n | class i] under the model's joint distribution averaged over ALL rows (not just the rows where the
        # MC landed in class i, which would condition on the outcome)
        p_i = probs.reshape(n, spec.topology.num_classes, K1)[:, i, :]
        pred_mean = float((p_i.sum(0) * ks).sum() / max(p_i.sum(), 1e-300))
        out["mean_count_by_exit_class"][c] = {"mc": float(nc[m].mean()), "pred_given_class": pred_mean,
                                              "rows": int(m.sum())}
    if E_in is not None:
        q = np.quantile(E_in, np.linspace(0, 1, n_ebins + 1))
        b = np.clip(np.searchsorted(q, E_in, side="right") - 1, 0, n_ebins - 1)
        rows = []
        for i in range(n_ebins):
            m = b == i
            mc_mean, pr_mean = float(nc[m].mean()), float((pn[m] * ks).sum(1).mean())
            se = float(nc[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else float("nan")
            rows.append({"E_lo": float(q[i]), "E_hi": float(q[i + 1]), "mc_mean": mc_mean, "pred_mean": pr_mean,
                         "z": (mc_mean - pr_mean) / se if se > 0 else 0.0})
        out["mean_count_by_E"] = rows
    return out


def plot_joint_checks(metrics, spec: JointExitSpec, path, title=""):
    import matplotlib.pyplot as plt

    K1 = spec.n_count_classes
    labels = [spec.count_label(k) for k in range(K1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    mc = np.array([metrics["count_pmf"][l]["mc"] for l in labels])
    pr = np.array([metrics["count_pmf"][l]["pred"] for l in labels])
    N = metrics["N"]
    x = np.arange(K1)
    ax.bar(x - 0.2, np.maximum(mc, 1e-9), 0.4, label="MC", color="k", alpha=0.7,
           yerr=np.sqrt(np.maximum(pr * (1 - pr), 0) / N))
    ax.bar(x + 0.2, np.maximum(pr, 1e-9), 0.4, label="classifier", color="C1")
    ax.set_yscale("log"), ax.set_xticks(x, labels), ax.set_xlabel("escaping secondary electrons")
    ax.set_ylabel("P(n)"), ax.legend(), ax.set_ylim(0.3 / N, 2)
    ax = axes[1]
    names = list(metrics["mean_count_by_exit_class"])
    mc_m = [metrics["mean_count_by_exit_class"][c]["mc"] for c in names]
    pr_m = [metrics["mean_count_by_exit_class"][c]["pred_given_class"] for c in names]
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, mc_m, 0.4, label="MC", color="k", alpha=0.7)
    ax.bar(xx + 0.2, pr_m, 0.4, label="classifier", color="C1")
    ax.set_xticks(xx, names), ax.set_ylabel("mean n | primary exit class"), ax.legend()
    ax = axes[2]
    if "mean_count_by_E" in metrics:
        r = metrics["mean_count_by_E"]
        c = [np.sqrt(e["E_lo"] * e["E_hi"]) for e in r]
        ax.plot(c, [e["mc_mean"] for e in r], "ko", label="MC")
        ax.plot(c, [e["pred_mean"] for e in r], "-", color="C1", label="classifier")
        ax.set_xscale("log"), ax.set_xlabel("E_in [MeV]"), ax.set_ylabel("mean n"), ax.legend()
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def _load(npz, norm_params, args):
    from physics.electron_data import load_electron_dataset

    resid = check_bank_complete(npz)
    df = load_electron_dataset(npz)
    if "n_sec_e" not in df:
        raise ValueError(f"{npz}: no secondary counts")
    print(f"  {npz}: energy-balance residual {resid:.1e} (banks complete)")
    return df


def main():
    from ml.transform import build_norm_params_2d_electron, load_normalization_params

    p = argparse.ArgumentParser(description="Train the joint exit-class x secondary-electron-count classifier")
    p.add_argument("--data_dim", default="2d_electron")
    p.add_argument("--electron_npz", required=True)
    p.add_argument("--test_npz", default=None, help="Independent MC file for the final evaluation")
    p.add_argument("--max_secondaries", type=int, default=DEFAULT_MAX_SECONDARIES,
                   help="Largest count class; it collects every count >= this value")
    p.add_argument("--norm_params", default=None, help="Share an existing flow's norm params (.npy base)")
    p.add_argument("--out", default=None,
                   help="Checkpoint (default ml/models/<data_type>/exit_secondary_classifier.pth)")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.data_dim != "2d_electron":
        raise ValueError("only 2d_electron is wired so far")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_type = "2d_electron_boundary"
    spec = JointExitSpec(get_topology("square", interior=("absorbed",)), args.max_secondaries)

    df = _load(args.electron_npz, None, args)
    if args.norm_params:
        norm_params, _ = load_normalization_params(args.norm_params)
    else:
        norm_params = build_norm_params_2d_electron(df)
    n_raw = df["n_sec_e"].to_numpy()
    merged = int((n_raw > spec.max_secondaries).sum())
    print(f"  secondary counts: max {n_raw.max()}, {merged} rows ({100 * merged / len(n_raw):.4f}%) above "
          f"{spec.max_secondaries} merged into the '>={spec.max_secondaries}' class")
    labels = torch.as_tensor(spec.encode(df["exit_class"].to_numpy(), n_raw), dtype=torch.long)
    cond, cond_dim = conditions_from_sim_data(df, norm_params, data_type)
    print(f"Joint classifier: {spec.topology.num_classes} exit classes x {spec.n_count_classes} count classes = "
          f"{spec.num_classes} classes, cond_dim={cond_dim}, N={len(df)}, device={device}")

    model, history, (_, val_idx) = train_exit_classifier(
        cond, labels, spec, val_ratio=args.val_ratio, hidden_dim=args.hidden_dim, depth=args.depth,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, prior_bias=True)

    out_path = args.out or f"ml/models/{data_type}/exit_secondary_classifier.pth"
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    evals = {}
    vi = val_idx.numpy()
    sets = [("val", df.iloc[vi].reset_index(drop=True), cond[val_idx])]
    if args.test_npz:
        tdf = _load(args.test_npz, None, args)
        tcond, _ = conditions_from_sim_data(tdf, norm_params, data_type)
        sets.append(("test", tdf, tcond))
    for name, d, c in sets:
        probs = model.predict_proba(c.to(device)).cpu().double().numpy()
        m = joint_metrics(probs, d["exit_class"].to_numpy(), d["n_sec_e"].to_numpy(), spec, d["E_in"].to_numpy())
        evals[name] = m
        plot_joint_checks(m, spec, os.path.join(out_dir, f"exit_secondary_check_{name}.png"),
                          title=f"{name}: {os.path.basename(args.test_npz if name == 'test' else args.electron_npz)}")
    lab = torch.bincount(labels, minlength=spec.num_classes).double()
    prior = lab[lab > 0] / lab.sum()
    metrics = {"prior_entropy": float(-(prior * prior.log()).sum()), "temperature": float(model.log_temperature.exp()),
               "merged_above_max": merged, **evals}
    save_exit_secondary_classifier(out_path, model, spec, data_type=data_type, norm_params=norm_params,
                                   metrics=metrics, extra={"history": history, "source": args.electron_npz,
                                                           "test": args.test_npz})
    print(f"\nSaved {out_path}")
    for name, m in evals.items():
        print(f"\n[{name}] N={m['N']}  NLL {m['nll']:.4f} (prior entropy {metrics['prior_entropy']:.4f})  "
              f"joint table max|z| {m['joint_max_abs_z']:.2f}")
        print("  P(n >= k)   " + "  ".join(f"{k}: mc {v['mc']:.2e} pred {v['pred']:.2e} (z {v['z']:+.1f}, "
                                          f"{v['mc_events']} ev)" for k, v in m["count_tail"].items()
                                          if v["mc"] > 0 or v["pred"] > 1e-6))
        print("  exit class  " + "  ".join(f"{c}: z {v['z']:+.1f}" for c, v in m["exit_class"].items()))
        print("  mean n | class  " + "  ".join(f"{c}: mc {v['mc']:.4f} pred {v['pred_given_class']:.4f}"
                                              for c, v in m["mean_count_by_exit_class"].items()))
        if "mean_count_by_E" in m:
            print("  mean n vs E_in, max|z| " + f"{max(abs(r['z']) for r in m['mean_count_by_E']):.2f}")


if __name__ == "__main__":
    main()
