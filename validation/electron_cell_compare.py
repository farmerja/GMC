#!/usr/bin/env python3
"""
Single-cell accuracy of 2D electron GMC models against held-out Monte Carlo.

Every predictor gets the entry states of a held-out MC file, draws one exit per particle, and is scored against the
MC exits of those same particles:
    * exit-class fractions (4 faces + absorbed), overall and in E_in bins,
    * per-face distributions of s (position along the face), psi (exit angle from the outward normal),
      E_out / E_in and log10(path length) as Wasserstein-1 distances,
    * deposited energy per primary (E_in - E_out for exits, E_in for absorbed), overall and in E_in bins,
    * baseline only: how many "exits" point back into the cell or sit inside it (the flow's 2D position is not
      constrained to the boundary).

Predictors (any subset):
    --baseline_ckpt/--baseline_norm             unconditioned 2d_electron flow; class = nearest face, or absorbed if
                                                its E_out is at the cutoff
    --face_ckpt/--face_norm/--classifier        exit classifier + face-conditioned 2d_electron_face flow
    --ref_npz                                   a second, independent MC file: the statistical noise floor

Example:
    python validation/electron_cell_compare.py --test_npz ../ElectronTransport/analog_tp_cu_1um_50k_7.npz \
        --baseline_ckpt ml/models/2d_electron_boundary_analog_cu/best_model.pth \
        --baseline_norm ml/data/norm_params_2d_electron_boundary_analog_cu.npy \
        --face_ckpt ml/models/2d_electron_face_boundary_analog_cu/best_model.pth \
        --face_norm ml/data/norm_params_2d_electron_face_boundary_analog_cu.npy \
        --classifier ml/models/2d_electron_analog_cu/exit_classifier.pth \
        --ref_npz ../ElectronTransport/analog_tp_cu_1um_20k_42.npz --out_dir validation/results/electron_cell_analog_cu
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.cell_topology import (SQUARE, UNIT_SQUARE_VERTS, polygon_nearest_face, polygon_signed_distances,
                              polygon_face_coordinate, direction_to_face_frame_2d)
from ml.exit_classifier import load_exit_classifier, check_norm_compatibility
from ml.model import get_model
from ml.transform import (SimDataset, load_normalization_params, face_exit_features_2d, denorm_position_relative,
                          denorm_angle_phi, denorm_radius_r, polar_to_uv, denorm_linear_scale_probit,
                          denorm_log_scale_probit, denorm_exit_angle, decode_energy_out)
from ml.utils import run_uniform_ode_inference
from physics.electron_data import load_electron_dataset

log = logging.getLogger("electron_cell_compare")
NF = SQUARE.num_faces
ABS = SQUARE.interior_class("absorbed")
CLASS_NAMES = SQUARE.class_names
# Baseline flow: an E_out within this fraction of (E_in - E_cut) above the cutoff counts as "stopped"
STOP_F_OUT = 1e-3


def load_flow(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    hidden, input_dim = sd["input_proj.weight"].shape
    target_dim = sd["output.weight"].shape[0]
    depth = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    expansion = sd["blocks.0.ff.0.weight"].shape[0] // hidden
    model = get_model(input_dim, input_dim - target_dim, hidden_dim=hidden, depth=depth, expansion_factor=expansion)
    model.load_state_dict(sd)
    return model.to(device).eval(), input_dim - target_dim


def ode(model, cond, target_dim, args, device):
    x0 = torch.cat([cond, torch.randn(cond.shape[0], target_dim)], 1).to(device)
    return run_uniform_ode_inference(model, x0, device, args.ode_method, args.ode_steps, show_timing=False)[
        :, cond.shape[1]:].double().numpy()


def empty(n):
    return {k: np.full(n, np.nan) for k in ("s", "psi", "r", "E_out", "path")} | {"cls": np.full(n, -1)}


def truth(df):
    out = empty(len(df))
    cls = df["exit_class"].to_numpy()
    out["cls"] = cls
    ex = cls < NF
    s, psi, r = face_exit_features_2d(df["x_out_norm"].to_numpy()[ex], df["y_out_norm"].to_numpy()[ex],
                                      df["u_out"].to_numpy()[ex], df["v_out"].to_numpy()[ex], cls[ex])
    out["s"][ex], out["psi"][ex], out["r"][ex] = s, psi, r
    out["E_out"][ex] = df["E_out"].to_numpy()[ex]
    out["path"] = df["path_length"].to_numpy().astype(np.float64)
    return out


def predict_baseline(df, args, device):
    norm, _ = load_normalization_params(args.baseline_norm)
    model, cond_dim = load_flow(args.baseline_ckpt, device)
    cond = SimDataset(df, norm, "2d_electron_boundary")._cached_x1[:, :cond_dim]
    z = ode(model, cond, 7, args, device)
    E_in, E_cut = df["E_in"].to_numpy(), float(norm["E_cutoff"])
    x, y = denorm_position_relative(z[:, 0]), denorm_position_relative(z[:, 1])
    u, v = polar_to_uv(denorm_angle_phi(z[:, 2]), denorm_radius_r(z[:, 3]))
    f_out = np.clip(denorm_linear_scale_probit(z[:, 4]), 0, 1)
    out = empty(len(df))
    pts = np.stack([x, y], 1)
    face = polygon_nearest_face(UNIT_SQUARE_VERTS, pts)
    stopped = f_out <= STOP_F_OUT
    out["cls"] = np.where(stopped, ABS, face)
    ex = ~stopped
    out["s"][ex] = polygon_face_coordinate(UNIT_SQUARE_VERTS, face[ex], pts[ex])
    mu_n, mu_t = direction_to_face_frame_2d(UNIT_SQUARE_VERTS, face[ex], u[ex], v[ex])
    out["psi"][ex] = np.arctan2(mu_t, mu_n)  # |psi| > pi/2 = pointing back into the cell
    out["r"][ex] = np.sqrt(mu_n ** 2 + mu_t ** 2)
    out["E_out"][ex] = f_out[ex] * (E_in[ex] - E_cut) + E_cut
    out["path"] = denorm_log_scale_probit(z[:, 6], norm["path_lower"], norm["path_upper"])
    depth = -polygon_signed_distances(UNIT_SQUARE_VERTS, pts).max(1)  # distance inside the cell
    out["diag"] = {"inward_direction_frac_of_exits": float((mu_n <= 0).mean()),
                   "exit_point_deeper_than_1e-3_frac": float((depth[ex] > 1e-3).mean()),
                   "exit_point_deeper_than_1e-2_frac": float((depth[ex] > 1e-2).mean())}
    return out


def predict_face(df, args, device):
    norm, _ = load_normalization_params(args.face_norm)
    clf, topo, ckpt = load_exit_classifier(args.classifier, device)
    check_norm_compatibility(ckpt, norm, ["E_in_lower", "E_in_upper"])
    assert topo.class_names == CLASS_NAMES, topo.class_names
    model, cond_dim = load_flow(args.face_ckpt, device)
    base_cond = SimDataset(df, norm, "2d_electron_boundary")._cached_x1[:, :5]
    cls = clf.sample(base_cond.to(device)).cpu().numpy()
    out = empty(len(df))
    out["cls"] = cls
    ex = np.flatnonzero(cls < NF)
    onehot = torch.nn.functional.one_hot(torch.as_tensor(cls[ex]), NF).float()
    z = ode(model, torch.cat([base_cond[ex], onehot], 1), 5, args, device)
    E_in, E_cut = df["E_in"].to_numpy()[ex], float(norm["E_cutoff"])
    out["s"][ex] = denorm_position_relative(z[:, 0])
    out["psi"][ex] = denorm_exit_angle(z[:, 1])
    out["r"][ex] = denorm_radius_r(z[:, 2])
    out["E_out"][ex] = decode_energy_out(z[:, 3], E_in, E_cut, norm)
    out["path"][ex] = denorm_log_scale_probit(z[:, 4], norm["path_lower"], norm["path_upper"])
    return out


def deposited(pred, E_in):
    return np.where(pred["cls"] == ABS, E_in, E_in - np.nan_to_num(pred["E_out"]))


def score(pred, ref, E_in, ebins):
    n = len(E_in)
    m = {}
    pf = np.bincount(pred["cls"], minlength=SQUARE.num_classes) / n
    rf = np.bincount(ref["cls"], minlength=SQUARE.num_classes) / n
    se = np.sqrt((pf * (1 - pf) + rf * (1 - rf)) / n).clip(1e-12)
    m["class_frac"] = dict(zip(CLASS_NAMES, pf.round(5).tolist()))
    m["class_frac_abs_err"] = dict(zip(CLASS_NAMES, np.abs(pf - rf).round(5).tolist()))
    m["class_frac_z"] = dict(zip(CLASS_NAMES, ((pf - rf) / se).round(2).tolist()))
    # Binned class fractions: chi^2 / dof over (E_in bin, class)
    b = np.digitize(E_in, ebins[1:-1])
    chi2, dof = 0.0, 0
    for i in range(len(ebins) - 1):
        sel = b == i
        nb = sel.sum()
        p = np.bincount(pred["cls"][sel], minlength=SQUARE.num_classes) / nb
        q = np.bincount(ref["cls"][sel], minlength=SQUARE.num_classes) / nb
        v = (p * (1 - p) + q * (1 - q)) / nb
        ok = v > 0
        chi2 += float((((p - q) ** 2)[ok] / v[ok]).sum())
        dof += int(ok.sum())
    m["class_frac_binned_chi2_per_dof"] = chi2 / max(dof, 1)
    # Per-face W1 of the continuous exit state
    w = {}
    for k in range(NF):
        a, c = pred["cls"] == k, ref["cls"] == k
        if a.sum() < 10 or c.sum() < 10:
            continue
        w[CLASS_NAMES[k]] = {
            "s": wasserstein_distance(pred["s"][a], ref["s"][c]),
            "psi": wasserstein_distance(np.clip(pred["psi"][a], -np.pi / 2, np.pi / 2), ref["psi"][c]),
            "r": wasserstein_distance(pred["r"][a], ref["r"][c]),
            "E_out/E_in": wasserstein_distance(pred["E_out"][a] / E_in[a], ref["E_out"][c] / E_in[c]),
            "log10_path": wasserstein_distance(np.log10(np.maximum(pred["path"][a], 1e-12)),
                                               np.log10(np.maximum(ref["path"][c], 1e-12))),
        }
    m["w1_per_face"] = {f: {q: round(x, 5) for q, x in d.items()} for f, d in w.items()}
    m["w1_mean_over_faces"] = {q: round(float(np.mean([d[q] for d in w.values()])), 5)
                               for q in next(iter(w.values()))}
    dp, dr = deposited(pred, E_in), deposited(ref, E_in)
    m["deposit_rel_bias"] = float(dp.sum() / dr.sum() - 1)
    m["deposit_rel_bias_per_Ebin"] = [round(float(dp[b == i].sum() / dr[b == i].sum() - 1), 4)
                                      for i in range(len(ebins) - 1)]
    if "diag" in pred:
        m["diagnostics"] = pred["diag"]
    return m


def plots(preds, ref, E_in, ebins, out_dir):
    import matplotlib.pyplot as plt

    colors = {"MC test": "k", "MC ref": "0.6", "baseline": "C3", "classifier+face": "C0"}
    b = np.digitize(E_in, ebins[1:-1])
    centers = np.sqrt(ebins[:-1] * ebins[1:])
    allp = {"MC test": ref, **preds}

    fig, axes = plt.subplots(1, SQUARE.num_classes, figsize=(3.4 * SQUARE.num_classes, 3.2))
    for name, p in allp.items():
        fr = np.array([np.bincount(p["cls"][b == i], minlength=SQUARE.num_classes) / (b == i).sum()
                       for i in range(len(ebins) - 1)])
        for j, ax in enumerate(axes):
            ax.plot(centers * 1e3, fr[:, j], "o-" if name == "MC test" else "-", ms=3, label=name,
                    color=colors.get(name))
    for j, ax in enumerate(axes):
        ax.set_title(CLASS_NAMES[j]), ax.set_xscale("log"), ax.set_xlabel("E_in [keV]"), ax.grid(alpha=.3)
    axes[0].set_ylabel("fraction"), axes[0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "class_fractions_vs_E.png"), dpi=200), plt.close(fig)

    qs = [("s", "s along face", (0, 1)), ("psi", "exit angle psi [rad]", (-np.pi / 2, np.pi / 2)),
          ("E_out/E_in", "E_out / E_in", (0, 1)), ("log10_path", "log10 path [cm]", None)]
    fig, axes = plt.subplots(len(qs), NF, figsize=(3.4 * NF, 2.6 * len(qs)))
    for k in range(NF):
        for i, (q, lab, rng) in enumerate(qs):
            ax = axes[i, k]
            for name, p in allp.items():
                sel = p["cls"] == k
                vals = {"s": p["s"], "psi": p["psi"], "E_out/E_in": p["E_out"] / E_in,
                        "log10_path": np.log10(np.maximum(p["path"], 1e-12))}[q][sel]
                ax.hist(vals, bins=60, range=rng, density=True, histtype="step", label=name, color=colors.get(name),
                        lw=1.6 if name == "MC test" else 1.0)
            ax.set_xlabel(lab, fontsize=8)
            if i == 0:
                ax.set_title(CLASS_NAMES[k])
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "per_face_distributions.png"), dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    dr = deposited(ref, E_in)
    for name, p in preds.items():
        dp = deposited(p, E_in)
        ax.plot(centers * 1e3, [dp[b == i].sum() / dr[b == i].sum() - 1 for i in range(len(ebins) - 1)], "o-", ms=3,
                label=name, color=colors.get(name))
    ax.axhline(0, color="k", lw=.8), ax.set_xscale("log"), ax.set_xlabel("E_in [keV]")
    ax.set_ylabel("deposited energy rel. bias"), ax.legend(fontsize=7), ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "deposit_bias_vs_E.png"), dpi=200), plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--test_npz", required=True)
    p.add_argument("--baseline_ckpt"), p.add_argument("--baseline_norm")
    p.add_argument("--face_ckpt"), p.add_argument("--face_norm"), p.add_argument("--classifier")
    p.add_argument("--ref_npz")
    p.add_argument("--ode_steps", type=int, default=100)
    p.add_argument("--ode_method", default="euler")
    p.add_argument("--n_ebins", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="validation/results/electron_cell")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_electron_dataset(args.test_npz)
    E_in = df["E_in"].to_numpy().astype(np.float64)
    ebins = np.quantile(E_in, np.linspace(0, 1, args.n_ebins + 1))
    ref = truth(df)
    preds = {}
    if args.ref_npz:
        rdf = load_electron_dataset(args.ref_npz)
        idx = np.random.default_rng(args.seed).choice(len(rdf), len(df), replace=len(rdf) < len(df))
        # Independent MC with the same source distribution: pair it with the test rows by E_in rank, so the
        # E_in-binned scores compare like with like (exit statistics depend on E_in much more than on the rest).
        rdf = rdf.iloc[idx].reset_index(drop=True)
        order_t, order_r = np.argsort(E_in), np.argsort(rdf["E_in"].to_numpy())
        rr = truth(rdf)
        mc_ref = {k: np.empty_like(v) for k, v in rr.items()}
        for k in rr:
            mc_ref[k][order_t] = rr[k][order_r]
        E_ref = np.empty_like(E_in)
        E_ref[order_t] = rdf["E_in"].to_numpy()[order_r]
        # E_out / E_in uses the test E_in, so rescale to keep the reference's own ratios
        mc_ref["E_out"] = mc_ref["E_out"] * E_in / E_ref
        preds["MC ref"] = mc_ref
    if args.baseline_ckpt:
        preds["baseline"] = predict_baseline(df, args, device)
    if args.face_ckpt:
        preds["classifier+face"] = predict_face(df, args, device)

    results = {name: score(pr, ref, E_in, ebins) for name, pr in preds.items()}
    results["MC test"] = {"class_frac": dict(zip(CLASS_NAMES, (np.bincount(ref["cls"], minlength=SQUARE.num_classes)
                                                              / len(df)).round(5).tolist())), "N": len(df)}
    plots(preds, ref, E_in, ebins, args.out_dir)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'':22s}" + "".join(f"{n:>18s}" for n in preds))
    rows = [("class |err| max", lambda m: max(m["class_frac_abs_err"].values())),
            ("class chi2/dof (E bins)", lambda m: m["class_frac_binned_chi2_per_dof"])]
    for q in ("s", "psi", "r", "E_out/E_in", "log10_path"):
        rows.append((f"W1 {q}", lambda m, q=q: m["w1_mean_over_faces"][q]))
    rows.append(("deposit rel bias", lambda m: m["deposit_rel_bias"]))
    rows.append(("deposit |bias| max bin", lambda m: max(abs(x) for x in m["deposit_rel_bias_per_Ebin"])))
    for lab, fn in rows:
        print(f"{lab:22s}" + "".join(f"{fn(results[n]):18.4g}" for n in preds))
    for n in preds:
        if "diagnostics" in results[n]:
            print(f"{n} diagnostics: {results[n]['diagnostics']}")
    print(f"\nWrote {args.out_dir}/metrics.json and plots")


if __name__ == "__main__":
    main()
