"""
Single-cell check of the secondary GMC (ml/secondary_gmc.py) against held-out analog MC.

The GMC is run on the exact entry states of an independent MC dataset (different seed) and compared per entry-energy
bin: exit-class fractions, relevant-daughter and photon rates, dose per transit, exit-state distributions and the
relevant daughters (energy, birth depth, direction). Both model sets are checked: BOUNDARY (electrons entering
through a face) and INTERIOR (relevant secondaries at birth).

    python validation/secondary_cell_compare.py --test_npz ../ElectronTransport/analog_cu_100um_1keV-10MeV_log_2M_7.npz \
        --model ml/models/secondary_gmc_cu_100um/secondary_gmc.pth --out_dir validation/results/secondary_cell_cu_100um
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sec_plot_style as S  # noqa: E402
from ml.secondary_gmc import SecondaryGMC, load_tables, NF, ABSORBED, exit_targets, sigmoid, describe  # noqa: E402

plt = S.plt
CLASS_NAMES = ("bottom", "right (transmitted)", "top", "left (back)", "absorbed")


def wasserstein1(a, b):
    a, b = np.sort(a), np.sort(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    q = np.linspace(0, 1, 512)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def binned(E, edges, *vals):
    idx = np.clip(np.digitize(E, edges) - 1, 0, len(edges) - 2)
    return idx


def frac_with_err(mask, idx, nb):
    n = np.bincount(idx, minlength=nb).astype(float)
    k = np.bincount(idx, weights=mask.astype(float), minlength=nb)
    p = k / np.maximum(n, 1)
    # Floor p at 0.5 / n (as ml/exit_classifier.binned_fraction_check): at p = 0 the binomial s.e. collapses and a
    # negligible leak would read as hundreds of sigma
    nn = np.maximum(n, 1)
    pf = np.clip(p, 0.5 / nn, 1 - 0.5 / nn)
    return p, np.sqrt(pf * (1 - pf) / nn), n


def mean_with_err(x, idx, nb):
    n = np.bincount(idx, minlength=nb).astype(float)
    s = np.bincount(idx, weights=x, minlength=nb)
    s2 = np.bincount(idx, weights=x * x, minlength=nb)
    m = s / np.maximum(n, 1)
    var = np.maximum(s2 / np.maximum(n, 1) - m * m, 0)
    return m, np.sqrt(var / np.maximum(n, 1)), n


def compare(which, T, rows, K, edges, out_dir, metrics, rng, title=None):
    """title: figure label (default: the model's description). The cell side of every row (T["L"]) is passed to the
    kernel; one-size models ignore it."""
    spec = T["spec"]
    E0 = T["E0"][rows]
    Lr = T["L"][rows] if "L" in T else None
    if which == "boundary":
        g = K.boundary(T["y0"][rows], T["u0"][rows], T["v0"][rows], np.full(len(rows), 0.5), E0, L=Lr)
    else:
        g = K.interior(T["x0"][rows], T["y0"][rows], T["u0"][rows], T["v0"][rows], np.full(len(rows), 0.5), E0,
                       L=Lr)
    # MC truth for these rows
    mc_cls, mc_n, mc_flag = T["cls"][rows], T["n_rel"][rows], T["flag"][rows]
    mc_dose, mc_q = T["dose"][rows], T["q"][rows]
    gm_cls = g["cls"]
    gm_n = np.bincount(g["daughters"]["parent"], minlength=len(rows))
    nb = len(edges) - 1
    idx = binned(E0, edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    M = {"n_rows": int(len(rows)), "E_bin_edges_MeV": edges.tolist()}

    # ---- 1. exit classes, daughters, photons, dose vs energy -------------------------------------------------------
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    zmax = {}
    for c in range(5):
        ax = axs.flat[c]
        pm, em, n = frac_with_err(mc_cls == c, idx, nb)
        pg, eg, _ = frac_with_err(gm_cls == c, idx, nb)
        ok = n > 50
        ax.fill_between(centers[ok], (pm - 2 * em)[ok], (pm + 2 * em)[ok], color=S.REF, alpha=0.15, lw=0)
        ax.plot(centers[ok], pm[ok], **S.style("mc", label="Analog MC (held out) ± 2σ"))
        ax.plot(centers[ok], pg[ok], **S.style("gmc_sec", label="GMC"))
        ax.set_title(f"P(exit: {CLASS_NAMES[c]})")
        z = np.abs(pg - pm)[ok] / np.sqrt(em[ok] ** 2 + eg[ok] ** 2)
        zmax[CLASS_NAMES[c]] = float(z.max()) if z.size else float("nan")
        M[f"P_{c}"] = {"mc": pm.tolist(), "gmc": pg.tolist(), "mc_se": em.tolist()}
    M["class_fraction_max_abs_z"] = zmax
    # relevant daughters
    ax = axs.flat[5]
    pm, em, n = frac_with_err(mc_n >= 1, idx, nb)
    pg, eg, _ = frac_with_err(gm_n >= 1, idx, nb)
    ok = n > 50
    zr = np.abs(pg - pm)[ok] / np.sqrt(em[ok] ** 2 + eg[ok] ** 2)
    M["P_relevant_max_abs_z"] = float(zr.max()) if zr.size else float("nan")
    ax.fill_between(centers[ok], (pm - 2 * em)[ok], (pm + 2 * em)[ok], color=S.REF, alpha=0.15, lw=0)
    ax.plot(centers[ok], pm[ok], **S.style("mc", label="Analog MC (held out) ± 2σ"))
    ax.plot(centers[ok], pg[ok], **S.style("gmc_sec", label="GMC"))
    ax.set_title("P(≥1 relevant secondary)")
    M["P_relevant"] = {"mc": pm.tolist(), "gmc": pg.tolist(), "mc_se": em.tolist()}
    # photons
    ax = axs.flat[6]
    mm, me, n = mean_with_err(mc_q / E0, idx, nb)
    mg, _, _ = mean_with_err(g["photon_E"] / E0, idx, nb)
    ax.fill_between(centers[ok], (mm - 2 * me)[ok], (mm + 2 * me)[ok], color=S.REF, alpha=0.15, lw=0)
    ax.plot(centers[ok], mm[ok], **S.style("mc", label="Analog MC (held out) ± 2σ"))
    ax.plot(centers[ok], mg[ok], **S.style("gmc_sec", label="GMC"))
    ax.set_title("Photon energy leaving / E_in")
    ax.set_yscale("log")
    M["photon_frac"] = {"mc": mm.tolist(), "gmc": mg.tolist(), "mc_se": me.tolist()}
    # dose ratio
    ax = axs.flat[7]
    dm, de, n = mean_with_err(mc_dose, idx, nb)
    dg, dge, _ = mean_with_err(g["dose"], idx, nb)
    ratio = dg / np.maximum(dm, 1e-300)
    rerr = ratio * np.sqrt((de / np.maximum(dm, 1e-300)) ** 2 + (dge / np.maximum(dg, 1e-300)) ** 2)
    ax.axhline(1.0, color=S.REF, lw=1.2)
    ax.fill_between(centers[ok], (1 - 2 * rerr)[ok], (1 + 2 * rerr)[ok], color=S.REF, alpha=0.15, lw=0,
                    label="± 2σ (both samples)")
    ax.plot(centers[ok], ratio[ok], **S.style("gmc_sec", label="GMC / MC"))
    ax.set_title("Mean dose per transit, GMC / MC")
    M["dose_ratio"] = ratio.tolist()
    M["dose_ratio_err"] = rerr.tolist()
    M["total_dose_ratio"] = float(g["dose"].sum() / mc_dose.sum())
    M["total_photon_ratio"] = float(g["photon_E"].sum() / max(mc_q.sum(), 1e-300))
    zd = np.abs(ratio - 1)[ok] / np.maximum(rerr[ok], 1e-300)
    M["dose_ratio_max_abs_z"] = float(zd.max()) if zd.size else float("nan")
    mc_ex_, gm_ex_ = mc_cls < NF, gm_cls < NF
    M["exit_energy_total_ratio"] = float(g["E_out"][gm_ex_].sum() / max(T["E_out"][rows][mc_ex_].sum(), 1e-300))
    for ax in axs.flat:
        ax.set_xscale("log")
    for ax in axs[1]:
        ax.set_xlabel("Entry energy (MeV)")
    axs.flat[0].legend(loc="best", fontsize=8)
    axs.flat[7].legend(loc="best", fontsize=8)
    fig.suptitle(f"{which.capitalize()} transits, {title or describe(spec).split(',')[0]}: GMC vs held-out analog MC "
                 f"({len(rows)} entries)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{which}_rates_vs_E.png"), dpi=130)
    plt.close(fig)

    # ---- 2. exit-state and dose distributions in energy bands ------------------------------------------------------
    bands = [(1e-3, 3e-2, "1–30 keV"), (3e-2, 1.0, "30 keV–1 MeV"), (1.0, 10.0, "1–10 MeV")]
    if 10 ** spec.logE_hi > 11.0:
        bands.append((10.0, 10 ** spec.logE_hi, f"10–{10 ** spec.logE_hi:.0f} MeV"))
    fig, axs = plt.subplots(len(bands), 4, figsize=(17, 10))
    mc_ex = mc_cls < NF
    gm_ex = gm_cls < NF
    mc_t = exit_targets(mc_cls[mc_ex], T["x1"][rows][mc_ex], T["y1"][rows][mc_ex], T["u1"][rows][mc_ex],
                        T["v1"][rows][mc_ex], E0[mc_ex], T["E_out"][rows][mc_ex], spec.E_cut)
    gm_t = exit_targets(gm_cls[gm_ex], g["x"][gm_ex], g["y"][gm_ex], g["u"][gm_ex], g["v"][gm_ex], E0[gm_ex],
                        g["E_out"][gm_ex], spec.E_cut)
    names = ["position along exit face s", "exit angle from normal ψ (deg)", "log10 energy-loss fraction λ",
             "log10 dose / E_in"]

    def phys(t):
        return [sigmoid(t[:, 0]), np.degrees(0.5 * np.pi * np.tanh(t[:, 1])), np.log10(sigmoid(t[:, 3]))]

    mcP, gmP = phys(mc_t), phys(gm_t)
    W = {}
    for b, (lo, hi, lab) in enumerate(bands):
        mb = (E0[mc_ex] >= lo) & (E0[mc_ex] < hi)
        gb = (E0[gm_ex] >= lo) & (E0[gm_ex] < hi)
        md = (E0 >= lo) & (E0 < hi)
        for k in range(4):
            ax = axs[b, k]
            if k < 3:
                a, c = mcP[k][mb], gmP[k][gb]
            else:
                a = np.log10(np.maximum(mc_dose[md] / E0[md], 1e-12))
                c = np.log10(np.maximum(g["dose"][md] / E0[md], 1e-12))
            if len(a) < 20:
                ax.set_visible(False)
                continue
            lo_, hi_ = np.quantile(np.concatenate([a, c]), [0.002, 0.998])
            bins = np.linspace(lo_, hi_, 60)
            ax.hist(a, bins, density=True, histtype="step", color=S.REF, lw=2.0, label="Analog MC (held out)")
            ax.hist(c, bins, density=True, histtype="step", color=S.SERIES["gmc_sec"]["color"], lw=1.8,
                    ls="--", label="GMC")
            W[f"{lab} | {names[k]}"] = wasserstein1(a, c)
            ax.set_title(f"{lab}: W1 = {W[f'{lab} | {names[k]}']:.3g}", fontsize=9, fontweight="normal")
            if b == len(bands) - 1:
                ax.set_xlabel(names[k])
            ax.set_yticks([])
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{which.capitalize()} transits: exit state (exits only) and dose per transit (all)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{which}_exit_distributions.png"), dpi=130)
    plt.close(fig)
    M["W1"] = W

    # ---- 3. relevant daughters --------------------------------------------------------------------------------------
    D = T["dau"]
    sel = np.isin(D["parent"], rows)
    gd = g["daughters"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    pairs = [("log10 E_daughter (MeV)", np.log10(D["E"][sel]), np.log10(np.maximum(gd["E"], 1e-12))),
             ("birth position x (cell units)", D["x"][sel], gd["x"]),
             ("birth position y (cell units)", D["y"][sel], gd["y"]),
             ("direction cosine u", D["u"][sel], gd["u"])]
    for ax, (lab, a, c) in zip(axs, pairs):
        if len(a) < 20 or len(c) < 20:
            ax.set_visible(False)
            continue
        lo_, hi_ = np.quantile(np.concatenate([a, c]), [0.002, 0.998])
        bins = np.linspace(lo_, hi_, 50)
        ax.hist(a, bins, density=True, histtype="step", color=S.REF, lw=2.0, label=f"Analog MC ({len(a)})")
        ax.hist(c, bins, density=True, histtype="step", color=S.SERIES["gmc_sec"]["color"], lw=1.8, ls="--",
                label=f"GMC ({len(c)})")
        W[f"daughter | {lab}"] = wasserstein1(a, c)
        ax.set_title(f"W1 = {W[f'daughter | {lab}']:.3g}", fontsize=9, fontweight="normal")
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Relevant secondaries made in {which} transits (MC {int(sel.sum())}, GMC {len(gd['E'])})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{which}_daughters.png"), dpi=130)
    plt.close(fig)
    M["n_daughters"] = {"mc": int(sel.sum()), "gmc": int(len(gd["E"]))}
    M["daughter_energy_total_ratio"] = float(gd["E"].sum() / max(D["E"][sel].sum(), 1e-300))
    metrics[which] = M
    print(f"[{which}] rows {len(rows)}: total dose GMC/MC {M['total_dose_ratio']:.4f}, daughters GMC/MC "
          f"{M['n_daughters']['gmc']}/{M['n_daughters']['mc']}, class max|z| "
          f"{max(zmax.values()):.2f}, P(relevant) max|z| {M['P_relevant_max_abs_z']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boundary", type=int, default=500000)
    ap.add_argument("--ode_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    S.apply()
    os.makedirs(a.out_dir, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed)
    T = load_tables(a.test_npz, NR=K.spec.NR)
    rng = np.random.default_rng(a.seed)
    edges = np.logspace(np.log10(K.spec.E_cut), K.spec.logE_hi, 13)
    metrics = {"test_npz": a.test_npz, "model": a.model}
    b = np.flatnonzero(T["boundary"])
    b = rng.choice(b, min(a.n_boundary, len(b)), replace=False)
    compare("boundary", T, np.sort(b), K, edges, a.out_dir, metrics, rng)
    i = np.flatnonzero(T["interior"])
    compare("interior", T, i, K, np.logspace(np.log10(K.spec.E_cut), K.spec.logE_hi, 9), a.out_dir, metrics, rng)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
