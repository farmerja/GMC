"""
Per-cell-size check of the size-conditioned GMC and SCR kernel (ml/size_stream.py) against held-out analog MC.

The held-out data (another seed of ElectronTransport/generate_size_sweep.py) span 100 um - 1 cm continuously; they are
split into log-L bands and each band is checked like one cell size. Optional one-size references (--ref) run the
size model AND the old one-size model on that size's own held-out data, so the size model is compared head-to-head
with what a dedicated model achieves.

  gmc  secondary_cell_compare.compare per band (exit classes, daughters, photons, dose per transit, exit-state and
       daughter distributions) + a summary across bands (summary_gmc.png / summary_gmc.json).
  scr  per band: expected sub-cell images of the kernel (flow samples) vs the true deposit images, on an r x r grid of
       the cell (r = 4, 16), as a Brier skill score against a uniform image (per transit), and the bias of the
       aggregated depth and lateral profiles of boundary transits vs their noise floor.

    python validation/size_cell_compare.py gmc --model ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth \
        --test_npz ../ElectronTransport/sweep_water_100um-1cm_30MeV_7/part_0001.npz --out_dir validation/results/size_gmc/cell \
        --ref 1mm ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_2M_7_estar.npz ml/models/secondary_gmc_water_1mm_30MeV_estar/secondary_gmc.pth
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sec_plot_style as S  # noqa: E402
from secondary_cell_compare import compare  # noqa: E402
from ml.secondary_gmc import SecondaryGMC, load_tables  # noqa: E402
from ml import subcell_kernel as sk  # noqa: E402

plt = S.plt
BAND_EDGES_UM = (100, 251, 631, 1585, 3981, 10000)


def band_label(lo, hi):
    f = lambda u: f"{u:.0f} µm" if u < 1000 else f"{u / 1e3:g} mm"  # noqa: E731
    return f"{f(lo)}–{f(hi)}"


def subset_T(T, rows):
    """T restricted to segment rows (daughters of other rows dropped, parent indices remapped)."""
    keep = np.zeros(len(T["E0"]), bool)
    keep[rows] = True
    new = np.full(len(keep), -1)
    new[np.flatnonzero(keep)] = np.arange(keep.sum())
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in T.items()
           if k != "dau"}
    D = T["dau"]
    m = keep[D["parent"]]
    out["dau"] = {k: v[m] for k, v in D.items()}
    out["dau"]["parent"] = new[D["parent"][m]]
    return out


def gmc_summary_row(M):
    def zmax(d):
        return max(d.values()) if isinstance(d, dict) else d
    W = M.get("W1", {})
    return {"n": M["n_rows"], "dose": M["total_dose_ratio"], "photon": M["total_photon_ratio"],
            "exit_E": M["exit_energy_total_ratio"], "class_z": zmax(M["class_fraction_max_abs_z"]),
            "rel_z": M["P_relevant_max_abs_z"], "dose_z": M["dose_ratio_max_abs_z"],
            "daughters": M["n_daughters"]["gmc"] / max(M["n_daughters"]["mc"], 1),
            "W1_angle_1-10MeV": W.get("1–10 MeV | exit angle from normal ψ (deg)", float("nan")),
            "W1_loss_1-10MeV": W.get("1–10 MeV | log10 energy-loss fraction λ", float("nan"))}


def run_gmc_bands(K, T, out_dir, n_max, rng, label_fn):
    rows_out = {}
    L = T["L"]
    edges = np.logspace(np.log10(K.spec.E_cut), K.spec.logE_hi, 13)
    for lo, hi in zip(BAND_EDGES_UM[:-1], BAND_EDGES_UM[1:]):
        m = (L >= lo * 1e-4) & (L < hi * 1e-4 * (1 + 1e-9))
        if not m.any():
            continue
        lab = label_fn(lo, hi)
        d = os.path.join(out_dir, lab.replace(" ", "").replace("µ", "u").replace("–", "-"))
        os.makedirs(d, exist_ok=True)
        rows = np.flatnonzero(m)
        Tb = subset_T(T, rows)
        metrics = {}
        b = np.flatnonzero(Tb["boundary"])
        b = np.sort(rng.choice(b, min(n_max, len(b)), replace=False))
        print(f"\n== band {lab}: {len(b)} boundary rows", flush=True)
        compare("boundary", Tb, b, K, edges, d, metrics, rng, title=lab)
        i = np.flatnonzero(Tb["interior"])
        if len(i) > 200:
            compare("interior", Tb, i, K, np.logspace(np.log10(K.spec.E_cut), K.spec.logE_hi, 9), d, metrics, rng,
                    title=lab)
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
        rows_out[lab] = {w: gmc_summary_row(metrics[w]) for w in metrics}
    return rows_out


def cmd_gmc(a):
    S.apply()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    K = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    T = load_tables(a.test_npz, NR=K.spec.NR)
    summary = {"model": a.model, "test_npz": a.test_npz, "bands": {}, "refs": {}}
    summary["bands"] = run_gmc_bands(K, T, a.out_dir, a.n_boundary, rng, band_label)
    del T
    for name, test, ref_model in a.ref or []:
        Tr = load_tables(test, NR=K.spec.NR)
        d = os.path.join(a.out_dir, f"ref_{name}")
        os.makedirs(d, exist_ok=True)
        b = np.flatnonzero(Tr["boundary"])
        b = np.sort(rng.choice(b, min(a.n_boundary, len(b)), replace=False))
        i = np.flatnonzero(Tr["interior"])
        res = {}
        for who, Km in (("size_model", K), ("one_size_model", SecondaryGMC(ref_model, dev, ode_steps=a.ode_steps,
                                                                            seed=a.seed))):
            metrics = {}
            dd = os.path.join(d, who)
            os.makedirs(dd, exist_ok=True)
            edges = np.logspace(np.log10(Km.spec.E_cut), Km.spec.logE_hi, 13)
            print(f"\n== reference {name}, {who}", flush=True)
            compare("boundary", Tr, b, Km, edges, dd, metrics, rng, title=f"{name}, {who.replace('_', ' ')}")
            compare("interior", Tr, i, Km, np.logspace(np.log10(Km.spec.E_cut), Km.spec.logE_hi, 9), dd, metrics,
                    rng, title=f"{name}, {who.replace('_', ' ')}")
            res[who] = {w: gmc_summary_row(metrics[w]) for w in metrics}
        summary["refs"][name] = res
        del Tr
    with open(os.path.join(a.out_dir, "summary_gmc.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot_gmc_summary(summary, a.out_dir)
    print_gmc_summary(summary)


def print_gmc_summary(s):
    cols = ("dose", "photon", "exit_E", "daughters", "class_z", "rel_z", "dose_z", "W1_angle_1-10MeV")
    print("\nGMC per cell size (ratios GMC/MC; z = max |GMC - MC| / sigma over energy bins):")
    print(f"{'':>34} " + " ".join(f"{c:>10}" for c in cols))
    for lab, r in s["bands"].items():
        for w, v in r.items():
            print(f"{lab + ' ' + w:>34} " + " ".join(f"{v[c]:10.4f}" for c in cols))
    for name, res in s["refs"].items():
        for who, r in res.items():
            for w, v in r.items():
                print(f"{name + ' ' + who + ' ' + w:>34} " + " ".join(f"{v[c]:10.4f}" for c in cols))


def plot_gmc_summary(s, out_dir):
    labs = list(s["bands"])
    fig, axs = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(labs))
    for which, mk in (("boundary", "o"), ("interior", "s")):
        for ax, key, title in zip(axs, ("dose", "photon", "daughters", "class_z"),
                                  ("Total dose per transit, GMC / MC", "Photon energy leaving, GMC / MC",
                                   "Relevant daughters, GMC / MC", "Exit-class fractions: max |z|")):
            v = [s["bands"][lb].get(which, {}).get(key, np.nan) for lb in labs]
            ax.plot(x, v, marker=mk, label=which)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(x, labs, rotation=30, fontsize=8)
    for ax in axs[:3]:
        ax.axhline(1.0, color="0.4", lw=1)
    axs[3].axhline(3.0, color="0.4", lw=1, ls="--")
    axs[0].legend(fontsize=8)
    fig.suptitle("Size-conditioned GMC vs held-out analog MC, by cell size", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_gmc.png"), dpi=130)
    plt.close(fig)


# ====================================================================================================================
# SCR
# ====================================================================================================================
def truth_points(ent, rows_sel, starts):
    """Deposit positions (moment means) and energies of the selected transits: (row index into rows_sel, x, y, E)."""
    out = []
    for k, r in enumerate(rows_sel):
        a_, b_ = starts[r], starts[r + 1]
        n = ent["n"][a_:b_].astype(np.float64)
        bins = ent["bin"][a_:b_].astype(np.int64)
        bx, by = bins // ent["n"][a_:b_], bins % ent["n"][a_:b_]
        mom = ent["mom"][a_:b_].astype(np.float64)
        out.append((np.full(b_ - a_, k), (bx + mom[:, 0]) / n, (by + mom[:, 1]) / n, ent["E"][a_:b_]))
    return [np.concatenate(c) for c in zip(*out)]


def images_from_points(k, x, y, w, n_rows, r):
    img = np.zeros((n_rows, r, r))
    np.add.at(img, (k, np.clip((x * r).astype(int), 0, r - 1), np.clip((y * r).astype(int), 0, r - 1)), w)
    return img


@torch.no_grad()
def scr_band_eval(spec, model, R, ent, rows, dev, n_samples, chunk=2000, resolutions=(4, 16)):
    """Per-transit Brier skill and boundary depth / lateral profile bias for the given transit rows."""
    starts = np.searchsorted(ent["seg"], np.arange(len(R["E0"]) + 1))
    Rd = sk.to_device(R, dev)
    acc = {r: {"model": 0.0, "uniform": 0.0} for r in resolutions}
    rmax = max(resolutions)
    prof = {k: np.zeros((2, rmax)) for k in ("depth", "lateral")}   # [truth, model] summed over boundary transits
    pvar = {k: np.zeros(rmax) for k in ("depth", "lateral")}
    for i in range(0, len(rows), chunk):
        rr = rows[i:i + chunk]
        z = model.sample(model.context(sk.cond_features(spec, sk.gather(Rd, torch.as_tensor(rr, device=dev)))),
                         n_samples, base="sobol").cpu().numpy()
        dose = R["dose"][rr]
        k = np.repeat(np.arange(len(rr)), n_samples)
        pm = images_from_points(k, z[..., 0].ravel(), z[..., 1].ravel(), np.repeat(dose / n_samples, n_samples),
                                len(rr), rmax)
        kt, xt, yt, Et = truth_points(ent, rr, starts)
        tm = images_from_points(kt, xt, yt, Et, len(rr), rmax)
        for r in resolutions:
            f = rmax // r
            co = lambda a_: a_.reshape(len(rr), r, f, r, f).sum((2, 4))  # noqa: E731
            t_, p_ = co(tm), co(pm)
            u_ = np.broadcast_to(dose[:, None, None] / (r * r), t_.shape)
            acc[r]["model"] += float(((t_ - p_) ** 2).sum())
            acc[r]["uniform"] += float(((t_ - u_) ** 2).sum())
        bnd = R["start"][rr] < 0.5
        for name, ax in (("depth", 2), ("lateral", 1)):
            tp, pp = tm[bnd].sum(ax), pm[bnd].sum(ax)
            prof[name][0] += tp.sum(0)
            prof[name][1] += pp.sum(0)
            pvar[name] += ((tp - pp) ** 2).sum(0)
    out = {f"skill_{r}": 1 - acc[r]["model"] / max(acc[r]["uniform"], 1e-300) for r in resolutions}
    for name in prof:
        t_, p_ = prof[name]
        rl = float(np.linalg.norm(t_ - p_) / max(np.linalg.norm(t_), 1e-300))
        nz = float(math.sqrt(pvar[name].sum()) / max(np.linalg.norm(t_), 1e-300))
        out[f"{name}_relL2"], out[f"{name}_noise"] = rl, nz
        out[f"{name}_bias"] = float(math.sqrt(max(rl * rl - nz * nz, 0.0)))
    out["n"] = int(len(rows))
    return out, prof


def cmd_scr(a):
    S.apply()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec, model, _ = sk.load_kernel(a.model, dev)
    rng = np.random.default_rng(a.seed)
    _, R, ent = sk.load_kernel_data(a.test_npz)
    res = {"model": a.model, "test_npz": a.test_npz, "bands": {}, "refs": {}}
    fig, axs = plt.subplots(2, len(BAND_EDGES_UM) - 1, figsize=(18, 6.5), squeeze=False)
    for j, (lo, hi) in enumerate(zip(BAND_EDGES_UM[:-1], BAND_EDGES_UM[1:])):
        m = (R["L"] >= lo * 1e-4) & (R["L"] < hi * 1e-4 * (1 + 1e-6))
        rows = np.flatnonzero(m)
        rows = np.sort(rng.choice(rows, min(a.n_transits, len(rows)), replace=False))
        lab = band_label(lo, hi)
        out, prof = scr_band_eval(spec, model, R, ent, rows, dev, a.n_samples)
        res["bands"][lab] = out
        print(f"{lab:>14}: skill 4x4 {out['skill_4']:.3f}, 16x16 {out['skill_16']:.3f}; boundary depth profile "
              f"relL2 {out['depth_relL2']:.4f} (noise {out['depth_noise']:.4f}), lateral {out['lateral_relL2']:.4f} "
              f"(noise {out['lateral_noise']:.4f})", flush=True)
        for row, name in enumerate(("depth", "lateral")):
            ax = axs[row, j]
            c = (np.arange(prof[name].shape[1]) + 0.5) / prof[name].shape[1]
            ax.step(c, prof[name][0] / prof[name][0].sum(), where="mid", color=S.REF, lw=2, label="analog MC")
            ax.step(c, prof[name][1] / prof[name][1].sum(), where="mid", color="#d9480f", lw=1.4, ls="--",
                    label="SCR kernel")
            ax.set_title(f"{lab}: {name}", fontsize=9)
            ax.set_yticks([])
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Boundary transits: depth (x) and lateral (y) dose profiles in the cell, by cell size",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "scr_profiles.png"), dpi=130)
    plt.close(fig)
    del R, ent
    for name, test, ref_model in a.ref or []:
        _, Rr, er = sk.load_kernel_data(test)
        n_fixed = int(np.load(test[:-4] + "_subcell.npz")["subcell_bins"])
        er["n"] = np.full(len(er["seg"]), n_fixed, np.int16)
        rows = np.sort(rng.choice(len(Rr["E0"]), min(a.n_transits, len(Rr["E0"])), replace=False))
        Lfix = float(np.load(test)["box_dims"][0])
        rr_ = {}
        for who, path in (("size_model", a.model), ("one_size_model", ref_model)):
            sp, mo, _ = sk.load_kernel(path, dev)
            R_ = dict(Rr)
            if getattr(sp, "size_cond", False):
                R_["L"] = np.full(len(Rr["E0"]), Lfix, np.float32)
            rr_[who], _ = scr_band_eval(sp, mo, R_, er, rows, dev, a.n_samples)
            o = rr_[who]
            print(f"ref {name} {who:>15}: skill 4x4 {o['skill_4']:.3f}, 16x16 {o['skill_16']:.3f}; depth relL2 "
                  f"{o['depth_relL2']:.4f} (noise {o['depth_noise']:.4f}), lateral {o['lateral_relL2']:.4f} "
                  f"(noise {o['lateral_noise']:.4f})", flush=True)
        res["refs"][name] = rr_
    with open(os.path.join(a.out_dir, "summary_scr.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gmc", "scr"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--test_npz", required=True, help="held-out generate_size_sweep.py file")
        p.add_argument("--out_dir", required=True)
        p.add_argument("--ref", nargs=3, action="append", metavar=("NAME", "TEST_NPZ", "ONE_SIZE_MODEL"),
                       help="one-size reference: its held-out data and its dedicated model")
        p.add_argument("--seed", type=int, default=7)
    sub.choices["gmc"].add_argument("--n_boundary", type=int, default=200000, help="boundary rows per band")
    sub.choices["gmc"].add_argument("--ode_steps", type=int, default=20)
    sub.choices["scr"].add_argument("--n_transits", type=int, default=20000, help="transits per band")
    sub.choices["scr"].add_argument("--n_samples", type=int, default=1024, help="flow samples per transit")
    a = ap.parse_args()
    (cmd_gmc if a.cmd == "gmc" else cmd_scr)(a)


if __name__ == "__main__":
    main()
