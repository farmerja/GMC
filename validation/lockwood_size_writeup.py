#!/usr/bin/env python3
"""
Write-up figures for the size-conditioned Al / Cu GMC + SCR on the Lockwood / Sandia depth-dose cases
(validation/lockwood_quadtree.py runs): the fewest-cells mesh (one cell), random-size tilings, the uniform 100 um
mesh and the old dedicated 100 um model (lockwood_slab.py), all vs the ESTAR analog MC.

  ratio_grid<sfx>.png      GMC / MC vs depth per case, MC 2-sigma band shaded
  speed_accuracy.png       speed-up vs MC against rms shape error and chi2/dof (K8 runs + the dedicated model)
  timing.png               ms per primary: MC vs each mesh per case; chain cost vs transits; chain / tally split
  meshes.png               the meshes of the Al 1 MeV case
  summary.json             every scored run (both K settings)

    .venv/bin/python validation/lockwood_size_writeup.py --out_dir validation/results/lockwood_sandia/size/writeup
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lockwood_quadtree import OUT, LW_OUT, load_gmc, load_cases, find_run, score_run  # noqa: E402

FAM = {"old": ("#7b7b7b", "dedicated 100 µm model, uniform 100 µm"), "uniform": ("#2a78d6", "size model, uniform 100 µm"),
       "random": ("#1baf7a", "size model, random sizes"), "one": ("#eb6834", "size model, one cell")}
RAND_LS = {"random:1": "-", "random:2": "--", "random:3": ":"}


def family(mesh):
    if "100um model" in mesh:
        return "old"
    return mesh.split(":")[0]


def collect(n_bins, min_frac):
    cases = load_cases()
    files = {"auto": sorted(glob.glob(str(OUT / "gmc_*_200k.npz"))),
             "k8": sorted(glob.glob(str(OUT / "gmc_*_200k_k8.npz"))),
             "old": sorted(glob.glob(str(LW_OUT / "gmc_*_200k_n20_estar.npz")))}
    out = {}
    for kset, fl in files.items():
        for f in fl:
            z, meta = load_gmc(f)
            key = meta["case"]
            c = cases[key]
            E = meta["E"]
            mc = find_run(key, E, 200000, 79, "estar")
            if mc is None:
                continue
            rho = float(z["rho"])
            R_g = meta["R_cm"] * rho / 1.1
            xm, g, m_v, sig, use, t = score_run(z, meta, mc, R_g, n_bins, min_frac)
            mc_ms = 1e3 * mc.meta["seconds"] / mc.meta["N"]
            t["total_ratio"] = float(z["prof"].sum(1).mean()) / mc.total("escape")[0]
            t["speedup_vs_mc"] = mc_ms / t["gmc_ms_per_primary"]
            t["mc_ms_per_primary"] = mc_ms
            t["material"], t["angle"], t["E"] = c["material"], c["angle"], E
            t["family"] = family(meta["mesh"])
            t["kset"] = kset
            t["L_min_um"] = float(1e4 * np.min(z["L"])) if "L" in z else 100.0
            t["L_max_um"] = float(1e4 * np.max(z["L"])) if "L" in z else 100.0
            out.setdefault(key, []).append({"file": Path(f).name, "stats": t, "xm": xm, "g": g, "m": m_v, "sig": sig,
                                            "use": use, "z": z, "meta": meta})
    return out


def ratio_grid(data, kset, png, plt):
    keys = [k for k in load_cases() if k in data and any(r["stats"]["kset"] == kset for r in data[k])]
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.3 * nr), squeeze=False, constrained_layout=True)
    for ax in axs.ravel()[len(keys):]:
        ax.set_visible(False)
    for ax, key in zip(axs.ravel(), keys):
        runs = [r for r in data[key] if r["stats"]["kset"] in (kset, "old")]
        ref = runs[0]
        mc_rel = None
        for r in sorted(runs, key=lambda r: ["old", "uniform", "random", "one"].index(r["stats"]["family"])):
            s = r["stats"]
            col, _ = FAM[s["family"]]
            u = r["use"]
            ratio = r["g"][u] / r["m"][u]
            ls = RAND_LS.get(s["mesh"], "--" if s["family"] == "old" else "-")
            lw = 1.6 if s["family"] in ("one", "old") else 1.1
            ax.plot(r["xm"][u], ratio, ls=ls, color=col, lw=lw, marker="o" if s["family"] == "one" else None, ms=2.5)
            if s["family"] == "uniform":
                mc_rel = (r["xm"][u], r["sig"][u] / r["m"][u])
        if mc_rel is not None:
            ax.fill_between(mc_rel[0], 1 - 2 * mc_rel[1], 1 + 2 * mc_rel[1], color="0.85", lw=0, zorder=0)
        ax.axhline(1, color="0.4", lw=0.8)
        s0 = ref["stats"]
        ax.set_title(f"{s0['material']} {s0['E']:g} MeV, {s0['angle']:g}°", fontsize=10)
        ax.set_ylim(0.9, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * (v - 1):+.0f} %"))
        ax.set_xlabel("depth / CSDA range")
    for ax in axs[:, 0]:
        ax.set_ylabel("GMC / MC − 1")
    h = [plt.Line2D([], [], color=c, lw=1.6, ls="--" if k == "old" else "-", marker="o" if k == "one" else None,
                    ms=3, label=l) for k, (c, l) in FAM.items()]
    h += [plt.Line2D([], [], color=FAM["random"][0], ls=ls, lw=1.1, label=m) for m, ls in RAND_LS.items()]
    h.append(plt.Rectangle((0, 0), 1, 1, color="0.85", label="± 2σ combined noise (uniform run)"))
    fig.legend(handles=h, loc="outside lower center", ncol=8, fontsize=8, frameon=False)
    ktxt = "K = max(8, 0.25 × auto)" if kset == "k8" else "K = max(32, auto)"
    fig.suptitle(f"Lockwood depth-dose, size-conditioned GMC + SCR vs analog MC (ESTAR), {ktxt}; bins with "
                 f"J > 10 % of max", fontweight="bold")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def speed_accuracy(data, png, plt):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    mk = {("Al", 0): "o", ("Al", 60): "^", ("Cu", 0): "s"}
    for key, runs in data.items():
        for r in runs:
            s = r["stats"]
            col, _ = FAM[s["family"]]
            m = mk[(s["material"], int(s["angle"]))]
            if s["kset"] == "auto":
                continue
            kw = dict(marker=m, s=40, zorder=3, edgecolor=col, lw=1.2, facecolor=col)
            ax[0].scatter(s["speedup_vs_mc"], 100 * s["rms_rel"], **kw)
            ax[1].scatter(s["speedup_vs_mc"], s["chi2_dof"], **kw)
    for x in ax:
        x.set_xscale("log")
        x.set_xlabel("speed-up vs analog MC (whole problem)")
        x.axvline(1, color="0.6", lw=0.8)
    ax[0].set_ylabel("rms of GMC / MC − 1 over the scored bins, %")
    ax[1].set_ylabel("χ²/dof of (GMC − MC) / σ")
    ax[1].set_yscale("log")
    ax[1].axhline(1, color="0.6", lw=0.8)
    h = [plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in FAM.values()]
    h += [plt.Line2D([], [], ls="", marker=m, color="0.3", mfc="0.3", label=f"{a} {b}°") for (a, b), m in mk.items()]
    ax[0].legend(handles=h, fontsize=8)
    ax[0].set_title("Shape error vs speed (size model: K8; dedicated model: auto K)", fontsize=10)
    ax[1].set_title("Within noise? (χ²/dof ≈ 1)", fontsize=10)
    fig.savefig(png, dpi=125)
    plt.close(fig)
    print(f"wrote {png}")


def timing(data, png, plt, kset="k8"):
    keys = [k for k in load_cases() if k in data]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True,
                           gridspec_kw={"width_ratios": [2.2, 1, 1]})
    order = ["old", "uniform", "random", "one"]
    w = 0.13
    for i, key in enumerate(keys):
        runs = [r for r in data[key] if r["stats"]["kset"] in (kset, "old")]
        if not runs:
            continue
        mc_ms = runs[0]["stats"]["mc_ms_per_primary"]
        ax[0].bar(i - 3 * w, mc_ms, w, color="0.15")
        rs = sorted(runs, key=lambda r: (order.index(r["stats"]["family"]), r["stats"]["mesh"]))
        for j, r in enumerate(rs[:6]):
            s = r["stats"]
            ax[0].bar(i + (j - 2) * w, s["gmc_ms_per_primary"], w, color=FAM[s["family"]][0],
                      alpha=1.0 if s["mesh"] != "random:2" and s["mesh"] != "random:3" else 0.6)
    ax[0].set_xticks(range(len(keys)))
    ax[0].set_xticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("ms per primary")
    h = [plt.Rectangle((0, 0), 1, 1, color="0.15", label="analog MC")]
    h += [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in FAM.values()]
    ax[0].legend(handles=h, fontsize=8, ncol=2)
    ax[0].set_title("Cost per primary (K8 runs; random: 3 bars)", fontsize=10)
    for key, runs in data.items():
        for r in runs:
            s = r["stats"]
            if s["kset"] not in (kset, "old"):
                continue
            col = FAM[s["family"]][0]
            ax[1].scatter(s["transits_per_primary"], s["chain_ms_per_primary"], color=col, s=25)
            ax[2].scatter(s["transits_per_primary"], s["tally_ms_per_primary"], color=col, s=25)
    for x, t in zip(ax[1:], ("GMC chain (flows)", "SCR tally")):
        x.set_xscale("log")
        x.set_yscale("log")
        x.set_xlabel("cell transits per primary")
        x.set_ylabel("ms per primary")
        x.set_title(t, fontsize=10)
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def meshes(data, png, plt, key="al_1.0_0", kset="k8"):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    runs = [r for r in data.get(key, []) if r["stats"]["kset"] == kset]
    runs = sorted(runs, key=lambda r: (["one", "random", "uniform"].index(r["stats"]["family"]), r["stats"]["mesh"]))
    fig, ax = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 4.2), constrained_layout=True, squeeze=False)
    for a, r in zip(ax[0], runs):
        z = r["z"]
        L = np.asarray(z["L"])
        lo, hi = np.log10(0.005), np.log10(0.6)        # the model range, 50 um - 6 mm
        cm = plt.get_cmap("viridis")
        pc = [Rectangle((x, y), l, l) for x, y, l in zip(z["x0"], z["y0"], L)]
        cols = [cm((np.log10(l) - lo) / max(hi - lo, 1e-9)) for l in L]
        a.add_collection(PatchCollection(pc, facecolor=cols, edgecolor="white", lw=0.3))
        W, H = z["box"]
        a.annotate("", (0.03, 0.5 * H), (-0.02, 0.5 * H), annotation_clip=False,
                   arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=1.5))
        a.set_xlim(0, W)
        a.set_ylim(0, H)
        a.set_aspect("equal")
        a.grid(False)
        s = r["stats"]
        a.set_title(f"{s['mesh']}: {s['n_cells']} cell{'s' if s['n_cells'] > 1 else ''}\n{s['L_min_um']:.0f}–{s['L_max_um']:.0f} µm, "
                    f"{s['transits_per_primary']:.1f} transits / e⁻", fontsize=9)
        a.set_xlabel("depth (cm)")
    ax[0, 0].set_ylabel("y (cm)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.matplotlib.colors.LogNorm(50, 6000))
    fig.colorbar(sm, ax=ax[0, -1], label="cell side (µm)")
    fig.suptitle(f"Meshes, {key.replace('_', ' ')}° (beam in red)", fontweight="bold")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(OUT / "writeup"))
    ap.add_argument("--n_bins", type=int, default=25)
    ap.add_argument("--min_frac", type=float, default=0.1)
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(a.n_bins, a.min_frac)
    for kset in ("auto", "k8"):
        if any(r["stats"]["kset"] == kset for rs in data.values() for r in rs):
            ratio_grid(data, kset, out / f"ratio_grid_{kset}.png", plt)
    speed_accuracy(data, out / "speed_accuracy.png", plt)
    ks = "k8" if any(r["stats"]["kset"] == "k8" for rs in data.values() for r in rs) else "auto"
    timing(data, out / "timing.png", plt, ks)
    meshes(data, out / "meshes.png", plt, kset=ks)
    summ = {k: {r["file"]: r["stats"] for r in rs} for k, rs in data.items()}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    # per-family aggregate: median over cases
    print(f"{'kset':>5} {'family':>8} {'n':>3} {'med rms %':>9} {'max rms %':>9} {'med chi2':>8} {'med speedup':>11} "
          f"{'|tot-1| max %':>13}")
    for kset in ("auto", "k8", "old"):
        for fam in ("old", "uniform", "random", "one"):
            st = [r["stats"] for rs in data.values() for r in rs if r["stats"]["kset"] == kset
                  and r["stats"]["family"] == fam]
            if not st:
                continue
            rms = np.array([s["rms_rel"] for s in st])
            print(f"{kset:>5} {fam:>8} {len(st):3d} {100 * np.median(rms):9.2f} {100 * rms.max():9.2f} "
                  f"{np.median([s['chi2_dof'] for s in st]):8.2f} {np.median([s['speedup_vs_mc'] for s in st]):11.2f} "
                  f"{100 * max(abs(s['total_ratio'] - 1) for s in st):13.2f}")


if __name__ == "__main__":
    main()
