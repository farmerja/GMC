#!/usr/bin/env python3
"""
Write-up figures for the size-conditioned Al / Cu GMC + SCR on the Lockwood / Sandia depth-dose cases
(validation/lockwood_quadtree.py runs): the fewest-cells mesh (one cell), random-size tilings, the uniform 100 um
mesh and the old dedicated 100 um model (lockwood_slab.py), all vs the ESTAR analog MC.

  ratio_grid<sfx>.png      GMC / MC vs depth per case, MC 2-sigma band shaded
  speed_accuracy.png       speed-up vs MC against rms shape error and chi2/dof (K8 runs + the dedicated model)
  timing.png               ms per primary: MC vs each mesh per case; chain cost vs transits; chain / tally split
  t1_vs_k8.png             targeted fine-tunes (_t1) vs base models, rms and chi2/dof per case and mesh
  meshes.png               the meshes of the Al 1 MeV case
  dose_density_<case>.png  GMC cell dose density on each mesh + depth-dose: MC vs cell level vs GMC + SCR
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


def collect(n_bins, min_frac, materials=None):
    cases = load_cases()
    files = {"auto": sorted(glob.glob(str(OUT / "gmc_*_200k.npz"))),
             "k8": sorted(glob.glob(str(OUT / "gmc_*_200k_k8.npz"))),
             # targeted fine-tunes (_t1 GMC + SCR); a _k8_t1q rerun (same settings, idle machine) replaces the timing-
             # contaminated _k8_t1 run of the same case and mesh
             "t1": sorted(f if not Path(f.replace("_k8_t1.npz", "_k8_t1q.npz")).exists()
                          else f.replace("_k8_t1.npz", "_k8_t1q.npz")
                          for f in glob.glob(str(OUT / "gmc_*_200k_k8_t1.npz"))),
             "old": sorted(glob.glob(str(LW_OUT / "gmc_*_200k_n20_estar.npz")))}
    out = {}
    for kset, fl in files.items():
        for f in fl:
            z, meta = load_gmc(f)
            key = meta["case"]
            c = cases[key]
            if materials and c["material"] not in materials:
                continue
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
    ktxt = {"k8": "K = max(8, 0.25 × auto)", "t1": "targeted fine-tunes (_t1), K = max(8, 0.25 × auto)"}.get(
        kset, "K = max(32, auto)")
    fig.suptitle(f"Lockwood depth-dose, size-conditioned GMC + SCR vs analog MC (ESTAR), {ktxt}; bins with "
                 f"J > 10 % of max", fontweight="bold")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def t1_vs_k8(data, png, plt):
    """Targeted fine-tunes vs the base models, per case and mesh: rms shape error and chi2/dof, _t1 against K8."""
    pairs = []
    for key, rs in data.items():
        by = {}
        for r in rs:
            by.setdefault(r["stats"]["mesh"], {})[r["stats"]["kset"]] = r["stats"]
        pairs += [(v["k8"], v["t1"]) for v in by.values() if "k8" in v and "t1" in v]
    if not pairs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, (f, lab, sc) in zip(axs, ((lambda s: 100 * s["rms_rel"], "rms shape error (%)", "linear"),
                                      (lambda s: s["chi2_dof"], "χ²/dof", "log"))):
        for fam in ("uniform", "random", "one"):
            for mat, mk in (("Al", "o"), ("Cu", "s")):
                pp = [(f(a), f(b)) for a, b in pairs if a["family"] == fam and a["material"] == mat]
                if pp:
                    x, y = np.array(pp).T
                    ax.plot(x, y, mk, color=FAM[fam][0], ms=6, mec=FAM[fam][0],
                            mfc=FAM[fam][0] if mat == "Cu" else "none", mew=1.4, ls="none",
                            label=f"{mat}, {FAM[fam][1].replace('size model, ', '')}")
        lo = min(min(f(a), f(b)) for a, b in pairs) * 0.9
        hi = max(max(f(a), f(b)) for a, b in pairs) * 1.1
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
        ax.set_xscale(sc)
        ax.set_yscale(sc)
        if sc == "log":
            from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
            t = [v for v in (1, 2, 3, 5, 10, 20, 30, 50, 100) if lo <= v <= hi]
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator(t))
                axis.set_minor_locator(NullLocator())
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(f"{lab}, base models (K8)")
        ax.set_ylabel(f"{lab}, targeted fine-tunes (_t1)")
        ax.set_title(f"{lab}: below the diagonal = better after fine-tuning", fontsize=10)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Lockwood: targeted Al / Cu fine-tunes vs base models, one point per case and mesh "
                 "(open = Al, filled = Cu)", fontweight="bold")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def speed_accuracy(data, png, plt, mk=None, title="Shape error vs speed (size model: K8; dedicated model: auto K)"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    mk = mk or {("Al", 0): "o", ("Al", 60): "^", ("Cu", 0): "s"}
    for key, runs in data.items():
        for r in runs:
            s = r["stats"]
            col, _ = FAM[s["family"]]
            m = mk[(s["material"], int(s["angle"]))]
            if s["kset"] in ("auto", "t1"):
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
    ax[0].set_title(title, fontsize=10)
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


def dose_density(data, png, plt, key="al_1.0_0", kset="k8", n_bins=25):
    """Per mesh: row 1 the GMC cell dose density (cell energy / cell area, log scale, relative to the row maximum) on
    the mesh; row 2 the depth-dose J vs depth / R: analog MC, the GMC cell-level dose (each cell's energy spread
    evenly over its depth extent) and GMC + SCR (the sub-cell reconstruction actually scored)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    from lockwood_quadtree import coarse_edges
    import sec_plot_style as S
    runs = [r for r in data.get(key, []) if r["stats"]["kset"] == kset]
    runs = sorted(runs, key=lambda r: (["one", "random", "uniform"].index(r["stats"]["family"]), r["stats"]["mesh"]))
    if not runs:
        return
    dens = [np.asarray(r["z"]["cell"]).mean(0) / np.asarray(r["z"]["L"]) ** 2 for r in runs]
    vmax = max(d.max() for d in dens)
    norm = LogNorm(vmax * 1e-3, vmax)
    fig, ax = plt.subplots(2, len(runs), figsize=(3.3 * len(runs), 7.4), constrained_layout=True, squeeze=False,
                           gridspec_kw={"height_ratios": [1.15, 1]})
    for j, (r, d) in enumerate(zip(runs, dens)):
        z, s = r["z"], r["stats"]
        rho = float(z["rho"])
        R_g = r["meta"]["R_cm"] * rho / 1.1
        a = ax[0, j]
        pc = [Rectangle((x, y), l, l) for x, y, l in zip(z["x0"], z["y0"], z["L"])]
        a.add_collection(PatchCollection(pc, facecolor=S.SEQ(norm(np.maximum(d, vmax * 1e-3))),
                                         edgecolor="white", lw=0.25))
        W, H = z["box"]
        a.set_xlim(0, W)
        a.set_ylim(0, H)
        a.set_aspect("equal")
        a.grid(False)
        a.set_title(f"{s['mesh']}: {s['n_cells']} cell{'s' if s['n_cells'] > 1 else ''}", fontsize=9)
        a.set_xlabel("depth (cm)")
        # depth-dose: cell-level (energy spread evenly over each cell's depth extent) vs SCR vs MC
        ed, _ = coarse_edges(z["edges"], R_g, n_bins)
        ed = ed[:len(r["xm"]) + 1]
        cl = np.zeros(len(ed) - 1)
        E = np.asarray(z["cell"]).mean(0)
        for x0, l, e in zip(np.asarray(z["x0"]) * rho, np.asarray(z["L"]) * rho, E):
            ov = np.clip(np.minimum(ed[1:], x0 + l) - np.maximum(ed[:-1], x0), 0, None)
            cl += e * ov / l
        cl /= np.diff(ed)
        b = ax[1, j]
        b.plot(r["xm"], r["m"], color="0.15", lw=2, label="analog MC")
        b.stairs(cl, ed / R_g, color="#eb6834", lw=1.2, label="GMC, cell level")
        b.plot(r["xm"], r["g"], color="#2a78d6", lw=1.5, ls="--", label="GMC + SCR")
        b.set_xlabel("depth / R")
        b.set_title(f"rms {100 * s['rms_rel']:.1f} %, {s['speedup_vs_mc']:.1f}× MC", fontsize=9)
    ax[1, 0].set_ylabel("J (MeV cm² / g per e⁻)")
    ax[0, 0].set_ylabel("y (cm)")
    ax[1, 0].legend(fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=S.SEQ, norm=LogNorm(1e-3, 1))
    fig.colorbar(sm, ax=ax[0, -1], label="cell energy / cell area (rel. to max)")
    fig.suptitle(f"Dose density, {key.replace('_', ' ')}° ({kset}): GMC cell dose on each mesh, and the depth-dose "
                 "before / after the sub-cell reconstruction", fontweight="bold")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(OUT / "writeup"))
    ap.add_argument("--n_bins", type=int, default=25)
    ap.add_argument("--min_frac", type=float, default=0.1)
    ap.add_argument("--materials", nargs="+", default=["Al", "Cu"],
                    help="cases of these materials only (Ta has its own write-up, lockwood_ta_writeup.py)")
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(a.n_bins, a.min_frac, a.materials)
    for kset in ("auto", "k8", "t1"):
        if any(r["stats"]["kset"] == kset for rs in data.values() for r in rs):
            ratio_grid(data, kset, out / f"ratio_grid_{kset}.png", plt)
    speed_accuracy(data, out / "speed_accuracy.png", plt)
    t1_vs_k8(data, out / "t1_vs_k8.png", plt)
    ks = "k8" if any(r["stats"]["kset"] == "k8" for rs in data.values() for r in rs) else "auto"
    timing(data, out / "timing.png", plt, ks)
    meshes(data, out / "meshes.png", plt, kset=ks)
    for key in ("al_1.0_0", "al_1.0_60", "cu_0.5_0"):
        dose_density(data, out / f"dose_density_{key}.png", plt, key=key, kset=ks, n_bins=a.n_bins)
    summ = {k: {r["file"]: r["stats"] for r in rs} for k, rs in data.items()}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    # per-family aggregate: median over cases
    print(f"{'kset':>5} {'family':>8} {'n':>3} {'med rms %':>9} {'max rms %':>9} {'med chi2':>8} {'med speedup':>11} "
          f"{'|tot-1| max %':>13}")
    for kset in ("auto", "k8", "t1", "old"):
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
