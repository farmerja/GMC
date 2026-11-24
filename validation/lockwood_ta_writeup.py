#!/usr/bin/env python3
"""
Write-up figures for the size-conditioned Ta GMC + SCR (cells 10 um - 1 cm, E 1 keV - 20 MeV) on the Lockwood /
Sandia semi-infinite tantalum cases (SAND79-0414 Tables V.G.1-6), vs the analog MC (ESTAR), the measurements and
the paper's TIGER calculations. Reuses the Al / Cu write-up functions (lockwood_size_writeup.py) on the Ta cases.

  depth_dose_vs_exp.png    per case: measurement (Method A, and Method B where taken), TIGER, analog MC, GMC + SCR
                           on the one-cell, random:3 and uniform 20 um meshes; lower panel: / measurement
  ratio_grid_k8.png        GMC / MC vs depth per case, MC 2-sigma band
  speed_accuracy.png       speed-up vs MC against rms shape error and chi2/dof
  timing.png               ms per primary per case and mesh; chain and tally cost vs transits
  meshes.png               the meshes of the 1 MeV 0 deg case
  dose_density_<case>.png  GMC cell dose density on each mesh + depth-dose before / after the SCR
  summary.json             every scored run (vs MC) and every model / MC / TIGER comparison with the measurement

    .venv/bin/python validation/lockwood_ta_writeup.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lockwood_size_writeup as W  # noqa: E402
from lockwood_quadtree import OUT  # noqa: E402
from lockwood_slab import load_cases, find_run, tiger_range  # noqa: E402

KEYS = ["ta_1.0_0", "ta_1.0_60", "ta_0.5_0", "ta_0.5_30", "ta_0.5_60", "ta_0.3_0"]
SHOW = {"one": ("#eb6834", "-"), "random:3": ("#1baf7a", "-"), "uniform:20": ("#2a78d6", "--")}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def window(edges, J, lo, hi):
    """Mean of a histogram J (per unit of edges) over [lo, hi]."""
    cum = np.r_[0.0, np.cumsum(J * np.diff(edges))]
    return (np.interp(hi, edges, cum) - np.interp(lo, edges, cum)) / (hi - lo)


def vs_exp(data, cases):
    """Mean / rms offset from the Method A measurement (calorimeter-averaged) for every GMC run, the MC and TIGER."""
    out = {}
    for k in KEYS:
        c = cases[k]
        t, R = c["calorimeter"], c["csda_range_exp"]
        x = np.array(c["exp"])
        lo, hi = np.maximum(x[:, 0] * R - t / 2, 0), x[:, 0] * R + t / 2
        J = x[:, 1]
        mask = J > 0.1 * J.max()
        mc = find_run(k, c["E_exp"], 200000, 79, "estar")
        vm, _ = mc.window("escape", lo, hi)
        th = np.array(c["theory"])
        th_e = np.r_[0.0, th[:, 0]]
        tg = window(th_e, th[:, 1], lo / R, hi / R)
        o = {"exp_total": c["exp_total"], "tiger_total": c["theory_total"], "mc_total": mc.total("escape")[0],
             "mc": [rms(vm[mask] / J[mask] - 1), float(np.mean(vm[mask] / J[mask] - 1))],
             "tiger": [rms(tg[mask] / J[mask] - 1), float(np.mean(tg[mask] / J[mask] - 1))], "gmc": {}}
        Rt = tiger_range(c)
        th_lo = np.r_[0.0, th[:-1, 0]]
        vt, _ = mc.window("escape", th_lo * Rt, th[:, 0] * Rt)
        tm = th[:, 1] > 0.1 * th[:, 1].max()
        o["mc_vs_tiger"] = [rms(vt[tm] / th[tm, 1] - 1), float(np.mean(vt[tm] / th[tm, 1] - 1))]
        for r in data.get(k, []):
            if r["stats"]["kset"] != "k8":
                continue
            z = r["z"]
            P = np.asarray(z["prof"]).mean(0)
            ed = np.asarray(z["edges"])
            vg = window(ed, P / np.diff(ed), lo, hi)
            o["gmc"][r["stats"]["mesh"]] = {"total": float(P.sum()),
                                            "vs_exp": [rms(vg[mask] / J[mask] - 1), float(np.mean(vg[mask] / J[mask] - 1))]}
        out[k] = o
    return out


def depth_dose(data, cases, png, plt):
    fig, axs = plt.subplots(4, 3, figsize=(15, 12.5), constrained_layout=True,
                            gridspec_kw={"height_ratios": [2.2, 1, 2.2, 1]})
    for i, k in enumerate(KEYS):
        c = cases[k]
        a, b = axs[2 * (i // 3), i % 3], axs[2 * (i // 3) + 1, i % 3]
        R, t = c["csda_range_exp"], c["calorimeter"]
        th = np.array(c["theory"])
        a.stairs(th[:, 1], np.r_[0.0, th[:, 0]], color="0.55", lw=1.2, label="TIGER (paper)")
        mc = find_run(k, c["E_exp"], 200000, 79, "estar")
        f = np.linspace(0, 0.75, 76)
        vm, em = mc.window("escape", f[:-1] * R, f[1:] * R)
        xm = 0.5 * (f[1:] + f[:-1])
        a.plot(xm, vm, color="0.1", lw=1.8, label="analog MC (ESTAR)")
        x = np.array(c["exp"])
        lo, hi = np.maximum(x[:, 0] * R - t / 2, 0), x[:, 0] * R + t / 2
        a.errorbar(x[:, 0], x[:, 1], yerr=c["exp_uncertainty"] * x[:, 1], fmt="o", color="#d6336c", ms=5, zorder=5,
                   label="measured, Method A")
        if "exp_b" in c:
            xb = np.array(c["exp_b"]["data"])
            a.plot(xb[:, 0], xb[:, 1], "o", mfc="none", mec="#d6336c", ms=5, zorder=5,
                   label=f"measured, Method B ({c['exp_b']['E']:g} MeV)")
        runs = {r["stats"]["mesh"]: r for r in data.get(k, []) if r["stats"]["kset"] == "k8"}
        J = x[:, 1]
        vmx, _ = mc.window("escape", lo, hi)
        b.plot(x[:, 0], vmx / J, "o-", color="0.1", ms=3, lw=1.2)
        thv = window(np.r_[0.0, th[:, 0]], th[:, 1], lo / R, hi / R)
        b.plot(x[:, 0], thv / J, "s-", color="0.55", ms=3, lw=1.0)
        for mesh, (col, ls) in SHOW.items():
            r = runs.get(mesh)
            if r is None:
                continue
            z = r["z"]
            P = np.asarray(z["prof"]).mean(0)
            ed = np.asarray(z["edges"])
            Jg = P / np.diff(ed)
            a.plot(0.5 * (ed[1:] + ed[:-1]) / R, Jg, color=col, ls=ls, lw=1.3,
                   label=f"GMC + SCR, {mesh} ({r['stats']['speedup_vs_mc']:.0f}× MC)")
            b.plot(x[:, 0], window(ed, Jg, lo, hi) / J, color=col, ls=ls, lw=1.3, marker=".", ms=4)
        b.fill_between([0, 0.75], 1 - c["exp_uncertainty"], 1 + c["exp_uncertainty"], color="#d6336c", alpha=0.12, lw=0)
        b.axhline(1, color="#d6336c", lw=0.8)
        a.set_xlim(0, 0.6)
        b.set_xlim(0, 0.6)
        b.set_ylim(0.6, 1.2)
        b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * (v - 1):+.0f} %"))
        a.set_title(f"Ta {c['E_nom']:g} MeV, {c['angle']}°", fontsize=11)
        b.set_xlabel("depth / CSDA range")
        if i % 3 == 0:
            a.set_ylabel("J (MeV cm² / g per e⁻)")
            b.set_ylabel("/ measured − 1")
        if i == 0:
            a.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Semi-infinite Ta (Lockwood et al., SAND79-0414): Ta size model GMC + SCR vs analog MC, TIGER and "
                 "measurement\nlower panels: calorimeter-averaged / measured − 1 (band: 2.2 % uncertainty)",
                 fontweight="bold", fontsize=11)
    fig.savefig(png, dpi=115)
    plt.close(fig)
    print(f"wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(OUT / "writeup_ta"))
    ap.add_argument("--n_bins", type=int, default=25)
    ap.add_argument("--min_frac", type=float, default=0.1)
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    # Ta legends: uniform mesh is 20 um, and there is no dedicated fixed-size Ta model
    W.FAM = {"uniform": (W.FAM["uniform"][0], "size model, uniform 20 µm"),
             "random": (W.FAM["random"][0], "size model, random sizes (10 µm lattice)"),
             "one": (W.FAM["one"][0], "size model, one cell")}
    data = W.collect(a.n_bins, a.min_frac, ["Ta"])
    depth_dose(data, cases, out / "depth_dose_vs_exp.png", plt)
    W.ratio_grid(data, "k8", out / "ratio_grid_k8.png", plt)
    W.speed_accuracy(data, out / "speed_accuracy.png", plt, mk={("Ta", 0): "o", ("Ta", 30): "D", ("Ta", 60): "^"},
                     title="Ta: shape error vs speed (K8)")
    W.timing(data, out / "timing.png", plt, "k8")
    W.meshes(data, out / "meshes.png", plt, key="ta_1.0_0", kset="k8")
    for key in ("ta_1.0_0", "ta_0.5_60"):
        W.dose_density(data, out / f"dose_density_{key}.png", plt, key=key, kset="k8", n_bins=a.n_bins)
    ve = vs_exp(data, cases)
    summ = {"vs_mc": {k: {r["file"]: r["stats"] for r in rs} for k, rs in data.items()}, "vs_exp": ve}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(f"{'family':>8} {'n':>3} {'med rms %':>9} {'max rms %':>9} {'med chi2':>8} {'med speedup':>11} {'|tot-1| max %':>13}")
    for fam in ("uniform", "random", "one"):
        st = [r["stats"] for rs in data.values() for r in rs if r["stats"]["kset"] == "k8" and r["stats"]["family"] == fam]
        if st:
            rr = np.array([s["rms_rel"] for s in st])
            print(f"{fam:>8} {len(st):3d} {100 * np.median(rr):9.2f} {100 * rr.max():9.2f} "
                  f"{np.median([s['chi2_dof'] for s in st]):8.2f} {np.median([s['speedup_vs_mc'] for s in st]):11.2f} "
                  f"{100 * max(abs(s['total_ratio'] - 1) for s in st):13.2f}")


if __name__ == "__main__":
    main()
