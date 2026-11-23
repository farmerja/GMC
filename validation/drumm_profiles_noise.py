#!/usr/bin/env python3
"""
GMC + SCR vs analog MC profiles on the Drumm cylinder with the statistical noise band: along the beam axis
(|y| < 0.5 mm) and across the beam at x = -0.5, 0, +0.5 cm, on 0.5 mm bins (the drumm_noise_check binning). Top row:
MC (seed mean, with its 2-sigma band) and each GMC run; bottom row: (GMC - MC) / MC with the combined 2-sigma band
(GMC + MC seed noise) of the first run shaded. Points inside the band are within noise.

    .venv/bin/python validation/drumm_profiles_noise.py --runs a.npz b.npz ... --png out.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drumm_quadtree import load_mc, coarse, H_OUT  # noqa: E402
from drumm_noise_check import seed_var, gmc_map, K  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c9a227", "#e34948")


def cut(A, which, pos):
    """Profile of a 0.5 mm map (and its variance, summed then divided by 4 for the 2-bin mean)."""
    n = A.shape[0]
    h = H_OUT * K
    if which == "along":
        return A[:, n // 2 - 1:n // 2 + 1].mean(1)
    i = int(np.floor((pos + 1.0) / h))
    return A[i, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--png", required=True)
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    mc = load_mc()
    M, vM = seed_var([coarse(d, K) for d in mc["loc"]])
    n = M.shape[0]
    g = (np.arange(n) + 0.5) * H_OUT * K - 1.0
    runs = [(p, *gmc_map(p)) for p in a.runs]
    labels = a.labels or [json.loads(str(np.load(p)["meta"]))["mesh"] for p in a.runs]
    panels = [("along", 0.0, "along the beam axis, |y| < 0.5 mm", "x (cm)"),
              ("across", -0.5, "across the beam at x = -0.5 cm", "y (cm)"),
              ("across", 0.0, "across the beam at x = 0", "y (cm)"),
              ("across", 0.5, "across the beam at x = +0.5 cm", "y (cm)")]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7.6), constrained_layout=True, sharex="col",
                           gridspec_kw={"height_ratios": [1.4, 1]})
    for j, (which, pos, title, xl) in enumerate(panels):
        # the along cut is the mean of 2 rows: its variance is (v1 + v2) / 4
        div = 2.0 if which == "along" else 1.0
        m, vm = cut(M, which, pos), cut(vM, which, pos) / div
        inside = np.abs(g) < np.sqrt(np.maximum(1.0 - (pos if which == "across" else 0.0) ** 2, 0)) - 0.025
        if which == "along":
            inside = np.abs(g) < 0.975
        keep = inside & (m > 0.01 * M.max())
        a0, a1 = ax[0, j], ax[1, j]
        a0.fill_between(g[keep], (m - 2 * np.sqrt(vm))[keep], (m + 2 * np.sqrt(vm))[keep], color="0.75", lw=0,
                        label="MC ± 2σ")
        a0.plot(g[keep], m[keep], color="0.15", lw=1.6, label="analog MC")
        for k, (p, G, vG) in enumerate(runs):
            gp, vg = cut(G, which, pos), cut(vG, which, pos) / div
            c = COLORS[k % len(COLORS)]
            a0.plot(g[keep], gp[keep], color=c, lw=1.2, label=labels[k])
            rel = (gp - m) / np.maximum(m, 1e-300)
            band = 2 * np.sqrt(vg + vm) / np.maximum(m, 1e-300)
            if k == 0:
                a1.fill_between(g[keep], -band[keep], band[keep], color=c, alpha=0.18, lw=0,
                                label=f"combined ± 2σ ({labels[0]})")
            a1.plot(g[keep], rel[keep], "o-", color=c, lw=1.0, ms=3)
        a1.axhline(0, color="0.4", lw=0.8)
        a0.set_title(title, fontsize=10)
        a0.set_yscale("log" if which == "across" else "linear")
        a1.set_xlabel(xl)
        a1.set_ylim(-0.15, 0.15)
        a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:+g} %"))
        if j == 0:
            a0.set_ylabel("dose (MeV/cm³ per e⁻/cm)")
            a1.set_ylabel("(GMC - MC) / MC")
            a0.legend(fontsize=8)
            a1.legend(fontsize=8, loc="lower left")
    fig.suptitle("Drumm cylinder, 0.5 mm bins: GMC + SCR vs analog MC with the statistical noise band",
                 fontweight="bold")
    fig.savefig(a.png, dpi=115)
    print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
