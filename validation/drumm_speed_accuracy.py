#!/usr/bin/env python3
"""
Speed vs accuracy of the Drumm meshes for the write-up: speed-up vs analog MC against (left) the noise-removed map bias
on 0.5 mm bins and (right) chi2/dof of the per-bin noise check away from the circle edge (r < 0.9 cm), with the
GMC-vs-GMC null-test value as the "within noise" line. One marker per mesh; filled = _sx3 model, open = _sx2.

    .venv/bin/python validation/drumm_speed_accuracy.py --summary .../summary.json --noise .../noise_check.json --png ...
"""
import argparse
import json

import numpy as np

COL = {"uniform": "#7b7b7b", "quad": "#2a78d6", "centre": "#8a5cd6", "sym": "#1baf7a", "min": "#eb6834"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="drumm_quadtree compare summary.json")
    ap.add_argument("--noise", required=True, help="drumm_noise_check noise_check.json (r < 0.9 cm)")
    ap.add_argument("--png", required=True)
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    sm = json.load(open(a.summary))["runs"]
    nz = json.load(open(a.noise))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, s in sm.items():
        mesh = s["mesh"]
        sx3 = "_sx3" in name
        c = COL[mesh.split(":")[0]]
        sp = s["timing"]["speedup_vs_mc"]
        kw = dict(s=70, color=c, edgecolor=c, lw=1.5, zorder=3) if sx3 else dict(s=70, facecolor="none",
                                                                                 edgecolor=c, lw=1.5, zorder=2)
        ax[0].scatter(sp, 100 * s["maps"]["loc"]["bins_500um"]["bias"], **kw)
        if name in nz["runs"]:
            ax[1].scatter(sp, nz["runs"][name]["chi2_dof"], **kw)
        if sx3:
            ax[0].annotate(mesh, (sp, 100 * s["maps"]["loc"]["bins_500um"]["bias"]), xytext=(6, 4),
                           textcoords="offset points", fontsize=8, color="0.25")
            if name in nz["runs"]:
                ax[1].annotate(mesh, (sp, nz["runs"][name]["chi2_dof"]), xytext=(6, 4), textcoords="offset points",
                               fontsize=8, color="0.25")
    if nz.get("null"):
        ax[1].axhline(nz["null"]["chi2_dof"], color="0.3", lw=1, ls="--")
        ax[1].text(49, nz["null"]["chi2_dof"] * 0.955, "within noise (GMC vs GMC null test)", fontsize=8, color="0.3",
                   ha="right")
    ax[1].axhline(1.0, color="0.6", lw=0.8)
    for x in ax:
        x.set_xscale("log")
        x.set_xlabel("speed-up vs analog MC (whole problem)")
        x.set_xlim(5, 50)
    ax[0].set_ylabel("map bias vs MC on 0.5 mm bins, % (noise removed)")
    ax[0].set_title("Bias of the dose map (filled: _sx3, open: _sx2)", fontsize=10)
    ax[1].set_ylabel("χ²/dof of (GMC − MC) / σ, 0.5 mm bins, r < 0.9 cm")
    ax[1].set_title("Is it within noise? (filled: _sx3 only)", fontsize=10)
    ax[1].set_yscale("log")
    ax[1].set_yticks([1, 1.5, 2, 3])
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.savefig(a.png, dpi=130)
    print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
