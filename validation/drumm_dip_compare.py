#!/usr/bin/env python3
"""
Drumm benchmark, like for like (1 mm staircase, photons escape): on-axis dip and within-cell row pattern of GMC + SCR
runs vs the 2-seed analog MC staircase mean. Writes a figure and prints the metrics.

Usage (GMC venv, GMC root):
    python validation/drumm_dip_compare.py --gmc "current=gmc_n8_500k_elsepa.npz" "new=gmc_n8_200k_..._mirror.npz" \
        --out validation/results/drumm_water_cylinder/phaseE/dip_compare.png
"""
import argparse
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "drumm_water_cylinder")
MC = ("mc_stair_n8_500k_s123_elsepa_split3.npz", "mc_stair_n8_500k_s1123_elsepa_split3.npz")


def metrics(G, M):
    sl = slice(8 * 2, 8 * 10)                                   # depth cells x = -8 .. 0 mm
    r = G[sl, 64:96].sum(0) / M[sl, 64:96].sum(0)               # 4 cell rows, y = -2 .. 2 mm
    fold = r.reshape(4, 8).mean(0)
    a_g = G[:, 79:81].mean(1).reshape(-1, 4).mean(1)            # 0.25 mm strip on y = 0, 0.5 mm bins in x
    a_m = M[:, 79:81].mean(1).reshape(-1, 4).mean(1)
    x = (np.arange(40) + 0.5) * 0.05 - 1.0
    sel = (x > -0.95) & (x < 0.95)
    ax = a_g[sel] / a_m[sel]
    m = M > 0.01 * M.max()
    cg, cm = G.reshape(20, 8, 20, 8).sum((1, 3)), M.reshape(20, 8, 20, 8).sum((1, 3))
    mm = cm > 0.01 * cm.max()
    return {"row_pattern": fold.tolist(), "row_rms": float(np.sqrt(np.mean((fold / fold.mean() - 1) ** 2))),
            "axis_x": x[sel].tolist(), "axis_ratio": ax.tolist(), "axis_mean": float(ax.mean()),
            "axis_min": float(ax.min()), "axis_rms": float(np.sqrt(np.mean((ax - 1) ** 2))),
            "fine_relL2": float(np.linalg.norm((G - M)[m]) / np.linalg.norm(M[m])),
            "cell_relL2": float(np.linalg.norm((cg - cm)[mm]) / np.linalg.norm(cm[mm]))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmc", nargs="+", required=True, help="label=file.npz (in the results folder)")
    p.add_argument("--out", required=True)
    p.add_argument("--mc", nargs=2, default=MC, help="the two staircase MC references (in the results folder); "
                   "default the ELSEPA-split ones (EEDL stopping). Use the *_estar pair for models trained on ESTAR data")
    a = p.parse_args()
    mc = [np.load(os.path.join(D, f))["energy"] / float(np.load(os.path.join(D, f))["N"]) for f in a.mc]  # per primary
    M = (mc[0] + mc[1]) / 2
    res = {"mc_seed_vs_seed": metrics(mc[0], mc[1])}
    for s in a.gmc:
        lab, f = s.split("=", 1)
        g = np.load(os.path.join(D, f), allow_pickle=True)
        res[lab] = {"file": f, "N": int(g["N"]), "seeds": int(g["scr"].shape[0]), **metrics(g["scr"].mean(0) / float(g["N"]), M)}
    for k, v in res.items():
        print(f"{k:<28} row-pattern rms {v['row_rms']:.4f}  axis mean {v['axis_mean']:.4f} min {v['axis_min']:.4f} "
              f"rms {v['axis_rms']:.4f}  fine relL2 {v['fine_relL2']:.4f}  cell relL2 {v['cell_relL2']:.4f}")
        print(f"{'':<28} row pattern {np.round(v['row_pattern'], 3)}")

    import matplotlib
    matplotlib.use("Agg")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, v in res.items():
        style = dict(color="0.55", linestyle="--") if k == "mc_seed_vs_seed" else {}
        lab = "MC seed 1 / seed 2 (noise)" if k == "mc_seed_vs_seed" else f"{k} ({v['N'] // 1000}k x {v['seeds']})"
        ax[0].plot(v["axis_x"], v["axis_ratio"], marker="o", ms=3, label=lab, **style)
        f = np.array(v["row_pattern"])
        ax[1].step(np.arange(9) / 8, np.r_[f, f[-1]], where="post", label=lab, **style)
    ax[0].axhline(1, color="0.5", lw=0.8)
    ax[0].set(xlabel="x (cm)", ylabel="GMC + SCR / analog MC", title="Energy on the beam axis (0.25 mm strip at y = 0)")
    ax[1].axhline(1, color="0.5", lw=0.8)
    ax[1].set(xlabel="position inside the 1 mm cell (y, cell units)", ylabel="GMC + SCR / analog MC",
              title="Within-cell row pattern (4 cells around the axis, x = -8..0 mm)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    with open(a.out[:-4] + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
