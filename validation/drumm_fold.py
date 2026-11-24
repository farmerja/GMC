#!/usr/bin/env python3
"""
In-cell dose pattern of GMC + SCR on the uniform Drumm meshes: the 125 um GMC / MC ratio map folded by position
inside the cell (L / 125 um positions per axis), i.e. the average error at each sub-cell position over all scored
cells. A sub-cell placement error of the SCR kernel (or a GMC exit-position error) shows up as a fixed pattern
(e.g. low at the cell edges, high in the middle); pure noise averages out.

Scored bins: inside the circle, MC > 1 % of its maximum. The GMC map is normalised to the MC over the scored
region (the fold tests the in-cell shape, not the total). Each sub-cell position is the MC-weighted mean ratio - 1.
Noise reference: the same fold of MC seed 1 against seed 2, divided by sqrt(2) (the error of a 2-seed mean);
the GMC run's own noise is of the same order, so the noise floor of a GMC fold is ~sqrt(2) x that.

    .venv/bin/python validation/drumm_fold.py --runs sx3:..._sx3_chord sx4:..._sx4_chord --png fold.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drumm_quadtree import load_mc, OUT  # noqa: E402


def fold(G, ref, hot, k):
    """MC-weighted mean (G / ref) - 1 at each of the k x k sub-cell positions, G normalised to ref over `hot`."""
    n = ref.shape[0]
    ix = np.arange(n) % k
    with np.errstate(divide="ignore", invalid="ignore"):
        r = G / ref * (ref[hot].sum() / G[hot].sum())
    f = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            s = hot & (ix[:, None] == a) & (ix[None, :] == b)
            f[a, b] = (r[s] * ref[s]).sum() / ref[s].sum()
    return f - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", nargs="+", default=["0.5", "1", "2"], help="uniform:<L mm> meshes")
    ap.add_argument("--runs", nargs="+", default=["_sx3:500k_K8_sx3_chord", "_sx4 GMC, v2 SCR:500k_K8_sx4_chord",
                                                  "_sx4 GMC, _t1 SCR:500k_K8_sx4t1_chord"],
                    help="label:file suffix; run file = gmc_uniform_<L>_<suffix>.npz in validation/results/drumm_size")
    ap.add_argument("--png", default=str(OUT / "tgt" / "fold.png"))
    ap.add_argument("--lim", type=float, default=0.03)
    a = ap.parse_args()
    import sec_plot_style as S
    from matplotlib.colors import LinearSegmentedColormap
    S.apply()
    plt = S.plt
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    mc = load_mc()
    m1, m2 = (np.asarray(x) for x in mc["loc"])
    M = 0.5 * (m1 + m2)
    n = M.shape[0]
    h = 2.0 / n
    gx = (np.arange(n) + 0.5) * h - 1
    hot = (np.hypot(gx[:, None], gx[None, :]) < 1.0) & (M > 0.01 * M.max())
    runs = [r.split(":", 1) for r in a.runs]
    res = {}
    rows = []
    for Lmm in a.meshes:
        k = int(round(float(Lmm) / 10 / h))
        tag = Lmm.replace(".", "p")
        nz = fold(m1, m2, hot, k) / np.sqrt(2)
        row = [("MC seed 1 vs 2 (÷√2)", nz)]
        for lab, suf in runs:
            p = OUT / f"gmc_uniform_{tag}_{suf}.npz"
            if p.exists():
                row.append((lab, fold(np.load(p)["loc"].mean(0), M, hot, k)))
        rows.append((Lmm, row))
        res[f"uniform:{Lmm}"] = {lab: {"rms": float(np.sqrt(np.mean(f ** 2))), "min": float(f.min()),
                                       "max": float(f.max())} for lab, f in row}
        print(f"uniform:{Lmm} ({k} x {k} positions): " + "  ".join(
            f"{lab} rms {100 * res[f'uniform:{Lmm}'][lab]['rms']:.2f} %" for lab, _ in row))
    nc = max(len(r) for _, r in rows)
    fig, ax = plt.subplots(len(rows), nc, figsize=(3.2 * nc, 3.0 * len(rows)), constrained_layout=True, squeeze=False)
    for i, (Lmm, row) in enumerate(rows):
        for j in range(nc):
            x = ax[i, j]
            x.grid(False)
            if j >= len(row):
                x.set_visible(False)
                continue
            lab, f = row[j]
            im = x.imshow(f.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=-a.lim, vmax=a.lim)
            x.set_title(f"{lab}\nrms {100 * np.sqrt(np.mean(f ** 2)):.2f} %, {100 * f.min():+.1f}…{100 * f.max():+.1f} %",
                        fontsize=9)
            x.set_xticks([0, 0.5, 1])
            x.set_yticks([0, 0.5, 1])
            if j == 0:
                x.set_ylabel(f"uniform:{Lmm} mm\ny / L in cell")
            if i == len(rows) - 1:
                x.set_xlabel("x / L in cell")
    fig.colorbar(im, ax=ax, shrink=0.8, label="GMC / MC − 1 at that in-cell position",
                 format=lambda v, _: f"{100 * v:+g} %")
    fig.suptitle("In-cell dose pattern on uniform Drumm meshes (125 µm bins folded by position in the cell)",
                 fontweight="bold")
    fig.savefig(a.png, dpi=120)
    plt.close(fig)
    json.dump(res, open(Path(a.png).with_suffix(".json"), "w"), indent=1)
    print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
