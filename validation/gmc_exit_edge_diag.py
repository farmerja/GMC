#!/usr/bin/env python3
"""
GMC exit positions near the cell corners: held-out analog boundary transits vs the SecondaryGMC sampled on the same
entry states.

Motivation: in the Drumm benchmark, GMC + SCR has the same within-cell error pattern in every cell along the beam
(sub-rows next to the lateral faces low, middle high), starting at the second cell, while the SCR kernel is exact on
true transits. Exits of one cell are the entries of the next, so a sub-cell error in the sampled exit points moves
the next cell's dose without changing cell totals.

Exit position as a coordinate along the cell perimeter in the canonical frame, t in [0, 4):
    right face  t = y1            (0 .. 1, from the bottom-right corner up)
    top face    t = 1 + (1 - x1)  (1 .. 2)
    left face   t = 2 + (1 - y1)  (2 .. 3)
    bottom face t = 3 + x1        (3 .. 4, ending at the bottom-right corner)
Histogrammed in fine bins; ratio GMC / analog per bin with its noise.

Usage (GMC venv, GMC root):
    python validation/gmc_exit_edge_diag.py --model ml/models/secondary_gmc_water_1mm_30MeV_elsepa \
        --test_npz ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_2M_7_elsepa.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)
from ml.cell_topology import classes_from_status  # noqa: E402
from ml.secondary_gmc import SecondaryGMC  # noqa: E402

BOTTOM, RIGHT, TOP, LEFT, ABSORBED = 0, 1, 2, 3, 4


def perimeter(cls, x1, y1):
    t = np.full(len(cls), np.nan)
    t = np.where(cls == RIGHT, y1, t)
    t = np.where(cls == TOP, 1 + (1 - x1), t)
    t = np.where(cls == LEFT, 2 + (1 - y1), t)
    t = np.where(cls == BOTTOM, 3 + x1, t)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--test_npz", required=True)
    p.add_argument("--E_min", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1000000)
    p.add_argument("--bins_per_face", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    D = np.load(a.test_npz, allow_pickle=False)
    seg = D["segments"]
    cols = [str(c) for c in D["segment_columns"]]
    c = cols.index
    L = float(D["box_dims"][0])
    cls = classes_from_status(seg[:, c("status")].astype(int), invalid=-1)
    keep = (cls >= 0) & (seg[:, c("start_type")] < 0.5) & (seg[:, c("origin")] == 0) & (seg[:, c("E0")] >= a.E_min)
    rows = np.flatnonzero(keep)
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(rows, size=min(a.n, len(rows)), replace=False))
    g = lambda k: seg[rows, c(k)].astype(np.float64)  # noqa: E731
    y0, u0, v0, w0, E0 = g("y0") / L, g("u0"), g("v0"), g("w0"), g("E0")
    cl_t = cls[rows]
    t_true = perimeter(cl_t, g("x1") / L, g("y1") / L)
    print(f"{len(rows)} boundary primaries with E0 >= {a.E_min} MeV; entry x0 max {np.abs(g('x0')).max():.2e}",
          flush=True)

    gmc = SecondaryGMC(a.model, "cuda", ode_steps=a.ode_steps, seed=a.seed + 1)
    t_gmc = np.full(len(rows), np.nan)
    cl_g = np.zeros(len(rows), dtype=int)
    for i in range(0, len(rows), 100000):
        sl = slice(i, i + 100000)
        out = gmc.boundary(y0[sl], u0[sl], v0[sl], w0[sl], E0[sl])
        cl_g[sl] = out["cls"]
        t_gmc[sl] = perimeter(np.asarray(out["cls"]), out["x"], out["y"])

    nb = a.bins_per_face
    edges = np.linspace(0, 4, 4 * nb + 1)
    res = {"model": a.model, "test_npz": a.test_npz, "n": int(len(rows)), "groups": []}
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    groups = [("all", np.ones(len(rows), bool)),
              ("E0 > 10 MeV, u0 > 0.9", (E0 > 10) & (u0 > 0.9)),
              ("E0 > 10 MeV, u0 > 0.9, y0 < 0.125", (E0 > 10) & (u0 > 0.9) & (y0 < 0.125)),
              ("E0 > 10 MeV, u0 > 0.9, y0 in [0.375, 0.625)", (E0 > 10) & (u0 > 0.9) & (np.abs(y0 - 0.5) < 0.125)),
              ("E0 > 10 MeV, all directions", E0 > 10),
              ("1 < E0 < 10 MeV", E0 < 10)]
    for name, m in groups:
        ht = np.histogram(t_true[m], edges)[0].astype(float)
        hg = np.histogram(t_gmc[m], edges)[0].astype(float)
        ncls_t = np.bincount(cl_t[m], minlength=5)
        ncls_g = np.bincount(cl_g[m], minlength=5)
        print(f"\n{name}: n = {m.sum()}; exit class GMC/MC (bottom, right, top, left, absorbed): "
              f"{ncls_g / np.maximum(ncls_t, 1)}")
        # right face, finely: the bins at both corners plus coarse middle
        r = slice(0, nb)
        rt, rg = ht[r], hg[r]
        sig = np.sqrt(rt + rg) / np.maximum(rt, 1)
        print("  right face y1 bins (ratio GMC/MC):", rg / np.maximum(rt, 1))
        print("                       sigma      :", sig)
        print("  right-face exits within 1/32, 1/16, 1/8 of the bottom / top corners, GMC/MC:",
              [f"{hg[:k].sum() / max(ht[:k].sum(), 1):.3f}/{hg[nb - k:nb].sum() / max(ht[nb - k:nb].sum(), 1):.3f}"
               for k in (1, 2, 4)])
        # bottom and top faces near the exit-side corners
        bt = slice(3 * nb, 4 * nb)
        tp = slice(nb, 2 * nb)
        print("  bottom face x1 bins (ratio):", hg[bt] / np.maximum(ht[bt], 1))
        print("  top face (1 - x1) bins    :", hg[tp] / np.maximum(ht[tp], 1))
        res["groups"].append({"name": name, "n": int(m.sum()), "hist_true": ht.tolist(), "hist_gmc": hg.tolist(),
                              "cls_true": ncls_t.tolist(), "cls_gmc": ncls_g.tolist()})
    out = a.out or os.path.join(a.model, f"exit_edge_diag_{os.path.basename(a.test_npz)[:-4]}.json")
    with open(out, "w") as f:
        json.dump({**res, "edges": edges.tolist()}, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
