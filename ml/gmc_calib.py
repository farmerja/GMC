#!/usr/bin/env python3
"""
Daughter calibration of a size-conditioned SecondaryGMC (see secondary_gmc.DaughterCalib).

Why: relevant daughters (delta rays that leave the cell) occur in a few % of transits, so the joint classifier and
the daughter flow can be a few % off in an (E, L) slice at almost no loss cost. On the Drumm cylinder that slice is
what carries dose out of the beam: water 100 um - 1 cm model, high-E near-normal entries: daughter energy per transit
GMC/MC +6 % at 0.6-1.6 mm, -3 % at 1.6-4 mm, and the chained wings at |y| 2.5-5 mm were +3..+10 % (1 mm cells) and
-2..-6 % (2-4 mm cells) vs analog MC (2026-11-22).

What: per (log10 E, log10 L) bin of the training transits (boundary and interior separately),
    1. nrel: logit offset on every classifier class with n_rel >= 1, iterated until the EXPECTED min(n_rel, NR) per
       bin (exact, from the softmax) matches the MC count;
    2. edau: offset on the daughter energy-share logit, iterated until the daughter energy per transit of the free-
       running GMC (classifier with offsets 1, exit flow, daughter flow) matches MC. Only transits whose sampled class
       has n_rel >= 1 are run through the flows (the others carry no daughter energy), so this is cheap.
Updates are shrunk towards 0 in bins with few MC daughters (w = n / (n + n0)).

Usage:
    python ml/gmc_calib.py --model ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth \
        --data "../ElectronTransport/sweep_water_100um-1cm_30MeV_42/part_*[0-9].npz" \
        --out ml/models/size_gmc_water_100um-1cm_30MeV/daughter_calib.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.secondary_gmc import (SecondaryGMC, load_tables, boundary_cond, interior_cond)  # noqa: E402

CHUNK = 1 << 20


def collect(files, cache):
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {k: {c[2:]: d[c] for c in d.files if c.startswith(k + "_")} for k in ("B", "I")}
    cols = ("x0", "y0", "u0", "v0", "E0", "L", "n_rel")
    out = {k: {c: [] for c in cols + ("edau",)} for k in ("B", "I")}
    for f in files:
        T = load_tables(f)
        dsum = np.bincount(T["dau"]["parent"], weights=T["dau"]["E"], minlength=len(T["E0"]))
        for k, m in (("B", T["boundary"]), ("I", T["interior"])):
            r = np.flatnonzero(m)
            for c in cols:
                out[k][c].append(T[c][r].astype(np.float32 if c != "n_rel" else np.int8))
            out[k]["edau"].append(dsum[r])
    out = {k: {c: np.concatenate(v) for c, v in d.items()} for k, d in out.items()}
    if cache:
        np.savez(cache, **{f"{k}_{c}": v for k, d in out.items() for c, v in d.items()})
    return out


def bins(E, L, logE, logL):
    i = np.clip(np.rint((np.log10(E) - logE[0]) / (logE[1] - logE[0])), 0, len(logE) - 1).astype(int)
    j = np.clip(np.rint((np.log10(L) - logL[0]) / (logL[1] - logL[0])), 0, len(logL) - 1).astype(int)
    return i * len(logL) + j


def cond_of(K, kind, D, sl):
    f = (lambda: boundary_cond(K.spec, D["y0"][sl], D["u0"][sl], D["v0"][sl], D["E0"][sl], D["L"][sl])) if kind == "B" \
        else (lambda: interior_cond(K.spec, D["x0"][sl], D["y0"][sl], D["u0"][sl], D["v0"][sl], D["E0"][sl], D["L"][sl]))
    return np.asarray(f(), np.float32)


def expected_nrel(K, kind, D, tab, calib_grid):
    """Per-row expected min(n_rel, NR) under the classifier with offsets tab (exact, no sampling)."""
    clf = K.m["clf_" + kind]["model"]
    nr_vec = torch.as_tensor(K.spec.decode(np.arange(clf.cfg.num_classes))[1], dtype=torch.float32, device=K.device)
    K.calib = calib_grid(tab)
    out = np.zeros(len(D["E0"]))
    with torch.no_grad():
        for s in range(0, len(out), CHUNK):
            sl = slice(s, s + CHUNK)
            c = torch.as_tensor(cond_of(K, kind, D, sl), device=K.device)
            off = torch.as_tensor(K.calib.offset("nrel_" + kind, D["E0"][sl], D["L"][sl]), dtype=torch.float32,
                                  device=K.device)
            for i in range(0, len(c), 65536):
                lg = clf(c[i:i + 65536]) + off[i:i + 65536, None] * (nr_vec >= 1)[None, :]
                out[s + i:s + i + len(lg)] = (torch.softmax(lg, -1) @ nr_vec).cpu().numpy()
    return out


def gmc_edau(K, kind, D, tab, calib_grid, reps):
    """Per-row daughter energy of the free-running GMC (mean of reps), offsets tab."""
    K.calib = calib_grid(tab)
    tot = np.zeros(len(D["E0"]))
    orig = K._classify
    for _ in range(reps):
        for s in range(0, len(tot), CHUNK):
            sl = slice(s, s + CHUNK)
            cond = cond_of(K, kind, D, sl)
            E0, L = D["E0"][sl].astype(np.float64), D["L"][sl].astype(np.float64)
            cls, nr, flag = orig("clf_" + kind, cond, E0, L)
            r = np.flatnonzero(nr >= 1)
            if r.size == 0:
                continue
            K._classify = lambda *a, **k: (cls[r], nr[r], flag[r])  # noqa: E731
            try:
                x0 = np.zeros(r.size) if kind == "B" else D["x0"][sl][r].astype(np.float64)
                g = K._transit(kind, x0, D["y0"][sl][r].astype(np.float64), D["u0"][sl][r].astype(np.float64),
                               D["v0"][sl][r].astype(np.float64), np.full(r.size, 0.5), E0[r], cond[r], L[r])
            finally:
                K._classify = orig
            dd = g["daughters"]
            tot[s:s + CHUNK] += np.bincount(r[dd["parent"]], weights=dd["E"], minlength=len(E0))
    return tot / reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="glob of training part files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="npz cache of the collected rows")
    ap.add_argument("--n_logE", type=int, default=14)
    ap.add_argument("--logE_lo", type=float, default=-2.0)
    ap.add_argument("--n_logL", type=int, default=9)
    ap.add_argument("--n0", type=float, default=300.0, help="shrinkage: MC daughters per bin for half weight")
    ap.add_argument("--iters_n", type=int, default=4)
    ap.add_argument("--iters_e", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2, help="GMC replicas per row in the energy step")
    ap.add_argument("--max_offset", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(a.data) if not f.endswith("_subcell.npz"))
    t0 = time.time()
    data = collect(files, a.cache)
    print(f"{len(files)} files: {len(data['B']['E0'])} boundary, {len(data['I']['E0'])} interior rows "
          f"({time.time() - t0:.0f} s)", flush=True)
    K = SecondaryGMC(a.model, a.device, seed=a.seed)
    spec = K.spec
    logE = np.linspace(a.logE_lo, spec.logE_hi, a.n_logE)
    logL = np.linspace(spec.logL_lo, spec.logL_hi, a.n_logL)
    shape = (len(logE), len(logL))
    tab = {k: np.zeros(shape) for k in ("nrel_B", "nrel_I", "edau_B", "edau_I")}
    from ml.secondary_gmc import DaughterCalib
    grid = lambda t: DaughterCalib({"logE": logE, "logL": logL, **t})  # noqa: E731
    diag = {}
    nb = shape[0] * shape[1]
    for kind in ("B", "I"):
        D = data[kind]
        b = bins(D["E0"], D["L"], logE, logL)
        mc_n = np.bincount(b, weights=np.minimum(D["n_rel"], spec.NR).astype(float), minlength=nb)
        mc_e = np.bincount(b, weights=D["edau"], minlength=nb)
        w = (mc_n / (mc_n + a.n0)).reshape(shape)
        for it in range(a.iters_n + 1):
            g_n = np.bincount(b, weights=expected_nrel(K, kind, D, tab, grid), minlength=nb)
            r = np.where((mc_n > 0) & (g_n > 0), mc_n / np.maximum(g_n, 1e-300), 1.0).reshape(shape)
            tot = g_n.sum() / mc_n.sum()
            print(f"[{kind}] nrel iter {it}: total GMC/MC {tot:.4f}; daughter-weighted |log ratio| "
                  f"{np.sum(mc_n * np.abs(np.log(r.ravel()))) / mc_n.sum():.4f}", flush=True)
            if it == 0:
                diag[f"nrel_ratio0_{kind}"] = 1 / r
            if it == a.iters_n:
                diag[f"nrel_ratio_{kind}"] = 1 / r
                break
            tab["nrel_" + kind] = np.clip(tab["nrel_" + kind] + w * np.log(r), -a.max_offset, a.max_offset)
        for it in range(a.iters_e + 1):
            g_e = np.bincount(b, weights=gmc_edau(K, kind, D, tab, grid, a.reps), minlength=nb)
            r = np.where((mc_e > 0) & (g_e > 0), mc_e / np.maximum(g_e, 1e-300), 1.0).reshape(shape)
            print(f"[{kind}] edau iter {it}: total GMC/MC {g_e.sum() / mc_e.sum():.4f}; daughter-weighted |log ratio| "
                  f"{np.sum(mc_n * np.abs(np.log(r.ravel()))) / mc_n.sum():.4f}", flush=True)
            if it == 0:
                diag[f"edau_ratio0_{kind}"] = 1 / r
            if it == a.iters_e:
                diag[f"edau_ratio_{kind}"] = 1 / r
                break
            tab["edau_" + kind] = np.clip(tab["edau_" + kind] + w * np.log(r), -a.max_offset, a.max_offset)
        diag[f"mc_n_{kind}"] = mc_n.reshape(shape)
    np.savez(a.out, logE=logE, logL=logL, model=os.path.abspath(a.model), **tab, **diag)
    print(f"saved {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
