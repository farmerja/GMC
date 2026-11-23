#!/usr/bin/env python3
"""
Analog-chain diagnostic for the Drumm cylinder on a mesh of square cells (validation/drumm_quadtree.py meshes).

The chained analog MC (chain_quadtree.MCSizeKernel: one analog run per cell transit) is exact by the Markov property,
so on a mesh it must reproduce the analog MC of the mesh's geometry. Two questions:

  1. Does the analog chain reproduce the true-circle MC cell totals (lateral profile of the beam wings)? If not, the
     wing bias of the GMC chain is in the chain / mesh bookkeeping, not in the GMC.
  2. Re-sampling the GMC on the analog chain's own boundary entries (exactly the entry states of the problem), does
     a whole cell transit (primary exit + relevant daughters transported to the cell boundary) send the same energy
     out through the OUTWARD side face (away from the beam axis) as the analog cell? Region by region.

    (ElectronTransport venv) ../ElectronTransport/.venv/bin/python validation/drumm_analog_chain.py mc --mesh uniform:1 --N 100000
    (GMC venv)               .venv/bin/python validation/drumm_analog_chain.py compare --run validation/results/drumm_size/analog_uniform_1_100k.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from chain_quadtree import chain_quadtree, VacuumKernel, LocalPhotons, MCSizeKernel, MirrorY  # noqa: E402
from chain_core import transform_from_canonical  # noqa: E402
from drumm_quadtree import make_mesh, mesh_tag, OUT, H_OUT, load_mc  # noqa: E402
from drumm_cylinder import triangle_source  # noqa: E402

LOG_COLS = ("cid", "start", "face", "gen", "y0", "u0", "v0", "w0", "E0", "cls", "x1", "y1", "E_out", "dose")
ABSORBED = 4


class Recorder:
    """Keeps the secondaries an analog kernel lets out of each cell, indexed by the kernel's own row counter (the
    order of the water rows of the chain's transit log)."""

    def __init__(self, k):
        self.k, self.n, self.rec = k, 0, []

    def _keep(self, out, n):
        X = out["extra_exits"]
        self.rec.append({"row": np.asarray(X["parent"], np.int64) + self.n, **{c: X[c] for c in ("x", "y", "E")}})
        self.n += n
        return out

    def boundary(self, y, u, v, w, E, L=None):
        return self._keep(self.k.boundary(y, u, v, w, E, L=L), len(E))

    def interior(self, x, y, u, v, w, E, L=None):
        return self._keep(self.k.interior(x, y, u, v, w, E, L=L), len(E))


def cmd_mc(a):
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    a.out = a.out and os.path.abspath(a.out)
    os.chdir(et)                      # the ElectronTransport data tables are found relative to it
    from material_data import build_library, named_material
    m, h = named_material("Water")
    lib = build_library([m], [h])
    mesh = make_mesh(a.mesh)
    src = triangle_source(a.N, a.src_seed)
    dose = np.zeros(mesh.n_cells)
    T, X = [], []
    off = 0
    t0 = time.time()
    for i, b in enumerate(range(0, a.N, a.batch)):
        rec = Recorder(MCSizeKernel(lib, a.E_cutoff, seed=a.seed + 100003 * i))
        c = chain_quadtree([LocalPhotons(rec), VacuumKernel()], src[b:b + a.batch], mesh, a.E_cutoff,
                           max_steps=100000, transit_log=True)
        dose += c["dose"]
        tr = c["transits"]
        w = tr["mat"] == 0
        if int(w.sum()) != rec.n:
            raise RuntimeError(f"recorder rows {rec.n} != water transits {int(w.sum())}")
        T.append({k: tr[k][w] for k in LOG_COLS})
        x = {k: np.concatenate([r[k] for r in rec.rec]) for k in ("row", "x", "y", "E")}
        x["row"] = x["row"] + off
        X.append(x)
        off += rec.n
        print(f"batch {i}: {b + len(src[b:b + a.batch])}/{a.N}, {time.time() - t0:.0f} s, "
              f"{rec.n / len(src[b:b + a.batch]):.1f} water transits/primary", flush=True)
    secs = time.time() - t0
    T = {k: np.concatenate([t[k] for t in T]) for k in LOG_COLS}
    X = {k: np.concatenate([x[k] for x in X]) for k in ("row", "x", "y", "E")}
    out = a.out or str(OUT / f"analog_{mesh_tag(a.mesh)}_{a.N // 1000}k.npz")
    f32 = lambda v: v.astype(np.float32) if v.dtype == np.float64 else v  # noqa: E731
    np.savez_compressed(out, cell=dose, x0=mesh.x0, y0=mesh.y0, L=mesh.L, mat=mesh.mat, N=a.N,
                        **{"T_" + k: (v if k in ("E0", "E_out", "dose") else f32(v)) for k, v in T.items()},
                        **{"X_" + k: v for k, v in X.items()},
                        meta=json.dumps({**vars(a), "seconds": secs}))
    print(f"analog chain {a.mesh}: {secs:.0f} s ({1e3 * secs / a.N:.2f} ms/primary) -> {out}")


# ------------------------------------------------------------------------------------------------------- compare
def cell_frame_face(x_c, y_c, face):
    """Physical face (0 bottom, 1 right, 2 top, 3 left) of canonical-frame boundary points of cells entered via face."""
    x, y, _, _ = transform_from_canonical(x_c, y_c, np.zeros_like(x_c), np.zeros_like(x_c), face)
    return np.argmin(np.stack([np.abs(y), np.abs(1 - x), np.abs(1 - y), np.abs(x)], 1), 1)


def gmc_cell(K, y, u, v, w, E, L, max_levels=6):
    """Whole-cell GMC transit (canonical frame): primary + relevant daughters followed to the cell boundary.
    Returns exits (row, x, y, E), dose per row (photon energy local) and the primary's own outcome (cls, x, y)."""
    o = K.boundary(y, u, v, w, E, L=L)
    prim = (o["cls"].copy(), o["x"].copy(), o["y"].copy())
    n = len(E)
    dose = o["dose"].copy()
    ex = o["cls"] < ABSORBED
    rows, xs, ys, Es = [np.flatnonzero(ex)], [o["x"][ex]], [o["y"][ex]], [o["E_out"][ex]]
    lv = [np.zeros(int(ex.sum()), np.int64)]
    D, par = o["daughters"], np.asarray(o["daughters"]["parent"], np.int64)
    for _ in range(max_levels):
        if len(D["E"]) == 0:
            break
        g = K.interior(D["x"], D["y"], D["u"], D["v"], D["w"], D["E"], L=L[par])
        dose += np.bincount(par, weights=g["dose"], minlength=n)
        ex = g["cls"] < ABSORBED
        rows.append(par[ex]); xs.append(g["x"][ex]); ys.append(g["y"][ex]); Es.append(g["E_out"][ex])  # noqa: E702
        lv.append(np.full(int(ex.sum()), len(lv), np.int64))
        par = par[np.asarray(g["daughters"]["parent"], np.int64)]
        D = g["daughters"]
    gmc_cell.level = np.concatenate(lv)
    return np.concatenate(rows), np.concatenate(xs), np.concatenate(ys), np.concatenate(Es), dose, prim


def face_energy(rows, x, y, E, face_of_row, outward_face, n):
    """Energy per row leaving through [outward side, inward side, downstream (+x), upstream (-x)]."""
    pf = cell_frame_face(x, y, face_of_row[rows])
    out = np.zeros((n, 4))
    side_out = pf == outward_face[rows]
    side_in = ((pf == 0) | (pf == 2)) & ~side_out
    for k, m in enumerate((side_out, side_in, pf == 1, pf == 3)):
        out[:, k] = np.bincount(rows[m], weights=E[m], minlength=n)
    return out


def cmd_compare(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    F = np.load(a.run, allow_pickle=False)
    meta = json.loads(str(F["meta"]))
    N = int(F["N"])
    x0, y0, L, mat = F["x0"], F["y0"], F["L"], F["mat"]
    xc, yc = x0 + L / 2 - 1.0, y0 + L / 2 - 1.0          # cell centres, cylinder axis at 0
    # 1. cell totals: analog chain vs true-circle MC (pixel sums over each cell) vs the GMC chain on the same mesh
    mc = load_mc()
    mcm = 0.5 * (mc["loc"][0] + mc["loc"][1]) * H_OUT ** 2   # energy per primary per pixel
    mcd = 0.5 * np.abs(mc["loc"][0] - mc["loc"][1]) * H_OUT ** 2
    gm_path = OUT / f"gmc_{mesh_tag(meta['mesh'])}_500k.npz"
    gm = np.load(gm_path) if gm_path.exists() else None
    gmm = None if gm is None else np.mean(gm["loc"], 0) / int(gm["N"])

    def cell_sums(img):
        s = np.zeros(len(L))
        for c in range(len(L)):
            i, j, k = int(round(x0[c] / H_OUT)), int(round(y0[c] / H_OUT)), int(round(L[c] / H_OUT))
            s[c] = img[i:i + k, j:j + k].sum()
        return s
    A = F["cell"] / N
    M, Md = cell_sums(mcm), cell_sums(mcd)
    G = None if gmm is None else cell_sums(gmm)
    ay = np.abs(yc)
    sel = (mat == 0) & (xc > a.x_lo) & (xc < a.x_hi)
    edges = np.unique(np.round(np.concatenate([np.abs(y0 - 1.0), np.abs(y0 + L - 1.0)]), 6))
    edges = edges[edges <= a.y_max + 1e-9]
    print(f"{os.path.basename(a.run)}: {meta['mesh']}, N {N}, {meta['seconds'] / N * 1e3:.2f} ms/primary\n")
    print(f"1. Cell totals over {a.x_lo} < x < {a.x_hi} cm by |y| of the cell centre (ratios to the true-circle MC; "
          f"analog-chain noise ~ 1/sqrt(transits))")
    print("   |y| band (mm)   analog chain / MC     GMC chain / MC    MC seed half-diff")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = sel & (ay >= lo) & (ay < hi)
        if not m.any():
            continue
        g = "" if G is None else f"{G[m].sum() / M[m].sum():.4f}"
        print(f"   {10 * lo:4.1f}-{10 * hi:4.1f}        {A[m].sum() / M[m].sum():.4f}              {g:8s}         "
              f"{Md[m].sum() / M[m].sum():.4f}")

    # 2. GMC whole-cell transit re-sampled on the analog boundary entries
    T = {k[2:]: F[k] for k in F.files if k.startswith("T_")}
    X = {k[2:]: F[k] for k in F.files if k.startswith("X_")}
    b = np.flatnonzero((T["start"] == 0) & (T["E0"] > a.E_min))
    if a.max_rows and len(b) > a.max_rows:
        b = np.sort(np.random.default_rng(0).choice(b, a.max_rows, replace=False))
    n = len(b)
    cid = T["cid"][b].astype(np.int64)
    face = T["face"][b].astype(np.int64)
    Lr = L[cid]
    outward = np.where(yc[cid] < 0, 0, 2)                    # physical face pointing away from the axis
    E0 = T["E0"][b]
    # analog: primary exit + recorded secondaries of these rows
    pos = np.full(len(T["E0"]), -1)
    pos[b] = np.arange(n)
    ex = T["cls"][b] < ABSORBED
    xr = pos[X["row"]]
    k = xr >= 0
    rows_a = np.concatenate([np.flatnonzero(ex), xr[k]])
    fa = face_energy(rows_a, np.concatenate([T["x1"][b][ex], X["x"][k]]).astype(float),
                     np.concatenate([T["y1"][b][ex], X["y"][k]]).astype(float),
                     np.concatenate([T["E_out"][b][ex], X["E"][k]]), face, outward, n)
    dose_a = T["dose"][b]
    sa = face_energy(xr[k], X["x"][k].astype(float), X["y"][k].astype(float), X["E"][k], face, outward, n)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fg, dose_g, sg = np.zeros((n, 4)), np.zeros(n), np.zeros((n, 4))
    pg_cls, pg_x, pg_y = np.zeros(n, np.int64), np.zeros(n), np.zeros(n)
    for s in range(0, n, a.chunk):
        sl = slice(s, s + a.chunk)
        K = LocalPhotons(MirrorY(SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed + s), a.seed + s + 3))
        r, xg, yg, Eg, dg, (pg_cls[sl], pg_x[sl], pg_y[sl]) = gmc_cell(K, T["y0"][b][sl].astype(float), T["u0"][b][sl].astype(float),
                                     T["v0"][b][sl].astype(float), T["w0"][b][sl].astype(float), E0[sl], Lr[sl])
        fg[sl] = face_energy(r, xg, yg, Eg, face[sl], outward[sl], len(dg))
        sec = gmc_cell.level > 0
        sg[sl] = face_energy(r[sec], xg[sec], yg[sec], Eg[sec], face[sl], outward[sl], len(dg))
        dose_g[sl] = dg
    print(f"\n2. GMC whole-cell transit re-sampled on {n} analog boundary entries (E0 > {a.E_min} MeV): energy per "
          f"entry leaving through each face, GMC / analog (z = (GMC - analog) / sigma)")
    print("   region (cell centre)          entries  out-side           in-side            downstream         "
          "upstream           deposited")
    regions = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    xr_ = xc[cid]
    for xlo, xhi in ((-1.0, 0.0), (0.0, 0.8)):
        for lo, hi in regions:
            m = (np.abs(yc[cid]) >= lo) & (np.abs(yc[cid]) < hi) & (xr_ > xlo) & (xr_ < xhi)
            if m.sum() < 200:
                continue
            line = f"   x {xlo:+.1f}..{xhi:+.1f} |y| {10 * lo:4.1f}-{10 * hi:4.1f}mm {m.sum():8d}"
            for j in range(5):
                va, vg = (fa[m, j], fg[m, j]) if j < 4 else (dose_a[m], dose_g[m])
                sd = np.sqrt(va.var() / m.sum() + vg.var() / m.sum())
                line += f"  {vg.mean() / max(va.mean(), 1e-300):.4f} ({(vg.mean() - va.mean()) / max(sd, 1e-300):+5.1f})"
            print(line, flush=True)


    # 3. primary exit drift away from the axis (continuous, much less noisy than face energies)
    # entry point in global coordinates (canonical entry (0, y0) -> cell frame)
    xe, ye, _, _ = transform_from_canonical(np.zeros(n), T["y0"][b].astype(float), np.zeros(n), np.zeros(n), face)
    gye = y0[cid] + ye * Lr
    def drift(cls, xc_, yc_):
        x, y, _, _ = transform_from_canonical(xc_.astype(float), yc_.astype(float), np.zeros(n), np.zeros(n), face)
        gy1 = y0[cid] + y * Lr
        d = (np.abs(gy1 - 1.0) - np.abs(gye - 1.0)) / Lr       # outward displacement, cell units
        return np.where(cls < ABSORBED, d, np.nan)
    da = drift(T["cls"][b], T["x1"][b], T["y1"][b])
    dg = drift(pg_cls, pg_x, pg_y)
    print(f"\n3. Primary exit drift away from the axis per transit (cell units; exits only), GMC - analog (z)")
    print("   region (cell centre)          E0 band (MeV)   entries   analog mean   GMC - analog (z)    "
          "analog E[d^2]   GMC/analog")
    for xlo, xhi in ((-1.0, 0.0), (0.0, 0.8)):
        for lo, hi in regions:
            for elo, ehi in ((0.0, 1.0), (1.0, 5.0), (5.0, 15.0), (15.0, 30.0)):
                m = ((np.abs(yc[cid]) >= lo) & (np.abs(yc[cid]) < hi) & (xr_ > xlo) & (xr_ < xhi) & (E0 >= elo)
                     & (E0 < ehi))
                ma, mg = m & np.isfinite(da), m & np.isfinite(dg)
                if ma.sum() < 500:
                    continue
                va, vg = da[ma], dg[mg]
                sd = np.sqrt(va.var() / len(va) + vg.var() / len(vg))
                print(f"   x {xlo:+.1f}..{xhi:+.1f} |y| {10 * lo:4.1f}-{10 * hi:4.1f}mm  {elo:4.0f}-{ehi:<4.0f}      "
                      f"{m.sum():8d}   {va.mean():+.5f}     {vg.mean() - va.mean():+.5f} ({(vg.mean() - va.mean()) / sd:+5.1f})"
                      f"    {np.mean(va ** 2):.5f}      {np.mean(vg ** 2) / np.mean(va ** 2):.4f}", flush=True)

    print(f"\n4. Secondaries only: energy per entry leaving through each face, GMC (relevant daughters + interior "
          f"model) / analog (escaping secondaries), z")
    print("   region (cell centre)          entries  out-side           in-side            downstream         upstream")
    for xlo, xhi in ((-1.0, 0.0), (0.0, 0.8)):
        for lo, hi in regions:
            m = (np.abs(yc[cid]) >= lo) & (np.abs(yc[cid]) < hi) & (xr_ > xlo) & (xr_ < xhi)
            if m.sum() < 2000:
                continue
            line = f"   x {xlo:+.1f}..{xhi:+.1f} |y| {10 * lo:4.1f}-{10 * hi:4.1f}mm {m.sum():8d}"
            for j in range(4):
                va, vg = sa[m, j], sg[m, j]
                sd = np.sqrt(va.var() / m.sum() + vg.var() / m.sum())
                line += f"  {vg.mean() / max(va.mean(), 1e-300):.4f} ({(vg.mean() - va.mean()) / max(sd, 1e-300):+5.1f})"
            print(line, flush=True)
    tot_a, tot_g = sa.sum(1), sg.sum(1)
    print(f"   all regions: secondary energy out per entry GMC/analog {tot_g.mean() / tot_a.mean():.4f}; "
          f"share of all energy leaving the cells: analog {tot_a.sum() / fa.sum():.4f}")


def cmd_flux(a):
    """GMC chain (same source, same mesh) with a transit log: boundary-entry flux per region / energy band /
    generation vs the analog chain's. The first region where they part shows where the GMC transport goes wrong."""
    import torch
    from ml.secondary_gmc import SecondaryGMC
    F = np.load(a.run, allow_pickle=False)
    meta = json.loads(str(F["meta"]))
    N = int(F["N"])
    mesh = make_mesh(meta["mesh"])
    src = triangle_source(N, meta["src_seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cols = ("cid", "start", "gen", "E0")
    G = {k: [] for k in cols}
    for i, b in enumerate(range(0, N, a.batch)):
        k0 = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed + 7 * i)
        c = chain_quadtree([LocalPhotons(MirrorY(k0, a.seed + 7 * i + 3)), VacuumKernel()], src[b:b + a.batch], mesh,
                           1e-3, max_steps=100000, transit_log=True)
        tr = c["transits"]
        w = tr["mat"] == 0
        for k in cols:
            G[k].append(tr[k][w])
    G = {k: np.concatenate(v) for k, v in G.items()}
    A = {k: F["T_" + k] for k in cols}
    xc, yc = mesh.x0 + mesh.L / 2 - 1.0, mesh.y0 + mesh.L / 2 - 1.0
    xb = np.arange(-1.0, 1.0001, 0.2)
    yb = np.unique(np.round(np.abs(np.concatenate([mesh.y0, mesh.y0 + mesh.L]) - 1.0), 6))
    yb = yb[yb <= a.y_max + 1e-9]
    for title, sel, wkey in (("primaries (gen 0), entries", lambda T: (T["start"] == 0) & (T["gen"] == 0), None),
                             ("secondaries (gen >= 1) entering through a face, energy",
                              lambda T: (T["start"] == 0) & (T["gen"] >= 1), "E0"),
                             ("primaries 1-5 MeV, entries",
                              lambda T: (T["start"] == 0) & (T["gen"] == 0) & (T["E0"] >= 1) & (T["E0"] < 5), None),
                             ("primaries 5-15 MeV, entries",
                              lambda T: (T["start"] == 0) & (T["gen"] == 0) & (T["E0"] >= 5) & (T["E0"] < 15), None)):
        print(f"\nGMC / analog, {title} (z); columns x from {xb[0]:.1f} to {xb[-1]:.1f} cm in 2 mm bins")
        for lo, hi in zip(yb[:-1], yb[1:]):
            line = f"  |y| {10 * lo:3.0f}-{10 * hi:<3.0f}mm"
            for x0_, x1_ in zip(xb[:-1], xb[1:]):
                v = []
                for T in (A, G):
                    m = sel(T)
                    cc = T["cid"][m].astype(np.int64)
                    r = (np.abs(yc[cc]) >= lo) & (np.abs(yc[cc]) < hi) & (xc[cc] > x0_) & (xc[cc] < x1_)
                    wv = np.ones(r.sum()) if wkey is None else T[wkey][m][r]
                    v.append((wv.sum(), np.sqrt((wv ** 2).sum())))
                (sa, ea), (sg, eg) = v
                line += "    -      " if sa <= 0 else f" {sg / sa:5.3f}({(sg - sa) / max(np.hypot(ea, eg), 1e-300):+4.1f})"
            print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mc")
    p.add_argument("--mesh", required=True)
    p.add_argument("--N", type=int, default=100000)
    p.add_argument("--batch", type=int, default=20000)
    p.add_argument("--src_seed", type=int, default=2007)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--run", required=True)
    c.add_argument("--model", default=str(ROOT / "ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth"))
    c.add_argument("--ode_steps", type=int, default=20)
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--E_min", type=float, default=0.0)
    c.add_argument("--max_rows", type=int, default=0)
    c.add_argument("--chunk", type=int, default=500000)
    c.add_argument("--x_lo", type=float, default=0.0)
    c.add_argument("--x_hi", type=float, default=0.8)
    c.add_argument("--y_max", type=float, default=0.6)
    f = sub.add_parser("flux")
    f.add_argument("--run", required=True)
    f.add_argument("--model", default=str(ROOT / "ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth"))
    f.add_argument("--ode_steps", type=int, default=20)
    f.add_argument("--seed", type=int, default=11)
    f.add_argument("--batch", type=int, default=100000)
    f.add_argument("--y_max", type=float, default=0.6)
    a = ap.parse_args()
    {"mc": cmd_mc, "compare": cmd_compare, "flux": cmd_flux}[a.cmd](a)


if __name__ == "__main__":
    main()
