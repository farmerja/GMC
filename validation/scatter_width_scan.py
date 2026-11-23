#!/usr/bin/env python3
"""
Scattering width of single boundary transits vs cell side L: analog MC vs the size-conditioned GMC.

Drumm-like entries (near-normal, energy E, entry point uniform on the face) into one water cell of side L; the
exit-position residual y1 - y0 - fit(v0 / u0) of far-face exits measures the multiple-scattering spread per transit
that a chained beam accumulates (2026-11-22: GMC/analog -3.9 % at 2 mm, +1.2 % at 1 mm for 15-30 MeV in the Drumm
chain logs -> beam wings too narrow / too broad).

    (ElectronTransport venv) ../ElectronTransport/.venv/bin/python validation/scatter_width_scan.py mc --out scan.npz
    (GMC venv)               .venv/bin/python validation/scatter_width_scan.py gmc --run scan.npz --model <gmc.pth>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

L_MM = (0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4, 3.0, 4.0, 5.0, 7.0, 10.0)
E_MEV = (20.0, 8.0)


def entries(n, seed):
    r = np.random.default_rng(seed)
    th = r.normal(0, 0.1, n)                       # in-plane angle (rad), Drumm-like beam spread
    ph = r.normal(0, 0.1, n)                       # out-of-plane
    u, v = np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph)
    w = np.sin(ph)
    return r.random(n), u, v, w


def cmd_mc(a):
    from chain_quadtree import MCSizeKernel
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    a.out = os.path.abspath(a.out)
    os.chdir(et)
    from material_data import build_library, named_material
    m, h = named_material("Water")
    K = MCSizeKernel(build_library([m], [h]), 1e-3, seed=a.seed)
    res = {}
    for E in E_MEV:
        for Lmm in L_MM:
            y, u, v, w = entries(a.n, a.seed + int(E * 1000 + Lmm * 10))
            o = K.boundary(y, u, v, w, np.full(a.n, E), L=Lmm / 10)
            tag = f"E{E:g}_L{Lmm:g}"
            res.update({f"{tag}_y0": y, f"{tag}_u0": u, f"{tag}_v0": v, f"{tag}_w0": w, f"{tag}_cls": o["cls"],
                        f"{tag}_y1": o["y"], f"{tag}_v1": o["v"]})
            print(f"{tag}: far {np.mean(o['cls'] == 1):.4f}", flush=True)
    np.savez_compressed(a.out, **res)


def resid(y0, u0, v0, y1):
    t = v0 / u0
    A = np.stack([t, np.ones_like(t)], 1)
    c = np.linalg.lstsq(A, y1 - y0, rcond=None)[0]
    return y1 - y0 - A @ c


def cmd_gmc(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    from chain_quadtree import MirrorY
    F = np.load(a.run)
    gm = SecondaryGMC(a.model, "cuda" if torch.cuda.is_available() else "cpu", ode_steps=a.ode_steps, seed=a.seed)
    if a.sigma:     # test: another sigma(E, L) table in the exit-flow decode only (conditioning features unchanged)
        import copy
        import ml.secondary_gmc as SG
        d = np.load(a.sigma)
        sp = copy.copy(gm.spec)
        sp.sig_logE, sp.sig_logL, sp.sig_val = d["sig_logE"].tolist(), d["sig_logL"].tolist(), d["sig_val"].tolist()
        orig_sw, orig_tr = SG.scatter_width, gm._transit

        def tr(*args, **kw):
            SG.scatter_width = lambda spec, E, L=None, which="feat": orig_sw(sp, E, L)  # noqa: E731
            try:
                return orig_tr(*args, **kw)
            finally:
                SG.scatter_width = orig_sw
        gm._transit = tr
    K = MirrorY(gm, a.seed + 3)
    rng = np.random.default_rng(0)
    print(f"far-exit position scattering width, GMC / analog ({os.path.basename(a.model)}, ode {a.ode_steps})")
    print("  E (MeV)  L (mm)   P(far) an  GMC/an    angle-resid std GMC/an      position-resid std GMC/an    q50 / q90 / q99")
    for E in E_MEV:
        for Lmm in L_MM:
            tag = f"E{E:g}_L{Lmm:g}"
            y0, u0, v0, w0 = (F[f"{tag}_{k}"] for k in ("y0", "u0", "v0", "w0"))
            o = K.boundary(y0, u0, v0, w0, np.full(len(y0), E), L=Lmm / 10)
            ma, mg = F[f"{tag}_cls"] == 1, o["cls"] == 1
            ra = resid(y0[ma], u0[ma], v0[ma], F[f"{tag}_y1"][ma])
            rg = resid(y0[mg], u0[mg], v0[mg], o["y"][mg])
            va = resid(y0[ma], u0[ma], v0[ma], F[f"{tag}_v1"][ma] - 0 * y0[ma])
            vg = resid(y0[mg], u0[mg], v0[mg], o["v"][mg])
            bs = np.std([np.std(rng.choice(rg, len(rg))) / np.std(rng.choice(ra, len(ra))) for _ in range(10)])
            bv = np.std([np.std(rng.choice(vg, len(vg))) / np.std(rng.choice(va, len(va))) for _ in range(10)])
            q = np.quantile(np.abs(rg), [.5, .9, .99]) / np.quantile(np.abs(ra), [.5, .9, .99])
            print(f"  {E:6.0f}  {Lmm:6.2f}   {ma.mean():.4f}  {mg.mean() / ma.mean():.4f}    {vg.std() / va.std():.4f} ± {bv:.4f}"
                  f"           {rg.std() / ra.std():.4f} ± {bs:.4f}             {np.round(q, 3)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mc")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--out", required=True)
    g = sub.add_parser("gmc")
    g.add_argument("--run", required=True)
    g.add_argument("--model", default=str(ROOT / "ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth"))
    g.add_argument("--ode_steps", type=int, default=20)
    g.add_argument("--seed", type=int, default=5)
    g.add_argument("--sigma", default=None, help="sigma(E, L) table .npz swapped into the exit decode (test)")
    a = ap.parse_args()
    cmd_mc(a) if a.cmd == "mc" else cmd_gmc(a)


if __name__ == "__main__":
    main()
