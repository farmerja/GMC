#!/usr/bin/env python3
"""
Cost of one cell transit vs cell side L: analog MC (ElectronTransport, all CPU threads) vs the size-conditioned GMC
(GPU) and the SCR placement (GPU), on the same Drumm-like entries (scatter_width_scan.entries: near-normal, 0.1 rad
spread, entry point uniform on the face) at one energy. The GMC transit includes the interior transits of the daughters
it emits; the SCR cost uses the Drumm settings (K = max(K_min, K_scale x auto_samples(L / 125 um))).

    (ElectronTransport venv) ../ElectronTransport/.venv/bin/python validation/timing_vs_size.py mc   --out t_mc.json
    (GMC venv)               .venv/bin/python validation/timing_vs_size.py gmc  --out t_gmc.json --model ... --kernel ...
    (GMC venv)               .venv/bin/python validation/timing_vs_size.py plot --mc t_mc.json --gmc t_gmc.json --png ...
Run on a quiet machine (no training / generation): these are wall-clock timings.
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

from scatter_width_scan import entries  # noqa: E402

L_MM = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
H_OUT = 0.0125


def cmd_mc(a):
    from chain_quadtree import MCSizeKernel
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    a.out = os.path.abspath(a.out)
    os.chdir(et)
    from material_data import build_library, named_material
    m, h = named_material("Water")
    K = MCSizeKernel(build_library([m], [h]), 1e-3, seed=a.seed)
    y, u, v, w = entries(2000, 1)
    K.boundary(y, u, v, w, np.full(2000, a.E), L=0.1)              # numba warm-up
    res = {}
    for Lmm in L_MM:
        y, u, v, w = entries(a.n_mc, a.seed + int(10 * Lmm))
        t0 = time.time()
        o = K.boundary(y, u, v, w, np.full(a.n_mc, a.E), L=Lmm / 10)
        dt = time.time() - t0
        res[str(Lmm)] = {"us_per_transit": 1e6 * dt / a.n_mc, "far_exit": float(np.mean(o["cls"] == 1)),
                         "secondaries_out_per_transit": float(len(o["extra_exits"]["E"]) / a.n_mc)}
        print(f"MC   L {Lmm:6.3f} mm: {1e6 * dt / a.n_mc:9.2f} us/transit", flush=True)
    json.dump({"E": a.E, "n": a.n_mc, "res": res}, open(a.out, "w"), indent=1)


def cmd_gmc(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, auto_samples, cond_features, to_device
    from chain_quadtree import MirrorY
    dev = "cuda"
    gm = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed)
    K = MirrorY(gm, a.seed + 3)
    spec, model, _ = load_kernel(a.kernel, dev)
    y, u, v, w = entries(a.n, 1)
    K.boundary(y, u, v, w, np.full(a.n, a.E), L=0.1)               # CUDA warm-up
    res = {}
    for Lmm in L_MM:
        L = Lmm / 10
        y, u, v, w = entries(a.n, a.seed + int(10 * Lmm))
        E = np.full(a.n, a.E)
        torch.cuda.synchronize()
        t0 = time.time()
        o = K.boundary(y, u, v, w, E, L=L)
        D = o.get("daughters")
        nd = 0
        while D is not None and len(D["E"]):
            nd += len(D["E"])
            o2 = K.interior(D["x"], D["y"], D["u"], D["v"], D["w"], D["E"], L=L)
            D = o2.get("daughters")
        torch.cuda.synchronize()
        t_gmc = time.time() - t0
        # SCR on the boundary transits, Drumm settings
        n_sub = max(1, int(round(L / H_OUT)))
        Ks = max(a.K_min, int(a.K_scale * auto_samples(n_sub)))
        cls = np.asarray(o["cls"])
        R = {"start": np.zeros(a.n), "x0": np.zeros(a.n), "y0": y, "u0": u, "v0": v, "E0": E, "cls": cls,
             "x1": o["x"], "y1": o["y"], "u1": o["u"], "v1": o["v"], "E_out": np.where(cls == 4, 0.0, o["E_out"]),
             "dose": o["dose"], "n_rel": np.zeros(a.n), "flag": np.zeros(a.n), "L": np.full(a.n, L)}
        Rd = to_device(R, dev)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            step = max(1, (1 << 20) // Ks)
            for i in range(0, a.n, step):
                rr = {k: v_[i:i + step] for k, v_ in Rd.items()}
                model.sample(model.context(cond_features(spec, rr)), Ks, base="sobol")
        torch.cuda.synchronize()
        t_scr = time.time() - t0
        res[str(Lmm)] = {"gmc_us_per_transit": 1e6 * t_gmc / a.n, "scr_us_per_transit": 1e6 * t_scr / a.n,
                         "scr_samples": Ks, "daughters_per_transit": nd / a.n}
        print(f"GMC  L {Lmm:6.3f} mm: GMC {1e6 * t_gmc / a.n:7.3f} us/transit ({nd / a.n:.3f} daughters), "
              f"SCR {1e6 * t_scr / a.n:7.3f} us/transit (K {Ks})", flush=True)
    json.dump({"E": a.E, "n": a.n, "model": a.model, "kernel": a.kernel, "res": res}, open(a.out, "w"), indent=1)


def cmd_plot(a):
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    mc, g = json.load(open(a.mc)), json.load(open(a.gmc))
    L = np.array([float(k) for k in g["res"]])
    t_mc = np.array([mc["res"][k]["us_per_transit"] for k in g["res"]])
    t_g = np.array([g["res"][k]["gmc_us_per_transit"] for k in g["res"]])
    t_s = np.array([g["res"][k]["scr_us_per_transit"] for k in g["res"]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for y_, lab, c in ((t_mc, "analog MC (CPU, all threads)", "0.15"), (t_g, "GMC transit (GPU)", "#2a78d6"),
                       (t_s, "SCR placement, Drumm K (GPU)", "#1baf7a"), (t_g + t_s, "GMC + SCR", "#eb6834")):
        ax[0].plot(L, y_, "o-", color=c, lw=2, ms=6, label=lab)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("cell side L (mm)")
    ax[0].set_ylabel("µs per cell transit")
    ax[0].set_title(f"Cost of one transit, {g['E']:g} MeV water, near-normal entries", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(L, t_mc / (t_g + t_s), "o-", color="#eb6834", lw=2, ms=6, label="MC / (GMC + SCR)")
    ax[1].plot(L, t_mc / t_g, "o--", color="#2a78d6", lw=1.5, ms=5, label="MC / GMC (no SCR)")
    ax[1].axhline(1, color="0.5", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("cell side L (mm)")
    ax[1].set_ylabel("speed-up per transit")
    ax[1].set_title("Per-transit speed-up vs cell size", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.savefig(a.png, dpi=130)
    print(f"{'L mm':>7} {'MC us':>9} {'GMC us':>8} {'SCR us':>8} {'speed-up':>9}")
    for i in range(len(L)):
        print(f"{L[i]:7.3f} {t_mc[i]:9.2f} {t_g[i]:8.3f} {t_s[i]:8.3f} {t_mc[i] / (t_g[i] + t_s[i]):9.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mc")
    p.add_argument("--out", required=True)
    p.add_argument("--E", type=float, default=19.0)
    p.add_argument("--n_mc", type=int, default=20000)
    p.add_argument("--seed", type=int, default=5)
    g = sub.add_parser("gmc")
    g.add_argument("--out", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--kernel", default=str(ROOT / "ml/models/size_scr_water_100um-1cm_30MeV_v2"))
    g.add_argument("--E", type=float, default=19.0)
    g.add_argument("--n", type=int, default=100000)
    g.add_argument("--ode_steps", type=int, default=20)
    g.add_argument("--K_min", type=int, default=8)
    g.add_argument("--K_scale", type=float, default=0.25)
    g.add_argument("--seed", type=int, default=5)
    q = sub.add_parser("plot")
    q.add_argument("--mc", required=True)
    q.add_argument("--gmc", required=True)
    q.add_argument("--png", required=True)
    a = ap.parse_args()
    {"mc": cmd_mc, "gmc": cmd_gmc, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
