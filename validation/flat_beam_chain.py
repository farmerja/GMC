#!/usr/bin/env python3
"""
Flat-beam chain test: does a GMC keep a laterally uniform forward beam uniform inside every cell?

A beam uniform in y (stratified, so no source noise) enters a water slab of NX x NY cells. The true problem is
translation invariant in y, so the density of electrons crossing every vertical cell face is exactly flat; a learned
cell model whose exit position / direction / lateral-exit errors depend on the entry point y0 makes it non-flat. For a
weakly scattering beam (20 MeV in 1 mm water: ~0.075 cell lateral spread per transit) an electron keeps its
within-cell coordinate for many cells, so per-transit errors of ~1e-3 cell accumulate coherently. Found 2026-11-16
as the cause of the Drumm GMC + SCR on-axis dip: +-5 % within-cell modulation at 2-8 mm depth, cell totals exact.

Modes:
    mc       analog chain (MCCellKernel, exact by the Markov property; ElectronTransport venv)
    gmc      GMC chain (GMC venv)
    compare  left-entry y0 pattern per cell for each run (flat = 1 for the truth), and, given an mc run, the
             one-transit check: the GMC re-sampled on the analog chain's own left-entry states, E[dy | y0],
             E[dv | y0] and P(lateral exit | y0) vs the analog outcomes.

Usage:
    (ElectronTransport venv) python ../GMC/validation/flat_beam_chain.py mc --out flat_mc.npz
    (GMC venv)               python validation/flat_beam_chain.py gmc --model ml/models/<gmc>/secondary_gmc.pth \
                                 --out flat_gmc.npz [--mirror]
    (GMC venv)               python validation/flat_beam_chain.py compare --runs flat_mc.npz flat_gmc.npz \
                                 [--transit_model ml/models/<gmc>/secondary_gmc.pth]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

GMC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET_ROOT = os.path.join(os.path.dirname(GMC_ROOT), "ElectronTransport")
sys.path[:0] = [GMC_ROOT, os.path.join(GMC_ROOT, "validation")]
from chain_secondaries import chain_secondaries  # noqa: E402

L = 0.1                 # cm, the water 1 mm models
E_SAVE = 5.0            # MeV: boundary transits of electrons above this are saved (the beam, not knock-ons)
COLS = ("cx", "cy", "face", "y0", "u0", "v0", "w0", "E0", "cls", "x1", "y1", "u1", "v1", "dose")


def source(N, ny, E0, seed):
    src = np.zeros((N, 8))
    src[:, 1] = (np.arange(N) + np.random.default_rng(seed).random(N)) / N * ny * L
    np.random.default_rng(seed + 1).shuffle(src[:, 1])
    src[:, 2], src[:, 4] = 1.0, E0
    return src


def cmd_run(a):
    src = source(a.N, a.ny, a.E0, a.src_seed)
    if a.cmd == "mc":
        sys.path.insert(0, ET_ROOT)
        os.chdir(ET_ROOT)
        from material_data import build_library, named_material
        from chain_secondaries import MCCellKernel
        m, h = named_material("Water")
        K = [MCCellKernel(build_library([m], [h]), L, a.E_cutoff, seed=a.seed, secondaries=True)]
    else:
        from ml.secondary_gmc import SecondaryGMC
        K = [SecondaryGMC(a.model, "cuda", seed=a.seed, ode_steps=a.ode_steps)]
        if a.mirror:
            from drumm_cylinder import MirrorY
            K = [MirrorY(K[0], a.seed + 3)]
    t0 = time.time()
    c = chain_secondaries(K, src, a.nx, a.ny, L, a.E_cutoff, log_every=0, transit_log=True, max_steps=100000,
                          material_map=np.zeros((a.nx, a.ny), np.int64))
    T = c["transits"]
    k = (np.asarray(T["start"]) == 0) & (np.asarray(T["E0"]) > E_SAVE)
    np.savez_compressed(a.out, **{n: np.asarray(T[n])[k] for n in COLS}, dose_map=c["dose"], N=a.N, ny=a.ny,
                        seconds=time.time() - t0, label=a.cmd + ("" if a.cmd == "mc" else f":{a.model}"
                                                                  + (" +MirrorY" if a.mirror else "")))
    print(f"{a.cmd}: {time.time() - t0:.0f} s, {k.sum()} boundary transits saved -> {a.out}")


def entry_patterns(T, ny, cells, nb=8):
    """Left-entry y0 histogram per cell column (middle rows only), normalized to mean 1."""
    mid = (T["cy"] >= 3) & (T["cy"] < ny - 3) & (T["face"] == 0)
    out = {}
    for cx in cells:
        h = np.histogram(T["y0"][mid & (T["cx"] == cx)], np.linspace(0, 1, nb + 1))[0].astype(float)
        out[cx] = h / h.mean()
    return out


def cmd_compare(a):
    np.set_printoptions(precision=3, suppress=True, linewidth=230)
    runs = [dict(np.load(p, allow_pickle=False)) for p in a.runs]
    cells = [int(c) for c in a.cells.split(",")]
    print("Left-entry y0 pattern (8 sub-rows, middle cell rows; the truth is flat = 1). rms = rms deviation from 1;"
          " pure counting noise is about 1 / sqrt(entries per bin).")
    for p, T in zip(a.runs, runs):
        print(f"\n{os.path.basename(p)} ({T['label']})")
        for cx, h in entry_patterns(T, int(T["ny"]), cells).items():
            print(f"  cell {cx + 1:>2}: {h}  rms {np.sqrt(np.mean((h - 1) ** 2)):.4f}")
    if not a.transit_model:
        return
    mc = next((T for T in runs if str(T["label"]) == "mc"), None)
    if mc is None:
        raise ValueError("--transit_model needs an mc run among --runs")
    from ml.secondary_gmc import SecondaryGMC
    m = (mc["face"] == 0) & (mc["cx"] >= 1)
    A = {k: mc[k][m] for k in ("y0", "u0", "v0", "w0", "E0", "cls", "y1", "v1")}
    n = len(A["y0"])
    gm = SecondaryGMC(a.transit_model, "cuda", seed=a.seed, ode_steps=a.ode_steps)
    G = {"cls": np.zeros(n, int), "y1": np.zeros(n), "v1": np.zeros(n)}
    for i in range(0, n, 250000):
        sl = slice(i, i + 250000)
        o = gm.boundary(A["y0"][sl], A["u0"][sl], A["v0"][sl], A["w0"][sl], A["E0"][sl])
        G["cls"][sl], G["y1"][sl], G["v1"][sl] = o["cls"], o["y"], o["v"]
    edges = np.linspace(0, 1, a.bins + 1)
    b = np.digitize(A["y0"], edges) - 1
    print(f"\nOne transit, {os.path.basename(a.transit_model)} re-sampled on the analog chain's {n} left-entry states "
          f"(cells >= 2, E > {E_SAVE} MeV): GMC - analog")
    print("   y0 bin      | E[dy|right] (z)   | E[dv|right] (z)   | P(bottom) an / GMC | P(top) an / GMC")
    zy, zv = [], []
    for k in range(a.bins):
        mm = b == k
        ra, rg = mm & (A["cls"] == 1), mm & (G["cls"] == 1)
        da, dg = A["y1"][ra] - A["y0"][ra], G["y1"][rg] - A["y0"][rg]
        va, vg = A["v1"][ra] - A["v0"][ra], G["v1"][rg] - A["v0"][rg]
        sy = np.hypot(da.std() / np.sqrt(len(da)), dg.std() / np.sqrt(len(dg)))
        sv = np.hypot(va.std() / np.sqrt(len(va)), vg.std() / np.sqrt(len(vg)))
        dy, dv = dg.mean() - da.mean(), vg.mean() - va.mean()
        zy.append(dy / sy)
        zv.append(dv / sv)
        print(f"  {edges[k]:.3f}-{edges[k + 1]:.3f} | {dy:+.5f} ({dy / sy:+5.1f}) | {dv:+.5f} ({dv / sv:+5.1f}) | "
              f"{(A['cls'][mm] == 0).mean():.4f} {(G['cls'][mm] == 0).mean():.4f} | "
              f"{(A['cls'][mm] == 2).mean():.4f} {(G['cls'][mm] == 2).mean():.4f}")
    print(f"  chi2/bin: dy {np.mean(np.square(zy)):.2f}, dv {np.mean(np.square(zv)):.2f} (1 = within noise)")


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    runs = [(lab, dict(np.load(p, allow_pickle=False))) for lab, p in (s.split("=", 1) for s in a.runs)]
    cells = list(range(1, 10))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for lab, T in runs:
        pat = entry_patterns(T, int(T["ny"]), cells)
        rms = [np.sqrt(np.mean((pat[c] - 1) ** 2)) * 100 for c in cells]
        ax[0].plot(np.array(cells) + 1, rms, "o-", label=lab)
        h = pat[a.cell - 1]
        ax[1].step(np.arange(9) / 8, np.r_[h, h[-1]], where="post", label=lab)
    ax[0].set(xlabel="cell along the beam", ylabel="rms deviation from flat (%)",
              title="Within-cell entry density of a uniform 20 MeV beam")
    ax[1].axhline(1, color="0.5", lw=0.8)
    ax[1].set(xlabel="entry position y0 inside the cell (cell units)", ylabel="entry density / mean",
              title=f"Cell {a.cell}: 8 sub-rows (truth = 1)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("mc", "gmc"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--N", type=int, default=300000)
        s.add_argument("--nx", type=int, default=10)
        s.add_argument("--ny", type=int, default=12)
        s.add_argument("--E0", type=float, default=20.0)
        s.add_argument("--E_cutoff", type=float, default=1e-3)
        s.add_argument("--src_seed", type=int, default=5)
        s.add_argument("--seed", type=int, default=77 if name == "mc" else 11)
        if name == "gmc":
            s.add_argument("--model", required=True)
            s.add_argument("--mirror", action="store_true", help="MirrorY symmetrization at inference")
            s.add_argument("--ode_steps", type=int, default=20, help="ODE steps of the GMC flows")
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--cells", default="1,2,3,5,7,9")
    c.add_argument("--transit_model", default=None)
    c.add_argument("--bins", type=int, default=16)
    c.add_argument("--seed", type=int, default=31)
    c.add_argument("--ode_steps", type=int, default=20, help="ODE steps of the exit flow in the one-transit check")
    q = sub.add_parser("plot")
    q.add_argument("--runs", nargs="+", required=True, help="label=path.npz ...")
    q.add_argument("--cell", type=int, default=6)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd in ("mc", "gmc"):
        a.out = os.path.abspath(a.out)
    {"compare": cmd_compare, "plot": cmd_plot}.get(a.cmd, cmd_run)(a)


if __name__ == "__main__":
    main()
