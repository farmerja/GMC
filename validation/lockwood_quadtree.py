"""
Lockwood / Sandia depth-dose (lockwood_slab.py: semi-infinite Al and Cu, SAND79-0414) with SIZE-CONDITIONED GMC + SCR
on square-cell meshes of any cell sizes (validation/chain_quadtree.py), vs the analog MC with ESTAR stopping.

Geometry as lockwood_slab.py gmc: a box of material (depth W >= 1.4 R, height H >= 2.5 R, R = 1.1 x the CSDA range at
the run energy), the beam entering mid-height of the left face at x = 0 (0 deg, or tilted 60 deg into the infinite z);
anything leaving the box is gone. Photons escape (the MC "escape" convention). The SCR tally is summed over y into a
depth profile, J in MeV cm^2/g per electron.

Meshes (--mesh):
  one             FEWEST CELLS: a single square of side 2.5 R (at least L_lo, at most L_hi = the model's range) with
                  the beam at the centre of its left face; the whole depth-dose comes from the SCR of one cell.
  uniform:<L um>  uniform squares (the box rounded up to whole cells, odd row count)
  random:<seed>   squares of random sides on an L_lo lattice (log-uniform k * L_lo, k = 1 .. R / L_lo, capped at
                  L_hi), placed at the first free lattice point, shrunk until they fit; odd lattice height so the beam
                  never enters on a cell edge

  run      .venv/bin/python validation/lockwood_quadtree.py run --mesh one --model_al <gmc.pth> --kernel_al <dir> ...
  compare  profiles, GMC/MC ratios with the combined 1-sigma noise, totals, timing; figures + summary.json
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_quadtree import QuadMesh, chain_quadtree, scr_tally_quadtree, MirrorY  # noqa: E402
from lockwood_slab import load_cases, find_run, OUT as LW_OUT  # noqa: E402

OUT = LW_OUT / "size"
L_LO, L_HI = 0.005, 0.6          # model cell-size range (cm): 50 um - 6 mm
RHO = {"Al": 2.699, "Cu": 8.96}


def case_range(c, E, rho):
    return 1.1 * c["csda_range_exp"] * (E / c["E_exp"]) ** 1.7 / rho          # cm, margin for our longer range


def mesh_one(R):
    S = min(max(2.5 * R, L_LO), L_HI)
    return QuadMesh([0.0], [0.0], [S], [0], (S, S), S)


def mesh_uniform(R, L):
    nx = max(1, math.ceil(1.4 * R / L))
    ny = max(1, math.ceil(2.5 * R / L))
    ny += 1 - ny % 2
    gx, gy = np.meshgrid(np.arange(nx) * L, np.arange(ny) * L, indexing="ij")
    return QuadMesh(gx.ravel(), gy.ravel(), np.full(nx * ny, L), np.zeros(nx * ny, int), (nx * L, ny * L), L)


def mesh_random(R, seed, h=L_LO):
    """Random square tiling on an h lattice: at the first free lattice point (x-major), a square of side k * h with k
    log-uniform in [1, k_max], shrunk until it fits (free and inside the box)."""
    rng = np.random.default_rng(seed)
    nx = max(1, math.ceil(1.4 * R / h))
    ny = max(1, math.ceil(2.5 * R / h))
    ny += 1 - ny % 2
    k_max = max(1, min(int(round(R / h)), int(L_HI / h), nx, ny))
    free = np.ones((nx, ny), bool)
    cells = []
    for i in range(nx):
        for j in range(ny):
            if not free[i, j]:
                continue
            k = int(np.clip(np.floor(np.exp(rng.uniform(0.0, np.log(k_max + 1)))), 1, k_max))
            while k > 1 and (i + k > nx or j + k > ny or not free[i:i + k, j:j + k].all()):
                k -= 1
            free[i:i + k, j:j + k] = False
            cells.append((i, j, k))
    c = np.array(cells, float)
    return QuadMesh(c[:, 0] * h, c[:, 1] * h, c[:, 2] * h, np.zeros(len(c), int), (nx * h, ny * h), h)


def make_mesh(spec, R):
    kind, *v = spec.split(":")
    if kind in ("one", "fewest"):
        return mesh_one(R)
    if kind == "uniform":
        return mesh_uniform(R, float(v[0]) * 1e-4)
    if kind == "random":
        return mesh_random(R, int(v[0]))
    raise ValueError(spec)


def tag(spec):
    return spec.replace(":", "_").replace(".", "p")


def run_path(key, E, spec, N, sfx=""):
    return OUT / f"gmc_{key}_{E:g}MeV_{tag(spec)}_{N // 1000}k{sfx}.npz"


# ----------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, auto_samples
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    cases = load_cases()
    keys = a.cases or [k for k, c in cases.items() if not c.get("missing")]
    models = {"Al": (a.model_al, a.kernel_al), "Cu": (a.model_cu, a.kernel_cu)}
    for key in keys:
        c = cases[key]
        model, kpath = models[c["material"]]
        if model is None:
            continue
        rho = RHO[c["material"]]
        E = c["E_exp"]
        out = Path(a.out) if a.out else run_path(key, E, a.mesh, a.N, a.tag)
        if out.exists() and not a.force:
            print(f"{key}: {out.name} exists, skipped", flush=True)
            continue
        R = case_range(c, E, rho)
        mesh = make_mesh(a.mesh, R)
        W, H = mesh.box
        # depth bins: the lattice cut into m parts, at most R / 40 wide
        m = max(1, math.ceil(mesh.h / (R / a.bins_per_range)))
        h_out = mesh.h / m
        spec_k, km, _ = load_kernel(kpath, dev)
        kern = {0: (spec_k, km)}
        samples = (lambda n: max(a.K_min, int(a.K_scale * auto_samples(n))))
        th = np.radians(c["angle"])
        src = np.zeros((a.N, 8))
        src[:, 1] = 0.5 * H
        src[:, 2], src[:, 7] = np.cos(th), np.sin(th)
        src[:, 4] = E
        prof, cellE, tc, tt, ntr = [], [], [], [], []
        nxo = int(round(W / h_out))
        for s in range(a.n_seeds):
            seed = a.seed + 100 * s
            p = np.zeros(nxo)
            cd = np.zeros(mesh.n_cells)
            t_c = t_t = 0.0
            n_t = 0
            gen = torch.Generator(device=dev)
            for i, b in enumerate(range(0, a.N, a.batch)):
                g = SecondaryGMC(model, dev, ode_steps=a.ode_steps, seed=seed + 7 * i)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                ch = chain_quadtree([MirrorY(g, seed + 7 * i + 3)], src[b:b + a.batch], mesh, a.E_cutoff,
                                    transit_log=True)
                t_c += time.time() - t0
                cd += ch["dose"]
                n_t += len(ch["transits"]["E0"])
                gen.manual_seed(seed + 5 + 13 * i)
                t0 = time.time()
                f, _ = scr_tally_quadtree(ch, mesh, kern, h_out, device=dev, gen=gen, samples=samples)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t_t += time.time() - t0
                p += f.sum(1)
                del ch
            prof.append(p / a.N)
            cellE.append(cd / a.N)
            tc.append(t_c)
            tt.append(t_t)
            ntr.append(n_t)
            print(f"{key} @ {E:g} MeV, {a.mesh}: {mesh.n_cells} cells (sides {1e4 * mesh.L.min():.0f}-"
                  f"{1e4 * mesh.L.max():.0f} um), box {1e4 * W:.0f} x {1e4 * H:.0f} um; seed {seed}: chain {t_c:.1f} s, "
                  f"SCR {t_t:.1f} s ({1e3 * (t_c + t_t) / a.N:.4f} ms/primary, {n_t / a.N:.2f} transits/primary); "
                  f"deposited {p.sum() / (a.N * E):.4f}", flush=True)
        np.savez_compressed(out, prof=np.array(prof), cell=np.array(cellE), edges=np.arange(nxo + 1) * h_out * rho,
                            x0=mesh.x0, y0=mesh.y0, L=mesh.L, box=np.array(mesh.box), rho=rho, E=E, N=a.N,
                            meta=json.dumps({**vars(a), "case": key, "E": E, "R_cm": R, "h_out": h_out,
                                             "n_cells": mesh.n_cells, "chain_s": tc, "tally_s": tt, "transits": ntr}))
        print(f"  saved {out}")


# ----------------------------------------------------------------------------------------------------------- compare
def rebin_mc(run, edges):
    """MC escape-convention J (MeV cm^2/g) averaged over the given g/cm^2 bins, with its batch 1-sigma error."""
    lo, hi = edges[:-1], edges[1:]
    return run.window("escape", lo, hi)


def coarse_edges(edges, R_g, n_bins):
    """Merge the run's depth bins into ~n_bins over one range (whole fine bins)."""
    w = edges[1] - edges[0]
    k = max(1, int(round(R_g / n_bins / w)))
    return edges[::k], k


def load_gmc(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    if "prof" not in z.files:
        # lockwood_slab.py gmc output (the dedicated fixed-size 100 um models of 2026-11-17): uniform grid of
        # nx x ny cells of side L, SCR depth profile with n sub-bins per cell
        L, n, rho = float(z["L"]), int(z["n"]), float(z["rho"])
        c = load_cases()[meta["case"]]
        meta = {**meta, "mesh": f"uniform:{1e4 * L:.0f}um, 100um model", "n_cells": int(np.prod(meta["grid"])),
                "R_cm": case_range(c, meta["E"], rho)}
        z = {"prof": z["scr"], "edges": np.arange(z["scr"].shape[1] + 1) * L * rho / n, "rho": rho}
    return z, meta


def score_run(z, meta, mc, R_g, n_bins=25, min_frac=0.1):
    """GMC depth profile on ~R / n_bins bins vs the MC reference: (depth / R, GMC J, MC J, combined 1-sigma, scored
    mask, stats). Scored bins: MC J > min_frac x its maximum."""
    ed, k = coarse_edges(z["edges"], R_g, n_bins)
    P = z["prof"]
    nb = (len(z["edges"]) - 1) // k
    Pc = P[:, :nb * k].reshape(len(P), nb, k).sum(2)
    ed = ed[:nb + 1]
    dz = np.diff(ed)
    g = Pc.mean(0) / dz
    g_err = (Pc.std(0, ddof=1) / np.sqrt(len(Pc)) / dz) if len(Pc) > 1 else np.zeros_like(g)
    inside = ed[1:] <= mc.edges[-1] + 1e-12            # the MC slab is 1.25 ranges deep, the GMC box 1.4
    ed = ed[:int(inside.sum()) + 1]
    g, g_err = g[:len(ed) - 1], g_err[:len(ed) - 1]
    m_v, m_e = rebin_mc(mc, ed)
    xm = 0.5 * (ed[:-1] + ed[1:]) / R_g
    use = m_v > min_frac * m_v.max()
    sig = np.hypot(m_e, g_err)
    zsc = (g - m_v)[use] / np.maximum(sig[use], 1e-300)
    ratio = g[use] / m_v[use]
    t = {"mesh": meta["mesh"], "n_cells": meta["n_cells"],
         "gmc_ms_per_primary": 1e3 * (np.mean(meta["chain_s"]) + np.mean(meta["tally_s"])) / meta["N"],
         "chain_ms_per_primary": 1e3 * np.mean(meta["chain_s"]) / meta["N"],
         "tally_ms_per_primary": 1e3 * np.mean(meta["tally_s"]) / meta["N"],
         "transits_per_primary": float(np.mean(meta["transits"])) / meta["N"],
         "K_min": meta.get("K_min"), "K_scale": meta.get("K_scale"),
         "rms_rel": float(np.sqrt(np.mean((ratio - 1) ** 2))),
         "rms_noise_rel": float(np.sqrt(np.mean((sig[use] / m_v[use]) ** 2))),
         "chi2_dof": float(np.mean(zsc ** 2)), "frac_within_2sigma": float(np.mean(np.abs(zsc) < 2)),
         "max_abs_z": float(np.abs(zsc).max()), "n_points": int(use.sum())}
    return xm, g, m_v, sig, use, t


def cmd_compare(a):
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    cases = load_cases()
    files = sorted(set(sum((glob.glob(p) for p in a.runs), [])))
    by_case = {}
    for f in files:
        z, meta = load_gmc(f)
        by_case.setdefault(meta["case"], []).append((f, z, meta))
    summary = {}
    keys = [k for k in cases if k in by_case]
    fig, axs = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7.2), squeeze=False,
                            gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#7048e8", "#e64980", "#f59f00", "#5c5c5c"]
    for j, key in enumerate(keys):
        c = cases[key]
        runs = sorted(by_case[key], key=lambda t: t[2]["mesh"])
        E = runs[0][2]["E"]
        mc = find_run(key, E, 200000, 79, "estar")
        if mc is None:
            print(f"{key}: no ESTAR MC reference at {E} MeV")
            continue
        rho = float(runs[0][1]["rho"])
        R_g = runs[0][2]["R_cm"] * rho / 1.1
        mc_ms = 1e3 * mc.meta["seconds"] / mc.meta["N"]
        summary[key] = {"E": E, "mc_ms_per_primary": mc_ms, "mc_total": mc.total("escape")[0], "runs": {}}
        ax, axr = axs[0, j], axs[1, j]
        drawn_mc = False
        for (f, z, meta), col in zip(runs, cols):
            xm, g, m_v, sig, use, t = score_run(z, meta, mc, R_g, a.n_bins, a.min_frac)
            ratio = g[use] / m_v[use]
            t["total_ratio"] = float(z["prof"].sum(1).mean()) / summary[key]["mc_total"]
            t["speedup_vs_mc"] = mc_ms / t["gmc_ms_per_primary"]
            summary[key]["runs"][Path(f).stem] = t
            if not drawn_mc:
                ax.step(xm, m_v, where="mid", color="0.15", lw=1.6, label="analog MC (ESTAR)")
                drawn_mc = True
            ax.step(xm, g, where="mid", color=col, lw=1.1, label=f"{meta['mesh']} ({meta['n_cells']} cells)")
            axr.errorbar(xm[use], ratio, yerr=sig[use] / m_v[use], fmt="o", ms=3, color=col, lw=0.8)
        axr.axhline(1.0, color="0.5", lw=0.8)
        ax.set_title(f"{key.replace('_', ' ')} deg, {E:g} MeV", fontsize=10)
        ax.set_ylabel("J (MeV cm²/g per e⁻)")
        axr.set_xlabel("depth / CSDA range")
        axr.set_ylabel("GMC / MC")
        axr.set_ylim(0.85, 1.15)
        ax.legend(fontsize=7)
    fig.suptitle("Lockwood depth-dose: size-conditioned GMC + SCR on different meshes vs analog MC "
                 "(error bars: combined 1-sigma)", fontweight="bold")
    os.makedirs(a.out_dir, exist_ok=True)
    fig.savefig(Path(a.out_dir) / "lockwood_meshes.png", dpi=120)
    plt.close(fig)
    with open(Path(a.out_dir) / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{'case':>10} {'mesh':>12} {'cells':>6} {'tr/prim':>7} {'ms/prim':>8} {'speedup':>8} {'total':>7} "
          f"{'rms rel':>8} {'noise':>7} {'chi2/dof':>8} {'|z|<2':>6}")
    for key, s in summary.items():
        for n_, t in s["runs"].items():
            print(f"{key:>10} {t['mesh']:>12} {t['n_cells']:6d} {t['transits_per_primary']:7.2f} "
                  f"{t['gmc_ms_per_primary']:8.4f} {t['speedup_vs_mc']:8.1f} {t['total_ratio']:7.4f} "
                  f"{t['rms_rel']:8.4f} {t['rms_noise_rel']:7.4f} {t['chi2_dof']:8.2f} {t['frac_within_2sigma']:6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mesh", required=True)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--model_al")
    p.add_argument("--kernel_al")
    p.add_argument("--model_cu")
    p.add_argument("--kernel_cu")
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=100000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--ode_steps", type=int, default=20)
    p.add_argument("--bins_per_range", type=float, default=40.0)
    p.add_argument("--K_scale", type=float, default=1.0)
    p.add_argument("--K_min", type=int, default=32)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="rerun cases whose output exists")
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--n_bins", type=int, default=25, help="depth bins per CSDA range for the comparison")
    c.add_argument("--min_frac", type=float, default=0.1, help="score bins with MC J > this x its maximum")
    c.add_argument("--out_dir", default=str(OUT))
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_compare)(a)


if __name__ == "__main__":
    main()
