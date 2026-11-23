"""
Drumm water-cylinder benchmark (drumm_cylinder.py: 20 MeV beam on a 2 cm water cylinder, ITS Figs. 8a) with the
SIZE-CONDITIONED GMC + SCR on meshes of different cell sizes, including quadtrees that refine only where the circle
cuts the cells (fine staircase at the surface, large cells inside and in the vacuum around it).

  run      one mesh: chained GMC (validation/chain_quadtree.py, MirrorY, photon energy kept in its transit as in the
           paper) + SCR tally on a 125 um grid, several seeds, timings. Both energy conventions from one tally:
           "loc" (photon energy deposited in the emitting transit, the paper's) and "esc" (photons leave).
  compare  runs vs the analog MC on the TRUE circle (two seeds: bias / noise split, drumm_cylinder conventions) and
           ITS: whole-map scores, profiles, timing; figures.

Meshes (--mesh): "uniform:<L mm>"; "quad:<L_max mm>:<L_min mm>" (uniform L_max tiling, cells cut by the circle
split down to L_min; leaf material = water if its centre is inside the circle); "centre:<L_c mm>:<L_min mm>[:<grade>]"
(one L_c square on the axis, graded squares shrinking towards the surface, see centre_mesh).

    python validation/drumm_quadtree.py run --mesh quad:4:0.125 --model ml/models/size_gmc_water_100um-1cm_30MeV/secondary_gmc.pth \
        --kernel ml/models/size_scr_water_100um-1cm_30MeV
    python validation/drumm_quadtree.py compare --runs validation/results/drumm_size/*.npz
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_quadtree import QuadMesh, chain_quadtree, scr_tally_quadtree, VacuumKernel, LocalPhotons, MirrorY  # noqa
from drumm_cylinder import triangle_source, E0, R  # noqa: E402
from scr_chain_compare import noise_split  # noqa: E402

OUT = ROOT / "validation/results/drumm_size"
DRUMM = ROOT / "validation/results/drumm_water_cylinder"
BOX = (2.0, 2.0)
H_OUT = 0.0125                      # SCR / MC tally bin (cm): 8 per mm, as the drumm_cylinder MC references
MC_REFS = ("mc_circle_n8_500k_s123_estar.npz", "mc_circle_n8_500k_s4123_src2011_estar.npz")


def material(xc, yc):
    return (np.hypot(xc - 1.0, yc - 1.0) >= R).astype(np.int64)      # 0 water, 1 vacuum


def cut_by_circle(x0, y0, L):
    dx = np.maximum(np.maximum(1.0 - (x0 + L), x0 - 1.0), 0.0)
    dy = np.maximum(np.maximum(1.0 - (y0 + L), y0 - 1.0), 0.0)
    dmin = np.hypot(dx, dy)
    dmax = np.hypot(np.maximum(np.abs(x0 - 1.0), np.abs(x0 + L - 1.0)), np.maximum(np.abs(y0 - 1.0),
                                                                                   np.abs(y0 + L - 1.0)))
    return (dmin < R) & (dmax > R)


def dist_to_surface(x0, y0, L):
    """Distance from a cell lying fully inside the circle to the circle (its farthest corner to the surface)."""
    dmax = np.hypot(np.maximum(np.abs(x0 - 1.0), np.abs(x0 + L - 1.0)), np.maximum(np.abs(y0 - 1.0),
                                                                                   np.abs(y0 + L - 1.0)))
    return R - dmax


def centre_mesh(L_c, L_min, grade=1.0):
    """One L_c square centred on the cylinder axis, the rest of the box tiled with L_c / 2 squares that are halved
    while the circle cuts them or (water cells) while they are larger than grade x their distance to the surface, down
    to L_min (the smallest L_c / 2^k >= L_min): cells get smaller and smaller towards the surface."""
    base = L_c / 2
    n = int(round(BOX[0] / base))
    if abs(n * base - BOX[0]) > 1e-9 or abs((BOX[0] - L_c) / 2 / base - round((BOX[0] - L_c) / 2 / base)) > 1e-9:
        raise ValueError(f"centre cell {L_c} cm does not align with a {base} cm tiling of the box")
    gx, gy = np.meshgrid(np.arange(n) * base, np.arange(n) * base, indexing="ij")
    x0, y0 = gx.ravel(), gy.ravel()
    lo = (BOX[0] - L_c) / 2
    centre = (x0 >= lo - 1e-9) & (x0 < lo + L_c - 1e-9) & (y0 >= lo - 1e-9) & (y0 < lo + L_c - 1e-9)
    done = [(np.array([lo]), np.array([lo]), np.array([L_c]))]
    x0, y0, L = x0[~centre], y0[~centre], np.full(int((~centre).sum()), base)
    while len(L):
        cut = cut_by_circle(x0, y0, L)
        inside = ~cut & (np.hypot(x0 + L / 2 - 1.0, y0 + L / 2 - 1.0) < R)
        split = (cut | (inside & (L > grade * dist_to_surface(x0, y0, L)))) & (L / 2 >= L_min * (1 - 1e-9))
        done.append((x0[~split], y0[~split], L[~split]))
        x0, y0, L = x0[split], y0[split], L[split] / 2
        x0, y0 = np.concatenate([x0, x0 + L, x0, x0 + L]), np.concatenate([y0, y0, y0 + L, y0 + L])
        L = np.tile(L, 4)
    x0, y0, L = (np.concatenate(v) for v in zip(*done))
    return QuadMesh(x0, y0, L, material(x0 + L / 2, y0 + L / 2), BOX, L.min())


def sym_mesh(L_max, h, L_vac=None, order="x"):
    """Fewest-cells tiling, mirror-symmetric about the beam axis (y = 1) but not in x: squares of any side k * h
    (corners on the h lattice; the chain's point location and the size-conditioned models do not need quadtree
    alignment or power-of-two sides). The water / vacuum interface is the h staircase of quad:<..>:h (lattice cell =
    water iff its centre is inside the circle); every square is homogeneous in it (water: side <= L_max, vacuum: side
    <= L_vac, default unlimited), so the surface geometry is exactly that of the h quadtree with far fewer cells.
    Greedy, largest square first; a square either straddles the axis centred on it or
    is placed together with its mirror image. Ties at one side: order 'x' takes the smallest x first (then the
    nearest to the axis), 'axis' the nearest to the axis first."""
    n = int(round(BOX[0] / h))
    if abs(n * h - BOX[0]) > 1e-9 or n % 2:
        raise ValueError(f"h = {h} cm must divide the box into an even number of cells")
    c_ = (np.arange(n) + 0.5) * h
    kind = material(c_[:, None], c_[None, :])                   # the h staircase: 0 water, 1 vacuum per lattice cell
    free = np.ones((n, n), bool)
    cells = []
    caps = {0: int(round(L_max / h)), 1: n if L_vac is None else int(round(L_vac / h))}

    def window_ok(ok, k):
        """ok[i, j] for all k x k windows starting at (i, j) that are entirely True."""
        c = np.zeros((n + 1, n + 1), np.int64)
        c[1:, 1:] = ok.cumsum(0).cumsum(1)
        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        return s == k * k

    def place(i, j, k, m):
        free[i:i + k, j:j + k] = False
        cells.append((i, j, k, m))

    for k in range(max(caps.values()), 1, -1):
        for m in (0, 1):
            if k > caps[m]:
                continue
            while True:
                ok = window_ok(free & (kind == m), k)
                ii, jj = np.nonzero(ok)
                if not len(ii):
                    break
                # symmetric placement: straddle the axis centred (j + k/2 = n/2) or stay on one side of it
                centred = 2 * jj + k == n
                one_side = (jj + k <= n // 2) | (jj >= n // 2)
                keep = centred | one_side
                ii, jj, centred = ii[keep], jj[keep], centred[keep]
                if not len(ii):
                    break
                dist_axis = np.abs(2 * jj + k - n)
                key = (np.lexsort((dist_axis, ii, ~centred)) if order == "x"
                       else np.lexsort((ii, dist_axis, ~centred)))
                i, j = int(ii[key[0]]), int(jj[key[0]])
                place(i, j, k, m)
                jm = n - j - k
                if jm != j:
                    place(i, jm, k, m)
    # unit cells: everything left (cut cells and the leftovers of the other kinds)
    ii, jj = np.nonzero(free)
    for i, j in zip(ii, jj):
        cells.append((int(i), int(j), 1, -1))
    c = np.array(cells, np.float64)
    x0, y0, L = c[:, 0] * h, c[:, 1] * h, c[:, 2] * h
    return QuadMesh(x0, y0, L, material(x0 + L / 2, y0 + L / 2), BOX, h)


def make_mesh(spec):
    kind, *v = spec.split(":")
    if kind == "sym":
        return sym_mesh(float(v[0]) / 10, float(v[1]) / 10, order=v[2] if len(v) > 2 else "x")
    if kind == "centre":
        return centre_mesh(float(v[0]) / 10, float(v[1]) / 10, float(v[2]) if len(v) > 2 else 1.0)
    if kind == "uniform":
        return QuadMesh.uniform(BOX, float(v[0]) / 10, material)
    if kind == "quad":
        return QuadMesh.quadtree(BOX, float(v[0]) / 10, float(v[1]) / 10, cut_by_circle, material)
    raise ValueError(spec)


def mesh_tag(spec):
    return spec.replace(":", "_").replace(".", "p")


# ----------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    import torch
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, auto_samples
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = make_mesh(a.mesh)
    water = mesh.mat == 0
    print(f"mesh {a.mesh}: {mesh.n_cells} cells ({int(water.sum())} water), sizes "
          + ", ".join(f"{L * 1e4:.0f} um x {int(((mesh.L == L) & water).sum())}" for L in np.unique(mesh.L)),
          flush=True)
    src = triangle_source(a.N, a.src_seed)
    spec, model, _ = load_kernel(a.kernel, dev)
    kern = {0: (spec, model)}
    samples = (lambda n: max(a.K_min, int(a.K_scale * auto_samples(n)))) if a.K is None else (lambda n: a.K)
    gx_n = int(round(BOX[0] / H_OUT))
    res = {k: [] for k in ("loc", "esc", "cell", "chain_s", "tally_s", "transits", "samples")}
    for s in range(a.n_seeds):
        seed = a.seed + 100 * s
        loc, esc = np.zeros((gx_n, gx_n)), np.zeros((gx_n, gx_n))
        cell = np.zeros(mesh.n_cells)
        tc = tt = 0.0
        ntr = nsm = 0
        gen = torch.Generator(device=dev)
        for i, b in enumerate(range(0, a.N, a.batch)):
            k0 = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=seed + 7 * i, calib=a.calib,
                              secondaries=not a.no_secondaries)
            K = [LocalPhotons(MirrorY(k0, seed + 7 * i + 3)), VacuumKernel()]
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            c = chain_quadtree(K, src[b:b + a.batch], mesh, a.E_cutoff, transit_log=True)
            tc += time.time() - t0
            cell += c["dose"]
            ntr += int((c["transits"]["mat"] == 0).sum())
            gen.manual_seed(seed + 5 + 13 * i)
            t0 = time.time()
            # one tally, two conventions from the same samples: dose (photon energy local, the paper's) and
            # dose_cond (the model's own dose: photons escape)
            tr = c["transits"]
            chord = (np.where(tr["cls"] != 4, tr["dose"] - tr["dose_cond"], 0.0) if a.photon_place == "chord"
                     else None)
            f_loc, info = scr_tally_quadtree(c, mesh, kern, H_OUT, device=dev, gen=gen, samples=samples,
                                             extra={"esc": tr["dose_cond"]}, chord=chord)
            if dev == "cuda":
                torch.cuda.synchronize()
            tt += time.time() - t0
            loc += f_loc
            esc += info["extra"]["esc"]
            nsm += info["samples"]
            del c
        res["loc"].append(loc)
        res["esc"].append(esc)
        res["cell"].append(cell)
        res["chain_s"].append(tc)
        res["tally_s"].append(tt)
        res["transits"].append(ntr)
        res["samples"].append(nsm)
        print(f"seed {seed}: chain {tc:.0f} s ({1e3 * tc / a.N:.3f} ms/primary, {ntr / a.N:.2f} water transits/primary)"
              f", SCR tally {tt:.0f} s ({1e3 * tt / a.N:.3f} ms/primary, {nsm / max(ntr, 1):.0f} samples/transit); "
              f"deposited {cell.sum() / (a.N * E0):.4f} of E0", flush=True)
    out = a.out or str(OUT / f"gmc_{mesh_tag(a.mesh)}_{a.N // 1000}k{a.tag}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, loc=np.array(res["loc"]), esc=np.array(res["esc"]), cell=np.array(res["cell"]),
                        x0=mesh.x0, y0=mesh.y0, L=mesh.L, mat=mesh.mat, N=a.N,
                        meta=json.dumps({**vars(a), **{k: res[k] for k in ("chain_s", "tally_s", "transits",
                                                                            "samples")}}))
    print(f"saved {out}")


# ----------------------------------------------------------------------------------------------------------- compare
def load_mc():
    F = [np.load(DRUMM / f) for f in MC_REFS if (DRUMM / f).exists()]
    if len(F) < 2:
        raise FileNotFoundError(f"need both MC references {MC_REFS} in {DRUMM}")
    h2 = H_OUT * H_OUT
    loc = [f["energy"] / (int(f["N"]) * h2) + f["photon"] / (int(f["N"]) * h2) for f in F]
    esc = [f["energy"] / (int(f["N"]) * h2) for f in F]
    secs = [json.loads(str(f["meta"]))["seconds"] / int(f["N"]) for f in F]
    return {"loc": loc, "esc": esc, "ms_per_primary": 1e3 * float(secs[0]), "ms_per_primary_all": secs}


def coarse(d, k):
    return d.reshape(d.shape[0] // k, k, d.shape[1] // k, k).mean((1, 3))


def profile(d, which, pos, strip=1, along_bin=4):
    i = int(round((pos + 1.0) / H_OUT))
    if which == "along":
        p = d[:, i - strip:i + strip].mean(1)
        return np.repeat(p.reshape(-1, along_bin).mean(1), along_bin)
    return d[i - strip:i + strip, :].mean(0)


PANELS = [("along", 0.0, "E_along", "along beam axis (y = 0)"), ("across", -0.5, "E_xm05", "across at x = -0.5"),
          ("across", 0.0, "E_x0", "across at x = 0"), ("across", 0.5, "E_xp05", "across at x = 0.5")]


def score_run(F, mc, its):
    N = int(F["N"])
    h2 = H_OUT * H_OUT
    meta = json.loads(str(F["meta"]))
    G = {c: [d / (N * h2) for d in F[c]] for c in ("loc", "esc")}
    gx = (np.arange(int(round(BOX[0] / H_OUT))) + 0.5) * H_OUT
    inside = np.hypot(gx[:, None] - 1, gx[None, :] - 1) < R
    out = {"mesh": meta["mesh"], "N": N, "n_seeds": len(F["loc"]), "maps": {}, "profiles": {}}
    for conv in ("loc", "esc"):
        m1, m2 = mc[conv]
        hot = inside & (m1 > 0.01 * m1.max())
        out["maps"][conv] = {"fine_125um": noise_split(G[conv], m1, m2, lambda d: d[hot]),
                             "bins_500um": noise_split([coarse(g, 4) for g in G[conv]], coarse(m1, 4), coarse(m2, 4),
                                                       lambda d: d[coarse(hot.astype(float), 4) > 0.99]),
                             "deposited_ratio": float(np.mean([g[inside].sum() for g in G[conv]]) / m1[inside].sum())}
    xf = gx - 1.0
    for which, pos, key, _ in PANELS:
        ref = np.mean([profile(m, which, pos) for m in mc["loc"]], 0)
        g = np.mean([profile(d, which, pos) for d in G["loc"]], 0)
        hot = ref > 0.1 * ref.max()
        it = its[key]
        ix, iy = np.asarray(it["x"]), np.asarray(it["y"])
        core = iy > 0.1 * iy.max()
        out["profiles"][key] = {
            "vs_mc_mean_rel": float(np.mean((g[hot] - ref[hot]) / ref[hot])),
            "vs_mc_rel_rms": float(np.sqrt(np.mean(((g[hot] - ref[hot]) / ref[hot]) ** 2))),
            "gmc_vs_its_mean_rel_core": float(np.mean((np.interp(ix, xf, g)[core] - iy[core]) / iy[core])),
            "mc_vs_its_mean_rel_core": float(np.mean((np.interp(ix, xf, ref)[core] - iy[core]) / iy[core]))}
    out["timing"] = {"chain_ms": 1e3 * float(np.mean(meta["chain_s"])) / N,
                     "tally_ms": 1e3 * float(np.mean(meta["tally_s"])) / N,
                     "water_transits_per_primary": float(np.mean(meta["transits"])) / N,
                     "samples_per_transit": float(np.mean(meta["samples"])) / max(float(np.mean(meta["transits"])), 1)}
    out["timing"]["total_ms"] = out["timing"]["chain_ms"] + out["timing"]["tally_ms"]
    out["timing"]["speedup_vs_mc"] = mc["ms_per_primary"] / out["timing"]["total_ms"]
    out["settings"] = {k: meta.get(k) for k in ("ode_steps", "K", "K_scale", "K_min", "batch", "model", "kernel")}
    return out, G


def cmd_compare(a):
    import matplotlib
    matplotlib.use("Agg")
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    mc = load_mc()
    its = json.load(open(DRUMM / "its_digitized.json"))
    runs = sorted(set(sum((glob.glob(p) for p in a.runs), [])))
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    os.makedirs(out_dir, exist_ok=True)
    scores, maps = {}, {}
    for p in runs:
        name = Path(p).stem
        scores[name], maps[name] = score_run(np.load(p), mc, its)
    # MC self-consistency: seed 2 scored like a run against seed 1 (its 'bias' is ~0 by construction)
    summary = {"mc_ms_per_primary": mc["ms_per_primary"], "runs": scores}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"{'run':>40} {'ms/prim':>8} {'speedup':>8} {'tr/prim':>8} {'loc bias':>9} {'noise':>7} {'mc noise':>8} "
          f"{'0.5mm bias':>10} {'esc bias':>9} {'dep loc':>8} {'along':>7} {'x0 rms':>7}")
    for n_, s in scores.items():
        ml_, me_ = s["maps"]["loc"], s["maps"]["esc"]
        print(f"{n_:>40} {s['timing']['total_ms']:8.3f} {s['timing']['speedup_vs_mc']:8.1f} "
              f"{s['timing']['water_transits_per_primary']:8.2f} {ml_['fine_125um']['bias']:9.4f} "
              f"{ml_['fine_125um']['noise']:7.4f} {ml_['fine_125um']['mc_noise']:8.4f} {ml_['bins_500um']['bias']:10.4f} "
              f"{me_['fine_125um']['bias']:9.4f} {ml_['deposited_ratio']:8.4f} "
              f"{s['profiles']['E_along']['vs_mc_mean_rel']:7.4f} {s['profiles']['E_x0']['vs_mc_rel_rms']:7.4f}")

    # Fig. 8a layout with the selected runs
    sel = a.plot or list(scores)[:4]
    cols = ["#d9480f", "#2f9e44", "#7048e8", "#e64980", "#f59f00"]
    gx = (np.arange(int(round(BOX[0] / H_OUT))) + 0.5) * H_OUT - 1.0
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    for a_, (which, pos, key, title) in zip(ax.ravel(), PANELS):
        it = its[key]
        a_.plot(it["x"], it["y"], color="0.1", lw=2.8, label="ITS/ACCEPT (paper, digitized)")
        ref = np.mean([profile(m, which, pos) for m in mc["loc"]], 0)
        a_.step(gx, ref, where="mid", color="#2a7ab0", lw=1.5, label="analog MC, true circle (2 seeds)")
        for c_, n_ in zip(cols, sel):
            g = np.mean([profile(d, which, pos) for d in maps[n_]["loc"]], 0)
            t = scores[n_]["timing"]
            a_.step(gx, g, where="mid", color=c_, lw=1.2,
                    label=f"GMC + SCR, {scores[n_]['mesh']} ({t['speedup_vs_mc']:.0f}x MC)")
        a_.set_title(title)
        a_.set_xlabel("position (cm)")
        a_.set_ylabel("energy deposition (MeV/cm³ per e⁻/cm)")
        if which == "across":
            a_.set_xlim(-0.6, 0.6)
    ax[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Drumm Fig. 8a: size-conditioned GMC + SCR on different meshes vs analog MC and ITS",
                 fontweight="bold")
    fig.savefig(out_dir / "fig8a_meshes.png", dpi=130)
    plt.close(fig)

    # speed vs accuracy
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for n_, s in scores.items():
        for ax_, key in zip(axs, ("fine_125um", "bins_500um")):
            ax_.plot(s["timing"]["speedup_vs_mc"], 100 * max(s["maps"]["loc"][key]["bias"], 0), "o")
            ax_.annotate(s["mesh"] + (f" ode{s['settings']['ode_steps']}" if s['settings']['ode_steps'] != 20 else "")
                         + (f" K{s['settings']['K']}" if s['settings']['K'] else ""),
                         (s["timing"]["speedup_vs_mc"], 100 * max(s["maps"]["loc"][key]["bias"], 0)), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
    for ax_, key in zip(axs, ("125 um bins", "0.5 mm bins")):
        ax_.set_xlabel("speed-up vs analog MC (GMC chain + SCR tally)")
        ax_.set_ylabel("map bias vs MC, % (noise removed)")
        ax_.set_title(f"Dose map inside the cylinder, {key}")
        ax_.set_xscale("log")
    fig.savefig(out_dir / "speed_vs_bias.png", dpi=130)
    plt.close(fig)

    # meshes + dose maps of the selected runs
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(2, len(sel) + 1, figsize=(4.3 * (len(sel) + 1), 8.6), constrained_layout=True,
                           squeeze=False)
    vmax = mc["loc"][0].max()
    ax[0, 0].imshow(np.maximum(mc["loc"][0].T, 1e-6), origin="lower", extent=[-1, 1, -1, 1],
                    norm=LogNorm(vmax * 1e-3, vmax), cmap=S.SEQ)
    ax[0, 0].set_title("analog MC, true circle")
    for a_ in ax.ravel():
        a_.grid(False)
    ax[1, 0].set_visible(False)
    for j, n_ in enumerate(sel, 1):
        F = np.load(runs[[Path(p).stem for p in runs].index(n_)])
        ax[0, j].imshow(np.maximum(maps[n_]["loc"][0].T, 1e-6), origin="lower", extent=[-1, 1, -1, 1],
                        norm=LogNorm(vmax * 1e-3, vmax), cmap=S.SEQ)
        ax[0, j].set_title(f"GMC + SCR, {scores[n_]['mesh']}")
        for x0, y0, L_, m_ in zip(F["x0"], F["y0"], F["L"], F["mat"]):
            ax[1, j].add_patch(plt.Rectangle((x0 - 1, y0 - 1), L_, L_, fill=m_ == 0, fc="#cfe3f3", ec="0.3",
                                             lw=0.3))
        ax[1, j].add_patch(plt.Circle((0, 0), 1, fill=False, color="#d9480f", lw=1))
        ax[1, j].set_xlim(-1, 1)
        ax[1, j].set_ylim(-1, 1)
        ax[1, j].set_aspect("equal")
        ax[1, j].set_title(f"{len(F['L'])} cells")
    fig.savefig(out_dir / "meshes_dose_maps.png", dpi=120)
    plt.close(fig)
    print(f"wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mesh", required=True)
    p.add_argument("--model", default=str(ROOT / "ml/models/size_gmc_water_100um-1cm_30MeV_sx2/secondary_gmc.pth"))
    p.add_argument("--kernel", default=str(ROOT / "ml/models/size_scr_water_100um-1cm_30MeV_v2"))
    p.add_argument("--calib", default=None, help="daughter calibration .npz (ml/gmc_calib.py)")
    p.add_argument("--no_secondaries", action="store_true",
                   help="primary-only GMC: relevant daughters deposit in their parent cell (diagnostic)")
    p.add_argument("--N", type=int, default=500000)
    p.add_argument("--src_seed", type=int, default=2007)
    p.add_argument("--batch", type=int, default=100000)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--ode_steps", type=int, default=20)
    p.add_argument("--K", type=int, default=None, help="SCR samples per transit (default auto per cell size)")
    p.add_argument("--K_scale", type=float, default=1.0, help="multiplier of the automatic K")
    p.add_argument("--K_min", type=int, default=32)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--photon_place", default="chord", choices=["chord", "kernel"],
                   help="where the local photon energy of an exiting transit goes: along its entry -> exit chord "
                        "(default since 2026-11-22) or with the SCR kernel's electron deposits (the 1 cm cell dip)")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--plot", nargs="*", default=None, help="run names (file stems) for the profile / map figures")
    c.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_compare)(a)


if __name__ == "__main__":
    main()
