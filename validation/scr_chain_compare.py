"""
SCR-GMC end to end: chained GMC with secondaries + sub-cell deposition kernel vs analog MC on a fine mesh (GMC venv).

    python validation/scr_chain_compare.py \
        --fine_ref validation/results/secondary_chain_cu_100um/mc_ref_mev_fine8.npz \
        --mc_ref validation/results/secondary_chain_cu_100um/mc_ref_mev.npz \
        --model ml/models/secondary_gmc_cu_100um/secondary_gmc.pth \
        --kernel ml/models/subcell_kernel_cu_100um_mom

The fine reference comes from validation/scr_mc_ref.py (same source, n x n fine voxels per transport cell). The GMC chain
runs on the coarse transport cells with transit_log=True; scr_tally (ml/subcell_kernel.py) then places every transit's
dose on the fine mesh. Several tallies (--tallies sampling:K, e.g. random:32,sobol:32,sobol:256) run on the SAME
chains, so they differ only in how the kernel is sampled. Scored against the fine MC at every --eval_n sub-grid size
(coarser ones are exact sums of the fine maps):
    split_fine / split_detail   relative noise of one analog-MC run, noise of one GMC + SCR run (from seed pairs)
                                and the bias left after removing both, for the fine map and for the within-cell
                                detail (fine map minus its cell mean: isolates the kernel from the GMC's cell doses)
    gmc_flat                    each cell's GMC dose spread uniformly (no kernel: what SCR must beat)
Several materials: pass one --model and one --kernel per material
(--kernel none = spread that material's transits uniformly).
Writes scr_metrics_<beam>_fine<n><tag>.json, scr_doses_<beam>_fine<n><tag>.npz and, for the first tally,
scr_maps_<beam>_fine<n><tag>.png and scr_profiles_<beam>_fine<n><tag>.png into --out_dir (default: the reference's directory).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sec_plot_style as S  # noqa: E402
from chain_secondaries import chain_secondaries  # noqa: E402

plt = S.plt


def rel_l2(d, ref):
    return float(np.sqrt(((d - ref) ** 2).sum() / (ref ** 2).sum()))


def coarsen(d, n):
    return d.reshape(d.shape[0] // n, n, d.shape[1] // n, n).sum((1, 3))


def upsample_flat(c, n):
    return np.repeat(np.repeat(c, n, 0), n, 1) / (n * n)


def detail(f, n):
    """Within-cell structure: fine map minus its cell mean."""
    return f - upsample_flat(coarsen(f, n), n)


def noise_split(G, mc, nz, T):
    """Error split for maps G (seeds) vs the MC truth mc, with nz an independent MC run, after transform T:
    relative noise of one MC run, of one G run (from seed pairs) and the bias left after removing both."""
    ss = lambda a: float((a ** 2).sum())  # noqa: E731
    R = ss(T(mc))
    mc_var = ss(T(nz) - T(mc)) / 2 / R
    if len(G) > 1:
        g_var = float(np.mean([ss(T(G[i]) - T(G[j])) for i in range(len(G)) for j in range(i + 1, len(G))])) / 2 / R
    else:
        g_var = float("nan")
    b2 = float(np.mean([ss(T(g) - T(mc)) for g in G])) / R - g_var - mc_var
    return {"mc_noise": float(np.sqrt(mc_var)), "noise": float(np.sqrt(g_var)),
            "bias": float(np.sign(b2) * np.sqrt(abs(b2)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fine_ref", required=True)
    ap.add_argument("--mc_ref", required=True, help="the coarse reference the fine one was made from (source)")
    ap.add_argument("--model", required=True, nargs="+", help="one GMC checkpoint per material")
    ap.add_argument("--kernel", required=True, nargs="+", help="one SCR kernel dir per material, or 'none'")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=("sample", "quad"), default="sample")
    ap.add_argument("--tallies", default="sobol:auto",
                    help="comma list of sampling:K tallies run on the SAME chains, sampling in random|jitter|sobol, "
                         "K an integer or 'auto' (subcell_kernel.auto_samples of the reference n), e.g. "
                         "random:32,sobol:32,sobol:256; the first one is plotted")
    ap.add_argument("--eval_n", default=None,
                    help="comma list of sub-grid sizes to score (each divides the reference n; coarser ones are "
                         "sums of the fine tally and of the fine MC); default: the reference n")
    ap.add_argument("--quad_sub", type=int, default=4)
    ap.add_argument("--ode_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix of the output names")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()
    S.apply()
    import torch
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, scr_tally
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    F = np.load(a.fine_ref)
    ref = np.load(a.mc_ref)
    meta = json.loads(str(ref["meta"]))
    n = int(F["n"])
    eval_n = [int(v) for v in a.eval_n.split(",")] if a.eval_n else [n]
    if any(n % r for r in eval_n):
        raise ValueError(f"--eval_n {eval_n} must divide the reference n = {n}")
    from ml.subcell_kernel import auto_samples
    tallies = [(t.split(":")[0], auto_samples(n) if t.split(":")[1] == "auto" else int(t.split(":")[1]))
               for t in a.tallies.split(",")]
    tnames = [f"{m}{k}" for m, k in tallies]
    src = ref["source"]
    N = len(src)
    nx, ny, L, E_cut = meta["nx"], meta["ny"], meta["cell_size"], meta["E_cutoff"]
    names = meta.get("materials", ["Cu"])
    mmap = ref["material_map"] if "material_map" in ref.files else np.zeros((nx, ny), dtype=np.int64)
    if len(a.model) != len(names) or len(a.kernel) != len(names):
        raise ValueError(f"reference has materials {names}: pass {len(names)} --model and --kernel in that order")
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.mc_ref))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta['beam']}_fine{n}{a.tag}"
    kernels = {}
    for m, k in enumerate(a.kernel):
        if k.lower() != "none":
            spec, model, _ = load_kernel(k, dev)
            if abs(spec.L - L) > 1e-9:
                raise ValueError(f"kernel {k} was trained on L = {spec.L} cm cells, the grid has {L}")
            kernels[m] = (spec, model)

    runs = {t: [] for t in tnames}
    runs["gmc_flat"] = []
    info = []
    gen = torch.Generator(device=dev)
    for s in range(a.n_seeds):
        seed = a.seed + 100 * s
        K = [SecondaryGMC(p, dev, ode_steps=a.ode_steps, seed=seed + 1009 * m) for m, p in enumerate(a.model)]
        t0 = time.time()
        c = chain_secondaries(K, src, nx, ny, L, E_cut, log_every=0, material_map=mmap, transit_log=True)
        t_chain = time.time() - t0
        n_t = len(c["transits"]["E0"])
        row = {"seed": seed, "chain_s": t_chain, "transits": n_t, "tallies": {}}
        msg = []
        for tn, (samp, k) in zip(tnames, tallies):
            gen.manual_seed(seed + 5)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            fine, ti = scr_tally(c, kernels, nx, ny, n, mode=a.mode, n_samples=k, quad_sub=a.quad_sub, device=dev,
                                 gen=gen, sampling=samp)
            if dev == "cuda":
                torch.cuda.synchronize()
            t_tally = time.time() - t0
            closure = float(np.abs(coarsen(fine, n) - c["dose"]).max() / c["dose"].max())
            row["tallies"][tn] = {"tally_s": t_tally, "tally_us_per_transit": 1e6 * t_tally / max(n_t, 1),
                                  "coarse_closure_max_rel": closure, **ti}
            runs[tn].append((fine / N).astype(np.float32))
            msg.append(f"{tn} {1e6 * t_tally / max(n_t, 1):.1f} us/transit (closure {closure:.0e})")
        runs["gmc_flat"].append((upsample_flat(c["dose"], n) / N).astype(np.float32))
        info.append(row)
        print(f"seed {seed}: chain {t_chain:.0f} s, {n_t} transits; " + ", ".join(msg), flush=True)
        del c

    mc = F["dose_fine"] / N
    nz = F["dose_fine_noise"] / N
    M = {"beam": meta["beam"], "E_range_MeV": [meta["E_low"], meta["E_high"]], "N": N, "grid": [nx, ny], "n_ref": n,
         "cell_cm": L, "materials": names, "models": a.model, "kernels": a.kernel, "mode": a.mode,
         "tallies": tnames, "runs": info, "by_n": {}}
    for r in eval_n:
        f_ = n // r
        T = (lambda d, f_=f_: coarsen(d, f_) if f_ > 1 else d)
        mc_r, nz_r = T(mc), T(nz)
        mc_c = coarsen(mc_r, r)
        hot_c = mc_c > 0.01 * mc_c.max()
        hot = np.repeat(np.repeat(hot_c, r, 0), r, 1)
        res = {"fine_um": L * 1e4 / r, "hot_dose_frac": float(mc_c[hot_c].sum() / mc_c.sum())}
        for k in tnames + ["gmc_flat"]:
            G = [T(g.astype(np.float64)) for g in runs[k]]
            res[k] = {"rel_L2_fine": float(np.mean([rel_l2(g, mc_r) for g in G])),
                      "rel_L2_fine_hot": float(np.mean([rel_l2(g[hot], mc_r[hot]) for g in G])),
                      "rel_L2_detail": float(np.mean([rel_l2(detail(g, r), detail(mc_r, r)) for g in G])),
                      "split_fine": noise_split(G, mc_r, nz_r, lambda x: x),
                      "split_detail": noise_split(G, mc_r, nz_r, lambda x, r=r: detail(x, r))}
        res["noise"] = {"rel_L2_fine": rel_l2(nz_r, mc_r), "rel_L2_detail": rel_l2(detail(nz_r, r), detail(mc_r, r))}
        M["by_n"][str(r)] = res
        print(f"\n{r}x{r} sub-cells ({L * 1e4 / r:.3g} um), one run each, relative to the fine MC:")
        print(f"  {'':12s} {'fine noise':>10s} {'fine bias':>10s} {'det noise':>10s} {'det bias':>10s}  "
              f"{'tally us/tr':>11s}")
        sp = res[tnames[0]]["split_fine"]
        sd = res[tnames[0]]["split_detail"]
        print(f"  {'analog MC':12s} {100 * sp['mc_noise']:9.2f}% {'':>10s} {100 * sd['mc_noise']:9.2f}%")
        for k in tnames:
            sp, sd = res[k]["split_fine"], res[k]["split_detail"]
            us = np.mean([row["tallies"][k]["tally_us_per_transit"] for row in info])
            print(f"  {k:12s} {100 * sp['noise']:9.2f}% {100 * sp['bias']:+9.2f}% {100 * sd['noise']:9.2f}% "
                  f"{100 * sd['bias']:+9.2f}%  {us:11.1f}")
        print(f"  {'gmc_flat':12s} rel-L2 fine {100 * res['gmc_flat']['rel_L2_fine']:.1f}%")
    path = os.path.join(out_dir, f"scr_metrics_{tag}.json")
    with open(path, "w") as fh:
        json.dump(M, fh, indent=1, default=float)
    np.savez_compressed(os.path.join(out_dir, f"scr_doses_{tag}.npz"), mc=mc.astype(np.float32),
                        noise=nz.astype(np.float32), **{k: np.array(v) for k, v in runs.items()})
    print(f"saved {path}")
    if not a.no_plots:
        hot_c = coarsen(mc, n) > 0.01 * coarsen(mc, n).max()
        plots(out_dir, tag, mc, nz, runs[tnames[0]][0].astype(np.float64), runs["gmc_flat"][0].astype(np.float64),
              n, nx, ny, L, hot_c, M)


def plots(out_dir, tag, mc, noise, scr, flat, n, nx, ny, L, hot_c, M):
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    um = L * 1e4 / n
    ix = np.flatnonzero(hot_c.any(1))
    iy = np.flatnonzero(hot_c.any(0))
    x0, x1 = ix.min() * n, (ix.max() + 1) * n
    y0, y1 = max(0, (iy.min() - 1) * n), min(ny * n, (iy.max() + 2) * n)
    crop = lambda d: d[x0:x1, y0:y1].T  # noqa: E731
    ext = [x0 * um, x1 * um, y0 * um, y1 * um]
    vmax = mc.max()
    norm = LogNorm(vmax * 1e-4, vmax)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for a_, d, t in ((ax[0, 0], mc, "analog MC (fine tally)"), (ax[0, 1], scr, "GMC + SCR kernel"),
                     (ax[0, 2], flat, "GMC, flat within cells")):
        im = a_.imshow(np.maximum(crop(d), 1e-300), origin="lower", extent=ext, norm=norm, cmap=S.SEQ, aspect="auto")
        a_.set_title(t)
    fig.colorbar(im, ax=ax[0, :], label="MeV per fine voxel per primary", shrink=0.8)
    spread = np.abs(noise - mc) / np.sqrt(2)
    s_c = np.sqrt(np.maximum(upsample_flat(coarsen(spread ** 2, n), n), 1e-300))  # local noise scale per cell
    for a_, d, t in ((ax[1, 0], noise, "MC 2nd seed - MC"), (ax[1, 1], scr, "GMC+SCR - MC"),
                     (ax[1, 2], flat, "GMC flat - MC")):
        z = np.where(mc > vmax * 1e-4, (d - mc) / s_c, 0.0)
        im2 = a_.imshow(crop(z), origin="lower", extent=ext, cmap=S.DIV, norm=TwoSlopeNorm(0, -5, 5), aspect="auto")
        a_.set_title(t + "  (/ MC noise)")
    fig.colorbar(im2, ax=ax[1, :], label="difference / per-voxel MC noise", shrink=0.8)
    for a_ in ax.ravel():
        a_.set_xlabel("x (µm)")
        a_.set_ylabel("y (µm)")
    fig.suptitle(f"SCR-GMC, {M['materials']} {L * 1e4:g} µm cells, {n}x{n} fine voxels ({um:g} µm), beam "
                 f"{M['E_range_MeV'][0]:g}-{M['E_range_MeV'][1]:g} MeV")
    fig.savefig(os.path.join(out_dir, f"scr_maps_{tag}.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    xs = (np.arange(nx * n) + 0.5) * um
    for d, lab, kw in ((mc, "analog MC", dict(color=S.REF, lw=2)), (noise, "MC 2nd seed", dict(color="0.6", lw=1)),
                       (scr, "GMC + SCR", dict(color=S.BLUE_RAMP[4], lw=1.4)),
                       (flat, "GMC flat", dict(color="#b52f2e", lw=1, ls="--"))):
        ax[0].plot(xs, d.sum(1), label=lab, **kw)
    ax[0].set_xlabel("depth x (µm)")
    ax[0].set_ylabel("MeV per fine column per primary")
    ax[0].set_title("depth profile (fine)")
    ax[0].legend()
    ys = (np.arange(ny * n) + 0.5) * um
    beam_c = int(np.argmax(coarsen(mc, n)[0]))
    for a_, cx in ((ax[1], 0), (ax[2], min(3, nx - 1))):
        sl = slice(cx * n, (cx + 1) * n)
        for d, lab, kw in ((mc, "analog MC", dict(color=S.REF, lw=2)), (noise, "MC 2nd seed", dict(color="0.6", lw=1)),
                           (scr, "GMC + SCR", dict(color=S.BLUE_RAMP[4], lw=1.4)),
                           (flat, "GMC flat", dict(color="#b52f2e", lw=1, ls="--"))):
            a_.plot(ys, d[sl].sum(0), label=lab, **kw)
        a_.set_xlim((beam_c - 4) * L * 1e4, (beam_c + 5) * L * 1e4)
        a_.set_yscale("log")
        a_.set_xlabel("y (µm)")
        a_.set_title(f"lateral profile, cell column {cx} (x = {cx * L * 1e4:g}-{(cx + 1) * L * 1e4:g} µm)")
    fig.savefig(os.path.join(out_dir, f"scr_profiles_{tag}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
