"""
Chained GMC with and without secondaries vs analog MC on a grid of 100 um cells, one or several materials (GMC venv).

    python validation/secondary_chain_compare.py --mc_ref validation/results/secondary_chain_cu_100um/mc_ref_mev.npz \
        --model ml/models/secondary_gmc_cu_100um/secondary_gmc.pth

Inputs come from validation/secondary_mc_refs.py. Every chain reuses the reference's source. Compared against the
full-grid analog MC (the truth):
    Analog MC, 2nd seed          noise floor
    MC-chained + secondaries     exact chain of the analog MC (validates the chain loop itself)
    MC-chained primary-only      best possible primary-only GMC
    GMC + secondaries            ml/secondary_gmc.py, relevant secondaries followed
    GMC primary-only             the same models, secondaries' energy left in the cell where they are born
Writes dose_maps_<beam>.png, profiles_<beam>.png, errors_<beam>.png and chain_metrics_<beam>.json next to --out_dir.

Several materials: a reference made with secondary_mc_refs.py --materials A,B --layout ... stores the material map;
pass one --model per material in the same order, e.g. for --materials Al,Cu:
    --model ml/models/secondary_gmc_al_100um/secondary_gmc.pth ml/models/secondary_gmc_cu_100um/secondary_gmc.pth
Errors are then also reported per material (cells of that material only), and interfaces are outlined in the plots.
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
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402


def rel_l2(d, ref):
    return float(np.sqrt(((d - ref) ** 2).sum() / (ref ** 2).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc_ref", required=True)
    ap.add_argument("--model", required=True, nargs="+", help="one GMC checkpoint per material of the reference")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ode_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="independent GMC chain runs; errors are reported as mean and spread, plots use the first")
    ap.add_argument("--replot", action="store_true",
                    help="reuse the GMC chain doses saved by an earlier run (gmc_doses_<beam>.npz) and only redo plots")
    a = ap.parse_args()
    S.apply()
    import torch
    from ml.secondary_gmc import SecondaryGMC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = np.load(a.mc_ref)
    meta = json.loads(str(ref["meta"]))
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.mc_ref))
    os.makedirs(out_dir, exist_ok=True)
    tag = meta["beam"]
    src = ref["source"]
    N = len(src)
    nx, ny, L, E_cut = meta["nx"], meta["ny"], meta["cell_size"], meta["E_cutoff"]
    names = meta.get("materials", ["Cu"])
    mmap = ref["material_map"] if "material_map" in ref.files else np.zeros((nx, ny), dtype=np.int64)
    if len(a.model) != len(names):
        raise ValueError(f"reference has materials {names}: pass {len(names)} --model paths in that order")
    multi = len(names) > 1
    cells_desc = f"{L * 1e4:g} µm " + ("/".join(names) if multi else names[0]) + " cells"
    if multi:
        cells_desc += f" ({names[1]} {meta.get('layout', '')} in {names[0]})"

    doses = {"mc": ref["dose_full"], "noise": ref["dose_full_noise"], "mcc_sec": ref["dose_mcchain_sec"],
             "mcc_prim": ref["dose_mcchain_prim"]}
    chain_info = {}
    runs = {"gmc_sec": [], "gmc_prim": []}  # one dose map per seed
    cache = os.path.join(out_dir, f"gmc_doses_{tag}.npz")
    if a.replot:
        c = np.load(cache)
        for name in runs:
            runs[name] = list(c[name])
        chain_info = json.loads(str(c["chain_info"]))
    for s in (() if a.replot else range(a.n_seeds)):
        for name, sec in (("gmc_sec", True), ("gmc_prim", False)):
            seed = a.seed + 100 * s + (0 if sec else 1)
            K = [SecondaryGMC(path, dev, ode_steps=a.ode_steps, seed=seed + 1009 * m, secondaries=sec)
                 for m, path in enumerate(a.model)]
            t0 = time.time()
            c = chain_secondaries(K, src, nx, ny, L, E_cut, log_every=0, material_map=mmap)
            tot = c["dose"].sum() + sum(c["escaped"].values()) + c["photon_E"] + c["in_flight"]
            info = {"seed": seed, "seconds": time.time() - t0, "steps": c["steps"],
                    "transits_boundary": c["transits_boundary"], "transits_interior": c["transits_interior"],
                    "daughters": c["daughters"], "escaped_MeV": c["escaped"], "photon_MeV": c["photon_E"],
                    "energy_closure": (tot - c["E_source"]) / c["E_source"],
                    "dose_from_secondary_transits_frac": float(c["dose_secondary_transits"].sum() /
                                                               max(c["dose"].sum(), 1e-300))}
            chain_info.setdefault(name, []).append(info)
            runs[name].append(c["dose"])
            print(f"{name} seed {seed}: {time.time() - t0:.0f} s, closure {info['energy_closure']:.1e}", flush=True)
    if not a.replot:
        np.savez_compressed(cache, gmc_sec=np.array(runs["gmc_sec"]), gmc_prim=np.array(runs["gmc_prim"]),
                            chain_info=json.dumps(chain_info, default=float), model=a.model)
    doses["gmc_sec"], doses["gmc_prim"] = runs["gmc_sec"][0], runs["gmc_prim"][0]
    D = {k: v / N for k, v in doses.items()}  # MeV per cell per primary
    ref_d = D["mc"]
    order = ["noise", "mcc_sec", "mcc_prim", "gmc_sec", "gmc_prim"]
    metrics = {"beam": tag, "E_range_MeV": [meta["E_low"], meta["E_high"]], "N": N, "grid": [nx, ny],
               "cell_cm": L, "materials": names, "layout": meta.get("layout", "uniform"), "models": a.model,
               "chains": chain_info, "vs_full_MC": {}}
    depth = {k: v.sum(1) for k, v in D.items()}
    lat = {k: v.sum(0) for k, v in D.items()}
    for k in order:
        metrics["vs_full_MC"][k] = {
            "rel_L2_map": rel_l2(D[k], ref_d), "total_dose_ratio": float(D[k].sum() / ref_d.sum()),
            "rel_L2_depth_profile": rel_l2(depth[k], depth["mc"]),
            "rel_L2_lateral_profile": rel_l2(lat[k], lat["mc"])}
        if multi:  # cells of one material only
            metrics["vs_full_MC"][k]["per_material"] = {
                n: {"rel_L2_map": rel_l2(D[k][mmap == m], ref_d[mmap == m]),
                    "total_dose_ratio": float(D[k][mmap == m].sum() / ref_d[mmap == m].sum()),
                    "dose_fraction_MC": float(ref_d[mmap == m].sum() / ref_d.sum())} for m, n in enumerate(names)}
    for k in ("gmc_sec", "gmc_prim"):  # spread over independent GMC runs
        l2 = [rel_l2(d / N, ref_d) for d in runs[k]]
        tr = [float(d.sum() / N / ref_d.sum()) for d in runs[k]]
        if multi:
            for m, n in enumerate(names):
                l2m = [rel_l2(d[mmap == m] / N, ref_d[mmap == m]) for d in runs[k]]
                trm = [float(d[mmap == m].sum() / N / ref_d[mmap == m].sum()) for d in runs[k]]
                metrics["vs_full_MC"][k]["per_material"][n].update({
                    "rel_L2_map_per_seed": l2m, "rel_L2_map_mean": float(np.mean(l2m)),
                    "rel_L2_map_std": float(np.std(l2m, ddof=1)) if len(l2m) > 1 else float("nan"),
                    "total_dose_ratio_mean": float(np.mean(trm))})
        metrics["vs_full_MC"][k].update({
            "rel_L2_map_per_seed": l2, "rel_L2_map_mean": float(np.mean(l2)),
            "rel_L2_map_std": float(np.std(l2, ddof=1)) if len(l2) > 1 else float("nan"),
            "total_dose_ratio_per_seed": tr, "total_dose_ratio_mean": float(np.mean(tr))})
    with open(os.path.join(out_dir, f"chain_metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    for k in order:
        m = metrics["vs_full_MC"][k]
        extra = (f"   [{len(m['rel_L2_map_per_seed'])} seeds: mean {m['rel_L2_map_mean']:.4f} ± "
                 f"{m['rel_L2_map_std']:.4f}]") if "rel_L2_map_mean" in m else ""
        print(f"  {S.SERIES[k]['label']:28s} rel-L2 map {m['rel_L2_map']:.4f}  total {m['total_dose_ratio']:.4f}  "
              f"depth {m['rel_L2_depth_profile']:.4f}  lateral {m['rel_L2_lateral_profile']:.4f}{extra}")
        for n, pm in m.get("per_material", {}).items():
            print(f"      {n:4s} cells: rel-L2 {pm.get('rel_L2_map_mean', pm['rel_L2_map']):.4f}  "
                  f"total {pm.get('total_dose_ratio_mean', pm['total_dose_ratio']):.4f}")

    title_E = f"{meta['E_low'] * 1e3:g} keV–{meta['E_high']:g} MeV" if meta["E_low"] < 1 else \
        f"{meta['E_low']:g}–{meta['E_high']:g} MeV"
    ext = [0, nx * L * 10, 0, ny * L * 10]  # mm

    def outline(ax):  # material interfaces
        if multi:
            g = np.arange(nx + 1) * L * 10, np.arange(ny + 1) * L * 10
            for i in range(nx):
                for j in range(ny):
                    x0, x1, y0, y1 = g[0][i], g[0][i + 1], g[1][j], g[1][j + 1]
                    if i + 1 < nx and mmap[i, j] != mmap[i + 1, j]:
                        ax.plot([x1, x1], [y0, y1], color="w", lw=1.2)
                    if j + 1 < ny and mmap[i, j] != mmap[i, j + 1]:
                        ax.plot([x0, x1], [y1, y1], color="w", lw=1.2)

    def shade(ax, axis):  # extent of material 1 along a profile axis
        if multi:
            cols = np.flatnonzero((mmap == 1).any(axis=1 - axis))
            if cols.size:
                ax.axvspan(cols[0] * L * 10, (cols[-1] + 1) * L * 10, color=S.GRID, alpha=0.6, lw=0,
                           label=f"{names[1]} extent")

    # ---- dose maps + ratio maps -------------------------------------------------------------------------------------
    vmax = ref_d.max()
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), layout="constrained")
    for ax, k in zip(axs[0], ["mc", "gmc_sec", "gmc_prim"]):
        im = ax.imshow(np.maximum(D[k].T, vmax * 1e-5), origin="lower", extent=ext, cmap=S.SEQ, norm=norm)
        ax.set_title(S.SERIES[k]["label"])
        ax.grid(False)
        outline(ax)
    fig.colorbar(im, ax=axs[0].tolist(), label="dose per primary (MeV / cell)", shrink=0.9)
    mask = ref_d < vmax * 1e-3
    rnorm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    for ax, k in zip(axs[1], ["noise", "gmc_sec", "gmc_prim"]):
        r = np.log2(np.maximum(D[k], 1e-300) / np.maximum(ref_d, 1e-300))
        r = np.ma.array(r, mask=mask)
        im2 = ax.imshow(r.T, origin="lower", extent=ext, cmap=S.DIV, norm=rnorm)
        ax.set_title(f"{S.SERIES[k]['label']} / analog MC\nrel-L2 {metrics['vs_full_MC'][k]['rel_L2_map']:.3f}",
                     fontsize=10)
        ax.grid(False)
        outline(ax)
    cb = fig.colorbar(im2, ax=axs[1].tolist(), label="log2(dose / analog MC)  (cells < 1e-3 of max hidden)",
                      shrink=0.9)
    cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    for ax in axs.flat:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle(f"Dose on a {nx}×{ny} grid of {cells_desc}, {title_E} beam, {N} primaries", fontweight="bold")
    fig.savefig(os.path.join(out_dir, f"dose_maps_{tag}.png"), dpi=130)
    plt.close(fig)

    # ---- profiles ----------------------------------------------------------------------------------------------------
    xc = (np.arange(nx) + 0.5) * L * 10
    yc = (np.arange(ny) + 0.5) * L * 10
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]})
    for col, (prof, coord, xlabel) in enumerate(((depth, xc, "depth x (mm)"), (lat, yc, "lateral y (mm)"))):
        ax, axr = axs[0, col], axs[1, col]
        shade(ax, col)
        shade(axr, col)
        ax.plot(coord, prof["mc"], **S.style("mc"))
        for k in order:
            ax.plot(coord, prof[k], **S.style(k))
            if k != "noise":
                axr.plot(coord, prof[k] / np.maximum(prof["mc"], 1e-300), **S.style(k))
        axr.plot(coord, prof["noise"] / np.maximum(prof["mc"], 1e-300), **S.style("noise"))
        axr.axhline(1.0, color=S.REF, lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel("dose per primary (MeV, summed over " + ("y" if col == 0 else "x") + ")")
        axr.set_ylabel("ratio to analog MC")
        axr.set_xlabel(xlabel)
        lo, hi = np.nanpercentile(np.concatenate([prof[k] / np.maximum(prof["mc"], 1e-300) for k in order]), [2, 98])
        axr.set_ylim(min(0.85, lo - 0.02), max(1.15, hi + 0.02))
    axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Depth and lateral dose profiles, {cells_desc}, {title_E} beam", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"profiles_{tag}.png"), dpi=130)
    plt.close(fig)

    # ---- error summary ------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.8))
    mv = metrics["vs_full_MC"]
    vals = [mv[k].get("rel_L2_map_mean", mv[k]["rel_L2_map"]) for k in order]
    errs = [mv[k].get("rel_L2_map_std", 0.0) for k in order]
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, vals, color=S.SERIES["gmc_sec"]["color"], height=0.55)
    for y_, k in zip(ypos, order):  # individual GMC runs as dots, spread as a bar
        if "rel_L2_map_per_seed" in mv[k]:
            ax.errorbar(vals[order.index(k)], y_, xerr=errs[order.index(k)], color=S.INK, capsize=4, lw=1.2)
            ax.plot(mv[k]["rel_L2_map_per_seed"], [y_] * len(mv[k]["rel_L2_map_per_seed"]), "o", ms=4,
                    color=S.INK, mec=S.SURFACE)
    for y_, v, e in zip(ypos, vals, errs):
        lab = f"  {v:.4f}" + (f" ± {e:.4f}  (mean ± sd, {len(runs['gmc_sec'])} runs)"
                              if e > 0 else "")
        k = order[list(ypos).index(y_)]
        right = max([v + e] + list(mv[k].get("rel_L2_map_per_seed", [])))  # clear of the whisker and the dots
        ax.text(right, y_, lab, va="center", color=S.INK, fontsize=9)
    ax.axvline(vals[0], color=S.INK_2, lw=1.2, ls="--")
    ax.text(vals[0], ypos[0] + 0.45, " MC noise floor", color=S.INK_2, fontsize=8)
    ax.set_yticks(ypos, [S.SERIES[k]["label"] for k in order])
    ax.set_xlabel("relative L2 error of the dose map vs full-grid analog MC")
    ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.7)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Dose-map error, {cells_desc}, {title_E} beam")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"errors_{tag}.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
