#!/usr/bin/env python3
"""
Is GMC + SCR within the noise of the analog MC on the Drumm cylinder? Per 0.5 mm bin (inside the circle, MC > 1 % of
its maximum): d = GMC - MC (seed means) and its 1-sigma noise sqrt(var_GMC + var_MC), each variance from the seed
spread of that side and smoothed over the 3 x 3 neighbouring bins (relative variance, so the smoothing does not mix
dose levels) to make a stable per-bin estimate from only a few seeds. If GMC = MC up to noise, z = d / sigma is
N(0, 1): chi2/dof ~ 1, 95 % of |z| < 2; a region whose MEAN z is many sigma-of-the-mean from 0 holds a systematic
offset.

Figures: GMC vs MC per bin (coloured by z), z histograms vs N(0, 1), z maps, profiles with the combined noise band.

    .venv/bin/python validation/drumm_noise_check.py --runs validation/results/drumm_size/gmc_*_sx3_chord.npz --out_dir ...
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drumm_quadtree import load_mc, H_OUT, R, coarse  # noqa: E402

K = 4                                      # 0.5 mm bins


def seed_var(maps):
    """Variance of the mean of the seed maps, as a RELATIVE variance smoothed over 3 x 3 bins."""
    from scipy.ndimage import uniform_filter
    A = np.array(maps)
    mean = A.mean(0)
    v = A.var(0, ddof=1) / len(A)
    rel = np.where(mean > 0, v / np.maximum(mean, 1e-300) ** 2, 0.0)
    return mean, uniform_filter(rel, 3, mode="nearest") * mean ** 2


def gmc_map(path):
    """Seed-mean GMC loc map on 0.5 mm bins (MC units) and its smoothed variance."""
    F = np.load(path)
    N = int(F["N"])
    return seed_var([coarse(d / (N * H_OUT ** 2), K) for d in F["loc"]])


def regions(gx):
    X, Y = np.meshgrid(gx, gx, indexing="ij")
    ay = np.abs(Y)
    return {"core |y| < 2 mm": ay < 0.2, "edge 2-3 mm": (ay >= 0.2) & (ay < 0.3), "wings |y| > 3 mm": ay >= 0.3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_frac", type=float, default=0.01)
    ap.add_argument("--r_max", type=float, default=R, help="only bins with r < r_max (cm), e.g. 0.9 to drop the "
                    "circle edge where a staircase mesh changes the geometry")
    ap.add_argument("--null", nargs=2, default=None, metavar=("RUN_A", "RUN_B"),
                    help="null test of the noise estimate: two independent GMC runs of the same model and mesh")
    a = ap.parse_args()
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    mc = load_mc()
    M, vM = seed_var([coarse(d, K) for d in mc["loc"]])
    gx = (np.arange(M.shape[0]) + 0.5) * H_OUT * K - 1.0
    hot = (np.hypot(gx[:, None], gx[None, :]) < min(R, a.r_max)) & (M > a.min_frac * M.max())
    regs = regions(gx)
    runs = sorted(set(sum((glob.glob(p) for p in a.runs), [])))
    res, Z = {}, {}
    for p in runs:
        meta = json.loads(str(np.load(p)["meta"]))
        G, vG = gmc_map(p)
        sig = np.sqrt(vG + vM)
        z = np.where(hot, (G - M) / np.maximum(sig, 1e-300), np.nan)
        zh = z[hot]
        r = {"mesh": meta["mesh"], "n_seeds": int(meta["n_seeds"]), "bins": int(hot.sum()),
             "chi2_dof": float(np.mean(zh ** 2)), "frac_abs_z_lt_2": float(np.mean(np.abs(zh) < 2)),
             "frac_abs_z_gt_3": float(np.mean(np.abs(zh) > 3)), "mean_z": float(zh.mean()),
             "noise_rel_rms": float(np.sqrt(np.mean((sig[hot] / M[hot]) ** 2))),
             "gmc_noise_rel_rms": float(np.sqrt(np.mean(vG[hot] / M[hot] ** 2))),
             "mc_noise_rel_rms": float(np.sqrt(np.mean(vM[hot] / M[hot] ** 2))), "regions": {}}
        for name, m in regs.items():
            s = hot & m
            if s.sum() < 3:
                continue
            zz = z[s]
            r["regions"][name] = {"bins": int(s.sum()), "mean_z": float(zz.mean()),
                                  "mean_z_sigma": float(zz.mean() * np.sqrt(len(zz))),   # significance of the mean
                                  "chi2_dof": float(np.mean(zz ** 2)),
                                  "energy_ratio": float(G[s].sum() / M[s].sum())}
        res[Path(p).stem] = r
        Z[Path(p).stem] = (G, sig, z)
    null = None
    if a.null:
        # same statistic between two independent GMC runs (different chain AND source seeds): z must be N(0, 1) if the
        # seed-spread variance estimate is right
        GA, vA = gmc_map(a.null[0])
        GB, vB = gmc_map(a.null[1])
        zn = ((GA - GB) / np.maximum(np.sqrt(vA + vB), 1e-300))[hot]
        null = {"runs": [Path(x).stem for x in a.null], "chi2_dof": float(np.mean(zn ** 2)),
                "frac_abs_z_lt_2": float(np.mean(np.abs(zn) < 2)), "frac_abs_z_gt_3": float(np.mean(np.abs(zn) > 3)),
                "mean_z": float(zn.mean()), "z": zn}
        print(f"null test {null['runs'][0]} vs {null['runs'][1]}: chi2/dof {null['chi2_dof']:.2f}, "
              f"|z|<2 {null['frac_abs_z_lt_2']:.3f}, |z|>3 {null['frac_abs_z_gt_3']:.3f}, mean z {null['mean_z']:+.2f}")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(Path(a.out_dir) / "noise_check.json", "w") as fh:
        json.dump({"mc_noise_rel_rms": float(np.sqrt(np.mean(vM[hot] / M[hot] ** 2))), "runs": res,
                   "null": None if null is None else {k: v for k, v in null.items() if k != "z"}}, fh, indent=1)

    names = list(res)
    nr = len(names)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#fcfcfb")
    fig, ax = plt.subplots(3, nr, figsize=(3.9 * nr, 11.5), constrained_layout=True, squeeze=False)
    zz = np.linspace(-5, 5, 201)
    for j, n_ in enumerate(names):
        G, sig, z = Z[n_]
        r = res[n_]
        a0 = ax[0, j]
        sc = a0.scatter(M[hot], G[hot], c=np.clip(z[hot], -4, 4), cmap=cmap, vmin=-4, vmax=4, s=7, lw=0)
        lim = [M[hot].min() * 0.8, M[hot].max() * 1.2]
        a0.plot(lim, lim, color="0.3", lw=0.8)
        a0.set_xscale("log")
        a0.set_yscale("log")
        a0.set_xlim(lim)
        a0.set_ylim(lim)
        a0.set_xlabel("analog MC (MeV/cm³ per e⁻/cm)")
        a0.set_ylabel("GMC + SCR")
        a0.set_title(f"{r['mesh']}\nchi2/dof {r['chi2_dof']:.2f}, |z|<2: {100 * r['frac_abs_z_lt_2']:.0f} %",
                     fontsize=9)
        a1 = ax[1, j]
        a1.hist(np.clip(z[hot], -6, 6), bins=np.linspace(-6, 6, 49), density=True, color="#2a78d6", alpha=0.75,
                label="z per 0.5 mm bin")
        if null is not None:
            a1.hist(np.clip(null["z"], -6, 6), bins=np.linspace(-6, 6, 49), density=True, histtype="step",
                    color="#eb6834", lw=1.5, label="null: GMC vs GMC")
        a1.plot(zz, np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), color="0.15", lw=1.5, label="N(0, 1)")
        a1.set_xlabel("z = (GMC - MC) / combined sigma")
        a1.set_title(f"mean z {r['mean_z']:+.2f}, |z|>3: {100 * r['frac_abs_z_gt_3']:.0f} %", fontsize=9)
        if j == 0:
            a1.legend(fontsize=8)
        a2 = ax[2, j]
        im = a2.imshow(z.T, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap, vmin=-4, vmax=4)
        a2.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.35", lw=0.6))
        a2.set_aspect("equal")
        a2.grid(False)
        a2.set_title("z map (|z| > 2: outside the noise)", fontsize=9)
    fig.colorbar(im, ax=ax[2, :], shrink=0.8, label="z")
    fig.colorbar(sc, ax=ax[0, :], shrink=0.8, label="z")
    fig.suptitle("GMC + SCR vs analog MC per 0.5 mm bin, in units of the combined statistical noise", fontweight="bold")
    fig.savefig(Path(a.out_dir) / "noise_check.png", dpi=115)
    plt.close(fig)
    print(f"MC noise (rel. rms, 0.5 mm bins) {np.sqrt(np.mean(vM[hot] / M[hot] ** 2)):.4f}")
    print(f"{'run':>44} {'chi2/dof':>8} {'|z|<2':>6} {'|z|>3':>6} {'mean z':>7} | region mean z (significance), energy ratio")
    for n_, r in res.items():
        reg = "  ".join(f"{k.split()[0]} {v['mean_z']:+.2f} ({v['mean_z_sigma']:+.0f}s) {v['energy_ratio']:.3f}"
                        for k, v in r["regions"].items())
        print(f"{n_[4:48]:>44} {r['chi2_dof']:8.2f} {r['frac_abs_z_lt_2']:6.2f} {r['frac_abs_z_gt_3']:6.2f} "
              f"{r['mean_z']:+7.2f} | {reg}")


if __name__ == "__main__":
    main()
