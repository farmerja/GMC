"""
Figures for the SCR-GMC results page (validation/results/subcell_kernel_cu_100um.md), from a saved
scr_chain_compare.py run (no transport is redone):

    python validation/scr_figures.py \
        --run validation/results/secondary_chain_cu_100um/scr_doses_full_fine32_strat.npz --tally sobol64

Writes into the run's directory:
    scr_mc_vs_gmc_<tag>.png      analog MC vs GMC + SCR energy density on the fine mesh, error maps (difference / per-voxel
                                 MC noise), error histogram, fine depth and lateral profiles, wall time
    scr_entry_zoom_<tag>.png     the beam entry region: MC, GMC + SCR, GMC flat within cells
    scr_samples_<tag>.png        tally study: noise of one run and accuracy per unit time vs samples per transit
                                 (Sobol vs random), per sub-grid size
--mc_seconds: wall time of one analog-MC run of the reference (default: the 'seconds' stored by scr_mc_ref.py; the
32x32 full-beam reference predates that and took 73.0 s for 200k primaries on 16 threads).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sec_plot_style as S  # noqa: E402

plt = S.plt
BLUE = S.SERIES["gmc_sec"]["color"]
ORANGE = S.SERIES["gmc_prim"]["color"]
GRAY = S.SERIES["noise"]["color"]


def coarsen(d, n):
    return d.reshape(d.shape[0] // n, n, d.shape[1] // n, n).sum((1, 3))


def cell_noise(mc, nz, n):
    """Per-voxel noise of ONE MC run, averaged over each transport cell (the two-run difference is too noisy per voxel)."""
    var = (nz - mc) ** 2 / 2.0
    c = coarsen(var, n) / (n * n)
    return np.sqrt(np.repeat(np.repeat(c, n, 0), n, 1))


def profile_sigma(a, b, w=9):
    """Noise of one run's profile from two independent runs, smoothed over w bins."""
    v = np.convolve((a - b) ** 2 / 2.0, np.ones(w) / w, mode="same")
    return np.sqrt(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="scr_doses_<tag>.npz written by scr_chain_compare.py")
    ap.add_argument("--tally", default=None, help="tally to show (default: the first one of the run)")
    ap.add_argument("--fine_ref", default=None, help="the fine MC reference (for its stored run time)")
    ap.add_argument("--mc_seconds", type=float, default=None)
    a = ap.parse_args()
    S.apply()
    out_dir = os.path.dirname(os.path.abspath(a.run))
    tag = os.path.basename(a.run)[len("scr_doses_"):-4]
    D = np.load(a.run)
    M = json.load(open(os.path.join(out_dir, f"scr_metrics_{tag}.json")))
    tally = a.tally or M["tallies"][0]
    n, (nx, ny), L, N = M["n_ref"], M["grid"], M["cell_cm"], M["N"]
    h_um = L * 1e4 / n
    mc_s = a.mc_seconds
    if mc_s is None:
        ref = a.fine_ref or os.path.join(out_dir, f"mc_ref_{M['beam']}_fine{n}.npz")
        F = np.load(ref)
        mc_s = float(np.mean(list(json.loads(str(F["seconds"])).values()))) if "seconds" in F.files else 73.0
    runs = M["runs"]
    chain_s = float(np.mean([r["chain_s"] for r in runs]))
    tally_s = float(np.mean([r["tallies"][tally]["tally_s"] for r in runs]))
    res = M["by_n"][str(n)][tally]["split_fine"]
    var_ratio = (res["noise"] / res["mc_noise"]) ** 2

    to_density = 1e6 / (h_um * h_um)              # MeV per voxel per primary -> eV / um^2 per primary
    mc = D["mc"].astype(np.float64)
    nz = D["noise"].astype(np.float64)
    scr = D[tally][0].astype(np.float64)
    flat = D["gmc_flat"][0].astype(np.float64)
    sig = cell_noise(mc, nz, n)
    mc_c = coarsen(mc, n)
    hot_c = mc_c > 0.01 * mc_c.max()
    ix, iy = np.flatnonzero(hot_c.any(1)), np.flatnonzero(hot_c.any(0))
    x0, x1 = ix.min() * n, (ix.max() + 1) * n
    y0, y1 = max(0, (iy.min() - 1) * n), min(ny * n, (iy.max() + 2) * n)
    crop = lambda d: d[x0:x1, y0:y1].T  # noqa: E731
    ext = [x0 * h_um, x1 * h_um, y0 * h_um, y1 * h_um]
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    vmax = mc.max() * to_density
    norm = LogNorm(vmax * 1e-4, vmax)
    valid = mc > mc.max() * 1e-4

    # ---------------------------------------------------------------- figure 1: maps, errors, profiles, timing
    fig = plt.figure(figsize=(17.5, 15.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    for a_, d, t in ((ax[0][0], mc, f"(a) Analog MC, {N // 1000}k primaries"),
                     (ax[0][1], scr, f"(b) GMC + SCR ({tally}), same primaries")):
        im = a_.imshow(np.maximum(crop(d) * to_density, 1e-300), origin="lower", extent=ext, norm=norm, cmap=S.SEQ,
                       aspect="equal", interpolation="nearest")
        a_.set_title(t)
        a_.set_xlabel("depth x (µm)")
        a_.set_ylabel("y (µm)")
        a_.grid(False)
    cb = fig.colorbar(im, ax=[ax[0][0], ax[0][1]], shrink=0.85, pad=0.01)
    cb.set_label("deposited energy (eV / µm² per primary, integrated over z)")

    # (c) timing
    a_ = ax[0][2]
    rows = [("Analog MC\n(16 CPU threads)", [(mc_s, S.REF, "MC transport")]),
            ("GMC + SCR\n(one GPU)", [(chain_s, BLUE, "GMC chain"), (tally_s, S.BLUE_RAMP[2], "SCR tally")]),
            ("GMC + SCR, time to\nreach the MC's noise", [(var_ratio * (chain_s + tally_s), S.BLUE_RAMP[5],
                                                         "")])]
    for i, (lab, segs) in enumerate(rows):
        left = 0.0
        for w, col, name in segs:
            a_.barh(i, w - 0.4, left=left, height=0.55, color=col, edgecolor=S.SURFACE, linewidth=2)
            if i < 2 and w > 15:
                a_.text(left + w / 2, i, f"{name}\n{w:.0f} s", ha="center", va="center", fontsize=8.5,
                        color="white" if col not in (S.BLUE_RAMP[2],) else S.INK)
            left += w
        a_.text(left + 1.5, i, f"{left:.0f} s", va="center", fontsize=10, color=S.INK, fontweight="bold")
    a_.set_axisbelow(True)
    a_.set_yticks(range(len(rows)), [r[0] for r in rows])
    a_.invert_yaxis()
    a_.set_xlim(0, max(mc_s, chain_s + tally_s) * 1.22)
    a_.set_xlabel(f"wall time for {N // 1000}k primaries (s)")
    a_.grid(axis="y", visible=False)
    sp = mc_s / (chain_s + tally_s)
    a_.set_title(f"(c) Timing: {sp:.1f}x wall time, {mc_s / (var_ratio * (chain_s + tally_s)):.1f}x at equal noise")
    a_.text(0.98, 0.02, f"fine-map noise of one run: MC {100 * res['mc_noise']:.2f}%, GMC + SCR "
                        f"{100 * res['noise']:.2f}%\n(variance ratio {var_ratio:.2f}); "
                        f"{runs[0]['transits'] / N:.2f} transits per primary", transform=a_.transAxes, ha="right",
            va="bottom", fontsize=8.5, color=S.INK_2)

    # (d, e) error maps
    for a_, d, t in ((ax[1][0], nz, "(d) MC 2nd seed − MC"), (ax[1][1], scr, f"(e) GMC + SCR − MC")):
        z = np.where(valid, (d - mc) / np.maximum(sig, 1e-300), 0.0)
        im2 = a_.imshow(crop(z), origin="lower", extent=ext, cmap=S.DIV, norm=TwoSlopeNorm(0, -4, 4), aspect="equal",
                        interpolation="nearest")
        a_.set_title(t + "  (÷ per-voxel MC noise)")
        a_.set_xlabel("depth x (µm)")
        a_.set_ylabel("y (µm)")
        a_.grid(False)
    cb2 = fig.colorbar(im2, ax=[ax[1][0], ax[1][1]], shrink=0.85, pad=0.01)
    cb2.set_label("difference / noise of one MC run per voxel")

    # (f) error histogram over voxels of the hot cells
    a_ = ax[1][2]
    hot = np.repeat(np.repeat(hot_c, n, 0), n, 1) & valid
    bins = np.linspace(-6, 6, 121)
    for d, col, ls, lab in ((nz, GRAY, "-", "MC 2nd seed − MC"), (scr, BLUE, "-", "GMC + SCR − MC")):
        z = ((d - mc) / np.maximum(sig, 1e-300))[hot]
        a_.hist(z, bins=bins, histtype="step", color=col, lw=2, ls=ls, density=True,
                label=f"{lab}: σ = {z.std():.2f}, mean = {z.mean():+.3f}")
    g = np.linspace(-6, 6, 400)
    a_.plot(g, np.exp(-g * g / 4) / np.sqrt(4 * np.pi), color=S.REF, lw=1, ls=":",
            label="expected for two MC runs (σ = √2)")
    a_.set_xlabel("difference / noise of one MC run")
    a_.set_ylabel("fraction of voxels (density)")
    a_.set_title(f"(f) Voxel errors, {hot.sum() / 1e3:.0f}k voxels in the dose region")
    a_.set_ylim(0, a_.get_ylim()[1] * 1.35)
    a_.legend(loc="upper left", fontsize=8.5)

    # (g, h, i) profiles
    xs = (np.arange(nx * n) + 0.5) * h_um
    a_ = ax[2][0]
    dm, dn, ds, df = (v.sum(1) * 1e6 / h_um for v in (mc, nz, scr, flat))
    s_ = profile_sigma(dm, dn)
    a_.fill_between(xs, dm - 2 * s_, dm + 2 * s_, color=GRAY, alpha=0.3, lw=0, label="MC ± 2σ (one run)")
    a_.plot(xs, dm, **S.style("mc", marker=None, label="analog MC"))
    a_.plot(xs, ds, color=BLUE, lw=1.4, label="GMC + SCR")
    a_.plot(xs, df, color=ORANGE, lw=1.2, ls="--", label="GMC, flat within cells")
    a_.set_xlim(0, 600)
    a_.set_xlabel("depth x (µm)")
    a_.set_ylabel("eV / µm depth per primary")
    a_.set_title(f"(g) Depth profile, {h_um:.3g} µm bins (first 600 µm)")
    a_.legend()
    ys = (np.arange(ny * n) + 0.5) * h_um
    p0 = mc[:n].sum(0)
    y_c = float((ys * p0).sum() / p0.sum())                 # beam centre (dose-weighted, first cell column)
    win = (ys > y_c - 450) & (ys < y_c + 450)
    for a_, cx, lab in ((ax[2][1], 0, "(h)"), (ax[2][2], 3, "(i)")):
        sl = slice(cx * n, (cx + 1) * n)
        lm, ln_, ls_, lf = (v[sl].sum(0) * 1e6 / h_um for v in (mc, nz, scr, flat))
        s_ = profile_sigma(lm, ln_)
        a_.fill_between(ys, np.maximum(lm - 2 * s_, 1e-30), lm + 2 * s_, color=GRAY, alpha=0.3, lw=0,
                        label="MC ± 2σ (one run)")
        a_.plot(ys, lm, **S.style("mc", marker=None, label="analog MC"))
        a_.plot(ys, ls_, color=BLUE, lw=1.4, label="GMC + SCR")
        a_.plot(ys, lf, color=ORANGE, lw=1.2, ls="--", label="GMC, flat within cells")
        a_.set_xlim(y_c - 450, y_c + 450)
        a_.set_yscale("log")
        lo, hi = min(lm[win].min(), ls_[win].min()), max(lm[win].max(), ls_[win].max(), lf[win].max())
        a_.set_ylim(lo / 1.4, hi * 1.4)
        a_.set_xlabel("y (µm)")
        a_.set_ylabel("eV / µm per primary")
        a_.set_title(f"{lab} Lateral profile, x = {cx * L * 1e4:g}–{(cx + 1) * L * 1e4:g} µm")
    ax[2][1].legend(fontsize=8.5)
    fig.suptitle(f"SCR-GMC vs analog MC: Cu, {L * 1e4:g} µm transport cells, {h_um:.3g} µm dose voxels "
                 f"({n}×{n} per cell), {M['E_range_MeV'][0]:g}–{M['E_range_MeV'][1]:g} MeV log-uniform beam",
                 fontsize=13, fontweight="bold")
    p1 = os.path.join(out_dir, f"scr_mc_vs_gmc_{tag}.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: beam entry zoom
    zx, zy = (0, 300), (850, 1150)
    sx = slice(int(zx[0] / h_um), int(zx[1] / h_um))
    sy = slice(int(zy[0] / h_um), int(zy[1] / h_um))
    fig, ax2 = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    znorm = LogNorm(vmax * 3e-3, vmax)
    for a_, d, t in ((ax2[0], mc, "Analog MC"), (ax2[1], scr, f"GMC + SCR ({tally})"),
                     (ax2[2], flat, "GMC, flat within cells")):
        im = a_.imshow(np.maximum(d[sx, sy].T * to_density, 1e-300), origin="lower", norm=znorm, cmap=S.SEQ,
                       extent=[*zx, *zy], aspect="equal", interpolation="nearest")
        for v in range(zx[0], zx[1] + 1, 100):
            a_.axvline(v, color="white", lw=0.6, alpha=0.6)
        for v in range(900, zy[1] + 1, 100):
            a_.axhline(v, color="white", lw=0.6, alpha=0.6)
        a_.set_title(t)
        a_.set_xlabel("depth x (µm)")
        a_.set_ylabel("y (µm)")
        a_.grid(False)
    cb = fig.colorbar(im, ax=ax2, shrink=0.9, pad=0.01)
    cb.set_label("eV / µm² per primary")
    fig.suptitle(f"Beam entry region at {h_um:.3g} µm (white lines: 100 µm transport-cell edges)", fontsize=12,
                 fontweight="bold")
    p2 = os.path.join(out_dir, f"scr_entry_zoom_{tag}.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3: samples per transit
    by = M["by_n"]
    sizes = sorted((int(k) for k in by), reverse=True)
    chain_us = float(np.mean([r["chain_s"] / r["transits"] for r in runs])) * 1e6
    tpp = float(np.mean([r["transits"] for r in runs])) / N
    mc_us = mc_s / N * 1e6
    tk = {t: float(np.mean([r["tallies"][t]["tally_us_per_transit"] for r in runs])) for t in M["tallies"]}
    kk = lambda t: int("".join(c for c in t if c.isdigit()))  # noqa: E731
    fig, ax3 = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    cols = {32: S.BLUE_RAMP[6], 16: S.BLUE_RAMP[4], 8: S.BLUE_RAMP[2]}
    for r in sizes:
        R = by[str(r)]
        um = L * 1e4 / r
        for meth, ls, mk in (("sobol", "-", "o"), ("random", "--", "s")):
            ts = sorted((t for t in M["tallies"] if t.startswith(meth)), key=kk)
            K = [kk(t) for t in ts]
            noise = [100 * R[t]["split_fine"]["noise"] for t in ts]
            eff = [mc_us * R[t]["split_fine"]["mc_noise"] ** 2 / (tpp * (chain_us + tk[t]) * R[t]["split_fine"]["noise"] ** 2)
                   for t in ts]
            kw = dict(color=cols[r], ls=ls, marker=mk, ms=7, lw=2, mfc=cols[r] if meth == "sobol" else S.SURFACE,
                      mec=S.SURFACE if meth == "sobol" else cols[r], mew=1.0 if meth == "sobol" else 1.8)
            ax3[0].plot(K, noise, label=f"{um:.3g} µm, {meth}", **kw)
            ax3[1].plot(K, eff, label=f"{um:.3g} µm, {meth}", **kw)
        ax3[0].axhline(100 * R[M["tallies"][0]]["split_fine"]["mc_noise"], color=cols[r], lw=1, ls=":")
    for a_ in ax3:
        a_.set_xscale("log", base=2)
        a_.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
        a_.set_xlabel("kernel samples per transit K")
    ax3[0].set_ylabel("fine-map noise of one run (%)")
    ax3[0].set_ylim(0, None)
    ax3[0].set_title("(a) Noise vs samples (dotted: one analog-MC run, same colour = same voxel size)")
    ax3[0].legend(fontsize=8.5, ncol=2, loc="lower left")
    ax3[1].axhline(1, color=S.REF, lw=1)
    ax3[1].set_ylabel("accuracy per unit time vs analog MC (×)")
    ax3[1].set_ylim(0, None)
    ax3[1].set_title(f"(b) Speedup at equal noise (MC {mc_us:.0f} µs/primary, chain {chain_us:.0f} µs/transit)")
    ax3[1].legend(fontsize=8.5, ncol=2, loc="lower left")
    fig.suptitle("Kernel tally: Sobol vs random sampling (same GMC chains, 3 seeds)", fontsize=12,
                 fontweight="bold")
    p3 = os.path.join(out_dir, f"scr_samples_{tag}.png")
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    print("\n".join(("wrote", p1, p2, p3)))
    print(f"MC {mc_s:.1f} s, chain {chain_s:.1f} s, tally {tally_s:.1f} s, variance ratio {var_ratio:.3f}")


if __name__ == "__main__":
    main()
