"""On-axis profile around the 1 cm cell entry of the centre:10:0.15625:2 mesh, before / after the photon-chord fix,
vs the true-circle MC (2026-11-22). Pools all seeds of the given runs; error bars from the seed spread."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drumm_quadtree import load_mc, H_OUT, OUT  # noqa: E402
import sec_plot_style as S  # noqa: E402

RUNS = {"before (photons with the kernel)": ["gmc_centre_10_0p15625_2_500k_K8_sx2.npz"],
        "after (photons on the chord)": ["gmc_centre_10_0p15625_2_500k_K8_sx2_chord.npz",
                                         "gmc_centre_10_0p15625_2_500k_K8_sx2_chord_s31.npz"]}
BAND = 0.1          # |y| < 1 mm
GROUP = 4           # 0.5 mm x bins


def main():
    S.apply()
    plt = S.plt
    mc = load_mc()
    n = mc["loc"][0].shape[0]
    c, k = n // 2, int(round(BAND / H_OUT))
    col = lambda d: d[:, c - k:c + k].sum(1).reshape(-1, GROUP).sum(1)  # noqa: E731
    x = ((np.arange(n) + 0.5) * H_OUT - 1.0).reshape(-1, GROUP).mean(1) * 10
    ref = np.mean([col(m) for m in mc["loc"]], 0)
    mc_err = 0.5 * np.abs(col(mc["loc"][0]) - col(mc["loc"][1])) / ref
    sel = (x > -8) & (x < 0)
    print(f"on-axis |y| < {10 * BAND:.0f} mm, 0.5 mm bins, GMC + SCR / MC (1 cm cell from x = -5 mm)")
    print(f"{'x (mm)':>34} " + " ".join(f"{v:6.2f}" for v in x[sel]))
    print(f"{'MC noise (seed half-diff)':>34} " + " ".join(f"{v:6.3f}" for v in mc_err[sel]))
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.axhline(1.0, color="0.5", lw=1)
    ax.axvspan(-5, 5, color="0.93", lw=0, zorder=0)
    ax.text(-4.9, 1.055, "1 cm cell", fontsize=9, color="0.35")
    ax.fill_between(x[sel], 1 - mc_err[sel], 1 + mc_err[sel], step="mid", color="#2a78d6", alpha=0.15, lw=0,
                    label="MC reference noise")
    for (lab, files), col_ in zip(RUNS.items(), ("#e34948", "#2f9e44")):
        G = []
        for f in files:
            F = np.load(OUT / f)
            G += [col(g / (int(F["N"]) * H_OUT ** 2)) for g in F["loc"]]
        G = np.array(G)
        r = G.mean(0) / ref
        e = G.std(0, ddof=1) / np.sqrt(len(G)) / ref
        print(f"{lab + f' ({len(G)} seeds)':>34} " + " ".join(f"{v:6.3f}" for v in r[sel]))
        ent = (x > -5) & (x < -3.5)
        print(f"{'':>34} first 1.5 mm of the 1 cm cell: mean {r[ent].mean():.4f} +- {np.sqrt((e[ent] ** 2).mean() / ent.sum()):.4f}")
        ax.errorbar(x[sel], r[sel], yerr=e[sel], fmt="o-", ms=4, lw=1.5, capsize=0, color=col_,
                    label=f"GMC + SCR, {lab} ({len(G)} seeds)")
    ax.set_xlim(-8, 0)
    ax.set_ylim(0.94, 1.07)
    ax.set_xlabel("x (mm), beam axis")
    ax.set_ylabel("dose ratio GMC + SCR / MC (|y| < 1 mm)")
    ax.set_title("centre:10:0.15625:2: on-axis dose at the 1 cm cell entry", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "dip_check.png", dpi=130)
    print(f"wrote {OUT / 'dip_check.png'}")


if __name__ == "__main__":
    main()
