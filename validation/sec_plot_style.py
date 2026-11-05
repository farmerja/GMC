"""Shared matplotlib style for the secondary-GMC validation plots (static PNGs for the repo).

Colors (validated categorical order; aqua and yellow sit below 3:1 on the light surface, so every series also has its
own line style / marker and a legend, and all numbers are written to the metrics JSON):
    analog MC (reference)       neutral ink, solid, with a 1-sigma band where shown
    GMC + secondaries           slot 1 blue
    GMC primary-only            slot 2 orange
    MC-chained + secondaries    slot 3 aqua
    MC-chained primary-only     slot 4 yellow
Dose maps use viridis (log scale): many hues for telling nearby levels apart, while lightness still rises
monotonically with dose, so it stays readable in grayscale and for color-vision deficiencies (unlike a rainbow).
Ratio maps use the blue <-> red diverging ramp with a gray midpoint.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REF = "#3d3c39"
SERIES = {
    "mc": dict(color=REF, ls="-", lw=2.0, label="Analog MC (full grid)"),
    "gmc_sec": dict(color="#2a78d6", ls="-", lw=2.0, marker="o", ms=4, label="GMC + secondaries"),
    "gmc_prim": dict(color="#eb6834", ls="--", lw=2.0, marker="s", ms=4, label="GMC primary-only"),
    "mcc_sec": dict(color="#1baf7a", ls="-.", lw=2.0, marker="^", ms=4, label="MC-chained + secondaries"),
    "mcc_prim": dict(color="#eda100", ls=":", lw=2.2, marker="v", ms=4, label="MC-chained primary-only"),
    "noise": dict(color="#8a8984", ls=(0, (1, 1)), lw=1.5, label="Analog MC, 2nd seed"),
}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = plt.get_cmap("viridis")
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ee8b8a", "#b52f2e"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "lines.markeredgecolor": SURFACE,
        "lines.markeredgewidth": 1.0,
    })


def style(name, **over):
    s = dict(SERIES[name])
    s.update(over)
    return s
