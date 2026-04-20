#!/usr/bin/env python3
"""
Single-cell 2D Monte Carlo exit distribution visualiser.

Simulates N neutrons inside one rectangular cell with isotropic scattering
(sigma_s=1, sigma_a=0) from a fixed initial position/direction.

Scattering is **3-D isotropic** (uniform on unit sphere); the x-y components
drive the 2-D cell geometry; the z component only contributes to track length.

With --scattered-only, the first free path is sampled from the truncated
exponential conditioned on occurring before the ballistic exit, so every
particle is guaranteed ≥1 scatter and none are discarded.

Usage:
  python validation/single_cell_mc_viz.py
  python validation/single_cell_mc_viz.py --source boundary --scattered-only
  python validation/single_cell_mc_viz.py --width 0.5 --height 1.0 --n 100000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def transport_to_boundary(
    x: float, y: float, u: float, v: float,
    width: float, height: float,
) -> tuple[float, float, float, int, float]:
    """
    Compute the distance and crossing point if a particle at (x,y) with
    direction (u,v) travels in a straight line to the cell boundary.

    Returns (distance, x_exit, y_exit, face, coord_on_face)
    face: 0=left, 1=right, 2=bottom, 3=top
    coord_on_face: position along the face (y for left/right, x for top/bottom)
    """
    tmin = math.inf
    face_hit = -1

    # Left  (x=0):  u < 0
    if u < -1e-14:
        t = -x / u
        if 0.0 < t < tmin:
            yc = y + t * v
            if 0.0 <= yc <= height:
                tmin, face_hit = t, 0
    # Right (x=W):  u > 0
    if u > 1e-14:
        t = (width - x) / u
        if 0.0 < t < tmin:
            yc = y + t * v
            if 0.0 <= yc <= height:
                tmin, face_hit = t, 1
    # Bottom (y=0): v < 0
    if v < -1e-14:
        t = -y / v
        if 0.0 < t < tmin:
            xc = x + t * u
            if 0.0 <= xc <= width:
                tmin, face_hit = t, 2
    # Top   (y=H):  v > 0
    if v > 1e-14:
        t = (height - y) / v
        if 0.0 < t < tmin:
            xc = x + t * u
            if 0.0 <= xc <= width:
                tmin, face_hit = t, 3

    x_exit = x + tmin * u
    y_exit = y + tmin * v
    # clamp to boundary
    x_exit = max(0.0, min(width, x_exit))
    y_exit = max(0.0, min(height, y_exit))
    coord = y_exit if face_hit in (0, 1) else x_exit
    return tmin, x_exit, y_exit, face_hit, coord


# ---------------------------------------------------------------------------
# Single-cell simulation
# ---------------------------------------------------------------------------
FACE_NAMES = {0: "Left", 1: "Right", 2: "Bottom", 3: "Top"}
FACE_COLORS = {0: "#4878cf", 1: "#d65f5f", 2: "#6acc65", 3: "#b47cc7"}


def sample_truncated_exp(rng: np.random.Generator, sigma_t: float, d_max: float) -> float:
    """
    Sample from Exp(sigma_t) conditioned on d < d_max (truncated exponential).
    Uses inverse-CDF: d = -log(1 - U*(1-exp(-sigma_t*d_max))) / sigma_t
    """
    # P(scatter before d_max) = 1 - exp(-sigma_t * d_max)
    p_scatter = -math.expm1(-sigma_t * d_max)  # numerically stable
    u = rng.uniform(0.0, p_scatter)
    return -math.log1p(-u) / sigma_t


def sample_3d_direction(rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample a uniformly random direction on the 3-D unit sphere."""
    # Marsaglia method: use two standard normals then normalise
    while True:
        x = rng.standard_normal()
        y = rng.standard_normal()
        z = rng.standard_normal()
        r = math.sqrt(x * x + y * y + z * z)
        if r > 1e-14:
            return x / r, y / r, z / r


def simulate(
    *,
    width: float,
    height: float,
    sigma_s: float,
    sigma_a: float,
    x0: float,
    y0: float,
    u0: float,
    v0: float,
    n_particles: int,
    seed: int,
    scattered_only: bool = False,
) -> dict:
    """
    Run N independent random walks from (x0,y0) with direction (u0,v0).

    Scattering is **3-D isotropic** (uniform on the unit sphere); the x-y
    components drive the 2-D geometry while the z component contributes to
    the physical track length (cell is infinite in z).

    If scattered_only=True, the first free path is sampled from the truncated
    Exp(sigma_t) conditioned on occurring before the ballistic exit distance,
    so *every* particle is guaranteed to scatter at least once — no particles
    are discarded.
    """
    sigma_t = sigma_s + sigma_a
    rng = np.random.default_rng(seed)

    exits = {f: {"coord": [], "u": [], "v": [], "track": []} for f in range(4)}

    for _ in range(n_particles):
        x, y, u, v = x0, y0, u0, v0
        track = 0.0
        first_step = True

        while True:
            # Distance to boundary in 2-D (geometry unchanged by z-component)
            d_boundary, _xb, _yb, face, coord = transport_to_boundary(x, y, u, v, width, height)

            if scattered_only and first_step:
                # Condition on first scatter happening before the ballistic exit.
                # Sample from Exp(sigma_t) truncated to (0, d_boundary).
                d_free = sample_truncated_exp(rng, sigma_t, d_boundary)
                first_step = False
            else:
                d_free = rng.exponential(1.0 / sigma_t)

            if d_free >= d_boundary:
                # Particle exits without scattering on this leg
                track += d_boundary
                exits[face]["coord"].append(coord)
                exits[face]["u"].append(u)
                exits[face]["v"].append(v)
                exits[face]["track"].append(track)
                break
            else:
                # Scatter: advance by d_free (3-D distance)
                track += d_free
                x = x + d_free * u
                y = y + d_free * v
                # New direction: 3-D isotropic, projected to 2-D for geometry
                u, v, _w = sample_3d_direction(rng)

    # Convert to arrays
    for f in exits:
        for k in exits[f]:
            exits[f][k] = np.asarray(exits[f][k], dtype=np.float64)

    return exits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_exits(exits: dict, *, width: float, height: float, sigma_s: float,
               x0: float, y0: float, u0: float, v0: float,
               n_particles: int, scattered_only: bool, outpath: Path) -> None:

    face_order = [0, 1, 2, 3]
    counts = {f: len(exits[f]["track"]) for f in face_order}
    total = sum(counts.values())
    scatter_tag = "  [scattered ≥1 only]" if scattered_only else ""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Single-cell MC exit distributions{scatter_tag}\n"
        f"W={width:.3f}  H={height:.3f}  σ_s={sigma_s}  σ_a=0\n"
        f"start=({x0:.3f},{y0:.3f})  dir=({u0:.3f},{v0:.3f})  N={n_particles:,}",
        fontsize=11,
    )

    # 4 faces × 4 quantities (coord, u, v, track) = 16 panels, arranged 4×4
    # rows = faces, cols = [coord, u, v, log10(track)]
    col_labels = ["Exit coord along face", "Exit direction u", "Exit direction v", "Track length (log₁₀)"]
    n_rows, n_cols = 4, 4

    for row, face in enumerate(face_order):
        data = exits[face]
        n_f = counts[face]
        pct = 100.0 * n_f / max(total, 1)
        face_label = f"{FACE_NAMES[face]} face  ({n_f:,} / {pct:.1f}%)"

        arrays = [
            data["coord"],
            data["u"],
            data["v"],
            np.log10(np.maximum(data["track"], 1e-12)),
        ]
        coord_range = (0.0, height) if face in (0, 1) else (0.0, width)
        ranges = [coord_range, (-1.05, 1.05), (-1.05, 1.05), None]

        for col, (arr, col_label, rng_) in enumerate(zip(arrays, col_labels, ranges)):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

            if len(arr) == 0:
                ax.text(0.5, 0.5, "no exits", ha="center", va="center", transform=ax.transAxes)
            else:
                bins = 50
                kw = {"color": FACE_COLORS[face], "edgecolor": "white", "linewidth": 0.3,
                      "density": True, "alpha": 0.85}
                if rng_ is not None:
                    ax.hist(arr, bins=bins, range=rng_, **kw)
                    ax.set_xlim(rng_)
                else:
                    ax.hist(arr, bins=bins, **kw)

                ax.axvline(np.median(arr), color="k", lw=1.0, ls="--", label=f"med={np.median(arr):.3f}")
                ax.legend(fontsize=7, handlelength=1)

            if row == 0:
                ax.set_title(col_label, fontsize=8)
            if col == 0:
                ax.set_ylabel(face_label, fontsize=7, labelpad=2)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


def plot_2d_scatter(exits: dict, *, width: float, height: float, outpath: Path) -> None:
    """Scatter plot of exit locations coloured by face."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for face in range(4):
        arr = exits[face]
        if len(arr["coord"]) == 0:
            continue
        if face == 0:   # left
            xs = np.zeros_like(arr["coord"]); ys = arr["coord"]
        elif face == 1:  # right
            xs = np.full_like(arr["coord"], width); ys = arr["coord"]
        elif face == 2:  # bottom
            xs = arr["coord"]; ys = np.zeros_like(arr["coord"])
        else:            # top
            xs = arr["coord"]; ys = np.full_like(arr["coord"], height)

        ax.scatter(xs, ys, s=1, alpha=0.3, color=FACE_COLORS[face],
                   label=f"{FACE_NAMES[face]} ({len(xs):,})", rasterized=True)

    ax.set_xlim(-0.05 * width, 1.05 * width)
    ax.set_ylim(-0.05 * height, 1.05 * height)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Exit positions (coloured by face)")
    ax.legend(markerscale=5, fontsize=8)
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, edgecolor="k", lw=1.5))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-cell MC exit distribution visualiser")
    parser.add_argument("--width",   type=float, default=0.5,  help="Cell width  (default 0.5)")
    parser.add_argument("--height",  type=float, default=0.5,  help="Cell height (default 0.5)")
    parser.add_argument("--sigma-s", type=float, default=1.0,  help="Scatter cross section (default 1.0)")
    parser.add_argument("--x0",      type=float, default=None, help="Initial x (default: centre)")
    parser.add_argument("--y0",      type=float, default=None, help="Initial y (default: centre)")
    parser.add_argument("--u0",      type=float, default=None, help="Initial direction u (default: random)")
    parser.add_argument("--v0",      type=float, default=None, help="Initial direction v (default: random)")
    parser.add_argument("--source",  choices=["internal", "boundary"], default="internal",
                        help="'internal': start at centre; 'boundary': start at left-face mid-point heading right")
    parser.add_argument("--n",       type=int,   default=500_0000, help="Number of particles (default 50000)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--scattered-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Only tally particles that had at least one scatter before exit (default: True)")
    parser.add_argument("--outdir",  default="validation/results/single_cell_mc_viz")
    args = parser.parse_args()

    W, H = args.width, args.height

    # Default start position / direction
    if args.source == "boundary":
        x0 = 0.0
        y0 = H / 2.0
        u0 = 1.0
        v0 = 0.0
    else:
        x0 = W / 2.0
        y0 = H / 2.0
        rng0 = np.random.default_rng(args.seed)
        theta0 = rng0.uniform(0.0, 2.0 * math.pi)
        u0 = float(np.cos(theta0))
        v0 = float(np.sin(theta0))

    # Override with explicit CLI args
    if args.x0 is not None: x0 = args.x0
    if args.y0 is not None: y0 = args.y0
    if args.u0 is not None: u0 = args.u0
    if args.v0 is not None: v0 = args.v0

    # Normalise direction
    mag = math.sqrt(u0 ** 2 + v0 ** 2)
    u0, v0 = u0 / mag, v0 / mag

    print(f"Cell: {W} × {H}   σ_s={args.sigma_s}  σ_a=0")
    print(f"Start: ({x0:.4f}, {y0:.4f})  dir: ({u0:.4f}, {v0:.4f})")
    print(f"Scattered-only: {args.scattered_only}")
    print(f"Running {args.n:,} particles …")

    exits = simulate(
        width=W, height=H,
        sigma_s=args.sigma_s, sigma_a=0.0,
        x0=x0, y0=y0, u0=u0, v0=v0,
        n_particles=args.n,
        seed=args.seed,
        scattered_only=args.scattered_only,
    )

    total = sum(len(exits[f]["track"]) for f in range(4))
    for f in range(4):
        n_f = len(exits[f]["track"])
        med_track = float(np.median(exits[f]["track"])) if n_f else float("nan")
        print(f"  {FACE_NAMES[f]:6s}: {n_f:6,}  ({100*n_f/total:.1f}%)  median track={med_track:.4f}")

    outdir = Path(args.outdir)
    tag = f"W{W}_H{H}_s{args.source}_N{args.n}{'_scattered' if args.scattered_only else ''}"
    plot_exits(exits, width=W, height=H, sigma_s=args.sigma_s,
               x0=x0, y0=y0, u0=u0, v0=v0, n_particles=args.n,
               scattered_only=args.scattered_only,
               outpath=outdir / f"histograms_{tag}.png")
   


if __name__ == "__main__":
    main()
