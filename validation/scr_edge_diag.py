#!/usr/bin/env python3
"""
SCR kernel near the cell faces: does the kernel put the right dose in the sub-cell rows next to each face?

Motivation: in the Drumm water-cylinder benchmark the y = 0 beam axis lies on transport-cell faces, and GMC + SCR reads
3-5 % low on that strip at 2-7 mm depth while cell totals agree. This script measures, on held-out analog data, the
dose the kernel places in a strip of width w along each of the four faces of the canonical cell (entered through the
left face), against the truth, per energy band and entry geometry.

Per held-out transit: true image (from the _subcell.npz) and the kernel's expected image (K Sobol samples). Sums over a
group of transits give strip masses; the ratio pred / true with its noise (transit-to-transit scatter of the
difference) tells whether a face strip is under- or over-filled.

Strips (canonical frame): entry = x < w (left), exit-side = x > 1 - w (right), bottom = y < w, top = y > 1 - w. Boundary
transits are folded so the entry point is in the lower half (y0 <= 0.5, y-mirror of image and features), which makes
"near" = the lateral face closest to the entry point.

Usage (GMC venv, from the GMC root):
    python validation/scr_edge_diag.py --model ml/models/subcell_kernel_water_1mm_30MeV_elsepa \
        --test_npz ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_2M_7_elsepa.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ml import subcell_kernel as sk  # noqa: E402

D_EDGES = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 0.5 + 1e-9)   # distance of the entry point to the nearest lateral face


def predicted_images(model, spec, Rd, rows, n, K, sampling, gen):
    """Expected images (len(rows), n, n) from K samples per transit."""
    chunk = max(1, (1 << 20) // K)
    out = []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        h = model.context(sk.cond_features(spec, sk.gather(Rd, r)))
        z = model.sample(h, K, gen=gen, base=sampling)
        b = (z[..., 0] * n).long().clamp(0, n - 1) * n + (z[..., 1] * n).long().clamp(0, n - 1)
        img = torch.zeros(len(r), n * n, device=z.device)
        img.scatter_add_(1, b, torch.full_like(b, 1.0 / K, dtype=torch.float32))
        out.append(img.reshape(len(r), n, n))
    return torch.cat(out)


def true_images(ent, sel, n_sub):
    """Dense true images for the selected transit rows (vectorized over the entry table)."""
    pos = np.full(int(ent["seg"].max()) + 1, -1, dtype=np.int64)
    pos[sel] = np.arange(len(sel))
    p = pos[ent["seg"]]
    ok = p >= 0
    img = np.zeros((len(sel), n_sub * n_sub))
    np.add.at(img, (p[ok], ent["bin"][ok].astype(np.int64)), ent["E"][ok])
    return img.reshape(len(sel), n_sub, n_sub)


def strip_masses(img, w_bins):
    """(N, n, n) images, index [ix, iy] -> (N, 4) masses: entry (left), exit side (right), bottom, top strips."""
    k = w_bins
    return np.stack([img[:, :k, :].sum((1, 2)), img[:, -k:, :].sum((1, 2)),
                     img[:, :, :k].sum((1, 2)), img[:, :, -k:].sum((1, 2))], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--test_npz", required=True)
    p.add_argument("--n_transits", type=int, default=300000, help="held-out transits per energy selection")
    p.add_argument("--E_min", type=float, default=1.0, help="MeV; only transits with E0 above this")
    p.add_argument("--K", type=int, default=1024, help="kernel samples per transit")
    p.add_argument("--sampling", default="sobol")
    p.add_argument("--widths", default="1,4", help="strip widths in data sub-bins")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="json output (default: next to the model)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec, model, _ = sk.load_kernel(a.model, dev)
    spec_t, R, ent = sk.load_kernel_data(a.test_npz)
    n = spec_t.n_sub
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    cand = np.flatnonzero(R["E0"] >= a.E_min)
    sel = np.sort(rng.choice(cand, size=min(a.n_transits, len(cand)), replace=False))
    print(f"{len(sel)} transits with E0 >= {a.E_min} MeV (of {len(cand)}), {n}x{n} grid, K = {a.K} {a.sampling}",
          flush=True)

    # fold boundary transits so the entry point is in the lower half: mirror y in features and in both images
    Rs = {k: np.asarray(v)[sel].copy() for k, v in R.items()}
    flip = (Rs["start"] < 0.5) & (Rs["y0"] > 0.5)
    Rf = sk.apply_d4(torch.as_tensor(np.where(flip, 6, 0)), {k: torch.as_tensor(np.asarray(v, dtype=np.float32))
                                                             for k, v in Rs.items()})
    Rf = {k: v.to(dev) for k, v in Rf.items()}
    T = true_images(ent, sel, n)
    T[flip] = T[flip][:, :, ::-1]
    P = predicted_images(model, spec, Rf, torch.arange(len(sel), device=dev), n, a.K, a.sampling, gen).cpu().numpy()
    P = P * Rs["dose"][:, None, None]

    band = np.digitize(Rs["E0"], sk.E_BANDS) - 1
    y0 = Rf["y0"].cpu().numpy()
    dist = np.where(Rs["start"] < 0.5, np.minimum(y0, 1 - y0), np.nan)
    dbin = np.digitize(dist, D_EDGES) - 1
    cls = Rs["cls"].astype(int)
    names = ("entry(left)", "exit-side(right)", "near(bottom)", "far(top)")
    widths = [int(w) for w in a.widths.split(",")]
    res = {"model": a.model, "test_npz": a.test_npz, "n": int(len(sel)), "K": a.K, "n_sub": n, "rows": []}

    def report(label, m):
        if m.sum() < 200:
            return
        row = {"group": label, "n": int(m.sum()), "dose": float(Rs["dose"][m].sum())}
        line = f"  {label:<44} n={int(m.sum()):>7}"
        for w in widths:
            st, sp = strip_masses(T[m], w), strip_masses(P[m], w)
            tt, pp = st.sum(0), sp.sum(0)
            sd = np.sqrt(((st - sp) ** 2).sum(0))          # noise of (pred - true) sum (transits independent)
            row[f"w{w}"] = {nm: {"true_frac": float(tt[j] / row["dose"]), "ratio": float(pp[j] / max(tt[j], 1e-300)),
                                 "sigma": float(sd[j] / max(tt[j], 1e-300))} for j, nm in enumerate(names)}
            line += f" | w={w}: " + " ".join(f"{nm.split('(')[0][:5]} {pp[j] / tt[j]:.3f}±{sd[j] / tt[j]:.3f}"
                                             for j, nm in enumerate(names))
        res["rows"].append(row)
        print(line, flush=True)

    bnd = Rs["start"] < 0.5
    print("ratio = kernel / truth of the dose in each face strip (sigma = noise of the ratio); w in sub-bins of "
          f"{spec_t.L * 1e4 / n:.1f} um")
    for b in range(sk.NB):
        mb = band == b
        if mb.sum() < 200:
            continue
        print(f"{sk.BAND_NAMES[b]}:")
        report(f"{sk.BAND_NAMES[b]} all", mb)
        report(f"{sk.BAND_NAMES[b]} interior starts", mb & ~bnd)
        for ce, cn in ((1, "right"), (0, "bottom(near)"), (2, "top(far)"), (3, "left"), (4, "absorbed")):
            report(f"{sk.BAND_NAMES[b]} boundary exit {cn}", mb & bnd & (cls == ce))
        for d in range(len(D_EDGES) - 1):
            report(f"{sk.BAND_NAMES[b]} boundary y0-dist [{D_EDGES[d]:.3f},{D_EDGES[d + 1]:.3f})",
                   mb & bnd & (dbin == d))
        # shallow entries (grazing: nearly parallel to the entry face) vs normal
        u0 = Rs["u0"]
        report(f"{sk.BAND_NAMES[b]} boundary u0 < 0.5 (grazing)", mb & bnd & (u0 < 0.5))
        report(f"{sk.BAND_NAMES[b]} boundary u0 >= 0.9", mb & bnd & (u0 >= 0.9))

    out = a.out or os.path.join(a.model, f"edge_diag_{os.path.basename(a.test_npz)[:-4]}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(out[:-5] + "_profiles.npz", true_lat=T.sum(1), pred_lat=P.sum(1), true_dep=T.sum(2),
                        pred_dep=P.sum(2), band=band, dist=dist, cls=cls, start=Rs["start"], u0=Rs["u0"],
                        dose=Rs["dose"], E0=Rs["E0"])
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
