"""Drumm water cylinder: analog MC with different physics generations vs ITS (no GMC).

Compares analog-MC runs of drumm_cylinder.py mc (true circle) made with different ElectronTransport physics, e.g. the
EEDL elastic tables + binary ionization deflection (before 2026-11-15) and ELSEPA elastic + close/distant ionization
deflection (after), against the digitized ITS curves, with the measures used in the results page's diagnosis:
along-axis energy (level, entry/exit), across-beam rms widths (and their scattering part after removing the
triangle's own rms 0.102 cm), lateral integrals in both photon conventions, and the charge trough at matched strips.

Usage:
    python validation/drumm_physics_compare.py --runs OLD.npz:"EEDL, binary" NEW.npz:"ELSEPA, split" \
        --out validation/results/drumm_water_cylinder/physics_compare
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "drumm_water_cylinder"
NC, L = 20, 0.1
W_TRI = 0.102  # rms of the triangular source profile (cm)


def maps(path):
    F = np.load(path, allow_pickle=False)
    n, N = int(F["n"]), int(F["N"])
    h = L / n
    v = lambda k: F[k] / (N * h * h)  # noqa: E731  per primary per cm^3 (per cm of beam)
    meta = json.loads(str(F["meta"]))
    return {"n": n, "h": h, "esc": v("energy"), "loc": v("energy") + v("photon"), "q_thr": v("charge_thr"),
            "meta": meta}


def its_strip(its_key, its, half_width):
    """ITS across-beam profile averaged over |y| < half_width (the value our strip of that width measures)."""
    x, y = np.asarray(its[its_key]["x"]), np.asarray(its[its_key]["y"])
    g = np.linspace(-half_width, half_width, 401)
    return float(np.interp(g, x, y).mean())


def rms_width(y, f, lim=0.5):
    k = np.abs(y) < lim
    w = np.clip(f[k], 0, None)
    return float(np.sqrt(np.sum(w * y[k] ** 2) / np.sum(w)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="path:label pairs")
    p.add_argument("--strip", type=int, default=1, help="fine bins on each side of a profile line")
    p.add_argument("--out", default=str(OUT / "physics_compare"))
    a = p.parse_args()
    its = json.load(open(OUT / "its_digitized.json"))
    runs = []
    for spec in a.runs:
        path, label = spec.split(":", 1)
        runs.append((label, maps(path)))
    n, h = runs[0][1]["n"], runs[0][1]["h"]
    xf = (np.arange(NC * n) + 0.5) * h - 1.0
    k = a.strip

    def across(d, pos, kk=k):
        i = int(round((pos + 1.0) / h))
        return d[i - kk:i + kk, :].mean(0)

    def along(d, kk=k):
        i = NC * n // 2
        return d[:, i - kk:i + kk].mean(1)

    res = {"runs": {}}
    for label, M in runs:
        r = {"physics": M["meta"].get("physics", {"elastic": "eedl", "ionization_deflection": "binary"})}
        # along-axis energy (photons local), 0.5 mm bins in x
        ix, iy = np.asarray(its["E_along"]["x"]), np.asarray(its["E_along"]["y"])
        ours = np.interp(ix, xf, np.repeat(along(M["loc"]).reshape(-1, 4).mean(1), 4))
        core = (ix > -0.95) & (ix < 0.95)
        rel = (ours - iy) / iy
        r["E_along"] = {"mean_rel": float(rel[core].mean()),
                        "entry_rel(-0.9..-0.6)": float(rel[(ix > -0.9) & (ix < -0.6)].mean()),
                        "exit_rel(0.6..0.9)": float(rel[(ix > 0.6) & (ix < 0.9)].mean())}
        for key, pos in (("E_xm05", -0.5), ("E_x0", 0.0), ("E_xp05", 0.5)):
            ix, iy = np.asarray(its[key]["x"]), np.asarray(its[key]["y"])
            yy = np.where(np.abs(ix) < 0.5, iy, 0.0)
            prof = across(M["loc"], pos)
            w_its, w_mc = rms_width(ix, iy), rms_width(xf, prof)
            sc = lambda w: np.sqrt(max(w * w - W_TRI ** 2, 0.0))  # noqa: E731
            r[key] = {"rms_width_its": w_its, "rms_width_mc": w_mc, "width_ratio": w_mc / w_its,
                      "scattering_width_ratio": float(sc(w_mc) / sc(w_its)),
                      "peak_its": float(iy.max()), "peak_mc": float(prof.max()),
                      "lateral_integral_its": float(np.trapezoid(yy, ix)),
                      "lateral_integral_mc_local": float(prof.sum() * h),
                      "lateral_integral_mc_escape": float(across(M["esc"], pos).sum() * h)}
        # charge: on-axis trough at x = -0.5, 0, 0.5 for strips 1, 0.5, 0.25 mm (x-bin 0.5 mm), vs ITS same strip
        q = {}
        for key, pos in (("Q_xm05", -0.5), ("Q_x0", 0.0), ("Q_xp05", 0.5)):
            for wmm in (1.0, 0.5, 0.25):
                kk = int(round(wmm / 10 / h / 2))
                prof = across(M["q_thr"], pos, kk=2)  # 0.5 mm in x (4 fine bins)
                c = NC * n // 2
                val = float(prof[c - kk:c + kk].mean())
                q[f"{key}_{wmm:g}mm"] = {"mc": val, "its": its_strip(key, its, wmm / 20)}
        r["charge_trough"] = q
        res["runs"][label] = r

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)

    # --- text summary ---
    labels = [lab for lab, _ in runs]
    print(f"{'':38s}" + "".join(f"{lab:>22s}" for lab in labels))
    R = res["runs"]
    for kname in ("mean_rel", "entry_rel(-0.9..-0.6)", "exit_rel(0.6..0.9)"):
        print(f"E along axis vs ITS, {kname:17s}" + "".join(f"{R[lab]['E_along'][kname]:+22.3f}" for lab in labels))
    for key in ("E_xm05", "E_x0", "E_xp05"):
        for kname in ("width_ratio", "scattering_width_ratio"):
            print(f"{key} {kname:33s}" + "".join(f"{R[lab][key][kname]:22.3f}" for lab in labels))
        print(f"{key} peak MC / ITS {R[labels[0]][key]['peak_its']:6.2f}          "
              + "".join(f"{R[lab][key]['peak_mc'] / R[lab][key]['peak_its']:22.3f}" for lab in labels))
        print(f"{key} lateral integral (ITS {R[labels[0]][key]['lateral_integral_its']:.3f}) local / escape "
              + "".join(f"{R[lab][key]['lateral_integral_mc_local']:11.3f}/{R[lab][key]['lateral_integral_mc_escape']:.3f}"
                        for lab in labels))
    for qk in R[labels[0]]["charge_trough"]:
        print(f"charge {qk:20s} ITS {R[labels[0]]['charge_trough'][qk]['its']:+.3f}       "
              + "".join(f"{R[lab]['charge_trough'][qk]['mc']:+22.3f}" for lab in labels))

    # --- figure: energy (photons local) and charge, ITS vs each run ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#8a8a8a", "#2a7ab0", "#d9480f", "#2f9e44"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    panels = [("E_along", None), ("E_xm05", -0.5), ("E_x0", 0.0), ("E_xp05", 0.5)]
    for j, (key, pos) in enumerate(panels):
        a_ = ax[0, j]
        a_.plot(its[key]["x"], its[key]["y"], color="0.1", lw=2.6, label="ITS/ACCEPT (paper)")
        for (lab, M), c in zip(runs, cols):
            prof = np.repeat(along(M["loc"]).reshape(-1, 4).mean(1), 4) if pos is None else across(M["loc"], pos)
            a_.step(xf, prof, where="mid", color=c, lw=1.4, label=f"analog MC, {lab}")
        a_.set_title(("energy along axis" if pos is None else f"energy across beam, x = {pos}") + " (brems local)")
        a_.set_xlim(-1, 1) if pos is None else a_.set_xlim(-0.6, 0.6)
        a_.set_xlabel("position (cm)")
    ax[0, 0].set_ylabel("MeV/cm³ per e⁻/cm")
    ax[0, 0].legend(fontsize=8)
    for j, (key, pos) in enumerate([("Q_along", None), ("Q_xm05", -0.5), ("Q_x0", 0.0), ("Q_xp05", 0.5)]):
        a_ = ax[1, j]
        a_.plot(its[key]["x"], its[key]["y"], color="0.1", lw=2.6, label="ITS/ACCEPT (paper)")
        hq = 4 * h
        xq = (np.arange(NC * n // 4) + 0.5) * hq - 1.0
        for (lab, M), col in zip(runs, cols):
            qm = M["q_thr"].reshape(NC * n // 4, 4, NC * n // 4, 4).mean((1, 3))
            if pos is None:  # the y = 0 line: the 2 coarse bins either side (1 mm strip)
                c = qm.shape[1] // 2
                prof = qm[:, c - 1:c + 1].mean(1)
            else:
                i = int(round((pos + 1.0) / hq))
                prof = qm[i - 1:i + 1, :].mean(0)
            a_.step(xq, prof, where="mid", color=col, lw=1.4, label=f"analog MC, {lab} (1 mm strip, 0.5 mm bins)")
        a_.set_title("charge along axis" if pos is None else f"charge across beam, x = {pos}")
        a_.set_xlim(-1, 1) if pos is None else a_.set_xlim(-0.6, 0.6)
        a_.set_xlabel("position (cm)")
    ax[1, 0].set_ylabel("e⁻/cm³ per e⁻/cm")
    ax[1, 0].legend(fontsize=8)
    fig.suptitle("Drumm water cylinder: analog MC physics generations vs ITS (charge: electrons born > 0.1 MeV)")
    fig.savefig(a.out + ".png", dpi=110)
    print(f"saved {a.out}.json / .png")


if __name__ == "__main__":
    main()
