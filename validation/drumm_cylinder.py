"""
Drumm, Fan, Lorence & Liscum-Powell, "An Analysis of the Extended-Transport Correction with Application to Electron
Beam Transport", Nucl. Sci. Eng. 155, 355-366 (2007), Sec. IV.A / Figs. 8a-8b: a 20 MeV electron beam on the side of
an infinite water cylinder (2D). Compares the analog MC, the chained GMC and GMC + SCR with the paper's ITS (ACCEPT)
profiles (digitized into validation/results/drumm_water_cylinder/its_digitized.json).

Problem: cylinder of diameter 2 cm, axis along z (our infinite-z 2D geometry). Beam along +x, uniform in z, triangular
in y with FWHM 0.25 cm (half-base 0.25 cm), 20 MeV. Paper: electron cutoff 0.1 MeV, no photon transport. Here: 1 keV
cutoff (the GMC training cutoff), photons banked at birth (their energy leaves, like no photon transport).
Coordinates: box [0, 2] x [0, 2] cm, cylinder centre (1, 1); the paper's position = ours - 1.
Units of the paper (checked: 4 /cm peak fluence x 2.0 MeV cm^2/g = 8 at entry): energy deposition in MeV/cm^3 and
charge deposition in electrons/cm^3, per source electron per cm of beam along z (= our tally per primary / voxel
area). Charge deposition = electrons stopped minus electrons created (negative on the beam axis, where knock-ons leave).

Subcommands:
  mc    (ElectronTransport venv) analog MC. --geom circle: the true cylinder on fine voxels (--geo_sub x finer than the
        tally voxels); --geom stair: the 1 mm staircase the GMC uses (cells whose centre is inside the circle are
        water, the rest vacuum), tallied --n x --n per cell. Writes energy and charge tallies (n per 1 mm cell).
  gmc   (GMC venv) chained GMC on the staircase (water model + a vacuum streaming kernel), in batches, several seeds;
        per seed: cell dose, cell charge, SCR fine dose (sub-cell kernel) and SCR fine charge (stop / birth points,
        all electrons and born above --q_th; with --charge_kernel also the irrelevant secondaries born above --q_th)
        on the same n x n grid.
  plot  Figs. 8a/8b layout plus metrics JSON.
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

OUT = ROOT / "validation/results/drumm_water_cylinder"
L, NC, R = 0.1, 20, 1.0           # transport cell (cm), cells per side, cylinder radius (cm)
E0, HALF_BASE = 20.0, 0.25        # MeV, triangle half-base (= FWHM) in cm


def triangle_source(N, seed):
    """Paper's sampling: y = -(1 - sqrt(2r))/4 (r < 0.5), (1 - sqrt(2 - 2r))/4 (r >= 0.5)."""
    r = np.random.default_rng(seed).random(N)
    y = np.where(r < 0.5, -(1 - np.sqrt(2 * r)) * HALF_BASE, (1 - np.sqrt(np.maximum(2 - 2 * r, 0))) * HALF_BASE)
    src = np.zeros((N, 8))
    src[:, 1] = 1.0 + y
    src[:, 2] = 1.0
    src[:, 4] = E0
    return src


def stair_map():
    c = (np.arange(NC) + 0.5) * L - 1.0
    return (np.hypot(c[:, None], c[None, :]) >= R).astype(np.int64)  # 0 water, 1 vacuum


def coarsen(d, n):
    return d.reshape(d.shape[0] // n, n, d.shape[1] // n, n).sum((1, 3))


# ----------------------------------------------------------------------------------------------------------- mc
def cmd_mc(a):
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    os.chdir(et)
    from analog_stepper import VoxelGeometry, run_transport, VACUUM
    from material_data import build_library, named_material
    m, h = named_material("Water")
    lib = build_library([m], [h])
    if a.geom == "circle":
        g = a.n * a.geo_sub
        c = (np.arange(NC * g) + 0.5) * (L / g) - 1.0
        mmap = np.where(np.hypot(c[:, None], c[None, :]) < R, 0, VACUUM)
    else:
        g = a.n
        mmap = np.where(np.repeat(np.repeat(stair_map(), g, 0), g, 1) == 0, 0, VACUUM)
    geo = VoxelGeometry(NC * L, NC * L, mmap)
    src = triangle_source(a.N, a.src_seed)
    acc, bal = {}, []
    edges = np.linspace(0, NC * L, NC * g + 1)
    t0 = time.time()
    for i, b in enumerate(range(0, a.N, a.batch)):
        r = run_transport(lib, geo, src[b:b + a.batch], E_cutoff=a.E_cutoff, seed=a.seed + 1000 * i, n_tracks=0,
                          charge_threshold=a.q_th)
        ph = r["photon_bank"]
        part = {"energy": r["energy_tally"], "charge": r["charge_tally"], "charge_thr": r["charge_tally_thr"],
                "photon": np.histogram2d(ph[:, 0], ph[:, 1], bins=[edges, edges], weights=ph[:, 5])[0]}
        for k, v in part.items():
            acc[k] = v if k not in acc else acc[k] + v
        bal.append(r["energy_balance"])
        print(f"batch {i}: {b + len(src[b:b + a.batch])}/{a.N} primaries, {time.time() - t0:.0f} s", flush=True)
    secs = time.time() - t0
    if g != a.n:
        acc = {k: coarsen(v, g // a.n) for k, v in acc.items()}
    Q = acc["charge"]
    E_in = sum(x["E_in"] for x in bal)
    summ = {k: float(sum(x[k] for x in bal) / E_in) for k in ("E_deposited", "E_escaped_electrons", "E_photons",
                                                              "E_overflow")}
    print(f"{a.geom}: {secs:.0f} s ({1e3 * secs / a.N:.2f} ms/primary); energy fractions {summ}; "
          f"net charge {Q.sum() / a.N:+.4f} per primary")
    out = a.out or str(OUT / f"mc_{a.geom}_n{a.n}_{a.N // 1000}k_s{a.seed}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **acc, n=a.n, N=a.N, source=src,
                        meta=json.dumps({**vars(a), "seconds": secs, "balance": summ,
                                         "physics": {"elastic": getattr(lib, "elastic", "eedl"),
                                                     "ionization_deflection": getattr(lib, "ionization_deflection",
                                                                                      "binary"),
                                                     "close_min_factor": getattr(lib, "close_min_factor", None)}}))
    print(f"saved {out}")


# ----------------------------------------------------------------------------------------------------------- gmc
class LocalPhotons:
    """Wraps a GMC kernel so the photon energy it emits stays in the transit's dose (the paper's convention: no photon
    transport, bremsstrahlung deposited locally). The SCR tally then spreads it with the transit's deposition kernel.
    The kernels still see the model's own dose (dose_cond): dose + photon energy is far outside their training range
    (it inflated the charge kernel's rate 1.75x before this was split off, 2026-11-15)."""

    def __init__(self, k):
        self.k = k

    def _f(self, out):
        out = dict(out)
        out["dose_cond"] = out["dose"]        # the model's own dose: what the SCR / charge kernels condition on
        out["dose"] = out["dose"] + out.get("photon_E", 0.0)
        out["photon_E"] = np.zeros_like(out["dose"])
        return out

    def boundary(self, *a):
        return self._f(self.k.boundary(*a))

    def interior(self, *a):
        return self._f(self.k.interior(*a))


class MirrorY:
    """Exact y-mirror symmetrization of a cell kernel at inference: a random half of the calls is made on the mirrored
    state (y -> 1 - y, v -> -v; x, u, w, E unchanged; in the canonical frame this keeps the entry face) and the outputs,
    daughters and extra exits are mirrored back (bottom <-> top exit classes). The square cell's physics has this
    symmetry, so the expected result is unchanged, while any odd bias of the model (e.g. a mean exit-direction drift)
    cancels. Needed because boundary transits are trained without mirror augmentation (secondary_gmc.classifier_data
    augments only interior rows); a ~6e-4 mean v drift per 1 mm transit moved a 20 MeV beam ~90 um in 2 cm."""

    def __init__(self, k, seed):
        self.k, self.rng = k, np.random.default_rng(seed)

    def _call(self, fn, x, y, u, v, w, E):
        m = self.rng.random(len(E)) < 0.5
        out = fn(x, np.where(m, 1 - y, y), u, np.where(m, -v, v), w, E)
        out = dict(out)
        out["y"] = np.where(m, 1 - out["y"], out["y"])
        out["v"] = np.where(m, -out["v"], out["v"])
        c = np.asarray(out["cls"])
        out["cls"] = np.where(m & (c == 0), 2, np.where(m & (c == 2), 0, c))
        for key in ("daughters", "extra_exits"):
            D = out.get(key)
            if D is not None and len(D["E"]):
                mp = m[np.asarray(D["parent"], dtype=np.int64)]
                D = dict(D)
                D["y"] = np.where(mp, 1 - D["y"], D["y"])
                D["v"] = np.where(mp, -D["v"], D["v"])
                out[key] = D
        return out

    def boundary(self, y_c, u_c, v_c, w, E):
        return self._call(lambda x, y, u, v, w_, e: self.k.boundary(y, u, v, w_, e), np.zeros_like(y_c), y_c, u_c,
                          v_c, w, E)

    def interior(self, x, y, u, v, w, E):
        return self._call(self.k.interior, x, y, u, v, w, E)


class VacuumKernel:
    """Straight streaming through an empty cell (no dose, no energy loss)."""

    def __init__(self):
        self.stuck = 0

    def _go(self, x, y, u, v, w, E):
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(u > 0, (1 - x) / u, np.where(u < 0, -x / u, np.inf))
            ty = np.where(v > 0, (1 - y) / v, np.where(v < 0, -y / v, np.inf))
        t = np.minimum(tx, ty)
        stuck = ~np.isfinite(t)
        self.stuck += int(stuck.sum())
        t = np.where(stuck, 0.0, t)
        xo, yo = np.clip(x + u * t, 0, 1), np.clip(y + v * t, 0, 1)
        cls = np.where(tx <= ty, np.where(u > 0, 1, 3), np.where(v > 0, 2, 0))
        cls = np.where(stuck, 4, cls)
        return {"cls": cls, "x": xo, "y": yo, "u": u, "v": v, "w": w, "E_out": E, "dose": np.zeros_like(E)}

    def boundary(self, y_c, u_c, v_c, w, E):
        return self._go(np.zeros_like(y_c), y_c, u_c, v_c, w, E)

    def interior(self, x, y, u, v, w, E):
        return self._go(x, y, u, v, w, E)


def chain_charge(c):
    """Cell charge (electrons stopped - created) from a transit log: +1 per absorbed water transit, -1 per relevant
    daughter (its own interior transit), +1 per electron entering a cell below the cutoff. Irrelevant secondaries
    are born and stop inside their cell (net 0)."""
    T, P = c["transits"], c["points"]
    q = np.zeros((NC, NC))
    w = T["mat"] == 0
    np.add.at(q, (T["cx"][w], T["cy"][w]), (T["cls"][w] == 4).astype(float) - (T["start"][w] == 1))
    np.add.at(q, (P["cx"], P["cy"]), 1.0)
    return q


def chain_charge_fine(c, n, q_th):
    """Sub-cell (SCR) charge from a transit log on the NC*n x NC*n fine grid, both tallies of the analog MC:
    layer 0 every electron, layer 1 only electrons born above q_th (source electrons always count; Eb is the birth
    energy the chain carries). +1 at the stop point of absorbed water transits (the GMC "stop" flow; boundary rows
    mapped from the canonical frame), -1 at the birth point of every relevant daughter (interior start), +1 at the
    entry point of electrons entering a cell below the cutoff. Irrelevant secondaries (< 10 keV, range ~2 um in
    water) are born and stop in place and are left out of both layers."""
    from chain_core import transform_from_canonical
    T, P = c["transits"], c["points"]
    G = NC * n
    out = np.zeros((2, G, G))

    def add(cx, cy, x, y, wt, thr):
        ix = np.clip((cx * n + np.floor(np.clip(x, 0, 1) * n)).astype(np.int64), cx * n, cx * n + n - 1)
        iy = np.clip((cy * n + np.floor(np.clip(y, 0, 1) * n)).astype(np.int64), cy * n, cy * n + n - 1)
        np.add.at(out[0], (ix, iy), wt)
        np.add.at(out[1], (ix[thr], iy[thr]), wt[thr])

    wat = T["mat"] == 0
    ab = np.flatnonzero(wat & (T["cls"] == 4))
    bnd = T["start"][ab] == 0
    x1, y1 = T["x1"][ab].copy(), T["y1"][ab].copy()
    xb, yb, _, _ = transform_from_canonical(x1[bnd], y1[bnd], np.zeros(bnd.sum()), np.zeros(bnd.sum()),
                                            T["face"][ab][bnd])
    x1[bnd], y1[bnd] = xb, yb
    add(T["cx"][ab], T["cy"][ab], x1, y1, np.ones(len(ab)), T["Eb"][ab] > q_th)
    bi = np.flatnonzero(wat & (T["start"] == 1))
    add(T["cx"][bi], T["cy"][bi], T["x0"][bi], T["y0"][bi], -np.ones(len(bi)), T["E0"][bi] > q_th)
    add(P["cx"], P["cy"], P["x"], P["y"], np.ones(len(P["E"])), P["Eb"] > q_th)
    return out


def cmd_gmc(a):
    import torch
    from chain_secondaries import chain_secondaries
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, scr_tally
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = np.load(a.source)["source"] if a.source else triangle_source(a.N, a.src_seed)
    N = len(src)
    mmap = stair_map()
    kern = {}
    if a.kernel:
        spec, model, _ = load_kernel(a.kernel, dev)
        kern = {0: (spec, model)}
    gen = torch.Generator(device=dev)
    ck = None
    if a.charge_kernel:
        from ml.charge_kernel import load_charge_kernel, charge_tally
        ck = load_charge_kernel(a.charge_kernel, dev)
        if abs(ck["q_th"] - a.q_th) > 1e-12:
            raise ValueError(f"charge kernel threshold {ck['q_th']} != --q_th {a.q_th}")
    res = {"dose": [], "charge": [], "scr": [], "scr_q": [], "scr_qi": [], "chain_s": [], "tally_s": [],
           "transits": []}
    for s in range(a.n_seeds):
        seed = a.seed + 100 * s
        K = [SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=seed), VacuumKernel()]
        dose, q, fine = np.zeros((NC, NC)), np.zeros((NC, NC)), np.zeros((NC * a.n, NC * a.n))
        fq = np.zeros((2, NC * a.n, NC * a.n))
        fqi = np.zeros((NC * a.n, NC * a.n))
        tc = tt = 0.0
        ntr = 0
        for i, b in enumerate(range(0, N, a.batch)):
            K[0] = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=seed + 7 * i)
            if a.mirror:
                K[0] = MirrorY(K[0], seed + 7 * i + 3)
            if a.local_photons:
                K[0] = LocalPhotons(K[0])
            t0 = time.time()
            c = chain_secondaries(K, src[b:b + a.batch], NC, NC, L, a.E_cutoff, log_every=0, material_map=mmap,
                                  transit_log=True)
            tc += time.time() - t0
            dose += c["dose"]
            q += chain_charge(c)
            fq += chain_charge_fine(c, a.n, a.q_th)
            if ck is not None:
                fqi += charge_tally(c, ck, NC, NC, a.n, device=dev, K=a.charge_K,
                                    gen=torch.Generator().manual_seed(seed + 17 + 13 * i))
            ntr += int((c["transits"]["mat"] == 0).sum())
            if kern:
                gen.manual_seed(seed + 5 + 13 * i)
                t0 = time.time()
                f, _ = scr_tally(c, kern, NC, NC, a.n, n_samples=a.K, device=dev, gen=gen, sampling="sobol")
                if dev == "cuda":
                    torch.cuda.synchronize()
                tt += time.time() - t0
                fine += f
            del c
        res["dose"].append(dose)
        res["charge"].append(q)
        res["scr_q"].append(fq)
        res["scr_qi"].append(fqi)
        res["scr"].append(fine)
        res["chain_s"].append(tc)
        res["tally_s"].append(tt)
        res["transits"].append(ntr)
        print(f"seed {seed}: chain {tc:.0f} s ({1e3 * tc / N:.2f} ms/primary, {ntr / N:.1f} water transits/primary), "
              f"SCR tally {tt:.0f} s; deposited {dose.sum() / (N * E0):.4f} of E0, net charge {q.sum() / N:+.4f}, "
              f"vacuum stuck {K[1].stuck}", flush=True)
    out = a.out or str(OUT / f"gmc_n{a.n}_{N // 1000}k.npz")
    np.savez_compressed(out, dose=np.array(res["dose"]), charge=np.array(res["charge"]),
                        scr=np.array(res["scr"]) if kern else np.zeros(0), n=a.n, N=N,
                        scr_charge=np.array(res["scr_q"])[:, 0], scr_charge_thr=np.array(res["scr_q"])[:, 1],
                        scr_charge_thr_irrelevant=np.array(res["scr_qi"]) if ck is not None else np.zeros(0),
                        meta=json.dumps({**vars(a), "chain_s": res["chain_s"], "tally_s": res["tally_s"],
                                         "transits": res["transits"]}))
    print(f"saved {out}")


# ----------------------------------------------------------------------------------------------------------- plot
def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    from scr_chain_compare import noise_split
    its = json.load(open(OUT / "its_digitized.json"))
    ld = lambda p: np.load(p, allow_pickle=False)  # noqa: E731
    C, S1, S2, G, GL = ld(a.circle), ld(a.stair), ld(a.stair_noise), ld(a.gmc), ld(a.gmc_lp)
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    os.makedirs(out_dir, exist_ok=True)
    n = int(C["n"])
    h = L / n
    fine = lambda F, k: F[k] / (int(F["N"]) * h * h)                    # noqa: E731  per primary per cm^3
    cell = lambda F, k: coarsen(F[k], n) / (int(F["N"]) * L * L)        # noqa: E731
    # energy: "esc" = photons leave (our convention), "loc" = photon energy deposited where emitted (the paper's)
    mc = {}
    for name, F in (("circle", C), ("stair", S1), ("stair2", S2)):
        mc[name] = {"esc": fine(F, "energy"), "loc": fine(F, "energy") + fine(F, "photon"),
                    "q": fine(F, "charge"), "q_thr": fine(F, "charge_thr"),
                    "esc_c": cell(F, "energy"), "loc_c": cell(F, "energy") + cell(F, "photon"), "q_c": cell(F, "charge")}
    gm = {}
    for conv, F in (("esc", G), ("loc", GL)):
        NG = int(F["N"])
        gm[conv] = {"scr": [d / (NG * h * h) for d in F["scr"]], "cell": [d / (NG * L * L) for d in F["dose"]],
                    "q_c": [d / (NG * L * L) for d in F["charge"]], "meta": json.loads(str(F["meta"]))}
        if "scr_charge_thr" in F.files:   # SCR charge (stop / birth points); same physics in both photon conventions
            irr = F["scr_charge_thr_irrelevant"] if "scr_charge_thr_irrelevant" in F.files else np.zeros(0)
            if a.no_irrelevant_charge or not len(irr):
                irr = np.zeros_like(F["scr_charge_thr"])
            gm[conv]["sq"] = [(d + di) / (NG * h * h) for d, di in zip(F["scr_charge_thr"], irr)]
    # charge does not depend on the photon convention, and both runs use the same seeds: take one run's seeds
    gsq = gm["esc"].get("sq") or gm["loc"].get("sq", [])
    xf = (np.arange(NC * n) + 0.5) * h - 1.0
    xc = (np.arange(NC) + 0.5) * L - 1.0

    def rb(p_, which):
        """Along-axis profiles averaged over --along_bin fine bins in x (gentle gradient; plotted on the fine grid)."""
        f = a.along_bin
        return np.repeat(p_.reshape(-1, f).mean(1), f) if which == "along" and f > 1 else p_

    qb = a.q_bin  # charge maps are binned q_bin x q_bin fine bins before profiles (0.1 MeV-threshold tally is noisy)
    hq = h * qb
    xq = (np.arange(NC * n // qb) + 0.5) * hq - 1.0

    def line(d, which, pos, coarse=False):
        """Profile along the beam axis (which='along', y = 0) or across it at x = pos: mean over the bins straddling
        the line (it sits on a bin edge): 2 * strip fine bins, or the 2 transport cells."""
        hh, k = (L, 1) if coarse else (h, a.strip)
        i = int(round((pos + 1.0) / hh))
        return d[:, i - k:i + k].mean(1) if which == "along" else d[i - k:i + k, :].mean(0)

    panels = [("along", 0.0, "along beam axis (y = 0)"), ("across", -0.5, "across beam at x = -0.5"),
              ("across", 0.0, "across beam at x = 0"), ("across", 0.5, "across beam at x = 0.5")]
    keys = {"E": ["E_along", "E_xm05", "E_x0", "E_xp05"], "Q": ["Q_along", "Q_xm05", "Q_x0", "Q_xp05"]}
    M = {"N_mc": int(C["N"]), "N_gmc": int(G["N"]), "fine_bin_cm": h, "strip_cm": 2 * a.strip * h, "profiles": {}}
    BLUE, GREY, ORANGE, GREEN, PURPLE = "#2a7ab0", "#8a8a8a", "#d9480f", "#2f9e44", "#7048e8"
    FINE_COLS = ["#0ca678", "#e64980", "#f59f00"]
    for qty in ("E", "Q"):
        strips, along_ax = {}, None
        fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
        for a_, (which, pos, title), key in zip(ax.ravel(), panels, keys[qty]):
            it = its[key]
            ix, iy = np.asarray(it["x"]), np.asarray(it["y"])
            a_.plot(ix, iy, color="0.1", lw=2.8, label="ITS/ACCEPT (paper, digitized)", zorder=1)
            m = {}
            if qty == "E":
                L_ = lambda d: rb(line(d, which, pos), which)  # noqa: E731
                c_loc, c_esc = L_(mc["circle"]["loc"]), L_(mc["circle"]["esc"])
                st_loc, st2_loc = L_(mc["stair"]["loc"]), L_(mc["stair2"]["loc"])
                st_esc, st2_esc = L_(mc["stair"]["esc"]), L_(mc["stair2"]["esc"])
                scr_loc = np.mean([L_(d) for d in gm["loc"]["scr"]], 0)
                scr_esc = np.mean([L_(d) for d in gm["esc"]["scr"]], 0)
                cell_loc = np.mean([line(d, which, pos, True) for d in gm["loc"]["cell"]], 0)
                a_.step(xf, c_esc, where="mid", color=BLUE, lw=0.9, alpha=0.45,
                        label="analog MC, circle, photons escape")
                a_.step(xf, c_loc, where="mid", color=BLUE, lw=1.5, label="analog MC, circle")
                a_.step(xf, st_loc, where="mid", color=GREY, lw=1.0, ls=":", label="analog MC, 1 mm staircase")
                a_.step(xf, scr_loc, where="mid", color=ORANGE, lw=1.4, label="GMC + SCR (1 mm cells)")
                a_.step(xc, cell_loc, where="mid", color=GREEN, lw=1.4, ls="--", label="GMC, cell average")
                ours = np.interp(ix, xf, c_loc)
                core = iy > 0.1 * iy.max()
                hot = st_loc > 0.1 * st_loc.max()
                rr = lambda u, v: float(np.sqrt(np.mean(((u[hot] - v[hot]) / v[hot]) ** 2)))  # noqa: E731
                m = {"mc_vs_its_mean_rel_core": float(np.mean((ours[core] - iy[core]) / iy[core])),
                     "mc_vs_its_mean_abs_rel_core": float(np.mean(np.abs(ours[core] - iy[core]) / iy[core])),
                     "its_peak": float(iy.max()), "mc_peak": float(c_loc.max()),
                     "scr_vs_stair_mean_rel": float(np.mean((scr_loc[hot] - st_loc[hot]) / st_loc[hot])),
                     "scr_vs_stair_rel_rms": rr(scr_loc, st_loc),
                     "scr_esc_vs_stair_esc_mean_rel": float(np.mean((scr_esc[hot] - st_esc[hot]) / st_esc[hot])),
                     "mc_seed_rel_rms_single_run": rr(st2_loc, st_loc) / np.sqrt(2),
                     "photon_share_of_mc_profile": float(np.mean(1 - c_esc[hot] / c_loc[hot]))}
                if which == "across":
                    yy = np.where(np.abs(ix) < 0.5, iy, 0.0)  # drop the digitized baseline offset of the tails
                    m["lateral_integral_MeV_per_cm"] = {"its": float(np.trapezoid(yy, ix)),
                                                        "mc_circle": float(c_loc.sum() * h),
                                                        "mc_circle_photons_escape": float(c_esc.sum() * h),
                                                        "gmc_scr": float(scr_loc.sum() * h)}
            else:
                qm = coarsen(mc["circle"]["q_thr"], qb) / (qb * qb)
                i = int(round((pos + 1.0) / hq))
                cq = qm[:, i - 1:i + 1].mean(1) if which == "along" else qm[i - 1:i + 1, :].mean(0)
                a_.step(xq, cq, where="mid", color=BLUE, lw=1.4,
                        label=f"analog MC, circle, {hq * 10:g} mm bins (electrons born > 0.1 MeV)")
                a_.step(xc, line(mc["stair"]["q_c"], which, pos, True), where="mid", color=PURPLE, lw=1.2, ls="-.",
                        label="analog MC, staircase, 1 mm cells (all electrons)")
                gq = np.mean([line(d, which, pos, True) for d in gm["loc"]["q_c"]], 0)
                a_.step(xc, gq, where="mid", color=GREEN, lw=1.4, ls="--", label="GMC, 1 mm cells")
                if gsq:
                    sm = np.mean([coarsen(d, qb) / (qb * qb) for d in gsq], 0)
                    sq = sm[:, i - 1:i + 1].mean(1) if which == "along" else sm[i - 1:i + 1, :].mean(0)
                    a_.step(xq, sq, where="mid", color=ORANGE, lw=1.4,
                            label=f"GMC + SCR, {hq * 10:g} mm bins (electrons born > 0.1 MeV)")
                    m["scr_extreme"] = float(sq[np.argmax(np.abs(sq))])
                    if which == "across":
                        m["scr_strip_1mm"] = float(sq[np.abs(xq) < hq * 1.0001].mean())
                ours = np.interp(ix, xq, cq)
                M_ = np.abs(iy).max()
                m = {"mc_vs_its_rms_abs": float(np.sqrt(np.mean((ours - iy) ** 2))), "its_scale": float(M_),
                     "its_extreme": float(iy[np.argmax(np.abs(iy))]), "mc_extreme": float(cq[np.argmax(np.abs(cq))])}
                # Like-for-like: ITS averaged the way our curves are. The on-axis trough is ~1.5 mm wide, so a 1 mm
                # strip (MC) or 2 mm strip (GMC cells) is visibly shallower than ITS's finely resolved line.
                if which == "across":
                    def its_binned(edges):
                        k_ = np.digitize(ix, edges) - 1
                        return np.array([iy[k_ == j].mean() if np.any(k_ == j) else np.nan
                                         for j in range(len(edges) - 1)])
                    ib_q = its_binned(np.linspace(-1, 1, len(xq) + 1))
                    ib_c = its_binned(np.linspace(-1, 1, NC + 1))
                    a_.step(xq, ib_q, where="mid", color="0.1", lw=1.0, ls="--",
                            label=f"ITS averaged over our {hq * 10:g} mm bins")
                    a_.step(xc, ib_c, where="mid", color="0.55", lw=1.0, ls="--",
                            label="ITS averaged over 1 mm cells")
                    ok_q, ok_c = np.isfinite(ib_q), np.isfinite(ib_c)
                    m["mc_vs_its_binned_rms_abs"] = float(np.sqrt(np.mean((cq[ok_q] - ib_q[ok_q]) ** 2)))
                    m["gmc_vs_its_cells_rms_abs"] = float(np.sqrt(np.mean((gq[ok_c] - ib_c[ok_c]) ** 2)))
                    m["mc_noise_per_bin"] = float(np.std(
                        (coarsen(mc["stair"]["q_thr"], qb) - coarsen(mc["stair2"]["q_thr"], qb))[
                            i - 1:i + 1, np.abs(xq) < 0.3].mean(0)) / (qb * qb) / np.sqrt(2))
                    m["its_on_axis"] = float(np.interp(0.0, ix, iy))
                    m["its_strip_1mm"] = float(iy[np.abs(ix) < 2 * hq / 2 * 1.0001].mean())
                    m["its_strip_2mm"] = float(iy[np.abs(ix) < L * 1.0001].mean())
                    m["mc_strip_1mm"] = float(cq[np.abs(xq) < hq * 1.0001].mean())
                    m["gmc_strip_2mm"] = float(gq[np.abs(xc) < L * 1.0001].mean())
                    strips[pos] = (m["its_strip_1mm"], m["its_strip_2mm"])
                else:
                    along_ax = a_
            M["profiles"][key] = m
            a_.set_title(title)
            a_.set_xlabel("position (cm)")
            a_.set_ylabel("energy deposition (MeV/cm³ per e⁻/cm)" if qty == "E"
                          else "charge deposition (e⁻/cm³ per e⁻/cm)")
            a_.set_xlim(-1, 1)
        if qty == "Q" and strips:
            # ITS trough averaged over the same strips as our along-axis curves, at the depths where ITS gives the
            # across-beam profile (the along-axis ITS line itself is a much narrower strip)
            ps = sorted(strips)
            along_ax.plot(ps, [strips[p_][0] for p_ in ps], "o", mfc="none", mec="0.1", ms=8, mew=1.6,
                          label=f"ITS across-profile averaged over our {2 * hq * 10:g} mm strip", zorder=5)
            along_ax.plot(ps, [strips[p_][1] for p_ in ps], "s", mfc="none", mec="0.55", ms=8, mew=1.6,
                          label="ITS across-profile averaged over 2 mm (GMC cells)", zorder=5)
        ax[0, 0].legend(fontsize=8, loc="lower left" if qty == "E" else "lower right")
        if qty == "Q":  # the ITS-averaged lines appear only in the across panels
            hs, ls = ax[0, 1].get_legend_handles_labels()
            ax[0, 1].legend([h_ for h_, l_ in zip(hs, ls) if l_.startswith("ITS averaged")],
                            [l_ for l_ in ls if l_.startswith("ITS averaged")], fontsize=8, loc="lower right")
        fig.suptitle(f"20 MeV triangular beam on the side of an infinite water cylinder (Drumm et al. 2007, Fig. 8"
                     f"{'a' if qty == 'E' else 'b'}); "
                     + (f"strips {2 * a.strip * h * 10:g} mm wide" if qty == "E" else f"strips {2 * hq * 10:g} mm wide")
                     + ("; bremsstrahlung deposited locally" if qty == "E" else ""))
        fig.savefig(out_dir / f"fig8{'a_energy' if qty == 'E' else 'b_charge'}.png", dpi=130)
        plt.close(fig)

    # Fine-resolution charge figure: our MC resolved only across the trough (in y), against ITS averaged the same way.
    # Along the axis: a y-strip 2 x q_fine fine bins wide, binned at --q_bin in x. Across the beam: y-bins of q_fine
    # fine bins in a 2 x q_bin x-strip (the gradient in x is gentle). Inside r < R_POOL the circle and the two staircase
    # runs agree within noise (same physics, the geometries only differ at the surface), so they are pooled there.
    R_POOL = 0.95
    xx = (np.arange(NC * n) + 0.5) * h - 1.0
    pool = np.hypot(xx[:, None], xx[None, :]) < R_POOL
    qp = np.where(pool, (mc["circle"]["q_thr"] + mc["stair"]["q_thr"] + mc["stair2"]["q_thr"]) / 3, mc["circle"]["q_thr"])
    n_pool = int(C["N"]) + int(S1["N"]) + int(S2["N"])

    def fine_profile(d, which, pos, qf):
        if which == "along":
            j = NC * n // 2  # y = 0 sits on this fine-bin edge
            return xq, d[:, j - qf:j + qf].mean(1).reshape(-1, qb).mean(1)
        i_ = int(round((pos + 1.0) / h))
        p_ = d[i_ - qb:i_ + qb, :].mean(0).reshape(-1, qf).mean(1)
        return (np.arange(len(p_)) + 0.5) * qf * h - 1.0, p_

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    fine_strips = {}
    for a_, (which, pos, title), key in zip(ax.ravel(), panels, keys["Q"]):
        it = its[key]
        ix, iy = np.asarray(it["x"]), np.asarray(it["y"])
        a_.plot(ix, iy, color="0.1", lw=2.4, label="ITS/ACCEPT (paper, digitized)", zorder=1)
        m = M["profiles"][key]
        for qf, col in zip(a.q_fine, FINE_COLS):
            fx, fp = fine_profile(qp, which, pos, qf)
            w_ = qf * h
            lab = f"{2 * w_ * 10:g} mm strip" if which == "along" else f"{w_ * 10:g} mm bins"
            a_.step(fx, fp, where="mid", color=col, lw=1.1, label=f"analog MC, {lab}")
            if which == "across":
                edges = np.linspace(-1, 1, len(fx) + 1)
                k_ = np.digitize(ix, edges) - 1
                ib = np.array([iy[k_ == j].mean() if np.any(k_ == j) else np.nan for j in range(len(fx))])
                a_.step(fx, ib, where="mid", color=col, lw=1.0, ls="--", alpha=0.9,
                        label=f"ITS averaged over {w_ * 10:g} mm bins")
                # noise of one pooled bin: single-run noise from the two staircase seeds, / sqrt(3 runs)
                s1 = fine_profile(mc["stair"]["q_thr"], which, pos, qf)[1]
                s2 = fine_profile(mc["stair2"]["q_thr"], which, pos, qf)[1]
                sig = float(np.std((s1 - s2)[np.abs(fx) < 0.3]) / np.sqrt(2) / np.sqrt(3))
                fine_strips.setdefault(qf, {})[pos] = float(iy[np.abs(ix) < w_ * 1.0001].mean())
                m[f"its_strip_{2 * w_ * 10:g}mm"] = fine_strips[qf][pos]
                m[f"mc_pooled_strip_{2 * w_ * 10:g}mm"] = float(fp[np.abs(fx) < w_ * 1.0001].mean())
                m[f"mc_pooled_noise_per_bin_{w_ * 10:g}mm"] = sig
                ok = np.isfinite(ib)
                m[f"mc_pooled_vs_its_rms_abs_{w_ * 10:g}mm"] = float(np.sqrt(np.mean((fp[ok] - ib[ok]) ** 2)))
        a_.set_title(title)
        a_.set_xlabel("position (cm)")
        a_.set_ylabel("charge deposition (e⁻/cm³ per e⁻/cm)")
        a_.set_xlim(-1, 1)
        if which == "along":
            along_ax = a_
        else:
            a_.set_xlim(-0.6, 0.6)  # the structure is within |y| < 0.6
    for qf, col in zip(a.q_fine, FINE_COLS):
        st = fine_strips[qf]
        along_ax.plot(sorted(st), [st[p_] for p_ in sorted(st)], "D", mfc="none", mec=col, ms=8, mew=1.8,
                      label=f"ITS across-profile averaged over {2 * qf * h * 10:g} mm", zorder=5)
    ax[0, 0].legend(fontsize=8, loc="lower right")
    ax[0, 1].legend(handles=ax[0, 1].get_legend_handles_labels()[0][1:],
                    labels=ax[0, 1].get_legend_handles_labels()[1][1:], fontsize=8, loc="lower right")
    fig.suptitle(f"Fig. 8b at fine resolution: analog MC ({n_pool / 1e6:g}M histories pooled inside r < {R_POOL} cm, "
                 f"electrons born > 0.1 MeV) vs ITS averaged over the same bins")
    fig.savefig(out_dir / "fig8b_charge_fine.png", dpi=130)
    plt.close(fig)
    M["fine_charge"] = {"pooled_runs": ["circle", "stair", "stair2"], "pool_radius_cm": R_POOL, "N_pooled": n_pool}

    # whole-map scores inside the cylinder (1 mm staircase, like for like): fine 125 um map and 1 mm cells
    inside_c = stair_map() == 0
    inside_f = np.repeat(np.repeat(inside_c, n, 0), n, 1)
    M["maps"] = {}
    for conv in ("loc", "esc"):
        ref, ref2 = mc["stair"][conv], mc["stair2"][conv]
        hot_f = inside_f & (ref > 0.01 * ref.max())
        refc, ref2c = mc["stair"][conv + "_c"], mc["stair2"][conv + "_c"]
        hot_c = inside_c & (refc > 0.01 * refc.max())
        flat = [np.repeat(np.repeat(g, n, 0), n, 1) for g in gm[conv]["cell"]]
        M["maps"][conv] = {
            "fine_gmc_scr": noise_split(gm[conv]["scr"], ref, ref2, lambda d: d[hot_f]),
            "fine_gmc_flat": noise_split(flat, ref, ref2, lambda d: d[hot_f]),
            "cell_gmc": noise_split(gm[conv]["cell"], refc, ref2c, lambda d: d[hot_c]),
            "fine_circle_vs_stair_rel_L2": float(np.sqrt(((mc["circle"][conv] - ref)[hot_f] ** 2).sum()
                                                         / (ref[hot_f] ** 2).sum())),
            "deposited_frac_of_E0": {"mc_stair": float(refc.sum() * L * L / E0),
                                     "gmc": [float(d.sum() * L * L / E0) for d in gm[conv]["cell"]]}}
    hot_c = inside_c & (np.abs(mc["stair"]["q_c"]) > 0.02 * np.abs(mc["stair"]["q_c"]).max())
    M["maps"]["charge_cell_gmc"] = noise_split(gm["loc"]["q_c"], mc["stair"]["q_c"], mc["stair2"]["q_c"],
                                               lambda d: d[hot_c])
    M["net_charge_per_primary"] = {"mc_stair": float(mc["stair"]["q_c"].sum() * L * L),
                                   "gmc": [float(d.sum() * L * L) for d in gm["loc"]["q_c"]]}
    if gsq:
        cq = lambda d: coarsen(d, qb) / (qb * qb)  # noqa: E731
        qref, qref2 = cq(mc["stair"]["q_thr"]), cq(mc["stair2"]["q_thr"])
        inside_q = np.repeat(np.repeat(inside_c, n // qb, 0), n // qb, 1)
        hot_q = inside_q & (np.abs(qref) > 0.05 * np.abs(qref).max())
        M["maps"]["charge_thr_scr"] = {**noise_split([cq(d) for d in gsq], qref, qref2, lambda d: d[hot_q]),
                                       "bins_mm": hq * 10, "n_bins": int(hot_q.sum())}
        M["net_charge_per_primary"]["gmc_scr_thr"] = [float(d.sum() * h * h) for d in gsq]
        M["net_charge_per_primary"]["mc_stair_thr"] = float(mc["stair"]["q_thr"].sum() * h * h)
    like_for_like(out_dir, M, mc, gm, gsq, panels, line, rb, xf, xc, xq, qb, hq, plt)
    M["mc_balance"] = {k_: json.loads(str(F["meta"]))["balance"] for k_, F in (("circle", C), ("stair", S1))}
    M["timing_s"] = {"mc_circle": json.loads(str(C["meta"]))["seconds"], "mc_stair": json.loads(str(S1["meta"]))["seconds"],
                     "gmc_chain": gm["loc"]["meta"]["chain_s"], "scr_tally": gm["loc"]["meta"]["tally_s"],
                     "water_transits": gm["loc"]["meta"]["transits"]}
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(M, fh, indent=1)
    print(json.dumps(M, indent=1))

    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    vmax = mc["circle"]["loc"].max()
    flat0 = np.repeat(np.repeat(gm["loc"]["cell"][0], n, 0), n, 1)
    for a_, d, t in ((ax[0], mc["circle"]["loc"], "analog MC, true circle"),
                     (ax[1], mc["stair"]["loc"], "analog MC, 1 mm staircase"),
                     (ax[2], gm["loc"]["scr"][0], "GMC + SCR (one seed)"), (ax[3], flat0, "GMC, cell average")):
        im = a_.imshow(np.maximum(d.T, 1e-6), origin="lower", extent=[-1, 1, -1, 1], norm=LogNorm(vmax * 1e-3, vmax),
                       cmap=S.SEQ)
        a_.add_patch(plt.Circle((0, 0), 1, fill=False, color="w", lw=0.6))
        a_.set_title(t)
        a_.set_xlabel("x (cm)")
    ax[0].set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, label="MeV/cm³ per e⁻/cm", shrink=0.85)
    fig.savefig(out_dir / "dose_maps.png", dpi=120)
    plt.close(fig)


def like_for_like(out_dir, M, mc, gm, gsq, panels, line, rb, xf, xc, xq, qb, hq, plt):
    """GMC vs the analog MC on the same 1 mm staircase, the GMC's own convention (photons escape; charge of electrons
    born > 0.1 MeV): MC = mean of the two staircase seeds with a +-2 sd band from their difference."""
    BLUE, ORANGE, GREEN = "#2a7ab0", "#d9480f", "#2f9e44"
    fig, ax = plt.subplots(2, 4, figsize=(21, 9), constrained_layout=True)
    M["like_for_like"] = {}
    for row, (qty, key_mc, key_g) in enumerate((("E", "esc", "scr"), ("Q", "q_thr", "sq"))):
        if qty == "Q" and not gsq:
            continue
        for a_, (which, pos, title) in zip(ax[row], panels):
            if qty == "E":
                f = lambda d: rb(line(d, which, pos), which)  # noqa: E731
                x, s1, s2 = xf, f(mc["stair"]["esc"]), f(mc["stair2"]["esc"])
                g = [f(d) for d in gm["esc"]["scr"]]
                cell = np.mean([line(d, which, pos, True) for d in gm["esc"]["cell"]], 0)
            else:
                def f(d):
                    dm = coarsen(d, qb) / (qb * qb)
                    i = int(round((pos + 1.0) / hq))
                    return dm[:, i - 1:i + 1].mean(1) if which == "along" else dm[i - 1:i + 1, :].mean(0)
                x, s1, s2 = xq, f(mc["stair"]["q_thr"]), f(mc["stair2"]["q_thr"])
                g = [f(d) for d in gsq]
                cell = None
            ref = (s1 + s2) / 2
            # sd of the 2-run mean: |s1 - s2| / 2 is a 1-dof estimate per bin, so smooth its square over 5 bins
            sd = np.sqrt(np.convolve(((s1 - s2) / 2) ** 2, np.ones(5) / 5, mode="same"))
            gmean = np.mean(g, 0)
            a_.fill_between(x, ref - 2 * sd, ref + 2 * sd, step="mid", color=BLUE, alpha=0.25, lw=0,
                            label="analog MC staircase, +-2 sd")
            a_.step(x, ref, where="mid", color=BLUE, lw=1.3, label="analog MC staircase (2 seeds)")
            a_.step(x, gmean, where="mid", color=ORANGE, lw=1.3, label=f"GMC + SCR ({len(g)} seeds)")
            if cell is not None:
                a_.step(xc, cell, where="mid", color=GREEN, lw=1.2, ls="--", label="GMC, cell average")
            hot = np.abs(ref) > 0.1 * np.abs(ref).max()
            gsd = (np.sqrt(np.convolve(np.var(g, 0, ddof=1) / len(g), np.ones(5) / 5, mode="same")) if len(g) > 1
                   else np.zeros_like(gmean))
            z = (gmean - ref)[hot] / np.sqrt(sd[hot] ** 2 + gsd[hot] ** 2 + 1e-30)
            M["like_for_like"][f"{qty}_{which}_{pos:g}"] = {
                "rel_L2": float(np.linalg.norm((gmean - ref)[hot]) / np.linalg.norm(ref[hot])),
                "mc_rel_noise": float(np.linalg.norm(sd[hot]) / np.linalg.norm(ref[hot])),
                "rms_z": float(np.sqrt(np.mean(z ** 2))), "max_abs_z": float(np.abs(z).max())}
            a_.set_title(("energy, photons escape: " if qty == "E" else "charge, born > 0.1 MeV: ") + title)
            a_.set_xlabel("position (cm)")
            a_.set_xlim(-1, 1) if which == "along" else a_.set_xlim(-0.6, 0.6)
        ax[row, 0].set_ylabel("MeV/cm³ per e⁻/cm" if qty == "E" else "e⁻/cm³ per e⁻/cm")
        ax[row, 0].legend(fontsize=8)
    fig.suptitle("Drumm water cylinder: GMC + SCR vs analog MC on the same 1 mm staircase (like for like; "
                 f"energy on {xf[1] - xf[0]:.4g} cm bins, charge on {hq * 10:g} mm bins)")
    fig.savefig(out_dir / "gmc_vs_mc.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("mc")
    p.add_argument("--geom", choices=("circle", "stair"), required=True)
    p.add_argument("--N", type=int, default=500000)
    p.add_argument("--n", type=int, default=8, help="tally bins per 1 mm cell per axis")
    p.add_argument("--geo_sub", type=int, default=4, help="circle: geometry voxels per tally bin per axis")
    p.add_argument("--batch", type=int, default=250000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--src_seed", type=int, default=2007)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--q_th", type=float, default=0.1, help="birth-energy threshold of the charge_thr tally (MeV)")
    p.add_argument("--out", default=None)
    p = sp.add_parser("gmc")
    p.add_argument("--model", required=True)
    p.add_argument("--kernel", default=None)
    p.add_argument("--source", default=None, help="npz with 'source' (an mc output); default: --N, --src_seed")
    p.add_argument("--N", type=int, default=500000)
    p.add_argument("--src_seed", type=int, default=2007)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--K", type=int, default=None, help="kernel samples per transit (default auto)")
    p.add_argument("--batch", type=int, default=100000)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--local_photons", action="store_true", help="photon energy deposited in the emitting transit")
    p.add_argument("--mirror", action="store_true", help="y-mirror symmetrization of the GMC (MirrorY)")
    p.add_argument("--q_th", type=float, default=0.1, help="birth-energy threshold of the scr_charge_thr tally (MeV)")
    p.add_argument("--charge_kernel", default=None,
                   help="ml/charge_kernel.py model: expected sub-cell charge of the irrelevant secondaries born above "
                        "--q_th, saved as scr_charge_thr_irrelevant (add it to scr_charge_thr)")
    p.add_argument("--charge_K", type=int, default=1, help="charge-kernel pair samples per transit")
    p.add_argument("--out", default=None)
    p = sp.add_parser("plot")
    for k in ("circle", "stair", "stair_noise", "gmc", "gmc_lp"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--strip", type=int, default=1, help="fine bins on each side of a profile line")
    p.add_argument("--along_bin", type=int, default=4, help="fine bins averaged in x for along-axis energy")
    p.add_argument("--q_bin", type=int, default=4, help="fine bins per charge bin (per axis)")
    p.add_argument("--out_dir", default=None, help="figures + metrics.json (default the results folder)")
    p.add_argument("--no_irrelevant_charge", action="store_true",
                   help="leave out the charge kernel's irrelevant-secondary term of the SCR charge")
    p.add_argument("--q_fine", type=int, nargs="*", default=[2, 1],
                   help="extra MC charge lines resolved in y: fine bins per y-bin across the beam (strip = 2x this "
                        "along the axis); x stays at the --q_bin binning")
    a = ap.parse_args()
    {"mc": cmd_mc, "gmc": cmd_gmc, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
