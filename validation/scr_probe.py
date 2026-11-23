"""
Targeted test of the size-conditioned SCR kernel on one cell size and a narrow entry distribution (the Drumm 1 cm cell
dip, 2026-11-22): analog MC data in the size-sweep training format, then kernel vs MC depth / lateral profiles of the
boundary transits at fine resolution.

  mc    (ElectronTransport venv) generate_size_sweep.make_file with one cell side L and a Drumm-like source on the
        left face: E log-uniform in [E_low, E_high], y around the face centre (+- y_half), in-plane and out-of-plane
        angles Gaussian (sigma_deg). Writes <out_dir>/part_0000(.npz, _subcell.npz).
  eval  kernel vs MC: depth profile (x, summed over y) and lateral profile near the entry face, per energy band of the
        boundary transits (primaries entering at x = 0 and the daughters / resumed segments are reported apart).

    ../ElectronTransport/.venv/bin/python validation/scr_probe.py mc --L 1.0 --N 20000 --out_dir <dir>
    .venv/bin/python validation/scr_probe.py eval --data <dir>/part_0000.npz --kernel ml/models/size_scr_water_100um-1cm_30MeV_v2
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ET = ROOT.parent / "ElectronTransport"


def cmd_mc(a):
    sys.path.insert(0, str(ET))
    out_dir = os.path.abspath(a.out_dir)
    a.entries = a.entries and os.path.abspath(a.entries)
    os.chdir(ET)
    import AnalogMC
    import generate_size_sweep as G
    from material_data import build_library, named_material

    def drumm_like(N, y_max, E_in_low, E_in_high, v_spread, benchmark_mode, rng, w_max_angle_deg):
        p = np.zeros((N, 8))
        p[:, 1] = y_max / 2 + rng.uniform(-a.y_half, a.y_half, N)
        th, ph = np.radians(a.sigma_deg) * rng.standard_normal((2, N))
        d = np.stack([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph)], 1)
        p[:, 2], p[:, 3], p[:, 7] = d[:, 0], d[:, 1], d[:, 2]
        p[:, 4] = rng.uniform(E_in_low, E_in_high, N)
        return p

    if a.entries:
        # the boundary entries an analog chain logged into one cell (canonical frame: entry face at x = 0)
        F = np.load(a.entries)
        mesh_L = F["L"]
        cid = int(np.argmax(mesh_L)) if a.cell < 0 else a.cell
        m = (F["T_cid"] == cid) & (F["T_start"] == 0)
        rows = np.flatnonzero(m)
        rows = np.random.default_rng(a.seed).permutation(rows)[:a.N]
        a.N = len(rows) // a.inner * a.inner
        rows = rows[:a.N]
        a.L = float(mesh_L[cid])
        pool = {k: F["T_" + k][rows].astype(np.float64) for k in ("y0", "u0", "v0", "w0", "E0")}
        print(f"cell {cid} (L = {10 * a.L:.3f} mm): {int(m.sum())} logged boundary entries, using {a.N}")
        state = {"i": 0}

        def logged(N, y_max, *args_):
            i = state["i"]
            state["i"] += N
            p = np.zeros((N, 8))
            p[:, 1] = pool["y0"][i:i + N] * y_max
            p[:, 2], p[:, 3], p[:, 7] = pool["u0"][i:i + N], pool["v0"][i:i + N], pool["w0"][i:i + N]
            p[:, 4] = pool["E0"][i:i + N]
            return p
        AnalogMC.generate_particles = logged
        orig = G.run_simulation
        G.run_simulation = lambda *x, **k: orig(*x, **{**k, "energy_sampling": "uniform"})
    else:
        AnalogMC.generate_particles = drumm_like
    args = SimpleNamespace(seed=a.seed, file_size=a.N, inner=a.inner, L_low=a.L, L_high=a.L, bin_um=10.0, min_bins=64,
                           material="Water", E_low=a.E_low, E_high=a.E_high, E_cutoff=1e-3, w_max_angle_deg=90.0,
                           v_spread=1.0, secondary_threshold=10, stack_size=4096, bank_factor=60.0,
                           segment_factor=400.0, photons_relevant=False, elastic="elsepa", stopping="estar",
                           charge_pair_min_E=0.05, out_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mat, h5 = named_material("Water")
    lib = build_library([mat], h5, E_max=100.0, elastic="elsepa", ionization_deflection="split", stopping="estar")
    csda = G.csda_table(lib, 1e-3, 100.0)
    print(G.make_file(0, args, lib, csda, a.N // a.inner))


def cmd_eval(a):
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "validation"))
    import ml.subcell_kernel as sk
    from size_cell_compare import truth_points
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, R, ent = sk.load_kernel_data(a.data)
    starts = np.searchsorted(ent["seg"], np.arange(len(R["E0"]) + 1))
    kernels = [(p, sk.load_kernel(p, dev)) for p in a.kernel]
    nb = a.bins
    bnd = (R["start"] < 0.5) & (R["L"] >= a.L_min)
    groups = [("primary-like entries, E0 > 10 MeV", bnd & (R["E0"] > 10)),
              ("boundary entries, 1-10 MeV", bnd & (R["E0"] > 1) & (R["E0"] <= 10)),
              ("boundary entries, 0.1-1 MeV", bnd & (R["E0"] > 0.1) & (R["E0"] <= 1)),
              ("boundary entries, < 0.1 MeV", bnd & (R["E0"] <= 0.1)),
              ("interior starts", (R["start"] >= 0.5) & (R["L"] >= a.L_min))]
    edges = np.linspace(0, 1, nb + 1)
    for name, m in groups:
        rows = np.flatnonzero(m)
        if len(rows) > a.max_rows:
            rows = np.sort(np.random.default_rng(0).choice(rows, a.max_rows, replace=False))
        if len(rows) == 0:
            continue
        kt, xt, yt, Et = truth_points(ent, rows, starts)
        # per-transit depth histograms (for the noise of the MC profile: transits are independent)
        T = np.zeros((len(rows), nb))
        np.add.at(T, (kt, np.clip((xt * nb).astype(int), 0, nb - 1)), Et)
        tot = T.sum()
        prof_t, err_t = T.sum(0), np.sqrt(((T - T.mean(0)) ** 2).sum(0) * len(rows) / max(len(rows) - 1, 1))
        lines = []
        for path, (spec, model, _) in kernels:
            P = np.zeros(nb)
            Rd = sk.to_device(R, dev)
            with torch.no_grad():
                for i in range(0, len(rows), 2000):
                    rr = rows[i:i + 2000]
                    z = model.sample(model.context(sk.cond_features(spec, sk.gather(Rd, torch.as_tensor(rr, device=dev)))),
                                     a.n_samples, base="sobol").cpu().numpy()
                    w = np.repeat(R["dose"][rr] / a.n_samples, a.n_samples)
                    P += np.histogram(z[..., 0].ravel(), edges, weights=w)[0]
            lines.append((Path(path).name, P))
        print(f"\n{name}: {len(rows)} transits, dose {tot / len(rows):.4g} MeV/transit; depth profile "
              f"(x / L, {nb} bins): kernel / MC (MC rel. error)")
        sel = list(range(min(12, nb))) + list(range(12, nb, max(1, nb // 16)))
        print("  x/L      " + " ".join(f"{(edges[i] + edges[i + 1]) / 2:6.3f}" for i in sel))
        print("  MC err   " + " ".join(f"{err_t[i] / max(prof_t[i], 1e-300):6.3f}" for i in sel))
        print("  MC share " + " ".join(f"{prof_t[i] / tot:6.4f}" for i in sel))
        for lab, P in lines:
            print(f"  {lab[-14:]:>8} " + " ".join(f"{P[i] / max(prof_t[i], 1e-300):6.3f}" for i in sel)
                  + f"   | total {P.sum() / tot:.4f}")


def cmd_gmc(a):
    """GMC transits (MirrorY, as in the chains) of the probe's own source entries + their daughters in the cell,
    placed by the SCR kernel, vs the MC images of the same entries (all segments): depth profile per primary, split
    into the entering transits and the interior starts (daughters)."""
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "validation"))
    import ml.subcell_kernel as sk
    from ml.secondary_gmc import SecondaryGMC
    from chain_quadtree import MirrorY
    from size_cell_compare import truth_points
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = np.load(a.data)
    _, R, ent = sk.load_kernel_data(a.data)
    L = float(R["L"][0])
    N = len(D["inputs"])
    nb = a.bins
    edges = np.linspace(0, 1, nb + 1)
    starts = np.searchsorted(ent["seg"], np.arange(len(R["E0"]) + 1))
    prof_mc = {}
    xq = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])               # depth bands of the lateral profiles
    yq = np.linspace(0.0, 1.0, 41)
    lat_mc = np.zeros((len(xq) - 1, len(yq) - 1))
    for name, m in (("entering", R["start"] < 0.5), ("interior", R["start"] >= 0.5)):
        rows = np.flatnonzero(m)
        kt, xt, yt, Et = truth_points(ent, rows, starts)
        prof_mc[name] = np.histogram(xt, edges, weights=Et)[0] / N
        lat_mc += np.histogram2d(xt, yt, (xq, yq), weights=Et)[0] / N
    spec, model, _ = sk.load_kernel(a.kernel, dev)
    gm = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=a.seed)
    K = MirrorY(gm, a.seed + 3)
    inp = D["inputs"]
    rows_all = {"entering": [], "interior": []}
    out = K.boundary(inp[:, 1] / L, inp[:, 2], inp[:, 3], D["w_in"], inp[:, 4], L=L)
    cur = ("entering", np.zeros(N), inp[:, 1] / L, inp[:, 2], inp[:, 3], inp[:, 4], out)
    for level in range(8):
        name, x0, y0, u0, v0, E0, o = cur
        n = len(E0)
        Dt = o.get("daughters")
        n_rel = (np.bincount(np.asarray(Dt["parent"], np.int64), minlength=n) if Dt is not None and len(Dt["E"])
                 else np.zeros(n, np.int64))
        flag = np.asarray(o["flag"]) if "flag" in o else (np.asarray(o.get("photon_E", np.zeros(n))) > 0)
        rows_all[name].append({"start": np.full(n, 0 if name == "entering" else 1), "x0": x0, "y0": y0, "u0": u0,
                               "v0": v0, "E0": E0, "cls": np.asarray(o["cls"]), "x1": o["x"], "y1": o["y"],
                               "u1": o["u"], "v1": o["v"], "E_out": np.where(np.asarray(o["cls"]) == 4, 0.0, o["E_out"]),
                               "dose": o["dose"], "n_rel": n_rel, "flag": flag.astype(np.int64),
                               "L": np.full(n, L)})
        if Dt is None or not len(Dt["E"]):
            break
        o2 = K.interior(Dt["x"], Dt["y"], Dt["u"], Dt["v"], Dt["w"], Dt["E"], L=L)
        cur = ("interior", Dt["x"], Dt["y"], Dt["u"], Dt["v"], Dt["E"], o2)
    prof_g = {}
    lat_g = np.zeros_like(lat_mc)
    for name, lst in rows_all.items():
        if not lst:
            prof_g[name] = np.zeros(nb)
            continue
        Rg = {k: np.concatenate([r[k] for r in lst]) for k in lst[0]}
        Rd = sk.to_device(Rg, dev)
        P = np.zeros(nb)
        with torch.no_grad():
            for i in range(0, len(Rg["E0"]), 2000):
                rr = torch.arange(i, min(i + 2000, len(Rg["E0"])), device=dev)
                z = model.sample(model.context(sk.cond_features(spec, sk.gather(Rd, rr))), a.n_samples,
                                 base="sobol").cpu().numpy()
                w = np.repeat(Rg["dose"][i:i + 2000] / a.n_samples, a.n_samples)
                P += np.histogram(z[..., 0].ravel(), edges, weights=w)[0]
                lat_g += np.histogram2d(z[..., 0].ravel(), z[..., 1].ravel(), (xq, yq), weights=w)[0] / N
        prof_g[name] = P / N
    tot_mc = prof_mc["entering"] + prof_mc["interior"]
    tot_g = prof_g["entering"] + prof_g["interior"]
    print(f"{a.data}: {N} source entries, L = {10 * L:.2f} mm; in-cell dose per primary MC {tot_mc.sum():.4f} MeV "
          f"(entering {prof_mc['entering'].sum():.4f}, interior {prof_mc['interior'].sum():.4f}), GMC "
          f"{tot_g.sum():.4f} (entering {prof_g['entering'].sum():.4f}, interior {prof_g['interior'].sum():.4f})")
    g = a.group
    cut = nb // g * g
    co = lambda p: p[:cut].reshape(-1, g).sum(1)  # noqa: E731
    xc = co(edges[:-1] + 0.5 / nb) / g
    print("  x/L             " + " ".join(f"{v:6.3f}" for v in xc))
    print("  GMC+SCR / MC    " + " ".join(f"{v:6.3f}" for v in co(tot_g) / co(tot_mc)))
    print("  entering only   " + " ".join(f"{v:6.3f}" for v in co(prof_g['entering']) / co(prof_mc['entering'])))
    print("  interior share MC  " + " ".join(f"{v:6.3f}" for v in co(prof_mc['interior']) / co(tot_mc)))
    print("  interior share GMC " + " ".join(f"{v:6.3f}" for v in co(prof_g['interior']) / co(tot_g)))
    yc = (yq[:-1] + yq[1:]) / 2 - 0.5
    print("lateral profiles (all transits): rms width y (cell units) MC vs GMC+SCR, and GMC/MC per |y - 0.5| band")
    bands = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    print("  depth x/L     rms MC  rms GMC  " + " ".join(f"|y|{lo:.1f}-{hi:.1f}" for lo, hi in bands))
    for j in range(len(xq) - 1):
        a_, b_ = lat_mc[j], lat_g[j]
        rm = np.sqrt((a_ * yc ** 2).sum() / a_.sum())
        rg = np.sqrt((b_ * yc ** 2).sum() / b_.sum())
        rat = [b_[(np.abs(yc) >= lo) & (np.abs(yc) < hi)].sum() / max(a_[(np.abs(yc) >= lo) & (np.abs(yc) < hi)].sum(),
                                                                        1e-300) for lo, hi in bands]
        print(f"  {xq[j]:.1f}-{xq[j + 1]:.1f}     {rm:7.4f} {rg:7.4f}   " + " ".join(f"{v:9.3f}" for v in rat))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mc")
    p.add_argument("--L", type=float, default=1.0, help="cell side (cm)")
    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--inner", type=int, default=10000)
    p.add_argument("--E_low", type=float, default=18.0)
    p.add_argument("--E_high", type=float, default=20.0)
    p.add_argument("--y_half", type=float, default=0.1, help="half width of the entry band around the face centre (cm)")
    p.add_argument("--sigma_deg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--entries", default=None, help="drumm_analog_chain mc output: replay its logged entries")
    p.add_argument("--cell", type=int, default=-1, help="cell id of --entries (default: the largest cell)")
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--kernel", nargs="+", required=True)
    e.add_argument("--bins", type=int, default=100)
    e.add_argument("--n_samples", type=int, default=1024)
    e.add_argument("--max_rows", type=int, default=40000)
    e.add_argument("--L_min", type=float, default=0.0, help="only transits in cells at least this large (cm)")
    g = sub.add_parser("gmc")
    g.add_argument("--data", required=True)
    g.add_argument("--kernel", default=str(ROOT / "ml/models/size_scr_water_100um-1cm_30MeV_v2"))
    g.add_argument("--model", default=str(ROOT / "ml/models/size_gmc_water_100um-1cm_30MeV_sx2/secondary_gmc.pth"))
    g.add_argument("--ode_steps", type=int, default=20)
    g.add_argument("--seed", type=int, default=5)
    g.add_argument("--bins", type=int, default=200)
    g.add_argument("--group", type=int, default=10)
    g.add_argument("--n_samples", type=int, default=256)
    a = ap.parse_args()
    {"mc": cmd_mc, "eval": cmd_eval, "gmc": cmd_gmc}[a.cmd](a)


if __name__ == "__main__":
    main()
