"""
Lockwood, Ruggles, Miller & Halbleib, "Calorimetric Measurement of Electron Energy Deposition in Extended Media -
Theory vs Experiment", SAND79-0414 (1980): measured depth-dose in semi-infinite Al and Cu, compared with the analog MC
(and the paper's TIGER calculations). Data transcribed in validation/results/lockwood_sandia/lockwood_tables.json
(units and conventions are documented there). PDF: papers/sandia_790414.pdf.

Cases: Al 1.0 / 0.5 / 0.3 MeV at 0 and 60 deg, Al 0.1 / 0.05 MeV surface calorimeter only, Cu 0.5 / 0.3 MeV at 0 deg.
Cu 1.0 MeV is missing from this scan (it can still be run, with nothing to compare to).

Geometry (our 2D code, infinite z): a slab of 1.25 CSDA ranges in x, beam entering at x = 0 along +x (60 deg: tilted
into z, u = cos 60, w = sin 60, so the lateral drift goes along the infinite axis), at mid-height of a 2.5-range-tall
box. Tally: n_bins depth bins x one voxel in y, i.e. integrated over y and z; by translation invariance that is the
broad-beam depth-dose per incident electron, MeV cm^2/g (the paper's J). Electrons are followed to 1 keV; photons are
banked at birth with their birth depth and origin (bremsstrahlung / fluorescence) histogrammed, so the plot can show
three photon conventions: escape (default, like the GMC), fluorescence deposited locally, all photons local. TIGER
transports photons; the truth is between the first two for the dose shape (Cu K x-rays travel ~0.02 g/cm^2,
bremsstrahlung mostly leaves).

Energies: Al was measured at 1.033 / 0.521 / 0.314 / 0.109 / 0.058 MeV and its FMR uses the CSDA range at those
energies, while TIGER ran at the nominal 1.0 / 0.5 / ... MeV. `mc --energy both` (default) runs each Al case at both,
and the plot compares each reference with the matching run. The FMR scale of the TIGER histograms is the range implied
by the paper's own numbers, R = total deposition / sum(J dFMR) (it reproduces the tabulated Cu ranges to 0.1 %),
cross-checked against the CSDA range of our stopping power.

Subcommands:
  mc    (ElectronTransport venv) runs the cases; one npz per case and energy, per-batch tallies for 1-sigma errors.
  plot  per-case figures (profile + ratio panel), an overview grid, metrics.json.
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

OUT = ROOT / "validation/results/lockwood_sandia"
TABLES = OUT / "lockwood_tables.json"
DEPTH_RANGES, HEIGHT_RANGES = 1.25, 2.5


def load_cases():
    return json.loads(TABLES.read_text())["cases"]


def run_name(key, E, N, seed, stopping="eedl"):
    tag = "" if stopping == "eedl" else f"_{stopping}"
    return f"mc_{key}_{E:g}MeV_{N // 1000}k_s{seed}{tag}.npz"


def tiger_range(c):
    """CSDA range (g/cm^2) behind the TIGER FMR axis: total / sum(J dFMR) over the upper-edge histogram."""
    t = np.array(c["theory"])
    return c["theory_total"] / float((t[:, 1] * np.diff(np.r_[0.0, t[:, 0]])).sum())


# ----------------------------------------------------------------------------------------------------------- mc
def csda_range_lib(lib, mat, E, E_lo=1e-3, n=400):
    """CSDA range (g/cm^2) from our tables' total stopping power, integrated from E_lo (log grid, trapezoid)."""
    Es = np.geomspace(E_lo, E, n)
    S = np.array([sum(lib.stopping_power(mat, e)) for e in Es])
    return float(np.trapezoid(1.0 / S, Es))


def cmd_mc(a):
    et = str(ROOT.parent / "ElectronTransport")
    sys.path.insert(0, et)
    os.chdir(et)
    from analog_stepper import VoxelGeometry, run_transport
    from material_data import build_library, named_material
    cases = load_cases()
    keys = a.cases or [k for k in cases]
    libs = {}
    for key in keys:
        c = cases[key]
        if c["material"] not in libs:
            m, h = named_material(c["material"])
            libs[c["material"]] = (m, build_library([m], [h], stopping=a.stopping))
        m, lib = libs[c["material"]]
        Es = {"exp": [c["E_exp"]], "nom": [c["E_nom"]], "both": sorted({c["E_exp"], c["E_nom"]})}[a.energy]
        for E in Es:
            rho = m.density
            R_lib = csda_range_lib(lib, 0, E)
            R = max(c["csda_range_exp"] * (E / c["E_exp"]) ** 1.7, R_lib)   # only sizes the box
            T, H = DEPTH_RANGES * R / rho, HEIGHT_RANGES * R / rho
            geo = VoxelGeometry(T, H, np.zeros((a.n_bins, 1), dtype=np.int64))
            th = np.radians(c["angle"])
            src = np.zeros((a.N, 8))
            src[:, 1] = 0.5 * H
            src[:, 2], src[:, 7] = np.cos(th), np.sin(th)
            src[:, 4] = E
            edges = np.linspace(0.0, T * rho, a.n_bins + 1)   # g/cm^2
            nb = a.n_batch
            energy = np.zeros((nb, a.n_bins))
            photon = np.zeros((nb, 2, a.n_bins))               # [brems, fluorescence] energy by birth depth
            bal = {k: 0.0 for k in ("E_in", "E_deposited", "E_escaped_electrons", "E_photons", "E_overflow")}
            back = {"E_front": 0.0, "E_side": 0.0, "n_primary_front": 0}
            t0 = time.time()
            for i, part in enumerate(np.array_split(src, nb)):
                r = run_transport(lib, geo, part, E_cutoff=a.E_cutoff, seed=a.seed + 1000 * i, n_tracks=0)
                energy[i] = r["energy_tally"][:, 0]
                ph = r["photon_bank"]
                for j, origin in enumerate((1.0, 2.0)):
                    s = ph[:, 7] == origin
                    photon[i, j] = np.histogram(ph[s, 0] * rho, bins=edges, weights=ph[s, 5])[0]
                for k in bal:
                    bal[k] += float(r["energy_balance"][k])
                eb = r["electron_bank"]
                front = eb[:, 0] <= 1e-9 * T
                back["E_front"] += float(eb[front, 5].sum())
                back["E_side"] += float(eb[~front, 5].sum())
                back["n_primary_front"] += int((front & (eb[:, 8] == 0)).sum())
            secs = time.time() - t0
            fr = {k: bal[k] / bal["E_in"] for k in bal if k != "E_in"}
            print(f"{key} @ {E:g} MeV: {secs:.0f} s ({1e3 * secs / a.N:.3f} ms/primary); "
                  f"deposited {fr['E_deposited']:.4f}, e- out front {back['E_front'] / bal['E_in']:.4f}, "
                  f"sides/back {back['E_side'] / bal['E_in']:.2e}, photons {fr['E_photons']:.4f}; "
                  f"eta_N {back['n_primary_front'] / a.N:.4f}; R_lib {R_lib:.4g} g/cm^2", flush=True)
            out = OUT / run_name(key, E, a.N, a.seed, a.stopping)
            np.savez_compressed(out, edges=edges, energy=energy, photon=photon, N=a.N, E=E,
                                meta=json.dumps({"case": key, "E": E, "angle": c["angle"], "N": a.N,
                                                 "seed": a.seed, "E_cutoff": a.E_cutoff, "seconds": secs,
                                                 "balance": bal, "backscatter": back, "csda_range_lib": R_lib,
                                                 "slab_cm": [T, H],
                                                 "physics": {"elastic": getattr(lib, "elastic", "eedl"),
                                                             "ionization_deflection":
                                                                 getattr(lib, "ionization_deflection", "binary"),
                                                             "stopping": getattr(lib, "stopping", "eedl")}}))
            print(f"  saved {out}")


# ----------------------------------------------------------------------------------------------------------- gmc
def gmc_name(key, E, N, n):
    return f"gmc_{key}_{E:g}MeV_{N // 1000}k_n{n}.npz"


def cmd_gmc(a):
    """Chained GMC (+ SCR sub-cell dose) on a slab of cells: the same problem as `mc`, at the cell size of the model.
    The grid has nx >= 1.4 ranges in depth and an odd number of rows (>= 2.5 ranges), and the beam enters the centre of
    the middle row's left face. Photons leave the problem, like the MC 'escape' convention."""
    import torch
    from chain_secondaries import chain_secondaries
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, scr_tally, auto_samples
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases = load_cases()
    keys = a.cases or [k for k in cases]
    models = {"Cu": (a.model_cu, a.kernel_cu), "Al": (a.model_al, a.kernel_al)}
    gen = torch.Generator(device=dev)
    for key in keys:
        c = cases[key]
        model, kpath = models[c["material"]]
        if model is None:
            print(f"{key}: no --model_{c['material'].lower()}, skipped")
            continue
        probe = SecondaryGMC(model, dev)
        L, rho = float(probe.spec.L), float(probe.spec.material_density)
        kern = {}
        if kpath:
            spec_k, km, _ = load_kernel(kpath, dev)
            kern = {0: (spec_k, km)}
        K = a.K or auto_samples(a.n)
        Es = {"exp": [c["E_exp"]], "nom": [c["E_nom"]], "both": sorted({c["E_exp"], c["E_nom"]})}[a.energy]
        for E in Es:
            R = 1.1 * c["csda_range_exp"] * (E / c["E_exp"]) ** 1.7 / rho       # cm, with margin for our longer range
            nx = max(3, int(np.ceil(1.4 * R / L)))
            ny = max(5, int(np.ceil(2.5 * R / L)))
            ny += 1 - ny % 2
            th = np.radians(c["angle"])
            src = np.zeros((a.N, 8))
            src[:, 1] = 0.5 * ny * L
            src[:, 2], src[:, 7] = np.cos(th), np.sin(th)
            src[:, 4] = E
            cell, fine, tc, tt, ntr, esc = [], [], [], [], [], []
            for s in range(a.n_seeds):
                seed = a.seed + 100 * s
                d_c, d_f = np.zeros(nx), np.zeros(nx * a.n)
                t_c = t_t = 0.0
                n_tr = 0
                e_s = {}
                for i, b in enumerate(range(0, a.N, a.batch)):
                    g = SecondaryGMC(model, dev, ode_steps=a.ode_steps, seed=seed + 7 * i)
                    t0 = time.time()
                    ch = chain_secondaries([g], src[b:b + a.batch], nx, ny, L, a.E_cutoff, log_every=0,
                                           transit_log=bool(kern))
                    t_c += time.time() - t0
                    d_c += ch["dose"].sum(1)
                    for k_, v_ in ch["escaped"].items():
                        e_s[k_] = e_s.get(k_, 0.0) + v_
                    if kern:
                        n_tr += len(ch["transits"]["E0"])
                        gen.manual_seed(seed + 5 + 13 * i)
                        t0 = time.time()
                        f, _ = scr_tally(ch, kern, nx, ny, a.n, n_samples=K, device=dev, gen=gen, sampling="sobol")
                        if dev == "cuda":
                            torch.cuda.synchronize()
                        t_t += time.time() - t0
                        d_f += f.sum(1)
                    del ch
                cell.append(d_c / a.N)
                fine.append(d_f / a.N)
                tc.append(t_c)
                tt.append(t_t)
                ntr.append(n_tr)
                esc.append({k_: v_ / (a.N * E) for k_, v_ in e_s.items()})
                print(f"{key} @ {E:g} MeV seed {seed}: grid {nx}x{ny}, chain {t_c:.0f} s ({1e3 * t_c / a.N:.3f} "
                      f"ms/primary, {n_tr / a.N:.1f} transits/primary), SCR {t_t:.0f} s (K {K}); deposited "
                      f"{d_c.sum() / (a.N * E):.4f} (SCR {d_f.sum() / (a.N * E):.4f}), e- out front "
                      f"{esc[-1].get('left', 0):.4f}", flush=True)
            out = OUT / gmc_name(key, E, a.N, a.n)
            np.savez_compressed(out, cell=np.array(cell), scr=np.array(fine), L=L, rho=rho, n=a.n, N=a.N, E=E,
                                meta=json.dumps({**vars(a), "case": key, "E": E, "grid": [nx, ny], "K": K,
                                                 "chain_s": tc, "tally_s": tt, "transits": ntr, "escaped": esc}))
            print(f"  saved {out}")


# ----------------------------------------------------------------------------------------------------------- plot
class Profile:
    """Depth tally: energy per primary per bin (MeV) for several independent batches / seeds on bin edges in g/cm^2.
    window() averages over arbitrary [lo, hi] windows (piecewise-constant within a bin) with a batch 1-sigma error."""

    def __init__(self, edges, batches):
        self.edges, self.b = np.asarray(edges, float), np.atleast_2d(batches)

    def window(self, lo, hi):
        cum = np.concatenate([np.zeros((self.b.shape[0], 1)), np.cumsum(self.b, 1)], 1)
        lo, hi = np.clip(lo, self.edges[0], self.edges[-1]), np.clip(hi, self.edges[0], self.edges[-1])
        v = np.array([(np.interp(hi, self.edges, cb) - np.interp(lo, self.edges, cb)) / (hi - lo) for cb in cum])
        e = v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros(v.shape[1])
        return v.mean(0), e

    def total(self):
        t = self.b.sum(1)
        return float(t.mean()), float(t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else 0.0


class Run:
    """One mc output: per-primary depth tallies with batch-to-batch 1-sigma errors, three photon conventions."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.edges = z["edges"]
        nb = z["energy"].shape[0]
        per = self.meta["N"] / nb
        self.p = {"escape": Profile(self.edges, z["energy"] / per),
                  "fluor_local": Profile(self.edges, (z["energy"] + z["photon"][:, 1]) / per),
                  "all_local": Profile(self.edges, (z["energy"] + z["photon"].sum(1)) / per)}
        self.E = float(z["E"])

    def window(self, conv, lo, hi):
        """Mean J (MeV cm^2/g per electron) over [lo, hi] g/cm^2 and its 1-sigma error; arrays lo, hi."""
        return self.p[conv].window(lo, hi)

    def total(self, conv):
        return self.p[conv].total()


class GmcRun:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        L, rho, n = float(z["L"]), float(z["rho"]), int(z["n"])
        nx = z["cell"].shape[1]
        self.cell = Profile(np.arange(nx + 1) * L * rho, z["cell"])
        self.scr = Profile(np.arange(nx * n + 1) * L * rho / n, z["scr"]) if z["scr"].size else None
        self.cell_gcm2 = L * rho
        self.ms = 1e3 * (np.mean(self.meta["chain_s"]) + np.mean(self.meta["tally_s"])) / self.meta["N"]
        self.ms_chain = 1e3 * np.mean(self.meta["chain_s"]) / self.meta["N"]


def find_run(key, E, N, seed, stopping="eedl"):
    p = OUT / run_name(key, E, N, seed, stopping)
    return Run(p) if p.exists() else None


def find_gmc(key, E, N, n):
    p = OUT / gmc_name(key, E, N, n)
    return GmcRun(p) if p.exists() else None


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def cmd_plot(a):
    import sec_plot_style as sty
    sty.apply()
    plt = sty.plt
    cases = load_cases()
    metrics, panels = {}, []
    C_MC, C_T, C_G, C_S = sty.REF, "#eb6834", "#2a78d6", "#1baf7a"
    C_B = "#8a8a8a"  # the EEDL-stopping MC, overlaid when plotting another stopping model
    # --stopping estar: the MC is the ESTAR-scaled run; figures go to a subfolder and the EEDL-stopping run is overlaid.
    # The GMC models were trained on EEDL-stopping data, so they are left out of that comparison.
    alt = a.stopping != "eedl"
    fig_dir = OUT / a.stopping if alt else OUT
    fig_dir.mkdir(exist_ok=True)
    for key, c in cases.items():
        if c.get("missing"):
            continue
        r_exp = find_run(key, c["E_exp"], a.N, a.seed, a.stopping)
        r_nom = find_run(key, c["E_nom"], a.N, a.seed, a.stopping)
        g = None if alt else find_gmc(key, c["E_exp"], a.gmc_N, a.gmc_n)
        base = find_run(key, c["E_exp"], a.N, a.seed) if alt else None
        if r_exp is None:
            print(f"{key}: no run at {c['E_exp']} MeV, skipped")
            continue
        E_in = r_exp.meta["balance"]["E_in"]
        m = {"E_exp": c["E_exp"], "E_nom": c["E_nom"], "angle": c["angle"], "material": c["material"],
             "csda_range_paper_exp": c["csda_range_exp"],
             "csda_range_lib_exp": r_exp.meta["csda_range_lib"],
             "backscatter_energy_fraction": r_exp.meta["backscatter"]["E_front"] / E_in,
             "backscatter_number_primaries": r_exp.meta["backscatter"]["n_primary_front"] / r_exp.meta["N"],
             "mc_ms_per_primary": 1e3 * r_exp.meta["seconds"] / r_exp.meta["N"]}
        if g is not None:
            m["gmc_ms_per_primary_chain"] = g.ms_chain
            m["gmc_ms_per_primary_with_scr"] = g.ms
            m["gmc_backscatter_energy_fraction"] = float(np.mean([e.get("left", 0) for e in g.meta["escaped"]]))
            m["gmc_total"] = g.cell.total()
            if g.scr is not None:
                m["scr_total"] = g.scr.total()
        t = c["calorimeter"]
        if c.get("surface_only"):
            j_exp = c["exp"][0][1]
            for conv in ("escape", "fluor_local", "all_local"):
                v, e = r_exp.window(conv, np.array([0.0]), np.array([t]))
                m[f"surface_J_{conv}"] = [float(v[0]), float(e[0])]
            m["surface_J_exp"], m["MC_over_exp"] = j_exp, m["surface_J_escape"][0] / j_exp
            if base is not None:
                m["MC_eedl_over_exp"] = float(base.window("escape", np.array([0.0]), np.array([t]))[0][0]) / j_exp
            if r_nom is not None:
                v, e = r_nom.window("escape", np.array([0.0]), np.array([t]))
                m["surface_J_nominal"] = [float(v[0]), float(e[0])]
                m["MC_over_TIGER"] = float(v[0]) / c["theory_surface"]
            m["TIGER_over_exp"] = c["theory_surface"] / j_exp
            if g is not None and g.scr is not None:
                v, e = g.scr.window(np.array([0.0]), np.array([t]))
                m["surface_J_scr"] = [float(v[0]), float(e[0])]
                m["SCR_over_exp"], m["SCR_over_MC"] = float(v[0]) / j_exp, float(v[0]) / m["surface_J_escape"][0]
            metrics[key] = m
            continue

        R_e = c["csda_range_exp"]
        x = np.array(c["exp"])
        fmr, J = x[:, 0], x[:, 1]
        d = fmr * R_e
        lo, hi = np.maximum(d - t / 2, 0), d + t / 2
        th = np.array(c["theory"])
        th_hi = th[:, 0]
        th_lo = np.r_[0.0, th_hi[:-1]]
        R_t = tiger_range(c)
        m["csda_range_tiger_inferred"] = R_t
        rt = r_nom if r_nom is not None else r_exp
        R_mc_t = R_t if r_nom is not None else R_e         # universal-curve fallback without a nominal run
        m["tiger_compared_at"] = "nominal" if r_nom is not None else "exp energy (universal curve)"
        if r_nom is not None:
            m["csda_range_lib_nom"] = r_nom.meta["csda_range_lib"]

        mask = J > a.min_frac * J.max()
        rows = {}
        for conv in ("escape", "fluor_local", "all_local"):
            v, e = r_exp.window(conv, lo, hi)
            rows[conv] = (v, e)
            m[f"total_{conv}"] = r_exp.total(conv)
            m[f"exp_rms_rel_{conv}"] = rms(v[mask] / J[mask] - 1)
            m[f"exp_mean_rel_{conv}"] = float(np.mean(v[mask] / J[mask] - 1))
        if base is not None:
            vb, eb_ = base.window("escape", lo, hi)
            m["eedl_exp_rms_rel_escape"] = rms(vb[mask] / J[mask] - 1)
            m["eedl_exp_mean_rel_escape"] = float(np.mean(vb[mask] / J[mask] - 1))
            m["eedl_total_escape"] = base.total("escape")
            m["csda_range_lib_exp_eedl"] = base.meta["csda_range_lib"]
            m["backscatter_energy_fraction_eedl"] = base.meta["backscatter"]["E_front"] / base.meta["balance"]["E_in"]
        v_t, _ = rt.window("escape", th_lo * R_mc_t, th_hi * R_mc_t)
        tmask = th[:, 1] > a.min_frac * th[:, 1].max()
        m["tiger_rms_rel_escape"] = rms(v_t[tmask] / th[tmask, 1] - 1)
        m["tiger_mean_rel_escape"] = float(np.mean(v_t[tmask] / th[tmask, 1] - 1))
        # TIGER itself vs the measurement (TIGER histogram averaged over each calorimeter window, universal curve)
        cum_t = np.r_[0.0, np.cumsum(th[:, 1] * (th_hi - th_lo))]
        f_lo, f_hi = lo / R_e, hi / R_e
        tig_at_exp = (np.interp(f_hi, np.r_[0.0, th_hi], cum_t) - np.interp(f_lo, np.r_[0.0, th_hi], cum_t)) / (f_hi - f_lo)
        m["tiger_vs_exp_rms_rel"] = rms(tig_at_exp[mask] / J[mask] - 1)
        m["tiger_vs_exp_mean_rel"] = float(np.mean(tig_at_exp[mask] / J[mask] - 1))
        m["total_exp"], m["total_tiger"] = c["exp_total"], c["theory_total"]
        if r_nom is not None:
            m["total_escape_nominal"] = r_nom.total("escape")

        # like-for-like grids against the analog MC (same energy, photons escape)
        fine = np.linspace(0, min(1.05, r_exp.edges[-1] / R_e), 211)
        fl, fh = fine[:-1] * R_e, fine[1:] * R_e
        xm = 0.5 * (fine[1:] + fine[:-1])
        v_mc, e_mc = r_exp.window("escape", fl, fh)
        cmask = v_mc > a.min_frac * v_mc.max()
        s_rows = None
        if g is not None:
            ce = g.cell.edges
            ce = ce[ce <= r_exp.edges[-1] + 1e-12]
            v_mc_c, _ = r_exp.window("escape", ce[:-1], ce[1:])
            v_g_c, e_g_c = g.cell.window(ce[:-1], ce[1:])
            ccm = v_mc_c > a.min_frac * v_mc_c.max()
            m["gmc_cell_vs_mc_rms_rel"] = rms(v_g_c[ccm] / v_mc_c[ccm] - 1)
            m["gmc_cell_vs_mc_mean_rel"] = float(np.mean(v_g_c[ccm] / v_mc_c[ccm] - 1))
            m["cells_in_range"] = float(R_e / g.cell_gcm2)
            if g.scr is not None:
                # compare on the SCR's own sub-bins (merged to >= 0.005 FMR): finer windows only resolve its
                # piecewise-constant bins (a sawtooth, not a model error)
                se = g.scr.edges
                f_ = max(1, int(np.ceil(0.005 * R_e / (se[1] - se[0]))))
                se = se[::f_]
                se = se[se <= r_exp.edges[-1] + 1e-12]
                v_s, e_s = g.scr.window(se[:-1], se[1:])
                v_ms, e_ms = r_exp.window("escape", se[:-1], se[1:])
                smask = v_ms > a.min_frac * v_ms.max()
                m["scr_bin_fmr"] = float((se[1] - se[0]) / R_e)
                m["scr_vs_mc_rms_rel"] = rms(v_s[smask] / v_ms[smask] - 1)
                m["scr_vs_mc_mean_rel"] = float(np.mean(v_s[smask] / v_ms[smask] - 1))
                m["scr_vs_mc_surface_bin_rel"] = float(v_s[0] / v_ms[0] - 1)
                m["mc_noise_rms_rel"] = rms(e_ms[smask] / v_ms[smask])
                m["scr_noise_rms_rel"] = rms(e_s[smask] / v_s[smask])
                s_rows = g.scr.window(lo, hi)
                m["scr_exp_rms_rel"] = rms(s_rows[0][mask] / J[mask] - 1)
                m["scr_exp_mean_rel"] = float(np.mean(s_rows[0][mask] / J[mask] - 1))
        metrics[key] = m

        # ---- figure: profile, ratio to experiment, ratio to analog MC
        fig, (ax, ar, am) = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True,
                                         gridspec_kw=dict(height_ratios=[3, 1.25, 1.25]))
        mc_label = "Analog MC" + (f" ({a.stopping.upper()} stopping)" if alt else "")
        ax.plot(xm, v_mc, color=C_MC, lw=1.8, label=mc_label)
        if base is not None:
            v_b, _ = base.window("escape", fl, fh)
            ax.plot(xm, v_b, color=C_B, lw=1.2, ls="--", label="Analog MC (EEDL stopping)")
        ax.fill_between(xm, v_mc - e_mc, v_mc + e_mc, color=C_MC, alpha=0.18, lw=0)
        ax.stairs(th[:, 1], np.r_[0.0, th_hi], color=C_T, lw=1.4, label=f"TIGER ({c['E_nom']:g} MeV)")
        if g is not None:
            ax.stairs(v_g_c, ce / R_e, color=C_G, lw=1.3, ls="--", label="GMC (100 um cells)")
            if g.scr is not None:
                ax.stairs(v_s, se / R_e, color=C_S, lw=1.5, ls="-.", label=f"GMC + SCR ({m['scr_bin_fmr']:.3f} FMR bins)")
        ax.errorbar(fmr, J, yerr=J * c["exp_uncertainty"], fmt="o", ms=6, mfc="none", mec=sty.INK, ecolor=sty.INK,
                    lw=1, label=f"Experiment ({c['E_exp']:g} MeV)")
        ax.plot(fmr, rows["escape"][0], "D", ms=3.5, color=C_MC, mec=C_MC,
                label=f"MC over the calorimeter ({t / R_e:.3f} FMR)")
        ax.set_ylabel("J (MeV cm$^2$/g per electron)")
        ax.set_title(f"{c['material']}, {c['E_exp']:g} MeV, {c['angle']}$^\\circ$: depth-dose vs Lockwood et al.")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.set_ylim(bottom=0)
        dx = 0.006
        ar.axhspan(1 - c["exp_uncertainty"], 1 + c["exp_uncertainty"], color=sty.GRID, alpha=0.8, lw=0)
        ar.axhline(1, color=sty.INK_2, lw=0.8)
        v, e = rows["escape"]
        ar.errorbar(fmr[mask] - dx, v[mask] / J[mask], yerr=e[mask] / J[mask], fmt="D", ms=4, color=C_MC, mec=C_MC,
                    lw=1, label="MC / exp")
        ar.plot(fmr[mask] + dx, tig_at_exp[mask] / J[mask], "x", color=C_T, mec=C_T, ms=6, mew=1.6,
                label="TIGER / exp")
        if base is not None:
            ar.errorbar(fmr[mask] + 2 * dx, vb[mask] / J[mask], yerr=eb_[mask] / J[mask], fmt="s", ms=3.5,
                        mfc="none", color=C_B, mec=C_B, lw=1, label="MC (EEDL) / exp")
        if s_rows is not None:
            ar.errorbar(fmr[mask], s_rows[0][mask] / J[mask], yerr=s_rows[1][mask] / J[mask], fmt="^", ms=4.5,
                        color=C_S, mec=C_S, lw=1, label="GMC+SCR / exp")
        ar.set_ylim(0.8, 1.2)
        ar.set_ylabel("/ experiment")
        ar.legend(fontsize=7, loc="lower left", ncol=3)
        am.axhline(1, color=sty.INK_2, lw=0.8)
        am.fill_between(xm, np.where(cmask, 1 - e_mc / v_mc, np.nan), np.where(cmask, 1 + e_mc / v_mc, np.nan),
                        color=C_MC, alpha=0.18, lw=0, label="MC 1$\\sigma$")
        am.stairs(np.where(tmask, th[:, 1] / v_t, np.nan), np.r_[0.0, th_hi], color=C_T, lw=1.2,
                  label="TIGER / MC")
        if base is not None:
            am.plot(xm, np.where(cmask, v_b / v_mc, np.nan), color=C_B, lw=1.2, ls="--", label="MC (EEDL) / MC")
        if g is not None:
            am.stairs(np.where(ccm, v_g_c / v_mc_c, np.nan), ce / R_e, color=C_G, lw=1.3, ls="--",
                      label="GMC cells / MC")
            if g.scr is not None:
                am.stairs(np.where(smask, v_s / v_ms, np.nan), se / R_e, color=C_S, lw=1.3, label="GMC+SCR / MC")
        am.set_ylim(0.8, 1.2)
        am.set_ylabel("/ analog MC")
        am.set_xlabel("depth / CSDA range (FMR)")
        am.legend(fontsize=7, loc="lower left", ncol=4)
        am.set_xlim(0, 1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{key}.png", dpi=130)
        plt.close(fig)
        panels.append((key, c, r_exp, g, fmr, J, th, th_hi))

    # ---- overview grid
    if panels:
        n = len(panels)
        nc = 4
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.8 * nr), squeeze=False)
        for axx, (key, c, r_exp, g, fmr, J, th, th_hi) in zip(axs.flat, panels):
            R_e = c["csda_range_exp"]
            fine = np.linspace(0, min(1.05, r_exp.edges[-1] / R_e), 106)
            xm = 0.5 * (fine[1:] + fine[:-1])
            v, _ = r_exp.window("escape", fine[:-1] * R_e, fine[1:] * R_e)
            axx.plot(xm, v, color=C_MC, lw=1.5)
            b = find_run(key, c["E_exp"], a.N, a.seed) if alt else None
            if b is not None:
                axx.plot(xm, b.window("escape", fine[:-1] * R_e, fine[1:] * R_e)[0], color=C_B, lw=1.0, ls="--")
            axx.stairs(th[:, 1], np.r_[0.0, th_hi], color=C_T, lw=1.0)
            if g is not None and g.scr is not None:
                se = g.scr.edges
                se = se[::max(1, int(np.ceil(0.005 * R_e / (se[1] - se[0]))))]
                se = se[se <= r_exp.edges[-1] + 1e-12]
                vs, _ = g.scr.window(se[:-1], se[1:])
                axx.stairs(vs, se / R_e, color=C_S, lw=1.3, ls="-.")
            axx.plot(fmr, J, "o", ms=3.5, mfc="none", mec=sty.INK)
            axx.set_title(f"{c['material']} {c['E_exp']:g} MeV {c['angle']}$^\\circ$", fontsize=9)
            axx.set_xlim(0, 1)
            axx.set_ylim(bottom=0)
        for axx in list(axs.flat)[n:]:
            axx.axis("off")
        fig.suptitle("Lockwood et al. (SAND79-0414): experiment (o), TIGER (orange), " +
                     (f"analog MC {a.stopping.upper()} stopping (black), EEDL stopping (grey dashed)" if alt else
                      "analog MC (black), GMC + SCR (green dash-dot)"), fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "overview.png", dpi=120)
        plt.close(fig)
    (fig_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    for key, m in metrics.items():
        if "MC_over_exp" in m:
            s = f", SCR/exp {m['SCR_over_exp']:.3f}, SCR/MC {m['SCR_over_MC']:.3f}" if "SCR_over_exp" in m else ""
            if "MC_eedl_over_exp" in m:
                s += f", EEDL-stopping MC/exp {m['MC_eedl_over_exp']:.3f}"
            print(f"{key:10s} surface J: MC/exp {m['MC_over_exp']:.3f}, TIGER/exp {m['TIGER_over_exp']:.3f}{s}")
            continue
        te = m["total_escape"][0]
        print(f"{key:10s} total MC {te:.3f} exp {m['total_exp']:.3f} TIGER {m['total_tiger']:.3f} | vs exp rms (mean): "
              f"MC {100 * m['exp_rms_rel_escape']:.1f}% ({100 * m['exp_mean_rel_escape']:+.1f}), TIGER "
              f"{100 * m['tiger_vs_exp_rms_rel']:.1f}% ({100 * m['tiger_vs_exp_mean_rel']:+.1f}) | MC/TIGER rms "
              f"{100 * m['tiger_rms_rel_escape']:.1f}% | R_lib/R_paper {m['csda_range_lib_exp'] / m['csda_range_paper_exp']:.3f}")
        if "eedl_exp_rms_rel_escape" in m:
            print(f"{'':10s} EEDL-stopping MC: total {m['eedl_total_escape'][0]:.3f}, vs exp rms "
                  f"{100 * m['eedl_exp_rms_rel_escape']:.1f}% ({100 * m['eedl_exp_mean_rel_escape']:+.1f}), "
                  f"R_lib/R_paper {m['csda_range_lib_exp_eedl'] / m['csda_range_paper_exp']:.3f}; backscatter E "
                  f"{m['backscatter_energy_fraction']:.3f} (EEDL {m['backscatter_energy_fraction_eedl']:.3f})")
        if "scr_vs_mc_rms_rel" in m:
            print(f"{'':10s} GMC: total {m['gmc_total'][0]:.3f} (SCR {m['scr_total'][0]:.3f}); cells/MC rms "
                  f"{100 * m['gmc_cell_vs_mc_rms_rel']:.1f}% ({m['cells_in_range']:.1f} cells/range); SCR/MC rms "
                  f"{100 * m['scr_vs_mc_rms_rel']:.1f}% on {m['scr_bin_fmr']:.4f}-FMR bins (mean {100 * m['scr_vs_mc_mean_rel']:+.1f}%, surface bin {100 * m['scr_vs_mc_surface_bin_rel']:+.1f}%, MC noise "
                  f"{100 * m['mc_noise_rms_rel']:.1f}%, SCR noise {100 * m['scr_noise_rms_rel']:.1f}%); SCR/exp rms "
                  f"{100 * m['scr_exp_rms_rel']:.1f}%; {m['gmc_ms_per_primary_with_scr']:.3f} vs MC "
                  f"{m['mc_ms_per_primary']:.3f} ms/primary")
    print(f"saved figures + metrics.json in {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("mc")
    p.add_argument("--cases", nargs="*", default=None, help="keys from lockwood_tables.json (default all)")
    p.add_argument("--energy", choices=("exp", "nom", "both"), default="both")
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--n_batch", type=int, default=10, help="batches (separate seeds) for the 1-sigma errors")
    p.add_argument("--n_bins", type=int, default=1000, help="depth bins over 1.25 CSDA ranges")
    p.add_argument("--seed", type=int, default=79)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p.add_argument("--stopping", choices=("eedl", "estar"), default="eedl",
                   help="material_data.build_library stopping model (estar: inelastic rescaled to ICRU 37)")
    p = sp.add_parser("gmc")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--energy", choices=("exp", "nom", "both"), default="exp")
    p.add_argument("--model_cu", default=None)
    p.add_argument("--model_al", default=None)
    p.add_argument("--kernel_cu", default=None, help="SCR sub-cell kernel (omit: cell dose only)")
    p.add_argument("--kernel_al", default=None)
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--n", type=int, default=20, help="SCR sub-bins per cell per axis")
    p.add_argument("--K", type=int, default=None, help="kernel samples per transit (default auto_samples(n))")
    p.add_argument("--batch", type=int, default=100000)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--ode_steps", type=int, default=20)
    p.add_argument("--E_cutoff", type=float, default=1e-3)
    p = sp.add_parser("plot")
    p.add_argument("--N", type=int, default=200000)
    p.add_argument("--seed", type=int, default=79)
    p.add_argument("--gmc_N", type=int, default=200000)
    p.add_argument("--gmc_n", type=int, default=20)
    p.add_argument("--min_frac", type=float, default=0.1, help="ratio metrics use points with J > this x max")
    p.add_argument("--stopping", choices=("eedl", "estar"), default="eedl",
                   help="which MC runs to plot; non-default: figures in a subfolder, EEDL-stopping MC overlaid")
    a = ap.parse_args()
    {"mc": cmd_mc, "gmc": cmd_gmc, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
