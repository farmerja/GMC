#!/usr/bin/env python3
"""
Accuracy / cost sweep of the GMC and the SCR kernel on the Drumm water-cylinder problem (20 MeV triangular beam, 1 mm
staircase, photons escape), scored against the analog MC staircase references like for like.

Modes:
  run      one GMC chain configuration (model, --ode_steps, --mirror), --n_seeds independent runs of --N primaries
           (each seed has its own source sample, --src_seed + seed index). Several SCR tallies (kernel, K) are applied
           to the SAME chains, so SCR-side variables cost only their tally time. Saves cell dose, cell charge, one fine
           map per tally per seed, and the chain / tally wall times.
  analyze  error split of every run and tally vs the mean of the MC references: per-run noise (from the spread of
           the seeds), bias (quadrature remainder) and time per primary; error-vs-time curves and speed-up vs the
           analog MC at fixed target errors. Writes figures + sweep_metrics.json.

Error metric: relative L2 over voxels above 1 % of the max (as in the results page), on the 1 mm cell map (GMC alone)
and on the 125 um map (GMC + SCR). For one run of N primaries the expected squared error is b^2 + s^2 (N0 / N), so
the time to reach an error eps is t(eps) = s^2 N0 t_p / (eps^2 - b^2) (infinite if eps <= b); the analog MC has b = 0.

Usage (GMC venv, GMC root):
  python validation/drumm_sweep.py run --model ml/models/<gmc>/secondary_gmc.pth --mirror --label x2 \
      --tally x2=ml/models/subcell_kernel_..._x2:32 --tally k16=ml/models/subcell_kernel_..._x2:16 --out runs/x2.npz
  python validation/drumm_sweep.py analyze --runs runs/*.npz --mc mc_stair_*.npz --out_dir results/.../sweep
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drumm_cylinder import (E0, L, NC, MirrorY, VacuumKernel, chain_charge, stair_map,  # noqa: E402
                            triangle_source)

N_FINE = 8


# ------------------------------------------------------------------------------------------------------------ run
def cmd_run(a):
    import torch
    from chain_secondaries import chain_secondaries
    from ml.secondary_gmc import SecondaryGMC
    from ml.subcell_kernel import load_kernel, scr_tally
    dev = "cuda"
    mmap = stair_map()
    tallies, cache = [], {}
    for t in a.tally:
        lab, rest = t.split("=", 1)
        path, K = rest.rsplit(":", 1)
        if path not in cache:
            spec, model, _ = load_kernel(path, dev)
            cache[path] = {0: (spec, model)}
        tallies.append((lab, path, int(K)))
    gen = torch.Generator(device=dev)
    out = {"dose": [], "charge": [], "chain_s": [], "transits": []}
    for lab, _, _ in tallies:
        out["fine_" + lab], out["tally_s_" + lab] = [], []
    # warm-up (CUDA context, kernels, numba) outside the timed region
    w = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=0)
    chain_secondaries([w, VacuumKernel()], triangle_source(2000, 1), NC, NC, L, a.E_cutoff, log_every=0,
                      material_map=mmap, transit_log=True)
    for s in range(a.n_seeds):
        seed = a.seed + 100 * s
        src = triangle_source(a.N, a.src_seed + s)
        dose, q = np.zeros((NC, NC)), np.zeros((NC, NC))
        fine = {lab: np.zeros((NC * N_FINE, NC * N_FINE)) for lab, _, _ in tallies}
        tt = {lab: 0.0 for lab, _, _ in tallies}
        tc, ntr = 0.0, 0
        for i, b in enumerate(range(0, a.N, a.batch)):
            K = SecondaryGMC(a.model, dev, ode_steps=a.ode_steps, seed=seed + 7 * i)
            if a.mirror:
                K = MirrorY(K, seed + 7 * i + 3)
            torch.cuda.synchronize()
            t0 = time.time()
            c = chain_secondaries([K, VacuumKernel()], src[b:b + a.batch], NC, NC, L, a.E_cutoff, log_every=0,
                                  material_map=mmap, transit_log=True)
            torch.cuda.synchronize()
            tc += time.time() - t0
            dose += c["dose"]
            q += chain_charge(c)
            ntr += int((c["transits"]["mat"] == 0).sum())
            for lab, path, Ks in tallies:
                gen.manual_seed(seed + 5 + 13 * i)
                t0 = time.time()
                f, _ = scr_tally(c, cache[path], NC, NC, N_FINE, n_samples=Ks, device=dev, gen=gen, sampling="sobol")
                torch.cuda.synchronize()
                tt[lab] += time.time() - t0
                fine[lab] += f
            del c
        out["dose"].append(dose)
        out["charge"].append(q)
        out["chain_s"].append(tc)
        out["transits"].append(ntr)
        for lab in fine:
            out["fine_" + lab].append(fine[lab])
            out["tally_s_" + lab].append(tt[lab])
        print(f"[{a.label}] seed {seed}: chain {tc:.0f} s ({1e3 * tc / a.N:.3f} ms/primary, {ntr / a.N:.1f} water "
              f"transits/primary); tallies " + ", ".join(f"{k} {v:.0f} s" for k, v in tt.items()) +
              f"; deposited {dose.sum() / (a.N * E0):.4f} of E0", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, N=a.N,
                        meta=json.dumps({**vars(a), "tallies": [list(t) for t in tallies]}))
    print(f"saved {a.out}")


# -------------------------------------------------------------------------------------------------------- analyze
def coarse(d):
    return d.reshape(NC, d.shape[0] // NC, NC, d.shape[1] // NC).sum((1, 3))


def split(G, M, mask, src_groups=None):
    """Error split of a set of runs G (n_g x map, per primary) vs a set of reference runs M (n_m x map).
    Returns per-run relative noise s_g (one run of the stored N), s_m, the bias b, and the total error of the means.
    src_groups: reference runs sharing a source sample (same id) are correlated; their pair differences are used only
    to estimate the non-source part, and the variance of the reference mean accounts for the shared part."""
    nrm = np.linalg.norm(M.mean(0)[mask])

    def pair_var(X, pairs):
        return np.mean([np.sum((X[i] - X[j])[mask] ** 2) / 2 for i, j in pairs]) / nrm ** 2

    ng, nm = len(G), len(M)
    sg2 = pair_var(G, [(i, j) for i in range(ng) for j in range(i + 1, ng)])
    grp = src_groups if src_groups is not None else list(range(nm))
    indep = [(i, j) for i in range(nm) for j in range(i + 1, nm) if grp[i] != grp[j]]
    same = [(i, j) for i in range(nm) for j in range(i + 1, nm) if grp[i] == grp[j]]
    if not indep:                                          # all references share one source: noise understated
        print("warning: no reference pair with independent sources; MC noise excludes source sampling")
        indep, same = same, []
    sm2 = pair_var(M, indep)
    cov = sm2 - pair_var(M, same) if same else 0.0          # source-noise part shared by same-source runs
    n_same = sum(1 for i in range(nm) for j in range(nm) if i != j and grp[i] == grp[j])
    var_mref = (nm * sm2 + n_same * cov) / nm ** 2
    e2 = np.sum((G.mean(0) - M.mean(0))[mask] ** 2) / nrm ** 2
    b2 = e2 - sg2 / ng - var_mref
    b2_seeds = [float(np.sum((g - M.mean(0))[mask] ** 2) / nrm ** 2 - sg2 - var_mref) for g in G]
    return {"b2_seeds": b2_seeds, "s_run": float(np.sqrt(sg2)), "s_mc_run": float(np.sqrt(sm2)), "s_src_frac": float(cov / sm2) if sm2 else 0,
            "err_means": float(np.sqrt(e2)), "bias": float(np.sqrt(max(b2, 0.0))), "bias2": float(b2),
            "bias_floor": float(np.sqrt(sg2 / ng + var_mref))}


def t_to(eps, b, s, tp_run):
    """Wall time to reach rms relative error eps with bias b and per-run noise s for a run costing tp_run seconds."""
    eps = np.asarray(eps, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(eps > b, s ** 2 * tp_run / (eps ** 2 - b ** 2), np.inf)


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import sec_plot_style as S
    S.apply()
    plt = S.plt
    os.makedirs(a.out_dir, exist_ok=True)
    mcs = [np.load(p, allow_pickle=True) for p in a.mc]
    Nm = [float(m["N"]) for m in mcs]
    grp = [json.loads(str(m["meta"])).get("src_seed", 2007) for m in mcs]
    Mf = np.array([m["energy"] / n for m, n in zip(mcs, Nm)])
    Mc = np.array([coarse(x) for x in Mf])
    Mq = np.array([coarse(m["charge"]) / n for m, n in zip(mcs, Nm)])
    mf = Mf.mean(0) > 0.01 * Mf.mean(0).max()
    mc_ = Mc.mean(0) > 0.01 * Mc.mean(0).max()
    mq = np.abs(Mq.mean(0)) > 0.02 * np.abs(Mq.mean(0)).max()
    mc_time = a.mc_ms * 1e-3 if a.mc_ms else np.mean([json.loads(str(m["meta"]))["seconds"] / n
                                                      for m, n in zip(mcs, Nm)])
    res = {"mc": {"files": a.mc, "s_per_primary": mc_time, "N_run": Nm[0]}}
    rows = []
    for p in sorted(sum((glob.glob(x) for x in a.runs), [])):
        g = np.load(p, allow_pickle=True)
        meta = json.loads(str(g["meta"]))
        N = float(g["N"])
        chain_tp = np.mean(g["chain_s"]) / N
        base = {"run": meta["label"], "file": os.path.basename(p), "N": N, "seeds": int(len(g["dose"])),
                "ode_steps": meta["ode_steps"], "mirror": meta["mirror"], "model": meta["model"],
                "chain_ms": 1e3 * chain_tp, "transits": float(np.mean(g["transits"]) / N)}
        cell = split(np.array([coarse(d) if d.shape[0] != NC else d for d in g["dose"]]) / N, Mc, mc_, grp)
        chg = split(g["charge"] / N, Mq, mq, grp)
        rows.append({**base, "tally": None, "kind": "cell", **cell, "t_ms": 1e3 * chain_tp})
        rows.append({**base, "tally": None, "kind": "charge", **chg, "t_ms": 1e3 * chain_tp})
        for lab, path, K in meta["tallies"]:
            tp = chain_tp + np.mean(g["tally_s_" + lab]) / N
            fe = split(g["fine_" + lab] / N, Mf, mf, grp)
            rows.append({**base, "tally": lab, "kernel": path, "K": K, "kind": "fine", **fe, "t_ms": 1e3 * tp,
                         "tally_ms": 1e3 * (tp - chain_tp)})
    # analog MC rows: the reference set itself (bias 0 by construction), noise from the seed spread
    mc_rows = {}
    for kind, X, m in (("cell", Mc, mc_), ("fine", Mf, mf), ("charge", Mq, mq)):
        sp = split(X, X, m, grp)
        mc_rows[kind] = {"s_run": sp["s_mc_run"], "t_ms": 1e3 * mc_time, "N": Nm[0], "s_src_frac": sp["s_src_frac"]}
    res["mc"]["metrics"] = mc_rows
    # bias uncertainty. The bias^2 estimate carries a cross term between the bias pattern and the noise (spatially
    # correlated: a 20 MeV track crosses the map). Its size is measured per row from the two seeds scored separately,
    # (b2_1 - b2_2) / 2 ~ the cross term of one seed; pooled over rows as f = rms[(b2_1 - b2_2) / (2 b floor)], then
    # sigma(b^2) = f sqrt((2 b floor)^2 + floor^4). The seeds' source samples are shared by all configurations, so part
    # of this is common to every row: differences between rows are better determined than absolute biases.
    kcal = {}
    for kind in ("cell", "fine", "charge"):
        z = [((r["b2_seeds"][0] - r["b2_seeds"][1]) / 2) / (max(r["bias"], 1e-4) * r["bias_floor"])
             for r in rows if r["kind"] == kind and len(r["b2_seeds"]) >= 2]
        kcal[kind] = float(np.sqrt(np.mean(np.square(z)))) if z else 0.5
    res["mc"]["cross_term_factor"] = kcal
    print("bias cross-term factor f (per-seed scores): " + ", ".join(f"{k} {v:.3f}" for k, v in kcal.items()))
    for r in rows:
        fl_ = r["bias_floor"]
        e2 = kcal[r["kind"]] * np.sqrt((2 * r["bias"] * fl_) ** 2 + fl_ ** 4)
        r["bias2_err"] = float(e2)
        r["bias_lo"] = float(np.sqrt(max(r["bias2"] - e2, 0.0)))
        r["bias_hi"] = float(np.sqrt(max(r["bias2"] + e2, 0.0)))
    eps_list = [float(e) for e in a.eps.split(",")]
    for r in rows:
        mr = mc_rows[r["kind"]]
        r["speedup"] = {}
        for e in eps_list:
            tg = t_to(e, r["bias"], r["s_run"], r["t_ms"] * r["N"])
            tm = t_to(e, 0.0, mr["s_run"], mr["t_ms"] * mr["N"])
            r["speedup"][f"{e:g}"] = float(tm / tg) if np.isfinite(tg) else 0.0
    res["rows"] = rows
    with open(os.path.join(a.out_dir, "sweep_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"MC: {1e3 * mc_time:.3f} ms/primary; per-run ({Nm[0] / 1e3:.0f}k) noise cell {100 * mc_rows['cell']['s_run']:.2f} %, "
          f"fine {100 * mc_rows['fine']['s_run']:.2f} %, charge {100 * mc_rows['charge']['s_run']:.2f} %, "
          f"source-noise share {mc_rows['fine']['s_src_frac']:.2f}")
    print(f"{'run':<14}{'tally':<12}{'kind':<7}{'ms/prim':>8}{'noise%':>8}{'bias%':>8} [1-sigma range]  speed-up at " +
          ", ".join(f"{100 * e:g} %" for e in eps_list))
    for r in rows:
        print(f"{r['run']:<14}{str(r['tally']):<12}{r['kind']:<7}{r['t_ms']:>8.3f}{100 * r['s_run'] * np.sqrt(r['N'] / 1e5):>8.2f}"
              f"{100 * r['bias']:>8.2f} [{100 * r['bias_lo']:.2f}-{100 * r['bias_hi']:.2f}]  " +
              ", ".join(f"{v:.2f}" for v in r["speedup"].values()))
    print("(noise% is per 100k primaries)")
    if a.flat:
        res["flat"] = flat_metrics(a.flat)
        with open(os.path.join(a.out_dir, "sweep_metrics.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    with open(a.groups) as fh:
        groups = json.load(fh)
    plot_all(res, groups, a.out_dir, plt, eps_list)
    if a.flat:
        plot_flat(res["flat"], groups.get("flat_order"), os.path.join(a.out_dir, "flat_beam_sweep.png"), plt)


def plot_flat(fl, order, path, plt):
    """Flat-beam within-cell entry-density rms (cells 4/6/8) per configuration, analog chain as the noise floor."""
    labs = [k for k in (order or fl.keys()) if k in fl and k != "mc"]
    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.75 * len(labs) + 2), 4.2))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(labs))
    for k, (c, off) in enumerate(zip((3, 5, 7), (-0.25, 0, 0.25))):
        ax.bar(x + off, [100 * fl[l]["rms_cells"][k] for l in labs], width=0.25, color=cols[k],
               label=f"cell {c + 1}")
    if "mc" in fl:
        for k, c in enumerate((3, 5, 7)):
            ax.axhline(100 * fl["mc"]["rms_cells"][k], color=cols[k], ls=":", lw=1)
        ax.text(len(labs) - 0.5, 100 * max(fl["mc"]["rms_cells"]) + 0.05, "analog chain (counting noise), dotted",
                ha="right", va="bottom", fontsize=7, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="within-cell entry density rms (%)",
           title="Flat 20 MeV beam, 300k primaries: within-cell pattern along the beam (truth = flat)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def flat_metrics(specs, cells=(3, 5, 7)):
    """Flat-beam chain test (validation/flat_beam_chain.py): rms deviation from flat of the left-entry density in 8
    sub-rows, averaged over cells 4, 6, 8 (0-based 3, 5, 7), for each label=file.npz."""
    from flat_beam_chain import entry_patterns
    out = {}
    for sp in specs:
        lab, f = sp.split("=", 1)
        T = dict(np.load(f, allow_pickle=False))
        pat = entry_patterns(T, int(T["ny"]), cells)
        rms = [float(np.sqrt(np.mean((pat[c] - 1) ** 2))) for c in cells]
        out[lab] = {"file": f, "rms_cells": rms, "rms": float(np.mean(rms)), "seconds": float(T["seconds"])}
        print(f"flat beam {lab:<14} rms cells 4/6/8: " + " ".join(f"{100 * x:.2f}" for x in rms) + " %")
    return out


def find(rows, run, kind, tally=None):
    for r in rows:
        if r["run"] == run and r["kind"] == kind and (kind != "fine" or r["tally"] == tally):
            return r
    raise KeyError((run, kind, tally))


def plot_all(res, figs, out_dir, plt, eps_list):
    """figs: {"figure name": {"panel title": {"kind": "cell"|"fine", "xlabel": ..., "points": [[x, run, tally], ...],
    "xlog": bool, "xticklabels": [...] (optional), "in_time_plot": bool}}}"""
    for name, groups in figs.items():
        if name.startswith("panels"):
            plot_panels(res, groups, os.path.join(out_dir, name + ".png"), plt)
    plot_time(res, figs.get("time", {}), out_dir, plt, eps_list)


def plot_panels(res, groups, path, plt):
    import matplotlib
    rows, mc = res["rows"], res["mc"]["metrics"]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # per-variable panels: bias (with the resolution floor), noise at 1M primaries and time per primary
    n = len(groups)
    has_train = any("train_min" in g for g in groups.values())
    nr = 3 if has_train else 2
    fig, ax = plt.subplots(nr, n, figsize=(3.6 * n, 3.3 * nr), squeeze=False)
    for j, (title, gdef) in enumerate(groups.items()):
        pts = [(x, find(rows, run, gdef["kind"], tal)) for x, run, tal in gdef["points"]]
        xs = np.array([float(x) for x, _ in pts])
        b = np.array([100 * r["bias"] for _, r in pts])
        fl = np.array([100 * r["bias_floor"] for _, r in pts])
        sn = np.array([100 * r["s_run"] * np.sqrt(r["N"] / 1e6) for _, r in pts])
        tms = np.array([r["t_ms"] for _, r in pts])
        lo = np.array([100 * r["bias_lo"] for _, r in pts])
        hi = np.array([100 * r["bias_hi"] for _, r in pts])
        a0 = ax[0, j]
        a0.errorbar(xs, b, yerr=[b - lo, hi - b], fmt="o-", color=cols[0], capsize=3, label="bias (1 sigma)")
        a0.plot(xs, sn, "s--", color=cols[1], label="noise, 1M primaries")
        a0.axhline(100 * mc[gdef["kind"]]["s_run"] * np.sqrt(mc[gdef["kind"]]["N"] / 1e6), color="0.3", ls=":",
                   label="analog MC noise, 1M primaries")
        a0.set(title=title, ylabel=f"rel. L2 error, {'1 mm cells' if gdef['kind'] == 'cell' else '125 um'} (%)")
        a0.set_ylim(bottom=0)
        a1 = ax[1, j]
        a1.plot(xs, tms, "o-", color=cols[2], label="GMC" + (" + SCR" if gdef["kind"] == "fine" else ""))
        a1.axhline(mc[gdef["kind"]]["t_ms"], color="0.3", ls=":", label="analog MC")
        a1.set(xlabel="" if has_train else gdef["xlabel"], ylabel="wall time per primary (ms)")
        a1.set_yscale("log")
        a1.set_ylim(0.1, 6)
        a1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a1.set_yticks([0.1, 0.2, 0.5, 1, 2, 4])
        a1.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "4"])
        for x_, t_ in zip(xs, tms):
            a1.annotate(f"{t_:.2f}", (x_, t_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        axs = [a0, a1]
        if has_train:
            a2 = ax[2, j]
            tr = gdef.get("train_min")
            if tr:
                a2.bar(xs if not gdef.get("xlog") else xs, tr, width=(0.25 * xs) if gdef.get("xlog") else 0.5,
                       color=cols[3])
                a2.set(ylabel=gdef.get("train_label", "training time (min)"))
            else:
                a2.text(0.5, 0.5, "inference only:\nno retraining", ha="center", va="center", transform=a2.transAxes,
                        color="0.4")
                a2.axis("off")
                a1.set(xlabel=gdef["xlabel"])
            if tr:
                a2.set(xlabel=gdef["xlabel"])
                axs.append(a2)
        for a_ in axs:
            if gdef.get("xlog"):
                a_.set_xscale("log")
            a_.set_xticks(xs)
            a_.set_xticklabels(gdef.get("xticklabels", [str(x) for x, _ in pts]), fontsize=8)
            a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if j == 0:
            a0.legend(fontsize=7)
            a1.legend(fontsize=7)
        a0.set_ylim(top=max(a0.get_ylim()[1], 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_time(res, sel_by_kind, out_dir, plt, eps_list):
    """sel_by_kind: {"cell": [[run, tally, label], ...], "fine": [...]}: error vs wall time, MC vs configurations."""
    rows, mc = res["rows"], res["mc"]["metrics"]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    t = np.logspace(0, 4.5, 200)                                       # seconds
    for kind, sel in sel_by_kind.items():
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.2))
        mr = mc[kind]
        ax.plot(t, 100 * mr["s_run"] * np.sqrt(mr["t_ms"] * 1e-3 * mr["N"] / t), color="k", lw=2,
                label=f"analog MC ({mr['t_ms']:.2f} ms/primary)")
        for k, (run, tal, lab) in enumerate(sel):
            r = find(rows, run, kind, tal)
            err = 100 * np.sqrt(r["bias"] ** 2 + r["s_run"] ** 2 * r["t_ms"] * 1e-3 * r["N"] / t)
            ax.plot(t, err, color=cols[k % len(cols)], ls="-" if k < len(cols) else "--",
                    label=f"{lab} ({r['t_ms']:.2f} ms/primary, bias {100 * r['bias']:.1f} %)")
        ax.set(xscale="log", yscale="log", xlabel="wall time (s)",
               ylabel=f"expected rel. L2 error vs truth, {'1 mm cells' if kind == 'cell' else '125 um map'} (%)",
               title=f"Drumm cylinder, photons escape: error vs cost ({'GMC' if kind == 'cell' else 'GMC + SCR'})")
        for e in eps_list:
            ax.axhline(100 * e, color="0.8", lw=0.7)
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"error_vs_time_{kind}.png"), dpi=130)
        plt.close(fig)
    print(f"saved figures in {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--tally", action="append", default=[], help="label=kernel_dir:K (repeatable)")
    r.add_argument("--N", type=int, default=200000)
    r.add_argument("--n_seeds", type=int, default=2)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--src_seed", type=int, default=5000)
    r.add_argument("--batch", type=int, default=100000)
    r.add_argument("--ode_steps", type=int, default=20)
    r.add_argument("--mirror", action="store_true")
    r.add_argument("--E_cutoff", type=float, default=1e-3)
    r.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--runs", nargs="+", required=True)
    z.add_argument("--mc", nargs="+", required=True)
    z.add_argument("--mc_ms", type=float, default=None, help="analog MC ms/primary (clean timing); default from meta")
    z.add_argument("--groups", required=True, help="json: panel definitions (see plot_all)")
    z.add_argument("--eps", default="0.01,0.02,0.03")
    z.add_argument("--flat", nargs="*", default=[], help="label=flat_beam_run.npz (flat-beam chain test)")
    z.add_argument("--out_dir", required=True)
    a = p.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
