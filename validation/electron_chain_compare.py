#!/usr/bin/env python3
"""
Chained GMC electron dose vs analog Monte Carlo on a multi-cell grid.

Needs the MC references from validation/mc_chain.py (ElectronTransport environment). Every GMC chain reuses the
primaries stored there, so all doses refer to the same source.

    Full MC        analog MC on the whole grid (truth)
    MC-chained     analog MC as the single-cell kernel (best possible GMC with this model structure)
    baseline       2d_electron flow chained (the original Macro_field kernel)
    face           exit classifier + face-conditioned flow chained
    --noise_ref    a second mc_chain.py file with another seed: MC-vs-MC statistical floor

Example:
    python validation/electron_chain_compare.py --mc_ref validation/results/chain_analog_cu/mc_ref.npz \
        --noise_ref validation/results/chain_analog_cu/mc_ref_seed456.npz \
        --baseline_ckpt ml/models/2d_electron_boundary_analog_cu/best_model.pth \
        --baseline_norm ml/data/norm_params_2d_electron_boundary_analog_cu.npy \
        --face_ckpt ml/models/2d_electron_face_boundary_analog_cu/best_model.pth \
        --face_norm ml/data/norm_params_2d_electron_face_boundary_analog_cu.npy \
        --classifier ml/models/2d_electron_analog_cu/exit_classifier.pth
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from chain_core import chain  # noqa: E402
from gmc_kernels import BaselineFlowKernel, FaceFlowKernel  # noqa: E402


def dose_metrics(d, ref, E_source):
    """Errors of dose map d against ref (same units). Cells below 1% of the ref max are excluded from the
    cellwise relative errors (they are dominated by MC noise)."""
    m = ref > 0.01 * ref.max()
    rel = (d[m] - ref[m]) / ref[m]
    depth_d, depth_r = d.sum(1), ref.sum(1)
    lat_d, lat_r = d.sum(0), ref.sum(0)
    return {
        "rel_L2": float(np.linalg.norm(d - ref) / np.linalg.norm(ref)),
        "cell_rel_err_mean_abs": float(np.abs(rel).mean()),
        "cell_rel_err_p95_abs": float(np.quantile(np.abs(rel), 0.95)),
        "total_deposit_rel": float(d.sum() / ref.sum() - 1),
        "deposited_frac_of_source": float(d.sum() / E_source),
        "depth_dose_rel_L2": float(np.linalg.norm(depth_d - depth_r) / np.linalg.norm(depth_r)),
        "lateral_rel_L2": float(np.linalg.norm(lat_d - lat_r) / np.linalg.norm(lat_r)),
        "depth_centroid_shift_cells": float((np.arange(len(depth_d)) * depth_d).sum() / depth_d.sum()
                                            - (np.arange(len(depth_r)) * depth_r).sum() / depth_r.sum()),
    }


def plot(doses, ref_name, meta, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    names = list(doses)
    ref = doses[ref_name]
    cs = meta["cell_size"] * 1e4  # um
    ext = [0, meta["nx"] * cs, 0, meta["ny"] * cs]
    vmax = max(d.max() for d in doses.values())
    vmin = vmax * 1e-4

    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 6.8))
    for k, n in enumerate(names):
        ax = axes[0, k]
        im = ax.imshow(np.maximum(doses[n].T, vmin), origin="lower", extent=ext, norm=LogNorm(vmin, vmax),
                       cmap="inferno")
        ax.set_title(n, fontsize=10), ax.set_xlabel("x [um]")
        ax = axes[1, k]
        rel = np.where(ref > 0.01 * ref.max(), (doses[n] - ref) / np.maximum(ref, 1e-300), np.nan)
        im2 = ax.imshow(rel.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_title(f"({n} - {ref_name}) / {ref_name}", fontsize=8), ax.set_xlabel("x [um]")
    axes[0, 0].set_ylabel("y [um]"), axes[1, 0].set_ylabel("y [um]")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, label="MeV / cm^2 / primary")
    fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, label="relative difference")
    fig.savefig(os.path.join(out_dir, "dose_maps.png"), dpi=170, bbox_inches="tight"), plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    x = (np.arange(meta["nx"]) + 0.5) * cs
    y = (np.arange(meta["ny"]) + 0.5) * cs
    for n, d in doses.items():
        st = dict(color="k", lw=2) if n == "full MC" else dict(color="0.5", lw=2, ls="--") if n == "MC-chained" \
            else {}
        a1.plot(x, d.sum(1), label=n, **st)
        a2.plot(y, d.sum(0), label=n, **st)
    a1.set_yscale("log"), a1.set_xlabel("depth x [um]"), a1.set_ylabel("dose summed over y"), a1.legend(fontsize=8)
    a2.set_yscale("log"), a2.set_xlabel("lateral y [um]"), a2.set_ylabel("dose summed over x")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout(), fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=170), plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Chained GMC dose vs analog MC")
    ap.add_argument("--mc_ref", required=True)
    ap.add_argument("--noise_ref", default=None)
    ap.add_argument("--baseline_ckpt"), ap.add_argument("--baseline_norm")
    ap.add_argument("--baseline_with_classifier", action="store_true",
                    help="Also chain the baseline flow with the classifier deciding the exit class (rejection)")
    ap.add_argument("--face_ckpt"), ap.add_argument("--face_norm"), ap.add_argument("--classifier")
    ap.add_argument("--ode_steps", type=int, default=100)
    ap.add_argument("--ode_method", default="euler")
    ap.add_argument("--stop_f_out", type=float, default=1e-3,
                    help="Baseline: E_out within this fraction of (E_in - E_cut) above the cutoff = absorbed")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = np.load(a.mc_ref)
    meta = json.loads(str(ref["meta"]))
    src = ref["source"]
    out_dir = a.out_dir or os.path.dirname(a.mc_ref)
    os.makedirs(out_dir, exist_ok=True)
    E_source = float(src[:, 4].sum())
    norm = 1.0 / (len(src) * meta["cell_size"] ** 2)  # MeV per cell -> MeV / cm^2 / primary

    doses = {"full MC": ref["dose_full"] * norm, "MC-chained": ref["dose_mcchain"] * norm}
    info = {}
    if a.noise_ref:
        nr = np.load(a.noise_ref)
        doses["MC-chained (seed 2)"] = nr["dose_mcchain"] * norm
    kernels = {}
    kw = dict(device=device, ode_steps=a.ode_steps, ode_method=a.ode_method, seed=a.seed)
    if a.baseline_ckpt:
        kernels["baseline"] = BaselineFlowKernel(a.baseline_ckpt, a.baseline_norm, stop_f_out=a.stop_f_out, **kw)
        if a.baseline_with_classifier:
            kernels["baseline+classifier"] = BaselineFlowKernel(a.baseline_ckpt, a.baseline_norm,
                                                                classifier=a.classifier, stop_f_out=a.stop_f_out, **kw)
    if a.face_ckpt:
        kernels["face"] = FaceFlowKernel(a.face_ckpt, a.face_norm, a.classifier, **kw)
    for name, k in kernels.items():
        t0 = time.time()
        res = chain(k, src, meta["nx"], meta["ny"], meta["cell_size"], max_steps=a.max_steps,
                    E_cutoff=meta["E_cutoff"], log=None)
        doses[name] = res["dose"] * norm
        info[name] = {"seconds": round(time.time() - t0, 1), "steps": res["steps"], "transits": res["transits"],
                      "escaped_frac": {s: v / E_source for s, v in res["escaped"].items()},
                      "in_flight_frac": res["in_flight"] / E_source}
        if getattr(k, "sampled", 0):
            info[name]["unmatched_frac"] = k.unmatched / k.sampled
        print(f"{name}: {info[name]}")
    info["MC-chained"] = {"escaped_frac": {s: v / E_source for s, v in json.loads(str(ref["escaped_mcchain"])).items()}}
    info["full MC"] = {"escaped_electron_frac": float(ref["escaped_full_electrons"]) / E_source,
                       "photon_frac": float(ref["photons_full"]) / E_source}

    E_src_n = E_source * norm  # source energy in the dose units, for deposited_frac_of_source
    results = {"vs_MC-chained (model error)": {n: dose_metrics(d, doses["MC-chained"], E_src_n)
                                               for n, d in doses.items() if n not in ("full MC", "MC-chained")},
               "vs_full_MC (total error)": {n: dose_metrics(d, doses["full MC"], E_src_n)
                                            for n, d in doses.items() if n != "full MC"},
               "runs": info, "meta": meta}
    with open(os.path.join(out_dir, "chain_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    plot(doses, "MC-chained", meta, out_dir)

    keys = ["rel_L2", "cell_rel_err_mean_abs", "cell_rel_err_p95_abs", "total_deposit_rel", "deposited_frac_of_source", "depth_dose_rel_L2",
            "lateral_rel_L2", "depth_centroid_shift_cells"]
    for title, block in results.items():
        if not title.startswith("vs_"):
            continue
        print(f"\n{title}")
        print(f"{'':28s}" + "".join(f"{n:>22s}" for n in block))
        for k in keys:
            print(f"{k:28s}" + "".join(f"{block[n][k]:22.4g}" for n in block))
    print(f"\nWrote {out_dir}/chain_metrics.json, dose_maps.png, profiles.png")


if __name__ == "__main__":
    main()
