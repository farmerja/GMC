#!/usr/bin/env python3
"""
SCR charge kernel: sub-cell charge deposition of the electrons the relevant-only GMC never simulates.

Charge deposition = electrons stopped - electrons created. The chained GMC knows where its own electrons are born
(relevant daughters, interior starts) and, with the "stop" flow of ml/secondary_gmc.py, where absorbed transits
stop. What it drops are the IRRELEVANT secondaries: electrons born and stopped inside the cell (their subtree never
leaves it). Each adds -1 at its birth point and +1 at its stop point. Cell totals are unaffected, but at sub-cell
resolution these dipoles matter: dropped, the downstream bins of every cell lose their stops and the upstream bins
their births (seen as a +-0.1 e/primary saw-tooth along the beam axis in the Drumm water cylinder).

Training data: <data>_subcell.npz from ElectronTransport/generate_analog_batches.py --subcell_bins n (2026-11-15 or
later): q_seg (anchor segment, where the GMC charges the subtree's dose) and q_pairs (birth x0, y0, stop x1, y1 in
cm, birth energy E). Only pairs born above --q_th count (the threshold tally of the analog MC / the paper's 0.1 MeV
electron cutoff); a birth above q_th needs a parent above ~2 q_th, so only those transits are modelled.

Two models on the SCR kernel's transit features (ml/subcell_kernel.cond_features, same frames and symmetries):
    rate   lambda(transit) = expected number of such pairs (Poisson regression)
    pair   flow-matching density of (logit birth x, logit birth y, stop x - birth x, stop y - birth y)
Tally (charge_tally): every transit places K sampled pairs with weight lambda / K: -w at birth, +w at stop. This is
unbiased for the expected charge image (linearity), with no extra noise from the pair count.

    python ml/charge_kernel.py train --train_npz ../ElectronTransport/<data>.npz --out ml/models/charge_kernel_<name>
    python ml/charge_kernel.py eval --model ml/models/charge_kernel_<name> --test_npz ../ElectronTransport/<test>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.subcell_kernel import (Spec, apply_d4, canonical_to_cell, cond_dim, cond_features, d4_points,  # noqa: E402
                               gather, load_kernel_data, random_sym, to_device, transit_log_tensors)
from ml.secondary_gmc import BOUNDARY_SYM, logit, sample_flow, sigmoid, train_flow  # noqa: E402
from ml.model import get_model  # noqa: E402


class RateNet(nn.Module):
    def __init__(self, cond_dim, hidden=256, depth=4):
        super().__init__()
        self.arch = dict(cond_dim=cond_dim, hidden=hidden, depth=depth)
        layers, d = [], cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(feats).squeeze(-1)          # log lambda


# ====================================================================================================================
# Data
# ====================================================================================================================
def load_charge_data(seg_npz, q_th, sub_npz=None, max_segments=None):
    """Kept transits (as the SCR kernel) + their irrelevant-electron pairs born above q_th (cell units)."""
    sub_npz = sub_npz or seg_npz[:-4] + "_subcell.npz"
    spec, R, _, rows = load_kernel_data(seg_npz, sub_npz, max_segments=max_segments, entries=False,
                                        return_rows=True)
    S = np.load(sub_npz, allow_pickle=False)
    if "q_pairs" not in S.files:
        raise ValueError(f"{sub_npz} has no charge pairs (made before 2026-11-15; regenerate)")
    cols = [str(c) for c in S["q_columns"]]
    P, qs = S["q_pairs"], S["q_seg"].astype(np.int64)
    m = P[:, cols.index("E")] > q_th
    new = np.full(int(S["n_segments"]), -1, dtype=np.int64)
    new[rows] = np.arange(len(rows))
    t = new[qs[m]]
    ok = t >= 0
    P = P[m][ok].astype(np.float64)
    L = spec.L
    pairs = {"t": t[ok], "x0": P[:, cols.index("x0")] / L, "y0": P[:, cols.index("y0")] / L,
             "x1": P[:, cols.index("x1")] / L, "y1": P[:, cols.index("y1")] / L, "E": P[:, cols.index("E")]}
    k = np.bincount(pairs["t"], minlength=len(rows))
    low = int((R["E0"][pairs["t"]] <= 2 * q_th).sum())
    print(f"charge pairs born > {q_th:g} MeV: {int(m.sum())} ({int((~ok).sum())} on transits the kernel drops), "
          f"{int((k > 0).sum())} transits with pairs, max {int(k.max()) if len(k) else 0} per transit; "
          f"{low} pairs on parents <= 2 q_th", flush=True)
    return spec, R, pairs, k


def pair_targets(x0, y0, x1, y1):
    return np.stack([logit(x0), logit(y0), x1 - x0, y1 - y0], 1)


def pair_points(z):
    bx, by = sigmoid(z[:, 0]), sigmoid(z[:, 1])
    return bx, by, np.clip(bx + z[:, 2], 0.0, 1.0), np.clip(by + z[:, 3], 0.0, 1.0)


def pair_flow_data(spec, R, pairs, device):
    """Pair rows with their transit's features under every allowed symmetry (interior: D4, boundary: y-mirror)."""
    t = pairs["t"]
    Rd = to_device(gather(R, t), device)
    is_int = R["start"][t] > 0.5
    conds, tgts = [], []
    for k in range(8):
        sel = np.ones(len(t), bool) if k in BOUNDARY_SYM else is_int
        if not sel.any():
            continue
        idx = torch.as_tensor(np.flatnonzero(sel), device=device)
        kk = torch.full((len(idx),), k, device=device)
        Rk = apply_d4(kk, gather(Rd, idx))
        f = cond_features(spec, Rk).cpu().numpy()
        pts = [torch.as_tensor(pairs[c][sel], device=device, dtype=torch.float32) for c in ("x0", "y0", "x1", "y1")]
        bx, by = d4_points(kk, pts[0], pts[1])
        sx, sy = d4_points(kk, pts[2], pts[3])
        conds.append(f)
        tgts.append(pair_targets(*(a.double().cpu().numpy() for a in (bx, by, sx, sy))))
    return np.concatenate(conds).astype(np.float32), np.concatenate(tgts).astype(np.float32)


# ====================================================================================================================
# Training
# ====================================================================================================================
def train_rate(spec, Rd, k, rows, device, epochs, batch=65536, lr=2e-3, val_frac=0.02, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n = len(rows)
    perm = torch.randperm(n, device=device, generator=g)
    n_val = max(1, int(val_frac * n))
    va, tr = rows[perm[:n_val]], rows[perm[n_val:]]
    batch = min(batch, len(tr))
    kd = torch.as_tensor(k, device=device, dtype=torch.float32)
    model = RateNet(cond_dim(spec)).to(device)
    with torch.no_grad():
        model.net[-1].bias.fill_(float(np.log(max(k[rows.cpu().numpy()].mean(), 1e-8))))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * max(1, len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def nll(idx, augment):
        Rb = gather(Rd, idx)
        if augment:
            Rb = apply_d4(random_sym(Rb["start"], g), Rb)
        s = model(cond_features(spec, Rb))
        return torch.exp(s) - kd[idx] * s           # Poisson NLL up to a constant

    @torch.no_grad()
    def val():
        model.eval()
        tot = sum(float(nll(va[i:i + 262144], False).sum()) for i in range(0, len(va), 262144))
        model.train()
        return tot / len(va)

    best, best_state, hist, t0 = float("inf"), None, [], time.time()
    for ep in range(epochs):
        p = tr[torch.randperm(len(tr), device=device, generator=g)]
        run = 0.0
        for i in range(len(tr) // batch):
            loss = nll(p[i * batch:(i + 1) * batch], True).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss.detach())
        v = val()
        hist.append((run / max(1, len(tr) // batch), v))
        if v < best:
            best, best_state = v, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        print(f"  [rate] epoch {ep + 1}/{epochs}: train {hist[-1][0]:.5f} val {v:.5f} ({time.time() - t0:.0f}s)",
              flush=True)
    model.load_state_dict(best_state)
    return model.eval(), hist


def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec, R, pairs, k = load_charge_data(args.train_npz, args.q_th, max_segments=args.max_segments)
    Rd = to_device(R, dev)
    rows = torch.as_tensor(np.flatnonzero(R["E0"] > 2 * args.q_th), device=dev)
    print(f"\n== rate: {len(rows)} transits with E0 > {2 * args.q_th:g} MeV, mean count "
          f"{k[rows.cpu().numpy()].mean():.4g}", flush=True)
    rate, rate_hist = train_rate(spec, Rd, k, rows, dev, args.rate_epochs)
    cond, tgt = pair_flow_data(spec, R, pairs, dev)
    print(f"\n== pair flow: {len(tgt)} rows (D4 / y-mirror copies of {len(pairs['t'])} pairs), cond {cond.shape[1]}",
          flush=True)
    F = train_flow(cond, tgt, dev, epochs=args.flow_epochs, batch=4096, label="pair", hidden=256, depth=6)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"spec": vars(spec), "q_th": args.q_th, "rate": {"arch": rate.arch, "state": rate.state_dict(),
                                                                "history": rate_hist},
                "pair": {"state": F["model"].state_dict(), "mu": F["mu"], "sd": F["sd"], "cond_dim": F["cond_dim"],
                         "target_dim": F["target_dim"], "arch": F["arch"], "history": F["history"]},
                "train_npz": args.train_npz, "n_pairs": int(len(pairs["t"]))},
               os.path.join(args.out, "charge_kernel.pth"))
    print(f"saved {args.out}/charge_kernel.pth")


def load_charge_kernel(path, device):
    if os.path.isdir(path):
        path = os.path.join(path, "charge_kernel.pth")
    b = torch.load(path, map_location=device, weights_only=False)
    rate = RateNet(**b["rate"]["arch"]).to(device)
    rate.load_state_dict(b["rate"]["state"])
    p = b["pair"]
    a = p["arch"]
    m = get_model(p["cond_dim"] + p["target_dim"], p["cond_dim"], hidden_dim=a["hidden"], depth=a["depth"],
                  expansion_factor=a["expansion"]).to(device)
    m.load_state_dict(p["state"])
    flow = {"model": m.eval(), "mu": p["mu"], "sd": p["sd"], "cond_dim": p["cond_dim"], "target_dim": p["target_dim"]}
    return {"spec": Spec(**b["spec"]), "q_th": float(b["q_th"]), "rate": rate.eval(), "pair": flow, "blob": b}


@torch.no_grad()
def sample_pairs(ck, Rd, device, K=1, ode_steps=50, gen=None):
    """Rows of Rd (transit features) -> lambda (N,) and K sampled pairs per row (N, K, 4: bx, by, sx, sy)."""
    feats = cond_features(ck["spec"], Rd)
    lam = torch.exp(ck["rate"](feats)).double().cpu().numpy()
    f = feats.cpu().numpy()
    z = sample_flow(ck["pair"], np.repeat(f, K, 0), device, ode_steps, "midpoint", gen)
    pts = np.stack(pair_points(z), 1).reshape(len(f), K, 4)
    return lam, pts


# ====================================================================================================================
# Chain tally
# ====================================================================================================================
@torch.no_grad()
def charge_tally(log, ck, nx, ny, n, device="cpu", K=1, ode_steps=50, gen=None, chunk=262144, material=0):
    """Expected sub-cell charge of the irrelevant electrons of a chain transit log (born above ck q_th) on the
    (nx * n, ny * n) fine grid: lambda / K at each of K sampled stop points, -lambda / K at their birth points."""
    T = log["transits"]
    fine = np.zeros((nx * n, ny * n))
    rows = np.flatnonzero((np.asarray(T["mat"]) == material) & (np.asarray(T["E0"]) > 2 * ck["q_th"])
                          & (np.asarray(T["dose"]) > 0))
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        lam, pts = sample_pairs(ck, transit_log_tensors(T, r, device), device, K, ode_steps, gen)
        face = torch.as_tensor(np.repeat(np.asarray(T["face"])[r], K))
        cx, cy = np.repeat(np.asarray(T["cx"])[r], K), np.repeat(np.asarray(T["cy"])[r], K)
        w = np.repeat(lam / K, K)
        P = torch.as_tensor(pts.reshape(-1, 4))
        for (jx, jy), sgn in (((0, 1), -1.0), ((2, 3), 1.0)):
            x, y = canonical_to_cell(P[:, jx], P[:, jy], face)
            ix = cx * n + np.clip((x.numpy() * n).astype(np.int64), 0, n - 1)
            iy = cy * n + np.clip((y.numpy() * n).astype(np.int64), 0, n - 1)
            np.add.at(fine, (ix, iy), sgn * w)
    return fine


# ====================================================================================================================
# Evaluation: aggregated charge images vs the held-out MC pairs
# ====================================================================================================================
def cmd_eval(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_charge_kernel(args.model, dev)
    spec, R, pairs, k = load_charge_data(args.test_npz, ck["q_th"], max_segments=args.max_segments)
    rows = np.flatnonzero(R["E0"] > 2 * ck["q_th"])
    rng = np.random.default_rng(args.seed)
    if len(rows) > args.n_eval:
        rows = np.sort(rng.choice(rows, args.n_eval, replace=False))
    n = args.n
    gen = torch.Generator().manual_seed(args.seed)
    bands = [(0.2, 1.0), (1.0, 5.0), (5.0, 30.0)]
    res = {"model": args.model, "test_npz": args.test_npz, "q_th": ck["q_th"], "n": n, "bands": {}}
    in_rows = np.zeros(len(R["E0"]), bool)
    in_rows[rows] = True
    lam_all = np.zeros(len(R["E0"]))
    img_p = {}
    for st in (0, 1):
        for lo, hi in bands:
            img_p[(st, lo)] = np.zeros((n, n))
    for i in range(0, len(rows), 262144):
        r = rows[i:i + 262144]
        lam, pts = sample_pairs(ck, to_device(gather(R, r), dev), dev, args.K, args.ode_steps, gen)
        lam_all[r] = lam
        st = (R["start"][r] > 0.5).astype(int)
        for s_ in (0, 1):
            for lo, hi in bands:
                m = (st == s_) & (R["E0"][r] > lo) & (R["E0"][r] <= hi)
                if not m.any():
                    continue
                P = pts[m].reshape(-1, 4)
                w = np.repeat(lam[m] / args.K, args.K)
                for (jx, jy), sgn in (((0, 1), -1.0), ((2, 3), 1.0)):
                    b = (np.clip((P[:, jx] * n).astype(int), 0, n - 1), np.clip((P[:, jy] * n).astype(int), 0, n - 1))
                    np.add.at(img_p[(s_, lo)], b, sgn * w)
    sel = in_rows[pairs["t"]]
    for s_ in (0, 1):
        for lo, hi in bands:
            t = pairs["t"][sel]
            m = ((R["start"][t] > 0.5).astype(int) == s_) & (R["E0"][t] > lo) & (R["E0"][t] <= hi)
            tr = (R["E0"] > lo) & (R["E0"] <= hi) & ((R["start"] > 0.5).astype(int) == s_) & in_rows
            if tr.sum() == 0:
                continue
            img_t = np.zeros((n, n))
            for (kx, ky), sgn in ((("x0", "y0"), -1.0), (("x1", "y1"), 1.0)):
                b = (np.clip((pairs[kx][sel][m] * n).astype(int), 0, n - 1),
                     np.clip((pairs[ky][sel][m] * n).astype(int), 0, n - 1))
                np.add.at(img_t, b, sgn)
            P_ = img_p[(s_, lo)]
            # noise of the true image: each pair is an independent +-1 dipole -> per-bin variance = number of
            # endpoints in the bin; profiles sum bins
            var = np.zeros((n, n))
            for kx, ky in (("x0", "y0"), ("x1", "y1")):
                b = (np.clip((pairs[kx][sel][m] * n).astype(int), 0, n - 1),
                     np.clip((pairs[ky][sel][m] * n).astype(int), 0, n - 1))
                np.add.at(var, b, 1.0)
            key = f"{'interior' if s_ else 'boundary'} {lo:g}-{hi:g} MeV"
            d_t, d_p, d_v = img_t.sum(1), P_.sum(1), var.sum(1)
            l_t, l_p, l_v = img_t.sum(0), P_.sum(0), var.sum(0)
            res["bands"][key] = {
                "transits": int(tr.sum()), "pairs_true": int(m.sum()), "pairs_pred": float(lam_all[tr].sum()),
                "depth_profile_chi2_per_bin": float(np.mean((d_p - d_t) ** 2 / np.maximum(d_v, 1))),
                "lateral_profile_chi2_per_bin": float(np.mean((l_p - l_t) ** 2 / np.maximum(l_v, 1))),
                "image_chi2_per_bin": float(np.mean((P_ - img_t) ** 2 / np.maximum(var, 1))),
                "depth_true": d_t.tolist(), "depth_pred": d_p.tolist()}
            r_ = res["bands"][key]
            print(f"{key:28s} transits {r_['transits']:8d}: pairs MC {r_['pairs_true']:6d} / model "
                  f"{r_['pairs_pred']:9.1f}; chi2/bin depth {r_['depth_profile_chi2_per_bin']:.2f}, lateral "
                  f"{r_['lateral_profile_chi2_per_bin']:.2f}, image {r_['image_chi2_per_bin']:.2f}", flush=True)
    out = args.out or os.path.join(args.model if os.path.isdir(args.model) else os.path.dirname(args.model),
                                   "eval_" + os.path.basename(args.test_npz)[:-4])
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k_ for k_ in res["bands"]]
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6), squeeze=False)
    xc = (np.arange(n) + 0.5) / n
    for a_, k_ in zip(ax[0], keys):
        b = res["bands"][k_]
        a_.step(xc, b["depth_true"], where="mid", color="k", label="MC (held out)")
        a_.step(xc, b["depth_pred"], where="mid", color="C3", label="charge kernel")
        a_.axhline(0, color="0.7", lw=0.6)
        a_.set_title(k_, fontsize=9)
        a_.set_xlabel("x (cell units; boundary: entry face at 0)")
    ax[0, 0].set_ylabel("net charge per depth bin")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "depth_profiles.png"), dpi=120)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train_npz", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--q_th", type=float, default=0.1, help="birth-energy threshold of the modelled pairs (MeV)")
    t.add_argument("--rate_epochs", type=int, default=30)
    t.add_argument("--flow_epochs", type=int, default=150)
    t.add_argument("--max_segments", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--test_npz", required=True)
    e.add_argument("--n", type=int, default=16, help="bins per cell side for the aggregated images")
    e.add_argument("--K", type=int, default=4, help="pair samples per transit")
    e.add_argument("--ode_steps", type=int, default=50)
    e.add_argument("--n_eval", type=int, default=1000000, help="transits evaluated (subsample)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--max_segments", type=int, default=None)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
