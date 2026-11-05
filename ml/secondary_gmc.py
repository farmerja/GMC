#!/usr/bin/env python3
"""
Relevance-based electron GMC with secondaries: 2D square cells, one material, one cell size.

Training data: ElectronTransport/generate_analog_batches.py --simulated_only --photons_irrelevant (gmc_segments.py
relevance scheme). Every cell transit of an electron is one segment. A secondary is RELEVANT when its subtree leaves
the cell; irrelevant subtrees are never simulated and their energy is dose of the segment that made them. Photons are
not transported: their energy leaves the problem (as in the analog MC, which banks them).

One transit, entry state -> outputs (all models share one cell frame: the cell is the unit square):
    1. joint classifier  -> exit class c (0 bottom, 1 right, 2 top, 3 left, 4 absorbed)
                            x n relevant daughters (0 .. NR, top class "NR or more")
                            x photon flag (does photon energy leave this transit)
    2. exit flow (c < 4) -> position along face s, angle from the outward normal psi, in-plane magnitude r (|w|),
                            energy loss fraction lambda = (E0 - E_out) / (E0 - E_cut)
    3. daughter flow     -> for j < n: birth point, direction (relative to the parent's w sign), energy
                            E_j = f_j * B_j, with B_0 = E0 - E_out and B_{j+1} = B_j - E_j (energy conserved)
    4. photon flow (flag)-> photon energy q = g * B_n
       dose = B_n - q (exactly what is left)
Two model sets: BOUNDARY (electron entering through a face, canonical frame: entering through the left face x = 0)
and INTERIOR (a relevant secondary at its birth point, in the cell frame). A relevant secondary is a biased sample
(its subtree is known to leave), so the interior models are trained on relevant secondaries only; an electron
entering a neighbour cell is a fresh boundary entry again. The daughter and photon flows are shared by both, with
a start-type flag.

Train (GMC venv, from the GMC root):
    python ml/secondary_gmc.py train --train_npz ../ElectronTransport/analog_cu_100um_1keV-10MeV_log_10M_42.npz \
        --out ml/models/secondary_gmc_cu_100um
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.cell_topology import (UNIT_SQUARE_VERTS, polygon_face_coordinate, polygon_face_point,  # noqa: E402
                              direction_to_face_frame_2d, direction_from_face_frame_2d, classes_from_status)
from ml.exit_classifier import ExitClassifier, ExitClassifierConfig, train_exit_classifier  # noqa: E402
from ml.model import get_model, get_flow_matching_path  # noqa: E402
from ml.utils import run_uniform_ode_inference  # noqa: E402

NF = 4                  # faces of the square
ABSORBED = 4
N_EXIT = 5              # 4 faces + absorbed
EPS = 1e-6
START_BOUNDARY, START_INTERIOR = 0.0, 1.0
SEG_COLS = None         # filled from the npz


# ====================================================================================================================
# Encodings
# ====================================================================================================================
def logit(p, eps=EPS):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Spec(SimpleNamespace):
    """Problem constants stored in every checkpoint: energy range, cell size, count cap."""

    def e_feat(self, E):
        """Energy feature in [-1, 1] (log scale between the cutoff and E_max)."""
        return 2.0 * (np.log10(np.maximum(E, 1e-300)) - self.logE_lo) / (self.logE_hi - self.logE_lo) - 1.0

    def n_classes(self):
        return N_EXIT * (self.NR + 1) * 2

    def encode(self, cls, n, flag):
        return (cls * (self.NR + 1) + np.minimum(n, self.NR)) * 2 + flag

    def decode(self, label):
        label = np.asarray(label)
        return label // (2 * (self.NR + 1)), (label // 2) % (self.NR + 1), label % 2


def dir_feats(u, v):
    """(cos phi, sin phi, 2 r - 1) of the in-plane direction; r = sqrt(u^2 + v^2) encodes |w|."""
    r = np.sqrt(u * u + v * v)
    phi = np.arctan2(v, u)
    return np.cos(phi), np.sin(phi), 2.0 * np.clip(r, 0.0, 1.0) - 1.0


def boundary_cond(spec, y, u, v, E):
    c, s, r = dir_feats(u, v)
    return np.stack([2.0 * y - 1.0, c, s, r, spec.e_feat(E)], 1).astype(np.float32)


def interior_cond(spec, x, y, u, v, E):
    c, s, r = dir_feats(u, v)
    return np.stack([2.0 * x - 1.0, 2.0 * y - 1.0, c, s, r, spec.e_feat(E)], 1).astype(np.float32)


def parent_feats(spec, start, x, y, u, v, E):
    """Shared parent description for the daughter / photon flows (boundary parents have x = 0)."""
    return np.concatenate([np.asarray(start, np.float32).reshape(-1, 1) * np.ones((len(E), 1), np.float32),
                           interior_cond(spec, x, y, u, v, E)], 1)


def exit_targets(cls, x1, y1, u1, v1, E0, E1, E_cut):
    """(s, psi, r, lambda) encoded for the exit flow (exits only)."""
    pts = np.stack([x1, y1], 1)
    s = polygon_face_coordinate(UNIT_SQUARE_VERTS, cls, pts)
    mu_n, mu_t = direction_to_face_frame_2d(UNIT_SQUARE_VERTS, cls, u1, v1)
    r = np.sqrt(u1 * u1 + v1 * v1)
    psi = np.arctan2(mu_t, np.maximum(mu_n, 1e-12))      # (-pi/2, pi/2): always outgoing
    lam = (E0 - E1) / np.maximum(E0 - E_cut, 1e-300)
    return np.stack([logit(s), np.arctanh(np.clip(psi / (0.5 * np.pi), -1 + EPS, 1 - EPS)), logit(r),
                     logit(lam, 1e-12)], 1)


def decode_exit(z, cls, E0, E_cut):
    s = sigmoid(z[:, 0])
    psi = 0.5 * np.pi * np.tanh(z[:, 1])
    r = sigmoid(z[:, 2])
    lam = sigmoid(z[:, 3])
    p = polygon_face_point(UNIT_SQUARE_VERTS, cls, s)
    u, v = direction_from_face_frame_2d(UNIT_SQUARE_VERTS, cls, r * np.cos(psi), r * np.sin(psi))
    E_out = np.maximum(E0 - lam * (E0 - E_cut), E_cut * (1 + 1e-9))
    return p[:, 0], p[:, 1], u, v, r, E_out


# ====================================================================================================================
# Square symmetry (D4) for data augmentation of interior rows
# ====================================================================================================================
def d4(k, x, y, u, v):
    """Element k (0..7) of the square's symmetry group: rotation by k % 4 quarter turns about the centre, then a
    reflection x -> 1 - x if k >= 4. Applied to points (x, y) and in-plane directions (u, v)."""
    x, y, u, v = x - 0.5, y - 0.5, u.copy(), v.copy()
    for _ in range(k % 4):
        x, y, u, v = -y, x, -v, u
    if k >= 4:
        x, u = -x, -u
    return x + 0.5, y + 0.5, u, v


def exit_class_from_point(x, y):
    """Nearest face of a point on the boundary (0 bottom, 1 right, 2 top, 3 left)."""
    d = np.stack([y, 1 - x, 1 - y, x], 1)
    return np.argmin(np.abs(d), 1)


# ====================================================================================================================
# Data
# ====================================================================================================================
def load_tables(path, NR=3, max_rows=None):
    """Segments + relevant daughters of a --simulated_only dataset -> dict of model tables (numpy).

    Positions are divided by the cell size; every segment (and its daughters) is mirrored to w0 >= 0 (z symmetry:
    w never enters the x-y transport, so only signs relative to the parent matter).
    """
    d = np.load(path, allow_pickle=False)
    if "segments" not in d.files or not bool(d.get("simulated_only", False)):
        raise ValueError(f"{path}: needs a generate_analog_batches.py --simulated_only dataset")
    cols = [str(c) for c in d["segment_columns"]]
    scol = [str(c) for c in d["secondary_columns"]]
    c, e = cols.index, scol.index
    seg, sec = d["segments"], d["secondaries"]
    L = float(d["box_dims"][0])
    E_cut = float(d["E_cutoff"])
    E_hi = float(d["inputs"][:, 4].max())
    spec = Spec(L=L, E_cut=E_cut, logE_lo=math.log10(E_cut), logE_hi=math.log10(max(E_hi, 10 * E_cut)) + 1e-9,
                NR=int(NR), material_density=float(np.atleast_1d(d["density"])[0]))
    status = seg[:, c("status")].astype(int)
    if np.any(status >= 6):
        bad = int((status >= 6).sum())
        print(f"[!] {bad} segments with status 6/7 (lost along z / split) dropped: not modelled at this cell size")
    keep = status <= 5
    if max_rows is not None:
        keep &= np.arange(len(seg)) < max_rows

    x0, y0 = seg[:, c("x0")] / L, seg[:, c("y0")] / L
    u0, v0, w0, E0 = seg[:, c("u0")], seg[:, c("v0")], seg[:, c("w0")], seg[:, c("E0")]
    x1, y1 = seg[:, c("x1")] / L, seg[:, c("y1")] / L
    u1, v1, E1 = seg[:, c("u1")], seg[:, c("v1")], seg[:, c("E1")]
    sgn = np.where(w0 < 0, -1.0, 1.0)
    cls = classes_from_status(status, invalid=-1)
    n_rel = seg[:, c("n_rel_e")].astype(int)
    E_out = np.where(cls == ABSORBED, 0.0, E1)
    E_rel = seg[:, c("E_rel_e")]
    q = np.maximum(E0 - E_out - E_rel - seg[:, c("E_dep_rel")], 0.0)  # photon energy leaving this transit
    flag = (q > 1e-12 * np.maximum(E0, 1e-300)).astype(int)
    boundary = (seg[:, c("generation")] == 0) & keep
    interior = (seg[:, c("generation")] > 0) & (seg[:, c("origin")] == 1) & keep
    resumed = int(((seg[:, c("origin")] == 2) & keep).sum())
    if resumed:
        print(f"[!] {resumed} resumed (post-split) segments ignored")

    T = {"spec": spec, "x0": x0, "y0": y0, "u0": u0, "v0": v0, "E0": E0, "x1": x1, "y1": y1, "u1": u1, "v1": v1,
         "E_out": E_out, "cls": cls, "n_rel": n_rel, "flag": flag, "q": q, "E_rel": E_rel,
         "dose": seg[:, c("E_dep_rel")], "boundary": boundary, "interior": interior, "sgn": sgn}

    # Daughters (secondary table rows are relevant electrons of simulated segments, in collision order)
    p = sec[:, e("segment")].astype(int)
    ok = keep[p]
    sec, p = sec[ok], p[ok]
    order = np.argsort(p, kind="stable")
    sec, p = sec[order], p[order]
    first = np.searchsorted(p, p, side="left")
    j = np.arange(len(p)) - first                     # index of the daughter within its parent segment
    Ed = sec[:, e("E")]
    cum = np.zeros(len(p))
    if len(p):
        cs = np.cumsum(Ed)
        cum = cs - Ed - (cs[first] - Ed[first])        # energy of earlier daughters of the same parent
    B = (E0[p] - E_out[p]) - cum                      # budget available to daughter j
    T["dau"] = {"parent": p, "j": j, "x": sec[:, e("x")] / L, "y": sec[:, e("y")] / L, "u": sec[:, e("u")],
                "v": sec[:, e("v")], "w": sec[:, e("w")] * sgn[p], "E": Ed, "B": B}
    print(f"Loaded {path}: {int(boundary.sum())} boundary transits, {int(interior.sum())} relevant-secondary "
          f"transits, {len(p)} relevant daughters; photon flag on {flag[keep].mean():.4f} of transits; "
          f"n_rel max {int(n_rel[keep].max())} (cap {NR})")
    return T


def _aug_interior(T, rows, k):
    """D4-transformed copies of interior segment rows: entry, exit and class."""
    x0, y0, u0, v0 = d4(k, T["x0"][rows], T["y0"][rows], T["u0"][rows], T["v0"][rows])
    x1, y1, u1, v1 = d4(k, T["x1"][rows], T["y1"][rows], T["u1"][rows], T["v1"][rows])
    cls = T["cls"][rows].copy()
    ex = cls < NF
    cls[ex] = exit_class_from_point(x1[ex], y1[ex])
    return x0, y0, u0, v0, x1, y1, u1, v1, cls


def classifier_data(T, kind, augment=True):
    spec = T["spec"]
    if kind == "boundary":
        r = np.flatnonzero(T["boundary"])
        cond = boundary_cond(spec, T["y0"][r], T["u0"][r], T["v0"][r], T["E0"][r])
        lab = spec.encode(T["cls"][r], T["n_rel"][r], T["flag"][r])
        return cond, lab
    r = np.flatnonzero(T["interior"])
    conds, labs = [], []
    for k in (range(8) if augment else [0]):
        x0, y0, u0, v0, _, _, _, _, cls = _aug_interior(T, r, k)
        conds.append(interior_cond(spec, x0, y0, u0, v0, T["E0"][r]))
        labs.append(spec.encode(cls, T["n_rel"][r], T["flag"][r]))
    return np.concatenate(conds), np.concatenate(labs)


def exit_flow_data(T, kind, augment=True):
    spec = T["spec"]
    rows = np.flatnonzero(T[kind] & (T["cls"] < NF))
    conds, tgts = [], []
    for k in ([0] if kind == "boundary" or not augment else range(8)):
        if kind == "boundary":
            x0, y0, u0, v0 = T["x0"][rows], T["y0"][rows], T["u0"][rows], T["v0"][rows]
            x1, y1, u1, v1, cls = T["x1"][rows], T["y1"][rows], T["u1"][rows], T["v1"][rows], T["cls"][rows]
            base = boundary_cond(spec, y0, u0, v0, T["E0"][rows])
        else:
            x0, y0, u0, v0, x1, y1, u1, v1, cls = _aug_interior(T, rows, k)
            base = interior_cond(spec, x0, y0, u0, v0, T["E0"][rows])
        extra = np.concatenate([np.eye(NF, dtype=np.float32)[cls],
                                (np.minimum(T["n_rel"][rows], spec.NR) / spec.NR).reshape(-1, 1),
                                T["flag"][rows].reshape(-1, 1)], 1).astype(np.float32)
        conds.append(np.concatenate([base, extra], 1))
        tgts.append(exit_targets(cls, x1, y1, u1, v1, T["E0"][rows], T["E_out"][rows], spec.E_cut))
    return np.concatenate(conds), np.concatenate(tgts).astype(np.float32)


def _parent_block(T, prow, k):
    """Parent features + class one-hot + n + lambda for parent segment rows prow under symmetry k (interior only)."""
    spec = T["spec"]
    is_int = T["interior"][prow]
    x0, y0, u0, v0 = T["x0"][prow].copy(), T["y0"][prow].copy(), T["u0"][prow].copy(), T["v0"][prow].copy()
    cls = T["cls"][prow].copy()
    if k:
        xa, ya, ua, va = d4(k, x0, y0, u0, v0)
        x0, y0, u0, v0 = np.where(is_int, xa, x0), np.where(is_int, ya, y0), np.where(is_int, ua, u0), \
            np.where(is_int, va, v0)
        x1, y1, _, _ = d4(k, T["x1"][prow], T["y1"][prow], T["u1"][prow], T["v1"][prow])
        ex = is_int & (cls < NF)
        cls[ex] = exit_class_from_point(x1[ex], y1[ex])
    E0 = T["E0"][prow]
    lam = (E0 - T["E_out"][prow]) / np.maximum(E0 - spec.E_cut, 1e-300)
    pf = parent_feats(spec, is_int.astype(np.float32), x0, y0, u0, v0, E0)
    blk = np.concatenate([pf, np.eye(N_EXIT, dtype=np.float32)[cls],
                          (np.minimum(T["n_rel"][prow], spec.NR) / spec.NR).reshape(-1, 1),
                          logit(lam, 1e-12).reshape(-1, 1) / 10.0], 1).astype(np.float32)
    return blk, is_int


def daughter_flow_data(T, augment=True):
    spec = T["spec"]
    D = T["dau"]
    p = D["parent"]
    conds, tgts = [], []
    for k in (range(8) if augment else [0]):
        blk, is_int = _parent_block(T, p, k)
        x, y, u, v = D["x"], D["y"], D["u"], D["v"]
        if k:
            xa, ya, ua, va = d4(k, x, y, u, v)
            x, y, u, v = np.where(is_int, xa, x), np.where(is_int, ya, y), np.where(is_int, ua, u), \
                np.where(is_int, va, v)
            if not is_int.any():
                continue
            sel = is_int                               # boundary parents only have the identity
        else:
            sel = np.ones(len(p), bool)
        extra = np.stack([D["j"] / spec.NR, spec.e_feat(D["B"])], 1).astype(np.float32)
        conds.append(np.concatenate([blk, extra], 1)[sel])
        f = D["E"] / np.maximum(D["B"], 1e-300)
        tgts.append(np.stack([logit(x), logit(y), u, v, D["w"], logit(f, 1e-9)], 1)[sel])
    return np.concatenate(conds), np.concatenate(tgts).astype(np.float32)


def photon_flow_data(T, augment=True):
    spec = T["spec"]
    rows = np.flatnonzero((T["boundary"] | T["interior"]) & (T["flag"] == 1))
    avail = T["E0"][rows] - T["E_out"][rows] - T["E_rel"][rows]
    g = T["q"][rows] / np.maximum(avail, 1e-300)
    conds, tgts = [], []
    for k in (range(8) if augment else [0]):
        blk, is_int = _parent_block(T, rows, k)
        sel = np.ones(len(rows), bool) if k == 0 else is_int
        if not sel.any():
            continue
        conds.append(np.concatenate([blk, spec.e_feat(avail).reshape(-1, 1).astype(np.float32)], 1)[sel])
        tgts.append(logit(g, 1e-9).reshape(-1, 1)[sel])
    return np.concatenate(conds), np.concatenate(tgts).astype(np.float32)


# ====================================================================================================================
# Flow training / sampling
# ====================================================================================================================
def train_flow(cond, target, device, hidden=256, depth=6, expansion=2, epochs=20, batch=8192, lr=1e-3,
               val_frac=0.05, seed=0, label="flow", max_steps=None):
    """Conditional flow matching (condOT path, ml/model.py network) on standardized targets."""
    g = torch.Generator().manual_seed(seed)
    n = len(cond)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    mu, sd = target.mean(0), target.std(0) + 1e-6
    X = torch.as_tensor(np.concatenate([cond, (target - mu) / sd], 1), dtype=torch.float32)
    tr, va = perm[n_val:], perm[:n_val]
    X = X.to(device)
    cd, td = cond.shape[1], target.shape[1]
    model = get_model(cd + td, cd, hidden_dim=hidden, depth=depth, expansion_factor=expansion).to(device)
    path = get_flow_matching_path(cond_dim=cd)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps_ep = max(1, len(tr) // batch)
    total = epochs * steps_ep if max_steps is None else min(epochs * steps_ep, max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)

    def loss_on(xb):
        x0 = torch.cat([xb[:, :cd], torch.randn(len(xb), td, device=device)], 1)
        t = torch.rand(len(xb), device=device)
        ps = path.sample(t=t, x_0=x0, x_1=xb)
        return ((model(ps.x_t, ps.t) - ps.dx_t)[:, cd:] ** 2).mean()

    @torch.no_grad()
    def val_loss():
        model.eval()
        torch.manual_seed(1234)                       # same noise every evaluation
        tot = sum(loss_on(X[va[i:i + 65536]]).item() * len(va[i:i + 65536]) for i in range(0, len(va), 65536))
        model.train()
        return tot / len(va)

    best, best_state, step, t0 = float("inf"), None, 0, time.time()
    hist = []
    for ep in range(epochs):
        p = tr[torch.randperm(len(tr), generator=g)].to(device)
        run = 0.0
        for i in range(steps_ep):
            loss = loss_on(X[p[i * batch:(i + 1) * batch]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step >= total:
                break
        vl = val_loss()
        hist.append((run / max(1, i + 1), vl))
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [{label}] epoch {ep + 1}/{epochs}: train {hist[-1][0]:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        if step >= total:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "mu": mu, "sd": sd, "cond_dim": cd, "target_dim": td, "history": hist,
            "arch": dict(hidden=hidden, depth=depth, expansion=expansion)}


@torch.no_grad()
def sample_flow(F, cond, device, steps=50, method="midpoint", gen=None, batch=262144):
    out = []
    for i in range(0, len(cond), batch):
        c = torch.as_tensor(cond[i:i + batch], dtype=torch.float32)
        z = torch.randn(len(c), F["target_dim"], generator=gen)
        x0 = torch.cat([c, z], 1).to(device)
        sol = run_uniform_ode_inference(F["model"], x0, device, method, steps, show_timing=False)
        out.append(sol[:, F["cond_dim"]:].double().numpy())
    z = np.concatenate(out) if out else np.zeros((0, F["target_dim"]))
    return z * F["sd"] + F["mu"]


# ====================================================================================================================
# Persistence
# ====================================================================================================================
def save_models(out_dir, spec, models):
    os.makedirs(out_dir, exist_ok=True)
    blob = {"spec": vars(spec)}
    for name, m in models.items():
        if m.get("kind") == "classifier":
            blob[name] = {"kind": "classifier", "cfg": vars(m["model"].cfg), "state": m["model"].state_dict()}
        else:
            blob[name] = {"kind": "flow", "state": m["model"].state_dict(), "mu": m["mu"], "sd": m["sd"],
                          "cond_dim": m["cond_dim"], "target_dim": m["target_dim"], "arch": m["arch"],
                          "history": m["history"]}
    torch.save(blob, os.path.join(out_dir, "secondary_gmc.pth"))


def load_models(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    spec = Spec(**blob["spec"])
    models = {}
    for name, b in blob.items():
        if name == "spec":
            continue
        if b["kind"] == "classifier":
            m = ExitClassifier(ExitClassifierConfig(**b["cfg"]))
            m.load_state_dict(b["state"])
            models[name] = {"kind": "classifier", "model": m.to(device).eval()}
        else:
            a = b["arch"]
            m = get_model(b["cond_dim"] + b["target_dim"], b["cond_dim"], hidden_dim=a["hidden"], depth=a["depth"],
                          expansion_factor=a["expansion"])
            m.load_state_dict(b["state"])
            models[name] = {"kind": "flow", "model": m.to(device).eval(), "mu": b["mu"], "sd": b["sd"],
                            "cond_dim": b["cond_dim"], "target_dim": b["target_dim"], "arch": a}
    return spec, models


# ====================================================================================================================
# Kernel: one cell transit for a batch of electrons
# ====================================================================================================================
class SecondaryGMC:
    """Samples cell transits. boundary(): canonical frame (entering through the left face). interior(): cell frame.

    Both return {"cls", "x", "y", "u", "v", "w", "E_out", "dose", "daughters"}; positions in the unit cell.
    daughters = {"parent", "x", "y", "u", "v", "w", "E"} (relevant secondaries to follow as interior particles).
    secondaries=False gives the primary-only GMC from the same models: daughters are not emitted and their energy is
    deposited in the cell (the n_rel / photon structure is still sampled, so this isolates the secondary transport).
    """

    def __init__(self, path, device="cpu", ode_steps=50, ode_method="midpoint", seed=0, secondaries=True):
        self.spec, self.m = load_models(path, device)
        self.device, self.steps, self.method = device, ode_steps, ode_method
        self.gen = torch.Generator().manual_seed(seed)                       # ODE noise (CPU)
        self.cgen = torch.Generator(device=device).manual_seed(seed + 7777)  # class sampling (model device)
        self.rng = np.random.default_rng(seed)
        self.secondaries = secondaries

    def _classify(self, name, cond):
        clf = self.m[name]["model"]
        lab = clf.sample(torch.as_tensor(cond, device=self.device), generator=self.cgen).cpu().numpy()
        return self.spec.decode(lab)

    def _flow(self, name, cond):
        return sample_flow(self.m[name], cond, self.device, self.steps, self.method, self.gen)

    def boundary(self, y, u, v, w, E):
        n = len(E)
        return self._transit("B", np.zeros(n), y, u, v, w, E, boundary_cond(self.spec, y, u, v, E))

    def interior(self, x, y, u, v, w, E):
        return self._transit("I", x, y, u, v, w, E, interior_cond(self.spec, x, y, u, v, E))

    def _transit(self, which, x0, y0, u0, v0, w0, E0, cond):
        spec = self.spec
        n = len(E0)
        cls, nr, flag = self._classify("clf_" + which, cond)
        out = {k: np.zeros(n) for k in ("x", "y", "u", "v", "w", "E_out", "dose")}
        out["cls"] = cls
        sgn = np.where(w0 < 0, -1.0, 1.0)
        ex = np.flatnonzero(cls < NF)
        if ex.size:
            extra = np.concatenate([np.eye(NF, dtype=np.float32)[cls[ex]], (nr[ex] / spec.NR).reshape(-1, 1),
                                    flag[ex].reshape(-1, 1)], 1).astype(np.float32)
            z = self._flow("exit_" + which, np.concatenate([cond[ex], extra], 1))
            x1, y1, u1, v1, r1, E1 = decode_exit(z, cls[ex], E0[ex], spec.E_cut)
            out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex], out["E_out"][ex] = x1, y1, u1, v1, E1
            out["w"][ex] = self.rng.choice([-1.0, 1.0], ex.size) * np.sqrt(np.clip(1 - r1 * r1, 0, 1))
        budget = E0 - out["E_out"]

        # Daughters, sequentially (energy shares of what is left)
        start = START_BOUNDARY if which == "B" else START_INTERIOR
        lam = budget / np.maximum(E0 - spec.E_cut, 1e-300)
        pblk = np.concatenate([parent_feats(spec, start, x0, y0, u0, v0, E0), np.eye(N_EXIT, dtype=np.float32)[cls],
                               (nr / spec.NR).reshape(-1, 1), (logit(lam, 1e-12) / 10.0).reshape(-1, 1)],
                              1).astype(np.float32)
        dau = {k: [] for k in ("parent", "x", "y", "u", "v", "w", "E")}
        local_dep = np.zeros(n)
        for j in range(spec.NR):
            rows = np.flatnonzero((nr > j) & (budget > spec.E_cut))
            if rows.size == 0:
                break
            c = np.concatenate([pblk[rows], np.stack([np.full(rows.size, j / spec.NR), spec.e_feat(budget[rows])],
                                                     1).astype(np.float32)], 1)
            z = self._flow("daughter", c)
            f = sigmoid(z[:, 5])
            Ed = f * budget[rows]
            budget[rows] -= Ed
            dx, dy = sigmoid(z[:, 0]), sigmoid(z[:, 1])
            du, dv, dw = z[:, 2], z[:, 3], z[:, 4]
            nrm = np.sqrt(du * du + dv * dv + dw * dw) + 1e-12
            keep = Ed > spec.E_cut
            if not self.secondaries:
                local_dep[rows] += Ed                  # primary-only: the daughter's energy stays in the cell
                continue
            local_dep[rows[~keep]] += Ed[~keep]
            r = rows[keep]
            for k, val in (("parent", r), ("x", dx[keep]), ("y", dy[keep]), ("u", (du / nrm)[keep]),
                           ("v", (dv / nrm)[keep]), ("w", (dw / nrm)[keep] * sgn[r]), ("E", Ed[keep])):
                dau[k].append(val)
        out["daughters"] = {k: (np.concatenate(v) if v else np.zeros(0, int if k == "parent" else float))
                            for k, v in dau.items()}

        # Photon energy leaving (fraction of what is left after the daughters)
        ph = np.flatnonzero(flag == 1)
        q = np.zeros(n)
        if ph.size:
            # Trained as a share of E0 - E_out - sum(E_daughters): the budget left after the daughters (in
            # primary-only mode too, where those daughters' energy is then added to the dose instead of emitted)
            avail = np.maximum(budget[ph], 0.0)
            c = np.concatenate([pblk[ph], spec.e_feat(np.maximum(avail, 1e-300)).reshape(-1, 1).astype(np.float32)], 1)
            g = sigmoid(self._flow("photon", c)[:, 0])
            q[ph] = g * avail
        out["dose"] = np.maximum(budget + local_dep - q, 0.0)
        out["photon_E"] = q
        return out


# ====================================================================================================================
# CLI
# ====================================================================================================================
def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = load_tables(args.train_npz, NR=args.NR, max_rows=args.max_rows)
    spec = T["spec"]
    models, info = {}, {}
    only = set(args.only.split(",")) if args.only else None
    prev = None
    if only is not None and os.path.exists(os.path.join(args.out, "secondary_gmc.pth")):
        _, prev = load_models(os.path.join(args.out, "secondary_gmc.pth"), dev)

    def want(name):
        return only is None or name in only

    for which, kind in (("B", "boundary"), ("I", "interior")):
        name = "clf_" + which
        if want(name):
            cond, lab = classifier_data(T, kind)
            print(f"\n== {name}: {len(lab)} rows, {spec.n_classes()} classes, "
                  f"occupied {len(np.unique(lab))}")
            topo = SimpleNamespace(num_classes=spec.n_classes())
            m, hist, _ = train_exit_classifier(torch.as_tensor(cond), torch.as_tensor(lab), topo, device=dev,
                                               epochs=args.clf_epochs, batch_size=16384, lr=2e-3, prior_bias=True,
                                               hidden_dim=256, depth=4, log_every=5)
            models[name] = {"kind": "classifier", "model": m}
            info[name] = {"rows": int(len(lab)), "val_nll": hist["val"][-1]}
        name = "exit_" + which
        if want(name):
            cond, tgt = exit_flow_data(T, kind)
            print(f"\n== {name}: {len(tgt)} rows, cond {cond.shape[1]}, target {tgt.shape[1]}")
            models[name] = {"kind": "flow", **train_flow(cond, tgt, dev, epochs=args.flow_epochs, label=name,
                                                         max_steps=args.max_steps)}
            info[name] = {"rows": int(len(tgt))}
    if want("daughter"):
        cond, tgt = daughter_flow_data(T)
        print(f"\n== daughter: {len(tgt)} rows (with D4 copies of interior parents), cond {cond.shape[1]}")
        models["daughter"] = {"kind": "flow", **train_flow(cond, tgt, dev, epochs=args.small_epochs, batch=4096,
                                                           label="daughter", max_steps=args.max_steps)}
        info["daughter"] = {"rows": int(len(tgt))}
    if want("photon"):
        cond, tgt = photon_flow_data(T)
        print(f"\n== photon: {len(tgt)} rows, cond {cond.shape[1]}")
        models["photon"] = {"kind": "flow", **train_flow(cond, tgt, dev, epochs=args.small_epochs, batch=4096,
                                                         label="photon", hidden=128, depth=4,
                                                         max_steps=args.max_steps)}
        info["photon"] = {"rows": int(len(tgt))}
    if prev is not None:
        for k, v in prev.items():
            models.setdefault(k, v)
    save_models(args.out, spec, models)
    with open(os.path.join(args.out, "train_info.json"), "w") as f:
        json.dump({"spec": vars(spec), "train_npz": args.train_npz, "models": info}, f, indent=2)
    print(f"\nSaved {os.path.join(args.out, 'secondary_gmc.pth')}")


MATERIAL_BY_DENSITY = {8.96: "Cu", 2.699: "Al"}  # g/cm^3 as stored by ElectronTransport/NIST_EPICS_convert.py


def describe(spec):
    """'100 µm Cu, 1 keV – 10 MeV' from a Spec or its dict (material named from its density)."""
    if isinstance(spec, dict):
        spec = SimpleNamespace(**spec)
    rho = float(getattr(spec, "material_density", float("nan")))
    mat = next((n for d, n in MATERIAL_BY_DENSITY.items() if abs(d - rho) < 1e-3), f"ρ = {rho:g} g/cm³")
    lo, hi = 10 ** spec.logE_lo, 10 ** spec.logE_hi
    fmt = lambda E: f"{E * 1e3:.3g} keV" if E < 1 else f"{E:.3g} MeV"  # noqa: E731
    return f"{spec.L * 1e4:g} µm {mat}, {fmt(lo)} – {fmt(hi)}"


def cmd_plot_history(args):
    """Train / validation loss per epoch of every flow in a checkpoint (classifier histories are not stored; their
    final validation NLL is in train_info.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blob = torch.load(args.model, map_location="cpu", weights_only=False)
    flows = [(k, v) for k, v in blob.items() if k != "spec" and v.get("kind") == "flow" and v.get("history")]
    titles = {"exit_B": "Exit flow, entering electrons", "exit_I": "Exit flow, relevant secondaries",
              "daughter": "Daughter flow", "photon": "Photon-energy flow"}
    fig, axs = plt.subplots(1, len(flows), figsize=(4.2 * len(flows), 3.6), squeeze=False)
    for ax, (k, v) in zip(axs[0], flows):
        h = np.array(v["history"])
        ep = np.arange(1, len(h) + 1)
        ax.plot(ep, h[:, 0], color="#2a78d6", lw=2, label="train")
        ax.plot(ep, h[:, 1], color="#eb6834", lw=2, ls="--", marker="o", ms=3, label="validation")
        ax.set_title(titles.get(k, k), fontsize=10, fontweight="bold")
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e4e3df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axs[0, 0].set_ylabel("flow-matching loss (standardized targets)")
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"Secondary GMC training, {describe(blob['spec'])}", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("plot_history")
    h.add_argument("--model", required=True)
    h.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train_npz", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--NR", type=int, default=3, help="relevant-daughter count cap (top class = NR or more)")
    t.add_argument("--clf_epochs", type=int, default=15)
    t.add_argument("--flow_epochs", type=int, default=20)
    t.add_argument("--small_epochs", type=int, default=60, help="epochs for the daughter / photon flows")
    t.add_argument("--max_steps", type=int, default=None, help="cap on optimizer steps per flow")
    t.add_argument("--max_rows", type=int, default=None, help="use only the first rows (smoke tests)")
    t.add_argument("--only", default=None, help="comma list of models to (re)train; others kept from --out")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "plot_history":
        cmd_plot_history(args)


if __name__ == "__main__":
    main()
