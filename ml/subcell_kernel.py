#!/usr/bin/env python3
"""
SCR-GMC sub-cell deposition kernel: a conditional rational-quadratic spline coupling flow p(x, y | transit).

The GMC charges each transit's dose (E_dep_rel) to the whole cell. This kernel learns WHERE inside the cell that dose
goes, conditioned on what the GMC knows about the transit: entry state, sampled exit (class, point, direction,
energy), dose and relevant-daughter count. After a run, every logged transit is tallied onto any finer mesh by
density quadrature: dose x integral of p over each fine cell.

Model (unit-square reference frame, so it carries over to quads / triangles through their reference maps):
    base       uniform on [0, 1]^2  (bounded domain: no logit Jacobians at the faces)
    layers     coupling layers alternating the active coordinate; each one is a monotone RQ spline on [0, 1]
               (Durkan et al. 2019) whose knots come from an MLP of (Fourier features of the passive coordinate,
               context embedding). Sampling and density are one pass each (no ODE).
    context    MLP embedding of the transit features (cond_features), computed once per transit.
Training: maximum likelihood on the deposit points, weighted by deposited energy. The data are 64 x 64 energy images
per segment (ElectronTransport/subcell_deposits.py); every non-empty bin is a point with weight E, dequantized
uniformly inside its bin. Points are drawn proportional to their weight (importance sampling, plain mean NLL):
    --seg_weight dose  weight = E           (every MeV counts the same: matches the dose error of a tally)
    --seg_weight unit  weight = E / E_dep   (every transit counts the same)
Symmetry: boundary entries are canonical (left face), so they are augmented with the y-reflection; interior starts
(relevant secondaries) with the full D4 group of the square.

Usage (GMC venv, from the GMC root):
    python ml/subcell_kernel.py train --train_npz ../ElectronTransport/analog_cu_100um_1keV-10MeV_log_10M_42.npz \
        --out ml/models/subcell_kernel_cu_100um
    python ml/subcell_kernel.py eval --model ml/models/subcell_kernel_cu_100um \
        --test_npz ../ElectronTransport/analog_cu_100um_1keV-10MeV_log_2M_7.npz
Each <data>.npz needs its <data>_subcell.npz (generate_analog_batches.py --subcell_bins 64 --simulated_only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from ml.cell_topology import classes_from_status  # noqa: E402

N_EXIT = 5            # 4 faces (0 bottom, 1 right, 2 top, 3 left) + absorbed
ABSORBED = 4
NR = 3                # relevant-daughter count one-hot: 0, 1, 2, NR or more
E_BANDS = (1e-3, 1e-2, 1e-1, 1.0, 10.0 + 1e-9, 100.0)
BAND_NAMES = ("1-10 keV", "10-100 keV", "0.1-1 MeV", "1-10 MeV", "10-100 MeV")
NB = len(BAND_NAMES)
# per-transit raw columns (float32), see load_kernel_data
RAW = ("start", "x0", "y0", "u0", "v0", "E0", "cls", "x1", "y1", "u1", "v1", "E_out", "dose", "n_rel", "flag")


# ====================================================================================================================
# Rational-quadratic spline on [0, 1]
# ====================================================================================================================
MIN_W, MIN_H, MIN_D = 1e-3, 1e-3, 1e-3
D_OFF = math.log(math.expm1(1.0 - MIN_D))      # raw 0 -> derivative 1 (identity spline at zero init)


_TRI = {}


def _cumsum(w):
    """Inclusive cumsum over the (short) last dim as a matmul with a triangular ones matrix: a GPU scan over an innermost
    dim of 32 is ~20x slower and was ~30% of the sampling time. w is split into a TF32-exact high part (low 13 mantissa
    bits zeroed) and the remainder, so the sum stays fp32-accurate (~1e-7) even with TF32 matmuls enabled. Inference
    only: with gradients on (training) the plain cumsum is kept, since the matmul's backward would carry TF32 error."""
    if w.dtype != torch.float32 or torch.is_grad_enabled():
        return torch.cumsum(w, -1)
    K = w.shape[-1]
    key = (K, w.device)
    tri = _TRI.get(key)
    if tri is None:
        tri = _TRI[key] = torch.triu(torch.ones(K, K, device=w.device, dtype=torch.float32))
    hi = (w.detach().contiguous().view(torch.int32) & -8192).view(torch.float32)
    return hi @ tri + (w - hi) @ tri


def _knots(u, min_size):
    K = u.shape[-1]
    w = min_size + (1.0 - min_size * K) * torch.softmax(u, -1)
    c = _cumsum(w)
    c = torch.cat([torch.zeros_like(c[..., :1]), c[..., :-1], torch.ones_like(c[..., :1])], -1)
    return w, c


def rq_spline(x, uw, uh, ud, inverse=False, logdet=True):
    """Monotone RQ spline [0, 1] -> [0, 1] with K bins; uw, uh (..., K), ud (..., K + 1) unconstrained.
    Returns (y, log|dy/dx|) of the forward map, or (x, -log|dy/dx|) of the inverse; logdet=False returns None for the
    log-determinant (sampling does not need it)."""
    w, cw = _knots(uw, MIN_W)
    h, ch = _knots(uh, MIN_H)
    d = MIN_D + F.softplus(ud + D_OFF)
    edges = ch if inverse else cw
    k = torch.searchsorted(edges[..., 1:-1].contiguous(), x.unsqueeze(-1).contiguous())
    g = lambda a, i=k: a.gather(-1, i).squeeze(-1)  # noqa: E731
    xk, wk, yk, hk, dk, dk1 = g(cw), g(w), g(ch), g(h), g(d), g(d, k + 1)
    s = hk / wk
    if not inverse:
        th = ((x - xk) / wk).clamp(0.0, 1.0)
        t1 = th * (1.0 - th)
        den = s + (dk1 + dk - 2.0 * s) * t1
        y = yk + hk * (s * th * th + dk * t1) / den
        dydx = s * s * (dk1 * th * th + 2.0 * s * t1 + dk * (1.0 - th) ** 2) / (den * den)
        return y.clamp(0.0, 1.0), torch.log(dydx)
    dy = x - yk
    a = hk * (s - dk) + dy * (dk1 + dk - 2.0 * s)
    b = hk * dk - dy * (dk1 + dk - 2.0 * s)
    c = -s * dy
    th = (2.0 * c / (-b - torch.sqrt((b * b - 4.0 * a * c).clamp_min(0.0)))).clamp(0.0, 1.0)
    if not logdet:
        return (xk + th * wk).clamp(0.0, 1.0), None
    t1 = th * (1.0 - th)
    den = s + (dk1 + dk - 2.0 * s) * t1
    dydx = s * s * (dk1 * th * th + 2.0 * s * t1 + dk * (1.0 - th) ** 2) / (den * den)
    return (xk + th * wk).clamp(0.0, 1.0), -torch.log(dydx)


# ====================================================================================================================
# Flow
# ====================================================================================================================
def fourier(t, n_freq):
    """[t, sin(2^k pi t), cos(2^k pi t)] for k < n_freq (t in [0, 1])."""
    f = (2.0 ** torch.arange(n_freq, device=t.device, dtype=t.dtype)) * math.pi
    a = t.unsqueeze(-1) * f
    return torch.cat([2.0 * t.unsqueeze(-1) - 1.0, torch.sin(a), torch.cos(a)], -1)


class Coupling(nn.Module):
    """Spline on coordinate `active`, knots from MLP(fourier(passive), context). The context enters the first layer
    additively, so it can be computed once per transit and broadcast over many query points."""

    def __init__(self, active, ctx_dim, hidden, n_bins, n_freq, depth=2):
        super().__init__()
        self.active, self.K, self.n_freq = active, n_bins, n_freq
        self.lin_x = nn.Linear(1 + 2 * n_freq, hidden)
        self.lin_h = nn.Linear(ctx_dim, hidden, bias=False)
        self.body = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, 3 * n_bins + 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def params(self, xp, h):
        z = F.silu(self.lin_x(fourier(xp, self.n_freq)) + self.lin_h(h))
        for lin in self.body:
            z = z + F.silu(lin(z))
        p = self.out(z)
        K = self.K
        return p[..., :K], p[..., K:2 * K], p[..., 2 * K:]

    def forward(self, x, h, inverse=False, logdet=True):
        a = self.active
        xa, xp = x[..., a], x[..., 1 - a]
        ya, ld = rq_spline(xa, *self.params(xp, h), inverse=inverse, logdet=logdet)
        y = torch.stack([ya, xp] if a == 0 else [xp, ya], -1)
        return y, ld


_SOBOL = {}


def base_points(rows, n, device, gen=None, method="random"):
    """(rows, n, 2) base points on [0, 1]^2, independent between rows.
    random  i.i.d. uniform
    jitter  jittered m x m grid (n = m^2): one uniform point in each grid cell
    sobol   the first n points of the 2D Sobol sequence (a (0, m, 2)-net when n = 2^m: every dyadic box of area 1/n,
            of any aspect ratio, holds exactly one point) with an independent random digital shift (XOR of the 32-bit
            coordinates) per row, plus uniform jitter below 2^-32. The digital shift keeps the net property and makes
            every point uniform on [0, 1]^2, so the tally stays unbiased."""
    if method == "random":
        return torch.rand(rows, n, 2, device=device, generator=gen)
    if method == "jitter":
        m = int(round(math.sqrt(n)))
        if m * m != n:
            raise ValueError(f"jitter needs a square number of samples, got {n}")
        g = torch.arange(m, device=device, dtype=torch.float32)
        ij = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(1, n, 2)
        return (ij + torch.rand(rows, n, 2, device=device, generator=gen)) / m
    if method == "sobol":
        key = (n, device)
        pts = _SOBOL.get(key)
        if pts is None:
            eng = torch.quasirandom.SobolEngine(2, scramble=False)
            pts = _SOBOL[key] = (eng.draw(n, dtype=torch.float64) * 2.0 ** 32).long().to(device)  # exact integers
        shift = torch.randint(0, 2 ** 32, (rows, 1, 2), device=device, generator=gen, dtype=torch.long)
        u = torch.bitwise_xor(pts.unsqueeze(0), shift).to(torch.float64)
        u = (u + torch.rand(rows, n, 2, device=device, generator=gen, dtype=torch.float64)) / 2.0 ** 32
        return u.float().clamp(0.0, 1.0 - 2 ** -24)
    raise ValueError(method)


class SubcellFlow(nn.Module):
    def __init__(self, cond_dim, ctx_dim=256, hidden=256, n_layers=8, n_bins=32, n_freq=6, depth=2):
        super().__init__()
        self.arch = dict(cond_dim=cond_dim, ctx_dim=ctx_dim, hidden=hidden, n_layers=n_layers, n_bins=n_bins,
                         n_freq=n_freq, depth=depth)
        self.enc = nn.Sequential(nn.Linear(cond_dim, ctx_dim), nn.SiLU(), nn.Linear(ctx_dim, ctx_dim), nn.SiLU(),
                                 nn.Linear(ctx_dim, ctx_dim), nn.SiLU(), nn.Linear(ctx_dim, ctx_dim))
        self.layers = nn.ModuleList([Coupling(i % 2, ctx_dim, hidden, n_bins, n_freq, depth)
                                     for i in range(n_layers)])

    def context(self, cond):
        return self.enc(cond)

    def log_prob(self, x, h):
        """x (..., 2) in [0, 1]^2, h (..., ctx) broadcastable: log density on the unit square."""
        lp = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for L in self.layers:
            x, ld = L(x, h)
            lp = lp + ld
        return lp

    @torch.no_grad()
    def sample(self, h, n, gen=None, base="random"):
        """n points per context row: (N, n, 2). base: how the n base points per row are placed on [0, 1]^2
        (base_points): 'random' (independent), 'jitter' (one point per cell of an m x m grid, n = m^2) or 'sobol'
        (digitally shifted Sobol net). All three are unbiased; the stratified ones put exactly one point in each
        1/n of the transit's dose (the flow maps base area to probability), which cuts the tally noise."""
        z = base_points(h.shape[0], n, h.device, gen, base).to(h.dtype)
        hh = h.unsqueeze(1)
        for L in reversed(self.layers):
            z, _ = L(z, hh, inverse=True, logdet=False)
        return z


# ====================================================================================================================
# Transit features and symmetry
# ====================================================================================================================
class Spec:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def e_feat(self, E):
        return 2.0 * (torch.log10(E.clamp_min(1e-300)) - self.logE_lo) / (self.logE_hi - self.logE_lo) - 1.0


def _dir(u, v):
    r = torch.sqrt(u * u + v * v)
    phi = torch.atan2(v, u)
    return torch.cos(phi), torch.sin(phi), 2.0 * r.clamp(0.0, 1.0) - 1.0


def _logfrac(f):
    return torch.log10(f.clamp(1e-6, 1.0)) / 3.0 + 1.0     # 1e-6 .. 1 -> -1 .. 1


def _interp(x, xp, fp):
    """np.interp for torch (xp increasing; clamped at the ends)."""
    xp = torch.as_tensor(xp, dtype=x.dtype, device=x.device)
    fp = torch.as_tensor(fp, dtype=x.dtype, device=x.device)
    k = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    t = ((x - xp[k - 1]) / (xp[k] - xp[k - 1])).clamp(0.0, 1.0)
    return fp[k - 1] + t * (fp[k] - fp[k - 1])


def size_features(spec, L, E0):
    """(log L between the training bounds, log10(L / R_CSDA(E0))) as secondary_gmc.size_feats, in torch."""
    ll = torch.log10(L)
    lf = 2.0 * (ll - spec.logL_lo) / (spec.logL_hi - spec.logL_lo) - 1.0
    lr = _interp(torch.log10(E0.clamp_min(1e-300)), spec.csda_logE, spec.csda_logR)
    return torch.stack([lf, ((ll - lr - 1.0) / 4.0).clamp(-2.0, 2.0)], -1)


def cond_features(spec, R):
    """Transit features from raw per-transit tensors R (dict of RAW columns, positions in cell units [0, 1]).
    A size-conditioned spec (size_cond) also needs R["L"], the cell side in cm."""
    E0, Eo, dose = R["E0"], R["E_out"], R["dose"]
    cls = R["cls"].long()
    ex = (cls != ABSORBED).float()
    c0, s0, r0 = _dir(R["u0"], R["v0"])
    c1, s1, r1 = _dir(R["u1"], R["v1"])
    n_rel = R["n_rel"].long().clamp(0, NR)
    cols = [R["start"], 2 * R["x0"] - 1, 2 * R["y0"] - 1, c0, s0, r0, spec.e_feat(E0)]
    oh = F.one_hot(cls, N_EXIT).float()
    cols2 = [ex * (2 * R["x1"] - 1), ex * (2 * R["y1"] - 1), ex * c1, ex * s1, ex * r1,
             ex * spec.e_feat(Eo.clamp_min(spec.E_cut)), _logfrac((E0 - Eo) / E0), _logfrac(dose / E0)]
    feats = [torch.stack(cols, -1), oh, torch.stack(cols2, -1), F.one_hot(n_rel, NR + 1).float(),
             R["flag"].unsqueeze(-1)]
    n_ph = getattr(spec, "phase_n", 0)
    if n_ph:
        # sub-bin phase of the entry / exit points on the TRAINING grid (diagnostic: lets the model reproduce the
        # dequantized bin boxes, whose edges jump as a point crosses a bin edge). Tangential coordinates only:
        # the face-normal coordinate of a face point sits on a bin edge.
        ph = lambda a: 2.0 * torch.frac(a * n_ph) - 1.0  # noqa: E731
        interior = R["start"]
        tan_x = ex * ((cls == 0) | (cls == 2)).float()
        tan_y = ex * ((cls == 1) | (cls == 3)).float()
        feats.append(torch.stack([interior * ph(R["x0"]), ph(R["y0"]), tan_x * ph(R["x1"]), tan_y * ph(R["y1"])],
                                 -1))
    if getattr(spec, "size_cond", False):
        feats.append(size_features(spec, R["L"], E0.float()).to(feats[0].dtype))
    return torch.cat(feats, -1)


COND_DIM = 7 + N_EXIT + 8 + (NR + 1) + 1


def cond_dim(spec):
    return COND_DIM + (4 if getattr(spec, "phase_n", 0) else 0) + (2 if getattr(spec, "size_cond", False) else 0)


def d4_points(k, x, y):
    """Square symmetry k (0..7): k % 4 quarter turns about the centre, then x -> 1 - x if k >= 4 (as secondary_gmc.d4)."""
    x, y = x - 0.5, y - 0.5
    q = k % 4
    for r in (1, 2, 3):
        m = q >= r
        x, y = torch.where(m, -y, x), torch.where(m, x, y)
    x = torch.where(k >= 4, -x, x)
    return x + 0.5, y + 0.5


def d4_dirs(k, u, v):
    q = k % 4
    for r in (1, 2, 3):
        m = q >= r
        u, v = torch.where(m, -v, u), torch.where(m, u, v)
    return torch.where(k >= 4, -u, u), v


def d4_face(k, cls):
    f = torch.where(cls < 4, (cls + k % 4) % 4, cls)
    return torch.where((k >= 4) & (cls < 4), (4 - f) % 4, f)


def apply_d4(k, R):
    out = dict(R)
    out["x0"], out["y0"] = d4_points(k, R["x0"], R["y0"])
    out["u0"], out["v0"] = d4_dirs(k, R["u0"], R["v0"])
    out["x1"], out["y1"] = d4_points(k, R["x1"], R["y1"])
    out["u1"], out["v1"] = d4_dirs(k, R["u1"], R["v1"])
    out["cls"] = d4_face(k, R["cls"].long()).to(R["cls"].dtype)
    return out


def random_sym(start, gen=None):
    """Boundary rows: identity or the y-reflection (k = 6); interior rows: any D4 element."""
    n = len(start)
    k_int = torch.randint(0, 8, (n,), device=start.device, generator=gen)
    k_bnd = 6 * torch.randint(0, 2, (n,), device=start.device, generator=gen)
    return torch.where(start > 0.5, k_int, k_bnd)


# ====================================================================================================================
# Data
# ====================================================================================================================
def load_kernel_data(seg_npz, sub_npz=None, check_sha=True, max_segments=None, entries=True, return_rows=False):
    """Simulated-only segment table + its sub-cell images -> per-transit raw columns and per-point entries.

    Kept transits: status 1-5 (exit or absorbed), boundary primaries (origin 0) and relevant secondaries at birth
    (origin 1), E_dep_rel > 0 (a transit without dose needs no kernel). Entries: (seg = kept-transit row, bin, E).
    entries=False skips the image entries (returns None for them); return_rows=True also returns the segment-table
    row of every kept transit."""
    sub_npz = sub_npz or seg_npz[:-4] + "_subcell.npz"
    D = np.load(seg_npz, allow_pickle=False)
    S = np.load(sub_npz, allow_pickle=False)
    seg = D["segments"]
    if int(S["n_segments"]) != len(seg):
        raise ValueError(f"{sub_npz} was built for {int(S['n_segments'])} segments, {seg_npz} has {len(seg)}")
    if check_sha:
        sha = hashlib.sha1(np.ascontiguousarray(seg).tobytes()).hexdigest()
        if sha != str(S["segments_sha1"]):
            raise ValueError(f"{sub_npz} does not belong to {seg_npz} (segments sha1 mismatch)")
    if str(S["scheme"]) != "rel":
        raise ValueError("the kernel is trained on scheme 'rel' images (sum = E_dep_rel)")
    cols = [str(c) for c in D["segment_columns"]]
    c = cols.index
    sweep = "seg_L" in D.files                       # generate_size_sweep.py: cell side and grid per segment
    Lseg = D["seg_L"] if sweep else np.full(len(seg), float(D["box_dims"][0]))
    E_cut = float(D["E_cutoff"])
    E_hi = float(seg[:, c("E0")].max())
    spec = Spec(L=float("nan") if sweep else float(Lseg[0]), E_cut=E_cut, logE_lo=math.log10(E_cut),
                logE_hi=math.log10(max(E_hi, 10 * E_cut)) + 1e-9, n_sub=0 if sweep else int(S["subcell_bins"]),
                density=float(np.atleast_1d(D["density"])[0]))
    if sweep:
        lo, hi = (float(v) for v in D["L_range"])
        spec.size_cond, spec.logL_lo, spec.logL_hi = True, math.log10(lo), math.log10(hi)
        spec.csda_logE = np.log10(D["csda_E"][1:]).tolist()
        spec.csda_logR = np.log10(D["csda_R"][1:]).tolist()
    status = seg[:, c("status")].astype(int)
    cls = classes_from_status(status, invalid=-1)
    origin = seg[:, c("origin")].astype(int)
    dose = seg[:, c("E_dep_rel")]
    keep = (cls >= 0) & (origin <= 1) & (dose > 0)
    if max_segments is not None:
        keep &= np.arange(len(seg)) < max_segments
    rows = np.flatnonzero(keep)
    L = Lseg[rows]
    E0 = seg[rows, c("E0")]
    E1 = seg[rows, c("E1")]
    cl = cls[rows]
    E_out = np.where(cl == ABSORBED, 0.0, E1)
    q = np.maximum(E0 - E_out - seg[rows, c("E_rel_e")] - dose[rows], 0.0)
    R = {"start": seg[rows, c("start_type")], "x0": seg[rows, c("x0")] / L, "y0": seg[rows, c("y0")] / L,
         "u0": seg[rows, c("u0")], "v0": seg[rows, c("v0")], "E0": E0, "cls": cl.astype(np.float64),
         "x1": seg[rows, c("x1")] / L, "y1": seg[rows, c("y1")] / L, "u1": seg[rows, c("u1")],
         "v1": seg[rows, c("v1")], "E_out": E_out, "dose": dose[rows], "n_rel": seg[rows, c("n_rel_e")],
         "flag": (q > 1e-12 * E0).astype(np.float64)}
    if sweep:
        R["L"] = L
    R = {k: v.astype(np.float32) for k, v in R.items()}
    R["E0"] = E0                                      # float64 for exact dose fractions on the host
    R["dose"] = dose[rows]
    del seg
    if not entries:
        print(f"Loaded {os.path.basename(seg_npz)}: {len(rows)} transits (no image entries)", flush=True)
        return (spec, R, None, rows) if return_rows else (spec, R, None)
    new = np.full(len(keep), -1, dtype=np.int64)
    new[rows] = np.arange(len(rows))
    s = S["sub_seg"]
    ns = new[s]
    ok = ns >= 0
    ent = {"seg": ns[ok].astype(np.int32),
           "bin": S["sub_bin"][ok].astype(np.int16 if 0 < spec.n_sub <= 128 else np.int32), "E": S["sub_E"][ok]}
    if sweep:
        ent["n"] = S["seg_n"][s[ok]].astype(np.int16)  # grid size of every entry's segment
    if "sub_mx" in S.files:  # in-bin mean offset / spread of the deposits (bin widths)
        ent["mom"] = np.stack([S[k][ok] for k in ("sub_mx", "sub_my", "sub_sx", "sub_sy")], 1).astype(np.float16)
    n_b = int((R["start"] < 0.5).sum())
    grid = (f"{int(ent['n'].min())}-{int(ent['n'].max())} bins per side" if sweep
            else f"{spec.n_sub}x{spec.n_sub} bins")
    size = f"{L.min() * 1e4:.0f}-{L.max() * 1e4:.0f} um" if sweep else f"{L[0] * 1e4:.0f} um"
    print(f"Loaded {os.path.basename(seg_npz)}: {len(rows)} transits ({n_b} boundary, {len(rows) - n_b} interior), "
          f"{len(ent['seg'])} image entries ({len(ent['seg']) / max(1, len(rows)):.1f} per transit), "
          f"{grid}{' + in-bin moments' if 'mom' in ent else ''}, L = {size}", flush=True)
    return (spec, R, ent, rows) if return_rows else (spec, R, ent)


def to_device(R, device):
    return {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device) for k, v in R.items()}


def beta_offsets(m, sd, min_spread):
    """In-bin offsets in [0, 1] drawn from the Beta distribution with mean m and spread sqrt(sd^2 + min_spread^2)
    (moment-matched dequantization; a uniformly filled bin gives Beta(1, 1)). Uses the global torch RNG.
    alpha, beta >= 1 keeps the target density bounded: a Beta with alpha < 1 (a deposit hugging a bin edge, e.g. a keV
    electron at the entry face) has an infinite density at the edge, which maximum likelihood chases until it
    diverges. Near an edge this only tightens the spread (alpha = 1 at m = 0.01 gives sd ~ 0.01 bin)."""
    m = m.float().clamp(1e-3, 1.0 - 1e-3)
    var = sd.float() ** 2 + min_spread ** 2
    var = torch.minimum(var, m * (1.0 - m) / 2.0)               # kappa >= 1
    k = m * (1.0 - m) / var - 1.0
    k = torch.maximum(k, torch.maximum(1.0 / m, 1.0 / (1.0 - m))).clamp_max(1e6)
    a, b = m * k, (1.0 - m) * k
    ga, gb = torch._standard_gamma(a), torch._standard_gamma(b)
    return (ga / (ga + gb).clamp_min(1e-30)).nan_to_num(0.5).clamp(0.0, 1.0)


def entry_points(spec, bins, gen=None, dequant=True, mom=None, min_spread=0.05, n=None):
    """Training points of image entries: bin + in-bin offset. mom (N, 4) = in-bin (mx, my, sx, sy) -> moment-matched
    Beta offsets; otherwise uniform offsets (or bin centres with dequant=False). n: per-entry grid size (size-sweep
    data, one grid per cell size); default spec.n_sub."""
    if n is not None and getattr(spec, "min_spread_cell", 0.0) > 0:
        # floor the spread at a fixed fraction of the CELL (not of the bin): with 1000 x 1000 bins (1 cm cells) the
        # bin floor alone leaves ~1e-4-cell-wide points, sharper than the splines' smallest bin (1e-3), and the
        # likelihood chased them into loss spikes (size_stream scr, 2026-11-22)
        min_spread = torch.clamp(spec.min_spread_cell * n.float(), min=min_spread)
    n = spec.n_sub if n is None else n.long()
    b = bins.long()
    bx, by = (b // n).float(), (b % n).float()
    if dequant and mom is not None:
        ux = beta_offsets(mom[:, 0], mom[:, 2], min_spread)
        uy = beta_offsets(mom[:, 1], mom[:, 3], min_spread)
    elif dequant:
        ux = torch.rand(len(b), device=b.device, generator=gen)
        uy = torch.rand(len(b), device=b.device, generator=gen)
    else:
        ux = uy = torch.full((len(b),), 0.5, device=b.device)
    return (bx + ux) / n, (by + uy) / n


def gather(R, idx):
    return {k: v[idx] for k, v in R.items()}


# ====================================================================================================================
# Training
# ====================================================================================================================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def batch_nll(model, spec, Rd, ent_seg, ent_bin, idx, gen=None, augment=True, dequant=True, ent_mom=None,
              points=None, ent_n=None):
    s = ent_seg[idx].long()
    Rb = gather(Rd, s)
    if points is not None:
        px, py = points
    else:
        px, py = entry_points(spec, ent_bin[idx], gen, dequant, None if ent_mom is None else ent_mom[idx],
                              getattr(spec, "min_spread", 0.05), None if ent_n is None else ent_n[idx])
    if augment:
        k = random_sym(Rb["start"], gen)
        Rb = apply_d4(k, Rb)
        px, py = d4_points(k, px, py)
    h = model.context(cond_features(spec, Rb))
    return -model.log_prob(torch.stack([px, py], -1).clamp(0.0, 1.0), h)


def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    spec, R, ent = load_kernel_data(args.train_npz, max_segments=args.max_segments, check_sha=not args.no_sha)
    n_t = len(R["E0"])
    rng = np.random.default_rng(args.seed)
    is_val = rng.random(n_t) < args.val_frac
    w = ent["E"].astype(np.float64)
    if args.seg_weight == "unit":
        w = w / R["dose"][ent["seg"]]
    tr = ~is_val[ent["seg"]]
    Rd = to_device({k: v for k, v in R.items()}, dev)
    ent_seg = torch.as_tensor(ent["seg"], device=dev)
    ent_bin = torch.as_tensor(ent["bin"], device=dev)
    use_mom = args.dequant == "moment"
    if use_mom and "mom" not in ent:
        raise ValueError("--dequant moment needs a _subcell.npz with in-bin moments (sub_mx ...); use --dequant uniform")
    ent_mom = torch.as_tensor(ent["mom"], device=dev) if use_mom else None
    spec.dequant, spec.min_spread = args.dequant, args.min_spread
    cdf = torch.as_tensor(np.cumsum(np.where(tr, w, 0.0)), device=dev)
    # fixed validation draw: entries of held-out transits, proportional to weight, fixed noise, no augmentation
    pv = np.where(~tr, w, 0.0)
    val_idx = torch.as_tensor(rng.choice(len(w), size=args.n_val, p=pv / pv.sum()), device=dev)
    vgen = torch.Generator(device=dev).manual_seed(1234)
    torch.manual_seed(1234)                                       # fixed validation points (Beta draws use it)
    val_pts = entry_points(spec, ent_bin[val_idx], vgen, True, None if ent_mom is None else ent_mom[val_idx],
                           spec.min_spread)
    torch.manual_seed(args.seed)
    del w, pv, tr
    spec.phase_n = spec.n_sub if args.phase_feats else 0
    model = SubcellFlow(cond_dim(spec), ctx_dim=args.ctx, hidden=args.hidden, n_layers=args.layers, n_bins=args.bins,
                        n_freq=args.n_freq, depth=args.depth).to(dev)
    print(f"Flow: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, {args.layers} couplings x "
          f"{args.bins} spline bins; weights '{args.seg_weight}'; {args.steps} steps x {args.batch}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    ema = EMA(model, args.ema)
    eval_model = SubcellFlow(**model.arch).to(dev).eval()

    @torch.no_grad()
    def val_nll():
        eval_model.load_state_dict(ema.shadow)
        tot = 0.0
        for i in range(0, len(val_idx), 65536):
            sl = slice(i, i + 65536)
            tot += batch_nll(eval_model, spec, Rd, ent_seg, ent_bin, val_idx[sl], augment=False,
                             points=(val_pts[0][sl], val_pts[1][sl])).sum().item()
        return tot / len(val_idx)

    os.makedirs(args.out, exist_ok=True)
    hist, best, t0, run, nrun, n_skip = [], float("inf"), time.time(), 0.0, 0, 0
    total = cdf[-1].item()
    for step in range(1, args.steps + 1):
        u = torch.rand(args.batch, device=dev, dtype=torch.float64, generator=gen) * total
        idx = torch.searchsorted(cdf, u).clamp_max(len(cdf) - 1)
        loss = batch_nll(model, spec, Rd, ent_seg, ent_bin, idx, gen, ent_mom=ent_mom).mean()
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            n_skip += 1
            if n_skip > 50:
                raise RuntimeError("more than 50 non-finite training losses: diverged")
            sched.step()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        ema.update(model)
        run += loss.item()
        nrun += 1
        if step % args.eval_every == 0 or step == args.steps:
            vl = val_nll()
            hist.append({"step": step, "train": run / nrun, "val": vl, "time": time.time() - t0})
            run, nrun = 0.0, 0
            print(f"  step {step}/{args.steps}: train NLL {hist[-1]['train']:.4f}  val NLL (EMA) {vl:.4f} nats  "
                  f"({hist[-1]['time']:.0f}s)" + (f"  [{n_skip} non-finite steps skipped]" if n_skip else ""),
                  flush=True)
            if vl < best:
                best = vl
                save_kernel(os.path.join(args.out, "subcell_kernel.pth"), spec, eval_model, hist, vars(args))
    save_kernel(os.path.join(args.out, "subcell_kernel_last.pth"), spec, eval_model, hist, vars(args))
    with open(os.path.join(args.out, "train_info.json"), "w") as f:
        json.dump({"spec": vars(spec), "args": vars(args), "best_val_nll": best, "history": hist}, f, indent=2)
    print(f"Saved {args.out} (best val NLL {best:.4f} nats; uniform kernel = 0)")


def save_kernel(path, spec, model, hist, args):
    torch.save({"spec": vars(spec), "arch": model.arch, "state": model.state_dict(), "history": hist,
                "args": args}, path)


def load_kernel(path, device):
    if os.path.isdir(path):
        path = os.path.join(path, "subcell_kernel.pth")
    blob = torch.load(path, map_location=device, weights_only=False)
    m = SubcellFlow(**blob["arch"]).to(device)
    m.load_state_dict(blob["state"])
    return Spec(**blob["spec"]), m.eval(), blob


# ====================================================================================================================
# Deterministic tally: expected image of a transit on an n x n grid of the cell
# ====================================================================================================================
@torch.no_grad()
def kernel_images(model, spec, Rd, rows, n, sub=2, max_points=1 << 21):
    """(len(rows), n, n) bin probabilities by midpoint quadrature with sub x sub points per bin, renormalized to sum 1.
    Also returns the raw quadrature mass (1 if the quadrature resolves the density)."""
    dev = Rd["E0"].device
    m = n * sub
    g = (torch.arange(m, device=dev, dtype=torch.float32) + 0.5) / m
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)          # (m*m, 2), index ix * m + iy
    chunk = max(1, max_points // len(pts))
    out, mass = [], []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        h = model.context(cond_features(spec, gather(Rd, r)))
        lp = model.log_prob(pts.unsqueeze(0).expand(len(r), -1, -1), h.unsqueeze(1))
        p = torch.exp(lp).reshape(len(r), n, sub, n, sub).mean((2, 4)) / (n * n)
        s = p.sum((1, 2), keepdim=True)
        out.append(p / s)
        mass.append(s.reshape(-1))
    return torch.cat(out), torch.cat(mass)


@torch.no_grad()
def kernel_images_sampled(model, spec, Rd, rows, n, n_samples=4096, max_points=1 << 20):
    """(len(rows), n, n) bin probabilities estimated from n_samples flow samples per transit (unbiased at any
    resolution, including sub-bin features the midpoint quadrature cannot resolve)."""
    chunk = max(1, max_points // n_samples)
    out = []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        z = model.sample(model.context(cond_features(spec, gather(Rd, r))), n_samples)
        b = ((z[..., 0] * n).long().clamp(0, n - 1) * n + (z[..., 1] * n).long().clamp(0, n - 1))
        h = torch.zeros(len(r), n * n, device=z.device)
        h.scatter_add_(1, b, torch.full_like(b, 1.0 / n_samples, dtype=torch.float32))
        out.append(h.reshape(len(r), n, n))
    return torch.cat(out), torch.ones(len(rows), device=Rd["E0"].device)


# ====================================================================================================================
# SCR tally: chain transit log -> dose on a fine mesh (n x n bins per transport cell)
# ====================================================================================================================
def canonical_to_cell(x, y, face):
    """Positions in the canonical frame (entered through the left face) -> cell frame of a cell entered through
    `face` (0 left, 1 right, 2 bottom, 3 top; -1 = interior start, already in the cell frame). Same map as
    chain_core.transform_from_canonical, on positions only."""
    f = face.reshape(-1, *([1] * (x.dim() - 1))).expand_as(x)
    xo = torch.where(f == 1, 1.0 - x, torch.where(f == 2, y, torch.where(f == 3, y, x)))
    yo = torch.where(f == 2, x, torch.where(f == 3, 1.0 - x, y))
    return xo, yo


def transit_log_tensors(T, rows, device):
    """Transit-log rows (chain_secondaries(..., transit_log=True)["transits"]) -> RAW tensors for cond_features."""
    g = lambda k: np.asarray(T[k])[rows]  # noqa: E731
    absorbed = g("cls") == ABSORBED
    R = {"start": g("start"), "x0": g("x0"), "y0": g("y0"), "u0": g("u0"), "v0": g("v0"), "E0": g("E0"),
         "cls": g("cls"), "x1": g("x1"), "y1": g("y1"), "u1": g("u1"), "v1": g("v1"),
         "E_out": np.where(absorbed, 0.0, g("E_out")), "dose": g("dose_cond") if "dose_cond" in T else g("dose"),
         "n_rel": g("n_rel"), "flag": g("flag")}
    return to_device(R, device)


@torch.no_grad()
def auto_samples(n):
    """Kernel samples per transit for an n x n sub-grid: 2n rounded up to a power of two, at least 32. Measured on
    100 um Cu with Sobol sampling (accuracy per unit time vs analog MC): n = 8 best at K = 32, n = 16 at 32-64,
    n = 32 at 64-128. The sampling variance scales as (n / K) when K < n, so K must grow with the sub-grid."""
    return max(32, 1 << math.ceil(math.log2(2 * n)))


def scr_tally(log, kernels, nx, ny, n, mode="sample", n_samples=None, quad_sub=4, device="cpu", gen=None,
              max_points=1 << 19, tf32=False, sampling="sobol"):
    """See _scr_tally. n_samples None = auto_samples(n); sampling 'sobol' (default) is about 2x more sample-efficient
    than 'random' at fine sub-grids. tf32: TF32 tensor-core matmuls for the flow's MLPs. Off by default: it gains only ~10-15% and
    moves samples by up to ~1e-2 cell widths at sharp (keV) densities. The global flag is restored afterwards."""
    if n_samples is None:
        n_samples = auto_samples(n)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        return _scr_tally(log, kernels, nx, ny, n, mode, n_samples, quad_sub, device, gen, max_points, sampling)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@torch.no_grad()
def _scr_tally(log, kernels, nx, ny, n, mode, n_samples, quad_sub, device, gen, max_points, sampling):
    """Fine-mesh dose (nx * n, ny * n) from a chain transit log.

    kernels: {material index: (spec, model)}; transits of a material without a kernel are spread uniformly over
    their cell. mode 'sample': n_samples flow samples per transit, each worth dose / n_samples (unbiased at any n;
    base points per SubcellFlow.sample, sampling = 'random' | 'jitter' | 'sobol');
    'quad': expected image by midpoint quadrature (quad_sub^2 points per fine bin, renormalized) - no sampling noise,
    but it under-resolves densities sharper than a fine bin. Point deposits (log["points"]) go to their bin.
    Returns (dose_fine, info) with the dose per route for closure checks."""
    T, P = log["transits"], log["points"]
    fine = torch.zeros(nx * n * ny * n, dtype=torch.float64, device=device)
    info = {"kernel": 0.0, "flat": 0.0, "points": 0.0, "quad_mass_min": 1.0}

    def add(cx, cy, x, y, wgt):
        """Cell (cx, cy) per row; x, y, wgt per row (1-D) or per row and point (2-D)."""
        if x.dim() > 1:
            cx, cy = cx.unsqueeze(-1), cy.unsqueeze(-1)
        ix = cx.long() * n + (x * n).long().clamp(0, n - 1)
        iy = cy.long() * n + (y * n).long().clamp(0, n - 1)
        fine.index_add_(0, (ix * (ny * n) + iy).reshape(-1), wgt.reshape(-1).double())

    dose_all = np.asarray(T["dose"], dtype=np.float64)
    mat = np.asarray(T["mat"])
    for m in np.unique(mat):
        rows_m = np.flatnonzero((mat == m) & (dose_all > 0))
        if rows_m.size == 0:
            continue
        if int(m) not in kernels:      # no kernel: uniform over the cell
            cx = torch.as_tensor(np.asarray(T["cx"])[rows_m], device=device)
            cy = torch.as_tensor(np.asarray(T["cy"])[rows_m], device=device)
            d = torch.as_tensor(dose_all[rows_m], device=device)
            c = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n
            gx, gy = torch.meshgrid(c, c, indexing="ij")
            add(cx, cy, gx.reshape(1, -1).expand(len(d), -1), gy.reshape(1, -1).expand(len(d), -1),
                (d / (n * n)).unsqueeze(-1).expand(-1, n * n))
            info["flat"] += float(d.sum())
            continue
        spec, model = kernels[int(m)]
        per = n_samples if mode == "sample" else (n * quad_sub) ** 2
        chunk = max(1, max_points // per)
        for i in range(0, rows_m.size, chunk):
            r = rows_m[i:i + chunk]
            Rd = transit_log_tensors(T, r, device)
            d = torch.as_tensor(dose_all[r], device=device)
            face = torch.as_tensor(np.asarray(T["face"])[r], device=device)
            cx = torch.as_tensor(np.asarray(T["cx"])[r], device=device)
            cy = torch.as_tensor(np.asarray(T["cy"])[r], device=device)
            h = model.context(cond_features(spec, Rd))
            if mode == "sample":
                z = model.sample(h, n_samples, gen=gen, base=sampling).clamp(0.0, 1.0)
                x, y = canonical_to_cell(z[..., 0], z[..., 1], face)
                wgt = (d / n_samples).unsqueeze(-1).expand(-1, n_samples)
            else:
                img, mass = kernel_images(model, spec, Rd, torch.arange(len(r), device=device), n, sub=quad_sub)
                info["quad_mass_min"] = min(info["quad_mass_min"], float(mass.min()))
                c = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
                gx, gy = torch.meshgrid(c, c, indexing="ij")
                x, y = canonical_to_cell(gx.reshape(1, -1).expand(len(r), -1), gy.reshape(1, -1).expand(len(r), -1),
                                         face)
                wgt = img.reshape(len(r), -1).double() * d.unsqueeze(-1)
            add(cx, cy, x, y, wgt)
            info["kernel"] += float(d.sum())
    if len(P["E"]):
        e = torch.as_tensor(np.asarray(P["E"], dtype=np.float64), device=device)
        add(torch.as_tensor(np.asarray(P["cx"]), device=device), torch.as_tensor(np.asarray(P["cy"]), device=device),
            torch.as_tensor(np.asarray(P["x"]), device=device), torch.as_tensor(np.asarray(P["y"]), device=device), e)
        info["points"] = float(e.sum())
    return fine.reshape(nx * n, ny * n).cpu().numpy(), info


def truth_images(ent, starts, rows, n_sub, n):
    """Dense true images (len(rows), n, n) from the sorted entry table (coarsened from n_sub if n < n_sub)."""
    f = n_sub // n
    out = np.zeros((len(rows), n, n))
    for k, r in enumerate(rows):
        a, b = starts[r], starts[r + 1]
        bins = ent["bin"][a:b].astype(np.int64)
        np.add.at(out[k], ((bins // n_sub) // f, (bins % n_sub) // f), ent["E"][a:b])
    return out


# ====================================================================================================================
# Evaluation
# ====================================================================================================================
@torch.no_grad()
def cmd_eval(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    spec, model, blob = load_kernel(args.model, dev)
    spec_t, R, ent = load_kernel_data(args.test_npz, sub_npz=args.test_sub, check_sha=not args.no_sha)
    if abs(spec_t.L - spec.L) > 1e-12:
        raise ValueError("test data cell size differs from the training data")
    n_sub = spec_t.n_sub                       # evaluation grid = the test data's grid (may be finer than training)
    grid = Spec(n_sub=n_sub)                   # decodes test-data bins
    n_t = len(R["E0"])
    Rd = to_device(R, dev)
    starts = np.searchsorted(ent["seg"], np.arange(n_t + 1))
    band = np.digitize(R["E0"], E_BANDS) - 1
    start = (R["start"] > 0.5).astype(int)
    out_dir = args.out or os.path.join(args.model if os.path.isdir(args.model) else os.path.dirname(args.model),
                                       "eval_" + os.path.basename(args.test_sub or args.test_npz)[:-4])
    os.makedirs(out_dir, exist_ok=True)
    res = {"model": args.model, "test_npz": args.test_npz, "n_transits": int(n_t)}

    # --- 1. held-out NLL per energy band (points drawn proportional to E, fixed dequantization noise) ----------------
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    w = ent["E"].astype(np.float64)
    idx = rng.choice(len(w), size=min(args.n_nll, len(w)), p=w / w.sum())
    nll = []
    ent_seg_d = torch.as_tensor(ent["seg"], device=dev)
    ent_bin_d = torch.as_tensor(ent["bin"], device=dev)
    ent_mom_d = torch.as_tensor(ent["mom"], device=dev) if "mom" in ent else None
    b_idx = band[ent["seg"][idx]]
    res["nll"] = {}
    for mode in ("uniform", "moment"):
        if mode == "moment" and ent_mom_d is None:
            continue
        torch.manual_seed(args.seed)
        nll = []
        for i in range(0, len(idx), 65536):
            ii = torch.as_tensor(idx[i:i + 65536], device=dev)
            px, py = entry_points(grid, ent_bin_d[ii], gen, True, None if mode == "uniform" else ent_mom_d[ii],
                                  getattr(spec, "min_spread", 0.05))
            nll.append(batch_nll(model, spec, Rd, ent_seg_d, ent_bin_d, ii, augment=False,
                                 points=(px, py)).cpu().numpy())
        nll = np.concatenate(nll)
        res["nll"][mode] = {"all": float(nll.mean()),
                            **{BAND_NAMES[b]: float(nll[b_idx == b].mean()) for b in range(NB)}}
    print(f"NLL (nats, dose-weighted, test points dequantized uniformly / moment-matched; uniform kernel = 0, "
          f"one-bin delta at {n_sub}x{n_sub} = -{math.log(n_sub * n_sub):.2f}):", json.dumps(res["nll"], indent=1))

    # --- 2. expected images on a transit subsample: Brier score, group aggregates ----------------------------------
    pick = rng.choice(n_t, size=min(args.n_images, n_t), replace=False)
    pick.sort()
    resolutions = [r for r in (256, 64, 16, 4) if r <= n_sub and n_sub % r == 0]
    if args.image_mode == "sample":
        images = lambda rr: kernel_images_sampled(model, spec, Rd, rr, n_sub, args.n_samples)  # noqa: E731
    else:
        images = lambda rr: kernel_images(model, spec, Rd, rr, n_sub, sub=args.quad_sub)  # noqa: E731
    brier = {r: {"model": np.zeros(NB), "uniform": np.zeros(NB), "norm": np.zeros(NB)} for r in resolutions}
    n_groups = NB * 2 * N_EXIT
    agg_t = np.zeros((n_groups, n_sub, n_sub))
    agg_p = np.zeros_like(agg_t)
    agg_v = np.zeros_like(agg_t)
    gid_all = (band * 2 + start) * N_EXIT + R["cls"].astype(int)
    # entry-aligned aggregates of boundary transits: lateral axis shifted so the entry bin sits at column n - 1
    W = 2 * n_sub - 1
    by0_all = np.clip((R["y0"] * n_sub).astype(int), 0, n_sub - 1)
    al_t = np.zeros(NB * N_EXIT * n_sub * W)
    al_p, al_v = np.zeros_like(al_t), np.zeros_like(al_t)
    mass_all = []
    t0 = time.time()
    for i in range(0, len(pick), args.img_chunk):
        rows = pick[i:i + args.img_chunk]
        p, mass = images(torch.as_tensor(rows, device=dev))
        mass_all.append(mass.cpu().numpy())
        p = p.cpu().numpy() * R["dose"][rows][:, None, None]
        t = truth_images(ent, starts, rows, n_sub, n_sub)
        u = np.broadcast_to(R["dose"][rows][:, None, None] / n_sub ** 2, t.shape)
        for r in resolutions:
            f = n_sub // r
            co = lambda a: a.reshape(len(rows), r, f, r, f).sum((2, 4))  # noqa: E731
            tc, pc, uc = co(t), co(p), co(u)
            for b in range(NB):
                m = band[rows] == b
                brier[r]["model"][b] += ((tc[m] - pc[m]) ** 2).sum()
                brier[r]["uniform"][b] += ((tc[m] - uc[m]) ** 2).sum()
                brier[r]["norm"][b] += (tc[m] ** 2).sum()
        g = gid_all[rows]
        np.add.at(agg_t, g, t)
        np.add.at(agg_p, g, p)
        np.add.at(agg_v, g, (t - p) ** 2)
        bnd = start[rows] == 0
        if bnd.any():
            rb = rows[bnd]
            ga = band[rb] * N_EXIT + R["cls"][rb].astype(int)
            col = np.arange(n_sub)[None, :] - by0_all[rb][:, None] + n_sub - 1          # (C, n) aligned column
            flat = ((ga[:, None, None] * n_sub + np.arange(n_sub)[None, :, None]) * W + col[:, None, :]).ravel()
            for acc, a in ((al_t, t[bnd]), (al_p, p[bnd]), (al_v, (t[bnd] - p[bnd]) ** 2)):
                acc += np.bincount(flat, weights=a.ravel(), minlength=len(acc))
    t_img = time.time() - t0
    mass_all = np.concatenate(mass_all)
    res["image_time_per_transit_ms"] = 1e3 * t_img / len(pick)
    res["quadrature_mass"] = {"mean": float(mass_all.mean()), "p01": float(np.quantile(mass_all, 0.01)),
                              "p99": float(np.quantile(mass_all, 0.99))}
    res["brier"] = {str(r): {BAND_NAMES[b]: {"model": float(brier[r]["model"][b] / max(brier[r]["norm"][b], 1e-300)),
                                             "uniform": float(brier[r]["uniform"][b] /
                                                              max(brier[r]["norm"][b], 1e-300)),
                                             "skill": float(1 - brier[r]["model"][b] /
                                                            max(brier[r]["uniform"][b], 1e-300))}
                             for b in range(NB)} for r in resolutions}
    how = (f"{args.n_samples} flow samples" if args.image_mode == "sample" else
           f"{args.quad_sub}^2 points per bin; quadrature mass {res['quadrature_mass']}")
    print(f"Images: {len(pick)} transits, {res['image_time_per_transit_ms']:.2f} ms each ({n_sub}x{n_sub}, {how})")
    print("Brier (sum |true - pred|^2 / sum |true|^2; skill = 1 - model/uniform):")
    for r in resolutions:
        print(f"  {r}x{r}: " + "  ".join(f"{BAND_NAMES[b]} {res['brier'][str(r)][BAND_NAMES[b]]['model']:.3f}/"
                                         f"{res['brier'][str(r)][BAND_NAMES[b]]['uniform']:.3f} "
                                         f"(skill {res['brier'][str(r)][BAND_NAMES[b]]['skill']:.3f})"
                                         for b in range(NB)))

    # group aggregates: bias of the expected image vs the noise floor of the true sum
    groups = []
    for gidx in range(n_groups):
        tot = agg_t[gidx].sum()
        n_g = int((gid_all[pick] == gidx).sum())
        if n_g < args.min_group or tot <= 0:
            continue
        b, rest = divmod(gidx, 2 * N_EXIT)
        st, cl = divmod(rest, N_EXIT)
        row = {"band": BAND_NAMES[b], "start": ("boundary", "interior")[st],
               "exit": ("bottom", "right", "top", "left", "absorbed")[cl], "n": n_g,
               "dose_sum_true": float(tot), "dose_sum_pred": float(agg_p[gidx].sum())}
        for r in resolutions:
            f = n_sub // r
            co = lambda a: a.reshape(r, f, r, f).sum((1, 3))  # noqa: E731
            d = co(agg_t[gidx]) - co(agg_p[gidx])
            row[f"relL2_{r}"] = float(np.linalg.norm(d) / np.linalg.norm(co(agg_t[gidx])))
            row[f"noise_{r}"] = float(math.sqrt(co(agg_v[gidx]).sum()) / np.linalg.norm(co(agg_t[gidx])))
        groups.append(row)
    res["groups"] = groups
    print(f"Group aggregates (>= {args.min_group} transits): relL2 vs noise floor at "
          + " / ".join(str(r) for r in resolutions))
    for g in groups:
        print(f"  {g['band']:>10} {g['start']:>8} {g['exit']:>8} n={g['n']:>6}: " +
              "  ".join(f"{g[f'relL2_{r}']:.3f}/{g[f'noise_{r}']:.3f}" for r in resolutions))

    # entry-aligned: relL2 of the aligned (n x 2n-1) sums and of their depth / lateral profiles, with noise floors
    al_t, al_p, al_v = (a.reshape(NB, N_EXIT, n_sub, W) for a in (al_t, al_p, al_v))
    aligned = []
    print("Entry-aligned boundary aggregates: relL2/noise of image, depth profile, lateral profile "
          "(bias = sqrt(relL2^2 - noise^2))")
    for b in range(NB):
        for cl in list(range(N_EXIT)) + [-1]:
            T, P, V = (a[b].sum(0) if cl < 0 else a[b, cl] for a in (al_t, al_p, al_v))
            if T.sum() <= 0:
                continue
            row = {"band": BAND_NAMES[b], "exit": "all" if cl < 0 else
                   ("bottom", "right", "top", "left", "absorbed")[cl]}
            for name, ax in (("image", None), ("depth", 1), ("lateral", 0)):
                tt, pp = (T, P) if ax is None else (T.sum(ax), P.sum(ax))
                vv = V if ax is None else V.sum(ax)      # transit-independent: profile variance = sum of variances
                rl = float(np.linalg.norm(tt - pp) / np.linalg.norm(tt))
                nz = float(math.sqrt(vv.sum()) / np.linalg.norm(tt))
                row[name] = {"relL2": rl, "noise": nz, "bias": float(math.sqrt(max(rl * rl - nz * nz, 0.0)))}
            aligned.append(row)
            if cl < 0 or T.sum() > 0.05 * al_t[b].sum():
                print(f"  {row['band']:>10} {row['exit']:>8}: " + "  ".join(
                    f"{k} {row[k]['relL2']:.3f}/{row[k]['noise']:.3f}" for k in ("image", "depth", "lateral")))
    res["aligned"] = aligned

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "aggregates.npz"), agg_t=agg_t, agg_p=agg_p, agg_v=agg_v,
                        al_t=al_t, al_p=al_p, al_v=al_v, band_edges=np.array(E_BANDS))
    if not args.no_plots:
        plot_eval(out_dir, spec, model, Rd, R, ent, starts, band, start, agg_t, agg_p, agg_v, args, dev, n_sub,
                  images)
        plot_aligned(out_dir, spec, al_t, al_p, al_v, n_sub)
    print(f"Saved {out_dir}")


def plot_eval(out_dir, spec, model, Rd, R, ent, starts, band, start, agg_t, agg_p, agg_v, args, dev, n, images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    L_um = spec.L * 1e4
    ext = [0, L_um, 0, L_um]

    # (a) examples: one realization vs the model's expected image (+ 1000 samples drawn from the flow)
    rng = np.random.default_rng(args.seed + 1)
    picks = []
    for b in (1, 2, 3):
        for cl in (1, 3, 4, 0):
            m = np.flatnonzero((band == b) & (start == 0) & (R["cls"].astype(int) == cl))
            if len(m):
                picks.append(int(rng.choice(m)))
    picks = picks[:9]
    p, _ = images(torch.as_tensor(picks, device=dev))
    p = p.cpu().numpy() * R["dose"][picks][:, None, None]
    t = truth_images(ent, starts, picks, n, n)
    smp = model.sample(model.context(cond_features(spec, gather(Rd, torch.as_tensor(picks, device=dev)))), 1000)
    smp = smp.cpu().numpy()
    fig, ax = plt.subplots(3, len(picks), figsize=(2.3 * len(picks), 7))
    names = ("bottom", "right", "top", "left", "absorbed")
    for j, r in enumerate(picks):
        vmax = max(t[j].max(), p[j].max())
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        ax[0, j].imshow(np.maximum(t[j].T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma")
        ax[1, j].imshow(np.maximum(p[j].T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma")
        ax[2, j].scatter(smp[j, :, 0] * L_um, smp[j, :, 1] * L_um, s=1, alpha=0.4)
        for a in ax[:, j]:
            a.plot([R["x0"][r] * L_um], [R["y0"][r] * L_um], "c>", ms=5)
            if R["cls"][r] != ABSORBED:
                a.plot([R["x1"][r] * L_um], [R["y1"][r] * L_um], "co", ms=5, mfc="none")
            a.set_xticks([]), a.set_yticks([])
        ax[2, j].set_xlim(0, L_um), ax[2, j].set_ylim(0, L_um), ax[2, j].set_aspect("equal")
        ax[0, j].set_title(f"E0 {R['E0'][r] * 1e3:.0f} keV, {names[int(R['cls'][r])]}\n"
                           f"dose {R['dose'][r] / R['E0'][r]:.2f} E0", fontsize=8)
    ax[0, 0].set_ylabel("one MC transit")
    ax[1, 0].set_ylabel("kernel E[image]")
    ax[2, 0].set_ylabel("flow samples")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "examples.png"), dpi=130)
    plt.close(fig)

    # (b) aggregate depth (x) and lateral (y) profiles per band, boundary transits, all exits pooled
    fig, ax = plt.subplots(2, NB, figsize=(3.75 * NB, 6))
    xc = (np.arange(n) + 0.5) / n * L_um
    for b in range(NB):
        gs = [(b * 2 + 0) * N_EXIT + cl for cl in range(N_EXIT)]
        T, P, V = agg_t[gs].sum(0), agg_p[gs].sum(0), agg_v[gs].sum(0)
        if T.sum() <= 0:
            continue
        for k, (axis, lab) in enumerate(((1, "depth x (um)"), (0, "lateral y (um)"))):
            tt, pp, sd = T.sum(axis), P.sum(axis), np.sqrt(V.sum(axis))
            a = ax[k, b]
            a.fill_between(xc, tt - 2 * sd, tt + 2 * sd, color="0.8", label="MC +- 2 sd")
            a.plot(xc, tt, "k-", lw=1, label="MC")
            a.plot(xc, pp, "C3--", lw=1.2, label="kernel")
            a.set_yscale("log")
            a.set_xlabel(lab)
            if k == 0:
                a.set_title(f"boundary, E0 {BAND_NAMES[b]}")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylabel("summed dose (MeV)")
    ax[1, 0].set_ylabel("summed dose (MeV)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=130)
    plt.close(fig)

    # (c) aggregate images per band: MC, kernel, (kernel - MC) / sd
    fig, ax = plt.subplots(3, NB, figsize=(3.5 * NB, 10))
    for b in range(NB):
        gs = [(b * 2 + 0) * N_EXIT + cl for cl in range(N_EXIT)]
        T, P, V = agg_t[gs].sum(0), agg_p[gs].sum(0), agg_v[gs].sum(0)
        if T.sum() <= 0:
            continue
        vmax = T.max()
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        ax[0, b].imshow(np.maximum(T.T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma")
        ax[1, b].imshow(np.maximum(P.T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma")
        z = (P - T) / np.sqrt(np.maximum(V, 1e-300))
        im = ax[2, b].imshow(np.clip(z.T, -5, 5), origin="lower", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
        ax[0, b].set_title(f"boundary, E0 {BAND_NAMES[b]}")
    ax[0, 0].set_ylabel("MC sum")
    ax[1, 0].set_ylabel("kernel sum")
    ax[2, 0].set_ylabel("(kernel - MC) / sd")
    fig.colorbar(im, ax=ax[2, :], shrink=0.8)
    fig.savefig(os.path.join(out_dir, "aggregate_images.png"), dpi=110)
    plt.close(fig)


def plot_aligned(out_dir, spec, al_t, al_p, al_v, n):
    """Entry-aligned boundary aggregates per band: depth and lateral profiles (MC +- 2 sd vs kernel) and images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    W = 2 * n - 1
    L_um = spec.L * 1e4
    xc = (np.arange(n) + 0.5) / n * L_um
    yc = (np.arange(W) - (n - 1)) / n * L_um
    fig, ax = plt.subplots(4, NB, figsize=(4 * NB, 14))
    for b in range(NB):
        T, P, V = al_t[b].sum(0), al_p[b].sum(0), al_v[b].sum(0)
        if T.sum() <= 0:
            continue
        for k, (axis, xs, lab) in enumerate(((1, xc, "depth x (um)"), (0, yc, "y - y_entry (um)"))):
            tt, pp, sd = T.sum(axis), P.sum(axis), np.sqrt(V.sum(axis))
            a = ax[k, b]
            a.fill_between(xs, np.maximum(tt - 2 * sd, tt * 1e-3), tt + 2 * sd, color="0.8", label="MC +- 2 sd")
            a.plot(xs, tt, "k-", lw=1, label="MC")
            a.plot(xs, pp, "C3--", lw=1.2, label="kernel")
            a.set_yscale("log")
            a.set_xlabel(lab)
            if k == 0:
                a.set_title(f"boundary entries, E0 {BAND_NAMES[b]}")
        vmax = T.max()
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        ext = [0, L_um, yc[0], yc[-1]]
        ax[2, b].imshow(np.maximum(T.T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma", aspect="auto")
        ax[3, b].imshow(np.maximum(P.T, 1e-30), origin="lower", extent=ext, norm=norm, cmap="magma", aspect="auto")
        for a in ax[2:, b]:
            a.set_xlabel("depth x (um)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_ylabel("summed dose (MeV)")
    ax[1, 0].set_ylabel("summed dose (MeV)")
    ax[2, 0].set_ylabel("MC: y - y_entry (um)")
    ax[3, 0].set_ylabel("kernel: y - y_entry (um)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aligned.png"), dpi=110)
    plt.close(fig)


# ====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train_npz", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seg_weight", choices=("dose", "unit"), default="dose")
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=16384)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--layers", type=int, default=8)
    t.add_argument("--bins", type=int, default=32)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--ctx", type=int, default=256)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--n_freq", type=int, default=6)
    t.add_argument("--val_frac", type=float, default=0.02)
    t.add_argument("--n_val", type=int, default=500000)
    t.add_argument("--eval_every", type=int, default=2000)
    t.add_argument("--dequant", choices=("moment", "uniform"), default="moment",
                   help="in-bin placement of training points: Beta matched to the recorded in-bin moments, or uniform")
    t.add_argument("--min_spread", type=float, default=0.05,
                   help="spread floor (bin widths) of the moment-matched dequantization")
    t.add_argument("--phase_feats", action="store_true",
                   help="diagnostic: add the sub-bin phase of entry / exit points on the training grid")
    t.add_argument("--max_segments", type=int, default=None)
    t.add_argument("--no_sha", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--test_npz", required=True)
    e.add_argument("--test_sub", default=None,
                   help="sub-cell file of the test data (default <test_npz>_subcell.npz); may use a finer grid")
    e.add_argument("--out", default=None)
    e.add_argument("--n_nll", type=int, default=2000000)
    e.add_argument("--n_images", type=int, default=40000)
    e.add_argument("--img_chunk", type=int, default=2000)
    e.add_argument("--image_mode", choices=("sample", "quad"), default="sample",
                   help="expected images from flow samples (unbiased at any resolution) or midpoint quadrature")
    e.add_argument("--n_samples", type=int, default=4096)
    e.add_argument("--quad_sub", type=int, default=2)
    e.add_argument("--min_group", type=int, default=200)
    e.add_argument("--no_plots", action="store_true")
    e.add_argument("--no_sha", action="store_true")
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
