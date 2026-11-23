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
    5. stop flow (c = 4, optional "stop" model) -> where an absorbed electron stops (logit x, logit y); only the
                            charge tally needs it (an electron stopping adds -e at that point)
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
EDGE_BAND = 1.0 / 16    # exit-flow oversampling: boundary entries this close (cell units) to a lateral face
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


EDGE_DELTA = 1e-4      # edge features: log10 distance to the bottom / top face, floored at 1e-4 cell


def edge_feats(y):
    """Log distance of the entry point to the two lateral faces, (log10(d + 1e-4) + 2) / 2 in [-1, 1]. The linear 2y - 1
    gives a network no resolution at a face. Physical edge effects are sharp there: a forward 20 MeV electron entering
    within 0.06 cell of the bottom face and still leaving through the right face was scattered upward (mean dv
    +0.004, ~0 by y0 = 0.2). With the linear feature only, the GMC smeared that selection effect into the interior, as
    a ~1e-3 focusing kick toward the cell middle. In a forward beam it accumulated over ~8 cells into a +-5 %
    within-cell density modulation (Drumm benchmark, 2026-11-16; validation/flat_beam_chain.py)."""
    return np.stack([(np.log10(y + EDGE_DELTA) + 2.0) / 2.0, (np.log10(1.0 - y + EDGE_DELTA) + 2.0) / 2.0], 1)


def _width_column(lg, dv, spec, n_bins, min_count):
    edges = np.linspace(spec.logE_lo, spec.logE_hi, n_bins + 1)
    b = np.clip(np.digitize(lg, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    sig = np.sqrt(np.bincount(b, weights=dv * dv, minlength=n_bins) / np.maximum(cnt, 1))
    sig = np.where(cnt >= min_count, sig, np.nan)
    if np.all(np.isnan(sig)):
        sig[:] = 1.0
    last = np.nanmax(sig)
    for k in range(n_bins - 1, -1, -1):              # fill gaps from the high-energy side, then make non-increasing
        if np.isnan(sig[k]):
            sig[k] = sig[k + 1] if k + 1 < n_bins else last
    sig = np.maximum.accumulate(sig[::-1])[::-1]
    return 0.5 * (edges[1:] + edges[:-1]), np.clip(sig, 1e-3, 1.0)


def scatter_width_table(T, n_bins=40, min_count=200, n_L=8):
    """Per-transit angular spread sigma(E) of boundary electrons crossing the cell: rms of v1 - v0 over right-face
    exits entering in the middle half (y0 in [0.25, 0.75], away from the selection at the faces), in n_bins log-E
    bins. Bins with too few such transits (low E: electrons rarely cross) take the value of the next higher-energy
    bin; the table is made non-increasing in E and clipped to [1e-3, 1]. Returns (log10 E centres, sigma).

    Size-conditioned data (spec.size_cond): one such column per log-L bin (n_L bins between the training bounds), so
    sigma = sigma(E, L); returns (log10 E centres, log10 L centres, sigma [n_L, n_bins])."""
    spec = T["spec"]
    m = T["boundary"] & (T["cls"] == 1) & (np.abs(T["y0"] - 0.5) < 0.25)
    lg = np.log10(np.maximum(T["E0"][m], 1e-300))
    dv = T["v1"][m] - T["v0"][m]
    if not getattr(spec, "size_cond", False):
        return _width_column(lg, dv, spec, n_bins, min_count)
    ll = np.log10(T["L"][m])
    l_edges = np.linspace(spec.logL_lo, spec.logL_hi, n_L + 1)
    bl = np.clip(np.digitize(ll, l_edges) - 1, 0, n_L - 1)
    cols = [_width_column(lg[bl == k], dv[bl == k], spec, n_bins, min_count) for k in range(n_L)]
    return cols[0][0], 0.5 * (l_edges[1:] + l_edges[:-1]), np.stack([c[1] for c in cols])


def set_scatter_table(spec, T):
    """Measure sigma(E) (or sigma(E, L)) on the tables T and store it in spec (sig_logE, sig_val[, sig_logL])."""
    tab = scatter_width_table(T)
    spec.sig_logE, spec.sig_val = tab[0].tolist(), tab[-1].tolist()
    if len(tab) == 3:
        spec.sig_logL = tab[1].tolist()
        rows = [(10 ** l, dict(zip(tab[0][::8], r[::8]))) for l, r in zip(tab[1], tab[2])][::3]
        print("scatter width sigma(E, L) (rms v1 - v0 per transit):")
        for Lc, r in rows:
            print(f"  L {Lc * 1e4:7.0f} um: " + ", ".join(f"{10 ** a:.3g} MeV {b:.3f}" for a, b in r.items()))
    else:
        print("scatter width sigma(E) (rms v1 - v0 per transit): " + ", ".join(
            f"{10 ** a:.3g} MeV {b:.3f}" for a, b in list(zip(tab[0], tab[1]))[::5]))


def smooth_scatter_table(lg, ll, dv, spec, n_E=40, n_L=16, s_E=0.15, s_L=0.25, min_count=200):
    """Smooth sigma(E, L): rms(dv) on a fine (log E, log L) grid, then count-weighted local-linear Gaussian smoothing of
    log sigma (widths s_E, s_L decades), gaps filled from higher E, non-increasing in E, clipped to [1e-3, 1].

    Why: the per-bin rms of v1 - v0 is dominated by the single-scattering tail, so the 8 x 40 table measured on 4 files
    was noisy (+-5-7 % at 20 MeV between neighbouring L nodes) and the scaled exit flow inherited most of that noise:
    the far-exit position spread of 20 MeV transits was +5 % at 1 mm, -5..-10 % at 2-4 mm vs analog MC, which a chained
    beam turns into +10 % / -5 % Drumm wings (2026-11-22, validation/scatter_width_scan.py).
    Returns (log10 E centres, log10 L centres, sigma [n_L, n_E])."""
    eE = np.linspace(spec.logE_lo, spec.logE_hi, n_E + 1)
    eL = np.linspace(spec.logL_lo, spec.logL_hi, n_L + 1)
    cE, cL = 0.5 * (eE[1:] + eE[:-1]), 0.5 * (eL[1:] + eL[:-1])
    nodes = getattr(spec, "_sigx_nodes", "edges")
    b = np.clip(np.digitize(lg, eE) - 1, 0, n_E - 1) * n_L + np.clip(np.digitize(ll, eL) - 1, 0, n_L - 1)
    cnt = np.bincount(b, minlength=n_E * n_L).reshape(n_E, n_L)
    raw = np.sqrt(np.bincount(b, weights=dv * dv, minlength=n_E * n_L).reshape(n_E, n_L) / np.maximum(cnt, 1))
    W = np.where(cnt >= min_count, cnt, 0).astype(float).ravel()
    lv = np.where(cnt >= min_count, np.log(np.maximum(raw, 1e-6)), 0.0).ravel()
    KE = np.exp(-0.5 * ((cE[:, None] - cE[None, :]) / s_E) ** 2)
    KL = np.exp(-0.5 * ((cL[:, None] - cL[None, :]) / s_L) ** 2)
    # evaluated on L nodes at the bin EDGES, so the end nodes sit on the trained range (0.1 mm / 1 cm) and the local-
    # linear fit extrapolates there; bin centres left the last node at 8.7 mm and the lookup clamped beyond it, 6 %
    # too narrow at 1 cm (20 MeV)
    gL = eL if nodes == "edges" else cL
    KL = np.exp(-0.5 * ((gL[:, None] - cL[None, :]) / s_L) ** 2)
    n_G = len(gL)
    sm = np.full((n_E, n_G), np.nan)
    for i in range(n_E):
        for j in range(n_G):
            w = (KE[i][:, None] * KL[j][None, :]).ravel() * W
            if w.sum() <= 0:
                continue
            X = np.stack([np.ones(n_E * n_L), np.repeat(cE - cE[i], n_L), np.tile(cL - gL[j], n_E)], 1)
            A = X * w[:, None]
            sm[i, j] = np.linalg.lstsq(A.T @ X + 1e-9 * np.eye(3), A.T @ lv, rcond=None)[0][0]
    sm = np.exp(sm)
    for j in range(n_G):
        col = sm[:, j]
        for i in range(n_E - 2, -1, -1):
            if not np.isfinite(col[i]):
                col[i] = col[i + 1]
        col[~np.isfinite(col)] = 1.0
        sm[:, j] = np.clip(np.maximum.accumulate(col[::-1])[::-1], 1e-3, 1.0)
    return cE, gL, sm.T


def scatter_width(spec, E, L=None, which="feat"):
    """sigma(E) of the training cell, or sigma(E, L) (bilinear in log E, log L) for a size-conditioned spec.
    which="exit": the exit-flow scale table (spec.sigx_*: a smooth table, smooth_scatter_table) when the spec has
    one, else the same table as the features."""
    lg = np.atleast_1d(np.log10(np.maximum(E, 1e-300)))
    if which == "exit" and getattr(spec, "sigx_val", None) is not None:
        logE, logL, val = spec.sigx_logE, spec.sigx_logL, spec.sigx_val
    else:
        logE, logL, val = spec.sig_logE, getattr(spec, "sig_logL", None), spec.sig_val
    if not logL:
        return np.interp(lg, logE, val)
    if L is None:
        raise ValueError("size-conditioned sigma(E, L): pass the cell side L (cm)")
    tab = np.asarray(val)                                 # [n_L, n_E]
    gl = np.asarray(logL)
    ll = np.broadcast_to(np.log10(np.asarray(L, np.float64)), np.shape(lg))
    cols = np.stack([np.interp(lg, logE, row) for row in tab])   # [n_L, n]
    k = np.clip(np.searchsorted(gl, ll) - 1, 0, len(gl) - 2)
    t = np.clip((ll - gl[k]) / (gl[k + 1] - gl[k]), 0.0, 1.0)
    idx = np.arange(len(lg))
    return (1 - t) * cols[k, idx] + t * cols[k + 1, idx]


def scale_feats(spec, y, E, L=None):
    """Distance of the entry point to the bottom / top face in units of the transit's angular spread sigma(E) (cells
    per radian, i.e. how many 'one-cell scattering widths' away the face is): clip(log10(d / sigma + 0.01) / 2, -1, 1).
    The face-selection effect on the exit (an electron entering near the bottom face that still leaves through the
    right face was scattered upward) is a near-universal function of d / sigma. With y0 and E as separate inputs the
    network leaked the strong low-energy profile (E[dv | y0 = 0.3] ~ +0.03 at 1-3 MeV) into the 20 MeV exits as a
    ~1e-3 focusing kick, which a forward beam accumulates over ~8 cells (Drumm benchmark, 2026-11-16)."""
    sg = scatter_width(spec, E, L)
    f = lambda d: np.clip(np.log10(d / sg + 0.01) / 2.0, -1.0, 1.0)  # noqa: E731
    return np.stack([f(y), f(1.0 - y)], 1)


def csda_range(spec, E):
    """CSDA range (cm) of the material at energy E, from the table stored in a size-conditioned spec."""
    return 10.0 ** np.interp(np.log10(np.maximum(E, 1e-300)), spec.csda_logE, spec.csda_logR)


def size_feats(spec, L, E):
    """Cell-size features of a size-conditioned model (spec.size_cond): the cell side L (cm) on a log scale between
    the training bounds, in [-1, 1], and log10(L / R_CSDA(E)), the side in units of the electron's range, scaled to
    about [-1, 1.5]. Transport through a square of one material depends on L mostly through L / R(E) (and through
    L / lambda_tr(E), which tracks it), so the ratio is given directly instead of being learned from (log L, log E)."""
    E = np.asarray(E, np.float64)
    L = np.broadcast_to(np.asarray(L, np.float64), E.shape)
    lf = 2.0 * (np.log10(L) - spec.logL_lo) / (spec.logL_hi - spec.logL_lo) - 1.0
    rf = np.clip((np.log10(L / csda_range(spec, E)) - 1.0) / 4.0, -2.0, 2.0)
    return np.stack([lf, rf], 1)


def _with_size(spec, base, L, E):
    if not getattr(spec, "size_cond", False):
        return base
    if L is None:
        raise ValueError("size-conditioned model: pass the cell side L (cm)")
    return np.concatenate([base, size_feats(spec, L, E)], 1)


def boundary_cond(spec, y, u, v, E, L=None):
    """Condition of an entering electron (canonical frame). L: cell side (cm, scalar or per row), used only by a
    size-conditioned spec."""
    c, s, r = dir_feats(u, v)
    base = np.stack([2.0 * y - 1.0, c, s, r, spec.e_feat(E)], 1)
    yc = np.clip(y, 0.0, 1.0)
    if getattr(spec, "edge_feats", False):
        base = np.concatenate([base, edge_feats(yc)], 1)
    if getattr(spec, "scale_feats", False):
        base = np.concatenate([base, scale_feats(spec, yc, E, L)], 1)
    return _with_size(spec, base, L, E).astype(np.float32)


def interior_cond(spec, x, y, u, v, E, L=None):
    c, s, r = dir_feats(u, v)
    base = np.stack([2.0 * x - 1.0, 2.0 * y - 1.0, c, s, r, spec.e_feat(E)], 1)
    return _with_size(spec, base, L, E).astype(np.float32)


def parent_feats(spec, start, x, y, u, v, E, L=None):
    """Shared parent description for the daughter / photon flows (boundary parents have x = 0)."""
    return np.concatenate([np.asarray(start, np.float32).reshape(-1, 1) * np.ones((len(E), 1), np.float32),
                           interior_cond(spec, x, y, u, v, E, L)], 1)


REF_S_CLIP = 0.01      # ray-relative exit encoding: reference face coordinate kept in [0.01, 0.99]
REF_PSI_CLIP = 0.95    # and reference angle in +-0.95 * pi/2


def exit_reference(cls, x0, y0, u0, v0):
    """Straight-line continuation of the entry state, in the exit flow's coordinates (ray-relative encoding).

    s_ref: face coordinate on exit face `cls` where the ray (x0, y0) + t (u0, v0), t > 0, meets that face's line, or,
    if the ray never meets it going forward, of the perpendicular projection of the entry point onto it; clipped to
    [REF_S_CLIP, 1 - REF_S_CLIP]. psi_ref: the entry direction's angle from that face's outward normal, clipped to
    +-REF_PSI_CLIP * pi/2. Both are deterministic in the condition, so the encoding is invertible.
    Motivation (2026-11-16, Drumm benchmark): with absolute targets the flow must place a narrow exit distribution
    (forward 20 MeV electron in 1 mm water: sd 0.075 cell) at a location that moves with y0. The residual y0-dependent
    mean error of ~1e-3 cell in position and direction accumulated coherently over ~8 cells of a forward beam (an
    electron keeps its within-cell coordinate) into a +-5 % within-cell dose modulation. Relative to the ray the
    target is almost independent of y0, so what error is left is nearly uniform (a drift, cancelled by symmetry)."""
    n = len(cls)
    vf = UNIT_SQUARE_VERTS[np.asarray(cls) % NF]
    ef = UNIT_SQUARE_VERTS[(np.asarray(cls) + 1) % NF] - vf
    nrm = np.stack([ef[:, 1], -ef[:, 0]], 1)                 # outward normal (counter-clockwise vertices)
    p0 = np.stack([x0, y0], 1)
    rr = np.maximum(np.sqrt(u0 * u0 + v0 * v0), 1e-12)
    d = np.stack([u0, v0], 1) / rr[:, None]
    dn = (d * nrm).sum(1)
    gap = ((vf - p0) * nrm).sum(1)                           # distance from the entry point to the face line (>= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dn > 1e-9, gap / dn, 0.0)
    q = p0 + t[:, None] * d
    s_ref = np.clip(((q - vf) * ef).sum(1), REF_S_CLIP, 1 - REF_S_CLIP)
    mu_n, mu_t = direction_to_face_frame_2d(UNIT_SQUARE_VERTS, np.asarray(cls), d[:, 0], d[:, 1])
    psi_ref = np.clip(np.arctan2(mu_t, mu_n) / (0.5 * np.pi), -REF_PSI_CLIP, REF_PSI_CLIP) * 0.5 * np.pi
    assert len(s_ref) == n
    return s_ref, psi_ref


SCALED_CLASS = 1      # scaled encoding: only exits through the face opposite the canonical entry (through-going)


def _scaled_map(cls, ref, scale):
    """(mask of scaled rows, s_ref, psi_ref, position factor, angle factor) of the scaled-relative encoding."""
    s_ref, psi_ref = exit_reference(cls, *ref)
    m = np.asarray(cls) == SCALED_CLASS
    fs = s_ref * (1.0 - s_ref) / scale          # logit difference x s_ref (1 - s_ref) ~ face-coordinate difference
    fp = 0.5 * np.pi / scale                    # atanh difference x pi/2 ~ angle difference
    return m, s_ref, psi_ref, fs, fp


def exit_targets(cls, x1, y1, u1, v1, E0, E1, E_cut, ref=None, scale=None):
    """(s, psi, r, lambda) encoded for the exit flow (exits only). ref = (x0, y0, u0, v0) of the entry: encode s and
    psi relative to the straight-line continuation (exit_reference) instead of absolutely.
    scale (per-row scattering width sigma(E), needs ref): through-going exits (cls == SCALED_CLASS) are encoded
    relative to the ray AND divided by sigma(E), so every energy's exit distribution is O(1) wide in the flow's
    standardized space; the other exit faces keep the absolute encoding (scaled, they would be far-out outliers).
    Why: with global standardization a 20 MeV through-going exit is ~0.1 wide in the flow's units, so the network's
    absolute error floor (~1e-3) was 1-2 % of its width, with a y0-dependent part a forward beam accumulates
    (Drumm benchmark, 2026-11-16)."""
    pts = np.stack([x1, y1], 1)
    s = polygon_face_coordinate(UNIT_SQUARE_VERTS, cls, pts)
    mu_n, mu_t = direction_to_face_frame_2d(UNIT_SQUARE_VERTS, cls, u1, v1)
    r = np.sqrt(u1 * u1 + v1 * v1)
    psi = np.arctan2(mu_t, np.maximum(mu_n, 1e-12))      # (-pi/2, pi/2): always outgoing
    lam = (E0 - E1) / np.maximum(E0 - E_cut, 1e-300)
    zs = logit(s)
    zp = np.arctanh(np.clip(psi / (0.5 * np.pi), -1 + EPS, 1 - EPS))
    if scale is not None:
        m, s_ref, psi_ref, fs, fp = _scaled_map(cls, ref, scale)
        zs = np.where(m, (zs - logit(s_ref)) * fs, zs)
        zp = np.where(m, (zp - np.arctanh(psi_ref / (0.5 * np.pi))) * fp, zp)
    elif ref is not None:
        s_ref, psi_ref = exit_reference(cls, *ref)
        zs = zs - logit(s_ref)
        zp = zp - np.arctanh(psi_ref / (0.5 * np.pi))
    return np.stack([zs, zp, logit(r), logit(lam, 1e-12)], 1)


def decode_exit(z, cls, E0, E_cut, ref=None, scale=None):
    zs, zp = z[:, 0], z[:, 1]
    if scale is not None:
        m, s_ref, psi_ref, fs, fp = _scaled_map(cls, ref, scale)
        zs = np.where(m, zs / fs + logit(s_ref), zs)
        zp = np.where(m, zp / fp + np.arctanh(psi_ref / (0.5 * np.pi)), zp)
    elif ref is not None:
        s_ref, psi_ref = exit_reference(cls, *ref)
        zs = zs + logit(s_ref)
        zp = zp + np.arctanh(psi_ref / (0.5 * np.pi))
    s = sigmoid(zs)
    psi = 0.5 * np.pi * np.tanh(zp)
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


# Boundary (entry) rows: the entry face x = 0 is kept only by the identity and the y-mirror y -> 1 - y, v -> -v, which
# is D4 element 6 (half turn, then x -> 1 - x). Without the mirrored copies the entry models are only approximately
# y-symmetric: in the Drumm cylinder benchmark (2026-11-12) that drifted a 20 MeV beam sideways by ~90 um over 2 cm.
K_YMIRROR = 6
BOUNDARY_SYM = (0, K_YMIRROR)


def sym_set(kind, augment):
    """D4 elements used to augment rows of this kind."""
    if not augment:
        return (0,)
    return BOUNDARY_SYM if kind == "boundary" else tuple(range(8))


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
    Size-sweep files (ElectronTransport/generate_size_sweep.py) carry the cell side of every segment (seg_L): the
    spec is then size-conditioned (size_cond, log L bounds, CSDA range table) and T["L"] holds it per segment
    (for one-size data T["L"] is that size on every row).
    """
    d = np.load(path, allow_pickle=False)
    if "segments" not in d.files or not bool(d.get("simulated_only", False)):
        raise ValueError(f"{path}: needs a generate_analog_batches.py --simulated_only dataset")
    cols = [str(c) for c in d["segment_columns"]]
    scol = [str(c) for c in d["secondary_columns"]]
    c, e = cols.index, scol.index
    seg, sec = d["segments"], d["secondaries"]
    sweep = "seg_L" in d.files
    L = d["seg_L"] if sweep else np.full(len(seg), float(d["box_dims"][0]))
    E_cut = float(d["E_cutoff"])
    E_hi = float(d["inputs"][:, 4].max())
    spec = Spec(L=float("nan") if sweep else float(L[0]), E_cut=E_cut, logE_lo=math.log10(E_cut),
                logE_hi=math.log10(max(E_hi, 10 * E_cut)) + 1e-9, NR=int(NR),
                material_density=float(np.atleast_1d(d["density"])[0]))
    if sweep:
        lo, hi = (float(v) for v in d["L_range"])
        cE, cR = d["csda_E"], d["csda_R"]
        spec.size_cond, spec.logL_lo, spec.logL_hi = True, math.log10(lo), math.log10(hi)
        spec.csda_logE, spec.csda_logR = np.log10(cE[1:]).tolist(), np.log10(cR[1:]).tolist()
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

    T = {"spec": spec, "L": L, "x0": x0, "y0": y0, "u0": u0, "v0": v0, "E0": E0, "x1": x1, "y1": y1, "u1": u1, "v1": v1,
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
    T["dau"] = {"parent": p, "j": j, "x": sec[:, e("x")] / L[p], "y": sec[:, e("y")] / L[p], "u": sec[:, e("u")],
                "v": sec[:, e("v")], "w": sec[:, e("w")] * sgn[p], "E": Ed, "B": B}
    print(f"Loaded {path}: {int(boundary.sum())} boundary transits, {int(interior.sum())} relevant-secondary "
          f"transits, {len(p)} relevant daughters; photon flag on {flag[keep].mean():.4f} of transits; "
          f"n_rel max {int(n_rel[keep].max())} (cap {NR})")
    return T


def _aug_interior(T, rows, k):
    """D4-transformed copies of segment rows: entry, exit and class (boundary rows: k in BOUNDARY_SYM only)."""
    x0, y0, u0, v0 = d4(k, T["x0"][rows], T["y0"][rows], T["u0"][rows], T["v0"][rows])
    x1, y1, u1, v1 = d4(k, T["x1"][rows], T["y1"][rows], T["u1"][rows], T["v1"][rows])
    cls = T["cls"][rows].copy()
    ex = cls < NF
    cls[ex] = exit_class_from_point(x1[ex], y1[ex])
    return x0, y0, u0, v0, x1, y1, u1, v1, cls


def classifier_data(T, kind, augment=True):
    spec = T["spec"]
    r = np.flatnonzero(T[kind])
    conds, labs = [], []
    for k in sym_set(kind, augment):
        x0, y0, u0, v0, _, _, _, _, cls = _aug_interior(T, r, k)
        conds.append(boundary_cond(spec, y0, u0, v0, T["E0"][r], T["L"][r]) if kind == "boundary"
                     else interior_cond(spec, x0, y0, u0, v0, T["E0"][r], T["L"][r]))
        labs.append(spec.encode(cls, T["n_rel"][r], T["flag"][r]))
    return np.concatenate(conds), np.concatenate(labs)


def exit_flow_data(T, kind, augment=True, rel=False, scaled=False, oversample=None):
    """rel: ray-relative position / angle targets (exit_reference); scaled: through-going exits ray-relative and
    divided by the scattering width sigma(E) (exit_targets).

    oversample = (k_flag, k_dau, k_edge): training multiplicity of rows with the photon flag, with >= 1 relevant
    daughter, and (boundary only) entering within EDGE_BAND of a lateral face; a row takes the largest that applies.
    All three are conditioning inputs, so the learned conditional is unchanged. Returns (cond, target, rep or None)."""
    spec = T["spec"]
    rows = np.flatnonzero(T[kind] & (T["cls"] < NF))
    conds, tgts, reps = [], [], []
    for k in sym_set(kind, augment):
        x0, y0, u0, v0, x1, y1, u1, v1, cls = _aug_interior(T, rows, k)
        base = (boundary_cond(spec, y0, u0, v0, T["E0"][rows], T["L"][rows]) if kind == "boundary"
                else interior_cond(spec, x0, y0, u0, v0, T["E0"][rows], T["L"][rows]))
        extra = np.concatenate([np.eye(NF, dtype=np.float32)[cls],
                                (np.minimum(T["n_rel"][rows], spec.NR) / spec.NR).reshape(-1, 1),
                                T["flag"][rows].reshape(-1, 1)], 1).astype(np.float32)
        conds.append(np.concatenate([base, extra], 1))
        if oversample is not None:
            kf, kd, ke = oversample
            r_ = np.ones(len(rows), np.int64)
            r_ = np.where(T["flag"][rows] == 1, np.maximum(r_, kf), r_)
            r_ = np.where(T["n_rel"][rows] >= 1, np.maximum(r_, kd), r_)
            if kind == "boundary":
                r_ = np.where(np.minimum(y0, 1 - y0) < EDGE_BAND, np.maximum(r_, ke), r_)
            reps.append(r_)
        tgts.append(exit_targets(cls, x1, y1, u1, v1, T["E0"][rows], T["E_out"][rows], spec.E_cut,
                                 ref=(x0, y0, u0, v0) if (rel or scaled) else None,
                                 scale=scatter_width(spec, T["E0"][rows], T["L"][rows], "exit") if scaled else None))
    return (np.concatenate(conds), np.concatenate(tgts).astype(np.float32),
            np.concatenate(reps) if oversample is not None else None)


def _parent_block(T, prow, k):
    """Parent features + class one-hot + n + lambda for parent segment rows prow under symmetry k. Returns the block
    and the mask of rows the symmetry applies to (interior parents: all k; boundary parents: k in BOUNDARY_SYM)."""
    spec = T["spec"]
    is_int = T["interior"][prow]
    ok = is_int | (k in BOUNDARY_SYM)
    x0, y0, u0, v0 = T["x0"][prow].copy(), T["y0"][prow].copy(), T["u0"][prow].copy(), T["v0"][prow].copy()
    cls = T["cls"][prow].copy()
    if k:
        xa, ya, ua, va = d4(k, x0, y0, u0, v0)
        x0, y0, u0, v0 = np.where(ok, xa, x0), np.where(ok, ya, y0), np.where(ok, ua, u0), np.where(ok, va, v0)
        x1, y1, _, _ = d4(k, T["x1"][prow], T["y1"][prow], T["u1"][prow], T["v1"][prow])
        ex = ok & (cls < NF)
        cls[ex] = exit_class_from_point(x1[ex], y1[ex])
    E0 = T["E0"][prow]
    lam = (E0 - T["E_out"][prow]) / np.maximum(E0 - spec.E_cut, 1e-300)
    pf = parent_feats(spec, is_int.astype(np.float32), x0, y0, u0, v0, E0, T["L"][prow])
    blk = np.concatenate([pf, np.eye(N_EXIT, dtype=np.float32)[cls],
                          (np.minimum(T["n_rel"][prow], spec.NR) / spec.NR).reshape(-1, 1),
                          logit(lam, 1e-12).reshape(-1, 1) / 10.0], 1).astype(np.float32)
    return blk, ok


def daughter_flow_data(T, augment=True):
    spec = T["spec"]
    D = T["dau"]
    p = D["parent"]
    conds, tgts = [], []
    for k in (range(8) if augment else [0]):
        blk, ok = _parent_block(T, p, k)
        x, y, u, v = D["x"], D["y"], D["u"], D["v"]
        if k:
            xa, ya, ua, va = d4(k, x, y, u, v)
            x, y, u, v = np.where(ok, xa, x), np.where(ok, ya, y), np.where(ok, ua, u), np.where(ok, va, v)
            if not ok.any():
                continue
            sel = ok                                   # boundary parents: identity and y-mirror only
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
        blk, ok = _parent_block(T, rows, k)
        sel = np.ones(len(rows), bool) if k == 0 else ok
        if not sel.any():
            continue
        conds.append(np.concatenate([blk, spec.e_feat(avail).reshape(-1, 1).astype(np.float32)], 1)[sel])
        tgts.append(logit(g, 1e-9).reshape(-1, 1)[sel])
    return np.concatenate(conds), np.concatenate(tgts).astype(np.float32)


def stop_flow_data(T, augment=True):
    """Absorbed transits: parent features + sampled n_rel / photon flag -> stop point (logit x1, logit y1).
    Interior rows get all D4 copies, boundary rows the y-mirror (BOUNDARY_SYM)."""
    spec = T["spec"]
    rows = np.flatnonzero((T["boundary"] | T["interior"]) & (T["cls"] == ABSORBED))
    is_int = T["interior"][rows]
    conds, tgts = [], []
    for k in (range(8) if augment else [0]):
        sel = is_int | (k in BOUNDARY_SYM)
        if not sel.any():
            continue
        r = rows[sel]
        x0, y0, u0, v0 = d4(k, T["x0"][r], T["y0"][r], T["u0"][r], T["v0"][r])
        x1, y1, _, _ = d4(k, T["x1"][r], T["y1"][r], T["u0"][r], T["v0"][r])
        conds.append(stop_cond(spec, is_int[sel].astype(np.float32), x0, y0, u0, v0, T["E0"][r],
                               T["n_rel"][r], T["flag"][r], T["L"][r]))
        tgts.append(np.stack([logit(x1), logit(y1)], 1))
    return np.concatenate(conds), np.concatenate(tgts).astype(np.float32)


def stop_cond(spec, start, x0, y0, u0, v0, E0, n_rel, flag, L=None):
    return np.concatenate([parent_feats(spec, start, x0, y0, u0, v0, E0, L),
                           (np.minimum(n_rel, spec.NR) / spec.NR).reshape(-1, 1),
                           np.asarray(flag, np.float32).reshape(-1, 1)], 1).astype(np.float32)


# ====================================================================================================================
# Flow training / sampling
# ====================================================================================================================
def train_flow(cond, target, device, hidden=256, depth=6, expansion=2, epochs=20, batch=8192, lr=1e-3,
               val_frac=0.05, seed=0, label="flow", max_steps=None, rep=None):
    """Conditional flow matching (condOT path, ml/model.py network) on standardized targets.

    rep: optional per-row integer multiplicity of the TRAINING rows (validation rows keep weight 1). Use it only with
    a function of the conditioning vector: it then reweights where the network spends capacity without changing the
    learned conditional distribution p(target | cond)."""
    g = torch.Generator().manual_seed(seed)
    n = len(cond)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    mu, sd = target.mean(0), target.std(0) + 1e-6
    X = torch.as_tensor(np.concatenate([cond, (target - mu) / sd], 1), dtype=torch.float32)
    tr, va = perm[n_val:], perm[:n_val]
    if rep is not None:
        tr = tr.repeat_interleave(torch.as_tensor(rep, dtype=torch.long)[tr])
        print(f"  [{label}] oversampled training rows: {n - n_val} -> {len(tr)}", flush=True)
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
                          "history": m.get("history", []), "rel": bool(m.get("rel", False)),
                          "scaled": bool(m.get("scaled", False))}
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
                            "cond_dim": b["cond_dim"], "target_dim": b["target_dim"], "arch": a,
                            "history": b.get("history", []), "rel": bool(b.get("rel", False)),
                            "scaled": bool(b.get("scaled", False))}
    return spec, models


# ====================================================================================================================
# Kernel: one cell transit for a batch of electrons
# ====================================================================================================================
class DaughterCalib:
    """Inference-time calibration of the relevant-daughter channel (ml/gmc_calib.py).

    Relevant daughters are rare (a few % of transits), so the joint classifier and the daughter flow can be off by a
    few % in an (E, L) slice at a negligible loss cost; chained, that moves the wide-angle delta-ray dose (Drumm wings).
    Tables of logit offsets on a (log10 E, log10 L) grid of bin centres, bilinear and clamped at the edges:
        nrel_B / nrel_I   added to the classifier logits of every class with n_rel >= 1 (boundary / interior)
        edau_B / edau_I   added to the daughter energy-share logit (z[:, 5]), by the parent's E0 and L
    """

    def __init__(self, src):
        d = dict(np.load(src)) if isinstance(src, (str, os.PathLike)) else dict(src)
        self.logE, self.logL = np.asarray(d["logE"], float), np.asarray(d["logL"], float)
        self.tab = {k: np.asarray(d[k], float) for k in ("nrel_B", "nrel_I", "edau_B", "edau_I") if k in d}

    def offset(self, key, E, L):
        t = self.tab.get(key)
        if t is None or E is None or len(E) == 0:
            return None
        lE = np.log10(np.maximum(np.asarray(E, float), 1e-300))
        lL = np.full_like(lE, self.logL[0]) if L is None else np.log10(np.asarray(L, float))

        def frac(x, g):
            i = np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2)
            return i, np.clip((x - g[i]) / (g[i + 1] - g[i]), 0.0, 1.0)
        i, a = frac(lE, self.logE)
        j, b = frac(lL, self.logL)
        return ((1 - a) * (1 - b) * t[i, j] + a * (1 - b) * t[i + 1, j] + (1 - a) * b * t[i, j + 1]
                + a * b * t[i + 1, j + 1])


class SecondaryGMC:
    """Samples cell transits. boundary(): canonical frame (entering through the left face). interior(): cell frame.

    Both return {"cls", "x", "y", "u", "v", "w", "E_out", "dose", "daughters"}; positions in the unit cell.
    daughters = {"parent", "x", "y", "u", "v", "w", "E"} (relevant secondaries to follow as interior particles).
    secondaries=False gives the primary-only GMC from the same models: daughters are not emitted and their energy is
    deposited in the cell (the n_rel / photon structure is still sampled, so this isolates the secondary transport).
    ode_steps: midpoint steps per flow sample. Default 20 since 2026-11-17 (Drumm sweep, water 1 mm _x2: 20 steps as
    accurate as 50 / 100 on the chained 125 um map, flat-beam test and single-cell check, 2.6x cheaper chain); results
    made before that used 50. Not yet re-checked for the 100 um Cu / Al / water models.
    """

    def __init__(self, path, device="cpu", ode_steps=20, ode_method="midpoint", seed=0, secondaries=True,
                 calib=None):
        """calib: optional daughter calibration (.npz from ml/gmc_calib.py, or its dict): logit offsets on (log10 E,
        log10 L) of the classifier classes with n_rel >= 1 and of the daughter energy share (see DaughterCalib)."""
        self.spec, self.m = load_models(path, device)
        self.device, self.steps, self.method = device, ode_steps, ode_method
        self.gen = torch.Generator().manual_seed(seed)                       # ODE noise (CPU)
        self.cgen = torch.Generator(device=device).manual_seed(seed + 7777)  # class sampling (model device)
        self.rng = np.random.default_rng(seed)
        self.secondaries = secondaries
        self.calib = None if calib is None else DaughterCalib(calib)

    def _classify(self, name, cond, E=None, L=None):
        clf = self.m[name]["model"]
        c = torch.as_tensor(cond, device=self.device)
        off = None if self.calib is None else self.calib.offset("nrel_" + name[-1], E, L)
        if off is None:
            lab = clf.sample(c, generator=self.cgen).cpu().numpy()
        else:
            nr_of = torch.as_tensor(self.spec.decode(np.arange(clf.cfg.num_classes))[1] >= 1, device=self.device)
            o = torch.as_tensor(off, dtype=torch.float32, device=self.device)
            labs = []
            with torch.no_grad():
                for i in range(0, len(c), 65536):
                    lg = clf(c[i:i + 65536]) + o[i:i + 65536, None] * nr_of[None, :]
                    labs.append(torch.multinomial(torch.softmax(lg, -1), 1, generator=self.cgen).squeeze(1))
            lab = (torch.cat(labs) if labs else torch.zeros(0, dtype=torch.long)).cpu().numpy()
        return self.spec.decode(lab)

    def _flow(self, name, cond):
        return sample_flow(self.m[name], cond, self.device, self.steps, self.method, self.gen)

    def _size(self, L, n):
        if not getattr(self.spec, "size_cond", False):
            return None
        if L is None:
            raise ValueError("size-conditioned GMC: pass the cell side L (cm, scalar or per electron)")
        return np.broadcast_to(np.asarray(L, np.float64), (n,)).copy()

    def boundary(self, y, u, v, w, E, L=None):
        """L: cell side (cm), scalar or per electron; required by a size-conditioned model, ignored otherwise."""
        n = len(E)
        L = self._size(L, n)
        return self._transit("B", np.zeros(n), y, u, v, w, E, boundary_cond(self.spec, y, u, v, E, L), L)

    def interior(self, x, y, u, v, w, E, L=None):
        L = self._size(L, len(E))
        return self._transit("I", x, y, u, v, w, E, interior_cond(self.spec, x, y, u, v, E, L), L)

    def _transit(self, which, x0, y0, u0, v0, w0, E0, cond, L=None):
        spec = self.spec
        n = len(E0)
        Lr = (lambda r: None) if L is None else (lambda r: L[r])  # noqa: E731
        cls, nr, flag = self._classify("clf_" + which, cond, E0, L)
        out = {k: np.zeros(n) for k in ("x", "y", "u", "v", "w", "E_out", "dose")}
        out["cls"] = cls
        sgn = np.where(w0 < 0, -1.0, 1.0)
        ab = np.flatnonzero(cls == ABSORBED)
        if ab.size and "stop" in self.m:     # stop point of absorbed electrons (charge tally); else left at 0
            start = START_BOUNDARY if which == "B" else START_INTERIOR
            z = self._flow("stop", stop_cond(spec, start, x0[ab], y0[ab], u0[ab], v0[ab], E0[ab], nr[ab], flag[ab],
                                             Lr(ab)))
            out["x"][ab], out["y"][ab] = sigmoid(z[:, 0]), sigmoid(z[:, 1])
        ex = np.flatnonzero(cls < NF)
        if ex.size:
            extra = np.concatenate([np.eye(NF, dtype=np.float32)[cls[ex]], (nr[ex] / spec.NR).reshape(-1, 1),
                                    flag[ex].reshape(-1, 1)], 1).astype(np.float32)
            z = self._flow("exit_" + which, np.concatenate([cond[ex], extra], 1))
            fm = self.m["exit_" + which]
            ref = (x0[ex], y0[ex], u0[ex], v0[ex]) if (fm.get("rel") or fm.get("scaled")) else None
            scale = scatter_width(spec, E0[ex], Lr(ex), "exit") if fm.get("scaled") else None
            x1, y1, u1, v1, r1, E1 = decode_exit(z, cls[ex], E0[ex], spec.E_cut, ref=ref, scale=scale)
            out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex], out["E_out"][ex] = x1, y1, u1, v1, E1
            out["w"][ex] = self.rng.choice([-1.0, 1.0], ex.size) * np.sqrt(np.clip(1 - r1 * r1, 0, 1))
        budget = E0 - out["E_out"]

        # Daughters, sequentially (energy shares of what is left)
        start = START_BOUNDARY if which == "B" else START_INTERIOR
        lam = budget / np.maximum(E0 - spec.E_cut, 1e-300)
        pblk = np.concatenate([parent_feats(spec, start, x0, y0, u0, v0, E0, L), np.eye(N_EXIT, dtype=np.float32)[cls],
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
            off = None if self.calib is None else self.calib.offset("edau_" + which, E0[rows],
                                                                     None if L is None else L[rows])
            f = sigmoid(z[:, 5] if off is None else z[:, 5] + off)
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
        out["n_rel"] = nr            # sampled relevant-daughter count and photon flag (SCR kernel conditioning)
        out["flag"] = flag
        return out


# ====================================================================================================================
# CLI
# ====================================================================================================================
def cmd_train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed:
        torch.manual_seed(args.seed)       # flow-matching noise / times (seed 0 keeps the old unseeded behaviour)
    T = load_tables(args.train_npz, NR=args.NR, max_rows=args.max_rows)
    spec = T["spec"]
    spec.edge_feats = bool(args.edge_feats)
    spec.scale_feats = bool(args.scale_feats)
    if spec.scale_feats or args.exit_scaled:
        set_scatter_table(spec, T)
    models, info = {}, {}
    only = set(args.only.split(",")) if args.only else None
    prev = None
    if only is not None and os.path.exists(os.path.join(args.out, "secondary_gmc.pth")):
        prev_spec, prev = load_models(os.path.join(args.out, "secondary_gmc.pth"), dev)
        for flag in ("edge_feats", "scale_feats"):
            if bool(getattr(prev_spec, flag, False)) != getattr(spec, flag) and not {"clf_B", "exit_B"} <= only:
                raise ValueError(f"--{flag} changes the boundary condition vector: retrain clf_B and exit_B together "
                                 "(--only clf_B,exit_B,...) or match the existing checkpoint's setting")

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
            scaled = args.exit_scaled and kind == "boundary"
            os_ = None if args.exit_oversample is None else tuple(int(v) for v in args.exit_oversample.split(","))
            cond, tgt, rep_ = exit_flow_data(T, kind, rel=args.exit_rel, scaled=scaled, oversample=os_)
            print(f"\n== {name}: {len(tgt)} rows, cond {cond.shape[1]}, target {tgt.shape[1]}"
                  f"{' (ray-relative s, psi)' if args.exit_rel else ''}"
                  f"{' (through-going exits ray-relative / sigma(E))' if scaled else ''}")
            models[name] = {"kind": "flow", **train_flow(cond, tgt, dev, epochs=args.flow_epochs, label=name,
                                                         max_steps=args.max_steps, hidden=args.exit_hidden,
                                                         depth=args.exit_depth, rep=rep_, seed=args.seed),
                            "rel": args.exit_rel, "scaled": scaled}
            info[name] = {"rows": int(len(tgt)), "oversample": args.exit_oversample, "seed": args.seed}
    if want("daughter"):
        cond, tgt = daughter_flow_data(T)
        print(f"\n== daughter: {len(tgt)} rows (D4 copies of interior parents, y-mirror of boundary parents), cond {cond.shape[1]}")
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
    if want("stop") and (only is not None or args.stop):
        cond, tgt = stop_flow_data(T)
        print(f"\n== stop: {len(tgt)} rows (absorbed transits; D4 / y-mirror copies), cond {cond.shape[1]}")
        models["stop"] = {"kind": "flow", **train_flow(cond, tgt, dev, epochs=args.flow_epochs, label="stop",
                                                       max_steps=args.max_steps)}
        info["stop"] = {"rows": int(len(tgt))}
    if prev is not None:
        for k, v in prev.items():
            models.setdefault(k, v)
    save_models(args.out, spec, models)
    info_path = os.path.join(args.out, "train_info.json")
    if prev is not None and os.path.exists(info_path):
        info = {**json.load(open(info_path)).get("models", {}), **info}
    with open(info_path, "w") as f:
        json.dump({"spec": vars(spec), "train_npz": args.train_npz, "models": info}, f, indent=2)
    print(f"\nSaved {os.path.join(args.out, 'secondary_gmc.pth')}")


MATERIAL_BY_DENSITY = {8.96: "Cu", 2.699: "Al", 1.0: "water"}  # g/cm^3 as stored by ElectronTransport/NIST_EPICS_convert.py


def describe(spec):
    """'100 µm Cu, 1 keV – 10 MeV' from a Spec or its dict (material named from its density)."""
    if isinstance(spec, dict):
        spec = SimpleNamespace(**spec)
    rho = float(getattr(spec, "material_density", float("nan")))
    mat = next((n for d, n in MATERIAL_BY_DENSITY.items() if abs(d - rho) < 1e-3), f"ρ = {rho:g} g/cm³")
    lo, hi = 10 ** spec.logE_lo, 10 ** spec.logE_hi
    fmt = lambda E: f"{E * 1e3:.3g} keV" if E < 1 else f"{E:.3g} MeV"  # noqa: E731
    if getattr(spec, "size_cond", False):
        size = f"{10 ** spec.logL_lo * 1e4:g} µm – {10 ** spec.logL_hi * 1e4:g} µm"
    else:
        size = f"{spec.L * 1e4:g} µm"
    return f"{size} {mat}, {fmt(lo)} – {fmt(hi)}"


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
    t.add_argument("--exit_hidden", type=int, default=256, help="hidden width of the exit flows (exit_B, exit_I)")
    t.add_argument("--exit_depth", type=int, default=6, help="residual blocks of the exit flows (exit_B, exit_I)")
    t.add_argument("--only", default=None, help="comma list of models to (re)train; others kept from --out")
    t.add_argument("--stop", action="store_true", help="also train the stop-point flow (charge tally) in a full run")
    t.add_argument("--seed", type=int, default=0,
                   help="exit flows: split / shuffle seed, and torch.manual_seed when nonzero (candidate runs)")
    t.add_argument("--exit_oversample", default=None, metavar="KF,KD,KE",
                   help="exit flows: training multiplicity of photon-flagged rows, rows with >= 1 relevant daughter, "
                        "and boundary entries within EDGE_BAND = 1/16 cell of a lateral face (e.g. 4,2,2). These are "
                        "conditioning inputs, so this shifts capacity without biasing the conditional. Added "
                        "2026-11-18: the ESTAR water exit flow under-sampled large losses on flagged transits "
                        "(photon energy -8 %) and pulled near-face entries inward")
    t.add_argument("--exit_rel", action="store_true",
                   help="exit flows learn position / angle relative to the entry ray's straight-line continuation "
                        "(exit_reference); stored per model, so old checkpoints keep the absolute encoding")
    t.add_argument("--edge_feats", action="store_true",
                   help="boundary models also see the log distance of the entry point to the lateral faces "
                        "(edge_feats; stored in the spec, affects clf_B and exit_B)")
    t.add_argument("--scale_feats", action="store_true",
                   help="boundary models also see the distance to the lateral faces in units of the per-transit "
                        "scattering width sigma(E) measured from the training data (scale_feats; affects clf_B, exit_B)")
    t.add_argument("--exit_scaled", action="store_true",
                   help="boundary exit flow: through-going exits encoded relative to the entry ray and divided by "
                        "sigma(E) (exit_targets); stored per model")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "plot_history":
        cmd_plot_history(args)


if __name__ == "__main__":
    main()
