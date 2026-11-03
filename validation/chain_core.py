"""
Cell-to-cell chaining of a single-cell transport kernel over a 2D grid of square cells (numpy only).

The loop is kernel-agnostic: a GMC flow, the exit classifier + face-conditioned flow, or the analog Monte Carlo run on
one cell can all be chained by the same code, so differences between their dose maps come from the kernels only.
It imports nothing beyond numpy, so it runs both in the GMC environment (torch) and in the ElectronTransport
environment (numba / h5py).

Kernel contract, all arrays in the canonical frame of the cell (particle entering through the left face x = 0,
coordinates normalized to the cell size):
    kernel(y_c, u_c, v_c, w, E, step) -> dict with
        cls   int    exit class (ml/cell_topology.py): 0 bottom, 1 right, 2 top, 3 left, 4 absorbed
        x, y  float  exit position in [0, 1]^2 (ignored when absorbed)
        u, v, w      exit direction (w is the out-of-plane cosine; ignored when absorbed)
        E_out float  exit energy (ignored when absorbed)
The energy deposited in the cell is E - E_out for an exit and E for an absorbed particle (GMC convention: the
primary's whole energy loss is charged to the cell it was crossing).
"""
import numpy as np

ABSORBED = 4
# Entry-face convention of the chaining loop (same as Macro_field.py): 0 = left (x=0), 1 = right, 2 = bottom, 3 = top
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3


def transform_to_canonical(x_local, y_local, u, v, face_idx):
    """Map cell entry states on any face to the canonical left-face frame (x'_in = 0).
    face_idx: 0=Left (x=0), 1=Right (x=1), 2=Bottom (y=0), 3=Top (y=1)"""
    y_c, u_c, v_c = np.zeros_like(y_local), np.zeros_like(u), np.zeros_like(v)
    m0, m1, m2, m3 = (face_idx == 0), (face_idx == 1), (face_idx == 2), (face_idx == 3)
    y_c[m0], u_c[m0], v_c[m0] = y_local[m0], u[m0], v[m0]
    y_c[m1], u_c[m1], v_c[m1] = y_local[m1], -u[m1], v[m1]
    y_c[m2], u_c[m2], v_c[m2] = x_local[m2], v[m2], u[m2]
    y_c[m3], u_c[m3], v_c[m3] = x_local[m3], -v[m3], u[m3]
    return np.zeros_like(x_local), y_c, u_c, v_c


def transform_from_canonical(x_out_c, y_out_c, u_out_c, v_out_c, face_idx):
    """Invert transform_to_canonical for exit states."""
    x_out, y_out = np.zeros_like(x_out_c), np.zeros_like(y_out_c)
    u_out, v_out = np.zeros_like(u_out_c), np.zeros_like(v_out_c)
    m0, m1, m2, m3 = (face_idx == 0), (face_idx == 1), (face_idx == 2), (face_idx == 3)
    x_out[m0], y_out[m0], u_out[m0], v_out[m0] = x_out_c[m0], y_out_c[m0], u_out_c[m0], v_out_c[m0]
    x_out[m1], y_out[m1], u_out[m1], v_out[m1] = 1.0 - x_out_c[m1], y_out_c[m1], -u_out_c[m1], v_out_c[m1]
    x_out[m2], y_out[m2], u_out[m2], v_out[m2] = y_out_c[m2], x_out_c[m2], v_out_c[m2], u_out_c[m2]
    x_out[m3], y_out[m3], u_out[m3], v_out[m3] = y_out_c[m3], 1.0 - x_out_c[m3], v_out_c[m3], -u_out_c[m3]
    return x_out, y_out, u_out, v_out


def make_source(N, y_max, E_low, E_high, v_spread=1.0, w_max_angle_deg=90.0, beam=None, seed=0):
    """Source on the left boundary x = 0, columns (x, y, u, v, E, path, status, w) as ElectronMC.generate_particles.

    Directions and energies follow generate_particles exactly (v ~ U(-v_spread, v_spread), the rest split between u
    and w with tilt beta ~ U(0, w_max_angle)), so with v_spread = 1, w_max_angle = 90 this is the GMC training source.
    beam: optional (y_lo, y_hi) in the same units as y_max; default the full height.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros((N, 8))
    y_lo, y_hi = (0.0, y_max) if beam is None else beam
    p[:, 1] = rng.uniform(y_lo, y_hi, N)
    v = rng.uniform(-v_spread, v_spread, N)
    rem = np.sqrt(np.maximum(0.0, 1.0 - v ** 2))
    beta = rng.uniform(0.0, np.radians(w_max_angle_deg), N)
    w_sign = rng.choice([-1.0, 1.0], N)
    p[:, 3] = v
    p[:, 2] = rem * np.cos(beta)
    p[:, 7] = w_sign * rem * np.sin(beta)
    p[:, 4] = rng.uniform(E_low, E_high, N)
    return p


def chain(kernel, source, nx, ny, cell_size, max_steps=2000, E_cutoff=0.0, log=print, log_every=10):
    """Chain ``kernel`` over an nx x ny grid of square cells of side ``cell_size``; ``source`` from make_source
    (positions in the same length unit as cell_size, entering through the grid's left boundary).

    Returns dict: dose [nx, ny] (energy per cell, summed over primaries), escaped energy per grid side, energy still
    in flight when max_steps ran out, number of steps and cell transits.
    """
    N = source.shape[0]
    cx = np.zeros(N, dtype=np.int64)
    cy = np.clip(np.floor(source[:, 1] / cell_size).astype(np.int64), 0, ny - 1)
    xl = np.zeros(N)
    yl = np.clip(source[:, 1] / cell_size - cy, 0.0, 1.0)
    u, v, w, E = source[:, 2].copy(), source[:, 3].copy(), source[:, 7].copy(), source[:, 4].copy()
    face = np.full(N, LEFT, dtype=np.int64)
    active = np.ones(N, dtype=bool)
    dose = np.zeros((nx, ny))
    escaped = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    transits = 0

    step = 0
    for step in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        transits += idx.size
        _, y_c, u_c, v_c = transform_to_canonical(xl[idx], yl[idx], u[idx], v[idx], face[idx])
        out = kernel(y_c, u_c, v_c, w[idx], E[idx], step)
        cls = np.asarray(out["cls"])
        absorbed = cls == ABSORBED
        E_out = np.where(absorbed, 0.0, out["E_out"])
        np.add.at(dose, (cx[idx], cy[idx]), E[idx] - E_out)
        active[idx[absorbed]] = False

        ex = ~absorbed
        j = idx[ex]
        x_p, y_p, u_p, v_p = transform_from_canonical(out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex],
                                                      face[j])
        # Which physical face: nearest one. The face-conditioned and MC kernels put the point exactly on the face;
        # for the unconstrained baseline flow this is the old Macro_field rule.
        f = np.argmin(np.stack([np.abs(x_p), np.abs(1 - x_p), np.abs(y_p), np.abs(1 - y_p)], 1), 1)
        # Leave through face f -> enter the neighbour through the opposite face
        dx_ = np.array([-1, 1, 0, 0])[f]
        dy_ = np.array([0, 0, -1, 1])[f]
        cx[j] += dx_
        cy[j] += dy_
        face[j] = np.array([RIGHT, LEFT, TOP, BOTTOM])[f]
        horiz = f < 2
        xl[j] = np.where(horiz, np.where(f == 0, 1.0, 0.0), np.clip(x_p, 0, 1))
        yl[j] = np.where(horiz, np.clip(y_p, 0, 1), np.where(f == 2, 1.0, 0.0))
        u[j], v[j], w[j], E[j] = u_p, v_p, out["w"][ex], out["E_out"][ex]

        gone = (cx[j] < 0) | (cx[j] >= nx) | (cy[j] < 0) | (cy[j] >= ny)
        for name, m in (("left", cx[j] < 0), ("right", cx[j] >= nx), ("bottom", cy[j] < 0), ("top", cy[j] >= ny)):
            escaped[name] += float(E[j][m & gone].sum())
        active[j[gone]] = False
        # A particle that left a cell at (or below) the cutoff stops in the cell it enters
        dead = ~gone & (E[j] <= E_cutoff)
        np.add.at(dose, (cx[j][dead], cy[j][dead]), E[j][dead])
        active[j[dead]] = False
        if log and log_every and step % log_every == 0:
            log(f"step {step:4d}: {idx.size:7d} active, mean E {E[idx].mean():.4g}")

    in_flight = float(E[active].sum())
    return {"dose": dose, "escaped": escaped, "in_flight": in_flight, "n_active_end": int(active.sum()),
            "steps": step + 1, "transits": transits, "E_source": float(source[:, 4].sum())}
