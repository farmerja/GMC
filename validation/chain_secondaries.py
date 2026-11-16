"""
Cell-to-cell chaining with secondaries over a 2D grid of square cells (numpy only, like chain_core.py).

Particles are either BOUNDARY entries (entering a cell through a face) or INTERIOR particles (relevant secondaries
at their birth point). A kernel provides
    kernel.boundary(y_c, u_c, v_c, w, E)  canonical frame of chain_core (entering through the left face x = 0)
    kernel.interior(x, y, u, v, w, E)     cell frame
each returning a dict (positions in the unit cell, frame of the call):
    cls, x, y, u, v, w, E_out   the transported particle (cls 4 = absorbed; exit state ignored then)
    dose                        energy left in the cell by this transit (MeV)
    daughters (optional)        {"parent", "x", "y", "u", "v", "w", "E"}: new interior particles in the same cell
    extra_exits (optional)      {"parent", "x", "y", "u", "v", "w", "E"}: further particles leaving through faces
                                (the MC kernel reports escaping secondaries this way)
Exits enter the neighbour cell as boundary entries; particles leaving the grid are tallied as escaped.

Transit log (transit_log=True, for SCR-GMC sub-cell dose reconstruction, ml/subcell_kernel.py): the result gets
    "transits"  one row per kernel call, in the frame of that call (canonical for boundary entries, cell frame for
                interior starts): cx, cy, mat, start (0 boundary / 1 interior), face (entry face, -1 interior), gen,
                x0, y0, u0, v0, w0, E0 (entry), cls, x1, y1, u1, v1, w1, E_out (exit; for cls 4 x1, y1 = stop point
                when the kernel reports it), dose, n_rel, flag, dose_cond (the dose the sub-cell kernels condition on:
                the kernel's own dose; differs from dose under drumm_cylinder.LocalPhotons), Eb (birth energy of the
                electron: inf for source
                electrons, E0 of its interior start for secondaries; carried across cells, for charge thresholds)
                (n_rel / flag: the kernel's sampled relevant-daughter count and photon flag when it reports them, else
                the number of daughters it emitted and photon_E > 0)
    "points"    point deposits at a known position, cell frame: cx, cy, x, y, E, Eb (electrons entering a cell below
                the cutoff, deposited at their entry point)
Map a transit's reference-frame points to its cell with transform_from_canonical(..., face) (boundary) or identity.

Several materials: pass material_map (nx, ny) of indices into a LIST of kernels, one per material. Each step, every
particle is transported by the kernel of the material of the cell it is in. An electron crossing an interface is an
ordinary boundary entry of the neighbour, so it uses the neighbour's entry models; relevant daughters stay in their
birth cell, so they always use the interior models of the material they were born in.

MCCellKernel (ElectronTransport venv) runs the analog MC on one cell; with secondaries=True it returns every
escaping electron, which makes the chain exact (Markov property of the analog MC), so its dose must match the
full-grid MC: that is the check of this file. secondaries=False reproduces the primary-only structure (escaping
secondaries' energy charged to the cell they were born in). Photons are never deposited (as in the analog MC).
"""
import numpy as np

from chain_core import transform_to_canonical, transform_from_canonical, make_source, LEFT, RIGHT, BOTTOM, TOP

ABSORBED = 4
_DX = np.array([-1, 1, 0, 0])
_DY = np.array([0, 0, -1, 1])
_ENTRY_FACE = np.array([RIGHT, LEFT, TOP, BOTTOM])  # leaving through physical face f -> entry face in the neighbour


def log_uniform_source(N, y_max, E_low, E_high, beam=None, seed=0, **kw):
    """make_source with log-uniform energies (the training source law)."""
    src = make_source(N, y_max, E_low, E_high, beam=beam, seed=seed, **kw)
    rng = np.random.default_rng(seed + 99991)
    src[:, 4] = np.exp(rng.uniform(np.log(E_low), np.log(E_high), N))
    return src


class _Bank:
    FIELDS = ("mode", "cx", "cy", "face", "x", "y", "u", "v", "w", "E", "gen", "Eb")

    def __init__(self):
        self.d = {k: [] for k in self.FIELDS}

    def add(self, **kw):
        n = len(kw["E"])
        if n == 0:
            return
        for k in self.FIELDS:
            v = kw.get(k, 0)
            self.d[k].append(np.broadcast_to(np.asarray(v), (n,)).copy())

    def take(self):
        out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.d.items()}
        for k in ("mode", "cx", "cy", "face", "gen"):
            out[k] = out[k].astype(np.int64)
        self.d = {k: [] for k in self.FIELDS}
        return out


def chain_secondaries(kernel, source, nx, ny, cell_size, E_cutoff, max_steps=5000, log=print, log_every=20,
                      max_generation=50, material_map=None, transit_log=False):
    N = source.shape[0]
    kernels = list(kernel) if isinstance(kernel, (list, tuple)) else [kernel]
    if material_map is None:
        if len(kernels) != 1:
            raise ValueError("several kernels need a material_map")
        material_map = np.zeros((nx, ny), dtype=np.int64)
    material_map = np.asarray(material_map, dtype=np.int64)
    if material_map.shape != (nx, ny) or material_map.min() < 0 or material_map.max() >= len(kernels):
        raise ValueError(f"material_map must be ({nx}, {ny}) with indices < {len(kernels)} kernels")
    dose = np.zeros((nx, ny))
    dose_by_gen = np.zeros((2, nx, ny))  # [0] primaries' transits, [1] secondaries' transits
    escaped = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    stats = {"transits_boundary": 0, "transits_interior": 0, "daughters": 0, "extra_exits": 0, "photon_E": 0.0,
             "cutoff_dep": 0.0}
    bank = _Bank()
    tlog = {"transits": [], "points": []} if transit_log else None
    cy0 = np.clip(np.floor(source[:, 1] / cell_size).astype(np.int64), 0, ny - 1)
    bank.add(mode=0, cx=0, cy=cy0, face=LEFT, x=0.0, y=np.clip(source[:, 1] / cell_size - cy0, 0, 1),
             u=source[:, 2], v=source[:, 3], w=source[:, 7], E=source[:, 4], gen=0, Eb=np.inf)

    def move_out(cx, cy, x, y, u, v, w, E, gen, Eb):
        """Particles on a face of cell (cx, cy) -> boundary entries of the neighbours (or escaped)."""
        if len(E) == 0:
            return
        f = np.argmin(np.stack([np.abs(x), np.abs(1 - x), np.abs(y), np.abs(1 - y)], 1), 1)
        ncx, ncy = cx + _DX[f], cy + _DY[f]
        gone = (ncx < 0) | (ncx >= nx) | (ncy < 0) | (ncy >= ny)
        for name, m in (("left", ncx < 0), ("right", ncx >= nx), ("bottom", ncy < 0), ("top", ncy >= ny)):
            escaped[name] += float(E[m & gone].sum())
        k = ~gone
        horiz = f < 2
        xl = np.where(horiz, np.where(f == 0, 1.0, 0.0), np.clip(x, 0, 1))
        yl = np.where(horiz, np.clip(y, 0, 1), np.where(f == 2, 1.0, 0.0))
        dead = k & (E <= E_cutoff)
        np.add.at(dose, (ncx[dead], ncy[dead]), E[dead])
        np.add.at(dose_by_gen, (np.minimum(gen[dead], 1), ncx[dead], ncy[dead]), E[dead])
        stats["cutoff_dep"] += float(E[dead].sum())
        if tlog is not None and dead.any():
            tlog["points"].append({"cx": ncx[dead], "cy": ncy[dead], "x": xl[dead], "y": yl[dead], "E": E[dead],
                                   "Eb": Eb[dead]})
        live = k & ~dead
        bank.add(mode=0, cx=ncx[live], cy=ncy[live], face=_ENTRY_FACE[f[live]], x=xl[live], y=yl[live], u=u[live],
                 v=v[live], w=w[live], E=E[live], gen=gen[live], Eb=Eb[live])

    step = 0
    for step in range(max_steps):
        P = bank.take()
        n_act = len(P["E"])
        if n_act == 0:
            break
        mat = material_map[P["cx"], P["cy"]]
        for mode, m in ((mode, m) for mode in (0, 1) for m in range(len(kernels))):
            idx = np.flatnonzero((P["mode"] == mode) & (mat == m))
            if idx.size == 0:
                continue
            kernel = kernels[m]
            cx, cy, face = P["cx"][idx], P["cy"][idx], P["face"][idx]
            w, E, gen, Eb = P["w"][idx], P["E"][idx], P["gen"][idx], P["Eb"][idx]
            if mode == 0:
                stats["transits_boundary"] += idx.size
                _, y_c, u_c, v_c = transform_to_canonical(P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx], face)
                out = kernel.boundary(y_c, u_c, v_c, w, E)

                def to_cell(x_, y_, u_, v_, parent):
                    return transform_from_canonical(x_, y_, u_, v_, face[parent])
                ref_in = (np.zeros(idx.size), y_c, u_c, v_c)
            else:
                stats["transits_interior"] += idx.size
                out = kernel.interior(P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx], w, E)

                def to_cell(x_, y_, u_, v_, parent):
                    return x_, y_, u_, v_
                ref_in = (P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx])
            if tlog is not None:
                tlog["transits"].append(_transit_rows(out, mode, m, cx, cy, face, gen, ref_in, w, E, Eb))
            np.add.at(dose, (cx, cy), out["dose"])
            np.add.at(dose_by_gen, (np.minimum(gen, 1), cx, cy), out["dose"])
            stats["photon_E"] += float(np.sum(out.get("photon_E", 0.0)))
            ex = np.flatnonzero(np.asarray(out["cls"]) != ABSORBED)
            x_, y_, u_, v_ = to_cell(out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex], ex)
            move_out(cx[ex], cy[ex], x_, y_, u_, v_, out["w"][ex], out["E_out"][ex], gen[ex], Eb[ex])
            for key in ("extra_exits", "daughters"):
                D = out.get(key)
                if D is None or len(D["E"]) == 0:
                    continue
                p = np.asarray(D["parent"], dtype=np.int64)
                x_, y_, u_, v_ = to_cell(D["x"], D["y"], D["u"], D["v"], p)
                g = np.minimum(gen[p] + 1, max_generation)
                if key == "extra_exits":
                    stats["extra_exits"] += len(p)
                    move_out(cx[p], cy[p], x_, y_, u_, v_, D["w"], D["E"], g, D["E"])
                else:
                    stats["daughters"] += len(p)
                    bank.add(mode=1, cx=cx[p], cy=cy[p], face=0, x=x_, y=y_, u=u_, v=v_, w=D["w"], E=D["E"], gen=g,
                             Eb=D["E"])
        if log and log_every and step % log_every == 0:
            log(f"step {step:4d}: {n_act:7d} active")
    left = bank.take()
    in_flight = float(left["E"].sum()) if len(left["E"]) else 0.0
    extra = {}
    if tlog is not None:
        extra = {"transits": _concat_rows(tlog["transits"], TRANSIT_COLUMNS),
                 "points": _concat_rows(tlog["points"], POINT_COLUMNS)}
    return {"dose": dose, "dose_primary_transits": dose_by_gen[0], "dose_secondary_transits": dose_by_gen[1],
            "escaped": escaped, "in_flight": in_flight, "steps": step + 1, "E_source": float(source[:, 4].sum()),
            **stats, **extra}


TRANSIT_COLUMNS = ("cx", "cy", "mat", "start", "face", "gen", "x0", "y0", "u0", "v0", "w0", "E0", "cls", "x1", "y1",
                   "u1", "v1", "w1", "E_out", "dose", "n_rel", "flag", "Eb", "dose_cond")
POINT_COLUMNS = ("cx", "cy", "x", "y", "E", "Eb")
_INT_COLUMNS = {"cx", "cy", "mat", "start", "face", "gen", "cls", "n_rel", "flag"}


def _transit_rows(out, mode, mat, cx, cy, face, gen, ref_in, w, E, Eb):
    """One kernel call -> transit-log columns (entry/exit in the frame of the call)."""
    n = len(E)
    if "n_rel" in out:
        n_rel = np.asarray(out["n_rel"])
    else:
        D = out.get("daughters")
        n_rel = np.bincount(np.asarray(D["parent"], dtype=np.int64), minlength=n) if D is not None and len(D["E"]) \
            else np.zeros(n, dtype=np.int64)
    flag = np.asarray(out["flag"]) if "flag" in out else (np.asarray(out.get("photon_E", np.zeros(n))) > 0)
    x0, y0, u0, v0 = ref_in
    return {"cx": cx, "cy": cy, "mat": np.full(n, mat), "start": np.full(n, mode),
            "face": face if mode == 0 else np.full(n, -1), "gen": gen, "x0": x0, "y0": y0, "u0": u0, "v0": v0,
            "w0": w, "E0": E, "cls": np.asarray(out["cls"]), "x1": out["x"], "y1": out["y"], "u1": out["u"],
            "v1": out["v"], "w1": out["w"], "E_out": out["E_out"], "dose": out["dose"], "n_rel": n_rel,
            "flag": flag.astype(np.int64), "Eb": Eb, "dose_cond": out.get("dose_cond", out["dose"])}


def _concat_rows(rows, cols):
    if not rows:
        return {c: np.zeros(0, dtype=np.int64 if c in _INT_COLUMNS else np.float64) for c in cols}
    return {c: np.concatenate([np.asarray(r[c]) for r in rows]).astype(np.int64 if c in _INT_COLUMNS else np.float64)
            for c in cols}


class MCCellKernel:
    """Analog MC (ElectronTransport analog_stepper.run_transport) on one cell, as a chain_secondaries kernel."""

    def __init__(self, library, cell_size, E_cutoff, seed=1000, secondaries=True, material=0):
        from analog_stepper import VoxelGeometry, run_transport
        self.run_transport = run_transport
        self.lib, self.L, self.E_cut, self.seed, self.secondaries = library, cell_size, E_cutoff, seed, secondaries
        self.geo = VoxelGeometry.uniform(cell_size, cell_size, 1, 1, material=material)
        self.calls = 0

    def _run(self, x, y, u, v, w, E):
        p = np.zeros((len(E), 8))
        p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 7] = x * self.L, y * self.L, u, v, E, w
        self.calls += 1
        return self.run_transport(self.lib, self.geo, p, E_cutoff=self.E_cut, seed=self.seed + 7919 * self.calls,
                                  n_tracks=0)

    def _pack(self, r, E):
        from ml.cell_topology import classes_from_status
        f = r["final"]
        status = f[:, 6].astype(int)
        if np.any(status == 6):
            raise RuntimeError("status 6 (lost along z) inside a filled cell")
        cls = classes_from_status(status)
        E_out = np.where(cls == ABSORBED, 0.0, f[:, 4])
        n = len(E)
        ph = r["photon_bank"]
        E_ph = np.bincount(ph[:, 6].astype(int), weights=ph[:, 5], minlength=n) if len(ph) else np.zeros(n)
        eb = r["electron_bank"]
        sec = eb[(eb[:, 8] > 0) & (eb[:, 6] >= 1) & (eb[:, 6] <= 4)] if len(eb) else np.zeros((0, 10))
        E_sec = np.bincount(sec[:, 7].astype(int), weights=sec[:, 5], minlength=n) if len(sec) else np.zeros(n)
        out = {"cls": cls, "x": f[:, 0] / self.L, "y": f[:, 1] / self.L, "u": f[:, 2], "v": f[:, 3], "w": f[:, 7],
               "E_out": E_out, "photon_E": E_ph}
        if self.secondaries:
            out["dose"] = E - E_out - E_sec - E_ph
            out["extra_exits"] = {"parent": sec[:, 7].astype(int), "x": sec[:, 0] / self.L, "y": sec[:, 1] / self.L,
                                  "u": sec[:, 2], "v": sec[:, 3], "w": sec[:, 4], "E": sec[:, 5]}
        else:
            out["dose"] = E - E_out - E_ph
        return out

    def boundary(self, y_c, u_c, v_c, w, E):
        return self._pack(self._run(np.zeros_like(y_c), y_c, u_c, v_c, w, E), E)

    def interior(self, x, y, u, v, w, E):
        return self._pack(self._run(x, y, u, v, w, E), E)
