"""
Cell-to-cell chaining with secondaries on a mesh of square cells of DIFFERENT sizes (quadtree / non-conforming), for
the size-conditioned GMC (ml/size_stream.py), plus the SCR-GMC fine-mesh tally of its transit log.

Same particle bookkeeping as chain_secondaries.py (boundary entries in the canonical left-face frame, relevant
daughters as interior starts in the cell frame, kernels per material), with three differences:
    * a cell is (x0, y0, L, material); neighbours are found by point location, so a face may border several smaller
      cells or part of a larger one (hanging nodes are fine: an electron leaving a face enters whichever cell
      contains the point just beyond it, through that cell's opposite face);
    * every kernel call gets the cell side: kernel.boundary(y_c, u_c, v_c, w, E, L=...) and
      kernel.interior(x, y, u, v, w, E, L=...), L per row (cm);
    * the transit log keeps the cell id ("cid") instead of (cx, cy); positions stay in the unit cell of that call.

Mesh: QuadMesh.quadtree(box, L_max, L_min, refine) splits a uniform L_max tiling recursively while refine(cell) says
so and L > L_min. Point location uses an index grid at the smallest cell size (box / L_min pixels).
"""
import numpy as np

from chain_core import transform_to_canonical, transform_from_canonical, LEFT, RIGHT, BOTTOM, TOP

ABSORBED = 4
_ENTRY_FACE = np.array([RIGHT, LEFT, TOP, BOTTOM])  # leaving through physical face f -> entry face of the neighbour
_NX = np.array([-1.0, 1.0, 0.0, 0.0])
_NY = np.array([0.0, 0.0, -1.0, 1.0])


class QuadMesh:
    def __init__(self, x0, y0, L, mat, box, h):
        self.x0, self.y0, self.L = (np.asarray(a, np.float64) for a in (x0, y0, L))
        self.mat = np.asarray(mat, np.int64)
        self.box, self.h = (float(box[0]), float(box[1])), float(h)
        self.nx, self.ny = int(round(self.box[0] / h)), int(round(self.box[1] / h))
        self.index = np.full((self.nx, self.ny), -1, np.int64)
        for c in range(len(self.L)):
            i0, j0 = int(round(self.x0[c] / h)), int(round(self.y0[c] / h))
            k = int(round(self.L[c] / h))
            self.index[i0:i0 + k, j0:j0 + k] = c
        if np.any(self.index < 0):
            raise ValueError("cells do not tile the box")

    @property
    def n_cells(self):
        return len(self.L)

    @classmethod
    def uniform(cls, box, L, material):
        nx, ny = int(round(box[0] / L)), int(round(box[1] / L))
        gx, gy = np.meshgrid(np.arange(nx) * L, np.arange(ny) * L, indexing="ij")
        x0, y0 = gx.ravel(), gy.ravel()
        return cls(x0, y0, np.full(len(x0), L), material(x0 + L / 2, y0 + L / 2), box, L)

    @classmethod
    def quadtree(cls, box, L_max, L_min, refine, material):
        """refine(x0, y0, L) -> bool per cell (vectorized); material(xc, yc) -> index per cell centre."""
        nx, ny = int(round(box[0] / L_max)), int(round(box[1] / L_max))
        gx, gy = np.meshgrid(np.arange(nx) * L_max, np.arange(ny) * L_max, indexing="ij")
        x0, y0, L = gx.ravel(), gy.ravel(), np.full(nx * ny, float(L_max))
        done = []
        while len(L):
            split = refine(x0, y0, L) & (L > L_min * (1 + 1e-9))
            done.append((x0[~split], y0[~split], L[~split]))
            x0, y0, L = x0[split], y0[split], L[split] / 2
            x0, y0 = (np.concatenate([x0, x0 + L, x0, x0 + L]), np.concatenate([y0, y0, y0 + L, y0 + L]))
            L = np.tile(L, 4)
        x0, y0, L = (np.concatenate(v) for v in zip(*done))
        return cls(x0, y0, L, material(x0 + L / 2, y0 + L / 2), box, L.min())

    def locate(self, gx, gy):
        """Cell id of global points (-1 outside the box)."""
        i, j = np.floor(gx / self.h).astype(np.int64), np.floor(gy / self.h).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        out = np.full(len(gx), -1, np.int64)
        out[ok] = self.index[i[ok], j[ok]]
        return out


class _Bank:
    FIELDS = ("mode", "cid", "face", "x", "y", "u", "v", "w", "E", "gen", "Eb")

    def __init__(self):
        self.d = {k: [] for k in self.FIELDS}

    def add(self, **kw):
        n = len(kw["E"])
        if n == 0:
            return
        for k in self.FIELDS:
            self.d[k].append(np.broadcast_to(np.asarray(kw.get(k, 0)), (n,)).copy())

    def take(self):
        out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.d.items()}
        for k in ("mode", "cid", "face", "gen"):
            out[k] = out[k].astype(np.int64)
        self.d = {k: [] for k in self.FIELDS}
        return out


TRANSIT_COLUMNS = ("cid", "mat", "start", "face", "gen", "x0", "y0", "u0", "v0", "w0", "E0", "cls", "x1", "y1", "u1",
                   "v1", "w1", "E_out", "dose", "n_rel", "flag", "Eb", "dose_cond")
POINT_COLUMNS = ("gx", "gy", "E", "Eb")
_INT = {"cid", "mat", "start", "face", "gen", "cls", "n_rel", "flag"}


def chain_quadtree(kernels, source, mesh, E_cutoff, max_steps=20000, max_generation=50, transit_log=False):
    """source: (N, 8) as chain_core.make_source, global positions; electrons enter the cell containing their start
    point through its LEFT face (the source must sit on the box's left edge, x = 0)."""
    N = len(source)
    dose = np.zeros(mesh.n_cells)
    escaped = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    stats = {"transits_boundary": 0, "transits_interior": 0, "daughters": 0, "photon_E": 0.0, "cutoff_dep": 0.0,
             "calls": 0}
    tlog = {"transits": [], "points": []} if transit_log else None
    bank = _Bank()
    cid0 = mesh.locate(np.full(N, 0.5 * mesh.h), source[:, 1])
    bank.add(mode=0, cid=cid0, face=LEFT, x=0.0, y=np.clip((source[:, 1] - mesh.y0[cid0]) / mesh.L[cid0], 0, 1),
             u=source[:, 2], v=source[:, 3], w=source[:, 7], E=source[:, 4], gen=0, Eb=np.inf)

    def move_out(cid, x, y, u, v, w, E, gen, Eb):
        if len(E) == 0:
            return
        f = np.argmin(np.stack([np.abs(x), np.abs(1 - x), np.abs(y), np.abs(1 - y)], 1), 1)
        Lc = mesh.L[cid]
        gx = mesh.x0[cid] + np.where(f == 0, 0.0, np.where(f == 1, 1.0, np.clip(x, 0, 1))) * Lc
        gy = mesh.y0[cid] + np.where(f == 2, 0.0, np.where(f == 3, 1.0, np.clip(y, 0, 1))) * Lc
        nb = mesh.locate(gx + 0.5 * mesh.h * _NX[f], gy + 0.5 * mesh.h * _NY[f])
        gone = nb < 0
        for name, k in (("left", 0), ("right", 1), ("bottom", 2), ("top", 3)):
            escaped[name] += float(E[gone & (f == k)].sum())
        k = ~gone
        nbk = np.maximum(nb, 0)
        Ln = mesh.L[nbk]
        xl = np.clip((gx - mesh.x0[nbk]) / Ln, 0, 1)
        yl = np.clip((gy - mesh.y0[nbk]) / Ln, 0, 1)
        horiz = f < 2
        xl = np.where(horiz, np.where(f == 0, 1.0, 0.0), xl)
        yl = np.where(horiz, yl, np.where(f == 2, 1.0, 0.0))
        dead = k & (E <= E_cutoff)
        np.add.at(dose, nb[dead], E[dead])
        stats["cutoff_dep"] += float(E[dead].sum())
        if tlog is not None and dead.any():
            tlog["points"].append({"gx": gx[dead], "gy": gy[dead], "E": E[dead], "Eb": Eb[dead]})
        live = k & ~dead
        bank.add(mode=0, cid=nb[live], face=_ENTRY_FACE[f[live]], x=xl[live], y=yl[live], u=u[live], v=v[live],
                 w=w[live], E=E[live], gen=gen[live], Eb=Eb[live])

    step = 0
    for step in range(max_steps):
        P = bank.take()
        if len(P["E"]) == 0:
            break
        mat = mesh.mat[P["cid"]]
        for mode, m in ((mode, m) for mode in (0, 1) for m in range(len(kernels))):
            idx = np.flatnonzero((P["mode"] == mode) & (mat == m))
            if idx.size == 0:
                continue
            kern = kernels[m]
            cid, face = P["cid"][idx], P["face"][idx]
            w, E, gen, Eb = P["w"][idx], P["E"][idx], P["gen"][idx], P["Eb"][idx]
            Lc = mesh.L[cid]
            stats["calls"] += 1
            if mode == 0:
                stats["transits_boundary"] += idx.size
                _, y_c, u_c, v_c = transform_to_canonical(P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx], face)
                out = kern.boundary(y_c, u_c, v_c, w, E, L=Lc)
                ref_in = (np.zeros(idx.size), y_c, u_c, v_c)

                def to_cell(x_, y_, u_, v_, parent):
                    return transform_from_canonical(x_, y_, u_, v_, face[parent])
            else:
                stats["transits_interior"] += idx.size
                out = kern.interior(P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx], w, E, L=Lc)
                ref_in = (P["x"][idx], P["y"][idx], P["u"][idx], P["v"][idx])

                def to_cell(x_, y_, u_, v_, parent):
                    return x_, y_, u_, v_
            if tlog is not None:
                tlog["transits"].append(_rows(out, mode, m, cid, face, gen, ref_in, w, E, Eb))
            np.add.at(dose, cid, out["dose"])
            stats["photon_E"] += float(np.sum(out.get("photon_E", 0.0)))
            ex = np.flatnonzero(np.asarray(out["cls"]) != ABSORBED)
            x_, y_, u_, v_ = to_cell(out["x"][ex], out["y"][ex], out["u"][ex], out["v"][ex], ex)
            move_out(cid[ex], x_, y_, u_, v_, out["w"][ex], out["E_out"][ex], gen[ex], Eb[ex])
            X = out.get("extra_exits")        # analog kernels: secondaries that left the cell (already transported)
            if X is not None and len(X["E"]):
                p = np.asarray(X["parent"], dtype=np.int64)
                x_, y_, u_, v_ = to_cell(X["x"], X["y"], X["u"], X["v"], p)
                move_out(cid[p], x_, y_, u_, v_, X["w"], X["E"], np.minimum(gen[p] + 1, max_generation), X["E"])
            D = out.get("daughters")
            if D is not None and len(D["E"]):
                p = np.asarray(D["parent"], dtype=np.int64)
                x_, y_, u_, v_ = to_cell(D["x"], D["y"], D["u"], D["v"], p)
                stats["daughters"] += len(p)
                bank.add(mode=1, cid=cid[p], face=0, x=x_, y=y_, u=u_, v=v_, w=D["w"], E=D["E"],
                         gen=np.minimum(gen[p] + 1, max_generation), Eb=D["E"])
    left = bank.take()
    res = {"dose": dose, "escaped": escaped, "in_flight": float(left["E"].sum()) if len(left["E"]) else 0.0,
           "steps": step + 1, "E_source": float(source[:, 4].sum()), **stats}
    if tlog is not None:
        res["transits"] = _concat(tlog["transits"], TRANSIT_COLUMNS)
        res["points"] = _concat(tlog["points"], POINT_COLUMNS)
    return res


def _rows(out, mode, mat, cid, face, gen, ref_in, w, E, Eb):
    n = len(E)
    if "n_rel" in out:
        n_rel = np.asarray(out["n_rel"])
    else:
        D = out.get("daughters")
        n_rel = (np.bincount(np.asarray(D["parent"], dtype=np.int64), minlength=n) if D is not None and len(D["E"])
                 else np.zeros(n, dtype=np.int64))
    flag = np.asarray(out["flag"]) if "flag" in out else (np.asarray(out.get("photon_E", np.zeros(n))) > 0)
    x0, y0, u0, v0 = ref_in
    return {"cid": cid, "mat": np.full(n, mat), "start": np.full(n, mode), "face": face if mode == 0 else np.full(n, -1),
            "gen": gen, "x0": x0, "y0": y0, "u0": u0, "v0": v0, "w0": w, "E0": E, "cls": np.asarray(out["cls"]),
            "x1": out["x"], "y1": out["y"], "u1": out["u"], "v1": out["v"], "w1": out["w"], "E_out": out["E_out"],
            "dose": out["dose"], "n_rel": n_rel, "flag": flag.astype(np.int64), "Eb": Eb,
            "dose_cond": out.get("dose_cond", out["dose"])}


def _concat(rows, cols):
    if not rows:
        return {c: np.zeros(0, dtype=np.int64 if c in _INT else np.float64) for c in cols}
    return {c: np.concatenate([np.asarray(r[c]) for r in rows]).astype(np.int64 if c in _INT else np.float64)
            for c in cols}


# ====================================================================================================================
# Kernel wrappers with the cell side
# ====================================================================================================================
class MCSizeKernel:
    """Analog MC (ElectronTransport analog_stepper) on one homogeneous cell of any side L, as a chain_quadtree kernel
    (chain_secondaries.MCCellKernel with the cell side per row: one run_transport call per distinct L). Exact by the
    Markov property, so a chain of it reproduces the analog MC on the mesh's geometry. Needs the ElectronTransport
    venv / path."""

    def __init__(self, library, E_cutoff, seed=1000, secondaries=True, material=0):
        from analog_stepper import VoxelGeometry, run_transport
        self.VG, self.run_transport = VoxelGeometry, run_transport
        self.lib, self.E_cut, self.seed, self.secondaries, self.material = library, E_cutoff, seed, secondaries, material
        self.geo, self.calls = {}, 0

    def _one(self, x, y, u, v, w, E, L):
        from ml.cell_topology import classes_from_status
        if L not in self.geo:
            self.geo[L] = self.VG.uniform(L, L, 1, 1, material=self.material)
        p = np.zeros((len(E), 8))
        p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 7] = x * L, y * L, u, v, E, w
        self.calls += 1
        r = self.run_transport(self.lib, self.geo[L], p, E_cutoff=self.E_cut, seed=self.seed + 7919 * self.calls,
                               n_tracks=0)
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
        out = {"cls": cls, "x": f[:, 0] / L, "y": f[:, 1] / L, "u": f[:, 2], "v": f[:, 3], "w": f[:, 7],
               "E_out": E_out, "photon_E": E_ph, "dose": E - E_out - E_sec - E_ph,
               "n_rel": np.bincount(sec[:, 7].astype(int), minlength=n),
               "extra_exits": {"parent": sec[:, 7].astype(int), "x": sec[:, 0] / L, "y": sec[:, 1] / L,
                               "u": sec[:, 2], "v": sec[:, 3], "w": sec[:, 4], "E": sec[:, 5]}}
        return out

    def _call(self, x, y, u, v, w, E, L):
        L = np.broadcast_to(np.asarray(L, np.float64), (len(E),))
        keys = ("cls", "x", "y", "u", "v", "w", "E_out", "photon_E", "dose", "n_rel")
        out = {k: np.zeros(len(E), np.int64 if k in ("cls", "n_rel") else np.float64) for k in keys}
        ex = {k: [] for k in ("parent", "x", "y", "u", "v", "w", "E")}
        for Lv in np.unique(L):
            r = np.flatnonzero(L == Lv)
            o = self._one(x[r], y[r], u[r], v[r], w[r], E[r], float(Lv))
            for k in keys:
                out[k][r] = o[k]
            X = o["extra_exits"]
            for k in ex:
                ex[k].append(r[X["parent"]] if k == "parent" else X[k])
        out["extra_exits"] = {k: (np.concatenate(v) if v else np.zeros(0, np.int64 if k == "parent" else float))
                              for k, v in ex.items()}
        return out

    def boundary(self, y_c, u_c, v_c, w, E, L=None):
        return self._call(np.zeros_like(y_c), y_c, u_c, v_c, w, E, L)

    def interior(self, x, y, u, v, w, E, L=None):
        return self._call(x, y, u, v, w, E, L)


class VacuumKernel:
    """Straight streaming through an empty cell (scale-free: the cell side does not matter)."""

    def __init__(self):
        self.stuck = 0

    def _go(self, x, y, u, v, w, E):
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(u > 0, (1 - x) / u, np.where(u < 0, -x / u, np.inf))
            ty = np.where(v > 0, (1 - y) / v, np.where(v < 0, -y / v, np.inf))
        t = np.minimum(tx, ty)
        stuck = ~np.isfinite(t)
        self.stuck += int(stuck.sum())
        t = np.where(stuck, 0.0, t)
        cls = np.where(tx <= ty, np.where(u > 0, 1, 3), np.where(v > 0, 2, 0))
        return {"cls": np.where(stuck, 4, cls), "x": np.clip(x + u * t, 0, 1), "y": np.clip(y + v * t, 0, 1),
                "u": u, "v": v, "w": w, "E_out": E, "dose": np.zeros_like(E)}

    def boundary(self, y_c, u_c, v_c, w, E, L=None):
        return self._go(np.zeros_like(y_c), y_c, u_c, v_c, w, E)

    def interior(self, x, y, u, v, w, E, L=None):
        return self._go(x, y, u, v, w, E)


class LocalPhotons:
    """drumm_cylinder.LocalPhotons with the cell side passed through (photon energy deposited in its transit)."""

    def __init__(self, k):
        self.k = k

    @staticmethod
    def _f(out):
        out = dict(out)
        out["dose_cond"] = out["dose"]
        out["dose"] = out["dose"] + out.get("photon_E", 0.0)
        out["photon_E"] = np.zeros_like(out["dose"])
        return out

    def boundary(self, *a, L=None):
        return self._f(self.k.boundary(*a, L=L))

    def interior(self, *a, L=None):
        return self._f(self.k.interior(*a, L=L))


class MirrorY:
    """drumm_cylinder.MirrorY (exact y-mirror symmetrization at inference) with the cell side passed through."""

    def __init__(self, k, seed):
        self.k, self.rng = k, np.random.default_rng(seed)

    def _call(self, fn, x, y, u, v, w, E):
        m = self.rng.random(len(E)) < 0.5
        out = dict(fn(x, np.where(m, 1 - y, y), u, np.where(m, -v, v), w, E))
        out["y"] = np.where(m, 1 - out["y"], out["y"])
        out["v"] = np.where(m, -out["v"], out["v"])
        c = np.asarray(out["cls"])
        out["cls"] = np.where(m & (c == 0), 2, np.where(m & (c == 2), 0, c))
        D = out.get("daughters")
        if D is not None and len(D["E"]):
            mp = m[np.asarray(D["parent"], dtype=np.int64)]
            D = dict(D)
            D["y"] = np.where(mp, 1 - D["y"], D["y"])
            D["v"] = np.where(mp, -D["v"], D["v"])
            out["daughters"] = D
        return out

    def boundary(self, y_c, u_c, v_c, w, E, L=None):
        return self._call(lambda x, y, u, v, w_, e: self.k.boundary(y, u, v, w_, e, L=L), np.zeros_like(y_c), y_c,
                          u_c, v_c, w, E)

    def interior(self, x, y, u, v, w, E, L=None):
        return self._call(lambda *a: self.k.interior(*a, L=L), x, y, u, v, w, E)


# ====================================================================================================================
# SCR tally on a fine uniform grid
# ====================================================================================================================
def scr_tally_quadtree(log, mesh, kernels, h_out, device="cpu", gen=None, samples=None, max_points=1 << 20,
                       sampling="sobol", extra=None):
    """Fine dose (box / h_out bins per axis) from a quadtree chain's transit log. kernels: {material: (spec, model)};
    materials without a kernel spread their dose uniformly over the cell. samples(n) -> flow samples per transit for
    a cell n = L / h_out output bins wide (default subcell_kernel.auto_samples). extra: {name: per-transit weight}
    tallied with the SAME samples (e.g. the dose without photons), returned in info["extra"] (point deposits are
    added to every map). Returns (fine, info)."""
    import torch
    from ml.subcell_kernel import auto_samples, canonical_to_cell, cond_features, transit_log_tensors
    samples = samples or auto_samples
    T, P = log["transits"], log["points"]
    gx_n, gy_n = int(round(mesh.box[0] / h_out)), int(round(mesh.box[1] / h_out))
    names = [None] + list(extra or {})
    fines = {nm: torch.zeros(gx_n * gy_n, dtype=torch.float64, device=device) for nm in names}
    wts = {None: np.asarray(T["dose"], dtype=np.float64),
           **{k: np.asarray(v, dtype=np.float64) for k, v in (extra or {}).items()}}
    info = {"kernel": 0.0, "flat": 0.0, "points": 0.0, "samples": 0}
    X0, Y0, LL = (torch.as_tensor(a, device=device) for a in (mesh.x0, mesh.y0, mesh.L))

    def add(cid, x, y, r, per):
        """Points (x, y) of transit rows r (cell ids cid); every map gets weight[r] * per (per: fraction per point)."""
        if x.dim() > 1:
            cid = cid.unsqueeze(-1)
        gx = X0[cid] + x.double() * LL[cid]
        gy = Y0[cid] + y.double() * LL[cid]
        ix = (gx / h_out).long().clamp(0, gx_n - 1)
        iy = (gy / h_out).long().clamp(0, gy_n - 1)
        flat = (ix * gy_n + iy).reshape(-1)
        for nm in names:
            w_ = torch.as_tensor(wts[nm][r], device=device).reshape(-1, *([1] * (x.dim() - 1))) * per
            fines[nm].index_add_(0, flat, w_.expand_as(x).reshape(-1).double())

    dose_all = np.asarray(T["dose"], dtype=np.float64)
    cid_all = np.asarray(T["cid"])
    mat = np.asarray(T["mat"])
    Lrow = mesh.L[cid_all]
    with torch.no_grad():
        for m in np.unique(mat):
            sel = (mat == m) & (dose_all > 0)
            if not sel.any():
                continue
            if int(m) not in kernels:
                r = np.flatnonzero(sel)
                k = 8
                c = (torch.arange(k, device=device, dtype=torch.float64) + 0.5) / k
                gxx, gyy = torch.meshgrid(c, c, indexing="ij")
                add(torch.as_tensor(cid_all[r], device=device), gxx.reshape(1, -1).expand(len(r), -1),
                    gyy.reshape(1, -1).expand(len(r), -1), r, 1.0 / (k * k))
                info["flat"] += float(dose_all[r].sum())
                continue
            spec, model = kernels[int(m)]
            for Lv in np.unique(Lrow[sel]):
                rows = np.flatnonzero(sel & (Lrow == Lv))
                K = samples(max(1, int(round(Lv / h_out))))
                chunk = max(1, max_points // K)
                for i in range(0, len(rows), chunk):
                    r = rows[i:i + chunk]
                    Rd = transit_log_tensors(T, r, device)
                    if getattr(spec, "size_cond", False):
                        Rd["L"] = torch.full((len(r),), float(Lv), device=device)
                    h = model.context(cond_features(spec, Rd))
                    z = model.sample(h, K, gen=gen, base=sampling).clamp(0.0, 1.0)
                    face = torch.as_tensor(np.asarray(T["face"])[r], device=device)
                    x, y = canonical_to_cell(z[..., 0], z[..., 1], face)
                    add(torch.as_tensor(cid_all[r], device=device), x, y, r, 1.0 / K)
                    info["kernel"] += float(dose_all[r].sum())
                    info["samples"] += len(r) * K
        if len(P["E"]):
            e = torch.as_tensor(np.asarray(P["E"], dtype=np.float64), device=device)
            ix = torch.as_tensor(np.clip((np.asarray(P["gx"]) / h_out).astype(np.int64), 0, gx_n - 1), device=device)
            iy = torch.as_tensor(np.clip((np.asarray(P["gy"]) / h_out).astype(np.int64), 0, gy_n - 1), device=device)
            for nm in names:
                fines[nm].index_add_(0, ix * gy_n + iy, e)
            info["points"] = float(e.sum())
    info["extra"] = {nm: fines[nm].reshape(gx_n, gy_n).cpu().numpy() for nm in names[1:]}
    return fines[None].reshape(gx_n, gy_n).cpu().numpy(), info
