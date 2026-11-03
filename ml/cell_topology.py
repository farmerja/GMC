#!/usr/bin/env python3
"""
Cell topology and face-local geometry for GMC exit classification.

A GMC cell model maps an entry state to an exit state. The exit state is discrete + continuous:
    class  in {face_0, ..., face_{F-1}, <interior outcomes>}
    then, given the class, a continuous state (position on that face, direction, energy, ...).

This module fixes ONE class convention for every geometry so the exit classifier, the face-conditioned flow and the
chaining code all agree:

    * 2D polygon cells (triangle, canonical quad, unit square): vertices are listed counter-clockwise and
      face k is the edge from vertex k to vertex (k + 1) % V. For the unit square / canonical quad
      (0,0),(1,0),(x3,y3),(x4,y4) that gives  0 = bottom, 1 = right, 2 = top, 3 = left.
    * 3D polyhedral cells (box, tetrahedron): faces are given by a vertex-index table whose vertices are ordered so
      that the right-hand normal points outward.
    * Interior outcomes come after the faces, in the order given by ``interior`` (e.g. "absorbed" for a particle
      that dropped below the energy cutoff inside the cell, "census" for a particle still inside at the end of a
      time step in time-dependent problems).

Class index = face index for faces, F + j for interior outcome j.

Face-local coordinates (what a face-conditioned flow should predict instead of a raw (x, y) point):
    * 2D: s in [0, 1], the fraction of the way along edge k from vertex k to vertex k + 1.
    * 3D: (a, b) affine coordinates in the frame of the face's first three vertices,
          p = v0 + a (v1 - v0) + b (v_{n-1} - v0). For parallelogram faces (boxes) a, b in [0, 1]; for triangular faces
          (tets) a, b >= 0, a + b <= 1.
Directions are expressed in the face frame (mu_n = cosine to the outward normal, then tangential components), so an
exiting particle always has mu_n > 0 by construction.

Everything is vectorized over a batch and works on numpy arrays or torch tensors. Vertices may be shared by the
whole batch (shape [V, D]) or given per sample (shape [N, V, D]), which is what unstructured quad / triangle meshes
need since every cell has its own canonical shape parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:  # optional, so the MC-side tools (numpy only) can import this module in a torch-free environment
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _lib(x):
    return torch if torch is not None and isinstance(x, torch.Tensor) else np


# --------------------------------------------------------------------------------------------------------------------
# Class specification
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CellTopology:
    """Discrete exit outcomes of one cell type.

    name:       identifier, stored in checkpoints.
    dim:        spatial dimension of the cell (2 or 3).
    face_names: one name per face, in face-index order.
    interior:   interior (non-exit) outcomes, appended after the faces.
    faces:      3D only, vertex-index table per face (outward right-hand ordering). 2D faces are implicit edges.
    """
    name: str
    dim: int
    face_names: tuple[str, ...]
    interior: tuple[str, ...] = ("absorbed",)
    faces: tuple[tuple[int, ...], ...] | None = None

    @property
    def num_faces(self) -> int:
        return len(self.face_names)

    @property
    def num_classes(self) -> int:
        return self.num_faces + len(self.interior)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.face_names) + tuple(self.interior)

    def interior_class(self, outcome: str) -> int:
        return self.num_faces + self.interior.index(outcome)

    def is_face(self, cls):
        return cls < self.num_faces

    def with_interior(self, *outcomes: str) -> "CellTopology":
        """Same faces, different interior outcomes (e.g. add 'census' for a time-dependent model)."""
        return CellTopology(self.name, self.dim, self.face_names, tuple(outcomes), self.faces)

    def to_dict(self) -> dict:
        return {"name": self.name, "dim": self.dim, "face_names": list(self.face_names),
                "interior": list(self.interior), "faces": None if self.faces is None else [list(f) for f in self.faces]}

    @classmethod
    def from_dict(cls, d: dict) -> "CellTopology":
        faces = d.get("faces")
        return cls(name=d["name"], dim=int(d["dim"]), face_names=tuple(d["face_names"]),
                   interior=tuple(d.get("interior", ("absorbed",))),
                   faces=None if faces is None else tuple(tuple(int(i) for i in f) for f in faces))


# Built-in topologies. Faces of polygons are CCW edges (face k: vertex k -> vertex k+1).
SQUARE = CellTopology("square", 2, ("bottom", "right", "top", "left"))
QUAD = CellTopology("quad", 2, ("edge0", "edge1", "edge2", "edge3"))
TRIANGLE = CellTopology("triangle", 2, ("edge0", "edge1", "edge2"))
# Box vertices: index = i + 2 j + 4 k for corner (i, j, k) in {0,1}^3.
BOX = CellTopology(
    "box", 3, ("x-", "x+", "y-", "y+", "z-", "z+"),
    faces=((0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)),
)
# Tetrahedron with vertices v0..v3; face k is the face opposite vertex k.
TETRA = CellTopology(
    "tetra", 3, ("opp0", "opp1", "opp2", "opp3"),
    faces=((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
)

TOPOLOGIES = {t.name: t for t in (SQUARE, QUAD, TRIANGLE, BOX, TETRA)}


def get_topology(name: str, interior: Sequence[str] | None = None) -> CellTopology:
    topo = TOPOLOGIES[name.lower()]
    return topo if interior is None else topo.with_interior(*interior)


UNIT_SQUARE_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
UNIT_BOX_VERTS = np.array([[i, j, k] for k in (0.0, 1.0) for j in (0.0, 1.0) for i in (0.0, 1.0)])


def canonical_quad_vertices(x3, y3, x4, y4):
    """[N, 4, 2] vertices of the canonical quad (0,0),(1,0),(x3,y3),(x4,y4) used by the 2d_quad models."""
    L = _lib(x3)
    x3, y3, x4, y4 = (L.asarray(a) if L is np else a for a in (x3, y3, x4, y4))
    zeros, ones = L.zeros_like(x3), L.ones_like(x3)
    xs = L.stack([zeros, ones, x3, x4], -1)
    ys = L.stack([zeros, zeros, y3, y4], -1)
    return L.stack([xs, ys], -1)


# --------------------------------------------------------------------------------------------------------------------
# Status-code adapters (MC codes -> class index)
# --------------------------------------------------------------------------------------------------------------------
# ElectronTransport (ElectronMC.py, analog_stepper.py) box status: 1 = left (x=0), 2 = right, 3 = bottom (y=0),
# 4 = top, 5 = absorbed (dropped to E_cut), 6 = lost (travelling exactly along z in vacuum; never reaches a face).
ELECTRON_MC_STATUS_TO_CLASS = {1: 3, 2: 1, 3: 0, 4: 2, 5: SQUARE.interior_class("absorbed")}


def classes_from_status(status, mapping: dict[int, int] = ELECTRON_MC_STATUS_TO_CLASS, invalid: int = -1):
    """Map an MC status array to class indices. Codes not in ``mapping`` become ``invalid`` (default -1)."""
    status = np.asarray(status).astype(np.int64)
    out = np.full(status.shape, invalid, dtype=np.int64)
    for code, cls in mapping.items():
        out[status == code] = cls
    return out


# --------------------------------------------------------------------------------------------------------------------
# 2D polygon geometry
# --------------------------------------------------------------------------------------------------------------------
def _batched_verts(verts, n, like):
    """Return vertices as [N, V, D] in the array library of ``like``."""
    L = _lib(like)
    if L is torch:
        v = torch.as_tensor(verts, dtype=like.dtype if like.is_floating_point() else torch.float64, device=like.device)
    else:
        v = np.asarray(verts, dtype=np.float64)
    if v.ndim == 2:
        v = v[None].expand(n, *v.shape) if L is torch else np.broadcast_to(v[None], (n, *v.shape))
    return v


def _gather_face(arr, face):
    """arr: [N, F, ...], face: [N] -> arr[i, face[i]]."""
    L = _lib(arr)
    idx = L.arange(arr.shape[0]) if L is np else torch.arange(arr.shape[0], device=arr.device)
    return arr[idx, face]


def polygon_edges(verts, n=None, like=None):
    """Edge start points, edge vectors and outward unit normals of a CCW polygon: each [N, V, 2]."""
    like = verts if like is None else like
    v = _batched_verts(verts, n if n is not None else (verts.shape[0] if np.ndim(verts) == 3 else 1), like)
    L = _lib(v)
    v_next = L.roll(v, -1, 1) if L is np else torch.roll(v, -1, 1)
    e = v_next - v
    length = L.sqrt((e ** 2).sum(-1, keepdims=True) if L is np else (e ** 2).sum(-1, keepdim=True))
    t = e / length
    normal = L.stack([t[..., 1], -t[..., 0]], -1)  # CCW polygon -> (t_y, -t_x) points outward
    return v, e, normal


def polygon_face_coordinate(verts, face, points):
    """Fraction s in [0, 1] along edge ``face`` of the projection of ``points`` ([N, 2]) onto that edge."""
    v, e, _ = polygon_edges(verts, points.shape[0], points)
    L = _lib(points)
    v0, ef = _gather_face(v, face), _gather_face(e, face)
    s = ((points - v0) * ef).sum(-1) / (ef ** 2).sum(-1)
    return L.clip(s, 0.0, 1.0) if L is np else s.clamp(0.0, 1.0)


def polygon_face_point(verts, face, s):
    """Point on edge ``face`` at fraction ``s``: [N, 2]."""
    v, e, _ = polygon_edges(verts, s.shape[0], s)
    return _gather_face(v, face) + s[:, None] * _gather_face(e, face)


def polygon_face_frame(verts, face, n, like):
    """(outward normal, tangent) unit vectors of edge ``face``: each [N, 2]."""
    _, e, normal = polygon_edges(verts, n, like)
    L = _lib(like)
    ef = _gather_face(e, face)
    t = ef / (L.sqrt((ef ** 2).sum(-1, keepdims=True)) if L is np else ef.norm(dim=-1, keepdim=True))
    return _gather_face(normal, face), t


def direction_to_face_frame_2d(verts, face, u, v):
    """In-plane direction (u, v) -> (mu_n, mu_t) in the frame of edge ``face`` (mu_n > 0 means outgoing).
    The out-of-plane cosine w is unchanged by this rotation, so it is not an argument."""
    nrm, t = polygon_face_frame(verts, face, u.shape[0], u)
    return u * nrm[:, 0] + v * nrm[:, 1], u * t[:, 0] + v * t[:, 1]


def direction_from_face_frame_2d(verts, face, mu_n, mu_t):
    nrm, t = polygon_face_frame(verts, face, mu_n.shape[0], mu_n)
    return mu_n * nrm[:, 0] + mu_t * t[:, 0], mu_n * nrm[:, 1] + mu_t * t[:, 1]


def polygon_signed_distances(verts, points):
    """Signed distance of ``points`` to every edge line (positive = outside): [N, V]."""
    v, _, normal = polygon_edges(verts, points.shape[0], points)
    return ((points[:, None, :] - v) * normal).sum(-1)


def polygon_nearest_face(verts, points):
    """Face whose edge line is closest to (or furthest past) each point. Use only to label data that lacks an
    explicit exit-face record; when the MC recorded the crossing, prefer that."""
    d = polygon_signed_distances(verts, points)
    L = _lib(points)
    return d.argmax(-1) if L is np else d.argmax(dim=-1)


# --------------------------------------------------------------------------------------------------------------------
# 3D polyhedron geometry (planar faces)
# --------------------------------------------------------------------------------------------------------------------
def polyhedron_face_frames(topo: CellTopology, verts, n, like):
    """Per-face origin o, affine axes (a_ax, b_ax) and outward unit normal: each [N, F, 3]."""
    if topo.faces is None:
        raise ValueError(f"Topology '{topo.name}' has no 3D face table")
    v = _batched_verts(verts, n, like)
    L = _lib(v)
    first = [f[0] for f in topo.faces]
    second = [f[1] for f in topo.faces]
    last = [f[-1] for f in topo.faces]
    o = v[:, first]
    a_ax = v[:, second] - o
    b_ax = v[:, last] - o
    nrm = L.cross(a_ax, b_ax) if L is np else torch.linalg.cross(a_ax, b_ax, dim=-1)
    nrm = nrm / (L.sqrt((nrm ** 2).sum(-1, keepdims=True)) if L is np else nrm.norm(dim=-1, keepdim=True))
    return o, a_ax, b_ax, nrm


def polyhedron_face_coordinates(topo: CellTopology, verts, face, points):
    """Affine (a, b) of ``points`` ([N, 3]) in the frame of face ``face``: p = o + a A + b B (least squares)."""
    o, A, B, _ = (_gather_face(x, face) for x in polyhedron_face_frames(topo, verts, points.shape[0], points))
    d = points - o
    aa, bb, ab = (A * A).sum(-1), (B * B).sum(-1), (A * B).sum(-1)
    da, db = (d * A).sum(-1), (d * B).sum(-1)
    det = aa * bb - ab * ab
    return (bb * da - ab * db) / det, (aa * db - ab * da) / det


def polyhedron_face_point(topo: CellTopology, verts, face, a, b):
    o, A, B, _ = (_gather_face(x, face) for x in polyhedron_face_frames(topo, verts, a.shape[0], a))
    return o + a[:, None] * A + b[:, None] * B


def polyhedron_signed_distances(topo: CellTopology, verts, points):
    o, _, _, nrm = polyhedron_face_frames(topo, verts, points.shape[0], points)
    return ((points[:, None, :] - o) * nrm).sum(-1)


def polyhedron_nearest_face(topo: CellTopology, verts, points):
    d = polyhedron_signed_distances(topo, verts, points)
    return d.argmax(-1) if _lib(points) is np else d.argmax(dim=-1)


def direction_to_face_frame_3d(topo: CellTopology, verts, face, dirs):
    """Direction [N, 3] -> (mu_n, t1, t2) with mu_n the cosine to the outward normal and (t1, t2) the components
    along an orthonormal tangent basis (t1 parallel to the face's first edge)."""
    o, A, _, nrm = (_gather_face(x, face) for x in polyhedron_face_frames(topo, verts, dirs.shape[0], dirs))
    L = _lib(dirs)
    t1 = A / (L.sqrt((A ** 2).sum(-1, keepdims=True)) if L is np else A.norm(dim=-1, keepdim=True))
    t2 = L.cross(nrm, t1) if L is np else torch.linalg.cross(nrm, t1, dim=-1)
    return (dirs * nrm).sum(-1), (dirs * t1).sum(-1), (dirs * t2).sum(-1)


# --------------------------------------------------------------------------------------------------------------------
# Valid-class masks (one classifier over mixed cell types)
# --------------------------------------------------------------------------------------------------------------------
def valid_class_mask(num_faces_per_sample, max_faces: int, interior: Sequence[str]):
    """Boolean [N, max_faces + len(interior)] mask of classes that exist for each sample.

    Lets one classifier with ``max_faces`` face slots serve cells with fewer faces (e.g. triangles and quads in one
    mesh, num_faces 3 or 4): logits of the missing face slots are masked to -inf before the softmax.
    """
    nf = np.asarray(num_faces_per_sample).astype(np.int64)
    faces = np.arange(max_faces)[None, :] < nf[:, None]
    return np.concatenate([faces, np.ones((nf.shape[0], len(interior)), dtype=bool)], axis=1)
