"""Boundary exit flow on true (cls, n_rel, flag): sampled energy loss E0 - E_out vs truth, flagged vs unflagged."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from ml.secondary_gmc import (load_tables, load_models, boundary_cond, sample_flow, decode_exit, scatter_width, NF)
model, npz = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
spec, M = load_models(model, "cuda")
T = load_tables(npz, NR=spec.NR, max_rows=1_000_000); T["spec"] = spec
r = np.flatnonzero(T["boundary"] & (T["cls"] < NF))
E0, cls, nr, fl = T["E0"][r], T["cls"][r], np.minimum(T["n_rel"][r], spec.NR), T["flag"][r]
cond = boundary_cond(spec, T["y0"][r], T["u0"][r], T["v0"][r], E0)
extra = np.concatenate([np.eye(NF, dtype=np.float32)[cls], (nr / spec.NR).reshape(-1, 1), fl.reshape(-1, 1)], 1)
z = sample_flow(M["exit_B"], np.concatenate([cond, extra], 1).astype(np.float32), "cuda", steps, "midpoint",
                torch.Generator().manual_seed(0))
fm = M["exit_B"]
ref = (T["x0"][r], T["y0"][r], T["u0"][r], T["v0"][r]) if (fm.get("rel") or fm.get("scaled")) else None
scale = scatter_width(spec, E0) if fm.get("scaled") else None
E1 = decode_exit(z, cls, E0, spec.E_cut, ref=ref, scale=scale)[5]
loss_g, loss_t = E0 - E1, E0 - T["E_out"][r]
avail_t = loss_t - T["E_rel"][r]
for name, m in (("all", np.ones(len(r), bool)), ("flag 0", fl == 0), ("flag 1", fl == 1)):
    print(f"{name:7s} n {m.sum():8d}  sum loss GMC/MC {loss_g[m].sum()/loss_t[m].sum():.4f}")
f = fl == 1
for a, b in ((0.001, 0.1), (0.1, 1), (1, 5), (5, 31)):
    m = f & (E0 >= a) & (E0 < b)
    print(f"  flag1 E0 {a}-{b}: n {m.sum():6d} loss GMC/MC {loss_g[m].sum()/loss_t[m].sum():.4f}"
          f"  P(loss>0.5E0) MC {(loss_t[m]>0.5*E0[m]).mean():.3f} GMC {(loss_g[m]>0.5*E0[m]).mean():.3f}")
print("flag1 photon q share of MC loss:", T["q"][r][f].sum() / loss_t[f].sum())
