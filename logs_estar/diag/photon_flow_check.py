"""Photon flow on the data's own conditions: sampled g vs true g (mean, mean q, tail fractions), by avail band."""
import sys, numpy as np
sys.path.insert(0, ".")
from ml.secondary_gmc import load_tables, load_models, photon_flow_data, sample_flow, sigmoid
model, npz = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
spec, M = load_models(model, "cuda")
T = load_tables(npz, NR=spec.NR); T["spec"] = spec
rows = np.flatnonzero((T["boundary"] | T["interior"]) & (T["flag"] == 1))
avail = T["E0"][rows] - T["E_out"][rows] - T["E_rel"][rows]
cond, tgt = photon_flow_data(T, augment=False)
g_true = sigmoid(tgt[:, 0].astype(float))
import torch
res = []
for s in range(3):
    z = sample_flow(M["photon"], cond, "cuda", steps, "midpoint", torch.Generator().manual_seed(s))
    res.append(sigmoid(z[:, 0]))
g_s = np.mean(res, 0)
print(f"rows {len(g_true)}  ODE steps {steps}")
print(f"mean g: true {g_true.mean():.4f} flow {np.mean([r.mean() for r in res]):.4f}")
print(f"sum q: flow/true {np.mean([(r*avail).sum() for r in res])/(g_true*avail).sum():.4f}")
for th in (0.5, 0.9, 0.97, 0.99):
    print(f"P(g>{th}): true {(g_true>th).mean():.4f} flow {np.mean([(r>th).mean() for r in res]):.4f}")
for a, b in ((0, 0.1), (0.1, 0.3), (0.3, 1), (1, 5), (5, 100)):
    m = (avail >= a) & (avail < b)
    print(f"avail {a}-{b} MeV n {m.sum():6d}: q flow/true {np.mean([(r[m]*avail[m]).sum() for r in res])/(g_true[m]*avail[m]).sum():.3f}"
          f"  P(g>0.9) true {(g_true[m]>0.9).mean():.3f} flow {np.mean([(r[m]>0.9).mean() for r in res]):.3f}")
