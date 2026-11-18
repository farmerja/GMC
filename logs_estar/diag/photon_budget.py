"""Photon energy budget of boundary transits, GMC vs MC, with errors: flag count, mean q | flag, total q.
MC errors by bootstrap over transits; GMC errors from the spread of independent sampling seeds."""
import sys, numpy as np
sys.path.insert(0, ".")
from ml.secondary_gmc import SecondaryGMC, load_tables
model, npz = sys.argv[1], sys.argv[2]
nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
T = load_tables(npz, max_rows=int(sys.argv[4]) if len(sys.argv) > 4 else None)
rows = np.flatnonzero(T["boundary"]); E0 = T["E0"][rows]
q_mc, f_mc = T["q"][rows], T["flag"][rows]
rng = np.random.default_rng(0)
hi = E0 > 5.0
def stats(q, f, m):
    return np.array([f[m].sum(), q[m].sum() / max(f[m].sum(), 1), q[m].sum()])
boot = {k: [] for k in ("all", "E>5")}
for _ in range(200):
    i = rng.integers(0, len(rows), len(rows))
    for k, m in (("all", np.ones(len(rows), bool)), ("E>5", hi)):
        boot[k].append(stats(q_mc[i], f_mc[i], m[i]))
gm = {k: [] for k in boot}
for s in range(nseed):
    K = SecondaryGMC(model, device="cuda", seed=100 + s)
    g = K.boundary(T["y0"][rows], T["u0"][rows], T["v0"][rows], np.full(len(rows), 0.5), E0)
    for k, m in (("all", np.ones(len(rows), bool)), ("E>5", hi)):
        gm[k].append(stats(g["photon_E"], g["flag"], m))
print(f"{model.split('/')[-2]}: {len(rows)} boundary transits, {nseed} GMC seeds")
for k, m in (("all", np.ones(len(rows), bool)), ("E>5", hi)):
    mc = stats(q_mc, f_mc, m); se_mc = np.std(boot[k], 0); G = np.array(gm[k]); gmean = G.mean(0); se_g = G.std(0, ddof=1) / np.sqrt(len(G))
    r = gmean / mc; se_r = r * np.sqrt((se_mc / mc) ** 2 + (se_g / gmean) ** 2)
    print(f"  {k:4s}: flag count {r[0]:.4f}±{se_r[0]:.4f}   mean q|flag {r[1]:.4f}±{se_r[1]:.4f}   total q {r[2]:.4f}±{se_r[2]:.4f}")
