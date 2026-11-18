"""Where do GMC relevant daughters / photon energy go missing? Boundary transits of a test set, MC vs GMC."""
import sys, numpy as np
sys.path.insert(0, ".")
from ml.secondary_gmc import SecondaryGMC, load_tables
model, npz = sys.argv[1], sys.argv[2]
K = SecondaryGMC(model, device="cuda", seed=3)
T = load_tables(npz, NR=K.spec.NR, max_rows=1_000_000)
rows = np.flatnonzero(T["boundary"]); E0 = T["E0"][rows]
g = K.boundary(T["y0"][rows], T["u0"][rows], T["v0"][rows], np.full(len(rows), 0.5), E0)
nr_mc = T["n_rel"][rows]
D = T["dau"]; inb = np.isin(D["parent"], rows)
print(f"transits {len(rows)}  E_cut {K.spec.E_cut}")
print(f"n>=1 transits: MC {np.sum(nr_mc>=1)}  GMC sampled {np.sum(g['n_rel']>=1)}")
print(f"daughters: MC {inb.sum()} (n_rel sum capped {np.minimum(nr_mc,K.spec.NR).sum()}, uncapped {nr_mc.sum()})"
      f"  GMC sampled slots {g['n_rel'].sum()}  emitted {len(g['daughters']['E'])}")
print(f"daughter E sum GMC/MC {g['daughters']['E'].sum()/D['E'][inb].sum():.4f}")
print(f"photon E sum GMC/MC {g['photon_E'].sum()/T['q'][rows].sum():.4f}; flag GMC {g['flag'].sum()} MC {T['flag'][rows].sum()}")
print(f"E_out sum GMC/MC {g['E_out'].sum()/T['E_out'][rows].sum():.5f}; dose GMC/MC {g['dose'].sum()/T['dose'][rows].sum():.4f}")
# per energy bin
edges = np.logspace(-3, np.log10(30), 8)
pe, pm = np.zeros(len(edges)-1), np.zeros(len(edges)-1)
for i,(a,b) in enumerate(zip(edges[:-1], edges[1:])):
    m = (E0>=a)&(E0<b)
    mq = T['q'][rows][m].sum()
    md = D['E'][inb][np.isin(D['parent'][inb], rows[m])]
    gm = np.isin(g['daughters']['parent'], np.flatnonzero(m))
    print(f"  E {a:.4f}-{b:.4f}: photon E {g['photon_E'][m].sum()/max(mq,1e-30):.3f}  "
          f"dau n {gm.sum()}/{len(md)}  dau E {g['daughters']['E'][gm].sum()/max(md.sum(),1e-30):.3f}")
# photon energy share g on flagged
mf = T['flag'][rows]==1
avail_mc = (T['E0']-T['E_out']-T['E_rel'])[rows][mf]
print("MC mean g", np.mean(T['q'][rows][mf]/avail_mc), " MC mean q|flag", T['q'][rows][mf].mean(), " GMC mean q|flag", g['photon_E'][g['flag']==1].mean())
