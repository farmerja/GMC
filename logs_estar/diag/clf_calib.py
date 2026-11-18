"""Classifier calibration: mean predicted P(n_rel>=1), P(flag) vs empirical, per energy bin, on test and train data."""
import sys, numpy as np, torch
sys.path.insert(0, "ml")
from secondary_gmc import load_tables, load_models, classifier_data
model, npzs = sys.argv[1], sys.argv[2:]
spec, M = load_models(model, "cuda")
for npz in npzs:
    T = load_tables(npz, NR=spec.NR, max_rows=4_000_000); T["spec"] = spec
    for which, kind in (("B", "boundary"), ("I", "interior")):
        cond, lab = classifier_data(T, kind, augment=False)
        p = M["clf_" + which]["model"].predict_proba(torch.as_tensor(cond, dtype=torch.float32, device="cuda")).cpu().numpy()
        cls, n, fl = spec.decode(np.arange(spec.n_classes()))
        pn = p[:, n >= 1].sum(1); pf = p[:, fl == 1].sum(1); pa = p[:, cls == 4].sum(1)
        _, nl, fll = spec.decode(lab); cl = spec.decode(lab)[0]
        E = T["E0"][np.flatnonzero(T[kind])]
        print(f"== {npz.split('/')[-1]} {kind} rows {len(lab)}  n>=1 pred/emp {pn.sum()/(nl>=1).sum():.4f} ({(nl>=1).sum()})"
              f"  flag {pf.sum()/(fll==1).sum():.4f} ({(fll==1).sum()})  absorbed {pa.sum()/(cl==4).sum():.4f}")
        edges = np.logspace(-3, np.log10(30), 8)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (E >= a) & (E < b)
            if (fll[m] == 1).sum() < 50 and (nl[m] >= 1).sum() < 50: continue
            print(f"   E {a:8.4f}-{b:8.4f}: n>=1 {pn[m].sum()/max((nl[m]>=1).sum(),1):.3f} ({(nl[m]>=1).sum():6d})"
                  f"  flag {pf[m].sum()/max((fll[m]==1).sum(),1):.3f} ({(fll[m]==1).sum():6d})")
