"""Digitize the solid ITS/ACCEPT curves of Drumm et al. (2007) Figs. 8a/8b from the 400 dpi PDF bitmaps.

Usage: pdfimages -f 9 -l 10 -png <paper.pdf> <dir>/img ; python validation/drumm_digitize.py <dir> \
           validation/results/drumm_water_cylinder/its_digitized.json
Panel frames and tick calibrations below were read off the bitmaps (img-002 = Fig. 8a, img-003 = Fig. 8b).
"""
import json
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

D = sys.argv[1]
# panel: frame (top, bottom, left, right) px; axis calibration (x at px a, x at px b), (y at px c, y at px d); legend boxes
PANELS = {
    "E_along": ("img-002.png", (19, 711, 128, 983), (129.25, -1, 982.75, 1), (710, 0, 20, 10), [(219, 867, 490, 571)]),
    "E_xm05": ("img-002.png", (33, 712, 1199, 2036), (1199, -1, 2035, 1), (710.4, 0, 34.5, 10), [(1224, 2030, 40, 128)]),
    "E_x0": ("img-002.png", (857, 1494, 129, 987), (130, -1, 986, 1), (1492.5, 0, 858.5, 10), [(209, 923, 880, 964)]),
    "E_xp05": ("img-002.png", (857, 1491, 1197, 2029), (1197.5, -1, 2028.5, 1), (1490, 0, 858, 10), [(1240, 2000, 915, 1005)]),
    "Q_along": ("img-003.png", (23, 716, 133, 988), (133.5, -1, 987.5, 1), (716, -2, 23, 0), [(229, 872, 485, 571)]),
    "Q_xm05": ("img-003.png", (37, 716, 1204, 2041), (1622 - 486.5, -1, 1622 + 486.5, 1), (716, -0.5, 230.5, 0),
               [(1219, 1555, 590, 678), (1615, 2030, 590, 678)]),
    "Q_x0": ("img-003.png", (861, 1497, 134, 992), (567 - 432, -1, 567 + 432, 1), (1497, -0.5, 1041.5, 0), [(199, 918, 1370, 1454)]),
    "Q_xp05": ("img-003.png", (861, 1495, 1202, 2034), (1618.5 - 483.5, -1, 1618.5 + 483.5, 1), (1495, -0.5, 1039, 0),
               [(1220, 1984, 1310, 1403)]),
}

out = {}
for name, (f, (t, b, l, r), (pa, xa, pb, xb), (pc, ya, pd, yb), legends) in PANELS.items():
    a = np.array(Image.open(f"{D}/{f}").convert("L")) >= 128
    m = np.zeros_like(a)
    m[t + 5:b - 4, l + 5:r - 4] = a[t + 5:b - 4, l + 5:r - 4]
    for (c0, c1, r0, r1) in legends:
        m[r0:r1, c0:c1] = False
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sl = ndimage.find_objects(lab)
    keep = np.zeros(n + 1, bool)
    for i, s in enumerate(sl):
        w, h = s[1].stop - s[1].start, s[0].stop - s[0].start
        keep[i + 1] = w >= 120 or h >= 120
    curve = keep[lab]
    cols = np.arange(l + 5, r - 4)
    ys = np.full(len(cols), np.nan)
    prev = None
    for k, c in enumerate(cols):
        rows = np.flatnonzero(curve[:, c])
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 2) + 1)
        cen = np.array([rr.mean() for rr in runs])
        j = 0 if prev is None else int(np.argmin(np.abs(cen - prev)))
        if prev is None:
            j = int(np.argmax([len(rr) for rr in runs]))
        ys[k] = cen[j]
        prev = cen[j]
    x = xa + (cols - pa) * (xb - xa) / (pb - pa)
    y = ya + (ys - pc) * (yb - ya) / (pd - pc)
    ok = np.isfinite(y)
    out[name] = {"x": np.round(x[ok], 4).tolist(), "y": np.round(y[ok], 4).tolist()}
    print(name, ok.sum(), "of", len(cols), "columns; y range", np.nanmin(y), np.nanmax(y))
json.dump(out, open(sys.argv[2], "w"))
