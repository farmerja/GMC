"""
SCR-GMC cost model: speedup at equal noise vs analog MC as a function of the dose resolution inside a transport block.

    python validation/scr_cost_model.py                       # the 1 mm water / Cu tables of the results page
    python validation/scr_cost_model.py --gmc_us 33 --mc_us 100 --block_um 1000 --h_um 10,5,2,1

Per transit, at equal noise and with the samples per transit K chosen optimally, GMC + SCR costs

    C(n) = ( sqrt(c * A) + sqrt(s * b * n) )^2,      n = L / h (sub-cells per block side),

because its variance relative to one analog-MC run is V(K) = A + b n / K and its cost is c + s K:
    c  GMC chain cost per transit (us)
    s  kernel cost per sample (0.33 us eager, ~0.23 us with torch.compile; RTX 5080)
    b  Sobol sampling-noise constant (~0.35, fitted on 100 um Cu at n = 32, K = 64-256)
    A  variance left with unlimited samples, relative to MC (0.17 at n = 32; it fell from 0.46 at n = 8, so this is
       conservative at larger n)
The optimum is K* = sqrt(b n c / (A s)), and the speedup is MC cost per transit / C(n). The analog MC's cost does not
depend on the tally resolution. Check: 100 um Cu at n = 32 is predicted at 4.4x; measured 4.0-4.1x.
"""
import argparse
import math

S_EAGER, S_COMPILED, B, A = 0.33, 0.23, 0.35, 0.17
# GMC us/transit (measured chains), MC us/transit (water 1 mm: chain_timing full beam 427 us/primary / 4.29 transits;
# Cu 1 mm: cell-size sweep, 322 us at 1 MeV - 678 us at 10 MeV)
CASES = {"water 1 mm": (33.0, 427.0 / 4.29), "Cu 1 mm": (30.0, 330.0)}


def cost(n, c, s, b=B, a=A):
    return (math.sqrt(c * a) + math.sqrt(s * b * n)) ** 2


def k_opt(n, c, s, b=B, a=A):
    return math.sqrt(b * n * c / (a * s))


def break_even_n(c, mc, s, b=B, a=A):
    return max(0.0, math.sqrt(mc) - math.sqrt(c * a)) ** 2 / (s * b)


def table(name, c, mc, block_um, hs):
    print(f"\n{name}: GMC {c:.0f} us/transit, MC {mc:.0f} us/transit")
    print(f"  {'voxel':>9s} {'n = L/h':>8s} {'K*':>6s} {'speedup':>8s} {'compiled':>9s}")
    for h in hs:
        n = block_um / h
        print(f"  {h:>7g}um {n:>8.0f} {k_opt(n, c, S_EAGER):>6.0f} {mc / cost(n, c, S_EAGER):>7.2f}x "
              f"{mc / cost(n, c, S_COMPILED):>8.2f}x")
    for s, lab in ((S_EAGER, "eager"), (S_COMPILED, "compiled")):
        ns = break_even_n(c, mc, s)
        print(f"  break-even ({lab}): n* = {ns:.0f}  ->  voxel {block_um / ns:.2f} um")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmc_us", type=float, default=None)
    ap.add_argument("--mc_us", type=float, default=None)
    ap.add_argument("--block_um", type=float, default=1000.0)
    ap.add_argument("--h_um", default="10,5,2,1,0.5,0.1,0.01")
    a = ap.parse_args()
    hs = [float(v) for v in a.h_um.split(",")]
    print(f"check, 100 um Cu at n = 32: predicted {(365 / 4.62) / cost(32, 31.6, S_EAGER):.1f}x (measured 4.0x)")
    if a.gmc_us is not None and a.mc_us is not None:
        table("custom", a.gmc_us, a.mc_us, a.block_um, hs)
    else:
        for name, (c, mc) in CASES.items():
            table(name, c, mc, a.block_um, hs)


if __name__ == "__main__":
    main()
