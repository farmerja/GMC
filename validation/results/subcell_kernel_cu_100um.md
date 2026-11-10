# SCR-GMC sub-cell deposition kernel: 100 µm Cu (2026-11-10)

Code: `ml/subcell_kernel.py` (conditional rational-quadratic spline coupling flow on the unit square, uniform base,
conditioned on the GMC transit: entry, sampled exit, dose, relevant-daughter count). Training data:
`ElectronTransport/analog_cu_100um_1keV-10MeV_log_10M_42_subcell.npz` (64×64 per-segment deposit images, scheme
"rel", now with in-bin moments). Test: `_2M_7` (64×64) and `_400k_7_s256` (256×256, independent histories).

## Models

| Model | Dequantization | Status |
|---|---|---|
| `ml/models/subcell_kernel_cu_100um` | uniform within the 64×64 bin | trained 40k steps; evaluated |
| `ml/models/subcell_kernel_cu_100um_phase` | uniform + sub-bin phase features (diagnostic, grid-tied) | trained 40k steps; evaluated |
| `ml/models/subcell_kernel_cu_100um_mom` | Beta matched to the recorded in-bin moments (α, β ≥ 1) | stopped at 18k of 40k steps (val NLL −2.684, still improving); **not evaluated yet** |

## Results (entry-aligned boundary aggregates, relL2 / MC noise floor, 64×64 test)

| E0 band | uniform: image | uniform: lateral | phase: image | phase: lateral |
|---|---|---|---|---|
| 1–10 keV | 0.354 / 0.006 | 0.354 / 0.006 | 0.014 / 0.002 | 0.014 / 0.002 |
| 10–100 keV | 0.203 / 0.018 | 0.164 / 0.014 | 0.015 / 0.018 | 0.011 / 0.014 |
| 0.1–1 MeV | 0.117 / 0.071 | 0.034 / 0.018 | 0.070 / 0.071 | 0.025 / 0.018 |
| 1–10 MeV | 0.117 / 0.083 | 0.028 / 0.018 | 0.085 / 0.084 | 0.020 / 0.018 |

- The flow fits the 64×64 data to the MC noise floor when it can see the grid phase. The uniform model's low-energy
  bias (70% of the dose in the entry bin vs 98.5% in MC at 1–10 keV) comes from uniform dequantization, whose bin
  edges jump with the entry point; it is a data artifact, not a flow limitation.
- Both uniform-dequantized models fail on the 4× finer 256×256 test (1–10 keV depth profile 86% off; 1–10 MeV image
  bias ~0.38): they learned 1.56 µm boxes. The in-bin moments are the fix; the moment model still needs evaluation.
- Figure: `ml/models/subcell_kernel_cu_100um/dose_mc_vs_scr.png` (0.1–10 MeV, MC on 1.56 µm cells vs kernel from
  one 100 µm cell, relL2 10%, mostly MC noise). Per-model figures: `eval_*/aligned.png`, `examples.png`, `profiles.png`.

## Cost (RTX 5080)

How the numbers were obtained:
- **GMC per transit**: measured once, with the 100 µm Cu model in 20×20 chains (`validation/chain_timing.py`):
  27 µs at 50 ODE steps, 6.5 µs at 10. The network does the same work whatever the cell size, and
  `gmc_size_timing.py` measured it flat across the Cu, Al and water 100 µm and water 1 mm models.
- **Kernel per transit**: measured once, with the 100 µm kernel (batches of 20k): 108 µs at 256 samples,
  1.7 ms at 4096. It depends on samples per transit, not cell size; finer meshes in bigger cells need more samples.
- **Analog MC per transit at each cell size**: measured by the cell-size × energy sweep
  (`ElectronTransport/sweep_cell_size_energy.py`, `figures/sweep_cu.json`):

| Cu cell | 1 MeV | 10 MeV |
|---|---|---|
| 10 µm | 7.2 µs | 6.3 µs |
| 100 µm | 73 µs | 67 µs (86–136 µs per transit in the chains) |
| 1 mm | 322 µs | 678 µs |

Break-even (GMC + kernel = MC per transit, interpolated in the sweep):
- Today (50 steps, 256 samples, ~135 µs): **~150–250 µm Cu** (~100–150 µm using the chain MC costs).
- Tuned (10 steps, 32–64 samples, ~20–35 µs; not yet validated): **~30–60 µm Cu**.

1 mm Cu block (GMC 27 µs + kernel 108–216 µs for 256–512 samples vs MC 322–678 µs):
**~1.5–2.5× at 1 MeV and ~3–5× at 10 MeV today; ~5–8× and ~10–17× with a tuned kernel (~40–70 µs).**
At 1 MeV the gain is range-limited (CSDA ~0.7 mm). Dose resolution realistically ~10–15 µm (64:1, the validated
ratio), ~4–8 µm with 128–256 bins plus moments (unproven).

## Next

1. Evaluate (or finish training) `subcell_kernel_cu_100um_mom` on the 64 and 256 test sets.
2. Transit log in `validation/chain_secondaries.py`; chained GMC + kernel vs fine-mesh MC with `--n_seeds`, timed.
3. 1 mm Cu data (128–256 bins with moments) and kernel.
