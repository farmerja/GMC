# 20 MeV beam on an infinite water cylinder: analog MC, GMC and SCR-GMC vs ITS (Drumm et al. 2007)

Generated 2026-11-11/12. Benchmark: C. R. Drumm, W. C. Fan, L. Lorence, J. Liscum-Powell, "An Analysis of the
Extended-Transport Correction with Application to Electron Beam Transport", Nucl. Sci. Eng. 155, 355–366 (2007),
Sec. IV.A, Figs. 8a/8b (ITS/ACCEPT Monte Carlo, 3.2 billion histories). Script: `validation/drumm_cylinder.py`
(`mc`, `gmc`, `plot`); the paper's ITS curves were digitized from the PDF bitmaps (`validation/drumm_digitize.py` →
`drumm_water_cylinder/its_digitized.json`, uncertain to about one line width: ±0.05 in energy, ±0.005 in charge).

## Problem and how it maps onto our code

| | paper | here |
|---|---|---|
| geometry | water cylinder, Ø 2 cm, infinite along z | our 2D infinite-z geometry: the circle in x–y |
| beam | 20 MeV, +x, uniform in z, triangular in y, FWHM 0.25 cm | identical (the paper's sampling formula) |
| electron cutoff | 0.1 MeV | 1 keV (the GMC training cutoff) |
| photons | "not transported" | banked at birth; shown both ways (energy escapes / deposited where emitted) |
| units | MeV/cm³ and e⁻/cm³ per source e⁻ per cm of beam | tally per primary / voxel area (checked: 4 /cm peak fluence × 2.05 MeV cm²/g ≈ 8) |
| charge deposition | electrons stopped − electrons created | same; fine maps count only electrons born > 0.1 MeV (new stepper tally) |

Runs (500k primaries each, same source):
- **analog MC, true circle**: the circle on 31 µm voxels (vacuum outside), tallied on 125 µm bins.
- **analog MC, 1 mm staircase** (2 seeds): the geometry the GMC sees (20 × 20 cells of 1 mm, a cell is water if its centre is
  inside the circle, vacuum otherwise), tallied on 125 µm bins. Like-for-like reference and noise floor for the GMC.
- **GMC** on the staircase: new model `ml/models/secondary_gmc_water_1mm_30MeV` (the old water 1 mm model stops at
  10 MeV), a vacuum streaming kernel for the empty cells, 2 seeds, both photon conventions.
- **GMC + SCR**: new kernel `ml/models/subcell_kernel_water_1mm_30MeV`, 8 × 8 sub-cells (125 µm), Sobol K = 32.

## Results

![](drumm_water_cylinder/fig8a_energy.png)
![](drumm_water_cylinder/fig8b_charge.png)

[Dose maps](drumm_water_cylinder/dose_maps.png). Numbers: `drumm_water_cylinder/metrics.json`.

**Analog MC vs ITS.**
- **Energy deposition depends on how bremsstrahlung is counted, and the paper does not say exactly.** Energy per cm
  of depth (lateral integral of the across-beam profiles): ITS 2.35 MeV/cm; ours 2.04–2.10 with photons escaping
  (−11 to −13 %), 2.41–2.43 with photon energy deposited where emitted (+3 to +4 %). Our photons-escape value is what
  the collision stopping power gives (ESTAR 2.05 MeV cm²/g at 20 MeV, plus obliquity), so ITS/CEPXS evidently keep
  most of the radiative loss locally ("no photon transport"). The figures use the local convention.
- In that convention the across-beam profiles match ITS in shape and height at all three depths (mean deviation in
  the core +2.6 %, +4.0 %, +6.6 % at x = −0.5, 0, 0.5; peaks 8.86 / 7.64 / 6.03 vs ITS 8.50 / 7.54 / 6.15).
  Along the axis our MC is steeper: about +8 % near entry, −9 % at the exit (mean −1.4 %). The cause is faster
  lateral broadening (see "Why ... differ" below).
- **Charge deposition** (electrons born > 0.1 MeV, 0.5 mm bins): matches ITS once ITS is averaged the same way (dashed
  steps and open markers in Fig. 8b). Across the beam, MC vs ITS-in-our-bins differ by 0.021–0.028 e⁻/cm³ rms against an
  MC noise of 0.017–0.019 per bin; the GMC's 1 mm cells vs ITS averaged over 1 mm cells differ by 0.011–0.015. Along
  the axis our 1 mm strip gives −0.350 / −0.270 / −0.139 at x = −0.5 / 0 / 0.5, ITS averaged over the same strip
  −0.349 / −0.285 / −0.209. Only x = +0.5 is off (see below).
- Geometry: true circle vs 1 mm staircase differ by 5.6 % rel-L2 on the 125 µm map (MC noise 1.3 %), mostly at the
  surfaces; the profiles in the figures are indistinguishable.

**GMC vs analog MC (1 mm staircase, like for like).** Error split of one run, relative L2 over voxels > 1 % of max:

| map | analog MC noise | GMC noise | GMC bias |
|---|---|---|---|
| 1 mm cells, photons local | 0.67 % | 0.60 % | **0.62 %** (at the noise floor) |
| 1 mm cells, photons escape | 0.20 % | 0.31 % | 1.73 % |
| 125 µm, GMC + SCR, photons escape | 0.94 % | 0.87 % | **3.3 %** |
| 125 µm, GMC flat in cells, photons escape | 0.94 % | 0.31 % | 15.7 % |
| 125 µm, GMC + SCR, photons local | 4.96 % | 1.32 % | 8.1 % |
| charge, 1 mm cells | 9.0 % | 9.7 % | 5.2 % (below noise) |

- **Energy closes.** Deposited fraction of E0, photons local: GMC 0.2370 / 0.2368, MC 0.2372. With photons escaping
  the GMC deposits 1.3 % more (0.2060 vs 0.2033) and emits correspondingly less photon energy (0.031 vs 0.034): the
  dose/photon split of the top-energy transits is off, the total is not (single cell: dose ratio 1.015 ± 0.009 in the
  12.7–30 MeV bin). Net charge per primary: GMC −0.072, MC −0.069.
- **SCR** turns the 15.7 % error of a flat-in-cell map into 3.3 % at 125 µm with MC-level noise; across-beam peaks
  within 0–4 %, rms widths within 1–3 %. Remaining bias: a 5 % dip on the axis strip at 4–8 mm depth. That strip
  (and x = ±0.5) lies on transport-cell faces, the kernel's weakest region.
- With photons local, the 125 µm comparison is dominated by where the photon energy goes: the MC puts it on the
  primary track (spiky, 5 % noise), the SCR kernel spreads it like collision dose, so the axis strip reads 6–8 % low
  in the first centimetre. Not a transport error, a consequence of emulating the paper's convention.
- **Cost** (500k primaries, CPU shared between runs, so ±30 %): analog MC 1561 s (3.1 ms/primary); GMC chain 375 s
  + SCR tally 143 s (25.8 water transits/primary, 11 µs tally per transit). 4.2× for the chain, 3.0× with the tally.

## Why the along-axis energy and the raw charge plots differ from ITS (diagnosed 2026-11-12)

**Energy along the axis** has two separate effects stacked:
1. *Level:* how bremsstrahlung energy is counted. It is ~15 % of the on-axis dose. ITS sits between our two conventions
   (local: +3 to +9 %; escaping: −11 to −13 %), closer to local deposition; the paper does not state its convention.
2. *Slope:* our beam broadens faster with depth. RMS width of the across-beam profiles vs ITS: +1 % / +5 % / +7 % at
   x = −0.5 / 0 / 0.5. Removing the triangle's own width (rms 0.102 cm), the scattering part is +12 / +17 / +14 %
   wider. The lateral integrals stay flat, so a wider beam means a lower axis value, and the gap grows with depth. This
   is the +8 % (entry) → −9 % (exit) trend.
   - Slab test (20 MeV pencil, plain water): our multiple-scattering θ0 (Gaussian fit to the central 98 %) is
     **4–5 % above Lynch–Dahl** at every thickness (28.5 vs 27.3 mrad at 1 mm; 106.6 vs 101.5 mrad at 10 mm). That
     accounts for roughly a third of the extra spread; the rest is open (part may be ITS's own multiple-scattering
     model).
   - GMC + SCR adds a separate ~5 % dip there: the y = 0 line lies on transport-cell faces, the kernel's weakest
     region.

**Charge** in the first version of Fig. 8b compared different things:
1. *Strip width (main cause):* the on-axis trough is ~1.5 mm wide. Our along-axis strip is 1 mm (needed for
   statistics), the GMC's is 2 mm (cells), while ITS resolves the line. ITS averaged over 1 mm is −0.349 / −0.285 /
   −0.209 instead of −0.44 / −0.33 / −0.22 on the line, and matches our MC at x = −0.5 and 0. Fig. 8b now overlays these
   strip-averaged ITS values.
2. *Statistics:* 500k histories vs ITS's 3.2 billion, and charge is a small difference of creations and stops.
   ±0.02 e⁻/cm³ per 0.5 mm bin across the beam, ±0.03 along the axis.
3. *Faster broadening:* the remaining gap at x = +0.5 (−0.139 vs −0.209 at equal strip width) is the same physics as
   the energy slope.
4. *Not the cause:* the 1 keV cutoff. Counting all electrons vs only those born > 0.1 MeV changes the axis values by
   < 0.02, except in the first 2 mm.

## Analog vs condensed history: method or data?

**In principle the transport method should not matter here.** Condensed history (ITS) is an approximation to the
same Boltzmann equation that analog (event-by-event) MC samples exactly. Its multiple-scattering distributions
(Goudsmit–Saunderson) are built from the single-scattering cross section, and the result converges to the analog one as
the step size shrinks. For a thick, homogeneous, 20 MeV problem like this, condensed-history step and boundary artifacts
are typically ≤ 1–2 % on dose profiles. Differences of 5–15 % point to the **cross-section data**, not the method.

**Test.** The quantity that sets multiple scattering is the transport cross section 1/λ₁ = Σ_channels Σ_c ⟨1 − μ⟩_c.
We computed what our analog code actually samples: elastic ⟨1 − μ⟩ integrated exactly from the tabulated EEDL pdfs with
the sampler's log-E mixing between tables, the small-angle channel analytically, and ionization deflection by sampling.
We compared it with Molière screened-Rutherford theory with Z(Z + 1), the basis of the condensed-history
multiple-scattering models and of Lynch–Dahl.

| E (MeV) | 0.1–0.256 | 0.3 | 0.5 | 1 | 2 | 3 | 5 | 7 | 9 | 10–25 |
|---|---|---|---|---|---|---|---|---|---|---|
| Water: ours / Molière | 1.05–1.10 | 1.16 | 1.63 | 2.70 | 4.40 | 5.02 | 4.76 | 3.48 | 1.84 | 0.98–1.01 |
| Cu | 1.03–1.12 | 1.16 | 1.64 | 2.64 | 4.23 | 5.47 | 5.72 | 4.40 | 2.26 | 1.00–1.07 |
| Al | 1.01–1.06 | 1.13 | 1.58 | 2.44 | 3.71 | 4.23 | 4.09 | 3.08 | 1.71 | 0.98–1.01 |

- **At EEDL's tabulated energies our scattering is right.** At 0.256, 10 and 21.25 MeV the ratio is 0.98–1.05. That
  validates the data, the small-angle channel and the ionization deflection (the electron–electron term), so the
  20 MeV primaries here scatter as in ITS.
- **Between 0.3 and 9 MeV it is wrong by up to 5.7×, in every material.** The EEDL files tabulate the large-angle
  elastic angular distributions (MF26/MT525) only at 0.128, 0.256, 10, 21.25 MeV …, with **nothing between 0.256 and
  10 MeV**. This is in the raw ENDF files (EPICS2017 O, EPICS2023 Cu); our converter reproduces them faithfully. The
  sampler mixes the two bracketing tables with log-E weights. At 5 MeV about one large-angle collision in five takes
  its angle from the 256 keV distribution, which is ~40× broader, because the width scales like 1/(pβ)².
- **Impact on this benchmark.** The 16–20 MeV primaries are fine. Knock-on electrons of 0.3–9 MeV, which carry roughly
  10–20 % of the collision energy and set where charge is deposited, scatter several times too strongly. That is a
  plausible contributor to the extra lateral spread and to the charge differences; the size has to be measured after
  the fix.
- **Impact on everything else.** Every analog data set and GMC model with electrons between 0.3 and 9 MeV inherits it:
  Cu, Al and water, 1 keV–10 MeV and 30 MeV. GMC-vs-MC comparisons stay valid (both share the physics), but comparisons
  with experiment between 0.3 and 9 MeV would be off. Examples: Lockwood depth-dose at 0.3–1 MeV, and MeV backscatter.
- **The 4–5 % θ0 excess over Lynch–Dahl at 20 MeV is not explained by λ₁**, which matches Molière within ~1 % there.
  It is more likely the angular shape: at 20 MeV, 8 % of large-angle collisions use the 10 MeV table. Lynch–Dahl's own
  accuracy (a few % for low Z) may also contribute. A Goudsmit–Saunderson comparison will settle it.

**Fix options** (checked 2026-11-12 against the exact transport-cross-section calculation):
- *Rescaling the tables in momentum transfer* (keep t·p² fixed, the Born-approximation scaling of screening) only halves
  the error: still up to 2.4× at 2 MeV, and it makes 15 MeV worse (1.30). Reason: at the two ends of the gap EEDL
  departs from screened Rutherford in different ways. At 0.256 MeV the large-angle cross section is 2.0× screened
  Rutherford and ⟨1 − μ⟩ ≈ 0.5×; at 10 MeV both are ≈ 1.0. Those are non-Born effects (Dirac partial waves, αZ/β), and
  no simple rescaling of the 256 keV table reproduces the MeV shape: at 2 MeV the two rescaled tables disagree by 4×.
- **Recommended, real fix: dense elastic data from ELSEPA** (Dirac partial-wave analysis; the source PENELOPE uses, and
  Geant4's DPWA model ships ELSEPA tables in its G4EMLOW data). Replace EEDL MT525/MT526 angular data with tables on a
  dense energy grid (e.g. 10–20 per decade, 1 keV–100 MeV), keeping the analog sampler unchanged. This also improves
  keV elastic. Cost: a new parser/converter, H, O, Al and Cu tables, and re-validation.
- *Interim fix if ELSEPA is not available:* in the gap, sample the analytic Wentzel / screened-Rutherford shape at the
  actual energy, times a correction ratio R(u, E) = p_EEDL / p_SR measured at the two bracketing tables in a
  screening-reduced angle u, interpolated in log E (rejection sampling). It is exact at the tabulated energies and
  follows the analytic energy dependence in between. Midgap accuracy cannot be verified without reference data
  (expect tens of %, vs up to 5.7× now).
- Validation after either fix: the transport-cross-section ratio table (≈ 1 apart from known non-Born effects), slab θ0
  vs Lynch–Dahl / Goudsmit–Saunderson at 1–20 MeV, MeV backscatter coefficients (Tabata), Lockwood depth-dose. Then
  regenerate data, retrain, and rerun this benchmark.

## Physics models: our analog MC vs the paper's codes

The paper compares CEPTRE (deterministic Boltzmann-CSD, multigroup cross sections from CEPXS, its Eqs. 1–9) with ITS/
ACCEPT (condensed-history MC). The ITS physics below is from the ITS documentation; the paper only cites the code.

**Same equations:**
- The knock-on deflection angle: the paper's deterministic cosine μ₀ = √[T_p(T + 2) / (T(T_p + 2))] (reduced
  energies) is exactly our binary-collision formula cos²θ = T(E + 2mc²) / (E(T + 2mc²)). Both use free-electron Møller
  kinematics.
- The screened-Rutherford shape: the paper's elastic test kernel is the Wentzel/Molière form our small-angle channel uses.
- Energy and charge deposition definitions (its Eqs. 12–13): ours are the same quantities, counted per collision.

**Different:**

| | CEPTRE (deterministic) | ITS/ACCEPT (condensed history) | ours (analog) |
|---|---|---|---|
| Elastic angles | screened Rutherford, truncated to a forward δ-function plus 8 Legendre terms (ETC P7: exact first 9 moments) | Goudsmit–Saunderson multiple scattering per step (screened Rutherford + Mott correction) | every collision sampled: EEDL tables (μ < 0.999999) + Wentzel/Molière below 1.4 mrad |
| Small energy losses | continuous slowing down (restricted stopping power): no straggling, no deflection | continuous loss per step + sampled straggling (Landau / Blunck–Leisegang) | every ionization and excitation sampled; natural straggling; excitation = EEDL mean loss, no deflection |
| Large energy losses | Møller knock-ons with deterministic cosine (ETC → δ-function downscatter) | Møller knock-ons and bremsstrahlung above thresholds, explicit | every knock-on above 1 keV, plus EADL relaxation (Auger, fluorescence) |
| Discretization | 40 log energy groups, S12, 6400 elements | log energy grid of steps | none |
| Cutoff | 0.1 MeV | 0.1 MeV | 1 keV |
| Data | CEPXS (condensed-phase stopping powers with density effect) | ICRU-37-type stopping powers, Seltzer–Berger bremsstrahlung | EEDL/EADL free atoms, Bragg additivity |
| Geometry | 2D | 3D, 30 cm cylinder, middle 10 cm | 2D infinite z (exact for this problem) |

At 20 MeV the data differences are: our water collision stopping power is 1.3–1.8 % above ESTAR at 10–30 MeV (free
atoms have no density effect), and our radiative stopping power is 0–7 % below ESTAR.

**Known trade-offs:**
- *Analog (ours).*
  - Pros: no step-size artifacts and no multiple-scattering theory, so it is exact for its data. Interfaces and µm–nm
    cells are handled correctly, which is why the GMC uses it.
  - Cons: cost scales with the number of collisions (~100–300× ITS per history here). The data are applied literally:
    gaps or errors in single-collision data (as above) are not smoothed by any theory. No condensed-phase effects.
    Excitation with mean loss only under-straggles slightly at keV.
- *Condensed history (ITS).*
  - Pros: cheap per history, well tuned for thick MeV targets.
  - Cons: results depend on step size and boundary crossing; multiple-scattering theory assumes small energy loss per
    step; Landau straggling is poor for thin steps; it breaks down when cells approach the mean free path (the GMC's
    regime).
- *Deterministic Boltzmann-CSD (CEPTRE).*
  - Pros: no noise, the whole field in one solve, adjoint and sensitivity calculations come naturally.
  - Cons: memory (6 GB here), ray effects (hence the paper's extended first-collision source), energy groups, no
    straggling with continuous slowing down. Angular truncation: the paper's Figs. 3–6 show a standard Legendre
    expansion underestimates the high moments and ETC overestimates them; P7 is inadequate for the Møller
    deterministic-cosine kernel at μ̄ = 0.99, where P15 is needed.

## Timing

One machine: 16 CPU threads, RTX 5080. The MC and GMC runs overlapped and shared the CPU, so treat them as ±30 %.

| step | wall time | notes |
|---|---|---|
| training data, 10M primaries, 1 keV–30 MeV | 17.5 min | analog MC with segment and sub-cell recording |
| test data, 2M | 3.5 min | |
| GMC training | ~9 min | shared the GPU with the kernel training |
| SCR kernel training (40k steps) | 27 min | |
| analog MC, 1 mm staircase, 500k | 26 min (3.1 ms/primary) | ~50 ms of thread time per primary |
| analog MC, true circle, 500k | 25.5 min (3.1 ms/primary) | 31 µm voxels cost nothing extra |
| GMC chain, 500k | 6.3 min (0.75 ms/primary) | 25.8 water transits/primary, ~29 µs each |
| SCR tally, 500k | 2.4 min | ~11 µs per transit (Sobol, K = 32) |

- **GMC per run:** the chain is 4.2× faster than the MC, 3.0× with the SCR tally. The one-time cost is ~1 h (data and
  training).
- **The paper** (32-processor 2.2 GHz cluster): CEPTRE took 38 min and 6 GB. ACCEPT took 3.2 billion histories in
  670 min, ≈ 0.4 ms of core time per history. Ours is ≈ 50 ms of thread time per history. Allowing for newer hardware,
  analog transport to 1 keV costs ~100–300× more per history than condensed history with a 0.1 MeV cutoff: each
  20 MeV primary makes ~170 knock-ons above 1 keV in 2 cm.

## Next session

1. **Fix the elastic angular data across the EEDL table gap (0.256–10 MeV)**, preferably with ELSEPA tables (see
   "Fix options").
   Re-check the transport-cross-section ratio table (should be ~1.0 everywhere). Then regenerate data, retrain, and
   rerun this benchmark to measure how much of the lateral-spread and charge differences it explains. Every analog data
   set and GMC model with electrons in 0.3–9 MeV needs regenerating.
2. **The remaining θ0 excess at 20 MeV** (4–5 % over Lynch–Dahl while λ₁ agrees with Molière): compare our slab
   angular distribution with a Goudsmit–Saunderson / Molière calculation from our own cross sections, and check the
   angular-shape mixing between the 10 and 21.25 MeV tables. Lower priority: Lynch–Dahl itself is only good to a few %.
3. **More histories for charge** (~5M): profiles on narrower strips (≤ 0.5 mm) to compare with ITS's resolved
   trough.

## Mirror symmetry bug found here (fixed at inference, needs a training fix)

The first GMC runs drifted sideways: the beam centroid moved to −90 µm by the exit (MC: within ±2 µm), growing
quadratically with depth, which made cells below the axis up to 8 % high and above it up to 12 % low (cell rel-L2
2.6 %). Cause: a constant mean exit-direction bias of about −6e-4 in v per 1 mm transit, under 2 % of the 20 MeV
angular spread per cell, accumulated over ~20 cells. `secondary_gmc.classifier_data` / the flow data augment only
**interior** rows with D4 copies; boundary (entry) rows are not mirror-augmented, so the entry models are only
approximately y-symmetric.

Fix used here: `MirrorY` in `drumm_cylinder.py` (`--mirror`) calls the model on the y-mirrored state (y → 1 − y,
v → −v) for a random half of the transits and mirrors the outputs, daughters and exit class back. Exact in
expectation, no cost. Centroid drift → within ±6 µm, cell bias 2.6 % → 0.6 %. All numbers above use it.
**To do:** add y-mirror augmentation of boundary rows in `ml/secondary_gmc.py` (and/or put `MirrorY` into
`SecondaryGMC`). The 100 µm Cu/Al chains used broad 2-cell beams and may hide the same drift; worth re-checking.

## Caveats

- The SCR kernel was trained on the 2M (seed 7) sub-cell set: the 10M sub-cell file was not written (the regenerating
  process was stopped for low memory), so there is no held-out kernel evaluation. The GMC itself was trained on the
  10M set and checked on the 2M set (`secondary_cell_water_1mm_30MeV/`: entry dose 1.004, daughters 12886 vs 12905).
- The first 10M training file had a single corrupted byte (zip CRC and segments SHA-1 failed; regenerated with the
  same seed, the rewrite differs from it in exactly one byte). Possibly a hardware/memory event; worth watching.
- Cutoff 1 keV vs the paper's 0.1 MeV: irrelevant for energy at these bins; for charge, emulated with the birth
  threshold. GMC charge is only available per transport cell (absorption positions are not modelled).

## Reproduce

```bash
# ElectronTransport venv: training / test data (1 keV - 30 MeV, 1 mm water, 32 x 32 sub-cell images)
for s in "10000000 42 10M" "2000000 7 2M"; do set -- $s
  (cd ../ElectronTransport && .venv/bin/python generate_analog_batches.py --N $1 --seed $2 \
      --out analog_water_1mm_1keV-30MeV_log_$3_$2 --E_low 1e-3 --E_high 30 --box 1e-1 --log_energy \
      --photons_irrelevant --simulated_only --batch_size 500000 --material Water --subcell_bins 32)
done
# GMC venv: models (GMC ~10 min, kernel ~27 min on the RTX 5080)
python ml/secondary_gmc.py train --train_npz ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_10M_42.npz \
    --out ml/models/secondary_gmc_water_1mm_30MeV
python ml/subcell_kernel.py train --train_npz ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_2M_7.npz \
    --out ml/models/subcell_kernel_water_1mm_30MeV --batch 32768
python validation/secondary_cell_compare.py --test_npz ../ElectronTransport/analog_water_1mm_1keV-30MeV_log_2M_7.npz \
    --model ml/models/secondary_gmc_water_1mm_30MeV/secondary_gmc.pth --out_dir validation/results/secondary_cell_water_1mm_30MeV
# analog MC references (ElectronTransport venv, ~26 min each)
for r in "stair 123" "stair 1123" "circle 123"; do set -- $r
  ../ElectronTransport/.venv/bin/python validation/drumm_cylinder.py mc --geom $1 --seed $2 --N 500000
done
# GMC + SCR (~18 min per convention, 2 seeds)
M=ml/models/secondary_gmc_water_1mm_30MeV/secondary_gmc.pth; K=ml/models/subcell_kernel_water_1mm_30MeV
D=validation/results/drumm_water_cylinder
python validation/drumm_cylinder.py gmc --model $M --kernel $K --N 500000 --n_seeds 2 --mirror --local_photons \
    --out $D/gmc_n8_500k_mirror_localph.npz
python validation/drumm_cylinder.py gmc --model $M --kernel $K --N 500000 --n_seeds 2 --mirror --out $D/gmc_n8_500k_mirror.npz
python validation/drumm_cylinder.py plot --circle $D/mc_circle_n8_500k_s123.npz --stair $D/mc_stair_n8_500k_s123.npz \
    --stair_noise $D/mc_stair_n8_500k_s1123.npz --gmc $D/gmc_n8_500k_mirror.npz --gmc_lp $D/gmc_n8_500k_mirror_localph.npz
```
