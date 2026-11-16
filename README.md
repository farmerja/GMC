# Generative Monte Carlo

> **Electron transport work** (branch `Aubrey_Electron`) is documented separately. Start with
> [`docs/electron_gmc_workflow.txt`](docs/electron_gmc_workflow.txt) (section 0 gives the reading order and a
> glossary), then the results page [`validation/results/secondary_gmc_cu_100um.md`](validation/results/secondary_gmc_cu_100um.md).
> The electron training data comes from the companion repository `ElectronTransport` (its README lists every script
> and dataset). The sections below up to "Electron Transport Notes" describe the neutron workflow.

This folder contains a generative Monte Carlo workflow for neutron transport problems. The codebase has three main parts:

- `physics/`: Monte Carlo data generation for 1D, 2D, 3D, and time-dependent transport problems.
- `ml/`: conditional flow matching training and inference code, plus saved models, normalization metadata, and output plots.
- `validation/`: lattice-style validation scripts that compare the generative model against reference Monte Carlo simulations.

## 🚦 Workflow

```mermaid
flowchart LR
    accTitle: Generative Monte Carlo Workflow
    accDescr: Training generates simulation data, normalization metadata, and model checkpoints. Testing loads the saved artifacts and writes comparison plots. Validation runs lattice benchmarks using the trained models.

    train["1. Train<br/>`python ml/train.py ...`"]
    norm["Normalization metadata<br/>`ml/data/*.npz`"]
    models["Model checkpoints<br/>`ml/models/.../*.pth`"]
    test["2. Test<br/>`python ml/test.py ...`"]
    test_out["Testing outputs<br/>`ml/testing_results/...`"]
    validate["3. Validate<br/>`python validation/lattice_steady.py`"]
    val_out["Validation outputs<br/>`validation/results/...`"]

    train --> norm
    train --> models
    norm --> test
    models --> test
    test --> test_out
    norm --> validate
    models --> validate
    validate --> val_out
```

## 📁 Folder Layout

- `physics/`: low-level Monte Carlo samplers and transport kernels.
- `ml/train.py`: generates simulation data, computes normalization metadata, trains the model, and saves checkpoints.
- `ml/test.py`: loads a saved model and normalization metadata, generates inference samples, and writes plots and metrics.
- `ml/data/`: saved normalization metadata such as `norm_params_2d_boundary.npz` and `norm_params_2d_internal.npz`.
- `ml/models/`: saved checkpoints such as `best_model.pth` and `trained_model_final.pth`.
- `ml/training_results/`: loss curves and loss histories from training.
- `ml/testing_results/`: plots and error summaries from testing.
- `validation/`: larger benchmark scripts, including `lattice_steady.py`.

## ▶️ Running Training

Run training from this folder:

```bash
python ml/train.py --data_dim 2d --source_type boundary --sampling random --num_epochs 50 --num_neutrons 100000
```

This command will:

- generate Monte Carlo training data on the fly,
- compute and save normalization metadata,
- train a conditional flow matching model,
- save best and final checkpoints,
- save loss curves and CSV/NumPy loss histories.

For the example above, the main outputs are:

- normalization metadata: `ml/data/norm_params_2d_boundary.npz`
- best checkpoint: `ml/models/2d_boundary/best_model.pth`
- final checkpoint: `ml/models/2d_boundary/trained_model_final.pth`
- training curves and histories: `ml/training_results/2d_boundary/`

Important detail: the training script uses `.npy` as a base path internally, but the normalization metadata is actually written as `.npz`. So when training says it is saving `norm_params_2d_boundary.npy`, the file you should expect on disk is `ml/data/norm_params_2d_boundary.npz`.

If you train with non-default settings, filenames can gain suffixes:

- non-default sampling adds `_<sampling>`, for example `best_model_qmc.pth` and `norm_params_2d_boundary_qmc.npz`
- non-default scheduler adds `_<scheduler>` to model filenames, for example `best_model_cosine.pth`
- if both are non-default, the model suffix becomes `_<scheduler>_<sampling>`

## 🧪 Running Testing

Run testing from this folder:

```bash
python ml/test.py --data_dim 2d --batch_size 30000 --source_type internal
```

For that example, `ml/test.py` will look for:

- model directory: `ml/models/2d_internal/`
- normalization metadata: `ml/data/norm_params_2d_internal.npz`

Testing writes plots and metric summaries to:

- `ml/testing_results/cfm_test_2d_internal_condOT_train-random_infer-random/`

If the model was trained with a non-default sampling strategy, pass it back to the test script so it loads the matching normalization file and checkpoint:

```bash
python ml/test.py --data_dim 2d --source_type internal --train_sampling qmc
```

Useful testing flags:

- `--train_sampling`: selects which training-time normalization/checkpoint suffix to load
- `--infer_sampling`: controls how inference conditions/noise are sampled during testing
- `--scheduler`: should match the scheduler used during training if it was not the default `condOT`
- `--batch_size`: number of inference samples for evaluation

## ✅ Running Validation

Run the lattice steady-state validation from this folder:

```bash
python validation/lattice_steady.py
```

This script uses the 2D trained models and normalization metadata already referenced in the file:

- boundary model: `ml/models/2d_boundary/`
- internal model: `ml/models/2d_internal/`
- boundary normalization base: `ml/data/norm_params_2d_boundary.npy`
- internal normalization base: `ml/data/norm_params_2d_internal.npy`

Those normalization base paths resolve to the saved `.npz` metadata files in `ml/data/`.

Validation outputs are written under:

- `validation/results/`

For `lattice_steady.py`, the current results directory is:

- `validation/results/lattice_2d_16/`

## 📝 Notes

- Run all commands from the root of this folder, not from inside `ml/` or `validation/`.
- Training and testing support several `data_dim` values, including `1d`, `2d`, `3d`, `1d_timedep`, `2d_timedep`, `3d_timedep`, `2d_triangle_timedep`, and `2d_quad_timedep`.
- The example commands above use the default scheduler `condOT` and default training sampling `random`, so they produce the unsuffixed filenames already present in this folder.

## ⚡ Electron Transport Notes

Electron training data comes from the companion repository `ElectronTransport`: 3D directions, 2D position, infinite in z.

- **Current data source:** the analog (event-by-event, EPICS-based) MC, `AnalogMC.py` / `generate_analog_batches.py`. All the `analog_*` datasets and every model trained since 2026-11-03 use it.
- **Legacy data source:** the condensed-history `ElectronMC.py` (stopping powers + Highland scattering). It produced the `gmc_tp_*` / `gmc_training_pairs_*` files and the original `2d_electron_boundary` model. The Highland bullet below applies only to it.

- Generate data with training inputs that include an out-of-plane component (`w_max_angle_deg`), so `r_in = sqrt(u² + v²)` spans `[cos(w_max_angle), 1]`. The hidden `w` is recoverable from `r` (`w² = 1 - r²`), and the model already conditions on `(phi, r)`.
- (Legacy `ElectronMC.py`) Scattering in the MC is memoryless (depends on energy and step only), which is required for cell-to-cell chaining to be consistent with a continuous MC track. Highland's log factor is frozen at `L_ref` (default: the training box size).
- Train: `python ml/train.py --data_dim 2d_electron --electron_npz <file.npz>`
- When chaining with `validation/macro_chaining_plot.py`, pass `--cell_size` equal to the training box (cm) and use `--ode_steps` of 50 or more (the default of 10 can narrow sampled distributions).

## 🚪 Exit-Class Classifier

`ml/exit_classifier.py` learns *which* face a particle leaves through (or whether it stops inside the cell), so the flow only has to model the continuous state. Its input is the flow's own conditioning vector, so it works for any `data_type`.

- Class convention (`ml/cell_topology.py`): faces first, then interior outcomes. 2D polygons use CCW edges (square/canonical quad: 0 bottom, 1 right, 2 top, 3 left); triangle, box and tetra topologies are built in. Use `topology.with_interior("absorbed", "census")` for time-dependent models, and `valid_class_mask` to share one model between triangles and quads.
- Train: `python ml/exit_classifier.py --data_dim 2d_electron --electron_npz <file.npz> [--norm_params ml/data/norm_params_2d_electron_boundary.npy]`. Pass `--norm_params` to share the flow's normalization. Writes the checkpoint plus `exit_classifier_check_*.png` (predicted vs MC class fractions per bin, with z-scores).
- Chain: `python validation/Macro_field.py ... --exit_classifier ml/models/2d_electron_boundary/exit_classifier.pth`. The face is sampled from the classifier, and the (not face-conditioned) flow is redrawn until it agrees.
- New data dims: add a branch to `_load_problem` in `ml/exit_classifier.py` that returns an integer `exit_class` column.

### Face-Conditioned Electron Flow

`--data_dim 2d_electron_face` trains a flow on exiting particles only. Its conditioning is the `2d_electron` conditioning plus a one-hot exit face. Its target is `(s, psi, r, f_out, path)`: position along the face, exit angle from the outward normal (always outgoing), in-plane magnitude, energy fraction and path length. At inference, sample the class from the exit classifier and run this flow only for the exits.

- Train: `python ml/train.py --data_dim 2d_electron_face --electron_npz <file.npz> --run_tag _analog_cu`. `--run_tag` suffixes the model, results and norm-param paths so the defaults are not overwritten.
- Compare against held-out MC: `python validation/electron_cell_compare.py --test_npz ... --baseline_ckpt ... --face_ckpt ... --classifier ... --ref_npz <independent MC>` (full example in the script docstring).

### Chaining Electron Cell Models

- `validation/chain_core.py` holds the cell-to-cell loop (numpy only). Every cell model is a kernel `kernel(y_c, u_c, v_c, w, E, step) -> {cls, x, y, u, v, w, E_out}` in the canonical left-face frame. `validation/gmc_kernels.py` has the baseline flow, flow + classifier and classifier + face-flow kernels.
- `validation/Macro_field.py` (old flags still work): `--checkpoint` for the baseline flow, add `--exit_classifier` to sample the face from the classifier, or use `--face_ckpt --face_norm --exit_classifier` for the face-conditioned model. `--norm_params` selects non-default flow norm params.
- To compare against analog MC on the same grid:
  1. `../ElectronTransport/.venv/bin/python validation/mc_chain.py --out validation/results/chain_analog_cu/mc_ref.npz` (needs the ElectronTransport venv). This writes the source, the full-grid MC dose and the MC-chained dose (the analog MC used as the cell kernel, i.e. the best any GMC with the primary-only / local-deposit structure can do).
  2. Run it again with `--seed 456` for the MC-vs-MC noise floor.
  3. `python validation/electron_chain_compare.py --mc_ref ... --noise_ref ... --baseline_ckpt ... --face_ckpt ... --classifier ...`
- `ml/train.py --energy_encoding log_loss` (`2d_electron_face` only) trains on log10 of the energy-loss fraction instead of probit(E_out fraction). It is an option, not the default: in the analog Cu test it did not remove the high-E deposit bias, which comes from the flow under-filling the heavy tail of the loss distribution.

### Electron GMC with Secondaries (100 µm Cu, 1 keV – 10 MeV)

**Results, figures and reproduction commands: [validation/results/secondary_gmc_cu_100um.md](validation/results/secondary_gmc_cu_100um.md).**
The design, including how the relevant-secondary classifier works, is in section 7b of `docs/electron_gmc_workflow.txt`.
Other materials: Al and Al/Cu ([secondary_gmc_alcu_100um.md](validation/results/secondary_gmc_alcu_100um.md)); water at
100 µm and 1 mm, cost vs cell size and energy, and the relevant-daughter count cap
([sweep_cost_water_cap.md](validation/results/sweep_cost_water_cap.md)). Timing tools: `validation/gmc_timing.py` (GMC cost per
transit), `validation/gmc_size_timing.py` (cost vs network size and ODE steps) and `validation/chain_timing.py` (full-grid MC vs
chained GMC, end to end).

- `ml/secondary_gmc.py` is the relevance-based model set. A secondary is followed only if its subtree leaves the cell. A joint classifier samples exit face × relevant daughters × photon energy; then exit, daughter and photon flows run; the dose is the energy that is left (conserved exactly). There are separate models for electrons entering a cell and for relevant secondaries starting inside it.
- Train: `python ml/secondary_gmc.py train --train_npz <generate_analog_batches.py --simulated_only file> --out ml/models/<name>`. Plot losses: `python ml/secondary_gmc.py plot_history --model ml/models/<name>/secondary_gmc.pth --out ml/training_results/<name>/training_loss.png`.
- Validate:
  - `validation/secondary_cell_compare.py`: single cell vs held-out MC.
  - `validation/secondary_mc_refs.py`: MC references (ElectronTransport venv).
  - `validation/secondary_chain_compare.py`: chained dose maps, profiles, errors over `--n_seeds` runs.
- Headline (20×20 chain, 1–10 MeV beam): dose-map error vs full analog MC is 1.18 ± 0.10 % with secondaries and 3.76 ± 0.24 % without, against a 1.10 % MC noise floor.

### SCR-GMC: Sub-Cell Dose Reconstruction

**Results, figures and reproduction commands: [validation/results/subcell_kernel_cu_100um.md](validation/results/subcell_kernel_cu_100um.md).**
A learned kernel (`ml/subcell_kernel.py`, a conditional spline flow) places each GMC transit's dose inside its cell.
The GMC transports on coarse cells, and dose is tallied afterwards on any finer mesh. On 100 µm Cu with 3.1 µm dose
voxels, it agrees with analog MC within noise and is 4× faster at equal noise. The speedup shrinks as the dose voxels
get finer relative to the block, because the MC tallies at any resolution for free and the SCR tally does not. For
example, a 1 mm water block breaks even with MC at about 1–2 µm voxels. See section 6 of the results page.

- Chain with `chain_secondaries(..., transit_log=True)`, then tally with `subcell_kernel.scr_tally`. The defaults are
  Sobol sampling with an automatic samples-per-transit setting.
- **Charge** (electrons stopped − created) on the same fine mesh: `drumm_cylinder.chain_charge_fine` places the
  chain's own events (relevant-daughter births; stops of absorbed transits from the GMC's optional "stop" flow,
  `secondary_gmc.py train --only stop`), and `ml/charge_kernel.py` adds the electrons the relevant-only GMC never
  simulates (born and stopped inside one cell): a Poisson rate plus a birth/stop pair flow, trained on the
  generator's `q_pairs`. Without that term the charge has a saw-tooth with the cell period.
- Benchmark: 20 MeV beam on a water cylinder vs ITS (Drumm et al. 2007), with energy and charge, analog MC, GMC and
  SCR-GMC: [validation/results/drumm_water_cylinder.md](validation/results/drumm_water_cylinder.md).
- Validate:
  - `validation/scr_mc_ref.py`: fine-voxel MC reference (ElectronTransport venv).
  - `validation/scr_chain_compare.py`: chained GMC + SCR vs the fine MC; noise/bias split over seeds; several
    tallies on the same chains.
  - `validation/scr_figures.py`: the results-page figures, from a saved run.
