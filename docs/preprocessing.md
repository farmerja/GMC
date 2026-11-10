# Pre‑Processing & Normalisation Logic

This document gives a high‑level overview of the data loading and normalisation pipelines used in the **GMC** project. The two key files involved are:

| File | Purpose |
|------|---------|
| `ml/transform.py` | Implements the :class:`SimDataset` that converts raw simulation tables into a tensor‑friendly format for training PyTorch models.
| `physics/electron_data.py` | Provides ``load_electron_dataset`` – a helper to ingest Monte‑Carlo electron transport data from an NPZ archive, validate it and expose a Pandas DataFrame ready for use with :class:`SimDataset`.

## 1. Normalisation Utilities (in `ml/transform.py`)

The project normalises all scalar quantities to a **log‑scale** or **linear‑scale** representation that lies in the interval ``[0, 1]``.

| Function | Normalisation type | Description |
|----------|--------------------|-------------|
| `normalize_log_scale_logit(value, lower, upper)` | Log scale (log10) | Maps a real value in `[lower, upper]` to ``[0, 1]`` using the log‑ratio of the bounds.
| `denorm_log_scale_logit(z_value, lower, upper)` | Inverse | Recovers the original value from its log‑scaled representation.
| `normalize_linear_scale_logit(value_01)` | Linear scale (logit) | Transforms a number in ``[0, 1]`` to an unbounded space via the logistic function; useful for continuous targets that should remain strictly positive/negative.
| `denorm_linear_scale_logit(logit_value)` | Inverse of above |

The project keeps backward‑compatibility wrappers:
```python
normalize_log_scale_logit   = normalize_log_scale_probit
denorm_log_scale_logit      = denorm_log_scale_probit
normalize_linear_scale_logit= normalize_linear_scale_probit
```
so older scripts that relied on the *probit* naming continue to run.

Additional specialised helpers:
| Helper | Purpose |
|--------|---------|
| `normalize_thickness`, `normalize_width`, `normalize_height`, `normalize_depth` | Thin‑film / box dimension normalisation (log scale). |
| `normalize_position_relative(pos)` | Normalises a 1‑D position that is already in the range ``[0, 1]``. |
| `normalize_angle_phi(phi)` / `normalize_radius_r(r)` | Convert polar coordinates to linear‑scaled logits on `[0, 1]`. |
| `denorm_angle_phi(logit_phi)` | Inverse of above |
| `uv_to_polar(u, v)`, `polar_to_uv(phi, r)` | Utility for converting between Cartesian and polar representations. |
| `uvw_to_axis_spherical_x(u, v, w)` | Convert 3‑D direction vectors to axis‑spherical coordinates (phi, mu). |

All of these functions are pure NumPy or Torch operations – they accept scalars, ``np.ndarray`` and ``torch.Tensor`` inputs seamlessly.

## 2. `SimDataset` class (`ml/transform.py`)

The dataset is the heart of the training pipeline.

### Construction
```python
dataset = SimDataset(
    sim_data=df,          # DataFrame from :func:`load_electron_dataset`
    norm_params=norms,    # Normalisation dictionary returned by ``load_norm_params``
    data_type="2d_boundary"  # One of the supported modalities
)
```

* **`sim_data`** – The raw simulation table. It must contain all columns referenced in `transform.py`, such as ``x_in``, ``y_in``, ``u_in``, ``v_in`` and their normalised versions.
* **`norm_params`** – A dictionary mapping field names to either linear bounds or a pre‑computed centre/scale for robust log scaling. The helper :func:`load_norm_params_for_data_dim` retrieves this from disk.
* **`data_type`** – Determines which subset of columns is used and how many conditioning / target features exist. Supported values include:
  - ``1d_boundary`` / ``1d_internal``
  - ``2d_boundary`` / ``2d_internal``
  - ``3d_boundary`` / ``3d_internal``
  - ``*_timedep_*`` variants for time‑dependent data
  - ``2d_electron_boundary`` – a special case for electron transport data.

The constructor calculates `cond_dim` and `target_dim` automatically based on the chosen type.

### Pre‑computation (`_precompute_tensors`)
To avoid per‑sample Python overhead during training, all normalisation is performed once and cached in ``self._cached_x1``. The method works as follows:

1. **Extract columns** – Uses the helper `col(name)` to fetch a NumPy column from the DataFrame.
2. **Build tensors** – Depending on `data_type`, it constructs two lists of columns: ``cond_cols`` and ``target_cols``. These are normalised using the utilities described above.
3. **Stack** – `stack_ct` concatenates conditioning and target tensors into a single shape `(N, cond_dim + target_dim)`.
4. **Validate dimensions** – Ensures that the resulting tensor’s second dimension matches ``self.input_dim`` (computed during initialization).

Specialised handling is present for *time‑dependent* data where an additional budget or length feature is encoded using either a log‑scaled value (`use_L_over_budget`) or the raw numeric field.

### Item Retrieval
When ``__getitem__(idx)`` is called:

```python
x1 = self._cached_x1[idx]
return x1, x1[:self.cond_dim], x1[self.cond_dim:]
```

The returned tuple consists of the full tensor, a conditioning slice and a target slice – exactly what the training loop expects.

## 3. Electron Data Loader (`physics/electron_data.py`)

This function reads an NPZ file with a very specific structure, validates the data shape, and augments it with normalised columns that the dataset can consume.

### Input format
The expected keys in the NPZ archive are documented in the docstring:

| Key | Shape | Description |
|-----|-------|-------------|
| `inputs` | ``[N,5]`` | Columns: ``x_in, y_in, u_in, v_in, E_in`` |
| `outputs` | ``[N,5]`` | Same order as inputs but for the exit state |
| `path_lengths` | ``[N]`` | Total track length of each particle |
| `status` | ``[N]`` | 1‑4 = exit face; 5 = absorbed |
| `energy_tally` | ``[grid_x, grid_y]`` | Spatial energy deposition map |
| `grid_dims` | ``[2]`` | Number of cells in X and Y |
| `box_dims` | ``[2]`` | Width & height of the simulation cell |
| `density` | scalar | Material density |

### Normalisation Steps
* **Position** – Normalised to `[0, 1]` by dividing by the box dimensions.
* **Energy deposition** – For each particle we compute ``energy_deposited = E_in - E_out`` (clipped at zero). A log‑scaled check is performed for absorbed particles.
* **Meta columns** – Adds additional fields such as `absorbed`, `exited`, and the normalized positions (`x_*_norm`, `y_*_norm`). These are the columns expected by :class:`SimDataset` for electron data.

### Logging
After loading, the function emits a detailed summary of the dataset: box size, density, grid dimensions, energy ranges, absorption/transmission statistics, etc. This is useful during debugging and model inspection.

### Return value
The function returns a **Pandas DataFrame** with all raw and normalised columns ready to be passed to ``SimDataset``.

---

## Common Pitfalls

1. **Missing Normalisation Parameters** – If `norm_params` is missing a key required for the chosen `data_type`, an exception will be raised during `_precompute_tensors`. Ensure that you load the correct normalisation file using :func:`load_norm_params_for_data_dim`.
2. **Incorrect NPZ Structure** – The electron loader expects exact column counts; otherwise it will raise a ValueError. Verify your simulation export matches the specification.
3. **Time‑Dependent Data** – For data types ending in ``_timedep_*`` you must provide the `use_L_over_budget` flag in `norm_params`. Otherwise the loader falls back to a simple log‑scaled representation of ``L``.

---

## Extending the Pipeline
If you need additional preprocessing steps:
1. **Add a new normalisation helper** in the top section of `ml/transform.py` and expose it in the dataset’s `_precompute_tensors` logic.
2. **Update the electron loader** if your NPZ format changes; simply adjust the key names or add new derived columns before returning the DataFrame.

Happy modeling!