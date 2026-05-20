# Anisotropic Scattering Implementation Summary

## Overview
The `johannes_exploration` branch implements **anisotropic scattering** for 2D steady-state neutron transport, extending the original isotropic GMC implementation. This is a proof-of-concept demonstrating how directional scattering bias (forward-peaked) affects both Monte Carlo physics and ML-based generative transport.

## Key Concepts

### 3D Direction in 2D Transport
- **Physical transport is 3D**: Particles carry full 3D direction vectors `(u, v, w)` where `u² + v² + w² = 1`
- **Geometry is 2D**: Only `(u, v)` components are used for cell boundaries and spatial transport
- **Projection relationship**: `u² + v² = 1 - w²` (perpendicular component to z-axis)
- **Why this matters**: The 2D problem represents an infinite extrusion along the z-axis with full 3D angular physics

### Scattering Models

#### Isotropic Scattering (baseline)
- **Physical model**: After scattering, new direction is sampled uniformly on the unit sphere
- **Sampling**: `mu_z ~ Uniform[-1,1]`, `phi ~ Uniform[0,2π]`
- **Formula**: 
  ```python
  mu_z = 2*unit_mu_z - 1
  t = sqrt(1 - mu_z²)
  u = t * cos(phi)
  v = t * sin(phi)
  w = mu_z
  ```

#### Anisotropic Scattering (new)
- **Physical model**: Forward-peaked scattering aligned with incident direction
- **Scattering distribution**: `mu ~ 2*sqrt(unit_x) - 1` (biased toward `mu=1`, i.e., forward direction)
- **Implementation**: Uses rotation matrices to scatter relative to incident direction
  1. Construct orthonormal basis `{d, e1, e2}` where `d` is incident direction
  2. Sample polar angle `mu` from forward-peaked distribution
  3. Sample azimuthal angle `phi ~ Uniform[0,2π]`
  4. Rotate scattered direction into global frame
- **Key difference**: Scattered direction correlated with incident direction (forward bias)

## Implementation Details

### 1. **Scattering Mode Infrastructure** (`physics/steady_2d_scattering.py`)

**Constants:**
```python
STEADY_2D_SCATTERING_ISOTROPIC = "isotropic"
STEADY_2D_SCATTERING_ANISOTROPIC = "anisotropic"
STEADY_2D_SCATTERING_TAG = "steady2d-v2"
STEADY_2D_ANISOTROPIC_ASSET_ROOT = "ml/steady_2d_anisotropic"
```

**Core Functions:**
- `normalize_steady_2d_scattering_mode(mode)` - Validates and normalizes mode string
- `steady_2d_scattering_mode_code(mode)` - Converts to integer: 0=isotropic, 1=anisotropic
- `steady_2d_scattering_asset_tag(mode)` - Returns file suffix for model/data paths
- `sample_scattered_direction_3d_from_unit(...)` - **JIT-compiled scattering kernel**

**Asset Path Resolution:**
- **Isotropic**: `ml/models/2d_boundary/`, `ml/data/norm_params_2d_boundary.npz`
- **Anisotropic**: `ml/steady_2d_anisotropic/models/2d_boundary/`, `ml/steady_2d_anisotropic/data/norm_params_2d_boundary.npz`

### 2. **Monte Carlo Integration** (`physics/mc2d.py`)

**Function Signatures Updated:**
```python
def multi_cell_mc_internal_source_2d_njit(
    ...,
    scattering_mode_code: int = 0,  # 0=isotropic, 1=anisotropic
    ...
)
```

**Scattering Call Site:**
```python
if scatter_event:
    u_new, v_new, w_new = sample_scattered_direction_3d_from_unit(
        u, v, w,                     # Current direction
        rng.random(),                # unit_mu (polar angle)
        rng.random(),                # unit_phi (azimuthal angle)
        scattering_mode_code,        # 0 or 1
        eps,
    )
```

**Wrapper Functions:**
- `run_mc_simulation_2d_njit_wrapper(...)` - Added `scattering_mode="isotropic"` parameter
- Converts string mode to integer code via `steady_2d_scattering_mode_code()`

### 3. **GMC Feature Encoding** (`physics/multigrid_steady_2d.py`)

**Direction Encoding: Polar Coordinates**

For anisotropic scattering, direction `(u, v)` is encoded as **polar coordinates** `(phi, r)`:
- `phi = atan2(v, u)` - azimuthal angle ∈ [0, 2π]
- `r = sqrt(u² + v²)` - projected radius ∈ [0, 1]

**Why polar?** Anisotropic scattering depends on the **magnitude** `r` (related to `w` component) and **angle** `phi`:
- High `r` (low `|w|`) → particle traveling mostly in x-y plane
- Low `r` (high `|w|`) → particle traveling mostly along z-axis
- Forward scattering preserves both magnitude and relative azimuth

**Boundary Source Conditioning** (5D):
```python
[
    normalize_width(W * sigma_s),       # Cell width × scattering cross-section
    normalize_height(H * sigma_s),      # Cell height × scattering cross-section
    normalize_position_relative(y_n),   # Position along face [0,1]
    normalize_angle_phi(phi_in),        # Incident azimuthal angle [0,2π] → [-1,1]
    normalize_radius_r(r_in),           # Projected radius [0,1] → [0,1]
]
```

**Internal Source Conditioning** (6D):
```python
[
    normalize_width(W * sigma_s),
    normalize_height(H * sigma_s),
    normalize_position_relative(x_rel), # x position in cell
    normalize_position_relative(y_rel), # y position in cell
    normalize_angle_phi(phi_in),
    normalize_radius_r(r_in),
]
```

**Isotropic Encoding (for comparison):**
- Direction encoded as `(u, v)` directly (Cartesian)
- Boundary: 5D = [W*σ_s, H*σ_s, y_n, u_in, v_in]
- Internal: 6D = [W*σ_s, H*σ_s, x_rel, y_rel, u, v]

### 4. **Model Loading** (`physics/multigrid_steady_2d.py`)

```python
def load_steady_gmc_models(
    device: Optional[str] = None,
    *,
    scattering_mode: str = "isotropic",
    train_sampling: str = "random",
) -> SteadyModelBundle:
```

**Returns:**
```python
SteadyModelBundle(
    device="cuda",
    scattering_mode="anisotropic",  # Stored for validation
    boundary_solver=ODESolver(...),
    internal_solver=ODESolver(...),
    boundary_norm={...},
    internal_norm={...},
    boundary_cond_dim=5,  # Same for both modes
    internal_cond_dim=6,  # Same for both modes
    boundary_target_dim=5,  # Unchanged (x, y, energy, u_out, v_out)
    internal_target_dim=5,  # Unchanged
)
```

**Validation:**
- `_assert_steady_2d_norm_mode()` ensures loaded normalization matches requested scattering mode
- Checks `norm_params["steady_2d_scattering_mode"]` and `norm_params["steady_2d_scattering_tag"]`

### 5. **S_n Refinement Compatibility**

**Current Status:** **NOT COMPATIBLE** with anisotropic scattering

**Reason:** The S_n (discrete ordinates) solver in `reconstruct_fine_flux_with_block_sn()` assumes:
1. **Isotropic scattering**: Scattering source is angle-independent
2. **Surface tally structure**: Only records `(phi, r)` at surfaces (no incident `r` stored)
3. **Angular discretization**: Uses uniform azimuthal angles, incompatible with forward-peaked distributions

**Code Handling:**
```python
if anisotropic_mode:
    print("Block-SN refinement: skipped (current steady 2D SN assumes isotropic scattering and surface tally omits projected radius r)")
    refined_gmc_block = None
    refined_mc_block = None
```

**Future Work:** Would require:
- Anisotropic scattering kernel in S_n solver
- Surface tally recording incident `r` (perpendicular component) for boundary conditions
- Gauss-Legendre angular quadrature compatible with forward scattering

## File Structure

```
ml/steady_2d_anisotropic/
├── models/
│   ├── 2d_boundary/
│   │   ├── best_model.pth
│   │   └── trained_model_final.pth
│   └── 2d_internal/
│       ├── best_model.pth
│       └── trained_model_final.pth
├── data/
│   ├── norm_params_2d_boundary.npz
│   └── norm_params_2d_internal.npz
└── training_results/
    ├── 2d_boundary/
    │   ├── loss_history.csv
    │   └── training_loss.png
    └── 2d_internal/
        ├── loss_history.csv
        └── training_loss.png
```

## Validation Script

**`validation/complete_comparison.py`**

**CLI:**
```bash
python validation/complete_comparison.py \
    --scattering-mode anisotropic \
    --coarse-particles 50000 \
    --fine-mc-particles 50000 \
    --device cuda
```

**What it does:**
1. **Coarse GMC (anisotropic)**: Runs GMC on 14×14 grid with anisotropic scattering models
2. **Coarse MC (anisotropic)**: Runs pure MC on 14×14 grid with anisotropic physics
3. **Block S_n refinement**: **Skipped** for anisotropic mode (incompatible)
4. **Fine MC (anisotropic)**: Reference solution on 56×56 grid with anisotropic physics
5. **Diagnostics**: 
   - Compares GMC vs MC surface tallies
   - Compares GMC vs MC scalar flux on coarse grid
   - Timing comparisons

**Output:**
```
Complete Method Comparison:
  Coarse grid: 14x14
  Fine grid:   56x56
  Scattering mode:  anisotropic
  Block-SN refinement: skipped (current steady 2D SN assumes isotropic scattering and surface tally omits projected radius r)

[1. Coarse-surface diagnostics: GMC vs MC]
  surface_gmc_vs_mc_active_bins:
    l1          : ...
    rel_l2      : ...
  GMC incoming current sum : ...
  MC incoming current sum  : ...

[5. Coarse scalar flux: GMC vs MC]
  l1          : ...
  rel_l2      : ...
```

## Key Insights

### Physics
1. **3D transport in 2D geometry**: The `w` component affects transport through `r = sqrt(u² + v²)`
2. **Forward scattering increases flux penetration**: Particles scatter less frequently at large angles, leading to longer effective path lengths
3. **Material interface behavior**: Anisotropic scattering affects how particles cross boundaries (more forward momentum preserved)

### ML Implications
1. **Polar encoding is essential**: Separating magnitude `r` and angle `phi` allows the model to learn correlation between incident and scattered directions
2. **Conditioning dimension unchanged**: Still 5D (boundary) and 6D (internal), just different encoding
3. **Model architecture unchanged**: Same ConditionalFlow with hidden_dim=160, depth=5
4. **Training data requirements**: Needs separate training data generated with anisotropic MC

### Computational
1. **GMC performance**: Similar speed to isotropic (same architecture, just different input encoding)
2. **MC performance**: Slightly slower due to rotation matrix calculations in scattering kernel
3. **S_n refinement**: **Not available** for anisotropic mode - requires fundamental algorithm changes

## Limitations & Future Work

### Current Limitations
1. **No S_n refinement**: Cannot use coarse-to-fine multigrid acceleration with anisotropic scattering
2. **Single anisotropic model**: Only implements `mu ~ 2*sqrt(x) - 1`, not general phase functions
3. **No time-dependence**: Only steady-state 2D implemented
4. **Polar singularity**: Encoding `(phi, r)` has undefined `phi` when `r→0` (particle moving along z-axis)

### Future Directions
1. **Anisotropic S_n solver**: Implement discrete ordinates with anisotropic scattering kernel
2. **General phase functions**: Support arbitrary `P_n` expansions or tabulated phase functions
3. **Hierarchical models**: Distillation from teacher models (see Marginalization branch)
4. **3D geometry**: Extend to true 3D transport (not just 2D projection)
5. **Adaptive quadrature**: Use importance sampling for angular discretization based on anisotropy

## Testing

**Recommended Tests:**
```bash
# 1. Isotropic baseline
python validation/complete_comparison.py \
    --scattering-mode isotropic \
    --coarse-particles 50000

# 2. Anisotropic comparison
python validation/complete_comparison.py \
    --scattering-mode anisotropic \
    --coarse-particles 50000

# 3. Compare surface tallies
# Look at validation/results/complete_comparison/surface_current_*.png

# 4. Verify models loaded correctly
# Check console output for "Loading model for 2d boundary source"
```

## Conclusion

The anisotropic implementation demonstrates:
- ✅ **Successful integration** of directional scattering bias into GMC framework
- ✅ **Polar coordinate encoding** for direction representation
- ✅ **Separate model assets** isolated in `ml/steady_2d_anisotropic/`
- ✅ **Validation framework** with GMC vs MC comparison
- ⚠️ **S_n refinement incompatible** - fundamental algorithmic limitation
- 🔬 **Proof-of-concept** ready for extension to more complex physics

This work provides the foundation for incorporating realistic scattering kernels (Henyey-Greenstein, Rayleigh, etc.) into ML-accelerated transport solvers.
