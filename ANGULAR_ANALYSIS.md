# Angular Treatment Analysis: 2D vs 3D with Symmetry

## Summary Answer

**YES**, this 2D problem represents a **3D problem with translation symmetry** along the third (z) axis, similar to an infinitely-extruded or slab geometry problem.

---

## Evidence from the Code

### 1. **Direction Sampling is 3D Isotropic Projected to 2D Plane**

In `physics/mc2d.py`, the direction sampling function is revealing:

```python
@njit(inline='always')
def _sample_isotropic_dir_uv_from_unit(unit_mu_z: float, unit_phi: float, eps: float):
    mu_z = 2.0 * unit_mu_z - 1.0        # Sample polar angle cosine: [-1, 1]
    phi = 2.0 * np.pi * unit_phi         # Sample azimuthal angle: [0, 2π]
    t = np.sqrt(max(0.0, 1.0 - mu_z * mu_z))  # Perpendicular component
    u = t * np.cos(phi)                  # x-component of direction
    v = t * np.sin(phi)                  # y-component of direction
    return u, v
```

**Key observation:** The function is named `_sample_isotropic_dir_uv_from_unit` and explicitly:
- Samples `mu_z` (the z-direction cosine) from the full range [-1, 1]
- Calculates the perpendicular component `t = sqrt(1 - mu_z²)`
- Projects this onto the x-y plane using azimuthal angle `phi`

**This is NOT a 2D angular distribution!** It's a 3D isotropic distribution where:
- The particle has a velocity component in all three dimensions (u, v, w)
- Only the x-y components (u, v) are tracked because the z-component doesn't affect transport in a translationally-invariant geometry
- The z-component (w = mu_z) exists implicitly but doesn't affect cell boundaries or path lengths

### 2. **Scattering is 3D Isotropic**

When a particle scatters (lines 184-187 of `mc2d.py`):

```python
mu = np.random.uniform(-1.0, 1.0)      # Full 3D polar angle cosine
phi = np.random.uniform(0.0, 2.0 * np.pi)  # Full 3D azimuthal angle
t = np.sqrt(1.0 - mu * mu)
u = t * np.cos(phi)
v = t * np.sin(phi)
```

This samples a **3D isotropic** scattering direction, not 2D isotropic scattering.

### 3. **Angular Quadrature Spans Full Azimuthal Circle**

In `multigrid_steady_2d.py`, the angular quadrature:

```python
theta_edges = np.linspace(0.0, 2.0 * np.pi, num_angles + 1)  # [0, 2π]
mu_x = np.cos(theta_centers)
mu_y = np.sin(theta_centers)
```

This creates directions uniformly distributed around the full azimuthal circle in the x-y plane. These represent the **projections** of 3D directions onto the x-y plane.

---

## Physical Interpretation

### What This Represents:

This is a **2D transport problem with 3D angular distribution**, equivalent to:

1. **Infinite Slab/Extrusion Geometry:**
   - The geometry extends infinitely in the z-direction
   - All spatial variations occur only in the x-y plane
   - Translation symmetry in z: ∂/∂z = 0

2. **3D Particle Motion:**
   - Particles move in full 3D space with velocities (u, v, w)
   - The w-component exists but doesn't affect spatial transport
   - Path lengths in the x-y plane are affected by the w-component through:
     - Effective 2D speed: √(u² + v²) < 1 when |w| > 0
     - Particles with large |w| travel "more slowly" through the x-y plane

3. **Physical Angular Variables:**
   - θ (theta): azimuthal angle in x-y plane ∈ [0, 2π]
   - μ_z = cos(polar angle): z-direction cosine ∈ [-1, 1]
   - Related by: u² + v² = 1 - μ_z²

---

## Comparison to True 2D Transport

### If this were TRUE 2D (no z-dimension):

1. **Direction sampling would be:**
   ```python
   theta = np.random.uniform(0.0, 2.0 * np.pi)
   u = np.cos(theta)
   v = np.sin(theta)
   # u² + v² = 1 always (unit vector in plane)
   ```

2. **No mu_z variable** would exist

3. **Scattering would preserve** u² + v² = 1

### Actual implementation (3D with z-symmetry):

1. **Direction sampling:**
   - First samples mu_z from [-1, 1]
   - Then projects perpendicular component onto x-y plane
   - Result: u² + v² = 1 - mu_z² < 1 (in general)

2. **Scattering:** Full 3D isotropic

3. **Physical meaning:** Particles can have velocity components pointing out of the x-y plane

---

## Implications for S_n Discretization

### Current Implementation:

The 16 uniform angles represent **azimuthal directions** in the x-y plane:
- θᵢ = 2πi/16, i = 0, 1, ..., 15
- Each direction (μₓ, μᵧ) = (cos θᵢ, sin θᵢ)

This is treating the **projection** of the 3D angular distribution onto the x-y plane.

### What's Missing:

For a full 3D → 2D reduction, you would typically **integrate over the polar angle** (μ_z):

```
ψ_2D(x, y, θ) = ∫₋₁¹ ψ_3D(x, y, z, μ_z, θ) dμ_z
```

However, with translation symmetry (∂/∂z = 0), the z-coordinate drops out, and you're left with an **effective 2D transport equation** with a 2D angular variable (azimuthal only).

### Standard S_n in 2D with 3D Symmetry:

Traditional 2D S_n methods for slab geometry would:
1. Use discrete azimuthal angles θᵢ
2. Use discrete polar angles μⱼ (cosines of angle from z-axis)
3. Quadrature points: (θᵢ, μⱼ) pairs

This code simplifies by treating the **azimuthal distribution only**, which is appropriate for problems where the polar angle distribution is handled implicitly (like in Monte Carlo sampling).

---

## Conclusion

✅ **YES**, this is a **3D problem with translation symmetry** along the z-axis (infinite extrusion).

**Key Features:**
- Particles move in 3D space but geometry is 2D
- Direction sampling is 3D isotropic projected to x-y plane
- Scattering is 3D isotropic
- The "2D" refers to spatial dimensions, not angular dimensions
- This is equivalent to slab geometry with azimuthal symmetry

**S_n Level:**
- 16 azimuthal angles by default
- No explicit polar angle discretization (handled via integration/projection)
- Not a standard S_n level, but closest to "S₁₆" in azimuthal resolution

This is a common approach in reactor physics and radiation transport for problems with one dimension of geometric symmetry.
