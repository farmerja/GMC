# Refinement Algorithm Analysis: Local vs Global

## Summary Answer

**YES**, the refinement runs **INDEPENDENTLY on each coarse cell** - there is **NO sweep through the whole domain**.

---

## Evidence from the Code

### The Main Refinement Function: `reconstruct_fine_flux_from_surface_tally`

Located in `physics/multigrid_steady_2d.py`, lines 982-1020:

```python
def reconstruct_fine_flux_from_surface_tally(
    hierarchy: MultigridHierarchy,
    tally: SurfaceTallyCollection,
    quadrature: AngularQuadrature2D,
    *,
    num_source_iterations: int = 6,
) -> Dict[str, object]:
    coarse = hierarchy.coarse
    fine = hierarchy.fine
    local_nx = hierarchy.fine_per_coarse_x
    local_ny = hierarchy.fine_per_coarse_y
    fine_flux = np.zeros((fine.num_cells_y, fine.num_cells_x), dtype=np.float64)
    
    # Loop over each coarse cell
    for iy in range(coarse.num_cells_y):
        for ix in range(coarse.num_cells_x):
            # Solve LOCAL SN problem for this coarse cell only
            solution = solve_local_sn(
                tally.angular_flux[iy, ix],  # BCs for THIS cell only
                quadrature,
                width=float(coarse.cell_widths[ix]),
                height=float(coarse.cell_heights[iy]),
                nx=local_nx,
                ny=local_ny,
                sigma_t=float(coarse.sigma_t_map[iy, ix]),
                sigma_s=float(coarse.sigma_s_map[iy, ix]),
                fixed_source_strength=fixed_source,
                num_source_iterations=num_source_iterations,
            )
            # Store results in global fine grid
            ys = iy * local_ny
            xs = ix * local_nx
            fine_flux[ys : ys + local_ny, xs : xs + local_nx] = solution.scalar_flux
```

### Key Observations:

1. **Simple nested loop** over coarse cells `(ix, iy)`
2. **Each call to `solve_local_sn`** is completely independent
3. **No information exchange** between neighboring coarse cells during refinement
4. **Boundary conditions** come from the surface tally (`tally.angular_flux[iy, ix]`)
5. **Results are assembled** into the global fine grid after each local solve

---

## The Local SN Solver: `solve_local_sn`

Located in `physics/multigrid_steady_2d.py`, lines 865-978:

```python
def solve_local_sn(
    incoming_angular_flux: np.ndarray,  # BCs: [4 faces, segments, angles]
    quadrature: AngularQuadrature2D,
    *,
    width: float,
    height: float,
    nx: int,  # Number of fine cells in x within this coarse cell
    ny: int,  # Number of fine cells in y within this coarse cell
    sigma_t: float,
    sigma_s: float,
    fixed_source_strength: float = 0.0,
    num_source_iterations: int = 6,
    positivity_clamp: bool = True,
) -> LocalSNSolution:
    """Solve a local 2D SN problem on a rectangular subgrid with diamond-difference sweeps."""
```

### What This Function Does:

1. **Creates a local grid** of size `(nx, ny)` fine cells
2. **Uses incoming angular flux** as boundary conditions on all 4 faces
3. **Performs source iteration** within this local domain only:
   - Diamond-difference sweeps for each angle
   - Scattering source updates
   - NO communication with neighboring coarse cells
4. **Returns local solution**:
   - Scalar flux on the local fine grid
   - Outgoing angular flux (not used for anything!)
   - Balance diagnostics

---

## Algorithm Structure

```
reconstruct_fine_flux_from_surface_tally():
    Initialize global fine_flux array
    
    FOR each coarse cell (iy, ix):
        
        Extract incoming BCs: tally.angular_flux[iy, ix]
        
        solution = solve_local_sn(incoming_BCs, ...):
            Create local (nx × ny) grid
            Initialize local scalar_flux = 0
            
            FOR iter in source_iterations:
                Update scatter_source from local flux
                
                FOR each angle:
                    Diamond-difference sweep through LOCAL grid
                    Use incoming_BCs at boundaries
                    Compute outgoing_flux at boundaries (discarded)
                
                Update local scalar_flux
            
            RETURN local solution
        
        Insert solution into global fine_flux array
    
    RETURN assembled fine_flux
```

---

## Key Implications

### ✅ **Advantages of This Approach:**

1. **Embarrassingly Parallel:**
   - Each coarse cell solve is completely independent
   - Could run all coarse cells in parallel with no synchronization
   - Perfect for multi-core or distributed computing

2. **Memory Efficient:**
   - Only need to store one local fine grid at a time
   - Don't need entire fine-grid angular flux in memory

3. **Simple Implementation:**
   - No complex global sweep logic
   - Easy to debug individual cells

### ⚠️ **Limitations/Issues:**

1. **No Global Coupling During Refinement:**
   - Outgoing flux from one coarse cell is NOT used as incoming flux for neighbors
   - Each cell sees only the BCs from the coarse GMC pass
   - Flux may be discontinuous at coarse cell boundaries

2. **Balance Issues at Interfaces:**
   - Outgoing current from local solve ≠ incoming current to neighbor
   - This explains the `balance_residual` diagnostic being tracked
   - Can see this in the test output: "max abs residual: 6.149412e-01"

3. **Not a True Multigrid Method:**
   - No restriction/prolongation operators
   - No coarse-grid correction
   - More like a "domain decomposition with fixed boundary conditions"

### 🔍 **What Gets Thrown Away:**

The outgoing angular flux from each local solve is computed but **never used**:

```python
outgoing = np.zeros_like(incoming_angular_flux)  # Computed in solve_local_sn
# ... sweeps update outgoing ...
return LocalSNSolution(
    scalar_flux=scalar_flux,
    outgoing_angular_flux=outgoing,  # Returned but never used!
    ...
)
```

This could potentially be used for:
- Interface corrections
- Iterative domain decomposition
- Multi-level refinement

---

## Comparison to Alternative Approaches

### What This Code Does (Local Solves):
```
Coarse GMC → Surface Tallies → Independent Local S_n → Assemble Fine Flux
```

### Alternative: Global Fine-Grid S_n:
```
Coarse GMC → Surface Tallies → Single Global S_n on Fine Grid
```
- Would require sweeping entire 56×56 grid
- Better coupling at coarse boundaries
- Much more memory intensive

### Alternative: Iterative Domain Decomposition:
```
Coarse GMC → Surface Tallies → Iterate {
    Solve each coarse cell with current BCs
    Update interface BCs from outgoing flux
} until convergence
```
- Would improve interface consistency
- More expensive (requires iteration)

---

## Verification

For a **14×14 coarse grid**, there are **196 independent local solves**.
Each solve works on a **4×4 fine subgrid** (for refinement 2→8).

For a **56×56 fine grid** (refinement 2→8):
- Still 196 independent solves
- Each on an **8×8 fine subgrid**

---

## Conclusion

✅ **Confirmed:** The refinement step consists of **independent local S_n solves** on each coarse cell.

**Characteristics:**
- No global sweep through the domain
- No coupling between coarse cells during refinement
- Perfectly parallelizable
- Simple but may have interface discontinuities

This is a **"one-shot" local refinement** approach, not an iterative multigrid method. The quality depends entirely on the accuracy of the surface tallies from the coarse GMC pass.
