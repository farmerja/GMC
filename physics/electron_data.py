# physics/electron_data.py

from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_electron_dataset(npz_path):
    """
    Load electron Monte Carlo data from an NPZ file and convert to GMC-compatible DataFrame.

    Expected NPZ keys:
        - inputs:          [N, 5] array [x_in, y_in, u_in, v_in, E_in]
        - outputs:         [N, 5] array [x_out, y_out, u_out, v_out, E_out]
        - path_lengths:    [N] array of path lengths
        - status:          [N] array where 1-4=exit face, 5=absorbed
        - energy_tally:    [grid_x, grid_y] energy deposited per grid cell (same for all particles)
        - grid_dims:       [2] array [n_grid_x, n_grid_y]
        - box_dims:        [2] array [width, height] of the simulation cell
        - density:         scalar density of the material

    Returns:
        pd.DataFrame with normalized and processed data ready for GMC training
    """
    npz_path = Path(npz_path)
    log.info(f"Loading electron dataset from {npz_path}")

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"], dtype=np.float32)
        outputs = np.asarray(data["outputs"], dtype=np.float32)
        path_lengths = np.asarray(data["path_lengths"], dtype=np.float32)
        status = np.asarray(data["status"], dtype=np.int32)
        energy_tally = np.asarray(data["energy_tally"], dtype=np.float32)  # [grid_x, grid_y]
        grid_dims = np.asarray(data["grid_dims"], dtype=np.int32)
        box_dims = np.asarray(data["box_dims"], dtype=np.float32)
        density = float(data["density"])

    # Validate shapes
    if inputs.ndim != 2 or inputs.shape[1] != 5:
        raise ValueError(f"Expected inputs with shape [N, 5], got {inputs.shape}")

    if outputs.ndim != 2 or outputs.shape[1] != 5:
        raise ValueError(f"Expected outputs with shape [N, 5], got {outputs.shape}")

    if path_lengths.ndim != 1:
        raise ValueError(f"Expected path_lengths with shape [N], got {path_lengths.shape}")

    if status.ndim != 1:
        raise ValueError(f"Expected status with shape [N], got {status.shape}")

    n = inputs.shape[0]
    if outputs.shape[0] != n or path_lengths.shape[0] != n or status.shape[0] != n:
        raise ValueError("inputs, outputs, path_lengths, and status must have same N")

    # Energy tally is [grid_x, grid_y], same for all particles
    if energy_tally.ndim != 2:
        raise ValueError(f"Expected energy_tally with shape [grid_x, grid_y], got {energy_tally.shape}")

    if grid_dims.ndim != 1 or grid_dims.shape[0] != 2:
        raise ValueError(f"Expected grid_dims with shape [2], got {grid_dims.shape}")

    if box_dims.ndim != 1 or box_dims.shape[0] != 2:
        raise ValueError(f"Expected box_dims with shape [2], got {box_dims.shape}")

    if not np.all(np.isin(status, [1, 2, 3, 4, 5])):
        raise ValueError("status must contain only 1-4 (exit faces) or 5 (absorbed)")

    # Parse inputs
    x_in = inputs[:, 0]
    y_in = inputs[:, 1]
    u_in = inputs[:, 2]
    v_in = inputs[:, 3]
    E_in = inputs[:, 4]

    # Parse outputs
    x_out = outputs[:, 0]
    y_out = outputs[:, 1]
    u_out = outputs[:, 2]
    v_out = outputs[:, 3]
    E_out = outputs[:, 4]

    # Parse metadata
    box_width, box_height = float(box_dims[0]), float(box_dims[1])
    grid_x, grid_y = int(grid_dims[0]), int(grid_dims[1])

    # Normalize positions to [0, 1]
    eps = 1e-12
    x_in_norm = np.clip(x_in / (box_width + eps), 0.0, 1.0)
    y_in_norm = np.clip(y_in / (box_height + eps), 0.0, 1.0)
    x_out_norm = np.clip(x_out / (box_width + eps), 0.0, 1.0)
    y_out_norm = np.clip(y_out / (box_height + eps), 0.0, 1.0)

    # Parse status
    absorbed = (status == 5).astype(np.float32)
    exited = (status != 5).astype(np.float32)
    exit_face = status.astype(np.float32)  # 1-4: face, 5: absorbed

    # Calculate total energy deposited from tally (sum over all grid cells)
    # This is a [grid_x, grid_y] array, so sum to get scalar per particle?
    # Actually, the energy_tally is the same for all particles (it's spatial tally data)
    # We need to compute per-particle energy deposited from E_in - E_out
    energy_deposited = np.clip(E_in - E_out, 0.0, None).astype(np.float32)

    # Also calculate total energy in tally for reference
    total_tally_energy = float(np.sum(energy_tally))

    # Validate energy conservation (approximately)
    # For absorbed particles: E_in ≈ E_out + energy_deposited
    # For transmitted particles: E_in ≈ E_out
    mask_absorbed = (absorbed == 1)
    if np.any(mask_absorbed):
        energy_lost_absorbed = E_in[mask_absorbed] - E_out[mask_absorbed] - energy_deposited[mask_absorbed]
        max_loss_error = np.max(np.abs(energy_lost_absorbed))
        if max_loss_error > 1e-4:
            log.warning(
                f"Energy conservation warning (absorbed): max error = {max_loss_error:.4e} MeV. "
                "Check if E_out + energy_deposited ≈ E_in."
            )

    df = pd.DataFrame({
        # Inputs (conditioning)
        "x_in": x_in,
        "y_in": y_in,
        "u_in": u_in,
        "v_in": v_in,
        "E_in": E_in,

        # Outputs (targets)
        "x_out": x_out,
        "y_out": y_out,
        "u_out": u_out,
        "v_out": v_out,
        "E_out": E_out,
        "path_length": path_lengths,
        "energy_deposited": energy_deposited,
        "absorbed": absorbed,
        "exited": exited,
        "exit_face": exit_face,

        # Normalized positions (for dataset transform)
        "x_in_norm": x_in_norm,
        "y_in_norm": y_in_norm,
        "x_out_norm": x_out_norm,
        "y_out_norm": y_out_norm,

        # Metadata
        "density": density,
        "grid_dims_x": grid_x,
        "grid_dims_y": grid_y,
        "box_width": box_width,
        "box_height": box_height,
    })

    log.info(f"✓ Loaded {len(df)} electron trajectories")
    log.info(f"  Box dimensions: {box_width:.4f} cm × {box_height:.4f} cm")
    log.info(f"  Material density: {density:.4f} g/cm³")
    log.info(f"  Grid dimensions: {grid_x} × {grid_y} cells")
    log.info(f"  Total energy in tally: {total_tally_energy:.6e} MeV")
    log.info(f"  Transmitted: {int(df['exited'].sum())} ({100 * df['exited'].mean():.1f}%)")
    log.info(f"  Absorbed: {int(df['absorbed'].sum())} ({100 * df['absorbed'].mean():.1f}%)")
    log.info(f"  E_in range: [{E_in.min():.4f}, {E_in.max():.4f}] MeV")
    log.info(f"  E_out range: [{E_out.min():.4f}, {E_out.max():.4f}] MeV")
    log.info(f"  Energy deposited range: [{energy_deposited.min():.4e}, {energy_deposited.max():.4e}] MeV")
    log.info(f"  Path length range: [{path_lengths.min():.4e}, {path_lengths.max():.4e}] cm")

    return df