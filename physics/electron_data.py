# physics/electron_data.py

from pathlib import Path
import numpy as np
import pandas as pd
import logging

from ml.cell_topology import SQUARE, classes_from_status

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_electron_dataset(npz_path):
    """
    Load electron Monte Carlo data from an NPZ file and convert to GMC-compatible DataFrame.

    Expected NPZ keys:
        - inputs:          [N, 5] array [x_in, y_in, u_in, v_in, E_in]
        - outputs:         [N, 5] array [x_out, y_out, u_out, v_out, E_out]
        - path_lengths:    [N] array of path lengths
        - status:          [N] array where 1-4=exit face (1 left, 2 right, 3 bottom, 4 top), 5=absorbed,
                           6=lost (moving exactly along z in vacuum; dropped here with a warning)
        - energy_tally:    [grid_x, grid_y] energy deposited per grid cell (same for all particles)
        - grid_dims:       [2] array [n_grid_x, n_grid_y]
        - box_dims:        [2] array [width, height] of the simulation cell
        - density:         density of the material (scalar, or one entry per material for analog_stepper data)
    Optional keys: w_in / w_out (out-of-plane direction cosines), E_cutoff (stored in df.attrs["E_cutoff"]),
        electron_bank (analog_stepper: every escaping electron; columns x, y, u, v, w, E, status, parent, generation,
        path) -> per-primary columns n_sec_e (escaping secondary electrons, all generations) and E_sec_e (their
        total energy).

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
        density_arr = np.atleast_1d(np.asarray(data["density"], dtype=np.float64))
        w_in = np.asarray(data["w_in"], dtype=np.float32) if "w_in" in data.files else None
        w_out = np.asarray(data["w_out"], dtype=np.float32) if "w_out" in data.files else None
        E_cutoff_file = float(data["E_cutoff"]) if "E_cutoff" in data.files else None
        n_sec_e = E_sec_e = None
        if "electron_bank" in data.files:
            # Escaping secondaries of each primary (generation > 0), counted BEFORE any rows are dropped so the
            # parent indices still line up with the input rows.
            bank = np.asarray(data["electron_bank"], dtype=np.float64)
            sec = bank[bank[:, 8] > 0] if bank.size else bank.reshape(0, 10)
            parent = sec[:, 7].astype(np.int64)
            n_sec_e = np.bincount(parent, minlength=inputs.shape[0]).astype(np.int64)
            E_sec_e = np.bincount(parent, weights=sec[:, 5], minlength=inputs.shape[0])
    if density_arr.size > 1:
        log.warning(f"density has {density_arr.size} entries (multi-material file); using the first, "
                    f"{density_arr[0]:.4f} g/cm^3, as the cell density")
    density = float(density_arr[0])

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

    if not np.all(np.isin(status, [1, 2, 3, 4, 5, 6])):
        raise ValueError("status must contain only 1-4 (exit faces), 5 (absorbed) or 6 (lost)")
    lost = status == 6
    if np.any(lost):
        # Status 6 particles never reach a face or the cutoff, so they have no exit state to learn; drop them.
        log.warning(f"Dropping {int(lost.sum())} status-6 (lost) particles ({100 * lost.mean():.3f}%)")
        keep = ~lost
        inputs, outputs, path_lengths, status = inputs[keep], outputs[keep], path_lengths[keep], status[keep]
        w_in = None if w_in is None else w_in[keep]
        w_out = None if w_out is None else w_out[keep]
        n_sec_e = None if n_sec_e is None else n_sec_e[keep]
        E_sec_e = None if E_sec_e is None else E_sec_e[keep]

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
    # Geometry-agnostic exit class (ml/cell_topology.py): 0 bottom, 1 right, 2 top, 3 left, 4 absorbed
    exit_class = classes_from_status(status)

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
        "exit_class": exit_class,

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

    if w_in is not None:
        df["w_in"] = w_in
    if w_out is not None:
        df["w_out"] = w_out
    if E_cutoff_file is not None:
        df.attrs["E_cutoff"] = E_cutoff_file
    if n_sec_e is not None:
        df["n_sec_e"] = n_sec_e
        df["E_sec_e"] = E_sec_e

    log.info(f"✓ Loaded {len(df)} electron trajectories")
    log.info(f"  Box dimensions: {box_width:.4f} cm × {box_height:.4f} cm")
    log.info(f"  Material density: {density:.4f} g/cm³")
    log.info(f"  Grid dimensions: {grid_x} × {grid_y} cells")
    log.info(f"  Total energy in tally: {total_tally_energy:.6e} MeV")
    log.info(f"  Transmitted: {int(df['exited'].sum())} ({100 * df['exited'].mean():.1f}%)")
    log.info(f"  Absorbed: {int(df['absorbed'].sum())} ({100 * df['absorbed'].mean():.1f}%)")
    counts = np.bincount(exit_class, minlength=SQUARE.num_classes)
    log.info("  Exit classes: " + ", ".join(f"{n} {100 * c / len(df):.2f}%" for n, c in zip(SQUARE.class_names, counts)))
    log.info(f"  E_in range: [{E_in.min():.4f}, {E_in.max():.4f}] MeV")
    log.info(f"  E_out range: [{E_out.min():.4f}, {E_out.max():.4f}] MeV")
    log.info(f"  Energy deposited range: [{energy_deposited.min():.4e}, {energy_deposited.max():.4e}] MeV")
    log.info(f"  Path length range: [{path_lengths.min():.4e}, {path_lengths.max():.4e}] cm")

    return df