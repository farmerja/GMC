"""Prototype 2D steady-state multigrid refinement workflow for lattice transport.

The prototype keeps the decomposition intentionally small:

1. A coarse-grid analog Monte Carlo transport stage estimates per-cell track
   length and bins coarse-cell surface crossings into face/segment/ordinate
   current tallies.
2. Each coarse cell is refined independently with a local discrete-ordinates
   source-iteration solve on the embedded fine subgrid.
3. The local refined solutions are renormalized to the coarse-cell track-length
   target so the assembled fine-grid scalar flux remains conservation-aware.

The implementation is designed for validation and integration work rather than
for production performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Sequence, Tuple

import numpy as np


FACE_LEFT = 0
FACE_RIGHT = 1
FACE_TOP = 2
FACE_BOTTOM = 3
OPPOSITE_FACE = np.array([FACE_RIGHT, FACE_LEFT, FACE_BOTTOM, FACE_TOP], dtype=np.int64)
OUTWARD_NORMALS = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=np.float64,
)
INWARD_NORMALS = -OUTWARD_NORMALS


@dataclass(frozen=True)
class RectangularMesh2D:
    """Uniform Cartesian mesh metadata."""

    widths: np.ndarray
    heights: np.ndarray
    boundaries_x: np.ndarray
    boundaries_y: np.ndarray
    total_width: float
    total_height: float
    areas: np.ndarray

    @classmethod
    def uniform(cls, num_cells_x: int, num_cells_y: int, total_width: float, total_height: float) -> "RectangularMesh2D":
        widths = np.full(num_cells_x, total_width / float(num_cells_x), dtype=np.float64)
        heights = np.full(num_cells_y, total_height / float(num_cells_y), dtype=np.float64)
        boundaries_x = np.concatenate(([0.0], np.cumsum(widths)))
        boundaries_y = np.concatenate(([0.0], np.cumsum(heights)))
        areas = np.outer(heights, widths)
        return cls(
            widths=widths,
            heights=heights,
            boundaries_x=boundaries_x,
            boundaries_y=boundaries_y,
            total_width=float(total_width),
            total_height=float(total_height),
            areas=areas,
        )


@dataclass(frozen=True)
class DiscreteOrdinates2D:
    """Global angular quadrature used by the local SN solves and surface bins."""

    angles: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    incoming_mu: np.ndarray
    incoming_mask: np.ndarray

    @classmethod
    def uniform(cls, num_ordinates: int) -> "DiscreteOrdinates2D":
        if num_ordinates < 4:
            raise ValueError("At least four ordinates are required in 2D.")
        angles = 2.0 * math.pi * (np.arange(num_ordinates, dtype=np.float64) + 0.5) / float(num_ordinates)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        weights = np.full(num_ordinates, 2.0 * math.pi / float(num_ordinates), dtype=np.float64)
        incoming_mu = np.maximum(0.0, INWARD_NORMALS @ directions.T)
        incoming_mask = incoming_mu > 1.0e-12
        return cls(
            angles=angles,
            directions=directions,
            weights=weights,
            incoming_mu=incoming_mu,
            incoming_mask=incoming_mask,
        )

    def nearest_incoming_ordinate(self, face: int, direction: np.ndarray) -> int:
        """Return the incoming ordinate index whose direction best matches `direction`."""

        candidates = np.flatnonzero(self.incoming_mask[face])
        if candidates.size == 0:
            raise ValueError(f"Face {face} has no incoming ordinates.")
        dots = self.directions[candidates] @ direction
        return int(candidates[int(np.argmax(dots))])


@dataclass(frozen=True)
class FaceCurrentRecord:
    """Sparse record view for a single binned incoming surface-current contribution."""

    cell_y: int
    cell_x: int
    face: int
    segment: int
    ordinate: int
    current: float
    crossings: int


@dataclass
class SurfaceCurrentTally:
    """Incoming face current bins for every coarse cell.

    The `current` array stores current-like contributions normalized per source
    particle. Its index order is `(cell_y, cell_x, face, segment, ordinate)`.
    """

    current: np.ndarray
    counts: np.ndarray
    segment_counts: np.ndarray
    segment_lengths: np.ndarray

    @classmethod
    def zeros(
        cls,
        num_cells_y: int,
        num_cells_x: int,
        segment_counts: Sequence[int],
        segment_lengths: Sequence[float],
        num_ordinates: int,
    ) -> "SurfaceCurrentTally":
        max_segments = int(max(segment_counts))
        return cls(
            current=np.zeros((num_cells_y, num_cells_x, 4, max_segments, num_ordinates), dtype=np.float64),
            counts=np.zeros((num_cells_y, num_cells_x, 4, max_segments, num_ordinates), dtype=np.int64),
            segment_counts=np.asarray(segment_counts, dtype=np.int64),
            segment_lengths=np.asarray(segment_lengths, dtype=np.float64),
        )

    def add(self, cell_y: int, cell_x: int, face: int, segment: int, ordinate: int, value: float) -> None:
        self.current[cell_y, cell_x, face, segment, ordinate] += value
        self.counts[cell_y, cell_x, face, segment, ordinate] += 1

    def active_bin_count(self) -> int:
        return int(np.count_nonzero(self.current > 0.0))

    def records_for_cell(self, cell_y: int, cell_x: int, min_current: float = 0.0) -> List[FaceCurrentRecord]:
        records: List[FaceCurrentRecord] = []
        current_block = self.current[cell_y, cell_x]
        count_block = self.counts[cell_y, cell_x]
        for face in range(4):
            seg_count = int(self.segment_counts[face])
            for segment in range(seg_count):
                for ordinate in range(current_block.shape[-1]):
                    value = float(current_block[face, segment, ordinate])
                    if value <= min_current:
                        continue
                    records.append(
                        FaceCurrentRecord(
                            cell_y=cell_y,
                            cell_x=cell_x,
                            face=face,
                            segment=segment,
                            ordinate=ordinate,
                            current=value,
                            crossings=int(count_block[face, segment, ordinate]),
                        )
                    )
        return records

    def boundary_angular_flux(self, cell_y: int, cell_x: int, ordinates: DiscreteOrdinates2D) -> np.ndarray:
        """Convert incoming current bins into incoming angular flux values."""

        block = np.zeros_like(self.current[cell_y, cell_x])
        for face in range(4):
            seg_count = int(self.segment_counts[face])
            seg_length = float(self.segment_lengths[face])
            for ordinate in np.flatnonzero(ordinates.incoming_mask[face]):
                denom = ordinates.weights[ordinate] * ordinates.incoming_mu[face, ordinate] * seg_length
                if denom > 0.0:
                    block[face, :seg_count, ordinate] = self.current[cell_y, cell_x, face, :seg_count, ordinate] / denom
        return block


@dataclass(frozen=True)
class Hierarchy2D:
    """Coarse/fine hierarchy and per-level material data."""

    coarse_mesh: RectangularMesh2D
    fine_mesh: RectangularMesh2D
    fine_sigma_t: np.ndarray
    fine_sigma_s: np.ndarray
    fine_sigma_a: np.ndarray
    fine_material_map: np.ndarray
    fine_source_mask: np.ndarray
    coarse_sigma_t: np.ndarray
    coarse_sigma_s: np.ndarray
    coarse_sigma_a: np.ndarray
    coarse_material_map: np.ndarray
    coarse_source_strength: np.ndarray
    fine_per_coarse_x: int
    fine_per_coarse_y: int

    def fine_slice(self, coarse_y: int, coarse_x: int) -> Tuple[slice, slice]:
        y0 = coarse_y * self.fine_per_coarse_y
        y1 = y0 + self.fine_per_coarse_y
        x0 = coarse_x * self.fine_per_coarse_x
        x1 = x0 + self.fine_per_coarse_x
        return slice(y0, y1), slice(x0, x1)

    @property
    def face_segment_counts(self) -> np.ndarray:
        return np.array(
            [
                self.fine_per_coarse_y,
                self.fine_per_coarse_y,
                self.fine_per_coarse_x,
                self.fine_per_coarse_x,
            ],
            dtype=np.int64,
        )

    @property
    def face_segment_lengths(self) -> np.ndarray:
        return np.array(
            [
                self.coarse_mesh.heights[0] / float(self.fine_per_coarse_y),
                self.coarse_mesh.heights[0] / float(self.fine_per_coarse_y),
                self.coarse_mesh.widths[0] / float(self.fine_per_coarse_x),
                self.coarse_mesh.widths[0] / float(self.fine_per_coarse_x),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CoarseTransportResult:
    """Outputs from the coarse transport and surface binning stage."""

    track_length: np.ndarray
    flux: np.ndarray
    energy_deposition: np.ndarray
    surface_tally: SurfaceCurrentTally
    leakage_weight: np.ndarray
    source_histories: int
    mean_events: float
    elapsed_sec: float


@dataclass(frozen=True)
class MultigridFluxResult:
    """Assembled fine-grid multigrid flux estimate and diagnostics."""

    hierarchy: Hierarchy2D
    ordinates: DiscreteOrdinates2D
    coarse_result: CoarseTransportResult
    fine_flux: np.ndarray
    fine_track_length: np.ndarray
    local_iterations: np.ndarray
    local_scale_factors: np.ndarray
    fixed_source: np.ndarray
    elapsed_sec: float


def _block_average(field: np.ndarray, coarse_shape: Tuple[int, int]) -> np.ndarray:
    coarse_y, coarse_x = coarse_shape
    if field.shape[0] % coarse_y != 0 or field.shape[1] % coarse_x != 0:
        raise ValueError("Fine grid must divide evenly into the requested coarse grid.")
    fine_per_coarse_y = field.shape[0] // coarse_y
    fine_per_coarse_x = field.shape[1] // coarse_x
    reshaped = field.reshape(coarse_y, fine_per_coarse_y, coarse_x, fine_per_coarse_x)
    return reshaped.mean(axis=(1, 3))


def _block_sum(field: np.ndarray, coarse_shape: Tuple[int, int]) -> np.ndarray:
    coarse_y, coarse_x = coarse_shape
    fine_per_coarse_y = field.shape[0] // coarse_y
    fine_per_coarse_x = field.shape[1] // coarse_x
    reshaped = field.reshape(coarse_y, fine_per_coarse_y, coarse_x, fine_per_coarse_x)
    return reshaped.sum(axis=(1, 3))


def _block_mode(material_map: np.ndarray, coarse_shape: Tuple[int, int]) -> np.ndarray:
    coarse_y, coarse_x = coarse_shape
    fine_per_coarse_y = material_map.shape[0] // coarse_y
    fine_per_coarse_x = material_map.shape[1] // coarse_x
    coarse = np.zeros((coarse_y, coarse_x), dtype=material_map.dtype)
    for cell_y in range(coarse_y):
        for cell_x in range(coarse_x):
            block = material_map[
                cell_y * fine_per_coarse_y : (cell_y + 1) * fine_per_coarse_y,
                cell_x * fine_per_coarse_x : (cell_x + 1) * fine_per_coarse_x,
            ]
            values, counts = np.unique(block, return_counts=True)
            coarse[cell_y, cell_x] = values[int(np.argmax(counts))]
    return coarse


def build_hierarchy_from_fine_maps(
    *,
    total_width: float,
    total_height: float,
    fine_sigma_t: np.ndarray,
    fine_sigma_s: np.ndarray,
    fine_sigma_a: np.ndarray,
    fine_material_map: np.ndarray,
    fine_source_mask: np.ndarray,
    coarse_shape: Tuple[int, int],
) -> Hierarchy2D:
    """Build a Cartesian hierarchy from fine-grid material arrays."""

    fine_num_cells_y, fine_num_cells_x = fine_sigma_t.shape
    coarse_num_cells_y, coarse_num_cells_x = coarse_shape
    if fine_sigma_s.shape != fine_sigma_t.shape or fine_sigma_a.shape != fine_sigma_t.shape:
        raise ValueError("Fine cross-section arrays must share the same shape.")
    if fine_material_map.shape != fine_sigma_t.shape or fine_source_mask.shape != fine_sigma_t.shape:
        raise ValueError("Fine material and source maps must match the fine cross-section shape.")
    if fine_num_cells_x % coarse_num_cells_x != 0 or fine_num_cells_y % coarse_num_cells_y != 0:
        raise ValueError("Fine grid dimensions must be integer multiples of the coarse grid dimensions.")

    fine_mesh = RectangularMesh2D.uniform(fine_num_cells_x, fine_num_cells_y, total_width, total_height)
    coarse_mesh = RectangularMesh2D.uniform(coarse_num_cells_x, coarse_num_cells_y, total_width, total_height)
    coarse_sigma_t = _block_average(fine_sigma_t, coarse_shape)
    coarse_sigma_s = _block_average(fine_sigma_s, coarse_shape)
    coarse_sigma_a = _block_average(fine_sigma_a, coarse_shape)
    coarse_material_map = _block_mode(fine_material_map, coarse_shape)
    source_counts = _block_sum(fine_source_mask.astype(np.float64), coarse_shape)

    return Hierarchy2D(
        coarse_mesh=coarse_mesh,
        fine_mesh=fine_mesh,
        fine_sigma_t=fine_sigma_t.astype(np.float64, copy=False),
        fine_sigma_s=fine_sigma_s.astype(np.float64, copy=False),
        fine_sigma_a=fine_sigma_a.astype(np.float64, copy=False),
        fine_material_map=fine_material_map.copy(),
        fine_source_mask=fine_source_mask.astype(bool, copy=False),
        coarse_sigma_t=coarse_sigma_t,
        coarse_sigma_s=coarse_sigma_s,
        coarse_sigma_a=coarse_sigma_a,
        coarse_material_map=coarse_material_map,
        coarse_source_strength=source_counts,
        fine_per_coarse_x=fine_num_cells_x // coarse_num_cells_x,
        fine_per_coarse_y=fine_num_cells_y // coarse_num_cells_y,
    )


def build_uniform_source_density(source_mask: np.ndarray, cell_areas: np.ndarray) -> np.ndarray:
    """Build a unit-total isotropic source density over the source-mask support."""

    source_density = np.zeros_like(cell_areas, dtype=np.float64)
    total_source_area = float(np.sum(cell_areas[source_mask]))
    if total_source_area <= 0.0:
        raise ValueError("Source mask is empty.")
    source_density[source_mask] = 1.0 / total_source_area
    return source_density


def _track_length_contribution(weight: float, sigma_a: float, distance: float) -> float:
    if distance <= 0.0 or weight <= 0.0:
        return 0.0
    if sigma_a <= 1.0e-12:
        return weight * distance
    optical_depth = sigma_a * distance
    if optical_depth < 1.0e-5:
        return weight * distance
    return weight * (1.0 - math.exp(-optical_depth)) / sigma_a


def _sample_isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    while True:
        theta = 2.0 * math.pi * float(rng.random())
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        if abs(direction[0]) > 1.0e-12 or abs(direction[1]) > 1.0e-12:
            return direction


def _distance_to_boundary(mesh: RectangularMesh2D, cell_y: int, cell_x: int, x: float, y: float, direction: np.ndarray) -> Tuple[float, int]:
    u, v = float(direction[0]), float(direction[1])
    distance = math.inf
    face = -1
    epsilon = 1.0e-12

    if u > epsilon:
        candidate = (mesh.boundaries_x[cell_x + 1] - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_RIGHT
    elif u < -epsilon:
        candidate = (mesh.boundaries_x[cell_x] - x) / u
        if candidate < distance:
            distance = candidate
            face = FACE_LEFT

    if v > epsilon:
        candidate = (mesh.boundaries_y[cell_y + 1] - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_TOP
    elif v < -epsilon:
        candidate = (mesh.boundaries_y[cell_y] - y) / v
        if candidate < distance:
            distance = candidate
            face = FACE_BOTTOM

    return max(distance, 0.0), face


def _face_segment_index(hierarchy: Hierarchy2D, cell_y: int, cell_x: int, face: int, x: float, y: float) -> int:
    if face in (FACE_LEFT, FACE_RIGHT):
        local_coordinate = y - hierarchy.coarse_mesh.boundaries_y[cell_y]
        face_length = hierarchy.coarse_mesh.heights[cell_y]
        segment_count = hierarchy.fine_per_coarse_y
    else:
        local_coordinate = x - hierarchy.coarse_mesh.boundaries_x[cell_x]
        face_length = hierarchy.coarse_mesh.widths[cell_x]
        segment_count = hierarchy.fine_per_coarse_x
    clipped = min(max(local_coordinate, 0.0), np.nextafter(face_length, 0.0))
    segment = int(clipped / (face_length / float(segment_count)))
    return min(max(segment, 0), segment_count - 1)


def run_coarse_surface_binned_transport(
    hierarchy: Hierarchy2D,
    *,
    num_histories: int,
    ordinates: DiscreteOrdinates2D,
    seed: int = 1,
    energy_threshold: float = 1.0e-8,
    max_events_per_history: int = 20000,
) -> CoarseTransportResult:
    """Run a coarse analog MC solve and bin inter-cell surface crossings."""

    start_time = time.perf_counter()
    coarse_mesh = hierarchy.coarse_mesh
    coarse_num_cells_y, coarse_num_cells_x = hierarchy.coarse_sigma_t.shape
    source_strength = hierarchy.coarse_source_strength.astype(np.float64).ravel()
    total_source_strength = float(source_strength.sum())
    if total_source_strength <= 0.0:
        raise ValueError("The hierarchy does not contain any source-support coarse cells.")

    source_cdf = np.cumsum(source_strength / total_source_strength)
    rng = np.random.default_rng(seed)

    track_length = np.zeros_like(hierarchy.coarse_sigma_t, dtype=np.float64)
    energy_deposition = np.zeros_like(hierarchy.coarse_sigma_t, dtype=np.float64)
    leakage_weight = np.zeros(4, dtype=np.float64)
    tally = SurfaceCurrentTally.zeros(
        num_cells_y=coarse_num_cells_y,
        num_cells_x=coarse_num_cells_x,
        segment_counts=hierarchy.face_segment_counts,
        segment_lengths=hierarchy.face_segment_lengths,
        num_ordinates=ordinates.directions.shape[0],
    )

    epsilon = 1.0e-10 * min(coarse_mesh.total_width, coarse_mesh.total_height)
    total_events = 0

    for _ in range(num_histories):
        source_index = int(np.searchsorted(source_cdf, float(rng.random()), side="right"))
        source_index = min(source_index, source_cdf.size - 1)
        cell_y, cell_x = np.unravel_index(source_index, hierarchy.coarse_sigma_t.shape)

        x = coarse_mesh.boundaries_x[cell_x] + coarse_mesh.widths[cell_x] * float(rng.random())
        y = coarse_mesh.boundaries_y[cell_y] + coarse_mesh.heights[cell_y] * float(rng.random())
        direction = _sample_isotropic_direction(rng)
        weight = 1.0

        for _event in range(max_events_per_history):
            total_events += 1
            sigma_s = max(0.0, float(hierarchy.coarse_sigma_s[cell_y, cell_x]))
            sigma_a = max(0.0, float(hierarchy.coarse_sigma_a[cell_y, cell_x]))

            free_path = math.inf if sigma_s <= 1.0e-12 else -math.log(max(1.0e-15, 1.0 - float(rng.random()))) / sigma_s
            boundary_distance, face = _distance_to_boundary(coarse_mesh, cell_y, cell_x, x, y, direction)
            travel_distance = min(free_path, boundary_distance)
            travel_distance = max(travel_distance, 0.0)

            if travel_distance <= 0.0 and face < 0:
                break

            weight_before = weight
            track_length[cell_y, cell_x] += _track_length_contribution(weight_before, sigma_a, travel_distance)
            weight *= math.exp(-sigma_a * travel_distance)
            energy_deposition[cell_y, cell_x] += weight_before - weight

            x += travel_distance * direction[0]
            y += travel_distance * direction[1]

            if weight <= energy_threshold:
                break

            boundary_event = face >= 0 and abs(travel_distance - boundary_distance) <= 1.0e-10 * max(1.0, boundary_distance)
            if boundary_event:
                next_cell_x = cell_x + (1 if face == FACE_RIGHT else -1 if face == FACE_LEFT else 0)
                next_cell_y = cell_y + (1 if face == FACE_TOP else -1 if face == FACE_BOTTOM else 0)

                if not (0 <= next_cell_x < coarse_num_cells_x and 0 <= next_cell_y < coarse_num_cells_y):
                    leakage_weight[face] += weight
                    break

                incoming_face = int(OPPOSITE_FACE[face])
                mu_in = float(np.dot(direction, INWARD_NORMALS[incoming_face]))
                if mu_in > 1.0e-12:
                    ordinate = ordinates.nearest_incoming_ordinate(incoming_face, direction)
                    segment = _face_segment_index(hierarchy, next_cell_y, next_cell_x, incoming_face, x, y)
                    tally.add(next_cell_y, next_cell_x, incoming_face, segment, ordinate, weight * mu_in)

                cell_x = next_cell_x
                cell_y = next_cell_y
                x = min(max(x + epsilon * direction[0], coarse_mesh.boundaries_x[cell_x] + epsilon), coarse_mesh.boundaries_x[cell_x + 1] - epsilon)
                y = min(max(y + epsilon * direction[1], coarse_mesh.boundaries_y[cell_y] + epsilon), coarse_mesh.boundaries_y[cell_y + 1] - epsilon)
            else:
                direction = _sample_isotropic_direction(rng)

    normalization = 1.0 / float(num_histories)
    track_length *= normalization
    energy_deposition *= normalization
    tally.current *= normalization
    leakage_weight *= normalization
    flux = np.divide(track_length, coarse_mesh.areas, out=np.zeros_like(track_length), where=coarse_mesh.areas > 0.0)

    return CoarseTransportResult(
        track_length=track_length,
        flux=flux,
        energy_deposition=energy_deposition,
        surface_tally=tally,
        leakage_weight=leakage_weight,
        source_histories=int(num_histories),
        mean_events=float(total_events) / float(max(1, num_histories)),
        elapsed_sec=time.perf_counter() - start_time,
    )


def solve_local_sn_block(
    *,
    sigma_t: np.ndarray,
    sigma_s: np.ndarray,
    fixed_source: np.ndarray,
    cell_widths: np.ndarray,
    cell_heights: np.ndarray,
    boundary_angular_flux: np.ndarray,
    ordinates: DiscreteOrdinates2D,
    target_track_length: float,
    max_iterations: int = 20,
    tolerance: float = 1.0e-5,
) -> Tuple[np.ndarray, int, float]:
    """Solve a local 2D SN refinement problem on a coarse-cell fine subgrid.

    The prototype assumes the subgrid shares a homogeneous material within the
    parent coarse cell, but the solver accepts array inputs so the interface
    remains easy to extend later.
    """

    num_cells_y, num_cells_x = sigma_t.shape
    num_ordinates = ordinates.directions.shape[0]
    scalar_flux = np.zeros((num_cells_y, num_cells_x), dtype=np.float64)
    angular_flux = np.zeros((num_ordinates, num_cells_y, num_cells_x), dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        isotropic_source = (fixed_source + sigma_s * scalar_flux) / (2.0 * math.pi)

        for ordinate in range(num_ordinates):
            omega_x, omega_y = ordinates.directions[ordinate]
            x_forward = omega_x > 0.0
            y_forward = omega_y > 0.0
            x_face = FACE_LEFT if x_forward else FACE_RIGHT
            y_face = FACE_BOTTOM if y_forward else FACE_TOP

            x_indices = range(num_cells_x) if x_forward else range(num_cells_x - 1, -1, -1)
            y_indices = range(num_cells_y) if y_forward else range(num_cells_y - 1, -1, -1)

            y_stream = np.array(boundary_angular_flux[y_face, :num_cells_x, ordinate], dtype=np.float64, copy=True)

            for local_y in y_indices:
                x_incoming = float(boundary_angular_flux[x_face, local_y, ordinate])
                for local_x in x_indices:
                    coeff_x = abs(omega_x) / max(cell_widths[local_x], 1.0e-14)
                    coeff_y = abs(omega_y) / max(cell_heights[local_y], 1.0e-14)
                    denom = sigma_t[local_y, local_x] + coeff_x + coeff_y
                    psi = 0.0
                    if denom > 0.0:
                        psi = (
                            isotropic_source[local_y, local_x]
                            + coeff_x * x_incoming
                            + coeff_y * y_stream[local_x]
                        ) / denom
                    angular_flux[ordinate, local_y, local_x] = psi
                    x_incoming = psi
                    y_stream[local_x] = psi

        updated_scalar_flux = np.tensordot(ordinates.weights, angular_flux, axes=(0, 0))
        denom = max(1.0e-12, float(np.max(np.abs(updated_scalar_flux))))
        delta = float(np.max(np.abs(updated_scalar_flux - scalar_flux))) / denom
        scalar_flux = updated_scalar_flux
        if delta < tolerance:
            break

    cell_areas = np.outer(cell_heights, cell_widths)
    local_track_length = float(np.sum(scalar_flux * cell_areas))
    if target_track_length <= 0.0:
        scalar_flux.fill(0.0)
        return scalar_flux, iteration, 0.0

    if local_track_length > 0.0:
        scale = target_track_length / local_track_length
        scalar_flux *= scale
        return scalar_flux, iteration, scale

    uniform_flux = target_track_length / float(np.sum(cell_areas))
    scalar_flux.fill(uniform_flux)
    return scalar_flux, iteration, 0.0


def run_multigrid_refinement(
    hierarchy: Hierarchy2D,
    *,
    num_histories: int,
    ordinates: DiscreteOrdinates2D,
    seed: int = 1,
    energy_threshold: float = 1.0e-8,
    max_coarse_events: int = 20000,
    sn_max_iterations: int = 20,
    sn_tolerance: float = 1.0e-5,
    homogeneous_local_blocks: bool = True,
) -> MultigridFluxResult:
    """Run the coarse transport stage followed by local-cell SN refinement.

    When `homogeneous_local_blocks` is true, each local solve uses the parent
    coarse-cell cross sections uniformly across its embedded fine subgrid. This
    matches the default `14x14 -> 56x56` lattice hierarchy where each coarse
    cell stays inside one base-material block.
    """

    start_time = time.perf_counter()
    coarse_result = run_coarse_surface_binned_transport(
        hierarchy,
        num_histories=num_histories,
        ordinates=ordinates,
        seed=seed,
        energy_threshold=energy_threshold,
        max_events_per_history=max_coarse_events,
    )

    fixed_source = build_uniform_source_density(hierarchy.fine_source_mask, hierarchy.fine_mesh.areas)
    fine_flux = np.zeros_like(hierarchy.fine_sigma_t, dtype=np.float64)
    local_iterations = np.zeros_like(hierarchy.coarse_sigma_t, dtype=np.int64)
    local_scale_factors = np.zeros_like(hierarchy.coarse_sigma_t, dtype=np.float64)

    coarse_num_cells_y, coarse_num_cells_x = hierarchy.coarse_sigma_t.shape
    for coarse_y in range(coarse_num_cells_y):
        for coarse_x in range(coarse_num_cells_x):
            ys, xs = hierarchy.fine_slice(coarse_y, coarse_x)
            boundary_flux = coarse_result.surface_tally.boundary_angular_flux(coarse_y, coarse_x, ordinates)
            if homogeneous_local_blocks:
                block_shape = hierarchy.fine_sigma_t[ys, xs].shape
                sigma_t_block = np.full(block_shape, hierarchy.coarse_sigma_t[coarse_y, coarse_x], dtype=np.float64)
                sigma_s_block = np.full(block_shape, hierarchy.coarse_sigma_s[coarse_y, coarse_x], dtype=np.float64)
            else:
                sigma_t_block = hierarchy.fine_sigma_t[ys, xs]
                sigma_s_block = hierarchy.fine_sigma_s[ys, xs]
            block_flux, iteration_count, scale = solve_local_sn_block(
                sigma_t=sigma_t_block,
                sigma_s=sigma_s_block,
                fixed_source=fixed_source[ys, xs],
                cell_widths=hierarchy.fine_mesh.widths[xs],
                cell_heights=hierarchy.fine_mesh.heights[ys],
                boundary_angular_flux=boundary_flux,
                ordinates=ordinates,
                target_track_length=float(coarse_result.track_length[coarse_y, coarse_x]),
                max_iterations=sn_max_iterations,
                tolerance=sn_tolerance,
            )
            fine_flux[ys, xs] = block_flux
            local_iterations[coarse_y, coarse_x] = iteration_count
            local_scale_factors[coarse_y, coarse_x] = scale

    fine_track_length = fine_flux * hierarchy.fine_mesh.areas
    return MultigridFluxResult(
        hierarchy=hierarchy,
        ordinates=ordinates,
        coarse_result=coarse_result,
        fine_flux=fine_flux,
        fine_track_length=fine_track_length,
        local_iterations=local_iterations,
        local_scale_factors=local_scale_factors,
        fixed_source=fixed_source,
        elapsed_sec=time.perf_counter() - start_time,
    )


def coarsen_fine_scalar_flux(hierarchy: Hierarchy2D, fine_flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Area-weight the fine scalar flux back onto the coarse mesh."""

    coarse_num_cells_y, coarse_num_cells_x = hierarchy.coarse_sigma_t.shape
    coarse_track = np.zeros((coarse_num_cells_y, coarse_num_cells_x), dtype=np.float64)
    coarse_flux = np.zeros((coarse_num_cells_y, coarse_num_cells_x), dtype=np.float64)
    for coarse_y in range(coarse_num_cells_y):
        for coarse_x in range(coarse_num_cells_x):
            ys, xs = hierarchy.fine_slice(coarse_y, coarse_x)
            block_track = float(np.sum(fine_flux[ys, xs] * hierarchy.fine_mesh.areas[ys, xs]))
            coarse_track[coarse_y, coarse_x] = block_track
            coarse_flux[coarse_y, coarse_x] = block_track / hierarchy.coarse_mesh.areas[coarse_y, coarse_x]
    return coarse_flux, coarse_track
