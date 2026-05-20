"""Helpers for iterative multigroup steady-state lattice solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

DOMAIN_SIZE = 7.0
BASE_CELLS = 7
REFINEMENT = 8
NG = 7

MAT_SCATTERING = 0
MAT_ABSORBING = 1
MAT_SOURCE = 2

MATERIAL_PROPERTIES = {
    MAT_SCATTERING: {"sigma_a": 0.0, "sigma_s": 1.0},
    MAT_ABSORBING: {"sigma_a": 9.5, "sigma_s": 0.5},
    MAT_SOURCE: {"sigma_a": 0.0, "sigma_s": 1.0},
}

LATTICE_PATTERN = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 2, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class SteadyLatticeGeometry:
    material_map: np.ndarray
    source_cells: np.ndarray
    cell_widths: np.ndarray
    cell_heights: np.ndarray
    cell_boundaries_x: np.ndarray
    cell_boundaries_y: np.ndarray
    cell_areas: np.ndarray
    num_cells_x: int
    num_cells_y: int
    refinement: int
    domain_size: float


@dataclass(frozen=True)
class MultigroupSteadyLatticeProblem:
    geometry: SteadyLatticeGeometry
    num_groups: int
    sigma_a_base: np.ndarray
    sigma_s_base: np.ndarray
    absorption_multipliers: np.ndarray
    scattering_matrix: np.ndarray
    external_source: np.ndarray


@dataclass(frozen=True)
class MixedSourceParticleBank:
    cell_y: np.ndarray
    cell_x: np.ndarray
    pos_rx: np.ndarray
    pos_ry: np.ndarray
    dir_unit_mu: np.ndarray
    dir_unit_phi: np.ndarray
    trajectory_seeds: np.ndarray
    component: np.ndarray
    source_component_count: int
    uniform_component_count: int
    source_fraction: float


@dataclass(frozen=True)
class MultigroupIterationResult:
    flux: np.ndarray
    source: np.ndarray
    convergence_history: np.ndarray
    source_strength_history: np.ndarray
    particle_counts_history: np.ndarray
    converged: bool
    outer_iterations: int


@dataclass(frozen=True)
class GroupCrossSections:
    sigma_s_within: np.ndarray
    sigma_a_effective: np.ndarray
    absorption_multipliers: np.ndarray
    scatter_matrix: np.ndarray


GroupSolve = Callable[[int, np.ndarray, np.ndarray, np.ndarray, int, int], Any]


def default_scattering_matrix(num_groups: int = NG) -> np.ndarray:
    num_groups = int(num_groups)
    matrix = np.zeros((num_groups, num_groups), dtype=np.float64)
    template = np.array([0.55, 0.30, 0.15], dtype=np.float64)
    for group_in in range(num_groups):
        if group_in == num_groups - 1:
            matrix[group_in, group_in] = 1.0
            continue
        diagonal = min(0.72, 0.58 + 0.02 * group_in)
        matrix[group_in, group_in] = diagonal
        targets = min(template.size, num_groups - group_in - 1)
        weights = template[:targets] / np.sum(template[:targets])
        matrix[group_in, group_in + 1 : group_in + 1 + targets] = (1.0 - diagonal) * weights
    return matrix


def default_scatter_matrix(num_groups: int = NG) -> np.ndarray:
    return default_scattering_matrix(num_groups)


def _build_sigma_s_base(
    geometry: SteadyLatticeGeometry,
    *,
    sigma_s_multiplier: float = 1.0,
) -> np.ndarray:
    multiplier = float(sigma_s_multiplier)
    if multiplier <= 0.0:
        raise ValueError(f"sigma_s_multiplier must be positive; got {sigma_s_multiplier}.")

    sigma_s_base = np.zeros_like(geometry.material_map, dtype=np.float64)
    for material_id, props in MATERIAL_PROPERTIES.items():
        mask = geometry.material_map == int(material_id)
        sigma_s_base[mask] = float(props["sigma_s"]) * multiplier
    return sigma_s_base


def build_steady_lattice_geometry(refinement: int = REFINEMENT) -> SteadyLatticeGeometry:
    if refinement <= 0:
        raise ValueError(f"refinement must be positive; got {refinement}.")

    num_cells_x = BASE_CELLS * int(refinement)
    num_cells_y = BASE_CELLS * int(refinement)
    material_map = np.zeros((num_cells_y, num_cells_x), dtype=np.int64)
    base_cell_size = DOMAIN_SIZE / BASE_CELLS

    def coord_to_idx(coord: float, total_cells: int) -> int:
        return int(coord * total_cells / DOMAIN_SIZE)

    for base_i in range(BASE_CELLS):
        for base_j in range(BASE_CELLS):
            iy0 = coord_to_idx(base_i * base_cell_size, num_cells_y)
            iy1 = coord_to_idx((base_i + 1) * base_cell_size, num_cells_y)
            ix0 = coord_to_idx(base_j * base_cell_size, num_cells_x)
            ix1 = coord_to_idx((base_j + 1) * base_cell_size, num_cells_x)
            material_map[iy0:iy1, ix0:ix1] = int(LATTICE_PATTERN[base_i, base_j])

    cell_widths = np.full(num_cells_x, DOMAIN_SIZE / num_cells_x, dtype=np.float64)
    cell_heights = np.full(num_cells_y, DOMAIN_SIZE / num_cells_y, dtype=np.float64)
    cell_boundaries_x = np.concatenate(([0.0], np.cumsum(cell_widths)))
    cell_boundaries_y = np.concatenate(([0.0], np.cumsum(cell_heights)))
    source_cells = np.argwhere(material_map == MAT_SOURCE).astype(np.int64)
    if source_cells.size == 0:
        raise RuntimeError("No source cells found in the benchmark lattice.")

    return SteadyLatticeGeometry(
        material_map=material_map,
        source_cells=source_cells,
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=cell_boundaries_x,
        cell_boundaries_y=cell_boundaries_y,
        cell_areas=np.outer(cell_heights, cell_widths),
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        refinement=int(refinement),
        domain_size=DOMAIN_SIZE,
    )


def build_multigroup_steady_lattice_problem(
    refinement: int = REFINEMENT,
    num_groups: int = NG,
    absorption_multipliers: np.ndarray | None = None,
    scattering_matrix: np.ndarray | None = None,
    external_source_strength: float = 1.0,
    sigma_s_multiplier: float = 1.0,
) -> MultigroupSteadyLatticeProblem:
    geometry = build_steady_lattice_geometry(refinement=refinement)
    sigma_a_base = np.zeros_like(geometry.material_map, dtype=np.float64)
    for material_id, props in MATERIAL_PROPERTIES.items():
        mask = geometry.material_map == int(material_id)
        sigma_a_base[mask] = float(props["sigma_a"])
    sigma_s_base = _build_sigma_s_base(geometry, sigma_s_multiplier=sigma_s_multiplier)

    multipliers = np.ones(int(num_groups), dtype=np.float64)
    if absorption_multipliers is not None:
        multipliers = np.asarray(absorption_multipliers, dtype=np.float64)
    if multipliers.shape != (int(num_groups),):
        raise ValueError(
            f"absorption_multipliers must have shape ({int(num_groups)},); got {multipliers.shape}."
        )

    scatter = default_scattering_matrix(num_groups) if scattering_matrix is None else np.asarray(scattering_matrix, dtype=np.float64)
    if scatter.shape != (int(num_groups), int(num_groups)):
        raise ValueError(f"scattering_matrix must have shape {(int(num_groups), int(num_groups))}; got {scatter.shape}.")
    if np.any(scatter < 0.0):
        raise ValueError("scattering_matrix entries must be non-negative.")
    if not np.allclose(scatter.sum(axis=1), 1.0, atol=1.0e-12):
        raise ValueError("Each scattering_matrix row must sum to 1.")

    external_source = np.zeros((int(num_groups), geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
    per_source_cell = float(external_source_strength) / float(geometry.source_cells.shape[0])
    external_source[0, geometry.source_cells[:, 0], geometry.source_cells[:, 1]] = per_source_cell

    return MultigroupSteadyLatticeProblem(
        geometry=geometry,
        num_groups=int(num_groups),
        sigma_a_base=sigma_a_base,
        sigma_s_base=sigma_s_base,
        absorption_multipliers=multipliers,
        scattering_matrix=scatter,
        external_source=external_source,
    )


def build_lattice_problem(refinement: int = REFINEMENT) -> SteadyLatticeGeometry:
    return build_steady_lattice_geometry(refinement=refinement)


def build_group_fixed_source(
    problem: SteadyLatticeGeometry | MultigroupSteadyLatticeProblem,
    num_groups: int,
    *,
    source_group: int = 0,
) -> np.ndarray:
    geometry = problem.geometry if isinstance(problem, MultigroupSteadyLatticeProblem) else problem
    source = np.zeros((int(num_groups), geometry.num_cells_y, geometry.num_cells_x), dtype=np.float64)
    source[int(source_group), geometry.source_cells[:, 0], geometry.source_cells[:, 1]] = 1.0
    return source


def build_group_cross_sections(
    problem: SteadyLatticeGeometry | MultigroupSteadyLatticeProblem,
    num_groups: int,
    *,
    absorption_multipliers: np.ndarray | None = None,
    scatter_matrix: np.ndarray | None = None,
    sigma_s_multiplier: float = 1.0,
) -> GroupCrossSections:
    mg_problem = (
        problem
        if isinstance(problem, MultigroupSteadyLatticeProblem)
        else build_multigroup_steady_lattice_problem(
            refinement=problem.refinement,
            num_groups=num_groups,
            absorption_multipliers=absorption_multipliers,
            scattering_matrix=scatter_matrix,
            sigma_s_multiplier=sigma_s_multiplier,
        )
    )
    sigma_s_within = np.zeros_like(mg_problem.external_source)
    sigma_a_effective = np.zeros_like(mg_problem.external_source)
    for group in range(mg_problem.num_groups):
        sigma_a_effective[group], sigma_s_within[group] = group_effective_xs(mg_problem, group)
    return GroupCrossSections(
        sigma_s_within=sigma_s_within,
        sigma_a_effective=sigma_a_effective,
        absorption_multipliers=mg_problem.absorption_multipliers,
        scatter_matrix=mg_problem.scattering_matrix,
    )


def build_lagged_scatter_source(
    problem: SteadyLatticeGeometry | MultigroupSteadyLatticeProblem,
    scatter_matrix: np.ndarray,
    flux: np.ndarray,
) -> np.ndarray:
    geometry = problem.geometry if isinstance(problem, MultigroupSteadyLatticeProblem) else problem
    scatter = np.asarray(scatter_matrix, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)
    source = np.zeros_like(flux_arr)
    sigma_s_base = (
        np.asarray(problem.sigma_s_base, dtype=np.float64)
        if isinstance(problem, MultigroupSteadyLatticeProblem)
        else _build_sigma_s_base(geometry)
    )
    for group in range(scatter.shape[0]):
        weights = scatter[:, group].copy()
        weights[group] = 0.0
        source[group] = np.sum(sigma_s_base[None, :, :] * weights[:, None, None] * flux_arr, axis=0)
    return source


def group_effective_xs(
    problem: MultigroupSteadyLatticeProblem,
    group: int,
) -> tuple[np.ndarray, np.ndarray]:
    g = int(group)
    s_gg = float(problem.scattering_matrix[g, g])
    sigma_s_within = problem.sigma_s_base * s_gg
    sigma_a_effective = problem.sigma_a_base * problem.absorption_multipliers[g] + problem.sigma_s_base * (1.0 - s_gg)
    return sigma_a_effective, sigma_s_within


def group_outscatter_source(
    problem: MultigroupSteadyLatticeProblem,
    flux: np.ndarray,
    group: int,
) -> np.ndarray:
    flux_arr = np.asarray(flux, dtype=np.float64)
    expected_shape = (problem.num_groups, problem.geometry.num_cells_y, problem.geometry.num_cells_x)
    if flux_arr.shape != expected_shape:
        raise ValueError(f"flux must have shape {expected_shape}; got {flux_arr.shape}.")
    weights = problem.scattering_matrix[:, int(group)].copy()
    weights[int(group)] = 0.0
    return np.sum(problem.sigma_s_base[None, :, :] * weights[:, None, None] * flux_arr, axis=0)


def group_source_strengths(problem: MultigroupSteadyLatticeProblem, source: np.ndarray) -> np.ndarray:
    source_arr = np.asarray(source, dtype=np.float64)
    return np.sum(source_arr * problem.geometry.cell_areas[None, :, :], axis=(1, 2))


def build_mixed_source_particle_bank(
    problem: MultigroupSteadyLatticeProblem,
    num_particles: int,
    seed: int,
    *,
    source_fraction: float = 0.5,
) -> MixedSourceParticleBank:
    total_particles = int(num_particles)
    if total_particles < 0:
        raise ValueError(f"num_particles must be non-negative; got {num_particles}.")
    if not 0.0 <= float(source_fraction) <= 1.0:
        raise ValueError(f"source_fraction must lie in [0, 1]; got {source_fraction}.")

    geometry = problem.geometry
    cell_areas = geometry.cell_areas.astype(np.float64)
    flat_all = np.arange(geometry.num_cells_y * geometry.num_cells_x, dtype=np.int64)
    all_probs = cell_areas.reshape(-1)
    all_probs /= np.sum(all_probs)

    source_count = int(np.floor(float(source_fraction) * total_particles))
    uniform_count = total_particles - source_count

    source_flat = geometry.source_cells[:, 0] * geometry.num_cells_x + geometry.source_cells[:, 1]
    source_probs = cell_areas[geometry.source_cells[:, 0], geometry.source_cells[:, 1]]
    source_probs /= np.sum(source_probs)

    sampled = np.empty(total_particles, dtype=np.int64)
    component = np.ones(total_particles, dtype=np.int64)
    source_mask = np.zeros(total_particles, dtype=bool)
    if total_particles > 0 and source_count > 0:
        src_cumulative = np.floor(np.arange(1, total_particles + 1, dtype=np.float64) * float(source_fraction)).astype(np.int64)
        src_prev = np.floor(np.arange(total_particles, dtype=np.float64) * float(source_fraction)).astype(np.int64)
        source_mask = src_cumulative > src_prev
        if int(np.count_nonzero(source_mask)) != source_count:
            source_mask[:] = False
            source_mask[:source_count] = True
    component[source_mask] = 0
    rng_source = np.random.default_rng(int(seed) + 11)
    rng_uniform = np.random.default_rng(int(seed) + 17)
    if source_count > 0:
        sampled[source_mask] = rng_source.choice(source_flat, size=source_count, replace=True, p=source_probs)
    if uniform_count > 0:
        sampled[~source_mask] = rng_uniform.choice(flat_all, size=uniform_count, replace=True, p=all_probs)

    cell_y, cell_x = np.divmod(sampled, geometry.num_cells_x)
    pos_rx = np.random.default_rng(int(seed) + 23).random(total_particles, dtype=np.float64)
    pos_ry = np.random.default_rng(int(seed) + 29).random(total_particles, dtype=np.float64)
    dir_unit_mu = np.random.default_rng(int(seed) + 31).random(total_particles, dtype=np.float64)
    dir_unit_phi = np.random.default_rng(int(seed) + 37).random(total_particles, dtype=np.float64)
    trajectory_seeds = np.random.default_rng(int(seed) + 41).integers(0, 2**31 - 1, size=total_particles, dtype=np.int64)

    return MixedSourceParticleBank(
        cell_y=cell_y.astype(np.int64),
        cell_x=cell_x.astype(np.int64),
        pos_rx=pos_rx,
        pos_ry=pos_ry,
        dir_unit_mu=dir_unit_mu,
        dir_unit_phi=dir_unit_phi,
        trajectory_seeds=trajectory_seeds,
        component=component,
        source_component_count=source_count,
        uniform_component_count=uniform_count,
        source_fraction=float(source_fraction),
    )


def mixed_source_particle_weights(
    problem: MultigroupSteadyLatticeProblem,
    source_map: np.ndarray,
    bank: MixedSourceParticleBank,
) -> np.ndarray:
    source_arr = np.asarray(source_map, dtype=np.float64)
    expected_shape = (problem.geometry.num_cells_y, problem.geometry.num_cells_x)
    if source_arr.shape != expected_shape:
        raise ValueError(f"source_map must have shape {expected_shape}; got {source_arr.shape}.")

    areas = problem.geometry.cell_areas
    source_volume = source_arr * areas
    source_strength = source_volume[bank.cell_y, bank.cell_x]
    if source_strength.size == 0:
        return np.zeros_like(source_strength, dtype=np.float64)

    source_total_area = float(np.sum(areas[problem.geometry.source_cells[:, 0], problem.geometry.source_cells[:, 1]]))
    domain_total_area = float(np.sum(areas))
    source_support = problem.geometry.material_map == MAT_SOURCE
    in_source_support = source_support[bank.cell_y, bank.cell_x]
    q_source = np.zeros_like(source_strength, dtype=np.float64)
    q_source[in_source_support] = areas[bank.cell_y[in_source_support], bank.cell_x[in_source_support]] / source_total_area
    q_uniform = areas[bank.cell_y, bank.cell_x] / domain_total_area

    mix_source = bank.source_fraction if bank.source_component_count > 0 else 0.0
    mix_uniform = (1.0 - bank.source_fraction) if bank.uniform_component_count > 0 else 0.0
    mix_total = mix_source + mix_uniform
    if mix_total <= 0.0:
        return np.zeros_like(source_strength, dtype=np.float64)
    q_mix = mix_source * q_source + mix_uniform * q_uniform

    return np.divide(
        source_strength,
        float(source_strength.size) * q_mix,
        out=np.zeros_like(source_strength, dtype=np.float64),
        where=q_mix > 0.0,
    )


def allocate_group_particle_counts(
    problem: MultigroupSteadyLatticeProblem,
    source: np.ndarray,
    num_particles: int,
) -> np.ndarray:
    total_particles = int(num_particles)
    counts = np.zeros(problem.num_groups, dtype=np.int64)
    if total_particles <= 0:
        return counts

    strengths = group_source_strengths(problem, source)
    active = strengths > 0.0
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        return counts

    if total_particles <= active_count:
        top = np.argsort(-strengths)[:total_particles]
        counts[top] = 1
        return counts

    counts[active] = 1
    remaining = total_particles - active_count
    weights = strengths[active]
    weights /= np.sum(weights)
    raw = remaining * weights
    base = np.floor(raw).astype(np.int64)
    counts[np.flatnonzero(active)] += base
    leftover = remaining - int(np.sum(base))
    if leftover > 0:
        frac = raw - base
        order = np.flatnonzero(active)[np.argsort(-frac)]
        counts[order[:leftover]] += 1
    return counts


def _extract_flux_and_meta(result: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(result, dict):
        flux = np.asarray(result["flux"], dtype=np.float64)
        meta = {key: value for key, value in result.items() if key != "flux"}
        return flux, meta
    return np.asarray(result, dtype=np.float64), {}


def iterate_multigroup_steady_lattice(
    problem: MultigroupSteadyLatticeProblem,
    solve_group: GroupSolve,
    *,
    num_particles: int,
    max_outer_iterations: int = 25,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-5,
    seed: int = 1,
    initial_flux: np.ndarray | None = None,
) -> MultigroupIterationResult:
    flux_prev = np.zeros_like(problem.external_source, dtype=np.float64)
    if initial_flux is not None:
        flux_prev = np.asarray(initial_flux, dtype=np.float64).copy()
    if flux_prev.shape != problem.external_source.shape:
        raise ValueError(f"initial_flux must have shape {problem.external_source.shape}; got {flux_prev.shape}.")

    convergence_history: list[float] = []
    source_strength_history: list[np.ndarray] = []
    particle_counts_history: list[np.ndarray] = []
    last_source = np.zeros_like(problem.external_source, dtype=np.float64)
    converged = False

    for _outer_idx in range(int(max_outer_iterations)):
        flux_next = np.zeros_like(flux_prev)
        source_next = problem.external_source + build_lagged_scatter_source(problem, problem.scattering_matrix, flux_prev)
        group_counts = allocate_group_particle_counts(problem, source_next, num_particles)
        source_strength_history.append(group_source_strengths(problem, source_next))
        particle_counts_history.append(group_counts.copy())

        for group in range(problem.num_groups):
            sigma_a_eff, sigma_s_within = group_effective_xs(problem, group)
            source_map = source_next[group]
            group_seed = int(seed) + group
            group_particles = int(group_counts[group])
            if group_particles == 0:
                continue
            solve_result = solve_group(group, source_map, sigma_a_eff, sigma_s_within, group_particles, group_seed)
            group_flux, meta = _extract_flux_and_meta(solve_result)
            if group_flux.shape != source_map.shape:
                raise ValueError(
                    f"group {group} solve returned flux shape {group_flux.shape}; expected {source_map.shape}."
                )
            flux_next[group] = group_flux
            _ = meta

        delta = flux_next - flux_prev
        max_abs = float(np.max(np.abs(delta)))
        flux_scale = float(np.max(np.abs(flux_next)))
        threshold = float(abs_tol) + float(rel_tol) * max(1.0, flux_scale)
        converged = max_abs <= threshold
        convergence_history.append(max_abs / max(1.0, flux_scale))
        flux_prev = flux_next
        last_source = source_next
        if converged:
            break

    return MultigroupIterationResult(
        flux=flux_prev,
        source=last_source,
        convergence_history=np.asarray(convergence_history, dtype=np.float64),
        source_strength_history=np.asarray(source_strength_history, dtype=np.float64),
        particle_counts_history=np.asarray(particle_counts_history, dtype=np.int64),
        converged=converged,
        outer_iterations=len(convergence_history),
    )


def iterate_multigroup_flux(
    problem: SteadyLatticeGeometry | MultigroupSteadyLatticeProblem,
    num_groups: int,
    group_solver: GroupSolve,
    *,
    source_group: int = 0,
    absorption_multipliers: np.ndarray | None = None,
    scatter_matrix: np.ndarray | None = None,
    sigma_s_multiplier: float = 1.0,
    initial_flux: np.ndarray | None = None,
    num_particles: int = 0,
    tolerance: float = 1.0e-5,
    max_iterations: int = 25,
    seed: int = 1,
) -> MultigroupIterationResult:
    mg_problem = (
        problem
        if isinstance(problem, MultigroupSteadyLatticeProblem)
        else build_multigroup_steady_lattice_problem(
            refinement=problem.refinement,
            num_groups=num_groups,
            absorption_multipliers=absorption_multipliers,
            scattering_matrix=scatter_matrix,
            sigma_s_multiplier=sigma_s_multiplier,
        )
    )
    if source_group != 0:
        mg_problem = MultigroupSteadyLatticeProblem(
            geometry=mg_problem.geometry,
            num_groups=mg_problem.num_groups,
            sigma_a_base=mg_problem.sigma_a_base,
            sigma_s_base=mg_problem.sigma_s_base,
            absorption_multipliers=mg_problem.absorption_multipliers,
            scattering_matrix=mg_problem.scattering_matrix,
            external_source=build_group_fixed_source(mg_problem, mg_problem.num_groups, source_group=source_group),
        )
    return iterate_multigroup_steady_lattice(
        mg_problem,
        lambda group, source_map, sigma_a_eff, sigma_s_within, group_particles, group_seed: group_solver(
            group, sigma_s_within, sigma_a_eff, source_map, group_particles, group_seed
        ),
        num_particles=num_particles,
        max_outer_iterations=max_iterations,
        abs_tol=1.0e-8,
        rel_tol=tolerance,
        seed=seed,
        initial_flux=initial_flux,
    )
