from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.multigroup_steady_lattice import (
    MixedSourceParticleBank,
    MultigroupSteadyLatticeProblem,
    SteadyLatticeGeometry,
    allocate_group_particle_counts,
    build_mixed_source_particle_bank,
    build_multigroup_steady_lattice_problem,
    default_scattering_matrix,
    group_effective_xs,
    iterate_multigroup_steady_lattice,
    mixed_source_particle_weights,
)


def make_one_cell_problem() -> MultigroupSteadyLatticeProblem:
    geometry = SteadyLatticeGeometry(
        material_map=np.array([[2]], dtype=np.int64),
        source_cells=np.array([[0, 0]], dtype=np.int64),
        cell_widths=np.array([1.0]),
        cell_heights=np.array([1.0]),
        cell_boundaries_x=np.array([0.0, 1.0]),
        cell_boundaries_y=np.array([0.0, 1.0]),
        cell_areas=np.array([[1.0]]),
        num_cells_x=1,
        num_cells_y=1,
        refinement=1,
        domain_size=1.0,
    )
    return MultigroupSteadyLatticeProblem(
        geometry=geometry,
        num_groups=2,
        sigma_a_base=np.array([[1.5]], dtype=np.float64),
        sigma_s_base=np.array([[2.0]], dtype=np.float64),
        absorption_multipliers=np.array([1.0, 2.0], dtype=np.float64),
        scattering_matrix=np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64),
        external_source=np.array([[[1.0]], [[0.0]]], dtype=np.float64),
    )


def test_build_multigroup_problem_uses_56x56_benchmark_defaults():
    problem = build_multigroup_steady_lattice_problem()

    assert problem.num_groups == 7
    assert problem.geometry.num_cells_x == 56
    assert problem.geometry.num_cells_y == 56
    assert problem.geometry.source_cells.shape == (64, 2)
    assert np.isclose(problem.external_source[0].sum(), 1.0)
    assert np.isclose(problem.external_source[1:].sum(), 0.0)
    assert np.allclose(problem.scattering_matrix.sum(axis=1), 1.0)
    assert np.array_equal(problem.absorption_multipliers, np.ones(7))


def test_build_multigroup_problem_scales_base_scattering_cross_sections():
    problem = build_multigroup_steady_lattice_problem(sigma_s_multiplier=10.0)

    sigma_s = problem.sigma_s_base
    material = problem.geometry.material_map

    assert np.allclose(sigma_s[material == 0], 10.0)
    assert np.allclose(sigma_s[material == 1], 5.0)
    assert np.allclose(sigma_s[material == 2], 10.0)


def test_group_effective_xs_keeps_within_group_scatter_implicit():
    problem = build_multigroup_steady_lattice_problem(
        absorption_multipliers=np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]),
        scattering_matrix=default_scattering_matrix(),
    )

    sigma_a_eff, sigma_s_within = group_effective_xs(problem, 3)
    s_gg = problem.scattering_matrix[3, 3]

    assert np.allclose(sigma_s_within, problem.sigma_s_base * s_gg)
    assert np.allclose(
        sigma_a_eff,
        problem.sigma_a_base * problem.absorption_multipliers[3] + problem.sigma_s_base * (1.0 - s_gg),
    )


def test_iteration_lags_out_of_group_scatter_and_uses_repeatable_group_seeds():
    problem = make_one_cell_problem()
    calls = []

    def solver(group, source_map, sigma_a_effective, sigma_s_within, num_particles, seed):
        calls.append(
            {
                "group": group,
                "source": source_map.copy(),
                "sigma_a": sigma_a_effective.copy(),
                "sigma_s": sigma_s_within.copy(),
                "num_particles": num_particles,
                "seed": seed,
            }
        )
        return {"flux": source_map.copy()}

    result = iterate_multigroup_steady_lattice(
        problem,
        solver,
        num_particles=4,
        max_outer_iterations=2,
        abs_tol=0.0,
        rel_tol=0.0,
        seed=11,
    )

    assert result.outer_iterations == 2
    assert np.allclose(calls[0]["source"], [[1.0]])
    assert np.allclose(calls[1]["source"], [[1.0]])
    assert np.allclose(calls[2]["source"], [[0.8]])
    assert [call["group"] for call in calls] == [0, 0, 1]
    assert [call["seed"] for call in calls] == [11, 11, 12]
    assert [call["num_particles"] for call in calls] == [4, 2, 2]
    assert np.allclose(calls[0]["sigma_s"], [[1.2]])
    assert np.allclose(calls[2]["sigma_s"], [[1.4]])
    assert np.allclose(result.convergence_history, [1.0, 0.8])
    assert result.particle_counts_history.tolist() == [[4, 0], [2, 2]]


def test_iteration_is_deterministic_for_a_fixed_outer_seed_schedule():
    problem = make_one_cell_problem()

    def noisy_solver(group, source_map, sigma_a_effective, sigma_s_within, num_particles, seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0e-4, size=source_map.shape)
        flux = source_map / (1.0 + sigma_a_effective) + noise
        return {"flux": flux}

    result_a = iterate_multigroup_steady_lattice(
        problem,
        noisy_solver,
        num_particles=5,
        max_outer_iterations=3,
        seed=5,
    )
    result_b = iterate_multigroup_steady_lattice(
        problem,
        noisy_solver,
        num_particles=5,
        max_outer_iterations=3,
        seed=5,
    )
    result_c = iterate_multigroup_steady_lattice(
        problem,
        noisy_solver,
        num_particles=5,
        max_outer_iterations=3,
        seed=6,
    )

    assert np.array_equal(result_a.flux, result_b.flux)
    assert np.array_equal(result_a.convergence_history, result_b.convergence_history)
    assert np.array_equal(result_a.particle_counts_history, result_b.particle_counts_history)
    assert not np.array_equal(result_a.flux, result_c.flux)


def test_allocate_group_particle_counts_uses_total_budget_once():
    problem = make_one_cell_problem()
    source = np.array([[[3.0]], [[1.0]]], dtype=np.float64)
    counts = allocate_group_particle_counts(problem, source, 5)
    assert counts.tolist() == [3, 2]


def test_mixed_source_particle_bank_is_prefix_stable_for_fixed_seed():
    problem = build_multigroup_steady_lattice_problem()
    bank_small = build_mixed_source_particle_bank(problem, 16, seed=9)
    bank_large = build_mixed_source_particle_bank(problem, 32, seed=9)

    assert np.array_equal(bank_small.component, bank_large.component[:16])
    assert np.array_equal(bank_small.cell_x, bank_large.cell_x[:16])
    assert np.array_equal(bank_small.cell_y, bank_large.cell_y[:16])
    assert np.allclose(bank_small.pos_rx, bank_large.pos_rx[:16])
    assert np.allclose(bank_small.pos_ry, bank_large.pos_ry[:16])
    assert np.allclose(bank_small.dir_unit_mu, bank_large.dir_unit_mu[:16])
    assert np.allclose(bank_small.dir_unit_phi, bank_large.dir_unit_phi[:16])
    assert np.array_equal(bank_small.trajectory_seeds, bank_large.trajectory_seeds[:16])


def test_mixed_source_particle_weights_match_total_source_for_one_cell_stratification():
    problem = make_one_cell_problem()
    bank = build_mixed_source_particle_bank(problem, 6, seed=4)
    weights = mixed_source_particle_weights(problem, np.array([[1.0]], dtype=np.float64), bank)

    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights > 0.0)


def test_mixed_source_particle_weights_use_full_mixture_density_outside_source_block():
    geometry = SteadyLatticeGeometry(
        material_map=np.array([[2, 0]], dtype=np.int64),
        source_cells=np.array([[0, 0]], dtype=np.int64),
        cell_widths=np.array([1.0, 1.0]),
        cell_heights=np.array([1.0]),
        cell_boundaries_x=np.array([0.0, 1.0, 2.0]),
        cell_boundaries_y=np.array([0.0, 1.0]),
        cell_areas=np.array([[1.0, 1.0]]),
        num_cells_x=2,
        num_cells_y=1,
        refinement=1,
        domain_size=2.0,
    )
    problem = MultigroupSteadyLatticeProblem(
        geometry=geometry,
        num_groups=1,
        sigma_a_base=np.zeros((1, 2), dtype=np.float64),
        sigma_s_base=np.ones((1, 2), dtype=np.float64),
        absorption_multipliers=np.ones(1, dtype=np.float64),
        scattering_matrix=np.ones((1, 1), dtype=np.float64),
        external_source=np.zeros((1, 1, 2), dtype=np.float64),
    )
    bank = MixedSourceParticleBank(
        cell_y=np.array([0, 0], dtype=np.int64),
        cell_x=np.array([0, 1], dtype=np.int64),
        pos_rx=np.array([0.25, 0.75], dtype=np.float64),
        pos_ry=np.array([0.5, 0.5], dtype=np.float64),
        dir_unit_mu=np.array([0.1, 0.2], dtype=np.float64),
        dir_unit_phi=np.array([0.3, 0.4], dtype=np.float64),
        trajectory_seeds=np.array([1, 2], dtype=np.int64),
        component=np.array([0, 1], dtype=np.int64),
        source_component_count=1,
        uniform_component_count=1,
        source_fraction=0.5,
    )
    source_map = np.array([[0.0, 1.0]], dtype=np.float64)

    weights = mixed_source_particle_weights(problem, source_map, bank)

    assert np.allclose(weights, [0.0, 2.0])
