import numpy as np
import pytest

from physics import multigrid_steady_2d as mg
from physics.steady_2d_scattering import guard_projected_direction_xy


def test_anisotropic_scatter_sampler_biases_forward_and_preserves_norm():
    rng = np.random.default_rng(1234)
    samples = np.array(
        [mg._sample_scattered_direction_3d(rng, (1.0, 0.0, 0.0), scattering_mode="anisotropic") for _ in range(4000)],
        dtype=np.float64,
    )

    assert np.allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1.0e-10)
    assert float(np.mean(samples[:, 0])) > 0.28
    assert abs(float(np.mean(samples[:, 1]))) < 0.03
    assert abs(float(np.mean(samples[:, 2]))) < 0.03


def test_build_multigrid_hierarchy_preserves_nested_material_blocks():
    hierarchy = mg.build_multigrid_hierarchy(coarse_refinement=2, fine_refinement=4)

    assert hierarchy.coarse.material_map.shape == (14, 14)
    assert hierarchy.fine.material_map.shape == (28, 28)
    assert hierarchy.fine_per_coarse_x == 2
    assert hierarchy.fine_per_coarse_y == 2

    for coarse_iy in range(hierarchy.coarse.num_cells_y):
        for coarse_ix in range(hierarchy.coarse.num_cells_x):
            ys = slice(coarse_iy * hierarchy.fine_per_coarse_y, (coarse_iy + 1) * hierarchy.fine_per_coarse_y)
            xs = slice(coarse_ix * hierarchy.fine_per_coarse_x, (coarse_ix + 1) * hierarchy.fine_per_coarse_x)
            fine_block = hierarchy.fine.material_map[ys, xs]
            assert np.all(fine_block == hierarchy.coarse.material_map[coarse_iy, coarse_ix])


def test_segment_index_and_surface_tally_account_for_synthetic_crossings():
    hierarchy = mg.build_multigrid_hierarchy(coarse_refinement=2, fine_refinement=4)
    quadrature = mg.build_uniform_angular_quadrature(16)
    tally = mg.initialize_surface_tallies(hierarchy, quadrature)

    coarse_height = hierarchy.coarse.cell_heights[0]
    coarse_width = hierarchy.coarse.cell_widths[0]
    segment_idx = mg._segment_index_for_face(
        np.array([mg.FACE_LEFT, mg.FACE_LEFT, mg.FACE_TOP, mg.FACE_TOP], dtype=np.int64),
        np.array([0.0, coarse_height - 1.0e-12, 0.0, coarse_width - 1.0e-12], dtype=np.float64),
        hierarchy,
    )
    assert segment_idx.tolist() == [0, 1, 0, 1]

    weights = np.array([0.6, 0.4], dtype=np.float64)
    u = np.array([1.0, 1.0], dtype=np.float64)
    v = np.array([0.0, 0.0], dtype=np.float64)
    pos_on_face = np.array([0.1, coarse_height - 1.0e-12], dtype=np.float64)
    angle_idx = int(mg._projection_to_angle_index(np.arctan2(v, u), quadrature)[0])

    mg.record_surface_crossings(
        tally,
        hierarchy,
        quadrature,
        target_ix=np.array([0, 0], dtype=np.int64),
        target_iy=np.array([0, 0], dtype=np.int64),
        target_face=np.array([mg.FACE_LEFT, mg.FACE_LEFT], dtype=np.int64),
        pos_on_face=pos_on_face,
        u=u,
        v=v,
        weight=weights,
        num_histories=10,
    )

    assert tally.crossing_weight[0, 0, mg.FACE_LEFT, 0, angle_idx] == pytest.approx(0.06)
    assert tally.crossing_weight[0, 0, mg.FACE_LEFT, 1, angle_idx] == pytest.approx(0.04)
    assert np.sum(tally.crossing_weight) == pytest.approx(0.10)

    mg.finalize_surface_angular_flux(tally, quadrature)

    segment_length = tally.segment_lengths[0, 0, mg.FACE_LEFT]
    measure = quadrature.face_bin_measure[mg.FACE_LEFT, angle_idx]
    assert measure > 0.0
    assert tally.angular_flux[0, 0, mg.FACE_LEFT, 0, angle_idx] == pytest.approx(0.06 / (segment_length * measure))
    assert tally.angular_flux[0, 0, mg.FACE_LEFT, 1, angle_idx] == pytest.approx(0.04 / (segment_length * measure))

    outgoing_bins = ~quadrature.face_incoming_mask[mg.FACE_LEFT]
    assert np.all(tally.angular_flux[0, 0, mg.FACE_LEFT, :, outgoing_bins] == 0.0)


def test_anisotropic_scatter_sampler_is_forward_biased_and_zero_guarded():
    rng = np.random.default_rng(123)
    current = np.array([0.35, -0.45, np.sqrt(1.0 - 0.35**2 - 0.45**2)], dtype=np.float64)
    samples = np.array(
        [mg._sample_scattered_direction_3d(rng, tuple(current), scattering_mode="anisotropic") for _ in range(4000)],
        dtype=np.float64,
    )

    dots = samples @ current
    norms = np.linalg.norm(samples, axis=1)

    assert np.allclose(norms, 1.0, atol=1.0e-10)
    assert np.mean(dots) == pytest.approx(1.0 / 3.0, abs=0.04)

    u_g, v_g = guard_projected_direction_xy(0.0, 0.0, 1.0e-6)
    assert u_g == pytest.approx(1.0e-6)
    assert v_g == pytest.approx(0.0)


def test_solve_local_sn_tiny_homogeneous_case_is_stable_and_symmetric():
    quadrature = mg.build_uniform_angular_quadrature(8)
    incoming = np.zeros((4, 2, quadrature.theta_centers.size), dtype=np.float64)

    zero_solution = mg.solve_local_sn(
        incoming,
        quadrature,
        width=1.0,
        height=1.0,
        nx=2,
        ny=2,
        sigma_t=1.0,
        sigma_s=0.5,
        fixed_source_strength=0.0,
        num_source_iterations=4,
    )
    assert np.allclose(zero_solution.scalar_flux, 0.0)
    assert zero_solution.balance_residual == pytest.approx(0.0)

    driven_solution = mg.solve_local_sn(
        incoming,
        quadrature,
        width=1.0,
        height=1.0,
        nx=2,
        ny=2,
        sigma_t=1.0,
        sigma_s=0.5,
        fixed_source_strength=1.0,
        num_source_iterations=6,
    )

    assert driven_solution.scalar_flux.shape == (2, 2)
    assert np.isfinite(driven_solution.scalar_flux).all()
    assert np.all(driven_solution.scalar_flux >= 0.0)
    assert np.allclose(driven_solution.scalar_flux, driven_solution.scalar_flux[0, 0])
    assert driven_solution.source_total == pytest.approx(1.0)
    assert driven_solution.absorbed_total > 0.0
    assert np.isfinite(driven_solution.balance_residual)


def test_reconstruct_fine_flux_from_surface_tally_smoke():
    hierarchy = mg.build_multigrid_hierarchy(coarse_refinement=2, fine_refinement=4)
    quadrature = mg.build_uniform_angular_quadrature(8)
    tally = mg.initialize_surface_tallies(hierarchy, quadrature)
    mg.finalize_surface_angular_flux(tally, quadrature)

    result = mg.reconstruct_fine_flux_from_surface_tally(
        hierarchy,
        tally,
        quadrature,
        num_source_iterations=2,
    )

    fine_flux = result["fine_flux"]
    local_balance = result["local_balance"]

    assert fine_flux.shape == hierarchy.fine.material_map.shape
    assert local_balance.shape == hierarchy.coarse.material_map.shape
    assert np.isfinite(fine_flux).all()
    assert np.isfinite(local_balance).all()
    assert np.all(fine_flux >= 0.0)

    coarse_block_sum = np.zeros_like(hierarchy.coarse.material_map, dtype=np.float64)
    for coarse_iy in range(hierarchy.coarse.num_cells_y):
        for coarse_ix in range(hierarchy.coarse.num_cells_x):
            ys = slice(coarse_iy * hierarchy.fine_per_coarse_y, (coarse_iy + 1) * hierarchy.fine_per_coarse_y)
            xs = slice(coarse_ix * hierarchy.fine_per_coarse_x, (coarse_ix + 1) * hierarchy.fine_per_coarse_x)
            coarse_block_sum[coarse_iy, coarse_ix] = np.sum(fine_flux[ys, xs])

    source_cells = hierarchy.coarse.material_map == mg.REGION_SOURCE
    assert np.all(coarse_block_sum[source_cells] > 0.0)
    assert np.allclose(coarse_block_sum[~source_cells], 0.0)


def test_coarse_mc_surface_tally_uses_target_cell_local_face_coordinates(monkeypatch):
    base_hierarchy = mg.build_multigrid_hierarchy(coarse_refinement=1, fine_refinement=4)
    coarse = base_hierarchy.coarse
    problem = mg.RectLatticeProblem(
        material_map=coarse.material_map.copy(),
        sigma_a_map=np.zeros_like(coarse.sigma_a_map),
        sigma_s_map=np.zeros_like(coarse.sigma_s_map),
        sigma_t_map=np.zeros_like(coarse.sigma_t_map),
        cell_widths=coarse.cell_widths.copy(),
        cell_heights=coarse.cell_heights.copy(),
        cell_boundaries_x=coarse.cell_boundaries_x.copy(),
        cell_boundaries_y=coarse.cell_boundaries_y.copy(),
        cell_areas=coarse.cell_areas.copy(),
        source_cells=np.array([[3, 3]], dtype=np.int64),
        num_cells_x=coarse.num_cells_x,
        num_cells_y=coarse.num_cells_y,
        refinement=coarse.refinement,
        domain_size=coarse.domain_size,
    )
    hierarchy = mg.MultigridHierarchy(
        coarse=problem,
        fine=base_hierarchy.fine,
        fine_per_coarse_x=base_hierarchy.fine_per_coarse_x,
        fine_per_coarse_y=base_hierarchy.fine_per_coarse_y,
    )
    quadrature = mg.build_uniform_angular_quadrature(8)

    monkeypatch.setattr(mg, "_sample_isotropic_direction_2d", lambda _rng: (1.0, 0.0))

    result = mg.run_coarse_mc_with_surface_tally(
        problem,
        hierarchy,
        quadrature,
        num_particles=1,
        randomize_start_pos=False,
        seed=7,
        max_events=1,
    )

    angle_idx = int(mg._projection_to_angle_index(np.array([0.0], dtype=np.float64), quadrature)[0])
    target_cell_y = 3
    target_cell_x = 4
    expected_segment = 2
    wrong_last_segment = hierarchy.fine_per_coarse_y - 1

    assert result["surface_tally"].crossing_weight[target_cell_y, target_cell_x, mg.FACE_LEFT, expected_segment, angle_idx] == pytest.approx(1.0)
    assert np.sum(result["surface_tally"].crossing_weight[target_cell_y, target_cell_x, mg.FACE_LEFT, wrong_last_segment, :]) == pytest.approx(0.0)
    assert np.sum(result["surface_tally"].crossing_weight) == pytest.approx(1.0)


def test_run_direct_fine_mc_reference_forwards_scattering_mode(monkeypatch):
    hierarchy = mg.build_multigrid_hierarchy(coarse_refinement=1, fine_refinement=2)
    calls = []

    def fake_wrapper(**kwargs):
        calls.append(kwargs["scattering_mode"])
        shape = kwargs["sigma_t_per_cell"].shape
        return np.zeros(shape), np.zeros(shape), (0, 0, 0, 0, kwargs["num_particles"], 0, 0, 0)

    monkeypatch.setattr(mg, "run_mc_simulation_2d_njit_wrapper", fake_wrapper)

    result = mg.run_direct_fine_mc_reference(
        hierarchy.fine,
        num_particles=5,
        seed=11,
        scattering_mode="anisotropic",
    )

    assert calls
    assert set(calls) == {"anisotropic"}
    assert result["flux"].shape == hierarchy.fine.material_map.shape
