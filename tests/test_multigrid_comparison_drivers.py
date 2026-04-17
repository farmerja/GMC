from pathlib import Path
from types import SimpleNamespace

import numpy as np

from physics import lattice_multigrid_2d as lmg
from physics import multigrid_steady_2d as legacy_mg
from validation import complete_comparison
from validation import lattice_steady_multigrid as steady_driver


def _legacy_rect_problem(num_cells_x, num_cells_y, *, source_cells):
    cell_widths = np.full(num_cells_x, 1.0, dtype=np.float64)
    cell_heights = np.full(num_cells_y, 1.0, dtype=np.float64)
    cell_boundaries_x = np.arange(num_cells_x + 1, dtype=np.float64)
    cell_boundaries_y = np.arange(num_cells_y + 1, dtype=np.float64)
    shape = (num_cells_y, num_cells_x)
    return legacy_mg.RectLatticeProblem(
        material_map=np.zeros(shape, dtype=np.int64),
        sigma_a_map=np.zeros(shape, dtype=np.float64),
        sigma_s_map=np.zeros(shape, dtype=np.float64),
        sigma_t_map=np.zeros(shape, dtype=np.float64),
        cell_widths=cell_widths,
        cell_heights=cell_heights,
        cell_boundaries_x=cell_boundaries_x,
        cell_boundaries_y=cell_boundaries_y,
        cell_areas=np.outer(cell_heights, cell_widths),
        source_cells=np.asarray(source_cells, dtype=np.int64),
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        refinement=1,
        domain_size=float(num_cells_x),
    )


def test_complete_comparison_mc_surface_path_records_true_crossings(monkeypatch):
    coarse = _legacy_rect_problem(2, 1, source_cells=[[0, 0]])
    fine = _legacy_rect_problem(4, 1, source_cells=[[0, 0], [0, 1]])
    hierarchy = legacy_mg.MultigridHierarchy(coarse=coarse, fine=fine, fine_per_coarse_x=2, fine_per_coarse_y=1)
    quadrature = legacy_mg.build_uniform_angular_quadrature(8)

    monkeypatch.setattr(legacy_mg, "_sample_isotropic_direction_2d", lambda _rng: (1.0, 0.0))

    result = complete_comparison.run_coarse_mc_with_surface_tally(
        coarse,
        hierarchy,
        quadrature,
        num_particles=4,
        randomize_start_pos=False,
        seed=5,
        max_events=4,
    )

    tally = result["surface_tally"]
    assert np.sum(tally.crossing_weight) == 1.0
    assert np.max(tally.angular_flux) > 0.0
    assert int(np.sum(complete_comparison._surface_current_map(tally) > 0.0)) == 1


def test_area_weighted_relative_metrics_match_hand_calculation():
    estimate = np.array([[2.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    areas = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    rel_l1 = steady_driver.area_weighted_relative_l1(estimate, reference, areas)
    rel_l2 = steady_driver.area_weighted_relative_l2(estimate, reference, areas)

    expected_l1 = (1.0 * 1.0 + 0.0 * 2.0 + 0.0 * 3.0 + 4.0 * 4.0) / (
        1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0
    )
    expected_l2 = np.sqrt((1.0**2 * 1.0 + 0.0 + 0.0 + 4.0**2 * 4.0) / (1.0**2 * 1.0 + 2.0**2 * 2.0 + 3.0**2 * 3.0 + 4.0**2 * 4.0))

    assert np.isclose(rel_l1, expected_l1)
    assert np.isclose(rel_l2, expected_l2)


def test_lattice_steady_multigrid_driver_saves_surface_diagnostics(tmp_path, monkeypatch):
    def fake_run_multigrid_refinement(hierarchy, *, ordinates, **_kwargs):
        fine_flux = np.full(hierarchy.fine_sigma_t.shape, 2.0, dtype=np.float64)
        fine_track_length = fine_flux * hierarchy.fine_mesh.areas
        coarse_flux, coarse_track = lmg.coarsen_fine_scalar_flux(hierarchy, fine_flux)
        surface_tally = lmg.SurfaceCurrentTally.zeros(
            num_cells_y=hierarchy.coarse_sigma_t.shape[0],
            num_cells_x=hierarchy.coarse_sigma_t.shape[1],
            segment_counts=hierarchy.face_segment_counts,
            segment_lengths=hierarchy.face_segment_lengths,
            num_ordinates=ordinates.directions.shape[0],
        )
        incoming_ordinate = int(np.flatnonzero(ordinates.incoming_mask[lmg.FACE_LEFT])[0])
        surface_tally.add(0, 0, lmg.FACE_LEFT, 0, incoming_ordinate, 0.25)
        coarse_result = SimpleNamespace(
            track_length=coarse_track,
            flux=coarse_flux,
            energy_deposition=np.zeros_like(coarse_track),
            surface_tally=surface_tally,
            leakage_weight=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            mean_events=3.5,
            elapsed_sec=0.01,
        )
        return SimpleNamespace(
            fine_flux=fine_flux,
            fine_track_length=fine_track_length,
            coarse_result=coarse_result,
            local_iterations=np.ones_like(coarse_track, dtype=np.int64),
            local_scale_factors=np.ones_like(coarse_track, dtype=np.float64),
            elapsed_sec=0.02,
        )

    def fake_reference_mc(*, sigma_t, **_kwargs):
        flux = np.full_like(sigma_t, 1.5, dtype=np.float64)
        cell_area = (7.0 / sigma_t.shape[0]) * (7.0 / sigma_t.shape[1])
        track = flux * cell_area
        energy = np.zeros_like(sigma_t, dtype=np.float64)
        return flux, track, energy, 0.03

    monkeypatch.setattr(steady_driver, "run_multigrid_refinement", fake_run_multigrid_refinement)
    monkeypatch.setattr(steady_driver, "run_internal_source_reference_mc", fake_reference_mc)
    monkeypatch.setattr(
        steady_driver,
        "parse_args",
        lambda: SimpleNamespace(
            coarse_refinement=1,
            fine_refinement=2,
            coarse_histories=10,
            reference_histories=12,
            num_angles=8,
            sn_iterations=4,
            sn_tolerance=1.0e-5,
            max_coarse_events=20,
            seed=123,
            output=Path(tmp_path) / "steady_multigrid_test.npz",
        ),
    )

    steady_driver.main()
    output = np.load(Path(tmp_path) / "steady_multigrid_test.npz")

    for key in (
        "surface_current",
        "surface_counts",
        "local_iterations",
        "local_scale_factors",
        "rel_l1",
        "rel_l2",
        "coarse_rel_l1",
    ):
        assert key in output.files

    assert output["surface_current"].ndim == 5
    assert output["surface_counts"].shape == output["surface_current"].shape
    assert output["local_iterations"].ndim == 2
