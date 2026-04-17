from types import SimpleNamespace

import numpy as np
import pytest

from physics import lattice_multigrid_2d as lmg


def _build_tiny_hierarchy(
    *,
    fine_shape=(2, 2),
    coarse_shape=(1, 1),
    sigma_a=0.0,
    sigma_s=0.0,
    source_mask=None,
):
    fine_shape = tuple(fine_shape)
    sigma_a_map = np.full(fine_shape, sigma_a, dtype=np.float64)
    sigma_s_map = np.full(fine_shape, sigma_s, dtype=np.float64)
    sigma_t_map = sigma_a_map + sigma_s_map
    material_map = np.zeros(fine_shape, dtype=np.int64)
    if source_mask is None:
        source_mask = np.ones(fine_shape, dtype=bool)
    return lmg.build_hierarchy_from_fine_maps(
        total_width=1.0,
        total_height=1.0,
        fine_sigma_t=sigma_t_map,
        fine_sigma_s=sigma_s_map,
        fine_sigma_a=sigma_a_map,
        fine_material_map=material_map,
        fine_source_mask=np.asarray(source_mask, dtype=bool),
        coarse_shape=tuple(coarse_shape),
    )


def test_surface_current_tally_records_and_flux_conversion():
    ordinates = lmg.DiscreteOrdinates2D.uniform(8)
    tally = lmg.SurfaceCurrentTally.zeros(
        num_cells_y=1,
        num_cells_x=1,
        segment_counts=[2, 2, 2, 2],
        segment_lengths=[0.5, 0.5, 0.5, 0.5],
        num_ordinates=ordinates.directions.shape[0],
    )
    ordinate = int(np.flatnonzero(ordinates.incoming_mask[lmg.FACE_LEFT])[0])

    tally.add(0, 0, lmg.FACE_LEFT, 1, ordinate, 0.3)

    records = tally.records_for_cell(0, 0, min_current=0.0)
    assert len(records) == 1
    assert records[0] == lmg.FaceCurrentRecord(
        cell_y=0,
        cell_x=0,
        face=lmg.FACE_LEFT,
        segment=1,
        ordinate=ordinate,
        current=0.3,
        crossings=1,
    )

    angular_flux = tally.boundary_angular_flux(0, 0, ordinates)
    expected = 0.3 / (ordinates.weights[ordinate] * ordinates.incoming_mu[lmg.FACE_LEFT, ordinate] * 0.5)
    assert angular_flux[lmg.FACE_LEFT, 1, ordinate] == pytest.approx(expected)

    outgoing = np.flatnonzero(~ordinates.incoming_mask[lmg.FACE_LEFT])
    assert np.all(angular_flux[lmg.FACE_LEFT, :, outgoing] == 0.0)


def test_coarse_surface_binned_transport_single_cell_conserves_to_leakage():
    hierarchy = _build_tiny_hierarchy(fine_shape=(1, 1), coarse_shape=(1, 1), sigma_a=0.0, sigma_s=0.0)
    ordinates = lmg.DiscreteOrdinates2D.uniform(8)

    result = lmg.run_coarse_surface_binned_transport(
        hierarchy,
        num_histories=32,
        ordinates=ordinates,
        seed=11,
        max_events_per_history=4,
    )

    assert result.source_histories == 32
    assert np.sum(result.energy_deposition) == pytest.approx(0.0)
    assert np.sum(result.leakage_weight) == pytest.approx(1.0)
    assert result.surface_tally.active_bin_count() == 0
    assert np.all(result.surface_tally.counts == 0)
    assert np.all(result.surface_tally.current == 0.0)


def test_run_multigrid_refinement_preserves_coarse_track_anchor():
    hierarchy = _build_tiny_hierarchy(fine_shape=(2, 2), coarse_shape=(1, 1), sigma_a=0.1, sigma_s=0.4)
    ordinates = lmg.DiscreteOrdinates2D.uniform(8)

    result = lmg.run_multigrid_refinement(
        hierarchy,
        num_histories=24,
        ordinates=ordinates,
        seed=7,
        max_coarse_events=16,
        sn_max_iterations=6,
        sn_tolerance=1.0e-6,
    )

    coarse_flux, coarse_track = lmg.coarsen_fine_scalar_flux(hierarchy, result.fine_flux)
    assert coarse_flux.shape == hierarchy.coarse_sigma_t.shape
    assert np.sum(result.fine_track_length) == pytest.approx(np.sum(result.coarse_result.track_length))
    assert coarse_track[0, 0] == pytest.approx(result.coarse_result.track_length[0, 0])
    assert result.local_iterations[0, 0] >= 1
    assert result.local_scale_factors[0, 0] >= 0.0
    assert np.all(np.isfinite(result.fine_flux))
