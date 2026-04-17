from pathlib import Path

import numpy as np

from physics.mc2d import calculate_boundaries_2d, run_mc_simulation_2d_njit_wrapper
from validation import lattice_multigrid_steady as multigrid_driver


def test_mc2d_wrapper_smoke_accounting():
    widths = np.array([0.5, 0.5], dtype=np.float64)
    heights = np.array([0.5, 0.5], dtype=np.float64)
    bx, by = calculate_boundaries_2d(widths, heights)
    energy, track, stats = run_mc_simulation_2d_njit_wrapper(
        source_mode="internal_start",
        num_cells_x=2,
        num_cells_y=2,
        cell_widths=widths,
        cell_heights=heights,
        cell_boundaries_x=bx,
        cell_boundaries_y=by,
        total_width=1.0,
        total_height=1.0,
        num_particles=20,
        start_cell_indices=[0, 0],
        start_position_relative=[0.5, 0.5],
        scatter_ratio=0.8,
        energy_threshold=1.0e-8,
        seed=123,
        reflected_boundaries=False,
        verbose=False,
    )
    assert energy.shape == (2, 2)
    assert track.shape == (2, 2)
    assert sum(stats[:5]) >= 20


def test_multigrid_driver_main_smoke_with_stubs(tmp_path, monkeypatch):
    class DummyHierarchy:
        def __init__(self):
            self.coarse = type(
                "Coarse",
                (),
                {
                    "num_cells_x": 2,
                    "num_cells_y": 2,
                    "cell_boundaries_x": np.array([0.0, 0.5, 1.0]),
                    "cell_boundaries_y": np.array([0.0, 0.5, 1.0]),
                },
            )()
            self.fine = type(
                "Fine",
                (),
                {
                    "num_cells_x": 4,
                    "num_cells_y": 4,
                    "cell_boundaries_x": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                    "cell_boundaries_y": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                },
            )()
            self.fine_per_coarse_x = 2
            self.fine_per_coarse_y = 2

    class DummyQuadrature:
        theta_centers = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)

    class DummyTally:
        crossing_weight = np.zeros((2, 2, 4, 2, 4))
        angular_flux = np.zeros((2, 2, 4, 2, 4))

    monkeypatch.setattr(multigrid_driver, "build_multigrid_hierarchy", lambda **_: DummyHierarchy())
    monkeypatch.setattr(multigrid_driver, "build_uniform_angular_quadrature", lambda *_args, **_kwargs: DummyQuadrature())
    monkeypatch.setattr(multigrid_driver, "load_steady_gmc_models", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        multigrid_driver,
        "run_coarse_gmc_with_surface_tally",
        lambda *_args, **_kwargs: {"surface_tally": DummyTally(), "coarse_flux": np.ones((2, 2))},
    )
    monkeypatch.setattr(
        multigrid_driver,
        "reconstruct_fine_flux_from_surface_tally",
        lambda *_args, **_kwargs: {
            "fine_flux": np.ones((4, 4)),
            "local_balance": np.zeros((2, 2)),
            "incoming_current": np.zeros((2, 2)),
            "outgoing_current": np.zeros((2, 2)),
        },
    )
    monkeypatch.setattr(multigrid_driver, "run_direct_fine_mc_reference", lambda *_args, **_kwargs: {"flux": np.ones((4, 4))})
    monkeypatch.setattr(
        multigrid_driver,
        "compute_flux_error_metrics",
        lambda *_args, **_kwargs: {"l1": 0.0, "l2": 0.0, "rel_l2": 0.0, "max_abs": 0.0},
    )
    monkeypatch.setattr(
        multigrid_driver,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "coarse_refinement": 2,
                "fine_refinement": 8,
                "coarse_particles": 10,
                "fine_mc_particles": 10,
                "num_angles": 4,
                "sn_iters": 1,
                "device": "cpu",
                "seed": 1,
                "output_dir": str(tmp_path),
            },
        )(),
    )

    multigrid_driver.main()
    assert (Path(tmp_path) / "lattice_multigrid_steady_results.npz").exists()
