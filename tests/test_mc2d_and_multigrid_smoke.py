from pathlib import Path

import numpy as np

from physics import multigrid_steady_2d as mg
from physics.mc2d import calculate_boundaries_2d, run_mc_simulation_2d_njit_wrapper
from validation import lattice_anisotropic_gmc_56x56 as anisotropic_poc_driver
from validation import lattice_multigrid_steady as multigrid_driver
from validation import steady2d_distilled_lattice as distilled_driver


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


def test_mc2d_wrapper_smoke_accounting_anisotropic():
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
        seed=321,
        reflected_boundaries=False,
        verbose=False,
        scattering_mode="anisotropic",
    )
    assert energy.shape == (2, 2)
    assert track.shape == (2, 2)
    assert np.isfinite(energy).all()
    assert np.isfinite(track).all()
    assert sum(stats[:5]) >= 20


def test_multigrid_driver_main_smoke_with_stubs(tmp_path, monkeypatch):
    seen = {"model_mode": None, "mc_mode": None}

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
    def fake_load_models(*_args, **kwargs):
        seen["model_mode"] = kwargs.get("scattering_mode")
        return object()

    monkeypatch.setattr(multigrid_driver, "load_steady_gmc_models", fake_load_models)
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
    def fake_run_direct_fine_mc_reference(*_args, **kwargs):
        seen["mc_mode"] = kwargs.get("scattering_mode")
        return {"flux": np.ones((4, 4))}

    monkeypatch.setattr(multigrid_driver, "run_direct_fine_mc_reference", fake_run_direct_fine_mc_reference)
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
                "scattering_mode": "anisotropic",
                "seed": 1,
                "output_dir": str(tmp_path),
            },
        )(),
    )

    multigrid_driver.main()
    output_path = Path(tmp_path) / "lattice_multigrid_steady_results.npz"
    assert output_path.exists()
    payload = np.load(output_path)
    assert not bool(payload["sn_compatible"])
    assert seen["model_mode"] == "anisotropic"
    assert seen["mc_mode"] == "anisotropic"


def test_anisotropic_56x56_driver_main_smoke_with_stubs(tmp_path, monkeypatch):
    seen = {"model_mode": None, "train_sampling": None, "mc_mode": None}
    problem = mg.build_rect_lattice_problem(refinement=1)
    hierarchy = mg.make_single_level_hierarchy(problem)

    class DummyQuadrature:
        theta_centers = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)

    class DummyTally:
        crossing_weight = np.zeros((problem.num_cells_y, problem.num_cells_x, 4, 1, 4))
        angular_flux = np.zeros((problem.num_cells_y, problem.num_cells_x, 4, 1, 4))

    monkeypatch.setattr(anisotropic_poc_driver, "build_multigrid_hierarchy", lambda **_: hierarchy)
    monkeypatch.setattr(anisotropic_poc_driver, "build_uniform_angular_quadrature", lambda *_args, **_kwargs: DummyQuadrature())

    def fake_load_models(*_args, **kwargs):
        seen["model_mode"] = kwargs.get("scattering_mode")
        seen["train_sampling"] = kwargs.get("train_sampling")
        return object()

    monkeypatch.setattr(anisotropic_poc_driver, "load_steady_gmc_models", fake_load_models)
    monkeypatch.setattr(
        anisotropic_poc_driver,
        "run_coarse_gmc_with_surface_tally",
        lambda *_args, **_kwargs: {
            "surface_tally": DummyTally(),
            "coarse_flux": np.ones_like(problem.sigma_t_map, dtype=np.float64),
            "totals": {"left": 1, "right": 2, "top": 3, "bottom": 4},
        },
    )

    def fake_run_direct_fine_mc_reference(*_args, **kwargs):
        seen["mc_mode"] = kwargs.get("scattering_mode")
        return {
            "flux": np.full_like(problem.sigma_t_map, 2.0, dtype=np.float64),
            "stats": (5, 6, 7, 8, 9, 0, 0, 0),
        }

    monkeypatch.setattr(anisotropic_poc_driver, "run_direct_fine_mc_reference", fake_run_direct_fine_mc_reference)
    monkeypatch.setattr(
        anisotropic_poc_driver,
        "compute_flux_error_metrics",
        lambda *_args, **_kwargs: {"l1": 1.0, "l2": 2.0, "rel_l2": 3.0, "max_abs": 4.0},
    )
    monkeypatch.setattr(
        anisotropic_poc_driver,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "gmc_particles": 10,
                "mc_particles": 10,
                "num_angles": 4,
                "sigma_s_scale": 1.5,
                "device": "cpu",
                "train_sampling": "lhs",
                "seed": 1,
                "output_dir": str(tmp_path),
            },
        )(),
    )

    anisotropic_poc_driver.main()
    output_path = Path(tmp_path) / "anisotropic_gmc_vs_mc_56x56.npz"
    assert output_path.exists()
    payload = np.load(output_path)
    assert payload["refinement"] == 8
    assert not bool(payload["sn_compatible"])
    assert float(payload["sigma_s_scale"]) == 1.5
    assert str(payload["train_sampling"]) == "lhs"
    assert payload["gmc_exit_counts"].tolist() == [1, 2, 3, 4]
    assert payload["mc_exit_counts"].tolist() == [5, 6, 7, 8]
    assert seen["model_mode"] == "anisotropic"
    assert seen["train_sampling"] == "lhs"
    assert seen["mc_mode"] == "anisotropic"


def test_distilled_lattice_driver_main_smoke_with_stubs(tmp_path, monkeypatch):
    seen_modes = []
    seen_artifact_roots = []
    problem = mg.build_rect_lattice_problem(refinement=1)
    hierarchy = mg.make_single_level_hierarchy(problem)

    class DummyQuadrature:
        theta_centers = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)

    monkeypatch.setattr(distilled_driver, "build_rect_lattice_problem", lambda refinement: problem)
    monkeypatch.setattr(distilled_driver, "make_single_level_hierarchy", lambda _problem: hierarchy)
    monkeypatch.setattr(distilled_driver, "build_uniform_angular_quadrature", lambda *_args, **_kwargs: DummyQuadrature())

    def fake_load_models(*_args, **kwargs):
        seen_modes.append(kwargs.get("model_mode"))
        seen_artifact_roots.append(kwargs.get("student_artifact_root"))
        return object()

    monkeypatch.setattr(distilled_driver, "load_steady_gmc_models", fake_load_models)

    def fake_run_gmc(*_args, **_kwargs):
        return {
            "coarse_flux": np.ones_like(problem.sigma_t_map, dtype=np.float64),
            "totals": {"left": 1, "right": 2, "top": 3, "bottom": 4, "absorb": 5},
        }

    monkeypatch.setattr(distilled_driver, "run_coarse_gmc_with_surface_tally", fake_run_gmc)
    monkeypatch.setattr(
        distilled_driver,
        "run_direct_fine_mc_reference",
        lambda *_args, **_kwargs: {
            "flux": np.full_like(problem.sigma_t_map, 2.0, dtype=np.float64),
            "stats": (6, 7, 8, 9, 10, 0, 0, 0),
        },
    )
    monkeypatch.setattr(
        distilled_driver,
        "compute_flux_error_metrics",
        lambda *_args, **_kwargs: {"l1": 1.0, "l2": 2.0, "rel_l2": 3.0, "max_abs": 4.0},
    )
    monkeypatch.setattr(
        distilled_driver,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "refinement": 1,
                "gmc_particles": 10,
                "mc_particles": 11,
                "num_angles": 4,
                "device": "cpu",
                "train_sampling": "random",
                "scattering_mode": "isotropic",
                "model_modes": ["teacher", "student_2"],
                "student_artifact_root": str(tmp_path / "students"),
                "seed": 1,
                "output_dir": str(tmp_path),
            },
        )(),
    )

    distilled_driver.main()
    assert (Path(tmp_path) / "summary.json").exists()
    payload = np.load(Path(tmp_path) / "steady2d_distilled_lattice_results.npz")
    assert "teacher_flux" in payload.files
    assert "student_2_flux" in payload.files
    assert "mc_flux" in payload.files
    assert seen_modes == ["teacher", "student_2"]
    assert all(root == str(tmp_path / "students") for root in seen_artifact_roots)
