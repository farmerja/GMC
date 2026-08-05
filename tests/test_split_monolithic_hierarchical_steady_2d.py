import numpy as np
import pytest

from ml.hierarchical_steady_2d.split_monolithic import (
    SplitMonolithicSteady2DSampler,
    SplitMonolithicTrainingConfig,
    run_split_monolithic_training,
)
from physics.hierarchical_steady_2d import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    direction_is_outward_for_face_2d,
)
from physics.mc_data_gen import get_2d_factorized_hierarchical_steady_scattered_only_data


def test_split_monolithic_training_smoke_writes_artifacts_and_runtime_loader(tmp_path):
    pytest.importorskip("torch")

    dataframe = get_2d_factorized_hierarchical_steady_scattered_only_data(
        source="internal",
        num_neutrons=48,
        sigma_s=1.0,
        sigma_a=0.3,
        sampling="lhs",
    )
    config = SplitMonolithicTrainingConfig(
        source="internal",
        epochs=1,
        batch_size=8,
        hidden_dim=16,
        depth=1,
        flow_steps=4,
        val_ratio=0.25,
        output_root=str(tmp_path),
        run_name="split_smoke",
        device="cpu",
        seed=11,
    )
    results = run_split_monolithic_training(config, dataframe=dataframe)

    run_dir = tmp_path / "split_smoke"
    assert results["run_dir"] == str(run_dir)
    assert (run_dir / "metadata" / "run_config.json").exists()
    assert (run_dir / "metadata" / "normalization.json").exists()
    assert (run_dir / "metadata" / "results.json").exists()
    assert (run_dir / "monolithic" / "models" / "best_model.pt").exists()
    assert (run_dir / "monolithic" / "models" / "final_model.pt").exists()
    assert (run_dir / "monolithic" / "metrics" / "metrics.json").exists()
    assert (run_dir / "monolithic" / "metrics" / "history.npz").exists()
    assert np.isfinite(results["monolithic"]["best_val_loss"])

    sampler = SplitMonolithicSteady2DSampler.from_run_dir(
        run_dir,
        source_type="internal",
        device="cpu",
        batch_size=4,
    )
    sample = sampler.sample_batch(
        width=dataframe["width"].to_numpy()[:4],
        height=dataframe["height"].to_numpy()[:4],
        initial_x=dataframe["initial_pos_x"].to_numpy()[:4],
        initial_y=dataframe["initial_pos_y"].to_numpy()[:4],
        initial_u=dataframe["initial_dir_u"].to_numpy()[:4],
        initial_v=dataframe["initial_dir_v"].to_numpy()[:4],
        sigma_t=(dataframe["sigma_s"] + dataframe["sigma_a"]).to_numpy()[:4],
        seed=5,
    )

    assert sample.x_out.shape == (4,)
    assert sample.y_out.shape == (4,)
    assert sample.u_out.shape == (4,)
    assert sample.v_out.shape == (4,)
    assert sample.track_length.shape == (4,)
    assert sample.exit_face.shape == (4,)
    assert sample.pos_on_face.shape == (4,)

    for values in (
        sample.x_out,
        sample.y_out,
        sample.u_out,
        sample.v_out,
        sample.track_length,
        sample.pos_on_face,
    ):
        assert np.isfinite(values).all()
    assert np.isfinite(list(sample.diagnostics.values())).all()
    assert np.all(sample.track_length >= 0.0)

    widths = dataframe["width"].to_numpy()[:4]
    heights = dataframe["height"].to_numpy()[:4]
    for idx, (face, x_out, y_out, pos, u_val, v_val) in enumerate(
        zip(
            sample.exit_face,
            sample.x_out,
            sample.y_out,
            sample.pos_on_face,
            sample.u_out,
            sample.v_out,
        )
    ):
        assert direction_is_outward_for_face_2d(int(face), float(u_val), float(v_val))
        if int(face) == FACE_LEFT:
            assert x_out == pytest.approx(0.0, abs=1.0e-5)
            assert 0.0 <= pos <= heights[idx] + 1.0e-6
        elif int(face) == FACE_RIGHT:
            assert x_out == pytest.approx(widths[idx], abs=1.0e-5)
            assert 0.0 <= pos <= heights[idx] + 1.0e-6
        elif int(face) == FACE_TOP:
            assert y_out == pytest.approx(heights[idx], abs=1.0e-5)
            assert 0.0 <= pos <= widths[idx] + 1.0e-6
        elif int(face) == FACE_BOTTOM:
            assert y_out == pytest.approx(0.0, abs=1.0e-5)
            assert 0.0 <= pos <= widths[idx] + 1.0e-6
        else:
            raise AssertionError(f"Unexpected face id: {face}")
