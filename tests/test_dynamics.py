"""Distribution and extrapolation checks for local training-only OT models."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from virtual_embryo_benchmark.dynamics import entropic_coupling, predict_ot


def snapshot(values, *, genes=("g1", "g2"), csr=False):
    matrix = np.asarray(values, dtype=np.float32)
    return ad.AnnData(
        X=sparse.csr_matrix(matrix) if csr else matrix,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(len(matrix))]),
        var=pd.DataFrame(index=genes),
    )


def forecast(source, history, **kwargs):
    return predict_ot(
        source, history, source_stage=9.5, history_stage=8.5,
        target_stage=11.5, n_components=2, **kwargs,
    )


def test_sinkhorn_uniform_marginals_for_unequal_snapshot_sizes():
    rng = np.random.default_rng(13)
    coupling = entropic_coupling(
        rng.normal(size=(5, 3)), rng.normal(size=(7, 3)),
        epsilon=0.2, iterations=500,
    )
    assert coupling.shape == (5, 7)
    assert np.isfinite(coupling).all()
    assert (coupling >= 0).all()
    np.testing.assert_allclose(coupling.sum(axis=1), np.full(5, 1 / 5), atol=1e-7)
    np.testing.assert_allclose(coupling.sum(axis=0), np.full(7, 1 / 7), atol=1e-7)


def test_sinkhorn_identical_states_still_have_uniform_valid_coupling():
    coupling = entropic_coupling(np.ones((3, 2)), np.ones((4, 2)))
    np.testing.assert_allclose(coupling, np.full((3, 4), 1 / 12), atol=1e-10)


@pytest.mark.parametrize("epsilon,iterations", [(0, 20), (-0.1, 20), (0.1, 0)])
def test_sinkhorn_rejects_nonpositive_solver_settings(epsilon, iterations):
    with pytest.raises(ValueError, match="positive"):
        entropic_coupling(np.ones((2, 2)), np.ones((2, 2)), epsilon=epsilon, iterations=iterations)


@pytest.mark.parametrize("epsilon", [np.nan, np.inf, -np.inf])
def test_sinkhorn_rejects_nonfinite_regularization_instead_of_producing_nan(epsilon):
    with pytest.raises(ValueError, match="positive"):
        entropic_coupling(np.ones((2, 2)), np.ones((2, 2)), epsilon=epsilon)


@pytest.mark.parametrize("csr", [False, True])
def test_linear_translation_extrapolates_mean_using_two_training_snapshots(csr):
    rng = np.random.default_rng(4)
    early = rng.uniform(1, 2, size=(12, 2)).astype(np.float32)
    shift = np.array([0.2, 0.4], dtype=np.float32)
    source = snapshot(early + shift, csr=csr)
    history = snapshot(early, csr=csr)
    # No target distribution is supplied. With the full two-gene PCA space,
    # the balanced coupling preserves means and the two-day horizon adds 2*shift.
    prediction = forecast(source, history, model="ot_linear", sinkhorn_iterations=500)
    np.testing.assert_allclose(
        prediction.X.mean(axis=0), (early + shift).mean(axis=0) + 2 * shift,
        atol=2e-5,
    )
    assert prediction.shape == source.shape
    assert list(prediction.var_names) == list(source.var_names)
    np.testing.assert_allclose(
        source.X.toarray() if csr else source.X, early + shift,
    )
    assert np.isfinite(prediction.X).all()
    assert (prediction.X >= 0).all()


def test_linear_forecast_aligns_history_gene_order_and_ignores_extra_gene():
    early = np.array([[1, 3], [2, 4], [3, 2], [4, 1]], dtype=np.float32)
    source = snapshot(early + [0.5, 0.25])
    reference = forecast(source, snapshot(early), model="ot_linear")
    reordered = snapshot(
        np.column_stack([early[:, 1], np.full(len(early), 99), early[:, 0]]),
        genes=("g2", "extra", "g1"), csr=True,
    )
    prediction = forecast(source, reordered, model="ot_linear")
    np.testing.assert_allclose(prediction.X, reference.X, atol=1e-6)
    assert list(prediction.var_names) == ["g1", "g2"]


def test_missing_history_gene_is_rejected_instead_of_zero_filled():
    with pytest.raises(ValueError, match="missing source genes"):
        forecast(
            snapshot([[1, 2], [2, 3]]),
            snapshot([[1, 2], [2, 3]], genes=("g1", "unrelated")),
        )


@pytest.mark.parametrize(
    "history_stage,source_stage,target_stage",
    [(9.5, 8.5, 10.5), (9.5, 9.5, 10.5), (8.5, 9.5, 9.5), (8.5, 9.5, 8.75)],
)
def test_forecast_requires_strict_forward_stage_order(history_stage, source_stage, target_stage):
    data = snapshot([[1, 2], [2, 3]])
    with pytest.raises(ValueError, match="history_stage < source_stage < target_stage"):
        predict_ot(
            data, data, history_stage=history_stage,
            source_stage=source_stage, target_stage=target_stage,
        )


@pytest.mark.parametrize("which", ["history", "source"])
@pytest.mark.parametrize("mixed", [False, True])
def test_declared_training_age_must_match_every_input_row(which, mixed):
    history = snapshot([[1, 2], [2, 3]])
    source = snapshot([[2, 3], [3, 4]])
    history.obs["stage"] = 8.5
    source.obs["stage"] = 9.5
    mismatched = history if which == "history" else source
    # Both an entirely mislabeled future snapshot and a file mixing ages must
    # be rejected before treating it as a declared training snapshot.
    if mixed:
        mismatched.obs.loc["cell1", "stage"] = 10.5
    else:
        mismatched.obs["stage"] = 10.5
    with pytest.raises(ValueError, match="only declared stage"):
        forecast(source, history)


def test_forecast_records_target_age_without_relabeling_training_inputs():
    history = snapshot([[1, 2], [2, 3], [3, 1]])
    source = snapshot([[2, 3], [3, 4], [4, 2]])
    history.obs["stage"] = "E8.5"
    source.obs["stage"] = "E9.5"
    prediction = forecast(source, history)
    assert (prediction.obs["stage"] == 11.5).all()
    assert prediction.uns["baseline"]["target_stage"] == 11.5
    assert (history.obs["stage"] == "E8.5").all()
    assert (source.obs["stage"] == "E9.5").all()


def test_negative_extrapolated_expression_is_clipped_to_zero():
    prediction = predict_ot(
        snapshot([[1, 1], [1, 1]]), snapshot([[6, 6], [6, 6]]),
        source_stage=9.5, history_stage=8.5, target_stage=10.5, n_components=1,
    )
    np.testing.assert_allclose(prediction.X, np.zeros((2, 2)), atol=1e-6)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1])
def test_nonfinite_or_negative_input_expression_is_rejected(invalid):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        forecast(snapshot([[invalid, 1], [2, 3]], csr=True), snapshot([[1, 1], [2, 2]]))


def test_t2_preserves_cell_coordinate_correspondence_and_does_not_modify_input():
    history = snapshot([[1, 2], [2, 1], [3, 4]])
    source = snapshot([[2, 3], [3, 2], [4, 5]])
    coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    source.obsm["spatial_3D"] = coordinates.copy()
    prediction = forecast(source, history, task="T2")
    np.testing.assert_array_equal(prediction.obsm["spatial_3D"], coordinates)
    prediction.obsm["spatial_3D"][0, 0] = 100
    np.testing.assert_array_equal(source.obsm["spatial_3D"], coordinates)


def test_flow_trains_and_extrapolates_on_small_synthetic_translation():
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    early = rng.uniform(1, 2, size=(12, 2)).astype(np.float32)
    source_values = early + np.array([0.2, 0.4], dtype=np.float32)
    prediction = predict_ot(
        snapshot(source_values), snapshot(early), model="ot_flow",
        source_stage=9.5, history_stage=8.5, target_stage=10.5,
        n_components=2, epochs=80, hidden_dim=16, batch_size=12,
        integration_steps=10, device="cpu", seed=2,
    )
    loss = prediction.uns["baseline"]["training_loss"]
    assert len(loss) == 80
    assert np.isfinite(loss).all()
    # Check that optimization actually learns rather than merely returning a
    # finite initial network; compare short windows to tolerate minibatch noise.
    assert loss[-5:].mean() < 0.5 * loss[:5].mean()
    assert np.isfinite(prediction.X).all()
    assert (prediction.X >= 0).all()
    assert np.all(prediction.X.mean(axis=0) > source_values.mean(axis=0))
    assert prediction.uns["baseline"]["method"] == "ot_flow"


@pytest.mark.parametrize("setting", ["epochs", "hidden_dim", "batch_size", "integration_steps"])
def test_flow_rejects_zero_training_and_integration_settings(setting):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must be positive"):
        forecast(
            snapshot([[1, 2], [2, 3]]), snapshot([[0.5, 1], [1, 2]]),
            model="ot_flow", **{setting: 0},
        )
