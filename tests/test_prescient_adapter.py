"""Tests for the locked-author PRESCIENT T1 adapter."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from virtual_embryo_benchmark.prescient_adapter import (
    AUTHOR_COMMIT,
    AUTHOR_HASHES,
    MAX_WORKING_BYTES,
    _dense_working_estimate,
    predict_prescient,
)


def snapshot(values, *, genes=("g1", "g2", "g3"), stage=None):
    matrix = np.asarray(values, dtype=np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(len(matrix))])
    if stage is not None:
        obs["stage"] = stage
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))


def toy_pair(*, n=10, stage_columns=False):
    rng = np.random.default_rng(12)
    history_values = rng.uniform(1, 2, size=(n, 3)).astype(np.float32)
    source_values = history_values + np.array([0.15, 0.05, 0.25], dtype=np.float32)
    return (
        snapshot(source_values, stage=9.5 if stage_columns else None),
        snapshot(history_values, stage=8.5 if stage_columns else None),
    )


def run_adapter(source, history, **kwargs):
    defaults = dict(
        root=".", source_stage=9.5, history_stage=8.5, target_stage=10.5,
        epochs=3, n_components=2, hidden_dim=8, batch_size=10,
        noise_sd=0.0, dt=0.25, seed=3, device="cpu",
    )
    defaults.update(kwargs)
    return predict_prescient(source, history, **defaults)


def test_adapter_imports_real_author_class_and_locks_source():
    prediction = run_adapter(*toy_pair())
    metadata = prediction.uns["baseline"]
    assert metadata["author_commit"] == AUTHOR_COMMIT
    assert metadata["author_model_class"] == "prescient.train.model.AutoGenerator"
    assert metadata["author_source_sha256"] == AUTHOR_HASHES
    assert metadata["contrastive_pretraining"] is False
    assert metadata["growth_weights"] == "uniform"


def test_adapter_returns_nonnegative_full_gene_forecast_and_target_stage():
    source, history = toy_pair(stage_columns=True)
    prediction = run_adapter(source, history, target_stage=11.5)
    assert prediction.shape == source.shape
    assert list(prediction.var_names) == list(source.var_names)
    assert np.isfinite(prediction.X).all()
    assert (prediction.X >= 0).all()
    np.testing.assert_allclose(prediction.obs["stage"], 11.5)
    assert prediction.uns["baseline"]["target_stage"] == 11.5
    # Predictions retain the observed source population's row identity/order.
    assert list(prediction.obs_names) == list(source.obs_names)
    np.testing.assert_allclose(source.X, toy_pair(stage_columns=True)[0].X)


def test_adapter_is_reproducible_with_seed_and_changes_with_seed():
    source, history = toy_pair()
    first = run_adapter(source, history, seed=4)
    second = run_adapter(source, history, seed=4)
    third = run_adapter(source, history, seed=5)
    np.testing.assert_allclose(first.X, second.X)
    assert not np.allclose(first.X, third.X)
    np.testing.assert_array_equal(first.uns["baseline"]["training_loss"].shape, (3,))


def test_adapter_checkpoint_contains_author_state_pca_and_fingerprints(tmp_path):
    source, history = toy_pair()
    checkpoint = tmp_path / "prescient" / "state.pt"
    prediction = run_adapter(source, history, checkpoint_path=checkpoint)
    assert checkpoint.exists()
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(
        [
            "model_state_dict", "config", "metadata", "pca_mean",
            "pca_components", "latent_scale", "gene_names",
            "history_fingerprint", "source_fingerprint",
        ]
    ) <= set(saved)
    assert saved["metadata"]["author_commit"] == AUTHOR_COMMIT
    assert saved["gene_names"] == ["g1", "g2", "g3"]
    assert saved["pca_components"].shape == (2, 3)
    assert len(saved["model_state_dict"]) > 0
    assert prediction.uns["baseline"]["training_loss"].shape == (3,)
    # Without clipping, a two-component forecast only changes the learned PCA
    # subspace; orthogonal full-gene residuals must survive prediction.
    components = saved["pca_components"].numpy()
    difference = prediction.X - source.X
    residual = difference - (difference @ components.T) @ components
    np.testing.assert_allclose(residual, 0, atol=2e-6)


def test_adapter_aligns_history_gene_order_without_zero_filling():
    source, history = toy_pair()
    values = history.X
    reordered = ad.AnnData(
        X=values[:, [2, 0, 1]], obs=history.obs.copy(),
        var=pd.DataFrame(index=["g3", "g1", "g2"]),
    )
    reference = run_adapter(source, history)
    aligned = run_adapter(source, reordered)
    np.testing.assert_allclose(reference.X, aligned.X)


def test_adapter_rejects_missing_genes_and_bad_expression():
    source, history = toy_pair()
    missing = ad.AnnData(
        X=history.X[:, :2], obs=history.obs.copy(),
        var=pd.DataFrame(index=["g1", "other"]),
    )
    with pytest.raises(ValueError, match="missing source genes"):
        run_adapter(source, missing)
    invalid = history.copy()
    invalid.X[0, 0] = -1
    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_adapter(source, invalid)


def test_adapter_rejects_mixed_or_reverse_stages():
    source, history = toy_pair(stage_columns=True)
    source.obs.loc["cell1", "stage"] = 8.5
    with pytest.raises(ValueError, match="only declared stage"):
        run_adapter(source, history)
    source, history = toy_pair()
    with pytest.raises(ValueError, match="history_stage < source_stage < target_stage"):
        run_adapter(source, history, target_stage=8.75)


@pytest.mark.parametrize("parameter", ["epochs", "n_components", "hidden_dim", "batch_size"])
def test_adapter_rejects_nonpositive_training_parameters(parameter):
    source, history = toy_pair()
    with pytest.raises(ValueError, match="positive integers"):
        run_adapter(source, history, **{parameter: 0})


def test_adapter_requires_locked_author_source(tmp_path):
    source, history = toy_pair()
    with pytest.raises(ValueError, match="source missing"):
        run_adapter(source, history, root=tmp_path)


def test_adapter_rejects_changed_locked_model_before_importing_it(tmp_path):
    source, history = toy_pair()
    installed = tmp_path / "third_party" / "prescient"
    train_path = installed / "prescient" / "train"
    train_path.mkdir(parents=True)
    (installed / ".source-lock.json").write_text(json.dumps({"commit": AUTHOR_COMMIT}))
    # A lockfile alone does not authorize substituted model code.
    (train_path / "model.py").write_text("raise RuntimeError('must never import')\n")
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        run_adapter(source, history, root=tmp_path)


def test_adapter_caps_cells_before_pca_or_author_import():
    source, history = toy_pair(n=5001)
    with pytest.raises(ValueError, match="5000 cells"):
        run_adapter(source, history)


def test_dense_budget_allows_two_thousand_cells_on_full_official_panel():
    # This matches the supplied runnable configuration, without allocating a
    # 2 GiB matrix solely to test the budget arithmetic.
    assert _dense_working_estimate(2000, 2000, 32285) < MAX_WORKING_BYTES
    assert _dense_working_estimate(5000, 5000, 32285) > MAX_WORKING_BYTES


def test_dense_allocation_guard_runs_before_author_import(monkeypatch):
    from virtual_embryo_benchmark import prescient_adapter

    monkeypatch.setattr(prescient_adapter, "MAX_WORKING_BYTES", 1)
    source, history = toy_pair()
    with pytest.raises(ValueError, match="dense PCA working estimate"):
        run_adapter(source, history)


def test_real_author_potential_training_learns_translation():
    rng = np.random.default_rng(4)
    early = rng.uniform(1, 2, size=(12, 2)).astype(np.float32)
    source = snapshot(early + [0.2, 0.4], genes=("g1", "g2"))
    history = snapshot(early, genes=("g1", "g2"))
    prediction = run_adapter(
        source, history, epochs=20, hidden_dim=16, batch_size=12,
        n_components=2, dt=0.1, seed=0,
    )
    losses = prediction.uns["baseline"]["training_loss"]
    assert np.isfinite(losses).all()
    assert losses[-3:].mean() < 0.5 * losses[:3].mean()
    assert np.all(prediction.X.mean(axis=0) > source.X.mean(axis=0))
