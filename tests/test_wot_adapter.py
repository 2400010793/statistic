"""WOT API, data alignment and explicit forecast-adaptation checks."""

from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from virtual_embryo_benchmark import wot_adapter


def snapshot(values, *, genes=("g1", "g2"), csr=False):
    values = np.asarray(values, dtype=np.float32)
    return ad.AnnData(
        X=sparse.csr_matrix(values) if csr else values,
        obs=pd.DataFrame(index=[f"cell{index}" for index in range(len(values))]),
        var=pd.DataFrame(index=genes),
    )


def forecast(source, history, **kwargs):
    return wot_adapter.predict_wot(
        source, history, source_stage=9.5, history_stage=8.5,
        target_stage=10.5, **kwargs,
    )


def test_calls_author_api_and_preserves_gene_identity(monkeypatch, tmp_path):
    calls = {}

    class AuthorModel:
        def __init__(self, training, **parameters):
            calls["training"] = training.copy()
            calls["parameters"] = parameters

        def compute_transport_map(self, first_stage, last_stage):
            calls["stages"] = first_stage, last_stage
            return ad.AnnData(np.eye(2, dtype=np.float32))

    author_module = SimpleNamespace(ot=SimpleNamespace(OTModel=AuthorModel))
    monkeypatch.setattr(wot_adapter, "_load_wot", lambda root: (author_module, "test_commit"))
    source = snapshot([[2, 4], [3, 5]], csr=True)
    history = snapshot([[3, 1], [4, 2]], genes=("g2", "g1"), csr=True)
    prediction = forecast(source, history, checkpoint_path=tmp_path / "transport.h5ad", n_iter=37)
    np.testing.assert_allclose(prediction.X, [[3, 5], [4, 6]])
    np.testing.assert_array_equal(calls["training"].X[:2], [[1, 3], [2, 4]])
    assert calls["parameters"]["solver"] == "fixed_iters"
    assert calls["parameters"]["scaling_iter"] == 37
    assert calls["parameters"]["extra_iter"] == 0
    assert calls["stages"] == (8.5, 9.5)
    assert prediction.uns["baseline"]["source_commit"] == "test_commit"
    np.testing.assert_allclose(source.X.toarray(), [[2, 4], [3, 5]])
    saved = ad.read_h5ad(tmp_path / "transport.h5ad")
    np.testing.assert_array_equal(saved.X, np.eye(2))
    assert saved.uns["adapter"]["forecast"] == "expression_barycentric_linear"


def test_unavailable_wot_reports_dependency_install_without_fallback(monkeypatch, tmp_path):
    def unavailable(name):
        raise ModuleNotFoundError("No module named 'ot'")

    monkeypatch.setattr(wot_adapter, "_verify_wot_source", lambda root: tmp_path)
    monkeypatch.setattr(wot_adapter.importlib, "import_module", unavailable)
    monkeypatch.delitem(sys.modules, "wot", raising=False)
    with pytest.raises(RuntimeError, match="pip install POT matplotlib"):
        wot_adapter._load_wot(tmp_path)


def test_missing_source_lock_is_not_replaced_by_installed_wot(monkeypatch, tmp_path):
    def must_not_import(name):
        pytest.fail("Unverified installed WOT must never replace missing locked source")

    monkeypatch.setattr(wot_adapter.importlib, "import_module", must_not_import)
    with pytest.raises(ValueError, match="locked source missing"):
        wot_adapter._load_wot(tmp_path)


@pytest.mark.parametrize("tamper", ["commit", "manifest", "file", "extra_file"])
def test_rejects_wrong_commit_incomplete_lock_and_changed_code(tmp_path, tamper):
    root = Path(__file__).resolve().parents[1]
    original = root / "third_party/wot"
    if not (original / ".source-lock.json").exists():
        pytest.skip("Locked WOT source has not been acquired")
    installed = tmp_path / "third_party/wot"
    shutil.copytree(original / "wot", installed / "wot")
    lock = json.loads((original / ".source-lock.json").read_text())
    if tamper == "commit":
        lock["commit"] = "different_author_version"
    elif tamper == "manifest":
        lock["file_sha256"].pop("wot/ot/ot_model.py")
    elif tamper == "file":
        path = installed / "wot/ot/ot_model.py"
        path.write_text(path.read_text() + "\n# local change\n")
    else:
        (installed / "wot/extra.py").write_text("# unregistered source\n")
    (installed / ".source-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="commit|hash lock|SHA256 mismatch|complete source lock"):
        wot_adapter._verify_wot_source(tmp_path)


def test_rejects_missing_gene_before_loading_solver(monkeypatch):
    def must_not_load(root):
        pytest.fail("Invalid training data must fail before loading WOT")

    monkeypatch.setattr(wot_adapter, "_load_wot", must_not_load)
    with pytest.raises(ValueError, match="missing source genes"):
        forecast(snapshot([[1, 2], [2, 3]]), snapshot([[1, 2], [2, 3]], genes=("g1", "wrong")))


def test_rejects_zero_mass_instead_of_fabricating_ancestry(monkeypatch):
    class AuthorModel:
        def __init__(self, training, **parameters):
            pass

        def compute_transport_map(self, first_stage, last_stage):
            return ad.AnnData(np.array([[1, 0], [1, 0]], dtype=np.float32))

    monkeypatch.setattr(
        wot_adapter, "_load_wot",
        lambda root: (SimpleNamespace(ot=SimpleNamespace(OTModel=AuthorModel)), "test_commit"),
    )
    with pytest.raises(RuntimeError, match="zero ancestral mass"):
        forecast(snapshot([[1, 2], [2, 3]]), snapshot([[1, 2], [2, 3]]))


def test_real_locked_author_wot_runs_small_unpaired_forecast(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / "third_party/wot/wot/__init__.py").exists():
        pytest.skip("Locked WOT source has not been acquired")
    pytest.importorskip("ot")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(7)
    early = rng.uniform(1, 2, size=(8, 2))
    source_values = early + [0.2, 0.3]
    prediction = forecast(
        snapshot(source_values), snapshot(early), root=root,
        n_components=2, n_iter=100, checkpoint_path=tmp_path / "real.transport.h5ad",
    )
    transport = ad.read_h5ad(tmp_path / "real.transport.h5ad")
    assert transport.shape == (8, 8)
    assert np.isfinite(transport.X).all() and (transport.X >= 0).all()
    assert prediction.shape == (8, 2)
    assert np.isfinite(prediction.X).all() and (prediction.X >= 0).all()
    assert prediction.uns["baseline"]["solver_api"] == "wot.ot.OTModel.compute_transport_map"
    assert prediction.uns["baseline"]["source_commit"] != "installed_version_unverified"
    assert np.all(prediction.X.mean(axis=0) > source_values.mean(axis=0))
