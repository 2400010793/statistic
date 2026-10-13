"""Data-contract tests for backed reading and prediction serialization."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from virtual_embryo_benchmark import io


def dataset(*, csr=False, rows=12):
    values = np.arange(rows * 3, dtype=np.float64).reshape(rows, 3)
    data = ad.AnnData(
        X=sparse.csr_matrix(values) if csr else values,
        obs=pd.DataFrame({"celltype": ["A", "B"] * (rows // 2)}, index=[f"c{i}" for i in range(rows)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    data.obsm["spatial_3D"] = np.column_stack([np.arange(rows), np.arange(rows) + 1, np.arange(rows) + 2])
    return data


def as_dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else matrix


@pytest.mark.parametrize("csr", [False, True])
def test_backed_sampling_preserves_sorted_row_expression_coordinate_alignment_and_closes(tmp_path, monkeypatch, csr):
    original = dataset(csr=csr)
    path = tmp_path / "input.h5ad"
    original.write_h5ad(path)
    reader = io.ad.read_h5ad
    opened = []

    def capture(*args, **kwargs):
        result = reader(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(io.ad, "read_h5ad", capture)
    result = io.read_data(str(path), max_cells=5, seed=17)
    expected = np.sort(np.random.default_rng(17).choice(original.n_obs, 5, replace=False))
    assert list(result.obs_names) == list(original.obs_names[expected])
    np.testing.assert_array_equal(as_dense(result.X), as_dense(original.X)[expected])
    np.testing.assert_array_equal(result.obsm["spatial_3D"], original.obsm["spatial_3D"][expected])
    assert not result.isbacked
    assert not opened[0].file.is_open


@pytest.mark.parametrize("failure", ["invalid_budget", "invalid_expression"])
def test_backed_reader_closes_hdf5_on_failure(tmp_path, monkeypatch, failure):
    original = dataset()
    if failure == "invalid_expression":
        original.X[0, 0] = np.nan
    path = tmp_path / "input.h5ad"
    original.write_h5ad(path)
    reader = io.ad.read_h5ad
    opened = []

    def capture(*args, **kwargs):
        result = reader(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(io.ad, "read_h5ad", capture)
    with pytest.raises(ValueError):
        io.read_data(str(path), max_cells=0 if failure == "invalid_budget" else None)
    assert not opened[0].file.is_open


@pytest.mark.parametrize("csr", [False, True])
def test_float32_expression_overflow_is_rejected_after_conversion(csr):
    original = dataset(csr=csr)
    original.X[0, 0] = 1e100
    io.check_data(original)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite and nonnegative"):
        io.clean_prediction(original, task="T1")


def test_float32_coordinate_overflow_is_rejected_after_conversion():
    original = dataset()
    original.obsm["spatial_3D"] = original.obsm["spatial_3D"].astype(np.float64)
    original.obsm["spatial_3D"][0, 0] = 1e100
    io.check_data(original, spatial=True)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="spatial_3D"):
        io.clean_prediction(original, task="T2")


@pytest.mark.parametrize("coordinates", [
    None,
    np.ones((12, 2)),
    np.ones((12, 4)),
    np.full((12, 3), np.nan),
    np.full((12, 3), np.inf),
])
def test_spatial_contract_rejects_missing_extra_dimensional_or_nonfinite_coordinates(coordinates):
    original = dataset()
    del original.obsm["spatial_3D"]
    if coordinates is not None:
        original.obsm["spatial_3D"] = coordinates
    with pytest.raises(ValueError, match="spatial_3D"):
        io.check_data(original, spatial=True)


def test_panel_selection_reorders_genes_without_changing_cell_coordinate_alignment():
    original = dataset(csr=True)
    selected = io.subset_panel(original, ["g3", "g1"])
    assert list(selected.var_names) == ["g3", "g1"]
    np.testing.assert_array_equal(as_dense(selected.X), as_dense(original.X)[:, [2, 0]])
    np.testing.assert_array_equal(selected.obsm["spatial_3D"], original.obsm["spatial_3D"])
    assert list(original.var_names) == ["g1", "g2", "g3"]


def test_panel_missing_genes_and_order_mismatches_are_explicit():
    original = dataset()
    with pytest.raises(ValueError, match="Missing 1 required genes"):
        io.subset_panel(original, ["g1", "absent"])
    with pytest.raises(ValueError, match="panel/order"):
        io.check_data(original, panel=["g3", "g2", "g1"])


def test_clean_prediction_omits_input_labels_raw_and_count_layers():
    original = dataset(csr=True)
    original.layers["counts"] = original.X.copy()
    original.raw = original.copy()
    original.uns["baseline"] = {"method": "copy_last"}
    cleaned = io.clean_prediction(original, task="T2")
    assert cleaned.X.dtype == np.float32
    assert not list(cleaned.obs.columns)
    assert not list(cleaned.layers.keys())
    assert cleaned.raw is None
    assert cleaned.uns["baseline"]["method"] == "copy_last"
    np.testing.assert_array_equal(cleaned.obsm["spatial_3D"], original.obsm["spatial_3D"])


def test_contract_enforces_cell_limits_and_unique_genes():
    original = dataset()
    with pytest.raises(ValueError, match="at least"):
        io.check_data(original, min_cells=13)
    with pytest.raises(ValueError, match="at most"):
        io.check_data(original, max_cells=11)
    original.var_names = ["g1", "g1", "g3"]
    with pytest.raises(ValueError, match="unique"):
        io.check_data(original)
