"""Behavioral tests for distribution baselines and their failure boundaries."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from virtual_embryo_benchmark.models import predict_t1, predict_t2, predict_t3


def snapshot(values, *, genes=("g1", "g2"), celltypes=None, stage=None, csr=False):
    matrix = np.asarray(values, dtype=np.float32)
    if celltypes is None:
        celltypes = ["A"] * len(matrix)
    obs = pd.DataFrame({"celltype": celltypes}, index=[f"cell{i}" for i in range(len(matrix))])
    if stage is not None:
        obs["stage"] = stage
    return ad.AnnData(
        X=sparse.csr_matrix(matrix) if csr else matrix,
        obs=obs,
        var=pd.DataFrame(index=genes),
    )


def dense(data):
    return data.X.toarray() if sparse.issparse(data.X) else data.X


def test_copy_last_keeps_source_distribution_and_does_not_modify_input():
    source = snapshot([[1, 3], [2, 4]], stage="E9.5", csr=True)
    output = predict_t1(source, source_stage=9.5, target_stage=10.5)
    assert sparse.issparse(output.X)
    np.testing.assert_array_equal(dense(output), dense(source))
    assert (source.obs.stage == "E9.5").all()
    assert (output.obs.stage == 10.5).all()
    output.X[0, 0] = 9
    assert source.X[0, 0] == 1


@pytest.mark.parametrize("csr", [False, True])
def test_temporal_shift_aligns_gene_order_and_uses_celltype_means(csr):
    history = snapshot(
        [[5, 1], [7, 3], [1, 6], [3, 8]],
        genes=("g2", "g1"),
        celltypes=["A", "A", "B", "B"],
        csr=csr,
    )
    source = snapshot(
        [[3, 4], [5, 6], [3, 5], [5, 7]],
        celltypes=["A", "A", "B", "B"],
        csr=csr,
    )
    output = predict_t1(
        source, method="pseudobulk_shift", history=history,
        source_stage=9.5, history_stage=8.5, target_stage=10.5,
    )
    np.testing.assert_allclose(dense(output), [[5, 3], [7, 5], [0, 9], [2, 11]])
    assert list(output.var_names) == ["g1", "g2"]
    assert sparse.issparse(output.X) == csr
    np.testing.assert_allclose(dense(source), [[3, 4], [5, 6], [3, 5], [5, 7]])


def test_time_scaling_is_opt_in_and_uses_forecast_horizon():
    history = snapshot([[1, 4], [1, 4]])
    source = snapshot([[2, 3], [2, 3]])
    arguments = dict(
        method="pseudobulk_shift", history=history,
        source_stage=9.5, history_stage=8.5, target_stage=12.5,
    )
    unscaled = predict_t1(source, **arguments)
    scaled = predict_t1(source, scale_by_time=True, **arguments)
    np.testing.assert_allclose(dense(unscaled), [[3, 2], [3, 2]])
    np.testing.assert_allclose(dense(scaled), [[5, 0], [5, 0]])


def test_unseen_celltype_is_frozen_instead_of_receiving_global_shift():
    history = snapshot([[1, 2]], celltypes=["A"], csr=True)
    source = snapshot([[2, 1]], celltypes=["B"], csr=True)
    output = predict_t1(
        source, method="pseudobulk_shift", history=history,
        source_stage=9.5, history_stage=8.5, target_stage=10.5,
    )
    np.testing.assert_allclose(dense(output), [[2, 1]])
    assert output.uns["baseline"]["frozen_celltypes"] == ["B"]


def test_single_cell_type_shifts_by_default_but_freezes_at_stricter_threshold():
    history = snapshot([[1, 2]], csr=True)
    source = snapshot([[2, 1]], csr=True)
    arguments = dict(
        method="pseudobulk_shift", history=history,
        source_stage=9.5, history_stage=8.5, target_stage=10.5,
    )
    shifted = predict_t1(source, **arguments)
    frozen = predict_t1(source, min_cells=2, **arguments)
    np.testing.assert_allclose(dense(shifted), [[3, 0]])
    np.testing.assert_allclose(dense(frozen), [[2, 1]])
    assert frozen.uns["baseline"]["frozen_celltypes"] == ["A"]


def test_missing_genes_are_rejected_instead_of_silently_filled():
    source = snapshot([[1, 2]])
    history = snapshot([[1, 2]], genes=("g1", "other"))
    with pytest.raises(ValueError, match="missing 1 source genes"):
        predict_t1(
            source, method="pseudobulk_shift", history=history,
            source_stage=9.5, history_stage=8.5, target_stage=10.5,
        )


def test_temporal_stages_are_explicit_and_cannot_mix_snapshots():
    source = snapshot([[1, 2], [3, 4]], stage=9.5)
    with pytest.raises(ValueError, match="explicit stage"):
        predict_t1(source, target_stage=10.5)
    with pytest.raises(ValueError, match="later than"):
        predict_t1(source, source_stage=9.5, target_stage=8.5)
    source.obs.loc["cell1", "stage"] = 8.5
    with pytest.raises(ValueError, match="only declared stage"):
        predict_t1(source, source_stage=9.5, target_stage=10.5)


def test_t2_keeps_coordinate_expression_row_correspondence():
    source = snapshot([[1, 2], [3, 4]])
    source.obsm["spatial_3D"] = np.array([[1, 4, 2], [9, 3, 5]], dtype=float)
    output = predict_t2(source, source_stage=7.0, target_stage=8.0)
    np.testing.assert_array_equal(output.obsm["spatial_3D"], source.obsm["spatial_3D"])
    assert output.uns["baseline"]["task"] == "T2"
    output.obsm["spatial_3D"][0, 0] = 0
    assert source.obsm["spatial_3D"][0, 0] == 1
    with pytest.raises(ValueError, match="spatial_3D"):
        predict_t2(snapshot([[1, 2]]), source_stage=7.0, target_stage=8.0)


def test_t3_shift_transfers_reference_effect_with_gene_alignment():
    source = snapshot([[5, 2], [7, 4]], stage=8.75, csr=True)
    wt = snapshot([[1, 3], [3, 5]], stage="E9.5", csr=True)
    perturbation = snapshot(
        [[2, 3], [4, 5]], genes=("g2", "g1"), stage=9.5, csr=True,
    )
    output = predict_t3(
        source, method="shift_transfer", observed_wt=wt,
        observed_perturbed=perturbation, stage=8.75, observed_stage=9.5,
    )
    np.testing.assert_allclose(dense(output), [[7, 1], [9, 3]])
    assert output.uns["baseline"]["stage"] == 8.75
    assert output.uns["baseline"]["observed_stage"] == 9.5
    perturbation.obs["stage"] = 8.75
    with pytest.raises(ValueError, match="only declared stage"):
        predict_t3(
            source, method="shift_transfer", observed_wt=wt,
            observed_perturbed=perturbation, stage=8.75, observed_stage=9.5,
        )


def test_t3_shift_is_global_and_does_not_require_celltype_annotations():
    source = snapshot([[5, 2], [7, 4]], celltypes=["A", "B"])
    wt = snapshot([[1, 3], [3, 5]], celltypes=["A", "B"])
    perturbation = snapshot([[2, 2], [6, 6]], celltypes=["A", "B"])
    # Per-type effects differ; the official baseline applies the same global
    # [2, 0] effect to both target rows, without using cell type labels.
    del source.obs["celltype"]
    del wt.obs["celltype"]
    del perturbation.obs["celltype"]
    output = predict_t3(
        source, method="shift_transfer", observed_wt=wt,
        observed_perturbed=perturbation, stage=8.75, observed_stage=9.5,
    )
    np.testing.assert_allclose(dense(output), [[7, 2], [9, 4]])
    assert output.uns["baseline"]["shift_scope"] == "global"


@pytest.mark.parametrize("csr", [False, True])
def test_official_gene_ko_transfers_global_effect_then_zeroes_target(csr):
    source = snapshot([[5, 2], [7, 4]], csr=csr)
    wt = snapshot([[1, 3], [3, 5]], csr=csr)
    perturbation = snapshot([[3, 2], [5, 4]], csr=csr)
    output = predict_t3(
        source, method="gene_ko", ko_genes="g2", stage=8.75,
        observed_wt=wt, observed_perturbed=perturbation, observed_stage=9.5,
    )
    np.testing.assert_allclose(dense(output), [[7, 0], [9, 0]])
    np.testing.assert_allclose(dense(source), [[5, 2], [7, 4]])
    assert output.uns["baseline"]["method"] == "gene_ko"


def test_t3_reference_stage_and_reference_distributions_are_required():
    source = snapshot([[5, 2]])
    for method in ["shift_transfer", "gene_ko"]:
        with pytest.raises(ValueError, match="requires observed_wt"):
            predict_t3(source, method=method, stage=8.75, ko_genes="g2")
        with pytest.raises(ValueError, match="explicit stage"):
            predict_t3(
                source, method=method, stage=8.75, observed_wt=source,
                observed_perturbed=source, ko_genes="g2",
            )


@pytest.mark.parametrize("csr", [False, True])
def test_supplemental_gene_zero_changes_only_target_and_identity_changes_nothing(csr):
    source = snapshot([[5, 2], [7, 4]], csr=csr)
    identity = predict_t3(source, stage=10.5)
    knockout = predict_t3(source, method="gene_zero", ko_genes="g2", stage=10.5)
    np.testing.assert_allclose(dense(identity), [[5, 2], [7, 4]])
    np.testing.assert_allclose(dense(knockout), [[5, 0], [7, 0]])
    np.testing.assert_allclose(dense(source), [[5, 2], [7, 4]])
    with pytest.raises(ValueError, match="absent"):
        predict_t3(source, method="gene_zero", ko_genes=["missing"], stage=10.5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0])
def test_invalid_log_expression_is_rejected(bad_value):
    source = snapshot([[1, bad_value]], csr=True)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        predict_t1(source, source_stage=9.5, target_stage=10.5)
