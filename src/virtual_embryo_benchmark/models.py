"""Small, transparent baselines on nonnegative, log-transformed expression.

These methods predict distributions; rows in different files are never treated
as matched cells. A stage argument identifies a whole input file. If the input
also has a stage column, it must contain only that declared stage.
"""

from __future__ import annotations

from typing import Any, Sequence

import anndata as ad
import numpy as np
from scipy import sparse


def _stage_number(value: Any) -> float:
    """Accept numerical embryonic days and labels such as ``E9.5``."""
    if isinstance(value, str):
        value = value.strip().removeprefix("E").removeprefix("e")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Stages must be finite numerical embryonic days.")
    return number


def _check_stage(data: ad.AnnData, stage: Any, stage_key: str, name: str) -> None:
    if stage is None:
        raise ValueError(f"An explicit stage is required for {name}.")
    declared = _stage_number(stage)
    if stage_key in data.obs:
        values = data.obs[stage_key].drop_duplicates()
        if len(values) != 1 or _stage_number(values.iloc[0]) != declared:
            raise ValueError(
                f"{name} must contain only declared stage {stage!r}; "
                f"check obs[{stage_key!r}] or split the input by stage."
            )


def _check_expression(data: ad.AnnData, name: str) -> None:
    if data.n_obs == 0 or data.n_vars == 0:
        raise ValueError(f"{name} must contain cells and genes.")
    if not data.var_names.is_unique:
        raise ValueError(f"{name} must have unique gene identifiers.")
    if data.X is None:
        raise ValueError(f"{name}.X must contain log-transformed expression.")
    # Validate in chunks; do not materialize a whole sparse/backed input.
    for start in range(0, data.n_obs, 1024):
        block = data.X[start : start + 1024]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name}.X must be finite and nonnegative.")


def _aligned_matrix(reference: ad.AnnData, genes, name: str):
    _check_expression(reference, name)
    indices = reference.var_names.get_indexer(genes)
    missing = np.asarray(genes)[indices < 0]
    if len(missing):
        raise ValueError(
            f"{name} is missing {len(missing)} source genes, including "
            f"{', '.join(map(str, missing[:5]))}. Gene values cannot be zero-filled."
        )
    return reference.X[:, indices]


def _labels(data: ad.AnnData, celltype_key: str, name: str) -> np.ndarray:
    if celltype_key not in data.obs:
        raise ValueError(f"{name} is missing obs[{celltype_key!r}].")
    if data.obs[celltype_key].isna().any():
        raise ValueError(f"{name} has missing cell type labels.")
    return data.obs[celltype_key].astype(str).to_numpy()


def _mean(matrix, rows: np.ndarray | None = None) -> np.ndarray:
    block = matrix if rows is None else matrix[rows]
    return np.asarray(block.mean(axis=0)).reshape(-1)


def _shifts(
    source: ad.AnnData,
    earlier: ad.AnnData,
    later: ad.AnnData,
    celltype_key: str,
    min_cells: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Compute later-minus-earlier means after explicit gene alignment."""
    if min_cells < 1:
        raise ValueError("min_cells must be at least one.")
    first = _aligned_matrix(earlier, source.var_names, "earlier/reference WT")
    second = _aligned_matrix(later, source.var_names, "later/reference perturbation")
    first_labels = _labels(earlier, celltype_key, "earlier/reference WT")
    second_labels = _labels(later, celltype_key, "later/reference perturbation")
    source_labels = _labels(source, celltype_key, "source")
    result, fallback = {}, []
    for label in np.unique(source_labels):
        first_rows = np.flatnonzero(first_labels == label)
        second_rows = np.flatnonzero(second_labels == label)
        if len(first_rows) >= min_cells and len(second_rows) >= min_cells:
            result[label] = _mean(second, second_rows) - _mean(first, first_rows)
        else:
            result[label] = np.zeros(source.n_vars, dtype=np.float32)
            fallback.append(label)
    return result, fallback


def _apply_shift(
    source: ad.AnnData,
    shifts: dict[str, np.ndarray] | np.ndarray,
    celltype_key: str = "celltype",
):
    labels = _labels(source, celltype_key, "source") if isinstance(shifts, dict) else None
    # A positive shift can mathematically make a sparse output dense. Bound
    # temporary dense allocations, while retaining sparse storage on output.
    chunk_rows = max(1, min(256, 1_000_000 // source.n_vars))
    is_sparse = sparse.issparse(source.X)
    chunks = []
    for start in range(0, source.n_obs, chunk_rows):
        stop = min(start + chunk_rows, source.n_obs)
        raw = source.X[start:stop]
        block = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
        block = block.astype(np.result_type(block.dtype, np.float32), copy=True)
        if labels is None:
            block += shifts
        else:
            chunk_labels = labels[start:stop]
            for label in np.unique(chunk_labels):
                block[chunk_labels == label] += shifts[label]
        np.maximum(block, 0, out=block)
        chunks.append(sparse.csr_matrix(block) if is_sparse else block)
    return sparse.vstack(chunks, format="csr") if is_sparse else np.concatenate(chunks)


def _result(source: ad.AnnData, matrix, *, task: str, method: str, stage_key: str, stage: Any):
    # Do not copy old counts/raw layers into a prediction with a new X matrix.
    output = ad.AnnData(
        X=matrix,
        obs=source.obs.copy(),
        var=source.var.copy(),
        obsm={key: value.copy() for key, value in source.obsm.items()},
    )
    if stage is not None:
        output.obs[stage_key] = stage
    output.uns["baseline"] = {"task": task, "method": method}
    return output


def predict_t1(
    source: ad.AnnData,
    *,
    method: str = "copy_last",
    history: ad.AnnData | None = None,
    source_stage: Any = None,
    history_stage: Any = None,
    target_stage: Any = None,
    stage_key: str = "stage",
    celltype_key: str = "celltype",
    scale_by_time: bool = False,
    time_scale: float = 1.0,
    min_cells: int = 1,
) -> ad.AnnData:
    """Copy the latest snapshot or extrapolate cell-type expression means.

    ``pseudobulk_shift`` applies ``source_mean - history_mean`` to each source
    cell of the same type, clipping negative log expression to zero. Groups
    missing from either reference are frozen, as in the official baseline.
    ``min_cells`` defaults to one; raising it also freezes smaller groups.
    Set ``scale_by_time`` to multiply by (target-source)/(source-history).
    ``time_scale`` supplies an additional explicit nonnegative multiplier.
    Cell counts, cell types and within-type residuals are inherited from source;
    this baseline cannot discover future cell types or infer proliferation.
    """
    if method not in {"copy_last", "pseudobulk_shift"}:
        raise ValueError(f"Unknown T1 method: {method!r}.")
    _check_expression(source, "source")
    _check_stage(source, source_stage, stage_key, "source")
    source_time = _stage_number(source_stage)
    if target_stage is None or _stage_number(target_stage) <= source_time:
        raise ValueError("target_stage must be explicitly later than source_stage.")
    if not np.isfinite(time_scale) or time_scale < 0:
        raise ValueError("time_scale must be finite and nonnegative.")
    fallback = []
    multiplier = float(time_scale)
    if method == "copy_last":
        matrix = source.X.copy()
    else:
        if history is None:
            raise ValueError("pseudobulk_shift requires history.")
        _check_stage(history, history_stage, stage_key, "history")
        history_time = _stage_number(history_stage)
        if history_time >= source_time:
            raise ValueError("history_stage must be earlier than source_stage.")
        if scale_by_time:
            multiplier *= (_stage_number(target_stage) - source_time) / (
                source_time - history_time
            )
        shifts, fallback = _shifts(source, history, source, celltype_key, min_cells)
        matrix = _apply_shift(
            source, {key: multiplier * value for key, value in shifts.items()}, celltype_key
        )
    output = _result(
        source, matrix, task="T1", method=method, stage_key=stage_key, stage=target_stage
    )
    output.uns["baseline"].update(
        source_stage=source_time,
        target_stage=_stage_number(target_stage),
        scale_by_time=bool(scale_by_time),
        time_scale=multiplier,
        fallback_celltypes=fallback,
        frozen_celltypes=fallback,
        missing_type_policy="freeze",
        min_cells=min_cells,
    )
    return output


def predict_t2(source: ad.AnnData, *, spatial_key: str = "spatial_3D", **kwargs) -> ad.AnnData:
    """Apply a T1 expression baseline, retaining observed 3D coordinates.

    These are deliberately simple spatial baselines; neither method predicts
    migration, tissue deformation or new spatial domains.
    """
    if spatial_key not in source.obsm:
        raise ValueError(f"T2 requires obsm[{spatial_key!r}].")
    coordinates = np.asarray(source.obsm[spatial_key])
    if coordinates.shape != (source.n_obs, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"obsm[{spatial_key!r}] must have finite shape (n_cells, 3).")
    output = predict_t1(source, **kwargs)
    output.uns["baseline"]["task"] = "T2"
    output.uns["baseline"]["spatial_method"] = "copy_last"
    return output


def predict_t3(
    source: ad.AnnData,
    *,
    method: str = "wt_identity",
    observed_wt: ad.AnnData | None = None,
    observed_perturbed: ad.AnnData | None = None,
    ko_genes: Sequence[str] | str | None = None,
    stage: Any = None,
    observed_stage: Any = None,
    stage_key: str = "stage",
    celltype_key: str = "celltype",
    min_cells: int = 1,
) -> ad.AnnData:
    """Official WT/shift/KO baselines plus a pure target-gene zeroing control.

    ``stage`` declares query WT age; ``observed_stage`` independently declares
    the common age of observed WT and KO references. ``shift_transfer`` adds
    the global observed KO-minus-WT mean to every query cell, then clips at
    zero. ``gene_ko`` applies that same shift and subsequently zeroes target
    genes. ``gene_zero`` is a supplemental baseline that only zeroes targets
    and needs no perturbation references. The global shift does not use
    ``celltype_key`` or ``min_cells``, retained as API compatibility parameters.
    """
    if method not in {"wt_identity", "shift_transfer", "gene_ko", "gene_zero"}:
        raise ValueError(f"Unknown T3 method: {method!r}.")
    _check_expression(source, "source")
    _check_stage(source, stage, stage_key, "source WT")
    if method in {"shift_transfer", "gene_ko"}:
        if observed_wt is None or observed_perturbed is None:
            raise ValueError(f"{method} requires observed_wt and observed_perturbed.")
        _check_stage(observed_wt, observed_stage, stage_key, "observed WT")
        _check_stage(observed_perturbed, observed_stage, stage_key, "observed perturbation")
        wt_matrix = _aligned_matrix(observed_wt, source.var_names, "observed WT")
        ko_matrix = _aligned_matrix(observed_perturbed, source.var_names, "observed perturbation")
        matrix = _apply_shift(source, _mean(ko_matrix) - _mean(wt_matrix))
    else:
        matrix = source.X.copy()
    if method in {"gene_ko", "gene_zero"}:
        genes = [ko_genes] if isinstance(ko_genes, str) else list(
            [] if ko_genes is None else ko_genes
        )
        if not genes:
            raise ValueError(f"{method} requires at least one ko_genes identifier.")
        indices = source.var_names.get_indexer(genes)
        missing = [gene for gene, index in zip(genes, indices) if index < 0]
        if missing:
            raise ValueError(f"Knockout genes are absent from the expression panel: {missing}.")
        if sparse.issparse(matrix):
            keep = np.ones(source.n_vars, dtype=matrix.dtype)
            keep[indices] = 0
            matrix = (matrix @ sparse.diags(keep)).tocsr()
            matrix.eliminate_zeros()
        else:
            matrix = np.array(matrix, copy=True)
            matrix[:, indices] = 0
    output = _result(source, matrix, task="T3", method=method, stage_key=stage_key, stage=stage)
    output.uns["baseline"].update(stage=_stage_number(stage))
    if method in {"shift_transfer", "gene_ko"}:
        output.uns["baseline"].update(
            observed_stage=_stage_number(observed_stage), shift_scope="global"
        )
    if method in {"gene_ko", "gene_zero"}:
        output.uns["baseline"]["ko_genes"] = genes
    return output
