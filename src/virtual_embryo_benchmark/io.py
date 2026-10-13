"""Data-contract checks and bounded sampling before model/scorer allocation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, content) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_data(path: str, *, max_cells=None, seed=0) -> ad.AnnData:
    """Read just the requested rows from backed storage; close HDF5 afterwards."""
    data = ad.read_h5ad(path, backed="r")
    try:
        if max_cells is not None and max_cells < 1:
            raise ValueError("max_cells must be positive")
        if max_cells and data.n_obs > max_cells:
            rows = np.sort(np.random.default_rng(seed).choice(data.n_obs, max_cells, replace=False))
            result = data[rows].to_memory()
        else:
            result = data.to_memory()
    finally:
        data.file.close()
    check_data(result)
    return result


def check_data(data: ad.AnnData, *, spatial=False, panel=None, min_cells=None, max_cells=None):
    if data.n_obs < 1 or data.n_vars < 1:
        raise ValueError("Expression must have cells and genes")
    if not data.var_names.is_unique:
        raise ValueError("Gene names must be unique")
    if panel is not None and list(data.var_names) != list(panel):
        raise ValueError("Gene panel/order differs from the selected board")
    if min_cells and data.n_obs < min_cells:
        raise ValueError(f"Board requires at least {min_cells} cells; got {data.n_obs}")
    if max_cells and data.n_obs > max_cells:
        raise ValueError(f"Board allows at most {max_cells} cells; got {data.n_obs}")
    for offset in range(0, data.n_obs, 128):
        block = data.X[offset:offset + 128]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expression must be finite and nonnegative (already log-normalized)")
    if spatial:
        if "spatial_3D" not in data.obsm:
            raise ValueError("Spatial tasks require obsm['spatial_3D']")
        coordinates = np.asarray(data.obsm["spatial_3D"])
        if coordinates.shape != (data.n_obs, 3) or not np.isfinite(coordinates).all():
            raise ValueError("spatial_3D must be finite [n_cells,3]")


def subset_panel(data: ad.AnnData, panel):
    missing = [gene for gene in panel if gene not in data.var_names]
    if missing:
        raise ValueError(f"Missing {len(missing)} required genes: {missing[:5]}")
    return data[:, list(panel)].copy()


def clean_prediction(data: ad.AnnData, *, task):
    """Write only predicted channels; omit input labels/counts/raw layers."""
    result = ad.AnnData(
        X=data.X.astype(np.float32),
        var=data.var.copy(),
        obsm={"spatial_3D": np.asarray(data.obsm["spatial_3D"], dtype=np.float32)}
        if task in {"T2", "T3"} else {},
        uns=data.uns.copy(),
    )
    check_data(result, spatial=task in {"T2", "T3"})
    return result
