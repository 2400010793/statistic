"""Author WOT transport maps followed by an explicit linear forecast.

WOT connects observed distributions. The barycentric velocity extrapolation
here is an adaptation for the challenge, not an original WOT prediction API.
"""
from __future__ import annotations

import importlib
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .io import check_data, sha256
from .models import _check_stage, _stage_number


AUTHOR_COMMIT = "ca5e94f05699997b01cf5ae13383f9810f0613f6"
# Canonical sorted path -> SHA256 manifest for all 39 Python files in the
# registered author archive. This trusted digest also detects rewritten or
# incomplete local lockfiles, rather than trusting their self-reported hashes.
AUTHOR_CODE_MANIFEST_SHA256 = "74a91acfe52bd6d6824ac881e5c84bbcc5b506a310d7fa692f13203158f14325"


def _verify_wot_source(root):
    source = Path(root).resolve() / "third_party" / "wot"
    lock_path = source / ".source-lock.json"
    if not lock_path.is_file() or not (source / "wot" / "__init__.py").is_file():
        raise ValueError("WOT locked source missing; run models install WOT under this root")
    lock = json.loads(lock_path.read_text())
    if lock.get("commit") != AUTHOR_COMMIT:
        raise ValueError("WOT adapter requires the registered locked author commit")
    all_hashes = lock.get("file_sha256")
    if not isinstance(all_hashes, dict):
        raise ValueError("WOT source hash lock is missing or incomplete; reacquire the locked source")
    hashes = {
        key: value for key, value in all_hashes.items()
        if key.startswith("wot/") and key.endswith(".py")
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != AUTHOR_CODE_MANIFEST_SHA256:
        raise ValueError("WOT source hash lock differs from the registered archive; reacquire the locked source")
    actual_files = {path.relative_to(source).as_posix() for path in (source / "wot").rglob("*.py")}
    if actual_files != set(hashes):
        raise ValueError("WOT Python files differ from the complete source lock")
    for filename, expected in hashes.items():
        if sha256(source / filename) != expected:
            raise ValueError(f"WOT author source SHA256 mismatch: {filename}")
    return source


def _load_wot(root):
    source = _verify_wot_source(root)
    existing = sys.modules.get("wot")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Another WOT version is already imported; restart with the locked source")
    old_path = list(sys.path)
    previously_loaded = {name for name in sys.modules if name == "wot" or name.startswith("wot.")}
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("wot")
    except ImportError as exc:
        for name in list(sys.modules):
            if (name == "wot" or name.startswith("wot.")) and name not in previously_loaded:
                del sys.modules[name]
        raise RuntimeError(
            "Author WOT or a dependency is unavailable. Run `python -m "
            "virtual_embryo_benchmark models install WOT`, then install "
            "POT and matplotlib in the benchmark environment: "
            "`python -m pip install POT matplotlib`. No substitute solver is used."
        ) from exc
    finally:
        sys.path[:] = old_path
    if not hasattr(module, "ot") or not hasattr(module.ot, "OTModel"):
        raise RuntimeError("The imported wot package does not expose the author OTModel API")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Imported WOT is outside the verified author source")
    return module, AUTHOR_COMMIT


def predict_wot(
    source,
    history,
    *,
    root=".",
    source_stage,
    history_stage,
    target_stage,
    seed=0,
    n_components=30,
    epsilon=0.1,
    n_iter=300,
    checkpoint_path=None,
):
    """Estimate author WOT coupling, then extrapolate ancestral mean velocity.

    Input rows have no longitudinal pairing. The WOT coupling determines an
    ancestral expression mean per source cell. Source composition is retained;
    growth priors are uniformly one, with author unbalanced marginal penalties.
    ``checkpoint_path`` saves the real observed-stage transport map as H5AD.
    ``seed`` records the surrounding experiment; author PCA uses its own fixed
    random_state=58951 and this adapter introduces no random cell matching.
    """
    check_data(source)
    check_data(history)
    _check_stage(source, source_stage, "stage", "source")
    _check_stage(history, history_stage, "stage", "history")
    first_stage = _stage_number(history_stage)
    last_stage = _stage_number(source_stage)
    future_stage = _stage_number(target_stage)
    if not first_stage < last_stage < future_stage:
        raise ValueError("WOT extrapolation requires history_stage < source_stage < target_stage")
    if not np.isfinite(epsilon) or epsilon <= 0 or n_iter < 1 or n_components < 1:
        raise ValueError("epsilon, n_iter and n_components must be positive")
    if source.n_obs > 5000 or history.n_obs > 5000:
        raise ValueError("WOT adapter is limited to 5000 cells per snapshot; subsample first")
    indices = history.var_names.get_indexer(source.var_names)
    if (indices < 0).any():
        raise ValueError("history is missing source genes; cannot zero-fill")
    dense_bytes = (source.n_obs + history.n_obs) * source.n_vars * 4
    if dense_bytes > 512 * 1024 ** 2:
        raise ValueError("WOT dense training expression exceeds 512 MiB; subsample first")
    early_matrix = history.X[:, indices]
    first = early_matrix.toarray() if sparse.issparse(early_matrix) else np.asarray(early_matrix)
    last = source.X.toarray() if sparse.issparse(source.X) else np.asarray(source.X)
    first = first.astype(np.float32, copy=False)
    last = last.astype(np.float32, copy=False)
    count = min(int(n_components), source.n_vars, source.n_obs + history.n_obs - 1)
    day = np.concatenate([
        np.full(history.n_obs, first_stage), np.full(source.n_obs, last_stage)
    ])
    cells = [f"history_{index}" for index in range(history.n_obs)]
    cells.extend(f"source_{index}" for index in range(source.n_obs))
    training = ad.AnnData(
        X=np.concatenate([first, last]),
        obs=pd.DataFrame({"day": day}, index=cells),
        var=source.var.copy(),
    )
    wot, commit = _load_wot(root)
    model = wot.ot.OTModel(
        training, day_field="day", solver="fixed_iters", local_pca=count,
        epsilon=float(epsilon), epsilon0=float(epsilon),
        scaling_iter=int(n_iter), extra_iter=0, growth_iters=1,
    )
    transport = model.compute_transport_map(first_stage, last_stage)
    if transport is None or transport.shape != (history.n_obs, source.n_obs):
        raise RuntimeError("Author WOT returned no valid observed-stage transport map")
    coupling = transport.X.toarray() if sparse.issparse(transport.X) else np.asarray(transport.X)
    if not np.isfinite(coupling).all() or (coupling < 0).any():
        raise RuntimeError("Author WOT transport is nonfinite or negative; adjust solver settings")
    masses = coupling.sum(axis=0)
    if (masses <= 0).any():
        raise RuntimeError("Author WOT transport leaves a source cell with zero ancestral mass")
    ancestors = (coupling.T @ first) / masses[:, None]
    time_ratio = (future_stage - last_stage) / (last_stage - first_stage)
    predicted = np.maximum(last + time_ratio * (last - ancestors), 0).astype(np.float32)
    if not np.isfinite(predicted).all():
        raise RuntimeError("WOT linear forecast is nonfinite; reduce the extrapolation horizon")
    metadata = {
        "method": "wot_linear", "task": "T1", "seed": int(seed),
        "history_stage": first_stage, "source_stage": last_stage,
        "target_stage": future_stage, "n_components": count,
        "epsilon": float(epsilon), "n_iter": int(n_iter),
        "source_commit": commit, "solver_api": "wot.ot.OTModel.compute_transport_map",
        "author_code_manifest_sha256": AUTHOR_CODE_MANIFEST_SHA256,
        "author_pca_seed": 58951, "experiment_seed_role": "provenance_only",
        "growth_prior": "uniform_one", "forecast": "expression_barycentric_linear",
        "note": "Author WOT observed-stage coupling plus local linear extrapolation; not a full paper reproduction",
    }
    output = ad.AnnData(X=predicted, obs=source.obs.copy(), var=source.var.copy())
    output.obs["stage"] = future_stage
    output.uns["baseline"] = metadata
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        transport.uns["adapter"] = metadata
        transport.write_h5ad(path)
    return output
