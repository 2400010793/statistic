"""Two-snapshot T1 adapter using the locked author's PRESCIENT classes.

This adapter uses AutoGenerator, OTLoss and AutoGenerator._step from the author
source. It fits a training-only PCA and a stochastic potential model with
uniform population weights. It omits the paper's contrastive pretraining,
growth estimation and multi-timepoint experiments, so it is an adapted training
run rather than a full paper reproduction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from .io import check_data, sha256
from .models import _check_stage, _stage_number


AUTHOR_COMMIT = "50971c7d495e8763eaa60f83af91f51555ed7ece"
AUTHOR_HASHES = {
    "model.py": "ec37d1b3bacfce6c2deb58b07bf5d4f6fb80e2e3e101452f7e15301761e39b20",
    "util.py": "97b55ea1979cd71b6116ec04db9c178f4c62f02372d37acc43d3301933f9b141",
}
MAX_WORKING_BYTES = 4 * 1024 ** 3


def _dense_working_estimate(n_history, n_source, n_genes):
    """Conservative float32 input/concatenation/PCA/output array budget.

    This is an allocation guard, not an RSS prediction: libraries, the model,
    sparse inputs and the operating system require additional memory.
    """
    return (n_history + n_source) * n_genes * np.dtype(np.float32).itemsize * 4


def _author_module(root):
    source = Path(root).resolve() / "third_party" / "prescient"
    lock_path = source / ".source-lock.json"
    if not lock_path.exists():
        raise ValueError("PRESCIENT source missing; run models install prescient under this root.")
    lock = json.loads(lock_path.read_text())
    if lock.get("commit") != AUTHOR_COMMIT:
        raise ValueError("PRESCIENT adapter requires the registered locked author commit.")
    train_path = source / "prescient" / "train"
    for filename, expected in AUTHOR_HASHES.items():
        path = train_path / filename
        if not path.exists() or sha256(path) != expected:
            raise ValueError(f"PRESCIENT author source SHA256 mismatch: {filename}.")
    namespace = "_virtual_embryo_prescient_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    model_name = namespace + ".train.model"
    if model_name in sys.modules:
        return sys.modules[model_name]
    for name, path in [(namespace, source / "prescient"), (namespace + ".train", train_path)]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    try:
        for filename in ["util", "model"]:
            name = namespace + ".train." + filename
            spec = importlib.util.spec_from_file_location(name, train_path / (filename + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise ValueError(
            "PRESCIENT adapter requires torch, geomloss and tqdm in the project environment."
        ) from exc
    return sys.modules[model_name]


def _data_fingerprint(matrix, genes):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(str, genes)), ensure_ascii=False).encode())
    for offset in range(0, len(matrix), 128):
        digest.update(np.ascontiguousarray(matrix[offset:offset + 128], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _step_sizes(duration, dt):
    count = math.ceil(duration / dt)
    return [min(dt, duration - index * dt) for index in range(count)]


def predict_prescient(
    source: ad.AnnData,
    history: ad.AnnData,
    *,
    root=".",
    source_stage,
    history_stage,
    target_stage,
    seed=0,
    epochs=20,
    n_components=30,
    device="cpu",
    checkpoint_path=None,
    hidden_dim=64,
    batch_size=64,
    dt=0.1,
    noise_sd=0.1,
    learning_rate=0.01,
    sinkhorn_blur=0.1,
    sinkhorn_scaling=0.7,
) -> ad.AnnData:
    """Train the author's stochastic potential on two snapshots and forecast T1.

    Ages use embryonic days and can be numerical or ``E9.5`` labels. All PCA,
    standardization and loss fitting use history/source only. Future dynamics
    are simulated from every source cell with the same learned potential and
    diffusion. Source expression outside PCA is retained; negative predicted
    log expression is clipped. ``checkpoint_path`` saves a CPU state_dict,
    PCA/normalization, source hashes, training losses and simulation settings.
    """
    check_data(source)
    check_data(history)
    _check_stage(source, source_stage, "stage", "source")
    _check_stage(history, history_stage, "stage", "history")
    history_time = _stage_number(history_stage)
    source_time = _stage_number(source_stage)
    target_time = _stage_number(target_stage)
    if not history_time < source_time < target_time:
        raise ValueError("PRESCIENT requires history_stage < source_stage < target_stage.")
    if any(not isinstance(value, (int, np.integer)) or value < 1 for value in
           [epochs, n_components, hidden_dim, batch_size]):
        raise ValueError("epochs, n_components, hidden_dim and batch_size must be positive integers.")
    if not all(np.isfinite(value) for value in [dt, noise_sd, learning_rate, sinkhorn_blur, sinkhorn_scaling]):
        raise ValueError("Simulation and loss parameters must be finite.")
    if dt <= 0 or noise_sd < 0 or learning_rate <= 0 or sinkhorn_blur <= 0 or not 0 < sinkhorn_scaling < 1:
        raise ValueError("Invalid simulation or Sinkhorn settings.")
    if source.n_obs > 5000 or history.n_obs > 5000:
        raise ValueError("PRESCIENT adapter is limited to 5000 cells per snapshot; subsample first.")
    estimated_bytes = _dense_working_estimate(history.n_obs, source.n_obs, source.n_vars)
    if estimated_bytes > MAX_WORKING_BYTES:
        raise ValueError("PRESCIENT dense PCA working estimate exceeds 4 GiB; subsample cells first.")
    train_duration, horizon = source_time - history_time, target_time - source_time
    if math.ceil(train_duration / dt) > 512 or math.ceil(horizon / dt) > 2048:
        raise ValueError("Too many stochastic steps; increase dt or reduce the forecast horizon.")
    indices = history.var_names.get_indexer(source.var_names)
    if (indices < 0).any():
        raise ValueError("history is missing source genes; cannot zero-fill.")
    author = _author_module(root)
    import torch

    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("PRESCIENT adapter supports cpu or cuda devices.")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA but PyTorch cannot see a CUDA device.")
    first_raw = history.X[:, indices]
    first = first_raw.toarray() if sparse.issparse(first_raw) else np.asarray(first_raw)
    last = source.X.toarray() if sparse.issparse(source.X) else np.asarray(source.X)
    first, last = first.astype(np.float32, copy=False), last.astype(np.float32, copy=False)
    training = np.concatenate([first, last])
    count = min(n_components, source.n_vars, len(training) - 1)
    pca = PCA(n_components=count, svd_solver="randomized", random_state=seed).fit(training)
    del training, first_raw
    early_z, last_z = pca.transform(first), pca.transform(last)
    scale = np.maximum(np.std(np.concatenate([early_z, last_z]), axis=0), 1e-3).astype(np.float32)
    configuration = SimpleNamespace(
        x_dim=count, k_dim=hidden_dim, layers=1, activation="softplus",
        sinkhorn_blur=sinkhorn_blur, sinkhorn_scaling=sinkhorn_scaling,
    )
    losses = []
    cuda_devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = author.AutoGenerator(configuration).to(torch_device)
        objective = author.OTLoss(configuration, torch_device)
        early = torch.as_tensor(early_z / scale, device=torch_device)
        current = torch.as_tensor(last_z / scale, device=torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for _ in range(epochs):
            early_rows = torch.randperm(len(early), device=torch_device)[:batch_size]
            current_rows = torch.randperm(len(current), device=torch_device)[:batch_size]
            simulated = early[early_rows].clone()
            target = current[current_rows]
            for step in _step_sizes(train_duration, dt):
                noise = torch.randn_like(simulated) * noise_sd
                simulated = model._step(simulated, dt=step, z=noise)
            early_weights = torch.full((len(simulated),), 1 / len(simulated), device=torch_device)
            target_weights = torch.full((len(target),), 1 / len(target), device=torch_device)
            loss = objective(early_weights, simulated, target_weights, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite PRESCIENT training loss; reduce learning_rate/noise_sd.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        position = current.detach().clone()
        for step in _step_sizes(horizon, dt):
            with torch.enable_grad():
                noise = torch.randn_like(position) * noise_sd
                position = model._step(position, dt=step, z=noise).detach()
        predicted_z = position.cpu().numpy() * scale
    predicted = np.maximum(last + (predicted_z - last_z) @ pca.components_, 0).astype(np.float32)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite PRESCIENT forecast; reduce training/noise/horizon settings.")
    metadata = {
        "method": "prescient", "task": "T1", "seed": seed, "epochs": epochs,
        "author_commit": AUTHOR_COMMIT,
        "author_model_class": "prescient.train.model.AutoGenerator",
        "author_source_sha256": AUTHOR_HASHES.copy(),
        "history_stage": history_time, "source_stage": source_time, "target_stage": target_time,
        "n_components": count, "hidden_dim": hidden_dim, "batch_size": batch_size,
        "dense_working_estimate_bytes": estimated_bytes,
        "dt": dt, "noise_sd": noise_sd, "learning_rate": learning_rate,
        "sinkhorn_blur": sinkhorn_blur, "sinkhorn_scaling": sinkhorn_scaling,
        "growth_weights": "uniform", "contrastive_pretraining": False,
        "training_loss": np.asarray(losses, dtype=float),
        "note": "Real author AutoGenerator/OTLoss/_step; adapted two-snapshot T1, not full paper reproduction",
    }
    output = ad.AnnData(X=predicted, obs=source.obs.copy(), var=source.var.copy())
    output.obs["stage"] = target_time
    output.uns["baseline"] = metadata
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata["training_loss"] = torch.tensor(losses)
        torch.save({
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": vars(configuration), "metadata": checkpoint_metadata,
            "pca_mean": torch.from_numpy(pca.mean_.copy()),
            "pca_components": torch.from_numpy(pca.components_.copy()),
            "latent_scale": torch.from_numpy(scale),
            "gene_names": list(map(str, source.var_names)),
            "history_fingerprint": _data_fingerprint(first, source.var_names),
            "source_fingerprint": _data_fingerprint(last, source.var_names),
        }, checkpoint)
    return output
