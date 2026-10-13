"""Small OT extrapolation references, not reproductions of named papers.

PCA and entropic coupling are fitted on training snapshots only. Neither model
uses a future target distribution. Counts/coordinates are copied from source.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from sklearn.decomposition import PCA

from .io import check_data


def entropic_coupling(first, second, *, epsilon=0.1, iterations=300):
    """Balanced Sinkhorn coupling with uniform masses, in the log domain."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("OT endpoints must be 2D matrices with matching feature dimensions")
    if not len(first) or not len(second) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("OT endpoints must be nonempty and finite")
    if not np.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon and iterations must be positive")
    costs = np.maximum(
        (first ** 2).sum(1)[:, None] + (second ** 2).sum(1)[None, :]
        - 2 * first @ second.T, 0,
    )
    scale = np.median(costs[costs > 0]) if (costs > 0).any() else 1.0
    log_kernel = -costs / (scale * epsilon)
    log_a = np.full(len(first), -np.log(len(first)))
    log_b = np.full(len(second), -np.log(len(second)))
    log_v = np.zeros(len(second))
    for _ in range(iterations):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
    coupling = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
    return coupling


def predict_ot(
    source: ad.AnnData,
    history: ad.AnnData,
    *,
    model="ot_linear",
    source_stage,
    history_stage,
    target_stage,
    seed=0,
    n_components=30,
    epsilon=0.1,
    sinkhorn_iterations=300,
    epochs=100,
    hidden_dim=64,
    batch_size=256,
    integration_steps=20,
    device="cpu",
    task="T1",
):
    check_data(source, spatial=task == "T2")
    check_data(history)
    from .models import _check_stage

    _check_stage(source, source_stage, "stage", "source")
    _check_stage(history, history_stage, "stage", "history")
    if not (float(history_stage) < float(source_stage) < float(target_stage)):
        raise ValueError("OT extrapolation requires history_stage < source_stage < target_stage")
    if model not in {"ot_linear", "ot_flow"}:
        raise ValueError(f"Unknown OT model {model}")
    if len(source) > 5000 or len(history) > 5000:
        raise ValueError("Local OT references are limited to 5000 cells per snapshot; subsample first")
    indices = history.var_names.get_indexer(source.var_names)
    if (indices < 0).any():
        raise ValueError("history is missing source genes; cannot zero-fill")
    first_raw = history.X[:, indices]
    first = first_raw.toarray() if sparse.issparse(first_raw) else np.asarray(first_raw)
    last = source.X.toarray() if sparse.issparse(source.X) else np.asarray(source.X)
    first, last = first.astype(np.float32), last.astype(np.float32)
    count = min(int(n_components), source.n_vars, len(first) + len(last) - 1)
    if count < 1:
        raise ValueError("At least one PCA component required")
    pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
    pca.fit(np.concatenate([first, last]))
    early_z, last_z = pca.transform(first), pca.transform(last)
    coupling = entropic_coupling(early_z, last_z, epsilon=epsilon, iterations=sinkhorn_iterations)
    row_residual = float(np.max(np.abs(coupling.sum(1) - 1 / len(early_z))))
    column_residual = float(np.max(np.abs(coupling.sum(0) - 1 / len(last_z))))
    # For each current cell, estimate its ancestral latent position.
    parents = (coupling.T @ early_z) / np.maximum(coupling.sum(0)[:, None], 1e-30)
    dt = float(source_stage) - float(history_stage)
    horizon = float(target_stage) - float(source_stage)
    velocity = (last_z - parents) / dt
    losses = []
    if model == "ot_linear":
        predicted_z = last_z + horizon * velocity
    else:
        if min(epochs, hidden_dim, batch_size, integration_steps) < 1:
            raise ValueError("Flow training and integration parameters must be positive")
        import torch

        torch.manual_seed(seed)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Requested CUDA but PyTorch cannot see a CUDA device")
        network = torch.nn.Sequential(
            torch.nn.Linear(count + 1, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, count),
        ).to(device)
        # Standardization stabilizes training without touching future data.
        scale = np.maximum(np.std(np.concatenate([early_z, last_z]), axis=0), 1e-3)
        start = torch.tensor(parents / scale, dtype=torch.float32, device=device)
        end = torch.tensor(last_z / scale, dtype=torch.float32, device=device)
        movement = end - start
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        for _ in range(epochs):
            rows = torch.randint(len(end), (min(batch_size, len(end)),), device=device)
            time = torch.rand(len(rows), 1, device=device)
            position = start[rows] + time * movement[rows]
            estimate = network(torch.cat([position, time], dim=1))
            loss = ((estimate - movement[rows]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        # Training time 0..1 is the observed interval. Future integration starts at 1.
        span = horizon / dt
        delta = span / integration_steps
        position = end.clone()
        network.eval()
        with torch.no_grad():
            for step in range(integration_steps):
                time = torch.full((len(position), 1), 1 + step * delta, device=device)
                half = position + 0.5 * delta * network(torch.cat([position, time], 1))
                position += delta * network(torch.cat([half, time + 0.5 * delta], 1))
        predicted_z = position.cpu().numpy() * scale
    # Preserve source residuals outside PCA; apply only the predicted latent change.
    predicted = np.maximum(last + (predicted_z - last_z) @ pca.components_, 0).astype(np.float32)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite OT forecast; reduce extrapolation/training settings")
    output = ad.AnnData(X=predicted, obs=source.obs.copy(), var=source.var.copy())
    output.obs["stage"] = float(target_stage)
    if task == "T2":
        output.obsm["spatial_3D"] = np.asarray(source.obsm["spatial_3D"]).copy()
    output.uns["baseline"] = {
        "method": model, "task": task, "seed": seed,
        "history_stage": float(history_stage), "source_stage": float(source_stage),
        "target_stage": float(target_stage), "n_components": count,
        "epsilon": epsilon, "sinkhorn_iterations": sinkhorn_iterations,
        "ot_row_mass_residual": row_residual, "ot_column_mass_residual": column_residual,
        "training_loss": np.asarray(losses, dtype=float),
        "note": "Local training-snapshot OT reference; spatial coordinates copied; not a paper reproduction",
    }
    return output
