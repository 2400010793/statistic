"""Materialize latest-stage proportions as equally weighted T1 prediction rows."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import argparse
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('research_root', type=Path)
parser.add_argument('output_dir', type=Path)
args = parser.parse_args()
ROOT = args.research_root.resolve()
SAVE = args.output_dir.resolve()
SAVE.mkdir(parents=True, exist_ok=True)
N_GROUPS = 12
SEED = 20261122

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def log_normalize(counts):
    totals = counts.sum(1, keepdims=True)
    assert totals.min() > 0
    return np.log1p(counts * (10000 / totals)).astype(np.float32)

BUDGET = 2000


def read(path):
    return json.loads(path.read_text())


def array(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def normalize_log(matrix):
    return log_normalize(np.expm1(array(matrix).astype(float)))


def row_indices(groups, latest):
    counts = np.bincount(groups, minlength=N_GROUPS)
    shares = latest * (counts > 0)
    shares /= shares.sum()
    exact = shares * BUDGET
    quotas = np.floor(exact).astype(int)
    order = np.argsort(-(exact - quotas), kind='stable')
    quotas[order[:BUDGET - quotas.sum()]] += 1
    rng = np.random.default_rng(SEED)
    rows = []
    for k in np.flatnonzero(quotas):
        pool = rng.permutation(np.flatnonzero(groups == k))
        rows.extend(np.resize(pool, quotas[k]))
    rows = rng.permutation(rows)
    assert np.array_equal(np.bincount(groups[rows], minlength=N_GROUPS), quotas)
    return rows, dict(before=counts.tolist(), latest=latest.tolist(), quotas=quotas.tolist(),
                      lost_mass=float(latest[counts == 0].sum()), distinct_parent_rows=len(np.unique(rows)))


def pack(name, values, genes, groups, latest):
    rows, stats = row_indices(groups, latest)
    path = SAVE / (name + '.h5ad')
    obs = pd.DataFrame(index=[f'prediction_{i}' for i in range(BUDGET)])
    output = ad.AnnData(X=values[rows], obs=obs, var=pd.DataFrame(index=genes))
    output.write_h5ad(path, compression='gzip', compression_opts=1)
    check = ad.read_h5ad(path)
    assert check.var_names.tolist() == list(genes) and check.var_names.is_unique
    assert check.obs_names.is_unique and not len(check.obs.columns) and check.shape==(BUDGET,len(genes))
    assert check.X.dtype == np.float32 and check.X.min() >= 0 and np.isfinite(check.X).all()
    for i in range(0, BUDGET, 128):
        chunk = check.X[i:i+128]
        assert np.array_equal(chunk, values[rows[i:i+128]])
        assert np.allclose(np.expm1(chunk.astype(float)).sum(1), 10000, rtol=1e-6)
    np.savez_compressed(SAVE / (name + '_indices.npz'), row=rows, parent_group=groups)
    return dict(**stats, file=path.name, shape=list(check.shape), sha256=digest(path))


def challenge():
    folder = ROOT / 'analysis/results/population_direction_round40'
    original = folder / 'neural_population_direction_E10.5.h5ad'
    source_hash = digest(original)
    assert source_hash == read(folder/'neural_population_direction_format.json')['sha256']
    spec = read(ROOT/'panels/index.json')['T1:val']
    genes = (ROOT/'panels'/spec['genes_file']).read_text().splitlines()
    base = ad.read_h5ad(original)
    assert base.var_names.tolist()==genes and spec['min_cells']<= BUDGET <=spec['max_cells']
    rng = np.random.default_rng(SEED)
    train = []
    inputs = {}
    for stage in ['E8.5', 'E9.5']:
        path = ROOT/f'data/raw/T1/{stage}_RNA.h5ad'
        inputs[stage] = digest(path)
        data = ad.read_h5ad(path)
        assert data.var_names.tolist() == genes
        train.append(normalize_log(data.X[rng.choice(data.n_obs,1500,replace=False)]))
    train = np.vstack(train)
    basis = PCA(20,svd_solver='randomized',random_state=SEED).fit(train)
    cluster = KMeans(N_GROUPS,n_init=10,random_state=SEED).fit(basis.transform(train))
    def labels(x):
        return np.concatenate([cluster.predict(basis.transform(normalize_log(x[i:i+256]))) for i in range(0,x.shape[0],256)])
    latest = np.bincount(labels(data.X),minlength=N_GROUPS)/data.n_obs
    return dict(**pack('r40_carry_E10.5',base.X,genes,labels(base.X),latest),
                parent_sha256=source_hash, training_inputs=inputs, panel_key=spec['key'])


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        print(json.dumps(challenge(), indent=2))
