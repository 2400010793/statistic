"""Independent frozen-artifact replay and preregistered group-specificity ablation.

No imports from historical project scripts. Only writes beside this script.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT/'analysis/results/p3_preparation/modules_substates_replay'
R40 = ROOT/'analysis/results/population_direction_round40'
SEED, K = 20261122, 12

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path, panel):
    a = ad.read_h5ad(path)
    assert list(a.var_names) == panel
    return a.X.tocsr() if sparse.issparse(a.X) else a.X

def means(x, labels):
    counts = np.bincount(labels, minlength=K)
    assert (counts>0).all()
    mu = np.stack([np.asarray(x[labels==k].sum(axis=0, dtype=np.float64)).ravel()/counts[k] for k in range(K)])
    return counts/len(labels), mu

def project(x, pca):
    return np.concatenate([pca.transform(x[i:i+256]) for i in range(0,len(x) if not sparse.issparse(x) else x.shape[0],256)])

def fit_space(panel):
    rng = np.random.default_rng(SEED)
    samples = []
    for s in ('E8.5','E9.5'):
        x = read(ROOT/f'data/raw/T1/{s}_RNA.h5ad',panel)
        block = x[rng.choice(x.shape[0],1500,replace=False)].toarray().astype(float)
        counts = np.expm1(block)
        samples.append(np.log1p(counts*10000/counts.sum(1,keepdims=True)).astype(np.float32))
        del x, block, counts
    train = np.vstack(samples)
    pca = PCA(20,svd_solver='randomized',random_state=SEED).fit(train)
    km = KMeans(K,n_init=10,random_state=SEED).fit(pca.transform(train))
    return pca, km

def decode(base, source_labels, recent_means, shift):
    ratio = np.divide(shift,recent_means,out=np.zeros_like(shift),where=recent_means>0)
    gain = np.clip(1+ratio,0.8,1.2)
    out = base.copy()
    for k in range(K):
        ix = source_labels==k
        counts = np.expm1(base[ix].astype(float)*gain[k])
        out[ix] = np.log1p(counts*(10000/counts.sum(1,keepdims=True)))
    assert np.isfinite(out).all() and out.min()>=0
    np.testing.assert_array_equal(out>0,base>0)
    for block in np.array_split(out,20):
        np.testing.assert_allclose(np.expm1(block.astype(float)).sum(1),10000,rtol=1e-5)
    return out

class Evaluator:
    def __init__(self,panel,pca,km,latest,z9):
        self.pca,self.km=pca,km
        rng=np.random.default_rng(SEED+21)
        self.bandwidth=float(np.median(pdist(z9[rng.choice(len(z9),1000,replace=False)],'sqeuclidean')))
        y=read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel)
        zy=project(y,pca)
        gy=km.predict(zy)
        self.weights,self.target=means(y,gy)
        self.ref=np.asarray(latest.mean(0,dtype=float)).ravel()
        self.actual=self.weights@self.target-self.ref
        self.cov=np.cov(zy.T)
        self.subs=[zy[ix] for ix in np.array_split(rng.permutation(len(zy))[:6000],3)]
        self.yy=[self.off(self.kernel(v,v)) for v in self.subs]
        self.counts=np.bincount(gy,minlength=K)

    def kernel(self,a,b):
        return np.exp(-cdist(a,b,'sqeuclidean')/(2*self.bandwidth))

    @staticmethod
    def off(a):
        return float((a.sum()-np.trace(a))/(len(a)*(len(a)-1)))

    def __call__(self,x,source_labels,copy_mean):
        z=project(x,self.pca)
        output_labels=self.km.predict(z)
        _,m=means(x,output_labels)
        _,fixed=means(x,source_labels)
        group_mse=np.mean((m-self.target)**2,axis=1)
        fixed_mse=np.mean((fixed-self.target)**2,axis=1)
        mean=np.asarray(x.mean(0,dtype=float)).ravel()
        d=mean-self.ref
        xx=self.off(self.kernel(z,z))
        mmd=[xx+b-2*self.kernel(z,t).mean() for b,t in zip(self.yy,self.subs)]
        return dict(pooled_rmse=float(np.sqrt(np.mean((d-self.actual)**2))),
            within_rmse=float(np.sqrt(self.weights@group_mse)),
            within_equal_rmse=float(np.sqrt(group_mse.mean())),
            fixed_within_rmse=float(np.sqrt(self.weights@fixed_mse)),
            fixed_equal_rmse=float(np.sqrt(fixed_mse.mean())),
            cosine=float(d@self.actual/(np.linalg.norm(d)*np.linalg.norm(self.actual))),
            mmd2=float(np.mean(mmd)),mmd2_target_batches=[float(v) for v in mmd],
            covariance_rel=float(np.linalg.norm(np.cov(z.T)-self.cov)/np.linalg.norm(self.cov)),
            group_rmse=np.sqrt(group_mse).tolist(),fixed_group_rmse=np.sqrt(fixed_mse).tolist(),
            reassigned_cells=int(np.sum(output_labels!=source_labels)),
            group_counts=np.bincount(output_labels,minlength=K).tolist(),
            realized_pooled_edit_rms=float(np.sqrt(np.mean((mean-copy_mean)**2))),
            mean_log_row_sum=float(mean.sum()),
            prediction_array_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())

def main():
    started=time.time()
    paths=[ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ('E8.5','E9.5')]+[
        ROOT/'data/p3/T1/E10.5_RNA.h5ad', OLD/'frozen.npz', OLD/'scores.json',
        R40/'forecast_parameters.npz',R40/'neural_population_direction_E10.5.h5ad',
        OUT/'initial_judgment.md',OUT/'protocol.txt',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    pca,km=fit_space(panel)
    print('Independent early PCA/clusters fitted',flush=True)
    frozen=np.load(OLD/'frozen.npz')
    source=frozen['sources']; full=frozen['step']; axes=frozen['module_axes']
    np.testing.assert_array_equal(source[0],np.load(R40/'forecast_parameters.npz')['source_indices'])
    early=read(ROOT/'data/raw/T1/E8.5_RNA.h5ad',panel)
    _,m8=means(early,km.predict(project(early,pca)))
    del early
    latest=read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    z9=project(latest,pca); g9=km.predict(z9)
    _,m9=means(latest,g9)
    delta=(m9-m8)@axes.T
    common=0.5*delta.mean(0)@axes
    reconstructed=0.5*(delta.mean(0)+frozen['reliability']*(delta-delta.mean(0)))@axes
    np.testing.assert_allclose(reconstructed,full,rtol=1e-6,atol=1e-7)
    print('Frozen step verified; freezing all new arms before target evaluation',flush=True)
    shifts={}; shift_audit={}
    for b,ix in enumerate(source):
        w=np.bincount(g9[ix],minlength=K)/len(ix)
        shared=w@full; deviation=full-shared
        arms=dict(modules_full=full,pooled_matched=np.broadcast_to(shared,full.shape),
                  common_only=np.broadcast_to(common,full.shape))
        if b==0:
            energy=float(w@np.sum(deviation**2,axis=1))
            for offset in range(1,K):
                d=np.roll(deviation,offset,axis=0)
                d=d-w@d
                d*=np.sqrt(energy/(w@np.sum(d**2,axis=1)))
                arms[f'mismatch_{offset:02d}']=shared+d
                np.testing.assert_allclose(w@d,0,atol=1e-12)
                np.testing.assert_allclose(w@np.sum(d**2,axis=1),energy,rtol=1e-12)
        shifts[b]=arms
        shift_audit[str(b)]={name:dict(pooled_target_rms=float(np.sqrt(np.mean((w@s)**2))),
            deviation_energy=float(w@np.sum((s-w@s)**2,axis=1))) for name,s in arms.items()}
    np.savez_compressed(OUT/'frozen_controls.npz',sources=source,source_groups=g9[source],recent_means=m9,
        early_means=m8,pca_components=pca.components_,pca_mean=pca.mean_,cluster_centers=km.cluster_centers_,
        **{f'b{b}_{name}':v for b,arms in shifts.items() for name,v in arms.items()})
    save('frozen_controls_audit.json',shift_audit)
    evaluator=Evaluator(panel,pca,km,latest,z9)
    result=[]
    for b,ix in enumerate(source):
        copy=latest[ix].toarray(); g=g9[ix]; copy_mean=copy.mean(0,dtype=float)
        rec={'copy':evaluator(copy,g,copy_mean)}
        if b==0:
            pred=read(R40/'neural_population_direction_E10.5.h5ad',panel)
            rec['r40']=evaluator(pred,g,copy_mean)
            del pred
        for name,shift in shifts[b].items():
            pred=decode(copy,g,m9,shift)
            rec[name]=evaluator(pred,g,copy_mean)
            print(b,name,{k:round(rec[name][k],7) for k in ('within_rmse','fixed_within_rmse','fixed_equal_rmse','cosine')},flush=True)
            del pred
        result.append(rec)
        save('results_partial.json',result)
    archived=json.loads((OLD/'scores.json').read_text())['results']
    verification=[]
    for b,rec in enumerate(result):
        for name,oldname in [('copy','copy'),('modules_full','modules')]+([('r40','r40')] if b==0 else []):
            errors={k:float(np.max(np.abs(np.asarray(rec[name][k])-np.asarray(archived[b][oldname][k]))))
                for k in ('pooled_rmse','within_rmse','cosine','mmd2','covariance_rel','group_rmse')}
            for k in errors:
                np.testing.assert_allclose(rec[name][k],archived[b][oldname][k],rtol=1e-6,atol=1e-9)
            verification.append(dict(batch=b,arm=name,max_absolute_error=errors))
    preservation={str(p.relative_to(ROOT)):digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths}
    assert all(preservation.values())
    report=dict(scope='Retrospective single-transition mechanism diagnostic; no target-fitted prediction parameters; no official score generated.',
        seed=SEED,n_genes=len(panel),source_shape=list(source.shape),source_unique_counts=[len(np.unique(v)) for v in source],
        target_counts=evaluator.counts.tolist(),target_weights=evaluator.weights.tolist(),bandwidth=evaluator.bandwidth,
        frozen_step_max_abs_error=float(np.max(abs(reconstructed-full))),results=result,verification=verification,
        inputs_sha256=hashes,input_files_unchanged=preservation,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,anndata=ad.__version__),
        elapsed_seconds=time.time()-started)
    save('results.json',report)
    print('COMPLETED',round(report['elapsed_seconds'],1),'seconds; all archived metrics reproduced',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
