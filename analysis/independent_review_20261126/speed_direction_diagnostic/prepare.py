"""Preprocess fixed representation and freeze all diagnostic cell splits."""
import joblib
from threadpoolctl import threadpool_limits
from helpers import OUT,REVIEW,JOINT,ROOT,prior,np,save

def main():
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    space=joblib.load(JOINT/'representation.joblib');early=np.load(JOINT/'early_latent.npz')
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel)
    z10=prior.project(x,space['pca']);del x
    original=np.load(JOINT/'frozen_predictions.npz');sources=original['sources'];groups=original['source_groups']
    rng=np.random.default_rng(2026112700)
    available=np.setdiff1d(np.arange(len(early['z9'])),np.unique(sources));perm=rng.permutation(available);n=int(.8*len(perm))
    train9=perm[:n];val9=perm[n:]
    halves=np.array_split(rng.permutation(len(z10)),2)
    arrays=dict(z9=early['z9'],z10=z10,sources=sources,source_groups=groups,train9=train9,valid9=val9)
    reports=[]
    for fold in range(2):
        held=halves[fold];cal=rng.permutation(halves[1-fold]);n=int(.8*len(cal))
        fit,valid=cal[:n],cal[n:]
        arrays.update({f'f{fold}_train10':fit,f'f{fold}_valid10':valid,f'f{fold}_test10':held,
            f'f{fold}_kernel_subset':np.arange(min(2000,len(held)))})
        assert not len(np.intersect1d(held,np.r_[fit,valid]))
        reports.append(dict(fold=fold,train10=len(fit),valid10=len(valid),test10=len(held)))
    assert not len(np.intersect1d(np.r_[train9,val9],sources.ravel()))
    np.savez_compressed(OUT/'data_splits.npz',**arrays)
    paths=[JOINT/'representation.joblib',JOINT/'early_latent.npz',JOINT/'frozen_predictions.npz',
        ROOT/'data/p3/T1/E10.5_RNA.h5ad',ROOT/'data/raw/T1/E9.5_RNA.h5ad',REVIEW/'frozen_controls.npz',OUT/'protocol.txt']
    save('split_manifest.json',dict(seed=2026112700,folds=reports,train9=len(train9),valid9=len(val9),
        excluded_unique_source_cells=len(np.unique(sources)),source_disjoint=True,target_outer_disjoint=True,
        data_sha256=prior.digest(OUT/'data_splits.npz'),input_sha256={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}))
    print('Prepared',reports,'source fit',len(train9),'validation',len(val9),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
