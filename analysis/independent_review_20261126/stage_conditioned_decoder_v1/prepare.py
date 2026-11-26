"""Freeze three-stage splits, early representation and calibration-only moments."""
import json
import joblib
from threadpoolctl import threadpool_limits
from model import OUT,REVIEW,JOINT,DIAG,ROOT,prior,np,save

def mean(x,rows):return np.asarray(x[rows].sum(0,dtype=float)).ravel()/len(rows)

def main():
    d=np.load(DIAG/'data_splits.npz');early=np.load(JOINT/'early_latent.npz')
    groups=np.load(REVIEW/'composition_forecast_trial/early_space.npz')
    arrays={k:d[k] for k in d.files};arrays.update(z8=early['z8'],train8=early['train0'],valid8=early['valid0'][:1024],test8=early['valid0'][1024:3024],
        test9=np.unique(d['sources']),groups8=groups['g8'],groups9=groups['g9'])
    assert len(arrays['test8'])==2000
    for stage in ['8','9']:
        assert not len(np.intersect1d(arrays['train'+stage],arrays['valid'+stage]))
        assert not len(np.intersect1d(arrays['train'+stage],arrays['test'+stage]))
        assert not len(np.intersect1d(arrays['valid'+stage],arrays['test'+stage]))
    np.savez_compressed(OUT/'data_splits.npz',**arrays)
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines();mom={}
    for stage,path in [('9',ROOT/'data/raw/T1/E9.5_RNA.h5ad'),('10',ROOT/'data/p3/T1/E10.5_RNA.h5ad')]:
        x=prior.read(path,panel)
        keys=['train9','valid9'] if stage=='9' else [f'f{fold}_{part}10' for fold in range(2) for part in ['train','valid']]
        for key in keys:mom[key+'_gene_mean']=mean(x,arrays[key]).astype(np.float32)
        del x
    np.savez_compressed(OUT/'target_moments.npz',**mom)
    paths=[JOINT/'representation.joblib',JOINT/'early_latent.npz',DIAG/'data_splits.npz',REVIEW/'frozen_controls.npz',OUT/'protocol.txt']
    paths += [ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]+[ROOT/'data/p3/T1/E10.5_RNA.h5ad']
    save('preparation.json',dict(scope='Two-interval E10.5-calibrated retrospective study',
        data_sha256=prior.digest(OUT/'data_splits.npz'),moments_sha256=prior.digest(OUT/'target_moments.npz'),
        input_sha256={str(p.relative_to(ROOT)):prior.digest(p) for p in paths},
        counts={k:len(arrays[k]) for k in ['train8','valid8','test8','train9','valid9','test9']},
        split_disjoint_checks=True))
    print('Prepared',[(k,len(arrays[k])) for k in ['train8','valid8','test8','train9','valid9','test9']],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
