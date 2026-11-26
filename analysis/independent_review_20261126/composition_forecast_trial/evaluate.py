"""Evaluate frozen forecast rules; include exact weights and actual 2,000-row outputs."""
import argparse
import json
import time
import hashlib
import joblib
import numpy as np
import anndata as ad
import pandas as pd
from threadpoolctl import threadpool_limits
from core import OUT,PREV,ROOT,prior,save,weights_for,materialize,gates

def prepare_weighted(x,z,ev):
    kernel=ev.kernel(z,z)
    cross=np.stack([ev.kernel(z,y).mean(1) for y in ev.subs])
    return kernel,cross

def weighted_score(x,z,g,w,src,prepared,ev):
    kernel,cross=prepared
    mean=w@x; change=mean-ev.ref
    u=w@z; centered=z-u
    cov=(centered.T*w)@centered
    target_cov=ev.cov*(sum(ev.counts)-1)/sum(ev.counts)
    masses=np.bincount(g,weights=w,minlength=12)
    group_means=np.stack([w[g==k]@x[g==k]/masses[k] for k in range(12)])
    mse=np.mean((group_means-ev.target)**2,axis=1)
    xx=float((w@kernel@w-w@w)/(1-w@w))
    mmd=xx+np.array(ev.yy)-2*(cross@w)
    _,inverse=np.unique(src,return_inverse=True)
    rawmass=np.bincount(inverse,weights=w)
    return dict(pooled_rmse=float(np.sqrt(np.mean((mean-ev.weights@ev.target)**2))),
        fixed_within_rmse=float(np.sqrt(ev.weights@mse)),fixed_equal_rmse=float(np.sqrt(mse.mean())),
        cosine=float(change@ev.actual/(np.linalg.norm(change)*np.linalg.norm(ev.actual))),
        covariance_rel=float(np.linalg.norm(cov-target_cov)/np.linalg.norm(target_cov)),
        mmd2=float(mmd.mean()),mmd2_batches=mmd.tolist(),composition_tv=float(abs(masses-ev.weights).sum()/2),
        group_proportions=masses.tolist(),effective_rows=float(1/(w@w)),effective_unique_source_cells=float(1/(rawmass@rawmass)),
        weight_min=float(w.min()),weight_max=float(w.max()))

def main(phase):
    start=time.time()
    fit=json.loads((OUT/f'{phase}_fit.json').read_text())
    assert prior.digest(OUT/f'{phase}_frozen.npz')==fit['parameters_sha256']
    assert prior.digest(OUT/'early_space.npz')==fit['early_space_sha256']
    paths=[OUT/f'{phase}_frozen.npz',OUT/f'{phase}_fit.json',OUT/'early_space.npz',OUT/'early_space.joblib',
        OUT/'protocol.txt',OUT/'evaluate.py',OUT/'core.py',PREV/'frozen_controls.npz',PREV/'results.json',
        ROOT/'data/raw/T1/E9.5_RNA.h5ad',ROOT/'data/p3/T1/E10.5_RNA.h5ad']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    frozen=np.load(PREV/'frozen_controls.npz'); sources=frozen['sources']; groups=frozen['source_groups']
    early=np.load(OUT/'early_space.npz'); trend=np.load(OUT/'trend_frozen.npz'); carry=trend['carry']
    if phase=='density':density=np.load(OUT/'density_frozen.npz')
    weights=[]; orders=[]; reconstruction={}
    for b,(src,g) in enumerate(zip(sources,groups)):
        q=dict(carry=carry); ratios={}
        if phase=='trend':q['trend']=trend['trend']
        else:
            q.update(density_full=density['density_shares'],density_group_only=density['density_shares'],density_within_only=carry)
            ratios={n:density['late_ratio'][src] for n in ['density_full','density_within_only']}
        w={n:weights_for(g,share,ratios.get(n)) for n,share in q.items()}
        order={n:materialize(g,q[n],v,b) for n,v in w.items()}
        w['original_shared']=np.full(len(src),1/len(src));order['original_shared']=np.arange(len(src))
        for n in w:
            reconstruction[f'b{b}_{n}_weights']=w[n]
            reconstruction[f'b{b}_{n}_pool_rows']=order[n]
            reconstruction[f'b{b}_{n}_raw_source_indices']=src[order[n]]
        weights.append(w);orders.append(order)
    # Every weighting and integer allocation is frozen before target loading.
    np.savez_compressed(OUT/f'{phase}_realizations.npz',sources=sources,source_groups=groups,**reconstruction)
    realization_hash=prior.digest(OUT/f'{phase}_realizations.npz')
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    estimators=joblib.load(OUT/'early_space.joblib');pca,km=estimators['pca'],estimators['clusters']
    latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    ev=prior.Evaluator(panel,pca,km,latest,early['z9'])
    archived=json.loads((PREV/'results.json').read_text())['results']
    results=[];checks=[]
    candidate='trend' if phase=='trend' else 'density_full'
    for b,src in enumerate(sources):
        base=latest[src].toarray(); g=groups[b]; copy_mean=base.mean(0,dtype=float)
        common=prior.decode(base,g,frozen['recent_means'],frozen[f'b{b}_common_only'])
        z=prior.project(common,pca);prepared=prepare_weighted(common,z,ev)
        weighted={n:weighted_score(common,z,g,w,src,prepared,ev) for n,w in weights[b].items()}
        materialized={}
        for n,order in orders[b].items():
            pred=common[order]
            assert pred.shape==(2000,len(panel)) and np.isfinite(pred).all() and pred.min()>=0
            for chunk in np.array_split(pred,20):np.testing.assert_allclose(np.expm1(chunk.astype(float)).sum(1),10000,rtol=1e-5)
            rec=ev(pred,g[order],copy_mean)
            p=np.bincount(g[order],minlength=12)/len(order)
            rec.update(composition_tv=float(abs(p-ev.weights).sum()/2),group_proportions=p.tolist(),
                unique_pool_rows=int(len(np.unique(order))),unique_raw_source_cells=int(len(np.unique(src[order]))))
            materialized[n]=rec
        materialized['copy_original']=ev(base,g,copy_mean)
        orig=materialized['original_shared']
        check={k:abs(orig[k]-archived[b]['common_only'][k]) for k in ['pooled_rmse','fixed_within_rmse','cosine','mmd2','covariance_rel']}
        for k in check:np.testing.assert_allclose(orig[k],archived[b]['common_only'][k],rtol=1e-7,atol=1e-9)
        for k in ['pooled_rmse','fixed_within_rmse','cosine','mmd2']:
            np.testing.assert_allclose(weighted['original_shared'][k],orig[k],rtol=1e-7,atol=1e-9)
        checks.append(dict(batch=b,archive_max_abs_error=max(check.values())))
        results.append(dict(batch=b,weighted=weighted,materialized=materialized))
        print(phase,b,'weighted',{n:{k:round(v[k],7) for k in ['composition_tv','covariance_rel','mmd2','pooled_rmse']} for n,v in weighted.items()},flush=True)
        del base,common,z,prepared
    decision=gates(results,candidate,phase)
    path_out=None
    if decision['passed']:
        src=sources[0];g=groups[0];base=latest[src].toarray()
        common=prior.decode(base,g,frozen['recent_means'],frozen['b0_common_only'])
        pred=common[orders[0][candidate]]
        path=OUT/f'{candidate}_E10.5.h5ad'
        ad.AnnData(pred,obs=pd.DataFrame(index=[f'prediction_{j}' for j in range(2000)]),var=pd.DataFrame(index=panel)).write_h5ad(path,compression='gzip')
        check=ad.read_h5ad(path);np.testing.assert_array_equal(check.X,pred);assert check.var_names.tolist()==panel
        path_out=dict(path=str(path),sha256=prior.digest(path))
    unchanged={str(p.relative_to(ROOT)):prior.digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths};assert all(unchanged.values())
    save(f'{phase}_results.json',dict(phase=phase,scope='Retrospective frozen E8.5/E9.5 forecast replay; E10.5 evaluation only; no official score',
        candidate=candidate,results=results,gate=decision,target_proportions=ev.weights.tolist(),
        verification=checks,realizations_sha256=realization_hash,prediction_artifact=path_out,
        covariance_conventions={'weighted':'population (ddof=0)','materialized':'sample (ddof=1), matching historical evaluator'},
        input_sha256=hashes,inputs_unchanged=unchanged,elapsed_seconds=time.time()-start))
    print('GATE',json.dumps(decision),'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['trend','density']);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.phase)
