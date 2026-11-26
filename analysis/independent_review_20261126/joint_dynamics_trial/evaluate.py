"""Read the target only after asserting all early-stage forecast artifacts frozen."""
import time
import json
import hashlib
import joblib
import anndata as ad
import pandas as pd
from threadpoolctl import threadpool_limits
from core import OUT,PREV,ROOT,SEEDS,prior,np,save,decode

KEYS=['pooled_rmse','fixed_within_rmse','cosine','mmd2','covariance_rel']

def score(x,z,g,w,src,ev,kernel,cross,actual=False):
    mean=w@x;d=mean-ev.ref
    centered=z-w@z;cov=(centered.T*w)@centered
    tcov=ev.cov if actual else ev.cov*(sum(ev.counts)-1)/sum(ev.counts)
    if actual:cov*=len(x)/(len(x)-1)
    p=np.bincount(g,weights=w,minlength=12)
    assert (p>0).all(),'Lost an entire source group; cannot silently omit within-group errors'
    means=np.stack([w[g==k]@x[g==k]/p[k] for k in range(12)])
    mse=np.mean((means-ev.target)**2,axis=1)
    xx=(w@kernel@w-w@w)/(1-w@w)
    mmd=xx+np.array(ev.yy)-2*cross@w
    output=ev.km.predict(z);op=np.bincount(output,weights=w,minlength=12)
    _,inv=np.unique(src,return_inverse=True);raww=np.bincount(inv,weights=w)
    return dict(pooled_rmse=float(np.sqrt(np.mean((mean-ev.weights@ev.target)**2))),
        fixed_within_rmse=float(np.sqrt(ev.weights@mse)),fixed_equal_rmse=float(np.sqrt(mse.mean())),
        cosine=float(d@ev.actual/(np.linalg.norm(d)*np.linalg.norm(ev.actual))),mmd2=float(mmd.mean()),mmd2_batches=mmd.tolist(),
        covariance_rel=float(np.linalg.norm(cov-tcov)/np.linalg.norm(tcov)),
        source_composition_tv=float(abs(p-ev.weights).sum()/2),output_composition_tv=float(abs(op-ev.weights).sum()/2),
        effective_rows=float(1/(w@w)),effective_unique_source_cells=float(1/(raww@raww)),
        group_proportions=p.tolist(),output_group_proportions=op.tolist(),
        zero_fraction=float(w@(x==0).mean(1)),reassigned_mass=float(w@(output!=g)),
        unique_raw_source_cells=int(len(np.unique(src))))

def gate(results,seed):
    arm=f'joint_{seed}_h1';checks={}
    for view in ['weighted','materialized']:
        for key in ['mmd2','covariance_rel']:
            delta=np.array([r[view][arm][key]-r[view]['shared'][key] for r in results])
            checks[f'{view}_{key}']=dict(mean_difference=float(delta.mean()),wins=int((delta<0).sum()),passed=bool(delta.mean()<0 and (delta<0).sum()>=4))
        for key in ['pooled_rmse','fixed_within_rmse']:
            ratio=results[0][view][arm][key]/results[0][view]['shared'][key]
            checks[f'{view}_{key}']=dict(ratio=ratio,passed=bool(ratio<=1.01))
        delta=results[0][view][arm]['cosine']-results[0][view]['shared']['cosine']
        checks[f'{view}_cosine']=dict(difference=delta,passed=bool(delta>=-.01))
    return dict(seed=seed,passed=all(c['passed'] for c in checks.values()),checks=checks)

def latent_metrics(z,w,ev):
    centered=z-w@z;cov=(centered.T*w)@centered
    target=ev.cov*(sum(ev.counts)-1)/sum(ev.counts)
    k=ev.kernel(z,z)
    xx=(w@k@w-w@w)/(1-w@w)
    mmd=[xx+yy-2*(w@ev.kernel(z,t).mean(1)) for t,yy in zip(ev.subs,ev.yy)]
    return dict(mmd2=float(np.mean(mmd)),covariance_rel=float(np.linalg.norm(cov-target)/np.linalg.norm(target)))

def main():
    start=time.time();training=json.loads((OUT/'training.json').read_text())
    for key,file in [('frozen_sha256','frozen_predictions.npz'),('representation_sha256','representation.joblib'),('early_latent_sha256','early_latent.npz')]:
        assert prior.digest(OUT/file)==training[key]
    paths=[PREV/'results.json',PREV/'frozen_controls.npz',OUT/'protocol.txt',OUT/'diagnostics_protocol.txt',OUT/'frozen_predictions.npz',OUT/'training.json',OUT/'representation.joblib',OUT/'early_latent.npz',
        ROOT/'data/raw/T1/E9.5_RNA.h5ad',ROOT/'data/p3/T1/E10.5_RNA.h5ad']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    frozen=np.load(OUT/'frozen_predictions.npz');old=np.load(PREV/'frozen_controls.npz')
    space=joblib.load(OUT/'representation.joblib');pca32=space['pca'];z9new=np.load(OUT/'early_latent.npz')['z9']
    evaluator_space=joblib.load(PREV/'composition_forecast_trial/early_space.joblib')
    pca,km=evaluator_space['pca'],evaluator_space['clusters']
    z9=np.load(PREV/'composition_forecast_trial/early_space.npz')['z9']
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    # All forecast model checkpoints, alpha values, weights and row allocations predate this call.
    ev=prior.Evaluator(panel,pca,km,latest,z9)
    archived=json.loads((PREV/'results.json').read_text())['results']
    results=[];checks=[];artifacts=[]
    for b,(src,g) in enumerate(zip(frozen['sources'],frozen['source_groups'])):
        base=latest[src].toarray();uniform=np.full(len(src),1/len(src));identity=np.arange(len(src))
        weighted={};materialized={};decode_diagnostics={};latent={}
        arms=['copy','shared']
        for seed in SEEDS:
            arms += [f'velocity_{seed}_h1',f'joint_{seed}_h1',f'joint_{seed}_h1_uniform',f'joint_{seed}_h1_open',f'velocity_{seed}_h0.5',f'joint_{seed}_h0.5']
        for arm in arms:
            if arm in ['copy','shared']:
                pred=base if arm=='copy' else prior.decode(base,g,old['recent_means'],old[f'b{b}_common_only'])
                w=uniform;order=identity;latent_delta=None
            else:
                prefix=arm.removesuffix('_uniform').removesuffix('_open')
                delta=frozen[prefix+'_future'][b]-z9new[src]
                residual=delta@pca32.components_
                pred=decode(base,residual,arm.endswith('_open'),float(frozen[prefix+'_open_alpha']) if arm.endswith('_open') else 1.)
                lg=frozen[prefix+'_logw'][b].astype(float);w=np.exp(lg);w/=w.sum();order=frozen[prefix+'_order'][b]
                if arm.endswith('_uniform'):w=uniform;order=identity
                latent_delta=delta
            for chunk in np.array_split(pred,20):np.testing.assert_allclose(np.expm1(chunk.astype(float)).sum(1),10000,rtol=1e-5)
            z=prior.project(pred,pca);kernel=ev.kernel(z,z);cross=np.stack([ev.kernel(z,t).mean(1) for t in ev.subs])
            weighted[arm]=score(pred,z,g,w,src,ev,kernel,cross)
            materialized[arm]=score(pred[order],z[order],g[order],uniform,src[order],ev,kernel[np.ix_(order,order)],cross[:,order],True)
            # Scalar diagnostics computed in chunks to avoid retaining dense float64 arrays.
            activated=0;closed=0;ss=0.
            for s in range(0,len(src),128):
                bb=base[s:s+128];pp=pred[s:s+128]
                activated+=np.sum((bb==0)&(pp>0));closed+=np.sum((bb>0)&(pp==0));ss+=np.square(pp.astype(float)-bb).sum()
            diag=dict(activated_entries=int(activated),closed_entries=int(closed),edit_rms=float(np.sqrt(ss/base.size)))
            if latent_delta is not None:
                realized=prior.project(pred,pca32)-z9new[src]
                diag.update(requested_latent_rms=float(np.sqrt(np.mean(latent_delta**2))),realized_latent_rms=float(np.sqrt(np.mean(realized**2))),
                    decoder_relative_residual=float(np.linalg.norm(realized-latent_delta)/np.linalg.norm(latent_delta)))
                intended=z9[src]+latent_delta@(pca32.components_@pca.components_.T)
                latent[arm]=latent_metrics(intended,w,ev)
            else:
                latent[arm]=latent_metrics(z,w,ev)
            decode_diagnostics[arm]=diag
            if arm in ['copy','shared']:
                oldname='copy' if arm=='copy' else 'common_only'
                errors={k:abs(materialized[arm][k]-archived[b][oldname][k]) for k in KEYS}
                for k in KEYS:np.testing.assert_allclose(materialized[arm][k],archived[b][oldname][k],rtol=1e-7,atol=3e-9)
                checks.append(dict(batch=b,arm=arm,max_abs_error=max(errors.values())))
            if b==0 and arm==f'joint_{SEEDS[0]}_h1':
                path=OUT/'joint_primary_E10.5.h5ad'
                a=ad.AnnData(pred[order],obs=pd.DataFrame(index=[f'prediction_{j}' for j in range(2000)]),var=pd.DataFrame(index=panel))
                a.write_h5ad(path,compression='gzip',compression_opts=1)
                check=ad.read_h5ad(path);np.testing.assert_array_equal(check.X,pred[order]);assert check.var_names.tolist()==panel
                artifacts.append(dict(path=str(path),sha256=prior.digest(path),role='Research output; not selected or officially submitted'))
            print(b,arm,{k:round(materialized[arm][k],7) for k in KEYS},flush=True)
            del pred,z,kernel,cross
        results.append(dict(batch=b,weighted=weighted,materialized=materialized,decoder=decode_diagnostics,latent_intended=latent))
        save('results_partial.json',results)
    decisions=[gate(results,seed) for seed in SEEDS]
    unchanged={str(p.relative_to(ROOT)):prior.digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths};assert all(unchanged.values())
    save('results.json',dict(scope='Retrospective E9.5→E10.5; fitted E8.5/E9.5 only. No official score.',
        results=results,gates=decisions,passed=all(d['passed'] for d in decisions),baseline_verification=checks,
        input_sha256=hashes,inputs_unchanged=unchanged,prediction_artifacts=artifacts,seconds=time.time()-start,
        covariance_conventions={'weighted':'ddof=0','materialized':'ddof=1'},target_weights=ev.weights.tolist()))
    print('COMPLETE',round(time.time()-start,1),'seconds',decisions,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
