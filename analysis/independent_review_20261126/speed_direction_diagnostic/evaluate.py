"""Outer-half evaluation after all calibrations are frozen. No parameter updates."""
import time
import json
import joblib
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits
from helpers import OUT,REVIEW,JOINT,ROOT,old,prior,np,SEEDS,MODES,metrics_latent,save

ARMS=['copy','shared','frozen','clock','reverse']+MODES

def gene_metrics(x,g,w,target,proportions,reference):
    mean=w@x;group_p=np.bincount(g,weights=w,minlength=12)
    group_means=np.stack([w[g==k]@x[g==k]/group_p[k] for k in range(12)])
    mse=np.mean((group_means-target)**2,axis=1);truth=proportions@target
    delta=mean-reference;actual=truth-reference
    return dict(pooled_rmse=float(np.sqrt(np.mean((mean-truth)**2))),
        fixed_within_rmse=float(np.sqrt(proportions@mse)),fixed_equal_rmse=float(np.sqrt(mse.mean())),
        cosine=float(delta@actual/(np.linalg.norm(delta)*np.linalg.norm(actual))),
        source_composition_tv=float(abs(group_p-proportions).sum()/2),source_proportions=group_p.tolist(),
        zero_fraction=float(w@(x==0).mean(1)))

def main():
    start=time.time();fit=json.loads((OUT/'calibration.json').read_text())
    assert prior.digest(OUT/'frozen_predictions.npz')==fit['frozen_sha256']
    assert prior.digest(OUT/'data_splits.npz')==fit['data_sha256']
    paths=[OUT/'calibration.json',OUT/'frozen_predictions.npz',OUT/'protocol.txt',OUT/'evaluation_details.txt',OUT/'evaluate.py']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    d=np.load(OUT/'data_splits.npz');f=np.load(OUT/'frozen_predictions.npz')
    rep=joblib.load(JOINT/'representation.joblib');pc32=rep['pca'];scale=rep['scale'];band32=rep['bandwidth']
    evspace=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib');pc20,km=evspace['pca'],evspace['clusters']
    z9_20=np.load(REVIEW/'composition_forecast_trial/early_space.npz')['z9']
    rng=np.random.default_rng(prior.SEED+21)
    band20=float(np.median(pdist(z9_20[rng.choice(len(z9_20),1000,False)],'sqeuclidean')))
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    truth=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);zt20=prior.project(truth,pc20);gt20=km.predict(zt20)
    targets={}
    for fold in range(2):
        rows=d[f'f{fold}_test10'];p,m=prior.means(truth[rows],gt20[rows])
        targets[fold]=dict(weights=p,means=m,latent20=zt20[rows],latent32=d['z10'][rows]/scale,subset=d[f'f{fold}_kernel_subset'])
    del truth
    latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    reference=np.asarray(latest.sum(0,dtype=float)).ravel()/latest.shape[0]
    controls=np.load(REVIEW/'frozen_controls.npz');mapping=pc32.components_@pc20.components_.T
    records=[];fullgenes=[]
    for batch,(src,g) in enumerate(zip(d['sources'],d['source_groups'])):
        base=latest[src].toarray();common=prior.decode(base,g,controls['recent_means'],controls[f'b{batch}_common_only'])
        baseline32=dict(copy=d['z9'][src],shared=prior.project(common,pc32))
        baseline20=dict(copy=z9_20[src],shared=prior.project(common,pc20))
        for fold in range(2):
            target=targets[fold]
            for seed in SEEDS:
                models={};gene_scores={}
                for arm in ARMS:
                    if arm in ['copy','shared']:
                        future=baseline32[arm];z20=baseline20[arm];w=np.full(2000,1/2000);lw=np.zeros(2000)
                    else:
                        key=f'f{fold}_s{seed}_{arm}';future=f[key+'_future'][batch];lw=f[key+'_logw'][batch].astype(float)
                        w=np.exp(lw-lw.max());w/=w.sum();z20=z9_20[src]+(future-d['z9'][src])@mapping
                    score32=metrics_latent(future/scale,w,target['latent32'],band32,target['subset'])
                    score20=metrics_latent(z20,w,target['latent20'],band20,target['subset'])
                    score32['logweight_near_bound_fraction']=float(np.mean(abs(lw)>=.95*np.log(4))) if arm in MODES else None
                    models[arm]=dict(latent32=score32,latent20=score20)
                    if batch==0:
                        pred=(base if arm=='copy' else common) if arm in ['copy','shared'] else old.decode(base,(future-d['z9'][src])@pc32.components_)
                        metrics=gene_metrics(pred,g,w,target['means'],target['weights'],reference)
                        decoded20=prior.project(pred,pc20)
                        metrics['decoded_latent20']=metrics_latent(decoded20,w,target['latent20'],band20,target['subset'])
                        metrics['decoder_relative_residual']=float(np.linalg.norm(prior.project(pred,pc32)-future)/max(np.linalg.norm(future-d['z9'][src]),1e-12)) if arm not in ['copy','shared'] else None
                        # Support preservation and CP10K, same decoder as the preceding experiment.
                        np.testing.assert_array_equal(base>0,pred>0)
                        for chunk in np.array_split(pred,20):np.testing.assert_allclose(np.expm1(chunk.astype(float)).sum(1),10000,rtol=1e-5)
                        gene_scores[arm]=metrics
                records.append(dict(batch=batch,fold=fold,seed=seed,models=models))
                if batch==0:fullgenes.append(dict(fold=fold,seed=seed,models=gene_scores))
                print('Evaluated',batch,fold,seed,{a:round(v['latent32']['objective'],6) for a,v in models.items()},flush=True)
        save('results_partial.json',dict(records=records,full_gene_main_pool=fullgenes))
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('results.json',dict(scope='Target-calibrated same-stage held-out diagnostic, not temporal forecasting',records=records,full_gene_main_pool=fullgenes,
        input_sha256=hashes,inputs_unchanged=unchanged,seconds=time.time()-start,
        representation='Frozen early-stage PCA32 primary; PCA20 secondary; three RBF bandwidths both; all covariances ddof0',
        metric_weights=dict(mean=.05,covariance=.01),target_counts=[len(d[f'f{fold}_test10']) for fold in range(2)]))
    print('EVALUATION COMPLETE',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
