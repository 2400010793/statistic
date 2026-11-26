"""Held-out evaluation of the post-hoc growth-bound check, same frozen splits."""
import time
import json
import joblib
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits
from helpers import OUT,REVIEW,JOINT,ROOT,old,prior,np,SEEDS,metrics_latent,save
import importlib.util
_spec=importlib.util.spec_from_file_location('speed_direction_evaluator',OUT/'evaluate.py')
_evaluation=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
gene_metrics=_evaluation.gene_metrics

def main():
    start=time.time();fit=json.loads((OUT/'unbounded_calibration.json').read_text())
    assert prior.digest(OUT/'unbounded_predictions.npz')==fit['parameters_sha256']
    assert prior.digest(OUT/'data_splits.npz')==fit['data_sha256']
    d=np.load(OUT/'data_splits.npz');f=np.load(OUT/'unbounded_predictions.npz')
    rep=joblib.load(JOINT/'representation.joblib');pc32=rep['pca'];scale=rep['scale'];band32=rep['bandwidth']
    space=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib');pc20,km=space['pca'],space['clusters']
    z9_20=np.load(REVIEW/'composition_forecast_trial/early_space.npz')['z9'];rng=np.random.default_rng(prior.SEED+21)
    band20=float(np.median(pdist(z9_20[rng.choice(len(z9_20),1000,False)],'sqeuclidean')))
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines();truth=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel)
    zt20=prior.project(truth,pc20);gt20=km.predict(zt20);targets={}
    for fold in range(2):
        rows=d[f'f{fold}_test10'];p,m=prior.means(truth[rows],gt20[rows])
        targets[fold]=dict(weights=p,means=m,latent20=zt20[rows],latent32=d['z10'][rows]/scale,subset=d[f'f{fold}_kernel_subset'])
    del truth
    latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel);ref=np.asarray(latest.sum(0,dtype=float)).ravel()/latest.shape[0]
    mapping=pc32.components_@pc20.components_.T;records=[];fullgenes=[]
    for batch,(src,g) in enumerate(zip(d['sources'],d['source_groups'])):
        if batch==0:base=latest[src].toarray()
        for fold in range(2):
            target=targets[fold]
            for seed in SEEDS:
                models={};genes={}
                for arm in ['flex_mass_unbounded','full_unbounded']:
                    key=f'f{fold}_s{seed}_{arm}';future=f[key+'_future'][batch];lw=f[key+'_logw'][batch].astype(float)
                    w=np.exp(lw-lw.max());w/=w.sum();z20=z9_20[src]+(future-d['z9'][src])@mapping
                    models[arm]=dict(latent32=metrics_latent(future/scale,w,target['latent32'],band32,target['subset']),
                        latent20=metrics_latent(z20,w,target['latent20'],band20,target['subset']))
                    if batch==0:
                        pred=old.decode(base,(future-d['z9'][src])@pc32.components_)
                        met=gene_metrics(pred,g,w,target['means'],target['weights'],ref)
                        met['decoded_latent20']=metrics_latent(prior.project(pred,pc20),w,target['latent20'],band20,target['subset'])
                        met['decoder_relative_residual']=float(np.linalg.norm(prior.project(pred,pc32)-future)/np.linalg.norm(future-d['z9'][src]))
                        for chunk in np.array_split(pred,20):np.testing.assert_allclose(np.expm1(chunk.astype(float)).sum(1),10000,rtol=1e-5)
                        genes[arm]=met
                records.append(dict(fold=fold,seed=seed,batch=batch,models=models))
                if batch==0:fullgenes.append(dict(fold=fold,seed=seed,models=genes))
                print('unbounded evaluation',batch,fold,seed,{a:round(m['latent32']['objective'],6) for a,m in models.items()},flush=True)
    save('unbounded_results.json',dict(scope='Post-hoc same-stage held-out growth-bound diagnostic',records=records,full_gene_main_pool=fullgenes,
        forecasts_unchanged=prior.digest(OUT/'unbounded_predictions.npz')==fit['parameters_sha256'],
        original_forecasts_unchanged=prior.digest(OUT/'frozen_predictions.npz')==fit['original_parameters_sha256'],seconds=time.time()-start))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
