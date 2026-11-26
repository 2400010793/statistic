"""Evaluate frozen models on both held-out observed intervals and all source pools."""
import time
import json
import hashlib
import joblib
import anndata as ad
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from model import OUT,REVIEW,JOINT,ROOT,prior,np,torch,SEEDS,ARMS,Decoder,save

class Target:
    def __init__(self,z,expression,groups,bandwidth,reference):
        self.z=z.astype(float);self.band=bandwidth;self.reference=reference
        self.mean=self.z.mean(0);c=self.z-self.mean;self.cov=c.T@c/len(c)
        self.proportions,self.group_means=prior.means(expression,groups)
        self.gene_mean=np.asarray(expression.sum(0,dtype=float)).ravel()/expression.shape[0]
        self.change=self.gene_mean-reference;self.sub=self.z[:min(2000,len(self.z))]
        dd=cdist(self.sub,self.sub,'sqeuclidean');self.yy=[]
        for m in [.25,1.,4.]:
            k=np.exp(-dd/(2*self.band*m));self.yy.append((float(k.mean()),float((k.sum()-len(k))/(len(k)*(len(k)-1)))))

    def score(self,x,z,groups,w,src):
        z=z.astype(float);w=w.astype(float);mu=w@z;c=z-mu;cov=(c.T*w)@c
        dxx=cdist(z,z,'sqeuclidean');dxy=cdist(z,self.sub,'sqeuclidean');mv=[];mu_u=[]
        for mult,(yy,yy_u) in zip([.25,1.,4.],self.yy):
            k=np.exp(-dxx/(2*self.band*mult));cross=np.exp(-dxy/(2*self.band*mult)).mean(1)
            xx=float(w@k@w);xy=float(w@cross)
            mv.append(xx+yy-2*xy);mu_u.append((xx-w@w)/(1-w@w)+yy_u-2*xy)
        gm=w@x;gene_mse=float(np.mean((gm-self.gene_mean)**2))
        p=np.bincount(groups,weights=w,minlength=12);assert (p>0).all()
        means=np.stack([w[groups==k]@x[groups==k]/p[k] for k in range(12)])
        group_error=np.mean((means-self.group_means)**2,axis=1)
        zmse=float(np.mean((mu-self.mean)**2));cmse=float(np.mean((cov-self.cov)**2));d=gm-self.reference
        _,inv=np.unique(src,return_inverse=True);unique_w=np.bincount(inv,weights=w)
        concentration=len(w)*float(w@w);objective=float(np.mean(mv))+.05*zmse+.01*cmse+10*gene_mse
        return dict(objective=objective,mmd_v=float(np.mean(mv)),mmd2=float(np.mean(mu_u)),
            covariance_rel=float(np.linalg.norm(cov-self.cov)/np.linalg.norm(self.cov)),pooled_rmse=float(np.sqrt(gene_mse)),
            fixed_within_rmse=float(np.sqrt(self.proportions@group_error)),fixed_equal_rmse=float(np.sqrt(group_error.mean())),
            cosine=float(d@self.change/(np.linalg.norm(d)*np.linalg.norm(self.change))),
            ess_rows=float(1/(w@w)),ess_fraction=float(1/concentration),ess_unique=float(1/(unique_w@unique_w)),
            concentration_penalty=.02*max(concentration-2,0)**2,
            source_composition_tv=float(abs(p-self.proportions).sum()/2),source_proportions=p.tolist(),
            weight_range=[float(len(w)*w.min()),float(len(w)*w.max())],zero_fraction=float(w@(x==0).mean(1)))

def materialize(w):
    rng=np.random.default_rng(2026112800);permutation=rng.permutation(len(w));u=(np.arange(len(w))+rng.random())/len(w)
    cdf=np.cumsum(w[permutation]);cdf[-1]=1
    return np.sort(permutation[np.searchsorted(cdf,u)])

def main():
    start=time.time();torch.set_num_threads(2)
    training=json.loads((OUT/'training.json').read_text());assert prior.digest(OUT/'frozen_predictions.npz')==training['frozen_sha256']
    assert prior.digest(OUT/'data_splits.npz')==training['data_sha256']
    paths=[OUT/'protocol.txt',OUT/'implementation_notes.txt',OUT/'training.json',OUT/'frozen_predictions.npz',OUT/'evaluate.py',REVIEW/'results.json',REVIEW/'frozen_controls.npz']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    d=np.load(OUT/'data_splits.npz');f=np.load(OUT/'frozen_predictions.npz');rep=joblib.load(JOINT/'representation.joblib')
    decoder=Decoder(rep['pca'],rep['scale']);pc32=rep['pca'];scale=rep['scale'];band=rep['bandwidth']
    evaluation_space=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib');km=evaluation_space['clusters'];pc20=evaluation_space['pca']
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    source=[prior.read(ROOT/f'data/raw/T1/{s}_RNA.h5ad',panel) for s in ['E8.5','E9.5']]
    references=[np.asarray(x.sum(0,dtype=float)).ravel()/x.shape[0] for x in source]
    test9=d['test9'];targets={(0,0):Target(d['z9'][test9]/scale,source[1][test9],d['groups9'][test9],band,references[0])}
    targets[(0,1)]=targets[(0,0)]
    truth=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);g10=km.predict(prior.project(truth,pc20))
    for fold in range(2):
        rows=d[f'f{fold}_test10'];targets[(1,fold)]=Target(d['z10'][rows]/scale,truth[rows],g10[rows],band,references[1])
    del truth
    controls=np.load(REVIEW/'frozen_controls.npz');records=[];baselines=[];artifacts=[];materialized=None
    pools=[(0,d['test8'])]+[(1,src) for src in d['sources']]
    for pool,(phase,src) in enumerate(pools):
        base=source[phase][src].toarray();zsource=d['z8'][src] if phase==0 else d['z9'][src]
        groups=d['groups8'][src] if phase==0 else d['groups9'][src];uniform=np.full(len(src),1/len(src))
        baseline_arrays={'copy':base}
        if phase==1:baseline_arrays['shared']=prior.decode(base,groups,controls['recent_means'],controls[f'b{pool-1}_common_only'])
        for fold in range(2):
            target=targets[(phase,fold)]
            base_scores={name:target.score(array,prior.project(array,pc32)/scale,groups,uniform,src) for name,array in baseline_arrays.items()}
            baselines.append(dict(pool=pool,phase=phase,fold=fold,models=base_scores))
            for seed in SEEDS:
                scores={}
                for arm in ARMS:
                    key=f'f{fold}_s{seed}_{arm}';future=f[key+'_future'][pool];lw=f[key+'_logw'][pool].astype(float)
                    w=np.exp(lw-lw.max());w/=w.sum()
                    with torch.no_grad():
                        pred=np.concatenate([decoder(torch.from_numpy(base[s:s+128]),torch.from_numpy(((future[s:s+128]-zsource[s:s+128])/scale).astype(np.float32))).numpy()
                            for s in range(0,len(src),128)])
                    np.testing.assert_array_equal(pred>0,base>0)
                    for chunk in np.array_split(pred,20):np.testing.assert_allclose(np.expm1(chunk.astype(float)).sum(1),10000,rtol=1e-5)
                    z=prior.project(pred,pc32)/scale;scores[arm]=target.score(pred,z,groups,w,src)
                    scores[arm]['decoder_relative_residual']=float(np.linalg.norm(z*scale-future)/max(np.linalg.norm(future-zsource),1e-12))
                    if pool==1 and fold==0 and seed==SEEDS[0] and arm=='temporal_decoded':
                        path=OUT/'temporal_decoded_weighted_pool.h5ad'
                        obs=pd.DataFrame({'source_index':src,'prediction_weight':w},index=[f'source_row_{j}' for j in range(len(src))])
                        ad.AnnData(pred,obs=obs,var=pd.DataFrame(index=panel)).write_h5ad(path,compression='gzip',compression_opts=1)
                        artifacts.append(dict(path=str(path),sha256=prior.digest(path),role='Weighted research pool; obs weights required'))
                        order=materialize(w);np.savez_compressed(OUT/'research_output_indices.npz',pool_rows=order,source_indices=src[order],pool_weights=w)
                        values=pred[order];dest=OUT/'temporal_decoded_E10.5_diagnostic.h5ad'
                        ad.AnnData(values,obs=pd.DataFrame(index=[f'prediction_{j}' for j in range(len(src))]),var=pd.DataFrame(index=panel)).write_h5ad(dest,compression='gzip',compression_opts=1)
                        check=ad.read_h5ad(dest);np.testing.assert_array_equal(check.X,values);assert check.var_names.tolist()==panel
                        materialized=target.score(values,z[order],groups[order],uniform,src[order])
                        materialized['unique_pool_rows']=int(len(np.unique(order)))
                        artifacts.append(dict(path=str(dest),sha256=prior.digest(dest),role='Same-stage calibrated research output; not an official forecast'))
                    print('score',pool,fold,seed,arm,{k:round(scores[arm][k],6) for k in ['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','ess_fraction']},flush=True)
                records.append(dict(pool=pool,phase=phase,fold=fold,seed=seed,models=scores))
        save('results_partial.json',dict(records=records,baselines=baselines))
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('results.json',dict(scope='E10.5-calibrated two-interval held-out-cell evaluation, not hidden-stage forecasting',
        records=records,baselines=baselines,materialized_research_output=materialized,artifacts=artifacts,
        input_sha256=hashes,inputs_unchanged=unchanged,seconds=time.time()-start,metric_space='Decoded log1p(CP10K), early PCA32, ddof0 covariance, three RBF bandwidths'))
    print('EVALUATION COMPLETE',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
