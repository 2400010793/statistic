"""Post-hoc technical check of mass rounding; never refits forecasts."""
import json
import time
import joblib
from threadpoolctl import threadpool_limits
from core import OUT,PREV,ROOT,SEEDS,prior,np,decode,save
from evaluate import score
REPEATS=[111,222,333]

def main():
    start=time.time();f=np.load(OUT/'frozen_predictions.npz')
    training=json.loads((OUT/'training.json').read_text());assert prior.digest(OUT/'frozen_predictions.npz')==training['frozen_sha256']
    orders={};audit=[]
    for b in range(5):
        for rep in REPEATS:
            rng=np.random.default_rng(2026112600+rep+10000*b);perm=rng.permutation(2000);u=rng.random()
            uniforms=(u+np.arange(2000))/2000
            identity=np.sort(perm[np.searchsorted(np.cumsum(np.full(2000,1/2000)),uniforms)])
            np.testing.assert_array_equal(identity,np.arange(2000))
            for seed in SEEDS:
                key=f'joint_{seed}_h1';w=np.exp(f[key+'_logw'][b].astype(float));w/=w.sum()
                cdf=np.cumsum(w[perm]);cdf[-1]=1
                order=np.sort(perm[np.searchsorted(cdf,uniforms)])
                orders[f'b{b}_s{seed}_r{rep}']=order
                counts=np.bincount(order,minlength=2000)
                audit.append(dict(batch=b,seed=seed,repeat=rep,uniform=u,distinct_pool_rows=int(np.sum(counts>0)),
                    identity=bool(np.array_equal(order,np.arange(2000))),l1_frequency_error=float(abs(counts/2000-w).sum()),
                    largest_remainder_identity=bool(np.array_equal(f[key+'_order'][b],np.arange(2000))),
                    weight_range=[float(2000*w.min()),float(2000*w.max())]))
    np.savez_compressed(OUT/'resampling_frozen.npz',**orders)
    frozen_hash=prior.digest(OUT/'resampling_frozen.npz')
    # Target first accessed below; sampling depends only on early-trained frozen weights.
    s=joblib.load(OUT/'representation.joblib');pc32=s['pca'];znew=np.load(OUT/'early_latent.npz')['z9']
    space=joblib.load(PREV/'composition_forecast_trial/early_space.joblib');pc,km=space['pca'],space['clusters']
    z9=np.load(PREV/'composition_forecast_trial/early_space.npz')['z9']
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines();latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    ev=prior.Evaluator(panel,pc,km,latest,z9);records=[];uniform=np.full(2000,1/2000)
    for b,(src,g) in enumerate(zip(f['sources'],f['source_groups'])):
        base=latest[src].toarray()
        for seed in SEEDS:
            key=f'joint_{seed}_h1';delta=(f[key+'_future'][b]-znew[src])@pc32.components_
            pred=decode(base,delta);z=prior.project(pred,pc);k=ev.kernel(z,z);cross=np.stack([ev.kernel(z,y).mean(1) for y in ev.subs])
            control=score(pred,z,g,uniform,src,ev,k,cross,True)
            for rep in REPEATS:
                order=orders[f'b{b}_s{seed}_r{rep}']
                res=score(pred[order],z[order],g[order],uniform,src[order],ev,k[np.ix_(order,order)],cross[:,order],True)
                records.append(dict(batch=b,seed=seed,repeat=rep,joint_systematic=res,joint_uniform=control))
            print('Systematic mass check',b,seed,flush=True)
    save('resampling_results.json',dict(scope='Post-hoc materialization diagnostic; frozen forecasts unchanged',
        records=records,allocation_audit=audit,frozen_orders_sha256=frozen_hash,original_forecast_sha256=training['frozen_sha256'],
        forecast_unchanged=prior.digest(OUT/'frozen_predictions.npz')==training['frozen_sha256'],seconds=time.time()-start))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
