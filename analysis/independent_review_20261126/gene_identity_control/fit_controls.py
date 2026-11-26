"""Fit and freeze all gene-identity controls using E9.5 source cells only."""
import time
import importlib.metadata
import numpy as np
import anndata as ad
from scipy.optimize import least_squares, brentq
from threadpoolctl import threadpool_limits
from control_core import OUT,PREV,ROOT,SEED,Decoder,digest,save

def fit_one(decoder,ratio,reference):
    scale=np.abs(reference)
    attempts=[]
    best=None
    for start in ([1.,0.],[.5,.01],[2.,.01]):
        cache={}
        def get(theta):
            key=tuple(theta)
            if cache.get('key')!=key:
                m,j=decoder.calculate(ratio,theta,jacobian=True)
                cache.update(key=key,res=(m-reference)/scale,jac=j/scale[:,None],moments=m)
            return cache
        fit=least_squares(lambda t:get(t)['res'],start,jac=lambda t:get(t)['jac'],
            bounds=([0,-.2],[4,.2]),max_nfev=40,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        item=dict(theta=fit.x.tolist(),nfev=int(fit.nfev),status=int(fit.status),
            **decoder.moment_audit(get(fit.x)['moments'],reference))
        attempts.append(item)
        if best is None or max(item['relative_errors'])<max(best['relative_errors']):best=item
        if max(item['relative_errors'])<=.005:break
    return best,attempts

def main():
    start=time.time()
    input_paths=[PREV/'frozen_controls.npz',PREV/'run_review.py',ROOT/'data/raw/T1/E9.5_RNA.h5ad',
                 OUT/'protocol.txt',OUT/'control_core.py',OUT/'fit_controls.py']
    input_hashes={str(p.relative_to(ROOT)):digest(p) for p in input_paths}
    old=np.load(PREV/'frozen_controls.npz')
    sources=old['sources']; labels=old['source_groups']; recent=old['recent_means']; common=old['b0_common_only'][0]
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    a=ad.read_h5ad(ROOT/'data/raw/T1/E9.5_RNA.h5ad')
    assert list(a.var_names)==panel
    x=a.X.tocsr(); x.eliminate_zeros()
    mean=np.asarray(x.mean(0,dtype=float)).ravel()
    detection=np.asarray((x>0).sum(0)).ravel()/x.shape[0]
    eligible=detection>0
    cap=np.minimum(detection,1-1/(2*x.shape[0]))
    bin_keys=np.full((len(panel),2),-999,dtype=int)
    bin_keys[eligible,0]=np.floor(np.log2(mean[eligible])).astype(int)
    bin_keys[eligible,1]=np.floor(np.log2(cap[eligible]/(1-cap[eligible]))).astype(int)
    _,bins=np.unique(bin_keys,axis=0,return_inverse=True)
    pools=[np.flatnonzero((bins==k)&eligible) for k in np.unique(bins)]
    movable=np.zeros(len(panel),bool)
    for ix in pools:
        if len(ix)>=2:movable[ix]=True
    coverage=float(np.sum(common[movable]**2)/np.sum(common[eligible]**2))
    permutations=[]
    for j in range(32):
        rng=np.random.default_rng(SEED+j); perm=np.arange(len(panel))
        for ix in pools:
            if len(ix)<2:continue
            order=rng.permutation(ix)
            perm[order]=np.roll(order,1)
        np.testing.assert_array_equal(bins,bins[perm])
        assert np.all(perm[movable]!=np.flatnonzero(movable))
        permutations.append(perm)
    permutations=np.asarray(permutations,dtype=np.int32)
    decoder=Decoder(x[sources[0]],labels[0],recent)
    del x,a
    reference=decoder.calculate(decoder.ratio(common),[1,0])
    # Numerical gradient check on real source data, far from clipping boundaries.
    check_theta=np.array([.8,.003]); ratio=decoder.ratio(common[permutations[0]])
    _,analytic=decoder.calculate(ratio,check_theta,jacobian=True)
    numeric=np.stack([(decoder.calculate(ratio,check_theta+np.eye(2)[j]*1e-6)-decoder.calculate(ratio,check_theta-np.eye(2)[j]*1e-6))/2e-6 for j in range(2)],axis=1)
    np.testing.assert_allclose(analytic,numeric,rtol=3e-3,atol=1e-5)
    print('Source-only reference:',reference.tolist(),'movable energy',coverage,flush=True)
    controls=[]; theta=[]
    for j,perm in enumerate(permutations):
        ratio=decoder.ratio(common[perm])
        raw=decoder.calculate(ratio,[1,0])
        best,attempts=fit_one(decoder,ratio,reference)
        valid=max(best['relative_errors'])<=.005
        controls.append(dict(id=j,seed=SEED+j,valid=valid,raw=decoder.moment_audit(raw,reference),fit=best,attempts=attempts,
            common_permutation_cosine=float(common@common[perm]/(np.linalg.norm(common)*np.linalg.norm(common[perm])))))
        theta.append(best['theta'])
        save('fit_partial.json',controls)
        print(j,'valid',valid,'theta',best['theta'],'relative errors',best['relative_errors'],'elapsed',round(time.time()-start,1),flush=True)
    zero=np.zeros_like(decoder.a)
    temperature=brentq(lambda beta:decoder.calculate(zero,[0,beta])[1]-reference[1],-.2,.2,xtol=1e-12)
    temperature_moments=decoder.calculate(zero,[0,temperature])
    np.savez_compressed(OUT/'frozen.npz',sources=sources,source_groups=labels,recent_means=recent,common=common,
        permutations=permutations,theta=np.array(theta),bins=bins,bin_keys=bin_keys,early_mean=mean,detection=detection,
        reference_moments=reference,temperature_beta=temperature,valid=np.array([c['valid'] for c in controls]))
    unchanged={str(p.relative_to(ROOT)):digest(p)==input_hashes[str(p.relative_to(ROOT))] for p in input_paths}
    assert all(unchanged.values())
    top_fixed=np.flatnonzero(eligible&~movable)
    top_fixed=top_fixed[np.argsort(-common[top_fixed]**2)[:15]]
    report=dict(phase='Source-only fit complete; E10.5 was not opened by this process.',
        reference_moments=reference.tolist(),valid_count=sum(c['valid'] for c in controls),controls=controls,
        binning=dict(expressed_genes=int(eligible.sum()),movable_genes=int(movable.sum()),
            expressed_bins=sum(len(p)>0 for p in pools),singleton_bins=sum(len(p)==1 for p in pools),
            movable_common_energy_fraction=coverage,coverage_passed=coverage>=.8,
            largest_unmoved_genes=[dict(gene=panel[g],shift=float(common[g]),mean=float(mean[g]),detection=float(detection[g])) for g in top_fixed]),
        gradient_check_max_abs_error=float(abs(analytic-numeric).max()),
        temperature_only=dict(beta=float(temperature),**decoder.moment_audit(temperature_moments,reference)),
        frozen_sha256=digest(OUT/'frozen.npz'),input_sha256=input_hashes,inputs_unchanged=unchanged,
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','anndata','scikit-learn']},
        elapsed_seconds=time.time()-start)
    save('fit_audit.json',report)
    print('FROZEN',report['frozen_sha256'],'valid',report['valid_count'],'elapsed',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
