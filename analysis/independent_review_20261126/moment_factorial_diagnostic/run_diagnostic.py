"""Oracle moment factorial diagnostic. All writes are confined to this directory."""
import sys
import json
import time
import itertools
import importlib.metadata
from pathlib import Path
import numpy as np
import anndata as ad
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
PREV=OUT.parent
ROOT=PREV.parents[1]
sys.path.insert(0,str(PREV))
import run_review as prior

FACTORS=('P','M','C')
MASK_NAMES={i:''.join(FACTORS[j] for j in range(3) if i&(1<<j)) or 'none' for i in range(8)}
ARCHIVE={}

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def mixture(s):
    p,u,c=s['p'],s['u'],s['c']
    avg=p@u
    within=np.einsum('k,kij->ij',p,c)
    d=u-avg
    between=(d.T*p)@d
    return dict(mean=avg,gene_mean=p@s['m'],within=within,between=between,cov=within+between)

def stats(x,z,labels,name):
    z=z.astype(float)
    p,m=prior.means(x,labels)
    n=np.bincount(labels,minlength=12)
    assert (n>1).all()
    u=np.stack([z[labels==k].mean(0) for k in range(12)])
    c=np.stack([(z[labels==k]-u[k]).T@(z[labels==k]-u[k])/n[k] for k in range(12)])
    s=dict(p=p,m=m,u=u,c=c,n=n)
    mixed=mixture(s)
    centered=z-z.mean(0)
    direct=centered.T@centered/len(z)
    rel=float(np.linalg.norm(mixed['cov']-direct)/np.linalg.norm(direct))
    np.testing.assert_allclose(mixed['cov'],direct,rtol=1e-11,atol=1e-10)
    # scipy sparse mean(dtype=float64) can still accumulate in float32;
    # explicit float64 summation is required for an exact moment identity.
    direct_mean=np.asarray(x.sum(0,dtype=np.float64)).ravel()/x.shape[0]
    np.testing.assert_allclose(mixed['gene_mean'],direct_mean,rtol=1e-10,atol=1e-10)
    for key,value in s.items():ARCHIVE[f'{name}_{key}']=value
    s['identity_error']=rel
    return s

def factorial(pred,oracle,target):
    t=mixture(target)
    denom=float(np.sum(t['cov']**2))
    base=mixture(pred)
    errw=base['within']-t['within']; errb=base['between']-t['between']
    decomposition=dict(within_squared=float(np.sum(errw**2)/denom),between_squared=float(np.sum(errb**2)/denom),
        cross_term=float(2*np.sum(errw*errb)/denom),total_squared=float(np.sum((errw+errb)**2)/denom),
        within_error_norm_over_total=float(np.linalg.norm(errw)/np.sqrt(denom)),
        between_error_norm_over_total=float(np.linalg.norm(errb)/np.sqrt(denom)))
    np.testing.assert_allclose(decomposition['within_squared']+decomposition['between_squared']+decomposition['cross_term'],decomposition['total_squared'],rtol=1e-12)
    rows={}
    for mask in range(8):
        s={k:(oracle if mask&(1<<j) else pred)[k] for j,ks in enumerate([['p'],['u','m'],['c']]) for k in ks}
        v=mixture(s)
        eigenvalue=float(np.linalg.eigvalsh(v['cov']).min())
        assert eigenvalue>=-1e-8
        loss=float(np.sum((v['cov']-t['cov'])**2)/denom)
        gdiff=v['gene_mean']-t['gene_mean']
        cdiff=s['c']-target['c']
        conditional=float(np.sqrt(np.einsum('k,kij,kij->',target['p'],cdiff,cdiff)/np.einsum('k,kij,kij->',target['p'],target['c'],target['c'])))
        rows[MASK_NAMES[mask]]=dict(covariance_relative=float(np.sqrt(loss)),covariance_squared=loss,
            gene_mean_rmse=float(np.sqrt(np.mean(gdiff**2))),gene_mean_mse=float(np.mean(gdiff**2)),
            pca_mean_rmse=float(np.sqrt(np.mean((v['mean']-t['mean'])**2))),
            conditional_covariance_relative=conditional,min_covariance_eigenvalue=eigenvalue)
    attribution={}
    for metric in ['covariance_squared','gene_mean_mse']:
        losses={mask:rows[MASK_NAMES[mask]][metric] for mask in range(8)}
        shapley=np.zeros(3)
        orders=[]
        for order in itertools.permutations(range(3)):
            mask=0; contributions={}
            for j in order:
                nex=mask|(1<<j)
                gain=losses[mask]-losses[nex]
                contributions[FACTORS[j]]=gain
                shapley[j]+=gain/6
                mask=nex
            orders.append(dict(order=''.join(FACTORS[j] for j in order),gains=contributions))
        reduction=losses[0]-losses[7]
        np.testing.assert_allclose(shapley.sum(),reduction,rtol=1e-12,atol=1e-15)
        gain={mask:losses[0]-losses[mask] for mask in range(8)}
        pair={MASK_NAMES[(1<<i)|(1<<j)]:gain[(1<<i)|(1<<j)]-gain[1<<i]-gain[1<<j] for i,j in itertools.combinations(range(3),2)}
        triple=gain[7]-sum(gain[1<<j] for j in range(3))-sum(pair.values())
        attribution[metric]=dict(baseline=losses[0],all_replaced=losses[7],total_reduction=reduction,
            single_replacement_reductions={FACTORS[j]:gain[1<<j] for j in range(3)},
            single_replacement_best=FACTORS[int(np.argmax([gain[1<<j] for j in range(3)]))],
            shapley=dict(zip(FACTORS,shapley.tolist())),
            shapley_fraction_of_reduction={f:float(v/reduction) if abs(reduction)>1e-20 else None for f,v in zip(FACTORS,shapley)},
            shapley_largest=FACTORS[int(np.argmax(shapley))],pair_interactions=pair,triple_interaction=triple,orders=orders)
    group_cov_error=np.linalg.norm(pred['c']-target['c'],axis=(1,2))/np.linalg.norm(target['c'],axis=(1,2))
    return dict(combinations=rows,attribution=attribution,error_decomposition=decomposition,
        pred_counts=pred['n'].tolist(),target_counts=target['n'].tolist(),
        composition_tv=float(abs(pred['p']-target['p']).sum()/2),
        group_conditional_covariance_relative=group_cov_error.tolist())

def leading_types(types,labels):
    rows=[]
    for k in range(12):
        names,counts=np.unique(types[labels==k],return_counts=True)
        j=int(np.argmax(counts))
        rows.append(dict(group=k,name=str(names[j]),purity=float(counts[j]/counts.sum())))
    return rows

def main():
    start=time.time()
    input_paths=[ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]+[
        ROOT/'data/p3/T1/E10.5_RNA.h5ad',prior.R40/'neural_population_direction_E10.5.h5ad',
        PREV/'frozen_controls.npz',PREV/'results.json',PREV/'run_review.py',OUT/'protocol.txt',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in input_paths}
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    frozen=np.load(PREV/'frozen_controls.npz'); old=json.loads((PREV/'results.json').read_text())['results']
    pca,km=prior.fit_space(panel)
    np.testing.assert_array_equal(pca.components_,frozen['pca_components'])
    y=ad.read_h5ad(ROOT/'data/p3/T1/E10.5_RNA.h5ad'); target_types=y.obs.celltype.astype(str).to_numpy()
    assert list(y.var_names)==panel
    truth=y.X.tocsr(); del y
    zt=prior.project(truth,pca); gt=km.predict(zt)
    target=stats(truth,zt,gt,'target_full')
    target_mix=mixture(target)
    half_stats=[]; split_indices={}
    for split in range(4):
        halves=np.array_split(np.random.default_rng(2026112630+split).permutation(truth.shape[0]),2)
        pair=[]
        for h,ix in enumerate(halves):
            split_indices[f'split{split}_half{h}']=ix
            pair.append(stats(truth[ix],zt[ix],gt[ix],f'target_s{split}_h{h}'))
        half_stats.append(pair)
    del truth,zt
    latest=ad.read_h5ad(ROOT/'data/raw/T1/E9.5_RNA.h5ad'); latest_types=latest.obs.celltype.astype(str).to_numpy()
    x9=latest.X.tocsr(); del latest
    z9=prior.project(x9,pca); g9=km.predict(z9)
    sources=frozen['sources']; np.testing.assert_array_equal(g9[sources],frozen['source_groups'])
    results=[]; half_results=[]; verification=[]
    identities=[target['identity_error']]+[s['identity_error'] for pair in half_stats for s in pair]
    for batch,source in enumerate(sources):
        base=x9[source].toarray(); labels=g9[source]
        for name in ['copy','common_only','modules_full']+(['r40'] if batch==0 else []):
            if name=='copy':pred=base
            elif name=='r40':pred=prior.read(prior.R40/'neural_population_direction_E10.5.h5ad',panel)
            else:pred=prior.decode(base,labels,frozen['recent_means'],frozen[f'b{batch}_{name}'])
            z=prior.project(pred,pca)
            output=km.predict(z)
            actual_cov=np.cov(z.T)
            target_cov=target_mix['cov']*sum(target['n'])/(sum(target['n'])-1)
            ddof1=float(np.linalg.norm(actual_cov-target_cov)/np.linalg.norm(target_cov))
            pooled=float(np.sqrt(np.mean((pred.mean(0,dtype=float)-target_mix['gene_mean'])**2)))
            np.testing.assert_allclose(ddof1,old[batch][name]['covariance_rel'],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(pooled,old[batch][name]['pooled_rmse'],rtol=1e-10,atol=1e-10)
            verification.append(dict(batch=batch,model=name,ddof1_covariance=ddof1,gene_mean_rmse=pooled,
                max_abs_archive_error=max(abs(ddof1-old[batch][name]['covariance_rel']),abs(pooled-old[batch][name]['pooled_rmse']))))
            for grouping,groups in [('fixed_source',labels),('output_reassigned',output)]:
                s=stats(pred,z,groups,f'b{batch}_{name}_{grouping}')
                identities.append(s['identity_error'])
                rec=factorial(s,target,target)
                assert rec['combinations']['PMC']['covariance_relative']<1e-12
                assert rec['combinations']['PMC']['gene_mean_rmse']<1e-12
                rec.update(batch=batch,model=name,grouping=grouping,reassigned_cells=int(np.sum(output!=labels)),
                    ddof1_covariance=ddof1,within_gene_rmse=float(np.sqrt(target['p']@np.mean((s['m']-target['m'])**2,axis=1))))
                results.append(rec)
                if batch==0 and grouping=='fixed_source':
                    for split,pair in enumerate(half_stats):
                        for h in [0,1]:
                            hr=factorial(s,pair[h],pair[1-h])
                            hr.update(model=name,split=split,oracle_half=h,evaluation_half=1-h)
                            half_results.append(hr)
            print(batch,name,'base',round(rec['combinations']['none']['covariance_relative'],6),'single',rec['attribution']['covariance_squared']['single_replacement_best'],'Shapley',rec['attribution']['covariance_squared']['shapley_fraction_of_reduction'],flush=True)
            del pred,z
        del base
    np.savez_compressed(OUT/'sufficient_statistics.npz',**ARCHIVE)
    np.savez_compressed(OUT/'sampling_indices.npz',sources=sources,source_groups=g9[sources],target_groups=gt,**split_indices)
    preservation={str(p.relative_to(ROOT)):prior.digest(p)==hashes[str(p.relative_to(ROOT))] for p in input_paths}
    assert all(preservation.values())
    report=dict(scope='E10.5 oracle moment diagnostic, not forecast performance. Population covariances (ddof=0) for exact mixture identity.',
        factors={'P':'group proportions','M':'conditional means','C':'conditional covariance'},
        results=results,target_half_sensitivity=half_results,verification=verification,
        max_covariance_identity_relative_error=max(identities),
        target_parts=dict(within_norm=float(np.linalg.norm(target_mix['within'])),between_norm=float(np.linalg.norm(target_mix['between'])),total_norm=float(np.linalg.norm(target_mix['cov']))),
        group_annotations=dict(E95=leading_types(latest_types,g9),E105=leading_types(target_types,gt)),
        target_counts=target['n'].tolist(),inputs_sha256=hashes,inputs_unchanged=preservation,
        versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','anndata']},elapsed_seconds=time.time()-start)
    save('results.json',report)
    print('COMPLETED',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
