"""Full-P3 models only: paired target-free candidates with ancestry audit."""
import anndata as ad
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def stats(expression,z,w,src,groups,reference,refz):
    gm=w@expression;mean=w@z;c=z-mean;cov=(c.T*w)@c
    mr=refz.mean(0);cr=refz-mr;covr=cr.T@cr/len(cr)
    _,inv=np.unique(src,return_inverse=True);ancestry=np.bincount(inv,weights=w)
    return dict(ess_component_rows=float(1/(w@w)),ess_unique_ancestors=float(1/(ancestry@ancestry)),
        unique_ancestors=int(len(np.unique(src))),pooled_gene_rmse_from_e105_source=float(np.sqrt(np.mean((gm-reference)**2))),
        covariance_relative_change_from_e105_source=float(np.linalg.norm(cov-covr)/np.linalg.norm(covr)),
        source_group_proportions=np.bincount(groups,weights=w,minlength=12).tolist(),
        zero_fraction=float(w@(expression==0).mean(1)),
        interpretation='Changes from the E10.5 source, not errors against E12.5')

def sample(w,n=2000):
    rng=np.random.default_rng(2026112603);perm=rng.permutation(len(w))
    u=(np.arange(n)+rng.random())/n;cdf=np.cumsum(w[perm]);cdf[-1]=1
    return perm[np.searchsorted(cdf,u)]

def main():
    torch.set_num_threads(2)
    train=json.loads((OUT/'full_training.json').read_text())
    decision=json.loads((OUT/'decision.json').read_text());assert decision['observed_gate_passed']
    rep=joblib.load(JOINT/'representation.joblib');decoder=Decoder(rep['pca'],rep['scale'])
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    limits=json.loads((ROOT/'panels/index.json').read_text())['T1:test']
    assert limits['min_cells']<=2000<=limits['max_cells'] and len(panel)==32285
    old=ROOT/'analysis/results/p3_preparation/damped_group_forecast'
    src=np.load(old/'source_rows_E10.5.npy');assert len(np.unique(src))==2000
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);base=x[src].toarray();del x
    copy=prior.read(old/'copylast_E12.5.h5ad',panel)
    if hasattr(copy,'toarray'):copy=copy.toarray()
    np.testing.assert_array_equal(copy,base);del copy
    paths=[old/'source_rows_E10.5.npy',old/'copylast_E12.5.h5ad',old/'damped_group_E12.5.h5ad',OUT/'full_training.json']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    refz=prior.project(base,rep['pca'])/rep['scale'];reference=base.astype(float).mean(0)
    grouping=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib')
    groups=grouping['clusters'].predict(prior.project(base,grouping['pca']))
    uniform=np.full(len(src),1/len(src));baselines={}
    for name in ['copylast','damped_group']:
        arr=prior.read(old/f'{name}_E12.5.h5ad',panel)
        if hasattr(arr,'toarray'):arr=arr.toarray()
        z=prior.project(arr,rep['pca'])/rep['scale']
        baselines[name]=stats(arr,z,uniform,src,groups,reference,refz)
    artifact_dir=OUT/'e125_candidates';artifact_dir.mkdir(exist_ok=True)
    records=[];mixed=[];artifacts=[]
    for arm in ARMS:
        for policy in ['hold','damped']:
            components=[]
            for seed in SEEDS:
                tag=f'full_s{seed}_{arm}';m=next(m for m in train['models'] if m['tag']==tag)
                assert prior.digest(OUT/(tag+'.pt'))==m['checkpoint_sha256']
                ck=torch.load(OUT/(tag+'.pt'),weights_only=True);assert ck['full_data']
                net=StageDynamics(True);net.load_state_dict(ck['state']);net.eval()
                r=predict(net,decoder,base,2,True,policy);w=probabilities(r['logw'])
                for s in range(0,len(src),128):
                    pred=r['expression'][s:s+128]
                    assert np.isfinite(pred).all() and (pred>=0).all()
                    np.testing.assert_array_equal(pred>0,base[s:s+128]>0)
                    np.testing.assert_allclose(np.expm1(pred.astype(float)).sum(1),10000,rtol=1e-5)
                metrics=stats(r['expression'],r['z'],w,src,groups,reference,refz)
                record=dict(arm=arm,policy=policy,seed=seed,metrics=metrics,internal=internal(r,base,src))
                records.append(record)
                # Save the exact probability distribution as a weighted research pool.
                dest=artifact_dir/f'{arm}_{policy}_s{seed}_weighted.h5ad'
                obs=pd.DataFrame(dict(source_index=src,prediction_weight=w,model_seed=np.full(len(src),seed)),index=[f'pool_{j}' for j in range(len(src))])
                obj=ad.AnnData(r['expression'],obs=obs,var=pd.DataFrame(index=panel))
                obj.uns['scope']='P3 E12.5 research candidate; no E12.5 truth';obj.uns['time_policy']=policy
                obj.write_h5ad(dest,compression='gzip',compression_opts=1)
                artifacts.append(dict(path=str(dest),sha256=prior.digest(dest),role='Weighted pool; requires obs.prediction_weight'))
                components.append((r,w,seed))
            values=np.concatenate([v[0]['expression'] for v in components]);z=np.concatenate([v[0]['z'] for v in components])
            weights=np.concatenate([v[1]/len(components) for v in components]);sources=np.tile(src,len(components))
            seeds=np.concatenate([np.full(len(src),v[2]) for v in components]);gg=np.tile(groups,len(components))
            exact=stats(values,z,weights,sources,gg,reference,refz)
            take=sample(weights);output=values[take]
            materialized=stats(output,z[take],uniform,sources[take],gg[take],reference,refz)
            dest=artifact_dir/f'{arm}_{policy}_E12.5.h5ad'
            obs=pd.DataFrame(dict(source_index=sources[take],component_row=take,model_seed=seeds[take]),index=[f'prediction_{j}' for j in range(2000)])
            obj=ad.AnnData(output,obs=obs,var=pd.DataFrame(index=panel))
            obj.uns['scope']='P3 full-data two-seed mixture; E12.5 truth unavailable'
            obj.uns['time_policy']=policy;obj.uns['resampling_seed']=2026112603
            obj.write_h5ad(dest,compression='gzip',compression_opts=1)
            check=ad.read_h5ad(dest);assert check.shape==(2000,32285) and check.var_names.tolist()==panel
            np.testing.assert_array_equal(check.X,output)
            np.savez_compressed(artifact_dir/f'{arm}_{policy}_provenance.npz',pool_source_indices=sources,pool_seed=seeds,pool_weights=weights,
                sampled_component_rows=take,sampled_source_indices=sources[take],sampled_seeds=seeds[take])
            # Disagreement describes sensitivity to fitting, not an uncertainty interval.
            mean0=components[0][1]@components[0][0]['expression'];mean1=components[1][1]@components[1][0]['expression']
            mixed.append(dict(arm=arm,policy=policy,exact=exact,materialized=materialized,
                seed_pooled_mean_disagreement_rmse=float(np.sqrt(np.mean((mean0-mean1)**2))),path=str(dest),format_readback_passed=True))
            artifacts.append(dict(path=str(dest),sha256=prior.digest(dest),role='Uniform 2000-row materialized mixture; local research candidate'))
            print(arm,policy,'ancestor ESS',round(exact['ess_unique_ancestors']),'materialized',round(materialized['ess_unique_ancestors']),flush=True)
            del components,values,z,weights,output,check,obj
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('e125_results.json',dict(scope='E12.5 candidate generation only; no accuracy estimate or official submission',
        records=records,mixtures=mixed,baselines=baselines,artifacts=artifacts,input_sha256=hashes,inputs_unchanged=unchanged,
        future_time='Condition t=2; state-dependent fields; hold versus fixed half second-day rates',
        source_indices_path=str(old/'source_rows_E10.5.npy'),format_readback_passed=True))
    save('model_card.json',dict(status='Research candidates; future mass concentration remains unresolved',
        training_stages=['E8.5','E9.5','E10.5'],hidden_target='E12.5',hidden_truth_available=False,
        observed_gate_passed=decision['observed_gate_passed'],future_stability_gate_passed=decision['future_stability_gate_passed'],
        parameters=29250,representation='Frozen early PCA32; log1p(CP10K), 32285 genes',
        update='Two generated daily outputs with re-encoding, terminal distribution loss and cumulative ancestry ESS penalty',
        time_policy='Hold time conditioning at E10.5; evaluate the vector fields at evolving states',
        future_multipliers={'hold':{'velocity':[1.,1.],'mass':[1.,1.]},'damped':{'velocity':[1.,.5],'mass':[1.,.5]}},
        edit_budget={'daily_pre_normalization_gain':[.8,1.2],'two_day_original_anchor_gain':[.64,1.44]},
        checkpoints=[dict(path=str(OUT/(m['tag']+'.pt')),sha256=m['checkpoint_sha256'],arm=m['arm'],seed=m['seed'],updates=m['best_step']) for m in train['models']],
        candidate_paths=[m['path'] for m in mixed],recommended_for_accuracy=None,
        provenance='Every exported row records its original E10.5 source and model seed; exact weighted pools also retained',
        limitations=['No future accuracy validation','Only two observed time intervals','Cell split is not a new developmental time point',
            'Source zero support preserved','Population dynamics do not identify unique cell lineages','ESS is a stability measure, not accuracy']))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
