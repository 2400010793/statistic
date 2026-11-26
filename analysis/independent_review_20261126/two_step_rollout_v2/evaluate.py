"""Observed-stage held-out-cell evaluation and target-free E12.5 stress."""
import time
import argparse
import joblib
from threadpoolctl import threadpool_limits
from rollout import *
sys.path.insert(0,str(V1))
spec_eval=importlib.util.spec_from_file_location('v1_evaluation',V1/'evaluate.py')
old_eval=importlib.util.module_from_spec(spec_eval);spec_eval.loader.exec_module(old_eval)
Target=old_eval.Target

def avg(rows):
    keys=['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine','ess_rows','ess_unique']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}

def main():
    start=time.time();torch.set_num_threads(2)
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    if not args.watch:
        training=json.loads((OUT/'training.json').read_text())
        for m in training['models']:assert prior.digest(OUT/(m['tag']+'.pt'))==m['checkpoint_sha256']
    d=np.load(V1/'data_splits.npz');rep=joblib.load(JOINT/'representation.joblib');decoder=Decoder(rep['pca'],rep['scale'])
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    xs=[prior.read(ROOT/f'data/raw/T1/{s}_RNA.h5ad',panel) for s in ['E8.5','E9.5']]
    xs.append(prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel))
    refs=[np.asarray(x.sum(0,dtype=float)).ravel()/x.shape[0] for x in xs]
    grouping=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib')
    groups=[d['groups8'],d['groups9'],grouping['clusters'].predict(prior.project(xs[2],grouping['pca']))]
    targets={}
    rows=d['test9'];targets[('early',0)]=Target(d['z9'][rows]/rep['scale'],xs[1][rows],groups[1][rows],rep['bandwidth'],refs[0])
    targets[('early',1)]=targets[('early',0)]
    for fold in range(2):
        rows=d[f'f{fold}_test10']
        for phase,reference in [('late',refs[1]),('two',refs[0])]:
            targets[(phase,fold)]=Target(d['z10'][rows]/rep['scale'],xs[2][rows],groups[2][rows],rep['bandwidth'],reference)
    source12=np.load(ROOT/'analysis/results/p3_preparation/damped_group_forecast/source_rows_E10.5.npy')
    base12=xs[2][source12].toarray();records=[];baseline=[];frozen={}
    pools=[('early',0,d['test8'],0),('two',0,d['test8'],0)]+[('late',1,s,p) for p,s in enumerate(d['sources'])]
    for phase,stage,src,pool in pools:
        base=xs[stage][src].toarray();g=groups[stage][src];u=np.full(len(src),1/len(src))
        zz=prior.project(base,rep['pca'])/rep['scale']
        for fold in range(2):baseline.append(dict(phase=phase,pool=pool,fold=fold,metrics=targets[(phase,fold)].score(base,zz,g,u,src)))
    tags=[f'f{fold}_s{seed}_{arm}' for fold in range(2) for seed in SEEDS for arm in ARMS]
    for tag in tags:
        if args.watch:
            while True:
                # Checkpoints are evaluated only after fit() has returned and published its hash.
                try:
                    partial=json.loads((OUT/'training_partial.json').read_text())
                    found=[m for m in partial['models'] if m['tag']==tag]
                except (FileNotFoundError,json.JSONDecodeError):found=[]
                if found:break
                time.sleep(10)
            m=found[0]
        else:m=next(m for m in training['models'] if m['tag']==tag)
        assert prior.digest(OUT/(m['tag']+'.pt'))==m['checkpoint_sha256']
        net=StageDynamics(True);ck=torch.load(OUT/(m['tag']+'.pt'),weights_only=True);net.load_state_dict(ck['state']);net.eval()
        observed=[]
        for phase,stage,src,pool in pools:
            base=xs[stage][src].toarray();r=predict(net,decoder,base,stage,phase=='two')
            w=probabilities(r['logw']);score=targets[(phase,m['fold'])].score(r['expression'],r['z'],groups[stage][src],w,src)
            observed.append(dict(phase=phase,pool=pool,metrics=score,internal=internal(r,base,src)))
            key=m['tag']+'_'+phase+str(pool)
            frozen[key+'_z']=r['z'];frozen[key+'_logw']=r['logw']
            print(m['tag'],phase,pool,'objective',round(score['objective'],6),'ESS',round(score['ess_rows']),flush=True)
            del r
        r=predict(net,decoder,base12,2,True);future=internal(r,base12,source12)
        midw=probabilities(r['mid_logw']);future['first_day_ess']=float(1/(midw@midw))
        frozen[m['tag']+'_future_z']=r['z'];frozen[m['tag']+'_future_logw']=r['logw']
        del r
        records.append(dict(tag=m['tag'],fold=m['fold'],seed=m['seed'],arm=m['arm'],observed=observed,future_hold=future))
        save('results_partial.json',dict(records=records,copylast=baseline))
    if args.watch:
        while not (OUT/'training.json').exists():time.sleep(2)
        training=json.loads((OUT/'training.json').read_text())
    np.savez_compressed(OUT/'frozen_evaluation.npz',**frozen)
    aggregates={arm:{phase:avg([o['metrics'] for r in records if r['arm']==arm for o in r['observed'] if o['phase']==phase]) for phase in ['early','late','two']} for arm in ARMS}
    pairs=[]
    for fold in range(2):
        for seed in SEEDS:
            rr={r['arm']:r for r in records if r['fold']==fold and r['seed']==seed}
            mm={a:{p:avg([o['metrics'] for o in rr[a]['observed'] if o['phase']==p]) for p in ['early','late','two']} for a in ARMS}
            pairs.append(dict(fold=fold,seed=seed,metrics=mm,future={a:rr[a]['future_hold'] for a in ARMS}))
    a=aggregates['single_step'];b=aggregates['two_step']
    criteria=dict(two_objective_at_least_ten_percent=b['two']['objective']<=.9*a['two']['objective'],
        two_objective_improves_three_configs=sum(p['metrics']['two_step']['two']['objective']<p['metrics']['single_step']['two']['objective'] for p in pairs)>=3,
        two_mmd_improves=b['two']['mmd2']<a['two']['mmd2'],two_covariance_improves=b['two']['covariance_rel']<a['two']['covariance_rel'],
        early_objective_cost_at_most_ten_percent=b['early']['objective']<=1.1*a['early']['objective'],
        late_objective_cost_at_most_ten_percent=b['late']['objective']<=1.1*a['late']['objective'],
        late_pooled_cost_at_most_five_percent=b['late']['pooled_rmse']<=1.05*a['late']['pooled_rmse'],
        late_within_cost_at_most_five_percent=b['late']['fixed_within_rmse']<=1.05*a['late']['fixed_within_rmse'],
        two_ess_all_at_least_800=all(p['metrics']['two_step']['two']['ess_rows']>=800 for p in pairs))
    futuremean={arm:float(np.mean([r['future_hold']['ess_rows'] for r in records if r['arm']==arm])) for arm in ARMS}
    futurecriteria=dict(mean_ess_improves_twenty_percent=futuremean['two_step']>=1.2*futuremean['single_step'],
        all_ess_at_least_500=all(r['future_hold']['ess_rows']>=500 for r in records if r['arm']=='two_step'))
    budgets={arm:int(np.ceil(np.median([m['best_step'] for m in training['models'] if m['arm']==arm]))) for arm in ARMS}
    decision=dict(observed_gate_passed=all(criteria.values()),criteria=criteria,future_stability_gate_passed=all(futurecriteria.values()),
        future_criteria=futurecriteria,future_mean_ess=futuremean,full_refit_updates=budgets,
        interpretation='Retrospective held-out cells, not held-out developmental time. Future stability is not future accuracy.')
    save('decision.json',decision)
    save('results.json',dict(scope='Observed-stage validation and target-free E12.5 stress',records=records,copylast=baseline,
        aggregates=aggregates,pairs=pairs,decision=decision,seconds=time.time()-start,
        checkpoint_sha256={m['tag']:m['checkpoint_sha256'] for m in training['models']},
        frozen_sha256=prior.digest(OUT/'frozen_evaluation.npz'),
        metrics='Exact weights; decoded log1p(CP10K); fixed early PCA32; ddof0 covariance; three RBF bands; target-proportion-weighted fixed-source-group within RMSE'))
    print(json.dumps(dict(aggregates=aggregates,decision=decision),indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
