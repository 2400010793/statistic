"""Post-hoc diagnostic: fixed expression trajectories, changed future mass only."""
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def main():
    torch.set_num_threads(2);rep=joblib.load(JOINT/'representation.joblib');decoder=Decoder(rep['pca'],rep['scale'])
    inputs=[Path(__file__),OUT/'rollout.py',JOINT/'representation.joblib',ROOT/'data/p3/T1/E10.5_RNA.h5ad',
        ROOT/'analysis/results/p3_preparation/damped_group_forecast/source_rows_E10.5.npy']
    fingerprints={str(p):prior.digest(p) for p in inputs};cache={}
    if (OUT/'mass_diagnostic.json').exists():
        previous=json.loads((OUT/'mass_diagnostic.json').read_text())
        if previous.get('input_sha256')==fingerprints:cache={r['tag']:r for r in previous['records']}
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    src=np.load(ROOT/'analysis/results/p3_preparation/damped_group_forecast/source_rows_E10.5.npy')
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);base=x[src].toarray();del x
    models=json.loads((OUT/'training.json').read_text())['models']
    decision=json.loads((OUT/'decision.json').read_text())
    if (OUT/'full_training.json').exists() and decision['observed_gate_passed']:
        models+=json.loads((OUT/'full_training.json').read_text())['models']
    records=[]
    for m in models:
        checkpoint_hash=prior.digest(OUT/(m['tag']+'.pt'))
        if m['tag'] in cache and cache[m['tag']].get('checkpoint_sha256')==checkpoint_hash:
            records.append(cache[m['tag']]);continue
        ck=torch.load(OUT/(m['tag']+'.pt'),weights_only=True);net=StageDynamics(True);net.load_state_dict(ck['state']);net.eval()
        r=predict(net,decoder,base,2,True);first=r['mid_logw'].astype(float);second=r['logw'].astype(float)-first
        original=probabilities(r['logw']);originalmean=original@r['expression'];policies={}
        for label,factor in [('continue_mass',1.),('half_second_day_mass',.5),('freeze_after_day1',0.)]:
            w=probabilities(first+factor*second)
            policies[label]=dict(ess=float(1/(w@w)),weight_l1_from_continue=float(abs(w-original).sum()),
                pooled_mean_rmse_from_continue=float(np.sqrt(np.mean((w@r['expression']-originalmean)**2))))
        a=first-first.mean();b=second-second.mean();var1=float(np.mean(a*a));var2=float(np.mean(b*b));cov=float(np.mean(a*b))
        records.append(dict(tag=m['tag'],checkpoint_sha256=checkpoint_hash,arm=m['arm'],full_data=m['full_data'],daily_logmass_correlation=float(np.corrcoef(first,second)[0,1]),
            variance_day1=var1,variance_day2=var2,twice_cross_covariance=2*cov,cumulative_variance=float(np.var(first+second)),policies=policies))
        np.testing.assert_allclose(np.var(first+second),var1+var2+2*cov,rtol=1e-12)
    save('mass_diagnostic.json',dict(scope='Post-hoc sensitivity, not predictive accuracy; expressions fixed across policies',records=records,input_sha256=fingerprints))
    print('Mass diagnostic completed',len(records),'checkpoints',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
