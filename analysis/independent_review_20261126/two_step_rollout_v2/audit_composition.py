"""Post-hoc ancestry concentration audit; these are not inferred future cell labels."""
import anndata as ad
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def main():
    result=json.loads((OUT/'e125_results.json').read_text())
    mainmix=next(r for r in result['mixtures'] if r['arm']=='two_step' and r['policy']=='hold')
    dominant=int(np.argmax(mainmix['exact']['source_group_proportions']))
    old=ROOT/'analysis/results/p3_preparation/damped_group_forecast'
    src=np.load(old/'source_rows_E10.5.npy');base=ad.read_h5ad(old/'copylast_E12.5.h5ad')
    sp=joblib.load(REVIEW/'composition_forecast_trial/early_space.joblib')
    groups=sp['clusters'].predict(prior.project(base.X,sp['pca']))
    truth=ad.read_h5ad(ROOT/'data/p3/T1/E10.5_RNA.h5ad',backed='r')
    counts=truth.obs.iloc[src[groups==dominant]].celltype.value_counts()
    labels={str(k):int(v) for k,v in counts.items() if v>0};truth.file.close()
    earlier=np.load(REVIEW/'composition_forecast_trial/early_space.npz')
    fractions={'E8.5_all':float((earlier['g8']==dominant).mean()),'E9.5_all':float((earlier['g9']==dominant).mean()),
        'E10.5_fixed_2000_source':float((groups==dominant).mean())}
    policies={r['arm']+'/'+r['policy']:r['exact']['source_group_proportions'][dominant] for r in result['mixtures']}
    save('composition_audit.json',dict(scope='Post-hoc source ancestry audit; no E12.5 labels or truth',
        dominant_source_group=dominant,observed_proportions=fractions,source_group_celltype_counts=labels,
        predicted_ancestry_weights=policies,
        interpretation='A large ancestry weight is not an independently inferred future cell-type fraction. The E10.5 value uses the fixed source pool, not all E10.5 cells.'))
    print('Dominant source group',dominant,labels,policies,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
