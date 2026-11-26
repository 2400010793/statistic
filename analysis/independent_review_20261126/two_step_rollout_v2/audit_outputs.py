"""Numerical rollout sensitivity and exact frozen-checkpoint replay."""
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def main():
    torch.set_num_threads(2)
    training=json.loads((OUT/'training.json').read_text());rep=joblib.load(JOINT/'representation.joblib')
    d=np.load(V1/'data_splits.npz');f=np.load(OUT/'frozen_evaluation.npz')
    for stage in ['8','9']:
        for a,b in [('train','valid'),('train','test'),('valid','test')]:
            assert len(np.intersect1d(d[a+stage],d[b+stage]))==0
    for fold in range(2):
        for a,b in [('train','valid'),('train','test'),('valid','test')]:
            assert len(np.intersect1d(d[f'f{fold}_{a}10'],d[f'f{fold}_{b}10']))==0
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    src12=np.load(ROOT/'analysis/results/p3_preparation/damped_group_forecast/source_rows_E10.5.npy')
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);base12=x[src12].toarray();del x
    x=prior.read(ROOT/'data/raw/T1/E8.5_RNA.h5ad',panel);base8=x[d['test8']].toarray();del x
    decoder=Decoder(rep['pca'],rep['scale']);records=[]
    for m in training['models']:
        assert prior.digest(OUT/(m['tag']+'.pt'))==m['checkpoint_sha256']
        ck=torch.load(OUT/(m['tag']+'.pt'),weights_only=True);net=StageDynamics(True);net.load_state_dict(ck['state']);net.eval()
        assert sum(p.numel() for p in net.parameters())==29250
        checks=[]
        for name,base,start,key in [('observed',base8,0,'_two0'),('future',base12,2,'_future')]:
            r=predict(net,decoder,base,start,True)
            # Same chunk size and exact weights; replay all 2000 rows.
            np.testing.assert_array_equal(r['z'],f[m['tag']+key+'_z'])
            np.testing.assert_array_equal(r['logw'],f[m['tag']+key+'_logw'])
            refined=predict(net,decoder,base[:128],start,True,steps=32)
            coarse=r['expression'][:128];delta=coarse-base[:128]
            relative=float(np.linalg.norm(refined['expression']-coarse)/max(np.linalg.norm(delta),1e-12))
            w=probabilities(r['logw'][:128]);wr=probabilities(refined['logw'])
            checks.append(dict(path=name,expression_change_relative_to_coarse_edit=relative,weight_l1=float(abs(w-wr).sum())))
        records.append(dict(tag=m['tag'],checks=checks,replay_max_abs_difference=0.))
    maxchange=max(c['expression_change_relative_to_coarse_edit'] for r in records for c in r['checks'])
    save('output_audit.json',dict(disjoint_splits=True,frozen_replay_exact=True,records=records,
        max_16_to_32_step_expression_change_relative_to_edit=maxchange,
        interpretation='Numerical sensitivity, not biological accuracy; 128-row integration probe per checkpoint and path'))
    print('Exact replay passed; maximum integration sensitivity',maxchange,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
