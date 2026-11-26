"""Target-free two-day rollout stress, using existing diagnostic checkpoints."""
import sys
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent
REVIEW=OUT.parent
V1=REVIEW/'stage_conditioned_decoder_v1'
sys.path.insert(0,str(V1))
from model import ROOT,JOINT,prior,torch,StageDynamics,Decoder

def evolve(net,z,days):
    lw=torch.zeros(len(z))
    with torch.no_grad():
        for _ in range(round(16*days)):
            v,g,_=net.field(z,2.)
            z=z+v/16;lw=lw+g/16
    return z,lw

def decode(decoder,base,delta):
    with torch.no_grad():
        return np.concatenate([decoder(torch.from_numpy(base[s:s+128]),delta[s:s+128]).numpy() for s in range(0,len(base),128)])

def score(base,pred,z0,z_future,logw,pca,scale,src,decoder):
    logw=np.asarray(logw,dtype=float);w=np.exp(logw-logw.max());w/=w.sum()
    _,inverse=np.unique(src,return_inverse=True);unique_w=np.bincount(inverse,weights=w)
    actual=(prior.project(pred,pca)/scale)-z0
    wanted=z_future-z0;edit_energy=0.;bound=0;nonzero=0
    for s in range(0,len(base),128):
        b=base[s:s+128];pp=pred[s:s+128]
        shift=((wanted[s:s+128])*scale)@pca.components_
        gain=1+shift/(b+.25);nz=b>0
        bound+=int(np.sum(nz&((gain<=.8)|(gain>=1.2))));nonzero+=int(nz.sum())
        edit_energy+=np.square(pp.astype(float)-b).sum()
        np.testing.assert_array_equal(pp>0,b>0)
        np.testing.assert_allclose(np.expm1(pp.astype(float)).sum(1),10000,rtol=1e-5)
    return dict(ess_rows=float(1/(w@w)),ess_unique=float(1/(unique_w@unique_w)),
        max_relative_weight=float(len(w)*w.max()),top_one_percent_mass=float(np.sort(w)[-max(1,len(w)//100):].sum()),
        requested_latent_rms=float(np.sqrt(np.mean(wanted**2))),realized_latent_rms=float(np.sqrt(np.mean(actual**2))),
        latent_decoder_relative_residual=float(np.linalg.norm(actual-wanted)/max(np.linalg.norm(wanted),1e-12)),
        gene_edit_rms=float(np.sqrt(edit_energy/base.size)),active_entry_gain_bound_fraction=float(bound/nonzero),
        note='Distance from source and internal decoder mismatch; not error against E12.5')

def main():
    torch.set_num_threads(2)
    rep=joblib.load(JOINT/'representation.joblib');decoder=Decoder(rep['pca'],rep['scale'])
    card=json.loads((V1/'model_card.json').read_text())
    baseline_dir=ROOT/'analysis/results/p3_preparation/damped_group_forecast'
    src=np.load(baseline_dir/'source_rows_E10.5.npy')
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    x=prior.read(ROOT/'data/p3/T1/E10.5_RNA.h5ad',panel);base=x[src].toarray();del x
    copy=prior.read(baseline_dir/'copylast_E12.5.h5ad',panel);np.testing.assert_array_equal(base,copy);del copy
    z0=prior.project(base,rep['pca'])/rep['scale'];zt=torch.from_numpy(z0.astype(np.float32))
    paths=[Path(c['path']) for c in card['checkpoints']]+[V1/'model.py',V1/'model_card.json',baseline_dir/'source_rows_E10.5.npy',baseline_dir/'copylast_E12.5.h5ad']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    reports=[]
    for entry in card['checkpoints']:
        ck=torch.load(entry['path'],weights_only=True);net=StageDynamics(True);net.load_state_dict(ck['state']);net.eval()
        one,lw1=evolve(net,zt,1);two,lw2=evolve(net,zt,2)
        day1=decode(decoder,base,one-zt);day2=decode(decoder,base,two-zt)
        record=dict(fold=entry['fold'],seed=entry['seed'],one_day=score(base,day1,z0,one.numpy(),lw1.numpy(),rep['pca'],rep['scale'],src,decoder),
            two_days_one_decode=score(base,day2,z0,two.numpy(),lw2.numpy(),rep['pca'],rep['scale'],src,decoder))
        reencoded=prior.project(day1,rep['pca'])/rep['scale'];mid=torch.from_numpy(reencoded.astype(np.float32))
        end,lw_second=evolve(net,mid,1);stepwise=decode(decoder,day1,end-mid)
        z_end=prior.project(stepwise,rep['pca'])/rep['scale'];wpath=np.exp((lw1+lw_second).numpy().astype(float));wpath/=wpath.sum()
        wdirect=np.exp(lw2.numpy().astype(float));wdirect/=wdirect.sum()
        record['decode_each_day']=dict(ess_rows=float(1/(wpath@wpath)),
            expression_rms_difference_vs_end_only=float(np.sqrt(sum(np.square(stepwise[s:s+128].astype(float)-day2[s:s+128]).sum() for s in range(0,len(base),128))/base.size)),
            pooled_gene_rmse_between_paths=float(np.sqrt(np.mean((wpath@stepwise-wdirect@day2)**2))),
            weight_l1_between_paths=float(abs(wpath-wdirect).sum()),
            latent_rms_difference_between_decoded_paths=float(np.sqrt(np.mean((z_end-prior.project(day2,rep['pca'])/rep['scale'])**2))),
            limitation='Repeated decoding also compounds the edit budget; no claim either path is biologically correct')
        reports.append(record)
        print(entry['fold'],entry['seed'],'ESS one/two',round(record['one_day']['ess_rows']),round(record['two_days_one_decode']['ess_rows']),
            'bounds',round(record['one_day']['active_entry_gain_bound_fraction'],3),round(record['two_days_one_decode']['active_entry_gain_bound_fraction'],3),flush=True)
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    summary=dict(scope='E12.5 readiness stress only; no E12.5 truth, forecast selection or accuracy estimate',
        source='Same 2000 E10.5 rows as existing P3 CopyLast and damped_group',source_indices=src.tolist(),
        time_policy='Hold time condition at E10.5, state-dependent fields continue',reports=reports,
        input_sha256=hashes,inputs_unchanged=unchanged)
    (OUT/'stress_results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
