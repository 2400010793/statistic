"""Paired single-day versus two-day training; no hidden-stage truth."""
import time
import copy
import argparse
import joblib
from threadpoolctl import threadpool_limits
from rollout import *

def moments(z):
    m=z.mean(0);c=z-m;return m,c.T@c/len(z)

def load_data():
    data=np.load(V1/'data_splits.npz');rep=joblib.load(JOINT/'representation.joblib')
    panel=(ROOT/'panels/T1__test.genes.txt').read_text().splitlines()
    paths=[ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]+[ROOT/'data/p3/T1/E10.5_RNA.h5ad']
    source=[prior.read(p,panel) for p in paths]
    return data,rep,source

def setup(data,rep,source,fold,full):
    srcrows=[np.arange(source[p].shape[0]) for p in range(2)] if full else [data['train8'],data['train9']]
    tarrows=[np.arange(source[p].shape[0]) for p in [1,2]] if full else [data['train9'],data[f'f{fold}_train10']]
    allz=[data['z9']/rep['scale'],data['z10']/rep['scale']]
    gene=[torch.from_numpy((np.asarray(source[p+1][rows].sum(0,dtype=float)).ravel()/len(rows)).astype(np.float32)) for p,rows in enumerate(tarrows)]
    valid=[]
    if not full:
        for p,(sr,tr) in enumerate([(data['valid8'],data['valid9']),(data['valid9'],data[f'f{fold}_valid10'])]):
            tz=torch.from_numpy(allz[p][tr].astype(np.float32))
            gm=np.asarray(source[p+1][tr].sum(0,dtype=float)).ravel()/len(tr)
            valid.append(dict(base=torch.from_numpy(source[p][sr[:768]].toarray()),target=tz[:768],moments=moments(tz),gene=torch.from_numpy(gm.astype(np.float32))))
    return srcrows,tarrows,allz,gene,valid

def validate(net,decoder,valid,band):
    with torch.no_grad():
        first,two=pair(net,decoder,valid[0]['base'],0)
        late=day(net,decoder,valid[1]['base'],1)
        records=[]
        for r,v in [(first,valid[0]),(late,valid[1]),(two,valid[1])]:
            q=torch.full((len(r['z']),),1/len(r['z']))
            _,report=loss(r,v['target'],v['gene'],q,band,v['moments'],False);records.append(report)
        return .5*sum(r['selection'] for r in records),records

def fit(seed,fold,arm,data,rep,source,steps=1000,full=False):
    tag=f'full_s{seed}_{arm}' if full else f'f{fold}_s{seed}_{arm}'
    torch.manual_seed(seed+100*fold);net=StageDynamics(True);decoder=Decoder(rep['pca'],rep['scale'])
    rng=np.random.default_rng(seed+10000*fold)
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    sr,tr,zall,gene,valid=setup(data,rep,source,fold,full)
    groups=[data['groups8'],data['groups9']];band=rep['bandwidth']
    best=float('inf');beststep=0;history=[];start=time.time();state=None
    for step in range(steps+1):
        if step:
            batches=[]
            # Draw all exogenous random samples before either arm's forward operations.
            for p in range(2):
                ix,qp=stratified_batch(rng,sr[p],groups[p]);yi=rng.choice(tr[p],384,False)
                batches.append((torch.from_numpy(source[p][ix].toarray()),torch.from_numpy(qp),torch.from_numpy(zall[p][yi].astype(np.float32))))
            b0,q0,y0=batches[0];b1,q1,y1=batches[1]
            if arm=='two_step':r0,r2=pair(net,decoder,b0,0)
            else:r0=day(net,decoder,b0,0)
            r1=day(net,decoder,b1,1)
            l0,_=loss(r0,y0,gene[0],q0,band);l1,_=loss(r1,y1,gene[1],q1,band)
            value=.5*(l0+l1)
            if arm=='two_step':
                l2,_=loss(r2,y1,gene[1],q0,band);value=value+.5*l2
            assert torch.isfinite(value)
            opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
            # Drop large graphs before validation or the next update.
            training_loss=float(value.detach())
            del value,l0,l1,r0,r1,batches
            if arm=='two_step':del l2,r2
        if step%100==0 or step==steps:
            if full:
                selection=None;details=[]
                if step==steps:state=copy.deepcopy(net.state_dict());beststep=step
            else:
                selection,details=validate(net,decoder,valid,band)
                if selection<best:best=selection;beststep=step;state=copy.deepcopy(net.state_dict())
            history.append(dict(step=step,selection=selection,details=details,training_loss=training_loss if step else None))
            print(tag,step,'selection',None if selection is None else round(selection,6),
                'ESS',[round(r['ess_fraction'],3) for r in details],'seconds',round(time.time()-start),flush=True)
    net.load_state_dict(state);net.eval()
    torch.save(dict(state=state,arm=arm,seed=seed,fold=fold,full_data=full,best_step=beststep),OUT/f'{tag}.pt')
    return dict(tag=tag,seed=seed,fold=fold,arm=arm,full_data=full,best_step=beststep,
        best_selection=None if full else best,history=history,seconds=time.time()-start,
        checkpoint_sha256=prior.digest(OUT/f'{tag}.pt'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    paths=[OUT/'protocol.txt',OUT/'rollout.py',OUT/'train.py',V1/'data_splits.npz',V1/'model.py',JOINT/'representation.joblib']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    data,rep,source=load_data();reports=[];start=time.time()
    if args.full:
        decision=json.loads((OUT/'decision.json').read_text());assert decision['observed_gate_passed']
        budget=decision['full_refit_updates'];folds=[0]
    else:budget={a:1000 for a in ARMS};folds=range(2)
    for fold in folds:
        for seed in SEEDS:
            for arm in ARMS:
                reports.append(fit(seed,fold,arm,data,rep,source,budget[arm],args.full))
                save(('full_' if args.full else '')+'training_partial.json',dict(models=reports))
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save(('full_' if args.full else '')+'training.json',dict(models=reports,input_sha256=hashes,inputs_unchanged=unchanged,
        seconds=time.time()-start,torch_version=torch.__version__,scope='P3 full fit' if args.full else 'Observed stages, held-out cells, retrospective'))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
