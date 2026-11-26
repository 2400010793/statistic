"""Paired stage/ESS/decoder ablations, trained on calibration partitions only."""
import time
import copy
import json
import joblib
from threadpoolctl import threadpool_limits
from model import OUT,REVIEW,JOINT,ROOT,prior,np,torch,SEEDS,ARMS,StageDynamics,Decoder,distribution,weights_and_penalty,stratified_batch,save

def target_moments(z):
    mean=z.mean(0);c=z-mean;return mean,c.T@c/len(z)

def validate(net,decoder,source,validation,band,with_ess):
    reports=[]
    with torch.no_grad():
        for phase,v in enumerate(validation):
            ix=v['source'];base=torch.from_numpy(source[phase][ix].toarray());z=v['zsource']
            future,lw,_,_,_=net(z,phase);q=torch.full((len(ix),),1/len(ix));w,pen,ess=weights_and_penalty(lw,q)
            # Chunking has no numerical or per-cell dependence on other rows.
            pred=torch.cat([decoder(base[s:s+128],future[s:s+128]-z[s:s+128]) for s in range(0,len(ix),128)])
            decoded_z=decoder.project(pred)
            dist=distribution(decoded_z,v['target'],w,band,v['moments'])
            gene=(w@pred-v['gene']).square().mean()
            objective=dist+10*gene
            reports.append(dict(phase=phase,decoded_objective=float(objective),gene_mean_mse=float(gene),
                selection=float(objective+(pen if with_ess else 0)),ess_fraction=float(ess)))
    return float(np.mean([v['selection'] for v in reports])),reports

def fit(seed,fold,arm,source,data,mom,rep):
    torch.manual_seed(seed+100*fold);net=StageDynamics(arm!='autonomous_latent');decoder=Decoder(rep['pca'],rep['scale'])
    with_ess=arm!='temporal_noess';rng=np.random.default_rng(seed+10000*fold)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    scale=rep['scale'];band=rep['bandwidth'];source_z=[data['z8']/scale,data['z9']/scale]
    srcrows=[data['train8'],data['train9']];groups=[data['groups8'],data['groups9']]
    targetrows=[data['train9'],data[f'f{fold}_train10']];targetall=[data['z9']/scale,data['z10']/scale]
    train_gene=[torch.from_numpy(mom['train9_gene_mean']),torch.from_numpy(mom[f'f{fold}_train10_gene_mean'])]
    validation=[]
    for phase,(sr,tar,gkey) in enumerate([(data['valid8'],data['valid9'],'valid9_gene_mean'),(data['valid9'],data[f'f{fold}_valid10'],f'f{fold}_valid10_gene_mean')]):
        fulltarget=torch.from_numpy(targetall[phase][tar].astype(np.float32));ix=sr[:768]
        validation.append(dict(source=ix,zsource=torch.from_numpy(source_z[phase][ix].astype(np.float32)),
            target=fulltarget[:768],moments=target_moments(fulltarget),gene=torch.from_numpy(mom[gkey])))
    history=[];best=float('inf');beststep=0;start=time.time()
    for step in range(2001):
        if step:
            phase=(step-1)%2;ix,qp=stratified_batch(rng,srcrows[phase],groups[phase])
            yix=rng.choice(targetrows[phase],384,False)
            base=torch.from_numpy(source[phase][ix].toarray());z=torch.from_numpy(source_z[phase][ix].astype(np.float32))
            y=torch.from_numpy(targetall[phase][yix].astype(np.float32));q=torch.from_numpy(qp)
            future,lw,energy,genergy,res=net(z,phase);w,pen,ess=weights_and_penalty(lw,q)
            latent=distribution(future,y,w,band)
            if arm=='temporal_decoded':
                pred=decoder(base,future-z);main=distribution(decoder.project(pred),y,w,band);gene_mean=w@pred
            else:main=latent;gene_mean=decoder.proxy_mean(base,future-z,w)
            gene=(gene_mean-train_gene[phase]).square().mean()
            loss=main+10*gene+.1*latent+.005*(q@energy+q@genergy)+.001*(q@res)
            if with_ess:loss=loss+pen
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
        if step%200==0:
            value,details=validate(net,decoder,source,validation,band,with_ess)
            history.append(dict(step=step,selection=value,phases=details,training_loss=float(loss.detach()) if step else None))
            if value<best:best=value;beststep=step;state=copy.deepcopy(net.state_dict())
            print('fit',fold,seed,arm,step,'selection',round(value,6),'ESS',[round(v['ess_fraction'],3) for v in details],'sec',round(time.time()-start),flush=True)
    net.load_state_dict(state);net.eval()
    key=f'f{fold}_s{seed}_{arm}';torch.save(dict(state=state,arm=arm,seed=seed,fold=fold,best_step=beststep),OUT/f'{key}.pt')
    return net,dict(fold=fold,seed=seed,arm=arm,best_step=beststep,best_selection=best,history=history,
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),seconds=time.time()-start)

def main():
    start=time.time();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    prep=json.loads((OUT/'preparation.json').read_text());assert prior.digest(OUT/'data_splits.npz')==prep['data_sha256']
    assert prior.digest(OUT/'target_moments.npz')==prep['moments_sha256']
    hashes=dict(prep['input_sha256'])
    for p in [OUT/'model.py',OUT/'train.py',OUT/'target_moments.npz']:hashes[str(p.relative_to(ROOT))]=prior.digest(p)
    data=np.load(OUT/'data_splits.npz');mom=np.load(OUT/'target_moments.npz');rep=joblib.load(JOINT/'representation.joblib')
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    source=[prior.read(ROOT/f'data/raw/T1/{s}_RNA.h5ad',panel) for s in ['E8.5','E9.5']]
    frozen={};reports=[]
    for fold in range(2):
        for seed in SEEDS:
            for arm in ARMS:
                net,report=fit(seed,fold,arm,source,data,mom,rep);reports.append(report)
                key=f'f{fold}_s{seed}_{arm}'
                pools=[(0,data['test8'])]+[(1,src) for src in data['sources']]
                futures=[];logw=[]
                for phase,src in pools:
                    zz=data['z8'][src] if phase==0 else data['z9'][src]
                    with torch.no_grad():out,lw,_,_,_=net(torch.from_numpy((zz/rep['scale']).astype(np.float32)),phase)
                    futures.append(out.numpy()*rep['scale']);logw.append(lw.numpy())
                frozen[key+'_future']=np.stack(futures);frozen[key+'_logw']=np.stack(logw)
                save('training_partial.json',dict(models=reports))
    np.savez_compressed(OUT/'frozen_predictions.npz',**frozen)
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('training.json',dict(scope='Two observed intervals; target-stage calibrated retrospective fits',models=reports,
        input_sha256=hashes,inputs_unchanged=unchanged,data_sha256=prep['data_sha256'],
        frozen_sha256=prior.digest(OUT/'frozen_predictions.npz'),seconds=time.time()-start,torch_version=torch.__version__))
    print('ALL MODELS FROZEN',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
