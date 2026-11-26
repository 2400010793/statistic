"""Post-hoc hard-growth-bound ablation; retain the original regularization."""
import copy
import time
import json
import joblib
from threadpoolctl import threadpool_limits
from helpers import OUT,JOINT,ROOT,old,prior,np,torch,SEEDS,Calibrated,load_original,save

class Unbounded(Calibrated):
    def __init__(self,original,mode):
        super().__init__(original,mode)
        with torch.no_grad():self.head.weight[32].mul_(np.log(4));self.head.bias[32].mul_(np.log(4))

    def field(self,z):
        out=self.head(self.trunk(z));g=out[:,32]
        if self.mode=='full':return out[:,:32],g,torch.ones(len(z))
        vo=self.original.head(self.original.trunk(z))[:,:32]
        s=8*torch.sigmoid(out[:,0]-np.log(7))
        return vo*s[:,None],g,s

def fit(seed,fold,mode,original,training,validation,band):
    torch.manual_seed(seed+fold*100);net=Unbounded(original,mode)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed+10000*fold);best=float('inf');beststep=0;history=[]
    for step in range(1201):
        if step:
            x=training[0][rng.choice(len(training[0]),384,False)];y=training[1][rng.choice(len(training[1]),384,False)]
            z,w,_,e,ge=net(x);dist,_=old.distribution_loss(z,y,w,band);loss=dist+.005*e+.005*ge
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
        if step%100==0:
            with torch.no_grad():
                z,w,_,_,_=net(validation[0]);val,parts=old.distribution_loss(z,validation[1],w,band)
            val=float(val);history.append(dict(step=step,validation=val,**parts))
            if val<best:best=val;beststep=step;state=copy.deepcopy(net.state_dict())
            if step%400==0:print('unbounded',fold,seed,mode,step,round(val,7),flush=True)
    net.load_state_dict(state);net.eval()
    name=f'f{fold}_s{seed}_{mode}_unbounded';torch.save(dict(state=state,mode=mode,seed=seed,fold=fold,best_step=beststep),OUT/f'{name}.pt')
    return net,dict(fold=fold,seed=seed,mode=mode,validation_best=best,best_step=beststep,history=history)

def main():
    start=time.time();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    original_fit=json.loads((OUT/'calibration.json').read_text());assert prior.digest(OUT/'data_splits.npz')==original_fit['data_sha256']
    original_hash=prior.digest(OUT/'frozen_predictions.npz')
    d=np.load(OUT/'data_splits.npz');rep=joblib.load(JOINT/'representation.joblib');scale=rep['scale'];band=rep['bandwidth']
    tensor=lambda x:torch.from_numpy((x/scale).astype(np.float32))
    train9=tensor(d['z9'][d['train9']]);val9=tensor(d['z9'][d['valid9'][:768]])
    arrays={};reports=[];diagnostics=[]
    for fold in range(2):
        tr=[train9,tensor(d['z10'][d[f'f{fold}_train10']])];val=[val9,tensor(d['z10'][d[f'f{fold}_valid10'][:768]])]
        for seed in SEEDS:
            original=load_original(seed)
            for mode in ['flex_mass','full']:
                net,report=fit(seed,fold,mode,original,tr,val,band);reports.append(report)
                future=[];logw=[]
                for b,src in enumerate(d['sources']):
                    x=tensor(d['z9'][src])
                    with torch.no_grad():
                        z,w,lw,_,_=net(x);v,g,s=net.field(x)
                        if b==0:z32,w32,_,_,_=net(x,steps=32)
                    future.append(z.numpy()*scale);logw.append(lw.numpy())
                    if b==0:
                        diagnostics.append(dict(fold=fold,seed=seed,mode=mode,growth_range=[float(g.min()),float(g.max())],
                            logweight_range=[float(lw.min()),float(lw.max())],effective_rows=float(1/w.square().sum()),
                            speed_near_upper_fraction=float((s>=7.6).float().mean()),
                            relative_displacement_16_to_32=float(torch.linalg.vector_norm(z32-z)/torch.linalg.vector_norm(z-x)),
                            weight_l1_16_to_32=float(abs(w32-w).sum())))
                prefix=f'f{fold}_s{seed}_{mode}_unbounded';arrays[prefix+'_future']=np.stack(future);arrays[prefix+'_logw']=np.stack(logw)
    np.savez_compressed(OUT/'unbounded_predictions.npz',**arrays)
    assert prior.digest(OUT/'frozen_predictions.npz')==original_hash
    save('unbounded_calibration.json',dict(scope='Post-hoc removal of growth hard bound; same-stage diagnostic',
        models=reports,diagnostics=diagnostics,parameters_sha256=prior.digest(OUT/'unbounded_predictions.npz'),
        original_parameters_sha256=original_hash,original_parameters_unchanged=True,data_sha256=original_fit['data_sha256'],
        amendment_sha256=prior.digest(OUT/'growth_cap_amendment.txt'),script_sha256=prior.digest(OUT/'unbounded_mass.py'),seconds=time.time()-start))
    print('UNBOUNDED MODELS FROZEN',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
