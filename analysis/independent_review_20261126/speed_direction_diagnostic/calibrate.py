"""Fit on the designated calibration cells; never score outer held-out cells."""
import time
import copy
import joblib
import json
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from helpers import OUT,REVIEW,JOINT,ROOT,old,prior,np,torch,SEEDS,MODES,STEPS,Calibrated,load_original,save

def choose_clock(net,validation,band,negative=False):
    x,y=validation
    grid=[-4,-3,-2,-1.5,-1,-.75,-.5,-.25,-.125,0] if negative else [0,.125,.25,.5,.75,1,1.5,2,3,4,6,8]
    curve=[]
    def fun(tau):
        with torch.no_grad():
            z,w,_,_,_=net(x,duration=float(tau),steps=STEPS)
            loss,parts=old.distribution_loss(z,y,w,band)
        value=float(loss);curve.append(dict(tau=float(tau),loss=value,**parts));return value
    losses=[fun(t) for t in grid];j=int(np.argmin(losses))
    lo,hi=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
    if hi>lo:minimize_scalar(fun,bounds=(lo,hi),method='bounded',options={'xatol':.002,'maxiter':40})
    best=min(curve,key=lambda r:r['loss'])
    return best['tau'],dict(selected=best,curve=curve,negative=negative)

def train(seed,fold,mode,original,training,validation,band):
    torch.manual_seed(seed+fold*100);net=Calibrated(original,mode)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed+10000*fold)
    history=[];best=float('inf');beststep=0;start=time.time()
    for step in range(1201):
        if step:
            x=training[0][rng.choice(len(training[0]),384,False)];y=training[1][rng.choice(len(training[1]),384,False)]
            z,w,_,energy,genergy=net(x);fitloss,_=old.distribution_loss(z,y,w,band)
            loss=fitloss+.005*energy+.005*genergy
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
        if step%100==0:
            with torch.no_grad():
                z,w,_,_,_=net(validation[0]);val,parts=old.distribution_loss(z,validation[1],w,band)
            value=float(val);history.append(dict(step=step,validation=value,training_loss=float(loss.detach()) if step else None,**parts))
            if value<best:best=value;beststep=step;beststate=copy.deepcopy(net.state_dict())
            if step%400==0:print('fit',fold,seed,mode,step,'val',round(value,6),'sec',round(time.time()-start,1),flush=True)
    net.load_state_dict(beststate);net.eval()
    name=f'f{fold}_s{seed}_{mode}'
    torch.save(dict(state=beststate,mode=mode,seed=seed,fold=fold,best_step=beststep),OUT/f'{name}.pt')
    return net,dict(seed=seed,fold=fold,mode=mode,best_step=beststep,validation_best=best,history=history,seconds=time.time()-start)

def field_report(net,original,x,mode):
    with torch.no_grad():
        vo=original.head(original.trunk(x))[:,:32]
        vn,_,scale=net.field(x)
    a=vo.numpy().astype(float);b=vn.numpy().astype(float);s=scale.numpy()
    an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1)
    cosine=np.sum(a*b,axis=1)/np.maximum(an*bn,1e-15)
    parallel=np.sum(a*b,axis=1)/np.maximum(an**2,1e-15)
    orth=b-parallel[:,None]*a
    return dict(mode=mode,cosine_quantiles=np.quantile(cosine,[0,.1,.5,.9,1]).tolist(),
        negative_direction_fraction=float(np.mean(cosine<0)),orthogonal_energy_fraction=float(np.sum(orth**2)/np.sum(b**2)),
        best_global_velocity_scale=float(np.sum(a*b)/np.sum(a*a)),
        speed_quantiles=np.quantile(s,[0,.1,.5,.9,.99,1]).tolist(),speed_near_upper_fraction=float(np.mean(s>=7.6)))

def main():
    start=time.time();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    splits=json.loads((OUT/'split_manifest.json').read_text());assert prior.digest(OUT/'data_splits.npz')==splits['data_sha256']
    hashes=dict(splits['input_sha256'])
    for p in [OUT/'helpers.py',OUT/'calibrate.py']+[JOINT/f'joint_{s}.pt' for s in SEEDS]:hashes[str(p.relative_to(ROOT))]=prior.digest(p)
    d=np.load(OUT/'data_splits.npz');space=joblib.load(JOINT/'representation.joblib');scale=space['scale'];band=space['bandwidth']
    tensor=lambda x:torch.from_numpy((x/scale).astype(np.float32))
    z9=d['z9'];sources=d['sources'];validation9=tensor(z9[d['valid9'][:768]])
    training9=tensor(z9[d['train9']]);probe=tensor(z9[np.unique(sources)])
    frozen={};reports=[];clocks=[];fields=[];integrator=[]
    for fold in range(2):
        training=[training9,tensor(d['z10'][d[f'f{fold}_train10']])]
        validation=[validation9,tensor(d['z10'][d[f'f{fold}_valid10'][:768]])]
        for seed in SEEDS:
            original=load_original(seed)
            pos,posreport=choose_clock(original,validation,band)
            neg,negreport=choose_clock(original,validation,band,True)
            clocks.append(dict(fold=fold,seed=seed,clock=posreport,reverse=negreport))
            print('clocks',fold,seed,'positive',pos,'reverse',neg,flush=True)
            for arm,tau in [('frozen',1.),('clock',pos),('reverse',neg)]:
                futures=[];logw=[]
                for src in sources:
                    with torch.no_grad():zz,_,lw,_,_=original(tensor(z9[src]),duration=tau,steps=STEPS)
                    futures.append(zz.numpy()*scale);logw.append(lw.numpy())
                key=f'f{fold}_s{seed}_{arm}';frozen[key+'_future']=np.stack(futures);frozen[key+'_logw']=np.stack(logw)
            with torch.no_grad():
                x=tensor(z9[sources[0]]);z4,_,l4,_,_=original(x,steps=4);z16,_,l16,_,_=original(x,steps=16)
            integrator.append(dict(fold=fold,seed=seed,original_4_to_16_relative_displacement=float(torch.linalg.vector_norm(z4-z16)/torch.linalg.vector_norm(z4-x))))
            for mode in MODES:
                net,report=train(seed,fold,mode,original,training,validation,band);reports.append(report)
                fields.append(dict(fold=fold,seed=seed,**field_report(net,original,probe,mode)))
                futures=[];logw=[]
                for src in sources:
                    with torch.no_grad():zz,_,lw,_,_=net(tensor(z9[src]))
                    futures.append(zz.numpy()*scale);logw.append(lw.numpy())
                key=f'f{fold}_s{seed}_{mode}';frozen[key+'_future']=np.stack(futures);frozen[key+'_logw']=np.stack(logw)
            save('calibration_partial.json',dict(models=reports,clocks=clocks,fields=fields))
    np.savez_compressed(OUT/'frozen_predictions.npz',**frozen)
    unchanged={p:prior.digest(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    save('calibration.json',dict(scope='Target-calibrated diagnostic; outer held-out target cells not used to select parameters',
        models=reports,clocks=clocks,fields=fields,integrator=integrator,input_sha256=hashes,inputs_unchanged=unchanged,
        frozen_sha256=prior.digest(OUT/'frozen_predictions.npz'),data_sha256=prior.digest(OUT/'data_splits.npz'),
        seconds=time.time()-start,torch_version=torch.__version__))
    print('ALL CALIBRATIONS FROZEN',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
