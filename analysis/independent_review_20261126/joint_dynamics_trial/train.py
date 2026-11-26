"""Fit and freeze using E8.5/E9.5 only. No import of evaluation code."""
import time
import copy
import platform
import joblib
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from core import OUT,PREV,ROOT,SEEDS,prior,np,torch,Dynamics,distribution_loss,save,decode,edit_rms,allocation

def prepare():
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    rng=np.random.default_rng(2026112600)
    x=[];sample=[];indices=[]
    for stage in ['E8.5','E9.5']:
        xx=prior.read(ROOT/f'data/raw/T1/{stage}_RNA.h5ad',panel)
        ix=rng.choice(xx.shape[0],1500,replace=False)
        sample.append(xx[ix].toarray());indices.append(ix);x.append(xx)
    pca=PCA(32,svd_solver='randomized',random_state=2026112600).fit(np.vstack(sample))
    z=[prior.project(xx,pca) for xx in x]
    scale=float(np.std(np.vstack(z),axis=0).mean())
    splits={}; train=[];valid=[]
    for j,zz in enumerate(z):
        perm=rng.permutation(len(zz));n=int(.8*len(zz))
        splits[f'train{j}']=perm[:n];splits[f'valid{j}']=perm[n:]
        train.append(torch.from_numpy((zz[perm[:n]]/scale).astype(np.float32)))
        valid.append(torch.from_numpy((zz[perm[n:n+768]]/scale).astype(np.float32)))
    zz=np.vstack([a.numpy() for a in train]);band=float(np.median(pdist(zz[rng.choice(len(zz),1000,False)],'sqeuclidean')))
    joblib.dump(dict(pca=pca,scale=scale,bandwidth=band),OUT/'representation.joblib')
    np.savez_compressed(OUT/'early_latent.npz',z8=z[0],z9=z[1],pca_sample8=indices[0],pca_sample9=indices[1],**splits)
    print('Representation ready',scale,band,flush=True)
    return panel,x[1],pca,scale,z[1],band,train,valid

def fit(seed,kind,train,valid,band):
    torch.manual_seed(seed);net=Dynamics(growth=kind=='joint')
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    with torch.no_grad():
        base,parts=distribution_loss(valid[0],valid[1],torch.full((len(valid[0]),),1/len(valid[0])),band)
    best=float('inf');history=[];beststep=0;start=time.time()
    for step in range(1,1201):
        a=train[0][rng.choice(len(train[0]),384,False)];b=train[1][rng.choice(len(train[1]),384,False)]
        zz,w,_,energy,genergy=net(a)
        fitloss,_=distribution_loss(zz,b,w,band)
        loss=fitloss+.005*energy+.005*genergy
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step()
        if step%100==0:
            with torch.no_grad():
                pred,w,lg,e,ge=net(valid[0]);val,parts=distribution_loss(pred,valid[1],w,band)
            val=float(val);history.append(dict(step=step,training_loss=float(loss.detach()),validation_loss=val,**parts))
            if val<best:
                best=val;beststep=step;beststate=copy.deepcopy(net.state_dict())
            print(seed,kind,step,'val',round(val,7),'identity',round(float(base),7),'sec',round(time.time()-start),flush=True)
    net.load_state_dict(beststate);net.eval()
    torch.save(dict(state=beststate,growth=kind=='joint',seed=seed,best_step=beststep),OUT/f'{kind}_{seed}.pt')
    return net,dict(seed=seed,kind=kind,parameters=sum(p.numel() for p in net.parameters()),best_step=beststep,
        validation_identity_loss=float(base),validation_best_loss=best,history=history,seconds=time.time()-start)

def main():
    start=time.time();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    paths=[ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]+[PREV/'frozen_controls.npz',OUT/'protocol.txt',OUT/'train.py',OUT/'core.py']
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    panel,latest,pca,scale,z9,band,train,valid=prepare()
    controls=np.load(PREV/'frozen_controls.npz');sources=controls['sources'];groups=controls['source_groups']
    frozen=dict(sources=sources,source_groups=groups);reports=[];decode_reports=[]
    base=latest[sources[0]].toarray()
    for seed in SEEDS:
        for kind in ['velocity','joint']:
            net,report=fit(seed,kind,train,valid,band);reports.append(report)
            for duration in [1.,.5]:
                prefix=f'{kind}_{seed}_h{duration:g}'
                future=[];logw=[];orders=[]
                for b,src in enumerate(sources):
                    with torch.no_grad():
                        zz,w,lw,_,_=net(torch.from_numpy((z9[src]/scale).astype(np.float32)),duration=duration)
                    future.append(zz.numpy()*scale);logw.append(lw.numpy())
                    weights=np.exp(lw.numpy().astype(float));weights/=weights.sum()
                    orders.append(allocation(weights,b))
                frozen[prefix+'_future']=np.stack(future);frozen[prefix+'_logw']=np.stack(logw);frozen[prefix+'_order']=np.stack(orders)
                if kind=='joint':
                    delta=(future[0]-z9[sources[0]])@pca.components_
                    bound=decode(base,delta);target=edit_rms(bound,base)
                    lo,hi=0.,1.
                    for _ in range(22):
                        mid=(lo+hi)/2
                        if edit_rms(decode(base,delta,True,mid),base)<target:lo=mid
                        else:hi=mid
                    alpha=(lo+hi)/2;realized=edit_rms(decode(base,delta,True,alpha),base)
                    frozen[prefix+'_open_alpha']=np.array(alpha)
                    decode_reports.append(dict(prefix=prefix,alpha=alpha,bounded_edit_rms=target,open_edit_rms=realized,
                        rms_relative_error=abs(realized-target)/target))
                    assert abs(realized-target)/target<1e-4,'Decoder energy matching outside alpha [0,1]'
    np.savez_compressed(OUT/'frozen_predictions.npz',**frozen)
    unchanged={str(p.relative_to(ROOT)):prior.digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths};assert all(unchanged.values())
    save('training.json',dict(scope='E8.5/E9.5 only; E10.5 not opened in train.py',models=reports,decoder=decode_reports,
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        frozen_sha256=prior.digest(OUT/'frozen_predictions.npz'),representation_sha256=prior.digest(OUT/'representation.joblib'),
        early_latent_sha256=prior.digest(OUT/'early_latent.npz'),input_sha256=hashes,inputs_unchanged=unchanged,seconds=time.time()-start))
    print('FROZEN ALL MODELS',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
