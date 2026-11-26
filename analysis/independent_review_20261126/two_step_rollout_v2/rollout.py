"""Two-day decoded feedback, cumulative ancestry mass, explicit edit budgets."""
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
V1=OUT.parent/'stage_conditioned_decoder_v1'
spec=importlib.util.spec_from_file_location('v1_model',V1/'model.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
torch=v1.torch;ROOT=v1.ROOT;REVIEW=v1.REVIEW;JOINT=v1.JOINT;prior=v1.prior
StageDynamics=v1.StageDynamics;distribution=v1.distribution
weights_and_penalty=v1.weights_and_penalty;stratified_batch=v1.stratified_batch
SEEDS=v1.SEEDS;ARMS=['single_step','two_step']

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class Decoder(v1.Decoder):
    def forward(self,base,delta,anchor=None,diagnostics=False):
        shift=(delta*self.scale)@self.axes
        gain=1+shift/(base+.25)
        edited=base*torch.clamp(gain,.8,1.2)
        total_bound=torch.zeros_like(base,dtype=torch.bool)
        if anchor is not None:
            total_bound=(edited<anchor*.64)|(edited>anchor*1.44)
            edited=torch.maximum(torch.minimum(edited,anchor*1.44),anchor*.64)
        linear=torch.expm1(edited);pred=torch.log1p(linear*(10000/linear.sum(1,keepdim=True)))
        if diagnostics:
            nz=base>0
            return pred,dict(daily_bound_count=int((nz&((gain<=.8)|(gain>=1.2))).sum()),
                total_bound_count=int((nz&total_bound).sum()),active_count=int(nz.sum()))
        return pred

def day(net,decoder,base,start,anchor=None,multipliers=(1.,1.),steps=16,diagnostics=False):
    z=decoder.project(base);z0=z
    lw=z.new_zeros(len(z));energy=lw.clone();growth=lw.clone();res=lw.clone()
    for j in range(steps):
        time=min(2.,start+j/steps)
        v,g,r=net.field(z,time);v=v*multipliers[0];g=g*multipliers[1]
        z=z+v/steps;lw=lw+g/steps
        energy=energy+v.square().mean(1)/steps;growth=growth+g.square()/steps
        res=res+(r[:,:32].square().mean(1)+r[:,32].square())/steps
    pred=decoder(base,z-z0,anchor=anchor,diagnostics=diagnostics)
    diag={}
    if diagnostics:pred,diag=pred
    return dict(expression=pred,z=decoder.project(pred),intended=z,z0=z0,logw=lw,
        energy=energy,growth=growth,res=res,diagnostics=diag)

def pair(net,decoder,base,start,policy='hold',steps=16,diagnostics=False):
    first=day(net,decoder,base,start,steps=steps,diagnostics=diagnostics)
    factors=(.5,.5) if policy=='damped' else (1.,1.)
    second=day(net,decoder,first['expression'],start+1,anchor=base,multipliers=factors,steps=steps,diagnostics=diagnostics)
    second['logw']=first['logw']+second['logw']
    for key in ['energy','growth','res']:second[key]=(first[key]+second[key])/2
    return first,second

def loss(result,target,gene,q,band,moments=None,auxiliary=True):
    w,pen,ess=weights_and_penalty(result['logw'],q)
    actual=distribution(result['z'],target,w,band,moments)+10*(w@result['expression']-gene).square().mean()
    value=actual+pen
    if auxiliary:
        value=value+.1*distribution(result['intended'],target,w,band,moments)
        value=value+.005*(q@result['energy']+q@result['growth'])+.001*(q@result['res'])
    return value,dict(objective=float(actual.detach()),selection=float((actual+pen).detach()),ess_fraction=float(ess.detach()))

def predict(net,decoder,base,start,two=False,policy='hold',steps=16):
    results=[];mid=[]
    with torch.no_grad():
        for s in range(0,len(base),128):
            b=torch.from_numpy(base[s:s+128])
            if two:
                a,r=pair(net,decoder,b,start,policy,steps,True);mid.append(a)
            else:r=day(net,decoder,b,start,steps=steps,diagnostics=True)
            results.append(r)
    keys=['expression','z','intended','z0','logw']
    out={k:np.concatenate([r[k].numpy() for r in results]) for k in keys}
    diag={k:sum(r['diagnostics'][k] for r in results) for k in results[0]['diagnostics']}
    out['diagnostics']={k:diag[k]/max(diag['active_count'],1) for k in ['daily_bound_count','total_bound_count']}
    if two:
        out['mid_logw']=np.concatenate([r['logw'].numpy() for r in mid])
        out['mid_z']=np.concatenate([r['z'].numpy() for r in mid])
    return out

def probabilities(logw):
    a=np.asarray(logw,dtype=float);w=np.exp(a-a.max());return w/w.sum()

def internal(result,base,src):
    w=probabilities(result['logw']);_,inv=np.unique(src,return_inverse=True)
    uw=np.bincount(inv,weights=w);wanted=result['intended']-result['z0']
    actual=result['z']-result['z0']
    return dict(ess_rows=float(1/(w@w)),ess_unique=float(1/(uw@uw)),
        max_relative_weight=float(len(w)*w.max()),top_one_percent_mass=float(np.sort(w)[-max(1,len(w)//100):].sum()),
        second_or_only_day_decoder_relative_residual=float(np.linalg.norm(actual-wanted)/max(np.linalg.norm(wanted),1e-12)),
        edit_rms=float(np.sqrt(sum(np.square(result['expression'][s:s+128].astype(float)-base[s:s+128]).sum() for s in range(0,len(base),128))/base.size)),
        **result['diagnostics'])
