"""Stage-residual population dynamics with a differentiable CP10K decoder."""
import sys
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
REVIEW=OUT.parent
JOINT=REVIEW/'joint_dynamics_trial'
DIAG=REVIEW/'speed_direction_diagnostic'
ROOT=REVIEW.parents[1]
sys.path.append(str(JOINT/'deps'))
sys.path.insert(0,str(REVIEW))
import run_review as prior
import torch
from torch import nn
SEEDS=[2026112601,2026112602]
ARMS=['autonomous_latent','temporal_noess','temporal_latent','temporal_decoded']
STEPS=16

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class StageDynamics(nn.Module):
    """Time 0=E8.5, 1=E9.5, 2=E10.5. No hidden-stage time rule is selected."""
    def __init__(self,temporal=True):
        super().__init__();self.temporal=temporal
        self.trunk=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.base=nn.Linear(128,33);self.residual=nn.Linear(128,33)
        for layer in [self.base,self.residual]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        if not temporal:
            for p in self.residual.parameters():p.requires_grad_(False)

    def field(self,z,time):
        h=self.trunk(z);b=self.base(h)
        r=self.residual(h) if self.temporal else torch.zeros_like(b)
        coeff=max(-1.,min(1.,float(time)-1.))
        out=b+coeff*r
        return out[:,:32],out[:,32],r

    def forward(self,z,start,steps=STEPS,duration=1.):
        if start<0 or start+duration>2:raise ValueError('This v1 is validated only on observed intervals [0,1] and [1,2].')
        lw=torch.zeros(len(z));energy=torch.zeros(len(z));growth=torch.zeros(len(z));res=torch.zeros(len(z))
        dt=duration/steps
        for j in range(steps):
            v,g,r=self.field(z,start+j*dt);z=z+dt*v;lw=lw+dt*g
            energy=energy+v.square().mean(1)/steps;growth=growth+g.square()/steps
            res=res+(r[:,:32].square().mean(1)+r[:,32].square())/steps
        return z,lw,energy,growth,res

class Decoder(nn.Module):
    def __init__(self,pca,scale):
        super().__init__();self.register_buffer('axes',torch.from_numpy(pca.components_.astype(np.float32)))
        self.register_buffer('center',torch.from_numpy(pca.mean_.astype(np.float32)));self.scale=float(scale)

    def forward(self,base,normalized_delta):
        shift=(normalized_delta*self.scale)@self.axes
        edited=base*torch.clamp(1+shift/(base+.25),.8,1.2)
        linear=torch.expm1(edited)
        return torch.log1p(linear*(10000/linear.sum(1,keepdim=True)))

    def project(self,expression):return ((expression-self.center)@self.axes.T)/self.scale

    def proxy_mean(self,base,delta,w):return w@base+((w@delta)*self.scale)@self.axes

def weights_and_penalty(logweight,base_probability):
    w=torch.softmax(logweight+torch.log(base_probability),dim=0)
    concentration=(w.square()/base_probability).sum()
    return w,.02*torch.relu(concentration-2).square(),1/concentration

def distribution(z,target,w,bandwidth,moments=None):
    dxx=torch.cdist(z,z).square();dxy=torch.cdist(z,target).square();dyy=torch.cdist(target,target).square()
    mmd=z.new_tensor(0.)
    for mult in [.25,1.,4.]:
        mmd=mmd+(w@torch.exp(-dxx/(2*bandwidth*mult))@w+torch.exp(-dyy/(2*bandwidth*mult)).mean()
            -2*w@torch.exp(-dxy/(2*bandwidth*mult)).mean(1))/3
    mx=w@z;cx=z-mx;covx=(cx.T*w)@cx
    if moments is None:
        my=target.mean(0);cy=target-my;covy=cy.T@cy/len(target)
    else:my,covy=moments
    mean=(mx-my).square().mean();cov=(covx-covy).square().mean()
    return mmd+.05*mean+.01*cov

def stratified_batch(rng,rows,groups,batch=192):
    pool=[rows[groups[rows]==k] for k in range(12)]
    sizes=np.array([len(p) for p in pool]);assert (sizes>0).all()
    p=sizes/sizes.sum();expected=(batch-48)*p;counts=4+np.floor(expected).astype(int)
    left=batch-counts.sum();counts[np.argsort(-(expected-np.floor(expected)))[:left]]+=1
    ix=np.concatenate([rng.choice(pool[k],counts[k],replace=len(pool[k])<counts[k]) for k in range(12)])
    q=np.concatenate([np.full(counts[k],p[k]/counts[k]) for k in range(12)]).astype(np.float32)
    assert len(ix)==batch and abs(q.sum()-1)<1e-6
    return ix,q
