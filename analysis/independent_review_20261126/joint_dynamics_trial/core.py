"""Local neural dynamics prototype; no target-stage data access here."""
import sys
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
PREV=OUT.parent
ROOT=PREV.parents[1]
sys.path.append(str(OUT/'deps'))  # Existing environment dependencies take precedence.
sys.path.insert(0,str(PREV))
import run_review as prior
import torch
from torch import nn
SEEDS=[2026112601,2026112602]

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class Dynamics(nn.Module):
    def __init__(self,growth):
        super().__init__()
        self.growth=growth
        self.trunk=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.head=nn.Linear(128,33)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,z,duration=1.,steps=4):
        logw=torch.zeros(len(z),device=z.device)
        energy=z.new_tensor(0.); genergy=z.new_tensor(0.)
        for _ in range(steps):
            out=self.head(self.trunk(z))
            v=out[:,:32]
            g=np.log(4)*torch.tanh(out[:,32]) if self.growth else out[:,32]*0
            z=z+(duration/steps)*v
            logw=logw+(duration/steps)*g
            energy=energy+v.square().mean()/steps
            genergy=genergy+g.square().mean()/steps
        return z,torch.softmax(logw,dim=0),logw,energy,genergy

def distribution_loss(x,y,w,bandwidth):
    dxx=torch.cdist(x,x).square(); dyy=torch.cdist(y,y).square();dxy=torch.cdist(x,y).square()
    mmd=x.new_tensor(0.)
    for mult in [.25,1.,4.]:
        kxx=torch.exp(-dxx/(2*bandwidth*mult))
        kyy=torch.exp(-dyy/(2*bandwidth*mult));kxy=torch.exp(-dxy/(2*bandwidth*mult))
        mmd=mmd+(w@kxx@w+kyy.mean()-2*(w@kxy.mean(1)))/3
    mx=w@x;my=y.mean(0)
    cx=x-mx;cy=y-my
    covx=(cx.T*w)@cx; covy=cy.T@cy/len(y)
    mean=(mx-my).square().mean();cov=(covx-covy).square().mean()
    return mmd+.05*mean+.01*cov,dict(mmd=float(mmd.detach()),mean=float(mean.detach()),cov=float(cov.detach()))

def normalize(values):
    counts=np.expm1(values.astype(np.float64))
    return np.log1p(counts*(10000/counts.sum(1,keepdims=True))).astype(np.float32)

def decode(base,delta,open_support=False,alpha=1.):
    out=np.empty_like(base)
    for start in range(0,len(base),128):
        stop=min(start+128,len(base));b=base[start:stop].astype(float);d=delta[start:stop]
        edited=np.maximum(b+alpha*d,0) if open_support else b*np.clip(1+d/(b+.25),.8,1.2)
        out[start:stop]=normalize(edited)
    assert np.isfinite(out).all() and out.min()>=0
    if not open_support:np.testing.assert_array_equal(base>0,out>0)
    return out

def edit_rms(a,b):
    return float(np.sqrt(sum(np.square(a[s:s+128].astype(float)-b[s:s+128]).sum() for s in range(0,len(a),128))/a.size))

def allocation(weights,batch):
    # Uniform weights produce the exact identity. Random tie breaking is shared across arms.
    expected=weights*len(weights);counts=np.floor(expected).astype(int)
    left=len(weights)-counts.sum();tie=np.random.default_rng(2026112640+batch).random(len(weights))
    order=np.lexsort((tie,-(expected-counts)))
    counts[order[:left]]+=1
    result=np.repeat(np.arange(len(weights)),counts)
    assert len(result)==len(weights)
    return result
