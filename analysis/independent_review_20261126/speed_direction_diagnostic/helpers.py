"""Nested calibration models; original artifacts and code are read-only."""
import sys
import copy
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
REVIEW=OUT.parent
JOINT=REVIEW/'joint_dynamics_trial'
ROOT=REVIEW.parents[1]
sys.path.insert(0,str(JOINT))
import core as old
torch,nn,prior=old.torch,old.nn,old.prior
SEEDS=old.SEEDS
MODES=['mass','global_mass','flex_mass','full']
STEPS=16

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def load_original(seed):
    checkpoint=torch.load(JOINT/f'joint_{seed}.pt',weights_only=True)
    net=old.Dynamics(True);net.load_state_dict(checkpoint['state']);net.eval()
    return net

class Calibrated(torch.nn.Module):
    def __init__(self,original,mode):
        super().__init__();self.mode=mode
        self.trunk=copy.deepcopy(original.trunk);self.head=copy.deepcopy(original.head)
        self.original=copy.deepcopy(original)
        for p in self.original.parameters():p.requires_grad_(False)
        self.clock=nn.Parameter(torch.tensor(0.))
        if mode=='flex_mass':
            with torch.no_grad():self.head.weight[0].zero_();self.head.bias[0].zero_()

    def field(self,z):
        out=self.head(self.trunk(z));g=np.log(4)*torch.tanh(out[:,32])
        if self.mode=='full':return out[:,:32],g,torch.ones(len(z))
        orig=self.original.head(self.original.trunk(z))[:,:32]
        if self.mode=='mass':scale=torch.ones(len(z))
        elif self.mode=='global_mass':scale=(8*torch.sigmoid(self.clock-np.log(7))).expand(len(z))
        else:scale=8*torch.sigmoid(out[:,0]-np.log(7))
        return orig*scale[:,None],g,scale

    def forward(self,z,steps=STEPS):
        logw=torch.zeros(len(z));energy=z.new_tensor(0.);genergy=z.new_tensor(0.)
        for _ in range(steps):
            v,g,_=self.field(z);z=z+v/steps;logw=logw+g/steps
            energy=energy+v.square().mean()/steps;genergy=genergy+g.square().mean()/steps
        return z,torch.softmax(logw,dim=0),logw,energy,genergy

def metrics_latent(z,w,target,bandwidth,subset):
    """Objective uses all target moments and a frozen target kernel subsample."""
    from scipy.spatial.distance import cdist
    z=z.astype(float);w=w.astype(float);target=target.astype(float)
    y=target[subset];mx=w@z;my=target.mean(0)
    cx=z-mx;cy=target-my;covx=(cx.T*w)@cx;covy=cy.T@cy/len(target)
    mean=float(np.mean((mx-my)**2));cov=float(np.mean((covx-covy)**2))
    dxx=cdist(z,z,'sqeuclidean');dxy=cdist(z,y,'sqeuclidean');dyy=cdist(y,y,'sqeuclidean')
    v=[];u=[]
    for mult in [.25,1.,4.]:
        kxx=np.exp(-dxx/(2*bandwidth*mult));kxy=np.exp(-dxy/(2*bandwidth*mult));kyy=np.exp(-dyy/(2*bandwidth*mult))
        xx=float(w@kxx@w);xy=float(w@kxy.mean(1));yy=float(kyy.mean())
        v.append(xx+yy-2*xy)
        u.append((xx-w@w)/(1-w@w)+(kyy.sum()-len(y))/(len(y)*(len(y)-1))-2*xy)
    mmd=float(np.mean(v));mmd_u=float(np.mean(u))
    return dict(objective=mmd+.05*mean+.01*cov,mmd_v=mmd,mmd_u=mmd_u,mean_mse=mean,cov_element_mse=cov,
        covariance_rel=float(np.linalg.norm(covx-covy)/np.linalg.norm(covy)),effective_rows=float(1/(w@w)),
        weight_range=[float(len(w)*w.min()),float(len(w)*w.max())])
