"""Source-only sparse decoder; never loads a future stage."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import sparse

OUT=Path(__file__).resolve().parent
PREV=OUT.parent
ROOT=PREV.parents[1]
SEED=2026112600

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

class Decoder:
    def __init__(self,x,labels,recent):
        self.x=sparse.csr_matrix(x).copy()
        self.x.eliminate_zeros(); self.x.sort_indices()
        self.labels=labels
        self.a=self.x.data.astype(float)
        self.row=np.repeat(np.arange(x.shape[0]),np.diff(self.x.indptr))
        self.gene=self.x.indices
        self.den=recent[labels[self.row],self.gene]
        assert (self.den>0).all()
        self.n,self.g=x.shape

    def ratio(self,shift):
        return shift[self.gene]/self.den

    def calculate(self,ratio,theta,jacobian=False,render=False):
        alpha,beta=theta
        raw=1+alpha*ratio+beta
        gain=np.clip(raw,.8,1.2)
        v=np.expm1(self.a*gain)
        sums=np.bincount(self.row,weights=v,minlength=self.n)
        y=np.log1p(v*(10000/sums)[self.row])
        diff=y-self.a
        rms=np.sqrt(diff@diff/(self.n*self.g))
        mass=diff.sum()/self.n
        moments=np.array([rms,mass])
        if render:
            result=sparse.csr_matrix((y.astype(np.float32),self.x.indices,self.x.indptr),shape=self.x.shape).toarray()
            assert np.isfinite(result).all() and result.min()>=0
            assert np.count_nonzero(result)==len(self.a)
            np.testing.assert_allclose(np.expm1(result.astype(float)).sum(1),10000,rtol=1e-5)
            return result,moments
        if not jacobian:
            return moments
        active=(raw>.8)&(raw<1.2)
        derivatives=[]
        denom=sums[self.row]/10000+v
        for dr in (ratio,1.):
            du=self.a*active*dr
            dv=(v+1)*du
            ds=np.bincount(self.row,weights=dv,minlength=self.n)
            dy=(dv-v*(ds/sums)[self.row])/denom
            derivatives.append([float(diff@dy/(self.n*self.g*rms)),float(dy.sum()/self.n)])
        return moments,np.array(derivatives).T

    def moment_audit(self,moments,reference):
        return dict(edit_rms=float(moments[0]),mean_log_total_change=float(moments[1]),
            relative_errors=(np.abs(moments-reference)/np.abs(reference)).tolist())
