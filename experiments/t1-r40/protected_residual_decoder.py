"""Gene-axis correction constrained to retain the base prediction's PCA state."""
import numpy as np
from scipy.optimize import minimize

LOG_BOUND=np.log(2.)
REG=.1
PCA_TOL=1e-5

def evaluate(theta,W,old,axis_target):
    t=np.tanh((W.T@theta)/LOG_BOUND);u=LOG_BOUND*t;du=1-t*t
    c=np.expm1(old)*np.exp(u);weights=c/c.sum();c=10000*weights;x=np.log1p(c)
    slope=c/(1+c);J=slope[:,None]*(du[:,None]*W.T-((weights*du)@W.T)[None,:])
    coord=W@x;jac=W@J;err=coord[-1]-axis_target;delta=x-old
    loss=.5*(err*err+REG*(delta@delta));gradient=err*jac[-1]+REG*(delta@J)
    return float(loss),gradient,x,coord,jac

def protected_decode(base,V,axis,axis_target):
    """Preserve V@base, support and total count; return base if guards fail."""
    base=np.asarray(base,dtype=np.float32);positive=base>0
    if np.linalg.norm(axis)<1e-10:return base.copy(),{'converged':True,'accepted':False,'reason':'zero_axis','iterations':0,'pca_error':0.}
    W=np.ascontiguousarray(np.vstack([V,axis])[:,positive]);old=base[positive].astype(float);z=W[:-1]@old
    initial=.5*(W[-1]@old-axis_target)**2
    if initial<1e-14:return base.copy(),{'converged':True,'accepted':False,'reason':'identity','iterations':0,'pca_error':0.}
    cache={}
    def ev(th):
        if 'th' not in cache or not np.array_equal(th,cache['th']):cache['th']=th.copy();cache['v']=evaluate(th,W,old,axis_target)
        return cache['v']
    def fun(th):v=ev(th);return v[0],v[1]
    def constraint(th):return ev(th)[3][:-1]-z
    def jac(th):return ev(th)[4][:-1]
    result=minimize(fun,np.zeros(len(W)),jac=True,method='SLSQP',constraints=[{'type':'eq','fun':constraint,'jac':jac}],options={'maxiter':80,'ftol':1e-10})
    _,_,x,_,_=ev(result.x);candidate=x.astype(np.float32);err=float(np.max(abs(W[:-1]@(candidate.astype(float)-old))))
    loss=.5*((W[-1]@candidate-axis_target)**2+REG*np.sum((candidate.astype(float)-old)**2))
    normal=float(np.expm1(candidate.astype(float)).sum())
    accepted=bool(result.success and np.isfinite(candidate).all() and candidate.min()>0 and err<=PCA_TOL and abs(normal/10000-1)<1e-6 and loss<initial)
    out=base.copy()
    if accepted:out[positive]=candidate
    return out,{'converged':bool(result.success),'accepted':accepted,'iterations':int(result.nit),'pca_error':err if accepted else 0.,'attempted_pca_error':err,'objective_before':float(initial),'objective_after':float(loss),'message':str(result.message)}
