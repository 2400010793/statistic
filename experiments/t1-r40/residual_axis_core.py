"""Empirical-Bayes historical residual axis and arbitrary-rank bounded decoder."""
import numpy as np
from scipy.optimize import minimize
LOG_GAIN_BOUND=np.log(2.);REGULARIZATION=.1

def mean_and_variance(x):
 x=x.astype(np.float64);mu=np.asarray(x.mean(0)).ravel();var=np.maximum(np.asarray(x.multiply(x).mean(0)).ravel()-mu*mu,0)
 return mu,var*x.shape[0]/max(x.shape[0]-1,1)

def estimate_axis(early,late,V):
 ma,va=mean_and_variance(early);mb,vb=mean_and_variance(late);d=mb-ma;se2=va/early.shape[0]+vb/late.shape[0]
 tau2=max(float(np.mean(d*d-se2)),0.);shrink=np.divide(tau2,tau2+se2,out=np.zeros_like(se2),where=(tau2+se2)>0);stable=shrink*d
 # Stored PCA vectors are only approximately orthonormal after float32 fitting.
 residual=stable-V.T@np.linalg.solve(V@V.T,V@stable);length=float(np.linalg.norm(residual));axis=residual/length if length>1e-10 else np.zeros_like(residual)
 return {'history':d,'sampling_variance':se2,'prior_variance':tau2,'shrinkage':shrink,'shrunken_history':stable,'residual':residual,'axis':axis,'length':length}

def evaluate(theta,W,old,target):
 h=W.T@theta;t=np.tanh(h/LOG_GAIN_BOUND);u=LOG_GAIN_BOUND*t;c=np.expm1(old)*np.exp(u);weights=c/c.sum();normalized=10000*weights;x=np.log1p(normalized);residual=W@x-target;dx=x-old
 objective=.5*(residual@residual+REGULARIZATION*(dx@dx));grad_x=W.T@residual+REGULARIZATION*dx;weighted=grad_x*normalized/(1+normalized);grad_u=weighted-weights*weighted.sum();grad=W@(grad_u*(1-t*t))
 return float(objective),grad,x

def decode(anchor,target,basis):
 positive=anchor>0;W=np.ascontiguousarray(basis[:,positive]);x0=anchor[positive].astype(float);initial=np.linalg.norm(W@x0-target)
 if initial<1e-10:return anchor.copy(),True,False,0
 def fun(theta):
  f,g,_=evaluate(theta,W,x0,target);return f,g
 res=minimize(fun,np.zeros(basis.shape[0]),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-10,'gtol':1e-6,'maxls':30});_,_,candidate=evaluate(res.x,W,x0,target);candidate=candidate.astype(np.float32);residual=W@candidate-target;delta=candidate.astype(float)-x0;loss=.5*(residual@residual+.1*(delta@delta))
 accepted=bool(np.isfinite(candidate).all() and candidate.min()>0 and loss<.5*initial**2 and np.linalg.norm(residual)<initial);out=anchor.copy()
 if accepted:out[positive]=candidate
 return out,bool(res.success),accepted,int(res.nit)

def extend_target(source,latent,V,fit,horizon=.25):
 if fit['length']<=1e-10:return V,latent
 return np.vstack([V,fit['axis']]),np.column_stack([latent,source@fit['axis']+horizon*fit['length']])
