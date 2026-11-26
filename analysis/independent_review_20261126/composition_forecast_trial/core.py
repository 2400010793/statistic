from pathlib import Path
import json
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
PREV=OUT.parent
ROOT=PREV.parents[1]
sys.path.insert(0,str(PREV))
import run_review as prior

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def weights_for(groups,q,ratio=None):
    r=np.ones(len(groups)) if ratio is None else ratio
    w=q[groups]*r/np.bincount(groups,weights=r,minlength=12)[groups]
    np.testing.assert_allclose(w.sum(),1,atol=1e-12)
    np.testing.assert_allclose(np.bincount(groups,weights=w,minlength=12),q,atol=1e-12)
    return w

def allocate(mass,total,tie):
    exact=total*mass/mass.sum(); n=np.floor(exact).astype(int)
    rest=total-n.sum()
    n[np.lexsort((tie,-(exact-n)))[:rest]]+=1
    assert n.sum()==total
    return n

def materialize(groups,q,w,batch):
    rng=np.random.default_rng(2026112640+batch)
    tie=rng.random(len(groups)); quota=allocate(q,2000,np.arange(12))
    counts=np.zeros(len(groups),int)
    for k in range(12):
        ix=np.flatnonzero(groups==k)
        counts[ix]=allocate(w[ix],int(quota[k]),tie[ix])
    order=np.repeat(np.arange(len(groups)),counts)
    assert len(order)==2000
    np.testing.assert_array_equal(np.bincount(groups[order],minlength=12),quota)
    return order

def gates(results,candidate,phase):
    out={}
    for mode in ['weighted','materialized']:
        rec=[r[mode] for r in results]
        stats={}
        for metric in ['covariance_rel','mmd2']:
            delta=[r[candidate][metric]-r['carry'][metric] for r in rec]
            stats[metric]=dict(deltas=delta,wins=sum(v<0 for v in delta),mean_delta=float(np.mean(delta)),passed=bool(sum(v<0 for v in delta)>=4 and np.mean(delta)<0))
        first=rec[0]; a=first[candidate]; b=first['carry']
        checks=dict(pooled_rmse_guard=a['pooled_rmse']<=1.01*b['pooled_rmse'],
            within_rmse_guard=a['fixed_within_rmse']<=1.01*b['fixed_within_rmse'],
            direction_guard=a['cosine']>=b['cosine']-.01,
            composition_guard=a['composition_tv']<b['composition_tv'] if phase=='trend' else a['composition_tv']<=b['composition_tv']+1e-12)
        out[mode]=dict(distribution=stats,guards=checks,passed=all(v['passed'] for v in stats.values()) and all(checks.values()))
    out['passed']=all(out[m]['passed'] for m in ['weighted','materialized'])
    return out
