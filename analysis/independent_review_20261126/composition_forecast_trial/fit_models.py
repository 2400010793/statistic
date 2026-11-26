"""Freeze early-only proportion or continuous-state models before target evaluation."""
import argparse
import time
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from core import OUT,PREV,ROOT,prior,save

def main(phase):
    start=time.time()
    paths=[OUT/'protocol.txt',OUT/'fit_models.py',OUT/'core.py',PREV/'frozen_controls.npz',PREV/'run_review.py']
    paths += [ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]
    hashes={str(p.relative_to(ROOT)):prior.digest(p) for p in paths}
    if phase=='trend':
        panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
        pca,km=prior.fit_space(panel)
        z=[]; groups=[]
        for stage in ['E8.5','E9.5']:
            x=prior.read(ROOT/f'data/raw/T1/{stage}_RNA.h5ad',panel)
            z.append(prior.project(x,pca));groups.append(km.predict(z[-1]));del x
        count=np.stack([np.bincount(g,minlength=12) for g in groups])
        freq=count/count.sum(1,keepdims=True)
        smooth=(count+.5)/(count.sum(1,keepdims=True)+6)
        logratio=np.log(smooth[1]/smooth[0])
        q=freq[1]*np.exp(np.clip(.5*logratio,-np.log(2),np.log(2)));q/=q.sum()
        old=np.load(PREV/'frozen_controls.npz')
        np.testing.assert_array_equal(groups[1][old['sources']],old['source_groups'])
        np.savez_compressed(OUT/'early_space.npz',z8=z[0],z9=z[1],g8=groups[0],g9=groups[1],counts=count,frequencies=freq)
        joblib.dump(dict(pca=pca,clusters=km),OUT/'early_space.joblib',compress=3)
        np.savez_compressed(OUT/'trend_frozen.npz',carry=freq[1],trend=q,logratio=logratio)
        extra=dict(stage_counts=count.tolist(),stage_frequencies=freq.tolist(),trend=q.tolist(),
            early_space_sha256=prior.digest(OUT/'early_space.npz'),estimator_sha256=prior.digest(OUT/'early_space.joblib'))
    else:
        decision=json.loads((OUT/'trend_results.json').read_text())['gate']
        assert not decision['passed'],'Fallback is only authorized by the predeclared failed-trend branch.'
        early=np.load(OUT/'early_space.npz'); meta=json.loads((OUT/'trend_fit.json').read_text())
        assert prior.digest(OUT/'early_space.npz')==meta['early_space_sha256']
        x=np.vstack([early['z8'],early['z9']]); n8=len(early['z8']); n9=len(early['z9'])
        y=np.r_[np.zeros(n8,dtype=int),np.ones(n9,dtype=int)]
        oof=np.empty(len(y)); folds=np.empty(len(y),int); models=[]
        cv=StratifiedKFold(5,shuffle=True,random_state=2026112640)
        for fold,(train,test) in enumerate(cv.split(x,y)):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=1000,solver='lbfgs'))
            model.fit(x[train],y[train]);oof[test]=model.decision_function(x[test]);folds[test]=fold;models.append(model)
            print('Density fold',fold,'iterations',model[-1].n_iter_.tolist(),flush=True)
        ratio=np.exp(.5*np.clip(oof[n8:],-np.log(4),np.log(4)))
        q=np.bincount(early['g9'],weights=ratio,minlength=12);q/=q.sum()
        np.savez_compressed(OUT/'density_frozen.npz',density_shares=q,late_ratio=ratio,oof_logit=oof,folds=folds)
        joblib.dump(models,OUT/'density_fold_models.joblib',compress=3)
        prob=1/(1+np.exp(-oof))
        extra=dict(oof_auc=float(roc_auc_score(y,prob)),oof_log_loss=float(log_loss(y,prob)),density_shares=q.tolist(),
            ratio_quantiles=np.quantile(ratio,[0,.1,.5,.9,1]).tolist(),
            density_models_sha256=prior.digest(OUT/'density_fold_models.joblib'),
            early_space_sha256=prior.digest(OUT/'early_space.npz'),fallback_reason='Preregistered trend gate failed')
    unchanged={str(p.relative_to(ROOT)):prior.digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths}
    assert all(unchanged.values())
    save(f'{phase}_fit.json',dict(phase=phase,scope='E8.5/E9.5 only; no E10.5 loaded in fitting process',
        **extra,parameters_sha256=prior.digest(OUT/f'{phase}_frozen.npz'),inputs_sha256=hashes,inputs_unchanged=unchanged,elapsed_seconds=time.time()-start))
    print('FROZEN',phase,'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['trend','density']);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.phase)
