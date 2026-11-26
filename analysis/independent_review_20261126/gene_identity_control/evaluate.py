"""Evaluate an already frozen source-only experiment on released E10.5."""
import sys
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from control_core import OUT,PREV,ROOT,Decoder,digest,save
sys.path.insert(0,str(PREV))
import run_review as prior

def actual_moments(x,copy):
    squares,total=0.,0.
    for start in range(0,len(x),128):
        d=x[start:start+128].astype(float)-copy[start:start+128]
        squares+=float(np.einsum('ij,ij->',d,d)); total+=float(d.sum())
    return np.array([np.sqrt(squares/x.size),total/len(x)])

def main():
    start=time.time()
    audit=json.loads((OUT/'fit_audit.json').read_text())
    assert digest(OUT/'frozen.npz')==audit['frozen_sha256']
    paths=[OUT/'frozen.npz',OUT/'fit_audit.json',OUT/'protocol.txt',OUT/'control_core.py',OUT/'evaluate.py',
           PREV/'results.json',ROOT/'data/p3/T1/E10.5_RNA.h5ad']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    frozen=np.load(OUT/'frozen.npz'); old=np.load(PREV/'frozen_controls.npz')
    sources=frozen['sources']; groups=frozen['source_groups']; recent=frozen['recent_means']; common=frozen['common']
    permutations=frozen['permutations']; theta=frozen['theta']
    panel=(ROOT/'panels/T1__val.genes.txt').read_text().splitlines()
    pca,km=prior.fit_space(panel)
    latest=prior.read(ROOT/'data/raw/T1/E9.5_RNA.h5ad',panel)
    z9=prior.project(latest,pca)
    np.testing.assert_array_equal(km.predict(z9)[sources],groups)
    # E10.5 is loaded for evaluation only after frozen-parameter integrity checks.
    evaluator=prior.Evaluator(panel,pca,km,latest,z9)
    archived=json.loads((PREV/'results.json').read_text())['results']
    all_results=[]; checks=[]
    for batch,ix in enumerate(sources):
        copy=latest[ix].toarray(); g=groups[batch]; copy_mean=copy.mean(0,dtype=float)
        decoder=Decoder(latest[ix],g,recent)
        correct,_=decoder.calculate(decoder.ratio(common),[1,0],render=True)
        reference=actual_moments(correct,copy)
        def evaluate(x):
            result=evaluator(x,g,copy_mean)
            moments=actual_moments(x,copy)
            result['edit_moments']=decoder.moment_audit(moments,reference)
            move=x.mean(0,dtype=float)-copy_mean
            goal=evaluator.weights@evaluator.target-copy_mean
            result['paired_edit_cosine']=float(move@goal/(np.linalg.norm(move)*np.linalg.norm(goal))) if np.linalg.norm(move)>0 else None
            return result
        rec=dict(copy=evaluate(copy),common_only=evaluate(correct))
        del correct
        full=prior.decode(copy,g,recent,old[f'b{batch}_modules_full'])
        rec['modules_full']=evaluate(full); del full
        if batch==0:
            r40=prior.read(prior.R40/'neural_population_direction_E10.5.h5ad',panel)
            rec['r40']=evaluate(r40); del r40
        for arm in ['copy','common_only','modules_full']+(['r40'] if batch==0 else []):
            differences={k:abs(rec[arm][k]-archived[batch][arm][k]) for k in
                         ['within_rmse','fixed_within_rmse','fixed_equal_rmse','pooled_rmse','cosine','mmd2','covariance_rel']}
            for key in differences:
                np.testing.assert_allclose(rec[arm][key],archived[batch][arm][key],atol=1e-8,rtol=1e-7)
            checks.append(dict(batch=batch,arm=arm,max_absolute_error=max(differences.values())))
        temp,_=decoder.calculate(np.zeros_like(decoder.a),[0,float(frozen['temperature_beta'])],render=True)
        rec['temperature_only']=evaluate(temp); del temp
        selected=range(32) if batch==0 else range(8)
        for j in selected:
            ratio=decoder.ratio(common[permutations[j]])
            pred,_=decoder.calculate(ratio,theta[j],render=True)
            stats=evaluate(pred); del pred
            stats['source_only_fit_valid']=bool(frozen['valid'][j])
            stats['realized_matching_valid']=max(stats['edit_moments']['relative_errors'])<=(.005 if batch==0 else .02)
            rec[f'matched_{j:02d}']=stats
            if batch==0:
                pred,_=decoder.calculate(ratio,[1,0],render=True)
                rec[f'raw_{j:02d}']=evaluate(pred); del pred
            print(batch,j,'fixed RMSE',stats['fixed_within_rmse'],'cosine',stats['cosine'],'match',stats['edit_moments']['relative_errors'],flush=True)
        all_results.append(rec)
        save('evaluation_partial.json',all_results)
        del decoder,copy
    unchanged={str(p.relative_to(ROOT)):digest(p)==hashes[str(p.relative_to(ROOT))] for p in paths}
    assert all(unchanged.values())
    save('results.json',dict(scope='Frozen source-only gene identity diagnostic; retrospective E9.5 to E10.5; no official scoring.',
        results=all_results,baseline_reproduction=checks,fit_audit_sha256=digest(OUT/'fit_audit.json'),
        target_counts=evaluator.counts.tolist(),bandwidth=evaluator.bandwidth,
        inputs_sha256=hashes,inputs_unchanged=unchanged,elapsed_seconds=time.time()-start))
    print('EVALUATION COMPLETE',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
