"""Freeze one new candidate after pooled and same-stage checks; never submit."""
from pathlib import Path
import argparse, hashlib,json
import numpy as np
import scipy.sparse as sp
from scipy.linalg import null_space
import anndata as ad
import joblib
from residual_axis_core import estimate_axis
from protected_residual_decoder import protected_decode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('research_root', type=Path, help='Local research workspace with baseline artifacts')
ROOT = parser.parse_args().research_root.resolve()
R = ROOT / 'analysis/results'
OUT = R / 'population_direction_round40'
NAME='neural_population_direction';BASE=R/'expanded_protection_round36'
read=lambda p:json.loads(p.read_text())
assert read(OUT/'review_summary.json')['comparisons']['population_direction']['passed']
assert read(OUT/'null_results.json')['passed']
PLAN={'architecture':'Start scored expanded50 candidate. Release one training-derived pooled historical direction inside Q50, freeze49 orthogonal coordinates. Move every cell toward a common projected increment correcting half the still-missing pooled shift. Same bounded positive-support decoder.',
      'horizon':.5,'missing_shift_fraction':.5,'seed_search':False,
      'fixed':['Source2000 cells and duplicates','Original neural parameters and donors','All other49 protected coordinates','Positive support and count-normalization10000'],
      'limitation':'Observed pooled historical change mixes type composition and within-type expression. Global motion is not lineage tracing or demonstrated future continuation. Source labels are only provenance.',
      'evaluation':'Five reused historical cell folds; no future truth. Candidate created locally only; no API call.'}
(OUT/'candidate_plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
best=ad.read_h5ad(BASE/'neural_expanded_protected_axis_E10.5.h5ad')
par=np.load(BASE/'forecast_parameters.npz');src=par['source_indices']
data=[ad.read_h5ad(ROOT/f'data/raw/T1/{s}_RNA.h5ad') for s in ['E8.5','E9.5']]
X=[sp.csr_matrix(a.X) for a in data];labels=data[1].obs['celltype'].astype(str).to_numpy()[src]
Q=joblib.load(BASE/'candidate_protection.joblib')['basis']
fit=estimate_axis(X[0],X[1],np.empty((0,Q.shape[1])))
coeff=Q@fit['shrunken_history'];length=float(np.linalg.norm(coeff));assert length>1e-10
axis=(coeff/length)@Q;guard=null_space((coeff/length)[None,:]).T@Q
assert guard.shape==(49,best.n_vars) and np.max(abs(guard@axis))<1e-10
source_mean=np.asarray(X[1][src].mean(0,dtype=np.float64)).ravel()
current=float(best.X.mean(0,dtype=float)@axis);desired=float(source_mean@axis+.5*length)
shift=.5*(desired-current)
unique,first,inverse=np.unique(src,return_index=True,return_inverse=True)
assert np.array_equal(best.X,best.X[first][inverse])
pred=best.X.copy();audits=[]
for n,(row,source_id) in enumerate(zip(first,unique)):
    target=float(best.X[row].astype(float)@axis+shift)
    value,diag=protected_decode(best.X[row],guard,axis,target)
    pred[inverse==n]=value;audits.append({'source_index':int(source_id),**diag})
    if n%400==0:print(json.dumps({'completed_unique':n,'total_unique':len(unique)}),flush=True)
error=float(np.max(abs((pred.astype(float)-best.X)@guard.T)))
assert error<1e-5 and np.array_equal(pred>0,best.X>0)
norms=np.expm1(pred.astype(float)).sum(1)
assert np.isfinite(pred).all() and np.allclose(norms,10000,rtol=1e-6)
assert sum(d['accepted'] for d in audits)>=.95*len(audits)
panel=read(ROOT/'panels/index.json')['T1:val']
assert best.var_names.tolist()==(ROOT/'panels'/panel['genes_file']).read_text().splitlines()
assert panel['min_cells']<=len(pred)<=panel['max_cells'] and len(best.obs.columns)==0
dest=OUT/f'{NAME}_E10.5.h5ad';ad.AnnData(X=pred,obs=best.obs.copy(),var=best.var.copy()).write_h5ad(dest,compression='gzip',compression_opts=1)
check=ad.read_h5ad(dest);assert np.array_equal(check.X,pred)
fit.update(axis=axis,length=length,guard=guard,horizon=.5,missing_shift_fraction=.5,target_shift=shift)
joblib.dump(fit,OUT/'candidate_axis.joblib',compress=3)
np.savez_compressed(OUT/'forecast_parameters.npz',source_indices=src,baseline_axis=best.X.astype(float)@axis,candidate_axis=pred.astype(float)@axis)
by_type=[{'celltype':c,'n_output':int((labels==c).sum()),'mean_cell_change_norm':float(np.linalg.norm(pred[labels==c].astype(float)-best.X[labels==c],axis=1).mean())} for c in sorted(set(labels))]
diagnostics={'shape':list(pred.shape),'unique_optimized':len(audits),'accepted':sum(d['accepted'] for d in audits),'converged':sum(d['converged'] for d in audits),
             'protected_dimensions':49,'max_preserved_coordinate_error':error,'historical_axis_length':length,'baseline_pooled_axis':current,'desired_pooled_axis':desired,
             'applied_axis_target_increment':shift,'realized_pooled_axis':float(pred.mean(0,dtype=float)@axis),'changed_rows':int(np.any(pred!=best.X,axis=1).sum()),
             'mean_cell_change_norm':float(np.linalg.norm(pred.astype(float)-best.X,axis=1).mean()),'row_sum_range':[float(norms.min()),float(norms.max())],
             'historical_mean_energy_in_Q50':float(np.linalg.norm(Q@fit['history'])**2/np.linalg.norm(fit['history'])**2)}
inputs=[BASE/'neural_expanded_protected_axis_E10.5.h5ad',BASE/'candidate_protection.joblib',BASE/'forecast_parameters.npz',Path(__file__).with_name('protected_residual_decoder.py'),Path(__file__).with_name('residual_axis_core.py')]+[ROOT/f'data/raw/T1/{s}_RNA.h5ad' for s in ['E8.5','E9.5']]
manifest={**PLAN,'models':{NAME:{'architecture':PLAN['architecture']}},'files':{NAME:{'path':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}},
          'diagnostics':diagnostics,'by_type':by_type,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'input_sha256':{(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'official_evaluation':'not_submitted; no API calls'}
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/'optimization_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
print(json.dumps(diagnostics),flush=True)
