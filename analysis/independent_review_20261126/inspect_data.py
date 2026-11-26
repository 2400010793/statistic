"""Independent inspection; does not import historical analysis code/results."""
import json
from pathlib import Path
import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
def main():
    result, early = {}, {}
    panel = (ROOT / 'panels/T1__val.genes.txt').read_text().splitlines()
    for stage, path in [('E8.5','data/raw/T1/E8.5_RNA.h5ad'), ('E9.5','data/raw/T1/E9.5_RNA.h5ad'), ('E10.5','data/p3/T1/E10.5_RNA.h5ad')]:
        a = ad.read_h5ad(ROOT / path)
        x = a.X.tocsr()
        rng = np.random.default_rng(261126)
        sample = x[rng.choice(x.shape[0], min(1000,x.shape[0]), replace=False)].copy()
        vals = sample.data.copy()
        sample.data = np.expm1(sample.data)
        labels = a.obs.celltype.astype(str).to_numpy()
        result[stage] = dict(shape=list(x.shape), storage=type(a.X).__name__, density=float(x.nnz/np.prod(x.shape)),
            celltypes=a.obs.celltype.value_counts().to_dict(), obs_columns=list(a.obs.columns),
            nonzero_quantiles=np.quantile(vals,[0,.25,.5,.75,.99,1]).tolist(),
            nonzero_integer_fraction=float(np.mean(np.abs(vals-np.round(vals))<1e-5)),
            log_space_row_sum_quantiles=np.quantile(np.asarray(x.sum(1)).ravel(),[0,.5,1]).tolist(),
            expm1_row_sum_quantiles=np.quantile(np.asarray(sample.sum(1)).ravel(),[0,.5,1]).tolist(),
            panel_order_exact=list(a.var_names)==panel)
        if stage != 'E10.5':
            means = {c: np.asarray(x[labels==c].mean(0)).ravel().astype(float) for c in np.unique(labels)}
            early[stage] = (np.asarray(x.mean(0)).ravel().astype(float),means)
        del a,x,sample
    m8,c8=early['E8.5']; m9,c9=early['E9.5']
    shared=sorted(set(c8)&set(c9))
    top=np.argsort(np.abs(m9-m8))[-20:][::-1]
    result['early_change'] = dict(global_mean_rmse=float(np.sqrt(np.mean((m9-m8)**2))),
        top_absolute_gene_changes=[dict(gene=panel[i],delta=float(m9[i]-m8[i])) for i in top],
        shared_celltype_mean_rmse={c:float(np.sqrt(np.mean((c9[c]-c8[c])**2))) for c in shared},
        only_E85=sorted(set(c8)-set(c9)),only_E95=sorted(set(c9)-set(c8)))
    (OUT/'data_inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
