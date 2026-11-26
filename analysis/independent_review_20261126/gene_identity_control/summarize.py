"""Report every preregistered control; no selection by target performance."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from control_core import OUT,save

def distribution(values):
    v=np.asarray(values,dtype=float)
    return dict(min=float(v.min()),q10=float(np.quantile(v,.1)),median=float(np.median(v)),
                q90=float(np.quantile(v,.9)),max=float(v.max()))

def main():
    data=json.loads((OUT/'results.json').read_text()); fit=json.loads((OUT/'fit_audit.json').read_text())
    batches=data['results']; first=batches[0]
    keys=['fixed_within_rmse','fixed_equal_rmse','pooled_rmse','cosine','mmd2','covariance_rel',
          'within_rmse','within_equal_rmse','paired_edit_cosine','realized_pooled_edit_rms']
    with (OUT/'metrics.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerow(['batch','arm',*keys,'edit_rms','log_total_change','rms_relative_error','log_total_relative_error'])
        for b,rec in enumerate(batches):
            for name,row in rec.items():
                m=row['edit_moments']
                writer.writerow([b,name,*[row[k] for k in keys],m['edit_rms'],m['mean_log_total_change'],*m['relative_errors']])
    controls=[row for name,row in first.items() if name.startswith('matched_') and row['source_only_fit_valid'] and row['realized_matching_valid']]
    raw=[row for name,row in first.items() if name.startswith('raw_')]
    higher={'cosine','paired_edit_cosine'}
    objectives=[k for k in keys if k!='realized_pooled_edit_rms']
    compared={k:dict(correct=first['common_only'][k],matched=distribution([r[k] for r in controls]),
        raw=distribution([r[k] for r in raw]),correct_better_count=sum(first['common_only'][k]>r[k] if k in higher else first['common_only'][k]<r[k] for r in controls)) for k in keys}
    compared['realized_pooled_edit_rms']['correct_better_count']=None
    qualified=fit['binning']['coverage_passed'] and len(controls)>=24
    supported=qualified and all(compared[k]['correct_better_count']/len(controls)>=.9 for k in ['fixed_within_rmse','cosine'])
    contradicted=qualified and all(compared[k]['correct_better_count']/len(controls)<=.5 for k in ['fixed_within_rmse','cosine'])
    decision='supports_gene_identity_information' if supported else 'supports_nonspecific_sufficiency' if contradicted else 'inconclusive'
    retention={k:distribution([(first['copy'][k]**2-c[k]**2)/(first['copy'][k]**2-first['common_only'][k]**2) for c in controls])
               for k in ['fixed_within_rmse','fixed_equal_rmse','pooled_rmse']}
    sensitivity=[]
    for b,rec in enumerate(batches[1:],1):
        rows=[v for name,v in rec.items() if name.startswith('matched_')]
        valid=[v for v in rows if v['realized_matching_valid']]
        sensitivity.append(dict(batch=b,n=len(rows),matching_within_2percent=len(valid),
            max_moment_relative_error=np.max([v['edit_moments']['relative_errors'] for v in rows],axis=0).tolist(),
            common={k:rec['common_only'][k] for k in keys},
            matched={k:distribution([v[k] for v in rows]) for k in keys},
            common_better_counts={k:sum(rec['common_only'][k]>v[k] if k in higher else rec['common_only'][k]<v[k] for v in rows) for k in objectives}))
    report=dict(preregistered_decision=decision,valid_matched_controls=len(controls),compared=compared,
        shuffled_retention_of_correct_mse_gain=retention,
        realized_matching_max_relative_error=np.max([r['edit_moments']['relative_errors'] for r in controls],axis=0).tolist(),
        sensitivity=sensitivity,max_baseline_reproduction_error=max(v['max_absolute_error'] for v in data['baseline_reproduction']),
        calibration_seconds=fit['elapsed_seconds'],evaluation_seconds=data['elapsed_seconds'],
        caution='Conditional diagnostic, not a randomization p value or independent future-stage validation. Only two source edit moments are matched.')
    save('summary.json',report)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    for ax,k,title in zip(axes,['fixed_within_rmse','cosine','covariance_rel'],
                         ['Fixed-group RMSE (lower better)','Pooled change cosine (higher better)','PCA covariance error (lower better)']):
        vals=np.array([v[k] for v in controls])
        ax.scatter(vals,np.arange(len(vals)),color='#5379a3',s=25,label='Matched gene permutations')
        ax.axvline(first['common_only'][k],color='#13846c',lw=2,label='Correct gene identity')
        ax.axvline(first['copy'][k],color='gray',ls=':',label='CopyLast')
        ax.axvline(first['temperature_only'][k],color='#b47625',ls='--',label='Temperature only (RMS unmatched)')
        ax.set(xlabel=title,ylabel='Permutation ID',yticks=[0,8,16,24,31])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f' if k!='cosine' else '%.2f'))
    axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('Gene identity control: identical source cells, matched edit RMS and mean log-total change\nRetrospective E9.5 to E10.5; 32 prespecified permutations',fontsize=12)
    fig.savefig(OUT/'gene_identity_figure.png',dpi=160);plt.close(fig)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
