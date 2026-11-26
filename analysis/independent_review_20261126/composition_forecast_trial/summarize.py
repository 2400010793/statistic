"""Report both predeclared attempts, preserving negative results and sampling costs."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from core import OUT,save

def main():
    trend=json.loads((OUT/'trend_results.json').read_text()); density=json.loads((OUT/'density_results.json').read_text())
    fit=json.loads((OUT/'density_fit.json').read_text());early=np.load(OUT/'early_space.npz')
    ratios=np.load(OUT/'density_frozen.npz')['late_ratio']
    keys=['composition_tv','pooled_rmse','fixed_within_rmse','cosine','mmd2','covariance_rel']
    phases={'trend':trend,'density':density};summary={}
    with (OUT/'metrics.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['phase','batch','mode','arm',*keys,'unique_source_cells','effective_unique_source_cells'])
        for phase,data in phases.items():
            candidate=data['candidate'];parts={}
            for mode in ['weighted','materialized']:
                candidate_rows=[r[mode][candidate] for r in data['results']]
                baseline_rows=[r[mode]['carry'] for r in data['results']]
                parts[mode]={k:dict(baseline_mean=float(np.mean([r[k] for r in baseline_rows])),candidate_mean=float(np.mean([r[k] for r in candidate_rows])),
                    delta_mean=float(np.mean([r[k]-b[k] for r,b in zip(candidate_rows,baseline_rows)])),
                    favorable_batches=sum(r[k]>b[k] if k=='cosine' else r[k]<b[k]-1e-12 for r,b in zip(candidate_rows,baseline_rows))) for k in keys}
            summary[phase]=dict(candidate=candidate,passed=data['gate']['passed'],gate=data['gate'],paired_summary=parts)
            for row in data['results']:
                for mode in ['weighted','materialized']:
                    for arm,v in row[mode].items():
                        if arm=='copy_original':continue
                        writer.writerow([phase,row['batch'],mode,arm,*[v[k] for k in keys],v.get('unique_raw_source_cells',''),v.get('effective_unique_source_cells','')])
    summary['density_classifier']=dict(oof_auc=fit['oof_auc'],upper_bound_fraction=float(np.mean(ratios==2)),lower_bound_fraction=float(np.mean(ratios==.5)),ratio_quantiles=fit['ratio_quantiles'])
    summary['main_batch']={mode:{**trend['results'][0][mode],**density['results'][0][mode]} for mode in ['weighted','materialized']}
    summary['composition']=dict(early=early['frequencies'][0].tolist(),carry=early['frequencies'][1].tolist(),
        trend=np.load(OUT/'trend_frozen.npz')['trend'].tolist(),density=fit['density_shares'],target=trend['target_proportions'])
    summary['max_old_metric_reproduction_error']=max(v['archive_max_abs_error'] for data in phases.values() for v in data['verification'])
    summary['decision']='Neither predeclared candidate passed; preserve existing shared-shift baseline. Trend is a distribution/structure tradeoff, density model is not an upgrade.'
    save('summary.json',summary)
    with (OUT/'composition.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['group','E85','E95_carry','trend','density','E105'])
        for k in range(12):writer.writerow([k,*[summary['composition'][name][k] for name in ['early','carry','trend','density','target']]])
    names=['original_shared','carry','trend','density_full'];labels=['Original shared','Carry shares','Share trend','Continuous density']
    colors=['#777777','#477cac','#16876b','#b76042']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    ax=axes[0]
    for name,label,color in zip(names,labels,colors):
        for mode,marker in [('weighted','o'),('materialized','s')]:
            r=summary['main_batch'][mode][name]
            ax.scatter(r['mmd2'],r['covariance_rel'],color=color,marker=marker,s=65,label=label+(' (weights)' if mode=='weighted' else ' (2,000 rows)'))
    ax.set(xlabel='MMD squared (lower better)',ylabel='Relative covariance error (lower better)',title='Distribution gains can cost covariance accuracy')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.legend(fontsize=7,loc='best')
    ax=axes[1];xx=np.arange(4)
    for mode,offset in [('weighted',-.18),('materialized',.18)]:
        vals=[summary['main_batch'][mode][n]['fixed_within_rmse'] for n in names]
        ax.bar(xx+offset,vals,width=.36,label='Exact weights' if mode=='weighted' else '2,000 equal rows',color='#477cac' if mode=='weighted' else '#b76042')
    ax.set(xticks=xx,xticklabels=['Original','Carry','Trend','Density'],ylim=(.068,.075),ylabel='Fixed-group mean RMSE (lower better)',title='Integer sampling adds a conditional-mean cost')
    ax.legend(fontsize=8)
    ax=axes[2];stages=['early','carry','trend','density','target'];values=[100*summary['composition'][n][6] for n in stages]
    ax.bar(range(5),values,color=['#aaaaaa','#477cac','#16876b','#b76042','#754b92'])
    for i,v in enumerate(values):ax.text(i,v+.2,f'{v:.2f}%',ha='center',fontsize=9)
    ax.set(xticks=range(5),xticklabels=['E8.5','E9.5','Trend','Density','E10.5'],ylabel='Share of frozen group 6 (%)',ylim=(0,18),title='Predominantly Blood: early trend misses the surge')
    fig.suptitle('Two early-only forecast trials on the same source pools — retrospective E10.5 evaluation\nWeighted covariance uses population moments; 2,000-row covariance uses the historical sample convention.',fontsize=12)
    fig.savefig(OUT/'forecast_trials.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='main_batch'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
