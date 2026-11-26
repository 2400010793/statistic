"""Summaries of a fixed oracle factorial diagnostic, not model selection."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent

def main():
    data=json.loads((OUT/'results.json').read_text())
    def row(batch,model,grouping='fixed_source'):
        return next(r for r in data['results'] if r['batch']==batch and r['model']==model and r['grouping']==grouping)
    main=row(0,'common_only'); attr=main['attribution']['covariance_squared']; factor=attr['shapley_largest']
    source_rows=[row(i,'common_only') for i in range(5)]
    halves=[r for r in data['target_half_sensitivity'] if r['model']=='common_only']
    half_wins=sum(r['attribution']['covariance_squared']['shapley_largest']==factor for r in halves)
    gates=dict(single_and_shapley_agree=attr['single_replacement_best']==factor,
        five_source_batches_agree=all(r['attribution']['covariance_squared']['shapley_largest']==factor for r in source_rows),
        output_grouping_agrees=row(0,'common_only','output_reassigned')['attribution']['covariance_squared']['shapley_largest']==factor,
        at_least_seven_target_halves_agree=half_wins>=7)
    summary=dict(primary_model='common_only',primary_grouping='fixed_source',recommended_factor=factor if all(gates.values()) else None,
        recommendation_gates=gates,primary_combinations=main['combinations'],primary_attributions=main['attribution'],
        primary_error_decomposition=main['error_decomposition'],
        source_sensitivity=[dict(batch=r['batch'],single_best=r['attribution']['covariance_squared']['single_replacement_best'],
            shapley_fraction=r['attribution']['covariance_squared']['shapley_fraction_of_reduction']) for r in source_rows],
        output_grouping_shapley=row(0,'common_only','output_reassigned')['attribution']['covariance_squared']['shapley_fraction_of_reduction'],
        target_halves=dict(matching_rank_count=half_wins,n=len(halves),
            single_P_best_count=sum(r['attribution']['covariance_squared']['single_replacement_best']=='P' for r in halves),
            factor_fraction_range=[min(r['attribution']['covariance_squared']['shapley_fraction_of_reduction'][factor] for r in halves),max(r['attribution']['covariance_squared']['shapley_fraction_of_reduction'][factor] for r in halves)],
            all_replaced_covariance_error_range=[min(r['combinations']['PMC']['covariance_relative'] for r in halves),max(r['combinations']['PMC']['covariance_relative'] for r in halves)]),
        baseline_reproduction_max_error=max(r['max_abs_archive_error'] for r in data['verification']),
        mixture_identity_max_error=data['max_covariance_identity_relative_error'])
    inspection=json.loads((OUT.parent/'data_inspection.json').read_text())
    summary['blood_annotation_frequencies_from_initial_inspection']={s:inspection[s]['celltypes']['Blood']/inspection[s]['shape'][0] for s in ['E8.5','E9.5','E10.5']}
    summary['single_covariance_squared_error_reduction_fraction']={k:v/attr['baseline'] for k,v in attr['single_replacement_reductions'].items()}
    summary['P_and_M_squared_error_reduction_fraction']=1-main['combinations']['PM']['covariance_squared']/attr['baseline']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    keys=['covariance_relative','covariance_squared','gene_mean_rmse','pca_mean_rmse','conditional_covariance_relative']
    with (OUT/'factorial_metrics.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['batch','model','grouping','oracle_factors',*keys])
        for r in data['results']:
            for name,values in r['combinations'].items():w.writerow([r['batch'],r['model'],r['grouping'],name,*[values[k] for k in keys]])
    with (OUT/'contributions.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['batch','model','grouping','loss','factor','single_gain','order_average_gain','fraction'])
        for r in data['results']:
            for loss,a in r['attribution'].items():
                for factor in ['P','M','C']:
                    w.writerow([r['batch'],r['model'],r['grouping'],loss,factor,a['single_replacement_reductions'][factor],a['shapley'][factor],a['shapley_fraction_of_reduction'][factor]])
    with (OUT/'group_composition.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['group','source_rows','source_fraction','target_rows','target_fraction','E95_leading_type','E95_purity','E105_leading_type','E105_purity','conditional_covariance_relative'])
        for k,(a,b) in enumerate(zip(main['pred_counts'],main['target_counts'])):
            labels=[data['group_annotations'][s][k] for s in ['E95','E105']]
            w.writerow([k,a,a/sum(main['pred_counts']),b,b/sum(main['target_counts']),labels[0]['name'],labels[0]['purity'],labels[1]['name'],labels[1]['purity'],main['group_conditional_covariance_relative'][k]])
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    names=['none','P','M','C','PM','PC','MC','PMC']
    ax=axes[0]
    vals=[main['combinations'][k]['covariance_relative'] for k in names]
    ax.bar(range(8),vals,color=['#777777','#16876b','#477cac','#ad7a25','#4a986f','#72aa95','#959da6','#cccccc'])
    ax.set(xticks=range(8),xticklabels=names,ylabel='Relative covariance error (lower better)',title='All eight oracle combinations',ylim=(0,.9))
    for i,v in enumerate(vals):ax.text(i,v+.012,f'{v:.3f}',ha='center',fontsize=8)
    ax=axes[1]; factors=['P','M','C']; colors=['#16876b','#477cac','#ad7a25']
    for j,(loss,label) in enumerate([('covariance_squared','Covariance error'),('gene_mean_mse','Gene mean error')]):
        a=main['attribution'][loss]['shapley_fraction_of_reduction'];left=0
        for k,c in zip(factors,colors):
            val=100*a[k]
            ax.barh(j,val,left=left,color=c,label={'P':'Proportions','M':'Group means','C':'Within-group covariance'}[k] if j==0 else None)
            if val>4:ax.text(left+val/2,j,f'{val:.1f}%',ha='center',va='center',color='white',fontsize=9)
            left+=val
    ax.set(yticks=[0,1],yticklabels=['Covariance','Gene means'],xlabel='Order-averaged allocation of squared-error reduction (%)',title='Priorities depend on the metric',xlim=(0,100))
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18))
    ax=axes[2]; k=np.arange(12)
    ax.bar(k-.18,np.array(main['pred_counts'])/sum(main['pred_counts'])*100,width=.36,label='Matched E9.5 source',color='#477cac')
    ax.bar(k+.18,np.array(main['target_counts'])/sum(main['target_counts'])*100,width=.36,label='Observed E10.5',color='#c17140')
    ax.set(xticks=k,xlabel='Frozen early cluster (6 is predominantly Blood)',ylabel='Cell share (%)',title='Large composition differences')
    ax.legend(fontsize=8)
    fig.suptitle('Oracle error diagnosis for the shared-shift model — not a forecast\nP: proportions; M: conditional means; C: conditional covariance. Frozen PCA20; population moments.',fontsize=12)
    fig.savefig(OUT/'factorial_diagnostic.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
