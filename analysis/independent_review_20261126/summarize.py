"""Generate tables and a static research figure from completed independent replay."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text())
batches=r['results']; main=batches[0]
keys=['within_rmse','within_equal_rmse','fixed_within_rmse','fixed_equal_rmse','pooled_rmse',
      'cosine','mmd2','covariance_rel','reassigned_cells','realized_pooled_edit_rms','mean_log_row_sum']
with (OUT/'metrics.tsv').open('w') as f:
    writer=csv.writer(f,delimiter='\t'); writer.writerow(['source_batch','arm',*keys])
    for i,b in enumerate(batches):
        for name,m in b.items(): writer.writerow([i,name,*[m[k] for k in keys]])
retention={k:[(b['copy'][k]**2-b['pooled_matched'][k]**2)/(b['copy'][k]**2-b['modules_full'][k]**2)
              for b in batches] for k in ['fixed_within_rmse','fixed_equal_rmse']}
summary=dict(
    max_archived_metric_absolute_error=max(max(v['max_absolute_error'].values()) for v in r['verification']),
    pooled_matched_fraction_of_full_mse_gain=retention,
    common_only_minus_full={k:[b['common_only'][k]-b['modules_full'][k] for b in batches] for k in keys},
    mismatches_better_than_full={k:[name for name,m in main.items() if name.startswith('mismatch') and m[k]<main['modules_full'][k]]
                               for k in ['fixed_within_rmse','fixed_equal_rmse','within_rmse']},
    full_beats_pooled_in_all_batches_and_both_weights=all(b['modules_full'][k]<b['pooled_matched'][k] for b in batches for k in retention),
    primary_retains_at_least_80_percent_in_both_weights=all(v[0]>=.8 for v in retention.values()),
    note='Cell samples and cyclic label controls are not independent developmental replicates; no inferential p values.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

colors=dict(copy='#777777',r40='#ab3c3c',modules_full='#3568a6',pooled_matched='#ac7c19',common_only='#17846a')
labels=dict(copy='CopyLast',r40='R40',modules_full='Full modules',pooled_matched='Pooled-matched',common_only='Common only')
fig,ax=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
for name,m in main.items():
    if name.startswith('mismatch'):
        ax[0].scatter(m['fixed_within_rmse'],m['cosine'],color='#bcbcbc',marker='x',s=28)
for name,c in colors.items():
    m=main[name]
    ax[0].scatter(m['fixed_within_rmse'],m['cosine'],color=c,s=60,label=labels[name])
ax[0].set(xlabel='Within-group RMSE (fixed source groups; lower better)',ylabel='Pooled change cosine (higher better)',title='Matched R40 source: 2,000 rows')
ax[0].legend(fontsize=8,loc='lower right')
for k,style in [('fixed_within_rmse','-'),('fixed_equal_rmse','--')]:
    ax[1].plot(np.arange(5),100*np.array(retention[k]),style,marker='o',label='Target weights' if style=='-' else 'Equal groups')
ax[1].axhline(80,color='gray',ls=':',label='Preregistered 80% threshold')
ax[1].axhline(100,color='gray',lw=.6)
ax[1].set(xlabel='Source batch (0 = R40 source)',ylabel='MSE gain retained by pooled-matched (%)',title='Shared shift retains much of the gain',xticks=range(5))
ax[1].legend(fontsize=8)
for name,c in colors.items():
    m=main[name]
    ax[2].scatter(m['mmd2'],m['covariance_rel'],color=c,s=60,label=labels[name])
ax[2].set(xlabel='MMD squared (lower better)',ylabel='Relative covariance error (lower better)',title='Distribution tradeoff in early PCA20')
fig.suptitle('Independent retrospective E9.5 to E10.5 audit — one developmental transition',fontsize=13)
fig.savefig(OUT/'audit_figure.png',dpi=160)
plt.close(fig)
print(json.dumps(summary,indent=2))
