"""Apply predeclared recovery thresholds and report every calibration configuration."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ARMS=['copy','shared','frozen','clock','reverse','mass','global_mass','flex_mass','full']
LABELS=['Copy','Shared','Frozen','Clock','Reverse','Mass','Global + mass','Flexible + mass','Full refit']

def main():
    r=json.loads((OUT/'results.json').read_text());c=json.loads((OUT/'calibration.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    table=[];configs=[]
    for rec in r['records']:
        for arm in ARMS:
            for space in ['latent32','latent20']:
                table.append(dict(fold=rec['fold'],seed=rec['seed'],batch=rec['batch'],arm=arm,space=space,
                    **{k:rec['models'][arm][space][k] for k in ['objective','mmd_v','mmd_u','covariance_rel','mean_mse','effective_rows']}))
    with (OUT/'latent_metrics.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');writer.writeheader();writer.writerows(table)
    for fold in range(2):
        for seed in [2026112601,2026112602]:
            rows=[x for x in r['records'] if x['fold']==fold and x['seed']==seed]
            models={arm:{key:float(np.mean([x['models'][arm]['latent32'][key] for x in rows])) for key in ['objective','mmd_v','mmd_u','covariance_rel','mean_mse','effective_rows']} for arm in ARMS}
            gap=models['frozen']['objective']-models['full']['objective']
            recovery={a:(models['frozen']['objective']-models[a]['objective'])/gap if gap>0 else None for a in ARMS}
            field=next(x for x in c['fields'] if x['fold']==fold and x['seed']==seed and x['mode']=='flex_mass')
            check=next(x for x in v['checks'] if x['fold']==fold and x['seed']==seed and x['mode']=='flex_mass')
            main=next(x for x in r['full_gene_main_pool'] if x['fold']==fold and x['seed']==seed)['models']
            clock=next(x for x in c['clocks'] if x['fold']==fold and x['seed']==seed)
            flags=dict(diagnostic_has_gain=gap>0,
                clock_sufficient=bool(gap>0 and recovery['clock']>=.8 and models['clock']['covariance_rel']<=models['frozen']['covariance_rel']),
                mass_sufficient=bool(gap>0 and recovery['mass']>=.8),
                global_mass_sufficient=bool(gap>0 and recovery['global_mass']>=.8),
                flex_mass_sufficient=bool(gap>0 and recovery['flex_mass']>=.8),
                direction_needed=bool(gap>0 and models['full']['objective']<=.8*models['flex_mass']['objective']
                    and models['full']['mmd_u']<models['flex_mass']['mmd_u'] and models['full']['covariance_rel']<models['flex_mass']['covariance_rel']
                    and field['speed_near_upper_fraction']<.05))
            configs.append(dict(fold=fold,seed=seed,models=models,recovery=recovery,flags=flags,
                clock=clock['clock']['selected'],reverse=clock['reverse']['selected'],
                full_gene_main_pool=main,flex_boundary=check))
    decision={k:all(x['flags'][k] for x in configs) for k in configs[0]['flags']}
    mass_boundary=max(x['flex_boundary']['endpoint_logweight_near_bound_fraction'] for x in configs)
    summary=dict(scope=r['scope'],decision=decision,configurations=configs,flex_max_growth_bound_fraction=mass_boundary,
        field_comparisons=c['fields'],old_replay=v['old_replay'],verification=v['checks'],
        preserved=all(c['inputs_unchanged'].values()) and all(r['inputs_unchanged'].values()) and v['frozen_unchanged'])
    extra=None
    if (OUT/'unbounded_results.json').exists():
        u=json.loads((OUT/'unbounded_results.json').read_text());uf=json.loads((OUT/'unbounded_calibration.json').read_text())
        extra=[]
        for cfg in configs:
            rows=[x for x in u['records'] if x['fold']==cfg['fold'] and x['seed']==cfg['seed']]
            models={arm:{k:float(np.mean([x['models'][arm]['latent32'][k] for x in rows])) for k in ['objective','mmd_u','covariance_rel','effective_rows']} for arm in ['flex_mass_unbounded','full_unbounded']}
            flex,full=models['flex_mass_unbounded'],models['full_unbounded']
            rec=(cfg['models']['frozen']['objective']-flex['objective'])/(cfg['models']['frozen']['objective']-full['objective'])
            gene=next(x for x in u['full_gene_main_pool'] if x['fold']==cfg['fold'] and x['seed']==cfg['seed'])['models']
            extra.append(dict(fold=cfg['fold'],seed=cfg['seed'],models=models,recovery_flex_unbounded=rec,
                full_beats_flex_on_loss_mmd_cov=bool(full['objective']<=.8*flex['objective'] and full['mmd_u']<flex['mmd_u'] and full['covariance_rel']<flex['covariance_rel']),
                full_gene_main_pool=gene))
        summary['growth_cap_check']=dict(scope=u['scope'],configurations=extra,fit_diagnostics=uf['diagnostics'],
            flex_recovers_80_percent_all=all(x['recovery_flex_unbounded']>=.8 for x in extra),
            full_still_beats_flex_all=all(x['full_beats_flex_on_loss_mmd_cov'] for x in extra),
            preservation=u['forecasts_unchanged'] and u['original_forecasts_unchanged'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['速度与方向诊断结果','',
        '明确使用E10.5拟合/选参，以下只是未参与当轮校准的E10.5细胞评价，不是新的未来预测成绩。',
        '两目标半区互换、两个原始种子、五源池；这些不构成独立发育时点或胚胎重复。','',
        '原有硬上限模型的预先判别规则：'+json.dumps(decision,ensure_ascii=False),
        '该规则不能忽略丰度硬上限的干扰；追加的去硬上限检验单独报告。',
        f'flex质量整段logw接近边界的最大比例：{mass_boundary:.4%}；方向解释需同时考虑该限制。','',
        '每配置先平均5源池，主空间为早期固定PCA32。表列：主分布损失 / MMD U代理 / 协方差相对误差 / 可恢复差距比例。']
    for cfg in configs:
        lines += ['',f'目标半区 {cfg["fold"]}，种子 {cfg["seed"]}；正时间尺度={cfg["clock"]["tau"]:.5f}，反转尺度={cfg["reverse"]["tau"]:.5f}']
        for a in ARMS:
            m=cfg['models'][a];recovery=cfg['recovery'][a]
            lines.append(f'{a:12s} {m["objective"]:.7f} / {m["mmd_u"]:.7f} / {m["covariance_rel"]:.7f} / {recovery:.2%}' if recovery is not None else f'{a} 无可恢复差距')
    lines += ['', '第一批R40相同2000来源，全基因解码代价；每项列出4配置均值（不把四配置当独立实验做显著性）：',
        '模型\tpooled RMSE\t固定簇内RMSE\t变化余弦\t解码后PCA20协方差误差']
    for a in ARMS:
        vals=[cfg['full_gene_main_pool'][a] for cfg in configs]
        data=[np.mean([q[k] for q in vals]) for k in ['pooled_rmse','fixed_within_rmse','cosine']]
        data.append(np.mean([q['decoded_latent20']['covariance_rel'] for q in vals]))
        lines.append(a+'\t'+'\t'.join(f'{x:.7f}' for x in data))
    if extra:
        lines+=['','追加去除丰度硬上限的结果（事后机制诊断，原增长正则保持不变）：']
        for cfg in extra:
            lines.append(f'fold={cfg["fold"]}, seed={cfg["seed"]}, 保方向恢复比例={cfg["recovery_flex_unbounded"]:.2%}')
            for arm,m in cfg['models'].items():
                lines.append(f'{arm}: loss={m["objective"]:.7f}, MMD={m["mmd_u"]:.7f}, covariance={m["covariance_rel"]:.7f}, ESS={m["effective_rows"]:.1f}')
        lines+=['去硬上限模型：第一批源池的4配置全基因均值代价（pooled RMSE / 群内RMSE / 余弦 / 解码协方差）：']
        for arm in ['flex_mass_unbounded','full_unbounded']:
            vals=[x['full_gene_main_pool'][arm] for x in extra]
            data=[np.mean([q[k] for q in vals]) for k in ['pooled_rmse','fixed_within_rmse','cosine']]
            data.append(np.mean([q['decoded_latent20']['covariance_rel'] for q in vals]))
            lines.append(arm+' / '+' / '.join(f'{x:.7f}' for x in data))
        lines+=['不应仅因误差下降就推荐无限制重加权：必须同时查看有效源细胞数ESS与表达解码代价。',
            '如果去硬界限保方向模型已恢复80%以上差距，则按原80%标准，不能再声称大部分改进必须改变方向；full优势只说明进一步拟合可由方向自由度获得。']
    lines += ['', '场方向读数：full重拟合场对同一E9.5状态的速度，与原场比较。不能当真实谱系速度。']
    for fld in c['fields']:
        if fld['mode']=='full':lines.append(f'fold={fld["fold"]}, seed={fld["seed"]}: 余弦中位数 {fld["cosine_quantiles"][2]:.5f}，负向细胞比例 {fld["negative_direction_fraction"]:.2%}，正交能量比例 {fld["orthogonal_energy_fraction"]:.2%}')
    lines += ['',f'原4步预测回放最大误差：{max(x["max_absolute_old_prediction_error"] for x in v["old_replay"]):.3g}',
        f'16→32步位移相对变化最大：{max(x["relative_displacement_16_to_32"] for x in v["checks"]):.3%}',
        f'记录的输入文件保持不变：{summary["preserved"]}。',
        '', '指标解释：主损失=3带宽RBF MMD V +0.05*均值MSE+0.01*协方差元素MSE；MMD U为去自配对的质量加权代理，不宣称数据依赖权重下无偏。',
        '全部协方差ddof0。目标核用固定2000细胞，其余目标矩用整个外层半阶段。全基因RMSE在log1p(CP10K)空间。群内按外层目标固定早期簇比例加权。',
        '当前实验与上一轮的单带宽/全部E10.5目标不同，不能直接横比原MMD数值。',
        '', '复跑命令：', 'cd .',
        *[f'.venv/bin/python analysis/independent_review_20261126/speed_direction_diagnostic/{name}.py' for name in ['prepare','calibrate','evaluate','verify','unbounded_mass','evaluate_unbounded','summarize']],
        '依赖沿用joint_dynamics_trial/deps中的torch==2.10.0，不修改旧文件。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for j,cfg in enumerate(configs):
        label=f'Fold {cfg["fold"]+1}, seed {cfg["seed"]-2026112600}'
        for ax,k in zip(axs[:2],['objective','covariance_rel']):
            ax.plot(range(len(ARMS)),[cfg['models'][a][k] for a in ARMS],'-o',alpha=.65,label=label)
    axs[0].set_yscale('log');axs[0].set_title('Held-out distribution objective (lower better)')
    axs[1].set_title('Held-out covariance error (lower better)')
    for ax in axs[:2]:ax.set_xticks(range(len(ARMS)),LABELS,rotation=50,ha='right');ax.grid(alpha=.2)
    for x in c['clocks']:
        curve=sorted(x['clock']['curve'],key=lambda v:v['tau'])
        axs[2].plot([v['tau'] for v in curve],[v['loss'] for v in curve],alpha=.7,label=f'Fold {x["fold"]+1}, seed {x["seed"]-2026112600}')
        best=x['clock']['selected'];axs[2].scatter([best['tau']],[best['loss']],s=35)
    axs[2].set(xlabel='Positive time scale',ylabel='Calibration selection loss',title='Clock calibration (not held-out test)');axs[2].legend(fontsize=8);axs[2].grid(alpha=.2)
    axs[0].legend(fontsize=7)
    fig.suptitle('Uses E10.5 calibration; held-out cells share the same stage; no independent temporal validation')
    fig.tight_layout();fig.savefig(OUT/'diagnostic_results.png',dpi=170);plt.close(fig)
    if extra:
        arms=['frozen','clock','mass','flex_mass','flex_mass_unbounded','full','full_unbounded']
        labels=['Frozen','Clock','Mass','Flex + mass','Flex, free mass','Full','Full, free mass']
        fig,axs=plt.subplots(1,3,figsize=(15,5))
        for cfg,post in zip(configs,extra):
            allmodels={**cfg['models'],**post['models']}
            axs[0].plot(range(len(arms)),[allmodels[a]['objective'] for a in arms],'-o',alpha=.7)
            axs[1].plot(range(len(arms)),[allmodels[a]['effective_rows'] for a in arms],'-o',alpha=.7)
            genes={**cfg['full_gene_main_pool'],**post['full_gene_main_pool']}
            axs[2].plot(range(len(arms)),[genes[a]['pooled_rmse'] for a in arms],'-o',alpha=.7)
        axs[0].set_yscale('log');axs[0].set_title('Held-out latent objective (lower better)')
        axs[1].set_title('Effective source rows / 2000');axs[1].set_ylim(0,2100)
        axs[2].set_title('Decoded full-gene pooled RMSE')
        for ax in axs:ax.set_xticks(range(len(arms)),labels,rotation=45,ha='right');ax.grid(alpha=.2)
        fig.suptitle('Post-hoc growth-bound check: better fit can cost source diversity and decoder consistency')
        fig.tight_layout();fig.savefig(OUT/'growth_bound_tradeoffs.png',dpi=170);plt.close(fig)
    print('\n'.join(lines[:23]))

if __name__=='__main__':main()
