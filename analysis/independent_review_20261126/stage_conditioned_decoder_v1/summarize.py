"""Report paired ablations, practical costs and the predeclared replacement gate."""
import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
ARMS=['autonomous_latent','temporal_noess','temporal_latent','temporal_decoded']
LABELS=['Autonomous + ESS','Stage, no ESS','Stage + ESS','Stage + ESS + decoded']
KEYS=['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine','ess_rows','ess_fraction','ess_unique']

def main():
    result=json.loads((OUT/'results.json').read_text());training=json.loads((OUT/'training.json').read_text());verification=json.loads((OUT/'verification.json').read_text())
    rows=[];configs=[]
    for r in result['records']:
        for arm,vals in r['models'].items():rows.append(dict(fold=r['fold'],seed=r['seed'],phase=r['phase'],pool=r['pool'],arm=arm,**{k:vals[k] for k in KEYS}))
    with (OUT/'metrics.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    for fold in range(2):
        for seed in [2026112601,2026112602]:
            rec=[v for v in result['records'] if v['fold']==fold and v['seed']==seed]
            early=next(v['models'] for v in rec if v['phase']==0)
            late=[v for v in rec if v['phase']==1]
            mean={a:{k:float(np.mean([v['models'][a][k] for v in late])) for k in KEYS} for a in ARMS}
            main_pool=next(v['models'] for v in late if v['pool']==1)
            configs.append(dict(fold=fold,seed=seed,early=early,late_mean=mean,late_main=main_pool))
    c,d='temporal_latent','temporal_decoded'
    checks={
        'late_objective_wins_at_least_3_of_4':sum(v['late_mean'][d]['objective']<v['late_mean'][c]['objective'] for v in configs)>=3,
        'mean_late_mmd_improves':np.mean([v['late_mean'][d]['mmd2']-v['late_mean'][c]['mmd2'] for v in configs])<0,
        'mean_late_covariance_improves':np.mean([v['late_mean'][d]['covariance_rel']-v['late_mean'][c]['covariance_rel'] for v in configs])<0,
        'main_pooled_within_2_percent_all':all(v['late_main'][d]['pooled_rmse']<=1.02*v['late_main'][c]['pooled_rmse'] for v in configs),
        'main_within_group_within_2_percent_all':all(v['late_main'][d]['fixed_within_rmse']<=1.02*v['late_main'][c]['fixed_within_rmse'] for v in configs),
        'late_mean_ess_fraction_at_least_point4_all':all(v['late_mean'][d]['ess_fraction']>=.4 for v in configs),
        'early_objective_within_5_percent_all':all(v['early'][d]['objective']<=1.05*v['early'][c]['objective'] for v in configs)}
    checks={k:bool(v) for k,v in checks.items()}
    aggregates={view:{a:{k:float(np.mean([v[view][a][k] for v in configs])) for k in KEYS} for a in ARMS} for view in ['early','late_mean','late_main']}
    ablations={}
    for name,a,b in [('stage','autonomous_latent','temporal_latent'),('ess','temporal_noess','temporal_latent'),('decoded','temporal_latent','temporal_decoded')]:
        ablations[name]={view:{k:float(aggregates[view][b][k]-aggregates[view][a][k]) for k in KEYS} for view in aggregates}
    summary=dict(scope=result['scope'],passed=all(checks.values()),checks=checks,configurations=configs,aggregates=aggregates,ablations=ablations,
        implementation_verification=verification,artifacts=result['artifacts'],materialized_research_output=result['materialized_research_output'],
        preserved=all(training['inputs_unchanged'].values()) and all(result['inputs_unchanged'].values()) and verification['frozen_unchanged'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['阶段相关、软丰度、解码一致训练v1：完成报告','',
        f'预先指定主比较 D vs C 的替换标准：{summary["passed"]}。',
        '本轮两个区间都参与拟合，明确使用E10.5校准部分。外层细胞留出不是独立时间外推；没有E12.5效果或官方分数。',
        '原有预测和记录只读，所有记录输入保持不变：'+str(summary['preserved']),
        '', '实现：共享128×128主干，阶段残差头；无丰度硬上限，增长平方正则和相对权重集中度软罚；全基因CP10K解码可微。',
        '四组共享随机初始化主干和小批量序列，来源按训练群分层并使用精确基础概率校正；评价源全部一致。',
        '所有组都根据实际解码输出选择检查点，避免选择空间不同掩盖训练目标比较。',
        '', '门槛逐项：'+json.dumps(checks,ensure_ascii=False),
        '', '晚区间：每配置先平均5源池，再平均两种子×两外层半区。指标全部来自实际解码表达。',
        '模型\t主损失\tMMD²代理\t协方差相对误差\tpooled RMSE\t固定簇内RMSE\tESS/2000']
    for a,m in aggregates['late_mean'].items():lines.append(a+'\t'+'\t'.join(f'{m[k]:.7f}' for k in ['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse'])+f'\t{m["ess_rows"]:.1f}')
    lines+=['','早区间：同一2000源池，目标为未参与本轮拟合/选择的E9.5源池并集；均值是4个拟合配置，不是独立时点。']
    for a,m in aggregates['early'].items():lines.append(a+'\t'+'\t'.join(f'{m[k]:.7f}' for k in ['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse'])+f'\t{m["ess_rows"]:.1f}')
    lines+=['','第一批R40原来源：4配置实际表达均值。']
    for a,m in aggregates['late_main'].items():lines.append(a+'\t'+'\t'.join(f'{m[k]:.7f}' for k in ['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine'])+f'\tESS={m["ess_rows"]:.1f}')
    lines+=['','消融差（后者减前者）：'+json.dumps(ablations,ensure_ascii=False),
        '', '表达尺度为全部32285基因log1p(CP10K)。主损失=3带宽MMD V+.05*PCA均值MSE+.01*PCA协方差MSE+10*全基因总体均值MSE。',
        'MMD²为去自配对的加权核代理；数据依赖权重下不声称无偏。协方差ddof0。群内RMSE使用固定早期群标签、按目标群比例加权，非真实谱系。',
        'ESS按源行计，同时保存合并重复原细胞后的ESS。软约束的目标不是逐次硬保证。',
        'D未学习基因开关，仍保留原零支持；它改的是经过这个解码器训练，而非使用新零值生成器。',
        '模型时间接口只开放已观察区间；尚未选定E10.5→E12.5方向外推规则。',
        '', f'解码器与旧NumPy实现最大差：{verification["decoder_max_absolute_difference"]:.3g}；方向有限差分梯度已通过。',
        f'加载检查点回放最大差：{max(x["replay_max_error"] for x in verification["checkpoint_replays"]):.3g}；16→32步最大位移变化：{max(x["integration_relative_change"] for x in verification["checkpoint_replays"]):.3%}。',
        '', '研究输出：temporal_decoded_weighted_pool.h5ad带有obs.prediction_weight；temporal_decoded_E10.5_diagnostic.h5ad通过系统重采样形成2000行。',
        '这两个文件都来自含E10.5校准数据的模型，不是早期盲测预测，也不代表推荐官方提交。',
        '', '复跑：','cd .',
        *[f'.venv/bin/python analysis/independent_review_20261126/stage_conditioned_decoder_v1/{cmd}' for cmd in ['prepare.py','verify.py','train.py','verify.py --after-training','evaluate.py','summarize.py']],
        '沿用joint_dynamics_trial/deps中的torch==2.10.0；其余为已有venv依赖。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    checkpoints=[]
    for model in training['models']:
        if model['arm']=='temporal_decoded':
            path=OUT/f'f{model["fold"]}_s{model["seed"]}_temporal_decoded.pt'
            checkpoints.append(dict(fold=model['fold'],seed=model['seed'],path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),best_step=model['best_step']))
    card=dict(recommended_research_arm='temporal_decoded' if summary['passed'] else None,gate_passed=summary['passed'],
        scope='Retrospective two-interval E10.5-calibrated research; not an E12.5 forecast or official submission',
        architecture=dict(latent_dimensions=32,hidden=[128,128],temporal_residual=True,
            relative_mass='unbounded with growth-square and concentration penalties',
            decoder='same bounded support-preserving CP10K decoder, differentiable'),
        observed_time_domain={'0':'E8.5','1':'E9.5','2':'E10.5'},future_time_policy_selected=False,
        checkpoints=checkpoints,artifacts=result['artifacts'],same_stage_validation_only=True)
    (OUT/'model_card.json').write_text(json.dumps(card,indent=2,ensure_ascii=False)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for v in configs:
        label=f'Fold {v["fold"]+1}, seed {v["seed"]-2026112600}'
        axs[0].plot(range(4),[v['late_mean'][a]['objective'] for a in ARMS],'-o',alpha=.7,label=label)
        axs[1].plot(range(4),[v['late_mean'][a]['ess_rows'] for a in ARMS],'-o',alpha=.7)
        axs[2].plot(range(4),[v['early'][a]['objective'] for a in ARMS],'-o',alpha=.7)
    axs[0].set_title('Late interval: decoded objective');axs[1].set_title('Late interval: effective source rows');axs[2].set_title('Early interval: decoded objective')
    axs[1].axhline(800,color='gray',ls='--',label='Gate ESS=800');axs[1].set_ylim(0,2050);axs[0].legend(fontsize=7)
    for ax in axs:ax.set_xticks(range(4),LABELS,rotation=35,ha='right');ax.grid(alpha=.2)
    fig.suptitle('Two-interval fit with E10.5 calibration; held-out cells are not independent future stages')
    fig.tight_layout();fig.savefig(OUT/'ablation_results.png',dpi=170);plt.close(fig)
    print('\n'.join(lines[:22]))

if __name__=='__main__':main()
