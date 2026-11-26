"""Summarize all predeclared arms, including failures, without selecting a new model."""
import csv
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
SEEDS=[2026112601,2026112602]
KEYS=['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','fixed_equal_rmse','cosine','source_composition_tv','output_composition_tv','zero_fraction','effective_unique_source_cells']

def main():
    r=json.loads((OUT/'results.json').read_text());t=json.loads((OUT/'training.json').read_text())
    rows=[];comparisons={}
    for rec in r['results']:
        for view in ['weighted','materialized']:
            for arm,v in rec[view].items():rows.append(dict(batch=rec['batch'],view=view,arm=arm,**{k:v[k] for k in KEYS}))
    with (OUT/'metrics.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['batch','view','arm']+KEYS,delimiter='\t');w.writeheader();w.writerows(rows)
    for view in ['weighted','materialized']:
        comparisons[view]={}
        for arm in r['results'][0][view]:
            comparisons[view][arm]={}
            for k in KEYS:
                vals=np.array([b[view][arm][k] for b in r['results']]);baseline=np.array([b[view]['shared'][k] for b in r['results']])
                comparisons[view][arm][k]=dict(mean=float(vals.mean()),min=float(vals.min()),max=float(vals.max()),
                    difference_mean=float((vals-baseline).mean()),wins=int(((vals>baseline) if k=='cosine' else (vals<baseline)).sum()))
    training=[{k:v[k] for k in ['seed','kind','parameters','best_step','validation_identity_loss','validation_best_loss']} for v in t['models']]
    summary=dict(passed=r['passed'],gates=r['gates'],training=training,comparisons=comparisons,primary=r['results'][0],
        baseline_verification_max_abs_error=max(v['max_abs_error'] for v in r['baseline_verification']),
        all_inputs_unchanged=all(r['inputs_unchanged'].values()) and all(t['inputs_unchanged'].values()))
    supplement=None
    if (OUT/'resampling_results.json').exists():
        supplement=json.loads((OUT/'resampling_results.json').read_text())
        rs={}
        for seed in SEEDS:
            recs=[v for v in supplement['records'] if v['seed']==seed]
            main=[v for v in recs if v['batch']==0]
            rs[str(seed)]=dict(main_mean={k:float(np.mean([v['joint_systematic'][k] for v in main])) for k in KEYS},
                main_min={k:float(np.min([v['joint_systematic'][k] for v in main])) for k in KEYS},
                main_max={k:float(np.max([v['joint_systematic'][k] for v in main])) for k in KEYS},
                differences_vs_uniform={k:float(np.mean([v['joint_systematic'][k]-v['joint_uniform'][k] for v in recs])) for k in KEYS})
        summary['resampling_diagnostic']=rs
        summary['largest_remainder_identity_count']=sum(all(v['largest_remainder_identity'] for v in supplement['allocation_audit'] if v['batch']==b and v['seed']==seed) for b in range(5) for seed in SEEDS)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['联合动力学框架本地实验报告','',f'预注册替换基线标准：{r["passed"]}。两个种子必须各自通过，不能择优报告。',
        '全部为本地代理指标，无本轮官方成绩；E10.5 被反复查看，本轮是回顾性模型比较。',
        '5批源细胞来自同一个E9.5，不能算5个独立发育时点；两个种子也不是生物学重复。','',
        '框架与旧实验的实质区别：32维连续状态、128×128共享网络、状态/相对质量共同优化终点分布损失。无类型标签或伪配对监督。',
        'velocity 是真正关闭质量头后重新训练；joint_uniform 则是同一个联合网络在预测时忽略质量。',
        'joint_open 是匹配编辑RMS的线性残差支持开放消融，尚非学习型随机基因解码器。',
        '模型只学相对分布权重，不能根据测序捕获细胞数推断真实增殖率。','',
        '阶段内验证（表示使用早期阶段抽样，验证不是新时间点）：']
    for v in training:lines.append(f'{v["seed"]} {v["kind"]}: 参数{v["parameters"]}，检查点{v["best_step"]}，恒等损失{v["validation_identity_loss"]:.7f} → {v["validation_best_loss"]:.7f}')
    lines += ['', '第一批R40相同2000源细胞，实际物化2000行：', '模型\tMMD²↓\t协方差相对误差↓\t总体RMSE↓\t固定簇内RMSE↓\t方向余弦↑']
    for arm,v in r['results'][0]['materialized'].items():lines.append(arm+'\t'+'\t'.join(f'{v[k]:.7f}' for k in ['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine']))
    lines+=['','重要物化限制：最大余数法可能把本轮小幅质量变化全部舍掉。joint与joint_uniform相同的实际矩阵不能证明质量头没有影响。',
        '精确加权视图保留了质量头，主批次数值如下（协方差ddof0）：',
        '模型\tMMD²↓\t协方差相对误差↓\t总体RMSE↓\t固定簇内RMSE↓\t方向余弦↑']
    for arm,v in r['results'][0]['weighted'].items():
        if arm in ['copy','shared'] or arm.endswith('_h1'):
            lines.append(arm+'\t'+'\t'.join(f'{v[k]:.7f}' for k in ['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine']))
    if supplement:
        lines+=['','追加随机系统重采样（事后技术诊断，3固定重复；未重拟合模型）：']
        for seed,res in summary['resampling_diagnostic'].items():
            lines.append(seed+' 主批3次均值：'+', '.join(f'{k}={res["main_mean"][k]:.7f}' for k in ['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine']))
            lines.append('跨5批×3次，相对同网络均匀质量的均值差：'+', '.join(f'{k}={res["differences_vs_uniform"][k]:+.7f}' for k in ['mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine']))
    lines+=['','五批次配对：相对shared，MMD和协方差胜出数分别为（均值差，胜出数/5）：']
    for view in comparisons:
        for arm,c in comparisons[view].items():
            if arm.startswith('joint_') and arm.endswith('_h1'):
                lines.append(f'{view} {arm}: '+', '.join(f'{k} ({c[k]["difference_mean"]:+.7f}, {c[k]["wins"]}/5)' for k in ['mmd2','covariance_rel']))
    lines+=['','第一批潜在目标与解码结果分离（精确加权，ddof0）：']
    for arm,v in r['results'][0]['latent_intended'].items():
        if arm in ['copy','shared'] or arm.endswith('_h1'):
            decoded=r['results'][0]['weighted'][arm]
            lines.append(f'{arm}: 潜在MMD {v["mmd2"]:.7f} /协方差 {v["covariance_rel"]:.7f}；解码后MMD {decoded["mmd2"]:.7f} /协方差 {decoded["covariance_rel"]:.7f}')
    lines += ['', '评价定义：全基因均值误差在 log1p(CP10K) 空间；固定簇内误差按真实目标簇比例加权。',
        'MMD²在早期冻结PCA20中计算，目标固定3批各2000；协方差参照全部E10.5。',
        '加权MMD为数据依赖质量的去自配对代理，不宣称无偏；实际行用U统计量。ddof分别为0与1。',
        '源码/输出簇TV均报告；固定簇不等于真实谱系，状态转移后固定簇内误差可能含重分配代价。',
        f'旧基线独立重算最大误差 {summary["baseline_verification_max_abs_error"]:.3g}；所有已记录输入保持不变：{summary["all_inputs_unchanged"]}。',
        '', '主张边界：通过某个训练区间不保证未来外推；本实验只约束这组表示、损失、正则和解码实现，不能否定所有更大模型。',
        '若继续，仅做一项诊断：在E10.5校准中比较“冻结早期场仅重估时间尺度”和“重新拟合速度/质量场”。区分变化速度错了与方向/分布结构变了。',
        '该诊断会使用E10.5，必须标为目标阶段校准；即使分半评价，也只检验同阶段细胞抽样稳定性，不能宣称独立时间外推。',
        '主时长1与预先固定0.5敏感性结果均保留，不能用评价后最佳时长替换主方案。','',
        '方法动机来源：https://arxiv.org/abs/2505.13413 （联合状态/质量）；本实验无半松弛OT、非VGFM复现。',
        '', '复跑：', 'cd .',
        '.venv/bin/python analysis/independent_review_20261126/joint_dynamics_trial/train.py',
        '.venv/bin/python analysis/independent_review_20261126/joint_dynamics_trial/evaluate.py',
        '.venv/bin/python analysis/independent_review_20261126/joint_dynamics_trial/check_resampling.py',
        '.venv/bin/python analysis/independent_review_20261126/joint_dynamics_trial/verify.py',
        '.venv/bin/python analysis/independent_review_20261126/joint_dynamics_trial/summarize.py',
        '已在本目录deps安装torch==2.10.0，core.py在现有环境搜索路径末尾添加deps，保留原venv依赖。',
        '如缺依赖：.venv/bin/python -m pip install --target analysis/independent_review_20261126/joint_dynamics_trial/deps -r analysis/independent_review_20261126/joint_dynamics_trial/requirements-local.txt',
        '结果：results.json（全部指标）、metrics.tsv、summary.json、training.json；frozen_predictions.npz冻结所有源/质量/位移/解码系数。',
        'joint_primary_E10.5.h5ad是第一种子主联合模型的研究输出，无论门槛是否通过都保存，不表示推荐提交。']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    names=['copy','shared',f'velocity_{SEEDS[0]}_h1',f'joint_{SEEDS[0]}_h1',f'joint_{SEEDS[0]}_h1_uniform',f'joint_{SEEDS[0]}_h1_open']
    labels=['Copy','Shared','Velocity','Joint','Joint uniform','Joint open']
    for ax,key,title in zip(axes[:2],['mmd2','covariance_rel'],['MMD squared (lower better)','Covariance relative error (lower better)']):
        for b,rec in enumerate(r['results']):
            color=f'C{b}'
            ax.plot(range(len(names)),[rec['weighted'][n][key] for n in names],'-o',alpha=.6,color=color)
            other=[n.replace(str(SEEDS[0]),str(SEEDS[1])) for n in names]
            ax.plot(range(len(names)),[rec['weighted'][n][key] for n in other],'--',alpha=.5,color=color)
        ax.set_xticks(range(len(names)),labels,rotation=35,ha='right');ax.set_title(title);ax.grid(alpha=.2)
    for model in t['models']:
        axes[2].plot([h['step'] for h in model['history']],[h['validation_loss'] for h in model['history']],label=f'{model["kind"]} seed {model["seed"]-SEEDS[0]+1}')
    axes[2].axhline(training[0]['validation_identity_loss'],ls='--',color='gray',label='Identity')
    axes[2].set(title='Observed-stage held-out fitting',xlabel='Training steps',ylabel='Distribution objective');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.suptitle('Exact weights: solid = seed 1, dashed = seed 2; colors = five source pools (not independent stages)')
    fig.tight_layout();fig.savefig(OUT/'joint_dynamics_results.png',dpi=170);plt.close(fig)
    print('\n'.join(lines[:26]))

if __name__=='__main__':main()
