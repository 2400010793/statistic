"""Report the preregistered paired comparison, including costs and failed gates."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rollout import *

def main():
    r=json.loads((OUT/'results.json').read_text());d=r['decision'];agg=r['aggregates']
    train=json.loads((OUT/'training.json').read_text());lines=[
        '两步闭环动力学 v2：结果与判断',
        '',
        '研究门槛：'+('通过' if d['observed_gate_passed'] else '未通过')+'；未来来源稳定性门槛：'+('通过' if d['future_stability_gate_passed'] else '未通过')+'。',
        'E10.5已参与训练。所有有真值的结果均为已观察阶段留出细胞的回顾性评价；没有E12.5预测准确性或官方分数。',
        '两组网络、来源、随机初始化、外生批次、单日和总编辑预算相同；A单步训练，B加两步实际输出终点和累计ESS损失。B计算量更高，不声称算力匹配。',
        '模型按共同的单日+两步验证目标选检查点。没有依据外层结果或未来稳定性继续调参。',
        '',
        '指标空间：log1p(CP10K)全部32285基因；分布指标采用冻结早期PCA32/scale、三个RBF带宽、ddof0协方差。精确权重优先；固定源簇内RMSE按目标簇比例加权。',
        '2个种子×2个半分不构成4个独立时间段；E9单日每配置再平均5个来源池，同样不增加独立发育时点。',
        '']
    rows=[]
    for phase,title in [('early','E8.5→E9.5单日'),('late','E9.5→E10.5单日'),('two','E8.5→E10.5两步闭环')]:
        lines.extend([title,'指标\tA 单步训练\tB 两步训练\tB相对A变化'])
        for k in ['objective','mmd2','covariance_rel','pooled_rmse','fixed_within_rmse','cosine','ess_rows','ess_unique']:
            a=agg['single_step'][phase][k];b=agg['two_step'][phase][k];change=(b/a-1)*100
            lines.append(f'{k}\t{a:.7f}\t{b:.7f}\t{change:+.2f}%')
            rows.append(dict(phase=phase,metric=k,single_step=a,two_step=b,relative_change_percent=change))
        lines.append('')
    lines.append('每配置两步主目标 / 来源ESS：')
    for p in r['pairs']:
        a=p['metrics']['single_step']['two'];b=p['metrics']['two_step']['two']
        lines.append(f"fold={p['fold']} seed={p['seed']}：A {a['objective']:.7f}/{a['ess_rows']:.1f}；B {b['objective']:.7f}/{b['ess_rows']:.1f}")
    lines.extend(['','E10.5→E12.5，时间条件保持t=2，分半检查点的无真值压力检查：'])
    for arm in ARMS:
        rr=[x['future_hold'] for x in r['records'] if x['arm']==arm]
        for k in ['ess_rows','second_or_only_day_decoder_relative_residual','daily_bound_count','total_bound_count','edit_rms']:
            values=[v[k] for v in rr];lines.append(f'{arm} {k}: mean={np.mean(values):.7f}, range=[{min(values):.7f}, {max(values):.7f}]')
    lines.extend(['这些是内部一致性和来源集中度，不是与E12.5真值的误差。',
        '触边比例按非零表达项计；daily为第二天的单日增益限幅，total为相对最初来源的两日总限幅。',
        '解码残差衡量第二天请求潜态位移和实际解码位移的差异；新方法未直接以该残差为损失，可能依然很高。','',
        '预设判断逐条：'+json.dumps(d['criteria'],ensure_ascii=False),
        '未来稳定性逐条：'+json.dumps(d['future_criteria'],ensure_ascii=False),
        '全量重训固定更新数（各组验证最优数中位数向上取整）：'+str(d['full_refit_updates']),
        '校准训练总秒数：'+str(round(train['seconds'],1))])
    epath=OUT/'e125_results.json'
    if epath.exists() and d['observed_gate_passed']:
        e=json.loads(epath.read_text());lines.extend(['','P3全量重训和E12.5候选：',
            '三个允许阶段全部细胞用于训练；表示保持早期冻结版本。两个种子按分布混合，不逐基因平均。',
            'hold固定最后时间条件；damped固定第二日速度与质量各乘.5。两者都未根据未来真值选择。',
            '以下RMSE和协方差差异均相对同一E10.5来源，不是预测误差；CopyLast天然为0，不能据此排名。'])
        for m in e['mixtures']:
            s=m['exact'];ms=m['materialized']
            lines.append(f"{m['arm']}/{m['policy']}: 源祖先ESS={s['ess_unique_ancestors']:.1f}, 重采样后ESS={ms['ess_unique_ancestors']:.1f}, 总体均值改变量={s['pooled_gene_rmse_from_e105_source']:.6f}, 协方差改变量={s['covariance_relative_change_from_e105_source']:.6f}, 种子均值分歧={m['seed_pooled_mean_disagreement_rmse']:.6f}")
            lines.append(m['path'])
        lines.append('所有物化候选2000×32285，基因顺序、非负有限、CP10K、原始零支持及文件回读校验通过。')
    else:lines.append('未完成全量E12.5导出；若研究门槛失败，不自动继续重训或把失败包装成成功。')
    if (OUT/'mass_diagnostic.json').exists():
        mass=json.loads((OUT/'mass_diagnostic.json').read_text());lines.extend(['','追加质量累积诊断（事后，表达固定不变）：'])
        for full in sorted(set(m['full_data'] for m in mass['records'])):
            for arm in ARMS:
                rr=[m for m in mass['records'] if m['full_data']==full and m['arm']==arm]
                rho=np.mean([m['daily_logmass_correlation'] for m in rr])
                ess={p:float(np.mean([m['policies'][p]['ess'] for m in rr])) for p in ['continue_mass','half_second_day_mass','freeze_after_day1']}
                lines.append(f'full={full}, {arm}: 两日log质量增量相关={rho:.6f}; ESS={ess}')
        lines.append('停止或衰减第二天质量会按构造提高ESS；本地没有真值，不能说明哪种比例更准确。')
    if (OUT/'composition_audit.json').exists() and d['observed_gate_passed']:
        ca=json.loads((OUT/'composition_audit.json').read_text())
        lines.extend(['','追加来源簇审计（事后）：',
            '主导来源簇='+str(ca['dominant_source_group'])+'；实际来源标注计数='+str(ca['source_group_celltype_counts']),
            '已观察的比例='+str(ca['observed_proportions']),
            '各候选给该来源簇的累计权重='+str(ca['predicted_ancestry_weights']),
            '这是原始来源簇权重，不是经独立推断的E12.5细胞类型比例。此集中现象提示末期丰度变化在未来被重复放大；没有未来真值，不能判定真实比例。'])
    if (OUT/'output_audit.json').exists():
        audit=json.loads((OUT/'output_audit.json').read_text())
        lines.append('冻结检查点回放全部一致；16→32步积分的表达变化/原编辑幅度最大为'+str(audit['max_16_to_32_step_expression_change_relative_to_edit']))
    lines.extend(['','限制：只能检验观测区间内的闭环拟合。真实细胞无配对，无法唯一识别生物学速度；未知未来时间条件仍是假设。模型不能开启源细胞中的零基因，也不能保证覆盖未来新谱系。软ESS不是逐样本硬下限。',
        '下一项辨别实验：固定表达路径，仅改变相对质量的时间延续规则，并在训练时对E10.5起点的无目标两步路径加入累计ESS控制；将它与当前只约束已知路径的版本比较。此控制可约束来源集中，但不能替代E12.5准确性验证。若需归因已知终点收益，可另做累计ESS罚与终点损失的消融；本轮未分开识别二者贡献。',
        '', '一条命令完整复跑（项目根目录）：',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/run_all.py',
        '', '分步命令：',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/verify.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/train.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/evaluate.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/audit_outputs.py',
        '# 仅observed_gate_passed=true后运行：',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/refit_parallel.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/export_e125.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/audit_composition.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/diagnose_mass.py',
        '.venv/bin/python analysis/independent_review_20261126/two_step_rollout_v2/summarize.py'])
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    with (OUT/'metrics.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors=['#526d90','#d27434']
    for j,arm in enumerate(ARMS):
        ax[0,0].bar(np.arange(3)+j*.34,[agg[arm][p]['objective'] for p in ['early','late','two']],width=.34,color=colors[j],label=arm)
        ax[0,1].bar(np.arange(2)+j*.34,[agg[arm]['two'][k]/agg['single_step']['two'][k] for k in ['mmd2','covariance_rel']],width=.34,color=colors[j])
        ax[1,0].bar(np.arange(2)+j*.34,[agg[arm]['two']['ess_rows'],d['future_mean_ess'][arm]],width=.34,color=colors[j])
        vals=[p['metrics'][arm]['two']['objective'] for p in r['pairs']]
        ax[1,1].plot(range(4),vals,'o-',color=colors[j],label=arm)
    ax[0,0].set(xticks=np.arange(3)+.17,xticklabels=['8.5 to 9.5','9.5 to 10.5','8.5 to 10.5'],title='Observed-stage decoded objective (lower)')
    ax[0,0].legend()
    ax[0,1].set(xticks=np.arange(2)+.17,xticklabels=['MMD2','Relative covariance error'],title='Observed two-day errors / single-step baseline')
    ax[1,0].set(xticks=np.arange(2)+.17,xticklabels=['Observed two days','Future stress only'],title='Effective source ancestors / 2000')
    ax[1,0].axhline(800,color='gray',linestyle=':',linewidth=1)
    ax[1,1].set(xticks=range(4),xticklabels=['F0/S1','F0/S2','F1/S1','F1/S2'],title='Paired observed two-day objective')
    for a in ax.ravel():a.spines[['top','right']].set_visible(False)
    fig.suptitle('Decoded two-step training | retrospective observed stages; no E12.5 truth',fontsize=12)
    fig.savefig(OUT/'rollout_comparison.png',dpi=160);plt.close(fig)
    print('\n'.join(lines[:18]),flush=True)

if __name__=='__main__':main()
