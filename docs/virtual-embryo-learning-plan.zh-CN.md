# NeurIPS 2026 Virtual Embryo Challenge：统计学习课程研究计划

**信息核查日期：2026-10-11。** 默认从当日开始，4 人团队，每人每周约 8–12 小时。比赛安排、数据契约和指标可能更新，提交前应重新核对官方规则、对应 board 的 gene panel 与官方评分代码。本文的实验设计和资源预算是建议，不是已经取得的实验结果。

**推荐课题：T1 时间外推中的细胞状态与群体组成预测。** 用官方简单基线建立可复现评价，再选择 PRESCIENT 或 TrajectoryNet 中的一个作为生成模型复现目标。课程贡献可以是一个经过严格比较的改进，也可以是对复杂模型何时失败的可信分析。T2、T3 作为可替换主线，不要求同时完成三个任务。

## 1. 问题背景：预测的是随时间变化的细胞群体

胚胎发育包含基因调控、细胞分化、增殖与死亡、迁移和组织形态形成。单细胞 RNA 测序给出每个细胞的表达状态，空间转录组进一步记录位置，但测量通常具有破坏性：不同阶段采集的是不同胚胎，不能把某个细胞在两个阶段的观测直接连成真实轨迹。

因此，这个项目属于**稀疏时间观测下的分布值时间序列预测**，兼有连续时间动力学学习、非平稳性、群体组成变化和域偏移。它不是常见的“同一对象每小时记录一次数值”的长序列预测。约百万细胞、11 个时间点描述的是整体资源；具体任务可用于训练的阶段远少于 11 个，不能把百万细胞视为百万条独立纵向序列。

记时间 $t$ 的第 $r$ 个胚胎观测为 $X_{t,r}\in\mathbb{R}^{n_{t,r}\times G}$，其经验分布为：

$$
\widehat P_{t,r}=\frac{1}{n_{t,r}}\sum_{i=1}^{n_{t,r}}\delta_{x_{i,t,r}}.
$$

T1 学习过去的 $P_t$ 如何产生未来的 $P_{t_*}$，输出一组预测细胞，而不是一个平均表达向量。E8.5 等标签表示小鼠胚胎发育日龄；实际时间与伪时间应分别记录。

| 任务 | 输入与输出 | 统计问题 | 当前官方划分 |
|---|---|---|---|
| T1 Temporal | 较早阶段的全转录组表达 → 未来表达分布 | 跨阶段分布预测、时间外推 | E8.5、E9.5 训练；E10.5 验证；E12.5 隐藏测试 |
| T2 心脏 | 表达与 3D 坐标 → 目标阶段联合分布 | 空间与时间联合建模 | E8.25、E8.75、E9.5 训练；E8.5 插值验证、E10.5 外推验证；E12.5 外推测试 |
| T2 全胚胎 | 原肠胚阶段表达与坐标 → 中间阶段联合分布 | 时间插值与组织形态重建 | E6.75、E7.25、E8.0 训练；E7.5 验证；E7.75 测试 |
| T3 Perturbation | 正常发育与已见 KO → 新 KO 的表达与坐标 | 未见干预泛化、条件生成 | Mab21l2 KO@E9.5 训练；Gata4 KO@E8.75 验证；β-catenin KO@E8.75 测试；使用同阶段 WT 对照 |

来源：[T1](https://virtualembryo.ai/challenge/tasks/temporal)、[T2](https://virtualembryo.ai/challenge/tasks/spatial)、[T3](https://virtualembryo.ai/challenge/tasks/perturbation)、[数据说明](https://virtualembryo.ai/challenge/data)。T1 实际是**以心脏为中心的解剖样本**，不能把其组成变化全解释为整个胚胎的生物学变化。

### 1.1 困难点及它们对实验设计的影响

| 困难 | 为什么难 | 对应处理 |
|---|---|---|
| 没有跨时间细胞配对 | 多种轨迹都能连接相同端点分布，动力学不可唯一识别 | 明确 OT、平滑性、增长率等建模假设；预测准确不等于恢复真实谱系 |
| 时间外推跨度长 | T1 从 E9.5 到 E12.5 是 3 天；短期位移规律未必延续 | 分别报告不同预测跨度；比较恒定、线性和非线性模型 |
| 新状态与组成变化 | 分化会改变细胞类型和比例；标签名称也可能变化 | 联合考察群体比例与群内表达；先统一标签词表 |
| 高维、稀疏与技术变异 | 32,285 个基因、掉零、解离偏差、不同胚胎与批次 | 使用稀疏矩阵和降维，但保留全基因解码；避免把时间信号当批次消除 |
| 细胞多、胚胎少 | 细胞不是独立生物学重复 | 按胚胎分层分析；细胞重采样仅反映样本内误差 |
| 空间坐标未跨阶段配准 | 两个胚胎各自有局部坐标系 | T2 使用配准或坐标不变描述；验证刚体变换与尺度的影响 |
| 一个已见 KO 难以识别新 KO | 基因作用、发育阶段和细胞背景共同影响响应 | T3 先做 WT 无效应基线；利用明确、合规的生物先验 |

**研究问题从一个可检验假设开始：**“简单模型在群内表达上表现尚可，但主要错在细胞组成或新状态出现；显式处理群体变化能否改善 T1 的分布与共变结构评分？”这比“尝试更大的网络”更容易产生可解释的课程结论。

## 2. 比赛节点与八周学习安排

官网核实的关键节点是 8 月 10 日验证 leaderboard 开放、8 月 15 日 starter kit 与参考基线发布、10 月 20 日 final test phase 开始并公开 validation ground truth、12 月 2 日最终提交截止、12 月 4 日正式最终评估开始、12 月 11 日公布结果。来源：[官方时间表](https://virtualembryo.ai/challenge/timeline)。截至本计划起始日处于 P2。

| 时间 | 学习与实验任务 | 必须留下的交付物/验收标准 |
|---|---|---|
| 10/11–10/13 | 确认参赛资格、Human/Agent 赛道；团队注册；理解数据与指标 | 注册状态记录；任务选择；数据文件清单、许可、SHA256 与元数据概览 |
| 10/14–10/19 | 学习 Scanpy/AnnData、分布指标；运行官方基线与格式检查 | 可重跑的 copy_last、pseudobulk_shift；争取 10/17 前完成一次有效验证提交并保存 submission ID |
| 10/20–10/26 | 下载新公开验证真值；先做冻结模型的 E10.5 回测 | 四项官方原始指标；模型选择记录；均值/组成/共变结构的错误分解 |
| 10/27–11/09 | 复现一个生成模型的原始实验；适配 T1 | 原论文核心图或指标的复现记录；T1 同数据、同预算比较；至少 3 个训练种子 |
| 11/10–11/19 | 只做一个主要改进，如时间条件或组成建模 | 改进假设、实现、对照；至少一个能否定假设的实验 |
| 11/20–11/26 | 消融、采样敏感性、失效案例与资源统计 | 消融表；细胞类型/基因程序分层结果；训练成本和限制说明 |
| 11/27–12/01 | 固定方案；使用允许阶段重训；格式检查与最终提交 | 本地预测文件、校验和、模型与配置；提前完成有效提交并保存官方回执 |
| 12/02 前 | 核查最终提交、方法说明、外部数据披露 | 至少一次有效官方提交得到确认；研究报告、贡献说明和复现入口 |
| 12/03–12/11 | 整理最终反馈，完善课程报告 | 对成功与失败的解释；明确隐藏测试真值不可获得 |

**10 月 20 日前的本地评估限制：**只有 E8.5、E9.5 两个 T1 训练阶段时，用两者拟合动力学，再评价 E9.5，是端点拟合，不是未来时间回测。随机拆分同阶段细胞也不能证明时间外推。此阶段可检查格式、端点分布拟合和复现原论文；真正的官方未来性能来自验证提交。若增加合规的早期时间点，可另建按时间顺序的前向验证。

**10 月 20 日后的回测：**先保存仅由 E8.5/E9.5 训练的模型，再对 E10.5 评分；随后按官方许可把 E10.5 纳入最终训练，预测 E12.5。只有一个新的可评估未来阶段，不能据此声称已经验证所有长跨度外推规律，也不能把反复用于调参的 E10.5 当作独立测试集。

## 3. 评价：以官方四项指标为主，补充指标解释失败

### 3.1 当前 T1 官方指标

以下权重和定义核对了[官方评分说明](https://virtualembryo.ai/challenge/evaluation?task=1&section=scoring)与公开 [veckit](https://github.com/aristoteleo/veckit) 的 `common/core_metrics.py`、`T1/metrics_v2.py`。核查代码版本为 `cc3859ab2a2305efc4544815d3f44f660efa77d4`。正式复现应保存所用版本；本地原始指标与官方总分不可混用。

| 指标 | 权重/方向 | 衡量什么 | 文献与实现依据 |
|---|---|---|---|
| DES：Differential Expression Score | 25%，越大越好 | 相对参考阶段，是否恢复正确上调/下调基因，并扣除仅利用基因表达量的机会水平 | 官方 `de_score`；多重检验依据 [R18] |
| DCS：Directional Concordance Score | 25%，越大越好 | 全基因表达变化的秩偏相关，控制参考阶段表达量 | 官方 `de_direction`；不能用绝对表达 Pearson 替代 |
| MMD：Maximum Mean Discrepancy | 30%，越小越好 | 预测与真实细胞状态分布及混合比例是否相近 | 核两样本检验 [R15]；官方 `mmd_unbiased` |
| CSS：Co-expression Structure Score | 20%，越小越好 | 基因对表达差的 variogram 是否匹配，约束共变结构 | variogram 评分思想 [R19]；官方 `variogram_score` |

设预测、目标和参考阶段的逐基因细胞平均分别为 $\mu_p,\mu_t,\mu_r$，令 $\Delta_p=\mu_p-\mu_r$、$\Delta_t=\mu_t-\mu_r$。这里官方称 pseudobulk 的量是**log-normalized 表达的细胞平均**，不同于按胚胎加总原始 counts 后用于 DESeq2/edgeR 的统计 pseudobulk。

**DES。** 真值相对参考经 Mann–Whitney U 检验、BH-FDR ≤0.05 和均值差阈值 0.25，获得上调集合 $U_t$、下调集合 $D_t$。预测按自己的变化排序，分别选出与真值集合相同数量的上/下调基因。令有方向的重叠比例为：

$$
A=\frac{|U_p\cap U_t|+|D_p\cap D_t|}{|U_t|+|D_t|},\qquad
DES=\frac{A-A_{\rm null}}{1-A_{\rm null}}.
$$

$A_{\rm null}$ 取仅按参考表达量构造正向或反向变化所得重叠的较大者。负 DES 表示比该机会基线更差。该官方细胞级检验用于比赛评分，不能据此忽略生物学重复要求。

**DCS。** 对 $\Delta_p,\Delta_t,\mu_r$ 做秩变换，计算预测与真实变化在控制参考表达量后的偏相关：

$$
DCS=\frac{r_{pt}-r_{pr}r_{tr}}{\sqrt{(1-r_{pr}^2)(1-r_{tr}^2)}}.
$$

无变化或退化情形按官方实现处理为 0；这是对变化方向的度量，不是变化幅度的完整评价。

**MMD。** 在目标拟合的 30 维 PCA 中，抽样至多 2,000 个细胞，对围绕目标 median heuristic 的五个 RBF 带宽求平均。每个核使用去掉对角项的无偏 $MMD^2$ 估计：

$$
\widehat{MMD}_u^2=
\frac{\sum_{i\ne i'}k(a_i,a_{i'})}{n(n-1)}+
\frac{\sum_{j\ne j'}k(b_j,b_{j'})}{m(m-1)}-
\frac{2\sum_{i,j}k(a_i,b_j)}{nm}.
$$

无偏估计在有限样本下可能为负，不应擅自取平方根或截成 0 后宣称与官方一致。官方 PCA 使用目标是评估器的定义；模型训练、HVG 筛选和潜空间拟合不能因此提前使用未开放目标。30 维投影会遗漏部分基因空间变化，需补充诊断。

**CSS。** 抽取约 20,000 个不同基因对，比较预测与真值上的：

$$
v_{ij}(P)=\mathbb E_{x\sim P}|x_i-x_j|^{0.5},\qquad
CSS=\operatorname{mean}_{(i,j)}\big(v_{ij}(P_p)-v_{ij}(P_t)\big)^2.
$$

这是共变结构诊断，不能把低 CSS 解读为恢复了真实因果基因调控网络。它也不唯一识别整个联合分布。文献 [R19] 提供评分思想；比赛的具体抽样、指数和归一化以官方实现为准。

**总分。** 每项原始指标以对应 board 的 `copy_last` floor 和目标 split-half ceiling 转为 skill，floor 对应 0.5、ceiling 对应 1，再按权重乘 100。floor 模型总分为 50；ceiling 是同一目标样本两半间的参考估计，不是独立胚胎间生物学上限，也不是所有模型必然低于它的理论界。记录官方原始指标、总分与评分版本，不能把自定义 MMD 或本地 proxy score 标成官方 leaderboard 分数。

### 3.2 推荐补充指标：与研究问题一一对应

| 问题 | 建议指标与操作 | 依据及局限 |
|---|---|---|
| 均值趋势和幅度是否正确 | 全基因/固定 marker 集合的均值 RMSE、相对 L2 误差、Pearson；另报告 $\Delta$ 的相关和幅度 | 与 [R10]、[R14] 的扰动比较思路一致；Pearson 对平移/缩放不敏感，不能单独使用 |
| 是否漏掉某类细胞 | 统一标签后计算组成 TV 和 JS；按类型报告比例及群内 MMD | JS 原始文献 [R20]；分类器在新状态上可能失准，报告 unknown/低置信度 |
| 是否只匹配了 PCA 主方向 | 固定训练拟合的空间中计算 Sinkhorn divergence，并报告全基因或固定基因集的 energy distance | OT [R16]、去偏 Sinkhorn [R17]、energy statistics [R28]；不是官方主分，尺度和投影须固定 |
| 基因相关结构是否保留 | 固定 200–500 个 marker/HVG 的相关矩阵差；并列官方 CSS | 不构造 32,285² 的全矩阵；相关可能来自细胞类型混合，增加群内结果 |
| 差异基因是否有生物意义 | 固定规则的带方向 top-k overlap；基因程序/通路响应对比 | BH [R18]；基因集来源与许可需披露，不能看真值后只选有利基因 |
| 模型结论是否稳定 | 至少 3 个训练种子；同一组重采样种子；报告均值、标准差和原始结果 | 训练随机性、评估抽样误差和胚胎间变异是不同来源 |

对统一细胞类型比例 $\pi,\hat\pi$，使用：

$$
TV=\tfrac12\sum_c|\pi_c-\hat\pi_c|,\quad
JS=\tfrac12 KL(\pi\|M)+\tfrac12 KL(\hat\pi\|M),\quad M=(\pi+\hat\pi)/2.
$$

所有方法使用相同注释器和标签映射，不能用预测时人工填入的类型标签评价类型恢复。官方使用自己的冻结 probe；本地注释与官方注释不一定相同。若有多个胚胎，先按胚胎报告，再做胚胎层面的配对或分层 bootstrap；只有一个胚胎时，细胞 bootstrap 只能表示该样本内抽样不确定性，不能制造独立生物学重复。

### 3.3 如改选 T2 或 T3

T2 当前四组各占 25%：表达变化（DES/DCS）、细胞分布（MMD/CSS）、组织形态与尺度（SDD/ODS/TSR）、局部空间组织（NFS）。SDD 比较归一化内部点对距离分布；ODS 比较标准化后的体素占据；TSR 是 RMS 半径比的对数；NFS 比较空间近邻平均表达的分布。来源：[T2 评分](https://virtualembryo.ai/challenge/evaluation?task=2&section=scoring)。可补充 marker 空间域、细胞类型邻接矩阵与配准敏感性。没有细胞对应时，不直接计算逐细胞坐标 RMSE；坐标标准化也不能把生长尺度完全抹掉。

T3 当前权重为 DES 30%、DCS 25%、PSS 25%、MMD/CSS 20%。PSS 衡量相对同阶段 WT 的响应幅度，由真值 DE 基因上的零截距回归斜率及拟合门槛构造。T3 仍须提交坐标，但当前形态组不参与排序。来源：[T3 评分](https://virtualembryo.ai/challenge/evaluation?task=3&section=scoring)。额外报告 WT→KO 的变化、固定 DEG 集合上的误差、组成和空间表型；必须包括 `wt_identity`，以防绝对表达的高相关掩盖未学到扰动效应。

## 4. 学术脉络与阅读顺序

### 4.1 从状态排序到群体动力学

| 阶段 | 代表工作 | 解决了什么 | 留下的缺口 |
|---|---|---|---|
| 概念起点 | Waddington 发育景观思想 | 把分化理解为状态、势垒与命运分支 | 景观本身不是可辨识的统计模型；入门可从综述 [R02] 阅读这一背景 |
| 2018–2019：表示与轨迹 | scVI [R05]；轨迹方法比较 [R03]；小鼠胚胎图谱 [R06] | 表示高维细胞状态、重建伪时间、认识真实发育细胞类型 | 伪时间排序不自动给出真实时间尺度或未来分布 |
| 2019：连接不配对时间快照 | Waddington-OT [R07] | 用最优传输连接阶段，纳入群体增殖/死亡估计 | 耦合依赖代价与增长假设，不能凭端点确定唯一真实轨迹 |
| 2020–2021：连续时间与随机分化 | TrajectoryNet [R08]、PRESCIENT [R09]；scVelo [R21] | 连续流、潜在景观、随机轨迹、局部 RNA 动力学 | 连续插值能力与长期外推能力必须分开评价；RNA velocity 需要额外观测 |
| 2023–2024：条件干预与增长 | CellOT [R10]、CPA [R11]、GEARS [R12]、TIGON [R13] | 非配对响应学习、组合干预、基因图先验、非守恒群体动力学 | 对已见干预/组合的泛化不等于只见一个 KO 后预测新 KO |
| 2024：空间组织动态 | Spateo [R22] | 多阶段空间重建、形态与分子状态联合分析 | 配准与已观测阶段重建仍不同于隐藏未来阶段预测 |
| 2023–2026：生成动力学前沿 | Flow Matching [R23]、[SF]²M [R24]、CytoBridge [R25]；综述 [R02] | 减少训练时的轨迹积分，结合随机性、非平衡增长和相互作用 | 两个时间快照的外推不可识别性仍存在，复杂性增加不保证改善 |
| 2025–2026：评价转向 | 简单基线比较 [R14]；虚拟胚胎评论 [R01] | 强调严格数据划分、变化量与群体/组织层面的评价 | 需要开放而可复核的基线与失效分析，而非只报高相关 |

这条脉络的关键转折是：**给细胞排顺序 → 连接两个分布 → 学习可生成的时间动力学 → 加入增长、空间和干预条件 → 用严格外推检验预测能力。** 不应把所有方法都称为“时间序列 Transformer”；许多通用长序列预测模型需要连续观测、固定实体和较多时间点，与本任务的采样机制不同。

### 4.2 必读与选读

第一轮按“背景 → 数据 → 评价 → 模型”读：[R01] 虚拟胚胎评论、[R04] 单细胞分析教程、[R03] 轨迹方法比较、[R15] MMD、[R07] WOT。每人用一页记录：观测单位、输入输出、关键假设、是否需要配对、评价划分、能否真正外推。

第二轮把 **[R02] 2026 生成细胞动力学综述**作为方法地图，精读 [R08] 或 [R09] 中一个，再读 [R13] 理解为什么细胞群体变化不能只靠平移解释。

第三轮按主线选读：T2 读空间技术综述 [R26] 与 Spateo [R22]；T3 读 [R10]–[R12]，并强制搭配 [R14] 的反向证据。前沿实现只在基线与核心复现完成后选择 [R24] 或 [R25]。

## 5. 如何复现论文并适配比赛

### 5.1 区分三种复现结果

1. **原论文复现：**使用论文作者公开数据、官方代码和原始划分，重做一个核心图/实验表，检查结果是否在合理误差范围内。
2. **算法移植：**把同一算法用于竞赛数据，记录输入、预处理、潜空间、时间单位、解码器等变化；这不是原论文原样复现。
3. **课程改进：**只增加一个可解释因素，和移植版本在相同数据与资源预算下比较。

原论文复现的数据用于独立学习实验，不自动并入比赛模型训练。只有完成许可、时间窗口、基因型和来源审查并披露后，外部数据才可能用于比赛。

### 5.2 最小可行复现组合

| 优先级 | 复现对象与真实代码 | 可交付实验 | 原始需求与适配风险 |
|---|---|---|---|
| 必须 | [官方基线 Colab](https://colab.research.google.com/drive/11kHZjd4IawoC6uwCqiHNuDGAYpoPlUQO?usp=sharing)、[veckit](https://github.com/aristoteleo/veckit) | copy_last、官方 pseudobulk_shift、全四项原始指标与格式检查 | 指标包与完整 baseline/notebook 是不同入口；先确认下载阶段和路径 |
| 数学入门 | WOT [R07]，[官方代码](https://github.com/broadinstitute/wot) | 选一对阶段复现 OT 耦合；比较有/无增长假设和正则化强度 | WOT 输出连接已观测分布的耦合，不是直接预测未知未来的完整模型；配对不等于谱系 |
| 推荐核心，二选一 | PRESCIENT [R09]，[模型](https://github.com/gifford-lab/prescient)、[原论文分析](https://github.com/gifford-lab/prescient-analysis) | 先复现一个公开分化数据实验，再从 E9.5 模拟 E10.5/E12.5，评价样本分布 | 随机势能景观与增长假设；竞争数据只含两阶段时动力学约束很弱；增长信息缺失要做敏感性分析 |
| 推荐核心，二选一 | TrajectoryNet [R08]，[官方代码](https://github.com/KrishnaswamyLab/TrajectoryNet) | 先复现简单轨迹/公开细胞实验，再移植连续流并做超出训练区间推断 | 原文重点包括分布间连续路径与插值；不能把论文插值优势直接当作比赛外推证据 |
| 增长方向扩展 | TIGON [R13]，[官方代码](https://github.com/yutongo/TIGON) | 在一个小规模数据上复现状态迁移与群体增长；对比是否纳入增长项 | 不守恒 OT 与 WFR 距离；全量重现复杂且计数受采样影响，建议部分复现并注明 |
| T2 替代方案 | Spateo [R22]，[官方代码](https://github.com/aristoteleo/spateo-release) | 先复现已知阶段的空间重建/对齐，再单独验证外推 | 需要坐标、组织尺度和空间表达；软件工作流不自动生成未来胚胎 |
| T3 替代方案 | CellOT [R10]，[代码](https://github.com/bunnech/cellot)；GEARS [R12]，[代码](https://github.com/snap-stanford/GEARS) | 分别复现非配对响应或组合扰动实验；再检验新基因 KO | CellOT 不直接解决未见干预身份；GEARS 需要基因图与丰富扰动监督，单个 KO 不能原样支撑原论文训练 |

**范围上限：**四人团队默认只复现官方基线、一个小 WOT 示例、一个核心生成模型；TIGON/前沿方法挑一个概念做改进。不要把表中所有仓库列成必须完成的任务。

官方 Colab 地址已核对为官网提供的链接，但本次环境无法直连其域名，未执行 notebook；公开 `veckit` 的代码与 CLI 已实际读取。开始复现时先验证 notebook 访问、数据路径与依赖；如果 notebook 不可访问，按官方基线定义自行实现并标明“重新实现”，不要声称完整复现。

### 5.3 统一复现流程

1. 保存论文版本、代码 commit、数据地址及校验和、许可证；记录 Python、CUDA、PyTorch、AnnData/Scanpy 与 OT 库版本。每个旧论文使用独立环境，先跑作者最小示例。
2. 在原论文数据上重现一个已指定的结果：WOT 可重做阶段间耦合/群体推断，PRESCIENT 可重做公开分化数据的命运偏好/分布结果，TrajectoryNet 可重做非线性路径与插值比较。数值与作者不一致时保留差异和原因，不用竞赛指标冒充原论文指标。
3. 在竞赛副本上核对 `.X` 是否已 log-normalized、基因顺序、阶段、胚胎 ID 与细胞类型；训练用 PCA/HVG/归一化参数只在允许的训练数据拟合。
4. 从约 2,000–5,000 个训练细胞的小实验开始。主结果按真实比例采样，稀有类型分层采样另作诊断；不能把平衡采样后的比例作为真实未来组成。
5. 所有方法使用相同输入阶段、基因、输出采样数量、评分种子和最大调参次数。额外数据、先验和模型参数量分别列出，不把“更多数据”效果归因于“更好模型”。
6. 保留至少 3 个训练种子的预测与日志；用一致的评估重采样种子重复评分。时间前向验证与同阶段细胞留出分开命名。
7. 固定方案后，用全部允许训练阶段重训、导出 full-panel `.h5ad`、本地检查、官方格式检查、再计分提交；保存 submission ID、时间和官方结果。

必须留下 `config`、代码 commit、数据 manifest、训练日志、预测文件校验和、原始指标 JSON、图表生成入口和模型选择理由。复现报告写清“完全复现 / 部分复现 / 重新实现 / 仅借鉴”，不可只写“已复现”。

### 5.4 本地评分示例：可运行管线，不代表官方未来得分

公开 `veckit` 的实际 CLI 要求自己提供 `--target` 和 `--reference`。官网页面中省略 target 的内部示例不能理解为本地会自动获得隐藏真值。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install veckit
mkdir -p artifacts
veckit --task T1 --input predictions/pred_E9.5.h5ad \
  --target data/T1/E9.5_RNA.h5ad \
  --reference data/T1/E8.5_RNA.h5ad \
  --out artifacts/local_proxy_metrics.json
```

安装与 CLI 来源：[公开 README](https://github.com/aristoteleo/veckit#cli)。依赖版本应另行锁定。上例用于已有阶段的 **proxy 检查**；若模型已见过 E9.5，只是拟合诊断。10 月 20 日后可对冻结的早期模型使用公开 E10.5 作真实前向回测。现在尚未安装这些环境或训练模型，本计划不声称已产生分数。

公开 scorer 主要核对所给文件之间的基因与基本结构，并不会自动从比赛平台获取最新 board panel、细胞上下限或校准锚点；仍需按官方 `index.json` 独立检查，并使用平台格式检查确认。不能把本地工具成功退出当作正式提交必定有效的证明。

### 5.5 统计基线与可研究的模型

**基线 A：copy_last。** 按最后已见阶段的真实样本分布输出。它保留异质性，是必须认真比较的强基线。

训练文件与正式评分器使用的样本数量不同；例如完整 E9.5 文件细胞数大于当前 T1 val 的提交上限。按官方 loader/采样种子复现其基线，或在允许上下限内无偏抽样并记录数量；不能直接把整个训练文件上传后期待平台自动替你缩减。

**基线 B：官方 pseudobulk_shift。** 每个类型计算两个已见阶段的平均变化，作用到末阶段的真实细胞；早期缺失该类型时保持原状态，结果裁剪为非负。首先按官方实现复现，再另加一个显式按目标时间跨度缩放的版本：

$$
\hat x_{i,c}(t_*)=\max\left(0,
x_{i,c}(t_b)+\frac{t_*-t_b}{t_b-t_a}
\big(\mu_c(t_b)-\mu_c(t_a)\big)\right).
$$

只有两个阶段共同可映射的类型能定义这种速度；新状态不应凭空获得人工真值标签。对 E12.5 的长时间倍数可能爆炸，需预先定义收缩/截断与敏感性分析，并与未缩放官方版分开命名。

**基线 C：组成与群内分布分开建模。** 写成 $P_t(x)=\sum_c\pi_c(t)P_t(x\mid c)$，用带收缩的 log-ratio 趋势估计已知类别比例，用基线 B 或生成模型预测群内变化。两个时间点无法可靠发现所有未来类别；新状态、标签映射和 unknown 必须单独诊断，不把固定类别模型当作完整发育机制。

**生成扩展：**在训练拟合的潜空间 $z$ 中学习连续时间随机过程：

$$
dz_t=f_\theta(z_t,t)\,dt+\sigma_\theta(z_t,t)\,dW_t.
$$

若研究增殖/死亡，可考虑非守恒连续性方程：

$$
\partial_t\rho_t+\nabla\!\cdot(\rho_t v_t)=g_t\rho_t.
$$

再经同一个全基因解码器生成 log-normalized 表达。时间条件、噪声项、增长项应分别消融，不能一次加入三项而无法解释来源。若没有 raw counts，不直接套用要求 counts 的 scVI/NB likelihood；可先用 PCA 或针对 log 表达的模型，但 PCA 重构可能有负值和过度平滑，需报告处理及影响。

## 6. 研究方向：什么值得做，什么适合探索

| 方向与优先级 | 可检验问题 | 最小实验 | 主要风险 |
|---|---|---|---|
| A：统计误差分解，最高优先 | 误差主要来自组成、群内均值还是异质性/共变结构？ | copy_last、shift、组成调整；四官方指标+类型比例+群内诊断 | 真值 oracle 组成只能在开放阶段用于诊断，不能送入未来预测 |
| B：时间条件与非平稳外推，推荐 | 时间依赖场能否优于恒定变化和自主动力学？ | 固定同一生成模型，比较有/无时间条件；短、长跨度对比 | 两时间点很容易拟合端点但无法约束未来；多参数可能更不稳 |
| C：增长与新状态，推荐但需限定 | 状态迁移和群体权重变化是否应分开建模？ | 平衡 OT 对不平衡 OT/重加权，按类型及全局评分 | 测序捕获数量不等于真实增殖率；增长与解剖偏差混淆 |
| D：Flow Matching / Schrödinger Bridge，前沿扩展 | 分布耦合+随机路径是否降低计算成本、改善异质性？ | 在同一小样本预算比较确定性 flow 与 [SF]²M [R24] | 端点生成质量不是外推保证；噪声和耦合正则须做敏感性分析 |
| E：细胞相互作用与非守恒动力学，研究性较强 | 局部相互作用能否解释新状态出现？ | 参考 CytoBridge [R25]，先做合成/小数据复现 | 参数不可识别、代价高；两阶段证据不足，适合作为后续课题 |
| F：T2 空间与表达联合生成，替代主线 | 形态相似但邻域分子状态错误的模型如何识别？ | 复制位置/尺度外推/联合模型；T2 四组分数与邻接统计 | 配准、尺度、采样密度和镜像盲点；工程成本显著增加 |
| G：T3 基因先验与跨干预泛化，替代主线 | 生物先验能否超过 WT identity 和同响应迁移？ | WT、shift_transfer、合规 gene graph 条件模型 | 单个 KO 极难估计新基因作用；外部图和预训练语料必须审查 |

“有希望”指这些方向对应已知缺口和可做的实验，不表示已经证实能提高本赛题排名。课程最稳妥的组合是 **A + B 或 A + C**；D/E 是完成基本工作后的拓展。预训练基础模型也可研究，但必须证明训练语料未含受保护阶段/基因型，并与从零训练及线性模型比较，不能默认它更强。

建议预先写下成功与失败标准：主要改进至少在目标官方指标上稳定优于复现基线，且其他关键指标没有明显倒退；若只在均值指标改善而 MMD/CSS 恶化，应报告它为模式收缩等失效，而不是宣布“整体更好”。结论允许是复杂模型未超越基线，但必须说明比较是否公平、样本是否足够、失败在哪里。

## 7. 分工、资源与交付

### 7.1 四人团队：共同完成一个任务

| 负责人 | 持续职责 | 主要交付 | 交叉检查 |
|---|---|---|---|
| A：数据与生物解释 | 数据 schema、标签映射、数据来源审查、marker/通路解释 | 数据 manifest、QC 报告、细胞类型词表、失效案例 | B 检查预处理是否泄漏；D 检查数据许可 |
| B：指标与统计基线 | 官方评分、copy_last/shift、补充指标、分层统计 | 独立评分入口、基线结果、误差分解与不确定性报告 | C 验证公式与模型输入一致 |
| C：模型复现与改进 | 原论文实验、T1 移植、一个主要改进与消融 | 可训练模型、原论文复现记录、种子与预算记录 | B 独立重新评分，不由模型作者手选有利结果 |
| D：工程与整合 | 环境、实验配置、任务运行、预测导出、版本与提交追踪 | 一次命令的复现入口、提交文件、实验表、报告整合 | A 核对基因和阶段，C 核对 checkpoint 与配置 |

每人至少精读一篇共同核心论文，并实际执行一次他人的复现步骤。每周一次 30–45 分钟讨论，只审查固定格式的实验表：问题、数据划分、改动、指标、成本、结论、下一步。代码审查和贡献说明按实际工作记录，不能仅按人数平均分配论文署名。

**两人版本：**A/B 合并为数据与统计，C/D 合并为模型与工程；只做 T1 + 一个生成模型，删去前沿扩展。**单人版本：**完成官方基线、组成误差分解、一个模型的小规模移植、至少一次有效提交，优先保留严格评价和失败分析。

### 7.2 资源预算与停止条件

以下是启动建议，需通过首轮 pilot 测量修正：CPU 8–16 核、RAM 32–64 GB 可先做稀疏预处理与统计基线；一个 16–24 GB GPU 可尝试小型潜空间模型，不保证所有论文的全规模训练都能完成。建议预留约 50–100 GB 本地空间用于环境、缓存、原始数据副本和预测；不要只按 `.h5ad` 文件大小估计内存。

第一轮限制约 2,000–5,000 个训练细胞、30–50 个潜维度与较小批量；记录 wall time、峰值 RAM/VRAM、样本生成时间、评分时间。公平调参按相同的最多 10 个配置或相同 GPU 小时执行，再对候选固定 3 个种子。若模型连续两轮明显劣于强基线，先检查 scale、reference、解码器和组成；若检查后仍无改善，转为失败分析，不无限扩展网络。

最小报告应包含：任务与假设、数据划分和规则、方法与基线、四项原始指标及官方结果、组成/群内分析、消融与资源成本、限制、真实参考文献、贡献说明。UMAP 图适合说明现象，不应代替定量评价。

## 8. 工程易错点与比赛规则

### 8.1 数据、内存和评分

| 易错点 | 应对与检查 |
|---|---|
| 把 `.X` 当 counts 或再次 log1p | 官方 T1 `.X` 已 log1p-normalized；T2/T3 也是 log 表达。先检查元数据、`layers`、数值范围与 loader，再选损失函数 |
| 假设 RNA velocity 数据总可用 | scVelo [R21] 常需 spliced/unspliced counts；没有相应层时不能原样复现，也不能从 log 表达伪造这些观测 |
| 把所有空间任务固定成 500 genes | 按 board 的 panel 导出。当前 T1 val 为 32,285；T2 embryo val 为 498；heart/T3 val 为 500，未来 test panel 以发布版本为准 |
| 只保证基因集合相同 | 基因**顺序**也必须正确，处理重复名与物种命名；内部模型可选 HVG，最终输出仍需完整要求的 panel |
| 在 million-cell 矩阵上 `.toarray()` | 百万×32,285×float32 约 129 GB，仅矩阵即超常见工作站；用 CSR/分块/backed 和 Incremental/Sparse-aware 方法，核对具体函数是否隐式 densify |
| 构造全量 OT/核/距离矩阵 | $10^6\times10^6$ 的 float32 约 4 TB；采用子样本、mini-batch OT 或近似方法，明确计算与统计偏差 |
| 在全基因上直接构造完整相关矩阵 | 32,285²×float64 约 8.3 GB，多个副本更大；固定少量基因或采样 gene pairs |
| PCA、batch correction、HVG 在真值上拟合 | 模型参数只用允许训练数据；官方评估器使用真值拟合 PCA 是评分行为，不能转化为模型训练许可 |
| 过强批次校正抹掉时间 | 时间与批次/胚胎可能混淆；同时检查生物 marker 和阶段差异，别把阶段全部回归掉 |
| 强制细胞数一致或把提交细胞数当增长 | 官方允许范围内的预测细胞数是统计样本量，不直接等于胚胎总细胞数量。T1 val 当前为 1,000–5,118；按每个 board 的最新 min/max 检查 |
| 本地评分没传 reference/WT | `veckit` 省略时默认 input 自身，真实模型的 DE/响应指标可能失去意义；T1/T2 传正确阶段，T3 传同阶段 WT |
| 删掉 NaN 指标重新平均 | 官方缺失/NaN 指标不会作为免责项删掉；保存错误并定位原因，不私改加权 |
| 输出均值复制、负值或 raw counts | 有限非负并不证明尺度正确；验证 log scale、方差、输出群体异质性与全基因维度 |
| 空间强制配准至目标或无限缩放 | 未公开目标不可用；保留训练预测的尺度，检查平移/旋转与尺度分别对评分的影响；官方空间指标存在左右镜像盲点 |
| 一个环境塞入所有旧代码 | 每篇论文单独环境，固定 commit 和依赖；先最小示例，避免 CUDA/NumPy/AnnData 版本冲突 |
| 把官方 proxy 当隐藏测试成绩 | 只有官方计分回执才能作为比赛结果；150-cell 示例只用于接口演示 |

当前机器可读契约：[官方 panels/index.json](https://virtualembryo.ai/challenge/panels/index.json)。本仓库保存了[核查时的公开契约快照](virtual-embryo/official-panels-2026-10-11.json)，只用于版本追踪；实际提交重新获取最新版本。公开工具的 densify 行为与服务端内存策略可能不同，不依赖网页描述来推断本地峰值内存。

### 8.2 参赛边界：读规则后再使用数据与模型

以下是与本项目直接相关的当前规定，来源为[官方 Rules，尤其第 8–11 节](https://virtualembryo.ai/challenge/rules)、[数据说明](https://virtualembryo.ai/challenge/data)和[Terms](https://virtualembryo.ai/challenge/terms)。这是学习计划中的执行要求，不表示本次已经注册或接受了比赛条款。

- **Human Team 更符合通常课程合作。** 人工设计、阅读中间结果、调整模型属于 Human 工作流；使用编码助手本身不自动构成 Agent Team。Agent Team 要在运行开始时锁定配置，运行中不得由人读取结果后指导、改配置或编辑最终预测；需保存真实 trajectory、prompts、harness 等证据，当前 Rules 要求至少两类且包括 trajectory。FAQ 部分旧表述不一致时以最新 Rules 和平台为准。
- **外部数据、预训练模型和公开代码可以使用，但必须披露来源并符合许可。** 不能直接或通过预训练权重使用 held-out 的实测阶段/基因型；“公开”或“换了数据名称”不构成豁免。
- **T1 外部保护窗口当前为 $(E9.5,E13.5]$。** 正好 E9.5 可用，正好 E13.5 不可用；窗口内数据不能训练，E10.5/E12.5 目标明确受保护。公开 E10.5 验证真值后，按 final phase 当时的新许可纳入训练，不能提前使用外部同阶段数据。
- **T2 插值禁止外部实测阶段落在括号内的区间：**心脏 $(E8.25,E8.75)$、胚胎 $(E7.25,E8.0)$；边界已发布训练阶段可以用。按体节数或 Theiler stage 标注的数据也要换算审查。外推窗口及跨任务 held-out 约束按最新 Rules 核对。
- **T3 held-out KO 不能由其他等位基因或相似表型扰动绕过。** 基因图、数据库和模型预训练语料同样记录来源；无法查清语料时，优先从允许数据训练。
- **提交额度要预留。** 当前 P2 为每队、每任务、每 UTC 日 8 次计分；P3 为每个 board 整个阶段至多 2 次官方提交；格式检查不限次数。单预测文件上限 1,200 MB。中国日期与 UTC 日不同，最终时刻以平台明确时间为准。
- **数据公开与论文发表有额外条件。** 数据用于研究教育；在比赛外使用、复用或发布必须按官方要求致谢。官网明确数据尚未发表，涉及该数据的论文或公开展示需事先获得 Neil Chi 与数据生成团队批准；仓库宜只发布代码、配置与获准的分析，数据不直接上传 GitHub。
- **允许用分数比较候选，不允许反演隐藏目标。** 不通过大量构造性提交恢复目标属性、标签或评估数据。保存每次候选与选择依据，正常模型调参与恢复真值是不同的行为。

为每个外部资源记录：URL/版本、许可、阶段/基因型、保护窗口过滤、是否用于预训练、来源披露文本。团队应自行完成真实身份验证、个人规则接受和资格确认；本学习计划不代替这些步骤。

## 9. 需要补全的基础知识

| 基础 | 学到什么程度即可启动 | 学习材料 | 小练习 |
|---|---|---|---|
| 单细胞与发育生物学 | counts/归一化/log 表达、稀疏性、marker、谱系、WT/条件 KO、采样偏差 | [R04]、[R06]、[R26]；[单细胞实践教材](https://www.sc-best-practices.org/) | 解释一张细胞类型比例表，区分“标签改名”和“新类型出现” |
| 时间序列与验证 | 插值/外推、真实时间/伪时间、分布漂移、前向验证、观测单位 | [R02]、[R03]、本计划的官方划分 | 写出只有两个阶段时哪些验证声明不成立 |
| 概率与统计推断 | 经验分布、混合分布、MMD、效应量、FDR、偏相关、分层 bootstrap | [R15]、[R18]、[R20] | 比较真实群体、均值复制和比例错误三个模型 |
| 最优传输 | 耦合与边缘约束、代价矩阵、熵正则、Sinkhorn、平衡/非平衡 OT | [R07]、[R16]、[R17]、[R27] | 小规模二维分支数据，观察正则强度如何改变耦合 |
| 连续时间模型 | ODE/SDE、数值求解、时间单位、势函数、连续性方程、flow matching | [R02]、[R08]、[R09]、[R23] | 在二维合成数据上分别测试插值与超过最后观测时刻的外推 |
| 工程复现 | Python/NumPy/SciPy、AnnData/Scanpy、稀疏运算、PyTorch、环境与日志 | [AnnData 文档](https://anndata.readthedocs.io/en/stable/)、[Scanpy 文档](https://scanpy.readthedocs.io/en/stable/)、[veckit](https://github.com/aristoteleo/veckit) | 导出一个小 `.h5ad`，核对 gene order 与 log scale，并重跑同一个 seed |

前两周优先补统计评价、数据语义和 OT 的基础。选 T2 再补 3D 配准、近邻图与几何不变量；选 T3 再补因果干预、遗传调控、Perturb-seq 与组合泛化。无需先学完所有深度生成模型再开始基线。

## 10. 参考文献：真实文章、可定位链接与用途

文献按正文编号列出。题名、年份和 DOI/正式论文入口经 Europe PMC、Crossref、PMLR、JMLR、NeurIPS proceedings 或 arXiv 元数据核查；存在开放全文时附 PMC。DOI 可定位文章，不保证出版社全文免费。来源与核查记录见 [sources.json](virtual-embryo/sources.json)。

| 编号 | 文章与入口 | 类型与在本计划中的用途 |
|---|---|---|
| R01 | Cao, Lu & Qiu (2026). [Towards predictive virtual embryos with genomics and AI](https://doi.org/10.1038/s41592-026-03055-4). *Nature Methods*. [官方 PDF 入口](https://virtualembryo.ai/challenge/paper/nature-methods-comment) | 评论文章；研究背景与从虚拟细胞到组织/胚胎的动机 |
| R02 | Richter, Wang, Palma & Theis (2026). [Generative models of cell dynamics: from Neural ODEs to flow matching](https://doi.org/10.1038/s42003-026-09758-w). *Communications Biology*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12976343/) | **核心综述**；连续时间模型、OT、生成路径与前沿问题 |
| R03 | Saelens et al. (2019). [A comparison of single-cell trajectory inference methods](https://doi.org/10.1038/s41587-019-0071-9). *Nature Biotechnology* | 方法比较与综述式阅读；理解轨迹推断的假设与评价 |
| R04 | Luecken & Theis (2019). [Current best practices in single-cell RNA-seq analysis: a tutorial](https://doi.org/10.15252/msb.20188746). *Molecular Systems Biology*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6582955/) | **基础教程/综述**；预处理、QC、注释与分析设计 |
| R05 | Lopez et al. (2018). [Deep generative modeling for single-cell transcriptomics](https://doi.org/10.1038/s41592-018-0229-2). *Nature Methods*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289068/) | scVI 原论文；表示学习与原始 counts 观测模型 |
| R06 | Pijuan-Sala et al. (2019). [A single-cell molecular map of mouse gastrulation and early organogenesis](https://doi.org/10.1038/s41586-019-0933-9). *Nature*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6522369/) | 胚胎图谱与细胞类型背景；不是不经审查即可训练的外部数据 |
| R07 | Schiebinger et al. (2019). [Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming](https://doi.org/10.1016/j.cell.2019.01.006). *Cell*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/) | Waddington-OT；连接不配对快照、增长假设 |
| R08 | Tong et al. (2020). [TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics](https://proceedings.mlr.press/v119/tong20a.html). *ICML*, PMLR 119 | 动态 OT/连续流；优先复现目标之一 |
| R09 | Yeo, Saksena & Gifford (2021). [Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions](https://doi.org/10.1038/s41467-021-23518-w). *Nature Communications*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8163769/) | 随机景观与轨迹模拟；优先复现目标之一 |
| R10 | Bunne et al. (2023). [Learning single-cell perturbation responses using neural optimal transport](https://doi.org/10.1038/s41592-023-01969-x). *Nature Methods*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10630137/) | CellOT；非配对干预分布与泛化边界 |
| R11 | Lotfollahi et al. (2023). [Predicting cellular responses to complex perturbations in high-throughput screens](https://doi.org/10.15252/msb.202211517). *Molecular Systems Biology*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/) | CPA；条件表示、剂量与组合泛化 |
| R12 | Roohani, Huang & Leskovec (2024；online 2023). [Predicting transcriptional outcomes of novel multigene perturbations with GEARS](https://doi.org/10.1038/s41587-023-01905-6). *Nature Biotechnology*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11180609/) | 基因图与组合扰动；训练监督条件须与比赛区分 |
| R13 | Sha et al. (2024；online 2023). [Reconstructing growth and dynamic trajectories from single-cell transcriptomics data](https://doi.org/10.1038/s42256-023-00763-w). *Nature Machine Intelligence*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10805654/) | TIGON；不平衡群体动力学与增长 |
| R14 | Ahlmann-Eltze, Huber & Anders (2025). [Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines](https://doi.org/10.1038/s41592-025-02772-6). *Nature Methods*. [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12328236/) | **评价必读**；强基线、真实变化和谨慎解释深度模型 |
| R15 | Gretton et al. (2012). [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html). *JMLR* 13:723–773 | MMD 的理论与无偏估计依据 |
| R16 | Cuturi (2013). [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html). *NeurIPS 2013* | 熵正则 OT 与快速计算；不是直接把正则 OT cost 等同 Wasserstein |
| R17 | Feydy et al. (2019). [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://proceedings.mlr.press/v89/feydy19a.html). *AISTATS*, PMLR 89 | 去偏 Sinkhorn divergence，作为补充分布指标 |
| R18 | Benjamini & Hochberg (1995). [Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). *JRSS Series B* | 多基因检验的 BH-FDR 依据 |
| R19 | Scheuerer & Hamill (2015). [Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities](https://doi.org/10.1175/MWR-D-14-00269.1). *Monthly Weather Review* | 共变结构评价的思想来源；官方 CSS 是其具体改编 |
| R20 | Lin (1991). [Divergence measures based on the Shannon entropy](https://doi.org/10.1109/18.61115). *IEEE Transactions on Information Theory* | JS divergence 的依据 |
| R21 | Bergen et al. (2020). [Generalizing RNA velocity to transient cell states through dynamical modeling](https://doi.org/10.1038/s41587-020-0591-3). *Nature Biotechnology*. [预印本入口](https://doi.org/10.1101/820936) | scVelo；局部 RNA 动力学与 spliced/unspliced 数据要求 |
| R22 | Qiu et al. (2024). [Spatiotemporal modeling of molecular holograms](https://doi.org/10.1016/j.cell.2024.10.011). *Cell* | Spateo 正式论文；空间重建和多尺度动态，不把早期预印本当正式题名 |
| R23 | Lipman et al. (2023). [Flow Matching for Generative Modeling](https://openreview.net/forum?id=PqvMRDCJT9t). *ICLR 2023*. [预印本](https://arxiv.org/abs/2210.02747) | 前沿训练框架；生成路径参数并不自动等同真实发育时间 |
| R24 | Tong et al. (2024). [Simulation-Free Schrödinger Bridges via Score and Flow Matching](https://proceedings.mlr.press/v238/tong24a.html). *AISTATS*, PMLR 238. [代码](https://github.com/atong01/conditional-flow-matching) | [SF]²M；不配对快照的随机动力学、训练效率 |
| R25 | Zhang et al. (2025). [Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](https://arxiv.org/abs/2505.11197). *NeurIPS 2025*（arXiv 页面标注已发表）. [代码](https://github.com/zhenyiizhang/CytoBridge-NeurIPS) | CytoBridge；细胞相互作用、增长和非守恒随机动力学 |
| R26 | Moses & Pachter (2022). [Museum of spatial transcriptomics](https://doi.org/10.1038/s41592-022-01409-2) | **空间技术综述**；测量方式、技术范围与 T2/T3 观测限制 |
| R27 | Peyré & Cuturi (2019；arXiv 初稿 2018). [Computational Optimal Transport](https://arxiv.org/abs/1803.00567) | OT 教材/长篇综述；代价、正则化、算法与计算边界 |
| R28 | Székely & Rizzo (2013). [Energy statistics: A class of statistics based on distances](https://doi.org/10.1016/j.jspi.2013.03.018). *Journal of Statistical Planning and Inference* | energy distance 的依据，用于补充检查投影之外的分布变化 |

核心阅读量建议为 R01–R04、R07、R15 和一个主要模型，约 7 篇；其余按方向选读。报告中的结论应区分文献原始证据、竞赛官网事实、自己复现实验以及尚待检验的假设。
