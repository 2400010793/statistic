# Virtual Embryo：本地数据、基线与复现入口

本文说明本仓库的实验入口，配套[中文学习计划](virtual-embryo-learning-plan.zh-CN.md)。命令均从仓库根目录运行。**公开资源清单、源码下载、模型训练、预测生成、官方比赛提交是不同步骤**；安装论文源码不会自动完成论文复现，也不会获得受保护的比赛真值。

## 1. 安装与快速检查

先为本项目创建独立 Python 环境，避免旧论文依赖污染已有环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m virtual_embryo_benchmark --help
python -m virtual_embryo_benchmark catalog
python -m virtual_embryo_benchmark models list
```

`catalog` 查看数据资源及访问条件；`models list` 查看本地基线与论文源码入口。首次运行应先确认路径、磁盘空间和数据许可。实验产物保存在数据或输出目录，不能将大规模 `.h5ad`、环境目录和模型权重加入 Git。

默认目录为：`data/panels/` 保存公开基因 panel，`data/samples/` 保存公开样例，`data/raw/` 保存授权原始数据，`third_party/` 保存论文源码，`artifacts/` 保存预测与评分结果。

`ot_flow` 另需 PyTorch；论文适配另需 GeomLoss、POT 等。需要全部当前可运行模型和测试时安装：

```bash
python -m pip install -e '.[flow,papers,dev]'
```

### 用公开样例检查整条管线

先获取公开资源，再运行小样例；需要测试神经流时加入 `--include-flow`：

```bash
python -m virtual_embryo_benchmark download --all --public-only
python -m virtual_embryo_benchmark demo --max-cells 150
python -m virtual_embryo_benchmark demo --include-flow --epochs 20 --max-cells 150
python -m virtual_embryo_benchmark demo --include-flow --include-papers --epochs 20 --max-cells 150
```

默认产物位于 `artifacts/demo/`。基础 demo 覆盖 8 个模型/任务组合；`--include-flow` 加入本仓库神经流，`--include-papers` 自动获取锁定的 WOT/PRESCIENT 源码并运行两个真实作者模型适配，共 11 个组合。PRESCIENT 和神经流只训练少量轮次。每阶段至多 150 个公开样例细胞适合**工程烟测**，不能证明论文指标复现或完整数据性能。

demo 中，T1 复制模型按 E8.5→E9.5 的相符阶段检查评分。其余 T1 使用 E8.5/E9.5 拟合变化后生成 E10.5 预测；没有 E10.5 真值时，指标写为 `null`、状态为 `target_unavailable`，不与 E9.5 错配评分。T2 检查样例空间评分；T3 使用已见 KO 检查 WT→KO 管线，不能证明新 KO 泛化。11 个组合中有 6 个阶段相符的评分案例。真实课程结论仍需完整合法训练数据、冻结模型的未来阶段评价和官方有效提交。

## 2. 数据下载与访问边界

```bash
python -m virtual_embryo_benchmark download --help
python -m virtual_embryo_benchmark download --all --public-only
python -m virtual_embryo_benchmark download --all
```

`--public-only` 仅处理公开 sample/panel。`--all` 还处理用户授权数据 manifest 中的 training 资源；**不代表无需登录即可下载全部比赛数据**。官方训练数据可能需要注册团队并登录；未授权资源记为 blocked，验证与测试数据必须服从各阶段发布安排。存在 blocked 或 failed 条目时退出码为 2，完整状态见 `artifacts/download-report.json`；不能因为部分公开文件成功就称为全量成功。程序校验 HDF5 文件头，避免把登录页 HTML 当成 `.h5ad`。

团队获得官方训练文件后，可建立本地 `data/training-manifest.local.json`。例如下面只登记一个已经合法获得的训练文件，其余训练项仍会记为 blocked：

```json
{
  "files": [
    {
      "id": "t1-e8.5",
      "authorized": true,
      "local_path": "/storage/virtual-embryo/E8.5_RNA.h5ad"
    }
  ]
}
```

```bash
python -m virtual_embryo_benchmark download --all \
  --manifest data/training-manifest.local.json
```

每项必须声明 `authorized: true`，并在 `local_path` 和 HTTPS `url` 中二选一；建议同时记录真实 `sha256` 和 `size_bytes`。manifest 只接受 catalog 中已发布的训练 ID，不能自造 held-out 文件或覆盖公开资源。可登记的训练 ID 为 `t1-e8.5`、`t1-e9.5`、`t2-embryo-e6.75`、`t2-embryo-e7.25`、`t2-embryo-e8.0`、`t2-heart-e8.25`、`t2-heart-e8.75`、`t2-heart-e9.5`、`t3-mab21l2-e9.5`。下载目标保持在所选项目根目录以内。不要把令牌、Cookie、带签名的下载地址或此本地 manifest 提交到 Git。

本次工作环境可用磁盘约 10 GB，全量矩阵、缓存、临时下载和论文环境可能超过这一空间。先阅读下载报告；需要完整数据时，将数据目录移到容量足够的磁盘，或先选择单任务、单阶段。下载完成后核对文件大小、校验和、AnnData 可读性、矩阵形状、基因标识和阶段元数据。中断或访问受限不能记作“全量下载完成”。

官方入口：[数据说明](https://virtualembryo.ai/challenge/data)、[竞赛规则](https://virtualembryo.ai/challenge/rules)。不要尝试恢复隐藏测试真值或使用规则禁止的阶段、基因型、预训练模型。外部论文数据只能在独立复现实验中使用，符合比赛规则并披露后才可考虑并入比赛训练。

## 3. 可直接运行的轻量基线

| 模型标识 | 任务 | 作用与解释 |
|---|---|---|
| `copy_last` | T1/T2 | 从最新允许阶段复制或重采样细胞，作为不发生后续变化的基线；T2 保留源坐标 |
| `pseudobulk_shift` | T1/T2 | 按细胞类型计算两个已见阶段的均值变化，再移位同类型源细胞；T2 保留源坐标 |
| `ot_linear` | T1/T2 | 在已见快照的低维空间拟合 OT 迁移并沿估计位移外推；属于本仓库的小规模实现 |
| `ot_flow` | T1/T2 | 根据已见快照的 OT 迁移拟合流场并积分外推；轨迹可识别性仍依赖建模假设 |
| `wot_linear` | T1 | 调用锁定作者 WOT 的 `OTModel.compute_transport_map`，在真实耦合上求祖先均值，再显式线性外推 |
| `prescient` | T1 | 调用锁定作者 `AutoGenerator`、`OTLoss`、`_step`，训练双阶段随机潜势并模拟未来 |
| `wt_identity` | T3 | 直接输出同阶段 WT，检查复杂模型是否优于“没有扰动效应” |
| `shift_transfer` | T3 | 把已观察 KO 相对其同阶段 WT 的全局平均变化，迁移到目标阶段 WT；不按细胞类型分组 |
| `gene_ko` | T3 | 先执行 `shift_transfer`，再将指定目标 KO 基因归零；下游变化来自迁移假设 |
| `gene_zero` | T3 | 仅将 WT 中指定基因的表达归零；不建模下游调控与组织响应 |

这些是透明、可复跑的比较基线。除非与固定版本的官方实现逐项一致，否则不能称为官方模型的完整复现。每次运行保存配置、随机种子、数据指纹、软件版本、预测文件与日志。

示例配置位于仓库 `configs/` 目录。将其中输入路径改成真实训练文件，再运行：

```bash
python -m virtual_embryo_benchmark run --config configs/t1-copy-last.json
python -m virtual_embryo_benchmark run --config configs/t1-pseudobulk-shift.json
python -m virtual_embryo_benchmark run --config configs/t3-shift-transfer.json
python -m virtual_embryo_benchmark models install WOT
python -m virtual_embryo_benchmark models install PRESCIENT
python -m virtual_embryo_benchmark run --config configs/t1-wot-linear.json
python -m virtual_embryo_benchmark run --config configs/t1-prescient.json
```

配置使用扁平 JSON 字段：`task`、`model`、`source`、`output`、`n_cells`、`seed` 为主要字段；`source` 是预测出发的最新阶段或目标 WT。T1 变化模型另用 `history`、`history_stage`、`source_stage`、`target_stage`；T3 迁移另用 `observed_wt`、`observed_perturbed`、`observed_stage` 与 `target_stage`。示例路径是约定路径，需要按实际下载文件修改。

`pseudobulk_shift` 示例的 `scale_by_time: false` 使用一次已见均值差；改为 `true` 时按 `(target_stage - source_stage) / (source_stage - history_stage)` 缩放。两者到 E10.5 的系数相同，到 E12.5 则不同，必须作为两个明确配置记录。`board` 标识目标 gene panel/任务板，不授予读取对应目标真值的权限。

T1 的类型均值位移依赖 `celltype_key`。某类型在历史或源阶段缺失，或任一阶段少于 `min_cells`，则冻结该类型；`min_cells` 默认是 1。模型继承源细胞组成，不能发现未来新类型或增殖规律；截断负表达也可能改变群内分布。提高 `min_cells` 是新的建模选择，必须记录。

T1 的 E10.5 示例用于验证阶段预测；最终预测 E12.5 前，应重新核对官方阶段许可，并显式修改目标时间和输出路径。T3 目标 Gata4 的阶段为 E8.75，已见 Mab21l2 KO 为 E9.5，需要两个对应阶段的 WT 对照；WT 复用 T2 心脏训练数据，避免重复下载。**把一个基因 KO 的全局平均变化迁移给另一个 KO，不是已学到新基因的因果响应**；组成差异也可能进入全局均值，该基线适合暴露数据和评价问题。

两个作者模型适配均只在 history/source 上拟合 PCA，要求 `history_stage < source_stage < target_stage`，保留源细胞组成。WOT 使用均匀增长先验和作者非平衡 OT；线性未来外推是本仓库增加的假设。PRESCIENT 使用均匀权重，省略论文预训练与增长估计，不能据此声称论文完整复现。WOT 保存 `.transport.h5ad`；PRESCIENT 保存 `.model.pt`，含作者权重、PCA、尺度和数据指纹；运行记录同时保存 checkpoint SHA256。源码 commit 与哈希不符合锁定版本时拒绝运行。

PRESCIENT 配置支持 `epochs`、`n_components`、`device`、`hidden_dim`、`batch_size`、`dt`、`noise_sd`、`learning_rate`、`sinkhorn_blur`、`sinkhorn_scaling`；WOT 支持 `n_components`、`epsilon`、`n_iter`。配置默认 CPU；示例 PRESCIENT 配置使用 `cuda:0`，无 CUDA 时改为 `cpu`。两者每个训练阶段最多抽样 5000 细胞，另有 dense 内存预算；全基因 2000 细胞双阶段实验建议预留至少 4 GiB RAM，8 GiB 更稳。当前实现用于可审查的小规模基准，尚未完成百万细胞扩展。

## 4. 输入与输出契约

- 输入为 AnnData `.h5ad`，`.X` 的处理状态须与比赛契约一致。若官方文件已 log-normalized，不再重复 normalize/log1p。
- 使用稳定、唯一的 `var_names`；所有阶段按基因标识对齐，不按列位置直接拼接。缺失基因不能默默补零后声称是完整 panel。
- T1 输出一组细胞的表达分布，保留提交所需全基因 panel；不能只输出均值、PCA 或 HVG。模型在训练中使用的降维表示与提交矩阵分开保存。
- T2/T3 需要的三维坐标、坐标键和单位须满足官方格式。保留或复制坐标不等于模型学到了形态响应。
- 输入阶段、胚胎编号、基因型与处理方式写入运行记录。同阶段随机拆分不称为时间外推验证。
- 仅对允许训练数据拟合 HVG、PCA、归一化和模型；当前未发布的目标不能提前用于模型选择。

导出后先检查矩阵可读、数值有限、基因顺序一致、坐标与细胞数匹配，再执行评分与官方格式检查。局部可运行不代表平台已经接收；有效官方提交需要保存 submission ID 与平台回执。

## 5. 本地评分

```bash
python -m virtual_embryo_benchmark score --task T1 \
  --input artifacts/T1/copy_last_E10.5.h5ad \
  --target data/raw/T1/E10.5_RNA.h5ad \
  --reference data/raw/T1/E9.5_RNA.h5ad \
  --out artifacts/T1/copy_last_metrics.json \
  --seed 0 --max-cells 2000
```

`--input` 是预测细胞，`--target` 是合法获得的真实目标，`--reference` 是对应参考阶段。T1 的变化指标以相应较早阶段为参考；T3 使用同阶段 WT，也可通过 `--wt` 指定 WT。`--setting` 用于选择评分场景，以 `score --help` 的可选值为准；`--seed` 固定评估抽样，`--max-cells` 控制采样规模。程序不会自动下载隐藏真值，也不能在没有真值时报告未来生物学性能。E10.5 真值正式开放前，本地可以验证格式或端点拟合；后者应明确标为端点检查。

官方 T1 评价含 DES、DCS、MMD、CSS，需报告四项原始指标、版本、参考文件与随机种子。官方总分还使用对应 board 的 floor/ceiling 和权重；不能把本地自定义分数写成官方 leaderboard 分数。评分定义与解释见[学习计划第 3 节](virtual-embryo-learning-plan.zh-CN.md#3-评价以官方四项指标为主补充指标解释失败)。

Pearson 相关不能单独说明外推成功；还应观察表达变化量、组成、新细胞状态和共变结构。不同阶段细胞无真实一一对应关系，不计算任意匹配后的逐细胞 RMSE 作为主结果。细胞 bootstrap 只表示样本内抽样误差；有多个胚胎时，以胚胎为独立单位报告不确定性。

## 6. 论文源码与独立环境

```bash
python -m virtual_embryo_benchmark models list
python -m virtual_embryo_benchmark models install WOT
python -m virtual_embryo_benchmark models install --all
```

模型名称以 `models list` 实际输出为准，安装名称不区分大小写。仓库提供 WOT、PRESCIENT、TrajectoryNet、TIGON、CPA、GEARS、CellOT、Spateo、veckit、torchcfm 共十个版本锁定源码入口，默认下载到 `third_party/`；`--all` 获取全部源码。它不等价于在当前环境安装所有依赖，更不等价于所有模型均已适配本比赛。当前真正可训练/预测的论文适配为 `wot_linear`、`prescient`；本仓库自编的 `ot_linear`、`ot_flow` 分别记录。

| 模型 | 作者代码 | 适合先做的复现 |
|---|---|---|
| WOT | <https://github.com/broadinstitute/wot> | 两个已观察阶段间的 OT 耦合；未来预测另需外推机制 |
| PRESCIENT | <https://github.com/gifford-lab/prescient> | 作者公开分化实验的随机动力学，再单独做 T1 适配 |
| TrajectoryNet | <https://github.com/KrishnaswamyLab/TrajectoryNet> | 作者连续路径/插值实验，另行评价超出训练区间的预测 |
| TIGON | <https://github.com/yutongo/TIGON> | 小规模状态迁移与群体增长，明确采样比例与生物增长的区别 |
| CPA | <https://github.com/theislab/cpa> | 作者条件/组合干预实验，不假设一个已见 KO 足以训练组合模型 |
| GEARS | <https://github.com/snap-stanford/GEARS> | 作者基因扰动划分与知识图使用，再检查竞赛合规性 |
| CellOT | <https://github.com/bunnech/cellot> | 非配对条件响应映射；未见干预身份需要额外机制 |
| Spateo | <https://github.com/aristoteleo/spateo-release> | 已见阶段空间重建/对齐；未来生成另行建模 |
| veckit | <https://github.com/aristoteleo/veckit> | 官方格式与评分代码；复现指标版本，不是生成模型 |
| torchcfm | <https://github.com/atong01/conditional-flow-matching> | 条件流匹配工具与作者示例；不自动适配胚胎预测 |

每篇论文使用独立环境并记录 Python、PyTorch/CUDA、AnnData、Scanpy 和 OT 库版本；先运行作者最小示例，再复现一个核心图表，最后移植竞赛数据。锁定 Git commit 只能固定源码，完整环境、数据与随机性还需另外记录。

## 7. 实验记录与验收

一次可信实验至少留下：允许的数据来源与校验和、模型/评分代码 commit、配置、随机种子、运行命令、预测文件校验和、原始指标 JSON、资源消耗和失败说明。比较复杂模型时，固定数据阶段、基因 panel、输出细胞数、训练预算及评分随机种子。至少报告三个训练种子的结果；纯确定性基线重复同一次运行不应冒充三个独立训练。

本工具的验收边界是：资源访问状态可追踪、允许的数据可读取、基线能生成合法预测、评分能对给定真值运行、论文源码能按指定版本获取。实际论文复现、官方团队注册和比赛提交要分别验收，不能由“命令存在”推断完成。
