# Virtual Embryo 本地验收记录：2026-10-13

本记录描述已经执行的工程检查。实验根目录为本机 `/home/chenxy/statistic`；数据与实验产物留在本机，GitHub 保存代码、配置、公开来源元数据和本文。

## 实际完成的模型接入

| 类别 | 模型 | 已执行检查 |
|---|---|---|
| 透明统计基线 | `copy_last`、`pseudobulk_shift`、`wt_identity`、`shift_transfer`、`gene_ko`、`gene_zero` | 官方样例生成预测，适用案例调用官方评分 |
| 本仓库 OT 实现 | `ot_linear`、`ot_flow` | 双阶段拟合、未来预测、有限非负数值与完整基因输出 |
| WOT 作者代码适配 | `wot_linear` | 真实 `OTModel.compute_transport_map`，保存 transport，显式线性未来外推 |
| PRESCIENT 作者代码适配 | `prescient` | 真实 `AutoGenerator`/`OTLoss`/`_step` 训练及模拟，保存权重、PCA、尺度和数据指纹 |

共 10 个模型标识，在 T1/T2/T3 形成 11 个样例模型/任务组合。T1 保留 32285 基因，其余样例保留 500 基因；每个输出均为 150 细胞。运行记录中的输出与 checkpoint SHA256 已核对。

WOT/PRESCIENT 为双快照比赛适配，尚未完成原论文数据、划分和图表复现。TrajectoryNet、TIGON、CellOT、CPA、GEARS、Spateo、TorchCFM 目前获取了固定版本作者源码，尚未接入本比赛训练入口。连同 WOT、PRESCIENT、veckit，共获取 10 份源码，保存在 `third_party/`，有 commit、archive SHA256 与文件校验记录。

## 数据实际状态

- 已下载并校验 6 个官方公开 `.h5ad` 样例，共 **16,718,536 字节（15.94 MiB）**，每文件 150 细胞。
- 已下载 7 个公开规范文件：panel index、5 个 gene panel、T1 composition。
- catalog 登记的 **9 个不重复完整训练文件均为 `blocked`**：没有官方授权下载直链。本次匿名访问[账号页](https://virtualembryo.ai/challenge/account)显示 `Sign-in is not available yet.`，公开数据页没有这些完整文件的下载地址。
- 未下载任何隐藏测试真值，未完成平台团队注册或比赛提交。

逐文件公开资源校验和及完整访问状态见 [download-audit-2026-10-13.json](virtual-embryo/download-audit-2026-10-13.json)。本机实时报告是 `artifacts/download-report.json`；`download --all` 本次退出码为 2，表示仍有受阻资源，不能称为完整下载成功。

后续取得已授权直链或本地文件后，按[使用文档](virtual-embryo-benchmark.zh-CN.md#2-数据下载与访问边界)建立私有 manifest，即可继续获取并校验完整训练集，无需修改模型代码。

## 已执行的验证

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python -m virtual_embryo_benchmark demo \
  --include-flow --include-papers --epochs 20

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q

.venv/bin/python -m virtual_embryo_benchmark download --all
```

测试结果：**113 passed**，包括模型数据对齐、阶段限制、稀疏输入、缺失基因、非有限数值、输出契约、断点下载、授权 manifest、真实作者模型、源码锁及禁止错阶段评分的回归检查。环境中已有 Bottleneck 版本提示、AnnData 索引转换提示和测试构造稀疏矩阵的效率提示，未影响通过结果；本次未修改其他项目的全局环境。

样例 demo 的 T1 copy、T2 copy 和 4 个 T3 案例共执行 6 次官方 `veckit==0.1.1` 评分。其余 5 个 T1 案例生成 E10.5 预测；由于 E10.5 真值不可用，只验证预测与 checkpoint，指标为 `null`、状态为 `target_unavailable`。不将这些预测与 E9.5 错阶段评分。

T3 已见 KO 案例只验证管线，不检验未见基因 KO 的泛化；150 细胞样例也不能满足正式 board 的输出规模。原始指标、预测和日志在 `artifacts/demo/`，不得据此报告官方 leaderboard 分数。

本次 Python 3.9 环境的直接依赖版本记录在 [python39-tested-constraints.txt](../configs/python39-tested-constraints.txt)。PyTorch 实际运行版本为 `2.6.0+cu124`，可见两张 RTX 2080 Ti；样例 demo 使用 CPU。PRESCIENT 全基因样例的独立内存检查峰值约 732 MiB；双阶段各 2000 细胞实验建议预留 4–8 GiB RAM，评分另需内存。当前没有执行完整训练集性能、百万细胞扩展或官方有效提交。
