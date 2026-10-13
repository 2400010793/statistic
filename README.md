# statistic

统计学习课程研究计划与复现实验记录。

## NeurIPS 2026 Virtual Embryo Challenge

[中文学习计划](docs/virtual-embryo-learning-plan.zh-CN.md)包含问题定义、比赛节点、官方与补充评价指标、文献脉络、论文复现、研究方向、团队分工、工程注意事项和基础知识清单。

建议以 **T1：未来发育阶段的单细胞表达分布预测**为课程主线，先复现官方基线，再选择一个生成动力学模型，最后围绕时间外推或细胞组成变化开展研究。

计划按 2026-10-11 至 2026-12-02 安排，默认 4 人团队；同时提供单人/双人版本。文献与官网核查记录见 [sources.json](docs/virtual-embryo/sources.json)。

## 可运行的 benchmark

[使用文档](docs/virtual-embryo-benchmark.zh-CN.md)说明下载、配置、模型和评分入口。已接入 10 个可运行模型标识：T1/T2 的 `copy_last`、`pseudobulk_shift`、`ot_linear`、`ot_flow`，T1 的作者代码适配 `wot_linear`、`prescient`，以及 T3 的 `wt_identity`、`shift_transfer`、`gene_ko`、`gene_zero`。本地评分使用官方 `veckit==0.1.1`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[flow,papers,dev]'
python -m virtual_embryo_benchmark download --public-only
python -m virtual_embryo_benchmark demo --include-flow --include-papers --epochs 20
```

2026-10-13 已获取全部 6 个官方公开样例（每个 150 细胞，共约 16 MB）及 7 个提交规范文件；11 个模型/任务组合生成预测，6 个阶段相符的样例案例运行官方评分。E10.5 真值不可用的 5 个未来预测只检查输出与数值，指标记为空。样例用于工程检查，不构成论文复现、未见 KO 验证或官方有效提交。验收记录见[本地验证报告](docs/virtual-embryo-validation-2026-10-13.zh-CN.md)。

WOT 适配实际计算作者 OT 耦合，再显式线性外推；PRESCIENT 适配实际训练作者随机潜势模型，保存权重和 PCA。二者均为双快照比赛适配，未声称完整论文复现。WOT、PRESCIENT、TrajectoryNet、TIGON、CellOT、CPA、GEARS、Spateo、TorchCFM 和官方 scorer 的固定版本源码已获取，可通过 `models install --all` 重新获取；其余论文目前提供源码入口。

完整训练数据列入 9 个不重复文件的 catalog，仍需要官方授权地址。当前公开注册页显示 `Sign-in is not available yet.`。取得数据后填写本地 manifest 再执行 `download --all --manifest data/training-manifest.local.json`；隐藏测试真值不会下载。数据、源码副本、预测和环境均不进入 Git。
