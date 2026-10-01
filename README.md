# MVForge · 多模态/多视图表征学习与融合系统

> 同一批样本的多份特征表示（视图/模态）如何对齐、检索与融合？MVForge 给出一套模块化、可验证、离线可跑的答案。
> **作者：晨星** · MIT License · Python 3.13

---

## 1. 这是什么

**MVForge** 是一个面向**多视图 / 多模态学习（Multiview / Multimodal Learning）**的端到端系统：

- **表征学习**：典型相关分析（CCA）家族把两个视图投影到共享语义空间，使对应样本在该空间中靠近。
- **跨模态检索**：在共享空间内以视图 A 检索视图 B 的对应样本（top-1 / top-5 / mAP）。
- **多视图融合预测**：早期融合（拼接）、晚期融合（Stacking）、CCA-KNN，以及旗舰 **ModaFuse** 共识融合。

设计原则：**顶级开源复用 + 纯 numpy 离线兜底 + 契约先行模块化**。无 GPU、无网络、无密钥时，全部 demo 与单测仍可确定性跑通。

---

## 2. 技术选型（随机抽选的顶级技术与项目）

| 角色 | 选型 | 说明 |
|------|------|------|
| 核心算法 | **CCA 家族**（Classical / Kernel / Deep CCA） | 多视图表征学习基石 |
| 顶级开源复用 | **`mvlearn`** | 多视图学习权威库（KMCCA / DCCA / 多视图分类） |
| 基线/兜底 | **`scikit-learn`**（`CCA`、`RandomForest`、`LogisticRegression`、`KNN`） | 离线默认后端 |
| 离线内核 | **`numpy` / `scipy`** | 纯手写 ClassicalCCA，零重型依赖 |
| 可选深度后端 | **`torch`**（经 `mvlearn.DCCA`） | 深度非线性共享表征 |

> 遵守 SOP：**禁止从零自研 SOTA 部分**——CCA/KMCCA/DCCA 直接复用 `mvlearn` 与 `sklearn`；仅在 SOTA 后端不可用时降级为纯 numpy 实现。

---

## 3. 安装

```bash
cd mvforge
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# （可选）启用 SOTA 后端
pip install mvlearn torch
```

---

## 4. 快速开始

```bash
# 端到端演示：生成多视图数据 → 表征 → 跨模态检索 → 融合 → 落盘 benchmark.json
python -m mvforge.examples.run_demo

# 命令行（含真实 digits 双视图数据集）
python -m mvforge.cli --digits --corr 0.3 0.6 0.9 --out benchmark.json
```

```python
from mvforge.data.synthetic import make_two_view_gaussian
from mvforge.pipeline.pipeline import MVForgePipeline

ds = make_two_view_gaussian(n=600, view_corr=0.6, random_state=42)
report = MVForgePipeline().run_full(ds)
print(report["retrieval"], report["fusion"])
```

---

## 5. 模块架构（单向无环：cli → pipeline → {data, represent, fusion, retrieval, eval} → core）

```
mvforge/
  core/         types · errors(E100~E500) · config(ENV_MVFORGE_* 覆盖) · interfaces(Protocol)
  data/         synthetic(双视图高斯 + digits 双视图) · loaders(切分)
  represent/    cca(纯 numpy ClassicalCCA + CCA 投影器) · kcca(mvlearn KMCCA 兜底) · dcca(torch DCCA 兜底)
  fusion/       base(单视图基线) · early(早期融合) · late(晚期 Stacking) · cca_knn(CCA 空间 KNN) · modafuse(旗舰)
  retrieval/    crossmodal(跨模态最近邻检索)
  eval/         metrics(准确率/topk/mAP/典型相关)
  pipeline/     MVForgePipeline.run_full() + benchmark()
  cli.py        argparse 入口
  examples/     run_demo.py 端到端演示
tests/          pytest 单测（含与 sklearn 参照交叉验证）
docs/architecture.md
```

---

## 6. 离线兜底（零下载可跑）

| SOTA 后端 | 不可用时降级为 |
|-----------|----------------|
| `mvlearn.KMCCA` | 纯 numpy **ClassicalCCA**（白化交叉协方差 SVD） |
| `mvlearn.DCCA`（torch） | 纯 numpy **ClassicalCCA** |
| `mvlearn` 多视图分类 | `sklearn` RF / LogisticRegression / KNN |

所有降级路径经 `available_kcca()` / `available_dcca()` 探测，benchmark 不伪造数字。

---

## 7. 性能基线（示例，真实值见 `benchmark.json`）

在 3 档合成难度 + 真实 digits 双视图上（`view_corr` 控制难度梯度）：

| dataset | canCorr | top1 | map | sv1 | sv2 | early | late | ccaK | moda |
|---------|--------:|----:|----:|----:|----:|------:|-----:|-----:|-----:|
| gaussian_c0.30 | — | — | — | — | — | — | — | — | — |
| gaussian_c0.60 | — | — | — | — | — | — | — | — | — |
| gaussian_c0.90 | — | — | — | — | — | — | — | — | — |
| digits_two_view | — | — | — | — | — | — | — | — | — |

> 表格数值由 `run_demo` 实时生成并写入 `benchmark.json`，此处留空以保持诚实（不编造数据）。

---

## 8. 创新点：ModaFuse 共识融合

在候选池 `{单视图0, 单视图1, 早期融合, 晚期融合, CCA-KNN}` 上做**贪心前向选择**（按验证集准确率，仅当提升 > `tol` 才纳入），并设**非劣守护**：若最终集成验证准确率低于最佳单视图 − `tol`，则回退到最佳单视图，保证 ModaFuse 至少不劣于单视图基线（诚实交付，不粉饰）。

---

## 9. 验证（不变量）

- **CCA 正确性**：`ClassicalCCA` 的典型相关系数与 `sklearn.cross_decomposition.CCA` 在容差内一致（`tests/test_cca.py` 参照交叉验证）。
- **检索难度梯度**：高 `view_corr` → top1 ≫ 随机基线；低 `view_corr` → 检索不虚高（`test_retrieval.py`）。
- **融合非劣**：ModaFuse 测试集准确率 ≥ 最佳单视图 − 0.05（`test_fusion.py`）。

---

## 10. 许可证与作者

MIT License · **作者：晨星（CJX0712）**
