# MVForge 架构设计文档

## 1. 领域定位

MVForge 解决**多视图 / 多模态学习**三类核心问题：

1. **表征对齐**：把同一批样本的两个（或多个）视图投影到共享语义空间，使对应样本靠近。
2. **跨模态检索**：在共享空间内以视图 A 检索视图 B 的对应样本。
3. **融合预测**：综合多视图信息提升分类/识别性能。

选择该域的理由：因果发现（causaldiscforge）、可解释性（explainforge）等相邻域已交付，而多视图表征学习尚未覆盖；且 CCA 家族有顶级开源 `mvlearn` 可复用，纯 numpy 离线兜底真实可行。

## 2. 调用单向无环

```
cli.py ──▶ pipeline.MVForgePipeline
                │
                ├─▶ data (synthetic / loaders)
                ├─▶ represent (cca / kcca / dcca)   ──▶ core
                ├─▶ fusion (base / early / late / cca_knn / modafuse)
                ├─▶ retrieval (crossmodal)
                └─▶ eval (metrics)
                                │
                                └─▶ core (types / errors / config)
```

依赖只向下流动，无环。

## 3. 核心算法

### 3.1 ClassicalCCA（纯 numpy，离线兜底）

中心化后构造白化交叉协方差：

```
M = Cxx^{-1/2} · Cxy · Cyy^{-1/2}
```

对 M 做 SVD，奇异值即**典型相关系数**，左右奇异向量经白化矩阵映射回各视图权重 `wx`, `wy`。投影：`Ux = (X − μx)·wx`, `Uy = (Y − μy)·wy`。

**不变量**：与 `sklearn.cross_decomposition.CCA` 的典型相关系数在 `atol=0.02` 内一致（已交叉验证）。

### 3.2 CrossModalRetrieval

以视图 A 投影为查询、视图 B 投影为画廊，欧氏距离最近邻排序；top-1 / top-5 命中率与 mAP 作为对齐质量度量。

### 3.3 融合模型

| 模型 | 方法 |
|------|------|
| SingleView | 仅用单视图训练 RF（基线） |
| EarlyFusion | 拼接原始视图 → RF |
| LateFusion | 各视图独立 RF → `cross_val_predict` 概率 → 元逻辑回归（防泄漏） |
| CCAKNN | 两视图 CCA 投影拼接 → KNN |
| **ModaFuse** | 候选池贪心前向选择 + 非劣守护 |

### 3.4 ModaFuse 旗舰

```
候选池 = {sv0, sv1, early, late, cca_knn}
best_single = argmax 单视图验证准确率
贪心: 每次加入使集成验证准确率提升 > tol 的候选
守护: 若 集成 < best_single − tol → 回退 best_single
最终: 在完整训练集重训被选中的模型, 预测 = 选中模型概率均值
```

## 4. 离线兜底矩阵

| SOTA 后端 | 降级实现 | 探测函数 |
|-----------|----------|----------|
| `mvlearn.KMCCA` | `ClassicalCCA`（numpy） | `available_kcca()` |
| `mvlearn.DCCA`（torch） | `ClassicalCCA`（numpy） | `available_dcca()` |

无 GPU / 无网络 / 无 `mvlearn` 时，系统零重型依赖可完整运行。

## 5. 难度梯度设计（防数据泄漏与虚高）

`make_two_view_gaussian` 通过 `view_corr ∈ (0,1]` 控制两视图共享信号占比：

```
view_i = sqrt(c)·shared_i + sqrt(1−c)·indep_i
```

- `c` 小 → 单视图弱、融合增益大；`c` 大 → 单视图已较完整。
- 标签由完整隐变量 `z` 决定，只有融合两视图才能最好恢复决策边界。
- 检索：`c` 高 → top1 ≫ 随机；`c` 低 → 检索不虚高（诚实）。

## 6. 基准与复现

```bash
python -m mvforge.examples.run_demo   # 落盘 benchmark.json (固定 random_state)
pytest -q -W ignore::UserWarning        # 全绿
```

量化基线（典型相关系数、检索 top1/mAP、各融合方法准确率）见 `benchmark.json`，固定 `random_state=42` 可复现。
