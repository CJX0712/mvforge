"""类型定义: 多视图数据集与各种结果载体.

所有跨模块数据结构在此集中定义, 保证接口语义一致:
- 多视图: 同一批样本的多份特征表示 (views), 共享样本数 n.
- 投影结果: 各视图在共享空间中的投影 + 典型相关 (correlations).
- 融合结果: 预测标签 + 概率 + 方法名.
- 检索结果: 跨模态检索的 top1/top5/mAP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MultiViewDataset:
    """同一批样本的多视图特征表示."""

    views: List[np.ndarray]                 # 每个元素 shape=(n, d_i)
    labels: Optional[np.ndarray] = None     # shape=(n,) 或 (n, c)
    view_names: List[str] = field(default_factory=list)
    name: str = "dataset"

    def __post_init__(self) -> None:
        if not self.views:
            raise ValueError("views 不能为空")
        n = self.views[0].shape[0]
        for v in self.views:
            if v.shape[0] != n:
                raise ValueError("所有视图必须共享相同样本数 n")
        if not self.view_names:
            self.view_names = [f"view_{i}" for i in range(len(self.views))]
        if self.labels is not None and self.labels.shape[0] != n:
            raise ValueError("labels 必须与视图样本数对齐")

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def n_samples(self) -> int:
        return self.views[0].shape[0]


@dataclass
class ProjectionResult:
    """表征学习投影结果."""

    projections: List[np.ndarray]   # 每个视图投影后的数组, shape=(n, k)
    correlations: np.ndarray        # 典型相关系数, shape=(k,)
    method: str = "cca"


@dataclass
class FusionResult:
    """融合模型预测结果."""

    predictions: np.ndarray
    proba: Optional[np.ndarray] = None
    method: str = "fusion"
    extra: Optional[Dict[str, object]] = None


@dataclass
class RetrievalResult:
    """跨模态检索结果."""

    top1_accuracy: float
    top5_accuracy: float
    map: float
    method: str = "crossmodal"
    gallery_size: int = 0
