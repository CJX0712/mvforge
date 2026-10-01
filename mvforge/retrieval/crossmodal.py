"""跨模态最近邻检索: 在共享表征空间中以视图A检索视图B的对应样本."""
from __future__ import annotations

import numpy as np

from ..core.errors import RetrievalError
from ..core.types import RetrievalResult


class CrossModalRetrieval:
    """以查询视图投影检索画廊视图投影的最近邻 (实现 Retriever 契约)."""

    method = "crossmodal"

    def __init__(self, topk: int = 5) -> None:
        self.topk = topk
        self.gallery_: np.ndarray | None = None

    def fit(self, gallery: np.ndarray) -> "CrossModalRetrieval":
        self.gallery_ = np.asarray(gallery, dtype=float)
        return self

    def query(self, queries: np.ndarray) -> np.ndarray:
        if self.gallery_ is None:
            raise RetrievalError("CrossModalRetrieval 尚未 fit")
        Q = np.asarray(queries, dtype=float)
        diff = Q[:, None, :] - self.gallery_[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        return np.argsort(dist, axis=1)  # 升序: 第 0 列=最近

    def evaluate(self, queries: np.ndarray, true_indices: np.ndarray) -> RetrievalResult:
        order = self.query(queries)
        m = order.shape[0]
        k = min(self.topk, order.shape[1])
        top1 = top5 = 0
        ap_sum = 0.0
        for i in range(m):
            if order[i, 0] == true_indices[i]:
                top1 += 1
            if true_indices[i] in order[i, :k]:
                top5 += 1
            pos = np.where(order[i] == true_indices[i])[0]
            if len(pos):
                ap_sum += 1.0 / (pos[0] + 1)
        return RetrievalResult(
            top1_accuracy=top1 / m,
            top5_accuracy=top5 / m,
            map=ap_sum / m,
            method=self.method,
            gallery_size=self.gallery_.shape[0],
        )
