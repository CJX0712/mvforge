"""晚期融合 (Stacking): 各视图独立基模型 -> 堆叠概率 -> 元逻辑回归.

元训练特征用 cross_val_predict 生成, 避免信息泄漏导致的虚高.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import _rf, _logreg


class LateFusionClassifier:
    """晚期融合 (Late Fusion / Stacking)."""

    name = "late_fusion"

    def __init__(self, n_estimators: int = 200) -> None:
        self.n_estimators = n_estimators
        self._bases: List[object] = []
        self._meta = None
        self._classes = None

    def fit(self, views: List[np.ndarray], y: np.ndarray) -> "LateFusionClassifier":
        from sklearn.model_selection import cross_val_predict

        n_class = len(np.unique(y))
        meta_X_parts = []
        for v in views:
            base = _rf(self.n_estimators)
            proba = cross_val_predict(base, np.asarray(v, float), y, cv=3, method="predict_proba")
            self._bases.append(base.fit(np.asarray(v, float), y))
            meta_X_parts.append(proba)
        meta_X = np.hstack(meta_X_parts)
        self._meta = _logreg().fit(meta_X, y)
        self._classes = self._meta.classes_
        return self

    def _stack(self, views: List[np.ndarray]) -> np.ndarray:
        parts = [b.predict_proba(np.asarray(v, float)) for b, v in zip(self._bases, views)]
        return np.hstack(parts)

    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        return self._meta.predict(self._stack(views))

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        return self._meta.predict_proba(self._stack(views))
