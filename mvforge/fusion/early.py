"""早期融合: 拼接原始视图特征后训练单一分类器."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import _rf


class EarlyFusionClassifier:
    """早期融合 (Early Fusion): 拼接全部视图原始特征 -> 随机森林."""

    name = "early_fusion"

    def __init__(self, n_estimators: int = 200) -> None:
        self.n_estimators = n_estimators
        self._model = None
        self._classes = None

    def _concat(self, views: List[np.ndarray]) -> np.ndarray:
        return np.hstack([np.asarray(v, float) for v in views])

    def fit(self, views: List[np.ndarray], y: np.ndarray) -> "EarlyFusionClassifier":
        self._model = _rf(self.n_estimators).fit(self._concat(views), y)
        self._classes = self._model.classes_
        return self

    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        return self._model.predict(self._concat(views))

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        return self._model.predict_proba(self._concat(views))
