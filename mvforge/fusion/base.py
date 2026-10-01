"""融合基类与单视图基线预测器."""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.errors import FusionError


def _rf(n_estimators: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=1, random_state=0
    )


def _logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, solver="lbfgs")


class SingleViewClassifier:
    """仅用第 idx 个视图训练的基线预测器 (实现 FusionModel 契约)."""

    name = "single_view"

    def __init__(self, view_index: int = 0, n_estimators: int = 200) -> None:
        self.view_index = view_index
        self.n_estimators = n_estimators
        self._model = None
        self._classes = None

    def fit(self, views: List[np.ndarray], y: np.ndarray) -> "SingleViewClassifier":
        if self.view_index >= len(views):
            raise FusionError(f"view_index={self.view_index} 超出视图数 {len(views)}")
        self._model = _rf(self.n_estimators).fit(views[self.view_index], y)
        self._classes = self._model.classes_
        return self

    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        return self._model.predict(views[self.view_index])

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        return self._model.predict_proba(views[self.view_index])

    @property
    def display_name(self) -> str:
        return f"{self.name}_{self.view_index}"
