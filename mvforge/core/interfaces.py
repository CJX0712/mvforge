"""接口契约 (Protocol), 调用单向无环: cli -> pipeline -> {data, represent, fusion, retrieval, eval} -> core."""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

from .types import (
    FusionResult,
    MultiViewDataset,
    ProjectionResult,
    RetrievalResult,
)


@runtime_checkable
class Projector(Protocol):
    """把多视图数据投影到共享表征空间."""

    def fit(self, dataset: MultiViewDataset) -> "Projector": ...

    def transform(self, dataset: MultiViewDataset) -> ProjectionResult: ...

    def project_view(self, X: np.ndarray, view: int) -> np.ndarray: ...


@runtime_checkable
class FusionModel(Protocol):
    """融合多视图特征做预测."""

    name: str

    def fit(self, views: List[np.ndarray], y: np.ndarray) -> "FusionModel": ...

    def predict(self, views: List[np.ndarray]) -> np.ndarray: ...

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray: ...


@runtime_checkable
class Retriever(Protocol):
    """在共享空间中做跨模态最近邻检索."""

    def fit(self, gallery: np.ndarray) -> "Retriever": ...

    def evaluate(self, queries: np.ndarray, true_indices: np.ndarray) -> RetrievalResult: ...
