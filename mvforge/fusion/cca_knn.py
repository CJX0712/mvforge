"""CCA-KNN: 把两视图投影到 CCA 共享空间后拼接, 用 KNN 分类 (利用跨视图对齐信息)."""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.errors import FusionError
from ..represent.cca import ClassicalCCA


class CCAKNNClassifier:
    """CCA 空间 KNN 融合分类器 (离线纯 numpy CCA + sklearn KNN)."""

    name = "cca_knn"

    def __init__(self, n_components: int = 10, n_neighbors: int = 5) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self._cca: ClassicalCCA | None = None
        self._knn = None
        self._classes = None

    def fit(self, views: List[np.ndarray], y: np.ndarray) -> "CCAKNNClassifier":
        if len(views) != 2:
            raise FusionError("CCAKNN 仅支持双视图")
        self._cca = ClassicalCCA(self.n_components, reg=1e-4).fit(views[0], views[1])
        Ux, Uy = self._cca.transform(views[0], views[1])
        feat = np.hstack([Ux, Uy])
        from sklearn.neighbors import KNeighborsClassifier

        self._knn = KNeighborsClassifier(n_neighbors=self.n_neighbors).fit(feat, y)
        self._classes = self._knn.classes_
        return self

    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        Ux = self._cca.project_view(views[0], 0)
        Uy = self._cca.project_view(views[1], 1)
        feat = np.hstack([Ux, Uy])
        return self._knn.predict(feat)

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        Ux = self._cca.project_view(views[0], 0)
        Uy = self._cca.project_view(views[1], 1)
        feat = np.hstack([Ux, Uy])
        return self._knn.predict_proba(feat)
