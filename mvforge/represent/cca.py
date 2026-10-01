"""经典典型相关分析 (Classical CCA), 纯 numpy 实现 (离线兜底).

数学核心: 对中心化后的 X, Y, 白化交叉协方差矩阵
    M = Cxx^{-1/2} Cxy Cyy^{-1/2}
其奇异值即为典型相关系数, 左/右奇异向量经白化矩阵映射回各视图权重.
不变量: 与 sklearn.cross_decomposition.CCA 的典型相关系数在容差内一致
(已在 tests/test_cca.py 中用参照实现交叉验证).
"""
from __future__ import annotations

import numpy as np

from ..core.errors import RepresentationError
from ..core.types import MultiViewDataset, ProjectionResult


class ClassicalCCA:
    """双视图经典 CCA (纯 numpy)."""

    def __init__(self, n_components: int = 10, reg: float = 1e-4) -> None:
        self.n_components = int(n_components)
        self.reg = float(reg)
        self.mean_x_: np.ndarray | None = None
        self.mean_y_: np.ndarray | None = None
        self.wx_: np.ndarray | None = None
        self.wy_: np.ndarray | None = None
        self.corr_: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ClassicalCCA":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape[0] != Y.shape[0]:
            raise RepresentationError("X 与 Y 样本数不一致")
        n = X.shape[0]
        self.mean_x_ = X.mean(axis=0)
        self.mean_y_ = Y.mean(axis=0)
        Xc = X - self.mean_x_
        Yc = Y - self.mean_y_
        Cxx = (Xc.T @ Xc) / n + self.reg * np.eye(Xc.shape[1])
        Cyy = (Yc.T @ Yc) / n + self.reg * np.eye(Yc.shape[1])
        Cxy = (Xc.T @ Yc) / n

        Ux, Sx, _ = np.linalg.svd(Cxx, hermitian=True)
        Uy, Sy, _ = np.linalg.svd(Cyy, hermitian=True)
        Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(np.maximum(Sx, 1e-12))) @ Ux.T
        Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(np.maximum(Sy, 1e-12))) @ Uy.T

        M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        k = min(self.n_components, U.shape[1], Vt.shape[0])
        self.wx_ = Cxx_inv_sqrt @ U[:, :k]
        self.wy_ = Cyy_inv_sqrt @ Vt[:k, :].T
        self.corr_ = S[:k]
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray | None = None):
        Xc = np.asarray(X, dtype=float) - self.mean_x_
        Ux = Xc @ self.wx_
        if Y is None:
            return Ux
        Yc = np.asarray(Y, dtype=float) - self.mean_y_
        Uy = Yc @ self.wy_
        return Ux, Uy

    def project_view(self, X: np.ndarray, view: int = 0) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - (self.mean_x_ if view == 0 else self.mean_y_)
        w = self.wx_ if view == 0 else self.wy_
        return Xc @ w


class CCA:
    """双视图 CCA 投影器 (实现 Projector 契约). 离线纯 numpy."""

    method = "classical_cca"

    def __init__(self, n_components: int = 10, reg: float = 1e-4) -> None:
        self.n_components = n_components
        self.reg = reg
        self._model: ClassicalCCA | None = None

    def fit(self, dataset: MultiViewDataset) -> "CCA":
        if dataset.n_views != 2:
            raise RepresentationError("ClassicalCCA 仅支持双视图; 多视图请用 mvlearn 后端")
        self._model = ClassicalCCA(self.n_components, self.reg).fit(
            dataset.views[0], dataset.views[1]
        )
        return self

    def transform(self, dataset: MultiViewDataset) -> ProjectionResult:
        if dataset.n_views != 2:
            raise RepresentationError("ClassicalCCA 仅支持双视图")
        assert self._model is not None
        Ux, Uy = self._model.transform(dataset.views[0], dataset.views[1])
        return ProjectionResult(
            projections=[Ux, Uy],
            correlations=self._model.corr_,
            method=self.method,
        )

    def project_view(self, X: np.ndarray, view: int = 0) -> np.ndarray:
        assert self._model is not None
        return self._model.project_view(X, view)
