"""深度典型相关分析 (Deep CCA) - 可选 mvlearn+torch 后端, 不可用时降级 ClassicalCCA.

DCCA 通过深度网络学习非线性共享表征, 是顶级多视图表征学习方法.
无 torch/mvlearn 时自动降级, available_dcca() 返回 False.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import RepresentationError
from ..core.types import MultiViewDataset, ProjectionResult
from .cca import ClassicalCCA


def available_dcca() -> bool:
    try:
        import torch  # noqa: F401
        import mvlearn  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


class DeepCCA:
    """深度 CCA 投影器 (torch+mvlearn 后端优先, 纯 numpy ClassicalCCA 兜底)."""

    method = "deep_cca"

    def __init__(self, n_components: int = 10, reg: float = 1e-3) -> None:
        self.n_components = n_components
        self.reg = reg
        self._backend: str = "classical"
        self._mv = None
        self._model: ClassicalCCA | None = None

    def fit(self, dataset: MultiViewDataset) -> "DeepCCA":
        if available_dcca():
            try:
                from mvlearn.embed import DCCA

                self._mv = DCCA(
                    layer_sizes1=[dataset.views[0].shape[1], 64, self.n_components],
                    layer_sizes2=[dataset.views[1].shape[1], 64, self.n_components],
                    reg=self.reg,
                    epochs=20,
                )
                views = [np.asarray(v, float) for v in dataset.views[:2]]
                self._mv.fit(views)
                self._backend = "mvlearn"
                return self
            except Exception:  # pragma: no cover
                pass
        self._model = ClassicalCCA(self.n_components, max(self.reg, 1e-4)).fit(
            dataset.views[0], dataset.views[1]
        )
        self._backend = "classical"
        return self

    def transform(self, dataset: MultiViewDataset) -> ProjectionResult:
        if self._backend == "mvlearn" and self._mv is not None:
            views = [np.asarray(v, float) for v in dataset.views[:2]]
            proj = self._mv.transform(views)
            corr = np.asarray(getattr(self._mv, "canon_corrs_", np.zeros(self.n_components)))
            return ProjectionResult(projections=proj, correlations=corr, method=self.method)
        assert self._model is not None
        Ux, Uy = self._model.transform(dataset.views[0], dataset.views[1])
        return ProjectionResult(
            projections=[Ux, Uy], correlations=self._model.corr_, method="classical_cca"
        )

    def project_view(self, X: np.ndarray, view: int = 0) -> np.ndarray:
        assert self._model is not None
        return self._model.project_view(X, view)
