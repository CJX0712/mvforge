"""核典型相关分析 (Kernel CCA) - 可选 mvlearn 后端, 不可用时降级 ClassicalCCA.

mvlearn.embed.KMCCA 是顶级开源核 CCA 实现; 若环境无 mvlearn,
本模块自动降级为 ClassicalCCA 并在 available_kcca() 返回 False, 不影响系统可运行性.
"""
from __future__ import annotations

import numpy as np

from ..core.errors import RepresentationError
from ..core.types import MultiViewDataset, ProjectionResult
from .cca import ClassicalCCA


def available_kcca() -> bool:
    try:
        import mvlearn  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


class KMCCA:
    """核 CCA 投影器 (sklearn/mvlearn 后端优先, 纯 numpy ClassicalCCA 兜底)."""

    method = "kernel_cca"

    def __init__(self, n_components: int = 10, gamma: float = 1.0, reg: float = 1e-3) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.reg = reg
        self._backend: str = "classical"
        self._mv = None
        self._model: ClassicalCCA | None = None

    def fit(self, dataset: MultiViewDataset) -> "KMCCA":
        if available_kcca():
            try:
                from mvlearn.embed import KMCCA as _KMCCA

                self._mv = _KMCCA(
                    n_components=self.n_components,
                    kernel="rbf",
                    gamma=self.gamma,
                    reg=self.reg,
                )
                views = [np.asarray(v, float) for v in dataset.views[:2]]
                self._mv.fit(views)
                self._backend = "mvlearn"
                return self
            except Exception:  # pragma: no cover
                pass
        # 降级
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
        if self._backend == "mvlearn" and self._mv is not None:
            arr = np.asarray(X, float)
            out = self._mv.transform([arr] if view == 0 else [np.zeros_like(arr), arr])
            return out[view]
        assert self._model is not None
        return self._model.project_view(X, view)
