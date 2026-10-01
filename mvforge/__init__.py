"""MVForge - 多模态/多视图表征学习与融合系统.

纯 numpy 离线兜底 (ClassicalCCA + CCA-KNN + sklearn 融合) + 可选 mvlearn / torch SOTA 后端.
作者: 晨星 (CJX0712).
"""

__version__ = "1.0.0"
__author__ = "晨星"

from .core.types import (
    MultiViewDataset,
    ProjectionResult,
    FusionResult,
    RetrievalResult,
)

__all__ = [
    "MultiViewDataset",
    "ProjectionResult",
    "FusionResult",
    "RetrievalResult",
    "__version__",
    "__author__",
]
