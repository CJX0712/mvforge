"""合成多视图数据生成.

make_two_view_gaussian: 由共享隐变量 z 生成两个视图, 通过 view_corr 控制两视图共享信号占比,
形成难度梯度 (低相关 -> 单视图弱, 融合增益大).
make_digits_two_view: 真实 sklearn digits, 上下半各作一个视图 (真实多视图).
"""
from __future__ import annotations

import numpy as np

from ..core.types import MultiViewDataset


def make_two_view_gaussian(
    n: int = 600,
    d_latent: int = 8,
    d_view: int = 20,
    view_corr: float = 0.6,
    label_noise: float = 0.1,
    random_state: int = 42,
) -> MultiViewDataset:
    """由共享隐变量 z 生成两视图二分类数据.

    view_i = sqrt(c) * shared_i + sqrt(1-c) * indep_i, 其中 shared_i = z @ A_i.
    c=view_corr 越大, 两视图共享信号越多, 单视图信息越完整; c 越小, 单视图越弱, 融合增益越大.
    标签由完整隐变量 z 决定, 因此只有融合两视图才能最好地恢复决策边界.
    """
    if not 0.0 < view_corr <= 1.0:
        raise ValueError("view_corr 必须在 (0, 1] 区间")
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal((n, d_latent))
    A1 = rng.standard_normal((d_latent, d_view))
    A2 = rng.standard_normal((d_latent, d_view))
    shared1 = z @ A1
    shared2 = z @ A2
    shared1 = shared1 / (shared1.std(0, keepdims=True) + 1e-8)
    shared2 = shared2 / (shared2.std(0, keepdims=True) + 1e-8)
    indep1 = rng.standard_normal((n, d_view))
    indep2 = rng.standard_normal((n, d_view))
    c = view_corr
    view1 = np.sqrt(c) * shared1 + np.sqrt(max(1e-6, 1.0 - c)) * indep1
    view2 = np.sqrt(c) * shared2 + np.sqrt(max(1e-6, 1.0 - c)) * indep2

    w = rng.standard_normal(d_latent)
    logit = z @ w
    y = (logit > 0).astype(int)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y).astype(int)
    return MultiViewDataset(
        views=[view1, view2],
        labels=y,
        view_names=["view1", "view2"],
        name=f"gaussian_c{view_corr:.2f}",
    )


def make_digits_two_view(random_state: int = 0) -> MultiViewDataset:
    """sklearn digits (8x8) 切分为上下两个视图 (真实多视图, 10 类)."""
    try:
        from sklearn.datasets import load_digits
    except Exception as exc:  # pragma: no cover
        raise ImportError("需要 scikit-learn 才能载入 digits 数据集") from exc
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)
    half = X.shape[1] // 2
    view_top = X[:, :half]
    view_bottom = X[:, half:]
    rng = np.random.default_rng(random_state)
    order = rng.permutation(X.shape[0])
    view_top = view_top[order]
    view_bottom = view_bottom[order]
    y = y[order]
    return MultiViewDataset(
        views=[view_top, view_bottom],
        labels=y,
        view_names=["top_half", "bottom_half"],
        name="digits_two_view",
    )
