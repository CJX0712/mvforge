"""数据集载入与切分工具."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.types import MultiViewDataset


def from_arrays(
    views: List[np.ndarray],
    labels: Optional[np.ndarray] = None,
    view_names: Optional[List[str]] = None,
    name: str = "from_arrays",
) -> MultiViewDataset:
    """由原始数组构造多视图数据集."""
    return MultiViewDataset(
        views=[np.asarray(v, float) for v in views],
        labels=None if labels is None else np.asarray(labels),
        view_names=view_names or [],
        name=name,
    )


def train_test_split_mv(
    dataset: MultiViewDataset,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[MultiViewDataset, MultiViewDataset]:
    """按比例切分多视图数据集 (标签可选)."""
    n = dataset.n_samples
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    train_views = [v[train_idx] for v in dataset.views]
    test_views = [v[test_idx] for v in dataset.views]
    train_labels = None if dataset.labels is None else dataset.labels[train_idx]
    test_labels = None if dataset.labels is None else dataset.labels[test_idx]
    train = MultiViewDataset(
        views=train_views, labels=train_labels,
        view_names=dataset.view_names, name=dataset.name + "_train",
    )
    test = MultiViewDataset(
        views=test_views, labels=test_labels,
        view_names=dataset.view_names, name=dataset.name + "_test",
    )
    return train, test
