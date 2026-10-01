"""评测指标: 分类准确率 / 跨模态检索 topk / mAP / 典型相关总和 / macro-F1."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def topk_retrieval_accuracy(order: np.ndarray, true_indices: np.ndarray, k: int) -> float:
    order = np.asarray(order)
    true_indices = np.asarray(true_indices)
    m = order.shape[0]
    kk = min(k, order.shape[1])
    hits = 0
    for i in range(m):
        if true_indices[i] in order[i, :kk]:
            hits += 1
    return hits / m


def mean_average_precision(order: np.ndarray, true_indices: np.ndarray) -> float:
    order = np.asarray(order)
    true_indices = np.asarray(true_indices)
    m = order.shape[0]
    s = 0.0
    for i in range(m):
        pos = np.where(order[i] == true_indices[i])[0]
        if len(pos):
            s += 1.0 / (pos[0] + 1)
    return s / m


def total_canonical_correlation(corr: np.ndarray) -> float:
    return float(np.sum(np.asarray(corr, dtype=float)))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:  # pragma: no cover
        return 0.0
