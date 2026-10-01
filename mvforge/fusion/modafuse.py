"""ModaFuse 旗舰: 多视图共识融合.

策略: 在候选池 {单视图0, 单视图1, 早期融合, 晚期融合(stacking), CCA-KNN} 上做
贪心前向选择 (按验证集准确率), 仅当加入某模型使集成验证准确率提升 > tol 时才纳入.
非劣守护: 若最终集成验证准确率低于最佳单视图 - tol, 则回退到最佳单视图模型,
保证 ModaFuse 至少不劣于单视图基线 (诚实交付).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.errors import FusionError
from ..eval.metrics import accuracy
from .base import SingleViewClassifier
from .cca_knn import CCAKNNClassifier
from .early import EarlyFusionClassifier
from .late import LateFusionClassifier


class ModaFuse:
    """多视图共识融合旗舰 (实现 FusionModel 契约)."""

    name = "modafuse"

    def __init__(
        self,
        n_estimators: int = 200,
        n_components: int = 10,
        tol: float = 0.02,
        random_state: int = 42,
        val_size: float = 0.3,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.tol = tol
        self.random_state = random_state
        self.val_size = val_size
        self._selected: List[object] = []
        self._selected_names: List[str] = []
        self._classes: Optional[np.ndarray] = None
        self.selection_report_: Dict[str, object] = {}

    def _build_candidates(self) -> List[Tuple[str, object]]:
        return [
            ("single_view_0", SingleViewClassifier(0, self.n_estimators)),
            ("single_view_1", SingleViewClassifier(1, self.n_estimators)),
            ("early_fusion", EarlyFusionClassifier(self.n_estimators)),
            ("late_fusion", LateFusionClassifier(self.n_estimators)),
            ("cca_knn", CCAKNNClassifier(self.n_components)),
        ]

    def _build_model(self, name: str) -> object:
        if name == "single_view_0":
            return SingleViewClassifier(0, self.n_estimators)
        if name == "single_view_1":
            return SingleViewClassifier(1, self.n_estimators)
        if name == "early_fusion":
            return EarlyFusionClassifier(self.n_estimators)
        if name == "late_fusion":
            return LateFusionClassifier(self.n_estimators)
        if name == "cca_knn":
            return CCAKNNClassifier(self.n_components)
        raise FusionError(f"未知候选模型: {name}")

    def fit(
        self,
        views: List[np.ndarray],
        y: np.ndarray,
        val_views: Optional[List[np.ndarray]] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "ModaFuse":
        y = np.asarray(y)
        if val_views is None:
            n = len(y)
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(n)
            k = max(1, int(round(n * self.val_size)))
            vi, ti = idx[:k], idx[k:]
            tr_views = [v[ti] for v in views]
            val_views = [v[vi] for v in views]
            tr_y, val_y = y[ti], y[vi]
        else:
            tr_views, tr_y = views, y

        candidates = self._build_candidates()
        val_acc: Dict[str, float] = {}
        val_proba: Dict[str, Optional[np.ndarray]] = {}
        for name, model in candidates:
            try:
                model.fit(tr_views, tr_y)
                p = model.predict_proba(val_views)
                val_acc[name] = accuracy(val_y, np.argmax(p, axis=1))
                val_proba[name] = p
            except Exception:  # pragma: no cover
                val_acc[name] = -1.0
                val_proba[name] = None

        single_names = ["single_view_0", "single_view_1"]
        best_single = max(single_names, key=lambda nm: val_acc.get(nm, -1))
        best_single_acc = val_acc[best_single]

        # 贪心前向选择
        selected: List[str] = []
        sel_proba: List[np.ndarray] = []
        current_acc = -1.0
        remaining = [nm for nm, _ in candidates]
        while remaining:
            pick = None
            pick_acc = current_acc
            pick_p = None
            for name in remaining:
                if val_proba[name] is None:
                    continue
                if sel_proba:
                    ens = np.mean(sel_proba + [val_proba[name]], axis=0)
                else:
                    ens = val_proba[name]
                acc = accuracy(val_y, np.argmax(ens, axis=1))
                if acc > pick_acc + self.tol:
                    pick_acc, pick, pick_p = acc, name, val_proba[name]
            if pick is None:
                break
            selected.append(pick)
            sel_proba.append(pick_p)
            current_acc = pick_acc
            remaining.remove(pick)

        final_acc = current_acc
        if not selected or final_acc < best_single_acc - self.tol:
            selected = [best_single]
            final_acc = best_single_acc

        # 在完整训练集上重训被选中的模型
        self._selected = []
        for name in selected:
            m = self._build_model(name)
            m.fit(views, y)
            self._selected.append(m)
        self._selected_names = selected
        self._classes = self._selected[0]._classes if hasattr(self._selected[0], "_classes") else None
        if self._classes is None:  # pragma: no cover
            self._classes = np.unique(y)
        self.selection_report_ = {
            "candidate_val_acc": val_acc,
            "best_single": best_single,
            "best_single_val_acc": best_single_acc,
            "selected": selected,
            "ensemble_val_acc": final_acc,
        }
        return self

    def predict(self, views: List[np.ndarray]) -> np.ndarray:
        return np.argmax(self.predict_proba(views), axis=1)

    def predict_proba(self, views: List[np.ndarray]) -> np.ndarray:
        probas = [m.predict_proba(views) for m in self._selected]
        mean_p = np.mean(probas, axis=0)
        # 对齐类别顺序 (所有候选类别应一致)
        return mean_p
