"""MVForge 端到端管线.

run_full: 表征(CCA) -> 跨模态检索 -> 多视图融合预测, 一站式产出评测报告.
benchmark: 跨多个数据集跑 run_full, 聚合指标并落盘 benchmark.json.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..core.config import Config
from ..core.types import MultiViewDataset
from ..eval.metrics import (
    accuracy,
    macro_f1,
    total_canonical_correlation,
)
from ..fusion.base import SingleViewClassifier
from ..fusion.cca_knn import CCAKNNClassifier
from ..fusion.early import EarlyFusionClassifier
from ..fusion.late import LateFusionClassifier
from ..fusion.modafuse import ModaFuse
from ..represent.cca import CCA
from ..represent.kcca import available_kcca
from ..retrieval.crossmodal import CrossModalRetrieval


class MVForgePipeline:
    """多模态/多视图表征学习与融合管线."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.from_env()

    def run_full(self, dataset: MultiViewDataset, test_size: float = 0.3) -> Dict:
        rng = np.random.default_rng(self.config.random_state)
        n = dataset.n_samples
        idx = rng.permutation(n)
        k = max(1, int(round(n * test_size)))
        test_idx = idx[:k]
        train_idx = idx[k:]
        train_views = [v[train_idx] for v in dataset.views]
        test_views = [v[test_idx] for v in dataset.views]
        has_labels = dataset.labels is not None
        train_labels = dataset.labels[train_idx] if has_labels else None
        test_labels = dataset.labels[test_idx] if has_labels else None

        # ---- 表征: 双视图 CCA ----
        cca = CCA(n_components=self.config.cca_components, reg=self.config.cca_reg)
        cca.fit(MultiViewDataset(views=train_views, name=dataset.name))
        u1_all = cca.project_view(dataset.views[0], 0)
        u2_all = cca.project_view(dataset.views[1], 1)
        proj_train = cca.transform(MultiViewDataset(views=train_views, name=dataset.name))
        total_corr = total_canonical_correlation(proj_train.correlations)

        report: Dict = {
            "dataset": dataset.name,
            "n_samples": n,
            "n_views": dataset.n_views,
            "representation": {
                "method": "classical_cca",
                "n_components": self.config.cca_components,
                "total_canonical_correlation": round(total_corr, 4),
                "mvlearn_kcca_available": available_kcca(),
            },
        }

        # ---- 跨模态检索: 以视图1检索视图2 ----
        retr = CrossModalRetrieval(topk=self.config.retrieval_topk)
        retr.fit(u2_all)
        res = retr.evaluate(u1_all[test_idx], test_idx)
        report["retrieval"] = {
            "method": res.method,
            "top1_accuracy": round(res.top1_accuracy, 4),
            "top5_accuracy": round(res.top5_accuracy, 4),
            "map": round(res.map, 4),
            "gallery_size": res.gallery_size,
        }

        # ---- 多视图融合预测 ----
        if has_labels:
            report["fusion"] = self._evaluate_fusion(
                train_views, train_labels, test_views, test_labels
            )
            report["modafuse_selection"] = report["fusion"].get("modafuse_report")
        return report

    def _evaluate_fusion(self, train_views, train_labels, test_views, test_labels):
        methods = {
            "single_view_0": SingleViewClassifier(0, self.config.fusion_n_estimators),
            "single_view_1": SingleViewClassifier(1, self.config.fusion_n_estimators),
            "early_fusion": EarlyFusionClassifier(self.config.fusion_n_estimators),
            "late_fusion": LateFusionClassifier(self.config.fusion_n_estimators),
            "cca_knn": CCAKNNClassifier(self.config.cca_components),
            "modafuse": ModaFuse(
                n_estimators=self.config.fusion_n_estimators,
                n_components=self.config.cca_components,
                tol=self.config.non_inferiority_tol,
                random_state=self.config.random_state,
            ),
        }
        fusion: Dict = {}
        for name, model in methods.items():
            try:
                model.fit(train_views, train_labels)
                pred = model.predict(test_views)
                fusion[name] = {
                    "accuracy": round(accuracy(test_labels, pred), 4),
                    "macro_f1": round(macro_f1(test_labels, pred), 4),
                }
                if name == "modafuse":
                    fusion["modafuse_report"] = model.selection_report_
            except Exception as exc:  # pragma: no cover
                fusion[name] = {"accuracy": None, "error": str(exc)}
        return fusion

    def benchmark(
        self, datasets: List[MultiViewDataset], out_path: str = "benchmark.json"
    ) -> Dict:
        rows = [self.run_full(ds) for ds in datasets]
        summary = self._summarize(rows)
        payload = {
            "system": "MVForge",
            "version": "1.0.0",
            "author": "晨星",
            "config": {
                "cca_components": self.config.cca_components,
                "retrieval_topk": self.config.retrieval_topk,
                "non_inferiority_tol": self.config.non_inferiority_tol,
            },
            "datasets": rows,
            "summary": summary,
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    @staticmethod
    def _summarize(rows: List[Dict]) -> Dict:
        def mean_of(key_path):
            vals = []
            for r in rows:
                node = r
                try:
                    for k in key_path:
                        node = node[k]
                    if isinstance(node, (int, float)):
                        vals.append(node)
                except Exception:
                    pass
            return round(float(np.mean(vals)), 4) if vals else None

        return {
            "avg_total_canonical_correlation": mean_of(
                ["representation", "total_canonical_correlation"]
            ),
            "avg_retrieval_top1": mean_of(["retrieval", "top1_accuracy"]),
            "avg_retrieval_map": mean_of(["retrieval", "map"]),
            "avg_modafuse_accuracy": mean_of(["fusion", "modafuse", "accuracy"]),
            "avg_early_fusion_accuracy": mean_of(["fusion", "early_fusion", "accuracy"]),
            "avg_best_single_accuracy": MVForgePipeline._avg_best_single(rows),
        }

    @staticmethod
    def _avg_best_single(rows: List[Dict]) -> Optional[float]:
        vals = []
        for r in rows:
            fus = r.get("fusion", {})
            s0 = fus.get("single_view_0", {}).get("accuracy")
            s1 = fus.get("single_view_1", {}).get("accuracy")
            if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
                vals.append(max(s0, s1))
        return round(float(np.mean(vals)), 4) if vals else None
