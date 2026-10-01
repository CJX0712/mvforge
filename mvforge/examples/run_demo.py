"""端到端演示: 生成多视图数据 -> 表征 -> 跨模态检索 -> 多视图融合 -> 落盘 benchmark.json.

用法:
    python -m mvforge.examples.run_demo
或:
    python -m mvforge.cli --digits --corr 0.3 0.6 0.9
"""
from __future__ import annotations

import json
import os
import sys

# 允许以脚本方式直接运行 (python mvforge/examples/run_demo.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mvforge.core.config import Config
from mvforge.data.synthetic import make_digits_two_view, make_two_view_gaussian
from mvforge.pipeline.pipeline import MVForgePipeline


def main() -> int:
    cfg = Config.from_env()
    datasets = [
        make_two_view_gaussian(n=600, view_corr=0.3, random_state=cfg.random_state),
        make_two_view_gaussian(n=600, view_corr=0.6, random_state=cfg.random_state),
        make_two_view_gaussian(n=600, view_corr=0.9, random_state=cfg.random_state),
        make_digits_two_view(random_state=cfg.random_state),
    ]
    pipe = MVForgePipeline(cfg)
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "benchmark.json")
    payload = pipe.benchmark(datasets, out_path=out)

    print("=" * 78)
    print("MVForge 演示 · 作者: 晨星")
    print("=" * 78)
    hdr = (f"{'dataset':<20}{'canCorr':>10}{'top1':>8}{'map':>8}"
           f"{'sv1':>7}{'sv2':>7}{'early':>7}{'late':>7}{'ccaK':>7}{'moda':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in payload["datasets"]:
        rep = r["representation"]; ret = r["retrieval"]; fus = r.get("fusion", {})
        a = lambda k: (fus.get(k, {}).get("accuracy") if isinstance(fus.get(k), dict) else None)
        print(
            f"{r['dataset']:<20}{rep['total_canonical_correlation']:>10}"
            f"{ret['top1_accuracy']:>8}{ret['map']:>8}"
            f"{(a('single_view_0') or 0):>7}{(a('single_view_1') or 0):>7}"
            f"{(a('early_fusion') or 0):>7}{(a('late_fusion') or 0):>7}"
            f"{(a('cca_knn') or 0):>7}{(a('modafuse') or 0):>7}"
        )
    print("\n摘要:", json.dumps(payload["summary"], ensure_ascii=False))
    print("ModaFuse 选择报告 (首个数据集):")
    sel = payload["datasets"][0].get("modafuse_selection")
    if sel:
        print("  candidate_val_acc:", json.dumps(sel["candidate_val_acc"], ensure_ascii=False))
        print("  selected:", sel["selected"], " ensemble_val_acc:", round(sel["ensemble_val_acc"], 4))
    print(f"\nbenchmark.json 已落盘: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
