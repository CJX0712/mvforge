"""MVForge 命令行入口 (argparse)."""
from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .data.synthetic import make_digits_two_view, make_two_view_gaussian
from .pipeline.pipeline import MVForgePipeline


def _build_datasets(args) -> list:
    datasets = []
    if args.digits:
        datasets.append(make_digits_two_view(random_state=args.random_state))
    for c in args.corr:
        datasets.append(
            make_two_view_gaussian(
                n=args.n,
                view_corr=c,
                random_state=args.random_state,
            )
        )
    if not datasets:
        datasets.append(make_two_view_gaussian(random_state=args.random_state))
    return datasets


def _print_table(payload: dict) -> None:
    hdr = f"{'dataset':<20}{'canCorr':>10}{'top1':>8}{'map':>8}{'sv1':>7}{'sv2':>7}{'early':>7}{'late':>7}{'ccaK':>7}{'moda':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in payload["datasets"]:
        rep = r["representation"]
        ret = r["retrieval"]
        fus = r.get("fusion", {})
        a = lambda k: (fus.get(k, {}).get("accuracy") if isinstance(fus.get(k), dict) else None)
        print(
            f"{r['dataset']:<20}"
            f"{rep['total_canonical_correlation']:>10}"
            f"{ret['top1_accuracy']:>8}"
            f"{ret['map']:>8}"
            f"{(a('single_view_0') or 0):>7}"
            f"{(a('single_view_1') or 0):>7}"
            f"{(a('early_fusion') or 0):>7}"
            f"{(a('late_fusion') or 0):>7}"
            f"{(a('cca_knn') or 0):>7}"
            f"{(a('modafuse') or 0):>7}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MVForge - 多模态/多视图表征学习与融合")
    p.add_argument("--n", type=int, default=600, help="合成样本数")
    p.add_argument("--corr", type=float, nargs="*", default=[0.3, 0.6, 0.9],
                   help="合成数据视图相关度列表")
    p.add_argument("--digits", action="store_true", help="加入 digits 真实多视图数据集")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out", default="benchmark.json", help="benchmark 落盘路径")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    cfg.random_state = args.random_state
    datasets = _build_datasets(args)
    pipe = MVForgePipeline(cfg)
    payload = pipe.benchmark(datasets, out_path=args.out)
    if not args.quiet:
        _print_table(payload)
        print("\n摘要:", json.dumps(payload["summary"], ensure_ascii=False))
        print(f"已落盘: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
