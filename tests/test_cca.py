"""CCA 单元测试: 与 sklearn 参照实现交叉验证 (不变量)."""
import numpy as np
import pytest

from mvforge.represent.cca import CCA, ClassicalCCA


def _sklearn_canonical_correlations(X, Y, k):
    from sklearn.cross_decomposition import CCA as SkCCA

    sk = SkCCA(n_components=k).fit(X, Y)
    xs, ys = sk.transform(X, Y)
    cors = [abs(np.corrcoef(xs[:, i], ys[:, i])[0, 1]) for i in range(k)]
    return np.sort(np.array(cors))


def test_cca_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 6))
    Y = rng.standard_normal((800, 5))
    k = 4
    m = ClassicalCCA(n_components=k, reg=1e-6).fit(X, Y)
    mine = np.sort(m.corr_)
    ref = _sklearn_canonical_correlations(X, Y, k)
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=0.02), f"mine={mine} ref={ref}"


def test_cca_projector_protocol():
    from mvforge.core.types import MultiViewDataset

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5))
    Y = rng.standard_normal((200, 5))
    ds = MultiViewDataset(views=[X, Y], name="t")
    proj = CCA(3, 1e-4).fit(ds).transform(ds)
    assert proj.projections[0].shape == (200, 3)
    assert len(proj.correlations) == 3
    # 投影自洽: 经验相关应与报告的典型相关系数一致
    c = np.corrcoef(proj.projections[0][:, 0], proj.projections[1][:, 0])[0, 1]
    assert abs(abs(c) - proj.correlations[0]) < 0.05


def test_cca_two_view_only():
    from mvforge.core.errors import RepresentationError
    from mvforge.core.types import MultiViewDataset

    ds = MultiViewDataset(views=[np.zeros((10, 2)), np.zeros((10, 2)), np.zeros((10, 2))])
    with pytest.raises(RepresentationError):
        CCA(2).fit(ds)
