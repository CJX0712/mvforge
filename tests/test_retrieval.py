"""跨模态检索单元测试: 高相关下 top1 应远优于随机基线."""
import numpy as np

from mvforge.represent.cca import CCA
from mvforge.retrieval.crossmodal import CrossModalRetrieval


def test_retrieval_high_corr():
    from mvforge.core.types import MultiViewDataset
    from mvforge.data.synthetic import make_two_view_gaussian

    ds = make_two_view_gaussian(n=300, view_corr=0.95, random_state=2)
    cca = CCA(10, 1e-4).fit(ds)
    u1 = cca.project_view(ds.views[0], 0)
    u2 = cca.project_view(ds.views[1], 1)
    retr = CrossModalRetrieval(topk=5).fit(u2)
    res = retr.evaluate(u1, np.arange(ds.n_samples))
    # 随机基线 top1 = 1/n ≈ 0.003; 高相关应 >> 此
    assert res.top1_accuracy > 0.4
    assert res.map > 0.4


def test_retrieval_low_corr_weak():
    from mvforge.data.synthetic import make_two_view_gaussian

    ds = make_two_view_gaussian(n=300, view_corr=0.15, random_state=3)
    cca = CCA(10, 1e-4).fit(ds)
    u1 = cca.project_view(ds.views[0], 0)
    u2 = cca.project_view(ds.views[1], 1)
    retr = CrossModalRetrieval(topk=5).fit(u2)
    res = retr.evaluate(u1, np.arange(ds.n_samples))
    # 低相关下检索不应虚高 (难度梯度诚实)
    assert res.top1_accuracy < 0.5
