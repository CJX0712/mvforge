"""融合模型单元测试: 基础可行性 + ModaFuse 非劣守护."""
import numpy as np

from mvforge.core.config import Config
from mvforge.data.loaders import train_test_split_mv
from mvforge.data.synthetic import make_two_view_gaussian
from mvforge.eval.metrics import accuracy
from mvforge.fusion.base import SingleViewClassifier
from mvforge.fusion.early import EarlyFusionClassifier
from mvforge.fusion.late import LateFusionClassifier
from mvforge.fusion.modafuse import ModaFuse


def _data(corr=0.5, seed=1):
    ds = make_two_view_gaussian(n=400, view_corr=corr, random_state=seed)
    return train_test_split_mv(ds, 0.3, seed)


def test_single_and_early_fusion_work():
    train, test = _data()
    for m in [
        SingleViewClassifier(0, 100),
        SingleViewClassifier(1, 100),
        EarlyFusionClassifier(100),
    ]:
        m.fit(train.views, train.labels)
        acc = accuracy(test.labels, m.predict(test.views))
        assert acc > 0.6


def test_late_fusion_runs():
    train, test = _data()
    m = LateFusionClassifier(100).fit(train.views, train.labels)
    acc = accuracy(test.labels, m.predict(test.views))
    assert acc > 0.6


def test_modafuse_non_inferiority():
    train, test = _data(corr=0.5)
    mf = ModaFuse(n_estimators=100, random_state=1).fit(train.views, train.labels)
    acc = accuracy(test.labels, mf.predict(test.views))
    sv = SingleViewClassifier(0, 100).fit(train.views, train.labels)
    sv_acc = accuracy(test.labels, sv.predict(test.views))
    # 非劣守护: ModaFuse 不应系统性劣于单视图基线
    assert acc >= sv_acc - 0.05
    assert len(mf._selected) >= 1


def test_modafuse_selection_present():
    train, test = _data()
    mf = ModaFuse(n_estimators=100, random_state=2).fit(train.views, train.labels)
    rep = mf.selection_report_
    assert "selected" in rep and "best_single" in rep
