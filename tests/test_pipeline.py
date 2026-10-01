"""管线与基准单元测试."""
import json
import os

from mvforge.core.config import Config
from mvforge.data.synthetic import make_two_view_gaussian
from mvforge.pipeline.pipeline import MVForgePipeline


def test_pipeline_run_full_keys():
    ds = make_two_view_gaussian(n=300, view_corr=0.6, random_state=3)
    pipe = MVForgePipeline(Config(random_state=3))
    rep = pipe.run_full(ds)
    assert "representation" in rep
    assert "retrieval" in rep
    assert "fusion" in rep
    assert rep["fusion"]["modafuse"]["accuracy"] is not None


def test_benchmark_writes_json(tmp_path):
    ds = make_two_view_gaussian(n=240, view_corr=0.6, random_state=4)
    pipe = MVForgePipeline(Config(random_state=4))
    out = os.path.join(tmp_path, "benchmark.json")
    payload = pipe.benchmark([ds], out_path=out)
    assert os.path.exists(out)
    assert payload["system"] == "MVForge"
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
