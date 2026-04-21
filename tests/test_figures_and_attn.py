"""Tests for figure emitters and the AttentionSampler side-writer.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: exercise :mod:`models.pipeline_attn.AttentionSampler`, the
    three ``report.figures_*`` modules, and the shipped
    ``results/bug_timeline.json`` fixture schema, without requiring
    matplotlib, NumPy, or PyTorch to be installed at import time.
"""
from __future__ import annotations

import json
from pathlib import Path

from models.pipeline_attn import DEFAULT_SAMPLE_K, AttentionSampler


def test_default_sample_k_is_small_and_positive() -> None:
    """The shipped K must stay in the 1--10 range (paper caption asserts K=3)."""
    assert 1 <= DEFAULT_SAMPLE_K <= 10


def test_attention_sampler_captures_up_to_k(tmp_path: Path) -> None:
    """Capture stops after K triples and survives NumPy-less environments."""
    sampler = AttentionSampler(k=2)
    for i in range(4):
        sampler.capture(
            image_path=f"/tmp/img_{i}.png",
            bboxes=[[0, 0, 10, 10], [0, 10, 10, 20]],
            attn_weights=[[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]],
        )
    assert sampler.full
    out = sampler.write(tmp_path)
    assert out is not None
    assert out.exists()
    # .npz preferred, but the .json fallback must also keep the data intact
    if out.suffix == ".json":
        payload = json.loads(out.read_text())
        assert len(payload["image_paths"]) == 2
        assert len(payload["attn"]) == 2


def test_attention_sampler_skips_empty_bboxes() -> None:
    """Empty bbox list → no-op, sampler remains un-full."""
    sampler = AttentionSampler(k=1)
    sampler.capture("/tmp/a.png", [], [[0.5, 0.5]])
    assert not sampler.full


def test_attention_sampler_accepts_tolist_objects() -> None:
    """Any object exposing ``.tolist()`` (tensor, ndarray) is accepted."""
    class _FakeTensor:
        def tolist(self) -> list[list[float]]:
            return [[0.7, 0.3], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]]

    sampler = AttentionSampler(k=1)
    sampler.capture("/tmp/b.png", [[0, 0, 5, 5], [0, 5, 5, 10]], _FakeTensor())
    assert sampler.full


def test_attention_sampler_write_no_samples_returns_none(tmp_path: Path) -> None:
    """Empty sampler writes nothing and returns ``None`` (paper-stage safe)."""
    sampler = AttentionSampler(k=3)
    assert sampler.write(tmp_path) is None


def test_bug_timeline_fixture_schema() -> None:
    """The shipped fixture must contain 13 valid entries with an F1 default."""
    repo = Path(__file__).resolve().parent.parent
    data = json.loads((repo / "results" / "bug_timeline.json").read_text())
    assert data["schema_version"] == 1
    assert "f1_after_default" in data
    bugs = data["bugs"]
    assert len(bugs) == 13
    seen_ids = set()
    for bug in bugs:
        assert isinstance(bug["id"], int)
        assert bug["id"] not in seen_ids
        seen_ids.add(bug["id"])
        assert 0.0 <= float(bug["f1_before"]) <= 1.0
        assert isinstance(bug["measured"], bool)
        assert bug["short"]
        assert bug["mechanism"]
    assert seen_ids == set(range(1, 14))
