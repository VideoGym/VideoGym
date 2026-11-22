"""Binary scoring, dataset identity and complete video decoding."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from evaluation.cross_benchmark.vbvr_pro.verification import engine as core
from evaluation.cross_benchmark.vbvr_pro.media import video as media
from evaluation.cross_benchmark.vbvr_pro.reports import report, summarize


def decision(status, task="G-135", split="Out_of_Domain"):
    return {"status": status, "task": task, "split": split}


def test_unresolved_samples_cannot_be_reported_as_zero_or_dropped():
    summary = summarize([decision("correct"), decision("incorrect"), decision("unverified"), decision("error")])
    assert summary["binary_pass_rate"] is None
    assert summary["binary_pass_rate_bounds"] == [0.25, 0.75]
    assert summary["resolved_binary_pass_rate"] == 0.5
    assert summary["coverage"] == 0.5
    assert not summary["complete"]


def test_missing_predictions_remain_in_denominator():
    summary = summarize([decision("correct"), decision("missing")])
    assert summary["binary_pass_rate"] == 0.5
    assert summary["num_missing"] == 1
    assert not summary["complete"]


def test_split_metrics_use_sample_denominators():
    summary = report([decision("correct"), decision("incorrect"), decision("correct", split="In_Domain")])
    assert summary["by_split"]["Out_of_Domain"]["binary_pass_rate"] == 0.5
    assert summary["by_split"]["In_Domain"]["binary_pass_rate"] == 1
    assert summary["overall"]["binary_pass_rate"] == pytest.approx(2 / 3)


def test_decode_includes_failure_after_frame_200(tmp_path):
    path = tmp_path / "long.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 16, (32, 32))
    assert writer.isOpened()
    for i in range(220):
        writer.write(np.full((32, 32, 3), 255 if i < 210 else 0, np.uint8))
    writer.release()
    decoded = media.decode(path)
    assert len(decoded.frames) == 220
    assert decoded.frames[-1].max() == 0




def test_no_semantic_implementation_abstains(monkeypatch):
    first = np.full((64, 64, 3), 255, np.uint8)
    monkeypatch.setattr(core, "engines", lambda: [])
    result = core.verify_frames("G-unknown", [first, first], first, first, {}, [first])
    assert result["status"] == "unverified"
    assert result["score"] is None


def test_semantic_checks_cannot_be_averaged(monkeypatch):
    first = np.full((64, 64, 3), 255, np.uint8)
    engine = SimpleNamespace(
        __name__="engine",
        verify=lambda *args: {
            "passed": False,
            "checks": {"good_quality": True, "correct_answer": False},
            "evidence": {},
        },
    )
    monkeypatch.setattr(core, "engines", lambda: [engine])
    result = core.verify_frames("G-135", [first, first], first, first, {}, [first])
    assert result["score"] == 0
    assert result["checks"]["correct_answer"] is False


def test_sequence_size_class_allows_in_class_variation():
    first = np.full((512, 512, 3), 255, np.uint8)
    slots = []
    radii = {"small": 13, "medium": 21, "large": 32}
    rgb = [44, 128, 209]
    for index, label in enumerate(["medium", "small", "large", "medium", "small", "large"]):
        x = 53 + 81 * index
        slots.append({"box": [x - 38, 200, x + 38, 312], "shape": {"size_name": label}})
        if index < 5:
            cv2.circle(first, (x, 256), radii[label], rgb[::-1], -1)
    # The repeated given triplet is medium, small, large, so the missing
    # label is large. A radius of 36 still belongs to the large size class.
    metadata = {"parameters": {"slots": slots[:-1], "next_slot": slots[-1],
                               "shape_color_rgb": rgb, "size_cycle": ["medium", "small", "large"]}}
    final = first.copy()
    cv2.circle(final, (x, 256), 32, rgb[::-1], -1)
    for radius, expected in ((36, 1), (23, 0)):
        observed = first.copy()
        cv2.circle(observed, (x, 256), radius, rgb[::-1], -1)
        result = core.verify_frames("G-193_draw_next_sized_shape_data-generator", [first, observed], first, final,
                                    metadata, [first, final])
        assert result["score"] == expected, result
