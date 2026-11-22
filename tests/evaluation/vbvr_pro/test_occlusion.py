import numpy as np

from evaluation.cross_benchmark.vbvr_pro.verification.scene.occlusion import _visible_ink_matches


def test_contour_can_straddle_occlusion_boundary():
    visible = np.zeros((32, 32), np.uint8)
    visible[:15] = 1
    expected = np.zeros_like(visible)
    expected[15, 5:15] = 1
    actual = np.zeros_like(visible)
    actual[14, 5:15] = 1

    assert _visible_ink_matches(actual, expected, visible, 1)

    actual[8, 5:15] = 1
    assert not _visible_ink_matches(actual, expected, visible, 1)
