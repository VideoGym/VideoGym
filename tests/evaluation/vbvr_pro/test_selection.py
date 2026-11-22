import cv2
import numpy as np
import pytest
from evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection import verify

TASK = "G-135_select_next_figure_small_large_alternating_sequence_data-generator"


def sample():
    first = np.full((512, 512, 3), 255, np.uint8)
    for center, radius in [((90, 100), 20), ((250, 100), 40), ((420, 100), 20), ((150, 370), 20), ((370, 370), 40)]:
        cv2.circle(first, center, radius, (0, 0, 0), -1)
    final = first.copy()
    cv2.circle(final, (370, 370), 51, (0, 0, 255), 3)
    return first, final


def check(frames, first, final):
    return verify(TASK, frames, first, final, {}, [])


def test_correct_answer_set_passes():
    a, b = sample()
    assert check([a, b], a, b)["passed"]


def test_wrong_answer_fails():
    a, b = sample()
    wrong = a.copy()
    cv2.circle(wrong, (150, 370), 31, (0, 0, 255), 3)
    result = check([a, wrong], a, b)
    assert not result["passed"]
    assert not result["checks"]["all_targets_matched_once"]


@pytest.mark.parametrize("failure", ["frozen", "erased_object", "corrupt_middle"])
def test_scene_and_completion_are_required(failure):
    first, final = sample()
    candidate = final.copy()
    frames = [first, candidate]
    if failure == "frozen":
        frames = [first, first]
    elif failure == "erased_object":
        cv2.circle(candidate, (90, 100), 25, (255, 255, 255), -1)
    else:
        frames = [first, np.zeros_like(first), final]
    assert not check(frames, first, final)["passed"]


def test_hollow_points_accept_larger_rings_and_reject_extra_selections():
    first = np.full((512, 512, 3), 255, np.uint8)
    points = [
        {"type": "hollow", "position": [95, 135]},
        {"type": "solid", "position": [250, 135]},
        {"type": "hollow", "position": [415, 135]},
    ]
    for point in points:
        thickness = -1 if point["type"] == "solid" else 2
        cv2.circle(first, point["position"], 22, (0, 0, 0), thickness)
    metadata = {"parameters": {"points": points, "canvas": {"width": 512, "height": 512}}}
    final = first.copy()
    larger = first.copy()
    for point in points[::2]:
        cv2.circle(final, point["position"], 27, (0, 0, 255), 3)
        cv2.circle(larger, point["position"], 39, (0, 0, 255), 3)

    def score(frames):
        return verify("G-158_identify_all_hollow_points_data-generator", frames, first, final, metadata, [])

    correct = score([first, larger])
    assert correct["passed"], correct
    extra = larger.copy()
    cv2.circle(extra, points[1]["position"], 39, (0, 0, 255), 3)
    assert not score([first, extra])["checks"]["no_extra_targets"]
    assert not score([first, extra, larger])["checks"]["annotations_stay_on_answer_regions"]


def test_wave_peak_markers_must_appear_left_to_right_one_at_a_time():
    first = np.full((512, 512, 3), 255, np.uint8)
    wave = [[50, 220], [100, 140], [180, 250], [280, 140], [380, 250]]
    cv2.polylines(first, [np.asarray(wave)], False, (0, 0, 0), 2)

    def mark(image, center):
        cv2.circle(image, center, 16, (0, 0, 255), 2)
        cv2.circle(image, center, 3, (0, 0, 255), -1)

    intermediate = first.copy()
    mark(intermediate, (100, 140))
    final = intermediate.copy()
    mark(final, (280, 140))
    metadata = {"parameters": {"wave_points": wave, "canvas": [512, 512]},
                "semantic_ground_truth": {"marker_ring_radius_px": 16}}
    correct = verify("G-202_wave", [first, intermediate, final], first, final, metadata, [])
    assert correct["passed"], correct
    simultaneous = verify("G-202_wave", [first, final], first, final, metadata, [])
    assert not simultaneous["passed"]
    assert not simultaneous["checks"]["peaks_marked_left_to_right_one_at_a_time"]
