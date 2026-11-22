"""Select the two labeled points with the smallest Euclidean distance."""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from data.geometry.core.point_target_base import (
    Point,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
    PointCandidate,
)


class ClosestPairGenerator(PointTargetPuzzleGenerator):
    """Find the unique closest pair among native labeled markers."""

    DEFAULT_RULE_PROMPT = (
        "A white canvas shows five labeled candidate circles A-E placed at general positions. "
        "Identify the two labeled points that form the closest pair, defined as the two markers whose straight-line distance is the smallest among all pairs of the five. "
        "Answer with the two point labels in alphabetical order."
    )

    DEFAULT_OUTPUT_DIR = "outputs/geometry/closest_pair"
    DEFAULT_TI2V_PROMPT = (
        "A square white canvas shows five labeled candidate markers A-E placed at general positions, with no other geometry. "
        "Each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the five markers, then draws one black solid segment of medium thickness between the two closest markers. "
        "In the final state, both endpoints of that segment change to pale red fill with dark red outlines, while the other three markers remain white. In portrait. Static camera."
    )
    DEFAULT_SOLUTION_IMAGE_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(self, *args, point_count=5, **kwargs):
        self.point_count = int(point_count)
        if self.point_count != 5:
            labels = tuple(chr(65+i) for i in range(self.point_count))
            kwargs["option_labels"] = labels
            for field in ("DEFAULT_RULE_PROMPT", "DEFAULT_TI2V_PROMPT", "DEFAULT_SOLUTION_IMAGE_PROMPT"):
                prompt = getattr(self, field).replace("five", str(self.point_count)).replace("A-E", f"A-{labels[-1]}").replace("other three", f"other {self.point_count-2}")
                setattr(self, field, prompt)
        super().__init__(*args, **kwargs)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        labels = list(self.option_labels)
        n_pts = len(labels)
        for _ in range(999):
            # Sample native markers with margin and minimum spacing
            pts: List[Point] = []
            spacing_min = self.canvas_short_side * 0.10
            ok = True
            if n_pts > 5:
                # Sample the unique close pair uniformly in the native canvas;
                # every other point must compete in the same all-pairs search.
                first = self.pick_target_point(0.82, padding=candidate_padding)
                second = self.point_on_ray(first, self._rng.uniform(0, 6.283185307179586), self._rng.uniform(42, 48))
                if not self.inside_canvas(second, padding=candidate_padding):
                    continue
                pts = [first, second]
                spacing_min = self.distance(first, second) * 1.60
            for _ in range(n_pts - len(pts)):
                inner_ok = False
                for _ in range(120):
                    cand = self.pick_target_point(0.82 if n_pts > 5 else 0.62, padding=candidate_padding)
                    if all(self.distance(cand, p) >= spacing_min for p in pts):
                        pts.append(cand)
                        inner_ok = True
                        break
                if not inner_ok:
                    ok = False
                    break
            if not ok:
                continue

            # Find closest pair
            min_d = float("inf")
            min_i, min_j = 0, 1
            second_d = float("inf")
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    d = self.distance(pts[i], pts[j])
                    if d < min_d:
                        second_d = min_d
                        min_d = d
                        min_i, min_j = i, j
                    elif d < second_d:
                        second_d = d
            # Force a clear gap between the closest pair and the next pair
            # so the answer is unambiguous. Distractors must NOT be too close
            # to the closest pair.
            if min_d == 0 or second_d / min_d < 1.55:
                continue
            # No three collinear / triangle gimmicks needed; just enforce gap

            shuffled_labels = list(labels)
            self._rng.shuffle(shuffled_labels)
            correct_label = "".join(sorted((shuffled_labels[min_i], shuffled_labels[min_j])))

            # Build candidates manually
            candidates = [PointCandidate(x=pts[i].x, y=pts[i].y, label=shuffled_labels[i])
                          for i in range(n_pts)]
            if not self.validate_candidate_layout(candidates, min_distance=(40 if n_pts > 5 else spacing_min * 0.95)):
                continue

            self.candidates = candidates
            self.correct_label = correct_label
            self.points = pts
            self.min_pair_idx = (min_i, min_j)
            self.target_point = Point(
                x=(pts[min_i].x + pts[min_j].x) / 2.0,
                y=(pts[min_i].y + pts[min_j].y) / 2.0,
            )
            return self.save_puzzle()
        raise RuntimeError("Failed to generate closest pair puzzle.")

    def _authored_observation(self) -> str:
        i, j = self.min_pair_idx
        first_region = self._canvas_region(self.points[i])
        second_region = self._canvas_region(self.points[j])
        if first_region == second_region:
            pair_location = f"with both of its members {first_region}"
        else:
            pair_location = (
                f"{self._canvas_region(self.target_point)}, one member {first_region} "
                f"and the other {second_region}"
            )
        return (
            f"Only the {len(self.points)} labeled markers occupy the canvas, so I compare their separations pairing by pairing. "
            f"One pairing stands apart, {pair_location}: those two centers sit closer together than any other "
            "two markers on the canvas, while every remaining pairing keeps a clearly wider gap."
        )

    def build_record_extra(self) -> dict[str, object]:
        i, j = self.min_pair_idx
        return {
            "difficulty_variant": "local_minimum_pair_search",
            "all_points": [p.to_list() for p in self.points],
            "closest_pair_indices": [int(i), int(j)],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        if highlight_label:
            i, j = self.min_pair_idx
            # Connect the closest pair, trimmed to stop at the marker rims
            t1, t2 = self.trim_segment(
                self.points[i], self.points[j],
                start_offset=float(self.point_radius),
                end_offset=float(self.point_radius),
            )
            self.draw_line(draw, [t1, t2])
        self.draw_candidates(draw, highlight_label=highlight_label)
        return base
