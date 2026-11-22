"""Multi-reflection mirror box: shared physics-family implementation.

A beam enters a walled box through a source port and reflects specularly off
two to four interior plane mirrors before ending on a wall, where exactly one
of four labeled detector disks contains the endpoint.  The reflection law is
stated executably in the prompt; the beam path itself is absent from the
input, so tracing the reflections is the load-bearing visual work.
"""

from __future__ import annotations

from data.common.assets import render_font

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageFont

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.physics.core.physical_scene_common import PhysicalScenePattern
from data.physics.core.source_replay import carrier_polyline_samples


BG = (247, 249, 250)
CARD = (253, 254, 254)
INK = (35, 43, 52)
RAY = (239, 151, 48)
BLUE = (57, 115, 182)
GREEN = (70, 154, 111)
IDLE = (231, 235, 238)
WALL = (66, 78, 90)
BOX_FLOOR = (249, 251, 252)

_BOX = (64.0, 64.0, 448.0, 448.0)
_DETECTOR_RADIUS = 20.0
_MIRROR_HALF = 40.0
_LABELS = ("A", "B", "C", "D")


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def _unit(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(*vector)
    if length <= 1e-12:
        raise ValueError("zero-length direction")
    return vector[0] / length, vector[1] / length


def _reflect(direction: Tuple[float, float], tangent: Tuple[float, float]) -> Tuple[float, float]:
    """Specular reflection of ``direction`` off a mirror with unit ``tangent``."""

    tx, ty = tangent
    dot = direction[0] * tx + direction[1] * ty
    return _unit((2.0 * dot * tx - direction[0], 2.0 * dot * ty - direction[1]))


def _wall_exit(point: Tuple[float, float], direction: Tuple[float, float]) -> Tuple[float, float]:
    """First intersection of the forward ray with the box walls."""

    x0, y0, x1, y1 = _BOX
    best = math.inf
    for bound, component, coordinate in ((x0, 0, point[0]), (x1, 0, point[0]), (y0, 1, point[1]), (y1, 1, point[1])):
        speed = direction[component]
        if abs(speed) <= 1e-12:
            continue
        time = (bound - coordinate) / speed
        if 1e-9 < time < best:
            candidate = (point[0] + direction[0] * time, point[1] + direction[1] * time)
            if x0 - 1e-6 <= candidate[0] <= x1 + 1e-6 and y0 - 1e-6 <= candidate[1] <= y1 + 1e-6:
                best = time
    if not math.isfinite(best):
        raise ValueError("ray does not exit the box")
    return (point[0] + direction[0] * best, point[1] + direction[1] * best)


def _segment_hit(
    point: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    tangent: Tuple[float, float],
    half_length: float,
) -> tuple[float, float] | None:
    """(ray_time, signed_offset) where the ray meets the mirror segment."""

    nx, ny = -tangent[1], tangent[0]
    denominator = direction[0] * nx + direction[1] * ny
    if abs(denominator) <= 1e-12:
        return None
    time = ((center[0] - point[0]) * nx + (center[1] - point[1]) * ny) / denominator
    if time <= 1e-9:
        return None
    hit = (point[0] + direction[0] * time, point[1] + direction[1] * time)
    offset = (hit[0] - center[0]) * tangent[0] + (hit[1] - center[1]) * tangent[1]
    if abs(offset) > half_length:
        return None
    return time, offset


def trace_beam(
    source: Tuple[float, float],
    entry_dir: Tuple[float, float],
    mirrors: Sequence[Mapping[str, Any]],
) -> List[Tuple[float, float]]:
    """Trace the beam against all mirror segments until it reaches a wall."""

    point = (float(source[0]), float(source[1]))
    direction = _unit((float(entry_dir[0]), float(entry_dir[1])))
    path = [point]
    for _ in range(len(mirrors) + 2):
        hits = []
        for index, mirror in enumerate(mirrors):
            center = (float(mirror["center"][0]), float(mirror["center"][1]))
            tangent = _unit((float(mirror["tangent"][0]), float(mirror["tangent"][1])))
            result = _segment_hit(point, direction, center, tangent, float(mirror["half_length"]))
            if result is not None:
                hits.append((result[0], index, tangent))
        exit_point = _wall_exit(point, direction)
        exit_time = math.hypot(exit_point[0] - point[0], exit_point[1] - point[1])
        if hits:
            hits.sort()
            time, _index, tangent = hits[0]
            if time < exit_time - 1e-6:
                point = (point[0] + direction[0] * time, point[1] + direction[1] * time)
                path.append(point)
                direction = _reflect(direction, tangent)
                continue
        path.append(exit_point)
        return path
    raise ValueError("beam did not terminate on a wall")


class MirrorAimDetectorPattern(PhysicalScenePattern):
    task_name = "mirror_aim_detector"
    task_mechanic = "multi_reflection_mirror_box"
    answer_overlay_mode = "local_opacity"
    verifier_profile = "mirror_box_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "mirror_box_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def _sample_task(self) -> Dict[str, Any]:
        bounce_range = {"min": (2, 2), "typical": (3, 3), "max": (4, 4)}.get(self.difficulty or "", (2, 4))
        # Fix the reflection count first so per-attempt acceptance speed cannot
        # skew the sampled difficulty distribution toward short traces.
        reflections = self.rng.randint(*bounce_range)
        for attempt in range(30000):
            if attempt == 24000:
                reflections = max(2, reflections - 1)
            task = self._try_scene(reflections)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid mirror-box scene")

    def _sample_wall_point(self, margin: float = 70.0) -> tuple[Tuple[float, float], Tuple[float, float]]:
        x0, y0, x1, y1 = _BOX
        wall = self.rng.randrange(4)
        if wall == 0:
            point = (x0, self.rng.uniform(y0 + margin, y1 - margin))
            inward = (1.0, 0.0)
        elif wall == 1:
            point = (x1, self.rng.uniform(y0 + margin, y1 - margin))
            inward = (-1.0, 0.0)
        elif wall == 2:
            point = (self.rng.uniform(x0 + margin, x1 - margin), y0)
            inward = (0.0, 1.0)
        else:
            point = (self.rng.uniform(x0 + margin, x1 - margin), y1)
            inward = (0.0, -1.0)
        return point, inward

    def _try_scene(self, reflections: int) -> Dict[str, Any] | None:
        rng = self.rng
        x0, y0, x1, y1 = _BOX
        source, inward = self._sample_wall_point()
        tilt = rng.uniform(-1.05, 1.05)
        cos_t, sin_t = math.cos(tilt), math.sin(tilt)
        direction = _unit(
            (inward[0] * cos_t - inward[1] * sin_t, inward[0] * sin_t + inward[1] * cos_t)
        )
        point = source
        mirrors: List[Dict[str, Any]] = []
        contacts: List[Tuple[float, float]] = []
        for _ in range(reflections):
            length = rng.uniform(95.0, 235.0)
            contact = (point[0] + direction[0] * length, point[1] + direction[1] * length)
            if not (x0 + 52.0 <= contact[0] <= x1 - 52.0 and y0 + 52.0 <= contact[1] <= y1 - 52.0):
                return None
            turn = rng.choice((-1.0, 1.0)) * rng.uniform(0.55, 1.5)
            reflected = _unit(
                (
                    direction[0] * math.cos(turn) - direction[1] * math.sin(turn),
                    direction[0] * math.sin(turn) + direction[1] * math.cos(turn),
                )
            )
            # Mirror tangent from the incoming and outgoing directions.
            normal = _unit((direction[0] - reflected[0], direction[1] - reflected[1]))
            tangent = (-normal[1], normal[0])
            mirror_ends = [
                (contact[0] - tangent[0] * _MIRROR_HALF, contact[1] - tangent[1] * _MIRROR_HALF),
                (contact[0] + tangent[0] * _MIRROR_HALF, contact[1] + tangent[1] * _MIRROR_HALF),
            ]
            if not all(
                x0 + 26.0 <= end[0] <= x1 - 26.0 and y0 + 26.0 <= end[1] <= y1 - 26.0
                for end in mirror_ends
            ):
                return None
            mirrors.append(
                {
                    "center": [round(contact[0], 2), round(contact[1], 2)],
                    "tangent": [round(tangent[0], 5), round(tangent[1], 5)],
                    "half_length": _MIRROR_HALF,
                }
            )
            contacts.append(contact)
            point = contact
            direction = reflected
        endpoint = _wall_exit(point, direction)
        if math.hypot(endpoint[0] - point[0], endpoint[1] - point[1]) < 95.0:
            return None
        for corner in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            if math.hypot(endpoint[0] - corner[0], endpoint[1] - corner[1]) < 62.0:
                return None
        if math.hypot(endpoint[0] - source[0], endpoint[1] - source[1]) < 90.0:
            return None

        # Re-trace against the complete mirror set: the intended contacts must
        # be exactly the realized ones (closed loop, no accidental hits).
        try:
            path = trace_beam(source, self._entry_dir(source, contacts[0]), mirrors)
        except ValueError:
            return None
        if len(path) != reflections + 2:
            return None
        for expected, realized in zip(contacts, path[1:-1]):
            if math.hypot(expected[0] - realized[0], expected[1] - realized[1]) > 0.5:
                return None
        if math.hypot(path[-1][0] - endpoint[0], path[-1][1] - endpoint[1]) > 0.5:
            return None
        if not self._margins_ok(path, mirrors):
            return None

        detectors = self._place_detectors(path, mirrors, endpoint)
        if detectors is None:
            return None
        return {
            "source": [round(source[0], 2), round(source[1], 2)],
            "entry_dir": [round(value, 5) for value in self._entry_dir(source, contacts[0])],
            "mirrors": mirrors,
            "detectors": detectors,
        }

    @staticmethod
    def _entry_dir(source: Tuple[float, float], first_contact: Tuple[float, float]) -> Tuple[float, float]:
        return _unit((first_contact[0] - source[0], first_contact[1] - source[1]))

    def _margins_ok(self, path: Sequence[Tuple[float, float]], mirrors: Sequence[Mapping[str, Any]]) -> bool:
        # Contacts stay well inside their mirror segments.
        for contact, mirror in zip(path[1:-1], mirrors):
            center = mirror["center"]
            offset = math.hypot(contact[0] - center[0], contact[1] - center[1])
            if offset > _MIRROR_HALF - 16.0:
                return False
        # Non-target mirrors keep clearance from every beam segment.
        for segment_index in range(len(path) - 1):
            start, end = path[segment_index], path[segment_index + 1]
            for mirror_index, mirror in enumerate(mirrors):
                if mirror_index in (segment_index - 1, segment_index):
                    continue
                if self._segment_distance(start, end, mirror) < 22.0:
                    return False
        # Mirrors keep clearance from one another.
        for first_index in range(len(mirrors)):
            for second_index in range(first_index + 1, len(mirrors)):
                if self._mirror_gap(mirrors[first_index], mirrors[second_index]) < 34.0:
                    return False
        # Consecutive contacts stay apart.
        for first, second in zip(path[1:-1], path[2:-1]):
            if math.hypot(first[0] - second[0], first[1] - second[1]) < 70.0:
                return False
        return True

    @staticmethod
    def _segment_distance(start: Tuple[float, float], end: Tuple[float, float], mirror: Mapping[str, Any]) -> float:
        center = mirror["center"]
        tangent = _unit((mirror["tangent"][0], mirror["tangent"][1]))
        half = float(mirror["half_length"])
        ends = [
            (center[0] - tangent[0] * half, center[1] - tangent[1] * half),
            (center[0] + tangent[0] * half, center[1] + tangent[1] * half),
        ]

        def _point_segment(point, seg_start, seg_end):
            vx, vy = seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]
            length_sq = vx * vx + vy * vy
            if length_sq <= 1e-12:
                return math.hypot(point[0] - seg_start[0], point[1] - seg_start[1])
            t = max(0.0, min(1.0, ((point[0] - seg_start[0]) * vx + (point[1] - seg_start[1]) * vy) / length_sq))
            proj = (seg_start[0] + vx * t, seg_start[1] + vy * t)
            return math.hypot(point[0] - proj[0], point[1] - proj[1])

        return min(
            _point_segment(ends[0], start, end),
            _point_segment(ends[1], start, end),
            _point_segment(start, ends[0], ends[1]),
            _point_segment(end, ends[0], ends[1]),
        )

    @staticmethod
    def _mirror_gap(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        return math.hypot(
            float(first["center"][0]) - float(second["center"][0]),
            float(first["center"][1]) - float(second["center"][1]),
        ) - float(first["half_length"]) - float(second["half_length"])

    def _place_detectors(
        self,
        path: Sequence[Tuple[float, float]],
        mirrors: Sequence[Mapping[str, Any]],
        endpoint: Tuple[float, float],
    ) -> List[Dict[str, Any]] | None:
        x0, y0, x1, y1 = _BOX
        rng = self.rng

        def _on_wall(point: Tuple[float, float]) -> bool:
            return (
                abs(point[0] - x0) <= 1e-6
                or abs(point[0] - x1) <= 1e-6
                or abs(point[1] - y0) <= 1e-6
                or abs(point[1] - y1) <= 1e-6
            )

        def _distance_to_path(point: Tuple[float, float]) -> float:
            best = math.inf
            for start, end in zip(path, path[1:]):
                vx, vy = end[0] - start[0], end[1] - start[1]
                length_sq = vx * vx + vy * vy
                t = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
                proj = (start[0] + vx * t, start[1] + vy * t)
                best = min(best, math.hypot(point[0] - proj[0], point[1] - proj[1]))
            return best

        candidates: List[Tuple[float, float]] = []
        # Mechanism-targeted decoys: skip the final reflection, skip
        # the whole mirror set, and flip the final reflection's turn side.
        try:
            before_last = _unit(
                (path[-2][0] - path[-3][0], path[-2][1] - path[-3][1])
            )
            candidates.append(_wall_exit(path[-2], before_last))
        except (ValueError, IndexError):
            pass
        try:
            entry = _unit((path[1][0] - path[0][0], path[1][1] - path[0][1]))
            candidates.append(_wall_exit(path[0], entry))
        except ValueError:
            pass
        try:
            last_in = _unit((path[-2][0] - path[-3][0], path[-2][1] - path[-3][1]))
            true_out = _unit((path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]))
            mirrored_out = _reflect(true_out, last_in)
            candidates.append(_wall_exit(path[-2], mirrored_out))
        except (ValueError, IndexError):
            pass
        for _ in range(60):
            point, _inward = self._sample_wall_point(margin=64.0)
            candidates.append(point)

        chosen: List[Tuple[float, float]] = [endpoint]
        for candidate in candidates:
            if len(chosen) == 4:
                break
            if not _on_wall(candidate):
                continue
            if any(
                math.hypot(candidate[0] - corner[0], candidate[1] - corner[1]) < 62.0
                for corner in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
            ):
                continue
            if math.hypot(candidate[0] - path[0][0], candidate[1] - path[0][1]) < 78.0:
                continue
            if any(
                math.hypot(candidate[0] - taken[0], candidate[1] - taken[1]) < 78.0
                for taken in chosen
            ):
                continue
            if _distance_to_path(candidate) < 44.0:
                continue
            chosen.append(candidate)
        if len(chosen) != 4:
            return None
        order = list(range(4))
        rng.shuffle(order)
        detectors = [None] * 4
        for label_index, source_index in enumerate(order):
            detectors[label_index] = {
                "label": _LABELS[label_index],
                "center": [round(chosen[source_index][0], 2), round(chosen[source_index][1], 2)],
            }
        # The true endpoint must sit in exactly one detector disk.
        containing = [
            detector["label"]
            for detector in detectors
            if math.hypot(detector["center"][0] - endpoint[0], detector["center"][1] - endpoint[1])
            <= _DETECTOR_RADIUS
        ]
        if len(containing) != 1:
            return None
        return detectors

    # ---------------------------------------------------------------- semantics

    @staticmethod
    def _path(task: Mapping[str, Any]) -> List[Tuple[float, float]]:
        return trace_beam(
            (float(task["source"][0]), float(task["source"][1])),
            (float(task["entry_dir"][0]), float(task["entry_dir"][1])),
            task["mirrors"],
        )

    @classmethod
    def _target_label(cls, task: Mapping[str, Any]) -> str:
        endpoint = cls._path(task)[-1]
        containing = [
            str(detector["label"])
            for detector in task["detectors"]
            if math.hypot(
                float(detector["center"][0]) - endpoint[0],
                float(detector["center"][1]) - endpoint[1],
            )
            <= _DETECTOR_RADIUS
        ]
        if len(containing) != 1:
            raise ValueError("mirror box requires exactly one containing detector")
        return containing[0]

    @classmethod
    def progress_carriers(
        cls, task: Mapping[str, Any]
    ) -> List[Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]]:
        """The beam, whose traced length only ever grows.

        ``_render_progress`` paints the prefix of the traced path by arc length
        and paints it last, over the mirrors and the detectors, so once a point
        of the path carries the ray it carries it in every later frame.  The
        declared samples are the traced path itself at one point per pixel, and
        the reading is the length of beam already drawn.
        """

        return [("beam", carrier_polyline_samples(cls._path(task)), BOX_FLOOR, None, 1)]

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for detector in task["detectors"]:
            cx, cy = float(detector["center"][0]), float(detector["center"][1])
            pad = _DETECTOR_RADIUS + 6.0
            boxes.append((f"detector_{detector['label']}", (cx - pad, cy - pad, cx + pad, cy + pad)))
        return boxes

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Which detector does the beam reach after all reflections?"

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        label = self._target_label(task)
        FOUR_CHOICE_CODEC.format(label)
        return label

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        path = self._path(task)
        lengths = [
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(path, path[1:])
        ]
        total = sum(lengths)
        states: List[tuple[float, str]] = []
        cumulative = 0.0
        for index in range(1, len(path) - 1):
            cumulative += lengths[index - 1]
            states.append((cumulative / total, self._contact_text(task, path, index)))
        states.append((1.0, self._terminal_text(task, path)))
        return states

    @staticmethod
    def _region_phrase(point: Tuple[float, float]) -> str:
        x0, y0, x1, y1 = _BOX
        column = ("left", "middle", "right")[min(2, int((point[0] - x0) / ((x1 - x0) / 3.0)))]
        row = ("upper", "middle", "lower")[min(2, int((point[1] - y0) / ((y1 - y0) / 3.0)))]
        if row == "middle" and column == "middle":
            return "center"
        if row == "middle":
            return f"middle {column}"
        if column == "middle":
            return f"{row} middle"
        return f"{row} {column}"

    @staticmethod
    def _heading_phrase(direction: Tuple[float, float]) -> str:
        horizontal = "right" if direction[0] > 0 else "left"
        vertical = "down" if direction[1] > 0 else "up"
        if abs(direction[0]) < 0.20:
            return f"almost straight {vertical}"
        if abs(direction[1]) < 0.20:
            return f"almost straight {horizontal}"
        return f"{vertical}-{horizontal}"

    def _contact_text(self, task: Mapping[str, Any], path: Sequence[Tuple[float, float]], index: int) -> str:
        contact = path[index]
        incoming = _unit((contact[0] - path[index - 1][0], contact[1] - path[index - 1][1]))
        outgoing = _unit((path[index + 1][0] - contact[0], path[index + 1][1] - contact[1]))
        where = self._region_phrase(contact)
        if index == 1:
            opening = (
                f"From the source the beam travels {self._heading_phrase(incoming)} and first strikes the mirror in the {where} of the box."
            )
        else:
            opening = (
                f"The reflected beam continues {self._heading_phrase(incoming)} and next strikes the mirror in the {where} of the box."
            )
        return (
            opening
            + f" Keeping the angle to that mirror equal on both sides sends the beam onward {self._heading_phrase(outgoing)}."
        )

    def _terminal_text(self, task: Mapping[str, Any], path: Sequence[Tuple[float, float]]) -> str:
        label = self._target_label(task)
        final_dir = _unit((path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]))
        return (
            f"After the last reflection the beam runs straight {self._heading_phrase(final_dir)} until it meets the wall, "
            f"where it lands inside detector {label}; that single detector lights up."
        )

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        path = self._path(task)
        return {
            "semantic_state": {
                "source_evidence": {
                    "source": list(task["source"]),
                    "entry_dir": list(task["entry_dir"]),
                    "mirrors": [dict(mirror) for mirror in task["mirrors"]],
                    "detectors": [dict(detector) for detector in task["detectors"]],
                }
            },
            "answer_schema": {
                "schema_version": "mirror_box_detector_v1",
                "kind": "detector_label",
                "parser_profile": "strict_abcd_answer",
                "format_template": FOUR_CHOICE_CODEC.schema,
                "format_example": FOUR_CHOICE_CODEC.example,
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "video_profile": self.video_verifier_profile,
                "semantic_source": "visible source port, entry arrow, mirror segments, and fixed labeled detector disks",
                "strict_binary_correctness": True,
                "strict_text_parser": FOUR_CHOICE_CODEC.full_pattern,
                "untrusted_redundant_fields": [
                    "path_points",
                    "target",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "exactly one detector disk contains the replayed beam endpoint",
            "path_points": [[round(x, 3), round(y, 3)] for x, y in path],
            "target": self._target_label(task),
            "reflection_count": len(task["mirrors"]),
            "semantic_state_trace": [
                "entry_hold",
                *[f"reflection_{index + 1}" for index in range(len(task["mirrors"]))],
                "detector_contact_final_hold",
            ],
        }

    # ------------------------------------------------------------------- render

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        x0, y0, x1, y1 = _BOX
        path = self._path(task)
        target = self._target_label(task)

        draw.rounded_rectangle((20, 20, 492, 492), radius=22, fill=CARD, outline=(214, 220, 224), width=2)
        draw.rectangle((x0 - 8, y0 - 8, x1 + 8, y1 + 8), fill=BOX_FLOOR, outline=WALL, width=6)

        # Mirrors: layered silver bars.
        for mirror in task["mirrors"]:
            center = mirror["center"]
            tangent = _unit((mirror["tangent"][0], mirror["tangent"][1]))
            half = float(mirror["half_length"])
            ends = (
                center[0] - tangent[0] * half,
                center[1] - tangent[1] * half,
                center[0] + tangent[0] * half,
                center[1] + tangent[1] * half,
            )
            draw.line(ends, fill=(75, 88, 101), width=11)
            draw.line(ends, fill=(184, 194, 203), width=7)
            draw.line(ends, fill=(231, 235, 238), width=2)

        # Detector disks with labels.
        label_font = _font(19 * supersample)
        for detector in task["detectors"]:
            cx, cy = float(detector["center"][0]), float(detector["center"][1])
            active = mark_answer and str(detector["label"]) == target
            fill = GREEN if active else IDLE
            draw.ellipse(
                (cx - _DETECTOR_RADIUS, cy - _DETECTOR_RADIUS, cx + _DETECTOR_RADIUS, cy + _DETECTOR_RADIUS),
                fill=fill,
                outline=INK,
                width=2,
            )
            draw.text(
                (cx, cy),
                str(detector["label"]),
                anchor="mm",
                fill=(255, 255, 255) if active else INK,
                font=label_font,
            )

        # Source port and entry arrow.
        sx, sy = float(task["source"][0]), float(task["source"][1])
        entry = _unit((task["entry_dir"][0], task["entry_dir"][1]))
        arrow_start = (sx + entry[0] * 16.0, sy + entry[1] * 16.0)
        arrow_end = (sx + entry[0] * 52.0, sy + entry[1] * 52.0)
        draw.line((*arrow_start, *arrow_end), fill=BLUE, width=3)
        angle = math.atan2(entry[1], entry[0])
        for delta in (math.radians(150), math.radians(-150)):
            draw.line(
                (
                    arrow_end[0],
                    arrow_end[1],
                    arrow_end[0] + 10 * math.cos(angle + delta),
                    arrow_end[1] + 10 * math.sin(angle + delta),
                ),
                fill=BLUE,
                width=3,
            )
        draw.ellipse((sx - 11, sy - 11, sx + 11, sy + 11), fill=BLUE, outline=INK, width=2)
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=(245, 249, 252))

        # Beam prefix by arc length.
        amount = max(0.0, min(1.0, float(progress)))
        if amount > 0.0:
            lengths = [
                math.hypot(second[0] - first[0], second[1] - first[1])
                for first, second in zip(path, path[1:])
            ]
            distance = amount * sum(lengths)
            cursor = path[0]
            for index, segment_length in enumerate(lengths):
                if distance >= segment_length - 1e-9:
                    nxt = path[index + 1]
                    draw.line((*cursor, *nxt), fill=RAY, width=5)
                    cursor = nxt
                    distance -= segment_length
                    continue
                fraction = max(0.0, distance / segment_length)
                if fraction > 0.0:
                    partial = (
                        cursor[0] + (path[index + 1][0] - cursor[0]) * fraction,
                        cursor[1] + (path[index + 1][1] - cursor[1]) * fraction,
                    )
                    draw.line((*cursor, *partial), fill=RAY, width=5)
                break
        return finish_scaled_canvas(image, (512, 512))

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        count = len(task["mirrors"])
        count_word = {2: "two", 3: "three", 4: "four"}[count]
        return VisualReasoningContract(
            input_description=(
                "The single image is a walled box seen from above. A blue port on one wall emits a beam whose entry direction is fixed by the short blue arrow, "
                f"{count_word} silver bars inside the box are plane mirrors reflective on both faces, and four labeled gray disks on the walls are detectors. No beam is drawn yet."
            ),
            symbol_legend=(
                "The blue arrow fixes only the beam's entry direction; its drawn length carries no meaning.",
                "A through D are fixed detector identities, all inactive gray in the input.",
            ),
            rules=(
                "The beam leaves the port along the arrow and travels in straight lines.",
                "When the beam meets a mirror bar, it reflects so that its angle to the mirror surface is the same on both sides of the contact; equivalently, with unit direction i and unit mirror normal n, the outgoing direction is i - 2(i dot n)n.",
                "A mirror reflects on both faces, but only where the bar actually is; the beam passes freely beside a bar's ends.",
                "Walls do not reflect: the beam ends at the first wall point it reaches, and the unique answer is the detector disk containing that endpoint.",
            ),
            modality_outputs={
                "ti2v": (
                    "Hold the input, then grow the orange beam continuously from the port: each straight segment extends until it meets its mirror, turns by the equal-angle law, and continues, one segment at a time. "
                    "After the beam lands on the wall, only the reached detector fills green while everything else stays fixed; hold that final state."
                ),
                "solution_image": (
                    "Preserve the exact input scene, add the complete orange beam from the port through every mirror contact to the wall, and fill only the reached detector green. Do not add formulas, coordinates, or any text."
                ),
            },
            preservation_constraints=(
                "Keep the walls, port, arrow, mirror bars, detector disks, labels, and colors fixed; add only the growing beam and the single green detector fill.",
            ),
            transition_constraints=(
                "Grow the beam monotonically with one partial segment at a time; do not retract, branch, skip a mirror, cross a wall, or light more than one detector.",
            ),
            ti2v_continuity={
                "moving_nouns": "the one growing orange beam",
                "static_nouns": "the walls, port, arrow, mirror bars, detector disks, labels, and camera",
                "hold_noun": "the complete beam with the single green detector",
            },
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_abcd_answer",
                verifier_profile=self.verifier_profile,
                format_example=FOUR_CHOICE_CODEC.example,
            ),
            render_profile="mirror_box_multi_reflection_v1",
        )


__all__ = ["MirrorAimDetectorPattern", "trace_beam"]
