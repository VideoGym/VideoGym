from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.pipeline.media import load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.common.contracts.multimodal import (
    AnswerContract,
    VisualReasoningContract,
    audit_contract_prompts,
)
from data.common.base import RenderedTaskGenerator
from data.physics.core.source_replay import carrier_polyline_samples


_CANVAS = (512, 512)
_ARENA_MARGIN = 42
_BALL_RADIUS = 14
_MIN_BOUNCES = 3
_MAX_BOUNCES = 7
_MIN_RENDERED_SEGMENT_LENGTH = 80.0
_LABELS = ("A", "B", "C", "D")

_BACKGROUND = (239, 244, 247)
_ARENA_FILL = (255, 254, 250)
_ARENA_INTERIOR = (255, 255, 253)
_ARENA_OUTLINE = (35, 45, 52)
_BALL_FILL = (58, 128, 197)
_BALL_OUTLINE = (24, 53, 76)
_ARROW = (31, 67, 132)
_TRACE = (220, 103, 54)
_CANDIDATE_FILL = (255, 249, 226)
_CANDIDATE_OUTLINE = (181, 132, 37)
_LABEL_FILL = (255, 255, 253)
_LABEL_INK = (50, 55, 58)


Point = Tuple[float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _rounded_point(point: Point) -> List[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _heading_phrase(vector: Point) -> str:
    """Natural screen-space travel phrase for one velocity vector."""

    horizontal = "right" if vector[0] > 0 else "left"
    vertical = "down" if vector[1] > 0 else "up"
    if abs(vector[0]) < 1e-9:
        return f"straight {vertical}"
    if abs(vector[1]) < 1e-9:
        return f"straight {horizontal}"
    return f"{vertical}-{horizontal}"


class BallBouncesGivenTimePattern(RenderedTaskGenerator):
    task_name = "ball_bounces_given_time"
    task_family = "physics"
    task_mechanic = "elastic_wall_collision_count"
    verifier_profile = "ball_bounces_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "ball_bounces_full_video_semantic_single_decode"
    video_fps = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        video_fps: int = 16,
        generate_videos: bool = True,
        video: bool | None = None,
        bounce_count: int | None = None,
        image_size: object | None = None,
        target_size: object | None = None,
        unique: bool = True,
        **_: Any,
    ) -> None:
        del image_size, target_size, unique
        resolved_output = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / "videogym_ball_bounces_given_time"
        )
        resolved_output.mkdir(parents=True, exist_ok=True)
        self.output_dir = resolved_output
        self.image_size = _CANVAS
        self.video_fps = int(video_fps)
        self.generate_videos = bool(generate_videos if video is None else video)
        effective_seed = seed if seed is not None else random_seed
        self.rng = random.Random(effective_seed)
        if bounce_count is not None and not _MIN_BOUNCES <= int(bounce_count) <= 9:
            raise ValueError(f"bounce_count must be in [{_MIN_BOUNCES}, 9]")
        self.fixed_bounce_count = int(bounce_count) if bounce_count is not None else None
        self._sample_index = 0

    @staticmethod
    def _center_bounds(width: int, height: int, margin: int, radius: int) -> Tuple[float, float, float, float]:
        return (
            float(margin + radius),
            float(width - margin - radius),
            float(margin + radius),
            float(height - margin - radius),
        )

    def _simulate(
        self,
        width: int,
        height: int,
        margin: int,
        radius: int,
        start: Tuple[int, int] | Point,
        velocity: Tuple[int, int] | Point,
        bounce_count: int,
    ) -> Tuple[List[Point], Point]:
        """Return the start followed by one exact point per wall collision.

        A vertical-wall collision reverses only vx, a horizontal-wall collision
        reverses only vy, and an exact corner reverses both components while
        counting as one collision event.
        """

        if bounce_count < 0:
            raise ValueError("bounce_count must be non-negative")
        x, y = float(start[0]), float(start[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        if abs(vx) < 1e-12 and abs(vy) < 1e-12:
            raise ValueError("velocity must be non-zero")
        left, right, top, bottom = self._center_bounds(width, height, margin, radius)
        epsilon = 1e-9
        if not left - epsilon <= x <= right + epsilon or not top - epsilon <= y <= bottom + epsilon:
            raise ValueError("start must keep the complete ball inside the arena")

        trace: List[Point] = [(x, y)]
        for _ in range(int(bounce_count)):
            time_to_x = (
                (right - x) / vx
                if vx > epsilon
                else (left - x) / vx
                if vx < -epsilon
                else math.inf
            )
            time_to_y = (
                (bottom - y) / vy
                if vy > epsilon
                else (top - y) / vy
                if vy < -epsilon
                else math.inf
            )
            collision_time = min(time_to_x, time_to_y)
            if not math.isfinite(collision_time) or collision_time <= epsilon:
                raise RuntimeError("ball simulation did not reach a future wall collision")

            x += vx * collision_time
            y += vy * collision_time
            x = min(right, max(left, x))
            y = min(bottom, max(top, y))
            trace.append((x, y))

            hit_vertical = abs(time_to_x - collision_time) <= epsilon
            hit_horizontal = abs(time_to_y - collision_time) <= epsilon
            if hit_vertical:
                vx = -vx
            if hit_horizontal:
                vy = -vy

        return trace, trace[-1]

    @staticmethod
    def _wall_hit_count(point: Point, bounds: Tuple[float, float, float, float]) -> int:
        left, right, top, bottom = bounds
        x, y = point
        return sum(
            (
                abs(x - left) <= 1e-7 or abs(x - right) <= 1e-7,
                abs(y - top) <= 1e-7 or abs(y - bottom) <= 1e-7,
            )
        )

    def _candidate_events(
        self,
        collision_points: Sequence[Point],
        bounce_count: int,
        bounds: Tuple[float, float, float, float],
    ) -> List[Tuple[Point, int | None, str]] | None:
        correct = collision_points[bounce_count - 1]
        ordered_counts: List[int] = []
        for offset in range(1, len(collision_points) + 1):
            for candidate_count in (bounce_count - offset, bounce_count + offset):
                if 1 <= candidate_count <= len(collision_points) and candidate_count != bounce_count:
                    ordered_counts.append(candidate_count)

        # Keep one mechanism-related distractor from another real collision,
        # then add two wall contacts that are not ball landings. This prevents
        # the option set from reducing to "which collision number among four
        # already-valid trace points" while keeping every marker physically on
        # a legal ball-center wall-contact locus.
        selected: List[Tuple[Point, int | None, str]] = [(correct, bounce_count, "target_collision")]
        for candidate_count in ordered_counts:
            point = collision_points[candidate_count - 1]
            if all(_distance(point, existing_point) >= 52.0 for existing_point, _count, _kind in selected):
                selected.append((point, candidate_count, "other_collision"))
                break
        if len(selected) != 2:
            return None

        left, right, top, bottom = bounds
        wall_points: List[Point] = []
        for fraction in (0.12, 0.22, 0.32, 0.42, 0.58, 0.68, 0.78, 0.88):
            x = left + (right - left) * fraction
            y = top + (bottom - top) * fraction
            wall_points.extend(((x, top), (x, bottom), (left, y), (right, y)))
        self.rng.shuffle(wall_points)
        for point in wall_points:
            if min(_distance(point, collision) for collision in collision_points) < 38.0:
                continue
            if any(_distance(point, existing_point) < 52.0 for existing_point, _count, _kind in selected):
                continue
            selected.append((point, None, "off_trace_wall_contact"))
            if len(selected) == 4:
                return selected
        return None

    def _generate_task(self) -> Dict[str, Any]:
        width, height = self.image_size
        bounce_count = self.fixed_bounce_count or self.rng.randint(_MIN_BOUNCES, _MAX_BOUNCES)
        margin = _ARENA_MARGIN
        radius = _BALL_RADIUS
        bounds = self._center_bounds(width, height, margin, radius)
        velocity_pool = (
            (7, 4),
            (6, 5),
            (5, 7),
            (7, 5),
            (-7, 4),
            (-6, 5),
            (-5, -7),
            (7, -5),
            (-7, -4),
            (6, -5),
            (4, 7),
            (-4, -7),
            (-5, 6),
            (5, -6),
        )

        for attempt in range(1, 129):
            start = (
                self.rng.randint(116, width - 116),
                self.rng.randint(116, height - 116),
            )
            velocity = self.rng.choice(velocity_pool)
            extended_count = max(12, bounce_count + 5)
            extended_trace, _ = self._simulate(
                width,
                height,
                margin,
                radius,
                start,
                velocity,
                extended_count,
            )
            collision_points = extended_trace[1:]
            # Corner events are fully defined by the contract, but ordinary
            # samples avoid them so each reflection is visually unambiguous.
            if any(self._wall_hit_count(point, bounds) != 1 for point in collision_points):
                continue
            trace = extended_trace[: bounce_count + 1]
            if any(
                _distance(start_point, end_point) < _MIN_RENDERED_SEGMENT_LENGTH
                for start_point, end_point in zip(trace, trace[1:])
            ):
                # Near-corner ricochets cannot be shown at approximately
                # constant speed while also retaining an exact collision frame
                # inside the compact video budget.
                continue
            candidate_events = self._candidate_events(collision_points, bounce_count, bounds)
            if candidate_events is None:
                continue

            self.rng.shuffle(candidate_events)
            correct_option = next(
                _LABELS[index]
                for index, (_point, event_count, kind) in enumerate(candidate_events)
                if event_count == bounce_count and kind == "target_collision"
            )
            candidates = [point for point, _event_count, _kind in candidate_events]
            event_counts = [event_count for _point, event_count, _kind in candidate_events]
            candidate_kinds = [kind for _point, _event_count, kind in candidate_events]
            return {
                "margin": margin,
                "radius": radius,
                "center_bounds": list(bounds),
                "start": start,
                "velocity": velocity,
                "bounce_count": bounce_count,
                "trace": trace,
                "final_point": trace[-1],
                "candidates": candidates,
                "candidate_event_counts": event_counts,
                "candidate_kinds": candidate_kinds,
                "correct_option": correct_option,
                "generation_attempts": attempt,
            }
        raise RuntimeError("unable to generate four distinct bounce-count candidates after 128 attempts")

    @staticmethod
    def _label_position(point: Point, bounds: Sequence[float]) -> Point:
        left, right, top, bottom = (float(value) for value in bounds)
        x, y = point
        inset = 27.0
        if abs(x - left) <= 1e-6:
            return x + inset, y
        if abs(x - right) <= 1e-6:
            return x - inset, y
        if abs(y - top) <= 1e-6:
            return x, y + inset
        return x, y - inset

    def _draw_arrow(self, draw: Any, start: Point, velocity: Point) -> None:
        magnitude = math.hypot(velocity[0], velocity[1])
        ux, uy = velocity[0] / magnitude, velocity[1] / magnitude
        start_distance = _BALL_RADIUS + 4
        end_distance = 56
        arrow_start = (start[0] + ux * start_distance, start[1] + uy * start_distance)
        arrow_end = (start[0] + ux * end_distance, start[1] + uy * end_distance)
        draw.line((*arrow_start, *arrow_end), fill=_ARROW, width=3)
        angle = math.atan2(uy, ux)
        head_length = 10
        for delta in (math.radians(150), math.radians(-150)):
            head = (
                arrow_end[0] + head_length * math.cos(angle + delta),
                arrow_end[1] + head_length * math.sin(angle + delta),
            )
            draw.line((*arrow_end, *head), fill=_ARROW, width=3)

    @staticmethod
    def _collision_text(point: Point, bounds: Sequence[float]) -> str:
        left, right, top, bottom = (float(value) for value in bounds)
        walls: List[str] = []
        if abs(point[0] - left) <= 1e-6:
            walls.append("left")
        elif abs(point[0] - right) <= 1e-6:
            walls.append("right")
        if abs(point[1] - top) <= 1e-6:
            walls.append("top")
        elif abs(point[1] - bottom) <= 1e-6:
            walls.append("bottom")
        if len(walls) == 2:
            return f"the {walls[0]}-{walls[1]} corner; both velocity components reverse"
        if walls and walls[0] in {"left", "right"}:
            return f"the {walls[0]} wall; only the horizontal velocity component reverses"
        if walls:
            return f"the {walls[0]} wall; only the vertical velocity component reverses"
        raise ValueError("collision point is not on the ball-center boundary")

    def _draw_frame(
        self,
        task: Dict[str, Any],
        *,
        path_points: Sequence[Point],
        current_position: Point,
    ) -> Image.Image:
        width, height = self.image_size
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(width, height, supersample, background=_BACKGROUND)

        draw.rounded_rectangle(
            (18, 18, width - 18, height - 18),
            radius=24,
            fill=_ARENA_FILL,
            outline=(204, 214, 219),
            width=2,
        )
        margin = int(task["margin"])
        draw.rectangle(
            (margin, margin, width - margin, height - margin),
            fill=_ARENA_INTERIOR,
            outline=_ARENA_OUTLINE,
            width=4,
        )

        if len(path_points) > 1:
            # Each vertex of the trace is a specular reflection, so the joins are
            # mitered rather than rounded.
            draw.open_path(
                [(float(px), float(py)) for px, py in path_points], fill=_TRACE, width=4
            )

        label_font = load_font(13 * supersample)
        for label, point in zip(_LABELS, task["candidates"]):
            x, y = float(point[0]), float(point[1])
            marker_radius = 7
            diamond = (
                (x, y - marker_radius),
                (x + marker_radius, y),
                (x, y + marker_radius),
                (x - marker_radius, y),
            )
            draw.polygon(diamond, fill=_CANDIDATE_FILL, outline=_CANDIDATE_OUTLINE, width=1)
            label_x, label_y = self._label_position(point, task["center_bounds"])
            half = 10
            draw.rounded_rectangle(
                (label_x - half, label_y - half, label_x + half, label_y + half),
                radius=6,
                fill=_LABEL_FILL,
                outline=_CANDIDATE_OUTLINE,
                width=1,
            )
            draw.text((label_x, label_y), label, font=label_font, fill=_LABEL_INK, anchor="mm")

        self._draw_arrow(draw, task["start"], task["velocity"])

        x, y = float(current_position[0]), float(current_position[1])
        radius = float(task["radius"])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_BALL_FILL,
            outline=_BALL_OUTLINE,
            width=2,
        )
        draw.ellipse(
            (x - 7, y - 8, x - 2, y - 3),
            fill=(184, 220, 244),
        )
        return finish_scaled_canvas(image, (width, height))

    @classmethod
    def progress_carriers(
        cls, task: Dict[str, Any]
    ) -> List[Tuple[str, List[Point], Tuple[int, int, int], Tuple[int, int, int] | None, int]]:
        """The drawn trace, whose arc length only ever grows.

        The ball's *position* is not monotone in time - it reverses at every
        wall - but the arc length it has travelled is, and the renderer draws
        exactly that: every frame paints the prefix of the trajectory up to the
        ball and never unpaints any of it.  The declared samples are the
        trajectory itself at one point per pixel, so the reading is the arc
        length already covered, counted in pixels of the path.

        Nothing takes ink back off those points.  The trace is drawn under the
        markers, the entry arrow and the ball, so a sample the scene permanently
        covers reads the same in every frame and a sample the ball passes over
        goes from arena to ball to trace without ever returning to arena.  The
        reading therefore rises with the ball and holds when it stops.

        The one exception is a point some *distant* leg of the trajectory passes
        within a ball radius of: the ball reaches it before the trace does and
        leaves again without drawing there, so that point genuinely dips.  Those
        points are dropped, because keeping them would raise the reader's floor
        for the whole path to cover a dip that carries no motion at all.
        """

        return [
            (
                "trace",
                carrier_polyline_samples(
                    [tuple(point) for point in task["trace"]],
                    self_clearance=float(task["radius"]) + 2.0,
                ),
                _ARENA_INTERIOR,
                None,
                1,
            )
        ]

    @staticmethod
    def _segment_frame_counts(trace: Sequence[Point], total_motion_frames: int) -> List[int]:
        """Allocate motion frames proportionally while retaining every collision frame."""

        segment_lengths = [_distance(trace[index], trace[index + 1]) for index in range(len(trace) - 1)]
        if not segment_lengths:
            return []
        minimum_frames = 2 * len(segment_lengths)
        if total_motion_frames < minimum_frames:
            raise ValueError("motion budget must allow at least two frames per segment")
        total_length = sum(segment_lengths)
        remaining = total_motion_frames - minimum_frames
        exact_extras = [remaining * length / total_length for length in segment_lengths]
        extras = [int(math.floor(value)) for value in exact_extras]
        leftover = remaining - sum(extras)
        order = sorted(
            range(len(segment_lengths)),
            key=lambda index: (exact_extras[index] - extras[index], segment_lengths[index]),
            reverse=True,
        )
        for index in order[:leftover]:
            extras[index] += 1
        counts = [2 + extra for extra in extras]
        if sum(counts) != total_motion_frames:
            raise AssertionError("segment frame allocation did not preserve the motion budget")
        return counts

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> int:
        if not self.generate_videos:
            return 0
        trace = task["trace"]
        puzzle = np.asarray(self._draw_frame(task, path_points=[trace[0]], current_position=trace[0]), dtype=np.uint8)
        solution = np.asarray(self._draw_frame(task, path_points=trace, current_position=trace[-1]), dtype=np.uint8)
        frames: List[np.ndarray] = [puzzle.copy() for _ in range(9)]
        motion_frames = 12 * int(task["bounce_count"])
        segment_frame_counts = self._segment_frame_counts(trace, motion_frames)
        for segment_index, segment_frames in enumerate(segment_frame_counts):
            start = trace[segment_index]
            end = trace[segment_index + 1]
            for local_frame in range(1, segment_frames + 1):
                fraction = local_frame / segment_frames
                current = (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
                if local_frame == segment_frames:
                    prefix = list(trace[: segment_index + 2])
                else:
                    prefix = [*trace[: segment_index + 1], current]
                frames.append(
                    np.asarray(
                        self._draw_frame(task, path_points=prefix, current_position=current),
                        dtype=np.uint8,
                    )
                )
        frames.extend(solution.copy() for _ in range(8))
        if len(frames) % 4 != 1 or len(frames) > 157:
            raise AssertionError("ball-bounce video must be WAN-compatible and at most 157 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _contract(self, bounce_count: int) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single supplied image shows one rectangular arena, one blue ball fully inside it, "
                "one navy arrow at the ball, and four equally styled gold diamond candidate markers labeled A through D."
            ),
            symbol_legend=(
                "The arrow fixes only the ball's initial travel direction; its drawn length is not a speed scale.",
                "Each gold diamond marks a possible final ball-center position, and its nearby letter is the option label.",
                "The dark rectangle is the physical wall; the ball collides when its circular edge touches that wall.",
            ),
            rules=(
                "The ball center moves in a straight line at constant speed between collisions.",
                "At a vertical wall, reverse only the horizontal velocity component; at a horizontal wall, reverse only the vertical component, so incidence and reflection angles are equal.",
                "An exact corner reverses both components and counts as one collision event.",
                f"Stop immediately at the {_ordinal(bounce_count)} wall collision and choose the unique marker at the ball center.",
            ),
            modality_outputs={
                "ti2v": f"Animate the one ball continuously along the elastic route, show all {bounce_count} collisions in order, stop at the matching marker, and hold the final state.",
                "solution_image": f"Draw the cumulative orange route through exactly {bounce_count} collisions and place the one blue ball at the final candidate marker.",
            },
            preservation_constraints=(
                "Keep the arena, candidate markers/labels, starting arrow, and palette fixed; move one unchanged ball only.",
            ),
            transition_constraints=(
                f"Use constant-speed straight segments and contact-only reflections; do not cross a wall or stop before collision {bounce_count}.",
            ),
            ti2v_continuity={
                "moving_nouns": "the one blue ball and its growing orange route",
                "static_nouns": "the arena walls, the four labeled candidate markers, the starting arrow, the background, and the camera",
                "hold_noun": "the stopped ball on the answer marker with the complete route",
            },
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_abcd_answer",
                verifier_profile=self.verifier_profile,
                format_example=FOUR_CHOICE_CODEC.example,
                accepts_multiple_valid_answers=False,
            ),
            render_profile="ball_bounces_single_arena_v3_supersampled",
        )

    @staticmethod
    def _answer_schema(bounce_count: int) -> Dict[str, Any]:
        return {
            "schema_version": "ball_bounces_option_v1",
            "kind": "option_label",
            "parser_profile": "strict_abcd_answer",
            "format_template": FOUR_CHOICE_CODEC.schema,
            "format_example": FOUR_CHOICE_CODEC.example,
            "judged_artifacts": {
                "ti2v": ["full_video"],
            },
            "accepts_multiple_valid_answers": False,
            "extra_text_allowed": False,
        }

    @classmethod
    def _verifier_contract(cls) -> Dict[str, Any]:
        return {
            "profile": cls.verifier_profile,
            "video_profile": cls.video_verifier_profile,
            "semantic_source": "visible start ball, direction arrow, arena center bounds, candidate markers, and prompt collision count",
            "strict_binary_correctness": True,
            "strict_text_parser": FOUR_CHOICE_CODEC.full_pattern,
            "untrusted_redundant_fields": [
                "correct_option",
                "true_coordinate",
                "trace",
                "final_point",
                "candidate_event_counts",
                "candidate_kinds",
                "semantic_state_trace",
            ],
        }

    def _prompts(self, bounce_count: int) -> Tuple[VisualReasoningContract, Dict[str, str]]:
        contract = self._contract(bounce_count)
        prompts = {
            key: contract.build_prompt(mode)
            for key, mode in (("ti2v_prompt", "ti2v"),
                ("solution_image_prompt", "solution_image"),
            )
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"invalid ball-bounces prompt contract: {'; '.join(issues)}")
        return contract, prompts

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        contract, prompts = self._prompts(int(task["bounce_count"]))
        trace: List[Point] = list(task["trace"])
        puzzle_image = self._draw_frame(task, path_points=[trace[0]], current_position=trace[0])
        solution_image = self._draw_frame(task, path_points=trace, current_position=trace[-1])

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        video_path: Path | None = None
        video_num_frames: int | None = None
        if self.generate_videos:
            candidate_video_path = sample_dir / f"{sample_id}__solution.webm"
            encoded_frames = self._render_video(task, output_path=candidate_video_path)
            if encoded_frames:
                video_path = candidate_video_path
                video_num_frames = encoded_frames

        candidates = [
            {
                "label": label,
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "collision_count": int(event_count) if event_count is not None else None,
                "candidate_kind": candidate_kind,
            }
            for label, point, event_count, candidate_kind in zip(
                _LABELS,
                task["candidates"],
                task["candidate_event_counts"],
                task["candidate_kinds"],
            )
        ]
        render_geometry = {
            "image_size": [512, 512],
            "arena_margin": int(task["margin"]),
            "ball_radius": int(task["radius"]),
            "center_bounds": [round(float(value), 3) for value in task["center_bounds"]],
            "trace_rgb": list(_TRACE),
            "ball_rgb": list(_BALL_FILL),
            "arrow_rgb": list(_ARROW),
            "candidate_fill_rgb": list(_CANDIDATE_FILL),
            "candidate_outline_rgb": list(_CANDIDATE_OUTLINE),
        }
        source_evidence = {
            "start": _rounded_point(task["start"]),
            "initial_velocity": _rounded_point(task["velocity"]),
            "bounce_count": int(task["bounce_count"]),
            "center_bounds": [round(float(value), 3) for value in task["center_bounds"]],
            "candidates": [
                {"label": candidate["label"], "x": candidate["x"], "y": candidate["y"]}
                for candidate in candidates
            ],
        }
        semantic_state_trace = [
            {
                "collision_index": index,
                "ball_center": _rounded_point(point),
                "route_prefix": [_rounded_point(prefix_point) for prefix_point in trace[: index + 1]],
            }
            for index, point in enumerate(trace)
        ]
        structured_answer = {
            "task_mechanic": self.task_mechanic,
            "source_evidence": source_evidence,
            "answer_schema": self._answer_schema(int(task["bounce_count"])),
            "verifier_contract": self._verifier_contract(),
            "media_contract": {
                "canvas": [512, 512],
                "video_fps": 16,
                "video_frame_count_rule": "4N+1",
                "maximum_video_frames": 157,
                "single_input": True,
            },
            "render_geometry": render_geometry,
            "correct_option": task["correct_option"],
            "candidates": candidates,
            "true_coordinate": {
                "x": round(float(task["final_point"][0]), 3),
                "y": round(float(task["final_point"][1]), 3),
            },
            "bounce_count": int(task["bounce_count"]),
            "start": _rounded_point(task["start"]),
            "initial_velocity": _rounded_point(task["velocity"]),
            "trace": [_rounded_point(point) for point in trace],
            "final_point": _rounded_point(task["final_point"]),
            "arena_margin": int(task["margin"]),
            "ball_radius": int(task["radius"]),
            "center_bounds": [round(float(value), 3) for value in task["center_bounds"]],
            "collision_rule": "elastic_axis_component_reflection",
            "corner_policy": "reverse_both_components_count_once",
            "multiple_solution_policy": "unique_candidate_required",
            "generation_attempts": int(task["generation_attempts"]),
            "semantic_state_trace": semantic_state_trace,
        }

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_mechanic": self.task_mechanic,
            "task_contract": contract.to_dict(),
            "question": f"Which marker contains the ball center immediately after wall collision {task['bounce_count']}?",
            "answer": task["correct_option"],
            "correct_option": task["correct_option"],
            "structured_answer": structured_answer,
            **prompts,
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [puzzle_path.as_posix()],
            "option_images": [],
            "option_image_paths": [],
            "option_labels": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": video_path.as_posix() if video_path is not None else None,
            "video_fps": self.video_fps if video_path is not None else None,
            "video_num_frames": video_num_frames,
        }
        return sample, puzzle_image, solution_image


__all__ = ["BallBouncesGivenTimePattern"]
