"""Equal-mass elastic collisions on a ticked 1D track.

Labeled balls of exactly one tick diameter roll along a straight rail at
stated constant integer speeds.  When two balls touch they instantly exchange
velocities (equal masses) and never pass through each other.  The layout
freezes at the prompt-stated whole-second time T, when every ball
center sits exactly on a tick, so the terminal layout is exactly typeable.
"""

from __future__ import annotations

from data.common.assets import render_font

from fractions import Fraction
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageFont

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.scalar_lists import (
    NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
    NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
    format_nonnegative_integer_list_payload,
)
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.physics.core.physical_scene_common import PhysicalScenePattern


BG = (246, 248, 250)
CARD = (252, 253, 254)
CARD_EDGE = (216, 221, 226)
INK = (40, 47, 56)
RAIL = (96, 108, 120)
TICK = (150, 160, 170)
ARROW = (57, 115, 182)
BALL_COLORS = {
    "A": (214, 105, 78),
    "B": (70, 132, 198),
    "C": (86, 158, 112),
    "D": (166, 118, 190),
}

_TICKS = 12
_TICK_X0, _TICK_DX = 58.0, 36.0
_RAIL_Y = 300.0
_BALL_R = 18.0          # diameter 36px = exactly one tick spacing
_LABELS = ("A", "B", "C", "D")


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def simulate_collisions(
    starts: Sequence[int],
    velocities: Sequence[int],
    horizon: int,
) -> Dict[str, Any]:
    """Exact event-driven replay with rational collision times.

    Balls are hard spheres of diameter one tick: adjacent balls collide when
    their centers are exactly one tick apart, and a collision exchanges the
    two velocities.  Returns per-ball piecewise trajectories, the collision
    events, and the integer terminal positions at ``horizon`` seconds.
    """

    count = len(starts)
    positions = [Fraction(int(value)) for value in starts]
    speeds = [Fraction(int(value)) for value in velocities]
    if sorted(positions) != positions or any(
        b - a < 1 for a, b in zip(positions, positions[1:])
    ):
        raise ValueError("balls must start ordered with at least one tick spacing")
    time = Fraction(0)
    horizon_f = Fraction(int(horizon))
    events: List[Dict[str, Any]] = []
    trajectories: List[List[Tuple[Fraction, Fraction]]] = [
        [(time, position)] for position in positions
    ]
    for _step in range(24):
        best: Tuple[Fraction, int] | None = None
        for index in range(count - 1):
            relative = speeds[index] - speeds[index + 1]
            if relative <= 0:
                continue
            gap = positions[index + 1] - positions[index] - 1
            candidate = time + gap / relative
            if candidate >= horizon_f:
                continue
            if best is None or candidate < best[0]:
                best = (candidate, index)
        if best is None:
            break
        event_time, index = best
        dt = event_time - time
        positions = [p + v * dt for p, v in zip(positions, speeds)]
        time = event_time
        speeds[index], speeds[index + 1] = speeds[index + 1], speeds[index]
        for ball in range(count):
            trajectories[ball].append((time, positions[ball]))
        events.append(
            {
                "time": event_time,
                "left_ball": index,
                "right_ball": index + 1,
                "left_position": positions[index],
                "right_position": positions[index + 1],
            }
        )
    dt = horizon_f - time
    positions = [p + v * dt for p, v in zip(positions, speeds)]
    for ball in range(count):
        trajectories[ball].append((horizon_f, positions[ball]))
    return {
        "events": events,
        "trajectories": trajectories,
        "final_positions": positions,
        "final_speeds": speeds,
    }


class ElasticCollisionTrackPattern(PhysicalScenePattern):
    task_name = "elastic_collision_track"
    task_family = "physics"
    task_mechanic = "equal_mass_velocity_exchange_track"
    answer_overlay_mode = "none"
    verifier_profile = "collision_track_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "collision_track_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def __init__(
        self,
        *args: Any,
        collision_count_range: Sequence[int] | None = None,
        speed_values: Sequence[int] | None = None,
        ball_count: int | None = None,
        horizon_range: Sequence[int] = (3, 6),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.collision_count_range = tuple(map(int, collision_count_range)) if collision_count_range is not None else None
        if self.collision_count_range is not None and not (
            len(self.collision_count_range) == 2 and 1 <= self.collision_count_range[0] <= self.collision_count_range[1] <= 4
        ):
            raise ValueError("collision_count_range must stay within 1..4")
        if ball_count not in (None, 3, 4):
            raise ValueError("ball_count must be 3 or 4")
        self.ball_count = ball_count
        self.speed_values = (
            tuple(speed_values) if speed_values is not None else (-2, -1, 0, 0, 1, 1, 2)
        )
        self.horizon_range = tuple(map(int, horizon_range))
        if len(self.horizon_range) != 2 or not (3 <= self.horizon_range[0] <= self.horizon_range[1] <= 8):
            raise ValueError("horizon_range must stay within 3..8")

    def _sample_task(self) -> Dict[str, Any]:
        shapes = {
            "min": ((3, 1, 2),),
            "typical": ((3, 2, 3), (4, 2, 3)),
            "max": ((4, 3, 4),),
        }.get(self.difficulty or "", ((3, 1, 2), (3, 2, 3), (4, 2, 3), (4, 3, 4)))
        count, min_events, max_events = self.rng.choice(shapes)
        if self.ball_count is not None:
            count = self.ball_count
        if self.collision_count_range is not None:
            min_events, max_events = self.collision_count_range
        for _ in range(8000):
            task = self._try_scene(count, min_events, max_events)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid collision schedule")

    def _try_scene(
        self, count: int, min_events: int, max_events: int
    ) -> Dict[str, Any] | None:
        rng = self.rng
        horizon = rng.randint(*self.horizon_range)
        positions = sorted(rng.sample(range(1, _TICKS + 1), count))
        if any(b - a < 2 for a, b in zip(positions, positions[1:])):
            return None
        velocities = [rng.choice(self.speed_values) for _ in range(count)]
        if all(v == 0 for v in velocities):
            return None
        try:
            replay = simulate_collisions(positions, velocities, horizon)
        except ValueError:
            return None
        events = replay["events"]
        if not min_events <= len(events) <= max_events:
            return None
        # Temporal margins: events well separated and clear of T.
        times = [float(event["time"]) for event in events]
        if any(second - first < 0.3 for first, second in zip(times, times[1:])):
            return None
        if times and times[-1] > horizon - 0.35:
            return None
        if times and times[0] < 0.35:
            return None
        # Track bounds over the whole run (positions are piecewise linear).
        for trajectory in replay["trajectories"]:
            for _t, position in trajectory:
                if not 1 <= float(position) <= _TICKS:
                    return None
        # Terminal positions are integers by construction; require them
        # distinct from the naive no-collision layout for every moving pair.
        final = [int(position) for position in replay["final_positions"]]
        naive = [start + velocity * horizon for start, velocity in zip(positions, velocities)]
        if final == naive:
            return None
        return {
            "starts": positions,
            "velocities": velocities,
            "horizon": horizon,
        }

    # ------------------------------------------------------- geometry derivation

    @staticmethod
    def _tick_x(position: float) -> float:
        return _TICK_X0 + (float(position) - 1.0) * _TICK_DX

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        replay = simulate_collisions(task["starts"], task["velocities"], int(task["horizon"]))
        boxes = []
        for index, position in enumerate(replay["final_positions"]):
            x = cls._tick_x(float(position))
            pad = _BALL_R + 5.0
            boxes.append(
                (
                    f"ball_{_LABELS[index]}",
                    (x - pad, _RAIL_Y - pad, x + pad, _RAIL_Y + pad),
                )
            )
        return boxes

    # ---------------------------------------------------------------- semantics

    def _replay(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return simulate_collisions(task["starts"], task["velocities"], int(task["horizon"]))

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        replay = self._replay(task)
        return format_nonnegative_integer_list_payload([int(position) for position in replay["final_positions"]])

    def _question(self, task: Mapping[str, Any]) -> str:
        return f"On which ticks do the balls sit after exactly {int(task['horizon'])} seconds?"

    @staticmethod
    def _speed_phrase(speed: int) -> str:
        if speed == 0:
            return "stays at rest"
        direction = "right" if speed > 0 else "left"
        return f"moves {direction} at speed {abs(speed)}"

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        replay = self._replay(task)
        horizon = int(task["horizon"])
        events = replay["events"]
        states: List[tuple[float, str]] = []
        speeds = [int(v) for v in task["velocities"]]
        for index, event in enumerate(events):
            left, right = int(event["left_ball"]), int(event["right_ball"])
            new_left, new_right = speeds[right], speeds[left]
            when = float(event["time"])
            left_label, right_label = _LABELS[left], _LABELS[right]
            text = (
                f"At about {when:.3g} seconds, balls {left_label} and {right_label} touch near tick "
                f"{float(event['left_position']) + 0.5:.3g}. They exchange velocities: {left_label} now "
                f"{self._speed_phrase(new_left)} and {right_label} now {self._speed_phrase(new_right)}."
            )
            speeds[left], speeds[right] = new_left, new_right
            states.append((when / horizon, text))
        final = [int(position) for position in replay["final_positions"]]
        summary = ", ".join(f"{label} on tick {position}" for label, position in zip(_LABELS, final))
        states.append(
            (
                1.0,
                f"No further collision happens before {horizon} seconds, so the balls freeze with {summary}.",
            )
        )
        return states

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        frames = max(31, 9 * int(task["horizon"]))
        while frames % 4 != 3:
            frames += 1
        return frames

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        replay = self._replay(task)
        return {
            "semantic_state": {
                "starts": [int(value) for value in task["starts"]],
                "velocities": [int(value) for value in task["velocities"]],
                "horizon": int(task["horizon"]),
            },
            "answer_schema": {
                "schema_version": "elastic_collision_track_v1",
                "kind": "per_ball_tick_list",
                "parser_profile": "strict_tick_list",
                "format_template": NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                "format_example": NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "video_profile": self.video_verifier_profile,
                "semantic_source": "visible tick positions, per-ball speed arrows, and the stated time T",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: \[([0-9]{1,2}(?:, [0-9]{1,2})*)\]\.$",
                "untrusted_redundant_fields": [
                    "events",
                    "final_positions",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "deterministic exchange dynamics; exactly one tick vector",
            "events": [
                {
                    "time": str(event["time"]),
                    "left_ball": _LABELS[int(event["left_ball"])],
                    "right_ball": _LABELS[int(event["right_ball"])],
                }
                for event in replay["events"]
            ],
            "final_positions": [int(position) for position in replay["final_positions"]],
            "semantic_state_trace": [
                f"collision_{index + 1}" for index in range(len(replay["events"]))
            ] + ["freeze_at_T"],
        }

    # ------------------------------------------------------------------- render

    def _positions_at(self, task: Mapping[str, Any], progress: float) -> List[float]:
        replay = self._replay(task)
        horizon = float(int(task["horizon"]))
        time = max(0.0, min(1.0, float(progress))) * horizon
        result = []
        for trajectory in replay["trajectories"]:
            position = float(trajectory[0][1])
            for (t0, p0), (t1, p1) in zip(trajectory, trajectory[1:]):
                f0, f1 = float(t0), float(t1)
                if time >= f1:
                    position = float(p1)
                elif time > f0:
                    position = float(p0) + (float(p1) - float(p0)) * (time - f0) / max(1e-9, f1 - f0)
                    break
                else:
                    break
            result.append(position)
        return result

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        del mark_answer  # the frozen layout at T is the terminal state itself
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        amount = max(0.0, min(1.0, float(progress)))
        positions = self._positions_at(task, amount)
        velocities = [int(value) for value in task["velocities"]]

        draw.rounded_rectangle((16, 16, 496, 496), radius=24, fill=CARD, outline=CARD_EDGE, width=2)
        # Rail and ticks.
        draw.line((_TICK_X0 - 22.0, _RAIL_Y + _BALL_R + 2.0, _TICK_X0 + (_TICKS - 1) * _TICK_DX + 22.0, _RAIL_Y + _BALL_R + 2.0), fill=RAIL, width=4)
        tick_font = _font(11 * supersample)
        for tick in range(1, _TICKS + 1):
            x = self._tick_x(tick)
            draw.line((x, _RAIL_Y + _BALL_R + 2.0, x, _RAIL_Y + _BALL_R - 8.0), fill=TICK, width=2)
            draw.text((x, _RAIL_Y + _BALL_R + 20.0), str(tick), anchor="mm", fill=INK, font=tick_font)

        # Velocity arrows fade out as motion begins.
        arrow_opacity = max(0.0, 1.0 - amount / 0.06)
        label_font = _font(15 * supersample)
        speed_font = _font(12 * supersample)

        def blend(color, opacity):
            return tuple(int(round(CARD[i] + (color[i] - CARD[i]) * opacity)) for i in range(3))

        for index, position in enumerate(positions):
            x = self._tick_x(position)
            label = _LABELS[index]
            color = BALL_COLORS[label]
            draw.ellipse(
                (x - _BALL_R, _RAIL_Y - _BALL_R, x + _BALL_R, _RAIL_Y + _BALL_R),
                fill=color,
                outline=INK,
                width=2,
            )
            draw.text((x, _RAIL_Y), label, anchor="mm", fill=(255, 255, 255), font=label_font)
            speed = velocities[index]
            if speed != 0 and arrow_opacity > 0.02:
                arrow_y = _RAIL_Y - _BALL_R - 18.0
                length = 14.0 * abs(speed)
                direction = 1.0 if speed > 0 else -1.0
                start = (x - direction * length / 2.0, arrow_y)
                end = (x + direction * length / 2.0, arrow_y)
                arrow_color = blend(ARROW, arrow_opacity)
                draw.line((*start, *end), fill=arrow_color, width=3)
                for dy in (-5.0, 5.0):
                    draw.line((end[0], end[1], end[0] - direction * 7.0, end[1] + dy), fill=arrow_color, width=3)
                draw.text(
                    (x, arrow_y - 14.0),
                    str(abs(speed)),
                    anchor="mm",
                    fill=blend(INK, arrow_opacity),
                    font=speed_font,
                )
        return finish_scaled_canvas(image, (512, 512))

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        count = len(task["starts"])
        horizon = int(task["horizon"])
        labels = ", ".join(_LABELS[:count])
        return VisualReasoningContract(
            input_description=(
                f"The single image shows a straight rail with numbered ticks and {count} colored balls labeled {labels} resting on ticks. "
                "The blue arrow above a ball shows its movement direction, and the small number above the arrow is its speed in ticks per second; a ball with no arrow is at rest."
            ),
            symbol_legend=(
                "Each ball is exactly one tick wide, so two balls touch exactly when they sit on adjacent tick positions.",
                "Tick numbers only name positions; ball labels only name balls.",
            ),
            rules=(
                "Every ball moves at its stated constant speed along the rail.",
                "When two balls touch, they instantly exchange their two velocities (equal masses) and never pass through each other; between touches, speeds never change.",
                f"The whole layout freezes after exactly {horizon} seconds, when every ball center is again exactly on a tick.",
            ),
            modality_outputs={
                "ti2v": (
                    "Hold the input while the speed arrows fade away, then move the balls continuously at their stated speeds; at each touch the two balls visibly exchange speeds and separate or push on accordingly. "
                    f"Freeze the layout at {horizon} seconds and hold it."
                ),
                "solution_image": (
                    f"Render the rail with every ball centered on its tick position at {horizon} seconds, with no arrows and nothing else changed."
                ),
            },
            preservation_constraints=(
                "Keep the rail, ticks, tick numbers, ball colors and labels, and the camera fixed; only the balls move and the input arrows fade out once at the start.",
            ),
            transition_constraints=(
                "Move every ball in straight constant-speed motion between touches, exchange velocities exactly at contact, and never let balls overlap, teleport, or change speed away from a contact.",
            ),
            ti2v_continuity={
                "moving_nouns": "the labeled balls",
                "static_nouns": "the rail, ticks, tick numbers, ball labels, and camera",
                "hold_noun": "the frozen layout at the stated time",
            },
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template=NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                parser_profile="strict_tick_list",
                verifier_profile=self.verifier_profile,
                format_example=NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
            ),
            render_profile="elastic_collision_track_v1",
        )


__all__ = ["ElasticCollisionTrackPattern", "simulate_collisions"]
