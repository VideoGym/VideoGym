from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.next_state_prediction.core.rendering import (
    discrete_video_frames,
    finish_state_canvas,
    new_state_canvas,
    scaled_font,
)
from data.common.typed_answers import (
    canonical_json_dumps,
    format_canonical_json_answer,
)


pattern_instructions = {
    "traffic_light_countdown": (
        "Four indexed traffic lights independently follow the explicit cycle green -> yellow -> red -> green. "
        "Each card shows its own color durations and an eight-cell remaining-time meter. Simulate one second at a time "
        "for the stated horizon and return every final color and countdown."
    )
}

VIDEOGEN_INSTRUCTION_COMMON = ""

_DIRECTIONS = ("N", "S", "E", "W")
_CYCLE = ("green", "yellow", "red")
_NEXT_STATE = {"green": "yellow", "yellow": "red", "red": "green"}
_CANVAS = (512, 512)
_BG = (246, 248, 247)
_CARD = (255, 255, 253)
_INK = (34, 40, 46)
_MUTED = (88, 96, 103)
_ROAD = (67, 72, 78)
_ROAD_MARK = (235, 238, 237)
_OFF = (209, 214, 216)
_LIGHT_COLORS = {
    "red": (224, 79, 73),
    "yellow": (222, 187, 55),
    "green": (77, 166, 104),
}
_ANCHORS = {
    "N": (256, 112),
    "S": (256, 400),
    "E": (400, 256),
    "W": (112, 256),
}
_LIGHT_X_OFFSETS = {"red": -28, "yellow": 0, "green": 28}
_LIGHT_Y_OFFSET = -19
_COUNTDOWN_METER_Y_OFFSET = 10
_COUNTDOWN_METER_SEGMENTS = 8
_COUNTDOWN_METER_LEFT = -27
_COUNTDOWN_METER_STEP = 7
_COUNTDOWN_METER_WIDTH = 5
_DURATION_Y_OFFSET = 40
_DURATION_X_OFFSETS = {"red": -28, "yellow": 0, "green": 28}
_METER_EMPTY = (231, 235, 234)
_METER_OUTLINE = (171, 181, 181)

_DIGIT_SEGMENTS = {
    0: frozenset("abcdef"),
    1: frozenset("bc"),
    2: frozenset("abdeg"),
    3: frozenset("abcdg"),
    4: frozenset("bcfg"),
    5: frozenset("acdfg"),
    6: frozenset("acdefg"),
    7: frozenset("abc"),
    8: frozenset("abcdefg"),
    9: frozenset("abcdfg"),
}


def _digit_segment_centers(center_x: float, center_y: float, *, scale: float = 1.0) -> Dict[str, Tuple[float, float]]:
    return {
        "a": (center_x, center_y - 8 * scale),
        "b": (center_x + 5 * scale, center_y - 4 * scale),
        "c": (center_x + 5 * scale, center_y + 4 * scale),
        "d": (center_x, center_y + 8 * scale),
        "e": (center_x - 5 * scale, center_y + 4 * scale),
        "f": (center_x - 5 * scale, center_y - 4 * scale),
        "g": (center_x, center_y),
    }


def _state_duration(state: str, durations: Dict[str, int]) -> int:
    return int(durations[state])


def _advance_light(state: str, countdown: int, durations: Dict[str, int]) -> Tuple[str, int]:
    if countdown > 1:
        return state, countdown - 1
    next_state = _NEXT_STATE[state]
    return next_state, _state_duration(next_state, durations)


_ANSWER_FORMAT_EXAMPLE = {
    "E": {"countdown": 99, "state": "yellow"},
    "N": {"countdown": 99, "state": "green"},
    "S": {"countdown": 99, "state": "red"},
    "W": {"countdown": 99, "state": "red"},
}


def _light_answer_payload(
    final_states: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_direction = {str(entry["direction"]): entry for entry in final_states}
    if set(by_direction) != set(_DIRECTIONS):
        raise ValueError("traffic answer requires exactly one N, S, E, and W light")
    return {
        direction: {
            "countdown": int(by_direction[direction]["countdown"]),
            "state": str(by_direction[direction]["state"]),
        }
        for direction in _DIRECTIONS
    }


class TrafficLightCountdownPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "traffic_light_countdown"
    task_family = "next_state_prediction"

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        video: bool | None = None,
        generate_videos: bool = True,
        video_fps: int = 16,
        image_size: int | Tuple[int, int] = _CANVAS,
        target_size: Tuple[int, int] = _CANVAS,
        unique: bool = True,
        **_kwargs: Any,
    ) -> None:
        del image_size, target_size, unique
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / "videogym_traffic_light_countdown"
        super().__init__(resolved)
        self.rng = random.Random(seed if seed is not None else random_seed)
        self.video = bool(generate_videos if video is None else video)
        self.generate_videos = self.video
        self.video_fps = int(video_fps)
        self.image_size = _CANVAS
        self._sample_index = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _generate_task(self) -> Dict[str, Any]:
        for _attempt in range(200):
            simulate_seconds = self.rng.randint(6, 10)
            initial: List[Dict[str, Any]] = []
            for direction in _DIRECTIONS:
                durations = {
                    "red": self.rng.randint(5, 8),
                    "yellow": self.rng.randint(2, 3),
                    "green": self.rng.randint(5, 8),
                }
                state = self.rng.choice(_CYCLE)
                countdown = self.rng.randint(1, durations[state])
                initial.append(
                    {
                        "direction": direction,
                        "state": state,
                        "visible_state": state,
                        "countdown": countdown,
                        "durations": durations,
                    }
                )
            timeline: List[List[Dict[str, Any]]] = [[dict(entry, durations=dict(entry["durations"])) for entry in initial]]
            for _second in range(simulate_seconds):
                next_state: List[Dict[str, Any]] = []
                for entry in timeline[-1]:
                    state, countdown = _advance_light(
                        str(entry["state"]),
                        int(entry["countdown"]),
                        dict(entry["durations"]),
                    )
                    next_state.append(
                        {
                            "direction": entry["direction"],
                            "state": state,
                            "visible_state": state,
                            "countdown": countdown,
                            "durations": dict(entry["durations"]),
                        }
                    )
                timeline.append(next_state)
            transition_count = sum(
                timeline[index - 1][light_index]["state"] != timeline[index][light_index]["state"]
                for index in range(1, len(timeline))
                for light_index in range(len(_DIRECTIONS))
            )
            if transition_count >= 4 and len({entry["state"] for entry in timeline[-1]}) >= 2:
                return {
                    "initial_states": timeline[0],
                    "timeline": timeline,
                    "simulate_seconds": simulate_seconds,
                    "final_states": timeline[-1],
                    "cycle": list(_CYCLE),
                    "direction_order": list(_DIRECTIONS),
                    "transition_count": transition_count,
                    "render_geometry": {
                        "anchors": {key: list(value) for key, value in _ANCHORS.items()},
                        "light_x_offsets": dict(_LIGHT_X_OFFSETS),
                        "light_y_offset": _LIGHT_Y_OFFSET,
                        "countdown_meter_y_offset": _COUNTDOWN_METER_Y_OFFSET,
                        "countdown_meter_segments": _COUNTDOWN_METER_SEGMENTS,
                        "countdown_meter_left": _COUNTDOWN_METER_LEFT,
                        "countdown_meter_step": _COUNTDOWN_METER_STEP,
                        "countdown_meter_width": _COUNTDOWN_METER_WIDTH,
                        "duration_y_offset": _DURATION_Y_OFFSET,
                        "duration_x_offsets": dict(_DURATION_X_OFFSETS),
                    },
                }
        raise RuntimeError("traffic_light_countdown: failed to sample a transition-rich timeline")

    def _contract(self, simulate_seconds: int) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The supplied image shows one fixed four-way crossroad with four independent cards labeled N, S, E, and W. "
                "Each card has red, yellow, and green lamps, an eight-cell remaining-time meter, and fixed R/Y/G duration digits."
            ),
            symbol_legend=(
                "Exactly one colored lamp is active on each card; gray lamps are off.",
                "The number of filled colored meter cells is the seconds remaining in the active color; empty meter cells are pale gray.",
                "The three small R/Y/G digits are that light's reset durations and never change.",
            ),
            rules=(
                "Each light independently follows green -> yellow -> red -> green.",
                "During one simulated second, a countdown above 1 decreases by exactly 1 without changing color.",
                "When a countdown is 1, the next second changes to the next color and resets to that card's displayed duration for the new color.",
                f"Apply one simultaneous update to all four lights per second for exactly {simulate_seconds} seconds.",
            ),
            modality_outputs={
                "ti2v": (
                    f"Show second 0, then commit every simultaneous one-second state through second {simulate_seconds} "
                    "in order as one complete crossroad state per second."
                ),
                "solution_image": "Render the exact final crossroad state after the required simultaneous updates.",
            },
            preservation_constraints=(
                "Keep the crossroad, the four cards, the direction labels, the lamp positions, and the R/Y/G duration digits unchanged.",
                "Change only the four active lamps and the filled cells of the four remaining-time meters.",
            ),
            transition_constraints=(
                "Commit each simulated second as one complete state: all four cards show their new lamps and meters together.",
                "Do not partially or asynchronously update a one-second state.",
                "Do not change any R/Y/G duration digit.",
            ),
            ti2v_continuity={
                "timing": "discrete",
                "moving_nouns": "the active lamps and the filled meter cells",
                "static_nouns": "the crossroad, the four cards, the direction labels, the duration digits, the background, and the camera",
                "hold_noun": "the exact final crossroad state",
            },
            answer=AnswerContract(
                kind="json_object",
                format_template="Answer: <canonical JSON object>.",
                parser_profile="canonical_json_object",
                verifier_profile="traffic_light_simultaneous_rollout_replay",
                format_example=format_canonical_json_answer(_ANSWER_FORMAT_EXAMPLE),
                accepts_multiple_valid_answers=False,
            ),
            render_profile="four_card_traffic_pip_meter_v4_complete_footprint_fade",
        )

    @staticmethod
    def _draw_digit(
        draw: ImageDraw.ImageDraw,
        center_x: float,
        center_y: float,
        digit: int,
        *,
        color: Tuple[int, int, int] = _INK,
        scale: float = 1.0,
    ) -> None:
        centers = _digit_segment_centers(center_x, center_y, scale=scale)
        thickness = max(2, int(round(3 * scale)))
        half = 4 * scale
        for segment in _DIGIT_SEGMENTS[int(digit)]:
            x, y = centers[segment]
            if segment in {"a", "d", "g"}:
                draw.line((x - half, y, x + half, y), fill=color, width=thickness)
            else:
                draw.line((x, y - half, x, y + half), fill=color, width=thickness)

    @staticmethod
    def _draw_countdown_meter(
        draw: ImageDraw.ImageDraw,
        center_x: float,
        center_y: float,
        countdown: int,
        state: str,
    ) -> None:
        value = int(countdown)
        if not 1 <= value <= _COUNTDOWN_METER_SEGMENTS:
            raise ValueError(f"traffic countdown {value} is outside the visible meter range")
        draw.rounded_rectangle(
            (center_x - 31, center_y - 10, center_x + 31, center_y + 10),
            radius=6,
            fill=(247, 249, 248),
            outline=(178, 187, 189),
            width=1,
        )
        for index in range(_COUNTDOWN_METER_SEGMENTS):
            x0 = center_x + _COUNTDOWN_METER_LEFT + index * _COUNTDOWN_METER_STEP
            draw.rounded_rectangle(
                (x0, center_y - 6, x0 + _COUNTDOWN_METER_WIDTH, center_y + 6),
                radius=2,
                fill=_LIGHT_COLORS[state] if index < value else _METER_EMPTY,
                outline=_METER_OUTLINE,
                width=1,
            )

    @classmethod
    def _draw_light_card(cls, draw: ImageDraw.ImageDraw, entry: Dict[str, Any]) -> None:
        direction = str(entry["direction"])
        anchor_x, anchor_y = _ANCHORS[direction]
        draw.rounded_rectangle(
            (anchor_x - 52, anchor_y - 58, anchor_x + 52, anchor_y + 58),
            radius=16,
            fill=_CARD,
            outline=(183, 199, 201),
            width=2,
        )
        draw.text((anchor_x, anchor_y - 48), direction, font=scaled_font(14), fill=_INK, anchor="mm")
        light_y = anchor_y + _LIGHT_Y_OFFSET
        for state in ("red", "yellow", "green"):
            center_x = anchor_x + _LIGHT_X_OFFSETS[state]
            fill = _LIGHT_COLORS[state] if entry["state"] == state else _OFF
            draw.ellipse((center_x - 10, light_y - 10, center_x + 10, light_y + 10), fill=fill, outline=(84, 91, 95), width=2)
        countdown_y = anchor_y + _COUNTDOWN_METER_Y_OFFSET
        cls._draw_countdown_meter(
            draw,
            anchor_x,
            countdown_y,
            int(entry["countdown"]),
            str(entry["state"]),
        )
        durations = dict(entry["durations"])
        duration_y = anchor_y + _DURATION_Y_OFFSET
        for state, label in (("red", "R"), ("yellow", "Y"), ("green", "G")):
            digit_x = anchor_x + _DURATION_X_OFFSETS[state]
            draw.text((digit_x - 8, duration_y), label, font=scaled_font(8), fill=_LIGHT_COLORS[state], anchor="mm")
            cls._draw_digit(draw, digit_x + 3, duration_y, int(durations[state]), scale=0.72)

    def _render_scene(self, task: Dict[str, Any], step_index: int) -> Image.Image:
        image, draw = new_state_canvas(_BG, (255, 255, 253), (216, 221, 218), card_box=(14, 14, 498, 498))
        road_half = 49
        draw.rectangle((26, 256 - road_half, 486, 256 + road_half), fill=_ROAD)
        draw.rectangle((256 - road_half, 60, 256 + road_half, 452), fill=_ROAD)
        for offset in range(35, 478, 34):
            draw.line((offset, 256, min(offset + 18, 486), 256), fill=_ROAD_MARK, width=2)
        for offset in range(68, 444, 34):
            draw.line((256, offset, 256, min(offset + 18, 452)), fill=_ROAD_MARK, width=2)
        snapshot = {str(entry["direction"]): entry for entry in task["timeline"][step_index]}
        for direction in _DIRECTIONS:
            self._draw_light_card(draw, snapshot[direction])
        return finish_state_canvas(image)

    def _render_video(self, task: Dict[str, Any], *, output_path: Path) -> Tuple[str | None, int | None]:
        if not self.video:
            return None, None
        renders = [self._render_scene(task, index) for index in range(len(task["timeline"]))]
        frames = [np.asarray(image, dtype=np.uint8) for image in discrete_video_frames(renders)]
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def _make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        answer_payload = _light_answer_payload(task["final_states"])
        answer_raw = canonical_json_dumps(answer_payload)
        contract = self._contract(int(task["simulate_seconds"]))
        prompts = {f"{mode}_prompt": contract.build_prompt(mode) for mode in contract.prompt_modes()}
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"Invalid traffic-light prompt contract: {'; '.join(issues)}")

        puzzle_image = self._render_scene(task, 0)
        solution_image = self._render_scene(task, len(task["timeline"]) - 1)
        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        puzzle_dir = sample_dir / "puzzles"
        solution_dir = sample_dir / "solutions"
        step_dir = sample_dir / "steps"
        video_dir = sample_dir / "videos"
        for directory in (puzzle_dir, solution_dir, step_dir, video_dir):
            directory.mkdir(parents=True, exist_ok=True)
        puzzle_path = puzzle_dir / "puzzle.png"
        solution_path = solution_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        # The four lights are independent, so the authored reasoning walks
        # each light's own countdown chain to the horizon; the only image is
        # the terminal answer image.


        solution_video_path, video_num_frames = self._render_video(task, output_path=video_dir / "solution.webm")
        structured = {
            "simulate_seconds": int(task["simulate_seconds"]),
            "cycle": list(task["cycle"]),
            "direction_order": list(task["direction_order"]),
            "initial_states": task["initial_states"],
            "timeline": task["timeline"],
            "final_states": task["final_states"],
            "terminal_answer": answer_payload,
            "transition_count": int(task["transition_count"]),
            "render_geometry": task["render_geometry"],
        }
        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": f"Simulate all four lights for exactly {task['simulate_seconds']} seconds and return their final states.",
            "answer": answer_raw,
            "structured_answer": structured,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [puzzle_path.as_posix()],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        return sample, puzzle_image, solution_image

    def create_puzzle(self) -> Dict[str, Any]:
        return self._make_sample()[0]

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        return self._make_sample()


__all__ = [
    "TrafficLightCountdownPattern",
    "VIDEOGEN_INSTRUCTION_COMMON",
    "_advance_light",
    "_digit_segment_centers",
    "_light_answer_payload",
]
