"""Graduated-container pouring: shared physics-family implementation.

Numbered open containers stand side by side, each with unit tick marks and a
starting water level.  A stated sequence of pours runs in order; one pour
moves water from its source into its target until the source is empty or the
target is full, whichever comes first, and the excess stays behind.  All
quantities are whole units, so the terminal levels are exactly readable and
exactly typeable.
"""

from __future__ import annotations

from data.common.assets import render_font

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
from data.physics.core.source_replay import carrier_column_samples


BG = (246, 248, 250)
CARD = (252, 253, 254)
CARD_EDGE = (216, 221, 226)
INK = (40, 47, 56)
GLASS = (108, 122, 136)
WATER = (96, 156, 214)
WATER_TOP = (58, 118, 182)
TICK = (150, 160, 170)
STREAM = (96, 156, 214)

_UNIT_PX = 34.0
_MIN_CAP, _MAX_CAP = 4, 7
_ORDINALS = ("first", "second", "third", "fourth")


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def simulate_pours(
    capacities: Sequence[int],
    levels: Sequence[int],
    pours: Sequence[Sequence[int]],
) -> tuple[List[List[int]], List[int]]:
    """Replay the pour schedule; returns (levels-after-each-pour, amounts)."""

    state = [int(value) for value in levels]
    history: List[List[int]] = []
    amounts: List[int] = []
    for source, target in pours:
        source_index, target_index = int(source) - 1, int(target) - 1
        if source_index == target_index:
            raise ValueError("a pour cannot target its own source")
        moved = min(state[source_index], int(capacities[target_index]) - state[target_index])
        moved = max(0, moved)
        state[source_index] -= moved
        state[target_index] += moved
        amounts.append(moved)
        history.append(list(state))
    return history, amounts


class WaterPourLevelsPattern(PhysicalScenePattern):
    task_name = "water_pour_levels"
    task_family = "physics"
    task_mechanic = "bounded_transfer_pour_sequence"
    answer_overlay_mode = "none"
    verifier_profile = "water_pour_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "water_pour_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def _sample_task(self) -> Dict[str, Any]:
        shapes = {
            "min": (3, 2),
            "typical": (3, 3),
            "max": (4, 4),
        }.get(self.difficulty or "")
        if shapes is None:
            shapes = self.rng.choice(((3, 2), (3, 3), (4, 3), (4, 4)))
        count, pour_count = shapes
        for _ in range(4000):
            task = self._try_scene(count, pour_count)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid pour schedule")

    def _try_scene(self, count: int, pour_count: int) -> Dict[str, Any] | None:
        rng = self.rng
        capacities = [rng.randint(_MIN_CAP, _MAX_CAP) for _ in range(count)]
        levels = [rng.randint(0, cap - 1) for cap in capacities]
        if sum(levels) < 3:
            return None
        pours: List[List[int]] = []
        previous: tuple[int, int] | None = None
        for _ in range(pour_count):
            source = rng.randint(1, count)
            target = rng.randint(1, count)
            if source == target:
                return None
            if previous == (source, target):
                return None
            previous = (source, target)
            pours.append([source, target])
        history, amounts = simulate_pours(capacities, levels, pours)
        # Every pour must actually move water (no dead steps).
        if any(amount < 1 for amount in amounts):
            return None
        # The capacity clip must bite at least once (target fills, residue
        # stays) and the source-empty stop at least once, so both halves of
        # the stop rule are load-bearing.
        clipped = empties = False
        state = list(levels)
        for (source, target), amount in zip(pours, amounts):
            source_index, target_index = source - 1, target - 1
            space = capacities[target_index] - state[target_index]
            if state[source_index] > space:
                clipped = True
            if state[source_index] <= space and state[source_index] == amount:
                empties = True
            state[source_index] -= amount
            state[target_index] += amount
        if not (clipped and empties):
            return None
        if history[-1] == list(levels):
            return None
        return {
            "capacities": capacities,
            "levels": levels,
            "pours": pours,
        }

    # ------------------------------------------------------- geometry derivation

    @staticmethod
    def _scene(task: Mapping[str, Any]) -> Dict[str, Any]:
        capacities = [int(value) for value in task["capacities"]]
        count = len(capacities)
        width = 74.0 if count == 3 else 64.0
        gap = (512.0 - count * width) / (count + 1)
        floor_y = 398.0
        containers = []
        for index, capacity in enumerate(capacities):
            left = gap + index * (width + gap)
            containers.append(
                {
                    "left": left,
                    "right": left + width,
                    "floor": floor_y,
                    "top": floor_y - capacity * _UNIT_PX,
                    "capacity": capacity,
                }
            )
        return {"containers": containers, "unit_px": _UNIT_PX, "floor": floor_y}

    @classmethod
    def progress_carriers(
        cls, task: Mapping[str, Any]
    ) -> List[Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]]:
        """Per container, its water level - which moves only as a pour says.

        A container's level is not one-way over the whole schedule, because the
        same container can receive water in one pour and give it away in the
        next.  What the rule does fix is that it never moves against the pour
        currently running: during a pour the source falls and the target rises,
        between pours nothing moves at all.  So the level travels exactly the
        distance the schedule's amounts add up to, and a candidate that rewinds,
        repeats or reorders frames moves it further than that - which is the
        gate this carrier is read for.

        Each carrier reads two columns inside one container, placed clear of the
        wall, of the tick marks along the left edge, and of the pour stream that
        falls down the container's centre, so the reading is that container's own
        water and nothing else.
        """

        geometry = cls._scene(task)
        carriers: List[
            Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]
        ] = []
        for index, container in enumerate(geometry["containers"], start=1):
            width = float(container["right"]) - float(container["left"])
            carriers.append(
                (
                    f"container_{index}",
                    carrier_column_samples(
                        (
                            float(container["left"]) + 0.28 * width,
                            float(container["left"]) + 0.72 * width,
                        ),
                        float(container["top"]),
                        float(container["floor"]) - 3.0,
                    ),
                    CARD,
                    None,
                    0,
                )
            )
        return carriers

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        geometry = cls._scene(task)
        boxes = []
        for index, container in enumerate(geometry["containers"], start=1):
            boxes.append(
                (
                    f"container_{index}",
                    (
                        container["left"] - 4.0,
                        container["top"] - 6.0,
                        container["right"] + 4.0,
                        container["floor"] + 4.0,
                    ),
                )
            )
        return boxes

    # ---------------------------------------------------------------- semantics

    def _final_levels(self, task: Mapping[str, Any]) -> List[int]:
        history, _amounts = simulate_pours(task["capacities"], task["levels"], task["pours"])
        return history[-1]

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return format_nonnegative_integer_list_payload(self._final_levels(task))

    def _question(self, task: Mapping[str, Any]) -> str:
        return "What is each container's water level after all pours?"

    def _pour_schedule_text(self, task: Mapping[str, Any]) -> str:
        clauses = [
            f"{_ORDINALS[index]} pour container {source} into container {target}"
            for index, (source, target) in enumerate(task["pours"])
        ]
        return "Then perform exactly these pours in order: " + "; ".join(clauses) + "."

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        capacities = [int(value) for value in task["capacities"]]
        history, amounts = simulate_pours(capacities, task["levels"], task["pours"])
        boundaries = self._time_boundaries(task)
        states: List[tuple[float, str]] = []
        state = [int(value) for value in task["levels"]]
        for index, ((source, target), amount) in enumerate(zip(task["pours"], amounts)):
            source_level = state[source - 1]
            target_level = state[target - 1]
            space = capacities[target - 1] - target_level
            if amount == space and source_level > space:
                stop = (
                    f"container {target} reaches its top tick first, so only {amount} unit{'s' if amount != 1 else ''} move and {source_level - amount} stay behind"
                )
            elif amount == source_level:
                stop = (
                    f"container {source} runs empty after {amount} unit{'s' if amount != 1 else ''}, all of which fit"
                )
            else:
                stop = f"{amount} units move"
            text = (
                f"The {_ORDINALS[index]} pour sends water from container {source} (level {source_level}) into container {target} "
                f"(level {target_level} of {capacities[target - 1]}): {stop}. "
                f"Levels now read {', '.join(str(v) for v in history[index])}."
            )
            if index == len(task["pours"]) - 1:
                text += (
                    " No pour remains, so the final levels are "
                    + ", ".join(str(v) for v in history[-1])
                    + "."
                )
            states.append((boundaries[index + 1], text))
            state = list(history[index])
        return states

    def _time_boundaries(self, task: Mapping[str, Any]) -> List[float]:
        _history, amounts = simulate_pours(task["capacities"], task["levels"], task["pours"])
        pause = 0.6
        total = sum(amounts) + pause * (len(amounts) - 1)
        boundaries = [0.0]
        elapsed = 0.0
        for index, amount in enumerate(amounts):
            elapsed += amount
            boundaries.append(elapsed / total)
            if index < len(amounts) - 1:
                elapsed += pause
        boundaries[-1] = 1.0
        return boundaries

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        frames = max(27, 9 * len(task["pours"]))
        while frames % 4 != 3:
            frames += 1
        return frames

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        history, amounts = simulate_pours(task["capacities"], task["levels"], task["pours"])
        return {
            "semantic_state": {
                "capacities": [int(v) for v in task["capacities"]],
                "levels": [int(v) for v in task["levels"]],
                "pours": [list(p) for p in task["pours"]],
            },
            "answer_schema": {
                "schema_version": "water_pour_levels_v1",
                "kind": "per_container_level_list",
                "parser_profile": "strict_level_list",
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
                "semantic_source": "visible capacities and tick levels plus the stated pour schedule",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: \[([0-9](?:, [0-9])*)\]\.$",
                "untrusted_redundant_fields": [
                    "levels_after_each_pour",
                    "pour_amounts",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "deterministic bounded transfers; exactly one level vector",
            "levels_after_each_pour": history,
            "pour_amounts": amounts,
            "semantic_state_trace": [
                {"pour": index + 1, "from": source, "to": target, "amount": amount}
                for index, ((source, target), amount) in enumerate(zip(task["pours"], amounts))
            ],
        }

    # ------------------------------------------------------------------- render

    def _levels_at(self, task: Mapping[str, Any], progress: float) -> tuple[List[float], int | None, float]:
        """Continuous levels, the active pour index, and its local fraction."""

        capacities = [int(value) for value in task["capacities"]]
        history, amounts = simulate_pours(capacities, task["levels"], task["pours"])
        pause = 0.6
        total = sum(amounts) + pause * (len(amounts) - 1)
        time = max(0.0, min(1.0, float(progress))) * total
        state = [float(value) for value in task["levels"]]
        for index, ((source, target), amount) in enumerate(zip(task["pours"], amounts)):
            if time <= 0.0:
                return state, None, 0.0
            if time < amount - 1e-9:
                fraction = time / amount
                state[source - 1] -= amount * fraction
                state[target - 1] += amount * fraction
                return state, index, fraction
            state = [float(v) for v in history[index]]
            time -= amount
            if index < len(amounts) - 1:
                if time < pause - 1e-9:
                    return state, None, 0.0
                time -= pause
        return state, None, 1.0

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        del mark_answer  # terminal state is the settled levels themselves
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        geometry = self._scene(task)
        levels, active, _fraction = self._levels_at(task, progress)

        draw.rounded_rectangle((16, 16, 496, 496), radius=24, fill=CARD, outline=CARD_EDGE, width=2)
        tick_font = _font(10 * supersample)
        id_font = _font(15 * supersample)

        containers = geometry["containers"]
        # Stream first (behind the glass walls of the target).
        if active is not None:
            source, target = task["pours"][active]
            source_box = containers[source - 1]
            target_box = containers[target - 1]
            start = ((source_box["left"] + source_box["right"]) / 2.0, source_box["top"] - 8.0)
            end = ((target_box["left"] + target_box["right"]) / 2.0, target_box["top"] - 2.0)
            mid = ((start[0] + end[0]) / 2.0, min(start[1], end[1]) - 34.0)
            previous = start
            for step in range(1, 13):
                t = step / 12.0
                point = (
                    (1 - t) * (1 - t) * start[0] + 2 * (1 - t) * t * mid[0] + t * t * end[0],
                    (1 - t) * (1 - t) * start[1] + 2 * (1 - t) * t * mid[1] + t * t * end[1],
                )
                draw.line((*previous, *point), fill=STREAM, width=4)
                previous = point
            target_level_y = target_box["floor"] - levels[target - 1] * _UNIT_PX
            draw.line(
                (end[0], end[1], end[0], target_level_y - 2.0),
                fill=STREAM,
                width=4,
            )

        for index, container in enumerate(containers, start=1):
            left, right = container["left"], container["right"]
            floor_y, top = container["floor"], container["top"]
            level = levels[index - 1]
            surface = floor_y - level * _UNIT_PX
            if level > 0.045:
                draw.rectangle(
                    (left + 2.0, min(surface, floor_y - 2.0), right - 2.0, floor_y - 1.0),
                    fill=WATER,
                )
                draw.line((left + 2.0, surface, right - 2.0, surface), fill=WATER_TOP, width=3)
            # Glass: open-top U shape.
            draw.line((left, top - 6.0, left, floor_y), fill=GLASS, width=4)
            draw.line((right, top - 6.0, right, floor_y), fill=GLASS, width=4)
            draw.line((left - 2.0, floor_y, right + 2.0, floor_y), fill=GLASS, width=5)
            for unit in range(1, container["capacity"] + 1):
                y = floor_y - unit * _UNIT_PX
                draw.line((left, y, left + 9.0, y), fill=TICK, width=2)
                draw.text((left - 12.0, y), str(unit), anchor="rm", fill=INK, font=tick_font)
            draw.text(((left + right) / 2.0, floor_y + 22.0), str(index), anchor="mm", fill=INK, font=id_font)
        return finish_scaled_canvas(image, (512, 512))

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        count = len(task["capacities"])
        return VisualReasoningContract(
            input_description=(
                f"The single image shows {count} numbered open containers on a shelf. Each container's side wall carries small tick marks with unit numbers counted from the bottom, "
                "its blue fill reaches its current whole-unit water level, and the number below its base is its identity."
            ),
            symbol_legend=(
                "A container's capacity is its highest tick number, and every water amount is a whole number of units.",
                "Container numbers are identities only; the pour schedule below refers to them.",
            ),
            rules=(
                "A pour moves water from its source container into its target container until the source is empty or the target's level reaches its top tick, whichever happens first; any excess simply stays in the source.",
                "No water is lost or created, and containers never move.",
                self._pour_schedule_text(task),
            ),
            modality_outputs={
                "ti2v": (
                    "Hold the input, then run the pours in order: during a pour a narrow blue stream arcs from the source's rim into the target while the source level falls and the target level rises together; the stream is absent between pours. "
                    "After the last pour, hold the settled levels."
                ),
                "solution_image": (
                    "Render the containers with each blue fill at its final whole-unit level and everything else unchanged."
                ),
            },
            preservation_constraints=(
                "Keep the containers, tick marks, numbers, palette, and shelf fixed; only the water fills and the transient pour stream change.",
            ),
            transition_constraints=(
                "Change levels continuously and simultaneously in source and target during each pour, in schedule order; never skip a pour, overfill past the top tick, or move water without the stream.",
            ),
            ti2v_continuity={
                "moving_nouns": "the rising and falling water fills and the transient pour stream",
                "static_nouns": "the containers, tick marks, unit numbers, container numbers, shelf, and camera",
                "hold_noun": "the settled final levels",
            },
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template=NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                parser_profile="strict_level_list",
                verifier_profile=self.verifier_profile,
                format_example=NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
            ),
            render_profile="graduated_containers_pour_v1",
        )


__all__ = ["WaterPourLevelsPattern", "simulate_pours"]
