"""Weir-chamber fill ordering: shared physics-family implementation.

A side-view tank is divided into chambers by interior walls of different
visible heights.  Water pours at a steady rate into the chamber under the
spout; a chamber rises until its level reaches its lowest side wall, spills
over into that neighbor while holding, and merges into one rising pool when
the fed chamber catches up.  The typed answer is the order in which the
chambers first receive water; an amber rank badge commits above a chamber the
moment it turns wet, binding the order to the objects.
"""

from __future__ import annotations

from data.common.assets import render_font

import itertools
from functools import lru_cache
from math import comb
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
WALL = (88, 100, 112)
WATER = (96, 156, 214)
WATER_TOP = (58, 118, 182)
BADGE_FILL = (247, 196, 92)
BADGE_EDGE = (176, 128, 38)
SPOUT = (120, 132, 144)

_FLOOR_Y = 402.0
_RIM_Y = 148.0
_MIN_CREST, _MAX_CREST = 56.0, 212.0    # crest heights above the floor
_CREST_MARGIN = 24.0                     # pairwise crest separation
_WALL_W = 10.0
_BADGE_R = 14.0


def _font(size: int) -> ImageFont.ImageFont:
    return render_font(size, bold=False)


def _draw_rank_badge(draw: Any, x: float, y: float, value: int, opacity: float, supersample: int) -> None:
    def blend(color):
        return tuple(int(round(CARD[i] + (color[i] - CARD[i]) * opacity)) for i in range(3))

    draw.ellipse(
        (x - _BADGE_R, y - _BADGE_R, x + _BADGE_R, y + _BADGE_R),
        fill=blend(BADGE_FILL), outline=blend(BADGE_EDGE), width=2,
    )
    draw.text((x, y), str(value), anchor="mm", fill=blend(INK), font=_font(14 * supersample))


def simulate_fill(
    widths: Sequence[float],
    crests: Sequence[float],
    source: int,
) -> Dict[str, Any]:
    """Event-driven replay of the weir cascade from visible facts only.

    ``widths`` are the chamber widths (area proxies), ``crests`` the interior
    wall heights above the common floor (len = chambers - 1), ``source`` the
    1-based chamber under the spout.  Returns first-wet order, event times,
    per-chamber piecewise-linear level trajectories, and the stop time.
    """

    count = len(widths)
    if not 2 <= count <= 6 or len(crests) != count - 1:
        raise ValueError("chamber shape is invalid")
    if not 1 <= source <= count:
        raise ValueError("spout chamber is out of range")

    def crest_between(a: int, b: int) -> float:
        return float(crests[min(a, b) - 1])

    # Chain of (group, hold_level); the last entry is the receiving group.
    chain: List[Dict[str, Any]] = [{"members": [source], "level": 0.0, "hold": None}]
    wet_order = [source]
    first_wet_times = {source: 0.0}
    time = 0.0
    levels = {chamber: 0.0 for chamber in range(1, count + 1)}
    trajectories: Dict[int, List[Tuple[float, float]]] = {
        chamber: [(0.0, 0.0)] for chamber in range(1, count + 1)
    }
    events: List[Dict[str, Any]] = []

    def record(chamber: int, at: float, level: float) -> None:
        trajectories[chamber].append((at, level))

    for _step in range(64):
        receiving = chain[-1]
        members = receiving["members"]
        area = sum(float(widths[m - 1]) for m in members)
        level = receiving["level"]
        low, high = min(members), max(members)
        candidates: List[Tuple[float, str, int]] = []
        # Spill over an outer crest to a dry neighbor.
        for edge, neighbor in ((low, low - 1), (high, high + 1)):
            if 1 <= neighbor <= count and neighbor not in first_wet_times:
                candidates.append((crest_between(edge, neighbor), "spill", neighbor))
        # Merge upward into the feeder chain.
        if len(chain) >= 2:
            feeder_hold = chain[-2]["hold"]
            candidates.append((float(feeder_hold), "merge", 0))
        if not candidates:
            # Fully merged single pool with every chamber wet: stop.
            break
        target_level, kind, payload = min(candidates)
        dt = (target_level - level) * area
        if dt <= 1e-9:
            raise ValueError("chamber events must strictly advance")
        time += dt
        for member in members:
            levels[member] = target_level
            record(member, time, target_level)
        if kind == "spill":
            receiving["level"] = target_level
            receiving["hold"] = target_level
            chain.append({"members": [payload], "level": 0.0, "hold": None})
            wet_order.append(payload)
            first_wet_times[payload] = time
            record(payload, time, 0.0)
            events.append({"time": time, "kind": "spill", "chamber": payload})
            if len(wet_order) == count:
                # The last dry chamber is now receiving; it fills exactly to
                # its entry crest, merges, and the pour stops.
                area_last = float(widths[payload - 1])
                dt_last = target_level * area_last
                time += dt_last
                levels[payload] = target_level
                record(payload, time, target_level)
                events.append({"time": time, "kind": "final_merge", "chamber": payload})
                return {
                    "wet_order": wet_order,
                    "first_wet_times": first_wet_times,
                    "events": events,
                    "trajectories": trajectories,
                    "total_time": time,
                    "final_levels": dict(levels),
                }
        else:
            merged = chain.pop()
            # The feeder held flat until this instant; give it an explicit
            # sample so interpolation never shows it creeping upward early.
            for member in chain[-1]["members"]:
                record(member, time, target_level)
                levels[member] = target_level
            chain[-1]["members"] = sorted(chain[-1]["members"] + merged["members"])
            chain[-1]["level"] = target_level
            chain[-1]["hold"] = None
            events.append({"time": time, "kind": "merge", "chamber": 0})
    raise ValueError("chamber cascade did not terminate")


@lru_cache(maxsize=None)
def _orders_by_crest_ranking(
    count: int, source: int
) -> Tuple[Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]], ...]:
    """Every first-wet order this tank can produce, with the rankings that do it.

    The cascade only ever compares crest heights against each other, so the
    order is a function of their ranking alone: widths and absolute heights
    change the timing and nothing else (verified over 400 random
    rankings).  Enumerating the rankings therefore enumerates the answer space
    exactly, which is what lets the sampler draw the answer uniformly instead
    of inheriting whatever bias random crest heights happen to carry - a bias
    a crest-blind side-first guess could otherwise exploit.
    """

    groups: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    unit_widths = [1.0] * count
    for ranking in itertools.permutations(range(count - 1)):
        crests = [_MIN_CREST + _CREST_MARGIN * (rank + 1) for rank in ranking]
        try:
            order = tuple(simulate_fill(unit_widths, crests, source)["wet_order"])
        except ValueError:
            continue
        groups.setdefault(order, []).append(ranking)
    return tuple((order, tuple(rankings)) for order, rankings in sorted(groups.items()))


class ChamberFlowFillPattern(PhysicalScenePattern):
    @staticmethod
    def numeric_glyph_templates(maximum: int = 9) -> tuple[tuple[int, Image.Image], ...]:
        """Source-font readings include digits that cannot be valid ranks."""
        templates = []
        for value in range(maximum + 1):
            image, draw = new_scaled_canvas(32, 32, DOCTRINE_SUPERSAMPLE, background=CARD)
            _draw_rank_badge(draw, 16, 16, value, 1.0, DOCTRINE_SUPERSAMPLE)
            patch = finish_scaled_canvas(image, (32, 32)).crop((5, 5, 28, 28))
            templates.append((value, patch))
        return tuple(templates)

    task_name = "chamber_flow_fill"
    task_family = "physics"
    task_mechanic = "weir_cascade_fill_order"
    answer_overlay_mode = "none"
    verifier_profile = "chamber_flow_visible_geometry_multimodal_hard_gate_v1"
    video_verifier_profile = "chamber_flow_full_video_semantic_single_decode"

    # ------------------------------------------------------------------ sampling

    def _sample_task(self) -> Dict[str, Any]:
        # Four to six chambers: every extra chamber doubles the number of
        # left/right interleavings the crests have to choose between, which is
        # what pushes a crest-blind guess down.  Six is the cap the tank width
        # allows at a readable chamber size; the 3-chamber form remains the
        # explicit "min" floor.
        counts = {"min": (3, 3), "typical": (5, 5), "max": (6, 6)}.get(self.difficulty or "", (4, 6))
        count = self.rng.randint(*counts)
        for _attempt in range(600):
            task = self._try_scene(count)
            if task is not None:
                return task
        raise RuntimeError("unable to sample a valid chamber layout")

    def _crest_heights(self, ranking: Sequence[int]) -> List[float]:
        """Crest heights realising ``ranking`` with the 7.5 pairwise margin.

        The gaps are cut out of the slack that remains after every consecutive
        pair is given its margin, so any ranking is realisable at any tank
        size without proposing heights and testing them.
        """

        crest_count = len(ranking)
        slack = (_MAX_CREST - _MIN_CREST) - _CREST_MARGIN * (crest_count - 1)
        cuts = sorted(self.rng.uniform(0.0, 1.0) for _ in range(crest_count))
        shares = [cuts[0]] + [cuts[index] - cuts[index - 1] for index in range(1, crest_count)]
        ordered: List[float] = []
        cursor = _MIN_CREST
        for index, share in enumerate(shares):
            cursor += share * slack + (_CREST_MARGIN if index else 0.0)
            ordered.append(cursor)
        return [ordered[rank] for rank in ranking]

    def _try_scene(self, count: int) -> Dict[str, Any] | None:
        rng = self.rng
        # An end spout would make the fill order pure adjacency (the
        # crest comparisons must decide the left/right interleaving), so the
        # spout always sits over an interior chamber, weighted toward the
        # centre because a central spout leaves the most orders to choose
        # between.
        interior = list(range(2, count))
        source = rng.choices(interior, weights=[comb(count - 1, seat - 1) for seat in interior])[0]
        # Draw the answer uniformly from the orders this tank can produce, then
        # a crest ranking that produces it: a crest-blind strategy then has no
        # better expectation than the size of that order space.
        groups = _orders_by_crest_ranking(count, source)
        if not groups:
            return None
        order, rankings = rng.choice(groups)
        available = 430.0 - (count + 1) * _WALL_W
        # The temporal-margin gate below depends on the heights and widths, not
        # on which order was drawn, so it is retried inside the drawn answer.
        # Rejecting the answer itself would hand back the bias this draw exists
        # to remove.
        for _ in range(96):
            crests = self._crest_heights(rng.choice(rankings))
            shares = [rng.uniform(0.80, 1.20) for _ in range(count)]
            total_share = sum(shares)
            # Widths fill the tank by construction rather than by
            # rescale-and-test.
            widths = [available * share / total_share for share in shares]
            try:
                replay = simulate_fill(widths, crests, source)
            except ValueError:
                continue
            if tuple(replay["wet_order"]) != order:
                continue
            # Distinct, well-separated events (temporal margin).
            times = sorted(
                [replay["first_wet_times"][chamber] for chamber in replay["wet_order"][1:]]
                + [event["time"] for event in replay["events"] if event["kind"] != "spill"]
            )
            total = replay["total_time"]
            if any(second - first < 0.04 * total for first, second in zip(times, times[1:])):
                continue
            if replay["first_wet_times"][replay["wet_order"][1]] < 0.10 * total:
                continue
            return {
                "widths": [round(w, 2) for w in widths],
                "crests": [round(c, 2) for c in crests],
                "source": source,
            }
        return None

    # ------------------------------------------------------- geometry derivation

    @staticmethod
    def _scene(task: Mapping[str, Any]) -> Dict[str, Any]:
        widths = [float(value) for value in task["widths"]]
        crests = [float(value) for value in task["crests"]]
        count = len(widths)
        total_width = sum(widths) + (count + 1) * _WALL_W
        left0 = (512.0 - total_width) / 2.0
        chambers = []
        cursor = left0 + _WALL_W
        for index in range(count):
            chambers.append({"left": cursor, "right": cursor + widths[index]})
            cursor += widths[index] + _WALL_W
        source = int(task["source"])
        spout_x = (chambers[source - 1]["left"] + chambers[source - 1]["right"]) / 2.0
        return {
            "chambers": chambers,
            "floor": _FLOOR_Y,
            "rim": _RIM_Y,
            "crest_y": [_FLOOR_Y - crest for crest in crests],
            "outer_left": left0,
            "outer_right": left0 + total_width,
            "spout_x": spout_x,
            "badge_y": _RIM_Y - 26.0,
        }

    @classmethod
    def progress_carriers(
        cls, task: Mapping[str, Any]
    ) -> List[Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]]:
        """Per chamber, the water standing in it - which only ever rises.

        Water arrives over a weir and never leaves: a chamber's level climbs to
        the crest that lets it spill onward, holds there while its neighbour
        fills, and climbs again when the pools merge.  Each carrier reads two
        columns inside one chamber, placed a quarter of its width in from each
        wall so that neither the weirs nor the spout stream - which falls down
        the source chamber's centre and shortens as that chamber fills - is ever
        sampled.  The reading is the wet height of that chamber alone, which is
        also what makes the first-wet order it answers with checkable per
        chamber.
        """

        geometry = cls._scene(task)
        floor, rim = float(geometry["floor"]), float(geometry["rim"])
        carriers: List[
            Tuple[str, List[Tuple[float, float]], Tuple[int, int, int], Tuple[int, int, int] | None, int]
        ] = []
        for index, chamber in enumerate(geometry["chambers"], start=1):
            width = float(chamber["right"]) - float(chamber["left"])
            carriers.append(
                (
                    f"chamber_{index}",
                    carrier_column_samples(
                        (
                            float(chamber["left"]) + 0.25 * width,
                            float(chamber["left"]) + 0.75 * width,
                        ),
                        rim + 2.0,
                        floor - 3.0,
                    ),
                    CARD,
                    None,
                    1,
                )
            )
        return carriers

    @classmethod
    def terminal_member_boxes(cls, task: Mapping[str, Any]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        geometry = cls._scene(task)
        boxes = []
        for index, chamber in enumerate(geometry["chambers"], start=1):
            boxes.append(
                (
                    f"chamber_{index}",
                    (
                        chamber["left"] - 3.0,
                        geometry["badge_y"] - _BADGE_R - 4.0,
                        chamber["right"] + 3.0,
                        geometry["floor"] + 4.0,
                    ),
                )
            )
        return boxes

    # ---------------------------------------------------------------- semantics

    def _replay(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return simulate_fill(task["widths"], task["crests"], int(task["source"]))

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return format_nonnegative_integer_list_payload(self._replay(task)["wet_order"])

    def _question(self, task: Mapping[str, Any]) -> str:
        return "In what order do the chambers first receive water?"

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        replay = self._replay(task)
        order = replay["wet_order"]
        total = replay["total_time"]
        count = len(order)
        states: List[tuple[float, str]] = []
        for rank in range(2, count + 1):
            chamber = order[rank - 1]
            feeder = self._feeder_of(order, rank)
            when = replay["first_wet_times"][chamber] / total
            text = (
                f"The pool rises to the top of the wall between chambers {feeder} and {chamber} "
                f"and spills over into chamber {chamber}, which becomes the number {rank} chamber to receive water; its badge appears."
            )
            if rank == 2:
                text = (
                    f"Water from the spout lands in chamber {order[0]} first, so its badge shows 1. "
                    + text
                )
            states.append((when, text))
        final = (
            f"Chamber {order[-1]} fills until its level reaches the crest it spilled from, the pools connect, and the pour stops. "
            f"Reading the badges, the chambers first received water in the order {', '.join(str(c) for c in order)}."
        )
        states.append((1.0, final))
        return states

    @staticmethod
    def _feeder_of(order: Sequence[int], rank: int) -> int:
        chamber = order[rank - 1]
        earlier = order[: rank - 1]
        if chamber - 1 in earlier:
            return chamber - 1
        return chamber + 1

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        frames = max(31, 11 * len(task["widths"]))
        while frames % 4 != 3:
            frames += 1
        return frames

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        replay = self._replay(task)
        return {
            "semantic_state": {
                "widths": [float(value) for value in task["widths"]],
                "crests": [float(value) for value in task["crests"]],
                "source": int(task["source"]),
            },
            "answer_schema": {
                "schema_version": "chamber_flow_fill_v1",
                "kind": "first_wet_order_list",
                "parser_profile": "strict_chamber_order_list",
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
                "semantic_source": "visible chamber widths, wall crest heights, and the spout position",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: \[([1-9][0-9]*(?:, [1-9][0-9]*)*)\]\.$",
                "untrusted_redundant_fields": [
                    "wet_order",
                    "first_wet_times",
                    "events",
                    "final_levels",
                    "state_trace",
                    "semantic_state_trace",
                ],
            },
            "multiple_solution_policy": "deterministic cascade; exactly one first-wet order",
            "wet_order": replay["wet_order"],
            "first_wet_times": {str(k): round(v, 4) for k, v in replay["first_wet_times"].items()},
            "final_levels": {str(k): round(v, 3) for k, v in replay["final_levels"].items()},
            "semantic_state_trace": [
                {"rank": rank + 1, "chamber": chamber}
                for rank, chamber in enumerate(replay["wet_order"])
            ],
        }

    # ------------------------------------------------------------------- render

    def _levels_at(self, task: Mapping[str, Any], progress: float) -> tuple[Dict[int, float], Dict[int, float]]:
        """Continuous per-chamber levels plus per-chamber badge opacities."""

        replay = self._replay(task)
        total = replay["total_time"]
        time = max(0.0, min(1.0, float(progress))) * total
        levels: Dict[int, float] = {}
        for chamber, points in replay["trajectories"].items():
            level = points[0][1]
            for (t0, v0), (t1, v1) in zip(points, points[1:]):
                if time >= t1:
                    level = v1
                elif time > t0:
                    level = v0 + (v1 - v0) * (time - t0) / max(1e-9, t1 - t0)
                    break
                else:
                    break
            levels[chamber] = level
        badges: Dict[int, float] = {}
        ramp = 0.035 * total
        for chamber, wet_time in replay["first_wet_times"].items():
            badges[chamber] = max(0.0, min(1.0, (time - wet_time) / ramp))
        return levels, badges

    def _render_progress(self, task: Mapping[str, Any], progress: float, *, mark_answer: bool) -> Image.Image:
        del mark_answer  # ranks commit inside the physical process itself
        supersample = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, supersample, background=BG)
        geometry = self._scene(task)
        replay = self._replay(task)
        levels, badges = self._levels_at(task, progress)
        rank_of = {chamber: rank + 1 for rank, chamber in enumerate(replay["wet_order"])}

        draw.rounded_rectangle((16, 16, 496, 496), radius=24, fill=CARD, outline=CARD_EDGE, width=2)
        floor, rim = geometry["floor"], geometry["rim"]
        chambers = geometry["chambers"]

        # Water fills (behind walls).
        for index, chamber in enumerate(chambers, start=1):
            level = levels.get(index, 0.0)
            if level > 1.2:
                surface = floor - level
                draw.rectangle(
                    (chamber["left"], min(surface, floor - 2.0), chamber["right"], floor - 1.0),
                    fill=WATER,
                )
                draw.line((chamber["left"], surface, chamber["right"], surface), fill=WATER_TOP, width=3)

        # Falling stream from the spout into the source chamber.
        amount = max(0.0, min(1.0, float(progress)))
        if amount < 1.0 - 1e-9:
            spout_x = geometry["spout_x"]
            source = int(task["source"])
            surface = floor - levels.get(source, 0.0)
            draw.line((spout_x, rim - 34.0, spout_x, surface - 2.0), fill=WATER, width=5)

        # Outer walls and floor.
        draw.rectangle((geometry["outer_left"], rim, geometry["outer_left"] + _WALL_W, floor), fill=WALL)
        draw.rectangle((geometry["outer_right"] - _WALL_W, rim, geometry["outer_right"], floor), fill=WALL)
        draw.rectangle((geometry["outer_left"], floor, geometry["outer_right"], floor + _WALL_W), fill=WALL)
        # Interior weirs.
        for index, crest_y in enumerate(geometry["crest_y"]):
            wall_left = chambers[index]["right"]
            draw.rectangle((wall_left, crest_y, wall_left + _WALL_W, floor), fill=WALL)

        # Spout.
        spout_x = geometry["spout_x"]
        draw.rectangle((spout_x - 13.0, rim - 46.0, spout_x + 13.0, rim - 34.0), fill=SPOUT)
        draw.rectangle((spout_x - 5.0, rim - 58.0, spout_x + 55.0, rim - 46.0), fill=SPOUT)

        # Chamber ids on the floor band.
        id_font = _font(15 * supersample)
        for index, chamber in enumerate(chambers, start=1):
            draw.text(
                ((chamber["left"] + chamber["right"]) / 2.0, floor + 26.0),
                str(index),
                anchor="mm",
                fill=INK,
                font=id_font,
            )

        # Rank badges (amber discs) above chambers, committed at first wet.
        badge_y = geometry["badge_y"]
        for index, chamber in enumerate(chambers, start=1):
            opacity = badges.get(index, 0.0)
            if opacity <= 0.01:
                continue
            x = (chamber["left"] + chamber["right"]) / 2.0
            _draw_rank_badge(draw, x, badge_y, rank_of[index], opacity, supersample)
        return finish_scaled_canvas(image, (512, 512))

    # ------------------------------------------------------------------ contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        count = len(task["widths"])
        return VisualReasoningContract(
            input_description=(
                f"The single image is a side view of one open tank divided into {count} numbered chambers by interior walls of clearly different heights. "
                "The chamber numbers sit under the floor, the spout above one chamber shows where water will pour in, and every chamber starts dry."
            ),
            symbol_legend=(
                "Interior wall heights are compared visually; their height differences are large and unambiguous.",
                "Chamber numbers are identities only. An amber rank badge above a chamber will show at which position that chamber first received water.",
            ),
            rules=(
                "Water pours at a steady rate into the chamber under the spout.",
                "A filling chamber's level rises until it reaches the top of its lower side wall; the water then spills over that wall into the neighbor, and the spilling chamber's level holds while the neighbor fills.",
                "When a fed chamber's level catches up to the level of the wall it was fed over, the two sides connect and rise together as one pool, which then spills at its next lowest wall in the same way.",
                "The pour stops the moment the last dry chamber's level reaches the top of the wall it was fed over; everything then holds.",
            ),
            modality_outputs={
                "ti2v": (
                    "Hold the dry input, then pour: levels rise continuously, hold during spills, and connect at merges exactly as the walls dictate, while the falling stream stays under the spout. "
                    "The moment a chamber first receives water, its amber rank badge fades in above it. After the last chamber's level reaches its feeding wall's top, the stream ends and the final state holds."
                ),
                "solution_image": (
                    "Render the stopped tank: every chamber holds its final water level and carries its amber rank badge above it; walls, numbers, and spout are unchanged."
                ),
            },
            preservation_constraints=(
                "Keep the tank walls, chamber numbers, spout, palette, and camera fixed; only water levels, the falling stream, and the amber rank badges change.",
            ),
            transition_constraints=(
                "Raise levels continuously in the receiving chamber only, hold spilling levels at their crests, and commit each rank badge exactly when its chamber first turns wet; never wet a chamber out of order or move water through a wall below its top.",
            ),
            ti2v_continuity={
                "moving_nouns": "the rising water levels, the falling stream, and the appearing rank badges",
                "static_nouns": "the tank walls, chamber numbers, spout, and camera",
                "hold_noun": "the stopped tank with all rank badges placed",
            },
            answer=AnswerContract(
                kind="nonnegative_integer_list",
                format_template=NONNEGATIVE_INTEGER_LIST_ANSWER_SCHEMA,
                parser_profile="strict_chamber_order_list",
                verifier_profile=self.verifier_profile,
                format_example=NONNEGATIVE_INTEGER_LIST_ANSWER_EXAMPLE,
            ),
            render_profile="weir_chamber_cascade_v1",
        )


__all__ = ["ChamberFlowFillPattern", "simulate_fill"]
