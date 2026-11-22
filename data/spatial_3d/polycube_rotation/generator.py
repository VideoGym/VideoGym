"""Polycube rotation: pick the pose plate that matches a stated turn sequence.

A small polycube (4-5 cubes, fully rotation-asymmetric) floats beside a
labeled 3D axis triad. The default distribution uses one to three quarter-turn
rotations; explicit programs support up to five rotations
about the marked axes; the four option plates are genuine rendered pose
hypotheses (correct pose plus mechanism-targeted distractors), and the typed
answer is the matching plate letter.  The reference video turns the structure
continuously, one rotation at a time, pausing after each quarter turn.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from PIL import Image, ImageDraw

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.pipeline.structured_grid import load_font
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
)
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.spatial_3d.core.iso_geometry import (
    AXIS_NAMES,
    Face,
    IsoProjection,
    UX,
    UY,
    UZ,
    apply_matrix,
    face_visibility,
    normalize_cells,
    rotate_cells_90,
    rotation_matrix,
    rotation_orbit,
    voxel_faces,
)
from data.spatial_3d.core.scene_base import Spatial3DScenePattern

BG = (246, 248, 251)
CARD = (255, 255, 253)
CARD_EDGE = (216, 221, 226)
CUBE_COLOR = (176, 196, 222)
AXIS_COLOR = (57, 115, 182)
AXIS_INK = (40, 47, 56)
LABEL_INK = (40, 47, 56)

_ROTATE_WEIGHT = 1.0
_HOLD_WEIGHT = 0.45
_DIRECTION_WORDS = {1: "counterclockwise", -1: "clockwise"}


def _bbox_center(cells: Sequence[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    return (
        (min(c[0] for c in cells) + max(c[0] for c in cells) + 1) / 2.0,
        (min(c[1] for c in cells) + max(c[1] for c in cells) + 1) / 2.0,
        (min(c[2] for c in cells) + max(c[2] for c in cells) + 1) / 2.0,
    )


def replay_polycube(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible pose program only."""

    cells = normalize_cells((int(x), int(y), int(z)) for x, y, z in source["cells"])
    rotations = [(str(axis), int(sign)) for axis, sign in source["rotations"]]
    options = [
        {
            "label": str(option["label"]),
            "cells": normalize_cells((int(x), int(y), int(z)) for x, y, z in option["cells"]),
        }
        for option in source["options"]
    ]
    if not 4 <= len(cells) <= 5:
        raise ValueError("polycube must contain four or five cubes")
    if not 1 <= len(rotations) <= 8:
        raise ValueError("rotation program must contain one to eight turns")
    if any(axis not in AXIS_NAMES or sign not in (1, -1) for axis, sign in rotations):
        raise ValueError("rotation program has an invalid axis or direction")
    if sorted(option["label"] for option in options) != ["A", "B", "C", "D"]:
        raise ValueError("options must be labeled A-D exactly once each")
    states = [cells]
    for axis, sign in rotations:
        states.append(rotate_cells_90(states[-1], axis, sign))
    final = states[-1]
    matches = [option["label"] for option in options if option["cells"] == final]
    if len(matches) != 1:
        raise ValueError("exactly one option plate must match the replayed final pose")
    return {"states": states, "answer": matches[0], "options": options}


class PolycubeRotationPattern(Spatial3DScenePattern):
    task_name = "polycube_rotation"
    task_family = "spatial_3d"
    task_mechanic = "polycube_quarter_turn_sequence"
    motion = True
    verifier_profile = "polycube_rotation_visible_program_v1"

    def __init__(
        self, *args: Any, rotation_count: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if rotation_count is not None and not 1 <= int(rotation_count) <= 8:
            raise ValueError("rotation_count must lie in 1..8")
        self.rotation_count = int(rotation_count) if rotation_count is not None else None

    # ------------------------------------------------------------- sampling

    def _sample_polycube(self) -> Tuple[Tuple[int, int, int], ...]:
        for _ in range(600):
            count = self.rng.choice((4, 4, 5))
            cells = [(0, 0, 0)]
            while len(cells) < count:
                base = self.rng.choice(cells)
                dx, dy, dz = self.rng.choice(
                    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
                )
                candidate = (base[0] + dx, base[1] + dy, base[2] + dz)
                if candidate not in cells:
                    cells.append(candidate)
            normalized = normalize_cells(cells)
            spans = tuple(max(c[i] for c in normalized) + 1 for i in range(3))
            if max(spans) > 3:
                continue
            if len(rotation_orbit(normalized)) != 24:
                continue
            if not any(self._tip_cube(normalized)):
                continue
            return normalized
        raise RuntimeError("polycube_rotation could not sample an asymmetric polycube")

    @staticmethod
    def _tip_cube(cells: Sequence[Tuple[int, int, int]]):
        occupied = set(cells)
        for cell in sorted(cells):
            neighbors = [
                n
                for n in (
                    (cell[0] + 1, cell[1], cell[2]), (cell[0] - 1, cell[1], cell[2]),
                    (cell[0], cell[1] + 1, cell[2]), (cell[0], cell[1] - 1, cell[2]),
                    (cell[0], cell[1], cell[2] + 1), (cell[0], cell[1], cell[2] - 1),
                )
                if n in occupied
            ]
            if len(neighbors) == 1:
                yield cell, neighbors[0]

    def _sample_task(self) -> Dict[str, Any]:
        rotation_counts = {
            "min": (1,),
            "typical": (2,),
            "max": (3,),
        }.get(self.difficulty or "", (1, 2, 2, 3))
        for _ in range(400):
            cells = self._sample_polycube()
            count = self.rotation_count if self.rotation_count is not None else self.rng.choice(rotation_counts)
            rotations = [
                (self.rng.choice(AXIS_NAMES), self.rng.choice((1, -1))) for _ in range(count)
            ]
            states = [cells]
            for axis, sign in rotations:
                states.append(rotate_cells_90(states[-1], axis, sign))
            # Every stated turn has to leave the structure in a pose it has not
            # held before.  A program that returns to an earlier pose contains a
            # turn that is immediately undone, which makes the stated schedule
            # longer than the transformation it performs and makes the reference
            # video retrace its own poses backwards - the same silhouette then
            # occurs at two different points of the process, so no reader, human
            # or verifier, can tell the second half from a rewind of the first.
            if len(set(states)) != len(states):
                continue
            distractors = self._distractors(cells, rotations, states)
            if distractors is None:
                continue
            option_cells = [states[-1], *distractors]
            order = list(range(4))
            self.rng.shuffle(order)
            labels = ["A", "B", "C", "D"]
            options = [
                {"label": labels[position], "cells": option_cells[source_index]}
                for position, source_index in enumerate(order)
            ]
            answer = labels[order.index(0)]
            task = self._build_task(cells, rotations, states, options, answer)
            if task is not None:
                return task
        raise RuntimeError("polycube_rotation could not sample a valid instance")

    def _distractors(
        self,
        cells: Tuple[Tuple[int, int, int], ...],
        rotations: List[Tuple[str, int]],
        states: List[Tuple[Tuple[int, int, int], ...]],
    ) -> Optional[List[Tuple[Tuple[int, int, int], ...]]]:
        final = states[-1]
        pool: List[Tuple[str, Tuple[Tuple[int, int, int], ...]]] = []
        # Inverted last turn.
        axis, sign = rotations[-1]
        pool.append(("inverted_last", rotate_cells_90(states[-2], axis, -sign)))
        # Wrong axis on the last turn.
        for other_axis in AXIS_NAMES:
            if other_axis != axis:
                pool.append(("wrong_axis", rotate_cells_90(states[-2], other_axis, sign)))
        # Missing last turn (or the untouched start pose).
        pool.append(("missing_last", states[-2]))
        # One extra turn.
        extra_axis = self.rng.choice(AXIS_NAMES)
        extra_sign = self.rng.choice((1, -1))
        pool.append(("extra_turn", rotate_cells_90(final, extra_axis, extra_sign)))
        # Mirror image.
        mirrored = normalize_cells((-x, y, z) for x, y, z in final)
        pool.append(("mirror", mirrored))
        self.rng.shuffle(pool)
        chosen: List[Tuple[Tuple[int, int, int], ...]] = []
        for _name, candidate in pool:
            if candidate == final or candidate in chosen:
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                return chosen
        return None

    def _build_task(
        self,
        cells: Tuple[Tuple[int, int, int], ...],
        rotations: List[Tuple[str, int]],
        states: List[Tuple[Tuple[int, int, int], ...]],
        options: List[Dict[str, Any]],
        answer: str,
    ) -> Optional[Dict[str, Any]]:
        # One fixed camera over all committed states and swept mid-turn poses:
        # every intermediate point stays inside the bounding sphere of radius
        # R about the turn center, so bound the projection by that sphere.
        radius = 0.0
        for state in states:
            center = _bbox_center(state)
            for x, y, z in state:
                for dx in (0.0, 1.0):
                    for dy in (0.0, 1.0):
                        for dz in (0.0, 1.0):
                            radius = max(
                                radius,
                                math.dist((x + dx, y + dy, z + dz), center),
                            )
        box = (96.0, 96.0, 428.0, 428.0)
        span_u = 2.0 * radius * UX * math.sqrt(2.0)
        span_v = 2.0 * radius * math.sqrt(2.0 * UY * UY + UZ * UZ)
        scale = min(46.0, (box[2] - box[0]) / span_u, (box[3] - box[1]) / span_v)
        if scale < 24.0:
            return None
        projection = IsoProjection(scale, ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
        task = {
            "cells": cells,
            "rotations": rotations,
            "states": states,
            "options": options,
            "answer": answer,
            "projection": projection,
        }
        # gates: every cube individually evidenced in every committed
        # state render and in every option-plate pose; plates pairwise
        # clearly different.
        for state in states:
            if min(self._per_cube_visibility(task, state)) < 0.22:
                return None
        for option in options:
            if min(self._per_cube_visibility(task, option["cells"])) < 0.22:
                return None
        margins = self._plate_margins(task)
        if min(margins) < 0.010:
            return None
        task["plate_min_diff_fraction"] = float(min(margins))
        replay_polycube(
            {
                "cells": [list(cell) for cell in cells],
                "rotations": [[axis, sign] for axis, sign in rotations],
                "options": [
                    {"label": option["label"], "cells": [list(c) for c in option["cells"]]}
                    for option in options
                ],
            }
        )
        return task

    # ------------------------------------------------------------- rendering

    def _pose_faces(
        self,
        state: Sequence[Tuple[int, int, int]],
        theta_axis: Optional[str] = None,
        theta: float = 0.0,
        sign: int = 1,
    ) -> List[Face]:
        center = _bbox_center(state)
        raw_faces = voxel_faces(state, default_color=CUBE_COLOR, only_camera_facing=False)
        matrix = rotation_matrix(theta_axis, sign * theta) if theta_axis else None
        transformed: List[Face] = []
        for index, face in enumerate(raw_faces):
            points = []
            for point in face.points:
                local = (point[0] - center[0], point[1] - center[1], point[2] - center[2])
                if matrix is not None:
                    local = apply_matrix(matrix, local)
                points.append(local)
            candidate = Face(
                points=tuple(points),
                color=face.color,
                edge_color=face.edge_color,
                edge_width=face.edge_width,
                bias=index * 1e-3,
            )
            normal = candidate.normal()
            if normal[0] + normal[1] + normal[2] * (2.0 * UY / UZ) <= 0:
                continue
            transformed.append(candidate)
        return transformed

    def _draw_axis_triad(self, canvas: Image.Image, supersample: int) -> None:
        draw = ScaledDraw(ImageDraw.Draw(canvas), supersample)
        anchor = (92.0, 442.0)
        probe = IsoProjection(1.0, (0.0, 0.0))
        origin2 = probe.project((0, 0, 0))
        length = 40.0
        font = load_font(15 * supersample)
        for axis, vector in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
            tip2 = probe.project(vector)
            dx, dy = tip2[0] - origin2[0], tip2[1] - origin2[1]
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            end = (anchor[0] + ux * length, anchor[1] + uy * length)
            draw.line((*anchor, *end), fill=AXIS_COLOR, width=3)
            head_angle = math.atan2(uy, ux)
            # One mitered polyline through the tip, so the two barbs meet in a
            # point instead of as two stroke ends butting against each other.
            barbs = [
                (end[0] + 9 * math.cos(head_angle + spread),
                 end[1] + 9 * math.sin(head_angle + spread))
                for spread in (math.radians(152.0), math.radians(-152.0))
            ]
            draw.open_path([barbs[0], end, barbs[1]], fill=AXIS_COLOR, width=3)
            label_pos = (anchor[0] + ux * (length + 13), anchor[1] + uy * (length + 13))
            draw.text(label_pos, axis, fill=AXIS_INK, font=font, anchor="mm")
        draw.ellipse(
            (anchor[0] - 3, anchor[1] - 3, anchor[0] + 3, anchor[1] + 3),
            fill=AXIS_INK,
        )

    def _render_pose(
        self,
        task: Mapping[str, Any],
        state: Sequence[Tuple[int, int, int]],
        theta_axis: Optional[str],
        theta: float,
        sign: int,
    ) -> Image.Image:
        supersample = DOCTRINE_SUPERSAMPLE
        canvas = Image.new("RGB", (512 * supersample, 512 * supersample), BG)
        backdrop = ImageDraw.Draw(canvas)
        backdrop.rounded_rectangle(
            (8 * supersample, 8 * supersample, 504 * supersample, 504 * supersample),
            radius=20 * supersample,
            fill=CARD,
            outline=CARD_EDGE,
            width=2 * supersample,
        )
        from data.spatial_3d.core.iso_geometry import render_faces

        faces = self._pose_faces(state, theta_axis, theta, sign)
        render_faces(canvas, faces, task["projection"], supersample)
        self._draw_axis_triad(canvas, supersample)
        return finish_scaled_canvas(canvas, (512, 512))

    def _schedule(self, task: Mapping[str, Any], progress: float) -> Tuple[int, Optional[str], float, int]:
        rotations = task["rotations"]
        total = len(rotations) * (_ROTATE_WEIGHT + _HOLD_WEIGHT)
        time = max(0.0, min(1.0, float(progress))) * total
        for index, (axis, sign) in enumerate(rotations):
            if time >= _ROTATE_WEIGHT + _HOLD_WEIGHT - 1e-9:
                time -= _ROTATE_WEIGHT + _HOLD_WEIGHT
                continue
            fraction = min(1.0, time / _ROTATE_WEIGHT)
            eased = 0.5 - 0.5 * math.cos(math.pi * fraction)
            if eased >= 1.0 - 1e-9:
                return index + 1, None, 0.0, 1
            return index, axis, eased * math.pi / 2.0, sign
        return len(rotations), None, 0.0, 1

    def _render_progress(self, task: Mapping[str, Any], progress: float) -> Image.Image:
        step, axis, theta, sign = self._schedule(task, progress)
        return self._render_pose(task, task["states"][step], axis, theta, sign)

    def _commit_progress(self, task: Mapping[str, Any], turn_number: int) -> float:
        total = len(task["rotations"]) * (_ROTATE_WEIGHT + _HOLD_WEIGHT)
        return ((turn_number - 1) * (_ROTATE_WEIGHT + _HOLD_WEIGHT) + _ROTATE_WEIGHT + _HOLD_WEIGHT * 0.5) / total

    def semantic_checkpoints(self, task: Mapping[str, Any]) -> Tuple[float, ...]:
        checkpoints = [0.0]
        checkpoints.extend(self._commit_progress(task, index + 1) for index in range(len(task["rotations"])))
        checkpoints[-1] = 1.0
        return tuple(sorted(set(checkpoints)))

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        return 16 * len(task["rotations"]) + 15

    # --------------------------------------------------------------- plates

    def _plate_image(self, task: Mapping[str, Any], cells: Sequence[Tuple[int, int, int]], label: str) -> Image.Image:
        supersample = DOCTRINE_SUPERSAMPLE
        canvas = Image.new("RGB", (512 * supersample, 512 * supersample), BG)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            (8 * supersample, 8 * supersample, 504 * supersample, 504 * supersample),
            radius=20 * supersample,
            fill=CARD,
            outline=CARD_EDGE,
            width=2 * supersample,
        )
        from data.spatial_3d.core.iso_geometry import render_faces

        plate_projection = IsoProjection(task["projection"].scale, (256.0, 238.0))
        faces = self._pose_faces(cells)
        render_faces(canvas, faces, plate_projection, supersample)
        font = load_font(26 * supersample)
        draw.text((256 * supersample, 478 * supersample), label, fill=LABEL_INK, font=font, anchor="mm")
        return finish_scaled_canvas(canvas, (512, 512))

    def _plate_margins(self, task: Mapping[str, Any]) -> List[float]:
        arrays = [
            np.asarray(self._plate_image(task, option["cells"], option["label"]).convert("L"), dtype=np.int16)
            for option in task["options"]
        ]
        margins = []
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                diff = np.abs(arrays[i] - arrays[j]) > 30
                margins.append(float(diff.mean()))
        return margins

    def _per_cube_visibility(self, task: Mapping[str, Any], state: Sequence[Tuple[int, int, int]]) -> List[float]:
        center = _bbox_center(state)
        occupied = set(state)
        faces: List[Face] = []
        owners: List[int] = []
        for cube_index, cell in enumerate(state):
            for face in voxel_faces([cell], default_color=CUBE_COLOR, only_camera_facing=True):
                # Skip faces shared with a neighboring cube of the structure.
                normal = face.normal()
                step = tuple(int(round(v)) for v in normal)
                neighbor = (cell[0] + step[0], cell[1] + step[1], cell[2] + step[2])
                if neighbor in occupied:
                    continue
                faces.append(
                    Face(
                        points=tuple(
                            (p[0] - center[0], p[1] - center[1], p[2] - center[2]) for p in face.points
                        ),
                        color=face.color,
                    )
                )
                owners.append(cube_index)
        fractions = face_visibility(faces, task["projection"])
        best = [0.0] * len(state)
        for owner, fraction in zip(owners, fractions):
            best[owner] = max(best[owner], fraction)
        return best

    def _option_plates(self, task: Mapping[str, Any], sample_dir, sample_id) -> Optional[Dict[str, Any]]:
        paths = []
        labels = []
        for option in task["options"]:
            path = sample_dir / f"{sample_id}__option_{option['label']}.png"
            self._plate_image(task, option["cells"], option["label"]).save(path)
            paths.append(path.as_posix())
            labels.append(option["label"])
        return {"paths": paths, "labels": labels}

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        label = str(task["answer"])
        FOUR_CHOICE_CODEC.format(label)
        return label

    def _question(self, task: Mapping[str, Any]) -> str:
        return "Which plate shows the structure after the stated quarter turns?"

    @staticmethod
    def _direction_phrase(vector: Tuple[int, int, int]) -> str:
        mapping = {
            (1, 0, 0): "along the X arrow",
            (-1, 0, 0): "against the X arrow",
            (0, 1, 0): "along the Y arrow",
            (0, -1, 0): "against the Y arrow",
            (0, 0, 1): "straight up",
            (0, 0, -1): "straight down",
        }
        return mapping[tuple(int(round(v)) for v in vector)]

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        states = task["states"]
        rotations = task["rotations"]
        texts: List[tuple[Any, str]] = []
        for index, (axis, sign) in enumerate(rotations):
            before, after = states[index], states[index + 1]
            tips = list(self._tip_cube(before))
            tip, neighbor = tips[0]
            direction = (tip[0] - neighbor[0], tip[1] - neighbor[1], tip[2] - neighbor[2])
            matrix = rotation_matrix(axis, sign * math.pi / 2.0)
            rotated = tuple(int(round(v)) for v in apply_matrix(matrix, direction))
            spans = tuple(max(c[i] for c in after) + 1 for i in range(3))
            sentence = (
                f"Turn {index + 1} rotates the whole structure a quarter turn about the {axis} axis, "
                f"{_DIRECTION_WORDS[sign]} as seen from the {axis} arrow tip. The single jutting cube that "
                f"pointed {self._direction_phrase(direction)} now points {self._direction_phrase(rotated)}, "
                f"and the structure spans {spans[0]} along X, {spans[1]} along Y, and {spans[2]} upward."
            )
            if index == 0:
                sentence = (
                    "I track the structure through each stated turn, watching its jutting cube and its "
                    "overall span. " + sentence
                )
            if index == len(rotations) - 1:
                sentence += (
                    f" Checking the four plates against this pose, plate {task['answer']} shows exactly "
                    f"this arrangement."
                )
            marker = self._commit_progress(task, index + 1) if index < len(rotations) - 1 else 1.0
            texts.append((marker, sentence))
        return texts

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "semantic_state": {
                "cells": [list(cell) for cell in task["cells"]],
                "rotations": [[axis, sign] for axis, sign in task["rotations"]],
                "options": [
                    {"label": option["label"], "cells": [list(c) for c in option["cells"]]}
                    for option in task["options"]
                ],
            },
            "pose_states": [[list(c) for c in state] for state in task["states"]],
            "correct_label": task["answer"],
            "plate_min_diff_fraction": task.get("plate_min_diff_fraction"),
            "multiple_solution_policy": "exactly one plate matches the deterministic replayed pose",
            "difficulty_evidence": {
                "cube_count": len(task["cells"]),
                "rotation_count": len(task["rotations"]),
            },
            "answer_schema": {
                "schema_version": "polycube_rotation_plate_v1",
                "kind": "option_letter",
                "parser_profile": "strict_option_letter_a_d",
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
                "semantic_source": "visible start pose, stated rotation program, and the four option plates",
                "strict_binary_correctness": True,
                "strict_text_parser": FOUR_CHOICE_CODEC.full_pattern,
                "untrusted_redundant_fields": ["pose_states", "correct_label"],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        del task
        inventory = {
            "schema_version": "polycube_rotation_rendered_glyph_inventory_v1",
            "fixed_labels": ["A", "B", "C", "D", "X", "Y", "Z"],
            "dynamic_labels": [],
            "all_rendered_text": ["A", "B", "C", "D", "X", "Y", "Z"],
        }
        schema = {
            "fixed_labels": "the three axis letters in the scene and one letter per option plate",
            "dynamic_labels": "none",
            "exclusions": "no task title, rule prose, coordinates, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        rotation_lines = " ".join(
            f"Turn {index + 1}: a quarter turn about the {axis} axis, {_DIRECTION_WORDS[sign]}."
            for index, (axis, sign) in enumerate(task["rotations"])
        )
        return VisualReasoningContract(
            input_description=(
                "The first input image shows a small structure of identical joined cubes floating beside a "
                "3D axis triad whose three arrows are labeled X, Y, and Z. Each additional input image shows "
                "exactly one candidate final pose with a single option letter centered below it: A, B, C, D."
            ),
            modality_input_descriptions={
                "ti2v": (
                    "The single input image shows a small structure of identical joined cubes floating beside "
                    "a 3D axis triad whose three arrows are labeled X, Y, and Z."
                ),
                "solution_image": (
                    "The single input image shows a small structure of identical joined cubes floating beside "
                    "a 3D axis triad whose three arrows are labeled X, Y, and Z."
                ),
            },
            symbol_legend=(
                "A rotation direction is always read looking from the tip of the named axis arrow back "
                "toward the structure: counterclockwise and clockwise mean what that viewpoint sees.",
                "The structure turns as one rigid piece about an axis through its own center; nothing "
                "detaches or bends.",
            ),
            rules=(
                "Apply the stated turns to the structure one after another, in order. "
                + rotation_lines,
                "The answer is the plate whose pose exactly matches the structure after the final turn.",
            ),
            modality_outputs={
                "ti2v": (
                    "The structure performs the stated turns in order: each turn is one continuous rigid "
                    "quarter rotation about the named axis through the structure's center, followed by a "
                    "short readable pause, and after the last turn the final pose holds."
                ),
                "solution_image": (
                    "Show the structure in its pose after the final stated turn, with the axis triad "
                    "unchanged."
                ),
            },
            preservation_constraints=(
                "the axis triad, the cube coloring, the cube count, and the background unchanged; only the "
                "structure's orientation changes.",
            ),
            transition_constraints=(
                "Rotate the whole structure rigidly about the stated axis, one turn at a time; never let a "
                "cube detach, resize, or lag behind, and never jump between poses.",
            ),
            ti2v_continuity={
                "moving_nouns": "the turning cube structure",
                "static_nouns": "the axis triad and its labels, the backdrop, and the camera",
                "hold_noun": "the structure resting in its final pose",
            },
            answer=AnswerContract(
                kind="option",
                format_template=FOUR_CHOICE_CODEC.schema,
                parser_profile="strict_option_letter_a_d",
                verifier_profile=self.verifier_profile,
                format_example=FOUR_CHOICE_CODEC.example,
            ),
            render_profile="polycube_rotation_v1",
        )


__all__ = ["PolycubeRotationPattern", "replay_polycube"]
