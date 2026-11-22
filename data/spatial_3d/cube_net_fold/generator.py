"""Cube-net folding: which pip face ends on top of the assembled box.

A flat hexomino net of six pip-marked squares lies in 3D view; the blue-shaded
square is the base and stays flat.  Folding every crease to 90 degrees
assembles the unit box above the base, so exactly one square becomes the lid.
The typed answer is that square's pip count.  The reference video folds the
five flaps up one at a time in increasing pip order (each flap carrying every
square attached beyond it), with a short readable pause after every fold.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


from PIL import Image

from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract
from data.common.render_primitives import DOCTRINE_SUPERSAMPLE, finish_scaled_canvas
from data.spatial_3d.core.iso_geometry import (
    CubeNet,
    Decal,
    Face,
    IsoProjection,
    all_cube_nets,
    fit_projection,
)
from data.spatial_3d.core.scene_base import Spatial3DScenePattern

BG = (246, 248, 251)
CARD = (255, 255, 253)
CARD_EDGE = (216, 221, 226)
PAPER = (247, 243, 233)
BASE_SHADE = (208, 224, 246)
PIP_INK = (40, 47, 56)

# Per-flap fold sweep plus a readable plateau, in schedule-progress units.
_FOLD_WEIGHT = 1.0
_HOLD_WEIGHT = 0.45


def _net_variants(cells: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    variants = set()
    current = [tuple(cell) for cell in cells]
    for _ in range(4):
        current = [(-r, c) for c, r in current]
        for mirrored in (current, [(-c, r) for c, r in current]):
            min_c = min(c for c, _ in mirrored)
            min_r = min(r for _, r in mirrored)
            variants.add(tuple(sorted((c - min_c, r - min_r) for c, r in mirrored)))
    return sorted(variants)


def replay_cube_net(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Independent forward replay from the visible net program only."""

    net_cells = [(int(c), int(r)) for c, r in source["net_cells"]]
    base_cell = (int(source["base_cell"][0]), int(source["base_cell"][1]))
    pip_triples = [(int(c), int(r), int(p)) for c, r, p in source["pip_by_cell"]]
    pips = {(c, r): p for c, r, p in pip_triples}
    if len(net_cells) != 6 or len(set(net_cells)) != 6:
        raise ValueError("cube net source must contain six distinct cells")
    if sorted(pips.values()) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("cube net pips must be a permutation of 1..6")
    if set(pips) != set(net_cells) or base_cell not in pips:
        raise ValueError("cube net pip cells must exactly cover the net")
    net = CubeNet(net_cells, base_cell)
    if not net.is_valid_cube_net():
        raise ValueError("the six cells do not fold into a cube")
    roles = net.fold_roles()
    top_cell = next(cell for cell, role in roles.items() if role == "top")
    return {
        "net": net,
        "pips": pips,
        "roles": roles,
        "top_cell": top_cell,
        "answer": pips[top_cell],
    }


class CubeNetFoldPattern(Spatial3DScenePattern):
    task_name = "cube_net_fold"
    task_family = "spatial_3d"
    task_mechanic = "cube_net_fold_assembly"
    motion = True
    verifier_profile = "cube_net_visible_fold_program_v1"

    # ------------------------------------------------------------- sampling

    def _sample_task(self) -> Dict[str, Any]:
        nets = all_cube_nets()
        for _ in range(64):
            shape = self.rng.choice(nets)
            cells = list(self.rng.choice(_net_variants(shape)))
            base_cell = self.rng.choice(cells)
            net = CubeNet(cells, base_cell)
            if not net.is_valid_cube_net():
                continue
            pip_values = [1, 2, 3, 4, 5, 6]
            self.rng.shuffle(pip_values)
            pips = {cell: pip_values[index] for index, cell in enumerate(cells)}
            task = self._build_task(cells, base_cell, pips)
            if task is not None:
                return task
        raise RuntimeError("cube_net_fold could not sample a valid net")

    def _build_task(
        self,
        cells: List[Tuple[int, int]],
        base_cell: Tuple[int, int],
        pips: Dict[Tuple[int, int], int],
    ) -> Optional[Dict[str, Any]]:
        replayed = replay_cube_net(
            {
                "net_cells": [list(cell) for cell in cells],
                "base_cell": list(base_cell),
                "pip_by_cell": [[c, r, pips[(c, r)]] for c, r in cells],
            }
        )
        net: CubeNet = replayed["net"]
        fold_order = sorted((cell for cell in cells if cell != base_cell), key=lambda cell: pips[cell])
        # Fixed camera over the whole schedule: fit over every commit state
        # plus swept mid-fold poses.
        points: List[Tuple[float, float, float]] = []
        for step in range(len(fold_order) + 1):
            for sweep in (0.0, 0.34, 0.67, 1.0):
                angles = self._angles(fold_order, step, sweep)
                for face in net.folded_faces(angles):
                    points.extend(face.corners)
        projection = fit_projection(points, box=(56.0, 56.0, 456.0, 430.0), max_scale=64.0)
        if projection.scale < 30.0:
            return None
        from data.common.prompt_paraphrase import active_profile
        if active_profile("numbered_faces"):
            projection=fit_projection(points,box=(38.0,38.0,474.0,474.0),max_scale=100.0)
        return {
            "cells": cells,
            "base": base_cell,
            "pips": pips,
            "net": net,
            "fold_order": fold_order,
            "roles": replayed["roles"],
            "top_cell": replayed["top_cell"],
            "answer": replayed["answer"],
            "projection": projection,
        }

    # ------------------------------------------------------------- schedule

    @staticmethod
    def _angles(
        fold_order: Sequence[Tuple[int, int]],
        committed: int,
        active_fraction: float,
    ) -> Dict[Tuple[int, int], float]:
        angles = {cell: math.pi / 2.0 for cell in fold_order[:committed]}
        if committed < len(fold_order) and active_fraction > 0.0:
            angles[fold_order[committed]] = (math.pi / 2.0) * min(1.0, active_fraction)
        return angles

    def _schedule_state(self, task: Mapping[str, Any], progress: float) -> Dict[Tuple[int, int], float]:
        fold_order = task["fold_order"]
        total = len(fold_order) * (_FOLD_WEIGHT + _HOLD_WEIGHT)
        time = max(0.0, min(1.0, float(progress))) * total
        committed = 0
        for _ in fold_order:
            if time >= _FOLD_WEIGHT + _HOLD_WEIGHT - 1e-9:
                time -= _FOLD_WEIGHT + _HOLD_WEIGHT
                committed += 1
                continue
            fraction = min(1.0, time / _FOLD_WEIGHT)
            # Smooth ease keeps per-frame pixel change moderate at fold ends.
            eased = 0.5 - 0.5 * math.cos(math.pi * fraction)
            return self._angles(fold_order, committed, eased)
        return self._angles(fold_order, len(fold_order), 0.0)

    def _commit_progress(self, task: Mapping[str, Any], fold_number: int) -> float:
        """Schedule progress at which fold ``fold_number`` (1-based) has committed."""

        fold_order = task["fold_order"]
        total = len(fold_order) * (_FOLD_WEIGHT + _HOLD_WEIGHT)
        return ((fold_number - 1) * (_FOLD_WEIGHT + _HOLD_WEIGHT) + _FOLD_WEIGHT + _HOLD_WEIGHT * 0.5) / total

    def semantic_checkpoints(self, task: Mapping[str, Any]) -> Tuple[float, ...]:
        checkpoints = [0.0]
        checkpoints.extend(self._commit_progress(task, index + 1) for index in range(len(task["fold_order"])))
        checkpoints[-1] = 1.0
        return tuple(sorted(set(checkpoints)))

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        del task
        return 79

    # ------------------------------------------------------------- rendering

    def _render_progress(self, task: Mapping[str, Any], progress: float) -> Image.Image:
        angles = self._schedule_state(task, progress)
        return self._render_angles(task, angles)

    def _render_angles(self, task: Mapping[str, Any], angles: Mapping[Tuple[int, int], float]) -> Image.Image:
        projection: IsoProjection = task["projection"]
        supersample = DOCTRINE_SUPERSAMPLE
        canvas = Image.new("RGB", (512 * supersample, 512 * supersample), BG)
        from PIL import ImageDraw

        backdrop = ImageDraw.Draw(canvas)
        from data.common.prompt_paraphrase import active_profile
        if active_profile("numbered_faces"):
            backdrop.rectangle((0,0,512*supersample,512*supersample),fill=(239,247,250))
            backdrop.rounded_rectangle((12*supersample,12*supersample,500*supersample,500*supersample),radius=18*supersample,fill=(252,251,244),outline=(191,213,218),width=2*supersample)
        else:
            backdrop.rounded_rectangle(
                (8*supersample,8*supersample,504*supersample,504*supersample),radius=20*supersample,fill=CARD,outline=CARD_EDGE,width=2*supersample,
            )
        from data.spatial_3d.core.iso_geometry import render_faces

        render_faces(canvas, self._mesh_faces(task, angles), projection, supersample)
        return finish_scaled_canvas(canvas, (512, 512))

    def _mesh_faces(self, task: Mapping[str, Any], angles: Mapping[Tuple[int, int], float]) -> List[Face]:
        """Native paper mesh and pips, shared with rigid-pose composition."""
        net: CubeNet = task["net"]
        pips: Dict[Tuple[int, int], int] = task["pips"]
        folded = net.folded_faces(angles)
        faces: List[Face] = []
        for face in folded:
            from data.common.prompt_paraphrase import active_profile
            numbered=active_profile("numbered_faces")
            color = ((83,178,206) if face.cell==net.base else (239,174,76)) if numbered else (BASE_SHADE if face.cell==net.base else PAPER)
            faces.append(
                Face(
                    points=face.corners,
                    color=color,
                    edge_color=(237,237,195) if numbered else (58,66,78),
                    edge_width=3.5 if numbered else 2.0,
                    decal=Decal(
                        str(pips[face.cell]),
                        face.origin,
                        face.axis_u,
                        face.axis_v,
                        color=PIP_INK,
                        kind="pips",
                    ),
                )
            )
        return faces

    # ------------------------------------------------------------- semantics

    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        return str(int(task["answer"]))

    def _question(self, task: Mapping[str, Any]) -> str:
        return "After the net folds into the box, how many pips are on the top face?"

    @staticmethod
    def _pose_phrase(face_center: Tuple[float, float, float], normal: Tuple[float, float, float]) -> str:
        nx, ny, nz = normal
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        nz /= magnitude
        if abs(nz) > 0.9:
            return "lies level at ground height" if face_center[2] < 0.5 else "lies level at lid height"
        return "stands upright"

    @staticmethod
    def _pip_list(pips: Sequence[int]) -> str:
        values = [f"{pip}-pip" for pip in pips]
        if len(values) == 1:
            return values[0]
        return ", ".join(values[:-1]) + " and " + values[-1]

    def _subtree(self, net: CubeNet, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        children: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for child, parent in net.parent.items():
            children.setdefault(parent, []).append(child)
        collected: List[Tuple[int, int]] = []
        stack = [cell]
        while stack:
            current = stack.pop()
            collected.append(current)
            stack.extend(children.get(current, ()))
        return collected

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        net: CubeNet = task["net"]
        pips: Dict[Tuple[int, int], int] = task["pips"]
        fold_order: List[Tuple[int, int]] = list(task["fold_order"])
        roles: Dict[Tuple[int, int], str] = task["roles"]
        role_words = {
            "top": "the lid of the box",
            "north": "a side wall",
            "south": "a side wall",
            "east": "a side wall",
            "west": "a side wall",
        }
        states: List[tuple[Any, str]] = []
        finals_so_far = 0
        for index, flap in enumerate(fold_order):
            committed = {cell: math.pi / 2.0 for cell in fold_order[: index + 1]}
            faces = {face.cell: face for face in net.folded_faces(committed)}
            face = faces[flap]
            carried = [cell for cell in self._subtree(net, flap) if cell != flap]
            parent = net.parent[flap]
            parent_name = (
                "the blue base"
                if parent == net.base
                else f"the {pips[parent]}-pip square"
            )
            carried_note = (
                ""
                if not carried
                else ", carrying the "
                + self._pip_list([pips[cell] for cell in sorted(carried, key=pips.get)])
                + (" square" if len(carried) == 1 else " squares")
            )
            ancestors_done = all(
                ancestor == net.base or ancestor in committed
                for ancestor in self._ancestors(net, flap)
            )
            pose = self._pose_phrase(face.center(), face.normal())
            sentence = (
                f"Fold {index + 1}: the {pips[flap]}-pip square folds up along its crease with {parent_name}"
                f"{carried_note}."
            )
            if ancestors_done:
                if finals_so_far == 0:
                    sentence += (
                        f" With every crease beneath it already folded, it now {pose} in its final place "
                        f"as {role_words[roles[flap]]}."
                    )
                else:
                    short_pose = "lying level at lid height" if "lid" in pose else "standing upright"
                    sentence += f" It ends final too, {short_pose} as {role_words[roles[flap]]}."
                finals_so_far += 1
            else:
                sentence += (
                    f" It now {pose}, but squares beneath it are still flat, so it will swing along when "
                    f"they fold."
                )
            if index == 0:
                sentence = (
                    "The blue-shaded square is the base and never moves; the five flaps fold up in increasing "
                    "pip order. " + sentence
                )
            if index == len(fold_order) - 1:
                top_pip = pips[task["top_cell"]]
                sentence += (
                    f" The box is now closed: the {top_pip}-pip square roofs it, so {top_pip} pips face up."
                )
            states.append((self._commit_progress(task, index + 1) if index < len(fold_order) - 1 else 1.0, sentence))
        return states

    @staticmethod
    def _ancestors(net: CubeNet, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        chain = []
        current = cell
        while current != net.base:
            current = net.parent[current]
            if current != net.base:
                chain.append(current)
        return chain

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "semantic_state": {
                "net_cells": [list(cell) for cell in task["cells"]],
                "base_cell": list(task["base"]),
                "pip_by_cell": [[c, r, task["pips"][(c, r)]] for c, r in task["cells"]],
            },
            "fold_order_pips": [task["pips"][cell] for cell in task["fold_order"]],
            "face_roles": {str(task["pips"][cell]): role for cell, role in task["roles"].items()},
            "top_pip": int(task["answer"]),
            "multiple_solution_policy": "the fold program is deterministic; exactly one pip count roofs the box",
            "difficulty_evidence": {
                "flap_count": len(task["fold_order"]),
                "max_hinge_depth": max(
                    len(self._ancestors(task["net"], cell)) + 1 for cell in task["fold_order"]
                ),
            },
            "answer_schema": {
                "schema_version": "cube_net_top_pip_v1",
                "kind": "pip_count_1_to_6",
                "parser_profile": "strict_single_digit_1_6",
                "format_template": "Answer: <1-6>.",
                "judged_artifacts": {
                    "ti2v": ["full_video"],
                },
                "accepts_multiple_valid_answers": False,
                "extra_text_allowed": False,
            },
            "verifier_contract": {
                "profile": self.verifier_profile,
                "semantic_source": "visible net cells, base shading, and pip marks",
                "strict_binary_correctness": True,
                "strict_text_parser": r"^Answer: ([1-6])\.$",
                "untrusted_redundant_fields": ["fold_order_pips", "face_roles", "top_pip"],
            },
        }

    def _glyph_inventory(self, task: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        del task
        inventory = {
            "schema_version": "cube_net_rendered_glyph_inventory_v1",
            "fixed_labels": [],
            "dynamic_labels": [],
            "all_rendered_text": [],
        }
        schema = {
            "fixed_labels": "none: faces carry drawn pip dots, not text",
            "dynamic_labels": "none",
            "exclusions": "no task title, rule prose, coordinates, status sentence, or answer banner",
        }
        return inventory, schema

    # -------------------------------------------------------------- contract

    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        return VisualReasoningContract(
            input_description=(
                "The single image shows a flat net of six connected paper squares in a three-quarter view. "
                "Each square carries a dice-style pip pattern from 1 to 6 pips, and every pip count appears on "
                "exactly one square. One square is shaded light blue."
            ),
            symbol_legend=(
                "Pips are printed identically on both sides of every square, so a square stays readable from "
                "either side.",
                "The blue-shaded square is the base: it stays flat on the table for the whole fold.",
                "Thin dark edges are the creases between neighboring squares.",
            ),
            rules=(
                "Every crease between two neighboring squares folds upward by a quarter turn, so the five "
                "unshaded squares rise off the table and close into a cube-shaped box sitting on the base.",
                "A square that folds carries every square attached beyond it rigidly along.",
                "When the box is closed, exactly one square lies level above the base as the lid; its pip "
                "count is the answer.",
            ),
            modality_outputs={
                "ti2v": (
                    "The five flaps fold up one at a time in increasing pip order of the flaps. Each fold is "
                    "one continuous quarter-turn about its crease that carries every square attached beyond "
                    "the moving flap, followed by a short readable pause; after the last fold the closed box "
                    "rests on the blue base with its lid pips facing up."
                ),
                "solution_image": (
                    "Show the completely folded box resting on the blue base square, with the lid and its "
                    "pips facing up and the visible side walls in place."
                ),
            },
            preservation_constraints=(
                "the blue base square flat in place, the pip pattern of every square, the paper coloring, and "
                "the background unchanged; only fold angles change.",
            ),
            transition_constraints=(
                "Fold exactly one crease at a time as a continuous rigid rotation; never detach, stretch, or "
                "re-print a square, and never cut between poses.",
            ),
            ti2v_continuity={
                "moving_nouns": "the folding paper squares and the pips printed on them",
                "static_nouns": "the blue base square, the paper colors, the pip patterns, the backdrop, and "
                "the camera",
                "hold_noun": "the closed box resting on the base with its lid pips facing up",
            },
            answer=AnswerContract(
                kind="nonnegative_integer",
                format_template="Answer: <1-6>.",
                parser_profile="strict_single_digit_1_6",
                verifier_profile=self.verifier_profile,
                format_example="Answer: 99.",
            ),
            render_profile="cube_net_fold_v1",
        )


__all__ = ["CubeNetFoldPattern", "replay_cube_net"]
