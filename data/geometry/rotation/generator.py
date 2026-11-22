from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.choice_cards import attach_visual_choice_card_inputs
from data.common.pipeline.media import LETTER_OPTIONS, load_font
from data.common.pipeline.rendered_task import RenderedTaskGenerator
from data.common.prompts import build_sectioned_prompt
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
    stroke_closed_ring,
)
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.typed_answers import FOUR_CHOICE_CODEC
from data.geometry.core.rendered_contract import finalize_geometry_sample
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph



_TRANSFORMS = ("identity", "rot90_cw", "rot180", "rot90_ccw")
_TARGET_TRANSFORMS = _TRANSFORMS[1:]
_SIGNED_CLOCKWISE_DEGREES = {
    "identity": 0,
    "rot90_cw": 90,
    "rot180": 180,
    "rot90_ccw": -90,
}
_ROTATION_RULE_TEXT = {
    "rot90_cw": "90 degrees clockwise",
    "rot180": "180 degrees (a half-turn)",
    "rot90_ccw": "90 degrees counterclockwise",
}
_VERIFIER_PROFILE = "rotation_visible_pose_multimodal_hard_gate_v2"
_RENDER_PROFILE = "native_asymmetric_rotation_cardless_v1"
_ANSWER_FORMAT_EXAMPLE = FOUR_CHOICE_CODEC.example
_POLYGON_FILL = (40, 120, 156)
_POLYGON_OUTLINE = (24, 43, 52)
_BACKGROUND = (247, 249, 252)
_CARD = (255, 255, 255)
_BORDER = (112, 123, 137)
_MUTED = (117, 128, 144)


def _answer_schema() -> Dict[str, Any]:
    return {
        "kind": "option_letter",
        "pattern": FOUR_CHOICE_CODEC.capture_pattern,
        "format_template": FOUR_CHOICE_CODEC.schema,
        "format_example": _ANSWER_FORMAT_EXAMPLE,
        "strict_full_match": True,
        "surrounding_whitespace_allowed": False,
        "extra_text_allowed": False,
    }


def _verifier_contract() -> Dict[str, Any]:
    return {
        "profile": _VERIFIER_PROFILE,
        "judged_artifacts": ["text", "final_image", "full_video"],
        "strict_text": True,
        "shape_only_pose_replay": True,
        "full_video_required": True,
    }


def _render_geometry() -> Dict[str, Any]:
    return {
        "canvas_size": [512, 512],
        "source_box": [28, 72, 236, 440],
        # Square tile slots: the authored tile is square, so a square paste
        # region keeps the silhouette undistorted in every pose.
        "source_tile_box": [48, 172, 216, 340],
        "target_box": [276, 72, 484, 440],
        "target_tile_box": [296, 172, 464, 340],
        "workspace_arrow_x": 256,
        "video_frame_count": 29,
        "video_fps": 16,
        "phase_frames": {
            "input_hold": [1, 5],
            "target_reveal": [6, 9],
            "rotation": [10, 25],
            "final_hold": [26, 29],
        },
    }


def _expected_state_trace() -> list[Dict[str, Any]]:
    return [
        {"phase": "input", "target_visible": False, "rotation_progress": 0.0},
        {"phase": "target_reveal", "target_visible": True, "rotation_progress": 0.0},
        {"phase": "rotation", "target_visible": True, "rotation_progress": 1.0},
        {"phase": "final", "target_visible": True, "rotation_progress": 1.0},
    ]


def _pose_digest(tile: Image.Image) -> str:
    return hashlib.sha256(tile.convert("RGB").tobytes()).hexdigest()


def _rotate_tile(tile: Image.Image, transform: str) -> Image.Image:
    if transform == "identity":
        return tile.convert("RGB").copy()
    degrees = {
        "rot90_cw": -90,
        "rot180": 180,
        "rot90_ccw": 90,
    }.get(transform)
    if degrees is None:
        raise ValueError(f"unknown rotation transform: {transform}")
    return tile.convert("RGB").rotate(
        degrees,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=_CARD,
    )


# Square tile whose polygon stays inside the inscribed circle, so every
# intermediate video rotation angle and all four right-angle poses render
# without corner clipping or aspect distortion.  Tile coordinates below stay in
# this logical space; the tile bitmap itself is kept supersampled so the
# silhouette outline, every intermediate rotation angle, and the eventual paste
# into the workspace all resolve at the family line-quality floor.
_TILE_SIZE = 300
_TILE_CONTENT_RADIUS = 128


def _make_polygon_tile(rng) -> Image.Image:
    """Sample one asymmetric angular silhouette per sample.

    The previous implementation drew a single hardcoded heptagon, so the
    entire dataset shared one memorizable shape.  The sampler now draws a
    fresh star-shaped polygon and leaves pose distinguishability to the
    caller's pixel-margin guard.
    """

    center = _TILE_SIZE / 2.0
    for _ in range(400):
        count = rng.randint(6, 8)
        step = 2.0 * math.pi / count
        base = rng.uniform(0.0, 2.0 * math.pi)
        angles = [
            base + index * step + rng.uniform(-0.5, 0.5) * step * 0.75
            for index in range(count)
        ]
        radii = [rng.uniform(58.0, _TILE_CONTENT_RADIUS - 4.0) for _ in range(count)]
        if max(radii) < 106.0:
            continue
        if max(radii) - min(radii) < 34.0:
            continue
        points = [
            (center + radius * math.cos(angle), center + radius * math.sin(angle))
            for angle, radius in zip(angles, radii)
        ]
        if any(
            math.hypot(
                points[index][0] - points[(index + 1) % count][0],
                points[index][1] - points[(index + 1) % count][1],
            )
            < 30.0
            for index in range(count)
        ):
            continue
        tile, draw = new_scaled_canvas(
            _TILE_SIZE, _TILE_SIZE, DOCTRINE_SUPERSAMPLE, background=_CARD
        )
        int_points = [(int(round(x)), int(round(y))) for x, y in points]
        draw.polygon(int_points, fill=_POLYGON_FILL)
        stroke_closed_ring(draw, int_points, fill=_POLYGON_OUTLINE, width=6)
        return tile
    raise RuntimeError("Failed to sample an asymmetric rotation silhouette.")


def _pairwise_pose_margin(poses: Dict[str, Image.Image]) -> float:
    """Smallest fraction of polygon pixels by which any two poses differ."""

    arrays = {
        name: np.asarray(tile.convert("L"), dtype=np.int16)
        for name, tile in poses.items()
    }
    filled = max(1, int((arrays["identity"] < 235).sum()))
    names = list(arrays)
    smallest = 1.0
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            differing = int(
                (np.abs(arrays[names[left]] - arrays[names[right]]) > 40).sum()
            )
            smallest = min(smallest, differing / filled)
    return smallest


def _option_plate(tile: Image.Image, label: str) -> Image.Image:
    # Match the shared Geometry hypothesis-plate style: a plain white plate
    # with only the pose content and one bottom-centered label, no card
    # rectangle or border.
    plate = Image.new("RGB", (512, 512), _CARD)
    fitted = tile.resize((360, 360), Image.Resampling.LANCZOS)
    plate.paste(fitted, ((512 - fitted.width) // 2, max(20, (438 - fitted.height) // 2)))
    draw = ImageDraw.Draw(plate)
    draw.text((256, 474), label, fill=(42, 46, 52), font=load_font(30), anchor="mm")
    return plate


def _replay_rotation(structured: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    target_transform = structured.get("target_transform")
    candidate_transforms = structured.get("candidate_transforms")
    if target_transform not in _TARGET_TRANSFORMS:
        return {}, False
    if (
        not isinstance(candidate_transforms, list)
        or len(candidate_transforms) != 4
        or set(candidate_transforms) != set(_TRANSFORMS)
        or len(set(candidate_transforms)) != 4
    ):
        return {}, False
    correct_index = candidate_transforms.index(target_transform)
    correct_option = LETTER_OPTIONS[correct_index]
    replay = {
        "target_transform": target_transform,
        "target_clockwise_degrees": _SIGNED_CLOCKWISE_DEGREES[target_transform],
        "candidate_transforms": list(candidate_transforms),
        "correct_index": correct_index,
        "correct_option": correct_option,
        "state_trace": _expected_state_trace(),
        "render_geometry": _render_geometry(),
        "answer_schema": _answer_schema(),
        "verifier_contract": _verifier_contract(),
    }
    metadata_valid = bool(
        structured.get("source_transform") == "identity"
        and structured.get("target_clockwise_degrees") == replay["target_clockwise_degrees"]
        and structured.get("answer_index") == correct_index
        and structured.get("correct_option") == correct_option
        and structured.get("state_trace") == replay["state_trace"]
        and structured.get("render_geometry") == replay["render_geometry"]
        and structured.get("answer_schema") == replay["answer_schema"]
        and structured.get("verifier_contract") == replay["verifier_contract"]
        and structured.get("multiple_solution_policy")
        == "exactly_one_visible_candidate_matches_the_rotated_source_pose"
    )
    return replay, metadata_valid


class RotationPattern(RenderedTaskGenerator):
    TASK_NAME = "rotation"
    TASK_FAMILY = "geometry"
    RENDER_SCALE = 1

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        image_size: Tuple[int, int] | int | None = None,
        video_fps: int | None = None,
        generate_videos: bool = True,
    ) -> None:
        normalized = (int(image_size), int(image_size)) if isinstance(image_size, int) else tuple(image_size or (512, 512))
        if normalized != (512, 512):
            raise ValueError("rotation renders are fixed at 512x512")
        if video_fps not in (None, 16):
            raise ValueError("rotation videos are fixed at 16 fps")
        super().__init__(
            output_dir=output_dir,
            seed=seed,
            image_size=(512, 512),
            video_fps=16,
            generate_videos=generate_videos,
        )

    def _generate_task(self) -> Dict[str, Any]:
        for _ in range(200):
            source_tile = _make_polygon_tile(self.rng)
            poses = {transform: _rotate_tile(source_tile, transform) for transform in _TRANSFORMS}
            if len({_pose_digest(tile) for tile in poses.values()}) != 4:
                continue
            # Perceptual margin: every pair of rigid poses must differ on a
            # substantial fraction of the polygon, so no near-symmetric shape
            # ever turns pose selection into a pixel-hunt.
            if _pairwise_pose_margin(poses) < 0.25:
                continue
            break
        else:
            raise RuntimeError("rotation polygon must have four clearly distinct rigid poses")
        target_transform = self.rng.choice(_TARGET_TRANSFORMS)
        candidate_transforms = list(_TRANSFORMS)
        self.rng.shuffle(candidate_transforms)
        return {
            "source_tile": source_tile,
            "poses": poses,
            "target_transform": target_transform,
            "target_clockwise_degrees": _SIGNED_CLOCKWISE_DEGREES[target_transform],
            "candidate_transforms": candidate_transforms,
            "answer_index": candidate_transforms.index(target_transform),
        }

    @staticmethod
    def _paste_tile(
        canvas: Image.Image,
        tile: Image.Image,
        box: Sequence[int],
        progress: float = 1.0,
        *,
        factor: int = 1,
    ) -> None:
        """Fit a supersampled tile into a workspace slot declared in 512-space."""

        left, top, right, bottom = [int(value) * factor for value in box]
        target_size = (right - left, bottom - top)
        fitted = tile.resize(target_size, Image.Resampling.LANCZOS)
        value = max(0.0, min(1.0, float(progress)))
        if value >= 1.0:
            canvas.paste(fitted, (left, top))
            return
        mask = Image.new("L", fitted.size, int(round(255 * value)))
        canvas.paste(fitted, (left, top), mask)

    def _render_workspace(
        self,
        task: Dict[str, Any],
        *,
        target_tile: Image.Image | None,
        target_reveal_progress: float = 1.0,
    ) -> Image.Image:
        geometry = _render_geometry()
        # Card borders and glyphs are laid down on a supersampled canvas and
        # LANCZOS-downsampled once, so no stroke is ever rasterized at final
        # resolution.  Slot geometry stays in the declared 512-space.
        factor = DOCTRINE_SUPERSAMPLE
        image, draw = new_scaled_canvas(512, 512, factor, background=_BACKGROUND)
        source_box = geometry["source_box"]
        target_box = geometry["target_box"]
        for box in (source_box, target_box):
            draw.rounded_rectangle(tuple(box), radius=18, fill=_CARD, outline=_BORDER, width=3)
        self._paste_tile(image, task["source_tile"], geometry["source_tile_box"], factor=factor)
        draw.text(
            (256, 256), "\N{RIGHTWARDS ARROW}", fill=_POLYGON_OUTLINE,
            font=load_font(40 * factor), anchor="mm",
        )
        if target_tile is None:
            draw.text(
                (380, 256), "?", fill=_MUTED,
                font=load_font(62 * factor), anchor="mm",
            )
        else:
            self._paste_tile(
                image,
                target_tile,
                geometry["target_tile_box"],
                progress=target_reveal_progress,
                factor=factor,
            )
        return finish_scaled_canvas(image, (512, 512))

    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_workspace(task, target_tile=None)

    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        return self._render_workspace(task, target_tile=task["poses"][task["target_transform"]])

    def _render_transition_image(self, task: Dict[str, Any], *, reveal_progress: float = 1.0, rotation_progress: float = 0.0) -> Image.Image:
        progress = max(0.0, min(1.0, float(rotation_progress)))
        if progress >= 1.0:
            tile = task["poses"][task["target_transform"]]
        else:
            degrees = -float(task["target_clockwise_degrees"]) * progress
            tile = task["source_tile"].rotate(
                degrees,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=_CARD,
            )
        return self._render_workspace(
            task,
            target_tile=tile,
            target_reveal_progress=reveal_progress,
        )

    def _render_video(self, task: Dict[str, Any], output_path: Path) -> int:
        frames: list[np.ndarray] = []
        puzzle = np.asarray(self._render_puzzle_image(task), dtype=np.uint8)
        frames.extend(puzzle.copy() for _ in range(5))
        for step in range(1, 5):
            frames.append(np.asarray(self._render_transition_image(task, reveal_progress=step / 4.0), dtype=np.uint8))
        for step in range(1, 17):
            frames.append(np.asarray(self._render_transition_image(task, rotation_progress=step / 16.0), dtype=np.uint8))
        final = np.asarray(self._render_solution_image(task), dtype=np.uint8)
        frames.extend(final.copy() for _ in range(4))
        if len(frames) != 29 or (len(frames) - 1) % 4 != 0:
            raise AssertionError("rotation video must contain 29 (4N+1) frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _answer_raw(self, task: Dict[str, Any]) -> str:
        return LETTER_OPTIONS[int(task["answer_index"])]

    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_transform": "identity",
            "target_transform": task["target_transform"],
            "target_clockwise_degrees": int(task["target_clockwise_degrees"]),
            "candidate_transforms": list(task["candidate_transforms"]),
            "answer_index": int(task["answer_index"]),
            "correct_option": self._answer_raw(task),
            "state_trace": _expected_state_trace(),
            "render_geometry": _render_geometry(),
            "answer_schema": _answer_schema(),
            "verifier_contract": _verifier_contract(),
            "multiple_solution_policy": "exactly_one_visible_candidate_matches_the_rotated_source_pose",
        }

    def _question_text(self, task: Dict[str, Any]) -> str:
        del task
        return "Rotate the source polygon as requested and choose the matching pose."

    def _visual_input_description(self, task: Dict[str, Any]) -> str:
        del task
        return "The single image is a centered source polygon workspace with an empty target slot."

    def _visual_output_description(self, task: Dict[str, Any]) -> str:
        return (
            f"{self._rules_description(task)} Show the completed source/target workspace without candidate cards "
            "or a selection border."
        )

    def _rules_description(self, task: Dict[str, Any]) -> str:
        return (
            f"Rotate the entire filled polygon exactly {_ROTATION_RULE_TEXT[task['target_transform']]} about the center of its tile. "
            "The outline is one rigid object: do not reflect, resize, translate, crop, recolor, or deform it."
        )

    def _build_prompts(self, task: Dict[str, Any], answer_raw: str) -> Dict[str, str]:
        rule = self._rules_description(task)
        ti2v_input = "The single image is the source workspace with one polygon and an empty target slot."
        return {
            "question": self._question_text(task),
            "ti2v_prompt": build_sectioned_prompt(
                ti2v_input,
                "Reveal one complete copy in the target slot and rotate it continuously through the requested motion. Do not show candidate cards or an answer label.",
                mode="ti2v",
                rule_description=rule,
                continuity_paragraph=build_ti2v_continuity_paragraph(
                    timing="continuous",
                    moving_nouns="the rotating copy",
                    static_nouns="everything the rules leave unchanged, including the source shape, labels, background, and camera",
                    hold_noun="the completed source and target workspace",
                ),
            ),
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        answer_raw = self._answer_raw(task)
        sample_id = f"{self.TASK_NAME}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        puzzle = self._render_puzzle_image(task)
        solution = self._render_solution_image(task)
        puzzle.save(puzzle_path)
        solution.save(solution_path)

        option_paths: list[str] = []
        for index, transform in enumerate(task["candidate_transforms"]):
            path = sample_dir / f"{sample_id}__option_{LETTER_OPTIONS[index]}.png"
            _option_plate(task["poses"][transform], LETTER_OPTIONS[index]).save(path)
            option_paths.append(path.as_posix())

        video_path: Path | None = None
        frame_count: int | None = None
        if self.generate_videos:
            video_path = sample_dir / f"{sample_id}__solution.webm"
            frame_count = self._render_video(task, video_path)
            if not frame_count:
                video_path = None

        structured = self._structured_answer(task)
        replay, valid = _replay_rotation(structured)
        if not valid or not replay:
            raise RuntimeError("Generated rotation sample failed independent semantic replay.")
        record: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.TASK_NAME,
            "task_family": self.TASK_FAMILY,
            "task_contract": {
                "input_count": 5,
                "input_media_policy": "visual_choice_cards_v1",
                "input_count_by_modality": { "ti2v": 1},
                "canvas_size": [512, 512],
                "render_profile": _RENDER_PROFILE,
                "answer": {
                    "kind": "option",
                    "format_template": FOUR_CHOICE_CODEC.schema,
                    "format_example": _ANSWER_FORMAT_EXAMPLE,
                    "verifier_profile": _VERIFIER_PROFILE,
                },
                "transition": "hold_input_then_reveal_target_copy_then_rotate_then_hold_solution",
            },
            "question": self._question_text(task),
            "answer": answer_raw,
            "correct_option": answer_raw,
            "structured_answer": structured,
            "verifier_contract": structured["verifier_contract"],
            "ti2v_prompt": self._build_prompts(task, answer_raw)["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": video_path.as_posix() if video_path else None,
            "video_fps": self.video_fps if video_path else None,
            "video_num_frames": frame_count,
        }
        attach_visual_choice_card_inputs(
            record,
            problem_image=puzzle_path.as_posix(),
            option_images=option_paths,
            option_labels=LETTER_OPTIONS[:4],
            ti2v_problem_image=puzzle_path.as_posix(),
        )
        return finalize_geometry_sample(record), puzzle, solution


__all__ = ["RotationPattern", "_replay_rotation", "_render_geometry"]
