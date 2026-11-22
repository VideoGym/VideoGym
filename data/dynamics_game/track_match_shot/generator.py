"""Insert one launched token into a linear track and clear its matching run."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.prompts import build_sectioned_prompt
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import CANVAS_SIZE, DEFAULT_FPS
from data.dynamics_game.core.scene import finish_scene, new_scene, scene_font
from data.common.video_encoding import (
    encode_rgb_frames_to_webm,
)
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the launched ball and the cleared run",
    static_nouns="everything the rules leave unchanged, including the board, labels, background, and camera",
    hold_noun="the final board state",
)


_INPUT_DESC = (
    "The single problem image shows a horizontal track of eight wells numbered 1 through 8. Each well is pale and empty or "
    "contains one red, green, blue, or yellow ball. Exactly one empty well has a gold outline. A launcher below the "
    "track contains one colored ball; that is the ball to insert into the gold-outlined well."
)
_RULE_SUMMARY = (
    "Move the launcher ball into the gold-outlined well without shifting any existing ball. Starting from the "
    "inserted ball, extend left and right through consecutive balls of the same color. If that one contiguous run "
    "contains at least three balls, remove the entire run; if it is shorter, nothing is removed. Every removed "
    "position becomes a pale empty well. Balls outside that run stay fixed; there is no gravity, refill, second "
    "clear, or movement along the track."
)
_TI2V_OUTPUT = (
    "Hold the input, then move the same launcher ball continuously to the center of the gold-outlined well; "
    "the launcher chamber becomes pale and empty as soon as the ball leaves. Hold the complete filled track "
    "briefly. If the contiguous same-color run through the inserted ball reaches three balls, shrink exactly "
    "that run together into pale empty wells; otherwise change nothing further. Hold the final track."
)


_BG = (252, 252, 254)
_TRACK_FILL = (245, 247, 251)
_TRACK_LINE = (210, 215, 224)
_LABEL_COLOR = (60, 70, 90)
_GAP_RING = (218, 165, 32)
_LAUNCHER = (64, 103, 180)
_LAUNCHER_DARK = (34, 64, 126)
_BLACK = (50, 60, 75)


@dataclass
class ZumaTrackShotRecord:
    id: str
    task_name: str = "track_match_shot"
    task_family: str = "dynamics_game"
    image: str = ""
    reasoning_image: str = ""
    solution_image_path: str = ""
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    ti2v_prompt: str = ""
    solution_image_prompt: str = ""
    answer: str = ""
    question: str = ""
    question_description: str = "track_shot"
    question_id: int = 0
    qa_type: str = "State Prediction"
    qa_level: str = "Medium"
    plot_level: str = "Medium"
    answer_grid: List[List[int]] = field(default_factory=list)
    structured_answer: Dict[str, Any] = field(default_factory=dict)
    answer_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "image": self.image,
            "reasoning_image": self.reasoning_image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "ti2v_prompt": self.ti2v_prompt,
            "answer": self.answer,
            "question": self.question,
            "question_description": self.question_description,
            "question_id": self.question_id,
            "qa_type": self.qa_type,
            "qa_level": self.qa_level,
            "plot_level": self.plot_level,
            "answer_grid": self.answer_grid,
            "structured_answer": self.structured_answer,
            "answer_payload": self.answer_payload,
        }


class ZumaTrackShotGenerator(AbstractPuzzleGenerator[ZumaTrackShotRecord]):
    COLOR_IDS = {"red": 1, "green": 2, "blue": 3, "yellow": 4}
    COLOR_RGB = {
        0: (255, 255, 255),
        1: (224, 86, 74),
        2: (77, 166, 106),
        3: (76, 124, 226),
        4: (244, 196, 77),
    }
    COLOR_DARK = {
        1: (130, 30, 25),
        2: (35, 100, 55),
        3: (30, 60, 150),
        4: (165, 120, 30),
    }

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        slots: int = 8,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
    ) -> None:
        super().__init__(output_dir)
        self.seed = seed
        self.rng = random.Random(seed)
        self.slots = slots
        self.video_fps = video_fps
        self.video = video
        self._counter = 0

        self.cell = max(40, (CANVAS_SIZE - 96) // self.slots)
        self.cell -= self.cell % 2  # even.
        track_width = self.slots * self.cell
        self.track_x0 = (CANVAS_SIZE - track_width) // 2
        self.track_x0 -= self.track_x0 % 2
        self.track_y0 = 155
        self.launcher_y = 386

    def record_to_dict(self, record: ZumaTrackShotRecord | Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(record)

    # ---- scenario ----------------------------------------------------------
    def _slot_box(self, idx: int) -> Tuple[int, int, int, int]:
        x0 = self.track_x0 + idx * self.cell
        return x0, self.track_y0, x0 + self.cell, self.track_y0 + self.cell

    def _slot_center(self, idx: int) -> Tuple[float, float]:
        x0, y0, x1, y1 = self._slot_box(idx)
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def _initial_state(self) -> Tuple[List[int], int, str]:
        names = list(self.COLOR_IDS)
        projectile = self.rng.choice(names)
        pid = self.COLOR_IDS[projectile]
        gap_slot = self.rng.randrange(2, self.slots - 2)
        track = [0 for _ in range(self.slots)]
        if self.rng.random() < 0.72:
            left_count = self.rng.choice((1, 2))
            right_count = self.rng.choice((1, 2))
        else:
            # Below-threshold instance: the insertion completes a run of one or
            # two, so the explicit no-clear rule branch is load-bearing.
            left_count, right_count = self.rng.choice(((0, 0), (1, 0), (0, 1)))
        for index in range(gap_slot - left_count, gap_slot):
            track[index] = pid
        for index in range(gap_slot + 1, gap_slot + right_count + 1):
            track[index] = pid

        protected = set(range(gap_slot - left_count, gap_slot + right_count + 1))
        rest = [i for i in range(self.slots) if i not in protected]
        self.rng.shuffle(rest)
        for i in rest:
            other = [n for n in names if self.COLOR_IDS[n] != pid]
            track[i] = self.COLOR_IDS[self.rng.choice(other)]
        return track, gap_slot, projectile

    def _simulate(self, track: Sequence[int], gap_slot: int, projectile: str) -> Tuple[List[int], List[int], int, int]:
        """Return (filled_track, final_track, run_left, run_right_exclusive)."""
        pid = self.COLOR_IDS[projectile]
        filled = list(track)
        filled[gap_slot] = pid
        left = gap_slot
        while left - 1 >= 0 and filled[left - 1] == pid:
            left -= 1
        right = gap_slot
        while right + 1 < len(filled) and filled[right + 1] == pid:
            right += 1
        final = list(filled)
        if right - left + 1 >= 3:
            for i in range(left, right + 1):
                final[i] = 0
        return filled, final, left, right + 1

    # ---- rendering ---------------------------------------------------------
    def _render_track(
        self,
        track: Sequence[int],
        *,
        gap_slot: Optional[int],
        projectile: str,
        flying_ball: Optional[Tuple[float, float, int]] = None,
        clear_scale_indices: Optional[Tuple[int, int, float]] = None,
        suppress_indices: Optional[Sequence[int]] = None,
        launcher_loaded: bool = True,
    ) -> Image.Image:
        image, draw = new_scene(_BG)
        font = scene_font(17)
        suppress = set(suppress_indices or ())
        draw.rounded_rectangle([28, 105, 484, 270], radius=24, fill=(255, 255, 255), outline=(210, 220, 234), width=2)

        # Slot boxes + labels.
        for idx in range(self.slots):
            x0, y0, x1, y1 = self._slot_box(idx)
            highlight = (gap_slot == idx)
            draw.rounded_rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], radius=11, fill=_TRACK_FILL, outline=_TRACK_LINE, width=2)
            if highlight:
                inset = 3
                draw.rounded_rectangle(
                    [x0 + inset, y0 + inset, x1 - inset, y1 - inset],
                    radius=8,
                    outline=_GAP_RING,
                    width=3,
                )
            draw.text((x0 + self.cell / 2, y0 - 18), str(idx + 1), fill=_LABEL_COLOR, font=font, anchor="mm")

        # Balls. Skip suppressed indices (used while a ball is flying out of the track).
        for idx in range(self.slots):
            if idx in suppress:
                continue
            v = int(track[idx])
            if v == 0:
                continue
            cx, cy = self._slot_center(idx)
            scale = 1.0
            if clear_scale_indices is not None:
                lo, hi, value = clear_scale_indices
                if lo <= idx < hi:
                    scale = value
            self._draw_ball(draw, (cx, cy), v, scale=scale)

        # Flying ball overlay.
        if flying_ball is not None:
            fx, fy, fcid = flying_ball
            self._draw_ball(draw, (fx, fy), fcid)

        self._draw_launcher(draw, projectile, loaded=launcher_loaded)

        return finish_scene(image)

    def _draw_ball(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        color_id: int,
        *,
        scale: float = 1.0,
    ) -> None:
        cx, cy = center
        r = self.cell * 0.34 * max(0.08, scale)
        base = self.COLOR_RGB.get(color_id, (200, 200, 200))
        outline = self.COLOR_DARK.get(color_id, _BLACK)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base, outline=outline, width=2)

    def _draw_launcher(self, draw: ImageDraw.ImageDraw, projectile: str, *, loaded: bool) -> None:
        cx = CANVAS_SIZE // 2
        cy = self.launcher_y
        draw.rounded_rectangle([cx - 42, cy - 34, cx + 42, cy + 42], radius=22, fill=_LAUNCHER, outline=_LAUNCHER_DARK, width=3)
        draw.polygon([(cx, cy - 26), (cx - 10, cy - 10), (cx + 10, cy - 10)], fill=(255, 255, 255))
        draw.ellipse([cx - 22, cy - 3, cx + 22, cy + 41], fill=(238, 243, 250), outline=_LAUNCHER_DARK, width=2)
        if not loaded:
            return
        pid = self.COLOR_IDS[projectile]
        self._draw_ball(draw, (cx, cy + 19), pid, scale=0.8)

    # ---- video -------------------------------------------------------------
    def _launcher_ball_origin(self) -> Tuple[float, float]:
        cx = CANVAS_SIZE / 2.0
        cy = self.launcher_y + 19
        return cx, cy

    def _write_video(
        self,
        track: Sequence[int],
        gap_slot: int,
        projectile: str,
        video_path: Path,
    ) -> int:
        pid = self.COLOR_IDS[projectile]
        filled, final, run_lo, run_hi = self._simulate(track, gap_slot, projectile)

        frames: List[np.ndarray] = []
        initial = self._render_track(track, gap_slot=gap_slot, projectile=projectile)
        frames.extend(np.asarray(initial, dtype=np.uint8).copy() for _ in range(8))
        ox, oy = self._launcher_ball_origin()
        gx, gy = self._slot_center(gap_slot)
        for k in range(1, 17):
            t = k / 16.0
            cx = ox + (gx - ox) * t
            cy = oy + (gy - oy) * t
            frame = self._render_track(
                track,
                gap_slot=gap_slot,
                projectile=projectile,
                flying_ball=(cx, cy, pid),
                launcher_loaded=False,
            )
            frames.append(np.asarray(frame, dtype=np.uint8))
        landed = self._render_track(filled, gap_slot=None, projectile=projectile, launcher_loaded=False)
        frames.extend(np.asarray(landed, dtype=np.uint8).copy() for _ in range(5))
        cleared = run_hi - run_lo >= 3
        if cleared:
            for scale in (0.82, 0.64, 0.46, 0.30, 0.18, 0.08, 0.08, 0.08):
                frame = self._render_track(
                    filled, gap_slot=None, projectile=projectile,
                    clear_scale_indices=(run_lo, run_hi, scale),
                    launcher_loaded=False,
                )
                frames.append(np.asarray(frame, dtype=np.uint8))
        final_image = self._render_track(final, gap_slot=None, projectile=projectile, launcher_loaded=False)
        frames.extend(np.asarray(final_image, dtype=np.uint8).copy() for _ in range(12))
        assert len(frames) == (49 if cleared else 41)
        if not encode_rgb_frames_to_webm(
            frames,
            video_path,
            fps=self.video_fps,
            extra_args=["-g", "1", "-bf", "0"],
        ):
            return 0
        return len(frames)

    # ---- prompts -----------------------------------------------------------
    def _prompts(self) -> Dict[str, str]:
        ti2v = build_sectioned_prompt(
            _INPUT_DESC,
            _TI2V_OUTPUT,
            mode="ti2v",
            rule_description=_RULE_SUMMARY,
            continuity_paragraph=_TI2V_CONTINUITY,
        )
        return {
            "ti2v_prompt": ti2v,
        }

    # ---- entry -------------------------------------------------------------
    def create_puzzle(self) -> ZumaTrackShotRecord:
        sample_id = f"track_match_shot-{self._counter:05d}"
        self._counter += 1

        track, gap_slot, projectile = self._initial_state()
        filled_track, final_track, run_lo, run_hi = self._simulate(track, gap_slot, projectile)

        puzzles_dir = self.output_dir / "puzzles"
        solutions_dir = self.output_dir / "solutions"
        videos_dir = self.output_dir / "videos"
        for d in (puzzles_dir, solutions_dir, videos_dir):
            d.mkdir(parents=True, exist_ok=True)

        puzzle_path = puzzles_dir / f"{sample_id}.png"
        solution_path = solutions_dir / f"{sample_id}.png"
        video_path = videos_dir / f"{sample_id}.webm"
        self._render_track(track, gap_slot=gap_slot, projectile=projectile).save(puzzle_path)
        self._render_track(final_track, gap_slot=None, projectile=projectile, launcher_loaded=False).save(solution_path)

        frame_count = self._write_video(track, gap_slot, projectile, video_path) if self.video else 0

        prompts = self._prompts()
        payload = {
            "state_grid": [final_track],
            "initial_grid": [list(track)],
            "state_trace": [[list(track)], [filled_track], [final_track]],
            "process_state_trace": [[list(track)], [filled_track], [final_track]],
            "filled_track": filled_track,
            "gap_slot": gap_slot,
            "projectile_color": projectile,
            "projectile_color_id": self.COLOR_IDS[projectile],
            "clear_run": [run_lo, run_hi],
        }

        return ZumaTrackShotRecord(
            id=sample_id,
            image=self.relativize_path(puzzle_path),
            reasoning_image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            solution_video_path=self.relativize_path(video_path) if frame_count > 0 else None,
            video_fps=self.video_fps if frame_count > 0 else None,
            video_num_frames=frame_count if frame_count > 0 else None,
            ti2v_prompt=prompts["ti2v_prompt"],


            answer=json.dumps([final_track]),


            question="Return the exact final track after inserting the launcher ball and applying the one allowed run clear.",
            answer_grid=[final_track],
            structured_answer=payload,
            answer_payload=payload,
        )


__all__ = ["ZumaTrackShotGenerator", "ZumaTrackShotRecord"]
