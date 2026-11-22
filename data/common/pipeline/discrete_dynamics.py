"""Shared record, option-shuffling, and video utilities for discrete dynamics tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.multiple_choice import save_labeled_option_images
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.video_encoding import (
    MOTION_VIDEO_FRAMES,
    encode_rgb_frames_to_webm,
    fit_frames_to_video_budget,
)

BG = (250, 250, 252)
GRID_LINE = (210, 215, 224)
LABEL_COLOR = (60, 70, 90)


def make_record_dataclass(task_name_value: str, task_family_value: str):
    """Build a per-task record dataclass with the canonical fields."""

    @dataclass
    class _DynamicsRecord:
        id: str
        task_name: str = task_name_value
        task_family: str = task_family_value
        image: str = ""
        reasoning_image: str = ""
        reasoning_images: List[str] = field(default_factory=list)
        solution_image_path: str = ""
        solution_video_path: Optional[str] = None
        video_fps: Optional[int] = None
        video_num_frames: Optional[int] = None
        ti2v_prompt: str = ""
        solution_image_prompt: str = ""
        answer: str = ""
        correct_option: str = ""
        question: str = ""
        question_description: str = task_name_value
        question_id: int = 0
        qa_type: str = "State Prediction"
        qa_level: str = "Medium"
        plot_level: str = "Medium"
        structured_answer: Dict[str, Any] = field(default_factory=dict)
        answer_payload: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "id": self.id,
                "task_name": self.task_name,
                "task_family": self.task_family,
                "image": self.image,
                "reasoning_image": self.reasoning_image,
                "reasoning_images": list(self.reasoning_images),
                "solution_image_path": self.solution_image_path,
                "solution_video_path": self.solution_video_path,
                "video_fps": self.video_fps,
                "video_num_frames": self.video_num_frames,
                "ti2v_prompt": self.ti2v_prompt,
                "answer": self.answer,
                "correct_option": self.correct_option,
                "question": self.question,
                "question_description": self.question_description,
                "question_id": self.question_id,
                "qa_type": self.qa_type,
                "qa_level": self.qa_level,
                "plot_level": self.plot_level,
                "structured_answer": self.structured_answer,
                "answer_payload": self.answer_payload,
            }

    _DynamicsRecord.__name__ = f"{task_name_value.title().replace('_', '')}Record"
    return _DynamicsRecord


def shuffle_with_correct_semantics(
    correct_tile: Image.Image,
    correct_semantic: Any,
    distractor_tiles: Sequence[Image.Image],
    distractor_semantics: Sequence[Any],
    rng: random.Random,
) -> Tuple[List[Image.Image], List[Any], int]:
    """Shuffle four rendered options together with their semantic payloads."""
    if len(distractor_tiles) < 3 or len(distractor_semantics) < 3:
        raise ValueError("need >=3 distractors to build a semantic 4-option MCQ")
    options = list(distractor_tiles[:3]) + [correct_tile]
    semantics = list(distractor_semantics[:3]) + [correct_semantic]
    indices = list(range(4))
    rng.shuffle(indices)
    return [options[i] for i in indices], [semantics[i] for i in indices], indices.index(3)


def default_dissolve_video(
    puzzle_image: Image.Image,
    solution_image: Image.Image,
    video_path: Path,
    fps: int = DEFAULT_FPS,
    transition_frames: int = 6,
    hold_frames: int = 12,
) -> int:
    """Alpha-dissolve video between puzzle and solution snapshots."""
    frames = smooth_state_dissolve(
        [puzzle_image, solution_image],
        transition_frames=transition_frames,
        hold_frames=hold_frames,
    )
    return encode_video(frames, video_path, fps=fps)


def write_artifacts(
    *,
    output_dir: Path,
    sample_id: str,
    puzzle_image: Image.Image,
    solution_image: Image.Image,
    option_tiles: Sequence[Image.Image],
    correct_index: int,
    payload: Dict[str, Any],
    write_video: Optional[Callable[[Path], int]] = None,
    enable_default_video: bool = False,
    video_keyframe_images: Optional[Sequence[Image.Image]] = None,
) -> Dict[str, Any]:
    """Save the canonical artifacts and return the path bundle."""
    puzzle_dir = output_dir / "puzzles"
    solution_dir = output_dir / "solutions"
    options_dir = output_dir / "reasoning_options"
    video_dir = output_dir / "videos"
    state_dir = output_dir / "states"
    for d in (puzzle_dir, solution_dir, options_dir, video_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    puzzle_path = puzzle_dir / f"{sample_id}.png"
    solution_path = solution_dir / f"{sample_id}.png"
    video_path = video_dir / f"{sample_id}.webm"
    puzzle_image.save(puzzle_path)
    solution_image.save(solution_path)
    saved_options = []
    if option_tiles:
        saved_options = save_labeled_option_images(
            list(option_tiles), options_dir, stem=sample_id, labels=("A", "B", "C", "D")
        )
    if (write_video is not None or enable_default_video) and video_keyframe_images:
        video_keyframe_paths: List[str] = []
        for index, state_image in enumerate(video_keyframe_images, start=1):
            state_image_path = state_dir / f"{sample_id}__video_keyframe_{index:02d}.png"
            state_image.save(state_image_path)
            video_keyframe_paths.append(state_image_path.relative_to(output_dir).as_posix())
        payload["video_keyframe_paths"] = video_keyframe_paths
    frame_count = 0
    if write_video is not None:
        frame_count = write_video(video_path)
    elif enable_default_video:
        frame_count = default_dissolve_video(puzzle_image, solution_image, video_path)

    return {
        "puzzle_path": puzzle_path,
        "solution_path": solution_path,
        "options_paths": saved_options,
        "video_path": video_path,
        "frame_count": frame_count,
    }


def encode_video(frames: List[np.ndarray], video_path: Path, fps: int = DEFAULT_FPS) -> int:
    """Budget, pad, encode VP9 lossless WebM."""
    frames = fit_frames_to_video_budget(
        frames,
        max_frames=MOTION_VIDEO_FRAMES,
        final_hold_frames=max(4, fps // 4),
    )
    if not encode_rgb_frames_to_webm(
        frames,
        video_path,
        fps=fps,
        extra_args=["-g", "1", "-bf", "0"],
    ):
        return 0
    return len(frames)


def smooth_state_dissolve(
    states: Sequence[Image.Image],
    *,
    transition_frames: int = 6,
    hold_frames: int = 8,
) -> List[np.ndarray]:
    """Build alpha-dissolve frames between successive snapshots."""
    if not states:
        return []
    arrays = [np.array(state.convert("RGB"), dtype=np.uint8) for state in states]
    frames: List[np.ndarray] = []
    for _ in range(hold_frames):
        frames.append(arrays[0].copy())
    for i in range(len(arrays) - 1):
        a = arrays[i].astype(np.float32)
        b = arrays[i + 1].astype(np.float32)
        for k in range(1, transition_frames + 1):
            t = k / (transition_frames + 1)
            blended = (a * (1.0 - t) + b * t).clip(0, 255).astype(np.uint8)
            frames.append(blended)
        for _ in range(hold_frames):
            frames.append(arrays[i + 1].copy())
    return frames


__all__ = [
    "BG",
    "GRID_LINE",
    "LABEL_COLOR",
    "shuffle_with_correct_semantics",
    "default_dissolve_video",
    "encode_video",
    "make_record_dataclass",
    "smooth_state_dissolve",
    "write_artifacts",
]
