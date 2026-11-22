"""Shared MCQ helpers for discrete board-state tasks. Provides a uniform record dataclass + create_puzzle scaffolding that takes:
- A puzzle PNG
- A solution PNG
- 4 candidate option tiles (PIL Images)
- Correct option index (0..3)
- Prompt parts (input desc, rule summary, outputs, etc.)
- Optional semantic transition video
- Optional structured payload So every board-game task only needs to focus on its mechanic and visuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image

from data.common.pipeline.board_motion import (
    DEFAULT_FPS,
    build_state_dissolve_video,
)
from data.common.multiple_choice import (
    option_answer_text,
    option_label,
    save_labeled_option_images,
)


@dataclass
class BoardChoiceRecord:
    id: str
    task_family: str
    task_name: str = ""
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
    question_description: str = ""
    question_id: int = 0
    qa_type: str = "State Prediction"
    qa_level: str = "Medium"
    plot_level: str = "Medium"
    structured_answer: Dict[str, Any] = field(default_factory=dict)
    answer_payload: Dict[str, Any] = field(default_factory=dict)
    answer_grid: List[List[int]] = field(default_factory=list)
    rendered_glyph_inventory: Dict[str, List[str]] = field(default_factory=dict)
    rendered_glyph_schema: Dict[str, str] = field(default_factory=dict)

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
            "answer_grid": list(self.answer_grid) if self.answer_grid else [],
            "rendered_glyph_inventory": dict(self.rendered_glyph_inventory),
            "rendered_glyph_schema": dict(self.rendered_glyph_schema),
        }


def write_board_choice_record(
    *,
    generator: Any,
    sample_id: str,
    task_name: str,
    task_family: str,
    question_description: str,
    puzzle_image: Image.Image,
    solution_image: Image.Image,
    option_tiles: Sequence[Image.Image],
    correct_option_index: Optional[int],
    prompts: Dict[str, str],
    structured_payload: Dict[str, Any],
    question_text: str,
    video_states: Optional[Sequence[Image.Image]] = None,
    video_verifier_states: Optional[Sequence[Image.Image]] = None,
    video_writer: Optional[Callable[[Path], int]] = None,
    video_transition_frames: int = 6,
    video_hold_frames: int = 8,
    qa_type: str = "State Prediction",
    qa_level: str = "Medium",
    single_input: bool = False,
    direct_answer_text: Optional[str] = None,
    direct_answer_value: Optional[str] = None,
) -> BoardChoiceRecord:
    """Write all assets to disk and return a fully-populated record."""

    puzzles_dir = generator.output_dir / "puzzles"
    solutions_dir = generator.output_dir / "solutions"
    options_dir = generator.output_dir / "reasoning_options"
    videos_dir = generator.output_dir / "videos"
    states_dir = generator.output_dir / "states"
    for d in (puzzles_dir, solutions_dir, options_dir, videos_dir, states_dir):
        d.mkdir(parents=True, exist_ok=True)

    puzzle_path = puzzles_dir / f"{sample_id}.png"
    solution_path = solutions_dir / f"{sample_id}.png"
    video_path = videos_dir / f"{sample_id}.webm"
    puzzle_image.save(puzzle_path)
    solution_image.save(solution_path)

    reasoning_images = [generator.relativize_path(puzzle_path)]
    if single_input:
        if option_tiles:
            raise ValueError("Single-input classical MCQ tasks must render candidates in the puzzle image.")
    else:
        if len(option_tiles) != 4:
            raise ValueError("Classical MCQ tasks require exactly 4 option tiles.")
        option_paths = save_labeled_option_images(
            list(option_tiles),
            options_dir,
            stem=sample_id,
            labels=("A", "B", "C", "D"),
        )
        reasoning_images.extend(generator.relativize_path(p) for p in option_paths)

    frame_count = 0
    if getattr(generator, "video", False) and video_writer is not None:
        frame_count = int(video_writer(video_path))
    elif getattr(generator, "video", False) and video_states:
        frame_count = build_state_dissolve_video(
            video_path=video_path,
            states=list(video_states),
            transition_frames=video_transition_frames,
            hold_frames=video_hold_frames,
            fps=getattr(generator, "video_fps", DEFAULT_FPS),
        )

    verifier_states = video_verifier_states if video_verifier_states is not None else video_states
    if getattr(generator, "video", False) and verifier_states:
        verifier_paths: List[str] = []
        for index, verifier_image in enumerate(verifier_states, start=1):
            verifier_path = states_dir / f"{sample_id}__video_keyframe_{index:02d}.png"
            verifier_image.save(verifier_path)
            verifier_paths.append(generator.relativize_path(verifier_path))
        structured_payload["video_keyframe_paths"] = verifier_paths

    if direct_answer_text is not None:
        answer_text = str(direct_answer_text)
        answer_label = str(direct_answer_value or direct_answer_text)
        correct_option_value = ""
    else:
        if correct_option_index is None:
            raise ValueError("Classical MCQ tasks require a correct option index.")
        answer_label = option_label(correct_option_index)
        answer_text = option_answer_text(correct_option_index)
        correct_option_value = answer_label
    solution_image_rel = generator.relativize_path(solution_path)

    return BoardChoiceRecord(
        id=sample_id,
        task_name=task_name,
        task_family=task_family,
        question_description=question_description,
        image=generator.relativize_path(puzzle_path),
        reasoning_image=generator.relativize_path(puzzle_path),
        reasoning_images=reasoning_images,
        solution_image_path=solution_image_rel,
        solution_video_path=generator.relativize_path(video_path) if frame_count > 0 else None,
        video_fps=getattr(generator, "video_fps", DEFAULT_FPS) if frame_count > 0 else None,
        video_num_frames=frame_count if frame_count > 0 else None,
        ti2v_prompt=prompts["ti2v_prompt"],

        answer=answer_text,
        correct_option=correct_option_value,
        question=question_text,
        structured_answer=structured_payload,
        answer_payload=structured_payload,
        qa_type=qa_type,
        qa_level=qa_level,
        plot_level=qa_level,
    )


__all__ = [
    "BoardChoiceRecord",
    "write_board_choice_record",
]
