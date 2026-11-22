from __future__ import annotations

import random
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image

from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.base import RenderedTaskGenerator as RenderedTaskProtocol
from data.common.pipeline.media import pad_image, save_visual_reasoning_video


def normalize_canvas_size(value: object) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return 512, 512

class RenderedTaskGenerator(RenderedTaskProtocol, ABC):
    """Base for task-owned rendering that returns a complete formal record."""

    TASK_NAME = "rendered_task"
    TASK_FAMILY: str | None = None
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)
    DEFAULT_VIDEO_FPS = 16

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        image_size: Tuple[int, int] | int | None = None,
        video_fps: int | None = None,
        generate_videos: bool = True,
    ) -> None:
        if not self.TASK_FAMILY:
            raise TypeError(f"{type(self).__name__} must define a non-empty TASK_FAMILY")
        resolved_output_dir = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / f"videogym_{self.TASK_NAME}"
        self.output_dir = Path(resolved_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.image_size = normalize_canvas_size(image_size if image_size is not None else self.DEFAULT_IMAGE_SIZE)
        self.video_fps = int(video_fps if video_fps is not None else self.DEFAULT_VIDEO_FPS)
        self.generate_videos = bool(generate_videos)
        self._sample_index = 0

    @abstractmethod
    def _generate_task(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _render_puzzle_image(self, task: Dict[str, Any]) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def _render_solution_image(self, task: Dict[str, Any]) -> Image.Image:
        raise NotImplementedError

    def _render_reasoning_option_tiles(self, task: Dict[str, Any]) -> list[Image.Image]:
        return []

    @abstractmethod
    def _question_text(self, task: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _answer_raw(self, task: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _structured_answer(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _visual_input_description(self, task: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _visual_output_description(self, task: Dict[str, Any]) -> str:
        raise NotImplementedError

    def _build_prompts(self, task: Dict[str, Any], answer_raw: str) -> Dict[str, str]:
        question = self._question_text(task)
        visual_input = self._visual_input_description(task)
        visual_output = self._visual_output_description(task)
        return {
            "question": question,
            "ti2v_prompt": build_sectioned_prompt(
                visual_input,
                visual_output,
                mode="ti2v",
                continuity_paragraph=build_ti2v_continuity_paragraph(
                    timing="continuous",
                    moving_nouns="the answer elements the rules add or change",
                    static_nouns="everything the rules leave unchanged, including the labels, background, and camera",
                    hold_noun="the final answer state",
                ),
            ),
        }

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        task = self._generate_task()
        answer_raw = self._answer_raw(task)
        prompts = self._build_prompts(task, answer_raw)
        rendered_puzzle = self._render_puzzle_image(task)
        rendered_solution = self._render_solution_image(task)
        puzzle_image = pad_image(rendered_puzzle.copy(), target_size=self.image_size)
        solution_image = pad_image(rendered_solution.copy(), target_size=self.image_size)

        sample_id = f"{self.TASK_NAME}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / "puzzle.png"
        solution_path = sample_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        reasoning_images = [puzzle_path.as_posix()]
        option_tiles = self._render_reasoning_option_tiles(task)
        if option_tiles:
            options_dir = sample_dir / "reasoning_options"
            options_dir.mkdir(parents=True, exist_ok=True)
            for option_index, option_image in enumerate(option_tiles):
                option_label = chr(ord("A") + option_index)
                option_path = options_dir / f"{option_label}.png"
                pad_image(option_image, target_size=self.image_size).save(option_path)
                reasoning_images.append(option_path.as_posix())

        solution_video_path = None
        video_num_frames = None
        if self.generate_videos:
            solution_video_path = (sample_dir / "solution.webm").as_posix()
            video_num_frames = save_visual_reasoning_video(puzzle_image, solution_image, solution_video_path, fps=self.video_fps)
            if not video_num_frames:
                solution_video_path = None
                video_num_frames = None

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.TASK_NAME,
            "task_family": self.TASK_FAMILY,
            "question": prompts["question"],
            "answer": answer_raw,
            "structured_answer": self._structured_answer(task),
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": reasoning_images,
            "option_images": reasoning_images[1:],
            "option_labels": [chr(ord("A") + index) for index in range(max(0, len(reasoning_images) - 1))],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        return sample, puzzle_image, solution_image

__all__ = [
    "RenderedTaskGenerator",
    "normalize_canvas_size",
]
