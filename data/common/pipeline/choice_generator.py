"""Shared generator for rendered four-choice tasks.

Subclasses implement ``_generate_task``, ``_draw_source``, and
``_make_candidate_drawers``; this base class handles the canonical sample
contract (TI2V prompt, MCQ candidate plates, reveal video,
and the canonical sample dict). All subclasses MUST keep render output
at 512x512.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from data.common.base import RenderedTaskGenerator
from data.common.multiple_choice import save_labeled_option_images
from data.common.pipeline.media import LETTER_OPTIONS
from data.common.pipeline.choice_layout import (
    CANVAS_SIZE,
    build_standard_prompts,
    compose_tile_panel,
    fit_pad,
    assemble_choice_order,
)


class ChoiceTaskGenerator(RenderedTaskGenerator):
    """Common scaffolding for four-choice rendered tasks."""

    task_name: str = "abstract_pattern"
    task_family: str = "in_context_learning"
    video_fps: int = 16
    generate_videos: bool = True

    # Three-section prompt blocks; subclasses override these.
    prompt_intro: str = "Inspect the source figure and the four candidates."
    prompt_rules: str = "Rules go here."
    prompt_output: str = (
        "Choose the candidate (A, B, C, D) that satisfies the rule."
    )

    def __init__(self, output_dir: str | Path | None = None, **kwargs: Any) -> None:
        seed = kwargs.get("random_seed", kwargs.get("seed"))
        self.rng = random.Random(seed)
        self.image_size: Tuple[int, int] = CANVAS_SIZE
        self.video_fps = int(kwargs.get("video_fps", self.video_fps))
        self.generate_videos = bool(kwargs.get("generate_videos", self.generate_videos))
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_index = 0

    # ----- subclass interface ----------------------------------------------

    def _generate_task(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _draw_source(
        self,
        img: Image.Image,
        box: Tuple[int, int, int, int],
        task: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def _make_candidate_drawers(
        self,
        task: Dict[str, Any],
    ) -> Tuple[
        Callable[[Image.Image, Tuple[int, int, int, int]], None],
        List[Callable[[Image.Image, Tuple[int, int, int, int]], None]],
        Dict[str, Any],
    ]:
        """Return (correct_drawer, [3 distractor drawers], extra_meta)."""
        raise NotImplementedError

    def _source_caption(self, task: Dict[str, Any]) -> str:
        return ""

    # ----- shared rendering/sample logic -----------------------------------

    def _compose_puzzle(
        self,
        task: Dict[str, Any],
        drawers: Sequence[Callable],
        *,
        highlight_letter: str | None = None,
    ) -> Image.Image:
        def src_drawer(img: Image.Image, box: Tuple[int, int, int, int]) -> None:
            self._draw_source(img, box, task)

        return compose_tile_panel(
            draw_source=src_drawer,
            candidate_drawers=drawers,
            source_caption=self._source_caption(task),
            highlight_letter=highlight_letter,
        )

    def _build_prompts(self) -> Dict[str, str]:
        return build_standard_prompts(
            self.task_name,
            intro=self.prompt_intro,
            rules=self.prompt_rules,
            output_format=self.prompt_output,
        )

    def _option_tile(
        self,
        task: Dict[str, Any],
        drawer: Callable,
    ) -> Image.Image:
        """Render a single candidate as its own labelled MCQ option tile."""
        from data.common.pipeline.choice_layout import (
            BACKGROUND_COLOR,
            PANEL_FILL_COLOR,
            PANEL_BORDER_COLOR,
        )

        img = Image.new("RGB", CANVAS_SIZE, BACKGROUND_COLOR)
        d = ImageDraw.Draw(img)
        margin = 24
        outer_box = (margin, margin, CANVAS_SIZE[0] - margin, CANVAS_SIZE[1] - margin)
        d.rounded_rectangle(
            outer_box, radius=12, fill=PANEL_FILL_COLOR, outline=PANEL_BORDER_COLOR, width=3
        )
        inner_box = (outer_box[0] + 18, outer_box[1] + 18, outer_box[2] - 18, outer_box[3] - 18)
        drawer(img, inner_box)
        return img

    def make_sample(self) -> Tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution

    def create_puzzle(self) -> Dict[str, Any]:
        for attempt in range(8):
            try:
                task = self._generate_task()
                correct_drawer, distractor_drawers, meta = self._make_candidate_drawers(task)
                if len(distractor_drawers) != 3:
                    raise RuntimeError(
                        f"{self.task_name}: expected 3 distractors, got {len(distractor_drawers)}"
                    )
                ordered_drawers, correct_letter, choice_indices = assemble_choice_order(
                    self.rng, correct_drawer, distractor_drawers
                )
                break
            except RuntimeError:
                if attempt == 7:
                    raise

        prompts = self._build_prompts()

        puzzle_image = fit_pad(self._compose_puzzle(task, ordered_drawers))
        solution_image = fit_pad(
            self._compose_puzzle(task, ordered_drawers, highlight_letter=correct_letter)
        )

        sample_id = f"{self.task_name}-{self._sample_index:05d}"
        self._sample_index += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / "puzzle.png"
        solution_path = sample_dir / "solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        # Per-option tile renders for the MCQ option grid.
        option_tiles: List[Image.Image] = []
        for drawer in ordered_drawers:
            option_tiles.append(self._option_tile(task, drawer))
        option_paths = save_labeled_option_images(
            option_tiles, sample_dir, stem=self.task_name
        )

        structured_meta = dict(meta)
        if "correct_value" in structured_meta and "distractor_values" in structured_meta:
            source_values = [structured_meta["correct_value"], *list(structured_meta["distractor_values"])]
            if len(source_values) == len(choice_indices):
                structured_meta["candidate_values_by_label"] = {
                    LETTER_OPTIONS[index]: source_values[source_index]
                    for index, source_index in enumerate(choice_indices)
                }

        solution_video_path: str | None = None
        video_num_frames: int | None = None

        sample: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "question": prompts["question"],
            "answer": correct_letter,
            "correct_option": correct_letter,
            "structured_answer": {
                "correct_option": correct_letter,
                "task_meta": structured_meta,
            },
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_path.as_posix(),
            "reasoning_image": puzzle_path.as_posix(),
            "reasoning_images": [puzzle_path.as_posix(), *(p.as_posix() for p in option_paths)],
            "option_image_paths": [p.as_posix() for p in option_paths],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        return sample


__all__ = [
    "ChoiceTaskGenerator",
]
