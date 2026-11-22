"""Media-safe generator shell for the Spatial 3D family."""

from __future__ import annotations

import random
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.choice_cards import attach_visual_choice_card_inputs
from data.common.contracts.multimodal import VisualReasoningContract, audit_contract_prompts
from data.common.video_encoding import encode_rgb_frames_to_webm


CANVAS = (512, 512)
STAGE_FADE_OPACITIES = (0.2, 0.4, 0.6, 0.8)


class Spatial3DScenePattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = "spatial_3d_scene"
    task_family = "spatial_3d"
    task_mechanic = "spatial_3d_state_replay"
    video_fps = 16
    # True: continuous-motion video via ``_render_progress``.
    # False: staged element-local fades via ``_render_stage``.
    motion: bool = True
    # True for tasks whose Rule legitimately lists the complete closed answer
    # vocabulary, where the payload word must be allowed to
    # appear in prompt prose.
    answer_vocabulary_in_prompt: bool = False

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: int | None = None,
        random_seed: int | None = None,
        video: bool | None = None,
        generate_videos: bool = True,
        video_fps: int = 16,
        image_size: int | Sequence[int] = CANVAS,
        difficulty: str | None = None,
        **_kwargs: Any,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        super().__init__(resolved)
        resolved_seed = seed if seed is not None else random_seed
        self.rng = random.Random(resolved_seed)
        self.seed = resolved_seed
        self.video = bool(generate_videos if video is None else video)
        self.video_fps = int(video_fps)
        if self.video_fps != 16:
            raise ValueError(f"{self.task_name} requires 16fps video")
        resolved_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if tuple(resolved_size) != CANVAS:
            raise ValueError(f"{self.task_name} requires a 512x512 canvas")
        if difficulty not in {None, "min", "typical", "max"}:
            raise ValueError("difficulty must be one of: min, typical, max")
        self.difficulty = difficulty
        self._counter = 0

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def _sample_task(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _contract(self, task: Mapping[str, Any]) -> VisualReasoningContract:
        raise NotImplementedError

    @abstractmethod
    def _question(self, task: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _answer_payload(self, task: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def _glyph_inventory(self, task: Mapping[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, str]]]:
        """Per-sample rendered-glyph inventory and schema."""

        del task
        return None

    # -- motion-model hooks --------------------------------------------------

    def _render_progress(self, task: Mapping[str, Any], progress: float) -> Image.Image:
        raise NotImplementedError(f"{self.task_name} is a staged task; use _render_stage")

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        """Motion-frame budget; must satisfy ``count % 4 == 3``."""

        del task
        return 31

    def _stage_count(self, task: Mapping[str, Any]) -> int:
        raise NotImplementedError(f"{self.task_name} is a motion task; use _render_progress")

    def _render_stage(self, task: Mapping[str, Any], index: int) -> Image.Image:
        raise NotImplementedError(f"{self.task_name} is a motion task; use _render_progress")

    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[Any, str]]:
        """0; staged tasks return ``(stage_index, text)`` pairs ending at the last stage. Texts are the sample-specific authored reasoning."""

        raise NotImplementedError

    def _option_plates(
        self, task: Mapping[str, Any], sample_dir: Path, sample_id: str
    ) -> Optional[Dict[str, Any]]:
        """Optional visual answer-hypothesis plates: {"paths": [...], "labels": [...]}."""

        del task, sample_dir, sample_id
        return None

    # -------------------------------------------------------------- rendering

    def _terminal_image(self, task: Mapping[str, Any]) -> Image.Image:
        if self.motion:
            return self._render_progress(task, 1.0).convert("RGB")
        return self._render_stage(task, self._stage_count(task) - 1).convert("RGB")

    def _input_image(self, task: Mapping[str, Any]) -> Image.Image:
        if self.motion:
            return self._render_progress(task, 0.0).convert("RGB")
        return self._render_stage(task, 0).convert("RGB")

    @staticmethod
    def _blend(first: np.ndarray, second: np.ndarray, opacity: float) -> np.ndarray:
        """Element-local fade frame: unchanged pixels stay bit-identical."""

        changed = np.any(first != second, axis=2)
        frame = first.copy()
        mixed = np.rint(
            first[changed].astype(np.float32) * (1.0 - opacity)
            + second[changed].astype(np.float32) * opacity
        ).astype(np.uint8)
        frame[changed] = mixed
        return frame

    def _render_video(self, task: Mapping[str, Any], output_path: Path) -> tuple[str | None, int | None]:
        if not self.video:
            return None, None
        frames: List[np.ndarray]
        if self.motion:
            motion_frames = int(self._motion_frame_count(task))
            if motion_frames < 3 or motion_frames % 4 != 3:
                raise AssertionError(f"{self.task_name} motion budget must be 3 mod 4")
            initial = np.asarray(self._render_progress(task, 0.0).convert("RGB"), dtype=np.uint8)
            terminal = np.asarray(self._render_progress(task, 1.0).convert("RGB"), dtype=np.uint8)
            frames = [initial.copy() for _ in range(5)]
            for index in range(1, motion_frames + 5):
                frames.append(
                    np.asarray(
                        self._render_progress(task, index / (motion_frames + 4)).convert("RGB"),
                        dtype=np.uint8,
                    )
                )
            frames.extend(terminal.copy() for _ in range(5))
        else:
            count = int(self._stage_count(task))
            if count < 2:
                raise AssertionError(f"{self.task_name} staged videos need at least two states")
            stages = [
                np.asarray(self._render_stage(task, index).convert("RGB"), dtype=np.uint8)
                for index in range(count)
            ]
            for previous, current in zip(stages, stages[1:]):
                if not bool(np.any(previous != current)):
                    raise AssertionError(f"{self.task_name} staged states must differ")
            frames = [stages[0].copy() for _ in range(5)]
            for previous, current in zip(stages, stages[1:]):
                for opacity in STAGE_FADE_OPACITIES:
                    frames.append(self._blend(previous, current, opacity))
                frames.extend(current.copy() for _ in range(4))
            frames.extend(stages[-1].copy() for _ in range(4))
        if len(frames) % 4 != 1 or len(frames) > 157:
            raise AssertionError(f"{self.task_name} video must keep 4N+1 within 157 frames")
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    # ------------------------------------------------------------- authoring

    # ----------------------------------------------------------------- record

    def create_puzzle(self) -> Dict[str, Any]:
        task = self._sample_task()
        contract = self._contract(task)
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        answer_payload = self._answer_payload(task)
        # Single-token payloads from a closed vocabulary (a digit, a letter,
        # or a declared vocabulary word listed in the Rule per 8.9)
        # legitimately occur inside prompt prose; the substring leak check is
        # meaningful only for structured payloads.
        leak_probe = (
            None
            if self.answer_vocabulary_in_prompt or len(answer_payload.strip()) < 3
            else answer_payload
        )
        issues = audit_contract_prompts(contract, prompts, answer_payload=leak_probe)
        if issues:
            raise ValueError(f"{self.task_name} prompt contract failed: {'; '.join(issues)}")

        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        self._input_image(task).save(puzzle_path)
        self._terminal_image(task).save(solution_path)

        solution_video_path, video_num_frames = self._render_video(task, video_path)

        structured = self._structured_answer(task)
        structured.setdefault("task_mechanic", self.task_mechanic)
        structured.setdefault("media_contract", {
            "canvas": [512, 512],
            "video_fps": 16,
            "video_num_frames": video_num_frames,
        })

        input_path = puzzle_path.as_posix()
        record: Dict[str, Any] = {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_mechanic": self.task_mechanic,
            "task_contract": contract.to_dict(),
            "question": self._question(task),
            "answer": answer_payload,
            "structured_answer": structured,
            **prompts,
            "image": input_path,
            "reasoning_image": input_path,
            "reasoning_images": [],
            "option_images": [],
            "option_image_paths": [],
            "option_labels": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }
        plates = self._option_plates(task, sample_dir, sample_id)
        if plates:
            attach_visual_choice_card_inputs(
                record,
                problem_image=input_path,
                option_images=[str(path) for path in plates["paths"]],
                option_labels=list(plates["labels"]),
            )
        glyphs = self._glyph_inventory(task)
        if glyphs is not None:
            record["rendered_glyph_inventory"], record["rendered_glyph_schema"] = glyphs
        return record

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution


__all__ = ["CANVAS", "STAGE_FADE_OPACITIES", "Spatial3DScenePattern"]
