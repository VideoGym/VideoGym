from __future__ import annotations

import random
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.typed_answers import format_answer_line
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.contracts.multimodal import VisualReasoningContract, audit_contract_prompts


CANVAS = (512, 512)


class PhysicalScenePattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    """Shared media-safe shell for the rebuilt physical-system tasks.

    Subclasses own the semantic state, renderer, and typed contract.  The shell
    only standardizes sample-unique paths, the single-input schema, cumulative
    keyframe states, and a compact 41-frame semantic video.
    """

    task_name = "physical_scene"
    task_family = "physics"
    task_mechanic = "physical_state_replay"
    video_fps = 16
    # Every concrete physical-scene renderer must explicitly declare whether
    # ``mark_answer=True`` adds a post-process answer overlay.  Leaving this as
    # ``None`` fails closed during solution/video rendering, so a newly added
    # task cannot silently reintroduce a terminal hard commit.  The explicit
    # ``semantic_terminal`` mode is reserved for a task-owned physical commit,
    # not an answer emphasis.
    answer_overlay_mode: str | None = None
    answer_fade_opacities = (0.25, 0.50, 0.75, 1.00)

    def _motion_frame_count(self, task: Mapping[str, Any]) -> int:
        """Motion-frame budget for this sample's physical process.

        The count must satisfy ``count % 4 == 3`` so the assembled video keeps
        the WAN-compatible ``4N+1`` total.  Tasks with longer multi-event
        processes override this from their own sampled event count.
        """

        del task
        return 27 if self.answer_overlay_mode == "local_opacity" else 31

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
    def _render_progress(
        self,
        task: Mapping[str, Any],
        progress: float,
        *,
        mark_answer: bool,
    ) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def _key_states(self, task: Mapping[str, Any]) -> Sequence[tuple[float, str]]:
        raise NotImplementedError

    def _structured_answer(self, task: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(task)

    def _render_with_answer_opacity(
        self,
        task: Mapping[str, Any],
        progress: float,
        *,
        answer_opacity: float,
    ) -> Image.Image:
        """Render a fixed-footprint local fade for a terminal answer overlay.

        Subclasses continue to own the binary ``mark_answer`` endpoint.  This
        adapter derives its exact source-owned footprint from the unmarked and
        marked renders, then changes only those pixels together.  True motion,
        rays, collisions, and propagation remain in ``progress`` and are never
        endpoint-blended here.
        """

        mode = self.answer_overlay_mode
        if mode not in {"local_opacity", "semantic_terminal", "none"}:
            raise ValueError(
                f"{self.task_name} must declare answer_overlay_mode as "
                "'local_opacity', 'semantic_terminal', or 'none'"
            )
        opacity = float(answer_opacity)
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("answer_opacity must lie in [0, 1]")
        if opacity > 0.0 and float(progress) < 1.0 - 1e-9:
            raise ValueError("answer overlays may appear only after physical progress completes")

        base = np.asarray(
            self._render_progress(task, float(progress), mark_answer=False).convert("RGB"),
            dtype=np.uint8,
        )
        marked = np.asarray(
            self._render_progress(task, float(progress), mark_answer=True).convert("RGB"),
            dtype=np.uint8,
        )
        if base.shape != marked.shape:
            raise ValueError(f"{self.task_name} answer-overlay endpoints have different shapes")
        changed = np.any(base != marked, axis=2)
        if mode == "none":
            if np.any(changed):
                raise ValueError(
                    f"{self.task_name} declares no answer overlay but mark_answer changes pixels"
                )
            return Image.fromarray(base, mode="RGB")
        if opacity == 0.0:
            # During the physical phase some renderers intentionally have no
            # mark-only delta yet.  The nonempty terminal footprint is still
            # validated when the solution or first positive-opacity frame is
            # rendered.
            return Image.fromarray(base, mode="RGB")
        if mode == "semantic_terminal":
            if not np.any(changed):
                raise ValueError(
                    f"{self.task_name} declares a semantic terminal commit but its footprint is empty"
                )
            if opacity != 1.0:
                raise ValueError("semantic terminal commits do not accept answer-opacity interpolation")
            return Image.fromarray(marked, mode="RGB")
        if not np.any(changed):
            raise ValueError(
                f"{self.task_name} declares a local-opacity answer overlay but its footprint is empty"
            )

        frame = base.copy()
        if opacity >= 1.0:
            frame[changed] = marked[changed]
        elif opacity > 0.0:
            frame[changed] = np.rint(
                base[changed].astype(np.float32) * (1.0 - opacity)
                + marked[changed].astype(np.float32) * opacity
            ).astype(np.uint8)
        return Image.fromarray(frame, mode="RGB")

    def _render_video(self, task: Mapping[str, Any], output_path: Path) -> tuple[str | None, int | None]:
        if not self.video:
            return None, None
        motion_frames = int(self._motion_frame_count(task))
        if motion_frames < 3 or motion_frames % 4 != 3:
            raise AssertionError(
                f"{self.task_name} motion budget must be 3 mod 4 to keep 4N+1 totals"
            )
        initial = np.asarray(self._render_progress(task, 0.0, mark_answer=False), dtype=np.uint8)
        final = np.asarray(
            self._render_with_answer_opacity(task, 1.0, answer_opacity=1.0),
            dtype=np.uint8,
        )
        frames: list[np.ndarray] = [initial.copy() for _ in range(5)]
        if self.answer_overlay_mode == "local_opacity":
            # Complete the physical process first, then reveal the already
            # fixed terminal answer footprint through several opacity levels.
            for index in range(1, motion_frames + 1):
                frames.append(
                    np.asarray(
                        self._render_progress(task, index / motion_frames, mark_answer=False),
                        dtype=np.uint8,
                    )
                )
            for opacity in self.answer_fade_opacities:
                frames.append(
                    np.asarray(
                        self._render_with_answer_opacity(
                            task,
                            1.0,
                            answer_opacity=opacity,
                        ),
                        dtype=np.uint8,
                    )
                )
        else:
            for index in range(1, motion_frames + 5):
                frames.append(
                    np.asarray(
                        self._render_progress(task, index / (motion_frames + 4), mark_answer=False),
                        dtype=np.uint8,
                    )
                )
        frames.extend(final.copy() for _ in range(5))
        if len(frames) % 4 != 1 or len(frames) > 157:
            raise AssertionError(
                f"{self.task_name} video must keep the 4N+1 rule within 157 frames"
            )
        if not encode_rgb_frames_to_webm(frames, output_path, fps=self.video_fps):
            return None, None
        return output_path.as_posix(), len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        task = self._sample_task()
        contract = self._contract(task)
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        issues = audit_contract_prompts(contract, prompts)
        if issues:
            raise ValueError(f"{self.task_name} prompt contract failed: {'; '.join(issues)}")

        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        puzzle = self._render_progress(task, 0.0, mark_answer=False).convert("RGB")
        solution = self._render_with_answer_opacity(
            task,
            1.0,
            answer_opacity=1.0,
        ).convert("RGB")
        puzzle.save(puzzle_path)
        solution.save(solution_path)

        answer_payload = self._answer_payload(task)
        answer_text = format_answer_line(answer_payload)
        key_states = list(self._key_states(task))
        if not key_states or abs(float(key_states[-1][0]) - 1.0) > 1e-9:
            raise ValueError(f"{self.task_name} key states must end at progress 1.0")

        solution_video_path, video_num_frames = self._render_video(task, video_path)
        structured = self._structured_answer(task)
        structured.setdefault("task_mechanic", self.task_mechanic)
        structured.setdefault("state_trace", [round(float(progress), 4) for progress, _ in key_states])
        structured.setdefault("media_contract", {
            "canvas": [512, 512],
            "video_fps": 16,
            "video_num_frames": video_num_frames,
            "single_input": True,
        })

        input_path = puzzle_path.as_posix()
        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_mechanic": self.task_mechanic,
            "task_contract": contract.to_dict(),
            "question": self._question(task),
            "answer": answer_text,
            "structured_answer": structured,
            **prompts,
            "image": input_path,
            "reasoning_image": input_path,
            # The primary puzzle image is already inserted by canonical scanning.
            # Keeping this list empty prevents a second copied input from appearing
            # beside the same scene in task prompts.
            "reasoning_images": [],
            "option_images": [],
            "option_image_paths": [],
            "option_labels": [],
            "solution_image_path": solution_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": self.video_fps if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        puzzle = Image.open(record["image"]).convert("RGB").copy()
        solution = Image.open(record["solution_image_path"]).convert("RGB").copy()
        return record, puzzle, solution


__all__ = ["CANVAS", "PhysicalScenePattern"]
