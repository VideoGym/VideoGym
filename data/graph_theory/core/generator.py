"""Shared generator shell for 16 labelled-graph task predicates."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from data.common.base import AbstractPuzzleGenerator
from data.common.contracts.multimodal import AnswerContract, VisualReasoningContract, audit_contract_prompts
from data.common.scalar_lists import NONNEGATIVE_INTEGER_ANSWER_SCHEMA
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.graph_theory.core.rendering import CANVAS
from data.graph_theory.core.replay import replay_unified_graph
from data.graph_theory.core.semantics import format_graph_answer, generate_graph_source
from data.graph_theory.core.task_specs import NEW_GRAPH_TASK_NAMES, get_unified_graph_spec



# Tasks whose input carries colored role rings; the preservation sentence
# names them only where they exist.
_RINGLESS_TASKS = frozenset({
    "euler_trail_endpoints",
    "bridge_edge_choice",
    "connected_component_count",
    "diameter_endpoints",
})


def _answer_contract(task_name: str) -> AnswerContract:
    answer_type = get_unified_graph_spec(task_name).answer_type
    templates = {
        "path": 'Answer: ["<start>", "<next>", ..., "<goal>"].',
        "edge": 'Answer: ["<smaller-number>", "<larger-number>"].',
        "integer": NONNEGATIVE_INTEGER_ANSWER_SCHEMA,
        "label": "Answer: <number-label>.",
        "token": "Answer: <uppercase comparison token>.",
        "color": "Answer: <uppercase color>.",
    }
    kinds = {
        "path": "string_list",
        "edge": "string_list",
        "integer": "nonnegative_integer",
        "label": "typed_text",
        "token": "typed_text",
        "color": "typed_text",
    }
    examples = {
        "path": 'Answer: ["98", "99", ...].',
        "edge": 'Answer: ["12", "13"].',
        "integer": "Answer: 99.",
        "label": "Answer: 99.",
        "token": "Answer: UNKNOWN.",
        "color": "Answer: GREEN.",
    }
    return AnswerContract(
        kind=kinds[answer_type],
        format_template=templates[answer_type],
        parser_profile=f"unified_graph_{answer_type}_v2",
        verifier_profile="unified_graph_source_replay_v2",
        format_example=examples[answer_type],
        accepts_multiple_valid_answers=task_name == "trace_all_edges_once",
    )


def graph_contract(task_name: str) -> VisualReasoningContract:
    spec = get_unified_graph_spec(task_name)
    answer = _answer_contract(task_name)
    if spec.kind == "tree":
        legend = (
            "Numbers inside the circles are the node labels.",
            "Each drawn segment is one parent-child edge.",
        )
    else:
        legend = (
            "Numbers inside the circles are the node labels.",
            "Each drawn segment is one undirected edge.",
        )
    if task_name == "degree_missing_value":
        preservation = (
            "Keep every node, its label, every edge, the purple ring, and the yellow badge exactly where they are; "
            "the background never changes and only the effects described above appear."
        )
        static_nouns = "every node, label, edge, ring, badge outline, and the background"
    elif task_name in _RINGLESS_TASKS:
        preservation = (
            "Keep every node, its label, and every edge exactly where they are; "
            "the background never changes and only the effects described above appear."
        )
        static_nouns = "every node, label, edge, and the background"
    else:
        preservation = (
            "Keep every node, its label, every edge, and the colored role rings exactly where they are; "
            "the background never changes and only the effects described above appear."
        )
        static_nouns = "every node, label, edge, ring, and the background"
    world = "tree" if spec.kind == "tree" else "graph"
    return VisualReasoningContract(
        input_description=spec.input_description,
        symbol_legend=legend,
        rules=spec.rules,
        modality_outputs={
            "ti2v": spec.process,
            "solution_image": spec.terminal,
        },
        preservation_constraints=(preservation,),
        transition_constraints=(spec.process,),
        answer=answer,
        render_profile="unified_labelled_graph_fixed_footprint_local_fade_v3",
        ti2v_continuity={
            "moving_nouns": spec.moving_nouns,
            "static_nouns": static_nouns,
            "hold_noun": f"the finished answer state of the {world}",
        },
    )


class UnifiedGraphPattern(AbstractPuzzleGenerator[Dict[str, Any]]):
    """One implementation shared by all task-level route modules."""

    task_name = "unified_graph"
    task_family = "graph_theory"
    task_mechanic = "unified_labelled_graph_source_replay"

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
        node_count: int | None = None,
        regime: str | None = None,
        minimum_answer: int | None = None,
        **_kwargs: Any,
    ) -> None:
        if self.task_name not in NEW_GRAPH_TASK_NAMES:
            raise ValueError(f"unknown unified graph task: {self.task_name}")
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / f"videogym_{self.task_name}"
        super().__init__(resolved)
        self.seed = seed if seed is not None else random_seed
        self.rng = random.Random(self.seed)
        self.video = bool(generate_videos if video is None else video)
        self.generate_videos = self.video
        self.video_fps = int(video_fps)
        if self.video_fps != 16:
            raise ValueError("unified Graph Theory videos require 16fps")
        resolved_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if resolved_size != (CANVAS, CANVAS):
            raise ValueError("unified Graph Theory media requires 512x512")
        if difficulty not in {None, "min", "typical", "max"}:
            raise ValueError("difficulty must be one of: min, typical, max")
        self.difficulty = difficulty
        if node_count is not None and not 6 <= int(node_count) <= 18:
            raise ValueError("node_count must lie in 6..18")
        self.regime = regime
        self.minimum_answer = minimum_answer
        self.node_count = int(node_count) if node_count is not None else None
        self._counter = 0

    def configure_runtime(
        self,
        *,
        output_dir: str | Path,
        seed: int | None,
        video: bool,
        video_fps: int,
    ) -> None:
        super().configure_runtime(
            output_dir=output_dir,
            seed=seed,
            video=video,
            video_fps=video_fps,
        )
        if int(video_fps) != 16:
            raise ValueError("unified Graph Theory videos require 16fps")
        self.rng = random.Random(seed)

    def _write_video(self, replay: Any, path: Path) -> tuple[str | None, int | None]:
        if not self.video:
            return None, None
        initial = np.asarray(replay.render_input().convert("RGB"), dtype=np.uint8)
        terminal = np.asarray(replay.render_progress(1.0).convert("RGB"), dtype=np.uint8)
        frames: list[np.ndarray] = [initial.copy() for _ in range(5)]
        frames.extend(
            np.asarray(replay.render_progress(index / 31.0).convert("RGB"), dtype=np.uint8)
            for index in range(1, 32)
        )
        frames.extend(terminal.copy() for _ in range(5))
        if len(frames) != 41 or (len(frames) - 1) % 4:
            raise AssertionError("unified graph reference video must contain 41 frames")
        if not encode_rgb_frames_to_webm(frames, path, fps=16):
            return None, None
        return path.as_posix(), len(frames)

    def create_puzzle(self) -> Dict[str, Any]:
        source = generate_graph_source(
            self.task_name,
            self.rng,
            self.difficulty,
            node_count=self.node_count,
            regime=self.regime,
            minimum_answer=self.minimum_answer,
        )
        replay = replay_unified_graph(self.task_name, source)
        solution = replay.solution
        contract = graph_contract(self.task_name)
        prompts = {
            "ti2v_prompt": contract.build_prompt("ti2v"),
        }
        prompt_issues = audit_contract_prompts(contract, prompts)
        if prompt_issues:
            raise ValueError(f"{self.task_name} prompt contract failed: {'; '.join(prompt_issues)}")

        sample_id = f"{self.task_name}-{self._counter:05d}"
        self._counter += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        input_path = sample_dir / f"{sample_id}__input.png"
        terminal_path = sample_dir / f"{sample_id}__terminal.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"
        replay.render_input().convert("RGB").save(input_path)
        replay.render_progress(1.0).convert("RGB").save(terminal_path)

        answer_text = format_graph_answer(self.task_name, solution["answer"])
        solution_video_path, video_num_frames = self._write_video(replay, video_path)

        return {
            "id": sample_id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "task_contract": contract.to_dict(),
            "question": get_unified_graph_spec(self.task_name).question,
            "answer": answer_text,
            # This object is deliberately source-only.  The solution, path,
            # terminal state, and replay checkpoints stay outside the verifier
            # trust domain and are recomputed by unified_replay.
            "structured_answer": dict(source),
            **prompts,
            "image": input_path.as_posix(),
            "solution_image_path": terminal_path.as_posix(),
            "solution_video_path": solution_video_path,
            "video_fps": 16 if solution_video_path else None,
            "video_num_frames": video_num_frames,
        }

    def make_sample(self) -> tuple[Dict[str, Any], Image.Image, Image.Image]:
        record = self.create_puzzle()
        with Image.open(record["image"]) as input_image:
            puzzle = input_image.convert("RGB").copy()
        with Image.open(record["solution_image_path"]) as answer_image:
            solution = answer_image.convert("RGB").copy()
        return record, puzzle, solution


def make_graph_task_class(task_name: str) -> type[UnifiedGraphPattern]:
    spec = get_unified_graph_spec(task_name)
    return type(spec.class_name, (UnifiedGraphPattern,), {"task_name": task_name, "__module__": __name__})


__all__ = ["UnifiedGraphPattern", "graph_contract", "make_graph_task_class"]
