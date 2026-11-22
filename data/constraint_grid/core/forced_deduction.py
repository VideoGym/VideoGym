"""Shared forced-deduction chain contracts for the Constraint Grid family.

Every task emits puzzles solvable by a chain of forced moves: at each round
the visible rules plus the committed state uniquely determine at least one new
commitment until the complete terminal state is reached with zero guessing.
Task generators own their deduction engines; this module owns the chain data
model, round-to-group budgeting, and cumulative-state video composition. The
proven chain is generator-side offline QA and never enters ``semantic_input``
or candidate verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from data.common.video_encoding import encode_rgb_frames_to_webm

MAX_COMMIT_GROUPS = 7


@dataclass(frozen=True)
class ForcedMove:
    """One forced commitment together with its sample-specific justification.

    ``cells`` lists the board positions the move decides (internal 0-based),
    ``value`` is the task-specific committed value, and ``reason`` is one or
    two authored sentences naming the visible facts that force the move.
    """

    kind: str
    cells: Tuple[Tuple[int, int], ...]
    value: object
    reason: str


@dataclass
class ForcedChain:
    """A complete forced solution: rounds of simultaneously forced moves."""

    rounds: List[List[ForcedMove]] = field(default_factory=list)

    @property
    def move_count(self) -> int:
        return sum(len(round_moves) for round_moves in self.rounds)

    @property
    def cell_count(self) -> int:
        return sum(len(move.cells) for round_moves in self.rounds for move in round_moves)

    def is_empty(self) -> bool:
        return not self.rounds


class ForcedDeductionError(RuntimeError):
    """Raised when a generator cannot construct a forced-solvable instance."""


def merge_rounds(
    rounds: Sequence[Sequence[ForcedMove]],
    *,
    max_groups: int = MAX_COMMIT_GROUPS,
) -> List[List[ForcedMove]]:
    """Merge adjacent rounds into at most ``max_groups`` ordered commit groups. Rounds are the canonical causal strata of the chain: every move in round ``k`` is forced only once the earlier rounds are committed."""

    normalized = [list(round_moves) for round_moves in rounds if round_moves]
    if not normalized:
        raise ForcedDeductionError("a forced chain must contain at least one round")
    if len(normalized) <= max_groups:
        return normalized
    boundaries = np.array_split(np.arange(len(normalized)), max_groups)
    groups: List[List[ForcedMove]] = []
    for chunk in boundaries:
        merged: List[ForcedMove] = []
        for index in chunk:
            merged.extend(normalized[int(index)])
        groups.append(merged)
    return [group for group in groups if group]


def group_paragraph(moves: Sequence[ForcedMove], *, opener: str = "") -> str:
    """Join the moves' authored reasons into one reasoning paragraph."""

    sentences: List[str] = []
    if opener:
        sentences.append(opener.strip())
    for move in moves:
        reason = str(move.reason or "").strip()
        if reason:
            sentences.append(reason)
    return " ".join(sentence if sentence.endswith((".", "!", "?")) else f"{sentence}." for sentence in sentences)


def build_sequential_commit_frames(
    states: Sequence[Image.Image],
    *,
    initial_hold: int = 5,
    fade_steps: int = 4,
    commit_hold: int = 2,
    minimum_final_hold: int = 4,
) -> List[np.ndarray]:
    """Render the forced chain as ordered element-local fade commits.

    ``states`` is ``[input, cumulative_1, ..., cumulative_G]`` where the last
    entry is the terminal state.  Consecutive states differ only in the pixels
    of the newly committed marks, so linear interpolation between them is an
    element-local fade: committed content and all static scenery stay
    byte-stable while the new group's marks rise from zero opacity.
    """

    if len(states) < 2:
        raise ForcedDeductionError("a commit video needs the input state and at least one commit state")
    arrays = [np.asarray(state.convert("RGB"), dtype=np.float32) for state in states]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ForcedDeductionError("all commit states must share one canvas geometry")

    frames: List[np.ndarray] = []
    frames.extend(arrays[0].astype(np.uint8).copy() for _ in range(max(1, int(initial_hold))))
    for previous, current in zip(arrays, arrays[1:]):
        for step in range(1, max(1, int(fade_steps)) + 1):
            alpha = step / float(max(1, int(fade_steps)))
            blended = np.rint(previous + alpha * (current - previous)).clip(0, 255).astype(np.uint8)
            frames.append(blended)
        frames.extend(current.astype(np.uint8).copy() for _ in range(max(0, int(commit_hold))))
    terminal = arrays[-1].astype(np.uint8)
    frames.extend(terminal.copy() for _ in range(max(1, int(minimum_final_hold))))
    while (len(frames) - 1) % 4 != 0:
        frames.append(terminal.copy())
    if len(frames) > 157:
        raise ForcedDeductionError(
            f"sequential commit video exceeds the 10-second media contract ({len(frames)} frames)"
        )
    return frames


def encode_commit_video(
    states: Sequence[Image.Image],
    video_path: Path,
    *,
    fps: int = 16,
) -> int:
    """Encode the ordered commit program; returns the frame count (0 on failure)."""

    frames = build_sequential_commit_frames(states)
    if not encode_rgb_frames_to_webm(frames, Path(video_path), fps=fps):
        return 0
    return len(frames)


def prove_complete_chain(
    chain: ForcedChain,
    *,
    expected_cells: int,
    task_name: str,
) -> None:
    """Fail generation loudly unless the chain decides every open cell."""

    if chain.is_empty():
        raise ForcedDeductionError(f"{task_name}: forced chain is empty")
    decided = chain.cell_count
    if decided != int(expected_cells):
        raise ForcedDeductionError(
            f"{task_name}: forced chain decides {decided} cells but the puzzle leaves "
            f"{int(expected_cells)} open"
        )


def reveal_budget_guard(
    *,
    revealed: int,
    open_before_reveals: int,
    max_fraction: float,
    task_name: str,
) -> None:
    """Reject degenerate near-given boards produced by excessive reveals."""

    if open_before_reveals <= 0:
        raise ForcedDeductionError(f"{task_name}: instance has no open deduction work")
    if revealed > max_fraction * open_before_reveals:
        raise ForcedDeductionError(
            f"{task_name}: {revealed} reveals against {open_before_reveals} open decisions "
            "would leave a near-complete board"
        )


__all__ = [
    "MAX_COMMIT_GROUPS",
    "ForcedChain",
    "ForcedDeductionError",
    "ForcedMove",
    "build_sequential_commit_frames",
    "encode_commit_video",
    "group_paragraph",
    "merge_rounds",
    "prove_complete_chain",
    "reveal_budget_guard",
]
