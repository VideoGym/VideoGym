"""Unique shortest L-route search on the shared neutral 5x5 board."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from data.board_game.core.answers import encode_visible_path
from data.board_game.core.coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.board_game.core.chess_engine import (
    BOARD_SIZE,
    Cell,
    ChessBoardRenderer,
    all_shortest_l_routes,
    build_l_waypoint_scenario,
    from_external_cell,
    l_search_board,
    to_external_cell,
    visited_waypoints_for_prefix,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.pipeline.board_motion import DEFAULT_FPS
from data.common.pipeline.board_records import BoardChoiceRecord, write_board_choice_record
from data.common.prompts import build_sectioned_prompt
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph
from data.common.typed_answers import format_answer_line
from data.common.video_encoding import (
    MAX_VIDEO_FRAMES,
    VIDEO_FPS,
    encode_rgb_frames_to_webm,
)

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="continuous",
    moving_nouns="the gliding L token and the waypoint cell fills",
    static_nouns="the board grid, the numeric labels, the green and red outlines, background, and camera",
    hold_noun="the final board with L on the red target and both waypoint fills light green",
)


TASK_NAME = "chess_l_waypoint_search"
TASK_FAMILY = "board_game"

_INPUT = (
    "The single input image shows one 5 by 5 board of square cells with row numbers 1-5 down the left and column "
    f"numbers 1-5 along the top. {GRID_XY_CONVENTION_SENTENCE} "
    "Exactly one circular white token labeled L sits on a cell. Two cells have green waypoint outlines, one cell has "
    "a red target outline, and every other cell is plain."
)

_RULES = (
    "In one move, L changes one coordinate by exactly 2 and the other by exactly 1, in any signed combination. It "
    "travels straight between the two cell centres, may pass over any cells between them, and must land on the 5 by 5 "
    "board. A green waypoint counts as visited only when L lands on its cell; the two waypoints may be visited in "
    "either order, and ordinary cells may be revisited. L must not enter the red target before both waypoints are "
    "visited, and the route ends the moment it lands there afterwards. The input has exactly one route with the "
    "minimum possible number of moves."
)

_TEXT_OUTPUT = (
    "Return the complete unique shortest route, including the starting position and every landing after it, as a "
    "coordinate list. For example: `Answer: [(1, 2), (3, 4), ...].` Use one space after every comma."
)

_IMAGE_OUTPUT = (
    "Render the solved board: L sits on the red target cell and both green waypoint cells have a light-green fill. "
    "Keep the board, numeric labels, outlines, colours, background, and camera unchanged, and draw no movement trail, "
    "route line, or extra board."
)

_VIDEO_OUTPUT = (
    "Move L along the unique shortest route: glide it continuously and directly from cell centre to cell centre, one "
    "legal L move at a time, leaving no trail or route line. Pause briefly at each landing. When L lands on a green "
    "waypoint, that cell's fill turns light green at the landing and stays. After the final landing on the red "
    "target, hold the finished board."
)


@dataclass(frozen=True)
class LSearchReplay:
    """Unique source-derived route; ``route`` excludes the visible start."""

    start: Cell
    waypoints: Tuple[Cell, Cell]
    target: Cell
    route: Tuple[Cell, ...]

    @property
    def full_path(self) -> Tuple[Cell, ...]:
        return (self.start,) + self.route


def _seeded_rng(seed: Optional[int]) -> random.Random:
    material = f"{TASK_NAME}:{0 if seed is None else int(seed)}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _visible_cell_list(cell: Cell) -> List[int]:
    return list(to_external_cell(cell))


def _parse_visible_cell(raw: object, *, name: str) -> Cell:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{name} must be a visible (x, y) pair")
    return from_external_cell((raw[0], raw[1]))


def replay_semantic_state(semantic_state: Mapping[str, object]) -> LSearchReplay:
    """Solve the search task using only visible source markers."""

    if set(semantic_state) != {"start", "waypoints", "target"}:
        raise ValueError("semantic_state must contain only start, waypoints, and target")
    start = _parse_visible_cell(semantic_state["start"], name="start")
    raw_waypoints = semantic_state["waypoints"]
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) != 2:
        raise ValueError("semantic waypoints must contain exactly two visible cells")
    waypoints = (
        _parse_visible_cell(raw_waypoints[0], name="first waypoint"),
        _parse_visible_cell(raw_waypoints[1], name="second waypoint"),
    )
    target = _parse_visible_cell(semantic_state["target"], name="target")
    routes = all_shortest_l_routes(start, waypoints, target)
    if len(routes) != 1:
        raise ValueError(f"search source must have exactly one shortest route, found {len(routes)}")
    return LSearchReplay(start, waypoints, target, routes[0])


def encode_full_path(path: Sequence[Cell]) -> str:
    return encode_visible_path(path, size=BOARD_SIZE)


def _prompts() -> Dict[str, str]:
    return {
        "ti2v_prompt": build_sectioned_prompt(
            _INPUT,
            _VIDEO_OUTPUT,
            mode="ti2v",
            rule_description=_RULES,
            continuity_paragraph=_TI2V_CONTINUITY,
        ),
    }


def _render_commits(replay: LSearchReplay, renderer: ChessBoardRenderer) -> Tuple[Image.Image, ...]:
    images: List[Image.Image] = []
    for landing_count, token in enumerate(replay.full_path):
        landed_prefix = replay.route[:landing_count]
        images.append(
            renderer.render(
                l_search_board(token),
                waypoints=replay.waypoints,
                visited_waypoints=visited_waypoints_for_prefix(replay.waypoints, landed_prefix),
                target=replay.target,
            )
        )
    return tuple(images)


def _write_video(
    path: Path,
    replay: LSearchReplay,
    renderer: ChessBoardRenderer,
    commits: Sequence[Image.Image],
) -> int:
    frames: List[Image.Image] = [commits[0].copy() for _ in range(5)]
    for hop_index, destination in enumerate(replay.route):
        source = replay.start if hop_index == 0 else replay.route[hop_index - 1]
        landed_prefix = replay.route[:hop_index]
        before_board = l_search_board(source)
        visited = visited_waypoints_for_prefix(replay.waypoints, landed_prefix)
        for motion_index in range(1, 9):
            frames.append(
                renderer.render(
                    before_board,
                    waypoints=replay.waypoints,
                    visited_waypoints=visited,
                    target=replay.target,
                    moving_piece=("N", True),
                    moving_from=source,
                    moving_to=destination,
                    progress=motion_index / 9.0,
                )
            )
        frames.extend(commits[hop_index + 1].copy() for _ in range(4))
    frames.extend(commits[-1].copy() for _ in range(8))
    if not frames or (len(frames) - 1) % 4 != 0 or len(frames) > MAX_VIDEO_FRAMES:
        raise ValueError("L-route video must have 4N+1 frames within the canonical budget")
    if not encode_rgb_frames_to_webm(frames, path, fps=VIDEO_FPS, extra_args=["-g", "1", "-bf", "0"]):
        return 0
    return len(frames)


def _token_grid(cell: Cell) -> List[List[int]]:
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    grid[cell[0]][cell[1]] = 4
    return grid


class ChessLWaypointSearchGenerator(AbstractPuzzleGenerator[BoardChoiceRecord]):
    """Generate uniquely solved two-waypoint L search programs."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        video_fps: int = DEFAULT_FPS,
        video: bool = False,
        difficulty: str = "typical",
    ) -> None:
        super().__init__(output_dir)
        if int(video_fps) != VIDEO_FPS:
            raise ValueError(f"{TASK_NAME} uses the canonical {VIDEO_FPS} fps")
        if difficulty not in {"minimum", "typical", "maximum"}:
            raise ValueError("difficulty must be minimum, typical, or maximum")
        self.rng = _seeded_rng(seed)
        self.video_fps = VIDEO_FPS
        self.video = bool(video)
        self.difficulty = difficulty
        self._counter = 0
        self.renderer = ChessBoardRenderer()

    def record_to_dict(self, record: BoardChoiceRecord | Dict[str, Any]) -> Dict[str, Any]:
        return record.to_dict() if hasattr(record, "to_dict") else dict(record)

    def create_puzzle(self) -> BoardChoiceRecord:
        sample_id = f'{TASK_NAME}-{self._counter:05d}'
        self._counter += 1
        scenario = build_l_waypoint_scenario(self.rng, difficulty=self.difficulty)
        semantic_state: Dict[str, object] = {'start': _visible_cell_list(scenario.start), 'waypoints': sorted((_visible_cell_list(cell) for cell in scenario.waypoints)), 'target': _visible_cell_list(scenario.target)}
        replay = replay_semantic_state(semantic_state)
        if replay.route != scenario.unique_route:
            raise AssertionError('source-only route replay diverged from the accepted scenario')
        commits = _render_commits(replay, self.renderer)
        puzzle = commits[0]
        solution = commits[-1]
        answer_value = encode_full_path(replay.full_path)
        answer_text = format_answer_line(answer_value)
        payload: Dict[str, Any] = {'semantic_state': semantic_state, 'answer_value': answer_value, 'optimal_route': [_visible_cell_list(cell) for cell in replay.full_path], 'optimal_length': len(replay.route), 'shortest_route_count': 1, 'initial_grid': _token_grid(replay.start), 'state_grid': _token_grid(replay.target), 'state_trace': [_token_grid(cell) for cell in replay.full_path], 'process_state_trace': [_token_grid(cell) for cell in replay.full_path]}
        qa_level = {'minimum': 'Easy', 'typical': 'Medium', 'maximum': 'Hard'}[self.difficulty]
        return write_board_choice_record(generator=self, sample_id=sample_id, task_name=TASK_NAME, task_family=TASK_FAMILY, question_description='unique_shortest_two_waypoint_l_route', puzzle_image=puzzle, solution_image=solution, option_tiles=(), correct_option_index=None, prompts=_prompts(), structured_payload=payload, question_text='Return the complete unique shortest L route through both waypoints to the target.', video_states=None, video_verifier_states=commits, video_writer=lambda path: _write_video(path, replay, self.renderer, commits), qa_type='Path Planning', qa_level=qa_level, single_input=True, direct_answer_text=answer_text, direct_answer_value=answer_value)
