from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    format_xy_path_answer,
)
from data.common.prompts import build_sectioned_prompt
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.dynamics_game.core.coordinate_format import display_cell_text, display_path_text
from data.dynamics_game.core.scene import SS, scene_font
from data.common.render_primitives import ScaledDraw, finish_scaled_canvas
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

_TI2V_CONTINUITY = build_ti2v_continuity_paragraph(
    timing="discrete",
    moving_nouns="the yellow agent and the pushed box",
    static_nouns="the walls, the green target mark, the grid, the labels, the background, and the camera",
    hold_noun="the solved board with the box on the target",
)

_INPUT_DESC = (
    "The input image shows a 7 by 7 coordinate grid with rows and columns labeled 1 through 7 on a white square canvas. Dark slate cells are fixed walls, pale cells are floor, "
    "the yellow circle is the movable agent, the brown crate is the only pushable box, and the green X cell is the target. "
    f"{GRID_XY_CONVENTION_SENTENCE}"
)
_RULE_SUMMARY = (
    "On each step the agent moves exactly one cell up, right, down, or left. It cannot leave the board or enter a wall. "
    "If the destination contains the box, the agent pushes it exactly one cell in the same direction; the push is legal "
    "only when that next box cell is inside the board and is neither a wall nor the agent. The box cannot be pulled, slide "
    "more than one cell, fall, refill, compact, or move without a push. Vacated cells return to ordinary floor, and every "
    "wall, the target mark, grid, labels, background, and camera remain fixed. The task ends when the box occupies the target."
)
_TI2V_OUTPUT = (
    "Generate a video that starts from the exact input and replays one shortest valid route as complete board "
    "states, one per move: in each state the yellow agent has advanced one cell, and on a push the box has "
    "advanced one cell in the same direction within the same state. End with the box on the green target."
)
_DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


class SokobanPlanGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_NAME = "sokoban"
    TASK_FAMILY = "dynamics_game"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: Optional[int] = None,
        width: int = 7,
        height: int = 7,
        wall_count_range: Tuple[int, int] = (3, 6),
        push_count_range: Tuple[int, int] = (2, 5),
        route_steps: Tuple[int, int] = (8, 24),
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        super().__init__(output_dir)
        self.rng = random.Random(seed)
        self.seed = seed
        self.width = width
        self.height = height
        self.wall_count_range = wall_count_range
        self.push_count_range = tuple(push_count_range)
        self.route_steps = tuple(route_steps)
        self.video = bool(video)
        self.video_fps = video_fps
        self._counter = 0

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    def _blank_board(self) -> List[List[str]]:
        board = [["." for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            board[row][0] = "#"
            board[row][self.width - 1] = "#"
        for col in range(self.width):
            board[0][col] = "#"
            board[self.height - 1][col] = "#"
        return board

    def _random_state(
        self,
    ) -> Tuple[List[List[str]], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        inner_cells = [
            (row, col)
            for row in range(1, self.height - 1)
            for col in range(1, self.width - 1)
        ]
        for _ in range(512):
            board = self._blank_board()
            wall_budget = self.rng.randint(*self.wall_count_range)
            self.rng.shuffle(inner_cells)
            for row, col in inner_cells[:wall_budget]:
                board[row][col] = "#"
            free = [(row, col) for row, col in inner_cells if board[row][col] == "."]
            if len(free) < 3:
                continue
            player, box, target = self.rng.sample(free, 3)
            solution = self._solve(board, player, box, target)
            if (
                not solution
                or not self.route_steps[0] <= len(solution) <= self.route_steps[1]
            ):
                continue
            trace = self._simulate(board, player, box, solution)
            push_count = sum(
                1
                for previous, current in zip(trace, trace[1:])
                if previous["box"] != current["box"]
            )
            if not self.push_count_range[0] <= push_count <= self.push_count_range[1]:
                continue
            open_solution = self._solve(self._blank_board(), player, box, target)
            if open_solution and len(solution) >= len(open_solution) + 2:
                return board, player, box, target
        raise RuntimeError("Failed to generate a wall-dependent multi-push Sokoban state.")

    def _is_free(self, board: Sequence[Sequence[str]], cell: Tuple[int, int], box: Tuple[int, int]) -> bool:
        row, col = cell
        return (
            0 <= row < self.height
            and 0 <= col < self.width
            and board[row][col] != "#"
            and cell != box
        )

    def _solve(
        self,
        board: Sequence[Sequence[str]],
        player: Tuple[int, int],
        box: Tuple[int, int],
        target: Tuple[int, int],
    ) -> List[str]:
        queue = deque([(player, box, [])])
        visited = {(player, box)}
        while queue:
            current_player, current_box, path = queue.popleft()
            if current_box == target:
                return path
            for move, (dr, dc) in _DIR_TO_DELTA.items():
                next_player = (current_player[0] + dr, current_player[1] + dc)
                next_box = current_box
                if next_player == current_box:
                    pushed_box = (current_box[0] + dr, current_box[1] + dc)
                    if not self._is_free(board, pushed_box, current_box):
                        continue
                    next_box = pushed_box
                elif not self._is_free(board, next_player, current_box):
                    continue
                state = (next_player, next_box)
                if state in visited:
                    continue
                visited.add(state)
                queue.append((next_player, next_box, path + [move]))
        return []

    def _simulate(
        self,
        board: Sequence[Sequence[str]],
        player: Tuple[int, int],
        box: Tuple[int, int],
        moves: Sequence[str],
    ) -> List[Dict[str, Tuple[int, int]]]:
        trace = [{"player": player, "box": box}]
        current_player = player
        current_box = box
        for move in moves:
            dr, dc = _DIR_TO_DELTA[move]
            next_player = (current_player[0] + dr, current_player[1] + dc)
            next_box = current_box
            if next_player == current_box:
                next_box = (current_box[0] + dr, current_box[1] + dc)
            current_player = next_player
            current_box = next_box
            trace.append({"player": current_player, "box": current_box})
        return trace

    @property
    def cell_size(self) -> int:
        side = max(self.width, self.height)
        return max(28, int(512 / (side + 1)) // 2 * 2)

    @property
    def margin(self) -> int:
        side = max(self.width, self.height)
        remaining = 512 - side * self.cell_size
        return remaining // 2 if remaining >= 24 else max(16, self.cell_size // 2)

    def _render_board(
        self,
        board: Sequence[Sequence[str]],
        player: Tuple[int, int],
        box: Tuple[int, int],
        target: Tuple[int, int],
        *,
        player_center_override: Optional[Tuple[float, float]] = None,
        box_center_override: Optional[Tuple[float, float]] = None,
    ) -> Image.Image:
        cell = self.cell_size
        margin = self.margin
        width = self.width * cell + margin * 2
        height = self.height * cell + margin * 2
        image = Image.new("RGB", (width * SS, height * SS), (255, 255, 255))
        draw = ScaledDraw(ImageDraw.Draw(image), SS)
        font = scene_font(max(16, cell // 3))

        for row in range(self.height):
            for col in range(self.width):
                x0 = margin + col * cell
                y0 = margin + row * cell
                x1 = x0 + cell
                y1 = y0 + cell
                fill = (245, 229, 204) if board[row][col] != "#" else (101, 109, 124)
                outline = (168, 130, 102) if board[row][col] != "#" else (66, 72, 84)
                draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=1)

        for row in range(self.height + 1):
            y = margin + row * cell
            draw.line([(margin, y), (margin + self.width * cell, y)], fill=(0, 0, 0), width=1)
        for col in range(self.width + 1):
            x = margin + col * cell
            draw.line([(x, margin), (x, margin + self.height * cell)], fill=(0, 0, 0), width=1)

        for row in range(self.height):
            draw.text((margin * 0.5, margin + row * cell + cell * 0.5), str(row + 1), fill=(0, 0, 0), font=font, anchor="mm")
        for col in range(self.width):
            draw.text((margin + col * cell + cell * 0.5, margin * 0.5), str(col + 1), fill=(0, 0, 0), font=font, anchor="mm")

        tr, tc = target
        tx0 = margin + tc * cell + 8
        ty0 = margin + tr * cell + 8
        tx1 = tx0 + cell - 16
        ty1 = ty0 + cell - 16
        draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=6, outline=(46, 160, 67), width=3)
        draw.line([(tx0, ty0), (tx1, ty1)], fill=(46, 160, 67), width=2)
        draw.line([(tx0, ty1), (tx1, ty0)], fill=(46, 160, 67), width=2)

        br, bc = box
        if box_center_override is not None:
            bcx, bcy = box_center_override
        else:
            bcx = margin + bc * cell + cell / 2
            bcy = margin + br * cell + cell / 2
        bhalf = (cell - 14) / 2
        draw.rounded_rectangle(
            [bcx - bhalf, bcy - bhalf, bcx + bhalf, bcy + bhalf],
            radius=6,
            fill=(160, 113, 74),
            outline=(102, 68, 43),
            width=2,
        )
        draw.line([(bcx - bhalf + 6, bcy - bhalf + 6), (bcx + bhalf - 6, bcy + bhalf - 6)], fill=(84, 57, 36), width=2)
        draw.line([(bcx - bhalf + 6, bcy + bhalf - 6), (bcx + bhalf - 6, bcy - bhalf + 6)], fill=(84, 57, 36), width=2)

        pr, pc = player
        if player_center_override is not None:
            pcx, pcy = player_center_override
        else:
            pcx = margin + pc * cell + cell / 2
            pcy = margin + pr * cell + cell / 2
        prad = (cell - 20) / 2
        draw.ellipse(
            [pcx - prad, pcy - prad, pcx + prad, pcy + prad],
            fill=(255, 214, 10),
            outline=(176, 135, 0),
            width=2,
        )
        return finish_scaled_canvas(image, (width, height))

    def _write_video(
        self,
        board: Sequence[Sequence[str]],
        trace: Sequence[Dict[str, Tuple[int, int]]],
        target: Tuple[int, int],
        video_path: Path,
    ) -> int:
        """Discrete world: hold each committed state equally, then switch directly."""

        if not trace:
            return 0
        hold = max(2, min(6, 156 // len(trace)))
        frames: List[np.ndarray] = []
        for state in trace:
            image = self._render_board(board, state["player"], state["box"], target)
            array = np.array(image.convert("RGB"), dtype=np.uint8)
            frames.extend(array.copy() for _ in range(hold))
        while len(frames) % 4 != 1:
            frames.append(frames[-1].copy())
        assert len(frames) <= 161
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _authored_route_narration(
        self,
        trace: Sequence[Dict[str, Tuple[int, int]]],
        target: Tuple[int, int],
    ) -> str:
        """Narrate one shortest route directly: observe, walk, push, confirm."""

        player0 = trace[0]["player"]
        box0 = trace[0]["box"]
        sentences = [
            f"The agent stands at {display_cell_text(player0)}, the box at {display_cell_text(box0)}, and the "
            f"green target at {display_cell_text(target)}, so the box has to travel from "
            f"{display_cell_text(box0)} to {display_cell_text(target)} one push at a time."
        ]
        walk: List[Tuple[int, int]] = [player0]
        for prev, state in zip(trace, trace[1:]):
            walk.append(state["player"])
            if prev["box"] == state["box"]:
                continue
            box_cell = display_cell_text(state["box"])
            if len(walk) > 2:
                sentences.append(
                    f"Follow the agent positions {display_path_text(walk[:-1])} to reach "
                    f"{display_cell_text(prev['player'])}, the cell the next push must start from."
                )
            sentences.append(
                f"Moving the agent from {display_cell_text(prev['player'])} to "
                f"{display_cell_text(state['player'])} pushes the box from "
                f"{display_cell_text(prev['box'])} to {box_cell}."
            )
            walk = [state["player"]]
        sentences.append(
            f"The box now rests on the green target at {display_cell_text(target)} after "
            f"{len(trace) - 1} moves, and the full agent path is "
            f"{display_path_text(step['player'] for step in trace)}."
        )
        return " ".join(sentences)

    def _state_grid(
        self,
        board: Sequence[Sequence[str]],
        player: Tuple[int, int],
        box: Tuple[int, int],
        target: Tuple[int, int],
    ) -> List[List[int]]:
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                if board[row][col] == "#":
                    grid[row][col] = 9
        tr, tc = target
        grid[tr][tc] = 4
        br, bc = box
        grid[br][bc] = 2 if (br, bc) != target else 5
        pr, pc = player
        grid[pr][pc] = 3
        return grid

    def _build_answer(self, trace: Sequence[Dict[str, Tuple[int, int]]]) -> str:
        return format_xy_path_answer([step["player"] for step in trace])

    def _prompts(self) -> Dict[str, str]:
        ti2v = build_sectioned_prompt(
            _INPUT_DESC.replace("7 by 7", f"{self.height} by {self.width}").replace("1 through 7", f"1 through {self.width}"),
            _TI2V_OUTPUT,
            mode="ti2v",
            rule_description=_RULE_SUMMARY,
            continuity_paragraph=_TI2V_CONTINUITY,
        )
        return {
            "ti2v_prompt": ti2v,
        }

    def create_puzzle(self) -> Dict[str, Any]:
        sample_id = f"{self.TASK_NAME}-{self._counter:05d}"
        self._counter += 1

        board, player, box, target = self._random_state()
        moves = self._solve(board, player, box, target)
        trace = self._simulate(board, player, box, moves)
        open_board_moves = self._solve(self._blank_board(), player, box, target)

        puzzles_dir = self.output_dir / "puzzles"
        solutions_dir = self.output_dir / "solutions"
        videos_dir = self.output_dir / "videos"
        for directory in (puzzles_dir, solutions_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.video:
            videos_dir.mkdir(parents=True, exist_ok=True)

        puzzle_path = puzzles_dir / f"{sample_id}.png"
        solution_path = solutions_dir / f"{sample_id}.png"
        video_path = videos_dir / f"{sample_id}.webm"
        puzzle_image = self._render_board(board, player, box, target)
        final_state = trace[-1]
        solution_image = self._render_board(board, final_state["player"], final_state["box"], target)
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        frame_count = self._write_video(board, trace, target, video_path) if self.video else 0

        push_count = sum(
            1
            for prev, nxt in zip(trace, trace[1:])
            if prev["box"] != nxt["box"]
        )
        prompts = self._prompts()
        answer_text = self._build_answer(trace)
        state_trace = [
            self._state_grid(board, step["player"], step["box"], target)
            for step in trace
        ]
        structured_answer = {
            # This is the only source-side board state consumed by the final
            # verifier.  The remaining fields are reference-authoring
            # derivatives and are scrubbed from canonical semantic_input.
            "initial_grid": state_trace[0],
            "move_sequence": list(moves),
            "state_trace": state_trace,
            "path_length": len(moves),
            "push_count": push_count,
            "difficulty_variant": "wall_detour_multi_push",
            "open_board_path_length": len(open_board_moves),
            "wall_detour_extra_steps": len(moves) - len(open_board_moves),
        }
        return {
            "id": sample_id,
            "answer": answer_text,
            "task_name": self.TASK_NAME,
            "task_family": self.TASK_FAMILY,
            "image": self.relativize_path(puzzle_path),
            "reasoning_image": self.relativize_path(puzzle_path),
            "solution_image_path": self.relativize_path(solution_path),
            "solution_video_path": self.relativize_path(video_path) if frame_count > 0 else None,
            "video_fps": self.video_fps if frame_count > 0 else None,
            "video_num_frames": frame_count if frame_count > 0 else None,
            "question": (
                "Return the complete agent-position path for one shortest valid route that pushes the box "
                "onto the target."
            ),
            "answer_payload": structured_answer,
            **prompts,
        }
