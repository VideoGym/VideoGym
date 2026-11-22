"""Shared base for multi-agent and dynamic Spatial Navigation tasks.

Each variant extends `MultiAgentSpatialNavigationBase` and only overrides:
- TASK_NAME / TASK_FAMILY / TASK_TITLE
- _input_desc / _rules_desc / _ti2v_output / _solution_image_output / _ti2t_output (text-only)
- sample_scenario_data() returning a dict with rich state
- _render_extras(draw, scenario, *, frame_state) for per-task icons
- _format_answer(scenario) producing the canonical `Answer: ...` string

The base handles:
- 512x512 canvas, even cell_size, integer-pixel alignment
- Coordinate-labelled grid background
- 4 prompt modes
- Exact integer-cell solution video with fixed-duration semantic state holds
- Standard payload shape
"""

from __future__ import annotations

from data.common.assets import render_font

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.common.grid_coordinates import (
    format_named_xy_paths_answer,
    format_xy_path_answer,
)
from data.common.base import AbstractPuzzleGenerator
from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    finish_scaled_canvas,
    new_scaled_canvas,
)
from data.common.video_encoding import encode_rgb_frames_to_webm, fit_frames_to_video_budget
from data.common.animation_policy import (
    DISCRETE_STATE_HOLD_FRAMES,
    STATE_HOLD_FRAMES,
    frame_budget_for_frame_count,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path
from data.spatial_navigation.core.prompt_contract import build_spatial_navigation_prompt_bundle

_DIR_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "stay": (0, 0),
}
_CANVAS_BG = (252, 252, 252)
_GRID_LINE = (210, 210, 210)
# Family-wide royal-blue 1-based axis numbers (same as the maze scenes).
_LABEL_COLOR = (30, 90, 220)
_WALL_FILL = (54, 64, 82)
_WALL_OUTLINE = (32, 38, 52)


@dataclass
class FrameState:
    """One render-frame snapshot. Tasks fill what they need."""
    primary_pos: Tuple[float, float]            # exact cell-coordinate position
    secondary_pos: Optional[Tuple[float, float]] = None
    primary_cell: Tuple[int, int] = (0, 0)      # integer cell at this frame (for icon placement)
    secondary_cell: Optional[Tuple[int, int]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    overlay_path: Optional[List[Tuple[int, int]]] = None


@dataclass
class MultiAgentScenario:
    task_name: str
    task_family: str
    title: str
    size: int
    walls: List[Tuple[int, int]] = field(default_factory=list)
    primary_path: List[Tuple[int, int]] = field(default_factory=list)  # The agent's full trajectory
    secondary_path: List[Tuple[int, int]] = field(default_factory=list)
    primary_moves: List[str] = field(default_factory=list)
    secondary_moves: List[str] = field(default_factory=list)
    answer_text: str = ""                       # Canonical "Answer: ..."
    extras: Dict[str, Any] = field(default_factory=dict)


def even(n: int) -> int:
    return int(n) // 2 * 2


def neighbors4(rc: Tuple[int, int], size: int, blocked: Iterable[Tuple[int, int]] = ()) -> List[Tuple[str, Tuple[int, int]]]:
    blocked_set = set(blocked)
    out = []
    for d, (dr, dc) in (("up", (-1, 0)), ("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1))):
        nr, nc = rc[0] + dr, rc[1] + dc
        if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in blocked_set:
            out.append((d, (nr, nc)))
    return out


def bfs_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    size: int,
    blocked: Iterable[Tuple[int, int]] = (),
) -> Optional[List[Tuple[int, int]]]:
    """Shortest path from start to goal (4-connected). Returns list of cells or None."""
    blocked_set = set(blocked)
    if start in blocked_set or goal in blocked_set:
        return None
    from collections import deque
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for _d, nb in neighbors4(cur, size, blocked_set):
            if nb in seen:
                continue
            seen.add(nb)
            parent[nb] = cur
            q.append(nb)
    return None


def cells_to_moves(cells: Sequence[Tuple[int, int]]) -> List[str]:
    out = []
    for a, b in zip(cells, cells[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        if (dr, dc) == (0, 0):
            out.append("stay")
        else:
            for name, (dr2, dc2) in _DIR_TO_DELTA.items():
                if (dr, dc) == (dr2, dc2):
                    out.append(name)
                    break
    return out


class MultiAgentSpatialNavigationBase(AbstractPuzzleGenerator[Dict[str, Any]]):
    TASK_NAME = "ma_grid_task"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Multi-Agent Grid Task"

    DEFAULT_SIZE = 8
    CANVAS_SIDE = 512
    # Family line-quality floor: scenes draw supersampled through the shared
    # render primitives and LANCZOS-downsample back to the 512 canvas.
    SCENE_SUPERSAMPLE = DOCTRINE_SUPERSAMPLE

    # Subclasses override these (small text-only):
    INPUT_DESC: str = ""
    RULES_DESC: str = ""
    TI2V_OUTPUT: str = ""
    SOLUTION_IMAGE_OUTPUT: str = ""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        video: bool = True,
        video_fps: int = 16,
    ) -> None:
        resolved = output_dir if output_dir is not None else Path(tempfile.gettempdir()) / f"videogym_spatial_navigation_{self.TASK_NAME}"
        super().__init__(resolved)
        self.rng = random.Random(seed)
        self.seed = seed
        self.size = int(size) if size is not None else self.DEFAULT_SIZE
        self.video = bool(video)
        self.video_fps = int(video_fps)
        # Compute cell_size, even, fitting 512×512 with margins.
        ideal = self.CANVAS_SIDE / (self.size + 1)
        self.cell_size = max(28, even(int(ideal)))
        remaining = self.CANVAS_SIDE - self.size * self.cell_size
        self.margin = max(16, remaining // 2)
        self._counter = 0
        self._font_label = self._load_font(max(14, self.cell_size // 4))
        self._font_id = self._load_font(max(11, self.cell_size // 5))
        self._font_big = self._load_font(max(18, self.cell_size // 3))

    def record_to_dict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return dict(record)

    # ----- subclass API -----

    def sample_scenario(self) -> MultiAgentScenario:
        raise NotImplementedError

    def render_static_layer(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        scenario: MultiAgentScenario,
    ) -> None:
        """Draw fixed elements (walls, goals, decor). Called before agents."""
        pass

    def render_dynamic_layer(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        scenario: MultiAgentScenario,
        frame_state: FrameState,
    ) -> None:
        """Draw state-dependent board elements before paths and agents."""
        pass

    def render_agents(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        scenario: MultiAgentScenario,
        frame_state: FrameState,
    ) -> None:
        """Draw moving agents at frame_state positions."""
        pass

    def render_solution_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        scenario: MultiAgentScenario,
    ) -> None:
        """Draw final solution path (red lines) on the static solution image."""
        pass

    def build_video_frame_states(self, scenario: MultiAgentScenario, substeps: int) -> List[FrameState]:
        """Build exact grid states with a uniform hold cadence."""
        states: List[FrameState] = []
        path = scenario.primary_path
        if not path:
            return states
        for cell in path:
            self._append_state_hold(
                states,
                FrameState(primary_pos=(cell[0], cell[1]), primary_cell=cell),
                DISCRETE_STATE_HOLD_FRAMES,
            )
        return states

    @staticmethod
    def _clone_frame_state(state: FrameState) -> FrameState:
        return FrameState(
            primary_pos=tuple(state.primary_pos),
            secondary_pos=tuple(state.secondary_pos) if state.secondary_pos is not None else None,
            primary_cell=tuple(state.primary_cell),
            secondary_cell=tuple(state.secondary_cell) if state.secondary_cell is not None else None,
            extras=dict(state.extras),
            overlay_path=list(state.overlay_path) if state.overlay_path is not None else None,
        )

    def _append_state_hold(
        self,
        states: List[FrameState],
        state: FrameState,
        count: int = STATE_HOLD_FRAMES,
    ) -> None:
        states.extend(self._clone_frame_state(state) for _ in range(max(0, int(count))))

    # ----- shared rendering -----

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        return render_font(int(size) * self.SCENE_SUPERSAMPLE, bold=False)

    def cell_box(self, rc: Tuple[int, int]) -> Tuple[int, int, int, int]:
        r, c = rc
        left = self.margin + c * self.cell_size
        top = self.margin + r * self.cell_size
        return left, top, left + self.cell_size, top + self.cell_size

    def cell_center(self, rc: Tuple[float, float]) -> Tuple[float, float]:
        r, c = rc
        return (self.margin + (c + 0.5) * self.cell_size, self.margin + (r + 0.5) * self.cell_size)

    def canvas_size(self) -> Tuple[int, int]:
        side = self.margin * 2 + self.size * self.cell_size
        return side, side

    def render_grid(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        scenario: MultiAgentScenario,
    ) -> None:
        wall_set = set(scenario.walls)
        for r in range(self.size):
            for c in range(self.size):
                left, top, right, bottom = self.cell_box((r, c))
                if (r, c) in wall_set:
                    draw.rectangle([left, top, right, bottom], fill=_WALL_FILL, outline=_WALL_OUTLINE, width=1)
                else:
                    draw.rectangle([left, top, right, bottom], outline=_GRID_LINE, width=1)
        # row/col labels
        for r in range(self.size):
            draw.text(
                (self.margin - self.cell_size * 0.32, self.margin + r * self.cell_size + self.cell_size * 0.5),
                str(r + 1),
                fill=_LABEL_COLOR,
                font=self._font_label,
                anchor="mm",
            )
        for c in range(self.size):
            draw.text(
                (self.margin + c * self.cell_size + self.cell_size * 0.5, self.margin - self.cell_size * 0.32),
                str(c + 1),
                fill=_LABEL_COLOR,
                font=self._font_label,
                anchor="mm",
            )

    def render_path_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        path_cells: Sequence[Tuple[int, int]],
        color: Tuple[int, int, int] = (214, 42, 42),
    ) -> None:
        if len(path_cells) < 2:
            return
        pts = [self.cell_center(c) for c in path_cells]
        draw_spatial_navigation_path(draw, pts, color=color, width=max(5, self.cell_size // 7))

    @staticmethod
    def _discrete_frame_state(frame_state: FrameState) -> FrameState:
        def exact_cell(raw_cell: Tuple[float, float], field: str) -> Tuple[int, int]:
            values = tuple(float(value) for value in raw_cell)
            rounded = tuple(int(round(value)) for value in values)
            if any(abs(value - target) > 1e-6 for value, target in zip(values, rounded)):
                raise ValueError(f"Spatial Navigation discrete state {field} contains a fractional cell: {raw_cell!r}")
            return rounded

        primary_cell = exact_cell(frame_state.primary_cell, "primary_cell")
        secondary_cell = (
            exact_cell(frame_state.secondary_cell, "secondary_cell")
            if frame_state.secondary_cell is not None
            else None
        )
        extras = dict(frame_state.extras)
        for key, value in list(extras.items()):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key == "clock_position":
                rounded = round(float(value))
                if abs(float(value) - rounded) > 1e-6:
                    raise ValueError(f"Spatial Navigation discrete state {key} is fractional: {value!r}")
                extras[key] = float(rounded)
            elif key == "world_progress" or key.endswith("_progress"):
                if float(value) not in (0.0, 1.0):
                    raise ValueError(f"Spatial Navigation discrete state {key} is fractional: {value!r}")
                extras[key] = float(value)
        if extras.get("world_progress") == 1.0 and "to_parity" in extras:
            extras["parity"] = extras["to_parity"]
        return FrameState(
            primary_pos=(primary_cell[0], primary_cell[1]),
            secondary_pos=(
                (secondary_cell[0], secondary_cell[1])
                if secondary_cell is not None
                else None
            ),
            primary_cell=primary_cell,
            secondary_cell=secondary_cell,
            extras=extras,
            overlay_path=frame_state.overlay_path,
        )

    @classmethod
    def _dedupe_discrete_states(cls, states: Sequence[FrameState]) -> List[FrameState]:
        out: List[FrameState] = []
        previous_key: Optional[Tuple[Any, ...]] = None
        for raw_state in states:
            state = cls._discrete_frame_state(raw_state)
            key = (
                state.primary_cell,
                state.secondary_cell,
                tuple(sorted((str(k), repr(v)) for k, v in state.extras.items())),
                tuple(state.overlay_path or ()),
            )
            if key == previous_key:
                continue
            out.append(state)
            previous_key = key
        return out

    def make_image(
        self,
        scenario: MultiAgentScenario,
        *,
        frame_state: Optional[FrameState] = None,
        show_path: bool = False,
        path_cells: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Image.Image:
        w, h = self.canvas_size()
        img, d = new_scaled_canvas(w, h, self.SCENE_SUPERSAMPLE, _CANVAS_BG)
        if frame_state is None:
            frame_state = FrameState(
                primary_pos=(scenario.primary_path[0][0] if scenario.primary_path else 0,
                             scenario.primary_path[0][1] if scenario.primary_path else 0),
                primary_cell=scenario.primary_path[0] if scenario.primary_path else (0, 0),
                secondary_pos=(scenario.secondary_path[0][0], scenario.secondary_path[0][1]) if scenario.secondary_path else None,
                secondary_cell=scenario.secondary_path[0] if scenario.secondary_path else None,
            )
        self.render_grid(d, img, scenario)
        self.render_static_layer(d, img, scenario)
        self.render_dynamic_layer(d, img, scenario, frame_state)
        if show_path and path_cells:
            self.render_path_overlay(d, path_cells)
        if show_path:
            self.render_solution_overlay(d, img, scenario)
        self.render_agents(d, img, scenario, frame_state)
        return finish_scaled_canvas(img, (w, h))

    def write_video(self, scenario: MultiAgentScenario, video_path: Path) -> int:
        raw_states = self.build_video_frame_states(scenario, 1)
        semantic_states = self._dedupe_discrete_states(raw_states)
        if not semantic_states:
            return 0
        frames: List[np.ndarray] = []
        protected_indices: List[int] = []
        for state in semantic_states:
            img = self.make_image(scenario, frame_state=state)
            start_index = len(frames)
            frame = np.array(img.convert("RGB"), dtype=np.uint8)
            frames.extend(frame.copy() for _ in range(DISCRETE_STATE_HOLD_FRAMES))
            protected_indices.append(start_index)
        budget = frame_budget_for_frame_count(len(frames))
        frames = fit_frames_to_video_budget(
            frames,
            max_frames=budget,
            final_hold_frames=DISCRETE_STATE_HOLD_FRAMES,
            protected_indices=protected_indices,
        )
        if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
            return 0
        return len(frames)

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> Optional[str]:
        """Sample-specific solving reasoning for a `text_cot` task.

        Observe the layout, name the binding mechanism, walk the joint
        timeline with its state changes, and confirm the terminal state.  The
        base returns ``None`` so tasks without an authored trace fail loudly
        at SFT reference build instead of shipping template prose.
        """

        return None

    def _authored_reasoning_segments(
        self,
        scenario: MultiAgentScenario,
    ) -> Optional[List[str]]:
        """Interleaved reasoning segments: one per key-state image plus a
        closing segment (``images + 1`` entries)."""

        return None


    def _cells_from_extra(self, raw: Any) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        if isinstance(raw, dict):
            if "cell" in raw:
                cells.extend(self._cells_from_extra(raw["cell"]))
            return cells
        if isinstance(raw, (list, tuple)):
            if len(raw) == 2 and all(isinstance(part, int) for part in raw):
                return [(int(raw[0]), int(raw[1]))]
            for item in raw:
                cells.extend(self._cells_from_extra(item))
        return cells


    # ----- prompts -----

    def _prompts(self, scenario: MultiAgentScenario) -> Dict[str, str]:
        return build_spatial_navigation_prompt_bundle(
            task_name=scenario.task_name,
            input_description=self.INPUT_DESC,
            rule_description=self.RULES_DESC,
            ti2v_output=self.TI2V_OUTPUT,
            moving_nouns="the agents and every object the rules move",
        )

    # ----- record assembly -----

    @staticmethod
    def _canonical_answer_text(scenario: MultiAgentScenario) -> str:
        """Serialize every rectangular-grid route at the shared model boundary."""

        if scenario.task_name == "agent_courier_relay":
            leg1 = [tuple(cell) for cell in scenario.extras.get("leg1", [])]
            leg2 = [tuple(cell) for cell in scenario.extras.get("leg2", [])]
            return format_named_xy_paths_answer({"A": leg1, "B": leg2})
        if scenario.task_name == "cooperative_open_door":
            return format_named_xy_paths_answer(
                {
                    "blue": [tuple(cell) for cell in scenario.primary_path],
                    "orange": [tuple(cell) for cell in scenario.secondary_path],
                }
            )
        return format_xy_path_answer(scenario.primary_path)

    def create_puzzle(self) -> Dict[str, Any]:
        scenario = self.sample_scenario()
        scenario.answer_text = self._canonical_answer_text(scenario)
        puzzle_id = f"{self.TASK_NAME}-{self._counter:05d}"
        self._counter += 1

        puzzle_img = self.make_image(scenario)
        terminal_states = self.build_video_frame_states(scenario, 1)
        if terminal_states:
            end_state = self._discrete_frame_state(terminal_states[-1])
        else:
            end_state = FrameState(
                primary_pos=(scenario.primary_path[-1][0], scenario.primary_path[-1][1]) if scenario.primary_path else (0, 0),
                primary_cell=scenario.primary_path[-1] if scenario.primary_path else (0, 0),
                secondary_pos=(scenario.secondary_path[-1][0], scenario.secondary_path[-1][1]) if scenario.secondary_path else None,
                secondary_cell=scenario.secondary_path[-1] if scenario.secondary_path else None,
            )
        solution_img = self.make_image(scenario, frame_state=end_state, show_path=True, path_cells=scenario.primary_path)

        img_path = self.output_dir / f"{puzzle_id}_puzzle.png"
        sol_path = self.output_dir / f"{puzzle_id}_solution.png"
        puzzle_img.save(img_path)
        solution_img.save(sol_path)

        prompts = self._prompts(scenario)
        answer_text = scenario.answer_text
        record: Dict[str, Any] = {
            "id": puzzle_id,
            "task_name": scenario.task_name,
            "task_family": scenario.task_family,
            "image": self.relativize_path(img_path),
            "reasoning_image": self.relativize_path(img_path),
            "solution_image_path": self.relativize_path(sol_path),
            "answer": answer_text,
            "answer_payload": {
                "grid_size": scenario.size,
                "walls": [[r, c] for r, c in scenario.walls],
                "primary_path": [[r, c] for r, c in scenario.primary_path],
                "secondary_path": [[r, c] for r, c in scenario.secondary_path],
                "primary_moves": list(scenario.primary_moves),
                "secondary_moves": list(scenario.secondary_moves),
                "extras": dict(scenario.extras),
            },
            "seed": self.seed,
            **prompts,
        }
        if self.video:
            video_path = self.output_dir / f"{puzzle_id}_solution.webm"
            n = self.write_video(scenario, video_path)
            record["solution_video_path"] = self.relativize_path(video_path)
            record["video_fps"] = self.video_fps
            record["video_num_frames"] = n
        return record

    # ----- shared icon helpers (product-grade) -----

    def draw_circle_agent(
        self,
        draw: ImageDraw.ImageDraw,
        center_xy: Tuple[float, float],
        *,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
        radius_factor: float = 0.32,
        ring_color: Optional[Tuple[int, int, int]] = (255, 255, 255),
    ) -> None:
        cx, cy = center_xy
        rad = self.cell_size * radius_factor
        if ring_color is not None:
            draw.ellipse([cx - rad - 2, cy - rad - 2, cx + rad + 2, cy + rad + 2], fill=ring_color)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=fill, outline=outline, width=2)
        # highlight
        hr = rad * 0.35
        draw.ellipse([cx - rad * 0.45, cy - rad * 0.55, cx - rad * 0.45 + hr, cy - rad * 0.55 + hr],
                     fill=tuple(min(255, c + 60) for c in fill))

    def draw_diamond_agent(
        self,
        draw: ImageDraw.ImageDraw,
        center_xy: Tuple[float, float],
        *,
        fill: Tuple[int, int, int],
        outline: Tuple[int, int, int],
    ) -> None:
        cx, cy = center_xy
        r = self.cell_size * 0.34
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=fill, outline=outline)
        # inner band
        draw.polygon([(cx, cy - r * 0.55), (cx + r * 0.55, cy), (cx, cy + r * 0.55), (cx - r * 0.55, cy)],
                     fill=tuple(min(255, c + 40) for c in fill))

    def draw_goal_flag(
        self,
        draw: ImageDraw.ImageDraw,
        rc: Tuple[int, int],
        *,
        fill: Tuple[int, int, int] = (228, 52, 52),
    ) -> None:
        cx, cy = self.cell_center(rc)
        # pole
        pole_h = self.cell_size * 0.55
        pole_x = cx - self.cell_size * 0.10
        draw.line([(pole_x, cy + pole_h * 0.5), (pole_x, cy - pole_h * 0.5)], fill=(70, 70, 70), width=3)
        # flag triangle
        pts = [
            (pole_x, cy - pole_h * 0.5),
            (pole_x + self.cell_size * 0.32, cy - pole_h * 0.30),
            (pole_x, cy - pole_h * 0.10),
        ]
        draw.polygon(pts, fill=fill, outline=(120, 20, 20))

    def draw_filled_cell(
        self,
        draw: ImageDraw.ImageDraw,
        rc: Tuple[int, int],
        *,
        fill: Tuple[int, int, int],
        outline: Optional[Tuple[int, int, int]] = None,
        inset: int = 4,
    ) -> None:
        l, t, r, b = self.cell_box(rc)
        ol = outline or tuple(max(0, c - 50) for c in fill)
        draw.rounded_rectangle([l + inset, t + inset, r - inset, b - inset],
                               radius=max(4, inset), fill=fill, outline=ol, width=2)
