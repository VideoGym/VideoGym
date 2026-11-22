"""Coloured balls must be collected and dropped into matching colour slots, one ball at a time."""

from __future__ import annotations

from collections import deque

from data.common.grid_coordinates import (
    GRID_PATH_STAY_ENCODING_SENTENCE,
    format_xy_path_answer,
    format_xy_position,
)
from data.spatial_navigation.core.multi_agent_base import (
    FrameState,
    MultiAgentSpatialNavigationBase,
    MultiAgentScenario,
    cells_to_moves,
)
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


_COLOR_TABLE = [
    ("red", (220, 70, 70), (140, 30, 30)),
    ("blue", (70, 130, 220), (30, 70, 150)),
    ("green", (70, 170, 90), (30, 100, 50)),
    ("yellow", (240, 200, 60), (170, 130, 20)),
]


def _shortest_collection_route(
    start,
    balls,
    slots,
    size,
    walls,
):
    """BFS the one-ball carrying state instead of concatenating waypoint legs.

    A shortest leg to one selected ball may cross another ball.  Concatenated
    point-to-point paths therefore do not implement the visible automatic
    pickup rule; this state-space search does.
    """

    ball_index = {tuple(cell): index for index, cell in enumerate(balls)}
    full_mask = (1 << len(balls)) - 1
    initial = (tuple(start), full_mask, -1, 0)
    queue = deque([initial])
    parent = {initial: None}
    while queue:
        state = queue.popleft()
        cell, remaining, carrying, delivered = state
        if delivered == full_mask and carrying < 0:
            states = [state]
            while parent[states[-1]] is not None:
                states.append(parent[states[-1]])
            return [item[0] for item in reversed(states)]
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                continue
            next_remaining = int(remaining)
            next_carrying = int(carrying)
            next_delivered = int(delivered)
            ball_idx = ball_index.get(nxt)
            if next_carrying < 0 and ball_idx is not None and next_remaining & (1 << ball_idx):
                next_remaining &= ~(1 << ball_idx)
                next_carrying = ball_idx
            if next_carrying >= 0 and nxt == tuple(slots[next_carrying]):
                next_delivered |= 1 << next_carrying
                next_carrying = -1
            next_state = (nxt, next_remaining, next_carrying, next_delivered)
            if next_state in parent:
                continue
            parent[next_state] = state
            queue.append(next_state)
    return None


class ColorMatchedCollectionGenerator(MultiAgentSpatialNavigationBase):
    TASK_NAME = "color_matched_collection"
    TASK_FAMILY = "spatial_navigation"
    TASK_TITLE = "Color Matched Collection"
    DEFAULT_SIZE = 8

    INPUT_DESC = (
        "The board has a dark grey circular agent and several coloured balls (red, blue, green, yellow) and "
        "matching coloured slots. Each ball is a shaded sphere; each slot is a hollow ring of the same "
        "colour. Dark cells are walls."
    )
    RULES_DESC = (
        "Pick up a ball by stepping on its cell, drop it into the matching colour slot by stepping on the "
        "slot, then continue to the next ball. The agent carries only one ball at a time. Each step may move up, "
        "right, down, or left, or use stay to remain on the current cell. Complete all pickup-then-slot pairs in "
        f"any legal order; route length is not scored. {GRID_PATH_STAY_ENCODING_SENTENCE}"
    )
    TI2V_OUTPUT = "Generate a short video showing the agent collecting each coloured ball into its matching slot."
    SOLUTION_IMAGE_OUTPUT = "Render the solved still image with the full collection route drawn in blue."

    @staticmethod
    def _pos(cell) -> str:
        return format_xy_position((cell[0] + 1, cell[1] + 1))

    def _authored_reasoning(self, scenario: MultiAgentScenario) -> str:
        from data.spatial_navigation.core.maze_style import route_run_breakdown

        path = [tuple(c) for c in scenario.primary_path]
        balls = [tuple(c) for c in scenario.extras["balls"]]
        slots = [tuple(c) for c in scenario.extras["slots"]]
        pair_text = "; ".join(
            f"the {name} ball at {self._pos(ball)} belongs in the {name} ring at {self._pos(slot)}"
            for (name, _f, _o), ball, slot in zip(_COLOR_TABLE, balls, slots)
        )
        sentences = [
            f"The collector starts at {self._pos(path[0])} holding nothing, and each colour must go home: "
            f"{pair_text}. Only one ball fits in hand at a time, so the tour is a chain of pickup-then-deliver "
            "pairs, and walking over another ball while carrying does nothing.",
        ]
        # Replay events to anchor legs.
        present = [True] * len(balls)
        carrying = None
        anchor = 0
        for step, cell in enumerate(path):
            if step == 0:
                continue
            if carrying is None and cell in balls:
                index = balls.index(cell)
                if present[index]:
                    present[index] = False
                    carrying = index
                    runs = route_run_breakdown(path[anchor : step + 1])
                    leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
                    sentences.append(
                        f"It walks {leg} and picks up the {_COLOR_TABLE[index][0]} ball."
                    )
                    anchor = step
                    continue
            if carrying is not None and cell == slots[carrying]:
                runs = route_run_breakdown(path[anchor : step + 1])
                leg = ", then ".join(f"{d} {len(r) - 1}" for d, r in runs)
                sentences.append(
                    f"Carrying it, the collector walks {leg} and drops the {_COLOR_TABLE[carrying][0]} ball into "
                    "its matching ring."
                )
                carrying = None
                anchor = step
        sentences.append(
            "All four balls now rest in their matching rings, each carried alone, so the whole collection is legal."
        )
        return " ".join(sentences)

    def __init__(self, *args, route_steps=(12, 30), **kwargs):
        self.route_steps = tuple(route_steps)
        super().__init__(*args, **kwargs)

    def sample_scenario(self) -> MultiAgentScenario:
        size = self.size
        for _ in range(600):
            cells_pool = [(r, c) for r in range(size) for c in range(size)]
            self.rng.shuffle(cells_pool)
            walls = cells_pool[: self.rng.randint(2, 4)]
            free = [c for c in cells_pool if c not in walls]
            # Need at least the agent + 4 ball/slot pairs = 9 free cells.
            if len(free) < 10:
                continue
            agent_start = free[0]
            # All 4 colors (red, blue, green, yellow) defined in _COLOR_TABLE must
            # be placed; the prompt explicitly anchors all four colors as visible.
            num_pairs = 4
            balls = []
            slots = []
            used = {agent_start}
            for k in range(num_pairs):
                b = next((c for c in free if c not in used), None)
                if b is None:
                    break
                used.add(b)
                s = next((c for c in free if c not in used), None)
                if s is None:
                    break
                used.add(s)
                balls.append(b)
                slots.append(s)
            if len(balls) != num_pairs:
                continue
            best_cells = _shortest_collection_route(
                agent_start, balls, slots, size, set(walls)
            )
            if best_cells is None:
                continue
            # Four identity-preserving pickup/drop pairs require multi-step
            # ordering, while the upper bound guarantees that every cell-to-cell
            # motion and all eight commits fit the native video contract without
            # temporal subsampling.
            if not self.route_steps[0] <= len(best_cells) - 1 <= self.route_steps[1]:
                continue
            scenario = MultiAgentScenario(
                task_name=self.TASK_NAME,
                task_family=self.TASK_FAMILY,
                title=self.TASK_TITLE,
                size=size,
                walls=walls,
                primary_path=best_cells,
                primary_moves=cells_to_moves(best_cells),
                extras={
                    "agent_start": agent_start,
                    "balls": balls,
                    "slots": slots,
                    "color_indices": list(range(num_pairs)),
                },
            )
            scenario.answer_text = format_xy_path_answer(scenario.primary_path)
            return scenario
        raise RuntimeError("color_matched_collection sample failed")

    def render_static_layer(self, draw, image, scenario):
        for i, (ball, slot) in enumerate(zip(scenario.extras["balls"], scenario.extras["slots"])):
            name, fill, outline = _COLOR_TABLE[i % len(_COLOR_TABLE)]
            # slot ring
            sx, sy = self.cell_center(slot)
            r2 = self.cell_size * 0.32
            draw.ellipse([sx - r2, sy - r2, sx + r2, sy + r2], outline=outline, width=4)
            draw.ellipse([sx - r2 * 0.55, sy - r2 * 0.55, sx + r2 * 0.55, sy + r2 * 0.55], outline=outline, width=2)

    def _draw_ball(self, draw, center, index, *, radius_factor=0.30):
        _name, fill, outline = _COLOR_TABLE[index % len(_COLOR_TABLE)]
        cx, cy = center
        radius = self.cell_size * radius_factor
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=fill,
            outline=outline,
            width=2,
        )
        draw.ellipse(
            [cx - radius * 0.45, cy - radius * 0.65, cx + radius * 0.05, cy - radius * 0.20],
            fill=tuple(min(255, channel + 80) for channel in fill),
        )

    def render_dynamic_layer(self, draw, image, scenario, frame_state):
        present = list(frame_state.extras.get("ball_present", [True] * len(scenario.extras["balls"])))
        delivered = set(int(index) for index in frame_state.extras.get("delivered", []))
        for index, ball in enumerate(scenario.extras["balls"]):
            if index < len(present) and present[index]:
                self._draw_ball(draw, self.cell_center(ball), index)
        for index in delivered:
            if 0 <= index < len(scenario.extras["slots"]):
                self._draw_ball(draw, self.cell_center(scenario.extras["slots"][index]), index, radius_factor=0.21)

    def render_solution_overlay(self, draw, image, scenario):
        if len(scenario.primary_path) >= 2:
            pts = [self.cell_center(c) for c in scenario.primary_path]
            draw_spatial_navigation_path(draw, pts, color=(56, 130, 220), width=max(5, self.cell_size // 7))

    def render_path_overlay(self, draw, path_cells, color=(214, 42, 42)) -> None:
        return

    def render_agents(self, draw, image, scenario, frame_state):
        cx, cy = self.cell_center(frame_state.primary_pos)
        self.draw_circle_agent(draw, (cx, cy), fill=(80, 80, 90), outline=(30, 30, 40))
        carrying = frame_state.extras.get("carrying")
        if isinstance(carrying, int):
            self._draw_ball(
                draw,
                (cx + self.cell_size * 0.22, cy - self.cell_size * 0.22),
                carrying,
                radius_factor=0.13,
            )

    def _discrete_collection_states(self, scenario):
        balls = [tuple(cell) for cell in scenario.extras["balls"]]
        slots = [tuple(cell) for cell in scenario.extras["slots"]]
        present = [True] * len(balls)
        delivered = set()
        carrying = None
        states = []
        for cell in scenario.primary_path:
            if carrying is None and cell in balls:
                index = balls.index(cell)
                if present[index] and index not in delivered:
                    present[index] = False
                    carrying = index
            if isinstance(carrying, int) and cell == slots[carrying]:
                delivered.add(carrying)
                carrying = None
            states.append(
                {
                    "ball_present": list(present),
                    "delivered": sorted(delivered),
                    "carrying": carrying,
                }
            )
        return states

    def build_video_frame_states(self, scenario, substeps):
        path = list(scenario.primary_path)
        if not path:
            return []
        discrete = self._discrete_collection_states(scenario)
        return [
            FrameState(primary_pos=cell, primary_cell=cell, extras=dict(extra))
            for cell, extra in zip(path, discrete)
        ]
