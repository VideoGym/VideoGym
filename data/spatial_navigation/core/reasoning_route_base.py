"""Shared helpers for compact reasoning-focused Spatial Navigation tasks."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from PIL import Image, ImageDraw

from data.spatial_navigation.core.mechanic_base import MechanicGridGeneratorBase, MechanicScenario
from data.spatial_navigation.core.path_rendering import draw_spatial_navigation_path


Coord = Tuple[int, int]


class ReasoningRouteGeneratorBase(MechanicGridGeneratorBase):
    """Small deterministic Spatial Navigation renderer for new reasoning-capability tasks."""

    DEFAULT_SIZE = 7

    def _build_scenario(
        self,
        *,
        task_name: str,
        task_family: str,
        title: str,
        input_desc: str,
        rules_desc: str,
        path: Sequence[Coord],
        walls: Sequence[Coord],
        extras: Dict[str, Any],
        structured_answer: Dict[str, Any],
    ) -> MechanicScenario:
        return MechanicScenario(
            task_name=task_name,
            task_family=task_family,
            title=title,
            input_desc=input_desc,
            rules_desc=rules_desc,
            ti2v_output="Generate a video that follows the solved route while preserving all walls, labels, and tokens.",
            solution_image_output="Render the solved still by drawing the solved route in red on the unchanged grid.",
            size=self.size,
            start=tuple(path[0]),
            goal=tuple(path[-1]),
            agent_path=list(path),
            walls=list(walls),
            extras=extras,
            structured_answer=structured_answer,
        )

    def _draw_extra_tokens(self, draw: ImageDraw.ImageDraw, scenario: MechanicScenario, *, solved: bool) -> None:
        token_font = self._load_font(max(12, self.cell_size // 5))
        for label, cell in (scenario.extras.get("labels") or {}).items():
            cx, cy = self._cell_center(tuple(cell))
            draw.text((cx, cy), str(label), fill=(30, 30, 30), font=token_font, anchor="mm")
        for cell in scenario.extras.get("highlight_cells") or []:
            cx, cy = self._cell_center(tuple(cell))
            radius = self.cell_size * 0.28
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(50, 120, 210), width=3)
        for cell in scenario.extras.get("orange_cells") or []:
            cx, cy = self._cell_center(tuple(cell))
            radius = self.cell_size * 0.26
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(244, 177, 88), outline=(143, 89, 34), width=2)
        if solved:
            for path in scenario.extras.get("secondary_paths") or []:
                draw_spatial_navigation_path(
                    draw,
                    [self._cell_center(tuple(cell)) for cell in path],
                    color=(220, 41, 41),
                    width=max(5, self.cell_size // 7),
                )

    def _render_scene(
        self,
        scenario: MechanicScenario,
        *,
        agent_cell: Coord,
        agent_center: Tuple[float, float] | None = None,
        show_path: bool = False,
    ):
        """Draw the full scene on the supersampled canvas, unfinished.

        Subclasses that add overlays (heading badges, extra markers) draw on
        the returned ScaledDraw and then call ``_finish_scene`` themselves so
        every stroke shares the same supersampled pass.
        """

        image, draw = self._begin_scene()
        self._draw_grid_and_labels(draw, wall_set=set(scenario.walls))
        if show_path:
            self._draw_path_overlay(draw, scenario.agent_path)
        self._draw_extra_tokens(draw, scenario, solved=show_path)
        self._draw_start_goal(draw, scenario)
        self._draw_agent(draw, agent_center if agent_center is not None else self._cell_center(agent_cell))
        return image, draw

    def _render(
        self,
        scenario: MechanicScenario,
        *,
        agent_cell: Coord,
        agent_center: Tuple[float, float] | None = None,
        show_path: bool = False,
    ) -> Image.Image:
        image, _draw = self._render_scene(
            scenario,
            agent_cell=agent_cell,
            agent_center=agent_center,
            show_path=show_path,
        )
        return self._finish_scene(image)

    def create_puzzle(self) -> Dict[str, Any]:  # type: ignore[override]
        scenario = self.sample_scenario()
        puzzle_image = self._render(scenario, agent_cell=scenario.start, show_path=False)
        solution_image = self._render(scenario, agent_cell=scenario.goal, show_path=True)

        def render_frame(agent_cell, agent_center, _step_index):
            return self._render(scenario, agent_cell=agent_cell, agent_center=agent_center, show_path=False)

        video_frames = self._animate_path(scenario, render_frame)
        record = self._assemble_record(
            scenario,
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            video_frames=video_frames,
        )
        return record


__all__ = ["ReasoningRouteGeneratorBase"]
