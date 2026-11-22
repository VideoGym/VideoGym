"""Filing cabinet with locked drawers — route papers via unlocked drawers.

A vertical filing cabinet has 5 alphabetic DRAWERS A,B,C,D,E (top→bottom).
Some drawers are LOCKED (shown with a padlock icon) and CANNOT be opened
(no paper may enter or leave a locked drawer for the duration of the
puzzle). The remaining drawers are UNLOCKED.

Each drawer holds a vertical stack of numbered PAPERS. Papers are LIFO:
each move pulls the TOP paper from one UNLOCKED source drawer and places
it on TOP of one UNLOCKED destination drawer.

Move notation: ``(src_drawer, dst_drawer)`` letters from {A,B,C,D,E},
with the constraint that both must be unlocked.

Mechanic load-bearing check (Tier B): we generate puzzles where the
plan's optimal length (under the locked-drawer restriction) is strictly
greater than the optimal length WITHOUT lock restrictions — i.e. removing
the locks would let you solve faster, so the lock pattern is load-bearing.
"""

from __future__ import annotations

from data.rearrangement_game.core.object_plan_base import RearrangementObjectPlanGenerator

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from data.common.pipeline.object_plans import (
    ITEM_PALETTE,
    _shade,
)

PANEL_BG = (245, 244, 240)
PANEL_BORDER = (204, 201, 194)
CABINET_FRAME = (150, 146, 136)
CABINET_FACE = (222, 219, 210)
DRAWER_FACE = (238, 236, 228)
DRAWER_FACE_LOCKED = (198, 199, 198)
DRAWER_BORDER = (166, 162, 152)
LOCK_BODY = (78, 82, 88)

DRAWERS = ["A", "B", "C", "D", "E"]


def _bfs(initial: Dict[str, Tuple[int, ...]], target: Dict[str, Tuple[int, ...]], unlocked: Set[str], max_moves: int) -> Optional[List[Tuple[str, str]]]:
    keys = sorted(initial.keys())
    start = tuple(initial[k] for k in keys)
    end = tuple(target[k] for k in keys)
    if start == end:
        return []
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        state, path = queue.popleft()
        if len(path) >= max_moves:
            continue
        for i, ki in enumerate(keys):
            if ki not in unlocked:
                continue
            if not state[i]:
                continue
            top = state[i][-1]
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                if kj not in unlocked:
                    continue
                new_i = state[i][:-1]
                new_j = state[j] + (top,)
                new_state = state[:i] + (new_i,) + state[i + 1:j] + (new_j,) + state[j + 1:] if i < j else\
                            state[:j] + (new_j,) + state[j + 1:i] + (new_i,) + state[i + 1:]
                if new_state in seen:
                    continue
                seen.add(new_state)
                np = path + [(ki, kj)]
                if new_state == end:
                    return np
                queue.append((new_state, np))
    return None


class FilingCabinetAlphabetLockGenerator(RearrangementObjectPlanGenerator):
    TASK_NAME = "filing_cabinet_alphabet_lock"
    TASK_FAMILY = "rearrangement_game"
    DEFAULT_OUTPUT_DIR = "outputs/rearrangement_game/filing_cabinet_alphabet_lock"
    SCENE_OUTER_MARGIN = 8
    SCENE_CHEVRON_BAND = 26
    SCENE_CHEVRON_SCALE = 0.72

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        seed: Optional[int] = None,
        video: bool = False,
        max_optimal_moves: int = 7,
    ) -> None:
        super().__init__(output_dir=output_dir, seed=seed, video=video)
        self.max_optimal_moves = int(max_optimal_moves)

    def _sample_problem(self) -> Tuple[Any, Any, List[Any]]:
        for _ in range(2000):
            # Choose 1-2 locked drawers (the locked ones contain no papers).
            n_locked = self.rng.randint(1, 2)
            locked = set(self.rng.sample(DRAWERS, n_locked))
            unlocked = set(DRAWERS) - locked
            n_papers = self.rng.randint(3, 4)
            paper_ids = list(range(1, n_papers + 1))
            self.rng.shuffle(paper_ids)
            # Distribute papers into unlocked drawers; locked drawers stay empty.
            initial = {d: [] for d in DRAWERS}
            for p in paper_ids:
                pick = self.rng.choice(sorted(unlocked))
                initial[pick].append(p)
            target = {d: [] for d in DRAWERS}
            target_ids = list(paper_ids)
            self.rng.shuffle(target_ids)
            # Distribute into unlocked drawers in DIFFERENT layout.
            for p in target_ids:
                pick = self.rng.choice(sorted(unlocked))
                target[pick].append(p)
            initial_t = {d: tuple(initial[d]) for d in DRAWERS}
            target_t = {d: tuple(target[d]) for d in DRAWERS}
            if initial_t == target_t:
                continue
            plan = _bfs(initial_t, target_t, unlocked, self.max_optimal_moves)
            if plan is None:
                continue
            # Check mechanic: with all drawers unlocked, optimal must be SHORTER.
            plan_unlocked = _bfs(initial_t, target_t, set(DRAWERS), self.max_optimal_moves)
            if plan_unlocked is None or len(plan_unlocked) >= len(plan):
                continue
            if 3 <= len(plan) <= self.max_optimal_moves:
                initial_state = {"drawers": {d: list(initial[d]) for d in DRAWERS}, "locked": sorted(locked)}
                target_state = {"drawers": {d: list(target[d]) for d in DRAWERS}, "locked": sorted(locked)}
                return initial_state, target_state, plan
        raise RuntimeError(
            "filing_cabinet_alphabet_lock rejection sampling exhausted its attempt budget; "
            "the admitted parameter ranges should make this unreachable"
        )

    def _state_signature(self, state):
        return tuple((d, tuple(state["drawers"][d])) for d in DRAWERS)

    def _serialize_state(self, state):
        return {
            "drawers": {d: list(state["drawers"][d]) for d in DRAWERS},
            "locked": list(state["locked"]),
        }

    def _apply_move(self, state, move):
        src, dst = move
        if src not in DRAWERS or dst not in DRAWERS or src == dst:
            raise ValueError(f"invalid drawer move: {move}")
        if src in state["locked"] or dst in state["locked"]:
            raise ValueError(f"locked drawer cannot be moved through: {move}")
        new_drawers = {d: list(state["drawers"][d]) for d in DRAWERS}
        if not new_drawers[src]:
            raise ValueError(f"source drawer {src} empty")
        top = new_drawers[src].pop()
        new_drawers[dst].append(top)
        return {"drawers": new_drawers, "locked": list(state["locked"])}

    def _paper_color(self, k: int) -> Tuple[int, int, int]:
        return ITEM_PALETTE[(k - 1) % len(ITEM_PALETTE)]

    def _layout(self, panel_w, panel_h):
        margin = max(8, int(min(panel_w, panel_h) * 0.05))
        cab_x0 = margin + int(panel_w * 0.05)
        cab_x1 = panel_w - margin - int(panel_w * 0.05)
        cab_y0 = margin
        cab_y1 = panel_h - margin
        n = len(DRAWERS)
        drawer_h = (cab_y1 - cab_y0) // n
        drawer_h -= drawer_h % 2
        return {
            "panel_w": panel_w, "panel_h": panel_h,
            "margin": margin,
            "cab_x0": cab_x0, "cab_x1": cab_x1,
            "cab_y0": cab_y0, "cab_y1": cab_y1,
            "drawer_h": drawer_h,
        }

    def _drawer_bbox(self, layout, idx: int) -> Tuple[int, int, int, int]:
        x0 = layout["cab_x0"]
        x1 = layout["cab_x1"]
        y0 = layout["cab_y0"] + idx * layout["drawer_h"]
        y1 = y0 + layout["drawer_h"]
        return (x0, y0, x1, y1)

    def _draw_lock(self, draw, cx: float, cy: float, size: float):
        half = size * 0.34
        draw.rounded_rectangle((cx - half, cy - half, cx + half, cy + half), radius=max(3, int(size * 0.10)), fill=LOCK_BODY)
        draw.line((cx - half * 0.45, cy - half * 0.45, cx + half * 0.45, cy + half * 0.45), fill=(224, 226, 228), width=2)
        draw.line((cx - half * 0.45, cy + half * 0.45, cx + half * 0.45, cy - half * 0.45), fill=(224, 226, 228), width=2)

    def _draw_paper(self, draw, bbox, k: int):
        x0, y0, x1, y1 = bbox
        color = self._paper_color(k)
        outline = _shade(color, 0.55)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(3, int((x1 - x0) * 0.12)), fill=color, outline=outline, width=2)
        font = self._draw_font(draw, max(9, int((y1 - y0) * 0.38)))
        text_color = (32, 30, 26) if sum(color) > 380 else (250, 248, 244)
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2 - 1), str(k), font=font, fill=text_color, anchor="mm")

    def _paper_geometry(self, layout, idx: int, count: int):
        x0, y0, x1, y1 = self._drawer_bbox(layout, idx)
        paper_w = layout["drawer_h"] * 0.29
        paper_h = layout["drawer_h"] * 0.58
        gap = 3.0
        right = x1 - 9
        centers = [
            (right - paper_w / 2 - (count - 1 - k) * (paper_w + gap), (y0 + y1) / 2)
            for k in range(count)
        ]
        return paper_w, paper_h, centers

    def _draw_drawer(self, draw, layout, idx: int, label: str, papers: Sequence[int], locked: bool):
        x0, y0, x1, y1 = self._drawer_bbox(layout, idx)
        face_color = DRAWER_FACE_LOCKED if locked else DRAWER_FACE
        # Drawer frame.
        draw.rectangle((x0, y0, x1, y1), fill=face_color, outline=DRAWER_BORDER, width=2)
        label_cx = x0 + layout["drawer_h"] * 0.34
        plate_cy = (y0 + y1) / 2
        font = self._draw_font(draw, max(12, int(layout["drawer_h"] * 0.28)))
        draw.text((label_cx, plate_cy - 1), label, font=font, fill=(70, 72, 74), anchor="mm")
        if locked:
            self._draw_lock(draw, x0 + layout["drawer_h"] * 0.86, plate_cy, layout["drawer_h"] * 0.42)
        else:
            paper_w, paper_h, centers = self._paper_geometry(layout, idx, len(papers))
            for p, (cx, cy) in zip(papers, centers):
                self._draw_paper(draw, (cx - paper_w / 2, cy - paper_h / 2, cx + paper_w / 2, cy + paper_h / 2), int(p))

    def _render_panel(self, state, panel_w, panel_h, *, moving=None):
        layout = self._layout(panel_w, panel_h)
        image, draw = self._panel_canvas(panel_w, panel_h, PANEL_BG)
        draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=PANEL_BORDER, width=2)
        # Cabinet frame.
        draw.rectangle((layout["cab_x0"] - 4, layout["cab_y0"] - 4, layout["cab_x1"] + 4, layout["cab_y1"] + 4), fill=CABINET_FRAME)
        draw.rectangle((layout["cab_x0"], layout["cab_y0"], layout["cab_x1"], layout["cab_y1"]), fill=CABINET_FACE)
        locked = set(state["locked"])
        drawers = state["drawers"]
        for idx, label in enumerate(DRAWERS):
            papers = list(drawers[label])
            if moving is not None and moving.get("hide_label") == label:
                if papers:
                    papers = papers[:-1]
            self._draw_drawer(draw, layout, idx, label, papers, label in locked)
        # Moving paper overlay.
        if moving is not None and "paper_pix" in moving and "paper_id" in moving:
            cx, cy = moving["paper_pix"]
            paper_w = layout["drawer_h"] * 0.29
            paper_h = layout["drawer_h"] * 0.58
            self._draw_paper(draw, (cx - paper_w / 2, cy - paper_h / 2, cx + paper_w / 2, cy + paper_h / 2), int(moving["paper_id"]))
        return self._panel_at_scene_scale(image, panel_w, panel_h)

    def _render_state(self, state, panel_width, *, panel_height=None, animation=None):
        ph = int(panel_height) if panel_height is not None else int(panel_width)
        return self._render_panel(state, panel_width, ph, moving=animation)

    def _animate_move(self, state, move, target_state):
        src, dst = move
        src_idx = DRAWERS.index(src)
        dst_idx = DRAWERS.index(dst)
        geom = self._panel_geometry()
        panel_w = geom["panel_w"]
        panel_h = geom["panel_h"]
        layout = self._layout(panel_w, panel_h)
        src_papers = list(state["drawers"][src])
        if not src_papers:
            return []
        paper_id = src_papers[-1]
        _, _, src_centers = self._paper_geometry(layout, src_idx, len(src_papers))
        _, _, dst_centers = self._paper_geometry(layout, dst_idx, len(state["drawers"][dst]) + 1)
        sx_paper, sy_paper = src_centers[-1]
        dx_paper, dy_paper = dst_centers[-1]
        right_image = self._render_panel(target_state, panel_w, panel_h, moving=None)
        frames: List[Image.Image] = []
        after = self._apply_move(state, move)
        for step in range(1, self.move_frames + 1):
            t = step / self.move_frames
            if step == self.move_frames:
                frames.append(self._render_static_scene(after, target_state))
                continue
            cx, cy = self._interpolate_pixel_linear((sx_paper, sy_paper), (dx_paper, dy_paper), t)
            anim = {"hide_label": src, "paper_pix": (cx, cy), "paper_id": paper_id}
            left = self._render_panel(state, panel_w, panel_h, moving=anim)
            frames.append(self._compose_scene(left, right_image))
        return frames


    # ------------------------------------------------------------------
    # Authored reasoning
    # ------------------------------------------------------------------

    AUTHORED_REASONING_HOOKS = True

    @staticmethod
    def _drawer_phrase(papers) -> str:
        if not papers:
            return "empty"
        if len(papers) == 1:
            return f"paper {papers[0]}"
        return "papers " + ", ".join(str(p) for p in papers) + " bottom to top"

    def _cabinet_readout(self, state) -> str:
        drawers = state["drawers"]
        locked = set(state.get("locked", []))
        parts = []
        for letter in DRAWERS:
            if letter in locked:
                parts.append(f"drawer {letter} is locked shut")
            else:
                parts.append(f"drawer {letter} holds {self._drawer_phrase(drawers[letter])}")
        return "; ".join(parts)

    def _author_observation(self, initial_state, target_state, moves) -> str:
        locked = ", ".join(initial_state.get("locked", []))
        return (
            f"Looking at the cabinet, {self._cabinet_readout(initial_state)}. The target layout wants "
            f"{self._cabinet_readout(target_state)}. Only a drawer's top paper can be pulled, nothing may "
            f"enter or leave the locked drawer{'s' if len(initial_state.get('locked', [])) != 1 else ''} "
            f"({locked}), and losing those drawers as workspace is what stretches the shuffle."
        )

    def _author_move_clause(self, move_index, move, previous_state, next_state, moves, target_state) -> str:
        src_letter, dst_letter = str(move[0]), str(move[1])
        paper = previous_state["drawers"][src_letter][-1]
        target_stack = target_state["drawers"][dst_letter]
        landed = next_state["drawers"][dst_letter]
        settled = (
            len(landed) <= len(target_stack)
            and list(target_stack[: len(landed)]) == list(landed)
        )
        uncovered = previous_state["drawers"][src_letter][:-1]
        if settled and target_stack:
            purpose = "which is exactly where the target files it"
        elif uncovered:
            purpose = f"which uncovers paper {uncovered[-1]}"
        else:
            purpose = "parking it out of the way"
        return (
            f"The top paper of drawer {src_letter} is {paper}; it moves onto drawer {dst_letter}, "
            f"{purpose}. Now {self._cabinet_readout(next_state)}."
        )

    def _author_confirmation(self, initial_state, target_state, moves) -> str:
        return (
            f"Every unlocked drawer now matches the target stack and the locked drawers were never "
            f"touched, after {len(moves)} top-paper pulls."
        )

    def _describe_input(self) -> str:
        return (
            "The input puzzle image shows two side-by-side filing cabinets. Each cabinet has FIVE drawers stacked "
            "top to bottom, each drawer marked only by a letter A, B, C, D, or E. Locked drawers use a flat dark "
            "square with a pale X. Each unlocked drawer holds a row of fixed-size numbered color tokens stacked "
            "LIFO (rightmost = top of stack). The left cabinet shows the starting "
            "layout; the right cabinet shows the desired layout. Padlocks appear in the SAME drawers in both "
            "cabinets (the lock pattern is fixed). A blue chevron arrow between the cabinets points from left to "
            "right. Brass plates, folded page corners, color bands, ledges, and detailed padlocks are omitted."
        )

    def _describe_rules(self) -> str:
        return (
            "Each move pulls the TOP paper (rightmost paper) out of one UNLOCKED source drawer and places it on "
            "top of one UNLOCKED destination drawer. A locked drawer cannot be opened — papers cannot enter or "
            "leave a locked drawer. The drawers themselves do not move; only papers do."
        )

    def _describe_output(self) -> str:
        return (
            "Return the shortest plan as `Answer: [(<source_drawer>, <destination_drawer>), ...].`, using quoted "
            "uppercase drawer labels."
        )

    def _format_move(self, move):
        return f"('{move[0]}', '{move[1]}')"

    def _extra_payload(self, initial_state, target_state, moves):
        return {
            "drawers": list(DRAWERS),
            "locked": list(initial_state["locked"]),
            "optimal_moves": [(str(m[0]), str(m[1])) for m in moves],
        }


__all__ = ["FilingCabinetAlphabetLockGenerator"]
