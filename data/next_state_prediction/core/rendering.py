from __future__ import annotations

from data.common.assets import render_font

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


from PIL import Image, ImageDraw, ImageFont

from data.common.render_primitives import (
    DOCTRINE_SUPERSAMPLE,
    ScaledDraw,
    finish_scaled_canvas,
)

from .semantics import DIRS, VELOCITY_BITS, visual_matrix


RGB = Tuple[int, int, int]
Matrix = List[List[str]]

SIZE = 512
FACTOR = DOCTRINE_SUPERSAMPLE

BACKGROUND: RGB = (239, 244, 248)
CARD: RGB = (252, 253, 254)
INK: RGB = (30, 42, 55)
MUTED: RGB = (91, 108, 124)
LINE: RGB = (190, 203, 214)

_ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

# Binary bit colours shared by every task whose cell state is exactly one bit.
_BIT_FILLS: Tuple[RGB, RGB] = ((245, 248, 250), (48, 79, 105))


@dataclass(frozen=True)
class VisualObject:
    row: float
    col: float
    label: str
    kind: str = "token"
    detail: str = ""


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    return render_font(size * FACTOR, bold=bold)


def _rgba(colour: RGB, opacity: float) -> Tuple[int, int, int, int]:
    amount = max(0.0, min(1.0, float(opacity)))
    return (*colour, int(round(255 * amount)))


def _text_size(draw: ScaledDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bounds = draw.raw.textbbox((0, 0), text, font=font)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def _center_text(
    draw: ScaledDraw,
    bounds: Sequence[float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB | Tuple[int, int, int, int] = INK,
) -> None:
    left, top, right, bottom = bounds
    # Subtract the glyph bbox origin so tall glyphs (arrows) center on their
    # visible ink instead of drifting toward the bottom of the badge.
    x0, y0, x1, y1 = draw.raw.textbbox((0, 0), text, font=font)
    width, height = x1 - x0, y1 - y0
    draw.raw.text(
        (
            (left + right) / 2 * FACTOR - width / 2 - x0,
            (top + bottom) / 2 * FACTOR - height / 2 - y0,
        ),
        text,
        font=font,
        fill=fill,
    )


def _draw_tracked_text(
    draw: ScaledDraw,
    bounds: Sequence[float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB | Tuple[int, int, int, int],
    tracking: float,
) -> float:
    """Centre ``text`` in ``bounds`` with ``tracking`` 512-space px per gap.

    PIL has no letter-spacing, so a compact numeric detail such as ``6/6`` can
    leave adjacent glyphs one pixel apart at 512.  Advancing each glyph by hand
    keeps a measurable gap between them.  Returns the achieved gap in 512-space
    pixels so callers can assert on it.
    """
    left, top, right, bottom = bounds
    step = tracking * FACTOR
    widths = [draw.raw.textlength(character, font=font) for character in text]
    total = sum(widths) + step * max(0, len(text) - 1)
    box = draw.raw.textbbox((0, 0), text, font=font)
    x = (left + right) / 2 * FACTOR - total / 2
    y = (top + bottom) / 2 * FACTOR - (box[3] - box[1]) / 2 - box[1]
    for character, width in zip(text, widths):
        draw.raw.text((x, y), character, font=font, fill=fill)
        x += width + step
    return tracking


def _fit_font(draw: ScaledDraw, text: str, max_width: float, preferred: int, minimum: int = 7, bold: bool = False) -> ImageFont.ImageFont:
    for size in range(preferred, minimum - 1, -1):
        candidate = _font(size, bold)
        if _text_size(draw, text, candidate)[0] <= max_width * FACTOR:
            return candidate
    return _font(minimum, bold)


def _operation_text(operation: Mapping[str, Any]) -> str:
    kind = operation["kind"]
    if kind in ("ADD", "MUL", "XOR", "MOD"):
        symbol = {"ADD": "+", "MUL": "×", "XOR": "xor", "MOD": "mod"}[kind]
        return f"{symbol} {operation['value']}"
    if kind == "CLAMP":
        return f"clip {operation['low']}..{operation['high']}"
    return "negate"


def _palette(task: str, value: str) -> Tuple[RGB, RGB]:
    numeric: int | None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    palettes: Dict[str, Sequence[RGB]] = {
        "rule_table_1d_rollout": _BIT_FILLS,
        "lifelike_rule_rollout": _BIT_FILLS,
        "xor_parity_lattice": _BIT_FILLS,
        "turn_write_agent_rollout": _BIT_FILLS,
        "finite_tape_machine_rollout": _BIT_FILLS,
        "cyclic_threshold_wave": ((81, 145, 184), (239, 130, 91), (100, 176, 139), (164, 122, 181), (230, 187, 80)),
        "excitable_signal_network": ((246, 248, 250), (232, 195, 81), (67, 144, 210), (210, 91, 96), (64, 74, 84)),
        "majority_consensus": ((224, 230, 235), (60, 131, 190), (221, 105, 91)),
        "forest_fire_barrier": ((246, 248, 250), (75, 156, 104), (232, 102, 61), (146, 116, 91), (65, 142, 197), (74, 82, 91)),
        "predator_prey_cycle": ((246, 248, 250), (103, 181, 106), (216, 86, 88), (110, 105, 200), (70, 79, 88)),
    }
    if numeric is not None and task in palettes:
        palette = palettes[task]
        fill = palette[numeric % len(palette)]
        luminance = sum(fill) / 3
        return fill, (255, 255, 255) if luminance < 145 else INK
    if value in ("", "—"):
        return (247, 249, 251), MUTED
    if value in ("#", "WALL"):
        return (64, 76, 86), (255, 255, 255)
    if value == "GREEN":
        return (89, 177, 118), (255, 255, 255)
    if value == "RED":
        return (210, 91, 91), (255, 255, 255)
    return (244, 247, 249), INK


# Minimal state encoding: each of these tasks encodes its
# complete per-cell state in the cell fill colour alone, so the renderer must
# not additionally print the state code as text inside the cell.  The Prompt
# legend carries the colour-to-code mapping.  Tasks absent from this set print
# their cell strings because those strings are values (heights, pile counts,
# machine labels), not a second encoding of a colour-coded state.
_COLOR_ONLY_CELL_TASKS = frozenset({
    "rule_table_1d_rollout",
    "lifelike_rule_rollout",
    "cyclic_threshold_wave",
    "excitable_signal_network",
    "xor_parity_lattice",
    "majority_consensus",
    "forest_fire_barrier",
    "predator_prey_cycle",
    "turn_write_agent_rollout",
    "finite_tape_machine_rollout",
    "lattice_gas_collision",
    "diffusion_limited_aggregation",
})


def _empty_matrix(rows: int, cols: int) -> Matrix:
    return [["" for _ in range(cols)] for _ in range(rows)]


def _base_and_objects(task: str, state: Mapping[str, Any], params: Mapping[str, Any]) -> Tuple[Matrix, Dict[str, VisualObject]]:
    objects: Dict[str, VisualObject] = {}
    if task == "alternating_block_rotation":
        grid = state["grid"]
        base = _empty_matrix(len(grid), len(grid[0]))
        for row, values in enumerate(grid):
            for col, token in enumerate(values):
                if token >= 0:
                    objects[f"T{token}"] = VisualObject(row, col, str(token), "token")
        return base, objects
    if task == "lattice_gas_collision":
        grid = state["grid"]
        base = _empty_matrix(len(grid), len(grid[0]))
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if params["walls"][row][col]:
                    base[row][col] = "WALL"
                for direction, bit in VELOCITY_BITS.items():
                    if grid[row][col] & bit:
                        objects[f"P{row}:{col}:{direction}"] = _particle_object(row, col, direction)
        return base, objects
    if task == "turn_write_agent_rollout":
        base = [[str(value) for value in row] for row in state["grid"]]
        row, col = state["head"]
        objects["HEAD"] = VisualObject(
            row, col, _ARROWS[state["direction"]], "agent", f"bit:{state['grid'][row][col]}",
        )
        return base, objects
    if task == "finite_tape_machine_rollout":
        base = [[str(value) for value in state["tape"]]]
        objects["HEAD"] = VisualObject(
            0, state["head"], str(state["control"]), "head", f"bit:{state['tape'][state['head']]}",
        )
        return base, objects
    if task == "simultaneous_agent_conflict":
        size = params["size"]
        base = _empty_matrix(size, size)
        for agent in state["agents"]:
            objects[agent["id"]] = VisualObject(agent["row"], agent["col"], f"{agent['id']}{_ARROWS[agent['direction']]}", "agent")
        return base, objects
    if task == "resource_pick_drop_swarm":
        base = [[str(value) if value else "" for value in row] for row in state["piles"]]
        for agent in state["agents"]:
            carrying = "●" if agent["carrying"] else "○"
            pile = state["piles"][agent["row"]][agent["col"]]
            objects[agent["id"]] = VisualObject(
                agent["row"], agent["col"], f"{agent['id']}{_ARROWS[agent['direction']]}", "agent", f"{carrying}·p{pile}",
            )
        return base, objects
    if task == "diffusion_limited_aggregation":
        size = params["size"]
        base = _empty_matrix(size, size)
        for row, col in state["aggregate"]:
            base[row][col] = "#"
        for walker in state["walkers"]:
            objects[walker["id"]] = VisualObject(walker["row"], walker["col"], f"{walker['id']}{_ARROWS[walker['direction']]}", "walker")
        return base, objects
    if task == "chained_function_pipeline":
        operations = params["operations"]
        base = [["IN"] + [_operation_text(item) for item in operations] + ["OUT"]]
        col = len(operations) + 1 if state["index"] >= len(operations) else state["index"]
        objects["PACKET"] = VisualObject(0, col, str(state["value"]), "packet")
        return base, objects
    if task == "packet_router_tick":
        # Drawn queue and port names are 1-based ordinals; the internal row,
        # column, and state indices below stay 0-based, and ``t`` is a TTL count.
        base = _empty_matrix(5, 5)
        for row in range(3):
            base[row][0] = f"I{row + 1}{'•' if row == state['pointer'] else ''}"
        for output in range(3):
            base[4][output + 1] = f"O{output + 1}"
        for input_index, queue in enumerate(state["inputs"]):
            for slot, packet in enumerate(queue[:4]):
                objects[packet["id"]] = VisualObject(input_index, slot + 1, f"{packet['id']}", "packet", f"O{packet['dest'] + 1}·t{packet['ttl']}")
        if state.get("last"):
            for transfer in state["last"]["transfers"]:
                packet = transfer["packet"]
                objects[packet["id"]] = VisualObject(4, transfer["output"] + 1, packet["id"], "packet", f"O{transfer['output'] + 1}·t{packet['ttl']}")
        return base, objects
    if task == "elevator_dispatch_tick":
        floors = params["floors"]
        base = _empty_matrix(floors, 4)
        for floor in range(floors):
            row = floors - 1 - floor
            base[row][0] = str(floor)
            if floor in state["calls"]:
                base[row][3] = "CALL"
        for index, car in enumerate(sorted(state["cars"], key=lambda item: item["id"])):
            row = floors - 1 - car["floor"]
            detail = f"{car['load']}/{car['capacity']}·{car['door'][0]}"
            objects[car["id"]] = VisualObject(row, index + 1, car["id"], "elevator", detail)
        return base, objects
    if task == "protocol_handshake_tick":
        base = _empty_matrix(4, 6)
        base[0] = ["CLIENT", state["client"], "", "SERVER", state["server"], ""]
        # Tick is part of the typed terminal state and determines which of two
        # otherwise identical idle states is terminal.  Keep it visible in
        # every stable render so source replay can distinguish those commits.
        base[1] = ["TIMEOUT", str(state["timeout"]), "", "TICK", str(state["tick"]), ""]
        base[2][0] = "QUEUE"
        for index, message in enumerate(state["queue"][:5]):
            identity = message.get("id", f"Q{index}:{message['kind']}")
            objects[identity] = VisualObject(
                2, index + 1, message["kind"], "message",
                f"{identity}→{message['to'][0].upper()}",
            )
        base[3] = ["OLDEST", "→", "", "", "", "NEWEST"]
        return base, objects
    return visual_matrix(task, state, params), objects


def _particle_object(row: int, col: int, direction: str) -> VisualObject:
    # Offset each channel toward its direction so a multi-channel cell shows
    # every particle separately; 0.26 keeps head-on pairs disjoint.
    dr, dc = DIRS[direction]
    return VisualObject(row + dr * 0.26, col + dc * 0.26, _ARROWS[direction], "particle")


# These tasks name grid positions in their Prompt, typed answer, or authored
# reasoning, so the render prints 1-based row/column ordinals along the top and
# left grid edges.  The numbers are static input legend: they
# are identical in every state, so no transition may move, restyle, or hide
# them.
_EDGE_NUMBERED_TASKS = frozenset({
    "lifelike_rule_rollout",
    "cyclic_threshold_wave",
    "excitable_signal_network",
    "alternating_block_rotation",
    "xor_parity_lattice",
    "sandpile_avalanche",
    "lattice_gas_collision",
    "majority_consensus",
    "forest_fire_barrier",
    "predator_prey_cycle",
    "turn_write_agent_rollout",
    "simultaneous_agent_conflict",
    "resource_pick_drop_swarm",
    "diffusion_limited_aggregation",
})

# One-row scenes whose answers or authored reasoning name slot positions print
# 1-based position ordinals along the top edge only.
_ROW_NUMBERED_TASKS = frozenset({
    "rule_table_1d_rollout",
    "finite_tape_machine_rollout",
})


def _draw_edge_numbers(
    draw: ScaledDraw,
    rows: int,
    cols: int,
    layout: Tuple[float, float, float, float],
    *,
    columns_only: bool = False,
) -> None:
    """Print compact 1-based row/column ordinals along the left and top edges."""
    x0, y0, cell_w, cell_h = layout
    font = _font(11)
    if not columns_only:
        for row in range(rows):
            _center_text(
                draw,
                (x0 - 26, y0 + row * cell_h, x0 - 8, y0 + (row + 1) * cell_h),
                str(row + 1),
                font,
                MUTED,
            )
    for col in range(cols):
        _center_text(
            draw,
            (x0 + col * cell_w, y0 - 26, x0 + (col + 1) * cell_w, y0 - 8),
            str(col + 1),
            font,
            MUTED,
        )


def _layout(rows: int, cols: int, *, numbered_edges: bool = False, numbered_columns: bool = False) -> Tuple[float, float, float, float]:
    top = 42.0
    bottom = 476.0
    left, right = 42.0, 470.0
    if numbered_edges or numbered_columns:
        # Reserve the printed 1-based number band outside the grid.
        top += 18
        if numbered_edges:
            left += 18
    available_width = right - left
    available_height = bottom - top
    cell_cap = 92 if max(rows, cols) <= 6 else 62
    cell = min(available_width / max(cols, 1), available_height / max(rows, 1), cell_cap)
    grid_width, grid_height = cell * cols, cell * rows
    x = left + (available_width - grid_width) / 2
    y = top + (available_height - grid_height) / 2
    return x, y, cell, cell


def rendered_glyph_inventory(
    task: str,
    state: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Return every string dynamically drawn by a completed state render.

    Header/status/rule prose is intentionally absent.  The categories are a
    bounded per-task schema: cell values come from the task's current visual
    matrix minus the colour-only suppression set, badge strings come from
    stable semantic objects, and edge-numbered tasks additionally declare the
    printed 1-based ordinals.
    """
    base, objects = _base_and_objects(task, state, params)
    if task in _COLOR_ONLY_CELL_TASKS:
        cell_values: List[str] = []
    else:
        cell_values = sorted({str(value) for row in base for value in row if str(value)})
    object_labels = sorted({obj.label for obj in objects.values() if obj.label})
    object_details = sorted({
        obj.detail for obj in objects.values()
        if obj.detail and not obj.detail.startswith("bit:")
    })
    inventory = {
        "cell_values": cell_values,
        "object_labels": object_labels,
        "object_details": object_details,
    }
    if task in _EDGE_NUMBERED_TASKS or task in _ROW_NUMBERED_TASKS:
        rows = len(base)
        cols = len(base[0]) if base else 0
        limit = cols if task in _ROW_NUMBERED_TASKS else max(rows, cols)
        inventory["edge_labels"] = [str(ordinal) for ordinal in range(1, limit + 1)]
    return inventory


def _draw_cell(
    draw: ScaledDraw,
    task: str,
    value: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    inset = max(1.5, min(width, height) * 0.045)
    left = x + inset
    right = x + width - inset
    top, bottom = y + inset, y + height - inset
    fill, text_fill = _palette(task, value)
    radius = min(10, width * 0.13, height * 0.13)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("disc_lattice") and task=="lifelike_rule_rollout":
        cx=x+width/2;cy=y+height/2;rw=width*.36;rh=height*.38
        draw.line((x,cy,x+width,cy),fill=(140,155,162),width=1)
        draw.line((cx,y,cx,y+height),fill=(140,155,162),width=1)
        draw.polygon([(cx-rw,cy-rh),(cx+rw,cy-rh),(cx+rw*.55,cy),(cx+rw,cy+rh),(cx-rw,cy+rh),(cx-rw*.55,cy)],fill=fill,outline=(67,83,96),width=2)
        if value=="1":
            draw.line((cx-rw*.4,cy+rh*.5,cx+rw*.4,cy-rh*.5),fill=(243,224,155),width=3)
    else:
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=radius,
            fill=fill,
            outline=LINE,
            width=1,
        )
    if value and task not in _COLOR_ONLY_CELL_TASKS and min(width, height) >= 17:
        preferred = max(8, min(17, int(min(width, height) * 0.30)))
        font = _fit_font(draw, value, right - left - 4, preferred, 6, value in ("GREEN", "RED"))
        _center_text(draw, (left, top, right, bottom), value, font, text_fill)


def _draw_base_matrix(
    draw: ScaledDraw,
    task: str,
    base: Matrix,
    layout: Tuple[float, float, float, float],
) -> None:
    x0, y0, cell_w, cell_h = layout
    rows, cols = len(base), len(base[0]) if base else 0
    from data.common.prompt_paraphrase import active_profile
    if not (active_profile("disc_lattice") or active_profile("register_chips")):
        draw.rounded_rectangle(
            (x0 - 6, y0 - 6, x0 + cols * cell_w + 6, y0 + rows * cell_h + 6),
            radius=14, fill=(235, 240, 244), outline=(205, 216, 225), width=1,
        )
    for row in range(rows):
        for col in range(cols):
            _draw_cell(draw, task, base[row][col], x0 + col * cell_w, y0 + row * cell_h, cell_w, cell_h)


def _draw_alternating_block_origins(
    draw: ScaledDraw,
    state: Mapping[str, Any],
    layout: Tuple[float, float, float, float],
) -> None:
    """Mark active 2x2 block origins without a PHASE/STEP status row.

    The marks are semantic: together they expose the current checkerboard
    partition, including the phase-1 origins whose blocks wrap at an edge.
    They carry no task name, prose, or progress counter.
    """
    size = len(state["grid"])
    phase = int(state["phase"])
    x0, y0, cell_w, cell_h = layout
    radius = max(2.5, min(cell_w, cell_h) * 0.075)
    inset = max(4.0, min(cell_w, cell_h) * 0.11)
    for row in range(phase, size, 2):
        for col in range(phase, size, 2):
            cx = x0 + col * cell_w + inset
            cy = y0 + row * cell_h + inset
            draw.polygon(
                [
                    (cx, cy - radius),
                    (cx + radius, cy),
                    (cx, cy + radius),
                    (cx - radius, cy),
                ],
                fill=(45, 116, 170),
            )


def _object_colour(kind: str) -> RGB:
    return {
        "particle": (45, 116, 170),
        "agent": (245, 167, 67),
        "head": (224, 108, 85),
        "walker": (132, 90, 177),
        "packet": (54, 152, 133),
        "message": (88, 125, 190),
        "elevator": (231, 139, 62),
        "token": (68, 121, 177),
    }.get(kind, (65, 126, 178))


def _draw_object(
    draw: ScaledDraw,
    obj: VisualObject,
    layout: Tuple[float, float, float, float],
) -> None:
    x0, y0, cell_w, cell_h = layout
    center_x = x0 + (obj.col + 0.5) * cell_w
    center_y = y0 + (obj.row + 0.5) * cell_h
    radius_factor = 0.24 if obj.kind == "particle" else (0.30 if obj.kind == "token" else 0.34)
    radius = min(cell_w, cell_h) * radius_factor
    if obj.detail:
        radius = min(cell_w, cell_h) * 0.40
    colour = _object_colour(obj.kind)
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=colour,
        outline=(255, 255, 255),
        width=max(1, radius * 0.09),
    )
    bit_detail = obj.detail.startswith("bit:") if obj.detail else False
    # A single-glyph label (arrow or control letter) fills most of its badge
    # so direction and control state stay readable at particle scale.
    label_scale = 1.3 if len(obj.label) == 1 else 0.64
    font = _fit_font(draw, obj.label, radius * 1.8, max(8, int(radius * label_scale)), 6, True)
    label_bounds = (
        center_x - radius,
        center_y - radius * (0.98 if obj.detail else 1),
        center_x + radius,
        center_y + radius * (0.06 if obj.detail else 1),
    )
    _center_text(draw, label_bounds, obj.label, font, (255, 255, 255))
    if not obj.detail or radius <= 7:
        return
    if bit_detail:
        # Joint occupancy: the head badge covers its cell, so
        # a small chip repeats the covered bit with the same two cell colours.
        bit = int(obj.detail.split(":", 1)[1])
        chip = radius * 0.30
        chip_cy = center_y + radius * 0.48
        draw.rounded_rectangle(
            (center_x - chip, chip_cy - chip * 0.72, center_x + chip, chip_cy + chip * 0.72),
            radius=chip * 0.3,
            fill=_BIT_FILLS[bit],
            outline=(255, 255, 255),
            width=1,
        )
        return
    # A load, capacity, TTL or endpoint printed inside the badge fill sat at
    # 0.36 * radius in white on the object colour: 4.2px digits at 2.57:1, no
    # pixel reaching solid white, and adjacent digits touching.  A quantity a
    # reader cannot resolve is not a perceptual margin, so the
    # detail hangs on its own opaque ink plate below the label.  Being wider
    # than the circle's chord at that height is what buys the glyph room, but
    # the plate is clamped to the drawn cell box -- _draw_cell insets each cell
    # by the same fraction, and a plate crossing that border would sit in a
    # neighbouring cell's region.
    cell_inset = max(1.5, min(cell_w, cell_h) * 0.045)
    plate_half_w = max(radius * 0.75, min(radius * 1.14, cell_w / 2 - cell_inset - 1.5))
    plate_cy = center_y + radius * 0.50
    plate_half_h = radius * 0.38
    plate = (
        center_x - plate_half_w,
        plate_cy - plate_half_h,
        center_x + plate_half_w,
        plate_cy + plate_half_h,
    )
    draw.rounded_rectangle(
        plate,
        radius=plate_half_h * 0.5,
        fill=INK,
        outline=(255, 255, 255),
        width=max(1, radius * 0.07),
    )
    # Track the glyphs apart so a compact numeric detail such as 6/6 keeps a
    # measurable gap; the fit allows for the tracking it is about to add.
    tracking = 1.5
    budget = plate_half_w * 1.86 - tracking * max(0, len(obj.detail) - 1)
    detail_font = _fit_font(draw, obj.detail, budget, max(9, int(radius * 0.66)), 8, True)
    _draw_tracked_text(draw, plate, obj.detail, detail_font, (255, 255, 255), tracking)


def new_state_canvas(
    background: RGB = BACKGROUND,
    card_fill: RGB = CARD,
    card_outline: RGB = (211, 221, 230),
    card_box: Tuple[float, float, float, float] = (18, 18, 494, 494),
) -> Tuple[Image.Image, ScaledDraw]:
    """Allocate the family's supersampled 512-target canvas with its card frame.

    Task renderers draw in unscaled 512-space through the returned
    :class:`ScaledDraw` proxy and finish with :func:`finish_state_canvas`.
    """
    canvas = Image.new("RGB", (SIZE * FACTOR, SIZE * FACTOR), background)
    draw = ScaledDraw(ImageDraw.Draw(canvas, "RGBA"), FACTOR)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("distance_beacons"):
        draw.rectangle((0,0,512,512),fill=(230,216,181))
        draw.rectangle(card_box,fill=(247,235,201),outline=(114,96,63),width=3)
    else:
        draw.rounded_rectangle(card_box, radius=22, fill=card_fill, outline=card_outline, width=2)
    return canvas, draw


def finish_state_canvas(canvas: Image.Image) -> Image.Image:
    """LANCZOS-downsample a family canvas back to the 512x512 media contract."""
    return finish_scaled_canvas(canvas, (SIZE, SIZE))


def scaled_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a font at the supersampled size matching :func:`new_state_canvas`."""
    return _font(size, bold)


def discrete_video_frames(states: Sequence[Image.Image]) -> List[Image.Image]:
    """Assemble the family's 4N+1 discrete video from ordered state renders.

    ``states[0]`` is the input; every later entry is one committed successor.
    Frames hold each state and switch directly: no
    interpolated position, opacity blend, or synthesized in-between frame.
    """
    if len(states) < 2:
        raise ValueError("a discrete transition video needs at least two states")
    frames: List[Image.Image] = [states[0].copy() for _ in range(_INPUT_HOLD_FRAMES)]
    for successor in states[1:]:
        frames.extend(successor.copy() for _ in range(_STATE_HOLD_FRAMES))
    frames.extend(states[-1].copy() for _ in range(_TERMINAL_EXTRA_FRAMES))
    if (len(frames) - 1) % 4 != 0:
        raise AssertionError(f"video frame count {len(frames)} is not 4N+1")
    # Reference Length grows with the number of committed states; do not truncate
    # deeper rollouts to the training-video frame budget.
    return frames


def gui_control_bboxes(
    state: Mapping[str, Any],
    params: Mapping[str, Any],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return source-derived control and panel bounds for the GUI renderer.

    The keys name semantic controls rather than sampled frame coordinates.  The
    verifier uses the same bounds to inspect the complete interface panel, but
    independently reconstructs the expected action sequence from ``params`` and
    the initial state.
    """
    matrix, _objects = _base_and_objects("gui_hierarchical_navigation", state, params)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("GUI navigation visual matrix must be exactly 3x3")
    x0, y0, cell_w, cell_h = _layout(3, 3)

    def cell(row: int, col: int) -> Tuple[float, float, float, float]:
        return (
            x0 + col * cell_w,
            y0 + row * cell_h,
            x0 + (col + 1) * cell_w,
            y0 + (row + 1) * cell_h,
        )

    bounds: Dict[str, Tuple[float, float, float, float]] = {
        "PANEL": (x0 - 8, y0 - 8, x0 + 3 * cell_w + 8, y0 + 3 * cell_h + 8),
        "BACK": cell(0, 0),
    }
    for col, tile in enumerate(params["pages"][state["page"]][:3]):
        bounds[f"CLICK:{tile['label']}"] = cell(1, col)
    return bounds


def _draw_gui_action_press(
    draw: ScaledDraw,
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    params: Mapping[str, Any],
    progress: float,
) -> None:
    """Draw a local press ring while leaving all old-page content visible."""
    event = new_state.get("last")
    if not isinstance(event, Mapping) or not isinstance(event.get("action"), Mapping):
        raise ValueError("GUI successor lacks its source-replayed action event")
    action = event["action"]
    key = "BACK" if action.get("kind") == "BACK" else f"CLICK:{action.get('label')}"
    bounds = gui_control_bboxes(old_state, params)
    if key not in bounds:
        raise ValueError(f"GUI action control {key!r} is not visible on page {old_state['page']!r}")

    amount = max(0.0, min(1.0, float(progress)))
    opacity = amount ** 0.5
    left, top, right, bottom = bounds[key]
    inset = 4.0
    colour = (24, 158, 158)
    draw.rounded_rectangle(
        (left + inset, top + inset, right - inset, bottom - inset),
        radius=8,
        outline=_rgba(colour, opacity),
        width=5,
    )
    # The small contact dot makes the press direction locally observable while
    # never covering the control label or replacing the page with an overlay.
    dot_radius = 5.0
    dot_x = right - inset - 7.0
    dot_y = top + inset + 7.0
    draw.ellipse(
        (dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius),
        fill=_rgba(colour, opacity),
        outline=(255, 255, 255, int(round(255 * opacity))),
        width=1,
    )


def _render_register_instruments(state):
    canvas,draw=new_state_canvas()
    draw.rectangle((0,0,512,512),fill=(205,223,212))
    positions=[(112,118),(400,118),(256,272),(112,414),(400,414)]
    values=[("Q1",state["q"][0]),("Q2",state["q"][1]),("Q3",state["q"][2]),("X",state["inputs"]["X"]),("Y",state["inputs"]["Y"])]
    for (cx,cy),(name,bit) in zip(positions,values):
        draw.polygon([(cx-87,cy),(cx-54,cy-62),(cx+54,cy-62),(cx+87,cy),(cx+54,cy+62),(cx-54,cy+62)],fill=(248,226,165),outline=(57,85,83),width=4)
        _center_text(draw,(cx-70,cy-54,cx+70,cy-18),name,_font(25,True),(37,63,65))
        draw.rounded_rectangle((cx-46,cy-8,cx+46,cy+42),radius=22,fill=(49,68,75),outline=(96,120,119),width=2)
        for value in (0,1):
            x=cx-23+46*value
            draw.ellipse((x-18,cy-2,x+18,cy+34),fill=(248,231,143) if value==bit else (73,88,94))
            _center_text(draw,(x-17,cy-2,x+17,cy+34),str(value),_font(22,True),(34,59,62) if value==bit else (186,201,194))
    return finish_state_canvas(canvas)


def _render(
    task: str,
    state: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    gui_press: Tuple[Mapping[str, Any], float] | None = None,
) -> Image.Image:
    from data.common.prompt_paraphrase import active_profile
    if task=="clocked_logic_register_tick" and active_profile("register_chips"):
        return _render_register_instruments(state)
    base, objects = _base_and_objects(task, state, params)
    rows, cols = len(base), len(base[0]) if base else 0
    numbered_edges = task in _EDGE_NUMBERED_TASKS
    numbered_columns = task in _ROW_NUMBERED_TASKS
    layout = _layout(rows, cols, numbered_edges=numbered_edges, numbered_columns=numbered_columns)
    canvas = Image.new("RGB", (SIZE * FACTOR, SIZE * FACTOR), BACKGROUND)
    draw = ScaledDraw(ImageDraw.Draw(canvas, "RGBA"), FACTOR)
    from data.common.prompt_paraphrase import active_profile
    if active_profile("disc_lattice") or active_profile("register_chips"):
        draw.rectangle((0,0,512,512),fill=(215,231,217))
        draw.rectangle((16,16,496,496),outline=(76,104,101),width=4)
    else:
        draw.rounded_rectangle((20, 18, 492, 494), radius=24, fill=CARD, outline=(211, 221, 230), width=2)
    _draw_base_matrix(draw, task, base, layout)
    if numbered_edges or numbered_columns:
        _draw_edge_numbers(draw, rows, cols, layout, columns_only=numbered_columns)
    if task == "alternating_block_rotation":
        _draw_alternating_block_origins(draw, state, layout)
    for identity in sorted(objects):
        _draw_object(draw, objects[identity], layout)
    if gui_press is not None:
        successor, progress = gui_press
        _draw_gui_action_press(draw, state, successor, params, progress)
    return finish_scaled_canvas(canvas, (SIZE, SIZE))


def stable_grid_layout(task: str, rows: int, cols: int) -> Tuple[float, float, float, float]:
    """Public 512-space grid layout ``(x0, y0, cell_w, cell_h)`` of a stable state.

    Source replay and the verifier share this geometry instead of duplicating
    the margin constants.
    """
    return _layout(
        rows,
        cols,
        numbered_edges=task in _EDGE_NUMBERED_TASKS,
        numbered_columns=task in _ROW_NUMBERED_TASKS,
    )


def render_state(
    task: str,
    state: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    mode: str = "input",
    total_steps: int | None = None,
) -> Image.Image:
    del mode, total_steps
    return _render(task, state, params)


def render_gui_action_press(
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    progress: float = 1.0,
) -> Image.Image:
    """Render the visible control press immediately before one atomic commit."""
    return _render(
        "gui_hierarchical_navigation",
        old_state,
        params,
        gui_press=(new_state, progress),
    )


# Discrete state switching: every non-GUI next-state world is
# a discrete transition system, so the reference video is a sequence of exact
# complete state renders.  Each committed successor holds for the same
# duration and the switch is direct: no positional interpolation, opacity
# blending, or synthesized sub-tick frame exists between two held states.
_INPUT_HOLD_FRAMES = 9
_STATE_HOLD_FRAMES = 12
_TERMINAL_EXTRA_FRAMES = 8


def render_video_frames(task: str, trace: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> List[Image.Image]:
    """Render 4N+1 discrete frames without interpolation, fades, or hard layout cuts."""
    if len(trace) < 2:
        raise ValueError("a transition video needs at least two semantic states")
    frames: List[Image.Image]
    if task == "gui_hierarchical_navigation":
        # GUI panels use atomic page transitions: the complete old page
        # stays readable while the source action's control press fades in
        # locally, then the page commits atomically and holds.
        frames = [_render(task, trace[0], params) for _ in range(5)]
        for old, new in zip(trace, trace[1:]):
            for progress in (0.25, 0.5, 0.75, 1.0):
                frames.append(render_gui_action_press(old, new, params, progress=progress))
            endpoint = _render(task, new, params)
            frames.extend(endpoint.copy() for _ in range(8))
    else:
        renders = [_render(task, state, params) for state in trace]
        frames = [renders[0].copy() for _ in range(_INPUT_HOLD_FRAMES)]
        for successor in renders[1:]:
            frames.extend(successor.copy() for _ in range(_STATE_HOLD_FRAMES))
        frames.extend(renders[-1].copy() for _ in range(_TERMINAL_EXTRA_FRAMES))
    if (len(frames) - 1) % 4 != 0:
        raise AssertionError(f"video frame count {len(frames)} is not 4N+1")
    # Reference Length grows with the number of committed states; do not truncate
    # deeper rollouts to the training-video frame budget.
    return frames


__all__ = [
    "discrete_video_frames",
    "finish_state_canvas",
    "gui_control_bboxes",
    "new_state_canvas",
    "render_gui_action_press",
    "render_state",
    "render_video_frames",
    "rendered_glyph_inventory",
    "scaled_font",
    "stable_grid_layout",
]
