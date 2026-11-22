from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from data.common.base import AbstractPuzzleGenerator
from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
    cell_to_xy,
    format_xy_position,
)
from data.common.typed_answers import (
    canonical_json_dumps,
)
from data.common.video_encoding import encode_rgb_frames_to_webm
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

from .catalog import EXPANSION_CLASS_NAMES
from .rendering import render_state, render_video_frames
from .semantics import DIRS, answer_payload, build_case


@dataclass(frozen=True)
class TransitionLexicon:
    """One task's time-unit vocabulary across Rule, CoT, and visual delivery."""

    unit: str
    plural: str | None = None

    def count_phrase(self, count: int) -> str:
        return _count_phrase(count, self.unit, self.plural)

    def opening(self, index: int, total: int) -> str:
        if total == 1:
            return f"For the committed {self.unit}"
        if index == 1:
            return f"For the first {self.unit}"
        if index == total:
            return f"For the final {self.unit}"
        return f"For the next {self.unit}"


_TRANSITION_LEXICONS: Dict[str, TransitionLexicon] = {
    "rule_table_1d_rollout": TransitionLexicon("generation"),
    "lifelike_rule_rollout": TransitionLexicon("generation"),
    "cyclic_threshold_wave": TransitionLexicon("generation"),
    "excitable_signal_network": TransitionLexicon("generation"),
    "alternating_block_rotation": TransitionLexicon("generation"),
    "xor_parity_lattice": TransitionLexicon("generation"),
    "sandpile_avalanche": TransitionLexicon("commit"),
    "lattice_gas_collision": TransitionLexicon("tick"),
    "majority_consensus": TransitionLexicon("generation"),
    "forest_fire_barrier": TransitionLexicon("tick"),
    "predator_prey_cycle": TransitionLexicon("tick"),
    "turn_write_agent_rollout": TransitionLexicon("tick"),
    "finite_tape_machine_rollout": TransitionLexicon("tick"),
    "simultaneous_agent_conflict": TransitionLexicon("tick"),
    "resource_pick_drop_swarm": TransitionLexicon("tick"),
    "diffusion_limited_aggregation": TransitionLexicon("tick"),
    "gui_hierarchical_navigation": TransitionLexicon("action"),
    "chained_function_pipeline": TransitionLexicon("block application"),
    "packet_router_tick": TransitionLexicon("tick"),
    "rail_signal_interlocking": TransitionLexicon("interlocking update"),
    "elevator_dispatch_tick": TransitionLexicon("controller tick"),
    "protocol_handshake_tick": TransitionLexicon("tick"),
    "register_machine_tick": TransitionLexicon("instruction"),
    "clocked_logic_register_tick": TransitionLexicon("clock edge"),
}


def _transition_lexicon(task: str) -> TransitionLexicon:
    try:
        return _TRANSITION_LEXICONS[task]
    except KeyError as exc:
        raise KeyError(f"{task}: missing TransitionLexicon") from exc


def _require_transition_unit(task: str, text: str, *, role: str) -> None:
    lexicon = _transition_lexicon(task)
    forms = (lexicon.unit, lexicon.plural or f"{lexicon.unit}s")
    if not any(form.lower() in text.lower() for form in forms):
        raise ValueError(
            f"{task}: {role} does not use its TransitionLexicon unit "
            f"{lexicon.unit!r}"
        )


def _canonical_json(value: Any) -> str:
    """Serialize public JSON with one readable space after commas and colons."""

    return canonical_json_dumps(value)


def _coords_where(old: Sequence[Sequence[Any]], new: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    return [
        {"cell": [row, col], "old": old[row][col], "new": new[row][col]}
        for row in range(len(old))
        for col in range(len(old[row]))
        if old[row][col] != new[row][col]
    ]


def _neighbours(grid: Sequence[Sequence[int]], row: int, col: int, offsets: Sequence[Tuple[int, int]], boundary: str, outside: int = 0) -> List[int]:
    rows, cols = len(grid), len(grid[0])
    values = []
    for dr, dc in offsets:
        rr, cc = row + dr, col + dc
        if boundary == "wrap":
            values.append(grid[rr % rows][cc % cols])
        elif 0 <= rr < rows and 0 <= cc < cols:
            values.append(grid[rr][cc])
        else:
            values.append(outside)
    return values


def _grid_transition_detail(task: str, old: Mapping[str, Any], new: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    old_grid, new_grid = old["grid"], new["grid"]
    changes = _coords_where(old_grid, new_grid)
    orthogonal = tuple(DIRS.values())
    moore = tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    )
    if task == "lifelike_rule_rollout":
        for change in changes:
            row, col = change["cell"]
            change["live_neighbours"] = sum(_neighbours(old_grid, row, col, moore, params["boundary"], 0))
            change["rule"] = "birth" if change["old"] == 0 else "survival"
    elif task == "cyclic_threshold_wave":
        offsets = orthogonal if params["neighbourhood"] == "orthogonal" else moore
        for change in changes:
            row, col = change["cell"]
            successor = (change["old"] + 1) % params["colours"]
            change["successor_state"] = successor
            change["successor_neighbours"] = sum(
                value == successor
                for value in _neighbours(old_grid, row, col, offsets, params["boundary"], -1)
            )
    elif task == "excitable_signal_network":
        for change in changes:
            row, col = change["cell"]
            change["old_head_neighbours"] = sum(
                value == 2 for value in _neighbours(old_grid, row, col, orthogonal, "fixed", 0)
            )
    elif task == "xor_parity_lattice":
        evaluations = []
        for row in range(len(old_grid)):
            for col in range(len(old_grid[row])):
                inputs = _neighbours(old_grid, row, col, orthogonal, "wrap", 0)
                if params["include_self"]:
                    inputs.append(old_grid[row][col])
                evaluations.append({"cell": [row, col], "inputs": inputs, "parity": sum(inputs) % 2})
        return {"evaluations": evaluations, "changed_cells": changes}
    elif task == "majority_consensus":
        offsets = orthogonal if params["neighbourhood"] == "orthogonal" else moore
        inspected = []
        for row in range(len(old_grid)):
            for col in range(len(old_grid[row])):
                if old_grid[row][col] == 0:
                    continue
                values = _neighbours(old_grid, row, col, offsets, params["boundary"], 0)
                one, two = values.count(1), values.count(2)
                if old_grid[row][col] != new_grid[row][col] or one == two:
                    inspected.append({"cell": [row, col], "opinion_1": one, "opinion_2": two, "result": new_grid[row][col]})
        return {"flip_or_tie_checks": inspected, "changed_cells": changes}
    elif task == "forest_fire_barrier":
        for change in changes:
            row, col = change["cell"]
            change["adjacent_old_flames"] = sum(
                value == 2 for value in _neighbours(old_grid, row, col, orthogonal, "fixed", 0)
            )
    elif task == "predator_prey_cycle":
        for change in changes:
            row, col = change["cell"]
            values = _neighbours(old_grid, row, col, moore, "fixed", 0)
            change["prey_like_neighbours"] = sum(value in (1, 3) for value in values)
            change["predator_neighbours"] = values.count(2)
    return {"changed_cells": changes}


def _trace_step(task: str, step: int, old: Mapping[str, Any], new: Mapping[str, Any], params: Mapping[str, Any]) -> Dict[str, Any]:
    event: Dict[str, Any]
    if task == "rule_table_1d_rollout":
        row = old["row"]
        cells = []
        for index in range(len(row)):
            if params["boundary"] == "wrap":
                triple = [row[(index - 1) % len(row)], row[index], row[(index + 1) % len(row)]]
            else:
                triple = [row[index - 1] if index else 0, row[index], row[index + 1] if index + 1 < len(row) else 0]
            rule_index = triple[0] * 4 + triple[1] * 2 + triple[2]
            cells.append({"index": index, "triple": triple, "rule_index": rule_index, "output": new["row"][index]})
        event = {"cell_evaluations": cells}
    elif task in {
        "lifelike_rule_rollout", "cyclic_threshold_wave", "excitable_signal_network",
        "xor_parity_lattice", "majority_consensus", "forest_fire_barrier", "predator_prey_cycle",
    }:
        event = _grid_transition_detail(task, old, new, params)
    elif task == "alternating_block_rotation":
        old_positions = {
            value: [row, col]
            for row, values in enumerate(old["grid"])
            for col, value in enumerate(values)
            if value >= 0
        }
        new_positions = {
            value: [row, col]
            for row, values in enumerate(new["grid"])
            for col, value in enumerate(values)
            if value >= 0
        }
        event = {
            "active_phase": old["phase"],
            "direction": params["direction"],
            "token_moves": [
                {"id": token, "from": old_positions[token], "to": new_positions[token]}
                for token in sorted(old_positions)
                if old_positions[token] != new_positions[token]
            ],
        }
    elif task == "sandpile_avalanche":
        if new.get("phase") == "added":
            event = {"operation": "add_one", "cell": params["add"], "height": new["grid"][params["add"][0]][params["add"][1]]}
        else:
            unstable = [
                [row, col]
                for row, values in enumerate(old["grid"])
                for col, value in enumerate(values)
                if value >= 4
            ]
            event = {
                "operation": "synchronous_topple_wave",
                "unstable_cells": unstable,
                "lost_this_wave": new["lost"] - old.get("lost", 0),
                "cumulative_lost": new["lost"],
                "changed_cells": _coords_where(old["grid"], new["grid"]),
            }
    elif task == "lattice_gas_collision":
        event = {
            "changed_velocity_masks": _coords_where(old["grid"], new["grid"]),
            "reflections": new["last_reflections"],
            "particle_count": sum(bin(value).count("1") for row in new["grid"] for value in row),
        }
    elif task in {"turn_write_agent_rollout", "finite_tape_machine_rollout"}:
        event = deepcopy(new["last"])
    elif task in {"simultaneous_agent_conflict", "diffusion_limited_aggregation"}:
        event = deepcopy(new["last"])
    elif task == "resource_pick_drop_swarm":
        event = {"ordered_agent_actions": deepcopy(new["last"])}
    elif task == "gui_hierarchical_navigation":
        event = deepcopy(new["last"])
    elif task == "chained_function_pipeline":
        event = deepcopy(new["last"])
    elif task in {"packet_router_tick", "rail_signal_interlocking", "elevator_dispatch_tick", "protocol_handshake_tick", "register_machine_tick"}:
        event = deepcopy(new["last"])
    elif task == "clocked_logic_register_tick":
        event = {"old_q": old["q"], "d_from_old_q": new["last_d"], "captured_q": new["q"]}
    else:
        event = {"transition": "replayed"}
    return {
        "step": step,
        "event": event,
        "successor": answer_payload(task, new, params),
    }


def build_typed_transition_trace(
    task: str,
    trace: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build the finite, sample-specific transition payload used by supervision.

    The verifier calls this only after replaying the transition system from the
    source ``params`` and ``initial_state``.  Cached traces never become an
    oracle merely because the generator serialized them.
    """
    return [
        _trace_step(task, index, old, new, params)
        for index, (old, new) in enumerate(zip(trace, trace[1:]), start=1)
    ]


def _ordinal_coord(value: Sequence[Any]) -> str:
    """Model-facing 1-based ``(x, y)`` name for a grid cell.

    Semantics, replay, and rendering keep internal 0-based indices; only this
    model-facing boundary shifts to the 1-based numbering the Prompt declares.
    """
    return format_xy_position(cell_to_xy(value))


def _human_join(items: Sequence[str]) -> str:
    clean = [item for item in items if item]
    if not clean:
        return "none"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


_DIRECTION_WORDS = {"N": "north", "E": "east", "S": "south", "W": "west"}


def _operation_text(operation: Mapping[str, Any]) -> str:
    kind = str(operation.get("kind") or "")
    if kind in {"ADD", "MUL", "XOR", "MOD"} and "value" in operation:
        return f"{kind} {operation['value']}"
    if kind == "CLAMP":
        return f"CLAMP to {operation['low']}..{operation['high']}"
    return kind


def _instruction_text(instruction: Mapping[str, Any]) -> str:
    """Name one register-machine instruction with a 1-based jump target.

    ``instruction['target']`` stays a 0-based list position in the semantics;
    every model-facing mention of a program line is numbered from 1.
    """
    kind = str(instruction.get("kind") or "")
    if kind in {"ADD", "XOR", "MOV"}:
        return f"{kind} {instruction['dst']},{instruction['src']}"
    if kind == "JZ":
        return f"JZ {int(instruction['target']) + 1}"
    if kind == "DECJNZ":
        return f"DECJNZ {instruction['reg']},{int(instruction['target']) + 1}"
    return kind


def _register_program_listing(program: Sequence[Mapping[str, Any]]) -> str:
    """Render the visible program as an explicitly 1-based numbered listing."""
    return "; ".join(
        f"{line}: {_instruction_text(instruction)}"
        for line, instruction in enumerate(program, start=1)
    )


def _logic_expression_text(expression: Mapping[str, Any]) -> str:
    """Name one clocked-logic expression with 1-based register slots.

    ``expression['index']`` stays a 0-based list position in the semantics; the
    visible register is Q1, Q2, or Q3 exactly as the state card labels it.
    """
    kind = str(expression.get("kind") or "")
    if kind == "Q":
        return f"Q{int(expression['index']) + 1}"
    if kind == "IN":
        return str(expression["name"])
    if kind == "NOT":
        return f"NOT {_logic_expression_text(expression['arg'])}"
    return (
        f"({_logic_expression_text(expression['left'])} {kind} "
        f"{_logic_expression_text(expression['right'])})"
    )


def _clocked_logic_listing(expressions: Sequence[Mapping[str, Any]]) -> str:
    """Render the three D expressions as an explicitly 1-based listing."""
    return "; ".join(
        f"D{index} = {_logic_expression_text(expression)}"
        for index, expression in enumerate(expressions, start=1)
    )


def _count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    plural = plural or f"{singular}s"
    return f"one {singular}" if int(count) == 1 else f"{int(count)} {plural}"


def _for_each(count: int, singular: str, plural: str | None = None) -> str:
    plural = plural or f"{singular}s"
    return (
        f"For the single {singular}"
        if int(count) == 1
        else f"For each of {int(count)} {plural}"
    )


def _rule_contract(task: str, params: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    """Return visible-token, transition, boundary, and object-fate clauses."""
    lexicon = _transition_lexicon(task)
    if task == "rule_table_1d_rollout":
        table = ", ".join(f"{code:03b}→{bit}" for code, bit in enumerate(params["rule"]))
        visible = "The state card contains one binary row: a pale cell is bit 0 and a navy cell is bit 1, and position numbers are printed above the row. The complete neighborhood table is given in the rule below."
        transition = (
            f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}, read every cell's old left-centre-right triple and write the bit the given table assigns to that exact triple: {table}. Compute the whole successor row synchronously from the same old row."
        )
        boundary = "At a fixed-zero edge, a missing outside neighbor is 0." if params["boundary"] == "fixed_zero" else "The row wraps: the last cell is the first cell's left neighbor and the first is the last cell's right neighbor."
        fate = "Every cell stays at its position and only its bit may change. Row length never changes, and no cell moves, falls, compacts, or refills."
        return visible, transition, boundary, fate
    if task == "lifelike_rule_rollout":
        visible = "The square grid uses pale cells for 0 (empty) and navy cells for 1 (live), with row and column numbers printed along the edges. The birth and survival sets are given in the rule below."
        transition = f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}, count the eight old-state surrounding cells for every location. An old 0 becomes 1 exactly when its count is in B={params['birth']}; an old 1 remains 1 exactly when its count is in S={params['survive']}; every other cell becomes 0. All cells update simultaneously."
        boundary = "Outside locations count as 0 and do not wrap." if params["boundary"] == "fixed_zero" else "Both axes wrap, so opposite edges and corners are adjacent."
        fate = "Each cell changes only in place. A birth does not travel from another cell, and a death leaves an ordinary pale cell. There is no movement, gravity, compaction, refill, or secondary update inside one generation."
        return visible, transition, boundary, fate
    if task == "cyclic_threshold_wave":
        colour_names = ("blue", "orange", "green", "purple", "amber")[: params["colours"]]
        colour_legend = ", ".join(
            f"state {index} is {name}" for index, name in enumerate(colour_names)
        )
        visible = (
            f"Each cell's color is its integer state from 0 through {params['colours'] - 1}: "
            f"{colour_legend}. Row and column numbers are printed along the edges."
        )
        transition = f"The cyclic successor of s is (s+1) mod {params['colours']}. {_for_each(params['steps'], lexicon.unit, lexicon.plural)}, count old-state {params['neighbourhood']} neighbors equal to that successor. Advance the cell only when the count is at least {params['threshold']}; otherwise retain s."
        boundary = "Missing neighbors outside the grid are ignored." if params["boundary"] == "fixed" else "Both axes wrap and opposite edges are neighbors."
        fate = "Cells change state in place only; a state never advances more than one step per update, and unchanged cells keep their exact state. Nothing moves, disappears, or refills."
        return visible, transition, boundary, fate
    if task == "excitable_signal_network":
        visible = "Cell colors are the states: pale is empty 0, yellow is conductor 1, blue is signal head 2, red is refractory tail 3, and dark is blocker 4. Row and column numbers are printed along the edges."
        transition = f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}: 2 becomes 3; 3 becomes 1; a 1 becomes 2 exactly when it has one old-state orthogonal 2 neighbor and otherwise remains 1; 0 and 4 remain unchanged."
        boundary = "The grid is fixed: outside is empty and never wraps. Only north, east, south, and west count; diagonal heads never trigger a conductor."
        fate = "Signal state changes in place. Conductors are not consumed, blockers and empty cells never change, and a multi-head collision leaves the contacted conductor a conductor. The network topology never changes."
        return visible, transition, boundary, fate
    if task == "alternating_block_rotation":
        visible = "Blue circular tokens carry stable numeric identities; blank cells are empty slots, and row and column numbers are printed along the edges. Small blue diamonds mark the top-left cells of the currently active disjoint 2×2 blocks, including wrapped edge blocks. Rotation direction is defined below."
        transition = f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}, partition the even square grid into disjoint 2×2 blocks: in phase 0 every block's top-left (x, y) has odd x and odd y, and in phase 1 it has even x and even y. Rotate all four contents of every active block one quarter-turn {params['direction']} simultaneously, then toggle phase 0↔1. Phase-1 edge blocks wrap across opposite edges."
        boundary = "Only phase-1 blocks wrap; the odd/even block membership given by the phase rule is exact. Tokens never leave their assigned rotating block during that step."
        fate = "Each token keeps its identity and label while it moves to the adjacent rotated corner of its own block. Empty slots rotate as vacancies and stay ordinary blank cells. No token is created, deleted, duplicated, or recolored."
        return visible, transition, boundary, fate
    if task == "xor_parity_lattice":
        visible = "Pale cells are bit 0 and navy cells are bit 1, with row and column numbers printed along the edges. Whether the centre bit participates is defined in the rule below."
        transition = f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}, XOR the old north, east, south, and west bits{' plus the old centre bit' if params['include_self'] else ''}; write the exact parity bit into the same cell."
        boundary = "Both axes wrap, including corner-to-opposite-edge adjacency through the specified orthogonal directions; there are no diagonal inputs."
        fate = "Every cell keeps its position and only its bit may change to the exact parity. No bit travels as an object, and no cell moves, falls, compacts, or refills."
        return visible, transition, boundary, fate
    if task == "sandpile_avalanche":
        # Model-facing row/column numbers start at 1; params["add"] stays 0-based.
        add_row, add_col = params["add"]
        visible = "Every cell prints its nonnegative grain height on a plain card, and row and column numbers are printed along the edges. The cell that receives the new grain is named in the rule below and is not marked in the image."
        transition = (
            f"For the first {lexicon.unit}, add one grain to cell {_ordinal_coord((add_row, add_col))}. "
            f"For every later {lexicon.unit}, while any cell is at least 4, form one synchronous wave from the entire old unstable set: "
            "each unstable cell loses 4 and sends one grain north, east, south, and west. "
            "Apply all wave transfers together and repeat until every height is below 4."
        )
        boundary = "The boundary is open: a grain sent outside is permanently lost and increments the loss total; it does not wrap or reflect."
        fate = "Grains move only through the toppling transfers, and a grain sent outside is gone permanently. Stable cells keep their heights except for received transfers; there is no refill, relocation, or hidden sequential toppling inside one wave."
        return visible, transition, boundary, fate
    if task == "lattice_gas_collision":
        visible = "A blue arrow is one directional particle channel; several differently directed arrows may coexist in a cell. Dark cells are impenetrable walls and blank pale cells are empty, and row and column numbers are printed along the edges. The provided rollout never sends two particles into the same directional channel of one destination, so every displayed arrow remains a distinct conserved particle."
        transition = f"{_for_each(params['steps'], 'synchronous tick')}, first turn an exact east+west pair into north+south and an exact north+south pair into east+west; other channel sets keep their directions. Then stream every resulting arrow one cell in its direction."
        boundary = "If the destination is outside the grid or a WALL, the particle remains in its source cell and reverses direction. Otherwise it enters the destination."
        fate = "Every particle is conserved: per tick it either turns in place, streams one cell, or reverses in its source cell at a wall or edge. Walls and empty cells never change, and no particle merges, duplicates, or disappears."
        return visible, transition, boundary, fate
    if task == "majority_consensus":
        visible = "Gray cells are immutable vacancies (state 0); blue cells hold opinion 1 and coral cells hold opinion 2. Row and column numbers are printed along the edges."
        transition = f"{_for_each(params['steps'], lexicon.unit, lexicon.plural)}, count opinion 1 and opinion 2 among the old {params['neighbourhood']} neighbors of each non-vacant cell. Adopt 1 if count1>count2, adopt 2 if count2>count1, and retain the old opinion on an exact tie. Vacancies stay 0."
        boundary = "Neighbors outside are ignored and do not wrap." if params["boundary"] == "fixed" else "Both axes wrap, so opposite edges participate in the majority."
        fate = "Opinion cells change opinion only in place; a vacancy never fills and contributes nothing to any count. Ties and unaffected opinions stay exactly as they were; nothing moves, disappears, or refills."
        return visible, transition, boundary, fate
    if task == "forest_fire_barrier":
        visible = "Cell colors are the states: pale is empty 0, green is tree 1, orange is flame 2, brown is ash 3, blue is water 4, and dark is rock 5. Row and column numbers are printed along the edges."
        transition = f"{_for_each(params['steps'], 'synchronous tick')}, every old flame 2 becomes ash 3, and every old tree 1 with at least one orthogonal old flame becomes flame 2. All other codes retain their state. Newly ignited trees cannot ignite another tree until the next tick."
        boundary = "Outside is empty, there is no wrap, and diagonal contact never ignites. Water and rock block propagation because they remain 4 and 5."
        fate = "State changes happen in place: ash remains ash, flames never move as objects, and water, rock, and empty cells never change. Fire never spreads farther than one cell within the same tick."
        return visible, transition, boundary, fate
    if task == "predator_prey_cycle":
        visible = "Cell colors are the states: pale is empty 0, green is prey 1, coral is predator 2, violet is sheltered prey 3, and dark is rock 4. Row and column numbers are printed along the edges."
        transition = f"{_for_each(params['steps'], 'synchronous tick')} using eight old surrounding neighbors: empty 0 becomes prey 1 when it has at least two prey-like neighbors (1 or 3) and zero predators; ordinary prey 1 becomes predator 2 when it has at least two predators; predator 2 becomes empty 0 when it has zero prey-like neighbors; every other state retains."
        boundary = "Outside is empty and does not wrap. Sheltered prey 3 and rock 4 are immutable even beside predators."
        fate = "Birth, predation, and starvation are in-place changes derived only from the old grid. No animal moves between cells, and nothing except an explicitly changed cell differs between consecutive states."
        return visible, transition, boundary, fate
    if task == "turn_write_agent_rollout":
        table = params["table"]
        visible = "Each cell is pale for bit 0 or navy for bit 1, with row and column numbers printed along the edges. The orange token is the unique head: its arrow is the facing direction, and the small chip on the token repeats the covered cell's bit color. The two read rules are given below."
        transition = f"Run exactly {_count_phrase(params['steps'], 'tick')}. On every tick in this exact order: read the bit under H; use table 0={table['0']} or 1={table['1']}; rotate H left or right; replace the departed cell with the specified bit; then move H one cell forward in its new direction."
        boundary = "Rows and columns wrap: moving beyond one edge enters the opposite edge at the same row or column."
        fate = "The head is the only moving object. The departed cell keeps the written bit, and every other cell keeps its bit until the head visits it. No cell moves, falls, compacts, or refills."
        return visible, transition, boundary, fate
    if task == "finite_tape_machine_rollout":
        visible = "The horizontal tape shows pale slots for bit 0 and navy slots for bit 1, with slot numbers printed above the tape. The red head badge carries control A, B, or C on exactly one slot, and its small chip repeats the covered slot's bit color. The six transition entries are given below."
        transition = f"Run exactly {_count_phrase(params['steps'], 'tick')}. At each tick, read the bit under the head, select the unique transition from table {_canonical_json(params['table'])}, write that bit in the same slot, change to the listed control, and only then move one slot L or R."
        boundary = "The tape is circular: L from slot 1 enters the last slot and R from the last slot enters slot 1."
        fate = "The head is the only moving object; the departed slot keeps the written bit and every unvisited slot keeps its bit. No slot shifts, appears, or disappears."
        return visible, transition, boundary, fate
    if task == "simultaneous_agent_conflict":
        visible = "Orange circular agents have stable ID letters and arrows showing their one-cell proposals. Blank cells are traversable empty cells, and row and column numbers are printed along the edges."
        transition = f"For the one tick, form every proposal from the old positions. A direct two-agent swap makes both wait. For a shared destination, apply policy {params['policy']}: either all contenders wait or only the lowest ID moves. A provisional mover into an old occupied cell succeeds only if that occupant also succeeds in leaving; repeatedly cancel blocked moves until stable. All remaining successful agents move simultaneously."
        boundary = "Both axes wrap; an edge proposal enters the opposite edge."
        fate = "Successful agents keep their ID and arrow while moving exactly one cell; waiting agents stay exactly where they are. Vacated cells become ordinary blank cells, and agents are never merged, duplicated, or deleted."
        return visible, transition, boundary, fate
    if task == "resource_pick_drop_swarm":
        visible = "A printed integer is the resource pile in that cell, and row and column numbers are printed along the edges. Orange agents A and B show direction arrows; ● on a badge means carrying one unit and ○ means carrying none, while pN on the badge repeats the covered cell's pile count so neither layer is hidden."
        transition = f"Run {_count_phrase(params['steps'], 'tick')}, processing agents in ascending ID order inside each tick. At its current cell, a carrying agent drops exactly one unit and becomes empty-handed; otherwise an empty-handed agent picks exactly one unit if the pile is positive and becomes carrying; if neither applies it does nothing. It then moves one cell in its fixed arrow direction. Later agents see earlier agents' pile update."
        boundary = "Movement wraps on both axes. Agents do not block one another, and their badges remain separately readable even when their paths interact."
        fate = "Each unit exists either in a displayed pile or on a carrying agent, so total pile units plus carried units never changes. Agents keep their identity and direction, and a vacated cell keeps its exact updated pile. No hidden resource exists."
        return visible, transition, boundary, fate
    if task == "diffusion_limited_aggregation":
        visible = "Dark cells are the fixed aggregate. Purple walkers have stable ID letters and arrows showing their one-cell proposals, and row and column numbers are printed along the edges."
        transition = "For the one tick, first freeze every walker orthogonally adjacent to the old aggregate: remove its walker badge and turn its cell into permanent #. Then form proposals for remaining walkers. A proposal into the now-expanded aggregate waits. For multiple walkers proposing the same empty destination, only the lowest ID moves; other contenders wait. Distinct destinations move simultaneously."
        boundary = "Adjacency and motion wrap on both axes. Old walker positions do not block other walkers, so a direct swap is allowed; only identical destinations conflict."
        fate = "A frozen walker becomes a permanent dark aggregate cell at that exact position; aggregate cells never move or disappear. Mobile walkers keep their ID and arrow, waiting walkers stay in place, and vacated cells become blank."
        return visible, transition, boundary, fate
    if task == "gui_hierarchical_navigation":
        visible = "The interface panel shows a stable ← BACK control, the active PAGE, stack DEPTH, a filled/empty bead strip for consumed and remaining prescribed actions, and up to three labeled current-page tiles with their target pages."
        action_rules = ", then ".join(
            "BACK" if action["kind"] == "BACK" else f"CLICK {action['label']}"
            for action in params["actions"]
        )
        transition = (
            "ROOT has ALPHA, BETA, and GAMMA tiles. On any top page TOP, the ONE, TWO, and THREE tiles "
            "open the leaf pages TOP-ONE, TOP-TWO, and TOP-THREE respectively; leaf pages have no child tiles. "
            f"Execute these five visible actions in order: {action_rules}. CLICK pushes the tile's target onto the "
            "stack; BACK pops one page when depth exceeds 1 and otherwise retains ROOT. After each action, PAGE is "
            "the stack top, DEPTH is its length, and the action index increments."
        )
        boundary = "A click never searches another page or sibling; BACK never underflows below ROOT. Pages with no tiles show normal dash placeholders."
        fate = "The BACK control, page map, panel geometry, and tile slots never change identity. A transition replaces only the active-page tile contents and the declared state fields; it never reorders, duplicates, invents, or deletes a hierarchy definition."
        return visible, transition, boundary, fate
    if task == "chained_function_pipeline":
        visible = "A green packet carries the current integer. Fixed blocks from IN through OUT show the exact left-to-right functions."
        transition = f"Starting with the input value, apply these operations exactly left to right: {_canonical_json(params['operations'])}. For each {lexicon.unit}, ADD and MUL use ordinary signed integers; XOR is integer bitwise XOR; MOD m returns the canonical remainder 0≤r<m; NEG changes the sign; CLAMP low-to-high returns low below the interval, high above it, and the input inside it. Commit each new value only at that block."
        boundary = "Operations are never reordered, skipped, repeated, or evaluated in parallel; every block receives the preceding block's committed output."
        fate = "The single green packet carries the current value and changes only at a block. The function blocks, their order, and their printed functions never change, and no second packet ever exists."
        return visible, transition, boundary, fate
    if task == "packet_router_tick":
        visible = "Rows I1–I3 are ordered queues from left to right; • marks the cyclic arbitration pointer. Each green packet has stable ID, destination O1–O3, and TTL tN. The bottom ports are O1–O3."
        transition = "For the one tick, remove every leading TTL-0 packet from each input. The surviving heads contend; for each output independently, scan inputs cyclically from the pointer and choose the first requester. Winners move to their outputs and losing heads stay; both decrement TTL once, non-heads do not. Advance the pointer to the next queue, wrapping from I3 back to I1."
        boundary = "A newly exposed head may contend immediately; an output with no requester stays idle."
        fate = "Queue survivors close toward the queue head without reordering. Packet IDs and destinations never change, and untouched packets, ports, and labels stay exactly as given."
        return visible, transition, boundary, fate
    if task == "rail_signal_interlocking":
        priority = [str(route_id) for route_id in params["priority"]]
        route_rules = "; ".join(
            f"route {route_id} uses blocks {_human_join([str(block) for block in route['blocks']])} "
            f"and requires switch {route['switch']}"
            for route_id, route in sorted(params["routes"].items())
        )
        visible = "The table lists routes A–C, request YES/NO, signal RED/GREEN, each route's numbered blocks, occupied blocks, and switch LEFT/RIGHT."
        transition = f"For the single {lexicon.unit}, inspect requests in priority order {_human_join(priority)}. Definitions: {route_rules}. A route is unsafe if any listed block is occupied. Accept only the first safe request, set its signal GREEN and the rest RED, and set its required switch; with no acceptance, retain the switch."
        boundary = "At most one route is accepted; unrequested routes stay RED and occupancy never changes."
        fate = "Only the signal aspects and possibly the switch change. Requests, route definitions, and occupancy stay exactly as given, and no train moves."
        return visible, transition, boundary, fate
    if task == "elevator_dispatch_tick":
        visible = "The vertical rows are numbered floors; shafts A and B contain orange car badges with load/capacity and door O/C. CALL marks the two ordered hall calls."
        transition = "Assign calls in displayed order, at most one per car. For each call choose the still-unassigned car with smallest absolute floor distance; an exact tie goes to the lowest ID. Then update both cars simultaneously for one controller tick: an assigned car below its call moves up one floor, one above moves down one, and one already at the call opens only when load<capacity. A full car at its call remains there with door CLOSED. An unassigned car remains still and closed."
        boundary = f"Valid floors are 0 through {params['floors'] - 1}; a car may not move outside this range. Assignment distance ignores load, while capacity affects only whether a door can open at the assigned floor."
        fate = "Each car keeps its identity, shaft, load, and capacity; in one tick it either moves exactly one floor, opens its door in place, or stays closed where it is. Calls remain displayed for this prediction tick, and no car swaps shafts, duplicates, or changes load."
        return visible, transition, boundary, fate
    if task == "protocol_handshake_tick":
        visible = "The panel shows CLIENT and SERVER finite states, the current timeout and tick, and an oldest-to-newest row of blue message badges. Each message has stable ID, kind, and target endpoint."
        transition = f"Run exactly {_count_phrase(params['steps'], 'oldest-first tick')}. Deliver at most the FIFO head, then append its response: START→client sets WAIT and timeout {params['timeout_reset']} and appends HELLO→server; HELLO→server sets READY and appends ACK→client; ACK→a WAIT client sets OPEN and timeout 0 and appends DATA→client. Other deliveries do not change endpoints. If the client remains WAIT without receiving ACK, decrement timeout; at 0 append retry HELLO and reset it to {params['timeout_reset']}."
        boundary = "An empty queue delivers nothing; a new response waits until a later tick and timeout never becomes negative."
        fate = "The delivered head leaves the queue, survivors keep their FIFO order and close toward the head, and responses join only at the tail. Message IDs and kinds never change, and the visible TICK count advances at each commit."
        return visible, transition, boundary, fate
    if task == "register_machine_tick":
        lines = len(params["program"])
        visible = "The state table shows PC, ZERO, and registers A–C. The complete program is given below rather than duplicated above the state table; its lines are numbered from 1 and PC holds the line number to execute next."
        transition = f"Execute exactly {params['steps']} instructions from the program [{_register_program_listing(params['program'])}]. ADD dst,src writes (dst+src) mod 16; XOR writes bitwise dst xor src; MOV copies src; all three set ZERO from the written result and normally advance PC to the next line, wrapping from line {lines} back to line 1. JZ target jumps to that line number when ZERO is true and otherwise advances. DECJNZ reg,target writes (reg−1) mod 16, sets ZERO from that result, jumps to that line number when nonzero, and otherwise advances. Each next instruction uses the fully committed preceding state."
        boundary = f"All register writes are integers 0–15 under modulo 16. PC always names a displayed line between 1 and {lines}; no instruction outside the program exists."
        fate = "Only the addressed register, the ZERO flag, and PC change per instruction; unaddressed registers keep their values. There is no parallel execution, hidden stack, or memory."
        return visible, transition, boundary, fate
    if task == "clocked_logic_register_tick":
        visible = "The table shows old Q1–Q3 and fixed inputs X/Y. The three D expressions are given below rather than duplicated in the image."
        transition = f"{_for_each(params['steps'], 'clock edge')}, evaluate the three expressions [{_clocked_logic_listing(params['expressions'])}] using the same complete old Q vector and current X/Y. Operators are bit NOT, XOR, AND, and OR. After every D value is known, capture D1,D2,D3 into Q1,Q2,Q3 simultaneously; then increment edge count. On a later edge, that new Q becomes the old vector."
        boundary = "Every signal is exactly bit 0 or 1. Never update Q1 early and feed it into D2 or D3 during the same edge."
        fate = "All three registers capture together only at the clock edge; X and Y never change. No register updates early, feeds its new value into another D expression on the same edge, or is skipped."
        return visible, transition, boundary, fate
    raise KeyError(task)


def _video_process_contract(task: str) -> str:
    """Per-task Required Output process prose for the discrete TI2V program.

    The hold/preservation/prohibition boilerplate lives only in the shared
    continuity paragraph; these sentences describe what each committed switch
    of this task's ordered visual program contains.
    """
    synchronous_grid = (
        f"Commit each {_transition_lexicon(task).unit} as one complete synchronous state: every cell that changes "
        f"in that {_transition_lexicon(task).unit} switches together, and the completed grid stays readable before "
        f"the next {_transition_lexicon(task).unit}."
    )
    if task in {
        "lifelike_rule_rollout", "cyclic_threshold_wave",
        "excitable_signal_network", "xor_parity_lattice", "majority_consensus",
        "forest_fire_barrier", "predator_prey_cycle",
    }:
        return synchronous_grid
    if task == "rule_table_1d_rollout":
        return (
            "Commit each generation as one complete synchronous row state: every position that changes "
            "switches together, and the completed row stays readable before the next generation."
        )
    if task == "alternating_block_rotation":
        return (
            f"Commit each {_transition_lexicon(task).unit} as one simultaneous quarter-turn: all contents of every marked block appear "
            "at their rotated corners together, and the phase diamonds toggle at the same switch."
        )
    if task == "sandpile_avalanche":
        return (
            "First commit the single grain addition on the named cell as its own state. Then commit each "
            "complete toppling wave as one state: every unstable cell topples together, and the updated "
            "heights stay readable before the next wave."
        )
    if task == "lattice_gas_collision":
        return (
            "Commit each tick as one simultaneous state: turned pairs, streamed arrows, and reversed "
            "arrows all appear at their resulting cells together."
        )
    if task in {"turn_write_agent_rollout", "finite_tape_machine_rollout"}:
        return (
            "Commit each tick as one state: the written bit, the head's updated badge, and the head's new "
            "position all change together."
        )
    if task == "simultaneous_agent_conflict":
        return (
            "Commit the single tick as one state: every successful agent appears at its resolved "
            "destination together while waiting agents stay in place."
        )
    if task == "resource_pick_drop_swarm":
        return (
            "Commit each tick as one state showing both ordered agents' results: the updated piles, the "
            "updated carrying marks, and both new agent positions appear together."
        )
    if task == "diffusion_limited_aggregation":
        return (
            "Commit the single tick as one state: frozen walkers appear as dark aggregate cells, accepted "
            "movers appear at their destinations, and waiting walkers stay in place."
        )
    if task == "gui_hierarchical_navigation":
        return (
            "For every prescribed action, keep the complete old page readable while a press ring and "
            "contact dot appear over the exact target tile or the stable ← BACK control. After that "
            "contact, replace the current-page tiles and update PAGE, DEPTH, and the action-progress beads "
            "together in one atomic commit, then keep the complete new page readable. Never wipe or "
            "dissolve the full page through a blank interface skeleton."
        )
    if task == "chained_function_pipeline":
        return (
            "Commit each block application as one state: the packet appears at the next block already "
            "showing its committed value."
        )
    if task == "packet_router_tick":
        return (
            "Commit the single tick as one state: expired heads are gone, winners appear at their output "
            "ports, surviving queues are closed toward their heads, TTLs are updated, and the pointer mark "
            "advances together."
        )
    if task == "elevator_dispatch_tick":
        return (
            "Commit the single controller tick as one state: both cars appear at their updated floors with "
            "their final door labels together."
        )
    if task == "protocol_handshake_tick":
        return (
            "Commit each tick as one state: the delivered message is gone, survivors are closed toward the "
            "head, the endpoint states, TIMEOUT, and TICK show their new values, and any response appears "
            "at the tail together."
        )
    if task == "rail_signal_interlocking":
        return (
            "Commit the single interlocking update as one state: every signal shows its final aspect and the switch "
            "shows its final position together."
        )
    if task == "register_machine_tick":
        return (
            "Commit each executed instruction as one state: the addressed register, ZERO, and PC show "
            "their new values together."
        )
    if task == "clocked_logic_register_tick":
        return (
            "Commit each clock edge as one state: Q1, Q2, and Q3 all show their captured values together."
        )
    raise KeyError(task)


# Per-task noun slots for the shared TI2V continuity paragraph.  Every
# non-GUI next-state world is a discrete transition system;
# the GUI task keeps continuous timing for its sanctioned element-local press
# and commit language.
_TI2V_CONTINUITY_SLOTS: Dict[str, Dict[str, str]] = {
    "rule_table_1d_rollout": {
        "moving": "the row cells the table changes",
        "static": "the row frame, the printed position numbers, the background, and the camera",
        "hold": "the exact final row",
    },
    "lifelike_rule_rollout": {
        "moving": "the grid cells the rule changes",
        "static": "the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final grid",
    },
    "cyclic_threshold_wave": {
        "moving": "the cells that advance state",
        "static": "the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final grid",
    },
    "excitable_signal_network": {
        "moving": "the signal heads and tails the rule changes",
        "static": "the conductors that stay conductors, the blockers, the empty cells, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final network state",
    },
    "alternating_block_rotation": {
        "moving": "the numbered tokens and the phase diamonds",
        "static": "the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final token layout",
    },
    "xor_parity_lattice": {
        "moving": "the cells whose parity bit changes",
        "static": "the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final grid",
    },
    "sandpile_avalanche": {
        "moving": "the printed heights the addition and the waves change",
        "static": "the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final stable heights",
    },
    "lattice_gas_collision": {
        "moving": "the directional arrows",
        "static": "the walls, the empty cells, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final arrow layout",
    },
    "majority_consensus": {
        "moving": "the opinion cells that flip",
        "static": "the vacancies, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final opinion grid",
    },
    "forest_fire_barrier": {
        "moving": "the tree and flame cells the tick changes",
        "static": "the water, rock, and empty cells, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final grid",
    },
    "predator_prey_cycle": {
        "moving": "the cells changed by birth, predation, or starvation",
        "static": "the sheltered prey, the rocks, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final grid",
    },
    "turn_write_agent_rollout": {
        "moving": "the head token and the bits it rewrites",
        "static": "the unvisited cells, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final board and head",
    },
    "finite_tape_machine_rollout": {
        "moving": "the head badge and the bits it rewrites",
        "static": "the unvisited slots, the tape frame, the printed slot numbers, the background, and the camera",
        "hold": "the exact final tape and head",
    },
    "simultaneous_agent_conflict": {
        "moving": "the agents that successfully move",
        "static": "the waiting agents, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact resolved agent layout",
    },
    "resource_pick_drop_swarm": {
        "moving": "the two agents, their carrying marks, and the pile counts they change",
        "static": "the untouched piles, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final piles and agents",
    },
    "diffusion_limited_aggregation": {
        "moving": "the walkers and the aggregate cells they become",
        "static": "the waiting walkers, the earlier aggregate, the grid frame, the printed edge numbers, the background, and the camera",
        "hold": "the exact final aggregate and walkers",
    },
    "gui_hierarchical_navigation": {
        "moving": "the pressed control cue and the committed page contents",
        "static": "the ← BACK control, the panel geometry, the tile slots, the background, and the camera",
        "hold": "the exact final page state",
    },
    "chained_function_pipeline": {
        "moving": "the single packet and its printed value",
        "static": "the function blocks, their printed functions, the background, and the camera",
        "hold": "the exact final packet value at OUT",
    },
    "packet_router_tick": {
        "moving": "the packets, their TTL counts, and the pointer mark",
        "static": "the queue rows, the port labels, the background, and the camera",
        "hold": "the exact final queue and port state",
    },
    "rail_signal_interlocking": {
        "moving": "the signal aspects and the switch value",
        "static": "the route table, the requests, the occupancy, the background, and the camera",
        "hold": "the exact final signal state",
    },
    "elevator_dispatch_tick": {
        "moving": "the two car badges and their door labels",
        "static": "the floor rows, the CALL marks, the background, and the camera",
        "hold": "the exact final car positions",
    },
    "protocol_handshake_tick": {
        "moving": "the message badges, the endpoint states, and the TIMEOUT and TICK values",
        "static": "the panel layout and its fixed labels, the background, and the camera",
        "hold": "the exact final protocol state",
    },
    "register_machine_tick": {
        "moving": "the register, ZERO, and PC values the instructions change",
        "static": "the state-table layout, the background, and the camera",
        "hold": "the exact final machine state",
    },
    "clocked_logic_register_tick": {
        "moving": "the captured Q values",
        "static": "the X and Y inputs, the table layout, the background, and the camera",
        "hold": "the exact final register state",
    },
}


def _build_prompts(task: str, params: Mapping[str, Any], answer: Mapping[str, Any], transition_count: int) -> Dict[str, str]:
    visible, transition, boundary, fate = _rule_contract(task, params)
    if task == "gui_hierarchical_navigation":
        input_prefix = "The input image is one complete initial interface state. "
    elif task == "finite_tape_machine_rollout":
        # This scene uses tape slots, not grid coordinates.  Model-facing slot
        # ordinals start at 1 at the left end.
        input_prefix = (
            "The input image is one complete initial-state card. Tape slots are numbered from 1 "
            "at the left end. "
        )
    elif task == "turn_write_agent_rollout":
        input_prefix = (
            f"The input image is one complete initial-state card. {GRID_XY_CONVENTION_SENTENCE} "
        )
    elif task == "elevator_dispatch_tick":
        # This scene uses numbered floors, not grid coordinates.  Do not retain
        # an inapplicable family boilerplate.
        input_prefix = "The input image is one complete initial-state card. "
    elif task == "packet_router_tick":
        # This scene numbers queues and ports, not grid coordinates.  Declare the
        # 1-based labels so the reported numbers match the drawn ones.
        input_prefix = (
            "The input image is one complete initial-state card. Input queues are numbered I1 to I3 "
            "and output ports O1 to O3; report every queue and port number with those same labels. "
        )
    elif task in {"register_machine_tick", "clocked_logic_register_tick"}:
        # These scenes use numbered program lines or numbered register slots,
        # not grid coordinates.  Do not retain an inapplicable family
        # boilerplate that also declares a conflicting zero-based convention.
        input_prefix = "The input image is one complete initial-state card. "
    elif task == "sandpile_avalanche":
        input_prefix = (
            f"The input image is one complete initial-state card. {GRID_XY_CONVENTION_SENTENCE} "
            "The printed edge numbers use that convention. "
        )
    elif task == "rule_table_1d_rollout":
        # This scene is a single row, not a grid.  Model-facing row positions
        # are numbered from 1 at the left end.
        input_prefix = (
            "The input image is one complete initial-state card. Row positions are numbered "
            "from 1 at the left end. "
        )
    elif task in {"chained_function_pipeline", "protocol_handshake_tick", "rail_signal_interlocking"}:
        # These scenes use labeled function blocks, endpoint/message badges, or
        # route tables, not grid coordinates.  Do not retain an inapplicable
        # family boilerplate.
        input_prefix = "The input image is one complete initial-state card. "
    elif task in {
        "simultaneous_agent_conflict", "resource_pick_drop_swarm",
        "diffusion_limited_aggregation",
    }:
        input_prefix = (
            f"The input image is one complete initial-state card. {GRID_XY_CONVENTION_SENTENCE} "
            "The printed edge numbers and every position in the typed answer use that convention. "
        )
    else:
        input_prefix = (
            f"The input image is one complete initial-state card. {GRID_XY_CONVENTION_SENTENCE} "
        )
    input_description = input_prefix + visible
    rules = " ".join((transition, boundary, fate))
    _require_transition_unit(task, rules, role="Rule")
    transition_lexicon = _transition_lexicon(task)
    transition_phrase = transition_lexicon.count_phrase(transition_count)
    process = _video_process_contract(task)
    _require_transition_unit(task, process, role="TI2V process")
    slots = _TI2V_CONTINUITY_SLOTS[task]
    ti2v = (
        "Input Image:\n" + input_description +
        "\n\nRule:\n" + rules +
        "\n\nRequired Output:\n"
        f"Create one video that replays {transition_phrase} in order. {process}"
        + "\n\n"
        + build_ti2v_continuity_paragraph(
            timing="continuous" if task == "gui_hierarchical_navigation" else "discrete",
            moving_nouns=slots["moving"],
            static_nouns=slots["static"],
            hold_noun=slots["hold"],
        )
    )
    return {
        "question": f"Replay the explicitly defined transition system for {transition_phrase} and predict its terminal state.",
        "ti2v_prompt": ti2v,
    }


class NextStateGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    task_name = ""
    task_family = "next_state_prediction"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        seed: int | None = None,
        video: bool = True,
        video_fps: int = 16,
        regime: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__(output_dir)
        if not self.task_name:
            raise ValueError("NextStateGenerator requires a concrete task_name")
        if video_fps != 16:
            raise ValueError("next-state videos are fixed at 16fps")
        self.rng = random.Random(seed)
        self.video = bool(video)
        self.video_fps = 16
        self._sample_counter = 0
        self.regime = regime

    def create_puzzle(self) -> Dict[str, Any]:
        case = build_case(self.task_name, self.rng, regime=self.regime)
        trace = case["trace"]
        params = case["params"]
        answer = case["answer"]
        transition_trace = build_typed_transition_trace(self.task_name, trace, params)
        answer_json = _canonical_json(answer)
        prompts = _build_prompts(self.task_name, params, answer, len(trace) - 1)

        token = self.rng.getrandbits(64)
        sample_id = f"{self.task_name}-{token:016x}-{self._sample_counter:04d}"
        self._sample_counter += 1
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=False)
        puzzle_path = sample_dir / f"{sample_id}__puzzle.png"
        solution_path = sample_dir / f"{sample_id}__solution.png"
        video_path = sample_dir / f"{sample_id}__solution.webm"

        render_state(
            self.task_name, trace[0], params,
            mode="input", total_steps=len(trace) - 1,
        ).save(puzzle_path)
        render_state(
            self.task_name, trace[-1], params,
            mode="result", total_steps=len(trace) - 1,
        ).save(solution_path)

        solution_video: str | None = None
        video_num_frames: int | None = None
        if self.video:
            frames = render_video_frames(self.task_name, trace, params)
            if not encode_rgb_frames_to_webm(frames, video_path, fps=self.video_fps):
                raise RuntimeError(f"failed to encode next-state video: {video_path}")
            solution_video = self.relativize_path(video_path)
            video_num_frames = len(frames)

        puzzle_rel = self.relativize_path(puzzle_path)
        solution_rel = self.relativize_path(solution_path)
        structured_answer = {
            "semantic_contract_version": "next_state_successor_keypoints_v3",
            "task": self.task_name,
            "params": deepcopy(params),
            "initial_state": deepcopy(case["initial_state"]),
            "replay_trace": deepcopy(trace),
            "typed_transition_trace": transition_trace,
            "terminal_state": deepcopy(trace[-1]),
            "terminal_answer": deepcopy(answer),
            "strict_replay_source": ["params", "initial_state"],
            "cached_terminal_is_not_oracle": True,
        }
        return {
            "id": sample_id,
            "task_family": self.task_family,
            "question": prompts["question"],
            "answer": answer_json,
            "structured_answer": structured_answer,
            "ti2v_prompt": prompts["ti2v_prompt"],
            "image": puzzle_rel,
            "reasoning_image": puzzle_rel,
            "reasoning_images": [puzzle_rel],
            "solution_image_path": solution_rel,
            "solution_video_path": solution_video,
            "video_fps": self.video_fps if solution_video else None,
            "video_num_frames": video_num_frames,
            "canvas_size": [512, 512],
        }


for _task_name, _class_name in EXPANSION_CLASS_NAMES.items():
    globals()[_class_name] = type(
        _class_name,
        (NextStateGenerator,),
        {"task_name": _task_name, "__module__": __name__},
    )


__all__ = [
    "NextStateGenerator",
    "build_typed_transition_trace",
    *EXPANSION_CLASS_NAMES.values(),
]
