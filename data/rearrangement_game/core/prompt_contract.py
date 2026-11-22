"""Natural three-section prompts for Rearrangement Game."""

from __future__ import annotations

import importlib
import inspect
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from data.common.grid_coordinates import (
    GRID_XY_CONVENTION_SENTENCE,
)
from data.common.prompts import build_sectioned_prompt
from data.registry import TASK_SPECS
from data.common.ti2v_continuity import build_ti2v_continuity_paragraph


_TYPED_OUTPUT_OVERRIDES: dict[str, str] = {
    "bookshelf_sort_by_height": (
        "Return any shortest swap plan as `Answer: [(<left_slot>, <right_slot>), ...].` "
        "Number the shelf slots 1 for the leftmost book onward, and use the slots as they stand before each swap. "
        "As a syntax-only example: `Answer: [(1, 3), (2, 5), ...].`"
    ),
    "bus_stops_dispatch": (
        "Return the complete visited-stop sequence for a shortest plan, including the initial stop and every arrival "
        "through the final stop. Use 1-based stop numbers with one space after every comma. As a syntax-only example: "
        "`Answer: [1, 2, 3, 2, ...].`"
    ),
    "chess_pawn_swap": (
        "Return any shortest plan as ordered `(source (x, y), destination (x, y))` entries, using one space "
        "after every comma. As a syntax-only example: "
        "`Answer: [((5, 1), (4, 1)), ((1, 3), (2, 3)), ...].`"
    ),
    "color_group_with_locks": (
        "Return any shortest adjacent-swap plan as `Answer: [<left_slot>, ...].`, where each integer names the "
        "current left slot of the swapped pair, counted from 1 at the leftmost printed slot number. "
        "As a syntax-only example: `Answer: [2, 4, 3, ...].`"
    ),
    "container_ship_loading_order": (
        "Return the shortest plan as `Answer: [(<action>, <container_id>), ...].`, where each quoted action is "
        "`\"ship\"`, `\"buffer\"`, or `\"unbuffer\"`. As a syntax-only example: "
        '`Answer: [("buffer", 98), ("ship", 99), ("unbuffer", 98), ...].`'
    ),
    "dock_yard_crane_swap": (
        "Return a shortest plan as `Answer: [(<crane>, <source_lane>, <destination_lane>), ...].` "
        "Each quoted crane value is `\"L\"` for a two-box flip or `\"R\"` for a one-box move. "
        'As a syntax-only example: `Answer: [("L", 1, 3), ("R", 2, 1), ...].`'
    ),
    "crane_block_stack": (
        "Return the shortest plan as `Answer: [(<source_column>, <destination_column>), ...].` "
        "As a syntax-only example: `Answer: [(1, 3), (2, 1), ...].`"
    ),
    "filing_cabinet_alphabet_lock": (
        "Return the shortest plan as `Answer: [(<source_drawer>, <destination_drawer>), ...].`, using quoted "
        "uppercase drawer labels. As a syntax-only example: `Answer: [('A', 'C'), ('B', 'D'), ...].`"
    ),
    "jumping_frog_swap": (
        "Return a globally shortest list of current frog source indices as `Answer: [<source_index>, ...].` "
        "As a syntax-only example: `Answer: [2, 4, 3, ...].`"
    ),
    "lever_crane_swap": (
        "Return the shortest plan as `Answer: [<action>, ...].`; the complete quoted action vocabulary is `PL`, "
        "`PR`, `DL`, and `DR`. As a syntax-only example: `Answer: ['PL', 'DL', 'PR', ...].`"
    ),
    "office_inbox_priority_pull": (
        "Return the complete legal plan as `Answer: [(<action>, <paper_index>), ...].`, using the quoted lowercase "
        "action names defined by the rule. As a syntax-only example: "
        '`Answer: [("work", 98), ("buffer", 99), ...].`'
    ),
    "pancake_flip": (
        "Return any shortest flip plan as `Answer: [<prefix_length>, ...].` "
        "As a syntax-only example: `Answer: [3, 5, 2, ...].`"
    ),
    "parking_2d_n_by_n": (
        "Return any shortest plan as `(car_id, (x, y))` entries in move order, using one space after every comma. "
        "As a syntax-only example: `Answer: [(4, (1, 2)), (7, (2, 2)), ...].`"
    ),
    "parking_lot_swap_minimal_moves": (
        "Return any shortest move plan as `Answer: [<car_id>, ...].`, where each integer is the printed ID of the "
        "car moved into the currently empty slot. As a syntax-only example using out-of-range IDs: "
        "`Answer: [98, 99, 100, ...].`"
    ),
    "pipeline_hold_cells_3": (
        "Return a minimum 2N-action plan. The complete move vocabulary is `('push', 1)`, `('push', 2)`, "
        "`('push', 3)`, `('pop', 1)`, `('pop', 2)`, and `('pop', 3)`, where the integer names a hold-cell slot. "
        "As a syntax-only example: `Answer: [('push', 2), ('push', 1), ('pop', 1), ...].`"
    ),
    "meal_train_pass_serve": (
        "Return the complete visited-seat sequence for the unique shortest route, including the starting seat and "
        "every landing through the final seat. Use 1-based seat numbers with one space after every comma. "
        "As a syntax-only example: `Answer: [3, 4, 5, 4, ...].`"
    ),
    "stack_reversal_minimal_moves": (
        "Return any shortest legal move sequence as "
        "`Answer: [(<source_column>, <destination_column>), ...].` "
        "Number the columns 1 for the leftmost column onward. "
        "As a syntax-only example: `Answer: [(1, 3), (2, 1), ...].`"
    ),
    "token_trade_matrix": (
        "Return any shortest legal plan as `Answer: [(<endpoint_a>, <endpoint_b>), ...].` "
        "Name the two sockets by their printed position numbers 1 through 5, not by the tokens they hold. "
        "As a syntax-only example: `Answer: [(1, 4), (2, 5), ...].`"
    ),
    "train_couple_swap": (
        "Return a globally shortest plan as `Answer: [(<action>, <integer_value>), ...].`, where each quoted "
        "lowercase action is `\"park\"` or `\"place\"`. As a syntax-only example: "
        '`Answer: [("park", 98), ("place", 3), ...].`'
    ),
    "vending_dispense_fifo": (
        "Return the shortest plan as `Answer: [(<action>, <can_id>), ...].`, where each quoted lowercase action is "
        "`\"dispense\"` or `\"reinsert\"`. As a syntax-only example: "
        '`Answer: [("dispense", 98), ("reinsert", 98), ...].`'
    ),
    "tower_of_hanoi": (
        "Return the shortest plan as `Answer: [(<source_peg>, <destination_peg>), ...].` "
        "As a syntax-only example: `Answer: [(1, 3), (1, 2), ...].`"
    ),
}

_RULE_TEXT_OVERRIDES: dict[str, str] = {
    "dock_yard_crane_swap": (
        "The three lanes are numbered 1, 2, and 3 from left to right, and source must differ from destination. Crane `L` requires at "
        "least two source boxes; it moves the top pair together and reverses their vertical order on the destination. "
        "Crane `R` requires a nonempty source and moves only its top box without flipping. Every other box stays fixed."
    ),
    "jumping_frog_swap": (
        "Exactly one pad is empty. Green frogs move only right and orange frogs only left: a frog may step into the "
        "adjacent empty pad or jump two pads into it only over the opposite color. It cannot move backward, wrap, or "
        "land on an occupied pad. The source becomes empty and every other frog stays fixed. Match the target with a "
        "globally shortest plan; any shortest plan is accepted."
    ),
    "train_couple_swap": (
        "Every car appears once. `park` moves one named main-track car into the sole empty siding and closes the remaining "
        "main-track cars without reordering them. `place` inserts the parked car at a current boundary slot, where slot 1 "
        "is before the leftmost car and the last slot is after the rightmost car, and shifts later cars right. "
        "Actions alternate because parking into an occupied siding and placing from an empty one "
        "are illegal. Match the target with the siding empty using any globally shortest plan."
    ),
}

_INPUT_TEXT_OVERRIDES: dict[str, str] = {
    "dock_yard_crane_swap": (
        "The image shows the current dock yard on the left and the target on the right. Each yard has three vertical "
        "stack lanes numbered 1, 2, and 3 from left to right. Numbered boxes keep stable identities. A wide blue `L` "
        "marker denotes the two-box flipping crane; a slim red `R` denotes the one-box crane."
    ),
}


def _compact(text: Any, *, sentence_limit: int = 8) -> str:
    value = " ".join(str(text or "").split()).strip()
    value = re.sub(r"\b512\s*[x×]\s*512\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+\s*fps\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b4\s*N\s*\+\s*1\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,")
    sentences = re.split(r"(?<=[.!?])\s+", value)
    return " ".join(sentences[:sentence_limit]).strip()


_TASK_MOTION_CLAUSES: dict[str, str] = {
    "bookshelf_sort_by_height": (
        "Exchange exactly the two books named by each swap with continuous identity-preserving motion. Every unselected book stays in its current shelf slot until the swap commits."
    ),
    "bus_stops_dispatch": (
        "Drive the same bus continuously by one adjacent stop per action. On arrival, matching onboard riders move to their destination slots before waiting riders board in order; riders at all other stops stay fixed."
    ),
    "chess_pawn_swap": (
        "Move only the named pawn one legal row to its destination. On a diagonal capture, the moving pawn reaches the occupied square before the captured pawn disappears; every other pawn stays fixed."
    ),
    "color_group_with_locks": (
        "Swap only the two adjacent unlocked orbs named by the action, using continuous opposing motion. Locked orbs and every other slot remain fixed, and no orb crosses a lock."
    ),
    "container_ship_loading_order": (
        "Carry only the accessible dock-top or buffered container to the ship, buffer, or dock-top named by the action and commit on contact. The one-place buffer and all other containers retain their order without automatic compaction."
    ),
    "crane_block_stack": (
        "Lift only the source column's top block, carry that same block under the crane to the destination column, and place it on contact. Lower blocks and all other columns stay fixed."
    ),
    "dock_yard_crane_swap": (
        "For a one-box action, carry only the source top to the destination top. For a two-box action, carry the exact top pair together and visibly complete the required flip before both land; every other box stays fixed."
    ),
    "filing_cabinet_alphabet_lock": (
        "Pull only the rightmost paper from an unlocked source drawer, carry it to the unlocked destination drawer, and place it at that drawer's right end. Locked drawers and all other papers stay fixed."
    ),
    "jumping_frog_swap": (
        "Move only the frog at the submitted source index into the unique legal empty destination by a continuous one-slot slide or two-slot jump. The empty slot transfers to the source; every other frog stays fixed."
    ),
    "lever_crane_swap": (
        "A pick lifts only the chosen side's top block into the empty crane buffer; a drop lowers only the buffered block onto the chosen side. Lower stack blocks remain fixed and the buffer changes only at contact."
    ),
    "meal_train_pass_serve": (
        "Pass the same platter continuously from each submitted seat to the next adjacent submitted seat. On landing, serve that seat only when required; all guests, served markers, and other seats stay fixed."
    ),
    "office_inbox_priority_pull": (
        "Carry only the accessible top paper named by the action to the top of work or the temporary buffer and commit on contact. Removing it merely exposes the paper below; every other paper stays fixed."
    ),
    "pancake_flip": (
        "Lift the chosen top prefix as one stack, turn it continuously, and commit the complete reversed prefix together. Pancakes below the prefix remain fixed throughout."
    ),
    "parking_2d_n_by_n": (
        "Drive only the named car straight into the chosen orthogonally adjacent empty cell. Its source becomes the new empty cell at commit; all other cars and the second empty cell stay fixed."
    ),
    "parking_lot_swap_minimal_moves": (
        "Drive only the car adjacent to the empty slot straight into that slot. The vacated source becomes the new empty slot, and all other cars stay fixed."
    ),
    "pipeline_hold_cells_3": (
        "For PUSH, carry only the infeed front to the selected cell top; for POP, carry only that cell's top to the next outfeed slot. Commit on contact without compacting infeed, lower stack items, or earlier outfeed slots."
    ),
    "stack_reversal_minimal_moves": (
        "Lift only the source column's top block, carry it to the destination column, and place it on top. Lower blocks, other columns, and fixed slots never compact or shift automatically."
    ),
    "token_trade_matrix": (
        "Exchange exactly the two endpoint tokens along the submitted visible edge and commit both socket occupancies together. Every other token, socket, edge, and position label stays fixed."
    ),
    "tower_of_hanoi": (
        "Lift only the source peg's top disk, carry it without resizing to the destination peg, and place it on the legal top surface. Every other disk and peg remains fixed."
    ),
    "train_couple_swap": (
        "A park action carries only the named main-track car into the empty siding while main-track survivors close the vacated position without reordering. A place action carries the parked car into the named boundary slot while affected survivors shift together."
    ),
    "vending_dispense_fifo": (
        "DISPENSE moves only the chute front to the tray's right end and closes the chute queue without reordering. REINSERT moves only the named tray can to the chute tail; all other cans retain relative order."
    ),
}


# Per-task TI2V continuity nouns: the shared final
# paragraph is built from each task's own visible vocabulary, never from a
# cross-task fallback noun list.  Every Rearrangement task must have an entry.
_TASK_CONTINUITY_NOUNS: dict[str, tuple[str, str]] = {
    "bookshelf_sort_by_height": (
        "the two books being exchanged",
        "every other book, the shelf, the slot numbers, and the target panel",
    ),
    "bus_stops_dispatch": (
        "the bus and the riders who board or alight at its current stop",
        "riders waiting at other stops, the stop layout, the stop numbers, and the target panel",
    ),
    "chess_pawn_swap": (
        "the single pawn that steps or captures",
        "every other pawn, the board grid, and the target panel",
    ),
    "color_group_with_locks": (
        "the two adjacent unlocked orbs being swapped",
        "the locked orbs, every other orb, the slot numbers, and the target panel",
    ),
    "container_ship_loading_order": (
        "the single container being carried",
        "every other container, the dock, the buffer, the ship, and the target panel",
    ),
    "crane_block_stack": (
        "the crane trolley and the single block it carries",
        "every other block, the column rails, the gantry, and the target panel",
    ),
    "dock_yard_crane_swap": (
        "the carried box or the flipped pair under the active crane",
        "every other box, the three lanes, the L and R crane markers, and the target panel",
    ),
    "filing_cabinet_alphabet_lock": (
        "the single paper being refiled",
        "every other paper, the drawers, the lock badges, and the target panel",
    ),
    "jumping_frog_swap": (
        "the single frog that slides or jumps",
        "every other frog, the row of pads, the pad numbers, and the target panel",
    ),
    "lever_crane_swap": (
        "the lever arm and the single block it picks or drops",
        "every other block, both stack platforms, and the target panel",
    ),
    "maze_rush_hour": (
        "the single sliding vehicle and the badge riding on it",
        "every other vehicle, the lot grid, the exit marker, and the other badges",
    ),
    "meal_train_pass_serve": (
        "the platter and the marker of the seat it has just served",
        "every other seat, both tables, and the target panel",
    ),
    "office_inbox_priority_pull": (
        "the single paper being pulled",
        "every other paper and the three lane outlines",
    ),
    "pancake_flip": (
        "the flipped top prefix of pancakes",
        "the pancakes below the prefix, both plates, and the target panel",
    ),
    "parking_2d_n_by_n": (
        "the single car that drives one cell",
        "every other car, the lot grid, and the target panel",
    ),
    "parking_lot_swap_minimal_moves": (
        "the single car that drives into the empty slot",
        "every other car, the slot row, and the target panel",
    ),
    "pipeline_hold_cells_3": (
        "the single product being pushed or popped",
        "every other product, the infeed, the hold cells, the outfeed, and the target panel",
    ),
    "sliding_tile_15": (
        "the single tile sliding into the blank",
        "every other tile and the board frame",
    ),
    "stack_reversal_minimal_moves": (
        "the single top block being carried",
        "every lower block, the other columns, the slots, and the target panel",
    ),
    "token_trade_matrix": (
        "the two endpoint tokens being exchanged",
        "every other token, the sockets, the edges, the position numbers, and the target panel",
    ),
    "tower_of_hanoi": (
        "the single carried disk",
        "every other disk, the pegs, the base, and the target panel",
    ),
    "train_couple_swap": (
        "the parked or placed car and the cars that close or shift around it",
        "every other car, the main track, the siding, and the target panel",
    ),
    "vending_dispense_fifo": (
        "the single dispensed or reinserted can",
        "every other can, the chute, the tray, and the target panel",
    ),
}


def _continuity_nouns(task_name: str) -> tuple[str, str]:
    try:
        return _TASK_CONTINUITY_NOUNS[task_name]
    except KeyError as exc:
        raise ValueError(f"{task_name} lacks task-owned TI2V continuity nouns") from exc


def _motion_clause(task_name: str) -> str:
    try:
        return _TASK_MOTION_CLAUSES[task_name]
    except KeyError as exc:
        raise ValueError(f"{task_name} lacks a task-owned motion clause") from exc


def prompts_from_generator(generator: Any) -> dict[str, str]:
    """Build compact prompts from a task generator's semantic hooks."""

    task_name = str(generator.TASK_NAME)
    input_text = _INPUT_TEXT_OVERRIDES.get(
        task_name,
        _compact(generator._describe_input(), sentence_limit=5),
    )
    rule_text = _RULE_TEXT_OVERRIDES.get(
        task_name,
        _compact(generator._describe_rules(), sentence_limit=9),
    )
    typed_output = _TYPED_OUTPUT_OVERRIDES.get(
        task_name,
        _compact(generator._describe_output(), sentence_limit=4),
    )
    if not input_text or not rule_text or not typed_output:
        raise ValueError(f"{task_name} has an incomplete prompt semantic hook")

    is_spatial_navigation = str(getattr(generator, "TASK_FAMILY", "")) == "spatial_navigation"
    if is_spatial_navigation:
        video_output = (
            f"Replay a complete legal solution. Show successive complete grid states for {_motion_clause(task_name)}"
        )
        video_continuity = build_ti2v_continuity_paragraph(
            timing="discrete",
            moving_nouns="the agent and every object the rules move",
            static_nouns="everything the rules leave unchanged, including the grid, labels, background, and camera",
            hold_noun="the solved final state",
        )
    else:
        moving_nouns, static_nouns = _continuity_nouns(task_name)
        video_output = f"Replay a complete legal solution. {_motion_clause(task_name)}"
        video_continuity = build_ti2v_continuity_paragraph(
            timing="continuous",
            moving_nouns=moving_nouns,
            static_nouns=f"everything the rules leave unchanged, including {static_nouns}, the background, and the camera",
            hold_noun="the solved final state",
        )
    return {
        "ti2v_prompt": build_sectioned_prompt(
            input_text,
            video_output,
            mode="ti2v",
            rule_description=rule_text,
            continuity_paragraph=video_continuity,
        ),
    }


def _special_prompts(task_name: str) -> dict[str, str]:
    if task_name == "sliding_tile_15":
        input_text = (
            "The image shows a 4 by 4 numbered sliding-tile board with one blank slot. "
            f"Tile numbers identify the pieces and their current cells. {GRID_XY_CONVENTION_SENTENCE}"
        )
        rule_text = (
            "A move swaps the blank with one orthogonally adjacent tile. The goal is row-major order 1 through 15 "
            "with the blank at bottom right, using the unique shortest sequence."
        )
        video = (
            "Replay the shortest solution. Slide one adjacent tile straight into the blank, commit only when it reaches "
            "the empty cell, and keep every other tile fixed."
        )
    elif task_name == "maze_rush_hour":
        input_text = (
            "The image shows a 6 by 6 lot of rigid vehicles and a marked exit on the red vehicle's row. Each "
            "vehicle carries one C-numbered badge, and its orientation, length, and occupied cells are visible. "
            "A vehicle's head is its top-left occupied grid position. "
            f"{GRID_XY_CONVENTION_SENTENCE}"
        )
        rule_text = (
            "A vehicle may slide only along its own axis through empty cells. It cannot rotate, overlap another vehicle, "
            "or leave the lot; only the red vehicle may finish fully through the exit. Use a shortest legal plan."
        )
        video = (
            "Replay the shortest plan with one rigid vehicle sliding continuously per move, its badge riding with it. "
            "Show contact-free travel and a stable commit after every slide; keep all unselected vehicles, the grid, "
            "exit, and badges fixed."
        )
    else:
        raise ValueError(f"no special Rearrangement prompt for {task_name!r}")
    moving_nouns, static_nouns = _continuity_nouns(task_name)
    video_continuity = build_ti2v_continuity_paragraph(
        timing="continuous",
        moving_nouns=moving_nouns,
        static_nouns=f"everything the rules leave unchanged, including {static_nouns}, the background, and the camera",
        hold_noun="the solved final state",
    )
    return {
        "ti2v_prompt": build_sectioned_prompt(
            input_text,
            video,
            mode="ti2v",
            rule_description=rule_text,
            continuity_paragraph=video_continuity,
        ),
    }


def _new_generator(task_name: str) -> Any:
    spec = TASK_SPECS[task_name]
    cls = getattr(importlib.import_module(spec.module), spec.class_name)
    signature = inspect.signature(cls)
    output = Path(tempfile.gettempdir()) / "videogym-rearrangement-prompt" / task_name
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = output
    else:
        args.append(output)
    if "seed" in signature.parameters:
        kwargs["seed"] = 0
    if "video" in signature.parameters:
        kwargs["video"] = False
    return cls(*args, **kwargs)


def normalize_rearrangement_prompts(record: Mapping[str, Any]) -> dict[str, Any]:
    """Publish task-specific prompts and their semantic delivery contract."""

    annotated = dict(record)
    task_name = str(annotated.get("task_name") or "")
    if task_name in {"sliding_tile_15", "maze_rush_hour"}:
        prompts = _special_prompts(task_name)
    else:
        prompts = prompts_from_generator(_new_generator(task_name))
    annotated.update(prompts)
    return annotated


__all__ = [
    "normalize_rearrangement_prompts",
    "prompts_from_generator",
]
