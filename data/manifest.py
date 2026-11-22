from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Optional, Sequence

from data.common.canonical_answers import parse_canonical_answer_line
from data.common.prompts import (
    remove_model_facing_serialization_recipes,
    soften_precise_video_timing,
)
from data.common.schemas import CanonicalSample
from data.common.io import write_jsonl
from data.common.media_integrity import (
    DeclaredArtifactError,
    verify_declared_images,
    verify_video_artifact,
)
from data.common.paths import to_absolute
from data.common.task_semantics import normalize_split
from data.common.typed_answers import parse_answer_line
from data.registry import get_task_spec
from data.common.sample_identity import build_canonical_sample_id
from data.common.ti2v_continuity import looks_like_ti2v_continuity_paragraph
from data.common.video_encoding import validate_video_contract

def _format_prompt_sections(text: str) -> str:
    result = str(text or "").strip()
    for section_label in ("Input Image:", "Rule:", "Required Output:"):
        result = re.sub(
            rf"\s*{re.escape(section_label)}\s*",
            f"\n\n{section_label} ",
            result,
            count=1,
            flags=re.IGNORECASE,
        )
    return result.strip()


def _extract_canonical_prompt(record: Dict) -> str:
    raw = record.get("ti2v_prompt")
    if raw is None:
        raw = record.get("prompt")
    text = str(raw).strip() if raw is not None else ""
    if not text:
        raise ValueError("canonical sample requires a non-empty prompt")
    # The shared continuity paragraph is an unlabeled final block.
    # The whitespace-collapsing normalizers below would fuse it into the
    # Required Output body, so detach it first and re-append it verbatim.
    continuity = ""
    if "\n\n" in text:
        head, _, tail = text.rpartition("\n\n")
        if looks_like_ti2v_continuity_paragraph(tail):
            text = head
            continuity = " ".join(tail.split())
    text = soften_precise_video_timing(text)
    text = remove_model_facing_serialization_recipes(text)
    text = _format_prompt_sections(text)
    if continuity:
        text = f"{text}\n\n{continuity}" if text else continuity
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if not blocks or not looks_like_ti2v_continuity_paragraph(blocks[-1]):
        raise ValueError(
            "canonical prompt must end with the shared unlabeled continuity paragraph"
        )
    return text


def _terminal_answer_line(raw_answer: object) -> Optional[str]:
    if not isinstance(raw_answer, str):
        return None
    normalized = raw_answer.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    if not normalized:
        return None
    answer_line = normalized.rsplit("\n", 1)[-1]
    if parse_answer_line(answer_line) is None:
        return None
    return answer_line


def _validate_answer_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            "formal Answer text must be an exact non-empty string with no "
            "outer whitespace"
        )
    return value


_NON_SEMANTIC_RECORD_KEYS = {
    "_reasoning_option_tiles",
    "answer",
    "answer_payload",
    "answer_grid",
    "correct_option",
    "dependency_contract",
    "image",
    "images",
    "input_media_policy",
    "option_image_paths",
    "option_images",
    "plot_level",
    "reasoning_image",
    "reasoning_images",
    "seed",
    "solution_image_path",
    "solution_video_path",
    "ti2v_input_images",
    "ti2v_prompt",
    "solution_image_prompt",
    "structured_answer",
    "task_contract",
    "video",
    "video_keyframe_paths",
    "verifier_contract",
    "rendered_glyph_inventory",
    "rendered_glyph_schema",
}

_DERIVED_TRACE_KEYS = {
    "action_sequence",
    "agent_path",
    "answer_state",
    "brace_steps",
    "canonical_path",
    "cached_terminal_is_not_oracle",
    "cart_path",
    "cell_state_sequence",
    "checkpoint_sequence",
    "event_indices",
    "event_steps",
    "expected_path",
    "expected_state",
    "final_board",
    "final_grid",
    "final_state",
    "terminal_state",
    "intermediate_states",
    "move_count",
    "moves_canonical",
    "moves",
    "move_sequence",
    "path",
    "optimal_path",
    "optimal_moves",
    "path_cells",
    "path_cell_ids",
    "path_length",
    "primary_moves",
    "primary_path",
    "secondary_moves",
    "secondary_path",
    "shortest_path",
    "shortest_length",
    "solution_grid",
    "solution_mask",
    "solution_path_cell_ids",
    "solution_paths",
    "state_trace",
    "state_grid",
    "boundary_path_cell_count",
    "wrap_jump_count",
    "uses_wraparound",
    "mechanic_disabled_path_cell_ids",
    "mechanic_detour_cells",
    "lock_chain_order",
    "min_solution_path_length_cells",
    "ti2t_state_answer",
}


_MANIFEST_CONTRACT_KEYS = {
    "answer_schema",
    "intermediate_supervision_schema",
    "verifier_component_scores",
    "verifier_contract",
    "verifier_profile",
}

# These fields are useful authoring diagnostics, but are recomputable outputs
# rather than visible source semantics.  Keep the list task-scoped so names
# such as ``rule`` or ``anchor`` remain available to unrelated tasks where
# they genuinely describe the input.
_ICL_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "alternating_size_sequence": {"correct_height", "heights"},
    "analogy_completion": {"matching_transforms", "transform"},
    "dot_count_progression": {"correct_count", "matching_rule_families", "rule", "sequence"},
    "growing_polygon_step": {"correct_count", "rule", "sequence", "step"},
    "hue": {"board_colors", "board_names", "col_step", "palette_cycle_direction", "row_step", "rule"},
    "increasing_size_sequence": {"correct_height", "heights", "step"},
    "legend_operator_program": {"operator_by_token", "output_matrix", "reversed_program_output"},
    "linear_sequence_next": {"correct_cells", "frames", "rule"},
    "object_subtraction": {"answer_index", "areas", "candidate_removed_slots", "removed_slot", "target_after", "target_areas", "target_removed_slot"},
    "odd_one_out_shape": {"matching_indices", "property_value", "target_value"},
    "rotate_matrix": {"output_matrix", "rotation_states"},
    "sequence_completion_poly": {"answer_transform", "step_transform"},
    "set_inclusion_diagram": {"answer_index", "candidate_values_by_label", "distractor_values", "relation", "shared_elements", "x_only", "y_only"},
    "spiral_matrix": {"spiral_coordinates", "spiral_order", "value_count"},
    "tile_color_sequence": {"colors", "correct_color_name", "cycle_length", "pattern"},
    "triple_intersection_red": {"anchor", "centroid", "circle_contributions", "pairwise_area", "pairwise_bbox", "radius", "rotation_quarters", "template", "triple_area", "triple_bbox"},
    "venn_intersection_count": {"counts", "distractor_counts", "distractor_values", "intersection_rank_among_disjoint_regions", "placement_seed"},
    "venn_symmetric_difference": {"all_regions", "candidate_values_by_label", "distractor_values"},
}


# Constraint-grid generators author one semantic record and historically put
# both the visible puzzle and the derived witness in ``structured_answer``.
# Canonical ``semantic_input`` is the verifier's source domain, so retaining
# the witness half here would make an otherwise independent replay silently
# depend on saved answers.  Blank locations, givens, clues, regions, option
# panels, and renderer geometry remain; solved fills, placements, paths,
# edges, hidden values, answer labels, and authored step order do not.
_CONSTRAINT_GRID_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "aquarium_completion": {"fill"},
    "campsite_completion": {"tents"},
    "fillomino_completion": {"solution"},
    "light_up_akari_completion": {"lights"},
    "masyu_completion": {"loop_path"},
    "maze_flow_pairs": {"solution_paths"},
    "nurikabe_completion": {"solution"},
    "shikaku_completion": {"rectangles"},
    "slitherlink_completion": {"edges", "inside_mask"},
    "star_battle_completion": {"difficulty_variant", "stars"},
    "yajilin_completion": {"black_cells", "loop_path"},
    "yin_yang_completion": {"solution"},
}


# Board-game raw records historically placed the visible position, authored
# answer, terminal state, and replay trace in one ``structured_answer`` map.
# Canonical semantic_input must expose only the position and visibly marked
# action/source; every verdict, path, capture set, component, terminal grid,
# and authored order is independently recomputed by the family verifier.
_BOARD_GAME_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "mancala_distribute_predict": {
        "distribution_destinations", "initial_grid",
    },
}


# Dynamics records share their visible initial scene with authoring-only
# rollout results.  Independently replayable terminal states, counters, and
# authored trajectories never belong to ``semantic_input``.
_DYNAMICS_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "lemmings_path_predict": {"final", "final_direction"},
    "merge_swipe_score": {"merge_count", "round_merge_counts"},
    "sokoban": {
        "difficulty_variant",
        "move_sequence",
        "open_board_path_length",
        "path_length",
        "push_count",
        "state_trace",
        "wall_detour_extra_steps",
    },
    "tetris_future_board": {"cleared_rows", "landing_grid", "landing_row", "locked_grid"},
    "track_match_shot": {"clear_run", "filled_track"},
}


# Next-state generators retain rich reference-authoring traces in their raw
# records.  Only the visible initial state and explicit rule parameters belong
# in canonical semantic_input; terminal grids, counts, component decompositions
# and timelines are independently replayed by the final-delivery verifier.
_NEXT_STATE_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "arc1d_compact_nonzero": {"compaction_states", "output_row", "rounds"},
    "binary_matrix": {
        "distance_matrix", "difficulty_variant", "influential_source_count",
        "max_distance", "source_count",
    },
    "connected_components": {
        "component_count", "component_order_is_semantic", "component_sizes",
        "component_states", "components", "filled_cell_count",
    },
    "grid_bfs": {
        "difficulty_variant", "distance_grid", "max_distance",
        "obstacle_ablation_changed_cells", "source_count", "unreachable_count",
        "wall_count", "wave_layer_count",
    },
    "grid_component": {
        "labels",
        "component_sizes", "components", "largest_component_label",
        "largest_component_size", "unique_largest",
    },
    "grid_shift": {
        "answer_matrix", "removed_values", "shift_states", "surviving_values",
        "target_matrix",
    },
    "rotten_oranges": {"minute_grid", "minutes"},
    "traffic_light_countdown": {"final_states", "timeline", "transition_count"},
}


# Spatial Navigation records intentionally carry rich authored trajectories so the
# reference builder can render cumulative stills and continuous videos.  None
# of those trajectories belongs to the verifier's source domain.  The source
# contract is the visible board, the explicit rule parameters, and the initial
# object state; candidate paths and terminal states are independently replayed.
_SPATIAL_NAVIGATION_COMMON_DERIVED_SOURCE_KEYS = {
    "agent_path_cell_ids",
    "target_state",
    "max_total_cost",
    "push_count",
    "difficulty_variant",
    "open_board_path_length",
    "wall_detour_extra_steps",
}

_SPATIAL_NAVIGATION_DERIVED_SOURCE_KEYS: Dict[str, set[str]] = {
    "directional_arrow_entry": {"minimum_arrow_commits"},
    "agent_courier_relay": {"leg1", "leg2", "leg1_len"},
    "bridge_drop_tile": {"drops"},
    "cooperative_open_door": {"press_step", "cross_step"},
    "heading_turn_budget_route": {"transform"},
    "lever_unlock_sequence": {"lever_steps"},
    "one_way_portal_chain": {"teleport_steps"},
    "switch_toggle_walls": {"switch_step"},
}


def _strip_task_derived_semantics(value: object, blocked: set[str]) -> object:
    if isinstance(value, dict):
        return {
            str(key): _strip_task_derived_semantics(item, blocked)
            for key, item in value.items()
            if str(key).lower() not in blocked and str(key).lower() not in _MANIFEST_CONTRACT_KEYS
        }
    if isinstance(value, list):
        return [_strip_task_derived_semantics(item, blocked) for item in value]
    return value


def _sanitize_semantic_value(value: object) -> object:
    if isinstance(value, dict):
        cleaned: Dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _NON_SEMANTIC_RECORD_KEYS or lowered in _DERIVED_TRACE_KEYS:
                continue
            # Palette constants describe how a visible artifact is encoded;
            # despite their historical names they are not saved answers.
            # Keep them so a verifier can parse the candidate artifact from
            # source-side geometry without falling back to ground truth.
            visible_palette_key = lowered in {"correct_badge_rgb", "correct_rgb"}
            if (
                lowered.startswith(("canonical_", "correct_", "expected_", "optimal_", "reference_", "solution_"))
                and not visible_palette_key
            ):
                continue
            if lowered.endswith(("_answer", "_keyframes", "_trace")):
                continue
            sanitized = _sanitize_semantic_value(item)
            # Empty semantic containers are state, not absence.  A queue, tray,
            # buffer, picked set, or exit list can be empty in the initial
            # state and later become load-bearing during independent replay.
            # Dropping those values forces verifiers to recover task-specific
            # defaults (or, worse, consult a saved solution) and makes the
            # canonical manifest cease to be an executable semantic contract.
            cleaned[key] = sanitized
        return cleaned
    if isinstance(value, list):
        return [_sanitize_semantic_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_semantic_value(item) for item in value]
    return value


def _extract_semantic_input(record: Dict) -> Dict[str, object]:
    task_name = str(record.get("task_name") or "")
    task_family = str(record.get("task_family") or "")
    if not task_family:
        task_spec = get_task_spec(task_name)
        task_family = str(task_spec.task_family if task_spec is not None else "")
    structured = record.get("structured_answer") or record.get("answer_payload")
    # Evaluation generators author a source-only envelope. Their new goals
    # share families with training tasks but are not governed by the latter's
    # task-specific source allowlists.
    from data.registry import TASK_SPECS
    if task_name not in TASK_SPECS:
        if not isinstance(structured, dict):
            raise ValueError("evaluation generator requires a source-only envelope")
        return {"semantic_state": _sanitize_semantic_value(structured)}
    if task_family == "geometry":
        from data.geometry.core.canonical_source import canonical_geometry_semantic_input

        return canonical_geometry_semantic_input(
            task_name,
            record,
            structured if isinstance(structured, dict) else {},
        )
    if task_name == "maze_aqueduct_pipes":
        # Aqueduct records retain solved rotations and flow schedules beside
        # their raw source, so the family keeps an explicit allowlist; the
        # PathTaskRecord stores its visible grid source at the record root.
        from data.spatial_navigation.core.canonical_source import (
            canonical_aqueduct_semantic_input,
        )

        return canonical_aqueduct_semantic_input(
            structured if isinstance(structured, dict) else record
        )
    if task_family == "physics":
        if not isinstance(structured, dict):
            raise ValueError(
                f"Physics task {task_name!r} must expose a structured source record"
            )
        # Physics reference authoring stores solved outcomes, trajectories,
        # and render schedules beside its raw source.  A family-owned
        # allowlist is the only safe canonical boundary: generic key-name
        # scrubbing cannot distinguish a visible detector from a saved target
        # or a source rotation from a corrected one.
        from data.physics.core.canonical_source import canonical_physics_semantic_input

        return canonical_physics_semantic_input(task_name, structured)
    if task_family == "rearrangement_game":
        # Rearrangement records keep authored paths, target schedules,
        # selected subsets, state traces, shortest-path diagnostics, and
        # reference-render metadata beside the two visible panels.  A
        # family-owned allowlist is the only safe boundary; generic key-name
        # scrubbing cannot tell a visible goal panel from a saved solution
        # schedule or a raw capacity from an optimal packing diagnostic.
        from data.rearrangement_game.core.canonical_source import (
            canonical_rearrangement_semantic_input,
        )

        return canonical_rearrangement_semantic_input(task_name, record)
    if task_family == "spatial_3d":
        if not isinstance(structured, dict):
            raise ValueError(
                f"Spatial 3D task {task_name!r} must expose a structured source record"
            )
        # Spatial 3D reference authoring stores unfolded matrices, state
        # chains, pose orbits, and view grids beside the raw source.  A
        # family-owned allowlist is the only safe canonical boundary.
        from data.spatial_3d.core.canonical_source import (
            canonical_spatial_3d_semantic_input,
        )

        return canonical_spatial_3d_semantic_input(task_name, structured)

    cleaned = _sanitize_semantic_value(record)
    if not isinstance(cleaned, dict):
        return {}
    if isinstance(structured, dict):
        # Board Game tasks that expose an explicit source-only semantic_state
        # mapping use it directly (after generic sanitization); other tasks use
        # the task-aware derived-key deny-list below.
        explicit_board_state = (
            task_family == "board_game"
            and isinstance(structured.get("semantic_state"), dict)
        )
        semantic_state = _sanitize_semantic_value(
            structured["semantic_state"] if explicit_board_state else structured
        )
        blocked = (
            _ICL_DERIVED_SOURCE_KEYS.get(task_name, set())
            | _BOARD_GAME_DERIVED_SOURCE_KEYS.get(task_name, set())
            | _CONSTRAINT_GRID_DERIVED_SOURCE_KEYS.get(task_name, set())
            | _DYNAMICS_DERIVED_SOURCE_KEYS.get(task_name, set())
            | _NEXT_STATE_DERIVED_SOURCE_KEYS.get(task_name, set())
            | (
                _SPATIAL_NAVIGATION_COMMON_DERIVED_SOURCE_KEYS
                | _SPATIAL_NAVIGATION_DERIVED_SOURCE_KEYS.get(task_name, set())
                if task_family == "spatial_navigation"
                else set()
            )
        )
        semantic_state = _strip_task_derived_semantics(semantic_state, blocked)
        if isinstance(semantic_state, dict) and semantic_state:
            cleaned["semantic_state"] = semantic_state
        if task_family == "spatial_navigation":
            # Spatial Navigation generators keep authored paths and solved render state
            # at several nesting levels for reference production.  Apply the
            # same task-aware deny-list to the complete canonical source, not
            # only the structured semantic_state copy, so a top-level saved
            # target or trace cannot silently survive canonicalization.
            cleaned = _strip_task_derived_semantics(cleaned, blocked)
    if task_family == "spatial_navigation":
        # Some object-plan Spatial Navigation records expose their visible state at the
        # record root instead of under structured_answer.  They still carry a
        # top-level authored target_state for reference rendering, so enforce
        # the family deny-list even when no structured envelope was present.
        cleaned = _strip_task_derived_semantics(
            cleaned,
            _SPATIAL_NAVIGATION_COMMON_DERIVED_SOURCE_KEYS
            | _SPATIAL_NAVIGATION_DERIVED_SOURCE_KEYS.get(task_name, set()),
        )
    for key in (
        "id",
        "question",
        "question_description",
        "question_id",
        "qa_level",
        "qa_type",
        "split",
        "task_family",
        "task_name",
        "title",
        "video_fps",
        "video_num_frames",
    ):
        cleaned.pop(key, None)
    if str(record.get("task_family") or "") == "board_game":
        # Board Game canonical source is deliberately minimal.  Structured
        # tasks need only the visible semantic_state.  Difficulty labels,
        # duplicated source/target fields, and authored output envelopes are
        # not executable task semantics.
        semantic_state = cleaned.get("semantic_state")
        cleaned = {"semantic_state": semantic_state} if isinstance(semantic_state, dict) else {}
    if task_family == "in_context_learning":
        # ICL generators keep renderer geometry, authored process schedules,
        # options, and answer-bearing diagnostics beside the visible semantic
        # primitives.  Candidate verification owns the task route already, so
        # the canonical source needs only the scrubbed visible state.  This
        # prevents a malformed/missing source from falling back to duplicated
        # top-level metadata or a saved option.
        semantic_state = cleaned.get("semantic_state")
        cleaned = {"semantic_state": semantic_state} if isinstance(semantic_state, dict) else {}
    return cleaned


def build_canonical_sample(
    record: Dict,
    *,
    task_dir: Path,
    source_task_name: str,
    seed: int,
    split: Optional[str] = None,
    record_index: Optional[int] = None,
    video_requested: bool = True,
) -> Optional[CanonicalSample]:
    """Build one canonical sample."""
    if type(seed) is not int:
        raise ValueError("build_canonical_sample requires an exact integer seed")
    task_name = str(record.get("task_name") or source_task_name)
    task_spec = get_task_spec(task_name)
    if task_spec is None:
        return None

    task_family = task_spec.task_family
    resolved_split = normalize_split(split if split is not None else record.get("split"), default="train")
    diagnostic_sample_id = (
        f"{task_name}@record-seed-{seed}"
        if record_index is None
        else f"{task_name}@record-{record_index:05d}-seed-{seed}"
    )
    prompt = _extract_canonical_prompt(record)

    image_value = record.get("image")
    solution_image_value = record.get("solution_image_path")
    if not image_value or not solution_image_value:
        return None

    puzzle_image = to_absolute(str(image_value), task_dir).as_posix()
    solution_image = to_absolute(str(solution_image_value), task_dir).as_posix()
    media_roles = [(puzzle_image, "primary input"), (solution_image, "terminal image")]
    checked_assets: set[str] = set()
    verify_declared_images(
        media_roles,
        task_name=task_name,
        sample_id=diagnostic_sample_id,
        already_verified=checked_assets,
    )

    solution_video = None
    video_contract: Optional[Dict[str, object]] = None
    raw_video = record.get("solution_video_path") or record.get("video")
    if raw_video:
        solution_video = to_absolute(str(raw_video), task_dir).as_posix()
        decoded_frames = verify_video_artifact(
            solution_video,
            task_name=task_name,
            sample_id=diagnostic_sample_id,
            role="solution video",
        )
        video_contract = validate_video_contract(Path(solution_video))
        if decoded_frames != int(video_contract["frame_count"]):
            raise DeclaredArtifactError(
                f"{task_name} sample {diagnostic_sample_id}: declared solution video decodes to "
                f"{decoded_frames} frames but its container header claims "
                f"{int(video_contract['frame_count'])}: {solution_video}"
            )
        declared_fps = record.get("video_fps")
        declared_frames = record.get("video_num_frames")
        if declared_fps is not None and int(declared_fps) != int(round(float(video_contract["fps"]))):
            raise ValueError(f"{task_name} video_fps metadata does not match the encoded asset")
        if declared_frames is not None and int(declared_frames) != int(video_contract["frame_count"]):
            raise ValueError(f"{task_name} video_num_frames metadata does not match the encoded asset")
    if video_requested and not solution_video:
        raise DeclaredArtifactError(
            f"{task_name} sample {diagnostic_sample_id}: video was requested but no "
            "solution video was produced; the encode failed or the asset was lost"
        )

    raw_answer_text = _validate_answer_text(record.get("answer"))
    if raw_answer_text is None:
        raise ValueError(f"{task_name} has no canonical Answer payload")
    answer_line = _terminal_answer_line(raw_answer_text)
    if answer_line is None:
        raise ValueError(f"{task_name} answer has no canonical final Answer line")
    answer = parse_canonical_answer_line(
        answer_line,
        kind=task_spec.answer_kind,
        choice_codec=task_spec.answer_codec,
    )

    semantic_input = _extract_semantic_input(record)
    sample_id = build_canonical_sample_id(
        task_name=task_name,
        seed=seed,
        semantic_input=semantic_input,
        ground_truth=answer.to_dict(),
        prompt=prompt,
        image=puzzle_image,
    )

    return CanonicalSample(
        id=sample_id,
        task_family=task_family,
        task_name=task_name,
        split=resolved_split,
        prompt=prompt,
        image=puzzle_image,
        video=solution_video,
        video_fps=int(round(float(video_contract["fps"]))) if video_contract else None,
        video_num_frames=int(video_contract["frame_count"]) if video_contract else None,
        seed=seed,
        ground_truth=answer,
        semantic_input=semantic_input,
    )


def write_manifest(path: Path, samples: Sequence[CanonicalSample]) -> None:
    base_dir = path.expanduser().resolve().parent
    write_jsonl(path, [sample.to_dict(base_dir=base_dir) for sample in samples])
