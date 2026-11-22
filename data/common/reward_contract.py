from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, Optional
from data.common.verifiers.keyframes import VERIFIER_PROTOCOL


RAW_REWARD_SCHEMA_VERSION = "videogym.raw_reward.v2"


@dataclass(frozen=True)
class RewardDimension:
    name: str
    semantic: str


RAW_REWARD_DIMENSIONS: tuple[RewardDimension, ...] = (
    RewardDimension(
        "whole_video_process",
        "Task-relevant key states preserve object identity and execute required legal transitions in order, with tolerance for codec noise and interpolation.",
    ),
    RewardDimension(
        "video_final_state",
        "The complete candidate video's final hold encodes an independently derived legal terminal state.",
    ),
    RewardDimension(
        "media_contract",
        "Every artifact required by the formal video delivery is present, decodable, and satisfies its media contract.",
    ),
)

RAW_REWARD_DIMENSION_NAMES: tuple[str, ...] = tuple(
    dimension.name for dimension in RAW_REWARD_DIMENSIONS
)

RAW_REWARD_HARD_GATES: tuple[str, ...] = (
    "whole_video_process",
    "video_final_state",
    "media_contract",
)

_LEGACY_REWARD_KEYS = frozenset(
    {
        "component_scores",
        "component_scores_by_artifact",
        "custom_reward_components",
        "reward_components",
        "reward_components_by_artifact",
        "reward_masks",
        "reward_masks_by_artifact",
        "raw_reward_vector",
        "reward_availability_mask",
        "reward_hard_gates",
        "reward_schema",
        "reward_schema_version",
        "verifier_component_scores",
    }
)


def _is_legacy_reward_key(key: object) -> bool:
    lowered = str(key).lower()
    return "custom_reward" in lowered or lowered in _LEGACY_REWARD_KEYS


def build_reward_schema() -> dict[str, object]:
    """Build the runtime raw-reward contract."""

    return {
        "version": RAW_REWARD_SCHEMA_VERSION,
        "dimensions": [
            {
                "name": dimension.name,
                "semantic": dimension.semantic,
                "range": [0.0, 1.0],
            }
            for dimension in RAW_REWARD_DIMENSIONS
        ],
        "hard_gates": list(RAW_REWARD_HARD_GATES),
        "diagnostics_are_rewards": False,
    }


def _iter_metric_items(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str, Any]]:
    if not isinstance(value, Mapping):
        return
    for raw_key, child in value.items():
        key = str(raw_key)
        lowered = key.lower()
        if _is_legacy_reward_key(lowered):
            continue
        yield path, lowered, child
        if isinstance(child, Mapping):
            yield from _iter_metric_items(child, path=(*path, lowered))


def _score_from_metrics(
    metrics: Mapping[str, Any],
    aliases: Sequence[str],
) -> Optional[float]:
    indexed: dict[str, list[Any]] = {}
    for _path, key, value in _iter_metric_items(metrics):
        indexed.setdefault(key, []).append(value)
    for alias in aliases:
        for value in indexed.get(alias, ()):
            if isinstance(value, bool):
                return float(value)
            if isinstance(value, (int, float)):
                numeric = float(value)
                if numeric == numeric:
                    return max(0.0, min(1.0, numeric))
    return None


def _direct_score_from_metrics(
    metrics: Mapping[str, Any],
    aliases: Sequence[str],
) -> Optional[float]:
    """Read only direct keys so one artifact cannot borrow another's score."""

    normalized = {str(key).lower(): value for key, value in metrics.items()}
    for alias in aliases:
        value = normalized.get(str(alias).lower())
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric == numeric:
                return max(0.0, min(1.0, numeric))
    return None


def _artifact_sections(
    metrics: Mapping[str, Any],
    names: Sequence[str],
) -> Iterable[Mapping[str, Any]]:
    """Yield mappings explicitly scoped to one formal artifact.

    Historical verifiers use several envelopes (``video``,
    ``component_scores_by_artifact`` and task-prefixed
    ``*_metrics_by_artifact``).  Restricting recursive alias lookup to those
    envelopes keeps diagnostic sections from standing in as process evidence.
    """

    wanted = {str(name).lower() for name in names}
    for raw_key, value in metrics.items():
        key = str(raw_key).lower()
        if key in wanted and isinstance(value, Mapping):
            yield value
        if not isinstance(value, Mapping):
            continue
        if not (
            key.endswith("_metrics_by_artifact")
            or key in {
                "artifact_metrics",
                "artifact_metrics_by_artifact",
                "component_scores_by_artifact",
                "metrics_by_artifact",
            }
        ):
            continue
        for raw_name, section in value.items():
            if str(raw_name).lower() in wanted and isinstance(section, Mapping):
                yield section


def _artifact_score(
    metrics: Mapping[str, Any],
    names: Sequence[str],
    aliases: Sequence[str],
    *,
    allow_section_binary: bool = False,
    allow_unscoped_direct: bool = False,
) -> Optional[float]:
    """Return independent evidence for one formal artifact, if exposed."""

    direct_aliases = (
        tuple(aliases)
        if allow_unscoped_direct
        else tuple(
            alias
            for alias in aliases
            if any(
                str(alias).lower().startswith(f"{str(name).lower()}_")
                for name in names
            )
        )
    )
    direct = _direct_score_from_metrics(metrics, direct_aliases)
    if direct is not None:
        return direct
    for section in _artifact_sections(metrics, names):
        score = _score_from_metrics(section, aliases)
        if score is not None:
            return score
        if allow_section_binary:
            score = _direct_score_from_metrics(section, ("binary_correct",))
            if score is not None:
                return score
    return None


def _top_component_score(
    metrics: Mapping[str, Any],
    aliases: Sequence[str],
) -> Optional[float]:
    components = metrics.get("component_scores")
    if not isinstance(components, Mapping):
        return None
    return _score_from_metrics(components, aliases)


def _artifact_result(
    metrics: Mapping[str, Any],
    names: Sequence[str],
) -> Optional[bool]:
    wanted = {name.lower() for name in names}
    for _path, key, value in _iter_metric_items(metrics):
        if not (
            key == "artifact_results" or key.endswith("_artifact_passes")
        ) or not isinstance(value, Mapping):
            continue
        for raw_name, result in value.items():
            if str(raw_name).lower() in wanted and isinstance(result, (bool, int, float)):
                return bool(result)
    direct_aliases = tuple(
        alias
        for name in names
        for alias in (
            f"{name}_pass",
            f"{name}_correct",
            f"{name}_semantic_pass",
            f"{name}_semantic_correct",
        )
    )
    score = _score_from_metrics(metrics, direct_aliases)
    return None if score is None else bool(score >= 1.0)


def _remove_legacy_reward_fields(value: Any) -> None:
    if isinstance(value, MutableSequence):
        for item in value:
            _remove_legacy_reward_fields(item)
        return
    if not isinstance(value, MutableMapping):
        return
    for key in list(value):
        if _is_legacy_reward_key(key):
            del value[key]
            continue
        _remove_legacy_reward_fields(value[key])


def normalize_eval_record_rewards(
    record: Any,
    *,
    artifact_set_complete: bool,
    artifact_presence: Mapping[str, bool],
    artifact_validity: Optional[Mapping[str, bool]] = None,
) -> None:
    """Seal one EvalRecord to the three-dimension v2 runtime contract.

    Family verifiers remain responsible for independent semantic replay.  This
    layer only gives their formal-artifact results one model-agnostic reward
    vocabulary and prevents historical per-route component names from becoming
    undeclared training rewards.
    """

    metrics = getattr(record, "offline_metrics", None)
    if not isinstance(metrics, dict):
        metrics = {}
        record.offline_metrics = metrics

    original_pass = bool(getattr(record, "offline_pass", False))
    video_present = bool(artifact_presence.get("video", False))
    validity = dict(artifact_validity or {})

    process_score = _artifact_score(
        metrics,
        ("video", "whole_video_process"),
        (
            "whole_video_process_correct",
            "video_whole_process_pass",
            "video_process_pass",
            "video_continuous_process_pass",
            "video_pass",
            "temporal_process_score",
        ),
        allow_section_binary=True,
        allow_unscoped_direct=True,
    )
    if process_score is None:
        process_score = _top_component_score(
            metrics,
            ("whole_video_process_score", "temporal_process_score"),
        )

    # Artifact-specific summary booleans are a last-resort independent signal;
    # never substitute the record-level offline_pass for a missing dimension.
    if process_score is None:
        result = _artifact_result(metrics, ("video", "whole_video_process"))
        process_score = None if result is None else float(result)

    video_result = bool(process_score is not None and process_score >= 1.0)

    if not video_present or not bool(validity.get("video", False)):
        video_result = False

    media_contract = float(
        artifact_set_complete and bool(validity.get("video", False))
    )

    video_final_score = _artifact_score(
        metrics,
        ("video", "whole_video_process"),
        (
            "video_final_state_correct",
            "video_final_state_score",
            "video_final_state_pass",
            "video_terminal_state_correct",
            "video_terminal_state_score",
            "video_terminal_state_pass",
            "video_terminal_semantic_correct",
            "video_terminal_semantic_pass",
            "video_final_hold_correct",
            "final_state_score",
        ),
        allow_section_binary=False,
        allow_unscoped_direct=True,
    )
    if video_final_score is None:
        video_final_score = _top_component_score(
            metrics,
            ("video_final_state_score", "final_state_score"),
        )
    video_final = float(video_final_score is not None and video_final_score >= 1.0)
    if not video_present or not bool(validity.get("video", False)):
        video_final = 0.0

    values = {
        "whole_video_process": float(video_result),
        "video_final_state": video_final,
        "media_contract": media_contract,
    }
    availability = {
        dimension.name: True
        for dimension in RAW_REWARD_DIMENSIONS
    }
    vector = {
        name: values[name]
        for name in RAW_REWARD_DIMENSION_NAMES
    }
    # Preserve the diagnostic when this sample's terminal state was not
    # independently decidable; retain the zero-valued axis so consumers still
    # receive the full schema.
    if metrics.get("video_final_state_available") is False:
        availability["video_final_state"] = False
        vector["video_final_state"] = 0.0
    hard_gates = list(RAW_REWARD_HARD_GATES)

    # Never promote a route-level semantic failure.  If a legacy verifier did
    # not expose the failing sub-dimension, attribute it to whole_video_process
    # so binary correctness remains fail closed.
    if not original_pass and all(vector.get(name) == 1.0 for name in hard_gates):
        if "whole_video_process" in vector:
            vector["whole_video_process"] = 0.0

    canonical_pass = bool(
        original_pass
        and artifact_set_complete
        and all(availability[name] for name in hard_gates)
        and all(vector.get(name) == 1.0 for name in hard_gates)
    )
    record.offline_pass = canonical_pass

    _remove_legacy_reward_fields(metrics)
    metrics.update(
        {
            "artifact_results": {"video": video_result},
            "binary_correct": canonical_pass,
            "raw_reward_vector": vector,
            "reward_availability_mask": availability,
            "reward_hard_gates": hard_gates,
            "reward_schema_version": RAW_REWARD_SCHEMA_VERSION,
            "reward_schema": build_reward_schema(),
            "diagnostics_are_rewards": False,
            "verifier_protocol": VERIFIER_PROTOCOL,
        }
    )


__all__ = [
    "RAW_REWARD_DIMENSIONS",
    "RAW_REWARD_DIMENSION_NAMES",
    "RAW_REWARD_HARD_GATES",
    "RAW_REWARD_SCHEMA_VERSION",
    "build_reward_schema",
    "normalize_eval_record_rewards",
]
