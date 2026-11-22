"""Shared, configurable catalog of training tasks and generalization levels."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import tomllib

PROTOCOL = "videogym.video.generalization"
TRACKS = ("id", "extrapolation", "composition", "cross_task")
TRACK_NAMES = {
    "id": "In-Domain",
    "prompt_paraphrase_text": "Text Prompt Paraphrase",
    "prompt_paraphrase_vision": "Vision Prompt Paraphrase",
    "extrapolation": "Extrapolation",
    "composition": "Composition",
    "cross_task": "Cross-Task",
    "cross_benchmark": "Cross-Benchmark",
}


@dataclass(frozen=True)
class EvaluationTask:
    name: str
    title: str
    track: str
    task_name: str
    config: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    description: str = ""
    extrapolation_axis: str | None = None
    primitives: tuple[str, ...] = ()
    role: str = "evaluation"
    tiers: tuple[dict, ...] = ()
    mechanisms: tuple[str, ...] = ()
    novelty: str = ""

    def __post_init__(self):
        if self.track not in TRACKS or self.role != "evaluation":
            raise ValueError("invalid evaluation track or role")
        if self.track == "cross_task" and self.primitives:
            raise ValueError("Cross-Task is family-derived, with no source-ID prerequisites")
        names = [tier["name"] for tier in self.tiers]
        if len(set(names)) != len(names):
            raise ValueError("duplicate difficulty tier")

    def difficulty_tiers(self):
        return self.tiers or (
            {"name": "default", "config": self.config, "constraints": self.constraints},
        )

    def at_tier(self, name):
        tier = next((t for t in self.difficulty_tiers() if t["name"] == name), None)
        if tier is None:
            raise ValueError(f"{self.name}: unknown difficulty tier {name}")
        return replace(
            self,
            config={**self.config, **tier.get("config", {})},
            constraints={**self.constraints, **tier.get("constraints", {})},
            tiers=(),
        )

    def accepts(self, features):
        return all(
            features.get(key) in values for key, values in self.constraints.items()
        )

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = Path(__file__).with_name("default.toml")


def load_config(config=None):
    if isinstance(config, dict):
        return config
    with Path(config or DEFAULT_CONFIG).expanduser().open("rb") as stream:
        return tomllib.load(stream)


def evaluation_tasks(config=None, training_tasks=None):
    """Resolve the configured catalog of primary evaluation tasks."""
    from data.registry import TASK_SPECS, get_task_spec
    from evaluation.composition.catalog import composition_tasks
    from evaluation.cross_task.catalog import cross_task_tasks
    from evaluation.extrapolation.catalog import extrapolation_tasks
    from evaluation.in_domain.catalog import in_domain_tasks

    settings = load_config(config)
    selection = settings.get("catalog", {})
    values = [
        *in_domain_tasks(),
        *extrapolation_tasks(),
        *composition_tasks(),
        *cross_task_tasks(),
    ]
    definitions = {}
    for case in values:
        if isinstance(case, dict):
            case = EvaluationTask(**case)
        if case.name in definitions:
            raise ValueError(f"duplicate evaluation task: {case.name}")
        if get_task_spec(case.task_name) is None:
            raise ValueError(f"unregistered generator: {case.task_name}")
        definitions[case.name] = case
    trained = set(training_tasks) if training_tasks is not None else set(TASK_SPECS)
    unknown = trained - TASK_SPECS.keys()
    if unknown:
        raise ValueError(f"training profile has unregistered tasks: {sorted(unknown)}")
    requested = set(selection.get("tasks", ()))
    if requested - definitions.keys():
        raise ValueError(
            f"unknown evaluation tasks: {sorted(requested - definitions.keys())}"
        )
    excluded = set(selection.get("exclude", ()))
    if excluded - definitions.keys():
        raise ValueError("unknown excluded evaluation task")
    tracks = set(selection.get("tracks", TRACKS))
    if tracks - set(TRACKS):
        raise ValueError("unknown generalization track")
    families = set(selection.get("families", ()))
    chosen = {
        name: case
        for name, case in definitions.items()
        if name not in excluded
        and case.track in tracks
        and (not requested or name in requested)
        and (not families or get_task_spec(case.task_name).task_family in families)
        and (case.track != "id" or case.task_name in trained)
    }
    if not chosen:
        raise ValueError("no evaluation tasks selected")
    return chosen


def catalog_summary(cases):
    return {
        "primary_tasks": len(cases),
        "tracks": dict(Counter(case.track for case in cases.values())),
        "difficulty_tiers": sum(
            len(case.difficulty_tiers()) for case in cases.values()
        ),
    }
