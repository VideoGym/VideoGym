from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from data.common.paths import to_absolute, to_relative
from data.task_spec import CANONICAL_ANSWER_KINDS
from data.common.sample_identity import (
    build_canonical_sample_id,
    validate_sample_identity_witness,
)
from data.common.task_semantics import VALID_SPLITS, normalize_split


_ANSWER_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _is_exact_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_exact_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_exact_json_value(item)
            for key, item in value.items()
        )
    return False


def _normalize_asset_path(value: object) -> str:
    return Path(str(value)).expanduser().as_posix()


def _exact_mapping(value: object, *, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Canonical {field_name} must be a mapping")
    return value


@dataclass
class CanonicalAnswer:
    kind: str
    typed_text: Optional[str] = None
    option: Optional[str] = None
    position: Optional[List[int]] = None
    cartesian_point: Optional[List[int]] = None
    positions: Optional[List[List[int]]] = None
    path_positions: Optional[List[List[int]]] = None
    named_path_positions: Optional[Dict[str, List[List[int]]]] = None
    answer_grid: Optional[List[List[Any]]] = None
    json_object: Optional[Dict[str, Any]] = None
    integer: Optional[int] = None
    nonnegative_integer: Optional[int] = None
    nonnegative_integer_list: Optional[List[int]] = None
    string_list: Optional[List[str]] = None

    def __post_init__(self) -> None:
        def _normalize_positions(
            positions: object,
            *,
            field_name: str,
        ) -> List[List[int]]:
            if not isinstance(positions, list):
                raise ValueError(f"Canonical {field_name} must be a list")
            normalized_positions: List[List[int]] = []
            for position in positions:
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    raise ValueError(
                        f"Canonical {field_name} must contain 1-based (x, y) pairs"
                    )
                x, y = position
                if type(x) is not int or type(y) is not int or x < 1 or y < 1:
                    raise ValueError(
                        f"Canonical {field_name} must contain positive integer pairs"
                    )
                normalized_positions.append([x, y])
            if not normalized_positions:
                raise ValueError(f"Canonical {field_name} must not be empty")
            return normalized_positions

        if self.path_positions is not None:
            self.path_positions = _normalize_positions(
                self.path_positions,
                field_name="path_positions",
            )
        if self.positions is not None:
            self.positions = _normalize_positions(
                self.positions,
                field_name="positions",
            )
        if self.position is not None:
            normalized_position = _normalize_positions(
                [self.position],
                field_name="position",
            )
            self.position = normalized_position[0]
        if self.cartesian_point is not None:
            if (
                not isinstance(self.cartesian_point, (list, tuple))
                or len(self.cartesian_point) != 2
                or any(type(value) is not int for value in self.cartesian_point)
            ):
                raise ValueError(
                    "Canonical cartesian_point must be one exact integer pair"
                )
            self.cartesian_point = list(self.cartesian_point)
        if self.named_path_positions is not None:
            if not isinstance(self.named_path_positions, dict) or not self.named_path_positions:
                raise ValueError("Canonical named_path_positions must be a non-empty mapping")
            normalized_named_paths: Dict[str, List[List[int]]] = {}
            for raw_name, positions in self.named_path_positions.items():
                if (
                    not isinstance(raw_name, str)
                    or not raw_name
                    or raw_name != raw_name.strip()
                    or _ANSWER_NAME_RE.fullmatch(raw_name) is None
                    or raw_name in normalized_named_paths
                ):
                    raise ValueError(
                        "Canonical named path names must be unique identifier-like strings"
                    )
                normalized_named_paths[raw_name] = _normalize_positions(
                    positions,
                    field_name=f"named_path_positions[{raw_name!r}]",
                )
            self.named_path_positions = normalized_named_paths

        if self.option is not None and (
            not isinstance(self.option, str)
            or len(self.option) != 1
            or not ("A" <= self.option <= "Z")
        ):
            raise ValueError(
                "Canonical option must be one exact uppercase A-Z label"
            )

        if self.typed_text is not None and (
            not isinstance(self.typed_text, str)
            or not self.typed_text
            or self.typed_text != self.typed_text.strip()
        ):
            raise ValueError(
                "Canonical typed_text must be a non-empty string with no outer whitespace"
            )
        if self.answer_grid is not None:
            if not isinstance(self.answer_grid, list) or not self.answer_grid:
                raise ValueError("Canonical answer_grid must be a non-empty row list")
            normalized_grid: List[List[Any]] = []
            for raw_row in self.answer_grid:
                if not isinstance(raw_row, list) or not raw_row:
                    raise ValueError("Canonical answer_grid rows must be non-empty lists")
                row: List[Any] = []
                for cell in raw_row:
                    if isinstance(cell, bool) or type(cell) not in {int, str}:
                        raise ValueError(
                            "Canonical answer_grid cells must be exact integers or strings"
                        )
                    row.append(cell)
                normalized_grid.append(row)
            if len({len(row) for row in normalized_grid}) != 1:
                raise ValueError("Canonical answer_grid must be rectangular")
            self.answer_grid = normalized_grid
        if self.json_object is not None and (
            not isinstance(self.json_object, dict)
            or not _is_exact_json_value(self.json_object)
        ):
            raise ValueError(
                "Canonical json_object must contain only exact finite JSON values"
            )
        if (
            self.integer is not None
            and type(self.integer) is not int
        ):
            raise ValueError("Canonical integer must be an exact integer")
        if (
            self.nonnegative_integer is not None
            and (
                type(self.nonnegative_integer) is not int
                or self.nonnegative_integer < 0
            )
        ):
            raise ValueError("Canonical nonnegative_integer must be an exact nonnegative integer")
        if self.nonnegative_integer_list is not None:
            if (
                not isinstance(self.nonnegative_integer_list, list)
                or not self.nonnegative_integer_list
                or any(
                    type(value) is not int or value < 0
                    for value in self.nonnegative_integer_list
                )
            ):
                raise ValueError(
                    "Canonical nonnegative_integer_list must contain exact nonnegative integers"
                )
            self.nonnegative_integer_list = list(self.nonnegative_integer_list)
        if self.string_list is not None:
            if (
                not isinstance(self.string_list, list)
                or not self.string_list
                or any(
                    not isinstance(value, str) or not value
                    for value in self.string_list
                )
            ):
                raise ValueError(
                    "Canonical string_list must contain non-empty strings"
                )
            self.string_list = list(self.string_list)

        if not isinstance(self.kind, str):
            raise ValueError("Canonical answer kind must be an exact string")
        normalized_kind = self.kind
        if normalized_kind not in CANONICAL_ANSWER_KINDS:
            raise ValueError(f"Unsupported canonical answer kind: {self.kind!r}")
        required_values = {
            "option": self.option,
            "grid": self.answer_grid,
            "path": self.path_positions,
            "position": self.position,
            "cartesian_point": self.cartesian_point,
            "positions": self.positions,
            "named_paths": self.named_path_positions,
            "json_object": self.json_object,
            "integer": self.integer,
            "nonnegative_integer": self.nonnegative_integer,
            "nonnegative_integer_list": self.nonnegative_integer_list,
            "string_list": self.string_list,
            "typed_text": self.typed_text,
        }
        if set(required_values) != set(CANONICAL_ANSWER_KINDS):
            raise RuntimeError(
                "CanonicalAnswer typed fields drifted from CANONICAL_ANSWER_KINDS"
            )
        required_value = required_values[normalized_kind]
        def _missing(value: object) -> bool:
            return value is None or value == "" or value == [] or value == {}

        if _missing(required_value):
            raise ValueError(
                f"Canonical answer kind {normalized_kind!r} lacks its typed value"
            )
        conflicting = [
            kind
            for kind, value in required_values.items()
            if kind != normalized_kind and not _missing(value)
        ]
        if conflicting:
            raise ValueError(
                f"Canonical answer kind {normalized_kind!r} also populates {conflicting}"
            )
        self.kind = normalized_kind

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"kind": self.kind}
        for key in (
            "typed_text",
            "option",
            "position",
            "cartesian_point",
            "positions",
            "path_positions",
            "named_path_positions",
            "answer_grid",
            "json_object",
            "integer",
            "nonnegative_integer",
            "nonnegative_integer_list",
            "string_list",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalAnswer":
        if not isinstance(payload, dict):
            raise ValueError("Canonical answer payload must be a mapping")
        allowed_fields = {
            "kind",
            "typed_text",
            "option",
            "position",
            "cartesian_point",
            "positions",
            "path_positions",
            "named_path_positions",
            "answer_grid",
            "json_object",
            "integer",
            "nonnegative_integer",
            "nonnegative_integer_list",
            "string_list",
        }
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "Canonical answer contains unknown field(s): "
                + ", ".join(unknown_fields)
            )
        return CanonicalAnswer(
            kind=payload.get("kind"),
            typed_text=payload.get("typed_text"),
            option=payload.get("option"),
            answer_grid=payload.get("answer_grid"),
            position=payload.get("position"),
            cartesian_point=payload.get("cartesian_point"),
            positions=payload.get("positions"),
            path_positions=payload.get("path_positions"),
            named_path_positions=payload.get("named_path_positions"),
            json_object=payload.get("json_object"),
            integer=payload.get("integer"),
            nonnegative_integer=payload.get("nonnegative_integer"),
            nonnegative_integer_list=payload.get("nonnegative_integer_list"),
            string_list=payload.get("string_list"),
        )


@dataclass
class CanonicalSample:
    id: str
    task_family: str
    task_name: str
    prompt: str
    image: str
    ground_truth: CanonicalAnswer
    split: str = "train"
    video: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None
    seed: int = 0
    semantic_input: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_input_image(self) -> str:
        return self.image

    @property
    def reference_video(self) -> Optional[str]:
        return self.video

    def to_dict(self, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        image = (
            to_relative(self.image, base_dir)
            if base_dir is not None and self.image
            else self.image
        )
        video = (
            to_relative(self.video, base_dir)
            if base_dir is not None and self.video
            else self.video
        )
        payload: Dict[str, Any] = {
            "id": self.id,
            "task_name": self.task_name,
            "task_family": self.task_family,
            "split": self.split,
            "prompt": self.prompt,
            "image": image,
        }
        if video is not None:
            payload["video"] = video
        if self.video_fps is not None:
            payload["video_fps"] = self.video_fps
        if self.video_num_frames is not None:
            payload["video_num_frames"] = self.video_num_frames
        payload["seed"] = self.seed
        payload["semantic_input"] = self.semantic_input
        payload["ground_truth"] = self.ground_truth.to_dict()
        return payload

    @staticmethod
    def from_dict(
        payload: Dict[str, Any],
        *,
        base_dir: Optional[Path] = None,
    ) -> "CanonicalSample":
        from data.registry import get_task_spec

        if not isinstance(payload, dict):
            raise ValueError("Canonical sample payload must be a mapping")

        def _resolve_media(raw: Any) -> Optional[str]:
            if not raw:
                return None
            if base_dir is not None:
                return to_absolute(str(raw), base_dir).as_posix()
            return _normalize_asset_path(raw)

        raw_split = payload.get("split")
        if raw_split not in VALID_SPLITS:
            raise ValueError(
                f"Canonical sample split must be one of {VALID_SPLITS}, got {raw_split!r}"
            )
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Canonical sample prompt must be a non-empty string")
        image = _resolve_media(payload.get("image"))
        if not image:
            raise ValueError("Canonical sample image must be a non-empty path")
        video = _resolve_media(payload.get("video"))

        sample = CanonicalSample(
            id=str(payload["id"]),
            task_family=str(payload["task_family"]),
            task_name=str(payload["task_name"]),
            split=normalize_split(raw_split),
            prompt=prompt,
            image=image,
            video=video,
            video_fps=int(payload["video_fps"]) if payload.get("video_fps") is not None else None,
            video_num_frames=int(payload["video_num_frames"]) if payload.get("video_num_frames") is not None else None,
            seed=payload.get("seed"),
            ground_truth=CanonicalAnswer.from_dict(
                _exact_mapping(
                    payload.get("ground_truth"),
                    field_name="ground_truth",
                )
            ),
            semantic_input=dict(
                _exact_mapping(
                    payload.get("semantic_input"),
                    field_name="semantic_input",
                )
            ),
        )
        task_spec = get_task_spec(sample.task_name)
        if task_spec is None:
            raise ValueError(
                f"Canonical sample names unregistered task {sample.task_name!r}"
            )
        if sample.task_family != task_spec.task_family:
            raise ValueError(
                f"{sample.task_name}: canonical task_family {sample.task_family!r} "
                f"does not match registry {task_spec.task_family!r}"
            )
        validate_sample_identity_witness(
            sample_id=sample.id,
            task_name=sample.task_name,
            seed=sample.seed,
        )
        expected_sample_id = build_canonical_sample_id(
            task_name=sample.task_name,
            seed=sample.seed,
            semantic_input=sample.semantic_input,
            ground_truth=sample.ground_truth.to_dict(),
            prompt=sample.prompt,
            image=sample.image,
        )
        if sample.id != expected_sample_id:
            raise ValueError(
                f"{sample.task_name}: canonical id does not match its persisted "
                "identity witnesses"
            )
        if sample.ground_truth.kind != task_spec.answer_kind:
            raise ValueError(
                f"{sample.task_name}: canonical answer kind "
                f"{sample.ground_truth.kind!r} does not match registry "
                f"{task_spec.answer_kind!r}"
            )
        if sample.ground_truth.kind == "option":
            codec = task_spec.answer_codec
            if codec is None or sample.ground_truth.option not in codec.labels:
                raise ValueError(
                    f"{sample.task_name}: canonical option "
                    f"{sample.ground_truth.option!r} is outside its registered domain"
                )
        if sample.ground_truth.kind == "positions":
            positions = [
                tuple(position)
                for position in (sample.ground_truth.positions or [])
            ]
            if len(positions) != len(set(positions)):
                raise ValueError(
                    f"{sample.task_name}: canonical position collection contains duplicates"
                )
            if positions != sorted(positions):
                raise ValueError(
                    f"{sample.task_name}: canonical position collection must be "
                    "serialized in increasing x order, then increasing y order"
                )
        return sample


@dataclass
class EvalRecord:
    sample_id: str
    task_family: str
    task_name: str
    input_asset: str
    split: str = "eval"
    prediction_asset: Optional[str] = None
    prediction_text: Optional[str] = None
    offline_metrics: Dict[str, Any] = field(default_factory=dict)
    offline_pass: Optional[bool] = None
    infer_meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "input_asset": self.input_asset,
            "task_family": self.task_family,
            "split": self.split,
            "prediction_asset": self.prediction_asset,
            "prediction_text": self.prediction_text,
            "offline_metrics": self.offline_metrics,
            "offline_pass": self.offline_pass,
            "infer_meta": self.infer_meta,
            "error": self.error,
        }
