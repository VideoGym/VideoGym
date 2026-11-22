from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from data.common.typed_answers import ChoiceCodec
from data.common.verifiers.routes import VerifierRoute

CANONICAL_ANSWER_KINDS: Tuple[str, ...] = (
    "option",
    "grid",
    "path",
    "position",
    "cartesian_point",
    "positions",
    "named_paths",
    "json_object",
    "integer",
    "nonnegative_integer",
    "nonnegative_integer_list",
    "string_list",
    "typed_text",
)
POSITION_COLLECTION_ORDERS: Tuple[str, ...] = (
    "unordered",
    "x_then_y",
)


@dataclass(frozen=True)
class TaskRoute:
    """Static generator and delivery contract owned by one Task Family."""

    name: str
    module: str
    class_name: str
    verifier: VerifierRoute
    answer_kind: str
    answer_codec: Optional[ChoiceCodec] = None
    position_collection_order: Optional[str] = None
    requires: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.answer_kind not in CANONICAL_ANSWER_KINDS:
            raise ValueError(
                f"{self.name}: answer_kind must be one of {CANONICAL_ANSWER_KINDS}, "
                f"got {self.answer_kind!r}"
            )
        if self.answer_kind == "option" and not isinstance(
            self.answer_codec, ChoiceCodec
        ):
            raise ValueError(
                f"{self.name}: option answers require one explicit ChoiceCodec"
            )
        if self.answer_kind != "option" and self.answer_codec is not None:
            raise ValueError(
                f"{self.name}: only option answers may declare a ChoiceCodec"
            )
        if self.answer_kind == "positions":
            if self.position_collection_order not in POSITION_COLLECTION_ORDERS:
                raise ValueError(
                    f"{self.name}: positions answers require one explicit "
                    f"position_collection_order from {POSITION_COLLECTION_ORDERS}"
                )
        elif self.position_collection_order is not None:
            raise ValueError(
                f"{self.name}: position_collection_order applies only to positions answers"
            )


@dataclass(frozen=True)
class TaskFamily:
    """Task family and its registered task inventory."""

    name: str
    description: str
    tasks: Tuple[TaskRoute, ...]


@dataclass(frozen=True)
class TaskSpec:
    """Resolved executable task contract."""

    name: str
    task_family: str
    module: str
    class_name: str
    verifier: VerifierRoute
    answer_kind: str
    answer_codec: Optional[ChoiceCodec] = None
    position_collection_order: Optional[str] = None
    requires: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.answer_kind not in CANONICAL_ANSWER_KINDS:
            raise ValueError(
                f"{self.name}: answer_kind must be one of {CANONICAL_ANSWER_KINDS}, "
                f"got {self.answer_kind!r}"
            )
        if self.answer_kind == "option" and not isinstance(
            self.answer_codec, ChoiceCodec
        ):
            raise ValueError(
                f"{self.name}: option answers require one explicit ChoiceCodec"
            )
        if self.answer_kind != "option" and self.answer_codec is not None:
            raise ValueError(
                f"{self.name}: only option answers may declare a ChoiceCodec"
            )
        if self.answer_kind == "positions":
            if self.position_collection_order not in POSITION_COLLECTION_ORDERS:
                raise ValueError(
                    f"{self.name}: positions answers require one explicit "
                    f"position_collection_order from {POSITION_COLLECTION_ORDERS}"
                )
        elif self.position_collection_order is not None:
            raise ValueError(
                f"{self.name}: position_collection_order applies only to positions answers"
            )
