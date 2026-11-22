"""Strict shared codecs for model-facing ``Answer:`` deliveries.

Task semantics decide *what* a payload means.  This module owns the syntax
that is identical across tasks: the one-line Answer envelope, canonical JSON
spacing, option labels, and rectangular matrices.  Candidate parsers use the
same codecs as reference serializers, so missing punctuation, alternate JSON
whitespace, boolean-as-integer coercion, and family-local label ranges cannot
drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Collection, Optional, Sequence


ANSWER_PREFIX = "Answer: "
ANSWER_SYNTAX_EXAMPLE_CUE = "As a syntax-only example:"


def format_answer_line(payload: str) -> str:
    """Wrap one already-semantic payload in the canonical Answer envelope."""

    if not isinstance(payload, str):
        raise ValueError(
            "Answer envelope requires a payload string from a typed codec"
        )
    if not payload or payload != payload.strip():
        raise ValueError("Answer payload must be non-empty with no outer whitespace")
    if payload.endswith("."):
        raise ValueError("Answer payload must not include the envelope's terminal period")
    if "\r" in payload or "\n" in payload:
        raise ValueError("Answer payload must occupy exactly one line")
    if "Answer:" in payload:
        raise ValueError("format_answer_line expects a payload, not an Answer line")
    if "<think>" in payload or "</think>" in payload:
        raise ValueError("think tags are not allowed in an Answer payload")
    return f"{ANSWER_PREFIX}{payload}."


def parse_answer_line(text: object) -> Optional[str]:
    """Return the payload only for one exact ``Answer: <payload>.`` line."""

    if not isinstance(text, str) or "\r" in text or "\n" in text:
        return None
    if not text.startswith(ANSWER_PREFIX) or not text.endswith("."):
        return None
    payload = text[len(ANSWER_PREFIX) : -1]
    if not payload or payload != payload.strip() or payload.endswith("."):
        return None
    if "Answer:" in payload or "<think>" in payload or "</think>" in payload:
        return None
    return payload


def canonical_json_dumps(value: Any, *, sort_keys: bool = True) -> str:
    """Serialize JSON with the repository-wide comma/colon spacing policy."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        allow_nan=False,
        separators=(", ", ": "),
    )


def format_canonical_json_answer(value: Any, *, sort_keys: bool = True) -> str:
    return format_answer_line(canonical_json_dumps(value, sort_keys=sort_keys))


def decode_canonical_json_answer(
    text: object,
    *,
    sort_keys: bool = True,
) -> tuple[Any, bool]:
    """Decode only JSON whose bytes already match the canonical serializer."""

    payload = parse_answer_line(text)
    if payload is None:
        return None, False
    try:
        value = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, False
    try:
        canonical = canonical_json_dumps(value, sort_keys=sort_keys)
    except (TypeError, ValueError):
        return None, False
    return (value, True) if payload == canonical else (None, False)


def _label_phrase(labels: Sequence[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


@dataclass(frozen=True)
class ChoiceCodec:
    """One option-label grammar shared by Prompt, serializer, and verifier."""

    labels: tuple[str, ...]
    example_label: str

    def __post_init__(self) -> None:
        normalized = self.labels
        example = self.example_label
        if (
            not isinstance(normalized, tuple)
            or not normalized
            or len(set(normalized)) != len(normalized)
            or any(
                not isinstance(label, str)
                or len(label) != 1
                or not ("A" <= label <= "Z")
                for label in normalized
            )
        ):
            raise ValueError("ChoiceCodec labels must be unique uppercase A-Z letters")
        if (
            not isinstance(example, str)
            or len(example) != 1
            or not ("A" <= example <= "Z")
            or example in normalized
        ):
            raise ValueError("ChoiceCodec example_label must be one out-of-domain uppercase letter")

    @property
    def schema(self) -> str:
        return "Answer: <uppercase option label>."

    @property
    def capture_pattern(self) -> str:
        """Regex source for code that must publish the executable grammar."""

        return rf"Answer: ([{''.join(self.labels)}])\."

    @property
    def full_pattern(self) -> str:
        return rf"^{self.capture_pattern}$"

    @property
    def example(self) -> str:
        return format_answer_line(self.example_label)

    @property
    def instruction(self) -> str:
        return (
            f"Return exactly one uppercase option label from {_label_phrase(self.labels)} and no extra text. "
            f"{ANSWER_SYNTAX_EXAMPLE_CUE} `{self.example}` The example label is outside the candidate set."
        )

    def format(self, label: object) -> str:
        if not isinstance(label, str) or label not in self.labels:
            raise ValueError(
                f"option label must be one of {_label_phrase(self.labels)}, got {label!r}"
            )
        return format_answer_line(label)

    def parse(self, text: object) -> Optional[str]:
        payload = parse_answer_line(text)
        return payload if payload in self.labels else None


FOUR_CHOICE_CODEC = ChoiceCodec(("A", "B", "C", "D"), "X")
FIVE_CHOICE_CODEC = ChoiceCodec(("A", "B", "C", "D", "E"), "Z")


def _normalize_matrix(
    value: object,
    *,
    rows: Optional[int],
    cols: Optional[int],
    cell_kind: str,
    allowed_values: Optional[Collection[object]],
) -> Optional[list[list[Any]]]:
    if not isinstance(value, list) or not value:
        return None
    parsed: list[list[Any]] = []
    for raw_row in value:
        if not isinstance(raw_row, list) or not raw_row:
            return None
        row: list[Any] = []
        for cell in raw_row:
            if cell_kind == "integer":
                if isinstance(cell, bool) or not isinstance(cell, int):
                    return None
            elif cell_kind == "string":
                if not isinstance(cell, str):
                    return None
            else:
                raise ValueError(f"unsupported matrix cell_kind {cell_kind!r}")
            if allowed_values is not None and cell not in allowed_values:
                return None
            row.append(cell)
        parsed.append(row)
    if len({len(row) for row in parsed}) != 1:
        return None
    if rows is not None and len(parsed) != int(rows):
        return None
    if cols is not None and len(parsed[0]) != int(cols):
        return None
    return parsed


def format_matrix_answer(
    matrix: object,
    *,
    cell_kind: str = "integer",
    allowed_values: Optional[Collection[object]] = None,
) -> str:
    parsed = _normalize_matrix(
        matrix,
        rows=None,
        cols=None,
        cell_kind=cell_kind,
        allowed_values=allowed_values,
    )
    if parsed is None:
        raise ValueError("matrix does not satisfy the declared Answer grammar")
    return format_canonical_json_answer(parsed)


def parse_matrix_answer(
    text: object,
    *,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    cell_kind: str = "integer",
    allowed_values: Optional[Collection[object]] = None,
) -> Optional[list[list[Any]]]:
    value, valid = decode_canonical_json_answer(text)
    if not valid:
        return None
    return _normalize_matrix(
        value,
        rows=rows,
        cols=cols,
        cell_kind=cell_kind,
        allowed_values=allowed_values,
    )


__all__ = [
    "ANSWER_PREFIX",
    "ANSWER_SYNTAX_EXAMPLE_CUE",
    "ChoiceCodec",
    "FIVE_CHOICE_CODEC",
    "FOUR_CHOICE_CODEC",
    "canonical_json_dumps",
    "decode_canonical_json_answer",
    "format_answer_line",
    "format_canonical_json_answer",
    "format_matrix_answer",
    "parse_answer_line",
    "parse_matrix_answer",
]
