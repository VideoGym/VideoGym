"""Shared generator and evaluator protocols for executable task semantics."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Mapping, Optional, Sequence, TypeVar, Union

PathLike = Union[str, Path]
RecordT = TypeVar("RecordT")


class AbstractPuzzleGenerator(ABC, Generic[RecordT]):
    """Single protocol implemented by every registered task generator."""

    def __init__(self, output_dir: PathLike) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_puzzle(self, *args, **kwargs) -> RecordT:
        """Create a puzzle from the provided resources."""

    def create_random_puzzle(self) -> RecordT:
        """Create a single randomized puzzle instance."""
        return self.create_puzzle()

    def configure_runtime(
        self,
        *,
        output_dir: PathLike,
        seed: Optional[int],
        video: bool,
        video_fps: int,
    ) -> None:
        """Apply the process-level runtime context without task-type routing.

        Constructors still receive every supported argument directly.  This
        method makes process-level runtime attributes consistent after
        construction, so orchestration never needs to branch
        on a generator family, renderer style, or verifier implementation.
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.video = bool(video)
        self.generate_videos = bool(video)
        self.record_video = bool(video)
        self.video_fps = int(video_fps)

    def record_to_dict(self, record: RecordT) -> Dict[str, Any]:
        """Dictionary serialization hook for puzzle records."""

        if isinstance(record, dict):
            return dict(record)
        if hasattr(record, "to_dict"):
            return getattr(record, "to_dict")()
        return dataclasses.asdict(record)

    def relativize_path(self, path: Path) -> str:
        """Map an absolute path into the generator output directory when possible."""

        try:
            return path.relative_to(self.output_dir).as_posix()
        except ValueError:
            return path.as_posix()


class RenderedTaskGenerator(AbstractPuzzleGenerator[Dict[str, Any]]):
    """Formal base for generators whose semantic authoring hook is ``make_sample``.

    ``make_sample`` may return a record directly or a
    ``(record, puzzle_image, solution_image)`` renderer tuple. Images are QA
    values only; the task-owned record must already contain its typed answer,
    prompt bundle, and persisted media paths.
    """

    def make_sample(self) -> object:
        raise NotImplementedError

    def create_puzzle(self) -> Dict[str, Any]:
        rendered = self.make_sample()
        record = rendered[0] if isinstance(rendered, tuple) and rendered else rendered
        if not isinstance(record, dict):
            raise TypeError(
                f"{type(self).__name__}.make_sample() must return a record dict "
                "or a tuple whose first item is a record dict"
            )
        return record


class AbstractPuzzleEvaluator(ABC):
    """Base class scaffolding for puzzle evaluation from in-memory records."""

    def __init__(
        self,
        records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        base_dir: Optional[PathLike] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self._records = self._load_records(records)

    @staticmethod
    def _coerce_records(
        records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if isinstance(records, Mapping):
            if "id" in records:
                raw_records: Sequence[Mapping[str, Any]] = [records]
            elif records and all(isinstance(value, Mapping) for value in records.values()):
                raw_records = list(records.values())
            else:
                raise ValueError("A single puzzle record must include an 'id'")
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            raw_records = records
        else:
            raise TypeError(
                "Puzzle evaluators require an in-memory record or sequence of records; "
                "metadata file paths are not supported"
            )

        normalized: List[Dict[str, Any]] = []
        for record in raw_records:
            if not isinstance(record, Mapping):
                raise TypeError("Each puzzle record must be a mapping")
            normalized.append(dict(record))
        return normalized

    def _load_records(
        self,
        raw_records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for record in self._coerce_records(raw_records):
            puzzle_id = record.get("id")
            if not puzzle_id:
                raise ValueError("Each puzzle record must include an 'id'")
            records[str(puzzle_id)] = record
        return records

    def get_record(self, puzzle_id: str) -> Dict[str, Any]:
        try:
            return self._records[puzzle_id]
        except KeyError as exc:
            raise KeyError(f"Puzzle id '{puzzle_id}' not found in source records") from exc

    def resolve_path(self, path_value: object) -> Path:
        candidate = Path(str(path_value))
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return candidate

__all__ = [
    "AbstractPuzzleGenerator",
    "AbstractPuzzleEvaluator",
    "PathLike",
    "RenderedTaskGenerator",
]
