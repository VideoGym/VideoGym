from __future__ import annotations

import importlib
import inspect
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from data.common.sample_identity import index_unique_samples


VerifierCallable = Callable[..., list[Any]]
VALID_MODALITIES = frozenset({"ti2v"})


@lru_cache(maxsize=None)
def _load_verifier(module: str, function: str) -> VerifierCallable:
    candidate = getattr(importlib.import_module(module), function)
    if not callable(candidate):
        raise TypeError(f"Verifier route {module}:{function} is not callable")
    return candidate


@dataclass(frozen=True)
class VerifierRoute:
    """Lazy, explicit route to one final-answer verifier capability."""

    module: str
    function: str

    @property
    def identifier(self) -> str:
        return f"{self.module}:{self.function}"

    def load(self) -> VerifierCallable:
        return _load_verifier(self.module, self.function)

    @staticmethod
    def _link_candidate(source: Optional[Path], destination: Path) -> bool:
        if source is None or not source.exists():
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(source.resolve(), destination)
        except OSError:
            shutil.copy2(source, destination)
        return True

    @staticmethod
    def _video_artifact_valid(source: Optional[Path]) -> bool:
        if source is None or not source.is_file():
            return False
        try:
            from data.common.video_encoding import validate_video_contract

            validate_video_contract(source)
        except (OSError, ValueError):
            return False
        return True

    def __call__(
        self,
        samples: Sequence[Any],
        pred_root: Path,
        *,
        modality: str = "ti2v",
        training_phase: Optional[str] = None,
    ) -> list[Any]:
        """Evaluate candidate video deliveries in an isolated candidate view."""

        normalized_modality = str(modality or "ti2v").strip().lower()
        if normalized_modality not in VALID_MODALITIES:
            raise ValueError(
                f"modality must be one of {sorted(VALID_MODALITIES)}, got {modality!r}"
            )
        samples_by_id = index_unique_samples(
            samples,
            context=f"verifier route {self.identifier}",
        )
        ordered_samples = list(samples_by_id.values())

        # Imports stay local so the lazy route module does not acquire a
        # verifier-module import cycle.
        from data.common.verifiers.candidates import resolve_candidate_video
        from data.common.reward_contract import normalize_eval_record_rewards

        artifact_presence: Dict[str, Dict[str, bool]] = {}
        artifact_validity: Dict[str, Dict[str, bool]] = {}
        with tempfile.TemporaryDirectory(prefix="videogym-verifier-") as tmp:
            isolated_root = Path(tmp)
            for sample in ordered_samples:
                sample_id = str(sample.id)
                split = str(getattr(sample, "split", "") or "").strip()
                sample_dir = isolated_root / split / sample_id if split else isolated_root / sample_id
                video_source = resolve_candidate_video(pred_root, sample_id, split or None)
                requested_sources = {"video": video_source}
                valid = {"video": self._video_artifact_valid(requested_sources["video"])}
                # Do not expose corrupt artifacts to legacy semantic decoders:
                # several route implementations assume that any discovered
                # candidate can be opened and would otherwise raise before a
                # fail-closed EvalRecord is produced.  Presence and validity
                # are tracked independently, so a supplied-but-corrupt file
                # still has a complete artifact set and earns media_contract=0.
                present = {
                    name: bool(source is not None and source.is_file())
                    for name, source in requested_sources.items()
                }
                self._link_candidate(
                    requested_sources["video"] if valid["video"] else None,
                    sample_dir / "prediction.webm",
                )
                artifact_validity[sample_id] = valid
                artifact_presence[sample_id] = present

            verifier = self.load()
            kwargs: Dict[str, Any] = {}
            verifier_parameters = inspect.signature(verifier).parameters
            if "modality" in verifier_parameters:
                kwargs["modality"] = normalized_modality
            if training_phase is not None and "training_phase" in verifier_parameters:
                kwargs["training_phase"] = training_phase
            records = verifier(ordered_samples, isolated_root, **kwargs)
            returned_ids = [str(getattr(record, "sample_id", "")) for record in records]
            if len(returned_ids) != len(samples_by_id) or set(returned_ids) != set(samples_by_id):
                raise RuntimeError(
                    f"Verifier {self.identifier} must return exactly one result for every supplied sample"
                )

        for record in records:
            sample_id = str(getattr(record, "sample_id", ""))
            present = artifact_presence.get(sample_id, {})
            complete = bool(present.get("video", False))
            metrics = getattr(record, "offline_metrics", None)
            if not complete:
                record.offline_pass = False
                record.error = "Missing required candidate video artifact."
                if isinstance(metrics, dict):
                    metrics["binary_correct"] = False
                    metrics["failure_reason"] = record.error
            normalize_eval_record_rewards(
                record,
                artifact_set_complete=complete,
                artifact_presence=present,
                artifact_validity=artifact_validity.get(sample_id, {}),
            )
        return records


__all__ = ["VALID_MODALITIES", "VerifierCallable", "VerifierRoute"]
