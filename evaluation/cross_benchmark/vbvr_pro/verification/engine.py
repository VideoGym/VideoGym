from evaluation.cross_benchmark.vbvr_pro.verification.registry import engines, engine_name
from evaluation.cross_benchmark.vbvr_pro.vision.bandwidth import raw_observations
import json
from pathlib import Path
import numpy as np
import cv2
from evaluation.cross_benchmark.vbvr_pro.media import video as media


def result(status: str, checks: dict, evidence: dict, **extra) -> dict:
    score = {"correct": 1, "incorrect": 0, "missing": 0, "error": None, "unverified": None}[status]
    return {"status": status, "score": score, "checks": checks, "evidence": evidence, **extra}


def verify_frames(
    task: str,
    frames: list[np.ndarray],
    first: np.ndarray,
    final: np.ndarray,
    metadata: dict,
    reference: list[np.ndarray],
    *,
    observation_fps: float = media.DEFAULT_FPS,
) -> dict:
    """All arrays are BGR; complete references retain canonical coordinates.

    Appearance templates use the candidate's registered observation bandwidth.

    A semantic engine must supply nonempty named boolean checks, including all
    task obligations. Scores, similarities and confidence cannot substitute for
    the checks. Unknown tasks and observation errors remain unresolved.
    """
    try:
        if not frames:
            return result("incorrect", {"decodable_video": False}, {})
        observed_height, observed_width = frames[0].shape[:2]
        target_height, target_width = first.shape[:2]
        if observed_width < min(512, target_width) or observed_height < min(512, target_height):
            return result(
                "incorrect",
                {"supported_observation_resolution": False},
                {},
                reason="Candidate is below the registered 512px observation resolution",
            )
        raw_frames, raw_first, raw_final, raw_reference = frames, first, final, reference
        frames = media.normalize(frames, first)
        if observed_width < target_width or observed_height < target_height:

            def same_bandwidth(frame):
                low = cv2.resize(frame, (observed_width, observed_height), interpolation=cv2.INTER_AREA)
                return cv2.resize(low, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            first, final = same_bandwidth(first), same_bandwidth(final)
            reference = [same_bandwidth(frame) for frame in reference]
        preserved, evidence = media.input_preserved(frames[0], first)
        evidence["observation_resolution"] = [observed_width, observed_height]
        evidence["observation_fps"] = observation_fps
        checks = {"conditioning_preserved": preserved}
        if not preserved:
            return result("incorrect", checks, evidence)
        for engine in engines():
            if engine.__name__ == "evaluation.cross_benchmark.vbvr_pro.verification.scene.verifier":
                # Keep lossless semantic roles and encoder pixels before
                # bandwidth registration. Observation arrays retain canonical
                # coordinates and their matched spatial bandwidth.
                with raw_observations(raw_first, raw_final, raw_reference, observation_fps=observation_fps):
                    decision = engine.verify(task, frames, first, final, metadata, reference)
            elif engine.__name__ == "evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection" and task.split("_", 1)[0] == "G-221":
                # Keep the established geometric grid, but retain actual native
                # colour evidence: interpolation can turn a cyan source edge
                # into an apparent extra blue component.
                size = (observed_width, observed_height)
                decision = engine.verify(
                    task, frames, first, final, metadata, reference,
                    native_observation=(
                        cv2.resize(raw_first, size, interpolation=cv2.INTER_AREA),
                        raw_frames[-1],
                    ),
                )
            elif engine.__name__ == "evaluation.cross_benchmark.vbvr_pro.verification.annotation.selection" and task.split("_", 1)[0] == "G-250":
                decision = engine.verify(
                    task, frames, first, final, metadata, reference,
                    observation_resolution=(observed_width, observed_height),
                )
            else:
                decision = engine.verify(task, frames, first, final, metadata, reference)
            if decision is None:
                continue
            semantic_checks = decision["checks"]
            if not semantic_checks or any(type(value) not in (bool, np.bool_) for value in semantic_checks.values()):
                raise TypeError("Semantic engines must return named boolean predicates")
            if type(decision["passed"]) not in (bool, np.bool_):
                raise TypeError("Semantic verdict must be boolean, never a numeric confidence")
            semantic_checks = {name: bool(value) for name, value in semantic_checks.items()}
            passed = all(semantic_checks.values())
            if bool(decision["passed"]) != passed:
                raise ValueError("Semantic engine verdict disagrees with its predicates")
            checks.update(semantic_checks)
            evidence.update(decision.get("evidence", {}))
            return result(
                "correct" if all(checks.values()) else "incorrect",
                checks,
                evidence,
                engine=engine_name(engine),
            )
        return result("unverified", checks, evidence, reason="No implemented semantic contract for this task")
    except media.InvalidMedia as exc:
        return result("incorrect", {"valid_media": False}, {}, reason=str(exc))
    except Exception as exc:
        return result("error", {}, {}, reason=f"{type(exc).__name__}: {exc}")


def evaluate_sample(row: dict, prediction: str | Path | None) -> dict:
    identity = {key: row[key] for key in ("sample_id", "task", "split", "instance")}
    if prediction is None or not Path(prediction).is_file():
        return {**identity, **result("missing", {"prediction_present": False}, {})}
    try:
        first, final = media.image(row["first_frame"]), media.image(row["final_frame"])
        reference = media.decode(row["reference_video"])
        metadata = json.loads(Path(row["metadata"]).read_text())
    except Exception as exc:
        return {**identity, **result("error", {}, {}, reason=f"Reference error: {exc}")}
    try:
        video = media.decode(prediction)
    except media.InvalidMedia as exc:
        return {**identity, **result("incorrect", {"valid_media": False}, {}, reason=str(exc))}
    decision = verify_frames(
        row["task"], video.frames, first, final, metadata, reference.frames, observation_fps=video.fps
    )
    decision["evidence"].update(
        {"decoded_frames": len(video.frames), "fps": video.fps, "reference_frames": len(reference.frames)}
    )
    return {**identity, **decision}
