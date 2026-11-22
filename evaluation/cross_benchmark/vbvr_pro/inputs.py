from data.common.paths import to_absolute, to_relative
from evaluation.cross_benchmark.vbvr_pro.reports import write_json, write_jsonl
import json
from pathlib import Path
import numpy as np
from PIL import Image
from inference.backends.ltx.delivery import Delivery

FIXED_LENGTH_FRAMES = 81


def timing_audit(timings):
    rows = list(timings)
    frames = np.array([row["frames"] for row in rows])
    if not len(rows) or any(row["fps"] != 16 for row in rows):
        raise ValueError("expected nonempty timing metadata at 16 FPS")
    fixed_delivery = Delivery.from_reference_timing(FIXED_LENGTH_FRAMES, 16)
    return {
        "samples": len(rows),
        "fps": 16,
        "reference_frames": {
            "min": int(frames.min()),
            "median": float(np.median(frames)),
            "p95": float(np.percentile(frames, 95)),
            "max": int(frames.max()),
        },
        "fixed_length_frames": FIXED_LENGTH_FRAMES,
        "fixed_length_seconds": FIXED_LENGTH_FRAMES / 16,
        "ltx_native_frames": fixed_delivery.native_frames,
        "ltx_native_fps": fixed_delivery.native_fps,
        "references_longer_than_fixed_length": int((frames > FIXED_LENGTH_FRAMES).sum()),
    }


def reference_clock(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        if not count or abs(fps - 16) > 0.01:
            raise ValueError(f"invalid reference clock: {path}")
        return count, 16
    finally:
        capture.release()


def prepare_vbvr_pro(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if (
        len(rows) != 500
        or len({row["sample_id"] for row in rows}) != 500
        or len({r["task"] for r in rows}) != 100
    ):
        raise ValueError(
            "VBVR-Pro preparation requires the complete 500-sample benchmark"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty")
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    manifest_dir = manifest.parent
    _PATH_FIELDS = ("first_frame", "final_frame", "reference_video", "metadata", "prompt")
    inputs, timings, mapping = [], [], []
    for row in rows:
        resolved = {
            **row,
            **{field: to_absolute(row[field], manifest_dir) for field in _PATH_FIELDS},
        }
        identity = "vbvr-" + row["sample_id"].replace("/", "--")
        image_path = output / "inputs" / f"{identity}.png"
        with Image.open(resolved["first_frame"]) as image:
            image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS).save(
                image_path
            )
        inputs.append(
            {
                "id": identity,
                "evaluation_task": row["task"],
                "prompt": resolved["prompt"].read_text().strip(),
                "image": to_relative(image_path, output),
            }
        )
        frames, fps = reference_clock(resolved["reference_video"])
        timings.append({"id": identity, "frames": frames, "fps": fps})
        relative_sample = {
            **row,
            **{field: to_relative(resolved[field], output) for field in _PATH_FIELDS},
        }
        mapping.append({"id": identity, "sample": relative_sample})
    write_jsonl(output / "model_inputs.jsonl", inputs)
    write_jsonl(output / "reference-timing.jsonl", timings)
    write_jsonl(output / "benchmark-map.jsonl", mapping)
    audit = timing_audit(timings)
    write_json(output / "timing-audit.json", audit)
    return audit
