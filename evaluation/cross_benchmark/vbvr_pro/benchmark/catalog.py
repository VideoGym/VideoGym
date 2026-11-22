"""Frozen benchmark identity and exhaustive enumeration."""

import json
from pathlib import Path

SOURCES = {
    "dataset_repo": "Video-Reason/VBVR-Pro-Bench",
    "dataset_revision": "fee09ecc6dcf78f44622cb7d80dc578e07ccf804",
}
SPLITS = {"In-Domain_50": "In_Domain", "Out-of-Domain_50": "Out_of_Domain"}
KNOWN_INCORRECT_REFERENCES = frozenset({
    "Out-of-Domain_50/O-62_gravity_physics_data-generator/00003",
})


def resolve_gt_root(root: str | Path) -> Path:
    resolved = Path(root).resolve()
    nested = resolved / "VBVR-Pro-Bench-Video"
    if not any((resolved / split).is_dir() for split in SPLITS) and any(
        (nested / split).is_dir() for split in SPLITS
    ):
        return nested
    return resolved


def catalog() -> dict[str, str]:
    return json.loads(Path(__file__).with_name("tasks.json").read_text())


def enumerate_samples(root: str | Path, *, complete: bool = True) -> list[dict]:
    root = resolve_gt_root(root)
    tasks = catalog()
    actual = {}
    for split in SPLITS:
        for folder in (root / split).glob("*"):
            if not folder.is_dir():
                continue
            if folder.name in actual:
                raise ValueError(f"Duplicate task across split directories: {folder.name}")
            actual[folder.name] = split
    if complete and actual != tasks:
        raise ValueError(
            f"Task catalog mismatch: missing={sorted(tasks.keys() - actual.keys())}, "
            f"extra={sorted(actual.keys() - tasks.keys())}; split identities must also match"
        )
    rows = []
    for task, split in sorted(actual.items()):
        if task not in tasks or tasks[task] != split:
            raise ValueError(f"Unknown task or wrong split: {task}")
        folder = root / split / task
        instances = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.isdigit())
        if complete and instances != [f"{i:05d}" for i in range(5)]:
            raise ValueError(f"Expected exactly five instances for {task}, got {instances}")
        for instance in instances:
            base = folder / instance
            row = {
                "sample_id": f"{split}/{task}/{instance}",
                "task": task,
                "split": SPLITS[split],
                "instance": instance,
            }
            for field, filename in [
                ("first_frame", "first_frame.png"),
                ("final_frame", "final_frame.png"),
                ("reference_video", "ground_truth.mp4"),
                ("metadata", "metadata.json"),
                ("prompt", "prompt.txt"),
            ]:
                path = base / filename
                if not path.is_file():
                    raise ValueError(f"Missing benchmark asset: {path}")
                row[field] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError("Empty benchmark")
    return rows
