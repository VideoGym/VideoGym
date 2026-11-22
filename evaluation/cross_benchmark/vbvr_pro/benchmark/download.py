"""Prepare the VBVR-Pro benchmark and its input manifest."""

import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from data.common.paths import to_relative
from evaluation.cross_benchmark.vbvr_pro.benchmark.catalog import SOURCES, enumerate_samples
from evaluation.cross_benchmark.vbvr_pro.reports import write_jsonl


def prepare_dataset(root: Path, endpoint: str = "https://huggingface.co") -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository, revision = SOURCES["dataset_repo"], SOURCES["dataset_revision"]
    with TemporaryDirectory(prefix="vbvr-download-", dir=root) as temporary:
        archive = Path(temporary) / "benchmark.tar.gz"
        urlretrieve(
            f"{endpoint}/datasets/{repository}/resolve/{revision}/VBVR-Pro-Bench-Video.tar.gz",
            archive,
        )
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(root, filter="data")
    dataset = root / "VBVR-Pro-Bench-Video"
    path_fields = ("first_frame", "final_frame", "reference_video", "metadata", "prompt")
    rows = [
        {
            **row,
            **{field: to_relative(row[field], root) for field in path_fields},
        }
        for row in enumerate_samples(dataset)
    ]
    manifest = root / "manifest.jsonl"
    write_jsonl(manifest, rows)
    print(json.dumps({"dataset": str(dataset), "samples": len(rows)}), flush=True)
    return manifest
