"""Task-text paraphrase variants of the in-domain evaluation set."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from data.common.io import matches_schema
from data.common.paths import to_absolute, to_relative
from evaluation.execution.datasets.records import (
    read_rows,
    rebase_sample_media,
    write_json,
)

TASK_SECTIONS = ("Input Image", "Rule", "Required Output")
PROTOCOL = "videogym.text-paraphrase"


def task_sections(text):
    blocks = text.split("\n\n")
    if len(blocks) < 3 or any(
        not blocks[i].startswith(name + ": ") for i, name in enumerate(TASK_SECTIONS)
    ):
        raise ValueError("expected Input Image, Rule and Required Output blocks")
    return {name: blocks[i][len(name) + 2:] for i, name in enumerate(TASK_SECTIONS)}, blocks[3:]


def _validate_sections(sections):
    if set(sections) != set(TASK_SECTIONS):
        raise ValueError("rewrite must cover all three task sections")
    for name in TASK_SECTIONS:
        after = sections[name]
        if not isinstance(after, str) or not after.strip() or "\n" in after:
            raise ValueError(f"{name} must be a nonempty paragraph")


def _load_rewrites(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {}
    for record in records:
        if set(record["source"]) != set(TASK_SECTIONS):
            raise ValueError("rewrite must cover all three task sections")
        _validate_sections(record["rewrite"])
        if not record["ids"]:
            raise ValueError("rewrite has no source IDs")
        for sid in record["ids"]:
            if sid in by_id:
                raise ValueError("duplicate rewrite sample ID")
            by_id[sid] = record
    return records, by_id


def prepare(dataset, output, rewrites):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    rows = [row for row in read_rows(dataset / "benchmark_manifest.jsonl")
            if row["track"] == "id" and row["role"] == "evaluation"]
    if not rows:
        raise ValueError("Text Prompt Paraphrase evaluation requires ID samples")
    if len({r["sample"]["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate ID sample")
    _, by_id = _load_rewrites(Path(rewrites))
    if set(by_id) != {row["sample"]["id"] for row in rows}:
        raise ValueError("rewrites must cover each ID sample exactly once")
    inputs, timing, pairs, presented = [], [], [], []
    for row in rows:
        sample = rebase_sample_media(row["sample"], from_dir=dataset, to_dir=output)
        sid = sample["id"]
        record = by_id[sid]
        original_text = sample["prompt"]
        source, suffix = task_sections(original_text)
        if source != record["source"] or sample["task_name"] != record["task"]:
            raise ValueError(f"rewrite does not match source sample: {sid}")
        text = "\n\n".join([f"{name}: {record['rewrite'][name]}" for name in TASK_SECTIONS] + suffix)
        inputs.append({"id": sid, "evaluation_task": row["evaluation_task"],
                       "prompt": text, "image": sample["image"]})
        timing.append({"id": sid, "frames": sample["video_num_frames"], "fps": sample["video_fps"]})
        pairs.append({"id": sid, "evaluation_task": row["evaluation_task"],
                      "baseline_prompt": original_text, "variant_prompt": text})
        presented.append({**row, "sample": sample})
    if output.exists() and any(output.iterdir()):
        raise ValueError("Text Prompt Paraphrase output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("model_inputs.jsonl", inputs), ("reference-timing.jsonl", timing),
                         ("pairs.jsonl", pairs), ("benchmark_manifest.jsonl", presented)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    record = {
        "protocol": PROTOCOL,
        "dataset": to_relative(dataset, output),
        "samples": len(rows),
        "tasks": len({r["evaluation_task"] for r in rows}),
        "families": dict(sorted(Counter(r["sample"]["task_family"] for r in rows).items())),
        "evaluation_protocol": "reference_length",
    }
    write_json(output / "variant.json", record)
    return record


def load_package(dataset):
    root = Path(dataset).resolve()
    metadata = json.loads((root / "variant.json").read_text())
    if not matches_schema(metadata.get("protocol"), PROTOCOL) or metadata.get("evaluation_protocol") != "reference_length":
        raise ValueError("expected the Reference Length text package")
    rows = read_rows(root / "benchmark_manifest.jsonl")
    inputs = read_rows(root / "model_inputs.jsonl")
    timings = read_rows(root / "reference-timing.jsonl")
    pairs = read_rows(root / "pairs.jsonl")
    source_root = to_absolute(metadata["dataset"], root)
    source_rows = read_rows(source_root / "benchmark_manifest.jsonl")
    source = {r["sample"]["id"]: r for r in source_rows if r["track"] == "id" and r["role"] == "evaluation"}
    if len(rows) != metadata["samples"] or not len(rows) == len(inputs) == len(timings) == len(pairs):
        raise ValueError("incomplete text package")
    seen = set()
    for row, public, clock, pair in zip(rows, inputs, timings, pairs):
        sample = row["sample"]
        sid = sample["id"]
        if sid in seen or sid not in source:
            raise ValueError("duplicate or unknown source sample")
        seen.add(sid)
        expected = {
            **source[sid],
            "sample": rebase_sample_media(source[sid]["sample"], from_dir=source_root, to_dir=root),
        }
        original = expected["sample"]["prompt"]
        before, suffix = task_sections(original)
        after, new_suffix = task_sections(public["prompt"])
        _validate_sections(after)
        if after == before:
            raise ValueError(f"variant prompt did not rewrite task sections: {sid}")
        invariant = lambda value: {key: item for key, item in value.items() if key != "features"}
        if invariant(row) != invariant(expected) or suffix != new_suffix:
            raise ValueError("text variant changed a source invariant")
        if public != {"id": sid, "evaluation_task": row["evaluation_task"], "prompt": public["prompt"], "image": sample["image"]}:
            raise ValueError("public condition differs from source sample")
        if clock != {"id": sid, "frames": sample["video_num_frames"], "fps": sample["video_fps"]}:
            raise ValueError("reference clock changed")
        if pair["id"] != sid or pair["evaluation_task"] != row["evaluation_task"] or pair["baseline_prompt"] != original or pair["variant_prompt"] != public["prompt"]:
            raise ValueError("pairing differs from source sample")
    if seen != set(source):
        raise ValueError("text package must cover the complete ID set")
    return metadata, rows


def compare(id_scores, variant):
    """Pair the ID baseline and text variant by sample ID."""
    def indexed(path, baseline):
        result = {}
        for row in read_rows(path):
            if row["track"] != "id":
                if baseline:
                    continue
                raise ValueError("Text Prompt Paraphrase results must contain only ID evaluation samples")
            sid = row["sample_id"]
            if sid in result or row["success"] not in (0, 1):
                raise ValueError("duplicate sample or nonbinary score")
            result[sid] = row
        return result
    originals, variants = indexed(id_scores, True), indexed(variant, False)
    if not variants or originals.keys() != variants.keys():
        raise ValueError("comparison requires the same complete ID sample set")
    groups = defaultdict(list)
    transitions = {f"{a}_to_{b}": 0 for a in (0, 1) for b in (0, 1)}
    for sid, row in variants.items():
        base = originals[sid]
        if base["evaluation_task"] != row["evaluation_task"]:
            raise ValueError("paired task identity differs")
        groups[row["evaluation_task"]].append((base["success"], row["success"]))
        transitions[f"{base['success']}_to_{row['success']}"] += 1
    def summary(values):
        count = len(values)
        a, b = map(sum, zip(*values))
        return {"samples": count, "id_correct": a, "variant_correct": b,
                "id_success_rate": a / count, "variant_success_rate": b / count,
                "success_rate_difference": (b - a) / count}
    tasks = {task: summary(values) for task, values in sorted(groups.items())}
    return {"tasks": tasks, "overall": summary([v for g in groups.values() for v in g]),
            "paired_transitions": dict(sorted(transitions.items())),
            "id_task_macro_success": sum(r["id_success_rate"] for r in tasks.values()) / len(tasks),
            "variant_task_macro_success": sum(r["variant_success_rate"] for r in tasks.values()) / len(tasks)}
