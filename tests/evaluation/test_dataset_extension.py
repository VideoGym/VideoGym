"""Dataset questions and their media survive budget expansion and resume."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.execution.datasets.generation import generate_dataset, _sample_generation
from evaluation.execution.datasets.records import read_rows, rebase_sample_media
from evaluation.execution.datasets.selection import subset_dataset
from evaluation.execution.datasets.validation import validate_relation
from evaluation.catalog.index import evaluation_tasks


class DatasetExtensionTests(unittest.TestCase):
    def test_seed_sampling_exhaustion_is_retried_but_runtime_bugs_propagate(self):
        for task, message in (
            ("directional_arrow_entry", "directional_arrow_entry: failed to sample scenario."),
            ("grid_bfs", "grid_bfs could not sample the requested distance/source constraints"),
            ("eval_ancestor_subtree_diameter", "ancestor_subtree_diameter: native graph sampling exhausted"),
        ):
            with patch("evaluation.execution.datasets.generation._generate",
                       side_effect=RuntimeError(message)):
                self.assertIsNone(_sample_generation(task, {}, 1, Path("unused"), video=False))
        with patch("evaluation.execution.datasets.generation._generate",
                   side_effect=RuntimeError("decoder failed")):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                _sample_generation("directional_arrow_entry", {}, 1, Path("unused"), video=False)

    def test_composition_exposure_checks_reject_trained_joint_mechanisms(self):
        case = evaluation_tasks({"catalog": {"tracks": ["composition"]}})["polygon_color_composition"]
        profile = {
            "task_counts": {source: 1 for source in case.primitives},
            "mechanism_sets": {source: [source] for source in case.primitives},
        }
        validate_relation(case, {}, profile)
        profile["task_counts"]["joint_source"] = 1
        profile["mechanism_sets"]["joint_source"] = list(case.mechanisms)
        with self.assertRaisesRegex(ValueError, "already exposed"):
            validate_relation(case, {}, profile)
        del profile["task_counts"][case.primitives[0]]
        with self.assertRaisesRegex(ValueError, "primitive is absent"):
            validate_relation(case, {}, profile)

    def test_extend_and_resume_preserve_parent_and_generate_distinct_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "schema": "videogym.video.training-coverage",
                "model_dataset": "fixture", "training_samples": 1,
                "task_counts": {"midpoint": 1},
                "training_sample_ids": [], "training_seeds": {},
            }))
            args = argparse.Namespace(
                config={"catalog": {"tasks": ["id_midpoint"], "tracks": ["id"]}},
                output_root=root / "parent", training_profile=profile,
                count_per_task=2, workers=1, seed=41, resume=False,
            )
            self.assertEqual(generate_dataset(args)["samples"], 2)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in args.output_root.rglob("*") if p.is_file()}
            original = read_rows(args.output_root / "benchmark_manifest.jsonl")
            args.extend_from = args.output_root
            args.output_root = root / "extended"
            args.count_per_task = 4
            self.assertEqual(generate_dataset(args)["samples"], 4)
            expanded = read_rows(args.output_root / "benchmark_manifest.jsonl")
            self.assertEqual(len({r["sample"]["seed"] for r in expanded}), 4)
            for row in original:
                rebased = {
                    **row,
                    "sample": rebase_sample_media(
                        row["sample"], from_dir=args.extend_from, to_dir=args.output_root
                    ),
                }
                self.assertIn(rebased, expanded)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            args.resume = True
            self.assertEqual(generate_dataset(args)["samples"], 4)
            self.assertEqual(read_rows(args.output_root / "benchmark_manifest.jsonl"), expanded)
            args.output_root = root / "smaller"
            args.resume = False
            args.count_per_task = 1
            with self.assertRaisesRegex(ValueError, "cannot discard"):
                generate_dataset(args)

    def test_task_subset_preserves_questions_for_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text(json.dumps({
                "schema": "videogym.video.training-coverage",
                "model_dataset": "fixture", "training_samples": 2,
                "task_counts": {"midpoint": 1, "parallel": 1},
                "training_sample_ids": [], "training_seeds": {},
            }))
            args = argparse.Namespace(
                config={"catalog": {"tasks": ["id_midpoint", "id_parallel"], "tracks": ["id"]}},
                output_root=root / "parent", training_profile=profile,
                count_per_task=2, workers=1, seed=41, resume=False,
            )
            self.assertEqual(generate_dataset(args)["samples"], 4)
            original = [r for r in read_rows(args.output_root / "benchmark_manifest.jsonl")
                        if r["evaluation_task"] == "id_midpoint"]
            subset = root / "subset"
            result = subset_dataset(args.output_root, subset, 2, tasks=["id_midpoint"])
            self.assertEqual(result["counts"], {"id_midpoint": 2})
            rebased_subset = [
                {
                    **row,
                    "sample": rebase_sample_media(
                        row["sample"], from_dir=args.output_root, to_dir=subset
                    ),
                }
                for row in original
            ]
            self.assertEqual(read_rows(subset / "benchmark_manifest.jsonl"), rebased_subset)
            with self.assertRaisesRegex(ValueError, "present in the dataset"):
                subset_dataset(args.output_root, root / "invalid", 2, tasks=["id_reflection"])
            args.config["catalog"]["tasks"] = ["id_midpoint"]
            args.extend_from = subset
            args.output_root = root / "extended"
            args.count_per_task = 4
            self.assertEqual(generate_dataset(args)["samples"], 4)
            expanded = read_rows(args.output_root / "benchmark_manifest.jsonl")
            for row in rebased_subset:
                rebased = {
                    **row,
                    "sample": rebase_sample_media(
                        row["sample"], from_dir=subset, to_dir=args.output_root
                    ),
                }
                self.assertIn(rebased, expanded)


if __name__ == "__main__":
    unittest.main()
