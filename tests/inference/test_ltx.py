"""Public-input isolation and fixed native-to-delivery contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference.backends.ltx.delivery import (
    Delivery,
    deliver_video,
    delivery_indices,
    pad_input,
    plan_deliveries,
)
from inference.backends.ltx.weights import (
    base_weight_map,
    load_transformer_weights,
    native_to_diffusers_key,
)
from inference.data.inputs import load_inputs
from inference.data.outputs import pending_inputs
from inference.data.timing import export_timing, load_timing, resolve_timing


class NativeLTXContractTests(unittest.TestCase):
    def test_fixed_length_and_reference_length_share_the_same_sampler_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs_path = root / "model_inputs.jsonl"
            timing_path = root / "reference-timing.jsonl"
            inputs_path.write_text(
                json.dumps(
                    {
                        "id": "case",
                        "evaluation_task": "id_grid_bfs",
                        "prompt": "rules",
                        "image": "/public/input.png",
                    }
                )
                + "\n"
            )
            timing_path.write_text(json.dumps({"id": "case", "frames": 81, "fps": 16}) + "\n")
            _, fixed_timings = resolve_timing(inputs_path, frames=81)
            _, ref_timings = resolve_timing(inputs_path, reference_timing=timing_path)
            fixed_delivery = plan_deliveries(fixed_timings)["case"]
            ref_delivery = plan_deliveries(ref_timings)["case"]
            self.assertEqual((fixed_delivery.native_frames, fixed_delivery.native_fps), (129, 24))
            self.assertEqual((fixed_delivery.frames, fixed_delivery.fps), (81, 16))
            self.assertEqual(fixed_delivery, ref_delivery)

    def test_reference_timing_roundtrip_preserves_every_source_state(self):
        for count in range(5, 158, 4):
            with self.subTest(frames=count):
                delivery = Delivery.from_reference_timing(count, 16)
                self.assertEqual(delivery.frames, count)
                self.assertEqual((delivery.native_frames - 1) % 8, 0)
                self.assertLessEqual(delivery.native_frames, 241)
                self.assertEqual(delivery.native_fps, 24)
                source = np.minimum(count - 1, np.floor(np.arange(delivery.native_frames) * 16 / 24 + .5).astype(int))
                np.testing.assert_array_equal(source[delivery_indices(delivery)], np.arange(count))
        self.assertEqual(Delivery.from_reference_timing(41, 16).native_frames, 65)
        self.assertEqual(Delivery.from_reference_timing(157, 16).native_frames, 241)

    def test_vbvr_long_and_nonbucket_lengths_preserve_every_reference_state(self):
        for count in (10, 12, 60, 161, 282):
            delivery = Delivery.from_reference_timing(count, 16)
            source = np.minimum(count - 1, np.floor(np.arange(delivery.native_frames) * 16 / 24 + .5).astype(int))
            np.testing.assert_array_equal(source[delivery_indices(delivery)], np.arange(count))

    def test_timing_export_and_load_never_read_reference_pixels_or_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [{"id": name, "evaluation_task": task,
                       "prompt": "public rules", "image": "/public/input.png"}
                      for name, task in (("id-case", "id_grid_bfs"), ("ood-case", "ext_grid_bfs"))]
            (root / "model_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
            frozen = [{"sample": {"id": row['id'], "video_num_frames": count, "video_fps": 16,
                                   "video": "/must-not-open.webm"}}
                      for row, count in zip(inputs, (41, 157))]
            manifest = root / "benchmark_manifest.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in frozen))
            output = root / "reference-timing.jsonl"
            self.assertEqual(export_timing(root, output)["samples"], 2)
            deliveries = plan_deliveries(load_timing(output, root / "model_inputs.jsonl"))
            self.assertEqual(deliveries['id-case'].native_frames, 65)
            self.assertEqual(deliveries['ood-case'].native_frames, 241)
            for line in output.read_text().splitlines():
                self.assertEqual(set(json.loads(line)), {"id", "frames", "fps"})
            export_timing(root, output)  # Idempotent for the same frozen metadata.
            frozen[0]['sample']['video_num_frames'] = 45
            manifest.write_text("".join(json.dumps(r) + "\n" for r in frozen))
            with self.assertRaisesRegex(ValueError, "different reference timing"):
                export_timing(root, output)
            output.write_text(json.dumps({"id": "id-case", "frames": 41, "fps": 16}) + '\n')
            with self.assertRaisesRegex(ValueError, "every public input"):
                load_timing(output, root / "model_inputs.jsonl")
            output.write_text(json.dumps({"id": "id-case", "frames": 41, "fps": 16, "answer": "secret"}) + '\n')
            with self.assertRaisesRegex(ValueError, "only id, frames, fps"):
                load_timing(output, root / "model_inputs.jsonl")

    def test_delivery_preserves_model_pixels_including_first_frame(self):
        delivery = Delivery(native_frames=9, native_fps=16, native_width=8, native_height=4,
                            frames=5, fps=16, width=4, height=4)
        puzzle = np.full((4, 4, 3), 253, np.uint8)
        canvas = pad_input(puzzle, delivery)
        self.assertTrue(np.array_equal(canvas[:, 2:6], puzzle))
        self.assertFalse(canvas[:, :2].any())
        self.assertFalse(canvas[:, 6:].any())
        decoded = np.stack([np.full((4, 8, 3), i, np.uint8) for i in range(9)])
        frames = deliver_video(decoded, delivery)
        self.assertFalse(np.array_equal(frames[0], puzzle))
        self.assertTrue(np.array_equal(frames[:, 0, 0, 0], [0, 1, 2, 3, 4]))

    def test_inputs_reject_reference_lengths_and_answers_and_partition_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = [{"id": f"puzzle-{i}", "evaluation_task": "id_task",
                     "prompt": "unchanged rules", "image": "input.png"}
                    for i in range(7)]
            path = Path(temporary) / "model_inputs.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            shards = [load_inputs(path, shard_index=i, shards=3) for i in range(3)]
            self.assertEqual(sorted(row["id"] for shard in shards for row in shard),
                             sorted(row["id"] for row in rows))
            rows[0]["video_num_frames"] = 123
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "public model_inputs"):
                load_inputs(path)
            del rows[0]["video_num_frames"]
            rows[0]["answer"] = "oracle answer"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "public model_inputs"):
                load_inputs(path)

    def test_base_weight_map_and_key_renaming(self):
        self.assertEqual(
            native_to_diffusers_key("adaln_single.linear.weight"),
            "time_embed.linear.weight",
        )
        self.assertEqual(
            native_to_diffusers_key("patchify_proj.weight"),
            "proj_in.weight",
        )
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "ltx"
            transformer_dir = model_dir / "transformer"
            transformer_dir.mkdir(parents=True)
            index = {"weight_map": {"proj_in.weight": "shard-00001.safetensors"}}
            (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(index)
            )
            folder, mapping = base_weight_map(model_dir)
            self.assertEqual(folder, transformer_dir)
            self.assertEqual(mapping, index["weight_map"])
            direct_folder, direct_mapping = base_weight_map(transformer_dir)
            self.assertEqual(direct_folder, transformer_dir)
            self.assertEqual(direct_mapping, index["weight_map"])

            class DummyTransformer:
                @staticmethod
                def state_dict():
                    return {"patchify_proj.weight": None, "adaln_single.linear.weight": None}

            shard_index = {
                "weight_map": {
                    "proj_in.weight": "shard-00001.safetensors",
                    "time_embed.linear.weight": "shard-00001.safetensors",
                }
            }
            (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(shard_index)
            )
            diffusers_tensors = {
                "proj_in.weight": "w_proj",
                "time_embed.linear.weight": "w_time",
            }
            native_tensors = {
                "patchify_proj.weight": "w_proj",
                "adaln_single.linear.weight": "w_time",
            }
            fake_st = type(
                "FakeSafetensorsTorch",
                (),
                {
                    "load_file": staticmethod(
                        lambda path, device="cpu": (
                            native_tensors
                            if str(path).endswith("epoch-10.safetensors")
                            else diffusers_tensors
                        )
                    )
                },
            )
            with patch.dict(
                "sys.modules",
                {"safetensors": type("S", (), {"torch": fake_st}), "safetensors.torch": fake_st},
            ):
                self.assertEqual(
                    load_transformer_weights(model_dir, None, DummyTransformer()),
                    native_tensors,
                )
                self.assertEqual(
                    load_transformer_weights(model_dir, transformer_dir, DummyTransformer()),
                    native_tensors,
                )
                self.assertEqual(
                    load_transformer_weights(
                        model_dir, Path(temporary) / "epoch-10.safetensors", DummyTransformer()
                    ),
                    native_tensors,
                )

    def test_pending_inputs_resumes_directly_from_valid_videos(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [
                {"id": "done", "evaluation_task": "t", "prompt": "p", "image": "/x.png"},
                {"id": "todo", "evaluation_task": "t", "prompt": "p", "image": "/x.png"},
            ]
            timings = {
                "done": {"id": "done", "frames": 41, "fps": 16},
                "todo": {"id": "todo", "frames": 81, "fps": 16},
            }
            with patch(
                "inference.data.outputs.valid_video",
                side_effect=lambda path, frames: path == output / "done" / "prediction.webm" and frames == 41,
            ):
                pending = pending_inputs(output, rows, timings)
            self.assertEqual([row["id"] for row in pending], ["todo"])
