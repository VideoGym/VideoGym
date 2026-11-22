"""Canonical integration of native scene composition and its full-video verifier."""

from data.common.base import AbstractPuzzleGenerator
from data.common.schemas import EvalRecord
from data.common.typed_answers import format_canonical_json_answer
from data.common.verifiers.candidates import resolve_candidate_video
from data.common.video_encoding import encode_rgb_frames_to_webm
from evaluation.shared.scenes.ablation import evidence
from evaluation.shared.scenes.catalog import FAMILIES, TASKS, module
from evaluation.shared.scenes.media import frames


class NativeCompositionGenerator(AbstractPuzzleGenerator):
    def __init__(self, output_dir, *, key, seed=None, video=True):
        super().__init__(output_dir)
        if key not in TASKS:
            raise ValueError("unknown native composition task")
        self.key, self.seed, self.video = key, seed, video

    def create_puzzle(self):
        m = module(self.key)
        source = m.sample(self.key, self.seed)
        if not evidence(source)["passed"]:
            raise ValueError("composition failed per-rule removal acceptance")
        from PIL import Image

        input_path = self.output_dir / "input.png"
        target_path = self.output_dir / "terminal.png"
        clip = frames(source) if self.video else None
        if clip is not None:
            first, last = Image.fromarray(clip[0]), Image.fromarray(clip[-1])
        elif TASKS[self.key].family == "navigation":
            first, last = m.render(source), m.render(source, m.solve(source)["states"][-1])
        else:
            first, last = m.render(source, 0), m.render(source, 1)
        first.save(input_path)
        last.save(target_path)
        video_path = self.output_dir / "reference.webm"
        if self.video and not encode_rgb_frames_to_webm(clip, video_path, fps=16):
            raise RuntimeError("native reference encoding failed")
        from data.common.ti2v_continuity import build_ti2v_continuity_paragraph

        prompt = m.prompt(source) + "\n\n" + build_ti2v_continuity_paragraph(
            timing="discrete" if TASKS[self.key].family == "next_state" else "continuous",
            moving_nouns="the task's native answer marks or moving objects",
            static_nouns="all source geometry, labels, background, and camera",
            hold_noun="the solved final state",
            allow_rule_teleports=TASKS[self.key].family == "navigation" and "portal" in self.key,
        )
        return {
            "id": "eval_" + self.key + "-source",
            "task_name": "eval_" + self.key,
            "task_family": FAMILIES[TASKS[self.key].family],
            "split": "eval",
            "ti2v_prompt": prompt,
            "answer": format_canonical_json_answer({"answer": m.solve(source)}),
            "structured_answer": source,
            "image": str(input_path),
            "solution_image_path": str(target_path),
            "solution_video_path": str(video_path) if self.video else None,
            "video_fps": 16 if self.video else None,
            "video_num_frames": len(clip) if self.video else None,
            "rendered_glyph_inventory": {"fixed_labels": []},
            "rendered_glyph_schema": {"fixed_labels": "source-native symbols"},
        }


def verify(samples, pred_root, *, modality="ti2v"):
    if modality != "ti2v":
        raise ValueError("TI2V only")
    results = []
    for sample in samples:
        record = EvalRecord(
            sample_id=sample.id,
            task_name=sample.task_name,
            task_family=sample.task_family,
            input_asset=sample.primary_input_image,
            split=sample.split,
        )
        try:
            path = resolve_candidate_video(pred_root, sample.id, sample.split)
            if path is None:
                raise FileNotFoundError("missing complete video")
            source = sample.semantic_input.get("semantic_state", sample.semantic_input)
            metrics = module(source["key"]).grade(source, path)
            record.prediction_asset = str(path)
            record.offline_pass = (
                metrics["whole_video_process_correct"]
                == metrics["video_final_state_correct"]
                == 1
            )
            record.offline_metrics = {
                **metrics,
                "artifact_results": {"video": record.offline_pass},
                "ground_truth_used": False,
                "reference_artifact_used": False,
            }
        except Exception as exc:
            record.offline_pass = False
            record.error = str(exc)
        results.append(record)
    return results
