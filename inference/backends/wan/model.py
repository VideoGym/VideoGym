from __future__ import annotations

import math
from pathlib import Path
from PIL import Image
from inference.backends.wan.weights import RELEASES

MODEL_RELEASES = {
    **RELEASES,
    "a14b-base": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "5b-base": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}

NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, low quality, motion blur, distorted, artifacts"


def prompt_budget(tokenizer, rows, clean, maximum=None, *, include_negative=True):
    texts = [row["prompt"] for row in rows]
    if include_negative:
        texts.append(NEGATIVE_PROMPT)
    tokens = tokenizer(
        [clean(text) for text in texts], truncation=False, add_special_tokens=True
    )["input_ids"]
    required = max(map(len, tokens))
    budget = (
        maximum if maximum is not None else max(512, 128 * math.ceil(required / 128))
    )
    if budget < required:
        raise ValueError(
            f"complete prompt needs {required} tokens, budget is {budget}; truncation is not allowed"
        )
    return {"max_sequence_length": budget, "longest_prompt_tokens": required}


def resolve_model_inputs(
    model_path,
    rows,
    maximum=None,
    model_key="a14b",
    *,
    local_files_only=False,
):
    from diffusers.pipelines.wan.pipeline_wan_i2v import prompt_clean
    from transformers import AutoTokenizer

    if model_key not in MODEL_RELEASES:
        raise ValueError(f"unknown Wan model key: {model_key}")
    model = Path(model_path).resolve()
    if model_key == "5b-rlvr" and not (model / "pipeline.py").is_file():
        raise ValueError("RLVR requires its bundled pipeline.py")
    for row in rows:
        with Image.open(row["image"]) as image:
            if image.size != (512, 512):
                raise ValueError("VideoGym public input must be 512x512")
    tokenizer = AutoTokenizer.from_pretrained(
        model / "tokenizer", local_files_only=local_files_only
    )
    return prompt_budget(
        tokenizer,
        rows,
        prompt_clean,
        maximum,
        include_negative=model_key != "5b-rlvr",
    )
