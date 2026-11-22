from __future__ import annotations
import gc
import json

NEGATIVE_PROMPT = ""


def encode_prompt_contexts(model_path, rows, device, *, local_files_only=False):
    import torch
    from diffusers import LTX2ImageToVideoPipeline

    # Skip both transformer and audio components: their state does not enter
    # text encoding, and audio=None remains native throughout denoising.
    pipe = LTX2ImageToVideoPipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        audio_vae=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    pipe.text_encoder.to(device).eval()
    pipe.connectors.to(device).eval()
    contexts = {}
    prompts = list(
        dict.fromkeys([NEGATIVE_PROMPT, *(row["prompt"] for row in rows)])
    )
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            if len(pipe.tokenizer(prompt, add_special_tokens=True).input_ids) > 1024:
                raise ValueError(
                    "public prompt exceeds native encoder budget; do not truncate task rules"
                )
            text, mask, _, _ = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                device=device,
                max_sequence_length=1024,
            )
            video, _, _ = pipe.connectors(text, mask, padding_side="left")
            contexts[prompt] = video.cpu()
            print(
                json.dumps(
                    {
                        "phase": "encode_public_prompts",
                        "completed": index + 1,
                        "total": len(prompts),
                    }
                ),
                flush=True,
            )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return contexts
