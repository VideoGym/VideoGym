# Prompt Paraphrase Evaluation

`Text Prompt Paraphrase` and `Vision Prompt Paraphrase` test generalization within familiar task patterns while preserving the exact underlying problem structure, rule meaning, answer, and output contract. Together they form two of the six evaluation levels in `VideoGym-Test` (and two of the seven generalization levels overall alongside `Cross-Benchmark`).

- **Text Prompt Paraphrase (246 tasks, 1,230 questions):** rewrites the `Input Image`, `Rule`, and `Required Output` sections synonymously on the exact same 1,230 In-Domain images and problems (`132` `Single-Step Group`, `114` `Multi-Step Group`).
- **Vision Prompt Paraphrase (33 tasks, 165 questions):** redraws visual symbols with matching prompt updates on 165 paired In-Domain questions (`19` `Single-Step Group`, `14` `Multi-Step Group`) while preserving the exact problem structure, rules, and answer. See the [Vision Prompt Paraphrase guide](vision_prompt_paraphrase/README.md).

## Text Prompt Paraphrase Workflow

Pass a source-bound rewrite corpus to `prepare-text`. Rewrites preserve images, semantic states, answers, shared closing constraints, and reference frame counts.

```bash
python -m evaluation.prompt_paraphrase prepare-text \
  --dataset <frozen-in-domain-dir> --rewrites <rewrites.jsonl> --output-root <text-package>
python -m evaluation.prompt_paraphrase qualify-text \
  --dataset <text-package> --output-root <text-controls> --workers 16
```

Run Reference Length inference on `model_inputs.jsonl` and `reference-timing.jsonl`, score completed shards, and pair against the In-Domain baseline under the exact same full-video verifier:

```bash
python -m evaluation.prompt_paraphrase score-text \
  --dataset <text-package> \
  --predictions-root <generation-run> --model <model-name> \
  --shard-index 0 --shards 1 --workers 4 --output-root <text-scores-shard>
python -m evaluation.prompt_paraphrase rescore-text-baseline \
  --dataset <text-package> --model <model-name> \
  --predictions-root <id-rollouts> --workers 16 --output-root <id-baseline-rescore>
python -m evaluation.prompt_paraphrase merge-text \
  --dataset <text-package> --shard-roots <text-scores-shard> \
  --id-scores <id-baseline-rescore>/scores.jsonl --output-root <text-comparison>
```

`rescore-text-baseline` also accepts `--baseline-scores <id-scores.jsonl>` instead of `--predictions-root` to reuse candidate paths recorded in an existing In-Domain score file.

## One-to-One Pairing

Each `Prompt Paraphrase` comparison evaluates paired source and rewritten questions under the same deterministic full-video verifier (`0_to_0`, `0_to_1`, `1_to_0`, `1_to_1` transitions and question-level retention). `benchmark_manifest.jsonl` retains the source question's canonical identity, `model_inputs.jsonl` contains the rewritten prompt, and `pairs.jsonl` records pairings and rewrite metadata.
