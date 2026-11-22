# Vision Prompt Paraphrase

`Vision Prompt Paraphrase` redraws visual symbols (with matching prompt and legend updates) on paired In-Domain questions while preserving the exact same problem structure, rules, answer, and reference length.

## Task Design & Coverage

The catalog in [catalog.py](catalog.py) defines **33 tasks (`19` `Single-Step Group`, `14` `Multi-Step Group`) × 5 paired In-Domain questions = 165 questions**, stratified by the 246 In-Domain family counts:

| Family | Tasks | Questions | Representative visual redesigns |
| :--- | ---: | ---: | :--- |
| `geometry` | 10 | 50 | Surveying bar, beacon baseline, light-beam rays, antenna-linked triangle, masonry enclosure, tension-cable triangle, glazed quadrilateral, projection guide rail, measuring strap, star-point field |
| `next_state_prediction` | 4 | 20 | Diamond transfer lattice, separate instrument gauges, diamond distance-wave network, hourglass binary signals |
| `spatial_navigation` | 4 | 20 | Transit rails, rooms and vault doors, send/receive emblems, pneumatic valve junctions |
| `rearrangement_game` | 3 | 15 | Radial measuring carousel, horizontal pegs, physical beam magazines |
| `in_context_learning` | 3 | 15 | Height gauges, open color-station ring, paper punchcards |
| `graph_theory` | 2 | 10 | Circuit-board plug-terminal banks with insulated cables (`shortest_path_choice`), beacon rosettes joined by violet connectors (`degree_missing_value`) |
| `dynamics_game` | 2 | 10 | Interlocking bow-tie clearing boards, over-water cargo docks and tugs |
| `constraint_grid` | 2 | 10 | Compass pieces on region boards, ceramic Yin–Yang grids |
| `board_game` | 1 | 5 | Diamond-mesh Go intersection board preserving orthogonal adjacency |
| `physics` | 1 | 5 | Open-spoke meshing gear assemblies on mechanical plates |
| `spatial_3d` | 1 | 5 | Hinged assembly panels with numeral plates replacing dice pips |

Each redesign preserves the original relational structure and operation rules without introducing extra scale marks or answer hints: graphs preserve exact isomorphism and endpoint roles across relaid-out connectors, geometric tasks retain metric relations, and grid puzzles preserve cell coordinates and local neighborhoods.

## Observation & Full-Video Verification

`pairs.jsonl` binds each source In-Domain question, authored redesign plan, input image, observation image, and reference video at the source frame count and FPS. `prompts.py` and task-specific descriptions synchronize visible symbols, coordinates, and actions across `Input Image`, `Rule`, `Required Output`, and delivery constraints.

Task verifiers evaluate every decoded frame under deterministic full-video verification (**Valid Video Format**, **Full-Video Process**, and **Final State**):
- For geometric views, horizontal pegs, and beam magazines, a fixed spatial pixel permutation maps each decoded frame into the observer's coordinate frame.
- Other redesigned layouts are inspected directly at their mapped positions.

## Workflow

Prepare and qualify from a frozen In-Domain dataset:

```bash
python -m evaluation.prompt_paraphrase prepare-visual \
  --dataset <frozen-in-domain-suite> --output-root <visual-package> --workers 16
python -m evaluation.prompt_paraphrase qualify-visual \
  --dataset <visual-package> --output-root <controls> --workers 16
```

Run Reference Length inference on `model_inputs.jsonl` and `reference-timing.jsonl`, then score candidate shards and pair against the In-Domain baseline:

```bash
python -m evaluation.prompt_paraphrase score-visual \
  --dataset <visual-package> \
  --predictions-root <generation-run> --model <model-name> \
  --shard-index 0 --shards 1 --workers 4 --output-root <scores>
python -m evaluation.prompt_paraphrase rescore-visual-baseline \
  --dataset <visual-package> \
  --model <model-name> --predictions-root <id-rollouts> \
  --workers 16 --output-root <full-id-rescore>
python -m evaluation.prompt_paraphrase merge-visual \
  --dataset <visual-package> --shard-roots <scores> \
  --id-scores <full-id-rescore>/scores.jsonl --output-root <comparison>
```

`rescore-visual-baseline` also accepts `--baseline-scores <id-scores.jsonl>` instead of `--predictions-root`.
