# Cross-Task Generalization

Cross-Task evaluates transfer to unseen tasks introducing new rules and solving objectives within the same 11 problem categories as training, without pairing to any In-Domain task.

## Coverage & Quotas

The [catalog](catalog.py) and [configuration](../catalog/cross_task.toml) define **33 tasks (23 `Single-Step Group`, 10 `Multi-Step Group`) × 5 questions = 165 questions** at seed `20261027` (supporting `--count-per-task 20` / `--extend-from` to generate the paper's separate 660-question robustness audit), allocated across all 11 families by the largest-remainder distribution of the 246 In-Domain tasks:

| Family | In-Domain tasks | Cross-Task tasks | Questions |
| :--- | ---: | ---: | ---: |
| `geometry` | 76 | 10 | 50 |
| `next_state_prediction` | 32 | 4 | 20 |
| `spatial_navigation` | 31 | 4 | 20 |
| `rearrangement_game` | 23 | 3 | 15 |
| `in_context_learning` | 19 | 3 | 15 |
| `graph_theory` | 17 | 2 | 10 |
| `dynamics_game` | 14 | 2 | 10 |
| `constraint_grid` | 13 | 2 | 10 |
| `board_game` | 8 | 1 | 5 |
| `physics` | 7 | 1 | 5 |
| `spatial_3d` | 6 | 1 | 5 |

## Domain Mechanisms

Each task is constructed so that local greedy heuristics or static-pattern shortcuts fail across the question set:

- **Geometry (10 tasks):** obstacle-avoiding Euclidean geodesic (`taut_obstacle_path`), minimum-weight octagon triangulation (`polygon_triangulation`), non-crossing minimum bichromatic matching (`bichromatic_matching`), minimum rectangle partition with holes (`rectangle_partition`), 7-half-plane intersection (`halfplane_intersection`), weighted Voronoi power cells (`power_cells`), projective homography landmark recovery (`projective_landmarks`), 3-disk union boundary (`disk_union_boundary`), discrete Minkowski sum (`minkowski_sum`), and minimum-width parallel support strip (`minimum_width_strip`).
- **Next State Prediction (4 tasks):** multi-input Petri net firing sequence (`petri_target_marking`), structural causal counterfactual inference (`causal_counterfactual`), Viterbi path on a hidden Markov model (`hidden_markov_path`), and signed delayed wave superposition (`delayed_wave_superposition`).
- **Spatial Navigation (4 tasks):** closed-loop pursuit evasion (`reactive_pursuer_escape`), collision-free synchronous two-robot exchange (`joint_robot_exchange`), maximum-reliability bridge routing (`reliable_bridge_route`), and topological winding-number routing (`winding_route`).
- **Rearrangement Game (3 tasks):** Gale–Shapley stable partner matching (`stable_partner_assignment`), makespan-optimal job-shop scheduling (`jobshop_schedule`), and exact tetromino packing (`polyomino_packing`).
- **In-Context Learning (3 tasks):** shared-bijection cryptogram decoding (`consistent_cryptogram`), two-state Mealy transducer induction (`hidden_transducer`), and parallel recursive grammar expansion (`recursive_rewrite`).
- **Graph Theory, Dynamics Game, Constraint Grid, Board Game, Physics & Spatial 3D (9 tasks):** integer maximum flow (`capacitated_max_flow`), minimum-weight directed arborescence (`minimum_arborescence`), wall-blocked bomb cascade timing (`bomb_chain_timing`), persistent rotor-router walk (`recurrent_rotor_router`), Nonogram (`nonogram_runs`), Kakuro (`kakuro_cross_sums`), adversarial Othello endgame (`othello_endgame`), multi-layer Snell refraction (`layered_refraction`), and 3D rigid polycube assembly (`polycube_assembly`).

## Rendering & Full-Video Verification

Question images use shared 4× supersampled primitives in `data/common/render_primitives.py` (`drawing.py`) and expose only essential numbers, symbols, and legends, placing full instructions in the prompt.

`engine.py` recomputes the valid solution set from public inputs (supporting multiple optimal plans, matchings, or packings) and enforces full-video verification across **Valid Video Format**, **Full-Video Process**, and **Final State**. Connected path or boundary answers may appear as a single monotone contour or segment-by-segment; independent markers must be committed without erasing or mutating the problem scene.

## Build & Qualify

Generate the package, audit exposure separation, run full-video positive/negative qualification controls, and export reference timing:

```bash
videogym benchmark generate --config evaluation/catalog/cross_task.toml \
  --training-profile <training-profile.json> \
  --output-root <cross-task-questions> --workers 8
videogym benchmark audit --dataset <cross-task-questions>
python -m evaluation.execution.scoring.qualification \
  --manifest <cross-task-questions>/canonical_manifest.jsonl \
  --output <qualification.jsonl> --workers 8
python -m inference.data.timing \
  --dataset <cross-task-questions> --output <cross-task-questions>/reference-timing.jsonl
```

Score generated candidate rollouts under deterministic full-video verification:

```bash
videogym benchmark score \
  --dataset <cross-task-questions> --predictions-root <rollouts-dir> \
  --model <model-name> --output-root <scores-dir> --workers 16
```
