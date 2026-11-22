# Composition

Composition evaluates whether two or three rules taught only in separate In-Domain tasks can be chained serially or combined jointly in a single task on the native canvas without added HUDs, status bars, or external counters. The [catalog](catalog.py) and [scene catalog](../shared/scenes/catalog.py) define **34 tasks (14 `Single-Step Group`, 20 `Multi-Step Group`) × 5 questions = 170 questions** at seed `20261027` (supporting `--count-per-task 20` / `--extend-from` to generate the paper's separate 680-question robustness audit), stratified by the 246 In-Domain family counts:

| Task family | Tasks | Questions |
| :--- | ---: | ---: |
| `geometry` | 11 | 55 |
| `next_state_prediction` | 4 | 20 |
| `spatial_navigation` | 4 | 20 |
| `rearrangement_game` | 3 | 15 |
| `in_context_learning` | 3 | 15 |
| `graph_theory` | 2 | 10 |
| `constraint_grid` | 2 | 10 |
| `dynamics_game` | 2 | 10 |
| `board_game` | 1 | 5 |
| `physics` | 1 | 5 |
| `spatial_3d` | 1 | 5 |

## Ablation Non-Degeneracy Contracts

Every composition question enforces strict ablation non-degeneracy across its constituent In-Domain rules:

- **Deterministic output:** removing any single constituent rule on the same public input produces a distinct answer (for 5-option geometric tasks, each ablated target maps to a different visible candidate at least 24 pixels away from the joint target; three-rule compositions such as `midpoint_reflect_rotate` verify all three single-rule removals).
- **Constraint conjunction:** the joint solution set is nonempty and strictly smaller than each relaxed set obtained by dropping a constituent constraint, witnessed by explicit layouts that satisfy the remaining rules while violating the dropped rule (e.g., `queens_regions` combining Star Battle region/adjacency rules with N-Queens diagonal exclusion; `queens_akari` combining Akari light/wall constraints with N-Queens exclusion). The verifier accepts all valid joint solutions.
- **Paths and staged optimal processes:** relaxed navigation witnesses admit paths that violate the full transition system, and `locked_trade` requires the joint shortest plan to exceed both relaxed shortest lengths. Staged rearrangement tasks (`crane_reverse`, `dock_reverse`) require reaching an explicit intermediate target configuration before completing the second stage, with stagewise optimality verified across all legal shortest plans.

## Domain Design Summary

- **Geometry (11 tasks):** composes metric point constructions (`midpoint`, `orthogonal_projection`), axial reflections/parallels/bisectors (`reflection`, `parallel`, `angle_bisector`), circle inversion (`circle_inversion`), and rotations (`point_rotation`) on the shared Euclidean canvas.
- **Graph Theory (2 tasks):** `subtree_leaves` combines descendant subtree membership with leaf counting; `ancestor_subtree_diameter` chains lowest common ancestor (LCA), induced subtree extraction (at least 6 nodes, 3 leaves, and a 4-edge diameter), and diameter endpoint identification.
- **Spatial Navigation (4 tasks):** combines bridge placement (`bridge_drop_tile`), one-way portals (`one_way_portal_chain`), conveyor belts (`conveyor_belt_path`), and directional gates (`directional_arrow_entry`) on the native grid (`bridge_conveyor`, `bridge_arrow`, `portal_conveyor`, `portal_arrow`).
- **In-Context Learning & Next State Prediction (7 tasks):** combines geometric progression rules (`increasing_size_sequence`, `alternating_size_sequence`, `growing_polygon_step`) with cyclic color schedules (`tile_color_sequence`) in `size_color`, `growth_color`, and `polygon_color`, or chains cellular-automaton transitions in `life_xor`, `xor_life`, `life_xor_life`, and `life_rowtable`.
- **Rearrangement Game, Dynamics Game, Board Game, Physics & Spatial 3D (8 tasks):** covers constrained swaps (`locked_trade`), two-stage crane/dock reversal (`crane_reverse`, `dock_reverse`), ordered clear-and-gravity interactions (`candy_gravity`, `gravity_candy`), Go stone-program execution followed by maximum-capture search (`program_max_capture`), elastic collision with reflecting boundaries (`collision_reflection`), and 3D net folding followed by rigid rotation (`fold_rotate`).
- **Constraint Grid (2 tasks):** combines N-Queens placement with region constraints (`queens_regions`) or Akari illumination (`queens_akari`).

## Build, Qualify & Score

Generate the Composition questions, verify ablation non-degeneracy, run full-video positive and negative controls (reference, alternative valid solution, rule violation, wrong answer, static input, reversal, skipped process, scene erasure, and transient foreign object), and export reference timing:

```bash
videogym benchmark generate \
  --config evaluation/catalog/composition.toml \
  --training-profile <training-profile.json> \
  --output-root <composition-package> --workers 24
videogym benchmark audit \
  --dataset <composition-package> \
  --training-profile <training-profile.json>
python -m evaluation.execution.scoring.qualification \
  --manifest <composition-package>/canonical_manifest.jsonl \
  --output <composition-controls.jsonl> --workers 24
python -m inference.data.timing \
  --dataset <composition-package> \
  --output <composition-package>/reference-timing.jsonl
```

Score model predictions on the generated questions:

```bash
videogym benchmark score \
  --dataset <composition-package> \
  --predictions-root <predictions> \
  --model <model-identity> --output-root <scores> --workers 24
```
