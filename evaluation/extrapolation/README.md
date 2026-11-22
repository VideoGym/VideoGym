# Scale and Mechanism Extrapolation

Extrapolation evaluates whether a single trained task rule transfers beyond the parameter ranges or rule conditions observed during training while keeping the native renderer, visual symbols, and output contract unchanged. The [catalog](catalog.py) and [configuration](../catalog/extrapolation.toml) define **52 tasks (29 Scale and 23 Mechanism; 25 `Single-Step Group`, 27 `Multi-Step Group`) × 5 questions = 260 questions** at seed `20261027` (supporting `--count-per-task 20` / `--extend-from` to generate the paper's separate 1,040-question robustness audit).

Family quotas follow the largest-remainder distribution of all 246 In-Domain tasks: `geometry` 16, `next_state_prediction` 7, `spatial_navigation` 6, `rearrangement_game` 5, `graph_theory` 4, `in_context_learning` 4, `dynamics_game` 3, `constraint_grid` 3, `board_game` 2, `physics` 1, and `spatial_3d` 1.

## Scale Extrapolation (29 Tasks, 145 Questions)

Each scale task in [catalog.py](catalog.py) expands the effective solving burden recomputed from public inputs (objects, constraints, mandatory mechanisms, or optimal steps) rather than merely enlarging blank canvases:

- **Geometry:** convex-hull counting grows to 26 points and 10 hull vertices; symmetry completion to 20 reflected cells on a 16×16 grid; closest-pair search to 12 markers (66 pairs); interior-lattice counting to 18–28 interior points in 7–8-sided polygons; ray tracing to four successive mirror reflections (`ext_ray_reflect`, `Multi-Step Group`).
- **Spatial Navigation:** 21×21 mazes require shortest paths of 95–125 cells; capacity-two transport and color-matched delivery require 44–60 and 64–80 legal shortest steps; mandatory bridges (`ext_bridge_drop_tile`), conveyor chains (`ext_conveyor_belt_path`), and directional gates (`ext_directional_arrow_entry`) grow to 5, 7, and 4.
- **Graph Theory:** 16-node trees require at least 8 leaves for counting or 10 edges in the queried path while keeping native node radii, fonts, and edge widths.
- **Programs & States:** BFS depth grows to 12; legend-operator programs to 5 steps; 7-line Go programs to 9 legal moves; rigid 3D programs to 8 rotations.
- **Rearrangement Game:** 10 books require 6–8 optimal swaps; 10 pancakes require 6–7 flips; 7 Hanoi disks require 20–28 moves; 9 stacked blocks require 10–12 transfers; container loading uses 8 boxes.
- **Dynamics Game, Constraint Grid & Board Game:** snakes have 7 foods and an 8-cell body; shortest Sokoban walks require 8–10 pushes; one jewel swap clears two 5-piece lines (10 cells); N-Queens uses 12×12; Shikaku and Akari use 10×10; Lights Out uses 9×9 with a unique 5-press solution.

## Mechanism Extrapolation (23 Tasks, 115 Questions)

Mechanism extrapolation retains the same governing rule and solver at trained scale while testing unseen conditions or structural boundary cases:

| Representative task | Unseen condition / boundary case | Governing rule |
| :--- | :--- | :--- |
| `count_intersection_points` | Three concurrent lines plus intersections with a fourth | Count distinct intersection coordinates rather than line pairs |
| `region_count` | Nested circles and disconnected boundaries | Count bounded faces across disconnected components |
| `diagonal_intersection` | Concave quadrilateral with an external diagonal crossing | Intersect the lines defined by opposite vertices |
| `circle_inversion` | Internal source point mapping to an external image | Preserve the common ray and radius-squared distance product |
| `homothety_center` | Negative homothety (inverted orientation across the center) | Connect corresponding vertices through the common center |

Additional geometric and multi-family extensions reverse spiral similarity direction (contracting to expanding), reflection axes (intersecting to parallel), circle tangency (disjoint to intersecting or external to internal contact), arc span (minor to major), size/arithmetic step sign (growing to shrinking), translation direction, tree query direction (ancestor vs. descendant), branch-count differences, cellular-automaton birth thresholds, internal sandpile addition, and initial state-machine phases.

## Build, Qualify & Score

Generate the Extrapolation package, audit parameter bounds against the training profile, run full-video positive and negative controls, and export reference timing:

```bash
videogym benchmark generate \
  --config evaluation/catalog/extrapolation.toml \
  --training-profile <training-profile.json> \
  --output-root <extrapolation-package> --workers 16
videogym benchmark audit \
  --dataset <extrapolation-package> \
  --training-profile <training-profile.json>
python -m evaluation.execution.scoring.qualification \
  --manifest <extrapolation-package>/canonical_manifest.jsonl \
  --output <extrapolation-controls.jsonl> --workers 16
python -m inference.data.timing \
  --dataset <extrapolation-package> \
  --output <extrapolation-package>/reference-timing.jsonl
```

Score model predictions on the generated questions:

```bash
videogym benchmark score \
  --dataset <extrapolation-package> \
  --predictions-root <predictions> \
  --model <model-identity> --output-root <scores> --workers 16
```
