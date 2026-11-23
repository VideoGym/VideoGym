(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );

  const prefersReducedMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ==================== 11 TASK FAMILIES MOSAIC DATA ==================== */
  const FAMILY_TILES = [
    {
      id: "graph_theory",
      name: "Graph Theory",
      tasks: 17,
      split: "17 SS",
      dominant: "ss",
      rate: 84.7,
      sample: "Trace All Edges Once",
      img: "media/tasks/0224.png",
      vid: "media/tasks/0224.mp4",
    },
    {
      id: "geometry",
      name: "Geometry",
      tasks: 76,
      split: "72 SS · 4 MS",
      dominant: "ss",
      rate: 78.9,
      sample: "Arc Center",
      img: "media/tasks/0010.png",
      vid: "media/tasks/0010.mp4",
    },
    {
      id: "in_context_learning",
      name: "In-Context Learning",
      tasks: 19,
      split: "16 SS · 3 MS",
      dominant: "ss",
      rate: 66.3,
      sample: "Growing Polygon Step",
      img: "media/tasks/0091.png",
      vid: "media/tasks/0091.mp4",
    },
    {
      id: "spatial_navigation",
      name: "Spatial Navigation",
      tasks: 31,
      split: "31 MS",
      dominant: "ms",
      rate: 52.9,
      sample: "Conveyor Belt Path",
      img: "media/tasks/0046.png",
      vid: "media/tasks/0046.mp4",
    },
    {
      id: "rearrangement_game",
      name: "Rearrangement Game",
      tasks: 23,
      split: "23 MS",
      dominant: "ms",
      rate: 52.2,
      sample: "Maze Rush Hour",
      img: "media/tasks/0134.png",
      vid: "media/tasks/0134.mp4",
    },
    {
      id: "board_game",
      name: "Board Game",
      tasks: 8,
      split: "1 SS · 7 MS",
      dominant: "ms",
      rate: 47.5,
      sample: "Go Max Capture",
      img: "media/tasks/0083.png",
      vid: "media/tasks/0083.mp4",
    },
    {
      id: "dynamics_game",
      name: "Dynamics Game",
      tasks: 14,
      split: "1 SS · 13 MS",
      dominant: "ms",
      rate: 42.9,
      sample: "Sokoban",
      img: "media/tasks/0204.png",
      vid: "media/tasks/0204.mp4",
    },
    {
      id: "physics",
      name: "Physics",
      tasks: 7,
      split: "1 SS · 6 MS",
      dominant: "ms",
      rate: 40.0,
      sample: "Mirror Aim Detector",
      img: "media/tasks/0141.png",
      vid: "media/tasks/0141.mp4",
    },
    {
      id: "constraint_grid",
      name: "Constraint Grid",
      tasks: 13,
      split: "13 SS",
      dominant: "ss",
      rate: 27.7,
      sample: "N-Queens Completion",
      img: "media/tasks/0146.png",
      vid: "media/tasks/0146.mp4",
    },
    {
      id: "next_state_prediction",
      name: "Next State Prediction",
      tasks: 32,
      split: "8 SS · 24 MS",
      dominant: "ms",
      rate: 19.4,
      sample: "Rotten Oranges",
      img: "media/tasks/0188.png",
      vid: "media/tasks/0188.mp4",
    },
    {
      id: "spatial_3d",
      name: "Spatial 3D",
      tasks: 6,
      split: "3 SS · 3 MS",
      dominant: "ms",
      rate: 13.3,
      sample: "Cube Net Fold",
      img: "media/tasks/0050.png",
      vid: "media/tasks/0050.mp4",
    },
  ];

  /* ==================== SYNCHRONIZED LIVE STAGES ==================== */
  const LIVE_STAGES = {
    hero: {
      eyebrow: "Synchronized Rollouts",
      title: "From In-Domain Mastery to Out-of-Domain Collapse",
      tabs: [
        {
          id: "id_vs_base",
          label: "In-Domain (56.3%)",
          caseId: "02_task_families_and_base_models__03_physics_mirror_aim_detector",
          finding:
            "In-Domain (Mirror Aim Detector): base LTX-2.3 bends light in mid-air (0.00%), whereas post-trained VideoGym-LTX-2.3 traces exact equal-angle reflections across four mirrors to trigger detector B (100% task accuracy).",
          panes: [
            {
              title: "Ground Truth",
              sub: "Reference · 41f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/reference.mp4",
              poster: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/input.png",
              frames: 41,
              note: "Four equal-angle bounces hit detector B.",
            },
            {
              title: "VideoGym-LTX-2.3",
              sub: "Post-Trained · 41f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/videogym_ltx23_PASS.mp4",
              poster: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/input.png",
              frames: 41,
              note: "Exact 4-mirror reflection; lights B green.",
            },
            {
              title: "LTX-2.3 Base",
              sub: "Base · 41f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/ltx23_base_FAIL.mp4",
              poster: "media/showcase/02_task_families_and_base_models/03_physics_mirror_aim_detector/input.png",
              frames: 41,
              note: "Curves ray in mid-air; lights wrong disks.",
            },
          ],
        },
        {
          id: "paraphrase",
          label: "Prompt Paraphrase",
          caseId: "01_paper_figures__08_fig4a_square_maze_paraphrase",
          finding:
            "Matched Square Maze: synonymous Text Prompt Paraphrase transfers cleanly (90.8% retained), whereas redrawing the walled maze as rail links collapses to 0.00% across all 14 Multi-Step tasks.",
          panes: [
            {
              title: "In-Domain",
              sub: "VideoGym-LTX-2.3 · 77f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/videogym_ltx23_in_domain.mp4",
              poster: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/input_in_domain.png",
              frames: 77,
              note: "Routes cleanly along open corridor.",
            },
            {
              title: "Text Paraphrase",
              sub: "VideoGym-LTX-2.3 · 77f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/videogym_ltx23_text_paraphrase.mp4",
              poster: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/input_in_domain.png",
              frames: 77,
              note: "Frozen LLM encoder preserves route.",
            },
            {
              title: "Vision Reference",
              sub: "Ground Truth · 77f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/reference_vision_paraphrase.mp4",
              poster: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/input_vision_paraphrase.png",
              frames: 77,
              note: "Identical underlying maze topology.",
            },
            {
              title: "Vision Paraphrase",
              sub: "VideoGym-LTX-2.3 · 77f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/videogym_ltx23_vision_paraphrase.mp4",
              poster: "media/showcase/01_paper_figures/08_fig4a_square_maze_paraphrase/input_vision_paraphrase.png",
              frames: 77,
              note: "Cuts straight across unlinked gap.",
            },
          ],
        },
        {
          id: "extrapolation",
          label: "Extrapolation",
          caseId: "04_within_familiar_patterns__03_extrap_multistep_scale_bridge_drop",
          finding:
            "Scale Extrapolation (Bridge Drop, 7×7 → 14×14): placing 2 bridges reaches 100% In-Domain accuracy, yet scaling to 5 water rows erases start S, recolors the agent, miscounts the bridge gauge, and extrudes a brown plank strip across dry floor on the way to G (0.00% across all Multi-Step Scale tasks).",
          panes: [
            {
              title: "7×7 In-Domain",
              sub: "VideoGym-LTX-2.3 · 21f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/in_domain_source_PASS.mp4",
              poster: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/input_in_domain.png",
              frames: 21,
              note: "Places 2 single-cell planks to reach G.",
            },
            {
              title: "14×14 Reference",
              sub: "Ground Truth · 33f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/extrapolation_reference.mp4",
              poster: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/input_extrapolation.png",
              frames: 33,
              note: "Steps cell by cell across 5 water rows.",
            },
            {
              title: "14×14 Rollout",
              sub: "VideoGym-LTX-2.3 · 33f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/extrapolation_rollout_FAIL.mp4",
              poster: "media/showcase/04_within_familiar_patterns/03_extrap_multistep_scale_bridge_drop/input_extrapolation.png",
              frames: 33,
              note: "Reaches G, but extrudes planks on dry land and miscounts pips.",
            },
          ],
        },
        {
          id: "composition",
          label: "Composition (0.59%)",
          caseId: "01_paper_figures__13_fig6a_bridge_conveyor_composition",
          finding:
            "Serial Multi-Step Composition: post-trained VideoGym-LTX-2.3 solves Conveyor Belt (100%) and Bridge Drop (100%) separately, yet ignores the conveyor push when both appear on one grid.",
          panes: [
            {
              title: "Rule A (Conveyor)",
              sub: "VideoGym-LTX-2.3 · 21f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/constituent_1_PASS.mp4",
              poster: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/input_constituent_1.png",
              frames: 21,
              note: "Obeys left-then-up conveyor pushes cleanly.",
            },
            {
              title: "Rule B (Bridge)",
              sub: "VideoGym-LTX-2.3 · 21f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/constituent_2_PASS.mp4",
              poster: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/input_constituent_2.png",
              frames: 21,
              note: "Drops two bridge tiles across water cleanly.",
            },
            {
              title: "Combined Reference",
              sub: "Ground Truth · 25f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/composition_reference.mp4",
              poster: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/input_composition.png",
              frames: 25,
              note: "Shifts left first, then bridges water.",
            },
            {
              title: "Combined Rollout",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/composition_FAIL.mp4",
              poster: "media/showcase/01_paper_figures/13_fig6a_bridge_conveyor_composition/input_composition.png",
              frames: 25,
              note: "Ignores conveyor; bridges wrong column.",
            },
          ],
        },
        {
          id: "cross_task",
          label: "Cross-Task (0.61%)",
          caseId: "05_beyond_familiar_patterns_and_n20_audit__02_crosstask_minimum_arborescence",
          finding:
            "Cross-Task (Minimum Directed Spanning Arborescence): despite 84.71% In-Domain accuracy across 17 Graph Theory tasks and familiar orange edge strokes, this unseen rule scores 0.00% across all 20 expanded test questions.",
          panes: [
            {
              title: "Ground Truth",
              sub: "Reference · 33f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/02_crosstask_minimum_arborescence/reference.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/02_crosstask_minimum_arborescence/input.png",
              frames: 33,
              note: "Minimum-weight directed tree from root S.",
            },
            {
              title: "VideoGym-LTX-2.3",
              sub: "Held-Out Rule · 33f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/02_crosstask_minimum_arborescence/videogym_ltx23_FAIL.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/02_crosstask_minimum_arborescence/input.png",
              frames: 33,
              note: "Two edges enter 1; leaves 2–5 unreachable (5 isolated).",
            },
          ],
        },
      ],
    },

    duration: {
      eyebrow: "Synchronized Rollouts",
      title: "Reference Length vs. Fixed 81-Frame Budget on the Same Input",
      tabs: [
        {
          id: "short_81f",
          label: "Shorter Budget (81f < 97f)",
          caseId: "01_paper_figures__06_fig3b_inbox_priority_short_81f",
          finding:
            "Inbox Priority Sort: when Fixed Length (81f) is shorter than Reference Length (97f), instead of moving cards faster one by one, post-trained VideoGym-LTX-2.3 illegally lifts multiple cards together.",
          panes: [
            {
              title: "Ground Truth",
              sub: "Reference · 97f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/reference.mp4",
              poster: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/input.png",
              frames: 97,
              note: "Moves one card at a time via buffer.",
            },
            {
              title: "Reference Length",
              sub: "VideoGym-LTX-2.3 · 97f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/videogym_ltx23_ref_PASS.mp4",
              poster: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/input.png",
              frames: 97,
              note: "Clean single-card buffer swaps.",
            },
            {
              title: "Fixed Length (81f)",
              sub: "VideoGym-LTX-2.3 · 81f",
              verdict: "warn",
              badge: "Shortcut Fail",
              src: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/videogym_ltx23_fixed81f_FAIL.mp4",
              poster: "media/showcase/01_paper_figures/06_fig3b_inbox_priority_short_81f/input.png",
              frames: 81,
              note: "Sorts stack by lifting 2 cards at once.",
            },
          ],
        },
        {
          id: "long_81f",
          label: "Longer Budget (81f > 17f)",
          caseId: "01_paper_figures__07_fig3b_symmetry_completion_long_81f",
          finding:
            "Symmetry Completion: when Fixed Length (81f) exceeds Reference Length (17f), post-trained VideoGym-LTX-2.3 finishes the mirror reflection early and then keeps painting spurious extra cells.",
          panes: [
            {
              title: "Ground Truth",
              sub: "Reference · 17f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/reference.mp4",
              poster: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/input.png",
              frames: 17,
              note: "Reflects cells across axis and holds still.",
            },
            {
              title: "Reference Length",
              sub: "VideoGym-LTX-2.3 · 17f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/videogym_ltx23_ref_PASS.mp4",
              poster: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/input.png",
              frames: 17,
              note: "Matches exact mirror reflection cleanly.",
            },
            {
              title: "Fixed Length (81f)",
              sub: "VideoGym-LTX-2.3 · 81f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/videogym_ltx23_fixed81f_FAIL.mp4",
              poster: "media/showcase/01_paper_figures/07_fig3b_symmetry_completion_long_81f/input.png",
              frames: 81,
              note: "Keeps painting extra cells after solving.",
            },
          ],
        },
        {
          id: "epochs_progression",
          label: "Epochs 2 → 10 vs. 81f",
          caseId: "03_epoch_and_duration_dynamics__01_multistep_epochs_2_to_10_ref_vs_fixed",
          finding:
            "Cooperative Open Door across Epochs 2 → 10: Reference Length converges from early tile and move errors (Epochs 2–4) to clean execution by Epochs 6–10, while Epoch 10 at Fixed 81f opens the door early and duplicates agents.",
          panes: [
            {
              title: "Epoch 2 · Ref.",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/epoch02_ref.mp4",
              poster: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/input.png",
              frames: 25,
              note: "Opens door but spawns block at (2, 4).",
            },
            {
              title: "Epoch 6 · Ref.",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/epoch06_ref.mp4",
              poster: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/input.png",
              frames: 25,
              note: "Valid plate press and passage emerge.",
            },
            {
              title: "Epoch 10 · Ref.",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/epoch10_ref.mp4",
              poster: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/input.png",
              frames: 25,
              note: "Clean synchronized multi-agent rollout.",
            },
            {
              title: "Epoch 10 · 81f",
              sub: "VideoGym-LTX-2.3 · 81f",
              verdict: "warn",
              badge: "Shortcut Fail",
              src: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/epoch10_fixed81f.mp4",
              poster: "media/showcase/03_epoch_and_duration_dynamics/01_multistep_epochs_2_to_10_ref_vs_fixed/input.png",
              frames: 81,
              note: "Opens door early; duplicates agents before flag.",
            },
          ],
        },
      ],
    },

    generalization: {
      eyebrow: "Synchronized Rollouts",
      title: "Symbol Redrawing, Trend Bias & Multi-Rule Composition",
      tabs: [
        {
          id: "3rule_comp",
          label: "3-Rule Composition (100% × 3 → 0%)",
          caseId: "05_beyond_familiar_patterns_and_n20_audit__01_comp_3rule_ancestor_subtree_diameter",
          finding:
            "Graph Theory 3-Rule Composition: post-trained VideoGym-LTX-2.3 reaches 100% In-Domain accuracy on Closest Common Ancestor, Subtree Size, and Diameter Endpoints individually (across 20 questions each), yet scores 0.00% when combined.",
          panes: [
            {
              title: "Rule 1: Ancestor",
              sub: "VideoGym-LTX-2.3 · 41f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/c1_cca_PASS.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/input_c1_cca.png",
              frames: 41,
              note: "Finds closest common ancestor.",
            },
            {
              title: "Rule 2: Subtree",
              sub: "VideoGym-LTX-2.3 · 41f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/c2_subtree_PASS.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/input_c2_subtree.png",
              frames: 41,
              note: "Isolates rooted subtree cleanly.",
            },
            {
              title: "Rule 3: Diameter",
              sub: "VideoGym-LTX-2.3 · 41f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/c3_diameter_PASS.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/input_c3_diameter.png",
              frames: 41,
              note: "Marks tree diameter endpoints.",
            },
            {
              title: "Combined Ref.",
              sub: "Ground Truth · 41f",
              verdict: "ref",
              badge: "Ref.",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/composition_reference.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/input_composition.png",
              frames: 41,
              note: "Diameter inside ancestor's subtree.",
            },
            {
              title: "Combined Rollout",
              sub: "VideoGym-LTX-2.3 · 41f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/composition_FAIL.mp4",
              poster: "media/showcase/05_beyond_familiar_patterns_and_n20_audit/01_comp_3rule_ancestor_subtree_diameter/input_composition.png",
              frames: 41,
              note: "Leaks above ancestor; rings 1 node.",
            },
          ],
        },
        {
          id: "comp_joint",
          label: "Joint Single-Step Comp.",
          caseId: "01_paper_figures__14_fig6b_polygon_color_composition",
          finding:
            "Joint Single-Step Composition: solving Polygon Side Step (100%) and Color Cycle (100%) separately still fails when card 4 must advance sides and cycle color together.",
          panes: [
            {
              title: "Rule A (+1 Side)",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/constituent_1_PASS.mp4",
              poster: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/input_constituent_1.png",
              frames: 25,
              note: "Advances polygon side count.",
            },
            {
              title: "Rule B (Color)",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/constituent_2_PASS.mp4",
              poster: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/input_constituent_2.png",
              frames: 25,
              note: "Continues period-4 color cycle.",
            },
            {
              title: "Combined Reference",
              sub: "Ground Truth · 25f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/composition_reference.mp4",
              poster: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/input_composition.png",
              frames: 25,
              note: "Draws orange octagon (+1 & cycle).",
            },
            {
              title: "Combined Rollout",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/composition_FAIL.mp4",
              poster: "media/showcase/01_paper_figures/14_fig6b_polygon_color_composition/input_composition.png",
              frames: 25,
              note: "Draws octagon, drops color cycle.",
            },
          ],
        },
        {
          id: "mech_extrap",
          label: "Mechanism Extrap. Bias",
          caseId: "01_paper_figures__11_fig5b_size_sequence_mech_extrap",
          finding:
            "Mechanism Extrapolation (Size Sequence): reversing a trained positive height step (100% In-Domain) to an unseen negative step (8, 6, 4 → 2) causes reversion to upward growth (draws height 6; 0% pass).",
          panes: [
            {
              title: "Growing Source",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/in_domain_source_PASS.mp4",
              poster: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/input_in_domain.png",
              frames: 25,
              note: "Continues upward height step cleanly.",
            },
            {
              title: "Shrinking Ref.",
              sub: "Ground Truth · 25f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/extrapolation_reference.mp4",
              poster: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/input_extrapolation.png",
              frames: 25,
              note: "Steps downward (8, 6, 4 → 2).",
            },
            {
              title: "Model Rollout",
              sub: "VideoGym-LTX-2.3 · 25f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/extrapolation_rollout.mp4",
              poster: "media/showcase/01_paper_figures/11_fig5b_size_sequence_mech_extrap/input_extrapolation.png",
              frames: 25,
              note: "Replays training bias; draws height 6.",
            },
          ],
        },
        {
          id: "yinyang_vision",
          label: "Single-Step Icon Binding",
          caseId: "04_within_familiar_patterns__01_paraphrase_singlestep_yinyang_symbol_fail",
          finding:
            "Single-Step Vision Paraphrase (Yin-Yang): even in the Single-Step Group, tasks bound to trained stone icons hold under Text Paraphrase (100%) yet drop to 0% when redrawn as tiles.",
          panes: [
            {
              title: "In-Domain Stones",
              sub: "VideoGym-LTX-2.3 · 45f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/in_domain_PASS.mp4",
              poster: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/input_in_domain.png",
              frames: 45,
              note: "Connects stones without 2×2 blocks.",
            },
            {
              title: "Text Paraphrase",
              sub: "VideoGym-LTX-2.3 · 45f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/text_paraphrase_PASS.mp4",
              poster: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/input_in_domain.png",
              frames: 45,
              note: "Preserves valid Yin-Yang solution.",
            },
            {
              title: "Redrawn Tiles Ref.",
              sub: "Ground Truth · 45f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/vision_paraphrase_reference.mp4",
              poster: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/input_vision_paraphrase.png",
              frames: 45,
              note: "Identical board logic with tile icons.",
            },
            {
              title: "Vision Paraphrase",
              sub: "VideoGym-LTX-2.3 · 45f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/vision_paraphrase_rollout.mp4",
              poster: "media/showcase/04_within_familiar_patterns/01_paraphrase_singlestep_yinyang_symbol_fail/input_vision_paraphrase.png",
              frames: 45,
              note: "Over-fills forbidden 2×2 dark blocks.",
            },
          ],
        },
      ],
    },

    transfer: {
      eyebrow: "Synchronized Rollouts",
      title: "External SFT (VBVR-Pro A14B) & RLVR (5B) Under Full-Video Verification",
      tabs: [
        {
          id: "vbvr_comp",
          label: "OOD Composition (0.00%)",
          caseId: "06_external_sft_rlvr_and_cross_benchmark__02_vbvr_composition_collapse_symbol_edit",
          finding:
            "VBVR-Pro trains symbol shifting and symbol insertion separately across 1.25M videos, yet combining both (shift right to make space, then insert hearts) scores 0.00% across all Composition tasks for both SFT and RLVR.",
          panes: [
            {
              title: "VBVR-Pro Ref.",
              sub: "Ground Truth · 54f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/reference.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/input.png",
              frames: 54,
              note: "Shifts suffix right, then inserts hearts.",
            },
            {
              title: "VBVR-Pro A14B",
              sub: "14B SFT · 54f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/vbvr_a14b.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/input.png",
              frames: 54,
              note: "Skips shift step; appends 1 heart.",
            },
            {
              title: "VBVR-Pro 5B RLVR",
              sub: "SFT + RLVR · 54f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/vbvr_5b_rlvr.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/02_vbvr_composition_collapse_symbol_edit/input.png",
              frames: 54,
              note: "Same single-rule failure despite RLVR.",
            },
          ],
        },
        {
          id: "vbvr_crosstask",
          label: "OOD Cross-Task (0.00%)",
          caseId: "06_external_sft_rlvr_and_cross_benchmark__03_vbvr_crosstask_collapse_gravity_physics",
          finding:
            "On unseen-rule Cross-Task problems in VBVR-Pro Bench (damped gravity bouncing with velocity arrows), both VBVR-Pro A14B (SFT) and VBVR-Pro 5B RLVR score 0.00% overall.",
          panes: [
            {
              title: "VBVR-Pro Ref.",
              sub: "Ground Truth · 192f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/reference.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/input.png",
              frames: 192,
              note: "Damped bounces with velocity arrows.",
            },
            {
              title: "VBVR-Pro A14B",
              sub: "14B SFT · 192f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/vbvr_a14b.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/input.png",
              frames: 192,
              note: "Hits floor without bouncing; garbles text.",
            },
            {
              title: "VBVR-Pro 5B RLVR",
              sub: "SFT + RLVR · 192f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/vbvr_5b_rlvr.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/03_vbvr_crosstask_collapse_gravity_physics/input.png",
              frames: 192,
              note: "Hits ground and sinks through floor; no bounce.",
            },
          ],
        },
        {
          id: "vbvr_id",
          label: "Cross-Bench: VBVR-Pro Task",
          caseId: "06_external_sft_rlvr_and_cross_benchmark__01_vbvr_in_domain_shortest_path",
          finding:
            "On VBVR-Pro's own trained Grid Shortest Path, both VBVR-Pro A14B (SFT) and 5B RLVR pass cleanly, whereas Cross-Benchmark VideoGym-LTX-2.3 (4.60% overall) replays its red-trail maze convention.",
          panes: [
            {
              title: "VBVR-Pro Ref.",
              sub: "Ground Truth · 60f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/reference.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/input.png",
              frames: 60,
              note: "Moves green dot along shortest route.",
            },
            {
              title: "VBVR-Pro A14B",
              sub: "Own In-Domain · 60f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/vbvr_a14b.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/input.png",
              frames: 60,
              note: "Solves trained grid convention.",
            },
            {
              title: "VBVR-Pro 5B RLVR",
              sub: "Own In-Domain · 60f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/vbvr_5b_rlvr.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/input.png",
              frames: 60,
              note: "Solves trained grid convention.",
            },
            {
              title: "VideoGym-LTX-2.3",
              sub: "Cross-Benchmark · 60f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/videogym_ltx23_crossbench.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/01_vbvr_in_domain_shortest_path/input.png",
              frames: 60,
              note: "Draws red trail along a long detour.",
            },
          ],
        },
        {
          id: "vg_crossbench",
          label: "Cross-Bench: VideoGym Task",
          caseId: "06_external_sft_rlvr_and_cross_benchmark__04_videogym_in_domain_shortest_path",
          finding:
            "On VideoGym-Test's Graph Shortest Path, post-trained VideoGym-LTX-2.3 passes cleanly (100% task accuracy), whereas Cross-Benchmark VBVR-Pro A14B (1.77% overall) cuts across vertex interiors and 5B RLVR (0.99%) colors only one edge.",
          panes: [
            {
              title: "VideoGym Ref.",
              sub: "Ground Truth · 41f",
              verdict: "ref",
              badge: "Reference",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/reference.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/input.png",
              frames: 41,
              note: "Traces shortest route 5 → 7 → 1 → 9.",
            },
            {
              title: "VideoGym-LTX-2.3",
              sub: "Own In-Domain · 41f",
              verdict: "pass",
              badge: "Pass",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/videogym_ltx23_PASS.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/input.png",
              frames: 41,
              note: "Exact route; preserves vertex labels.",
            },
            {
              title: "VBVR-Pro A14B",
              sub: "Cross-Benchmark · 41f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/vbvr_a14b_crossbench_FAIL.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/input.png",
              frames: 41,
              note: "Cuts strokes across node labels 7, 1 & 9.",
            },
            {
              title: "VBVR-Pro 5B RLVR",
              sub: "Cross-Benchmark · 41f",
              verdict: "fail",
              badge: "Fail",
              src: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/vbvr_5b_rlvr_crossbench_FAIL.mp4",
              poster: "media/showcase/06_external_sft_rlvr_and_cross_benchmark/04_videogym_in_domain_shortest_path/input.png",
              frames: 41,
              note: "Colors only the final edge 1 → 9; skips 5 → 7 → 1.",
            },
          ],
        },
      ],
    },
  };

  const stageControllers = [];

  function pauseOverviewMedia() {
    stageControllers.forEach((ctrl) => ctrl.suspend());
    document.querySelectorAll("#family-mosaic video").forEach((v) => {
      try {
        v.pause();
      } catch (_) {}
    });
  }

  /* ==================== VIEW SWITCHING ==================== */
  function switchView(mode, opts = {}) {
    const overviewEl = $("overview-view");
    const explorerEl = $("explorer-view");
    const navEl = $("overview-nav");
    const openBtn = $("open-explorer-btn");
    const backBtn = $("back-to-overview");

    if (mode === "explorer") {
      pauseOverviewMedia();
      overviewEl.hidden = true;
      navEl.hidden = true;
      explorerEl.hidden = false;
      backBtn.hidden = false;
      openBtn.setAttribute("aria-pressed", "true");
      window.scrollTo({ top: 0, behavior: "instant" });
      window.VideoGymExplorer.ensureLoaded(opts, Boolean(opts.pushUrl));
    } else {
      window.VideoGymExplorer.closeAll();
      explorerEl.hidden = true;
      backBtn.hidden = true;
      overviewEl.hidden = false;
      navEl.hidden = false;
      openBtn.setAttribute("aria-pressed", "false");

      const url = new URL(location.href);
      for (const k of [
        "view",
        "tab",
        "track",
        "family",
        "taskGroup",
        "q",
        "package",
        "taskFilter",
        "model",
        "protocol",
        "group",
        "task",
      ]) {
        url.searchParams.delete(k);
      }
      if (opts.pushUrl) history.pushState(null, "", url);
      stageControllers.forEach((ctrl) => ctrl.resumeIfVisible());
    }
  }

  /* ==================== BUILD 11-FAMILY MOSAIC (4x3 COMPLETE GRID) ==================== */
  function initFamilyMosaic() {
    const root = $("family-mosaic");
    if (!root) return;
    const tilesHtml = FAMILY_TILES.map(
      (f) => `
      <div role="button" tabindex="0" class="family-tile" data-jump-family="${esc(f.id)}" data-vid-src="${esc(f.vid)}" title="Browse ${esc(f.name)} (${f.tasks} In-Domain tasks)">
        <div class="ft-thumb">
          <img src="${esc(f.img)}" alt="${esc(f.name)} sample: ${esc(f.sample)}" loading="lazy" />
          <video muted loop playsinline preload="none" poster="${esc(f.img)}"></video>
          <span class="ft-group-dot ${f.dominant === "ss" ? "is-ss" : "is-ms"}">${esc(f.split)}</span>
          <span class="ft-play-hint">&#9654; Preview</span>
        </div>
        <div class="ft-body">
          <div class="ft-title-row">
            <strong>${esc(f.name)}</strong>
            <span>${f.tasks} tasks</span>
          </div>
          <div class="ft-rate-row" title="Post-trained VideoGym-LTX-2.3 In-Domain pass rate across ${f.tasks} tasks">
            <div class="ft-bar"><i style="width: ${f.rate}%" class="${f.dominant === "ss" ? "fill-blue" : "fill-green"}"></i></div>
            <span>${f.rate.toFixed(1)}% ID Pass</span>
          </div>
        </div>
      </div>`,
    ).join("");

    const exploreAllTile = `
      <button type="button" class="family-tile is-cta-tile" data-open-all-id>
        <div class="ft-cta-inner">
          <span class="ft-cta-eyebrow">Full Task Catalog</span>
          <strong>Explore All 744 Evaluation Tasks &rarr;</strong>
          <p>246 In-Domain tasks + 498 derived &amp; external evaluation entries with prompts and reference videos.</p>
        </div>
      </button>`;

    root.innerHTML = tilesHtml + exploreAllTile;

    root.querySelectorAll(".family-tile[data-jump-family]").forEach((tile) => {
      const video = tile.querySelector("video");
      const startPreview = () => {
        if (!video) return;
        if (!video.getAttribute("src")) {
          video.preload = "auto";
          video.src = tile.dataset.vidSrc;
          video.load();
        }
        tile.classList.add("is-playing");
        video.muted = true;
        video.currentTime = 0;
        video.play().catch(() => {});
      };
      const stopPreview = () => {
        if (!video) return;
        tile.classList.remove("is-playing");
        video.pause();
      };
      const openFamily = () => {
        stopPreview();
        switchView("explorer", {
          resetFilters: true,
          family: tile.dataset.jumpFamily,
          track: "all",
          pushUrl: true,
        });
      };
      tile.addEventListener("mouseenter", startPreview);
      tile.addEventListener("pointerenter", startPreview);
      tile.addEventListener("mouseleave", stopPreview);
      tile.addEventListener("pointerleave", stopPreview);
      tile.addEventListener("focusin", startPreview);
      tile.addEventListener("focusout", stopPreview);
      tile.addEventListener("click", openFamily);
      tile.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openFamily();
        }
      });
    });

    root.querySelector("[data-open-all-id]")?.addEventListener("click", () => {
      switchView("explorer", { resetFilters: true, track: "all", pushUrl: true });
    });
  }

  /* ==================== BUILD LIVE SYNCHRONIZED STAGES ==================== */
  function initLiveStages() {
    document.querySelectorAll("[data-live-stage]").forEach((mount) => {
      const key = mount.dataset.liveStage;
      const cfg = LIVE_STAGES[key];
      if (!cfg) return;

      let activeIdx = 0;
      let rate = 1;
      let userPaused = prefersReducedMotion;
      let isVisible = false;
      let holdTimer = null;
      let currentFrame = 0;

      mount.innerHTML = `
        <div class="ls-header">
          <div class="ls-heading">
            <span class="ls-eyebrow">${esc(cfg.eyebrow)}</span>
            <h3 class="ls-title">${esc(cfg.title)}</h3>
          </div>
          <div class="ls-tabs" role="tablist" aria-label="${esc(cfg.title)} scenarios">
            ${cfg.tabs
              .map(
                (t, idx) =>
                  `<button type="button" role="tab" class="ls-tab ${idx === 0 ? "is-active" : ""}" aria-selected="${idx === 0}" data-tab-idx="${idx}">${esc(t.label)}</button>`,
              )
              .join("")}
          </div>
        </div>
        <div class="ls-finding-bar">
          <p class="ls-finding" data-ls-finding></p>
          <button type="button" class="ls-case-btn" data-ls-jump>Inspect Full Case &rarr;</button>
        </div>
        <div class="ls-grid" data-ls-grid></div>
        <div class="ls-controls">
          <div class="ls-btns">
            <button type="button" class="ls-ctrl-btn" data-ls-act="toggle">${userPaused ? "&#9654; Play" : "&#10074;&#10074; Pause"}</button>
            <button type="button" class="ls-ctrl-btn" data-ls-act="restart" title="Restart synchronized playback">&#8634; Restart</button>
            <button type="button" class="ls-ctrl-btn" data-ls-act="prev" title="Step back 1 frame">&minus;1f</button>
            <button type="button" class="ls-ctrl-btn" data-ls-act="next" title="Step forward 1 frame">+1f</button>
            <select class="ls-speed" data-ls-act="speed" aria-label="Playback speed">
              <option value="0.5">0.5&times;</option>
              <option value="1" selected>1&times;</option>
            </select>
          </div>
          <input type="range" class="ls-scrubber" data-ls-act="scrub" min="0" max="24" step="1" value="0" aria-label="Synchronized frame scrubber" />
          <span class="ls-readout" data-ls-readout>Frame 1 / 25</span>
        </div>
      `;

      const findingEl = mount.querySelector("[data-ls-finding]");
      const jumpBtn = mount.querySelector("[data-ls-jump]");
      const gridEl = mount.querySelector("[data-ls-grid]");
      const toggleBtn = mount.querySelector('[data-ls-act="toggle"]');
      const scrubber = mount.querySelector('[data-ls-act="scrub"]');
      const readout = mount.querySelector("[data-ls-readout]");
      const speedSel = mount.querySelector('[data-ls-act="speed"]');

      const clearHold = () => {
        if (holdTimer) {
          clearTimeout(holdTimer);
          holdTimer = null;
        }
      };

      const getVideos = () => [...gridEl.querySelectorAll("video")];
      const getTab = () => cfg.tabs[activeIdx];
      const getMaxFrames = () =>
        Math.max(1, ...getTab().panes.map((p) => p.frames || 25));

      const updateToggleLabel = () => {
        const playing = !userPaused && getVideos().some((v) => !v.paused && !v.ended);
        toggleBtn.innerHTML =
          !userPaused && (playing || holdTimer)
            ? "&#10074;&#10074; Pause"
            : "&#9654; Play";
      };

      const updateReadout = (frameIdx) => {
        const maxIdx = Math.max(0, getMaxFrames() - 1);
        const clamped = Math.min(maxIdx, Math.max(0, Math.round(frameIdx)));
        currentFrame = clamped;
        scrubber.max = String(maxIdx);
        scrubber.value = String(clamped);
        readout.textContent = `Frame ${clamped + 1} / ${maxIdx + 1}`;
      };

      const ensureSourcesLoaded = () => {
        getVideos().forEach((v) => {
          if (!v.getAttribute("src") && v.dataset.src) {
            v.preload = "auto";
            v.src = v.dataset.src;
            v.playbackRate = rate;
          }
        });
      };

      const playAllFrom = (startSec = null) => {
        clearHold();
        ensureSourcesLoaded();
        getVideos().forEach((v) => {
          v.muted = true;
          v.playbackRate = rate;
          if (startSec !== null) {
            try {
              v.currentTime = startSec;
            } catch (_) {}
          }
          v.play().catch(() => {});
        });
        updateToggleLabel();
      };

      const pauseAll = () => {
        clearHold();
        getVideos().forEach((v) => {
          try {
            v.pause();
          } catch (_) {}
        });
        updateToggleLabel();
      };

      const seekFrame = (targetIdx) => {
        userPaused = true;
        pauseAll();
        ensureSourcesLoaded();
        const maxIdx = Math.max(0, getMaxFrames() - 1);
        const snapped = Math.min(maxIdx, Math.max(0, Math.round(targetIdx)));
        getVideos().forEach((v) => {
          const paneMax = Math.max(0, (Number(v.dataset.frames) || 25) - 1);
          const t = Math.min(snapped, paneMax) / 16 + 0.002;
          if (v.readyState >= 1) {
            try {
              v.currentTime = t;
            } catch (_) {}
          } else {
            v.addEventListener(
              "loadedmetadata",
              () => {
                try {
                  v.currentTime = t;
                } catch (_) {}
              },
              { once: true },
            );
          }
        });
        updateReadout(snapped);
        updateToggleLabel();
      };

      const checkAllEnded = () => {
        if (userPaused || !isVisible || holdTimer) return;
        const vids = getVideos();
        if (!vids.length) return;
        if (vids.every((v) => v.ended)) {
          updateReadout(getMaxFrames() - 1);
          holdTimer = setTimeout(() => {
            holdTimer = null;
            if (!userPaused && isVisible) {
              updateReadout(0);
              playAllFrom(0);
            }
          }, 900);
        }
      };

      const renderActiveTab = () => {
        clearHold();
        getVideos().forEach((v) => {
          try {
            v.pause();
            v.removeAttribute("src");
            v.load();
          } catch (_) {}
        });

        const tab = getTab();
        findingEl.textContent = tab.finding;
        jumpBtn.dataset.targetCase = tab.caseId;
        gridEl.className = `ls-grid cols-${tab.panes.length}`;
        gridEl.innerHTML = tab.panes
          .map(
            (p) => `
          <div class="ls-pane is-${esc(p.verdict)}">
            <div class="ls-pane-head">
              <div class="ls-pane-titles">
                <strong>${esc(p.title)}</strong>
                <span>${esc(p.sub)}</span>
              </div>
              <span class="ls-badge badge-${esc(p.verdict)}">${esc(p.badge)}</span>
            </div>
            <div class="ls-video-frame">
              <video data-src="${esc(p.src)}" data-frames="${esc(p.frames)}" poster="${esc(p.poster)}" muted playsinline preload="none"></video>
            </div>
            <p class="ls-pane-note">${esc(p.note)}</p>
          </div>`,
          )
          .join("");

        updateReadout(0);

        const vids = getVideos();
        vids.forEach((v) => {
          v.addEventListener("timeupdate", () => {
            if (!v.paused) {
              const maxCur = Math.max(...vids.map((x) => x.currentTime || 0));
              updateReadout(Math.floor(maxCur * 16 + 1e-3));
            }
          });
          v.addEventListener("ended", checkAllEnded);
        });

        if (isVisible && !userPaused) {
          playAllFrom(0);
        } else if (isVisible) {
          ensureSourcesLoaded();
          updateToggleLabel();
        }
      };

      mount.querySelector(".ls-tabs").addEventListener("click", (e) => {
        const btn = e.target.closest("[data-tab-idx]");
        if (!btn) return;
        const nextIdx = Number(btn.dataset.tabIdx);
        if (nextIdx === activeIdx) return;
        activeIdx = nextIdx;
        mount.querySelectorAll(".ls-tab").forEach((b, idx) => {
          b.classList.toggle("is-active", idx === activeIdx);
          b.setAttribute("aria-selected", String(idx === activeIdx));
        });
        userPaused = false;
        renderActiveTab();
      });

      jumpBtn.addEventListener("click", () => {
        const caseId = jumpBtn.dataset.targetCase;
        if (caseId) {
          switchView("explorer", {
            tab: "rollouts",
            showcaseCaseId: caseId,
            pushUrl: true,
          });
        }
      });

      toggleBtn.addEventListener("click", () => {
        if (userPaused) {
          userPaused = false;
          const atEnd = getVideos().every((v) => v.ended);
          playAllFrom(atEnd ? 0 : null);
        } else {
          userPaused = true;
          pauseAll();
        }
      });

      mount.querySelector('[data-ls-act="restart"]').addEventListener("click", () => {
        userPaused = false;
        updateReadout(0);
        playAllFrom(0);
      });

      mount.querySelector('[data-ls-act="prev"]').addEventListener("click", () => {
        seekFrame(currentFrame - 1);
      });

      mount.querySelector('[data-ls-act="next"]').addEventListener("click", () => {
        seekFrame(currentFrame + 1);
      });

      speedSel.addEventListener("change", () => {
        rate = Number(speedSel.value) || 1;
        getVideos().forEach((v) => {
          v.playbackRate = rate;
        });
      });

      scrubber.addEventListener("input", () => {
        seekFrame(Number(scrubber.value) || 0);
      });

      renderActiveTab();

      if ("IntersectionObserver" in window) {
        const io = new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              isVisible = entry.isIntersecting;
              if (isVisible) {
                ensureSourcesLoaded();
                if (!userPaused) playAllFrom(null);
              } else {
                pauseAll();
              }
            }
          },
          { rootMargin: "120px 0px", threshold: 0.15 },
        );
        io.observe(mount);
      }

      stageControllers.push({
        suspend: () => pauseAll(),
        resumeIfVisible: () => {
          if (isVisible && !userPaused) playAllFrom(null);
        },
      });
    });
  }

  /* ==================== PAPER FIGURE ZOOM LIGHTBOX ==================== */
  function initFigureZoom() {
    const dlg = $("fig-zoom-dialog");
    const zoomImg = $("fig-zoom-img");
    const zoomCap = $("fig-zoom-caption");
    const zoomTitle = $("fig-zoom-title");
    const closeBtn = $("fig-zoom-close");
    if (!dlg || !zoomImg) return;

    document.querySelectorAll("[data-zoomable-fig]").forEach((fig) => {
      const imgWrap = fig.querySelector(".fig-img-wrap");
      const img = fig.querySelector("img");
      const cap = fig.querySelector("figcaption");
      if (!imgWrap || !img) return;
      imgWrap.tabIndex = 0;
      imgWrap.setAttribute("role", "button");
      imgWrap.setAttribute("aria-label", `Enlarge ${img.alt || "figure"}`);
      const openZoom = () => {
        zoomImg.src = img.src;
        zoomImg.alt = img.alt || "";
        const leadStrong = cap?.querySelector("strong")?.textContent || "Paper Figure";
        zoomTitle.textContent = leadStrong.replace(/\.$/, "");
        zoomCap.innerHTML = cap ? cap.innerHTML : "";
        dlg.showModal();
      };
      imgWrap.addEventListener("click", openZoom);
      imgWrap.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openZoom();
        }
      });
    });

    closeBtn?.addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg) dlg.close();
    });
  }

  /* ==================== GLOBAL NAVIGATION & EVENTS ==================== */
  $("open-explorer-btn")?.addEventListener("click", () => {
    const isOpen = $("open-explorer-btn").getAttribute("aria-pressed") === "true";
    if (isOpen) switchView("overview", { pushUrl: true });
    else switchView("explorer", { pushUrl: true });
  });

  $("back-to-overview")?.addEventListener("click", () => {
    switchView("overview", { pushUrl: true });
  });

  document.querySelector("[data-nav-overview]")?.addEventListener("click", (e) => {
    if (!$("explorer-view").hidden) {
      e.preventDefault();
      switchView("overview", { pushUrl: true });
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });

  document.querySelectorAll("[data-open-explorer]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchView("explorer", { resetFilters: true, track: "all", pushUrl: true });
    });
  });

  document.querySelectorAll("[data-open-rollouts]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchView("explorer", { tab: "rollouts", pushUrl: true });
    });
  });

  document.querySelectorAll("[data-jump-track]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchView("explorer", {
        resetFilters: true,
        track: btn.dataset.jumpTrack,
        pushUrl: true,
      });
    });
  });

  $("copy-bibtex-btn")?.addEventListener("click", async () => {
    const text = $("bibtex-code")?.textContent || "";
    const btn = $("copy-bibtex-btn");
    try {
      await navigator.clipboard.writeText(text);
      const prev = btn.textContent;
      btn.textContent = "Copied ✓";
      setTimeout(() => {
        btn.textContent = prev;
      }, 2000);
    } catch (_) {
      btn.textContent = "Select & Copy Below";
    }
  });

  if ("IntersectionObserver" in window) {
    const navLinks = [...document.querySelectorAll("#overview-nav a")];
    const sections = navLinks
      .map((a) => document.querySelector(a.getAttribute("href")))
      .filter(Boolean);
    const spy = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const id = "#" + entry.target.id;
            navLinks.forEach((a) =>
              a.classList.toggle("is-active", a.getAttribute("href") === id),
            );
          }
        }
      },
      { rootMargin: "-20% 0px -65% 0px" },
    );
    sections.forEach((sec) => spy.observe(sec));
  }

  initFamilyMosaic();
  initLiveStages();
  initFigureZoom();

  function restoreFromUrl() {
    const q = new URL(location.href).searchParams;
    if (q.get("view") === "explorer" || q.get("task") || q.get("track") || q.get("tab") || q.get("family")) {
      switchView("explorer", { fromUrl: true });
    } else {
      switchView("overview", { pushUrl: false });
    }
  }

  window.addEventListener("popstate", restoreFromUrl);
  window.VideoGymApp = { switchView };

  restoreFromUrl();
})();
