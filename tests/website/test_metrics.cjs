const { test } = require("node:test");
const assert = require("node:assert/strict");

globalThis.VIDEOGYM_EXPLORER = {
  families: [{ name: "in_context_learning", display_name: "In-Context Learning" }],
  task_groups: [
    { id: "single_step", label: "Single-Step Group" },
    { id: "multi_step", label: "Multi-Step Group" },
  ],
};

const {
  displayName,
  taskLabel,
  questionCount,
  taskQuestions,
} = require("../../docs/assets/metrics.js");

test("Display names retain acronyms and replace identifier separators", () => {
  assert.equal(displayName("board_game"), "Board Game");
  assert.equal(displayName("grid_bfs"), "Grid BFS");
  assert.equal(displayName("spatial_3d"), "Spatial 3D");
  assert.equal(displayName("vbvr_pro"), "VBVR-Pro");
  assert.equal(displayName("In_Domain"), "In-Domain");
  assert.equal(displayName("Out_of_Domain"), "Out-of-Domain");
  assert.equal(displayName("in_context_learning"), "In-Context Learning");
});

test("Task labels and question counts format cleanly across generalization tracks", () => {
  assert.equal(
    taskLabel({
      track: "extrapolation",
      name: "ext_bookshelf_sort_by_height",
      task_name: "bookshelf_sort_by_height",
    }),
    "Bookshelf Sort By Height",
  );
  assert.equal(
    taskLabel({
      track: "composition",
      name: "bridge_conveyor_composition",
      task_name: "eval_bridge_conveyor",
    }),
    "Bridge Conveyor",
  );
  assert.equal(
    taskLabel({
      track: "cross_task",
      name: "halfplane_intersection",
      task_name: "eval_cross_halfplane_intersection",
    }),
    "Halfplane Intersection",
  );
  assert.equal(
    taskLabel({
      track: "cross_benchmark",
      name: "vbvr_O-75_communicating_vessels_data-generator",
      task_name: "O-75_communicating_vessels_data-generator",
    }),
    "O-75 Communicating Vessels",
  );
  assert.equal(questionCount(246, 1230), "246×5 Questions");
  assert.equal(questionCount(744, 3720), "744×5 Questions");
  assert.equal(taskQuestions({ question_count: 5, samples: [{}] }), 5);
});
