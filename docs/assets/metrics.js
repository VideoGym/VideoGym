/* Shared display names and question-count helpers for the VideoGym Task Explorer. */
(function (root) {
  "use strict";
  const acronyms = {
    id: "ID",
    ood: "OOD",
    bfs: "BFS",
    dfs: "DFS",
    hmm: "HMM",
    xor: "XOR",
    bfs2: "BFS2",
    gt: "GT",
    hud: "HUD",
    vbvr: "VBVR",
    videogym: "VideoGym",
    "1d": "1D",
    arc1d: "ARC 1D",
    fifo: "FIFO",
    gui: "GUI",
    "2d": "2D",
    "3d": "3D",
    rlvr: "RLVR",
    nqueens: "N-Queens",
    lightsout: "Lights Out",
  };

  function catalog() {
    return root.VIDEOGYM_EXPLORER || null;
  }

  function displayName(value) {
    if (value === "vbvr_pro") return "VBVR-Pro";
    if (value === "In_Domain") return "In-Domain";
    if (value === "Out_of_Domain") return "Out-of-Domain";
    const family = catalog()?.families?.find((entry) => entry.name === value);
    if (family) return family.display_name;
    return String(value ?? "")
      .replace(/_/g, " ")
      .replace(
        /\b[a-zA-Z0-9]+\b/g,
        (word) =>
          acronyms[word.toLowerCase()] ||
          (word === word.toUpperCase() && word.length > 1
            ? word
            : word[0].toUpperCase() + word.slice(1)),
      );
  }

  function taskLabel(task) {
    if (task.track === "cross_benchmark") {
      return displayName(
        String(task.task_name ?? task.name).replace(/[_-]data-generator$/i, ""),
      );
    }
    if (task.track === "composition") {
      return displayName(String(task.name ?? "").replace(/_composition$/, ""));
    }
    if (
      [
        "id",
        "prompt_paraphrase_text",
        "prompt_paraphrase_vision",
        "extrapolation",
      ].includes(task.track)
    ) {
      return displayName(task.task_name);
    }
    return displayName(task.name);
  }

  const taskQuestions = (task) =>
    Number.isFinite(task?.question_count)
      ? task.question_count
      : (task?.samples || []).length;

  const questionCount = (tasks, samples) =>
    tasks && samples % tasks === 0
      ? `${tasks}×${samples / tasks} Questions`
      : `${samples.toLocaleString()} Questions`;

  const api = {
    questionCount,
    taskQuestions,
    displayName,
    taskLabel,
  };
  if (typeof module !== "undefined") module.exports = api;
  root.ExplorerMetrics = api;
})(globalThis);
