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

  const NOTES = {
    all: "Browse all 744 level-by-task benchmark entries across seven generalization levels (246 In-Domain + 246 Text Paraphrase + 33 Vision Paraphrase + 52 Extrapolation + 34 Composition + 33 Cross-Task + 100 Cross-Benchmark; 5 questions each = 3,720 questions). Click any card to inspect its initial image, rule prompt, reference video, and paired source task(s).",
    id: "Unseen problem instances within trained task rules and parameter ranges (246 tasks, 1,230 questions).",
    prompt_paraphrase_text:
      "Synonymous instruction rewrites on the exact same 1,230 In-Domain images, rules, answers, and reference lengths.",
    prompt_paraphrase_vision:
      "Redrawn visual symbols with matching legend updates on the exact same problem structure, answer, and reference length (33 tasks, 165 questions).",
    extrapolation:
      "Scale (29 tasks) and Mechanism (23 tasks) extrapolation testing larger instances or unseen boundary conditions of a trained rule.",
    composition:
      "Serial chaining or joint combination of 2–3 rules taught only in separate In-Domain tasks (34 tasks, 170 questions).",
    cross_task:
      "Held-out tasks introducing new rules and solving objectives within the same 11 problem categories (33 tasks, 165 questions).",
    cross_benchmark:
      "VBVR-Pro Bench tasks under deterministic full-video verification (100 tasks, 500 questions). Generalization labels reflect VBVR-Pro training exposure; all tasks are Cross-Benchmark for VideoGym models.",
  };

  let catalogPromise = null;
  let showcasePromise = null;
  let catalogReady = false;
  let rolloutsReady = false;
  let explorerMode = "tasks";

  let state = {
    track: "all",
    family: "",
    taskGroup: "",
    query: "",
    packageKey: "",
    taskKey: "",
    activeTask: null,
  };

  function loadScriptOnce(src, globalName, cacheHolder) {
    if (window[globalName]) return Promise.resolve(window[globalName]);
    if (cacheHolder.p) return cacheHolder.p;
    cacheHolder.p = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => {
        if (window[globalName]) resolve(window[globalName]);
        else reject(new Error(`${globalName} missing after load`));
      };
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
    return cacheHolder.p;
  }

  function loadCatalog() {
    if (!catalogPromise) catalogPromise = { p: null };
    return loadScriptOnce("data/explorer-data.js", "VIDEOGYM_EXPLORER", catalogPromise);
  }

  function loadShowcase() {
    if (!showcasePromise) showcasePromise = { p: null };
    return loadScriptOnce("data/showcase-data.js", "VIDEOGYM_SHOWCASE", showcasePromise);
  }

  function syncUrl(push = false) {
    if ($("explorer-view")?.hidden) return;
    const url = new URL(location.href);
    url.searchParams.set("view", "explorer");
    if (explorerMode === "rollouts") {
      url.searchParams.set("tab", "rollouts");
    } else {
      url.searchParams.delete("tab");
    }
    const pairs = {
      track: state.track === "all" ? "" : state.track,
      family: state.family,
      taskGroup: state.taskGroup,
      q: state.query,
      package: state.packageKey,
      taskFilter: state.taskKey,
      task: state.activeTask || "",
    };
    for (const [k, val] of Object.entries(pairs)) {
      if (val && explorerMode === "tasks") url.searchParams.set(k, val);
      else url.searchParams.delete(k);
    }
    history[push ? "pushState" : "replaceState"](null, "", url);
  }

  function showcaseMatchesForTask(task) {
    const cases = window.VIDEOGYM_SHOWCASE?.cases || [];
    return cases.filter((c) => c.task_name === task.task_name);
  }

  /* ==================== SUB-PANEL B: ROLLOUT CASES ==================== */
  let activeChapter = "featured";
  let chapterBeforeSearch = null;
  let scGroup = "";
  let scQuery = "";
  let cardObserver = null;

  const CHECK_LABELS = {
    all_minority_objects_removed: [1, "minority objects unremoved"],
    zero_wall_crossings: [1, "crossed obstacle"],
    one_constant_ball_on_fixed_vertical_axis: [1, "ball lost or off vertical axis"],
    exact_object_count: [1, "wrong final symbol count"],
    declared_sequence_inventory: [1, "target sequence mismatch"],
    original_labels_have_distinct_observed_support: [2, "unshifted/missing symbols"],
    shortest_legal_route: [2, "non-shortest/invalid route"],
    scene_preserved_throughout: [2, "unrequested scene edit"],
    no_other_objects_or_marks: [2, "non-majority object remaining"],
    removed_regions_match_retained_scene: [3, "removal region uncleared"],
    slot_frames_labels_and_reference_panel_preserved: [3, "slot/label drift"],
    candidate_panel_preserved_throughout: [3, "reference panel altered"],
    adjacent_orthogonal_steps: [3, "non-adjacent step jump"],
    one_readable_agent_every_frame: [4, "agent flicker/loss"],
  };

  function resolveCheckSpec(key) {
    if (CHECK_LABELS[key]) return CHECK_LABELS[key];
    if (/^frame_\d+\.no_isolated_added_content$/.test(key)) {
      return [2, "spurious added symbols"];
    }
    return [5, key.replace(/_/g, " ")];
  }

  function formatFailedChecks(rawChecks) {
    const ordered = [...rawChecks].sort(
      (a, b) => resolveCheckSpec(a)[0] - resolveCheckSpec(b)[0],
    );
    const labels = [];
    for (const key of ordered) {
      const lbl = resolveCheckSpec(key)[1];
      if (!labels.includes(lbl)) labels.push(lbl);
    }
    return labels;
  }

  const SHORT_CHAPTER_NAMES = {
    "01_paper_figures": "01 · Paper Figures",
    "02_task_families_and_base_models": "02 · Families & Base",
    "03_epoch_and_duration_dynamics": "03 · Epochs & 81f",
    "04_within_familiar_patterns": "04 · Paraphrase & Extrap.",
    "05_beyond_familiar_patterns_and_n20_audit": "05 · Comp. & Cross-Task",
    "06_external_sft_rlvr_and_cross_benchmark": "06 · External SFT & RLVR",
  };

  function cleanupRolloutVideos() {
    if (cardObserver) {
      cardObserver.disconnect();
      cardObserver = null;
    }
    $("showcase-list")
      ?.querySelectorAll("video")
      .forEach((v) => {
        try {
          v.pause();
          v.removeAttribute("src");
          v.load();
        } catch (_) {}
      });
  }

  function initRolloutsPanel(showcase) {
    function matchesCase(c, ignoreChapter = false) {
      if (!ignoreChapter) {
        if (activeChapter === "featured" && !c.featured) return false;
        if (
          activeChapter !== "featured" &&
          activeChapter !== "all" &&
          c.section_id !== activeChapter
        ) {
          return false;
        }
      }
      if (scGroup && c.group !== scGroup) return false;
      if (scQuery) {
        const hay = [
          c.id,
          c.title,
          c.paper_ref,
          c.level,
          c.task_name,
          c.family,
          c.group,
          c.finding,
          ...c.inputs.map((inp) => `${inp.label} ${inp.sample_id}`),
          ...c.videos.map((v) => `${v.label} ${v.model} ${v.protocol} ${v.verdict} ${v.note || ""}`),
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(scQuery)) return false;
      }
      return true;
    }

    function renderMetricsChips(v) {
      const m = v.metrics || {};
      const fCount = v.frames || m.frames;
      if (v.role === "reference") {
        return `<div class="metric-chips"><span class="mchip">${fCount ? `${esc(fCount)} frames` : "Reference"} &middot; 16 FPS</span></div>`;
      }
      const chips = [];
      if (Object.hasOwn(m, "process_legality")) {
        const shortcut = m.final_state_pass === 1 && m.process_legality === 0;
        chips.push(
          `<span class="mchip ${shortcut ? "mchip-warn" : ""}">Process: ${m.process_legality === 1 ? "Pass" : "Fail"}</span>`,
          `<span class="mchip ${shortcut ? "mchip-warn" : ""}">Final State: ${m.final_state_pass === 1 ? "Pass" : "Fail"}</span>`,
        );
      }
      if (Array.isArray(m.failed_checks) && m.failed_checks.length) {
        const readable = formatFailedChecks(m.failed_checks);
        const shown = readable.slice(0, 2).join(", ");
        const extra = readable.length > 2 ? ` (+${readable.length - 2} more)` : "";
        chips.push(
          `<span class="mchip" title="Failed verifier checks: ${esc(readable.join("; "))}">Failed: ${esc(shown)}${esc(extra)}</span>`,
        );
      }
      return chips.length ? `<div class="metric-chips">${chips.join("")}</div>` : "";
    }

    function ensureCardLoaded(card) {
      const rateVal = Number(card.querySelector('[data-act="speed"]')?.value) || 1;
      card.querySelectorAll("[data-sync-video]").forEach((v) => {
        if (!v.getAttribute("src") && v.dataset.src) {
          v.preload = "metadata";
          v.defaultPlaybackRate = rateVal;
          v.src = v.dataset.src;
          v.playbackRate = rateVal;
        }
      });
    }

    function bindShowcaseCard(card) {
      const videos = [...card.querySelectorAll("[data-sync-video]")];
      if (!videos.length) return;
      const slider = card.querySelector('[data-act="seek"]');
      const readout = card.querySelector("[data-readout]");
      const speedSel = card.querySelector('[data-act="speed"]');

      card.addEventListener("pointerenter", () => ensureCardLoaded(card), { once: true });
      card.addEventListener("focusin", () => ensureCardLoaded(card), { once: true });

      const getVidFrames = (v) => {
        const tagged = Number(v.dataset.frames);
        if (tagged > 0) return tagged;
        if (Number.isFinite(v.duration) && v.duration > 0) {
          return Math.max(1, Math.round(v.duration * 16));
        }
        return 1;
      };
      const getMaxIdx = () => Math.max(0, Math.max(...videos.map(getVidFrames)) - 1);
      let sharedFrame = 0;

      const updateReadout = (fIdx) => {
        const maxIdx = getMaxIdx();
        const clamped = Math.min(maxIdx, Math.max(0, Math.round(fIdx)));
        sharedFrame = clamped;
        readout.textContent = `Frame ${clamped + 1} / ${maxIdx + 1}`;
        slider.max = String(maxIdx);
        slider.value = String(clamped);
      };

      const seekFrames = (targetFrameIdx) => {
        ensureCardLoaded(card);
        const maxIdx = getMaxIdx();
        const snapped = Math.min(maxIdx, Math.max(0, Math.round(targetFrameIdx)));
        videos.forEach((v) => {
          v.pause();
          const vLast = Math.max(0, getVidFrames(v) - 1);
          const targetTime = Math.min(snapped, vLast) / 16 + 0.002;
          if (v.readyState >= 1) v.currentTime = targetTime;
        });
        updateReadout(snapped);
      };

      videos.forEach((v) => {
        v.addEventListener("loadedmetadata", () => {
          slider.max = String(getMaxIdx());
        });
        v.addEventListener("timeupdate", () => {
          if (!v.paused) {
            const cur = Math.max(...videos.map((x) => x.currentTime || 0));
            updateReadout(Math.floor(cur * 16 + 1e-3));
          }
        });
      });

      card.querySelector('[data-act="play"]').onclick = () => {
        ensureCardLoaded(card);
        const rateVal = Number(speedSel.value) || 1;
        videos.forEach((v) => {
          v.muted = true;
          v.playbackRate = rateVal;
          if (v.ended || sharedFrame >= getMaxIdx()) v.currentTime = 0;
          v.play().catch(() => {});
        });
      };
      card.querySelector('[data-act="pause"]').onclick = () =>
        videos.forEach((v) => v.pause());
      card.querySelector('[data-act="restart"]').onclick = () => {
        ensureCardLoaded(card);
        const rateVal = Number(speedSel.value) || 1;
        updateReadout(0);
        videos.forEach((v) => {
          v.muted = true;
          v.currentTime = 0;
          v.playbackRate = rateVal;
          v.play().catch(() => {});
        });
      };
      card.querySelector('[data-act="prev"]').onclick = () =>
        seekFrames(sharedFrame - 1);
      card.querySelector('[data-act="next"]').onclick = () =>
        seekFrames(sharedFrame + 1);
      speedSel.onchange = () => {
        const rateVal = Number(speedSel.value) || 1;
        videos.forEach((v) => {
          v.playbackRate = rateVal;
        });
      };
      slider.oninput = () => seekFrames(Number(slider.value) || 0);
    }

    function renderRollouts() {
      cleanupRolloutVideos();
      const featuredCnt = showcase.cases.filter(
        (c) => c.featured && matchesCase(c, true),
      ).length;
      const allCnt = showcase.cases.filter((c) => matchesCase(c, true)).length;
      $("showcase-tabs").innerHTML = [
        `<button type="button" class="chapter-tab ${activeChapter === "featured" ? "is-active" : ""}" data-chapter="featured">&#9733; Featured (${featuredCnt})</button>`,
        ...showcase.sections.map((s) => {
          const cnt = showcase.cases.filter(
            (c) => c.section_id === s.id && matchesCase(c, true),
          ).length;
          return `<button type="button" class="chapter-tab ${activeChapter === s.id ? "is-active" : ""}" data-chapter="${esc(s.id)}">${esc(SHORT_CHAPTER_NAMES[s.id] || s.title)} <span class="tab-cnt">(${cnt})</span></button>`;
        }),
        `<button type="button" class="chapter-tab ${activeChapter === "all" ? "is-active" : ""}" data-chapter="all">All (${allCnt})</button>`,
      ].join("");

      const filtered = showcase.cases.filter((c) => matchesCase(c, false));
      const totalCases = showcase.stats?.cases || showcase.cases.length;
      const totalVideos = showcase.stats?.videos || showcase.cases.reduce((n, c) => n + c.videos.length, 0);
      const noteEl = $("showcase-chapter-note");
      if (activeChapter === "featured") {
        noteEl.innerHTML = `<strong>&#9733; Featured Highlight Cases (${filtered.length})</strong><span>Synchronized reference and model rollouts across Figures 2–6. Switch tabs above to inspect all ${totalCases} cases.</span>`;
      } else if (activeChapter === "all") {
        noteEl.innerHTML = `<strong>All Curated Rollout Cases (${filtered.length})</strong><span>Complete multi-model comparison set (${totalVideos} MP4 videos).</span>`;
      } else {
        const sec = showcase.sections.find((s) => s.id === activeChapter);
        if (sec) {
          noteEl.innerHTML = `<strong>${esc(sec.number)}. ${esc(sec.title)} (${filtered.length})</strong><span>${esc(sec.summary)}</span>`;
        }
      }

      const listEl = $("showcase-list");
      if (!filtered.length) {
        listEl.innerHTML = '<div class="empty-box">No rollout cases match this filter.</div>';
        return;
      }

      listEl.innerHTML = filtered
        .map((c) => {
          const grpClass =
            c.group === "Single-Step Group"
              ? "badge-ss"
              : c.group === "Multi-Step Group"
                ? "badge-ms"
                : "";
          const defaultPoster = c.inputs[0]?.image || "";
          const knownFrames = c.videos
            .map((v) => Number(v.frames || v.metrics?.frames || 0))
            .filter((n) => n > 0);
          const initialMaxIdx = Math.max(
            1,
            (knownFrames.length ? Math.max(...knownFrames) : 81) - 1,
          );
          const videosHtml = c.videos
            .map((v) => {
              const stateCls =
                v.role === "reference" ? "" : v.passed ? "is-pass" : "is-fail";
              const pillCls =
                v.role === "reference"
                  ? "verdict-ref"
                  : v.passed
                    ? "verdict-pass"
                    : "verdict-fail";
              return `
                <div class="video-pane ${stateCls}">
                  <div class="pane-head">
                    <div>
                      <div class="pane-title">${esc(v.label)}</div>
                      <div class="pane-sub">${esc(v.model)} &middot; ${esc(v.protocol)}</div>
                    </div>
                    <span class="verdict-pill ${pillCls}">${esc(v.verdict === "PASS" ? "Pass" : v.verdict === "FAIL" ? "Fail" : v.verdict === "REF" ? "Reference" : v.verdict)}</span>
                  </div>
                  <video data-sync-video data-frames="${esc(Number(v.frames || v.metrics?.frames || 0))}" data-src="${esc(v.video)}" muted controls playsinline preload="none" poster="${esc(v.poster || defaultPoster)}"></video>
                  ${renderMetricsChips(v)}
                  ${v.note ? `<p class="pane-note">${esc(v.note)}</p>` : ""}
                </div>`;
            })
            .join("");
          const inputsHtml = c.inputs
            .map(
              (inp) => `
              <div class="input-card">
                <a href="${esc(inp.image)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(inp.image)}" alt="${esc(inp.label)}" /></a>
                <div>
                  <h4>${esc(inp.label)}</h4>
                  <div class="input-meta">Sample ID: ${esc(inp.sample_id)}</div>
                  <pre class="prompt-box">${esc(inp.prompt)}</pre>
                </div>
              </div>`,
            )
            .join("");
          return `
            <article class="case-card" data-case-id="${esc(c.id)}">
              <div class="case-head">
                <h3 class="case-title">${esc(c.title)}</h3>
                <div class="badges">
                  <span class="badge badge-ref">${esc(c.paper_ref)}</span>
                  <span class="badge">${esc(c.level)}</span>
                  <span class="badge">${esc(c.family)}</span>
                  ${c.group ? `<span class="badge ${grpClass}">${esc(c.group)}</span>` : ""}
                </div>
              </div>
              <p class="case-finding">${esc(c.finding)}</p>
              <div class="sync-bar">
                <button type="button" class="sync-btn" data-act="play">&#9654; Play All</button>
                <button type="button" class="sync-btn" data-act="pause">&#10074;&#10074; Pause</button>
                <button type="button" class="sync-btn" data-act="restart">&#8634; Restart</button>
                <button type="button" class="sync-btn" data-act="prev">&minus;1 Frame</button>
                <button type="button" class="sync-btn" data-act="next">+1 Frame</button>
                <label>Speed
                  <select data-act="speed">
                    <option value="0.25">0.25&times;</option>
                    <option value="0.5">0.5&times;</option>
                    <option value="1" selected>1&times;</option>
                    <option value="2">2&times;</option>
                  </select>
                </label>
                <input type="range" class="sync-slider" data-act="seek" min="0" max="${initialMaxIdx}" step="1" value="0" aria-label="Shared frame timeline" />
                <span class="time-readout" data-readout>Frame 1 / ${initialMaxIdx + 1}</span>
              </div>
              <div class="video-grid">${videosHtml}</div>
              <div class="case-bottom-bar">
                <details class="inputs-drawer">
                  <summary>Inspect Input Image(s) I&#8320; &amp; Exact Task Prompt P (${c.inputs.length})</summary>
                  <div class="inputs-grid">${inputsHtml}</div>
                </details>
                <button type="button" class="btn-subtle" data-inspect-task="${esc(c.task_name)}">
                  View Task in Catalog &rarr;
                </button>
              </div>
            </article>`;
        })
        .join("");

      const cards = [...listEl.querySelectorAll(".case-card")];
      cards.forEach(bindShowcaseCard);
      if ("IntersectionObserver" in window) {
        cardObserver = new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (entry.isIntersecting) {
                ensureCardLoaded(entry.target);
                cardObserver.unobserve(entry.target);
              }
            }
          },
          { rootMargin: "700px 0px" },
        );
        cards.forEach((card) => cardObserver.observe(card));
      }
    }

    if (!rolloutsReady) {
      rolloutsReady = true;
      $("showcase-tabs").addEventListener("click", (e) => {
        const btn = e.target.closest("[data-chapter]");
        if (!btn) return;
        activeChapter = btn.dataset.chapter;
        if (scQuery) chapterBeforeSearch = activeChapter;
        renderRollouts();
      });
      document.querySelectorAll("[data-sc-group]").forEach((btn) => {
        btn.addEventListener("click", () => {
          scGroup = btn.dataset.scGroup;
          document
            .querySelectorAll("[data-sc-group]")
            .forEach((b) => b.classList.toggle("active", b === btn));
          renderRollouts();
        });
      });
      $("showcase-search-input").addEventListener("input", (e) => {
        const next = e.target.value.trim().toLowerCase();
        if (next && !scQuery) {
          chapterBeforeSearch = activeChapter;
          activeChapter = "all";
        } else if (!next && scQuery && chapterBeforeSearch) {
          activeChapter = chapterBeforeSearch;
          chapterBeforeSearch = null;
        }
        scQuery = next;
        renderRollouts();
      });
      $("showcase-list").addEventListener("click", (e) => {
        const btn = e.target.closest("[data-inspect-task]");
        if (!btn) return;
        setExplorerMode("tasks", { taskName: btn.dataset.inspectTask });
      });
    }

    window.VideoGymExplorer._renderRollouts = (caseId) => {
      if (caseId) {
        const target = showcase.cases.find((c) => c.id === caseId);
        if (target) {
          activeChapter = target.section_id;
          scGroup = "";
          scQuery = "";
          $("showcase-search-input").value = "";
        }
      }
      renderRollouts();
      if (caseId) {
        const scrollToCase = () => {
          const el = $("showcase-list")?.querySelector(
            `.case-card[data-case-id="${CSS.escape(caseId)}"]`,
          );
          if (!el) return;
          $("showcase-list")
            .querySelectorAll(".case-card.is-targeted")
            .forEach((c) => c.classList.remove("is-targeted"));
          el.classList.add("is-targeted");
          const top = Math.max(0, el.getBoundingClientRect().top + window.scrollY - 68);
          window.scrollTo({ top, behavior: "instant" });
        };
        requestAnimationFrame(() => {
          scrollToCase();
          setTimeout(scrollToCase, 80);
        });
      }
    };
  }

  /* ==================== SUB-PANEL A: 744-TASK CATALOG ==================== */
  function initCatalogPanel(catalog) {
    if (catalogReady) return;
    catalogReady = true;

    const {
      displayName,
      taskLabel,
      questionCount,
      taskQuestions,
    } = window.ExplorerMetrics;

    const names = catalog.evaluation.tracks;
    const vbvrLevels = { ...names, prompt_paraphrase: "Vision Prompt Paraphrase" };
    const baseTasks = new Map(
      catalog.families.flatMap((f) => f.tasks).map((t) => [t.id, t]),
    );
    const tasks = catalog.evaluation.tasks;
    const byId = new Map(tasks.map((t) => [t.name, t]));

    const sourceIds = (task) =>
      task.track === "cross_task" ? [] : task.source_task_ids || [];

    const description = (task) =>
      task.track === "id"
        ? baseTasks.get(task.task_name)?.design_intent || task.description
        : task.description;

    const taskOptions = [...new Set(tasks.map((t) => t.task_name))]
      .map((id) => [
        id,
        taskLabel(
          tasks.find((t) => t.task_name === id && t.track === "id") ||
            tasks.find((t) => t.task_name === id),
        ),
      ])
      .sort(([, a], [, b]) => a.localeCompare(b));

    $("exp-task-filter").innerHTML =
      '<option value="">All Tasks</option>' +
      taskOptions
        .map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`)
        .join("");

    let moreObserver = null;
    function renderSingleCard(t) {
      const sample = (t.samples || [])[0];
      const groupLabel =
        catalog.task_groups.find((g) => g.id === t.task_group)?.label ||
        displayName(t.task_group);
      const scMatches = showcaseMatchesForTask(t);
      const ids = t.track === "id" ? [] : sourceIds(t);
      const sourceNote =
        ids.length > 1
          ? ` &middot; ${ids.length} Source Rules`
          : ids.length === 1
            ? ` &middot; Paired Source`
            : "";
      return `<button type="button" class="exp-card" data-open-task="${esc(t.name)}">
        <div class="exp-preview">
          ${sample?.image ? `<img src="${esc(sample.image)}" alt="${esc(taskLabel(t))} input" loading="lazy" />` : "<span>No preview</span>"}
          <span class="exp-track-badge ${esc(t.track)}">${esc(names[t.track])}</span>
          ${scMatches.length ? '<span class="exp-rollout-flag">&#9733; Rollouts</span>' : ""}
        </div>
        <div class="exp-card-body">
          <div class="exp-card-meta">${esc(displayName(t.task_family))} &middot; ${esc(groupLabel)}</div>
          <h4>${esc(taskLabel(t))}</h4>
          <p class="exp-card-desc">${esc(description(t))}</p>
          <div class="exp-card-foot">
            <span>${taskQuestions(t)} Questions${sourceNote}</span>
            <span>Inspect Task &rarr;</span>
          </div>
        </div>
      </button>`;
    }

    function renderCardsPaged(filtered) {
      if (moreObserver) {
        moreObserver.disconnect();
        moreObserver = null;
      }
      const container = $("exp-cards");
      if (!filtered.length) {
        container.innerHTML =
          '<div class="empty-box" style="grid-column:1/-1">No tasks match the active filters. Click "Reset" to clear.</div>';
        return;
      }
      const PAGE = 48;
      let shown = 0;
      container.innerHTML = "";

      const appendBatch = () => {
        if (moreObserver) {
          moreObserver.disconnect();
          moreObserver = null;
        }
        container.querySelector("[data-load-more-wrap]")?.remove();
        const slice = filtered.slice(shown, shown + PAGE);
        shown += slice.length;
        container.insertAdjacentHTML(
          "beforeend",
          slice.map(renderSingleCard).join(""),
        );
        if (shown < filtered.length) {
          container.insertAdjacentHTML(
            "beforeend",
            `<div data-load-more-wrap style="grid-column:1/-1;text-align:center;padding:12px 0"><button type="button" class="cta-btn" data-load-more>Show More Tasks (${shown} of ${filtered.length} shown)</button></div>`,
          );
          const wrap = container.querySelector("[data-load-more-wrap]");
          wrap.querySelector("[data-load-more]").onclick = appendBatch;
          if ("IntersectionObserver" in window) {
            moreObserver = new IntersectionObserver(
              (entries) => {
                if (entries.some((en) => en.isIntersecting)) appendBatch();
              },
              { rootMargin: "600px 0px" },
            );
            moreObserver.observe(wrap);
          }
        }
      };

      appendBatch();
    }

    function renderList() {
      const inGroup = (t) => !state.taskGroup || t.task_group === state.taskGroup;
      const inFamily = (t) => !state.family || t.task_family === state.family;
      const inTrack = (t) => state.track === "all" || t.track === state.track;

      $("exp-tracks").innerHTML = Object.entries({ all: "All", ...names })
        .filter(([k]) => k === "all" || tasks.some((t) => t.track === k))
        .map(
          ([k, label]) =>
            `<button type="button" data-exp-track="${esc(k)}" aria-pressed="${state.track === k}">${esc(label)}<span>${tasks.filter((t) => (k === "all" || t.track === k) && inFamily(t) && inGroup(t)).length}</span></button>`,
        )
        .join("");

      $("exp-task-groups").innerHTML = [
        { id: "", label: "All Groups" },
        ...catalog.task_groups,
      ]
        .map(
          (g) =>
            `<button type="button" data-exp-group="${esc(g.id)}" aria-current="${state.taskGroup === g.id}">${esc(g.label)}<span>${tasks.filter((t) => (!g.id || t.task_group === g.id) && inFamily(t) && inTrack(t)).length}</span></button>`,
        )
        .join("");

      $("exp-families").innerHTML = [
        ["", "All Families"],
        ...(tasks.some((t) => t.task_family === "vbvr_pro")
          ? [["vbvr_pro", "VBVR-Pro Bench"]]
          : []),
        ...catalog.families.map((f) => [f.name, f.display_name]),
      ]
        .map(
          ([id, label]) =>
            `<button type="button" data-exp-family="${esc(id)}" aria-current="${state.family === id}">${esc(label)}<span>${tasks.filter((t) => (!id || t.task_family === id) && inTrack(t) && inGroup(t)).length}</span></button>`,
        )
        .join("");

      const qLower = state.query.toLowerCase();
      const filtered = tasks.filter(
        (t) =>
          inTrack(t) &&
          inFamily(t) &&
          inGroup(t) &&
          (!state.packageKey || t.package === state.packageKey) &&
          (!state.taskKey || t.task_name === state.taskKey) &&
          (!qLower ||
            `${t.name} ${taskLabel(t)} ${t.title} ${description(t)} ${sourceIds(t).join(" ")} ${sourceIds(t).map(displayName).join(" ")}`
              .toLowerCase()
              .includes(qLower)),
      );

      $("exp-track-note").textContent = NOTES[state.track] || NOTES.all;
      const totalQ = filtered.reduce((n, t) => n + taskQuestions(t), 0);
      $("exp-count").textContent = `${filtered.length} Tasks · ${questionCount(filtered.length, totalQ)}`;

      renderCardsPaged(filtered);
    }

    function stopModalVideos() {
      $("exp-detail-body")
        ?.querySelectorAll("video")
        .forEach((v) => {
          try {
            v.pause();
          } catch (_) {}
        });
    }

    function renderVideoBox(sample, titleLabel) {
      if (!sample?.video) {
        return `<div class="td-media-card"><div class="td-card-head"><strong>${esc(titleLabel)}</strong></div><div class="td-square-frame"><span class="td-muted">No reference video</span></div></div>`;
      }
      const fps = sample.fps || 16;
      const totalF = Number(sample.frames) || 0;
      return `
        <div class="td-media-card" data-video-unit>
          <div class="td-card-head">
            <strong>${esc(titleLabel)}</strong>
            <span>${esc(totalF)} frames @ ${esc(fps)} FPS</span>
          </div>
          <div class="td-square-frame">
            <video data-td-video data-fps="${esc(fps)}" data-frames="${esc(totalF)}" controls muted playsinline preload="metadata" poster="${esc(sample.image || "")}" src="${esc(sample.video)}"></video>
          </div>
          <div class="td-video-controls">
            <div class="td-step-btns">
              <button type="button" class="td-mini-btn" data-vact="play" title="Play from start">&#9654;</button>
              <button type="button" class="td-mini-btn" data-vact="pause" title="Pause">&#10074;&#10074;</button>
              <button type="button" class="td-mini-btn" data-vact="prev" title="Previous frame">&minus;1f</button>
              <button type="button" class="td-mini-btn" data-vact="next" title="Next frame">+1f</button>
              <select class="td-speed-sel" data-vact="speed" aria-label="Playback speed">
                <option value="0.25">0.25&times;</option>
                <option value="0.5">0.5&times;</option>
                <option value="1" selected>1&times;</option>
                <option value="2">2&times;</option>
              </select>
            </div>
            <span class="td-time-readout" data-vreadout>Frame 1${totalF ? ` / ${esc(totalF)}` : ""}</span>
          </div>
        </div>`;
    }

    function renderImageBox(sample, titleLabel, altText) {
      if (!sample?.image) {
        return `<div class="td-media-card"><div class="td-card-head"><strong>${esc(titleLabel)}</strong></div><div class="td-square-frame"><span class="td-muted">No preview</span></div></div>`;
      }
      return `
        <div class="td-media-card">
          <div class="td-card-head">
            <strong>${esc(titleLabel)}</strong>
            <span>512 &times; 512 px</span>
          </div>
          <a class="td-square-frame" href="${esc(sample.image)}" target="_blank" rel="noopener" title="Open full-resolution image in new tab">
            <img src="${esc(sample.image)}" alt="${esc(altText)}" />
          </a>
          <div class="td-card-foot">
            <span class="td-sample-id" title="${esc(sample.id || "")}">${esc(sample.id || "Sample")}</span>
            <span>Open original &nearr;</span>
          </div>
        </div>`;
    }

    function bindModalVideoUnits(container) {
      container.querySelectorAll("[data-video-unit]").forEach((unit) => {
        const video = unit.querySelector("[data-td-video]");
        if (!video) return;
        const fps = Number(video.dataset.fps) || 16;
        const totalF = Number(video.dataset.frames) || 0;
        const readout = unit.querySelector("[data-vreadout]");
        const updateReadout = () => {
          const cur = video.currentTime || 0;
          const idx1 = Math.max(1, Math.round(cur * fps) + 1);
          const clamped1 = totalF > 0 ? Math.min(totalF, idx1) : idx1;
          if (readout) {
            readout.textContent = totalF > 0 ? `Frame ${clamped1} / ${totalF}` : `Frame ${clamped1}`;
          }
        };
        video.addEventListener("timeupdate", updateReadout);
        video.addEventListener("seeked", updateReadout);
        unit.querySelector('[data-vact="play"]')?.addEventListener("click", () => {
          if (video.ended || (video.duration && video.currentTime >= video.duration - 0.02)) {
            video.currentTime = 0;
          }
          video.muted = true;
          video.play().catch(() => {});
        });
        unit.querySelector('[data-vact="pause"]')?.addEventListener("click", () => {
          video.pause();
        });
        unit.querySelector('[data-vact="prev"]')?.addEventListener("click", () => {
          video.pause();
          video.currentTime = Math.max(0, (video.currentTime || 0) - 1 / fps);
          updateReadout();
        });
        unit.querySelector('[data-vact="next"]')?.addEventListener("click", () => {
          video.pause();
          const maxT = Number.isFinite(video.duration) && video.duration > 0 ? video.duration - 0.002 : 10;
          video.currentTime = Math.min(maxT, (video.currentTime || 0) + 1 / fps);
          updateReadout();
        });
        unit.querySelector('[data-vact="speed"]')?.addEventListener("change", (e) => {
          video.playbackRate = Number(e.target.value) || 1;
        });
      });
    }

    function renderSourceSection(ids) {
      if (!ids.length) return "";
      if (ids.length === 1) {
        const id = ids[0];
        const srcTask = baseTasks.get(id);
        const srcSample = srcTask?.samples?.[0];
        return `
          <section class="td-sources-section">
            <div class="td-section-head">
              <h3>Paired Source In-Domain Task: ${esc(displayName(id))}</h3>
              <button type="button" class="btn-subtle btn-xs" data-open-task="${esc("id_" + id)}">Inspect Source Task &rarr;</button>
            </div>
            <div class="td-hero-grid">
              ${renderImageBox(srcSample, "Source Image (I₀)", `${displayName(id)} source input`)}
              ${renderVideoBox(srcSample, "Source Reference Video")}
              <div class="td-spec-card">
                <div class="td-badges">
                  <span class="badge">In-Domain Source</span>
                  <span class="badge">${esc(displayName(srcTask?.task_family || ""))}</span>
                </div>
                <div class="td-spec-body">
                  ${srcTask?.design_intent ? `<div class="td-spec-block"><h4>Source Design Intent</h4><p>${esc(srcTask.design_intent)}</p></div>` : ""}
                  ${srcTask?.logic ? `<div class="td-spec-block"><h4>Source Rule Logic</h4><p>${esc(srcTask.logic)}</p></div>` : ""}
                  ${
                    srcSample?.prompt
                      ? `<details class="td-source-prompt-details"><summary>View Source Prompt (P)</summary><pre class="prompt-box">${esc(srcSample.prompt)}</pre></details>`
                      : ""
                  }
                </div>
              </div>
            </div>
          </section>`;
      }

      const cardsHtml = ids
        .map((id, idx) => {
          const srcTask = baseTasks.get(id);
          const srcSample = srcTask?.samples?.[0];
          return `
            <div class="td-constituent-card">
              <div class="td-constituent-head">
                <div>
                  <span class="eyebrow">Constituent Rule ${String.fromCharCode(65 + idx)} &middot; In-Domain</span>
                  <h4>${esc(displayName(id))}</h4>
                </div>
                <button type="button" class="btn-subtle btn-xs" data-open-task="${esc("id_" + id)}">Inspect &rarr;</button>
              </div>
              <div class="td-constituent-media">
                ${renderImageBox(srcSample, "Source I₀", `${displayName(id)} input`)}
                ${renderVideoBox(srcSample, "Source Reference Video")}
              </div>
              <p class="td-constituent-logic">${esc(srcTask?.logic || srcTask?.design_intent || "")}</p>
            </div>`;
        })
        .join("");

      return `
        <section class="td-sources-section">
          <div class="td-section-head">
            <h3>Constituent In-Domain Tasks (${ids.length} Rules Combined)</h3>
          </div>
          <div class="td-constituent-grid cols-${Math.min(ids.length, 3)}">${cardsHtml}</div>
        </section>`;
    }

    function openDetail(taskId, push = true) {
      const task = byId.get(taskId);
      if (!task) return;
      stopModalVideos();
      state.activeTask = taskId;

      const sample = (task.samples || [])[0];
      const baseTask = baseTasks.get(task.task_name);
      const ids = task.track === "id" ? [] : sourceIds(task);
      const groupLabel =
        catalog.task_groups.find((g) => g.id === task.task_group)?.label ||
        displayName(task.task_group);

      $("exp-breadcrumb").textContent = `${displayName(task.task_family)} · ${groupLabel} · ${names[task.track]}`;
      $("exp-detail-title").textContent = taskLabel(task);

      const scMatches = showcaseMatchesForTask(task);
      $("exp-dialog-rollout-slot").innerHTML = scMatches.length
        ? `<button type="button" class="cta-btn cta-primary" data-jump-showcase="${esc(scMatches[0].id)}">&#9654; Watch Model Rollouts (${esc(scMatches[0].paper_ref)})</button>`
        : "";

      let specBlocksHtml = "";
      if (task.track === "id" && baseTask) {
        specBlocksHtml = `
          <div class="td-spec-block">
            <h4>Design Intent</h4>
            <p>${esc(baseTask.design_intent)}</p>
          </div>
          <div class="td-spec-block">
            <h4>Rule &amp; Execution Logic</h4>
            <p>${esc(baseTask.logic)}</p>
          </div>`;
      } else {
        specBlocksHtml = `
          <div class="td-spec-block">
            <h4>Task Specification</h4>
            <p>${esc(description(task))}</p>
          </div>
          ${
            sample?.plan?.visual_change
              ? `<div class="td-spec-block"><h4>Visual Redesign</h4><p>${esc(sample.plan.visual_change)}</p></div>`
              : ""
          }
          ${
            sample?.plan?.semantic_contract
              ? `<div class="td-spec-block"><h4>Preserved Invariant</h4><p class="td-muted">${esc(sample.plan.semantic_contract)}</p></div>`
              : ""
          }
          ${
            task.vbvr_level
              ? `<div class="td-spec-block"><h4>Benchmark Provenance</h4><p class="td-muted">VBVR-Pro Training-Relative Level: ${esc(vbvrLevels[task.vbvr_level] || task.vbvr_level)} &middot; Official Split: ${esc(displayName(task.official_split))}${task.training_neighbors?.length ? ` &middot; Nearest Training Task(s): ${esc(task.training_neighbors.map(displayName).join(", "))}` : ""}</p></div>`
              : ""
          }`;
      }

      const sourceChipsHtml = ids.length
        ? `<div class="td-source-chips">
            <span>${ids.length > 1 ? "Combines:" : "Source Task:"}</span>
            ${ids
              .map(
                (id) =>
                  `<button type="button" data-open-task="${esc("id_" + id)}">${esc(displayName(id))} &rarr;</button>`,
              )
              .join("")}
          </div>`
        : "";

      const inlinePrompt = task.track !== "id" && Boolean(sample?.prompt);
      const body = $("exp-detail-body");
      body.innerHTML = `
        <div class="td-hero-grid ${inlinePrompt ? "is-derived" : "is-id"}">
          ${renderImageBox(sample, "Initial Image (I₀)", `${taskLabel(task)} initial input`)}
          ${renderVideoBox(sample, "Ground-Truth Video")}
          <div class="td-spec-card">
            <div class="td-badges">
              <span class="badge badge-ref">${esc(names[task.track])}</span>
              <span class="badge ${task.task_group === "single_step" ? "badge-ss" : "badge-ms"}">${esc(groupLabel)}</span>
              <span class="badge">${taskQuestions(task)} Questions</span>
            </div>
            <div class="td-spec-body">
              ${specBlocksHtml}
              ${
                inlinePrompt
                  ? `<div class="td-inline-prompt">
                      <div class="td-prompt-head">
                        <strong>Self-Contained Input Prompt (P)</strong>
                        <button type="button" class="btn-subtle btn-xs" data-copy-prompt>Copy Prompt</button>
                      </div>
                      <pre class="prompt-box td-prompt-pre">${esc(sample.prompt)}</pre>
                    </div>`
                  : ""
              }
            </div>
            ${sourceChipsHtml}
          </div>
        </div>
        ${
          !inlinePrompt && sample?.prompt
            ? `<div class="td-prompt-card">
                <div class="td-prompt-head">
                  <strong>Self-Contained Input Prompt (P)</strong>
                  <button type="button" class="btn-subtle btn-xs" data-copy-prompt>Copy Prompt</button>
                </div>
                <pre class="prompt-box td-prompt-pre">${esc(sample.prompt)}</pre>
              </div>`
            : ""
        }
        ${renderSourceSection(ids)}
      `;

      bindModalVideoUnits(body);
      const copyBtn = body.querySelector("[data-copy-prompt]");
      if (copyBtn && sample?.prompt) {
        copyBtn.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(sample.prompt);
            copyBtn.textContent = "Copied";
            setTimeout(() => {
              copyBtn.textContent = "Copy Prompt";
            }, 1500);
          } catch (_) {}
        });
      }

      const dlg = $("exp-detail-dialog");
      if (!dlg.open) dlg.showModal();
      syncUrl(push);
    }

    // Bind events
    $("exp-tracks").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-exp-track]");
      if (!btn) return;
      state.track = btn.dataset.expTrack;
      renderList();
      syncUrl(true);
    });
    $("exp-task-groups").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-exp-group]");
      if (!btn) return;
      state.taskGroup = btn.dataset.expGroup;
      renderList();
      syncUrl(true);
    });
    $("exp-families").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-exp-family]");
      if (!btn) return;
      state.family = btn.dataset.expFamily;
      renderList();
      syncUrl(true);
    });
    $("exp-search-input").addEventListener("input", (e) => {
      state.query = e.target.value.trim();
      renderList();
      syncUrl(false);
    });
    $("exp-task-filter").addEventListener("change", (e) => {
      state.taskKey = e.target.value;
      renderList();
      syncUrl(true);
    });
    $("exp-reset-btn").addEventListener("click", () => {
      state = {
        track: "all",
        family: "",
        taskGroup: "",
        query: "",
        packageKey: "",
        taskKey: "",
        activeTask: null,
      };
      $("exp-search-input").value = "";
      $("exp-task-filter").value = "";
      renderList();
      syncUrl(true);
    });
    $("exp-cards").addEventListener("click", (e) => {
      const card = e.target.closest("[data-open-task]");
      if (card) openDetail(card.dataset.openTask, true);
    });
    $("exp-detail-dialog").addEventListener("click", (e) => {
      const openBtn = e.target.closest("[data-open-task]");
      if (openBtn) {
        openDetail(openBtn.dataset.openTask, true);
        return;
      }
      const jumpSc = e.target.closest("[data-jump-showcase]");
      if (jumpSc) {
        const caseId = jumpSc.dataset.jumpShowcase;
        $("exp-detail-dialog").close();
        setExplorerMode("rollouts", { showcaseCaseId: caseId });
      }
    });

    const dlg = $("exp-detail-dialog");
    $("exp-dialog-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => {
      if (e.target === dlg) dlg.close();
    });
    dlg.addEventListener("close", () => {
      stopModalVideos();
      if (state.activeTask) {
        state.activeTask = null;
        syncUrl(false);
      }
    });

    window.VideoGymExplorer._applyCatalogState = (opts = {}, push = false) => {
      const namesObj = catalog.evaluation.tracks;
      if (opts.fromUrl) {
        const q = new URL(location.href).searchParams;
        state.track = Object.hasOwn(namesObj, q.get("track")) ? q.get("track") : "all";
        state.family = q.get("family") || "";
        state.taskGroup = q.get("taskGroup") || "";
        state.query = q.get("q") || "";
        state.packageKey = q.get("package") || "";
        state.taskKey = q.get("taskFilter") || "";
        state.activeTask = q.get("task") || null;
      } else {
        if (opts.resetFilters) {
          state.track = "all";
          state.family = "";
          state.taskGroup = "";
          state.query = "";
          state.packageKey = "";
          state.taskKey = "";
          state.activeTask = null;
        }
        if (opts.track !== undefined) {
          state.track =
            opts.track === "all" || Object.hasOwn(namesObj, opts.track)
              ? opts.track
              : "all";
        }
        if (opts.family !== undefined) {
          state.family = opts.family || "";
        }
        if (opts.taskGroup !== undefined) {
          state.taskGroup = opts.taskGroup || "";
        }
        if (opts.taskName !== undefined) {
          const exact =
            tasks.find((t) => t.task_name === opts.taskName && t.track === "id") ||
            tasks.find((t) => t.task_name === opts.taskName) ||
            byId.get(opts.taskName);
          if (exact) state.activeTask = exact.name;
        }
        if (opts.task !== undefined) {
          state.activeTask = byId.has(opts.task) ? opts.task : null;
        }
      }

      $("exp-search-input").value = state.query;
      $("exp-task-filter").value = state.taskKey;

      renderList();
      if (state.activeTask && byId.has(state.activeTask)) {
        openDetail(state.activeTask, false);
      } else {
        if (dlg.open) dlg.close();
        syncUrl(push);
      }
    };
  }

  async function setExplorerMode(mode, opts = {}, push = true) {
    explorerMode = mode === "rollouts" ? "rollouts" : "tasks";
    $("exp-tab-tasks").classList.toggle("is-active", explorerMode === "tasks");
    $("exp-tab-rollouts").classList.toggle("is-active", explorerMode === "rollouts");

    if (explorerMode === "rollouts") {
      $("exp-tasks-panel").hidden = true;
      $("exp-rollouts-panel").hidden = false;
      const loadingEl = $("rollouts-loading");
      const contentEl = $("rollouts-content");
      if (!window.VIDEOGYM_SHOWCASE) {
        loadingEl.hidden = false;
        contentEl.hidden = true;
      }
      const showcase = await loadShowcase();
      loadingEl.hidden = true;
      contentEl.hidden = false;
      initRolloutsPanel(showcase);
      window.VideoGymExplorer._renderRollouts(opts.showcaseCaseId);
      syncUrl(push);
    } else {
      cleanupRolloutVideos();
      $("exp-rollouts-panel").hidden = true;
      $("exp-tasks-panel").hidden = false;
      const loadingEl = $("exp-loading");
      const contentEl = $("exp-content");
      if (!window.VIDEOGYM_EXPLORER) {
        loadingEl.hidden = false;
        contentEl.hidden = true;
      }
      const [catalog] = await Promise.all([loadCatalog(), loadShowcase()]);
      initCatalogPanel(catalog);
      loadingEl.hidden = true;
      contentEl.hidden = false;
      window.VideoGymExplorer._applyCatalogState(opts, push);
    }
  }

  document.querySelectorAll("[data-exp-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setExplorerMode(btn.dataset.expMode, {}, true);
    });
  });

  async function ensureLoaded(opts = {}, push = false) {
    const q = new URL(location.href).searchParams;
    const requestedTab = opts.tab || (opts.fromUrl ? q.get("tab") : "") || "tasks";
    await setExplorerMode(requestedTab, opts, push);
  }

  function closeAll() {
    cleanupRolloutVideos();
    const dlg = $("exp-detail-dialog");
    if (dlg?.open) dlg.close();
  }

  window.VideoGymExplorer = {
    ensureLoaded,
    closeAll,
    _applyCatalogState: () => {},
    _renderRollouts: () => {},
  };
})();
