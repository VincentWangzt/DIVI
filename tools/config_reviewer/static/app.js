const state = {
  metadata: null,
  comparison: null,
  snapshotHash: null,
};

const els = {
  subtitle: document.getElementById("subtitle"),
  liveStatus: document.getElementById("liveStatus"),
  refreshButton: document.getElementById("refreshButton"),
  modeSelect: document.getElementById("modeSelect"),
  methodSelect: document.getElementById("methodSelect"),
  methodGroupSelect: document.getElementById("methodGroupSelect"),
  targetSelect: document.getElementById("targetSelect"),
  groupSelect: document.getElementById("groupSelect"),
  changedOnly: document.getElementById("changedOnly"),
  viewSelect: document.getElementById("viewSelect"),
  multiPicker: document.getElementById("multiPicker"),
  methodControl: document.getElementById("methodControl"),
  methodGroupControl: document.getElementById("methodGroupControl"),
  targetControl: document.getElementById("targetControl"),
  groupControl: document.getElementById("groupControl"),
  comparisonTitle: document.getElementById("comparisonTitle"),
  comparisonSummary: document.getElementById("comparisonSummary"),
  pathFilter: document.getElementById("pathFilter"),
  configColumns: document.getElementById("configColumns"),
  errorBox: document.getElementById("errorBox"),
  keyTableHead: document.getElementById("keyTableHead"),
  keyTableBody: document.getElementById("keyTableBody"),
  tableView: document.getElementById("tableView"),
  rawView: document.getElementById("rawView"),
  matrixView: document.getElementById("matrixView"),
  rawPanels: document.getElementById("rawPanels"),
  matrixPanel: document.getElementById("matrixPanel"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function setStatus(message, mode = "ok") {
  els.liveStatus.textContent = message;
  els.liveStatus.style.background = mode === "error" ? "#fff0f0" : "#e9f8ee";
  els.liveStatus.style.color = mode === "error" ? "#a42020" : "#166534";
}

function showError(error) {
  els.errorBox.textContent = error ? String(error.message || error) : "";
  els.errorBox.classList.toggle("hidden", !error);
}

function option(value, label) {
  const el = document.createElement("option");
  el.value = value;
  el.textContent = label;
  return el;
}

function populateControls() {
  const metadata = state.metadata;
  const current = {
    method: els.methodSelect.value || metadata.defaults.method,
    target: els.targetSelect.value || metadata.defaults.target,
    methodGroup: els.methodGroupSelect.value || metadata.defaults.method_group,
    group: els.groupSelect.value || metadata.defaults.target_group,
  };
  els.methodSelect.replaceChildren(...metadata.methods.map((item) => option(item.name, item.label)));
  els.targetSelect.replaceChildren(...metadata.targets.map((item) => option(item.name, item.label)));
  els.methodGroupSelect.replaceChildren(
    ...Object.keys(metadata.method_groups).map((name) => option(name, name))
  );
  els.groupSelect.replaceChildren(
    ...Object.keys(metadata.target_groups).map((name) => option(name, name))
  );
  els.methodSelect.value = metadata.methods.some((item) => item.name === current.method)
    ? current.method
    : metadata.defaults.method;
  els.targetSelect.value = metadata.targets.some((item) => item.name === current.target)
    ? current.target
    : metadata.defaults.target;
  els.methodGroupSelect.value = Object.hasOwn(metadata.method_groups, current.methodGroup)
    ? current.methodGroup
    : metadata.defaults.method_group;
  els.groupSelect.value = Object.hasOwn(metadata.target_groups, current.group)
    ? current.group
    : metadata.defaults.target_group;
  renderMultiPicker();
  updateControlVisibility();
}

function selectedGroupTargets() {
  const group = els.groupSelect.value || "all";
  return state.metadata.target_groups[group] || state.metadata.target_groups.all;
}

function selectedGroupMethods() {
  const group = els.methodGroupSelect.value || "all";
  return state.metadata.method_groups[group] || state.metadata.method_groups.all;
}

function checkItem(name, value, label, checked) {
  const row = document.createElement("div");
  row.className = "multi-item";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.name = name;
  input.value = value;
  input.checked = checked;
  input.addEventListener("change", loadComparison);
  const text = document.createElement("label");
  text.textContent = label;
  row.append(input, text);
  return row;
}

function renderMultiPicker() {
  const mode = els.modeSelect.value;
  els.multiPicker.innerHTML = "";

  if (mode === "methods_for_target") {
    const title = document.createElement("label");
    title.textContent = "Methods";
    els.multiPicker.appendChild(title);
    const methods = selectedGroupMethods();
    for (const method of state.metadata.methods) {
      const checked = methods.includes(method.name);
      els.multiPicker.appendChild(checkItem("multiMethod", method.name, method.label, checked));
    }
    return;
  }

  if (mode === "targets_for_method") {
    const title = document.createElement("label");
    title.textContent = "Targets";
    els.multiPicker.appendChild(title);
    const targets = selectedGroupTargets();
    for (const target of state.metadata.targets) {
      const checked = targets.includes(target.name);
      els.multiPicker.appendChild(checkItem("multiTarget", target.name, target.label, checked));
    }
  }
}

function checkedValues(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((el) => el.value);
}

function updateControlVisibility() {
  const mode = els.modeSelect.value;
  // In "methods_for_target" mode: show target picker + method group, hide method picker + target group
  // In "targets_for_method" mode: show method picker + target group, hide target picker + method group
  els.methodControl.classList.toggle("hidden", mode === "methods_for_target");
  els.methodGroupControl.classList.toggle("hidden", mode !== "methods_for_target");
  els.targetControl.classList.toggle("hidden", mode === "targets_for_method");
  els.groupControl.classList.toggle("hidden", mode !== "targets_for_method");
}

function compareUrl() {
  const params = new URLSearchParams();
  const mode = els.modeSelect.value;
  params.set("mode", mode);
  if (mode === "methods_for_target") {
    params.set("target", els.targetSelect.value);
    const methods = checkedValues("multiMethod");
    params.set("methods", methods.length ? methods.join(",") : selectedGroupMethods().join(","));
  } else if (mode === "targets_for_method") {
    params.set("method", els.methodSelect.value);
    const targets = checkedValues("multiTarget");
    params.set("targets", targets.length ? targets.join(",") : selectedGroupTargets().join(","));
  }
  return `/api/compare?${params.toString()}`;
}

async function loadMetadata() {
  state.metadata = await fetchJson("/api/metadata");
  populateControls();
}

async function loadComparison() {
  if (!state.metadata) {
    return;
  }
  showError(null);
  try {
    state.comparison = await fetchJson(compareUrl());
    render();
    setStatus("up to date");
  } catch (error) {
    showError(error);
    setStatus("error", "error");
  }
}

function render() {
  const comparison = state.comparison;
  if (!comparison) {
    return;
  }
  els.comparisonTitle.textContent = comparison.title;
  const summary = comparison.summary;
  els.comparisonSummary.textContent =
    `${summary.config_count} configs, ${summary.row_count} keys, ` +
    `${summary.changed_count} changed, ${summary.missing_file_count} missing files, ${summary.error_count} errors`;
  renderConfigCards(comparison.configs);
  renderTable(comparison);
  renderRaw(comparison);
  renderMatrix(comparison);
  updateView();
}

function configTitle(config) {
  return `${config.method_label} / ${config.target}`;
}

function renderConfigCards(configs) {
  els.configColumns.innerHTML = configs.map((config) => {
    const badge = config.error
      ? `<span class="badge error">error</span>`
      : config.exists
        ? `<span class="badge">available</span>`
        : `<span class="badge">missing</span>`;
    return `
      <article class="config-card">
        <h3>${escapeHtml(configTitle(config))}</h3>
        <p>${escapeHtml(config.display_path || config.path)}</p>
        ${badge}
      </article>
    `;
  }).join("");
}

function renderTable(comparison) {
  const headers = [
    `<tr><th class="key-cell">Key</th>${comparison.configs
      .map((config) => `<th>${escapeHtml(configTitle(config))}</th>`)
      .join("")}</tr>`,
  ];
  els.keyTableHead.innerHTML = headers.join("");

  const filter = els.pathFilter.value.trim().toLowerCase();
  const rows = comparison.rows.filter((row) => {
    if (els.changedOnly.checked && !row.changed) {
      return false;
    }
    if (filter && !row.path.toLowerCase().includes(filter)) {
      return false;
    }
    return true;
  });

  els.keyTableBody.innerHTML = rows.map((row) => {
    const pad = Math.min(row.depth, 10) * 16;
    const cells = row.cells.map((cell) => {
      if (cell.status === "missing_file") {
        return `<td class="value-cell cell-missing">missing file</td>`;
      }
      if (cell.status === "missing_key") {
        return `<td class="value-cell cell-missing">missing key</td>`;
      }
      if (cell.status === "error") {
        return `<td class="value-cell cell-error">load error</td>`;
      }
      return `<td class="value-cell">${escapeHtml(cell.display)}</td>`;
    }).join("");
    return `
      <tr class="${row.changed ? "changed" : ""}">
        <td class="key-cell" style="padding-left: ${10 + pad}px">${escapeHtml(row.path)}</td>
        ${cells}
      </tr>
    `;
  }).join("");
}

async function renderRaw(comparison) {
  els.rawPanels.innerHTML = comparison.configs.map((config) => `
    <article class="raw-panel">
      <h3>${escapeHtml(configTitle(config))}</h3>
      <pre id="raw-${escapeHtml(config.id.replaceAll(":", "-"))}">Loading...</pre>
    </article>
  `).join("");

  for (const config of comparison.configs) {
    const pre = document.getElementById(`raw-${config.id.replaceAll(":", "-")}`);
    try {
      const raw = await fetchJson(`/api/raw?method=${encodeURIComponent(config.method)}&target=${encodeURIComponent(config.target)}`);
      pre.textContent = raw.error || raw.text || "missing";
    } catch (error) {
      pre.textContent = error.message || String(error);
    }
  }
}

function renderMatrix(comparison) {
  const rows = comparison.rows;
  const changed = rows.filter((row) => row.changed).length;
  const missingKeys = rows.filter((row) => row.cells.some((cell) => cell.status === "missing_key")).length;
  const cards = [
    ["Displayed configs", comparison.configs.length],
    ["Key rows", rows.length],
    ["Changed rows", changed],
    ["Rows with missing keys", missingKeys],
    ["Missing files", comparison.summary.missing_file_count],
    ["Load errors", comparison.summary.error_count],
  ];
  els.matrixPanel.innerHTML = `
    <div class="matrix-grid">
      ${cards.map(([label, value]) => `
        <article class="matrix-card">
          <h3>${escapeHtml(label)}</h3>
          <dl><dt>Count</dt><dd>${escapeHtml(value)}</dd></dl>
        </article>
      `).join("")}
    </div>
  `;
}

function updateView() {
  const view = els.viewSelect.value;
  els.tableView.classList.toggle("hidden", view !== "table");
  els.rawView.classList.toggle("hidden", view !== "raw");
  els.matrixView.classList.toggle("hidden", view !== "matrix");
}

async function pollSnapshot() {
  try {
    const snapshot = await fetchJson("/api/snapshot");
    if (state.snapshotHash && snapshot.hash !== state.snapshotHash) {
      state.snapshotHash = snapshot.hash;
      setStatus("files changed; reloading");
      await loadMetadata();
      await loadComparison();
      return;
    }
    state.snapshotHash = snapshot.hash;
    setStatus("up to date");
  } catch (error) {
    setStatus("poll error", "error");
  }
}

function bindEvents() {
  for (const el of [
    els.modeSelect,
    els.methodSelect,
    els.targetSelect,
    els.changedOnly,
  ]) {
    el.addEventListener("change", () => {
      if (el === els.modeSelect) {
        updateControlVisibility();
        renderMultiPicker();
      }
      loadComparison();
    });
  }
  els.groupSelect.addEventListener("change", () => {
    renderMultiPicker();
    loadComparison();
  });
  els.methodGroupSelect.addEventListener("change", () => {
    renderMultiPicker();
    loadComparison();
  });
  els.viewSelect.addEventListener("change", updateView);
  els.pathFilter.addEventListener("input", () => {
    if (state.comparison) {
      renderTable(state.comparison);
    }
  });
  els.refreshButton.addEventListener("click", async () => {
    await loadMetadata();
    await loadComparison();
    await pollSnapshot();
  });
}

async function boot() {
  bindEvents();
  try {
    await loadMetadata();
    await loadComparison();
    await pollSnapshot();
    setInterval(pollSnapshot, 3000);
  } catch (error) {
    showError(error);
    setStatus("startup error", "error");
  }
}

boot();
