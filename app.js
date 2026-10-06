// Graphspace site: playground (real package in Pyodide) + benchmark dashboard
// (numbers read from benchmarks/results at build time).
const $ = (s) => document.querySelector(s);
const SVG = "http://www.w3.org/2000/svg";
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const bytes = (b) => (b == null ? "—" : b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KiB` : `${(b / 1048576).toFixed(2)} MiB`);
const fmt = (v) => (v == null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v));
function el(name, attrs = {}, parent) { const n = document.createElementNS(SVG, name); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); parent?.appendChild(n); return n; }
function text(parent, x, y, content, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = content; return t; }
function table(container, columns, rows) {
  container.innerHTML = `<div class="table-scroll"><table><thead><tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${columns.map((c) => `<td class="${c.num ? "num" : ""}">${c.html ? c.html(r) : esc(c.value(r))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
const chartWidth = (c) => Math.max(280, Math.round(c.clientWidth || 640));
function responsive(container, draw) { draw(); let last = container.clientWidth; new ResizeObserver(() => { if (Math.abs(container.clientWidth - last) > 4) { last = container.clientWidth; draw(); } }).observe(container); }
function tooltip(container) {
  const tip = document.createElement("div"); tip.className = "tooltip"; tip.hidden = true; container.appendChild(tip);
  return { show(html, e) { const b = container.getBoundingClientRect(); tip.innerHTML = html; tip.hidden = false; tip.style.left = `${Math.min(Math.max(0, e.clientX - b.left + 12), b.width - tip.offsetWidth)}px`; tip.style.top = `${Math.max(0, e.clientY - b.top - tip.offsetHeight - 12)}px`; }, hide() { tip.hidden = true; } };
}

const [examples, bench, build] = await Promise.all(["data/examples.json", "data/benchmarks.json", "data/build.json"].map((u) => fetch(u).then((r) => r.json())));
document.querySelectorAll("[data-version]").forEach((n) => { n.textContent = build.graphspace; });
$("#pyodide-version").textContent = build.pyodide;
$("#package-hash").textContent = `sha256 ${build.package_sha256.slice(0, 16)}…`;

// ---------- stats ----------
{
  const r4 = bench.rounds.find((r) => r.round === 4);
  const cov = bench.calibration.versions;
  $("#stats").innerHTML = [
    ["0", "runtime dependencies: the core is pure standard-library Python"],
    [`${bench.rounds.length} rounds`, `of preregistered benchmark thresholds; round 1 failed and is kept`],
    [`${r4.T0}/${r4.T1}/${r4.T2}/${r4.T3}`.replaceAll("PASS", "✓"), "round 4 model blocks: correctness, time, memory ≤ estimate, memory < NumPy"],
    [`${Math.min(...cov.map((v) => v.covered))}–${Math.max(...cov.map((v) => v.covered))}/300`, "held-out first calls within the memory estimate, CPython 3.10–3.14"],
  ].map(([v, l]) => `<div class="stat"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join("");
}

// ---------- playground ----------
const BASIS = [
  ["declared", "Stated by the user"], ["proven", "Established by graph construction rules"], ["inferred", "Derived from input sizes at execution"],
  ["estimated", "Computed from a model, not observed"], ["runtime_checked", "Verified during execution"], ["measured", "Observed during execution"], ["backend_reported", "Supplied by the backend"],
];
table($("#basis-table"), [{ label: "basis", html: (r) => `<span class="basis ${r[0]}">${r[0]}</span>` }, { label: "meaning", value: (r) => r[1] }], BASIS);

const FAILS = new Set(["04_shape_error", "05_memory_contract", "06_unbound_dims", "07_integer_overflow"]);
const LABELS = { "01_classifier": "Classifier", "02_mlp": "MLP · symbolic batch", "03_attention": "Attention block", "04_shape_error": "Shape error", "05_memory_contract": "Memory contract", "06_unbound_dims": "Unbound dimension", "07_integer_overflow": "Integer overflow", "08_reshape_symbolic": "Symbolic reshape" };
const codeEl = $("#code");
let current = examples[0], lastReport = null, activeTab = "graph";
$("#examples").innerHTML = examples.map((e, i) => `<button type="button" role="tab" aria-selected="${i === 0}" data-id="${e.id}">${esc(LABELS[e.id] ?? e.id)}${FAILS.has(e.id) ? '<span class="tag">fails</span>' : ""}</button>`).join("");
const loadExample = (id) => {
  current = examples.find((e) => e.id === id);
  document.querySelectorAll("#examples button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.id === id)));
  codeEl.value = current.code; $("#editor-title").textContent = `${current.id}.py`;
};
$("#examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; loadExample(b.dataset.id); if (ready) run(); });
$("#reset").addEventListener("click", () => loadExample(current.id));
codeEl.addEventListener("keydown", (e) => {
  if (e.key === "Tab") { e.preventDefault(); const s = codeEl.selectionStart; codeEl.setRangeText("    ", s, codeEl.selectionEnd, "end"); }
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
});
loadExample(examples[0].id);

const worker = new Worker("worker.js");
let ready = false, seq = 0;
const pending = new Map();
const runtime = $("#runtime"), runtimeText = $("#runtime-text");
worker.onmessage = ({ data }) => {
  if (data.type === "status") runtimeText.textContent = data.text;
  else if (data.type === "ready") { ready = true; runtime.classList.add("ready"); runtimeText.textContent = `Ready · graphspace ${data.version} on CPython ${data.python} (WebAssembly)`; $("#run").disabled = false; run(); }
  else if (data.type === "failed") { runtime.classList.add("error"); runtimeText.textContent = `Python failed to load: ${data.error}`; }
  else if (data.type === "result") { const p = pending.get(data.id); pending.delete(data.id); p(data); }
};
worker.onerror = (e) => { runtime.classList.add("error"); runtimeText.textContent = `Python failed to load: ${e.message || "worker error"}`; };
function run() {
  if (!ready) return;
  $("#run").disabled = true; $("#run").textContent = "Running…";
  const id = ++seq;
  pending.set(id, (msg) => {
    $("#run").disabled = false; $("#run").textContent = "Run";
    if (!msg.ok) { lastReport = { stages: [], failure: { stage: "runtime", type: "Error", message: msg.error } }; } else lastReport = msg.report;
    renderReport();
  });
  worker.postMessage({ id, code: codeEl.value });
}
$("#run").addEventListener("click", run);
document.querySelector(".tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return; activeTab = b.dataset.tab;
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  renderPanel();
});

function renderReport() {
  const r = lastReport, stages = ["construction", "validation", "execution"];
  $("#pipeline").innerHTML = stages.map((s) => { const st = r.stages.find((x) => x.stage === s); return `<li class="${st ? (st.ok ? "ok" : "fail") : ""}">${s}${st ? (st.ok ? " ✓" : " ✗") : ""}</li>`; }).join("");
  const f = r.failure;
  $("#failure").innerHTML = f ? `<div class="failure"><h3>${esc(f.type)} at ${esc(f.stage)}</h3><div>${esc(f.message)}</div><dl>${["code", "graph", "node", "expected", "actual", "remediation"].filter((k) => f[k] != null).map((k) => `<dt>${k}</dt><dd>${esc(fmt(f[k]))}</dd>`).join("")}</dl></div>` : "";
  if (f && activeTab === "record" && !r.record) activeTab = r.graph ? "graph" : "console";
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x.dataset.tab === activeTab)));
  renderPanel();
}

function renderPanel() {
  const panel = $("#panel"), r = lastReport;
  if (!r) return;
  const g = r.graph;
  if (activeTab === "graph") {
    if (!g) { panel.innerHTML = `<p class="muted">No graph was built.</p>`; return; }
    panel.innerHTML = `<dl class="kv"><dt>graph</dt><dd>${esc(g.name)}</dd><dt>inputs</dt><dd>${g.inputs.length}</dd><dt>nodes</dt><dd>${g.nodes.length}</dd><dt>contract</dt><dd>${esc(g.resources.max_memory_bytes != null ? `max ${bytes(g.resources.max_memory_bytes)}` : "no memory limit")}${g.resources.deterministic ? " · deterministic" : ""}</dd></dl><div id="graph-view" class="chart"></div>`;
    drawGraph($("#graph-view"), g);
  } else if (activeTab === "memory") {
    if (!g?.memory_plan) { panel.innerHTML = g?.memory_plan_error ? `<p>${esc(g.memory_plan_error.message)}</p>` : `<p class="muted">No memory plan.</p>`; return; }
    const mp = g.memory_plan;
    panel.innerHTML = `<dl class="kv"><dt>peak memory</dt><dd>${mp.peak_memory_bytes == null ? "unknown: bind the symbolic dimensions (dims)" : `${bytes(mp.peak_memory_bytes)} <span class="basis estimated">estimated</span>`}</dd><dt>bookkeeping</dt><dd>${bytes(mp.bookkeeping_bytes)}</dd><dt>reused buffers</dt><dd>${mp.reusable_buffers}</dd></dl><p class="muted small">Each row is a buffer; each bar is a value living in it from its first to its last use. Several bars in one row show a buffer being reused. Grey rows are caller inputs, which are never reused.</p><div id="memory-view" class="chart"></div>`;
    drawMemory($("#memory-view"), mp, new Set(g.inputs.map((i) => i.name)));
  } else if (activeTab === "claims") {
    const claims = r.record?.claims ?? g?.analysis;
    if (!claims) { panel.innerHTML = `<p class="muted">No claims.</p>`; return; }
    panel.innerHTML = `<p class="muted small">${r.record ? "Execution claims from the ExecutionRecord." : "Analysis claims from graph.analyze()."}</p><div id="claims-table"></div>`;
    table($("#claims-table"), [{ label: "claim", html: (c) => `<code>${esc(c.name)}</code>` }, { label: "value", value: (c) => (typeof c.value === "string" && c.value.length > 40 ? `${c.value.slice(0, 40)}…` : fmt(c.value)) }, { label: "basis", html: (c) => `<span class="basis ${esc(c.basis)}">${esc(c.basis)}</span>` }], claims);
  } else if (activeTab === "record") {
    if (!r.record) { panel.innerHTML = `<p class="muted">${r.failure ? "Nothing was executed." : "Define <code>values</code> to execute the graph."}</p>`; return; }
    const rec = r.record, out = Array.isArray(r.result) ? r.result : [r.result];
    panel.innerHTML = `<dl class="kv"><dt>output</dt><dd class="mono">[${out.slice(0, 12).map((v) => (typeof v === "number" ? Number(v.toPrecision(6)) : v)).join(", ")}${out.length > 12 ? `, … (${out.length})` : ""}]</dd><dt>backend</dt><dd>${esc(rec.backend)}</dd><dt>graph sha256</dt><dd class="hash">${esc(rec.graph_sha256)}</dd><dt>output sha256</dt><dd class="hash">${esc(rec.output_sha256 ?? "—")}</dd><dt>python</dt><dd>${esc(rec.python_version)}</dd></dl><pre>${esc(JSON.stringify(rec, null, 2))}</pre>`;
  } else {
    panel.innerHTML = `<pre>${esc(r.stdout || "(no output)")}</pre>`;
  }
}

function drawGraph(container, g) {
  const nodes = [...g.inputs.map((i) => ({ id: i.name, op: "input", shape: i.spec.shape, dtype: i.spec.dtype, inputs: [] })),
    ...g.nodes.map((n) => ({ id: n.output, op: n.operation, shape: n.spec.shape, dtype: n.spec.dtype, inputs: n.inputs }))];
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const depth = (n) => (n.d ??= n.inputs.length ? 1 + Math.max(...n.inputs.map((i) => (byId[i] ? depth(byId[i]) : 0))) : 0);
  nodes.forEach(depth);
  const layers = [];
  nodes.forEach((n) => (layers[n.d] ??= []).push(n));
  const width = chartWidth(container), boxW = 116, boxH = 44, gapY = 22;
  const colW = layers.length > 1 ? Math.max(boxW + 24, (width - boxW - 8) / (layers.length - 1)) : boxW;
  const svgW = Math.max(width, (layers.length - 1) * colW + boxW + 8);
  const height = Math.max(...layers.map((l) => l.length)) * (boxH + gapY) + 10;
  layers.forEach((l, li) => l.forEach((n, ni) => { n.x = 4 + li * colW; n.y = 6 + ni * (boxH + gapY) + (height - l.length * (boxH + gapY)) / 2; }));
  const svg = el("svg", { viewBox: `0 0 ${svgW} ${height}`, class: "graph-svg", role: "img", "aria-label": `Graph ${g.name}` }, container);
  if (svgW > width) { container.style.overflowX = "auto"; svg.style.width = `${svgW}px`; }
  for (const n of nodes) for (const i of n.inputs) { const s = byId[i]; if (!s) continue; const x1 = s.x + boxW, y1 = s.y + boxH / 2, x2 = n.x, y2 = n.y + boxH / 2, mx = (x1 + x2) / 2; el("path", { class: "edge", d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}` }, svg); }
  for (const n of nodes) {
    const gEl = el("g", { class: `node ${n.op === "input" ? "input" : ""} ${n.id === g.output ? "output" : ""}`, transform: `translate(${n.x},${n.y})` }, svg);
    el("rect", { width: boxW, height: boxH, rx: 9 }, gEl);
    text(gEl, 10, 18, n.op === "input" ? n.id : n.op, { class: "op" });
    text(gEl, 10, 34, `(${n.shape.join(", ")}) ${n.dtype}`, { class: "shape" });
  }
}

function drawMemory(container, mp, inputNames) {
  const values = mp.values, steps = Math.max(1, ...values.map((v) => v.last_use));
  const buffers = [...new Set(values.map((v) => v.buffer))];
  const width = chartWidth(container), rowH = 30, lw = 96, h = buffers.length * rowH + 30;
  const svg = el("svg", { viewBox: `0 0 ${width} ${h}`, role: "img", "aria-label": "Buffer liveness" }, container);
  const x = (s) => lw + (s / (steps + 1)) * (width - lw - 10);
  for (let s = 0; s <= steps; s += 1) { el("line", { class: "gridline", x1: x(s), x2: x(s), y1: 0, y2: h - 22 }, svg); text(svg, x(s) + (x(1) - x(0)) / 2, h - 6, `step ${s}`, { "text-anchor": "middle" }); }
  const tip = tooltip(container);
  buffers.forEach((b, bi) => {
    const y = bi * rowH + 4;
    text(svg, lw - 8, y + 16, b, { "text-anchor": "end", class: "strong" });
    for (const v of values.filter((v) => v.buffer === b)) {
      const isInput = inputNames.has(v.name);   // caller inputs are never reused
      const half = (x(1) - x(0)) / 2, x0 = x(v.first_use) + half - 4, x1 = x(v.last_use) + half + 4;
      el("rect", { x: x0, y, width: Math.max(4, x1 - x0), height: rowH - 8, rx: 4, fill: isInput ? "var(--text-muted)" : "var(--series-1)", opacity: isInput ? 0.45 : 0.85 }, svg);
      if (x1 - x0 > v.name.length * 7 + 14) text(svg, x0 + 6, y + 15, v.name, { fill: "#fff", "font-size": 11.5, "font-weight": 600 });
      const hit = el("rect", { x: x0, y, width: Math.max(4, x1 - x0), height: rowH - 8, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => tip.show(`<b>${esc(v.name)}</b> in buffer <b>${esc(b)}</b><br><span class="k">size</span> ${bytes(v.bytes)}<br><span class="k">live</span> step ${v.first_use}–${v.last_use}`, e));
      hit.addEventListener("pointerleave", () => tip.hide());
    }
  });
}

// ---------- benchmarks ----------
$("#env-line").textContent = `${bench.environment.cpu}, ${bench.environment.python}, NumPy ${bench.environment.numpy}`;
table($("#rounds-table"), [
  { label: "round", num: true, value: (r) => r.round }, { label: "setup", value: (r) => r.machine },
  ...["T0", "T1", "T2", "T3"].map((t) => ({ label: t, html: (r) => (r[t] ? `<span class="verdict ${r[t]}">${r[t]}</span>` : `<span class="muted">—</span>`) })),
], bench.rounds);

function ratioChart(container, cells) {
  container.innerHTML = "";
  const width = chartWidth(container), compact = width < 560, lw = compact ? 140 : 210, rowH = 30, h = cells.length * rowH + 34;
  const svg = el("svg", { viewBox: `0 0 ${width} ${h}`, role: "img", "aria-label": "Time ratio to NumPy" }, container);
  const lo = 0.75, hi = 1.25, x = (v) => lw + ((v - lo) / (hi - lo)) * (width - lw - 50);
  for (const t of [0.8, 0.9, 1.0, 1.1, 1.2]) { el("line", { class: t === 1.1 ? "ref" : "gridline", x1: x(t), x2: x(t), y1: 0, y2: h - 26 }, svg); text(svg, x(t), h - 10, `${t.toFixed(2)}×`, { "text-anchor": "middle" }); }
  const tip = tooltip(container);
  cells.forEach((c, i) => {
    const cy = i * rowH + 16;
    text(svg, lw - 10, cy + 4, `${c.workload.replaceAll("_", " ")} · ${c.size.replace("model_large", "large model").replace("model", "model")}`, { "text-anchor": "end", class: "strong" });
    el("line", { x1: x(c.min_ratio), x2: x(c.max_ratio), y1: cy, y2: cy, stroke: "var(--series-1)", "stroke-width": 4, "stroke-linecap": "round", opacity: 0.35 }, svg);
    el("circle", { cx: x(c.median_ratio), cy, r: 6, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
    text(svg, x(Math.max(c.max_ratio, c.median_ratio)) + 10, cy + 4, `${c.median_ratio.toFixed(3)}×`);
    const hit = el("rect", { x: 0, y: cy - rowH / 2, width, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => tip.show(`<b>${esc(c.workload)} · ${esc(c.size)}</b><br><span class="k">median</span> ${c.median_ratio.toFixed(3)}×<br><span class="k">range</span> ${c.min_ratio.toFixed(3)}–${c.max_ratio.toFixed(3)}×<br><span class="k">runs ≤ 1.10</span> ${c.runs_within} of 10`, e));
    hit.addEventListener("pointerleave", () => tip.hide());
  });
}
const drawRatio = (key) => {
  const cells = bench[key];
  responsive($("#chart-ratio"), () => ratioChart($("#chart-ratio"), cells));
  table($("#table-ratio"), [{ label: "workload", value: (c) => c.workload }, { label: "size", value: (c) => c.size }, { label: "median", num: true, value: (c) => c.median_ratio.toFixed(3) }, { label: "min", num: true, value: (c) => c.min_ratio.toFixed(3) }, { label: "max", num: true, value: (c) => c.max_ratio.toFixed(3) }, { label: "runs ≤ 1.10", num: true, value: (c) => `${c.runs_within}/10` }], cells);
};
$("#seg-round").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; $("#seg-round").querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b))); drawRatio(b.dataset.round); });
drawRatio("round4");

function memoryChart(container, cells) {
  container.innerHTML = "";
  const width = chartWidth(container), lw = 118, groupH = 62, h = cells.length * groupH + 28;
  const svg = el("svg", { viewBox: `0 0 ${width} ${h}`, role: "img", "aria-label": "Estimated and measured memory" }, container);
  const max = Math.max(...cells.flatMap((c) => [c.estimated, c.graphspace_max, c.numpy_min]));
  const x = (v) => lw + (v / max) * (width - lw - 70);
  const tip = tooltip(container);
  cells.forEach((c, i) => {
    const y = i * groupH + 4;
    text(svg, lw - 10, y + 22, c.workload.replaceAll("_", " "), { "text-anchor": "end", class: "strong" });
    text(svg, lw - 10, y + 37, c.size === "model" ? "model" : "large model", { "text-anchor": "end" });
    [["estimated", "var(--series-3)"], ["graphspace_max", "var(--series-1)"], ["numpy_min", "var(--series-2)"]].forEach(([k, color], j) => {
      const yy = y + j * 17, w = Math.max(2, x(c[k]) - lw);
      el("path", { d: `M${lw},${yy} H${lw + w - 3} Q${lw + w},${yy} ${lw + w},${yy + 3} V${yy + 10} Q${lw + w},${yy + 13} ${lw + w - 3},${yy + 13} H${lw} Z`, fill: color }, svg);
      text(svg, lw + w + 6, yy + 11, bytes(c[k]), { "font-size": 11 });
    });
    const hit = el("rect", { x: 0, y, width, height: groupH - 8, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (e) => tip.show(`<b>${esc(c.workload)} · ${esc(c.size)}</b><br><span class="k">estimated</span> ${bytes(c.estimated)}<br><span class="k">graphspace (max of 10)</span> ${bytes(c.graphspace_max)}<br><span class="k">NumPy (min of 10)</span> ${bytes(c.numpy_min)}`, e));
    hit.addEventListener("pointerleave", () => tip.hide());
  });
}
responsive($("#chart-memory"), () => memoryChart($("#chart-memory"), bench.round4_memory));
table($("#table-memory"), [{ label: "workload", value: (c) => c.workload }, { label: "size", value: (c) => c.size }, { label: "estimated", num: true, value: (c) => bytes(c.estimated) }, { label: "graphspace max", num: true, value: (c) => bytes(c.graphspace_max) }, { label: "NumPy min", num: true, value: (c) => bytes(c.numpy_min) }], bench.round4_memory);

function calibrationChart(container, data) {
  container.innerHTML = "";
  const pts = data.points, width = chartWidth(container), h = Math.min(320, Math.max(240, width * 0.62)), m = { t: 10, r: 14, b: 40, l: 56 };
  const svg = el("svg", { viewBox: `0 0 ${width} ${h}`, role: "img", "aria-label": "Measured overhead against the allowance" }, container);
  const maxX = Math.max(...pts.map((p) => p.allowance)) * 1.05, maxY = Math.max(...pts.map((p) => Math.max(p.cold, p.allowance))) * 1.05;
  const x = (v) => m.l + (v / maxX) * (width - m.l - m.r), y = (v) => h - m.b - (v / maxY) * (h - m.t - m.b);
  for (let i = 0; i <= 4; i += 1) { const v = (maxY * i) / 4; el("line", { class: "gridline", x1: m.l, x2: width - m.r, y1: y(v), y2: y(v) }, svg); text(svg, m.l - 6, y(v) + 4, bytes(Math.round(v)), { "text-anchor": "end", "font-size": 11 }); }
  for (let i = 0; i <= 3; i += 1) { const v = (maxX * i) / 3; text(svg, x(v), h - 22, bytes(Math.round(v)), { "text-anchor": "middle", "font-size": 11 }); }
  text(svg, (m.l + width - m.r) / 2, h - 4, "allowance for the graph (inputs, nodes)", { "text-anchor": "middle" });
  el("line", { class: "ref", x1: x(0), y1: y(0), x2: x(Math.min(maxX, maxY)), y2: y(Math.min(maxX, maxY)) }, svg);
  for (const [k, color] of [["warm", "var(--series-1)"], ["cold", "var(--series-2)"]]) for (const p of pts) el("circle", { cx: x(p.allowance), cy: y(Math.max(0, p[k])), r: 3, fill: color, opacity: 0.7 }, svg);
}
$("#scatter-py").textContent = bench.calibration.scatter.python.replace("CPython ", "");
responsive($("#chart-calibration"), () => calibrationChart($("#chart-calibration"), bench.calibration.scatter));
{
  const pts = bench.calibration.scatter.points, negCold = pts.filter((p) => p.cold < 0).length, negWarm = pts.filter((p) => p.warm < 0).length;
  const overCold = pts.filter((p) => p.cold > p.allowance).length;
  $("#calibration-note").textContent = `${negCold} cold and ${negWarm} warm measurements were below zero (memory was freed during the call) and are drawn at 0. Cold calls above the allowance with these constants: ${overCold} of ${pts.length}. Held-out coverage by CPython version:`;
}
$("#coverage").innerHTML = bench.calibration.versions.map((v) => `<div><strong>${v.covered}/${v.graphs}</strong>${esc(v.python.replace("CPython ", "py"))}</div>`).join("");
