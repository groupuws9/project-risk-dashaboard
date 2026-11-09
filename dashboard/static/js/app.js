// Dashboard logic: tabs, dynamic forms, client-side validation, resilient API
// calls (timeout + retry + backoff), prediction gauge, SHAP charts, explorer,
// insights, what-if simulator.  (Spec 3 §3.6 — runtime validation on the client)
let META = {features: [], labels: ["Low", "Medium", "High"]};

// Sector is presented as its raw WERS `tsector3` code with a neutral label
// ("Sector 1" / "Sector 2" / "Sector 3") rather than any real-world name, so
// the extract's labels are not surfaced in the UI.  Blank or non-numeric
// values render as "Unknown" (the backend already emits "Unknown" for a blank
// sector group), keeping the chart axis unambiguous.
function sectorLabel(value) {
  if (value === "" || value === null || value === undefined) return "Unknown";
  const n = Number(value);
  return Number.isNaN(n) ? String(value) : ("Sector " + n);
}

// [name, label, {req, min, max, type}]  — ranges mirror dashboard/validation.py
const FIELD_DEFS = [
  ["org_size_category_3", "Org size (3-cat: 1 small - 3 large)", {req: 1, min: 1, max: 3, group: "org"}],
  ["sector", "Sector code (1, 2, 3)", {req: 1, min: 1, max: 10, group: "org"}],
  ["sector_detailed", "Detailed sector code", {min: 0, max: 99, group: "org"}],
  ["multinational", "Multinational (0/1/2)", {req: 1, min: 0, max: 2, group: "org"}],
  ["org_age_years", "Org age band", {min: 0, max: 99, group: "org"}],
  ["trading_status", "Trading status", {min: 0, max: 99, group: "org"}],
  ["union_coverage_pct", "Union coverage % (0-100)", {req: 1, min: 0, max: 100, pct: 1, group: "hr"}],
  ["collective_bargaining", "Collective bargaining code", {req: 1, min: 0, max: 99, group: "hr"}],
  ["formal_hr_strategy", "Formal HR strategy (1 yes, 2 no)", {min: 0, max: 99, group: "hr"}],
  ["performance_appraisal_pct", "Appraisal coverage code", {req: 1, min: 1, max: 99, group: "hr"}],
  ["training_budget_exists", "Training budget (1 yes, 2 no)", {min: 0, max: 99, group: "hr"}],
  ["job_security_policy", "Job-security policy code", {min: 0, max: 99, group: "hr"}],
  ["jcc_exists", "Joint consultative committee (1 yes, 2 no)", {req: 1, min: 0, max: 99, group: "hr"}],
  ["information_sharing_score", "Info-sharing score", {req: 1, min: 1, max: 99, group: "hr"}],
  ["consultation_depth", "Consultation depth code", {min: 0, max: 99, group: "hr"}],
  ["employee_voice_channels", "Voice channels code", {min: 0, max: 99, group: "hr"}],
  ["grievance_procedure_formal", "Formal grievance proc (1 yes, 2 no)", {min: 0, max: 99, group: "hr"}],
  ["pct_part_time", "% part-time (0-100)", {req: 1, min: 0, max: 100, pct: 1, group: "labor"}],
  ["pct_female", "% female (0-100)", {min: 0, max: 100, pct: 1, group: "labor"}],
  ["pct_temporary", "% temporary code", {req: 1, min: 0, max: 99, group: "labor"}]
];
const EXAMPLE = {
  org_size_category_3: "2",
  sector: "1",
  sector_detailed: "6",
  multinational: "2",
  org_age_years: "4",
  trading_status: "5",
  union_coverage_pct: "30",
  collective_bargaining: "2",
  formal_hr_strategy: "1",
  performance_appraisal_pct: "3",
  training_budget_exists: "1",
  job_security_policy: "1",
  jcc_exists: "2",
  information_sharing_score: "3",
  consultation_depth: "2",
  employee_voice_channels: "3",
  grievance_procedure_formal: "1",
  pct_part_time: "20",
  pct_female: "45",
  pct_temporary: "5"
};

// ---------------------------------------------------------------------------
// Client-side validation (spec 3.6.2) — required / type / range / consistency
// ---------------------------------------------------------------------------
function fieldErrors(formId) {
  const errs = {};
  document.querySelectorAll("#" + formId + " input").forEach(i => {
    const def = (FIELD_DEFS.find(f => f[0] === i.name) || [])[2] || {};
    i.classList.remove("bad"); delErr(i);
    const v = i.value.trim();
    if (def.req && v === "") {
      errs[i.name] = "Required field"; return;
    }
    if (v === "") return;                       // optionals may stay empty
    const n = Number(v);
    if (!Number.isFinite(n)) { errs[i.name] = "Must be a number"; return; }
    if (def.min !== undefined && n < def.min) {
      errs[i.name] = `Must be ≥ ${def.min}`; return;
    }
    if (def.max !== undefined && n > def.max) {
      errs[i.name] = `Must be ≤ ${def.max}`; return;
    }
    if (def.pct && n > 100) errs[i.name] = "Percentage cannot exceed 100";
  });
  return errs;
}
function showErrors(formId, errs) {
  Object.entries(errs).forEach(([name, msg]) => {
    const i = document.querySelector(`#${formId} input[name="${name}"]`);
    if (i) { i.classList.add("bad"); addErr(i, msg); }
  });
}
function addErr(input, msg) {
  delErr(input);
  const e = document.createElement("span");
  e.className = "field-err"; e.textContent = msg;
  input.insertAdjacentElement("afterend", e);
}
function delErr(input) {
  const e = input.nextElementSibling;
  if (e && e.classList.contains("field-err")) e.remove();
}
function toast(msg, kind) {
  const t = document.createElement("div");
  t.className = "toast " + (kind || "error");
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 5000);
}
// ---------------------------------------------------------------------------
// Resilient API client (spec 3.6.4): 10s timeout, 2 retries with exponential
// backoff on network failures / 5xx / 429; never retries 4xx validation errors.
// ---------------------------------------------------------------------------
async function apiFetch(url, opts = {}, {timeout = 10000, retries = 2} = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const r = await fetch(url, Object.assign({}, opts,
        {signal: ctrl.signal}));
      clearTimeout(timer);
      if (r.status >= 500 || r.status === 429) {          // retryable
        lastErr = {status: r.status, body: await r.json().catch(() => ({}))};
        if (attempt < retries) {
          await new Promise(res => setTimeout(res, 300 * Math.pow(2, attempt)));
          continue;
        }
        throw lastErr;
      }
      const body = await r.json().catch(() => null);
      if (!r.ok) throw {status: r.status, body: body || {}};
      if (body === null) {
        throw {status: r.status, body: {error: "Invalid JSON response from server"}};
      }
      return body;
    } catch (e) {
      clearTimeout(timer);
      if (e && e.status && e.status < 500 && e.status !== 429) throw e; // 4xx
      lastErr = e && e.body ? e : {status: 0, body: {
        error: e && e.name === "AbortError"
          ? "Request timed out (10s). The server may be busy."
          : "Network error — check your connection.", code: "NETWORK"}};
      if (attempt < retries) {
        await new Promise(res => setTimeout(res, 300 * Math.pow(2, attempt)));
      }
    }
  }
  throw lastErr;
}
function setLoading(btn, on, label) {
  if (on) { btn.dataset.label = btn.textContent; btn.textContent = "Working…"; btn.disabled = true; }
  else { btn.textContent = btn.dataset.label || label || btn.textContent; btn.disabled = false; }
}
// Map structured API errors to inline field errors + toast (spec 3.6.3 table)
function handleApiError(err, formId, outEl) {
  const b = (err && err.body) || {};
  // what-if errors carry a "block" (baseline/scenario) — highlight there
  if (!formId && b.block) formId = b.block + "-form";
  let msg = b.error || "Unexpected error";
  if (b.code === "RATE_LIMITED") msg = "Too many requests — wait a moment and retry.";
  if (b.code === "NETWORK") msg = b.error;
  if (b.code === "INTERNAL_ERROR" || err.status === 503)
    msg = "Service unavailable. Please try again shortly.";
  if (b.code === "MODEL_ERROR")
    msg = "Unable to generate prediction for this input.";
  if (formId && b.field) {
    const i = document.querySelector(`#${formId} input[name="${b.field}"]`);
    if (i) { i.classList.add("bad"); addErr(i, msg); }
  }
  toast((b.code ? `[${b.code}] ` : "") + msg, "error");
  if (outEl) outEl.innerHTML = `<div class="err-box"><b>Error (${b.code || err.status}):</b> ${msg}</div>`;
}
// ---------------------------------------------------------------------------
// Tabs, forms, gauge
// ---------------------------------------------------------------------------
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  const tab = document.getElementById("tab-" + b.dataset.tab);
  if (tab) tab.classList.add("active");
});

function buildForm(el, values) {
  values = values || {};
  const f = document.getElementById(el); f.innerHTML = "";
  const groups = {};
  FIELD_DEFS.forEach(([name, label, def]) => {
    const groupName = def.group || "default";
    if (!groups[groupName]) groups[groupName] = [];
    groups[groupName].push([name, label, def]);
  });
  const groupLabels = {
    org: "Organizational Characteristics",
    hr: "HR Management Practices",
    labor: "Labor Market Factors",
    default: "Other"
  };
  Object.entries(groups).forEach(([groupName, fields]) => {
    if (fields.length > 0) {
      const groupDiv = document.createElement("div");
      groupDiv.className = "form-group";
      if (groupName !== "default") {
        const groupTitle = document.createElement("h4");
        groupTitle.textContent = groupLabels[groupName] || groupName;
        groupTitle.style.marginTop = "16px";
        groupTitle.style.marginBottom = "8px";
        groupTitle.style.color = "#cbd5e1";
        groupTitle.style.fontSize = "0.95rem";
        groupDiv.appendChild(groupTitle);
      }
      fields.forEach(([name, label, def]) => {
        const l = document.createElement("label"); l.textContent = label + (def.req ? " *" : "");
        const i = document.createElement("input");
        i.name = name;
        i.type = "number";                       // HTML5 type check (spec 3.6.2)
        if (def.min !== undefined) i.min = def.min;
        if (def.max !== undefined) i.max = def.max;
        if (def.req) i.required = true;
        i.inputMode = "numeric";
        i.value = values[name] || "";
        i.placeholder = name;
        i.addEventListener("input", () => { i.classList.remove("bad"); delErr(i); });
        l.appendChild(i); groupDiv.appendChild(l);
      });
      f.appendChild(groupDiv);
    }
  });
}
function readForm(el) {
  const o = {};
  document.querySelectorAll("#" + el + " input").forEach(i => {
    const v = i.value.trim();
    if (v !== "") o[i.name] = v; // Only send non-empty values
  });
  return o;
}

function ensureRequiredFields(payload) {
  // Ensure all required fields are present in the payload
  const required = FIELD_DEFS.filter(f => f[2].req).map(f => f[0]);
  required.forEach(field => {
    if (!(field in payload) || payload[field] === "" || payload[field] === undefined) {
      // Check if field is in EXAMPLE
      if (EXAMPLE[field]) {
        payload[field] = EXAMPLE[field];
      }
    }
  });
  return payload;
}
function validateOrExplain(formId, outEl) {
  const errs = fieldErrors(formId);
  document.querySelectorAll(`#${formId} input`).forEach(i => {
    i.classList.remove("bad"); delErr(i);
  });
  if (Object.keys(errs).length) {
    showErrors(formId, errs);
    const n = Object.keys(errs).length;
    toast(`${n} field${n > 1 ? "s" : ""} need attention before submitting.`, "error");
    if (outEl) outEl.innerHTML = `<div class="err-box">Fix the highlighted field${n > 1 ? "s" : ""} and try again (* = required).</div>`;
    return false;
  }
  return true;
}
function gauge(probs, labels) {
  const lvls = labels || ["Low", "Medium", "High"];
  const p = probs || {};
  const idx = lvls.indexOf(Object.keys(p).reduce(
    (a, b) => (p[a] || 0) >= (p[b] || 0) ? a : b, lvls[0]));
  Plotly.newPlot("predict-gauge", [{
    type: "indicator", mode: "gauge+number+delta",
    value: 100 * Math.max(...lvls.map(l => p[l] || 0)),
    number: {suffix: "%"},
    title: {text: "Max class probability"},
    gauge: {
      axis: {range: [0, 100]},
      bar: {color: ["#4ade80", "#facc15", "#f87171"][Math.max(idx, 0)]},
      steps: [
        {range: [0, 33], color: "rgba(74,222,128,.25)"},
        {range: [33, 66], color: "rgba(250,204,21,.25)"},
        {range: [66, 100], color: "rgba(248,113,113,.25)"}]
    }
  }], {margin: {t: 40, b: 10}});
}
function barChart(el, labels, values, title, color) {
  Plotly.newPlot(el, [{type: "bar", x: labels, y: values,
    marker: {color: color || ["#4ade80", "#facc15", "#f87171"]}}],
    {title: title, margin: {t: 40}, paper_bgcolor: "rgba(0,0,0,0)",
     plot_bgcolor: "rgba(0,0,0,0)", font: {color: "#cbd5e1"}});
}
// ---------------------------------------------------------------------------
// Risk Prediction page
// ---------------------------------------------------------------------------
const ALT_PRESET = Object.assign({}, EXAMPLE,
  {formal_hr_strategy: "1", jcc_exists: "1", performance_appraisal_pct: "5"});

function ensureForms() {
  // (Re)build any form that has no inputs yet — repairs pages where init()
  // has not finished (or failed) so no handler can ever read an empty form.
  if (!document.querySelector("#predict-form input")) buildForm("predict-form");
  if (!document.querySelector("#base-form input")) buildForm("base-form", EXAMPLE);
  if (!document.querySelector("#alt-form input")) buildForm("alt-form", ALT_PRESET);
}

async function init() {
  // Build ALL forms FIRST — synchronous and independent of metadata, so a
  // click can never beat form construction (empty-form race) or send {}.
  ensureForms();
  try { const r = await fetch("/api/metadata"); if (r.ok) META = await r.json(); } catch (e) {}
  const fs = document.getElementById("f-sector");
  ["1", "2", "3"].forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = sectorLabel(v); fs.appendChild(o); });
  const fz = document.getElementById("f-size");
  ["1", "2", "3"].forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = "Size " + v; fz.appendChild(o); });
  // Auto-load insights on page load
  loadInsights();
}
document.getElementById("predict-btn").onclick = async (ev) => {
  const out = document.getElementById("predict-out");
  ensureForms();
  if (!validateOrExplain("predict-form", out)) return;
  let payload = readForm("predict-form");
  payload = ensureRequiredFields(payload);
  const btn = ev.currentTarget; setLoading(btn, true);
  try {
    const j = await apiFetch("/api/predict", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    out.innerHTML = "<h3>Predicted: <span class='risk-" + j.risk_level + "'>" +
      j.risk_level + "</span> (" + (j.probability * 100).toFixed(1) + "%)</h3>" +
      "<p class='hint'>Probabilistic estimate from historical WERS data — use professional judgement.</p>";
    const lv = META.labels || ["Low", "Medium", "High"];
    gauge(j.probabilities, lv);
    barChart("predict-probs", lv, lv.map(l => j.probabilities[l] || 0),
      "Class probabilities");
    renderLocalExplanation(j.local_explanation);
  } catch (e) { handleApiError(e, "predict-form", out); }
  finally { setLoading(btn, false); }
};
document.getElementById("fill-btn").onclick = () => buildForm("predict-form", EXAMPLE);

// Local SHAP explanation for the last prediction (spec 3.4 force plot)
function renderLocalExplanation(loc) {
  const el = document.getElementById("predict-shap");
  if (!loc || !loc.contributions || !loc.contributions.length) {
    el.innerHTML = "<div class='card'><p class='hint'>Local SHAP " +
      "explanation unavailable for this prediction.</p></div>";
    return;
  }
    const c = loc.contributions.slice(0, 12).reverse();
    Plotly.newPlot("predict-shap", [{
      type: "bar", orientation: "h",
      x: c.map(d => d.shap), y: c.map(d => d.feature),
      marker: {color: c.map(d => d.shap >= 0 ? "#f87171" : "#4ade80")},
      customdata: c.map(d => d.value),
      hovertemplate: "<b>%{y}</b><br>input: %{customdata}<br>SHAP: %{x:.4f}<extra></extra>"}],
      {title: "Why? Local SHAP contributions for class “" + loc.class_explained +
        "” (red pushes that class up, green down)",
       margin: {l: 190, t: 40}, paper_bgcolor: "rgba(0,0,0,0)",
       plot_bgcolor: "rgba(0,0,0,0)", font: {color: "#cbd5e1"},
       xaxis: {title: "SHAP value (log-odds)", gridcolor: "#334155"}},
      {responsive: true, displayModeBar: false});
}
// ---------------------------------------------------------------------------
// Data Explorer (filters + CSV export) and Model Insights (incl. SHAP)
// ---------------------------------------------------------------------------
document.getElementById("upload-btn").onclick = async (ev) => {
  const fileInput = document.getElementById("data-upload");
  const statusEl = document.getElementById("upload-status");
  const btn = ev.currentTarget;

  if (!fileInput.files || fileInput.files.length === 0) {
    statusEl.innerHTML = "<div class='err-box'>Please select a CSV file to upload.</div>";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  setLoading(btn, true);
  statusEl.innerHTML = "<div class='card'>Uploading and processing data...</div>";

  try {
    const response = await fetch("/api/data/upload", {
      method: "POST",
      body: formData
    });

    const result = await response.json();

    if (response.ok) {
      statusEl.innerHTML = "<div class='card' style='background:#064e3b;border-color:#4ade80'><p>✓ Data uploaded successfully!</p><p>Records: " + result.n + "</p></div>";
      toast("Data uploaded successfully", "success");
      // Refresh the explorer
      document.getElementById("explore-btn").click();
    } else {
      statusEl.innerHTML = "<div class='err-box'>Upload failed: " + (result.error || "Unknown error") + "</div>";
      toast("Upload failed", "error");
    }
  } catch (e) {
    statusEl.innerHTML = "<div class='err-box'>Upload error: " + e.message + "</div>";
    toast("Upload error", "error");
  } finally {
    setLoading(btn, false);
  }
};

document.getElementById("load-sample-btn").onclick = async (ev) => {
  const statusEl = document.getElementById("upload-status");
  const btn = ev.currentTarget;

  setLoading(btn, true);
  statusEl.innerHTML = "<div class='card'>Loading sample data...</div>";

  try {
    const response = await fetch("/api/data/load-sample", { method: "POST" });
    const result = await response.json();

    if (response.ok) {
      statusEl.innerHTML = "<div class='card' style='background:#064e3b;border-color:#4ade80'><p>✓ Sample data loaded successfully!</p><p>Records: " + result.n + "</p></div>";
      toast("Sample data loaded", "success");
      document.getElementById("explore-btn").click();
    } else {
      statusEl.innerHTML = "<div class='err-box'>Failed to load sample data: " + (result.error || "Unknown error") + "</div>";
      toast("Failed to load sample", "error");
    }
  } catch (e) {
    statusEl.innerHTML = "<div class='err-box'>Error: " + e.message + "</div>";
    toast("Error loading sample", "error");
  } finally {
    setLoading(btn, false);
  }
};

document.getElementById("explore-btn").onclick = async (ev) => {
  const btn = ev.currentTarget; setLoading(btn, true);
  const s = document.getElementById("f-sector").value;
  const z = document.getElementById("f-size").value;
  try {
    const j = await apiFetch("/api/data/summary?sector=" + s + "&size=" + z);
    // The backend exposes the descriptive turnover-rate mean (not a model
    // score): turnover_rate_mean / turnover_rate_by_sector.
    const meanTurnover = (j.turnover_rate_mean === null ||
      j.turnover_rate_mean === undefined)
      ? "n/a"
      : Number(j.turnover_rate_mean).toFixed(2);
    document.getElementById("explore-out").innerHTML =
      "<div class='card'><b>n=" + j.n + "</b> &bull; mean turnover rate=" +
      meanTurnover + " &bull; " +
      JSON.stringify(j.risk_distribution) + "</div>";
    const d = j.risk_distribution || {};
    barChart("chart-dist", Object.keys(d), Object.values(d),
      "Risk distribution (filtered)");
    const bs = j.turnover_rate_by_sector || {};
    barChart("chart-sector", Object.keys(bs).map(sectorLabel), Object.values(bs),
      "Mean turnover rate by sector", "#60a5fa");
  } catch (e) { handleApiError(e, null, document.getElementById("explore-out")); }
  finally { setLoading(btn, false); }
};
document.getElementById("export-btn").onclick = () => {
  const s = document.getElementById("f-sector").value;
  const z = document.getElementById("f-size").value;
  // direct download — browser handles Content-Disposition
  window.location.href = "/api/data/export?sector=" + s + "&size=" + z;
};

async function loadInsights() {
  const outEl = document.getElementById("insights-out");
  outEl.innerHTML = "<div class='card'><p>Loading insights...</p></div>";

  try {
    const j = await apiFetch("/api/insights");
    if (!j) {
      throw {status: 0, body: {error: "Invalid response from server"}};
    }
    let html = "";

    // Best model info with formatted metrics
    if (j.best_model) {
      html += "<div class='card'><h3>Best Model</h3>" +
        "<p><b>" + j.best_model + "</b></p>";
      if (j.metrics && Object.keys(j.metrics).length > 0) {
        html += "<h4>Balanced Accuracy by Model</h4>";
        html += "<div style='margin:8px 0;padding-left:20px'>";
        Object.entries(j.metrics).forEach(([model, acc]) => {
          const isBest = model === j.best_model;
          // Models without a held-out score arrive as null → render "n/a".
          const value = (typeof acc === "number" && isFinite(acc))
            ? acc.toFixed(3)
            : "n/a";
          html += "<div style='margin:6px 0;color:#cbd5e1;line-height:1.4" +
            (isBest ? ";color:#4ade80;font-weight:bold" : "") + "'>" +
            "• " + model + ": " + value +
            (isBest ? " ✓" : "") + "</div>";
        });
        html += "</div>";
      }
      html += "</div>";
    } else {
      html += "<div class='card'><p>No model information available. Please train models first with <code>python ml_pipeline/04_model_training.py</code></p></div>";
    }

    // Model comparison table with highlighting
    if (j["model_comparison.csv"] && Array.isArray(j["model_comparison.csv"]) && j["model_comparison.csv"].length > 0) {
      html += "<h3>Model Comparison</h3><div class='card'><table>";
      const cols = Object.keys(j["model_comparison.csv"][0]);
      html += "<thead><tr>";
      cols.forEach(c => html += "<th>" + c + "</th>");
      html += "</tr></thead><tbody>";
      j["model_comparison.csv"].forEach(row => {
        const isBest = row["Model"] === j.best_model;
        html += "<tr" + (isBest ? " style='background:#064e3b'" : "") + ">";
        cols.forEach(c => {
          let val = row[c];
          if (typeof val === 'number') val = val.toFixed(4);
          html += "<td" + (isBest ? " style='font-weight:bold'" : "") + ">" + val + "</td>";
        });
        html += "</tr>";
      });
      html += "</tbody></table></div>";
    }

    // Evaluation summary
    if (j["evaluation_summary.csv"] && Array.isArray(j["evaluation_summary.csv"]) && j["evaluation_summary.csv"].length > 0) {
      html += "<h3>Evaluation Summary</h3><div class='card'><table>";
      const cols = Object.keys(j["evaluation_summary.csv"][0]);
      html += "<thead><tr>";
      cols.forEach(c => html += "<th>" + c + "</th>");
      html += "</tr></thead><tbody>";
      j["evaluation_summary.csv"].forEach(row => {
        html += "<tr>";
        cols.forEach(c => {
          let val = row[c];
          if (typeof val === 'number') val = val.toFixed(4);
          html += "<td>" + val + "</td>";
        });
        html += "</tr>";
      });
      html += "</tbody></table></div>";
    }

    if (!html) {
      html = "<div class='card'><p>No insights data available. Please train models first with <code>python ml_pipeline/04_model_training.py</code></p></div>";
    }

    outEl.innerHTML = html;

    // Feature importance chart with interactivity
    const fi = j["feature_importance.csv"];
    if (fi && Array.isArray(fi) && fi.length > 0) {
      const rows = fi.slice(0, 20).reverse();
      Plotly.newPlot("chart-importance",
        [{
          type: "bar",
          orientation: "h",
          x: rows.map(x => x.importance),
          y: rows.map(x => x.feature),
          marker: {color: "#60a5fa"},
          hovertemplate: "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
          text: rows.map(x => x.importance.toFixed(4)),
          textposition: "auto"
        }],
        {
          title: {text: "Top 20 Features (Model Importance)", font: {color: "#cbd5e1", size: 16}},
          margin: {l: 220, t: 50, r: 80, b: 40},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {color: "#cbd5e1"},
          xaxis: {title: "Importance Score", gridcolor: "#334155"},
          yaxis: {gridcolor: "#334155"},
          hovermode: "closest"
        },
        {responsive: true, displayModeBar: false}
      );
    }

    // SHAP global importance with interactivity
    const shap = j["shap_importance.csv"];
    if (shap && Array.isArray(shap) && shap.length > 0) {
      const rows = shap.slice(0, 20).reverse();
      Plotly.newPlot("chart-shap",
        [{
          type: "bar",
          orientation: "h",
          x: rows.map(x => x.mean_abs_shap),
          y: rows.map(x => x.feature),
          marker: {color: "#f59e0b"},
          hovertemplate: "<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
          text: rows.map(x => x.mean_abs_shap.toFixed(4)),
          textposition: "auto"
        }],
        {
          title: {text: "Top 20 Features (Mean |SHAP|)", font: {color: "#cbd5e1", size: 16}},
          margin: {l: 220, t: 50, r: 80, b: 40},
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {color: "#cbd5e1"},
          xaxis: {title: "Mean |SHAP| Value", gridcolor: "#334155"},
          yaxis: {gridcolor: "#334155"},
          hovermode: "closest"
        },
        {responsive: true, displayModeBar: false}
      );
    }

    toast("Insights loaded successfully", "success");
  } catch (e) {
    const msg = (e.body && e.body.error) || (e.message || e);
    console.error("Insights load error:", e);
    toast("Insights unavailable: " + msg, "error");
    outEl.innerHTML =
      "<div class='err-box'>Unable to load insights. Please train models first with <code>python ml_pipeline/04_model_training.py</code></div>";
  }
}
document.getElementById("insights-refresh-btn").onclick = async (ev) => {
  const btn = ev.currentTarget;
  setLoading(btn, true);
  await loadInsights();
  setLoading(btn, false);
};
// ---------------------------------------------------------------------------
// What-If Simulator — client-side validation on BOTH forms, delta chart,
// changed-fields diff (auditable comparison), causality-aware interpretation
// ---------------------------------------------------------------------------
document.getElementById("whatif-btn").onclick = async (ev) => {
  const out = document.getElementById("whatif-out");
  const btn = ev.currentTarget;
  ensureForms();   // never validate/read a form that was never built
  // spec 3.6.2: validate both forms inline BEFORE any network call —
  // no silent defaults for cleared required fields
  const errsBase = fieldErrors("base-form");
  const errsAlt = fieldErrors("alt-form");
  document.querySelectorAll("#base-form input, #alt-form input").forEach(i => {
    i.classList.remove("bad"); delErr(i);
  });
  const nErr = Object.keys(errsBase).length + Object.keys(errsAlt).length;
  if (nErr) {
    showErrors("base-form", errsBase);
    showErrors("alt-form", errsAlt);
    toast(`${nErr} field${nErr > 1 ? "s" : ""} need attention before comparing (* = required).`, "error");
    out.innerHTML = "<div class='err-box'>Fix the highlighted fields in both forms and try again (* = required).</div>";
    return;
  }
  const base = readForm("base-form");
  const alt = readForm("alt-form");
  setLoading(btn, true);
  try {
    const j = await apiFetch("/api/whatif", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({baseline: base, scenario: alt})
    });
    renderWhatIf(j, base, alt);
    toast("Scenarios compared", "success");
  } catch (e) { handleApiError(e, null, out); }
  finally { setLoading(btn, false); }
};

function renderWhatIf(j, base, alt) {
  const out = document.getElementById("whatif-out");
  const labels = META.labels || ["Low", "Medium", "High"];
  const bp = j.baseline.probabilities || {};
  const sp = j.scenario.probabilities || {};
  const d = {};
  labels.forEach(l => { d[l] = (sp[l] || 0) - (bp[l] || 0); });
  // scalar risk movement: Medium +1, High +2 weight (Low +0)
  const riskDelta = (d.Medium || 0) + 2 * (d.High || 0);
  const dir = riskDelta > 0.01 ? "worse" : riskDelta < -0.01 ? "better" : "flat";
  const dirStyle = {
    better: ["#4ade80", "▼ Scenario lowers modelled risk"],
    worse: ["#f87171", "▲ Scenario raises modelled risk"],
    flat: ["#facc15", "→ Scenario barely changes modelled risk"]
  }[dir];

  const rows = labels.map(l => {
    const dp = 100 * d[l];
    const good = l === "Low" ? dp >= 0 : dp <= 0;  // risk semantics
    const color = Math.abs(dp) < 0.05 ? "#cbd5e1" : (good ? "#4ade80" : "#f87171");
    return "<tr><td>" + l + "</td><td>" + (100 * (bp[l] || 0)).toFixed(1) +
      "%</td><td>" + (100 * (sp[l] || 0)).toFixed(1) +
      "%</td><td style='color:" + color + ";font-weight:600'>" +
      (dp >= 0 ? "+" : "") + dp.toFixed(1) + " pp</td></tr>";
  }).join("");

  out.innerHTML =
    "<div class='card'><h3>Baseline</h3><p class='risk-" + j.baseline.risk_level + "'>" +
      j.baseline.risk_level + " (" + (j.baseline.probability * 100).toFixed(1) + "%)</p></div>" +
    "<div class='card'><h3>Scenario</h3><p class='risk-" + j.scenario.risk_level + "'>" +
      j.scenario.risk_level + " (" + (j.scenario.probability * 100).toFixed(1) + "%)</p>" +
      "<p style='color:" + dirStyle[0] + "'><b>" + dirStyle[1] + "</b></p></div>" +
    "<div class='card' style='grid-column:1/-1'><h3>Probability shift by class</h3>" +
      "<table><thead><tr><th>Class</th><th>Baseline</th><th>Scenario</th><th>Δ</th></tr></thead>" +
      "<tbody>" + rows + "</tbody></table>" +
      "<div id='whatif-chart' style='height:280px;margin-top:12px'></div>" +
      "<p class='hint'>Note: results are model-learned statistical associations from " +
      "WERS/WIRS (logistic regression, balanced accuracy 0.389 held-out / 0.385 CV) — " +
      "not causal effects. P(High) stays low for most inputs (550 High observations " +
      "in training), so treat High as a triage signal.</p></div>" +
    whatIfDiff(base, alt);

  const colors = {Low: "#4ade80", Medium: "#facc15", High: "#f87171"};
  Plotly.newPlot("whatif-chart", labels.map(l => ({
    name: l, type: "bar",
    x: ["Baseline", "Scenario"],
    y: [100 * (bp[l] || 0), 100 * (sp[l] || 0)],
    marker: {color: colors[l]}
  })), {
    barmode: "group",
    title: {text: "Class probability comparison", font: {color: "#cbd5e1", size: 15}},
    font: {color: "#cbd5e1"},
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    yaxis: {title: "Probability (%)", gridcolor: "#334155"},
    legend: {orientation: "h", y: 1.12},
    margin: {t: 50, b: 40}
  }, {responsive: true, displayModeBar: false});
}

function whatIfDiff(base, alt) {
  const changed = [];
  Object.keys(Object.assign({}, base, alt)).forEach(k => {
    const b = base[k] !== undefined ? String(base[k]) : "(absent)";
    const a = alt[k] !== undefined ? String(alt[k]) : "(absent)";
    if (b !== a) changed.push([k, b, a]);
  });
  if (!changed.length) {
    return "<div class='card' style='grid-column:1/-1'><h3>Changed inputs</h3>" +
      "<p class='hint'>No differences — the scenario is identical to the baseline. " +
      "Edit scenario fields to simulate an intervention.</p></div>";
  }
  return "<div class='card' style='grid-column:1/-1'><h3>Changed inputs (" +
    changed.length + ")</h3><ul class='diff-list'>" +
    changed.map(([k, b, a]) =>
      "<li><code>" + k + "</code>: " + b + " → <b>" + a + "</b></li>").join("") +
    "</ul></div>";
}
// <<CHUNK_F>>

// ---------------------------------------------------------------------------
// Bootstrap: the script loads at the end of <body>, so start init() as soon
// as the document is ready.  Without this call the forms were never built
// (latent bug: base-form/alt-form stayed empty -> MISSING_FIELD on compare).
// ---------------------------------------------------------------------------
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}




