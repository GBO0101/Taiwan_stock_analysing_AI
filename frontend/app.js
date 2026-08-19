"use strict";

const API_BASE = "http://127.0.0.1:8000";

const els = {
  input: document.getElementById("question-input"),
  submit: document.getElementById("submit-btn"),
  showJson: document.getElementById("show-json"),
  error: document.getElementById("error-banner"),
  results: document.getElementById("results"),
  steps: document.getElementById("steps"),
  chartContainer: document.getElementById("chart-container"),
  chartImg: document.getElementById("chart-img"),
  chartMessage: document.getElementById("chart-message"),
};

let lastResult = null;

function showError(message) {
  els.error.textContent = message;
  els.error.classList.remove("hidden");
}

function clearError() {
  els.error.textContent = "";
  els.error.classList.add("hidden");
}

function showChartMessage(message, isError = false) {
  els.chartMessage.textContent = message;
  els.chartMessage.classList.toggle("error", isError);
  els.chartMessage.classList.remove("hidden");
}

function clearChartMessage() {
  els.chartMessage.textContent = "";
  els.chartMessage.classList.remove("error");
  els.chartMessage.classList.add("hidden");
}

function statusClass(status) {
  if (status === "completed") return "badge completed";
  if (status === "skipped") return "badge skipped";
  if (status === "failed") return "badge failed";
  return "badge";
}

function renderStep(step, showJson) {
  const wrap = document.createElement("div");
  wrap.className = "step";

  const header = document.createElement("div");
  header.className = "step-header";

  const title = document.createElement("span");
  title.className = "step-title";
  title.textContent = `Step: ${step.step}`;

  const badge = document.createElement("span");
  badge.className = statusClass(step.status);
  badge.textContent = step.status;

  header.appendChild(title);
  header.appendChild(badge);
  wrap.appendChild(header);

  if (showJson) {
    const pre = document.createElement("pre");
    pre.className = "step-json";
    pre.textContent = JSON.stringify(step.output, null, 2);
    wrap.appendChild(pre);
  }

  return wrap;
}

function findStep(steps, name) {
  return steps.find((s) => s.step === name);
}

async function renderChart(boundary) {
  const stockCodes = boundary && boundary.stock_codes ? boundary.stock_codes : [];
  const showArea = () => els.chartContainer.classList.remove("hidden");
  const hideImg = () => els.chartImg.classList.add("hidden");
  const showImg = () => els.chartImg.classList.remove("hidden");

  if (!stockCodes.length) {
    showArea();
    hideImg();
    showChartMessage("查詢未解析出股票代碼，無法產生圖表。", true);
    return;
  }
  // The boundary output (validated at Step 1) is the single source of truth
  // for charting; use its chart_data_requirements rather than the classification.
  if (!boundary || !boundary.chart_data_requirements) {
    els.chartContainer.classList.add("hidden");
    return;
  }

  const body = {
    stock_codes: stockCodes,
    chart_data_requirements: boundary.chart_data_requirements,
    chart_type: boundary.chart_type || "line",
    date_range: boundary.date_range ? boundary.date_range : null,
  };

  try {
    const resp = await fetch(`${API_BASE}/chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const msg = `圖表渲染失敗: ${err.error || resp.status}`;
      showError(msg);
      showArea();
      hideImg();
      showChartMessage(msg, true);
      return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    els.chartImg.src = url;
    showArea();
    showImg();
    clearChartMessage();
  } catch (e) {
    const msg = `圖表請求失敗: ${e.message}`;
    showError(msg);
    showArea();
    hideImg();
    showChartMessage(msg, true);
  }
}

function renderSteps(data, showJson) {
  els.steps.innerHTML = "";
  (data.steps || []).forEach((step) => {
    els.steps.appendChild(renderStep(step, showJson));
  });
  els.results.classList.remove("hidden");
}

async function submitQuestion() {
  const question = els.input.value.trim();
  if (!question) return;

  clearError();
  els.results.classList.add("hidden");
  els.chartContainer.classList.add("hidden");
  els.submit.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      showError(`後端錯誤: ${err.error || resp.status}`);
      return;
    }

    const data = await resp.json();
    lastResult = data;
    const showJson = els.showJson.checked;
    renderSteps(data, showJson);

    const boundary = findStep(data.steps, "boundary");
    if (boundary) {
      await renderChart(boundary.output);
    }
  } catch (e) {
    showError(`請求失敗: ${e.message}`);
  } finally {
    els.submit.disabled = false;
  }
}

els.submit.addEventListener("click", submitQuestion);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitQuestion();
});
els.showJson.addEventListener("change", () => {
  if (!lastResult) return;
  renderSteps(lastResult, els.showJson.checked);
});
