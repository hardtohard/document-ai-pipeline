const form = document.getElementById("uploadForm");
const input = document.getElementById("imageInput");
const statusEl = document.getElementById("status");
const preview = document.getElementById("preview");
const emptyPreview = document.getElementById("emptyPreview");
const zoomLevel = document.getElementById("zoomLevel");
const summaryGrid = document.getElementById("summaryGrid");
const taskBadge = document.getElementById("taskBadge");
const allDataView = document.getElementById("allDataView");
const tablesView = document.getElementById("tablesView");
const apiHeaderBody = document.getElementById("apiHeaderBody");
const apiLinesView = document.getElementById("apiLinesView");
const jsonView = document.getElementById("jsonView");
const downloadJson = document.getElementById("downloadJson");
const fieldSearch = document.getElementById("fieldSearch");
const fieldCount = document.getElementById("fieldCount");
const customPrompt = document.getElementById("customPrompt");

let currentResult = null;
let uploadTimer = null;
let uploadStartedAt = 0;
let imageZoom = 1;
let imageRotation = 0;

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

const promptTemplates = {
  delivery: "请重点提取：单据编号、单据日期、发货方、收货方、客户名称、物料/货物名称、规格型号、单位、数量、批次号、订单号、备注。表格必须保留原始列名和每一行数据。",
  table: "请重点提取图片里的所有表格。每个表格输出 table_name、columns 和 rows。列名使用图片上的原始列名，空值输出 null，不要合并不同行。",
  acceptance: "请重点提取：合同编号、甲方、乙方、货物名称、型号/配置、SN、数量、验收结论、验收状态、盖章方、收货人签字、签字日期。没有看到的字段输出 null。",
};

document.querySelectorAll("[data-prompt-template]").forEach((button) => {
  button.addEventListener("click", () => {
    const template = button.dataset.promptTemplate;
    customPrompt.value = template === "clear" ? "" : promptTemplates[template] || "";
    customPrompt.focus();
  });
});

document.querySelectorAll("[data-image-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.imageAction;
    if (action === "zoom-in") imageZoom = Math.min(3, imageZoom + 0.25);
    if (action === "zoom-out") imageZoom = Math.max(0.5, imageZoom - 0.25);
    if (action === "rotate-left") imageRotation -= 90;
    if (action === "rotate-right") imageRotation += 90;
    if (action === "reset") {
      imageZoom = 1;
      imageRotation = 0;
    }
    updatePreviewTransform();
  });
});

fieldSearch.addEventListener("input", () => {
  if (currentResult) renderAllData(currentResult, fieldSearch.value);
});

input.addEventListener("change", () => {
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  emptyPreview.style.display = "none";
  imageZoom = 1;
  imageRotation = 0;
  updatePreviewTransform();
  statusEl.textContent = `已选择：${file.name}`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) {
    statusEl.textContent = "请先选择图片";
    return;
  }

  statusEl.textContent = "模型识别中，请稍候...";
  taskBadge.textContent = "识别中";
  setLoading(true);
  startTimer("上传完成，正在调用多模态模型");

  const body = new FormData();
  const uploadFile = await buildUploadFile(file);
  const extractionMode = document.querySelector("input[name='extraction_mode']:checked")?.value || "full";
  body.append("image", uploadFile);
  body.append("custom_prompt", customPrompt.value.trim());
  body.append("extraction_mode", extractionMode);

  try {
    const response = await fetch("/api/recognize", { method: "POST", body });
    statusEl.textContent = "模型已返回，正在渲染识别结果";
    const payload = await response.json();
    if (!payload.ok) {
      statusEl.textContent = payload.error || "识别失败";
      taskBadge.textContent = "失败";
      renderJson(payload.debug || payload);
      return;
    }

    currentResult = payload.result;
    statusEl.textContent = `识别完成：${payload.task_id}，JSON 已保存，Excel 正在后台更新`;
    taskBadge.textContent = payload.result.document_category || "已识别";
    if (payload.image_url) {
      preview.src = payload.image_url;
      preview.style.display = "block";
      emptyPreview.style.display = "none";
    }
    downloadJson.href = payload.json_url;
    downloadJson.classList.remove("disabled");
    renderResult(payload.result);
  } catch (error) {
    statusEl.textContent = `请求失败：${error.message}`;
    taskBadge.textContent = "异常";
  } finally {
    stopTimer();
    setLoading(false);
  }
});

function setLoading(loading) {
  form.querySelector("button").disabled = loading;
}

function startTimer(prefix) {
  uploadStartedAt = Date.now();
  stopTimer();
  uploadTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - uploadStartedAt) / 1000);
    statusEl.textContent = `${prefix}，已等待 ${seconds} 秒`;
  }, 1000);
}

function stopTimer() {
  if (uploadTimer) {
    window.clearInterval(uploadTimer);
    uploadTimer = null;
  }
}

function updatePreviewTransform() {
  const normalizedRotation = ((imageRotation % 360) + 360) % 360;
  preview.style.width = `${Math.round(imageZoom * 100)}%`;
  preview.style.transform = `rotate(${normalizedRotation}deg)`;
  zoomLevel.textContent = `${Math.round(imageZoom * 100)}%`;
}

function buildUploadFile(file) {
  const normalizedRotation = ((imageRotation % 360) + 360) % 360;
  if (normalizedRotation === 0) {
    return Promise.resolve(file);
  }

  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      const swapSize = normalizedRotation === 90 || normalizedRotation === 270;
      canvas.width = swapSize ? image.naturalHeight : image.naturalWidth;
      canvas.height = swapSize ? image.naturalWidth : image.naturalHeight;

      const context = canvas.getContext("2d");
      context.translate(canvas.width / 2, canvas.height / 2);
      context.rotate((normalizedRotation * Math.PI) / 180);
      context.drawImage(image, -image.naturalWidth / 2, -image.naturalHeight / 2);

      canvas.toBlob((blob) => {
        if (!blob) {
          resolve(file);
          return;
        }
        const rotatedName = withSuffix(file.name, "_rotated", ".jpg");
        resolve(new File([blob], rotatedName, { type: "image/jpeg" }));
      }, "image/jpeg", 0.95);
    };
    image.onerror = () => resolve(file);
    image.src = URL.createObjectURL(file);
  });
}

function withSuffix(name, suffix, fallbackExtension) {
  const dotIndex = name.lastIndexOf(".");
  if (dotIndex <= 0) return `${name}${suffix}${fallbackExtension}`;
  return `${name.slice(0, dotIndex)}${suffix}${name.slice(dotIndex)}`;
}

function renderResult(result) {
  renderSummary(result);
  renderAllData(result, fieldSearch.value);
  renderTables(result.tables || []);
  renderApiView(result);
  renderJson(result);
}

function renderSummary(result) {
  const common = result.common_fields || {};
  const tableCount = Array.isArray(result.tables) ? result.tables.length : 0;
  const rowCount = (result.tables || []).reduce((count, table) => {
    return count + (Array.isArray(table.rows) ? table.rows.length : 0);
  }, 0);
  const rawFieldCount = Object.keys(result.raw_fields || {}).length;
  const rows = [
    ["单据类型", result.document_category || ""],
    ["单据编号", common.document_no || ""],
    ["单据日期", common.document_date || ""],
    ["发货方", common.sender || ""],
    ["收货方", common.receiver || ""],
    ["总数量", common.total_quantity || ""],
    ["字段数量", rawFieldCount],
    ["表格/行数", `${tableCount} / ${rowCount}`],
  ];
  summaryGrid.innerHTML = rows.map(([label, value]) => `
    <div class="summary-item">
      <div class="summary-label">${escapeHtml(label)}</div>
      <div class="summary-value">${escapeHtml(String(value || "-"))}</div>
    </div>
  `).join("");
}

function renderAllData(result, query = "") {
  const normalizedQuery = query.trim().toLowerCase();
  const rows = flatten(result).filter(([key, value]) => {
    if (!normalizedQuery) return true;
    return `${key} ${value}`.toLowerCase().includes(normalizedQuery);
  });
  fieldCount.textContent = `${rows.length} 条`;
  allDataView.innerHTML = `
    <table class="kv-table">
      <thead>
        <tr><th>字段路径</th><th>值</th></tr>
      </thead>
      <tbody>
        ${rows.map(([key, value]) => `
          <tr>
            <td>${escapeHtml(key)}</td>
            <td>${escapeHtml(String(value ?? ""))}</td>
          </tr>
        `).join("") || `<tr><td colspan="2">没有匹配数据</td></tr>`}
      </tbody>
    </table>
  `;
}

function renderTables(tables) {
  if (!tables.length) {
    tablesView.innerHTML = `<div class="empty-state">没有识别到表格明细</div>`;
    return;
  }

  tablesView.innerHTML = tables.map((table, index) => {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    const columns = Array.isArray(table.columns) && table.columns.length
      ? table.columns
      : collectColumns(rows);
    const body = rows.map((row) => `
      <tr>${columns.map((column) => `<td>${escapeHtml(String(row?.[column] ?? ""))}</td>`).join("")}</tr>
    `).join("");
    return `
      <section class="table-block">
        <div class="table-head">
          <h2>${escapeHtml(table.table_name || `表格 ${index + 1}`)}</h2>
          <span>${rows.length} 行</span>
        </div>
        <table class="data-table">
          <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </section>
    `;
  }).join("");
}

function renderApiView(result) {
  const common = result.common_fields || {};
  const headerRows = {
    document_category: result.document_category,
    document_no: common.document_no,
    document_date: common.document_date,
    sender: common.sender,
    receiver: common.receiver,
    total_quantity: common.total_quantity,
  };
  apiHeaderBody.innerHTML = Object.entries(headerRows).map(([key, value]) => `
    <tr><td>${escapeHtml(key)}</td><td>${escapeHtml(String(value ?? ""))}</td></tr>
  `).join("");

  const lines = [];
  (result.tables || []).forEach((table) => {
    (table.rows || []).forEach((row, index) => {
      lines.push({
        table_name: table.table_name || "table",
        row_index: index + 1,
        data: row,
      });
    });
  });
  apiLinesView.innerHTML = lines.length
    ? `<pre>${escapeHtml(JSON.stringify(lines, null, 2))}</pre>`
    : `<div class="empty-state">没有明细行</div>`;
}

function renderJson(result) {
  jsonView.textContent = JSON.stringify(result, null, 2);
}

function collectColumns(rows) {
  const columns = new Set();
  rows.forEach((row) => {
    if (row && typeof row === "object") {
      Object.keys(row).forEach((key) => columns.add(key));
    }
  });
  return Array.from(columns);
}

function flatten(value, prefix = "") {
  if (value === null || value === undefined) {
    return [[prefix, ""]];
  }
  if (typeof value !== "object") {
    return [[prefix, value]];
  }
  if (Array.isArray(value)) {
    if (!value.length) return [[prefix, "[]"]];
    return value.flatMap((child, index) => flatten(child, `${prefix}[${index + 1}]`));
  }
  const entries = Object.entries(value);
  if (!entries.length) return [[prefix, "{}"]];
  return entries.flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return flatten(child, path);
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
