/**
 * SigLIP Multimodal Evaluation Suite - Vanilla JavaScript Client
 * Handles dataset upload, pre-run validation, live SSE stream processing,
 * offline calibration sweeps, confusion matrix heatmaps, canvas charts, and exports.
 */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    apiKey: sessionStorage.getItem("siglip_api_key") || "",
    currentJobId: null,
    jobStatus: "idle",
    eventSource: null,
    rows: [],
    summary: null,
    validation: null,
    activeFilter: "all",
    searchQuery: "",
    sortCol: "row_id",
    sortAsc: true,
    expandedRows: new Set(),
    activeTemp: 1.0,
    blobUrlCache: new Map(),
    effectiveSettings: null,
  };

  const compareState = {
    runA: null,
    runB: null,
    data: null,
    activeFilter: "all",
  };

  // ── DOM Elements ───────────────────────────────────────────────────────────
  const apiKeyInput = document.getElementById("apiKeyInput");
  const saveKeyBtn = document.getElementById("saveKeyBtn");
  const pastRunsSelect = document.getElementById("pastRunsSelect");
  const newEvalBtn = document.getElementById("newEvalBtn");
  const compareNavBtn = document.getElementById("compareNavBtn");

  const uploadSection = document.getElementById("uploadSection");
  const dropzone = document.getElementById("dropzone");
  const zipFileInput = document.getElementById("zipFileInput");
  const currentJobIdBadge = document.getElementById("currentJobIdBadge");
  const quickLoadBtn = document.getElementById("quickLoadBtn");
  const localPathInput = document.getElementById("localPathInput");
  const loadLocalBtn = document.getElementById("loadLocalBtn");

  const validationBox = document.getElementById("validationBox");
  const valTotalRows = document.getElementById("valTotalRows");
  const valValidRows = document.getElementById("valValidRows");
  const valMissingImages = document.getElementById("valMissingImages");
  const valCorruptImages = document.getElementById("valCorruptImages");
  const valUnknownServices = document.getElementById("valUnknownServices");
  const valDuplicateRows = document.getElementById("valDuplicateRows");
  const valEmptyDesc = document.getElementById("valEmptyDesc");
  const valNoneRows = document.getElementById("valNoneRows");
  const valAlerts = document.getElementById("valAlerts");
  const showSettingsBtn = document.getElementById("showSettingsBtn");

  const settingsSection = document.getElementById("settingsSection");
  const presetRerankerOnBtn = document.getElementById("presetRerankerOnBtn");
  const presetRerankerOffBtn = document.getElementById("presetRerankerOffBtn");
  const descWeightInput = document.getElementById("descWeightInput");
  const descWeightVal = document.getElementById("descWeightVal");
  const confThreshInput = document.getElementById("confThreshInput");
  const confThreshVal = document.getElementById("confThreshVal");
  const gapThreshInput = document.getElementById("gapThreshInput");
  const gapThreshVal = document.getElementById("gapThreshVal");
  const floorThreshInput = document.getElementById("floorThreshInput");
  const floorThreshVal = document.getElementById("floorThreshVal");
  const enableRerankerInput = document.getElementById("enableRerankerInput");
  const enableRerankerVal = document.getElementById("enableRerankerVal");
  const resetSettingsBtn = document.getElementById("resetSettingsBtn");
  const startRunBtn = document.getElementById("startRunBtn");
  const stopRunBtn = document.getElementById("stopRunBtn");

  const progressSection = document.getElementById("progressSection");
  const progressBarFill = document.getElementById("progressBarFill");
  const statusText = document.getElementById("statusText");
  const statusSpinner = document.getElementById("statusSpinner");
  const progressFraction = document.getElementById("progressFraction");
  const metaSpeed = document.getElementById("metaSpeed");
  const metaElapsed = document.getElementById("metaElapsed");

  const summarySection = document.getElementById("summarySection");
  const rerankerOffBanner = document.getElementById("rerankerOffBanner");
  const runRecordSnapshot = document.getElementById("runRecordSnapshot");
  const snapshotCatalogHashBadge = document.getElementById("snapshotCatalogHashBadge");
  const snapshotGrid = document.getElementById("snapshotGrid");
  const summaryWarnings = document.getElementById("summaryWarnings");
  const kpiTop1 = document.getElementById("kpiTop1");
  const kpiTop1Sub = document.getElementById("kpiTop1Sub");
  const kpiTop35 = document.getElementById("kpiTop35");
  const kpiPoolRecall = document.getElementById("kpiPoolRecall");
  const kpiAutoPrec = document.getElementById("kpiAutoPrec");
  const kpiAutoPrecSub = document.getElementById("kpiAutoPrecSub");
  const kpiWrongAuto = document.getElementById("kpiWrongAuto");
  const wrongAutoCard = document.getElementById("wrongAutoCard");
  const kpiThroughput = document.getElementById("kpiThroughput");
  const kpiMeanLatency = document.getElementById("kpiMeanLatency");

  const subDescAcc = document.getElementById("subDescAcc");
  const subImageOnlyAcc = document.getElementById("subImageOnlyAcc");
  const subNoneAcc = document.getElementById("subNoneAcc");
  const subAgreeRate = document.getElementById("subAgreeRate");

  const actionAutoCount = document.getElementById("actionAutoCount");
  const actionDraftCount = document.getElementById("actionDraftCount");
  const actionNoActionCount = document.getElementById("actionNoActionCount");
  const actionAutoCoverage = document.getElementById("actionAutoCoverage");

  const latP50 = document.getElementById("latP50");
  const latP95 = document.getElementById("latP95");
  const latMax = document.getElementById("latMax");
  const latSiglipMean = document.getElementById("latSiglipMean");

  const latencyCanvas = document.getElementById("latencyCanvas");
  const confusionsList = document.getElementById("confusionsList");
  const servicesTableBody = document.getElementById("servicesTableBody");
  const signalQualityTableBody = document.getElementById("signalQualityTableBody");
  const matrixContainer = document.getElementById("matrixContainer");
  const thresholdsTableBody = document.getElementById("thresholdsTableBody");

  const tempSelectorBtns = document.getElementById("tempSelectorBtns");
  const reliabilityCanvas = document.getElementById("reliabilityCanvas");
  const calibEceVal = document.getElementById("calibEceVal");
  const calibCovVal = document.getElementById("calibCovVal");
  const calibPrecVal = document.getElementById("calibPrecVal");
  const calibBinsTableBody = document.getElementById("calibBinsTableBody");

  const resultsSection = document.getElementById("resultsSection");
  const resultsTableBody = document.getElementById("resultsTableBody");
  const tableRowCounter = document.getElementById("tableRowCounter");
  const tableSearchInput = document.getElementById("tableSearchInput");

  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");
  const exportMdBtn = document.getElementById("exportMdBtn");
  const copyMdBtn = document.getElementById("copyMdBtn");

  const summaryFindings = document.getElementById("summaryFindings");

  // NONE Ground Truth Panel Elements (3 Levels: Requirement 4)
  const tabNoneBadge = document.getElementById("tabNoneBadge");
  const kpiNoneN = document.getElementById("kpiNoneN");
  const kpiNoneCorrectRate = document.getElementById("kpiNoneCorrectRate");
  const kpiNoneCorrectSub = document.getElementById("kpiNoneCorrectSub");
  const kpiNoneDraftRate = document.getElementById("kpiNoneDraftRate");
  const kpiNoneDraftSub = document.getElementById("kpiNoneDraftSub");
  const kpiNoneWrongAutoCount = document.getElementById("kpiNoneWrongAutoCount");
  const kpiNoneWrongAutoSub = document.getElementById("kpiNoneWrongAutoSub");
  const cardNoneFalseMatch = document.getElementById("cardNoneFalseMatch");
  const noneHistogramCanvas = document.getElementById("noneHistogramCanvas");
  const noneFalseMatchesTable = document.getElementById("noneFalseMatchesTable");
  const noneFalseMatchesTableBody = document.getElementById("noneFalseMatchesTableBody");
  const noneFalseMatchesHeaderCount = document.getElementById("noneFalseMatchesHeaderCount");

  // Compare Runs View Elements (Requirement 3)
  const compareSection = document.getElementById("compareSection");
  const closeCompareBtn = document.getElementById("closeCompareBtn");
  const compareRunASelect = document.getElementById("compareRunASelect");
  const compareRunBSelect = document.getElementById("compareRunBSelect");
  const runComparisonBtn = document.getElementById("runComparisonBtn");
  const compareWarnings = document.getElementById("compareWarnings");
  const compareResultsArea = document.getElementById("compareResultsArea");
  const compareColRunAHeader = document.getElementById("compareColRunAHeader");
  const compareColRunBHeader = document.getElementById("compareColRunBHeader");
  const compareDeltasTableBody = document.getElementById("compareDeltasTableBody");
  const compareFlipFilterPills = document.getElementById("compareFlipFilterPills");
  const compareCountAll = document.getElementById("compareCountAll");
  const compareCountImproved = document.getElementById("compareCountImproved");
  const compareCountRegressed = document.getElementById("compareCountRegressed");
  const compareChangedTableBody = document.getElementById("compareChangedTableBody");

  // Enhanced Modal Elements
  const imageModal = document.getElementById("imageModal");
  const modalImg = document.getElementById("modalImg");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalPrevBtn = document.getElementById("modalPrevBtn");
  const modalNextBtn = document.getElementById("modalNextBtn");
  const modalLoadingSpinner = document.getElementById("modalLoadingSpinner");
  const modalRowIdx = document.getElementById("modalRowIdx");
  const modalFilename = document.getElementById("modalFilename");
  const modalActionBadge = document.getElementById("modalActionBadge");
  const modalResultBadge = document.getElementById("modalResultBadge");
  const modalExpectedSvc = document.getElementById("modalExpectedSvc");
  const modalPredictedSvc = document.getElementById("modalPredictedSvc");
  const modalConfGap = document.getElementById("modalConfGap");
  const modalDescription = document.getElementById("modalDescription");

  // ── Helper: Fetch with Auth ────────────────────────────────────────────────
  async function apiFetch(url, options = {}) {
    options.headers = options.headers || {};
    if (state.apiKey) {
      options.headers["Authorization"] = `Bearer ${state.apiKey}`;
    }
    const res = await fetch(url, options);
    if (res.status === 401) {
      alert("Unauthorized (401). Please check the API key entered in the header.");
    }
    return res;
  }

  // ── Initialization ─────────────────────────────────────────────────────────
  function init() {
    if (state.apiKey) {
      apiKeyInput.value = state.apiKey;
    }

    // Load past runs
    loadPastRuns();

    // Setup input sliders
    bindSlider(descWeightInput, descWeightVal);
    bindSlider(confThreshInput, confThreshVal);
    bindSlider(gapThreshInput, gapThreshVal);
    bindSlider(floorThreshInput, floorThreshVal);

    if (enableRerankerInput) {
      enableRerankerInput.addEventListener("change", updateRerankerToggleDisplay);
    }

    // Preset buttons: Reranker ON / OFF (Requirement 6)
    if (presetRerankerOnBtn) {
      presetRerankerOnBtn.addEventListener("click", () => {
        if (enableRerankerInput) {
          enableRerankerInput.checked = true;
          updateRerankerToggleDisplay();
        }
      });
    }
    if (presetRerankerOffBtn) {
      presetRerankerOffBtn.addEventListener("click", () => {
        if (enableRerankerInput) {
          enableRerankerInput.checked = false;
          updateRerankerToggleDisplay();
        }
      });
    }

    // Compare view buttons & events (Requirement 3)
    if (compareNavBtn) {
      compareNavBtn.addEventListener("click", () => {
        if (!compareSection) return;
        const isHidden = compareSection.style.display === "none";
        compareSection.style.display = isHidden ? "block" : "none";
        if (isHidden) {
          compareSection.scrollIntoView({ behavior: "smooth" });
        }
      });
    }
    if (closeCompareBtn) {
      closeCompareBtn.addEventListener("click", () => {
        if (compareSection) compareSection.style.display = "none";
      });
    }
    if (runComparisonBtn) {
      runComparisonBtn.addEventListener("click", runCompare);
    }
    if (compareFlipFilterPills) {
      compareFlipFilterPills.querySelectorAll(".pill-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          compareFlipFilterPills.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          compareState.activeFilter = btn.dataset.flipFilter || "all";
          renderCompareChangedRows();
        });
      });
    }

    // Check for saved job in session
    const savedJobId = sessionStorage.getItem("siglip_eval_active_job");
    if (savedJobId) {
      loadJob(savedJobId);
    }
  }

  function bindSlider(input, label) {
    input.addEventListener("input", () => {
      label.textContent = parseFloat(input.value).toFixed(2);
    });
  }

  // ── API Key Management ─────────────────────────────────────────────────────
  saveKeyBtn.addEventListener("click", () => {
    state.apiKey = apiKeyInput.value.trim();
    if (state.apiKey) {
      sessionStorage.setItem("siglip_api_key", state.apiKey);
    } else {
      sessionStorage.removeItem("siglip_api_key");
    }
    alert("API Key updated in session.");
    loadPastRuns();
  });

  apiKeyInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      saveKeyBtn.click();
    }
  });

  // ── Past Runs ──────────────────────────────────────────────────────────────
  async function loadPastRuns() {
    try {
      const res = await apiFetch("/v1/eval/runs");
      if (!res.ok) return;
      const data = await res.json();
      const runs = data.runs || [];

      pastRunsSelect.innerHTML = '<option value="">Select past run...</option>';
      if (compareRunASelect && compareRunBSelect) {
        const curA = compareRunASelect.value;
        const curB = compareRunBSelect.value;
        compareRunASelect.innerHTML = '<option value="">Select Baseline Run A...</option>';
        compareRunBSelect.innerHTML = '<option value="">Select Comparison Run B...</option>';

        runs.forEach((r) => {
          const acc = r.top1_accuracy !== undefined && r.top1_accuracy !== null ? `Acc: ${r.top1_accuracy}%` : r.status;
          const label = `${r.job_id} (${r.completed_rows} rows) - ${acc}`;

          const opt = document.createElement("option");
          opt.value = r.job_id;
          opt.textContent = label;
          pastRunsSelect.appendChild(opt);

          const optA = document.createElement("option");
          optA.value = r.job_id;
          optA.textContent = label;
          compareRunASelect.appendChild(optA);

          const optB = document.createElement("option");
          optB.value = r.job_id;
          optB.textContent = label;
          compareRunBSelect.appendChild(optB);
        });

        if (curA) compareRunASelect.value = curA;
        if (curB) compareRunBSelect.value = curB;
      } else {
        runs.forEach((r) => {
          const opt = document.createElement("option");
          opt.value = r.job_id;
          const acc = r.top1_accuracy !== undefined && r.top1_accuracy !== null ? `Acc: ${r.top1_accuracy}%` : r.status;
          opt.textContent = `${r.job_id} (${r.completed_rows} rows) - ${acc}`;
          pastRunsSelect.appendChild(opt);
        });
      }
    } catch (e) {
      console.warn("Could not load past runs:", e);
    }
  }

  pastRunsSelect.addEventListener("change", () => {
    const jId = pastRunsSelect.value;
    if (jId) {
      loadJob(jId);
    }
  });

  // ── Drag & Drop ZIP Upload ─────────────────────────────────────────────────
  // Prevent browser from navigating away if user drops a file outside the dropzone
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  dropzone.addEventListener("click", () => {
    if (state.uploading) return;
    zipFileInput.click();
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (!state.uploading) zipFileInput.click();
    }
  });

  dropzone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = "copy";
    }
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (state.uploading) return;
    const dt = e.dataTransfer;
    if (!dt) return;

    if (dt.files && dt.files.length > 0) {
      uploadZipFile(dt.files[0]);
      return;
    }

    // Handle dragged links/paths from VS Code or text (passes file:/// or path string)
    const text = dt.getData("text/uri-list") || dt.getData("text/plain") || "";
    if (text) {
      const firstLine = text.split("\n")[0].trim();
      if (firstLine.includes("file://") || firstLine.includes(".zip") || firstLine.includes(":\\") || firstLine.includes("/")) {
        loadFromLocalPath(firstLine);
        return;
      }
    }
  });

  zipFileInput.addEventListener("change", () => {
    if (zipFileInput.files && zipFileInput.files.length > 0) {
      const file = zipFileInput.files[0];
      // Reset value so selecting the same file again triggers change event
      zipFileInput.value = "";
      uploadZipFile(file);
    }
  });

  if (quickLoadBtn) {
    quickLoadBtn.addEventListener("click", () => {
      loadFromLocalPath("");
    });
  }

  if (loadLocalBtn) {
    loadLocalBtn.addEventListener("click", () => {
      loadFromLocalPath(localPathInput ? localPathInput.value.trim() : "");
    });
  }

  if (localPathInput) {
    localPathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        loadLocalBtn.click();
      }
    });
  }

  async function loadFromLocalPath(customPath) {
    const path = (customPath !== undefined ? customPath : (localPathInput ? localPathInput.value.trim() : "")).trim();

    state.uploading = true;
    dropzone.innerHTML = `
      <div class="dropzone-icon">⏳</div>
      <div class="dropzone-text">Loading local dataset from host machine...</div>
      <div class="dropzone-sub">Reading: ${escapeHtml(path || "workspace default (eval_dataset.zip)")}</div>
    `;

    try {
      const res = await apiFetch("/v1/eval/load-local", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path || null }),
      });

      if (!res.ok) {
        let errMessage = `Local dataset load failed (HTTP ${res.status})`;
        try {
          const err = await res.json();
          errMessage = err.detail?.message || (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) || errMessage;
        } catch (_) {
          try {
            const raw = await res.text();
            if (raw) errMessage = raw.slice(0, 300);
          } catch (__) {}
        }
        throw new Error(errMessage);
      }

      const data = await res.json();
      state.currentJobId = data.job_id;
      sessionStorage.setItem("siglip_eval_active_job", data.job_id);
      currentJobIdBadge.textContent = `Job: ${data.job_id}`;

      renderValidationReport(data.validation, data.total_images_in_zip);
    } catch (exc) {
      alert(`Local dataset load error: ${exc.message}`);
      dropzone.innerHTML = `
        <div class="dropzone-icon">📁</div>
        <div class="dropzone-text">Drag & drop your evaluation ZIP file here, or click to browse</div>
        <div class="dropzone-sub">Accepts ZIP containing images (.jpg, .jpeg, .png, .webp) + ground_truth.csv</div>
      `;
    } finally {
      state.uploading = false;
    }
  }

  async function uploadZipFile(file) {
    if (!file) return;

    const fileName = file.name || "";
    const lowerName = fileName.toLowerCase();
    const isZip = lowerName.endsWith(".zip") ||
                  file.type === "application/zip" ||
                  file.type === "application/x-zip-compressed" ||
                  file.type === "multipart/x-zip";

    if (!isZip) {
      alert("Please upload a .zip file containing images and ground_truth.csv.");
      return;
    }

    state.uploading = true;
    dropzone.innerHTML = `
      <div class="dropzone-icon">⏳</div>
      <div class="dropzone-text">Uploading and validating ${escapeHtml(fileName)}...</div>
      <div class="dropzone-sub">Inspecting archive structure and safety constraints</div>
    `;

    const formData = new FormData();
    formData.append("file", file, fileName);

    try {
      const res = await apiFetch("/v1/eval/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let errMessage = `Upload validation failed (HTTP ${res.status})`;
        try {
          const err = await res.json();
          errMessage = err.detail?.message || (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)) || errMessage;
        } catch (_) {
          try {
            const raw = await res.text();
            if (raw) errMessage = raw.slice(0, 300);
          } catch (__) {}
        }
        throw new Error(errMessage);
      }

      const data = await res.json();
      state.currentJobId = data.job_id;
      sessionStorage.setItem("siglip_eval_active_job", data.job_id);
      currentJobIdBadge.textContent = `Job: ${data.job_id}`;

      renderValidationReport(data.validation, data.total_images_in_zip);
    } catch (exc) {
      alert(`Validation error: ${exc.message}`);
      dropzone.innerHTML = `
        <div class="dropzone-icon">📁</div>
        <div class="dropzone-text">Drag & drop your evaluation ZIP file here, or click to browse</div>
        <div class="dropzone-sub">Accepts ZIP containing images (.jpg, .jpeg, .png, .webp) + ground_truth.csv</div>
      `;
    } finally {
      state.uploading = false;
    }
  }

  function renderValidationReport(val, totalImgsInZip) {
    state.validation = val;
    valTotalRows.textContent = val.total_rows;
    valValidRows.textContent = val.valid_rows_count;
    valMissingImages.textContent = val.missing_images ? val.missing_images.length : 0;
    valCorruptImages.textContent = val.corrupt_images ? val.corrupt_images.length : 0;
    valUnknownServices.textContent = val.unknown_services ? val.unknown_services.length : 0;
    valDuplicateRows.textContent = val.duplicate_rows ? val.duplicate_rows.length : 0;
    valEmptyDesc.textContent = val.empty_description_count || 0;
    valNoneRows.textContent = val.none_count || 0;

    // Highlight errors/warnings
    if (val.missing_images && val.missing_images.length > 0) {
      valMissingImages.classList.add("error");
    } else {
      valMissingImages.classList.remove("error");
    }

    if (val.unknown_services && val.unknown_services.length > 0) {
      valUnknownServices.classList.add("warning");
    } else {
      valUnknownServices.classList.remove("warning");
    }

    valAlerts.innerHTML = "";

    // Critical errors
    if (val.error_messages && val.error_messages.length > 0) {
      val.error_messages.forEach((msg) => {
        const div = document.createElement("div");
        div.className = "val-error-pill";
        div.textContent = `❌ ${msg}`;
        valAlerts.appendChild(div);
      });
    }

    // Unknown services with suggestions
    if (val.unknown_services && val.unknown_services.length > 0) {
      val.unknown_services.forEach((us) => {
        const div = document.createElement("div");
        div.className = "val-warning-pill";
        const sugg = us.suggestion ? ` (Did you mean "${us.suggestion}"?)` : "";
        div.textContent = `⚠️ Row ${us.row}: Unknown service "${us.service}"${sugg}`;
        valAlerts.appendChild(div);
      });
    }

    // Warnings
    if (val.warnings && val.warnings.length > 0) {
      val.warnings.forEach((w) => {
        const div = document.createElement("div");
        div.className = "val-warning-pill";
        div.textContent = `⚠️ ${w}`;
        valAlerts.appendChild(div);
      });
    }

    validationBox.style.display = "block";
    showSettingsBtn.style.display = val.has_critical_errors ? "none" : "inline-flex";

    dropzone.innerHTML = `
      <div class="dropzone-icon" style="color: var(--accent-green);">✓</div>
      <div class="dropzone-text">Dataset extracted & verified (${val.valid_rows_count} usable images)</div>
      <div class="dropzone-sub">Click to upload a different ZIP file</div>
    `;
  }

  showSettingsBtn.addEventListener("click", () => {
    settingsSection.style.display = "block";
    settingsSection.scrollIntoView({ behavior: "smooth" });
  });

  function updateRerankerToggleDisplay() {
    if (!enableRerankerInput || !enableRerankerVal) return;
    const isEnabled = Boolean(enableRerankerInput.checked);
    if (isEnabled) {
      enableRerankerVal.textContent = "ENABLED";
      enableRerankerVal.style.color = "var(--accent-green)";
    } else {
      enableRerankerVal.textContent = "BYPASSED";
      enableRerankerVal.style.color = "var(--accent-amber)";
    }
    if (presetRerankerOnBtn) presetRerankerOnBtn.classList.toggle("active", isEnabled);
    if (presetRerankerOffBtn) presetRerankerOffBtn.classList.toggle("active", !isEnabled);
  }

  resetSettingsBtn.addEventListener("click", () => {
    descWeightInput.value = 0.70;
    descWeightVal.textContent = "0.70";
    confThreshInput.value = 0.40;
    confThreshVal.textContent = "0.40";
    gapThreshInput.value = 0.10;
    gapThreshVal.textContent = "0.10";
    floorThreshInput.value = 0.25;
    floorThreshVal.textContent = "0.25";
    if (enableRerankerInput) {
      enableRerankerInput.checked = true;
      updateRerankerToggleDisplay();
    }
  });

  // ── Run Execution & SSE Streaming ──────────────────────────────────────────
  startRunBtn.addEventListener("click", async () => {
    if (!state.currentJobId) return;

    startRunBtn.disabled = true;
    stopRunBtn.style.display = "inline-flex";
    progressSection.style.display = "block";
    summarySection.style.display = "block";
    resultsSection.style.display = "block";

    // Clear previous results
    state.rows = [];
    state.summary = null;
    renderResultsTable();

    const overrides = {
      description_weight: parseFloat(descWeightInput.value),
      confidence_threshold: parseFloat(confThreshInput.value),
      gap_threshold: parseFloat(gapThreshInput.value),
      no_action_floor: parseFloat(floorThreshInput.value),
      enable_reranker: enableRerankerInput ? enableRerankerInput.checked : true,
    };

    try {
      const res = await apiFetch(`/v1/eval/jobs/${state.currentJobId}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(overrides),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to start job.");
      }

      connectSSE(state.currentJobId);
    } catch (exc) {
      alert(`Could not start evaluation: ${exc.message}`);
      startRunBtn.disabled = false;
      stopRunBtn.style.display = "none";
    }
  });

  stopRunBtn.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      await apiFetch(`/v1/eval/jobs/${state.currentJobId}/stop`, { method: "POST" });
      statusText.textContent = "Stopping evaluation...";
    } catch (e) {
      console.warn("Stop request failed:", e);
    }
  });

  function connectSSE(jobId) {
    if (state.eventSource) {
      state.eventSource.close();
    }

    // Build SSE URL; if API key exists, include in query param or rely on header
    let sseUrl = `/v1/eval/jobs/${jobId}/stream`;
    if (state.apiKey) {
      // Browsers don't support custom headers in standard EventSource; pass via query
      sseUrl += `?token=${encodeURIComponent(state.apiKey)}`;
    }

    const es = new EventSource(sseUrl);
    state.eventSource = es;

    es.addEventListener("init", (e) => {
      const data = JSON.parse(e.data);
      state.jobStatus = data.status;
      if (data.rows && data.rows.length > 0) {
        state.rows = data.rows;
        renderResultsTable();
      }
      if (data.summary) {
        state.summary = data.summary;
        renderSummaryPanel(data.summary);
      }
      updateProgressDisplay(data.current_index, data.total_rows);
    });

    es.addEventListener("status", (e) => {
      const data = JSON.parse(e.data);
      statusText.textContent = data.message || data.status;
    });

    es.addEventListener("row", (e) => {
      const data = JSON.parse(e.data);
      state.rows.push(data.row);
      appendTableRow(data.row);
      updateProgressDisplay(data.index, data.total);
      if (data.summary) {
        state.summary = data.summary;
        renderSummaryPanel(data.summary);
      }
    });

    es.addEventListener("complete", (e) => {
      const data = JSON.parse(e.data);
      state.jobStatus = "completed";
      statusSpinner.textContent = "✅";
      statusText.textContent = `Evaluation complete (${data.total_rows} images evaluated in ${data.elapsed_sec}s).`;
      stopRunBtn.style.display = "none";
      startRunBtn.disabled = false;
      if (data.summary) {
        state.summary = data.summary;
        renderSummaryPanel(data.summary);
      }
      es.close();
      loadPastRuns();
    });

    es.addEventListener("stopped", (e) => {
      state.jobStatus = "stopped";
      statusSpinner.textContent = "⏹";
      statusText.textContent = "Evaluation stopped by user.";
      stopRunBtn.style.display = "none";
      startRunBtn.disabled = false;
      es.close();
      loadPastRuns();
    });

    es.addEventListener("error", (e) => {
      console.warn("SSE connection error / closed:", e);
      es.close();
      startRunBtn.disabled = false;
      stopRunBtn.style.display = "none";
    });
  }

  function updateProgressDisplay(curr, total) {
    if (!total || total <= 0) return;
    const pct = Math.min(100, Math.round((curr / total) * 100));
    progressBarFill.style.width = `${pct}%`;
    progressFraction.textContent = `${curr} / ${total} (${pct}%)`;
    if (state.summary && state.summary.latency) {
      metaSpeed.textContent = `${state.summary.latency.throughput_fps || 0} img/sec`;
    }
  }

  // ── Load Existing Job ──────────────────────────────────────────────────────
  async function loadJob(jobId) {
    try {
      const res = await apiFetch(`/v1/eval/jobs/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();

      state.currentJobId = data.job_id;
      sessionStorage.setItem("siglip_eval_active_job", data.job_id);
      currentJobIdBadge.textContent = `Job: ${data.job_id}`;
      state.jobStatus = data.status;
      state.rows = data.rows || [];
      state.summary = data.summary || null;

      // Restore effective settings display
      if (data.effective_settings) {
        descWeightInput.value = data.effective_settings.description_weight || 0.70;
        descWeightVal.textContent = parseFloat(descWeightInput.value).toFixed(2);
        confThreshInput.value = data.effective_settings.confidence_threshold || 0.40;
        confThreshVal.textContent = parseFloat(confThreshInput.value).toFixed(2);
        gapThreshInput.value = data.effective_settings.gap_threshold || 0.10;
        gapThreshVal.textContent = parseFloat(gapThreshInput.value).toFixed(2);
        floorThreshInput.value = data.effective_settings.no_action_floor || 0.25;
        floorThreshVal.textContent = parseFloat(floorThreshInput.value).toFixed(2);
        if (enableRerankerInput && data.effective_settings.enable_reranker !== undefined) {
          enableRerankerInput.checked = Boolean(data.effective_settings.enable_reranker);
          updateRerankerToggleDisplay();
        }
      }

      settingsSection.style.display = "block";
      progressSection.style.display = "block";
      summarySection.style.display = "block";
      resultsSection.style.display = "block";

      updateProgressDisplay(data.current_index || data.total_rows, data.total_rows);
      statusText.textContent = `Loaded run ${data.job_id} (${data.status})`;
      statusSpinner.textContent = data.status === "completed" ? "✅" : "📄";

      if (state.summary) {
        renderSummaryPanel(state.summary);
      }
      renderResultsTable();

      // If running, reconnect SSE
      if (data.status === "running") {
        connectSSE(jobId);
        startRunBtn.disabled = true;
        stopRunBtn.style.display = "inline-flex";
      }
    } catch (e) {
      console.warn("Failed to load job:", e);
    }
  }

  // ── Render Summary Dashboard ───────────────────────────────────────────────
  function renderSummaryPanel(s) {
    if (!s || !s.overall) return;
    const ov = s.overall;
    const act = s.actions || {};
    const lat = s.latency || {};
    const sub = s.subsets || {};

    kpiTop1.textContent = `${ov.top1_accuracy}%`;
    kpiTop1Sub.textContent = `n = ${s.n} (Excl. warmup)`;

    kpiTop35.textContent = `${ov.top3_accuracy}% / ${ov.top5_accuracy}%`;
    kpiPoolRecall.textContent = `${ov.pool_recall}%`;

    kpiAutoPrec.textContent = `${act.auto_precision}%`;
    kpiAutoPrecSub.textContent = `${act.auto_create_count} auto (${act.coverage_pct}% cov)`;

    const wrongAutos = act.wrong_auto_count || 0;
    kpiWrongAuto.textContent = wrongAutos;
    if (wrongAutos > 0) {
      kpiWrongAuto.style.color = "var(--accent-red)";
      wrongAutoCard.classList.add("highlight-danger");
    } else {
      kpiWrongAuto.style.color = "var(--accent-green)";
      wrongAutoCard.classList.remove("highlight-danger");
    }

    kpiThroughput.textContent = `${lat.throughput_fps || 0} img/s`;
    kpiMeanLatency.textContent = `Mean: ${lat.mean_ms || 0} ms`;

    // Subsets
    subDescAcc.textContent = sub.with_description ? `${sub.with_description.accuracy}% (n=${sub.with_description.n})` : "-";
    subImageOnlyAcc.textContent = sub.image_only ? `${sub.image_only.accuracy}% (n=${sub.image_only.n})` : "-";
    subNoneAcc.textContent = sub.none_expected ? `${sub.none_expected.accuracy}% (n=${sub.none_expected.n})` : "-";
    subAgreeRate.textContent = `${ov.agree_rate}% (Agree acc: ${ov.accuracy_when_agree}%, Disagree: ${ov.accuracy_when_disagree}%)`;

    // Actions
    actionAutoCount.textContent = act.auto_create_count || 0;
    actionDraftCount.textContent = act.draft_count || 0;
    actionNoActionCount.textContent = act.no_action_count || 0;
    actionAutoCoverage.textContent = `${act.coverage_pct}%`;

    // Latency
    latP50.textContent = `${lat.p50_ms || 0} ms`;
    latP95.textContent = `${lat.p95_ms || 0} ms`;
    latMax.textContent = `${lat.max_ms || 0} ms`;
    if (lat.stages && lat.stages.siglip_encode_ms) {
      latSiglipMean.textContent = `${lat.stages.siglip_encode_ms.mean_ms} ms`;
    }

    // Findings Box
    if (summaryFindings) {
      summaryFindings.innerHTML = "";
      if (s.findings && s.findings.length > 0) {
        const box = document.createElement("div");
        box.className = "findings-box";
        let fHtml = '<div style="font-weight:700;margin-bottom:6px;">📌 Key Findings & Evaluation Insights:</div><ul style="margin:0;padding-left:18px;">';
        s.findings.forEach((f) => {
          fHtml += `<li>${escapeHtml(f)}</li>`;
        });
        fHtml += "</ul>";
        box.innerHTML = fHtml;
        summaryFindings.appendChild(box);
      }
    }

    // Warnings Box
    summaryWarnings.innerHTML = "";
    if (s.warnings && s.warnings.length > 0) {
      s.warnings.forEach((w) => {
        const div = document.createElement("div");
        div.className = w.includes("CRITICAL") ? "val-error-pill" : "val-warning-pill";
        div.textContent = `⚠️ ${w}`;
        summaryWarnings.appendChild(div);
      });
    }

    const eff = s.effective_settings || state.effectiveSettings || {};

    // 1. Visible RERANKER OFF Banner (Requirement 1)
    if (rerankerOffBanner) {
      if (eff && eff.enable_reranker === false) {
        rerankerOffBanner.style.display = "block";
      } else {
        rerankerOffBanner.style.display = "none";
      }
    }

    // 2. Run Record Snapshot (Requirement 1)
    renderRunRecordSnapshot(eff);

    // 3. Dedicated NONE Panel (3 Levels: Requirement 4)
    renderNonePanel(s.none_panel, s);

    // 4. Signal Quality Table (AUROC: Requirement 5)
    renderSignalQuality(s.signal_quality);

    // Render Canvas Latency Chart
    renderLatencyBarChart(lat.stages || {});

    // Render Top Confusions
    renderTopConfusions(s.top_confusions || []);

    // Render Per-Service Table
    renderServicesTable(s.services || []);

    // Render Confusion Matrix Heatmap
    renderConfusionMatrix(s.confusion_matrix || { labels: [], matrix: [] });

    // Render Threshold Sensitivity Grid
    renderThresholdsTable(s.threshold_grid || []);

    // Render Calibration
    renderCalibration(s.calibration || []);
  }

  // ── Render Run Record Snapshot Panel (Requirement 1) ──────────────────────
  function renderRunRecordSnapshot(eff) {
    if (!runRecordSnapshot || !snapshotGrid) return;
    if (!eff || (eff.enable_reranker === undefined && !eff.catalog_hash)) {
      runRecordSnapshot.style.display = "none";
      return;
    }

    runRecordSnapshot.style.display = "block";
    if (snapshotCatalogHashBadge) {
      snapshotCatalogHashBadge.textContent = `Catalog Hash: ${eff.catalog_hash || "N/A"}`;
      snapshotCatalogHashBadge.title = "SHA-256 fingerprint of active catalog title + description + visual_line";
    }

    const items = [
      {
        label: "Reranker Mode",
        value: eff.enable_reranker ? "🟢 ENABLED (Production)" : "⚪ BYPASSED (Bi-Encoder Only)",
        isHighlighted: !eff.enable_reranker,
      },
      {
        label: "Description Weight",
        value: eff.description_weight !== undefined ? Number(eff.description_weight).toFixed(2) : "-",
      },
      {
        label: "Bi-Encoder Top-N",
        value: `Text: ${eff.text_top_n ?? "-"} | Image: ${eff.image_top_n ?? "-"}`,
      },
      {
        label: "Pool Max Candidates",
        value: eff.candidate_pool_max ?? "-",
      },
      {
        label: "Temperatures",
        value: `Reranker: ${eff.reranker_temperature ?? "-"} | Bi-enc: ${eff.biencoder_temperature ?? "-"}`,
      },
      {
        label: "Agreement Boost",
        value: eff.agreement_boost !== undefined ? Number(eff.agreement_boost).toFixed(2) : "-",
      },
      {
        label: "Confidence Threshold",
        value: eff.confidence_threshold !== undefined ? Number(eff.confidence_threshold).toFixed(2) : "-",
      },
      {
        label: "Margin / Gap Threshold",
        value: eff.gap_threshold !== undefined ? Number(eff.gap_threshold).toFixed(2) : "-",
      },
      {
        label: "No-Action Floor",
        value: eff.no_action_floor !== undefined ? Number(eff.no_action_floor).toFixed(2) : "-",
      },
      {
        label: "Require Agree",
        value: eff.require_agree !== undefined ? (eff.require_agree ? "true" : "false") : "-",
      },
      {
        label: "Min Description Words",
        value: eff.min_description_words ?? "-",
      },
      {
        label: "Auto-Create Enabled",
        value: eff.auto_create_enabled !== undefined ? (eff.auto_create_enabled ? "true" : "false") : "-",
      },
      {
        label: "Always-Draft Services",
        value: Array.isArray(eff.always_draft_services)
          ? (eff.always_draft_services.length ? escapeHtml(eff.always_draft_services.join(", ")) : "[] (None)")
          : escapeHtml(eff.always_draft_services || "[]"),
      },
      {
        label: "Active Catalog Hash",
        value: eff.catalog_hash ? `${escapeHtml(eff.catalog_hash.substring(0, 16))}...` : "-",
      },
      {
        label: "Model IDs",
        value: eff.model_ids
          ? `Text: ${escapeHtml(eff.model_ids.text_model || "-")}<br>Vision: ${escapeHtml(eff.model_ids.image_model || "-")}<br>Rerank: ${escapeHtml(eff.model_ids.reranker_model || "-")}`
          : "-",
      },
      {
        label: "Run Overrides",
        value: eff.overrides && Object.keys(eff.overrides).length > 0
          ? Object.entries(eff.overrides).map(([k, v]) => `<code>${escapeHtml(k)}: ${escapeHtml(v)}</code>`).join(", ")
          : '<span style="color:var(--text-muted);">None (Production Defaults)</span>',
      },
    ];

    snapshotGrid.innerHTML = "";
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "snapshot-card";
      if (item.isHighlighted) {
        card.style.borderColor = "var(--accent-amber)";
        card.style.background = "rgba(245, 158, 11, 0.08)";
      }
      card.innerHTML = `
        <div class="snapshot-card-label">${escapeHtml(item.label)}</div>
        <div class="snapshot-card-val">${item.value}</div>
      `;
      snapshotGrid.appendChild(card);
    });
  }

  // ── Render Signal Quality Table (AUROC: Requirement 5) ────────────────────
  function renderSignalQuality(sq) {
    if (!signalQualityTableBody) return;
    signalQualityTableBody.innerHTML = "";

    if (!sq || !sq.signals) {
      signalQualityTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 18px;">
            No signal quality data available for this evaluation run.
          </td>
        </tr>
      `;
      return;
    }

    const sigs = sq.signals;
    const signalOrder = [
      "softmax_confidence",
      "top1_gap",
      "raw_top1_reranker_score",
      "top1_text_similarity",
      "top1_image_similarity",
    ];

    signalOrder.forEach((k) => {
      const sInfo = sigs[k];
      if (!sInfo) return;

      const tr = document.createElement("tr");

      // Signal Name + Description
      const nameCol = `
        <div style="font-weight:600;">${escapeHtml(sInfo.name)}</div>
        <div style="font-size:11px; color:var(--text-muted);">${escapeHtml(sInfo.description || "")}</div>
      `;

      // n Normal
      const nNorm = sq.n_normal !== undefined ? sq.n_normal : "-";

      // AUROC Top-1 Correct
      let aurocNormStr = "-";
      if (k === "raw_top1_reranker_score" && (!sq.enable_reranker || sInfo.top1_correct_auroc === null)) {
        aurocNormStr = '<span style="color:var(--text-muted);font-size:11px;">N/A (Reranker OFF)</span>';
      } else if (sInfo.top1_correct_auroc !== null && sInfo.top1_correct_auroc !== undefined) {
        const v = sInfo.top1_correct_auroc;
        const color = v >= 0.70 ? "var(--accent-green)" : v >= 0.55 ? "var(--accent-amber)" : "var(--accent-red)";
        aurocNormStr = `<span style="font-family:monospace;font-weight:700;color:${color};">${v.toFixed(3)}</span>`;
      }

      // Warning Normal (< 10 rows in a class)
      const warnNorm = sq.warning_normal
        ? `<span class="val-warning-pill" style="font-size:10px;padding:2px 6px;">⚠️ ${escapeHtml(sq.warning_normal)}</span>`
        : '<span style="color:var(--accent-green);font-size:11px;">✓ n &ge; 10</span>';

      // n All
      const nAll = sq.n_all !== undefined ? sq.n_all : "-";

      // AUROC Normal vs NONE
      let aurocNoneStr = "-";
      if (k === "raw_top1_reranker_score" && (!sq.enable_reranker || sInfo.none_distinction_auroc === null)) {
        aurocNoneStr = '<span style="color:var(--text-muted);font-size:11px;">N/A (Reranker OFF)</span>';
      } else if (sInfo.none_distinction_auroc !== null && sInfo.none_distinction_auroc !== undefined) {
        const v = sInfo.none_distinction_auroc;
        const color = v >= 0.70 ? "var(--accent-green)" : v >= 0.55 ? "var(--accent-amber)" : "var(--accent-red)";
        aurocNoneStr = `<span style="font-family:monospace;font-weight:700;color:${color};">${v.toFixed(3)}</span>`;
      }

      // Warning NONE (< 10 rows in a class)
      const warnNone = sq.warning_none
        ? `<span class="val-warning-pill" style="font-size:10px;padding:2px 6px;">⚠️ ${escapeHtml(sq.warning_none)}</span>`
        : '<span style="color:var(--accent-green);font-size:11px;">✓ n &ge; 10</span>';

      tr.innerHTML = `
        <td>${nameCol}</td>
        <td style="font-family:monospace;">${nNorm}</td>
        <td>${aurocNormStr}</td>
        <td>${warnNorm}</td>
        <td style="font-family:monospace;">${nAll}</td>
        <td>${aurocNoneStr}</td>
        <td>${warnNone}</td>
      `;

      signalQualityTableBody.appendChild(tr);
    });
  }

  // ── Render Dedicated NONE Ground Truth Panel (3 Levels: Requirement 4) ────
  function renderNonePanel(np, s) {
    if (!tabNoneBadge) return;
    const n = np ? (np.n || 0) : 0;
    tabNoneBadge.textContent = n;

    if (kpiNoneN) kpiNoneN.textContent = n;
    if (kpiNoneCorrectRate) {
      kpiNoneCorrectRate.textContent = np ? `${np.correct_rate}%` : "-";
    }
    if (kpiNoneCorrectSub) {
      kpiNoneCorrectSub.textContent = np ? `${np.correct_count} / ${n} No Action` : "Correctly Suppressed";
    }
    if (kpiNoneDraftRate) {
      kpiNoneDraftRate.textContent = np ? `${np.draft_rate || 0}%` : "-";
    }
    if (kpiNoneDraftSub) {
      kpiNoneDraftSub.textContent = np ? `${np.draft_count || 0} / ${n} Human Draft Review` : "Human Draft Review";
    }

    const wrongAutos = np ? (np.wrong_auto_count || 0) : 0;
    if (kpiNoneWrongAutoCount) {
      kpiNoneWrongAutoCount.textContent = wrongAutos;
      kpiNoneWrongAutoCount.style.color = wrongAutos > 0 ? "var(--accent-red)" : "var(--accent-green)";
    }
    if (kpiNoneWrongAutoSub) {
      kpiNoneWrongAutoSub.textContent = np ? `${np.wrong_auto_rate || 0}% (${wrongAutos} / ${n} Harmful)` : "Critical False Auto-Create";
    }
    if (cardNoneFalseMatch) {
      if (wrongAutos > 0) {
        cardNoneFalseMatch.classList.add("highlight-danger");
        cardNoneFalseMatch.style.borderColor = "var(--accent-red)";
      } else {
        cardNoneFalseMatch.classList.remove("highlight-danger");
        cardNoneFalseMatch.style.borderColor = "var(--border-color)";
      }
    }

    // Populate False Matches Table (10 columns: #, thumb, img, desc, pred, action, level/severity, conf, gap, action)
    if (noneFalseMatchesTableBody) {
      noneFalseMatchesTableBody.innerHTML = "";
      const fMatches = (np && np.false_matches) ? np.false_matches : [];
      if (noneFalseMatchesHeaderCount) noneFalseMatchesHeaderCount.textContent = fMatches.length;

      if (fMatches.length === 0) {
        noneFalseMatchesTableBody.innerHTML = `
          <tr>
            <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 18px;">
              ${n === 0 ? "No NONE ground-truth rows in dataset." : "✓ All NONE rows were correctly returned as No Action (0 false matches)."}
            </td>
          </tr>
        `;
      } else {
        fMatches.forEach((fm) => {
          const tr = document.createElement("tr");
          const actClass = fm.action === "auto_create" ? "badge-auto" : fm.action === "draft" ? "badge-draft" : "badge-no-action";
          const actLabel = fm.action === "auto_create" ? "AUTO" : fm.action === "draft" ? "DRAFT" : "NO ACTION";
          const thumbHtml = fm.thumbnail_b64
            ? `<img src="${fm.thumbnail_b64}" class="row-thumb" alt="${escapeHtml(fm.image)}" data-row-id="${fm.row_id}" title="Click to view large preview" style="cursor: pointer;">`
            : `<div class="row-thumb" data-row-id="${fm.row_id}" style="cursor: pointer;">IMG</div>`;

          const levelBadge = fm.is_harmful
            ? '<span class="badge badge-harmful">Level 3: HARMFUL WRONG-AUTO</span>'
            : '<span class="badge badge-soft">Level 2: Draft (Soft Match)</span>';

          if (fm.is_harmful) {
            tr.style.background = "rgba(239, 68, 68, 0.08)";
          }

          tr.innerHTML = `
            <td>${fm.row_id}</td>
            <td>${thumbHtml}</td>
            <td style="font-family: monospace; font-size: 11px;">${escapeHtml(fm.image)}</td>
            <td style="max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(fm.description || '<empty>')}">
              ${escapeHtml(fm.description || '<empty>')}
            </td>
            <td style="font-weight: 600; color: var(--accent-red);">${escapeHtml(fm.predicted_service || "NONE")}</td>
            <td><span class="badge ${actClass}">${actLabel}</span></td>
            <td>${levelBadge}</td>
            <td>${fm.confidence !== null ? (fm.confidence * 100).toFixed(1) + "%" : "-"}</td>
            <td>g: ${fm.gap !== null ? (fm.gap * 100).toFixed(1) + "%" : "-"}</td>
            <td>
              <button class="btn btn-secondary btn-preview-modal" style="padding: 2px 8px; font-size: 11px;" data-row-id="${fm.row_id}">
                Preview 🔍
              </button>
            </td>
          `;
          noneFalseMatchesTableBody.appendChild(tr);
        });
      }
    }

    // Render Canvas Histogram
    renderNoneHistogram(np ? np.histogram : []);
  }

  // ── Render Top-1 Confidence Distribution Histogram (NONE vs Correct Normal) ─
  function renderNoneHistogram(hist) {
    if (!noneHistogramCanvas) return;
    const ctx = noneHistogramCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = noneHistogramCanvas.getBoundingClientRect();
    if (rect.width === 0) return;

    noneHistogramCanvas.width = rect.width * dpr;
    noneHistogramCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    if (!hist || hist.length === 0) {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No NONE evaluation data available", w / 2, h / 2);
      return;
    }

    const padLeft = 45;
    const padRight = 20;
    const padTop = 38;
    const padBottom = 42;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Find max count for Y scale
    let maxCount = 1;
    hist.forEach((b) => {
      if (b.none_count > maxCount) maxCount = b.none_count;
      if (b.normal_correct_count > maxCount) maxCount = b.normal_correct_count;
    });
    maxCount = Math.ceil(maxCount * 1.15);

    // Draw horizontal grid lines & Y labels
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const yTicks = 4;
    for (let t = 0; t <= yTicks; t++) {
      const val = Math.round((maxCount / yTicks) * t);
      const y = padTop + plotH - (val / maxCount) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
      ctx.fillText(String(val), padLeft - 8, y);
    }

    // Legend at top
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    // Blue legend (Normal Correct)
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(padLeft + 8, 14, 12, 12);
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "11px sans-serif";
    ctx.fillText("Correct Normal Rows (True Services)", padLeft + 26, 20);

    // Red legend (NONE rows)
    const leg2X = padLeft + 250;
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(leg2X, 14, 12, 12);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("NONE Rows (Out-of-Scope Ground Truth)", leg2X + 18, 20);

    // Bars per bin
    const binCount = hist.length;
    const binSlotW = plotW / binCount;
    const barW = Math.max(3, (binSlotW - 8) / 2);

    hist.forEach((b, i) => {
      const binCenterX = padLeft + i * binSlotW + binSlotW / 2;
      const xNorm = binCenterX - barW - 1;
      const xNone = binCenterX + 1;

      // Normal Correct Bar (Blue)
      if (b.normal_correct_count > 0) {
        const normH = (b.normal_correct_count / maxCount) * plotH;
        const normY = padTop + plotH - normH;
        ctx.fillStyle = "#3b82f6";
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(xNorm, normY, barW, normH, [2, 2, 0, 0]);
        } else {
          ctx.rect(xNorm, normY, barW, normH);
        }
        ctx.fill();

        ctx.fillStyle = "#93c5fd";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(b.normal_correct_count), xNorm + barW / 2, normY - 4);
      }

      // NONE Bar (Red)
      if (b.none_count > 0) {
        const noneH = (b.none_count / maxCount) * plotH;
        const noneY = padTop + plotH - noneH;
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(xNone, noneY, barW, noneH, [2, 2, 0, 0]);
        } else {
          ctx.rect(xNone, noneY, barW, noneH);
        }
        ctx.fill();

        ctx.fillStyle = "#fca5a5";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(b.none_count), xNone + barW / 2, noneY - 4);
      }

      // X Axis Label
      ctx.fillStyle = "#64748b";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(b.label, binCenterX, padTop + plotH + 8);
    });
  }

  // ── Canvas Bar Chart for Stage Latency ─────────────────────────────────────
  function renderLatencyBarChart(stages) {
    if (!latencyCanvas) return;
    const ctx = latencyCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = latencyCanvas.getBoundingClientRect();
    latencyCanvas.width = rect.width * dpr;
    latencyCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    const keys = Object.keys(stages);
    if (keys.length === 0) return;

    const labels = keys.map((k) => stages[k].label || k);
    const vals = keys.map((k) => stages[k].mean_ms || 0);
    const maxVal = Math.max(...vals, 1);

    const padLeft = 140;
    const padRight = 40;
    const padTop = 15;
    const padBottom = 20;

    const chartW = w - padLeft - padRight;
    const barHeight = Math.max(12, Math.floor((h - padTop - padBottom) / keys.length) - 8);

    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";

    keys.forEach((k, i) => {
      const y = padTop + i * (barHeight + 8);
      const val = vals[i];
      const barW = Math.max(2, (val / maxVal) * chartW);

      // Label
      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "right";
      ctx.fillText(labels[i], padLeft - 10, y + barHeight / 2 + 4);

      // Bar
      ctx.fillStyle = k === "siglip_encode_ms" ? "#8b5cf6" : k === "rerank_ms" ? "#3b82f6" : "#10b981";
      ctx.beginPath();
      ctx.roundRect(padLeft, y, barW, barHeight, 3);
      ctx.fill();

      // Value text
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.fillText(`${val} ms`, padLeft + barW + 6, y + barHeight / 2 + 4);
    });
  }

  // ── Top Confusions ─────────────────────────────────────────────────────────
  function renderTopConfusions(confList) {
    confusionsList.innerHTML = "";
    if (confList.length === 0) {
      confusionsList.innerHTML = '<span style="color: var(--accent-green);">No misclassifications recorded!</span>';
      return;
    }
    confList.slice(0, 10).forEach((c) => {
      const div = document.createElement("div");
      div.style.cssText = "display: flex; justify-content: space-between; padding: 4px 8px; background: var(--bg-secondary); border-radius: 4px;";
      div.innerHTML = `
        <span><strong style="color: var(--text-primary);">${escapeHtml(c.expected)}</strong> → <span style="color: var(--accent-red);">${escapeHtml(c.predicted)}</span></span>
        <span style="font-weight: 700; color: var(--accent-amber);">${c.count} row(s)</span>
      `;
      confusionsList.appendChild(div);
    });
  }

  // ── Per-Service Accuracy Table ─────────────────────────────────────────────
  function renderServicesTable(services) {
    servicesTableBody.innerHTML = "";
    services.sort((a, b) => a.accuracy - b.accuracy); // lowest accuracy first
    services.forEach((s) => {
      const tr = document.createElement("tr");
      const warningHtml = s.small_sample ? '<span class="badge" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">n &lt; 5 (Low Sample)</span>' : '<span style="color: var(--accent-green); font-size: 11px;">OK</span>';
      tr.innerHTML = `
        <td style="font-weight: 600;">${escapeHtml(s.service)}</td>
        <td>${s.n}</td>
        <td style="font-weight: 700; color: ${s.accuracy >= 70 ? "var(--accent-green)" : s.accuracy >= 40 ? "var(--accent-amber)" : "var(--accent-red)"};">${s.accuracy}%</td>
        <td style="font-family: monospace;">[${s.ci_95[0]}% – ${s.ci_95[1]}%]</td>
        <td>${warningHtml}</td>
      `;
      servicesTableBody.appendChild(tr);
    });
  }

  // ── Confusion Matrix Heatmap ───────────────────────────────────────────────
  function renderConfusionMatrix(matrixData) {
    const labels = matrixData.labels || [];
    const mat = matrixData.matrix || [];
    if (labels.length === 0) return;

    let html = '<table class="matrix-table"><thead><tr><th>Expected \\ Pred</th>';
    labels.forEach((l) => {
      html += `<th title="${escapeHtml(l)}">${escapeHtml(l.length > 15 ? l.substring(0, 14) + "…" : l)}</th>`;
    });
    html += '</tr></thead><tbody>';

    for (let i = 0; i < labels.length; i++) {
      const expL = labels[i];
      html += `<tr><th title="${escapeHtml(expL)}">${escapeHtml(expL.length > 15 ? expL.substring(0, 14) + "…" : expL)}</th>`;
      for (let j = 0; j < labels.length; j++) {
        const count = mat[i] ? mat[i][j] : 0;
        let cls = "";
        if (count > 0) {
          cls = (i === j) ? "correct-diag" : "wrong-cell";
        }
        html += `<td class="${cls}" title="${escapeHtml(labels[i])} -> ${escapeHtml(labels[j])}: ${count}">${count > 0 ? count : "-"}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody></table>';
    matrixContainer.innerHTML = html;
  }

  // ── Threshold Sensitivity Grid ─────────────────────────────────────────────
  function renderThresholdsTable(grid) {
    thresholdsTableBody.innerHTML = "";
    grid.forEach((item) => {
      const tr = document.createElement("tr");
      const fmCount = (item.none_false_matches !== undefined) ? item.none_false_matches : "-";
      const fmColor = (item.none_false_matches > 0) ? "var(--accent-amber)" : "var(--accent-green)";
      tr.innerHTML = `
        <td style="font-weight: 600;">${item.confidence_threshold.toFixed(2)}</td>
        <td>${item.gap_threshold.toFixed(2)}</td>
        <td>${item.auto_count}</td>
        <td>${item.coverage_pct}%</td>
        <td style="font-weight: 700; color: ${item.precision_pct >= 90 ? "var(--accent-green)" : item.precision_pct >= 70 ? "var(--accent-amber)" : "var(--accent-red)"};">${item.precision_pct}%</td>
        <td style="font-weight: 600; color: ${fmColor};">${fmCount}</td>
      `;
      thresholdsTableBody.appendChild(tr);
    });
  }

  // ── Offline Calibration Sweeps ─────────────────────────────────────────────
  function renderCalibration(calibList) {
    if (!calibList || calibList.length === 0) return;

    // Render selector buttons
    tempSelectorBtns.innerHTML = "";
    calibList.forEach((c) => {
      const btn = document.createElement("button");
      btn.className = `btn btn-secondary ${c.temperature === state.activeTemp ? "active" : ""}`;
      btn.textContent = `T = ${c.temperature}`;
      btn.addEventListener("click", () => {
        state.activeTemp = c.temperature;
        document.querySelectorAll("#tempSelectorBtns button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderSelectedCalib(c);
      });
      tempSelectorBtns.appendChild(btn);
    });

    const activeCalib = calibList.find((c) => c.temperature === state.activeTemp) || calibList[0];
    renderSelectedCalib(activeCalib);
  }

  function renderSelectedCalib(c) {
    calibEceVal.textContent = `${c.ece}%`;
    calibCovVal.textContent = `${c.coverage_pct}%`;
    calibPrecVal.textContent = `${c.precision_pct}%`;

    // Render reliability canvas
    renderReliabilityDiagram(c.bins || []);

    // Render bins table
    calibBinsTableBody.innerHTML = "";
    (c.bins || []).forEach((b) => {
      const tr = document.createElement("tr");
      const warningHtml = b.small_sample ? '<span class="badge" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24;">n &lt; 30 (Small Sample)</span>' : '<span style="color: var(--accent-green); font-size: 11px;">OK</span>';
      tr.innerHTML = `
        <td style="font-weight: 600;">${b.label}</td>
        <td>${b.n}</td>
        <td>${b.avg_confidence}%</td>
        <td style="font-weight: 700;">${b.n > 0 ? b.accuracy + "%" : "-"}</td>
        <td>${warningHtml}</td>
      `;
      calibBinsTableBody.appendChild(tr);
    });
  }

  function renderReliabilityDiagram(bins) {
    if (!reliabilityCanvas) return;
    const ctx = reliabilityCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = reliabilityCanvas.getBoundingClientRect();
    reliabilityCanvas.width = rect.width * dpr;
    reliabilityCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const padLeft = 40;
    const padRight = 20;
    const padTop = 15;
    const padBottom = 30;

    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    // Diagonal perfect calibration reference line
    ctx.strokeStyle = "#475569";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop);
    ctx.stroke();
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Ticks & Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("100%", padLeft - 6, padTop + 8);
    ctx.fillText("0%", padLeft - 6, padTop + plotH);
    ctx.textAlign = "center";
    ctx.fillText("Confidence", padLeft + plotW / 2, h - 5);

    // Plot observed points / bars
    const barW = Math.max(4, plotW / bins.length - 4);
    bins.forEach((b, i) => {
      if (b.n === 0) return;
      const x = padLeft + (i / bins.length) * plotW + 2;
      const accRatio = Math.min(1.0, b.accuracy / 100.0);
      const barH = accRatio * plotH;
      const y = padTop + plotH - barH;

      ctx.fillStyle = "#3b82f6";
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, 2);
      ctx.fill();
    });
  }

  // ── Per-Image Table Rendering ──────────────────────────────────────────────
  function renderResultsTable() {
    resultsTableBody.innerHTML = "";
    const filtered = getFilteredRows();
    tableRowCounter.textContent = `Showing ${filtered.length} of ${state.rows.length} rows`;
    filtered.forEach((r) => appendTableRow(r));
  }

  function getFilteredRows() {
    return state.rows.filter((r) => {
      // Warmup exclusion
      if (r.is_warmup) return false;

      // Filter category
      if (state.activeFilter === "wrong" && r.correct) return false;
      if (state.activeFilter === "none" && !r.is_none) return false;
      if (state.activeFilter === "empty_desc" && !r.is_empty_desc) return false;
      if (state.activeFilter === "auto" && r.action !== "auto_create") return false;
      if (state.activeFilter === "wrong_auto" && !(r.action === "auto_create" && !r.correct)) return false;

      // Text query
      if (state.searchQuery) {
        const q = state.searchQuery.toLowerCase();
        const str = `${r.image} ${r.description} ${r.expected_service} ${r.predicted_service}`.toLowerCase();
        if (!str.includes(q)) return false;
      }
      return true;
    }).sort((a, b) => {
      let vA = a[state.sortCol];
      let vB = b[state.sortCol];
      if (vA === null || vA === undefined) vA = -999999;
      if (vB === null || vB === undefined) vB = -999999;
      if (typeof vA === "string") {
        return state.sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
      }
      return state.sortAsc ? vA - vB : vB - vA;
    });
  }

  function appendTableRow(r) {
    if (r.is_warmup) return;

    // Check filter match
    if (state.activeFilter === "wrong" && r.correct) return;
    if (state.activeFilter === "none" && !r.is_none) return;
    if (state.activeFilter === "empty_desc" && !r.is_empty_desc) return;
    if (state.activeFilter === "auto" && r.action !== "auto_create") return;
    if (state.activeFilter === "wrong_auto" && !(r.action === "auto_create" && !r.correct)) return;

    if (state.searchQuery) {
      const q = state.searchQuery.toLowerCase();
      const str = `${r.image} ${r.description} ${r.expected_service} ${r.predicted_service}`.toLowerCase();
      if (!str.includes(q)) return;
    }

    const tr = document.createElement("tr");
    tr.id = `row-${r.row_id}`;

    // Thumbnail
    const thumbHtml = r.thumbnail_b64
      ? `<img src="${r.thumbnail_b64}" class="row-thumb" alt="${r.image}" data-img="${r.thumbnail_b64}" data-row-id="${r.row_id}" title="Click to view large preview" style="cursor: pointer;">`
      : `<div class="row-thumb" data-row-id="${r.row_id}" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:10px;cursor:pointer;">IMG</div>`;

    // Action Badge
    const actClass = r.action === "auto_create" ? "badge-auto" : r.action === "draft" ? "badge-draft" : "badge-no-action";
    const actLabel = r.action === "auto_create" ? "AUTO" : r.action === "draft" ? "DRAFT" : "NO ACTION";

    // Result Badge: NONE row returned as auto or draft is marked FALSE MATCH
    let resBadge = "";
    if (r.is_none) {
      if (r.correct) {
        resBadge = '<span class="badge-correct">✓ CORRECT</span>';
      } else {
        resBadge = '<span class="badge-false-match">✗ FALSE MATCH</span>';
      }
    } else {
      resBadge = r.correct ? '<span class="badge-correct">✓ CORRECT</span>' : '<span class="badge-wrong">✗ WRONG</span>';
    }

    // Pool Ranks
    const tRankStr = r.text_pool_rank !== null ? `#${r.text_pool_rank}` : '<span class="badge-not-in-pool">not in pool</span>';
    const iRankStr = r.image_pool_rank !== null ? `#${r.image_pool_rank}` : '<span class="badge-not-in-pool">not in pool</span>';
    const rRankStr = r.reranker_rank !== null ? `#${r.reranker_rank}` : '<span class="badge-not-in-pool">not in pool</span>';

    const agreeBadge = r.agree === true
      ? '<span style="color:var(--accent-green);font-size:11px;">AGREE</span>'
      : r.agree === false
      ? '<span style="color:var(--accent-amber);font-size:11px;">DISAGREE</span>'
      : '<span style="color:var(--text-muted);font-size:11px;">-</span>';

    const confStr = r.confidence !== null ? `${(r.confidence * 100).toFixed(1)}%` : "-";
    const gapStr = r.gap !== null ? `g: ${(r.gap * 100).toFixed(1)}%` : "";

    tr.innerHTML = `
      <td>${r.row_id}</td>
      <td>${thumbHtml}</td>
      <td style="font-family:monospace;font-size:11px;">${escapeHtml(r.image)}</td>
      <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(r.description || '(empty description)')}">${escapeHtml(r.description || '<empty>')}</td>
      <td style="font-weight:600;">${escapeHtml(r.is_none ? "NONE" : r.expected_service)}</td>
      <td style="font-weight:600;color:${r.correct ? "var(--text-primary)" : "var(--accent-red)"};">${escapeHtml(r.predicted_service || "NONE")}</td>
      <td><span class="badge ${actClass}">${actLabel}</span></td>
      <td><div>${confStr}</div><div style="font-size:11px;color:var(--text-muted);">${gapStr}</div></td>
      <td>${resBadge}</td>
      <td style="font-size:11px;">T: ${tRankStr}<br>I: ${iRankStr}<br>R: ${rRankStr}</td>
      <td>${agreeBadge}</td>
      <td>${r.timings?.total_ms || 0} ms</td>
      <td><button class="btn btn-secondary btn-expand" style="padding:2px 8px;font-size:11px;" data-id="${r.row_id}">Details ▾</button></td>
    `;

    resultsTableBody.appendChild(tr);

    // If expanded, append details row
    if (state.expandedRows.has(r.row_id)) {
      renderDetailsRow(r);
    }
  }

  function renderDetailsRow(r) {
    const existing = document.getElementById(`details-${r.row_id}`);
    if (existing) existing.remove();

    const dTr = document.createElement("tr");
    dTr.id = `details-${r.row_id}`;
    dTr.className = "details-row";

    let candsHtml = '<div style="font-weight:600;margin-bottom:6px;">Candidate Pool & Rerank Scores:</div><table class="cand-table"><thead><tr><th>Rerank Rank</th><th>Service</th><th>Confidence</th><th>Raw Logit</th><th>Text Rank</th><th>Image Rank</th><th>Pool Rank</th></tr></thead><tbody>';

    (r.candidates || []).forEach((c) => {
      const isExp = c.title === r.expected_service;
      const isPred = c.title === r.predicted_service;
      const rowStyle = isExp ? "background: rgba(16, 185, 129, 0.15);" : isPred ? "background: rgba(59, 130, 246, 0.15);" : "";
      candsHtml += `
        <tr style="${rowStyle}">
          <td>#${c.rerank_rank || "-"}</td>
          <td style="font-weight:600;">${escapeHtml(c.title)} ${isExp ? "🎯 [EXPECTED]" : ""} ${isPred ? "👑 [TOP-1]" : ""}</td>
          <td>${c.confidence ? (c.confidence * 100).toFixed(1) + "%" : "-"}</td>
          <td style="font-family:monospace;">${c.raw_score !== null && c.raw_score !== undefined ? Number(c.raw_score).toFixed(3) : "-"}</td>
          <td>${c.text_rank ? "#" + c.text_rank : "-"}</td>
          <td>${c.image_rank ? "#" + c.image_rank : "-"}</td>
          <td>${c.pool_rank ? "#" + c.pool_rank : "-"}</td>
        </tr>
      `;
    });
    candsHtml += '</tbody></table>';

    // Explicit Top-1 Scores (Requirement 2)
    const top1RawStr = r.top1_raw_score !== null && r.top1_raw_score !== undefined
      ? Number(r.top1_raw_score).toFixed(4)
      : "N/A";
    const top1ImgSimStr = r.top1_image_similarity !== null && r.top1_image_similarity !== undefined
      ? Number(r.top1_image_similarity).toFixed(4)
      : "N/A";
    const top1TxtSimStr = r.top1_text_similarity !== null && r.top1_text_similarity !== undefined
      ? Number(r.top1_text_similarity).toFixed(4)
      : "N/A";

    let explicitScoresHtml = `
      <div style="display:flex;gap:18px;font-size:12px;margin-top:10px;padding:8px;background:var(--bg-secondary);border-radius:4px;border-left:3px solid var(--accent-blue);">
        <span><strong>Top-1 Raw Score:</strong> <span style="font-family:monospace;font-weight:600;">${top1RawStr}</span></span>
        <span><strong>Top-1 Image Sim:</strong> <span style="font-family:monospace;font-weight:600;">${top1ImgSimStr}</span></span>
        <span><strong>Top-1 Text Sim:</strong> <span style="font-family:monospace;font-weight:600;">${top1TxtSimStr}</span></span>
      </div>
    `;

    // Diagnostics full catalog ranks
    let diagHtml = `
      <div style="display:flex;gap:20px;font-size:12px;margin-top:8px;padding:8px;background:var(--bg-secondary);border-radius:4px;">
        <span><strong>Full Catalog Text Rank:</strong> ${r.full_text_rank ? "#" + r.full_text_rank : "N/A"}</span>
        <span><strong>Full Catalog Image Rank:</strong> ${r.full_image_rank ? "#" + r.full_image_rank : "N/A"}</span>
        <span><strong>Text Top-1:</strong> ${escapeHtml(r.text_top1 || "None")}</span>
        <span><strong>Image Top-1:</strong> ${escapeHtml(r.image_top1 || "None")}</span>
      </div>
    `;

    // All active services scores (Requirement 2)
    let allScoresHtml = "";
    const allTxt = r.all_service_text_scores || {};
    const allImg = r.all_service_image_scores || {};
    const allSvcs = Array.from(new Set([...Object.keys(allTxt), ...Object.keys(allImg)])).sort();

    if (allSvcs.length > 0) {
      allScoresHtml = `
        <details style="margin-top:8px; font-size:12px;">
          <summary style="cursor:pointer; color:var(--accent-blue); font-weight:600; outline:none;">
            📊 View All Active Service Scores (Full Catalog: ${allSvcs.length} services)
          </summary>
          <div style="max-height:180px; overflow-y:auto; margin-top:6px; border:1px solid var(--border-color); border-radius:4px;">
            <table class="cand-table" style="font-size:11px; margin:0;">
              <thead>
                <tr>
                  <th>Service</th>
                  <th>Text Score</th>
                  <th>Image Similarity</th>
                </tr>
              </thead>
              <tbody>
                ${allSvcs.map((svc) => {
                  const tScore = allTxt[svc] !== undefined ? Number(allTxt[svc]).toFixed(4) : "-";
                  const iScore = allImg[svc] !== undefined ? Number(allImg[svc]).toFixed(4) : "-";
                  const isTop = svc === r.predicted_service;
                  const isExp = svc === r.expected_service;
                  const rowBg = isExp ? "background:rgba(16, 185, 129, 0.12);" : isTop ? "background:rgba(59, 130, 246, 0.12);" : "";
                  return `
                    <tr style="${rowBg}">
                      <td style="font-weight:600;">${escapeHtml(svc)} ${isExp ? '🎯' : ''} ${isTop ? '👑' : ''}</td>
                      <td style="font-family:monospace;">${tScore}</td>
                      <td style="font-family:monospace;">${iScore}</td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
        </details>
      `;
    }

    // Timings
    const tm = r.timings || {};
    let tmHtml = `
      <div style="display:flex;gap:14px;font-size:11px;color:var(--text-secondary);margin-top:8px;">
        <span>Decode: ${tm.image_decode_ms || 0}ms</span>
        <span>SigLIP: ${tm.siglip_encode_ms || 0}ms</span>
        <span>TEI Embed: ${tm.text_embed_ms || 0}ms</span>
        <span>pgvector: ${tm.pgvector_search_ms || 0}ms</span>
        <span>Rerank: ${tm.rerank_ms || 0}ms</span>
        <span>Decision: ${tm.decision_ms || 0}ms</span>
        <span><strong>Total: ${tm.total_ms || 0}ms</strong></span>
      </div>
    `;

    dTr.innerHTML = `
      <td colspan="13">
        <div class="details-panel">
          ${candsHtml}
          ${explicitScoresHtml}
          ${diagHtml}
          ${allScoresHtml}
          ${tmHtml}
        </div>
      </td>
    `;

    const targetRow = document.getElementById(`row-${r.row_id}`);
    if (targetRow) {
      targetRow.after(dTr);
    }
  }

  // ── Table Interactions: Expand, Lightbox, Filter, Sort ─────────────────────
  // ── Table Interactions: Expand, Lightbox, Filter, Sort ─────────────────────
  let modalCurrentRowId = null;

  async function openImageModal(rowId) {
    const row = state.rows.find((r) => r.row_id === rowId);
    if (!row) return;

    modalCurrentRowId = rowId;

    // Set textual metadata
    if (modalRowIdx) modalRowIdx.textContent = `Row #${row.row_id}`;
    if (modalFilename) modalFilename.textContent = row.image;
    if (modalExpectedSvc) {
      modalExpectedSvc.textContent = row.is_none ? "NONE" : row.expected_service;
    }
    if (modalPredictedSvc) {
      modalPredictedSvc.textContent = row.predicted_service || "NONE";
      modalPredictedSvc.style.color = row.correct ? "var(--text-primary)" : "var(--accent-red)";
    }

    // Action badge
    const actClass = row.action === "auto_create" ? "badge-auto" : row.action === "draft" ? "badge-draft" : "badge-no-action";
    const actLabel = row.action === "auto_create" ? "AUTO" : row.action === "draft" ? "DRAFT" : "NO ACTION";
    if (modalActionBadge) {
      modalActionBadge.innerHTML = `<span class="badge ${actClass}">${actLabel}</span>`;
    }

    // Result badge
    if (modalResultBadge) {
      if (row.is_none) {
        modalResultBadge.innerHTML = row.correct
          ? '<span class="badge-correct">✓ CORRECT (NO ACTION)</span>'
          : '<span class="badge-false-match">✗ FALSE MATCH</span>';
      } else {
        modalResultBadge.innerHTML = row.correct
          ? '<span class="badge-correct">✓ CORRECT</span>'
          : '<span class="badge-wrong">✗ WRONG</span>';
      }
    }

    // Confidence & Gap
    const cStr = row.confidence !== null ? `${(row.confidence * 100).toFixed(1)}%` : "-";
    const gStr = row.gap !== null ? ` (gap: ${(row.gap * 100).toFixed(1)}%)` : "";
    if (modalConfGap) {
      modalConfGap.textContent = `${cStr}${gStr}`;
    }

    // Description
    if (modalDescription) {
      modalDescription.textContent = row.description || "<No description - Image only>";
    }

    // Show thumbnail immediately as placeholder while fetching full preview
    if (row.thumbnail_b64) {
      modalImg.src = row.thumbnail_b64;
    } else {
      modalImg.src = "";
    }

    imageModal.style.display = "flex";
    imageModal.focus();

    // Fetch larger preview image from authenticated endpoint
    const jobId = state.currentJobId;
    if (jobId) {
      const cacheKey = `${jobId}_${row.row_id}`;
      if (state.blobUrlCache && state.blobUrlCache.has(cacheKey)) {
        modalImg.src = state.blobUrlCache.get(cacheKey);
        if (modalLoadingSpinner) modalLoadingSpinner.style.display = "none";
      } else {
        if (modalLoadingSpinner) modalLoadingSpinner.style.display = "block";
        try {
          const res = await apiFetch(`/v1/eval/jobs/${jobId}/preview/${row.row_id}`);
          if (res.ok) {
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            if (!state.blobUrlCache) state.blobUrlCache = new Map();
            state.blobUrlCache.set(cacheKey, blobUrl);
            if (modalCurrentRowId === row.row_id) {
              modalImg.src = blobUrl;
            }
          }
        } catch (err) {
          console.warn("Could not fetch full preview:", err);
        } finally {
          if (modalLoadingSpinner) modalLoadingSpinner.style.display = "none";
        }
      }
    }
  }

  function closeImageModal() {
    imageModal.style.display = "none";
    modalCurrentRowId = null;
  }

  function navigateModal(direction) {
    if (modalCurrentRowId === null) return;
    const visibleRows = getFilteredRows();
    if (visibleRows.length === 0) return;

    const currIdx = visibleRows.findIndex((r) => r.row_id === modalCurrentRowId);
    let nextIdx = (currIdx !== -1) ? currIdx + direction : 0;
    if (nextIdx < 0) nextIdx = visibleRows.length - 1;
    if (nextIdx >= visibleRows.length) nextIdx = 0;

    openImageModal(visibleRows[nextIdx].row_id);
  }

  // Keyboard navigation for modal
  window.addEventListener("keydown", (e) => {
    if (imageModal && imageModal.style.display === "flex") {
      if (e.key === "Escape") {
        e.preventDefault();
        closeImageModal();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        navigateModal(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        navigateModal(1);
      }
    }
  });

  if (modalPrevBtn) modalPrevBtn.addEventListener("click", () => navigateModal(-1));
  if (modalNextBtn) modalNextBtn.addEventListener("click", () => navigateModal(1));
  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeImageModal);
  if (imageModal) {
    imageModal.addEventListener("click", (e) => {
      if (e.target === imageModal) {
        closeImageModal();
      }
    });
  }

  resultsTableBody.addEventListener("click", (e) => {
    // Lightbox click on thumbnail
    const thumb = e.target.closest(".row-thumb");
    if (thumb) {
      const rowId = parseInt(thumb.dataset.rowId, 10);
      if (!isNaN(rowId)) {
        openImageModal(rowId);
        return;
      }
    }

    // Details expand toggle
    const btn = e.target.closest(".btn-expand");
    if (btn) {
      const rowId = parseInt(btn.dataset.id, 10);
      if (state.expandedRows.has(rowId)) {
        state.expandedRows.delete(rowId);
        const el = document.getElementById(`details-${rowId}`);
        if (el) el.remove();
        btn.textContent = "Details ▾";
      } else {
        state.expandedRows.add(rowId);
        const row = state.rows.find((r) => r.row_id === rowId);
        if (row) renderDetailsRow(row);
        btn.textContent = "Hide ▴";
      }
    }
  });

  // False Matches table clicks (thumbnails or preview buttons)
  if (noneFalseMatchesTable) {
    noneFalseMatchesTable.addEventListener("click", (e) => {
      const target = e.target.closest(".row-thumb, .btn-preview-modal");
      if (target) {
        const rowId = parseInt(target.dataset.rowId, 10);
        if (!isNaN(rowId)) {
          openImageModal(rowId);
        }
      }
    });
  }

  // Filter Buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeFilter = btn.dataset.filter;
      renderResultsTable();
    });
  });

  // Search input
  tableSearchInput.addEventListener("input", (e) => {
    state.searchQuery = e.target.value.trim();
    renderResultsTable();
  });

  // Sorting
  document.querySelectorAll("#resultsTable th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (state.sortCol === col) {
        state.sortAsc = !state.sortAsc;
      } else {
        state.sortCol = col;
        state.sortAsc = true;
      }
      renderResultsTable();
    });
  });

  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => (c.style.display = "none"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) {
        target.style.display = "block";
        if (btn.dataset.tab === "tabNonePanel") {
          renderNoneHistogram(state.summary?.none_panel?.histogram || []);
        } else if (btn.dataset.tab === "tabCalibration") {
          const activeCalib = (state.summary?.calibration || []).find((c) => c.temperature === state.activeTemp) || (state.summary?.calibration || [])[0];
          if (activeCalib) renderReliabilityDiagram(activeCalib.bins || []);
        }
      }
    });
  });

  // ── "New Evaluation" Button Handler ────────────────────────────────────────
  if (newEvalBtn) {
    newEvalBtn.addEventListener("click", async () => {
      if (state.jobStatus === "running") {
        const confirmed = window.confirm(
          "An evaluation is currently running. Do you want to stop it and start a new evaluation?"
        );
        if (!confirmed) return;
        try {
          if (state.currentJobId) {
            await apiFetch(`/v1/eval/jobs/${state.currentJobId}/stop`, { method: "POST" });
          }
        } catch (e) {
          console.warn("Error stopping job on new eval:", e);
        }
      }

      // Close SSE stream
      if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
      }

      // Revoke any created preview blob URLs
      if (state.blobUrlCache) {
        state.blobUrlCache.forEach((url) => URL.revokeObjectURL(url));
        state.blobUrlCache.clear();
      }

      // Clear internal state
      state.currentJobId = null;
      state.jobStatus = "idle";
      state.rows = [];
      state.summary = null;
      state.validation = null;
      state.activeFilter = "all";
      state.searchQuery = "";
      state.expandedRows.clear();

      // Reset form controls
      zipFileInput.value = "";
      tableSearchInput.value = "";

      // Reset DOM sections
      currentJobIdBadge.textContent = "";
      validationBox.style.display = "none";
      settingsSection.style.display = "none";
      progressSection.style.display = "none";
      summarySection.style.display = "none";
      resultsSection.style.display = "none";
      resultsTableBody.innerHTML = "";

      if (summaryWarnings) summaryWarnings.innerHTML = "";
      if (summaryFindings) summaryFindings.innerHTML = "";

      // Reset Dropzone
      dropzone.innerHTML = `
        <div class="dropzone-icon">📁</div>
        <div class="dropzone-text">Drag & drop your evaluation ZIP file here, or click to browse</div>
        <div class="dropzone-sub">Accepts ZIP containing images (.jpg, .jpeg, .png, .webp) + ground_truth.csv</div>
      `;

      // Show upload section
      uploadSection.style.display = "block";
      uploadSection.scrollIntoView({ behavior: "smooth" });

      // Reset filter buttons
      document.querySelectorAll(".filter-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.filter === "all");
      });

      // Reset and reload past runs list (preserves past runs!)
      pastRunsSelect.value = "";
      loadPastRuns();
    });
  }

  // ── Compare Evaluation Runs (Requirement 3) ───────────────────────────────
  async function runCompare() {
    if (!compareRunASelect || !compareRunBSelect) return;
    const runA = compareRunASelect.value;
    const runB = compareRunBSelect.value;

    if (!runA || !runB) {
      alert("Please select both Baseline Run A and Comparison Run B.");
      return;
    }
    if (runA === runB) {
      alert("Baseline Run A and Comparison Run B must be different runs.");
      return;
    }

    if (runComparisonBtn) {
      runComparisonBtn.disabled = true;
      runComparisonBtn.textContent = "Comparing...";
    }

    try {
      const res = await apiFetch(`/v1/eval/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Run comparison failed.");
      }
      const data = await res.json();
      compareState.data = data;
      compareState.runA = runA;
      compareState.runB = runB;
      renderCompareResults(data);
    } catch (err) {
      alert(`Could not compare runs: ${err.message}`);
    } finally {
      if (runComparisonBtn) {
        runComparisonBtn.disabled = false;
        runComparisonBtn.textContent = "Compare Runs";
      }
    }
  }

  function renderCompareResults(data) {
    if (!compareResultsArea) return;
    compareResultsArea.style.display = "block";

    // Update column headers with Run IDs
    if (compareColRunAHeader) {
      compareColRunAHeader.textContent = `Run A (${data.run_a.job_id})`;
    }
    if (compareColRunBHeader) {
      compareColRunBHeader.textContent = `Run B (${data.run_b.job_id})`;
    }

    // Warnings Box (Dataset / Catalog Mismatch: Requirement 3)
    if (compareWarnings) {
      compareWarnings.innerHTML = "";
      if (data.warnings && data.warnings.length > 0) {
        compareWarnings.style.display = "block";
        data.warnings.forEach((w) => {
          const div = document.createElement("div");
          div.className = "val-warning-pill";
          div.style.marginBottom = "8px";
          div.style.fontSize = "13px";
          div.style.padding = "10px 14px";
          div.textContent = `⚠️ ${w}`;
          compareWarnings.appendChild(div);
        });
      } else {
        compareWarnings.style.display = "none";
      }
    }

    // Render Metric Deltas
    renderCompareDeltas(data.deltas || {});

    // Update flip counters
    if (compareCountAll) compareCountAll.textContent = data.counts?.changed_count || 0;
    if (compareCountImproved) compareCountImproved.textContent = data.counts?.improved_count || 0;
    if (compareCountRegressed) compareCountRegressed.textContent = data.counts?.regressed_count || 0;

    // Render Changed Correctness Rows
    renderCompareChangedRows();
  }

  function renderCompareDeltas(deltas) {
    if (!compareDeltasTableBody) return;
    compareDeltasTableBody.innerHTML = "";

    const metricDefs = [
      { key: "top1_accuracy", label: "Top-1 Accuracy", unit: "%", lowerIsBetter: false },
      { key: "top3_accuracy", label: "Top-3 Accuracy", unit: "%", lowerIsBetter: false },
      { key: "top5_accuracy", label: "Top-5 Accuracy", unit: "%", lowerIsBetter: false },
      { key: "pool_recall", label: "Candidate Pool Recall", unit: "%", lowerIsBetter: false },
      { key: "auto_precision", label: "Auto-Create Precision", unit: "%", lowerIsBetter: false },
      { key: "wrong_auto_count", label: "WRONG-AUTO Count (Harmful)", unit: "", lowerIsBetter: true, isCritical: true },
      { key: "coverage_pct", label: "Auto-Create Coverage", unit: "%", lowerIsBetter: false },
      { key: "none_correct_rate", label: "NONE L1: Correct Suppression Rate", unit: "%", lowerIsBetter: false },
      { key: "none_draft_rate", label: "NONE L2: Draft Rate (Soft Match)", unit: "%", lowerIsBetter: null },
      { key: "none_wrong_auto_count", label: "NONE L3: Harmful Wrong-Auto Count", unit: "", lowerIsBetter: true, isCritical: true },
      { key: "none_false_match_count", label: "Total NONE False Matches", unit: "", lowerIsBetter: true },
      { key: "mean_latency_ms", label: "Mean Total Latency", unit: " ms", lowerIsBetter: true },
      { key: "p50_latency_ms", label: "P50 Latency", unit: " ms", lowerIsBetter: true },
      { key: "p95_latency_ms", label: "P95 Latency", unit: " ms", lowerIsBetter: true },
    ];

    metricDefs.forEach((m) => {
      const item = deltas[m.key];
      if (!item) return;

      const vA = item.a !== undefined ? `${item.a}${m.unit}` : "-";
      const vB = item.b !== undefined ? `${item.b}${m.unit}` : "-";
      const deltaNum = item.delta;

      let deltaClass = "";
      let deltaStr = "-";
      let badge = '<span style="color:var(--text-muted);font-size:11px;">No Change</span>';

      if (deltaNum > 0) {
        deltaStr = `+${deltaNum}${m.unit}`;
        if (m.lowerIsBetter === true) {
          deltaClass = "delta-neg";
          badge = '<span class="badge badge-regressed">Regressed</span>';
        } else if (m.lowerIsBetter === false) {
          deltaClass = "delta-pos";
          badge = '<span class="badge badge-improved">Improved</span>';
        } else {
          badge = '<span style="font-size:11px;color:var(--text-muted);">Shift</span>';
        }
      } else if (deltaNum < 0) {
        deltaStr = `${deltaNum}${m.unit}`;
        if (m.lowerIsBetter === true) {
          deltaClass = "delta-pos";
          badge = '<span class="badge badge-improved">Improved</span>';
        } else if (m.lowerIsBetter === false) {
          deltaClass = "delta-neg";
          badge = '<span class="badge badge-regressed">Regressed</span>';
        } else {
          badge = '<span style="font-size:11px;color:var(--text-muted);">Shift</span>';
        }
      }

      const tr = document.createElement("tr");
      if (m.isCritical) {
        tr.style.background = "rgba(239, 68, 68, 0.06)";
      }
      tr.innerHTML = `
        <td style="font-weight:${m.isCritical ? '700' : '600'}; color:${m.isCritical ? 'var(--accent-red)' : 'var(--text-primary)'};">
          ${escapeHtml(m.label)}
        </td>
        <td style="font-family:monospace;">${vA}</td>
        <td style="font-family:monospace;">${vB}</td>
        <td class="${deltaClass}" style="font-weight:700; font-family:monospace;">${deltaStr}</td>
        <td>${badge}</td>
      `;
      compareDeltasTableBody.appendChild(tr);
    });
  }

  function renderCompareChangedRows() {
    if (!compareChangedTableBody) return;
    compareChangedTableBody.innerHTML = "";

    const data = compareState.data;
    if (!data || !data.changed_rows) return;

    const rows = data.changed_rows.filter((r) => {
      if (compareState.activeFilter === "improved") return r.change_type === "improved";
      if (compareState.activeFilter === "regressed") return r.change_type === "regressed";
      return true;
    });

    if (rows.length === 0) {
      compareChangedTableBody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 22px;">
            ${data.changed_rows.length === 0 ? "✓ Both runs produced identical correctness classifications across all matched rows." : "No rows match the selected filter."}
          </td>
        </tr>
      `;
      return;
    }

    rows.forEach((r, idx) => {
      const tr = document.createElement("tr");

      const thumbHtml = r.thumbnail_b64
        ? `<img src="${r.thumbnail_b64}" class="row-thumb" alt="${escapeHtml(r.image)}" style="cursor: pointer;" data-compare-idx="${idx}" title="Click to view large preview">`
        : `<div class="row-thumb" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:10px;">IMG</div>`;

      const expBadge = r.is_none
        ? '<span class="badge badge-no-action">NONE</span>'
        : `<span style="font-weight:600;">${escapeHtml(r.expected_service)}</span>`;

      const actA = r.run_a.action === "auto_create" ? "badge-auto" : r.run_a.action === "draft" ? "badge-draft" : "badge-no-action";
      const actB = r.run_b.action === "auto_create" ? "badge-auto" : r.run_b.action === "draft" ? "badge-draft" : "badge-no-action";

      const predA = `
        <div style="font-weight:600; color:${r.run_a.correct ? 'var(--text-primary)' : 'var(--accent-red)'};">
          ${escapeHtml(r.run_a.prediction || "NONE")}
        </div>
        <div style="font-size:11px; margin-top:2px;">
          <span class="badge ${actA}" style="font-size:10px; padding:1px 5px;">${r.run_a.action}</span>
          <span style="color:var(--text-muted);">${r.run_a.confidence !== null ? (r.run_a.confidence * 100).toFixed(1) + '%' : '-'}</span>
          ${r.run_a.correct ? '<span style="color:var(--accent-green);font-size:11px;margin-left:4px;">✓</span>' : '<span style="color:var(--accent-red);font-size:11px;margin-left:4px;">✗</span>'}
        </div>
      `;

      const predB = `
        <div style="font-weight:600; color:${r.run_b.correct ? 'var(--text-primary)' : 'var(--accent-red)'};">
          ${escapeHtml(r.run_b.prediction || "NONE")}
        </div>
        <div style="font-size:11px; margin-top:2px;">
          <span class="badge ${actB}" style="font-size:10px; padding:1px 5px;">${r.run_b.action}</span>
          <span style="color:var(--text-muted);">${r.run_b.confidence !== null ? (r.run_b.confidence * 100).toFixed(1) + '%' : '-'}</span>
          ${r.run_b.correct ? '<span style="color:var(--accent-green);font-size:11px;margin-left:4px;">✓</span>' : '<span style="color:var(--accent-red);font-size:11px;margin-left:4px;">✗</span>'}
        </div>
      `;

      const changeBadge = r.change_type === "improved"
        ? '<span class="badge badge-improved">✓ A wrong &rarr; B right</span>'
        : '<span class="badge badge-regressed">✗ A right &rarr; B wrong</span>';

      tr.innerHTML = `
        <td>${thumbHtml}</td>
        <td style="font-family:monospace; font-size:11px;" title="${escapeHtml(r.description || '')}">
          ${escapeHtml(r.image)}
          ${r.description ? `<div style="color:var(--text-muted); font-size:10px; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(r.description)}</div>` : ''}
        </td>
        <td>${expBadge}</td>
        <td>${predA}</td>
        <td>${predB}</td>
        <td>${changeBadge}</td>
      `;

      compareChangedTableBody.appendChild(tr);
    });

    // Wire thumbnail preview clicks
    compareChangedTableBody.querySelectorAll("img.row-thumb").forEach((img) => {
      img.addEventListener("click", () => {
        const idx = parseInt(img.dataset.compareIdx, 10);
        const cr = rows[idx];
        if (cr) openCompareImageModal(cr);
      });
    });
  }

  function openCompareImageModal(cr) {
    if (modalRowIdx) modalRowIdx.textContent = cr.change_label;
    if (modalFilename) modalFilename.textContent = cr.image;
    if (modalExpectedSvc) modalExpectedSvc.textContent = cr.is_none ? "NONE" : cr.expected_service;
    if (modalPredictedSvc) {
      modalPredictedSvc.innerHTML = `Run A: <strong>${escapeHtml(cr.run_a.prediction || "NONE")}</strong> (${cr.run_a.correct ? "✓ Correct" : "✗ Wrong"})<br>` +
                                    `Run B: <strong>${escapeHtml(cr.run_b.prediction || "NONE")}</strong> (${cr.run_b.correct ? "✓ Correct" : "✗ Wrong"})`;
      modalPredictedSvc.style.color = "var(--text-primary)";
    }
    if (modalConfGap) {
      modalConfGap.textContent = `A: ${cr.run_a.confidence !== null ? (cr.run_a.confidence * 100).toFixed(1) + '%' : '-'} (${cr.run_a.action}) | ` +
                                 `B: ${cr.run_b.confidence !== null ? (cr.run_b.confidence * 100).toFixed(1) + '%' : '-'} (${cr.run_b.action})`;
    }
    if (modalDescription) modalDescription.textContent = cr.description || "<No description>";
    if (modalActionBadge) {
      modalActionBadge.innerHTML = `<span class="badge ${cr.change_type === 'improved' ? 'badge-improved' : 'badge-regressed'}">${escapeHtml(cr.change_label)}</span>`;
    }
    if (modalResultBadge) {
      modalResultBadge.innerHTML = cr.change_type === "improved"
        ? '<span class="badge-correct">✓ IMPROVED IN B</span>'
        : '<span class="badge-wrong">✗ REGRESSED IN B</span>';
    }
    if (modalImg) modalImg.src = cr.thumbnail_b64 || "";
    if (modalLoadingSpinner) modalLoadingSpinner.style.display = "none";

    if (imageModal) {
      imageModal.style.display = "flex";
      imageModal.focus();
    }
  }

  // ── Export Handlers ────────────────────────────────────────────────────────
  exportCsvBtn.addEventListener("click", () => {
    if (!state.currentJobId) return;
    downloadFile(`/v1/eval/runs/${state.currentJobId}/export/csv`);
  });

  exportJsonBtn.addEventListener("click", () => {
    if (!state.currentJobId) return;
    downloadFile(`/v1/eval/runs/${state.currentJobId}/export/json`);
  });

  exportMdBtn.addEventListener("click", () => {
    if (!state.currentJobId) return;
    downloadFile(`/v1/eval/runs/${state.currentJobId}/export/md`);
  });

  copyMdBtn.addEventListener("click", async () => {
    if (!state.currentJobId) return;
    try {
      const res = await apiFetch(`/v1/eval/runs/${state.currentJobId}/export/md`);
      if (!res.ok) throw new Error("Could not fetch markdown report.");
      const text = await res.text();
      await navigator.clipboard.writeText(text);
      copyMdBtn.textContent = "✓ Copied!";
      setTimeout(() => (copyMdBtn.textContent = "📋 Copy Report"), 2000);
    } catch (e) {
      alert(`Failed to copy: ${e.message}`);
    }
  });

  function downloadFile(url) {
    if (state.apiKey) {
      // Use fetch + blob for authorization support
      apiFetch(url)
        .then((res) => {
          if (!res.ok) throw new Error("Export download failed.");
          const disp = res.headers.get("Content-Disposition");
          let filename = "export";
          if (disp && disp.includes("filename=")) {
            filename = disp.split("filename=")[1].replace(/["']/g, "").trim();
          }
          return res.blob().then((blob) => ({ blob, filename }));
        })
        .then(({ blob, filename }) => {
          const a = document.createElement("a");
          const objUrl = window.URL.createObjectURL(blob);
          a.href = objUrl;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
          window.URL.revokeObjectURL(objUrl);
        })
        .catch((e) => alert(e.message));
    } else {
      window.location.href = url;
    }
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  window.addEventListener("DOMContentLoaded", init);
})();

