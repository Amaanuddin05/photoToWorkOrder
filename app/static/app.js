/**
 * Photo-to-Work-Order — Operator Console Logic
 * Plain vanilla JavaScript, zero dependencies, strictly safe DOM manipulation (textContent only).
 * Adapted for /v1 API endpoints with session-based Bearer token authentication.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    selectedFile: null,
    selectedFileUrl: null,
    submittedDesc: "",
    submittedLoc: "",
    currentDecision: null,
    catalogServices: [],
    demoServices: [],
  };

  // Known emergency services requiring mandatory human review
  const EMERGENCY_SERVICES = [
    "Emergency Response & Area Securing",
    "Emergency Reactive Cleaning (Spills, Vomit, Incidents)",
  ];

  // DOM Elements
  const els = {
    // Header & Auth
    apiKeyInput: document.getElementById("api-key-input"),
    saveKeyBtn: document.getElementById("save-key-btn"),
    systemStatus: document.getElementById("system-status"),
    statusText: document.getElementById("status-text"),

    // Screen 1: New Post
    screenNewPost: document.getElementById("screen-new-post"),
    postForm: document.getElementById("post-form"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    previewContainer: document.getElementById("preview-container"),
    previewImg: document.getElementById("preview-img"),
    removePhotoBtn: document.getElementById("remove-photo-btn"),
    photoError: document.getElementById("photo-error"),
    descInput: document.getElementById("desc-input"),
    noDescNotice: document.getElementById("no-desc-notice"),
    locationInput: document.getElementById("location-input"),
    submitBtn: document.getElementById("submit-btn"),
    submitSpinner: document.getElementById("submit-spinner"),
    submitBtnText: document.getElementById("submit-btn-text"),
    formError: document.getElementById("form-error"),

    // Screen 2: Result
    screenResult: document.getElementById("screen-result"),
    actionBadge: document.getElementById("action-badge"),
    shadowNote: document.getElementById("shadow-note"),
    agreeBadge: document.getElementById("agree-badge"),
    restartTopBtn: document.getElementById("restart-top-btn"),
    reasonText: document.getElementById("reason-text"),
    submittedImgWrapper: document.getElementById("submitted-img-wrapper"),
    submittedImg: document.getElementById("submitted-img"),
    submittedNoPhoto: document.getElementById("submitted-no-photo"),
    submittedDescText: document.getElementById("submitted-desc-text"),
    submittedLocTitle: document.getElementById("submitted-loc-title"),
    submittedLocText: document.getElementById("submitted-loc-text"),
    decisionDetails: document.getElementById("decision-details"),
    candidateTbody: document.getElementById("candidate-tbody"),
    toggleAddServiceBtn: document.getElementById("toggle-add-service-btn"),
    resultError: document.getElementById("result-error"),

    // Screen 3: History
    historyTable: document.getElementById("history-table"),
    historyTbody: document.getElementById("history-tbody"),
    historyEmpty: document.getElementById("history-empty"),
    refreshHistoryBtn: document.getElementById("refresh-history-btn"),

    // Screen 4: Add Demo Service
    screenAddService: document.getElementById("screen-add-service"),
    addServiceForm: document.getElementById("add-service-form"),
    svcTitle: document.getElementById("svc-title"),
    titleCharCounter: document.getElementById("title-char-counter"),
    svcDesc: document.getElementById("svc-desc"),
    descCharCounter: document.getElementById("desc-char-counter"),
    svcKeywords: document.getElementById("svc-keywords"),
    svcVisual: document.getElementById("svc-visual"),
    visualCharCounter: document.getElementById("visual-char-counter"),
    noVisualWarning: document.getElementById("no-visual-warning"),
    svcSimBox: document.getElementById("svc-sim-box"),
    svcSimStatus: document.getElementById("svc-sim-status"),
    svcSimList: document.getElementById("svc-sim-list"),
    svcReactivateBox: document.getElementById("svc-reactivate-box"),
    svcReactivateMsg: document.getElementById("svc-reactivate-msg"),
    svcReactivateBtn: document.getElementById("svc-reactivate-btn"),
    svcErrorAlert: document.getElementById("svc-error-alert"),
    svcSuccessAlert: document.getElementById("svc-success-alert"),
    saveSvcBtn: document.getElementById("save-svc-btn"),
    saveSvcSpinner: document.getElementById("save-svc-spinner"),
    saveSvcBtnText: document.getElementById("save-svc-btn-text"),
    checkSimBtn: document.getElementById("check-sim-btn"),
    checkSimSpinner: document.getElementById("check-sim-spinner"),
    clearSvcBtn: document.getElementById("clear-svc-btn"),

    // Screen 5: Demo Services
    screenDemoServices: document.getElementById("screen-demo-services"),
    demoCountBadge: document.getElementById("demo-count-badge"),
    refreshDemosBtn: document.getElementById("refresh-demos-btn"),
    deleteAllDemosBtn: document.getElementById("delete-all-demos-btn"),
    demoServicesTable: document.getElementById("demo-services-table"),
    demoServicesTbody: document.getElementById("demo-services-tbody"),
    demosEmpty: document.getElementById("demos-empty"),

    // Feedback
    toast: document.getElementById("toast-notification"),
  };

  // Helper: Get Request Headers
  function getHeaders(extraHeaders = {}) {
    const headers = { ...extraHeaders };
    const savedKey = (sessionStorage.getItem("siglip_api_key") || "").trim();
    const inputKey = els.apiKeyInput ? els.apiKeyInput.value.trim() : "";
    const key = savedKey || inputKey;
    if (key) {
      headers["Authorization"] = `Bearer ${key}`;
    }
    return headers;
  }

  // Toast Helper
  let toastTimer = null;
  function showToast(message) {
    if (toastTimer) {
      clearTimeout(toastTimer);
    }
    els.toast.textContent = message;
    els.toast.classList.add("show");
    toastTimer = setTimeout(() => {
      els.toast.classList.remove("show");
    }, 3000);
  }

  // Error message formatting helper (safely extracts error strings from FastAPI dicts/arrays, preventing [object Object])
  function formatErrorMessage(errData, fallbackMessage = "An unexpected error occurred.") {
    if (!errData) return fallbackMessage;

    if (errData.detail !== undefined && errData.detail !== null) {
      if (typeof errData.detail === "string") {
        return errData.detail;
      }
      if (Array.isArray(errData.detail)) {
        const msgs = errData.detail.map((item) => {
          const loc = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
          return (loc ? `${loc}: ` : "") + (item.msg || JSON.stringify(item));
        });
        return msgs.join("; ") || fallbackMessage;
      }
      if (typeof errData.detail === "object") {
        if (errData.detail.message) {
          return errData.detail.message;
        }
        return JSON.stringify(errData.detail);
      }
    }

    if (errData.message) return errData.message;
    if (errData.error && errData.error.message) return errData.error.message;
    if (typeof errData === "string") return errData;
    return fallbackMessage;
  }

  // Reason mapping helper (maps raw/server reasons to readable human sentences)
  function formatReason(rawReason) {
    if (!rawReason) {
      return "No specific triage notes available.";
    }
    const lower = rawReason.toLowerCase();

    if (lower.includes("description missing") || lower.includes("no_description")) {
      return "No description was provided. This post requires manual operator review (ranked by image only).";
    }
    if (lower.includes("word(s)") || lower.includes("short_description") || lower.includes("words required")) {
      return "Description is very brief (fewer than 3 words). Manual review is required.";
    }
    if (lower.includes("mandatory human") || lower.includes("emergency") || lower.includes("safety")) {
      return "Emergency or safety-critical service detected: mandatory human review required.";
    }
    if (lower.includes("below floor") || lower.includes("below_threshold") || lower.includes("confidence is below")) {
      return "Confidence score is below the floor threshold. Routing to manual review.";
    }
    if (lower.includes("shadow mode")) {
      return "Shadow mode active: automatic creation criteria met, routed to draft for operator verification.";
    }
    if (lower.includes("high confidence")) {
      return "High confidence and score separation across candidate services.";
    }
    if (lower.includes("criteria for auto_create not met")) {
      return "Score confidence or separation did not meet automatic creation thresholds.";
    }
    return rawReason;
  }

  // File Validation
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

  function validateFile(file) {
    if (!file) return { valid: false, error: "No file selected." };

    if (file.size > MAX_FILE_SIZE) {
      return {
        valid: false,
        error: `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed is 10 MB.`,
      };
    }

    const name = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));
    const hasValidMime = ALLOWED_MIME_TYPES.includes(file.type);

    if (!hasValidMime && !hasValidExt) {
      return {
        valid: false,
        error: "Invalid file type. Only JPEG, PNG, and WebP images are supported.",
      };
    }

    return { valid: true };
  }

  function handleFileSelection(file) {
    clearPhotoError();

    const validation = validateFile(file);
    if (!validation.valid) {
      showPhotoError(validation.error);
      clearPhotoInput();
      updateSubmitButtonState();
      return;
    }

    if (state.selectedFileUrl) {
      URL.revokeObjectURL(state.selectedFileUrl);
      state.selectedFileUrl = null;
    }

    state.selectedFile = file;
    state.selectedFileUrl = URL.createObjectURL(file);

    els.previewImg.src = state.selectedFileUrl;
    els.previewContainer.style.display = "block";
    els.dropZone.style.display = "none";

    updateSubmitButtonState();
  }

  function clearPhotoInput() {
    if (state.selectedFileUrl) {
      URL.revokeObjectURL(state.selectedFileUrl);
      state.selectedFileUrl = null;
    }
    state.selectedFile = null;
    els.fileInput.value = "";
    els.previewImg.removeAttribute("src");
    els.previewContainer.style.display = "none";
    els.dropZone.style.display = "block";
    updateSubmitButtonState();
  }

  function showPhotoError(msg) {
    els.photoError.textContent = msg;
    els.photoError.style.display = "block";
  }

  function clearPhotoError() {
    els.photoError.textContent = "";
    els.photoError.style.display = "none";
  }

  function showFormError(msg) {
    els.formError.textContent = msg;
    els.formError.style.display = "block";
  }

  function clearFormError() {
    els.formError.textContent = "";
    els.formError.style.display = "none";
  }

  function showResultError(msg) {
    els.resultError.textContent = msg;
    els.resultError.style.display = "block";
  }

  function clearResultError() {
    els.resultError.textContent = "";
    els.resultError.style.display = "none";
  }

  function updateSubmitButtonState() {
    const hasPhoto = state.selectedFile !== null;
    const descText = els.descInput.value.trim();
    const hasDesc = descText.length > 0;

    if (hasDesc) {
      els.noDescNotice.style.display = "none";
    } else {
      els.noDescNotice.style.display = "flex";
    }

    els.submitBtn.disabled = !hasPhoto && !hasDesc;
  }

  // Drag and Drop Events
  ["dragenter", "dragover"].forEach((eventName) => {
    els.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.classList.remove("dragover");
    });
  });

  els.dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelection(dt.files[0]);
    }
  });

  els.dropZone.addEventListener("click", () => {
    els.fileInput.click();
  });

  els.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });

  els.fileInput.addEventListener("change", () => {
    if (els.fileInput.files && els.fileInput.files.length > 0) {
      handleFileSelection(els.fileInput.files[0]);
    }
  });

  els.removePhotoBtn.addEventListener("click", () => {
    clearPhotoInput();
  });

  els.descInput.addEventListener("input", () => {
    updateSubmitButtonState();
  });

  // Submit New Post -> POST /v1/posts/process
  els.postForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFormError();

    const descText = els.descInput.value.trim();
    const locText = els.locationInput.value.trim();
    const hasPhoto = state.selectedFile !== null;
    const hasDesc = descText.length > 0;

    if (!hasPhoto && !hasDesc) {
      showFormError("Please provide a photo or a description.");
      return;
    }

    state.submittedDesc = descText;
    state.submittedLoc = locText;

    els.submitBtn.disabled = true;
    els.submitSpinner.style.display = "inline-block";
    els.submitBtnText.textContent = "Analyzing...";
    els.descInput.disabled = true;
    els.locationInput.disabled = true;
    els.removePhotoBtn.disabled = true;

    const formData = new FormData();
    if (hasPhoto) {
      formData.append("image", state.selectedFile, state.selectedFile.name);
    }
    if (hasDesc) {
      formData.append("description", descText);
    }
    if (locText) {
      formData.append("location", locText);
    }
    const agreeModeEl = document.getElementById("agree-mode-input");
    if (agreeModeEl && agreeModeEl.value) {
      formData.append("agree_mode", agreeModeEl.value);  // test-only per-request override
    }
    const intentEl = document.getElementById("intent-input");
    if (intentEl) {
      formData.append("intent_enabled", intentEl.checked ? "on" : "off");  // test-only per-request override
    }
    formData.append("post_id", `post_${Date.now()}`);

    try {
      const resp = await fetch("/v1/posts/process", {
        method: "POST",
        headers: getHeaders(),
        body: formData,
      });

      if (!resp.ok) {
        let errDetail = `Server error (${resp.status})`;
        try {
          const errData = await resp.json();
          if (errData && errData.detail) {
            errDetail = typeof errData.detail === "string" ? errData.detail : (errData.detail.message || JSON.stringify(errData.detail));
          } else if (errData && errData.error && errData.error.message) {
            errDetail = errData.error.message;
          }
        } catch (_) {}
        throw new Error(errDetail);
      }

      const decision = await resp.json();
      state.currentDecision = decision;

      renderResultScreen(decision);
    } catch (err) {
      showFormError(err.message || "Failed to analyze post. Please check connection.");
    } finally {
      els.submitSpinner.style.display = "none";
      els.submitBtnText.textContent = "Analyze Post";
      els.descInput.disabled = false;
      els.locationInput.disabled = false;
      els.removePhotoBtn.disabled = false;
      updateSubmitButtonState();
    }
  });

  // ── Result view helpers ──────────────────────────────────────────────
  const fmt = (v, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
  const fmtPct = (v) => (typeof v === "number" && isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
  const fmtRank = (v) => (typeof v === "number" && isFinite(v) ? `#${Math.round(v)}` : "—");

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function addDetail(label, value, cls) {
    const tr = document.createElement("tr");
    tr.appendChild(el("th", label));
    const td = el("td", value, cls);
    tr.appendChild(td);
    els.decisionDetails.appendChild(tr);
    return td;
  }

  function passFail(value, min, label) {
    if (typeof value !== "number") return ["—", "muted"];
    const ok = value >= min;
    return [`${fmt(value, 3)} (${ok ? "≥" : "<"} ${min} ${label})`, ok ? "ok" : "bad"];
  }

  // Render Screen 2 (Result View)
  function renderResultScreen(decision) {
    clearResultError();

    // 1. Action badge + "would be" note
    els.actionBadge.className = "action-badge";
    const labels = { auto_create: "Auto-create", no_action: "No action", draft: "Draft" };
    els.actionBadge.textContent = labels[decision.action] || decision.action;
    els.actionBadge.classList.add(
      decision.action === "auto_create" ? "badge-auto-create" : decision.action === "no_action" ? "badge-no-action" : "badge-draft"
    );
    if (decision.would_be_action && decision.would_be_action !== decision.action) {
      els.shadowNote.textContent = `would be: ${labels[decision.would_be_action] || decision.would_be_action}${decision.shadow ? " (shadow mode)" : ""}`;
      els.shadowNote.style.display = "inline";
    } else {
      els.shadowNote.style.display = "none";
    }

    // 2. Agreement badge
    if (decision.agree === true || decision.agree === false) {
      els.agreeBadge.style.display = "inline-flex";
      els.agreeBadge.className = `agree-badge ${decision.agree ? "agree-true" : "agree-false"}`;
      els.agreeBadge.textContent = decision.agree ? "✓ Text and photo agree" : "⚠ Text and photo differ";
    } else {
      els.agreeBadge.style.display = "none";
    }

    // 3. Reason
    els.reasonText.textContent = formatReason(decision.reason);

    // 4. Submitted input
    if (state.selectedFileUrl) {
      els.submittedImg.src = state.selectedFileUrl;
      els.submittedImgWrapper.style.display = "block";
      els.submittedNoPhoto.style.display = "none";
    } else {
      els.submittedImgWrapper.style.display = "none";
      els.submittedNoPhoto.style.display = "block";
    }
    els.submittedDescText.textContent = state.submittedDesc || "(No description provided)";
    const hasLoc = !!state.submittedLoc;
    els.submittedLocTitle.style.display = hasLoc ? "block" : "none";
    els.submittedLocText.style.display = hasLoc ? "block" : "none";
    els.submittedLocText.textContent = state.submittedLoc || "";

    // 5. Decision details (key / value)
    els.decisionDetails.replaceChildren();
    const top = (decision.top_candidates || [])[0];
    addDetail("Decision ID", decision.decision_id != null ? `#${decision.decision_id}` : "—");
    addDetail("Top service", top ? top.title : "—");
    const [cTxt, cCls] = passFail(decision.confidence, 0.4, "auto threshold");
    addDetail("Confidence", cTxt, cCls);
    const [gTxt, gCls] = passFail(decision.gap, 0.1, "gap threshold");
    addDetail("Gap to #2", gTxt, gCls);
    const tg = decision.text_guard;
    if (tg && typeof tg.score === "number") {
      const [tTxt, tCls] = passFail(tg.score, tg.auto_min, `auto min, floor ${tg.no_action_floor}`);
      addDetail("Text match score", tTxt, tCls);
    } else {
      addDetail("Text match score", "— (no description)", "muted");
    }
    addDetail("Text #1", decision.text_top1 || "—");
    addDetail("Photo #1", decision.image_top1 || "—");
    const agreeTxt = decision.agree === true ? "Yes" : decision.agree === false ? "No" : "— (photo or text missing)";
    addDetail("Agreement", `${agreeTxt} · strictness: ${decision.agree_mode || "strict"}`,
      decision.agree === true ? "ok" : decision.agree === false ? "bad" : "muted");
    const it = decision.intent;
    if (it) {
      const names = { issue: "Issue report", done: "Work completed", other: "Not maintenance" };
      addDetail("Message intent", `${names[it.label] || it.label} (${fmtPct(it.prob)})${it.acted ? " · blocked work order" : ""}`,
        it.acted ? "bad" : "ok");
    } else {
      addDetail("Message intent", "— (gate off or no description)", "muted");
    }
    const t = decision.timings || {};
    addDetail("Latency", `${fmt((t.total_ms || 0) / 1000, 2)} s  (text ${fmt(t.text_ms, 0)} ms · photo ${fmt(t.image_ms, 0)} ms)`);
    if (decision.idempotent_replay) addDetail("Replay", "Same post seen before: stored result returned", "muted");

    // 6. Candidate scores table (whole pool, sorted by confidence)
    els.candidateTbody.replaceChildren();
    const cands = (decision.all_candidates && decision.all_candidates.length ? decision.all_candidates : decision.top_candidates || [])
      .slice()
      .sort((x, y) => (y.confidence || 0) - (x.confidence || 0));
    cands.forEach((c, idx) => {
      const tr = document.createElement("tr");
      if (idx === 0) tr.className = "top-row";
      tr.appendChild(el("td", String(idx + 1), "num"));
      const tdName = el("td", c.title || `Service #${c.service_id}`);
      if (c.title === decision.text_top1) tdName.appendChild(el("span", "text #1", "tag"));
      if (c.title === decision.image_top1) tdName.appendChild(el("span", "photo #1", "tag"));
      tr.appendChild(tdName);
      const tdConf = document.createElement("td");
      const wrap = el("div", null, "conf-cell");
      const bar = el("div", null, "conf-bar");
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, (c.confidence || 0) * 100))}%`;
      bar.appendChild(fill);
      wrap.appendChild(bar);
      wrap.appendChild(el("span", fmtPct(c.confidence)));
      tdConf.appendChild(wrap);
      tr.appendChild(tdConf);
      tr.appendChild(el("td", fmt(c.text_score, 3), "num"));
      tr.appendChild(el("td", fmtRank(c.text_rank), "num"));
      tr.appendChild(el("td", fmtRank(c.image_rank), "num"));
      els.candidateTbody.appendChild(tr);
    });
    if (!cands.length) {
      const tr = document.createElement("tr");
      const td = el("td", "No candidates", "muted");
      td.colSpan = 6;
      tr.appendChild(td);
      els.candidateTbody.appendChild(tr);
    }

    els.screenNewPost.style.display = "none";
    els.screenResult.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // Reset & Next Post
  function resetFormAndGoToScreen1() {
    clearPhotoInput();
    els.descInput.value = "";
    els.locationInput.value = "";
    state.submittedDesc = "";
    state.submittedLoc = "";
    state.currentDecision = null;
    clearFormError();
    clearResultError();

    updateSubmitButtonState();
    els.screenResult.style.display = "none";
    els.screenNewPost.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  els.restartTopBtn.addEventListener("click", resetFormAndGoToScreen1);

  // Screen 3: History Table (GET /v1/decisions?limit=20)
  async function fetchHistory() {
    try {
      const resp = await fetch("/v1/decisions?limit=20", {
        headers: getHeaders(),
      });
      if (!resp.ok) return;

      const decisions = await resp.json();
      renderHistoryTable(decisions);
    } catch (err) {
      console.warn("Failed to load decision history:", err);
    }
  }

  function renderHistoryTable(decisions) {
    while (els.historyTbody.firstChild) {
      els.historyTbody.removeChild(els.historyTbody.firstChild);
    }

    if (!decisions || decisions.length === 0) {
      els.historyTable.style.display = "none";
      els.historyEmpty.style.display = "block";
      return;
    }

    els.historyTable.style.display = "table";
    els.historyEmpty.style.display = "none";

    decisions.forEach((row) => {
      const tr = document.createElement("tr");

      // 1. Time
      const tdTime = document.createElement("td");
      const timeVal = row.created_at || row.createdAt;
      if (timeVal) {
        try {
          const d = new Date(timeVal);
          tdTime.textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
          tdTime.title = d.toLocaleString();
        } catch (_) {
          tdTime.textContent = String(timeVal);
        }
      } else {
        tdTime.textContent = "-";
      }
      tr.appendChild(tdTime);

      // 2. Description (truncated, full text on hover)
      const tdDesc = document.createElement("td");
      tdDesc.className = "td-desc";
      const descVal = row.description ? row.description.trim() : "";
      if (descVal) {
        tdDesc.textContent = descVal.length > 35 ? `${descVal.substring(0, 35)}...` : descVal;
        tdDesc.title = descVal;
      } else {
        tdDesc.textContent = "(none)";
        tdDesc.title = "No description provided";
      }
      tr.appendChild(tdDesc);

      // 3. Suggested Service
      const tdSuggested = document.createElement("td");
      const topTitle = row.top_service_title || row.topServiceTitle;
      const topId = row.top_service_id != null ? row.top_service_id : row.topServiceId;
      tdSuggested.textContent = topTitle || (topId ? `Service #${topId}` : "-");
      tr.appendChild(tdSuggested);

      // 4. Confidence
      tr.appendChild(el("td", fmtPct(row.confidence)));

      // 6. Action Badge
      const tdAction = document.createElement("td");
      const actionBadge = document.createElement("span");
      actionBadge.className = "action-badge";
      if (row.action === "auto_create") {
        actionBadge.textContent = "Auto-create";
        actionBadge.classList.add("badge-auto-create");
      } else if (row.action === "no_action") {
        actionBadge.textContent = "No action";
        actionBadge.classList.add("badge-no-action");
      } else {
        actionBadge.textContent = "Draft";
        actionBadge.classList.add("badge-draft");
      }
      tdAction.appendChild(actionBadge);
      tr.appendChild(tdAction);

      // 7. Latency in seconds
      const tdLatency = document.createElement("td");
      const lat = row.latency_ms != null ? row.latency_ms : row.latencyMs;
      if (typeof lat === "number") {
        tdLatency.textContent = `${(lat / 1000).toFixed(2)}s`;
      } else {
        tdLatency.textContent = "-";
      }
      tr.appendChild(tdLatency);

      els.historyTbody.appendChild(tr);
    });
  }

  els.refreshHistoryBtn.addEventListener("click", () => {
    fetchHistory();
  });

  // Catalog Services loader -> GET /v1/services
  async function fetchCatalogServices() {
    try {
      const resp = await fetch("/v1/services", {
        headers: getHeaders(),
      });
      if (!resp.ok) return;

      const services = await resp.json();
      state.catalogServices = services || [];
    } catch (err) {
      console.warn("Could not fetch catalog services:", err);
    }
  }

  // --- Auth Handler ---
  function initAuth() {
    if (!els.apiKeyInput) return;
    const savedKey = sessionStorage.getItem("siglip_api_key") || "";
    if (savedKey) {
      els.apiKeyInput.value = savedKey;
    }
    if (els.saveKeyBtn) {
      els.saveKeyBtn.addEventListener("click", () => {
        const val = els.apiKeyInput.value.trim();
        if (val) {
          sessionStorage.setItem("siglip_api_key", val);
          showToast("API Key saved for session");
        } else {
          sessionStorage.removeItem("siglip_api_key");
          showToast("API Key cleared");
        }
      });
      els.apiKeyInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          els.saveKeyBtn.click();
        }
      });
    }
  }

  // --- Render Similar Services Helper ---
  function renderSimilarServices(similarServices, highestScore) {
    if (!els.svcSimBox || !els.svcSimList) return;
    els.svcSimList.innerHTML = "";
    if (!similarServices || similarServices.length === 0) {
      els.svcSimBox.style.display = "none";
      return;
    }
    els.svcSimBox.style.display = "block";
    if (highestScore > 0.85) {
      if (els.svcSimStatus) {
        els.svcSimStatus.textContent = "High similarity (> 85%) detected";
        els.svcSimStatus.style.color = "#dc2626";
      }
    } else {
      if (els.svcSimStatus) {
        els.svcSimStatus.textContent = "Top active catalog matches";
        els.svcSimStatus.style.color = "var(--text-muted)";
      }
    }

    similarServices.forEach((item) => {
      const itemEl = document.createElement("div");
      itemEl.className = "sim-item";

      const titleSpan = document.createElement("span");
      titleSpan.textContent = `#${item.id} ${item.title}`;
      itemEl.appendChild(titleSpan);

      const scoreSpan = document.createElement("span");
      const pct = (item.score * 100).toFixed(1);
      scoreSpan.textContent = `${pct}%`;
      if (item.score > 0.85) {
        scoreSpan.className = "sim-badge-high";
      } else {
        scoreSpan.style.color = "var(--text-secondary)";
        scoreSpan.style.fontWeight = "600";
      }
      itemEl.appendChild(scoreSpan);
      els.svcSimList.appendChild(itemEl);
    });
  }

  // --- Add Demo Service Form Handler ---
  function initAddService() {
    if (!els.addServiceForm) return;

    function updateCounters() {
      if (els.svcTitle && els.titleCharCounter) {
        const len = els.svcTitle.value.length;
        els.titleCharCounter.textContent = `${len} / 200`;
        els.titleCharCounter.classList.toggle("near-limit", len >= 180);
      }
      if (els.svcDesc && els.descCharCounter) {
        const len = els.svcDesc.value.length;
        els.descCharCounter.textContent = `${len} / 1000`;
        els.descCharCounter.classList.toggle("near-limit", len >= 900);
      }
      if (els.svcVisual && els.visualCharCounter) {
        const len = els.svcVisual.value.length;
        els.visualCharCounter.textContent = `${len} / 500`;
        els.visualCharCounter.classList.toggle("near-limit", len >= 450);
        if (els.noVisualWarning) {
          els.noVisualWarning.style.display = len === 0 ? "block" : "none";
        }
      }
    }

    if (els.svcTitle) els.svcTitle.addEventListener("input", updateCounters);
    if (els.svcDesc) els.svcDesc.addEventListener("input", updateCounters);
    if (els.svcVisual) els.svcVisual.addEventListener("input", updateCounters);
    updateCounters();

    // Clear form
    if (els.clearSvcBtn) {
      els.clearSvcBtn.addEventListener("click", () => {
        els.addServiceForm.reset();
        updateCounters();
        if (els.svcSimBox) els.svcSimBox.style.display = "none";
        if (els.svcReactivateBox) els.svcReactivateBox.style.display = "none";
        if (els.svcErrorAlert) els.svcErrorAlert.style.display = "none";
        if (els.svcSuccessAlert) els.svcSuccessAlert.style.display = "none";
      });
    }

    // Check similarity button
    if (els.checkSimBtn) {
      els.checkSimBtn.addEventListener("click", async () => {
        const title = (els.svcTitle ? els.svcTitle.value : "").trim();
        const desc = (els.svcDesc ? els.svcDesc.value : "").trim();

        if (els.svcErrorAlert) els.svcErrorAlert.style.display = "none";
        if (!title || !desc) {
          if (els.svcErrorAlert) {
            els.svcErrorAlert.textContent = "Please enter both Service Name and Description to check similarity.";
            els.svcErrorAlert.style.display = "block";
          }
          return;
        }

        try {
          els.checkSimBtn.disabled = true;
          if (els.checkSimSpinner) els.checkSimSpinner.style.display = "inline-block";

          const resp = await fetch("/v1/services/check-similarity", {
            method: "POST",
            headers: getHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ title, description: desc }),
          });

          const data = await resp.json().catch(() => ({}));
          if (!resp.ok) {
            if (els.svcErrorAlert) {
              els.svcErrorAlert.textContent = formatErrorMessage(data, "Failed to check similarity.");
              els.svcErrorAlert.style.display = "block";
            }
            return;
          }

          renderSimilarServices(data.similar_services, data.highest_score || 0);
        } catch (err) {
          console.error("Check similarity error:", err);
          if (els.svcErrorAlert) {
            els.svcErrorAlert.textContent = "Network error while checking similarity.";
            els.svcErrorAlert.style.display = "block";
          }
        } finally {
          els.checkSimBtn.disabled = false;
          if (els.checkSimSpinner) els.checkSimSpinner.style.display = "none";
        }
      });
    }

    // Submit / Save
    async function submitService(reactivateIfExists = false) {
      const title = (els.svcTitle ? els.svcTitle.value : "").trim();
      const desc = (els.svcDesc ? els.svcDesc.value : "").trim();
      const keywords = (els.svcKeywords ? els.svcKeywords.value : "").trim();
      const visual = (els.svcVisual ? els.svcVisual.value : "").trim();

      if (els.svcErrorAlert) els.svcErrorAlert.style.display = "none";
      if (els.svcSuccessAlert) els.svcSuccessAlert.style.display = "none";
      if (els.svcReactivateBox) els.svcReactivateBox.style.display = "none";

      if (!title || !desc) {
        if (els.svcErrorAlert) {
          els.svcErrorAlert.textContent = "Service Name and Description are required.";
          els.svcErrorAlert.style.display = "block";
        }
        return;
      }

      try {
        if (els.saveSvcBtn) els.saveSvcBtn.disabled = true;
        if (els.saveSvcSpinner) els.saveSvcSpinner.style.display = "inline-block";
        if (els.saveSvcBtnText) els.saveSvcBtnText.textContent = "Saving...";

        const payload = {
          title,
          description: desc,
          keywords,
          visual_line: visual,
          visual_description: visual,
          reactivate_if_exists: reactivateIfExists,
        };

        const resp = await fetch("/v1/services/create", {
          method: "POST",
          headers: getHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(payload),
        });

        const data = await resp.json().catch(() => ({}));

        if (resp.status === 409) {
          const detailObj = (typeof data.detail === "object" && data.detail !== null) ? data.detail : {};
          const canReactivate = Boolean(data.can_reactivate || detailObj.can_reactivate);
          const errorMsg = formatErrorMessage(data, "A service with this title already exists.");

          if (canReactivate && els.svcReactivateBox) {
            if (els.svcReactivateMsg) els.svcReactivateMsg.textContent = errorMsg;
            if (els.svcReactivateBtn) els.svcReactivateBtn.onclick = () => submitService(true);
            els.svcReactivateBox.style.display = "block";
          } else if (els.svcErrorAlert) {
            els.svcErrorAlert.textContent = errorMsg;
            els.svcErrorAlert.style.display = "block";
          }
          return;
        }

        if (!resp.ok) {
          if (els.svcErrorAlert) {
            els.svcErrorAlert.textContent = formatErrorMessage(data, `Server error (${resp.status})`);
            els.svcErrorAlert.style.display = "block";
          }
          return;
        }

        // Success
        if (els.svcSuccessAlert) {
          let msg = `Created demo service "${data.title}" (ID: ${data.id}). Embeddings computed (${data.embedding_dimensions || 768}d).`;
          if (!data.has_image_profile) {
            msg += " Notice: No image profile was created (visual description was empty).";
          }
          if (data.warning) {
            msg += " Notice: " + data.warning;
          } else if (data.warnings && data.warnings.length > 0) {
            msg += " Warnings: " + (Array.isArray(data.warnings) ? data.warnings.join("; ") : data.warnings);
          }
          els.svcSuccessAlert.textContent = msg;
          els.svcSuccessAlert.style.display = "block";
        }

        if (data.similar_services && data.similar_services.length > 0) {
          renderSimilarServices(data.similar_services, data.highest_similarity || 0);
        }

        showToast(`Saved demo service "${data.title}"`);
        els.addServiceForm.reset();
        updateCounters();
        fetchDemoServices();
        fetchCatalogServices();
      } catch (err) {
        console.error("Save service error:", err);
        if (els.svcErrorAlert) {
          els.svcErrorAlert.textContent = "Network error while saving service.";
          els.svcErrorAlert.style.display = "block";
        }
      } finally {
        if (els.saveSvcBtn) els.saveSvcBtn.disabled = false;
        if (els.saveSvcSpinner) els.saveSvcSpinner.style.display = "none";
        if (els.saveSvcBtnText) els.saveSvcBtnText.textContent = "Save Demo Service";
      }
    }

    els.addServiceForm.addEventListener("submit", (e) => {
      e.preventDefault();
      submitService(false);
    });
  }

  // --- Demo Services List & Delete Handler ---
  async function fetchDemoServices() {
    if (!els.demoServicesTable || !els.demoServicesTbody) return;

    try {
      const resp = await fetch("/v1/services/demo", {
        headers: getHeaders(),
      });

      if (!resp.ok) {
        console.warn("Could not fetch demo services, status:", resp.status);
        return;
      }

      const demos = await resp.json();
      state.demoServices = demos || [];

      if (els.demoCountBadge) {
        els.demoCountBadge.textContent = `${state.demoServices.length} Demo`;
      }
      if (els.deleteAllDemosBtn) {
        els.deleteAllDemosBtn.disabled = state.demoServices.length === 0;
      }

      els.demoServicesTbody.innerHTML = "";

      if (state.demoServices.length === 0) {
        if (els.demosEmpty) els.demosEmpty.style.display = "block";
        els.demoServicesTable.style.display = "none";
        return;
      }

      if (els.demosEmpty) els.demosEmpty.style.display = "none";
      els.demoServicesTable.style.display = "table";

      state.demoServices.forEach((item) => {
        const tr = document.createElement("tr");

        // 1. Service Title + ID + Badge
        const tdTitle = document.createElement("td");
        const titleDiv = document.createElement("div");
        titleDiv.style.fontWeight = "600";
        titleDiv.textContent = item.title;

        const metaDiv = document.createElement("div");
        metaDiv.style.display = "flex";
        metaDiv.style.alignItems = "center";
        metaDiv.style.gap = "6px";
        metaDiv.style.marginTop = "4px";

        const badge = document.createElement("span");
        badge.className = "badge-demo";
        badge.textContent = "Demo";
        metaDiv.appendChild(badge);

        const idSpan = document.createElement("span");
        idSpan.style.fontSize = "0.75rem";
        idSpan.style.color = "var(--text-muted)";
        idSpan.textContent = `#${item.id}`;
        metaDiv.appendChild(idSpan);

        tdTitle.appendChild(titleDiv);
        tdTitle.appendChild(metaDiv);
        tr.appendChild(tdTitle);

        // 2. Description + Keywords
        const tdDesc = document.createElement("td");
        const descDiv = document.createElement("div");
        descDiv.style.fontSize = "0.85rem";
        descDiv.textContent = item.description || "-";
        tdDesc.appendChild(descDiv);

        if (item.keywords) {
          const kwDiv = document.createElement("div");
          kwDiv.style.fontSize = "0.75rem";
          kwDiv.style.color = "var(--text-muted)";
          kwDiv.style.marginTop = "4px";
          kwDiv.textContent = `Keywords: ${item.keywords}`;
          tdDesc.appendChild(kwDiv);
        }
        tr.appendChild(tdDesc);

        // 3. Visual Description & Warning Badge
        const tdVisual = document.createElement("td");
        if (item.has_image_profile && item.visual_description) {
          const visDiv = document.createElement("div");
          visDiv.style.fontSize = "0.85rem";
          visDiv.textContent = item.visual_description;
          tdVisual.appendChild(visDiv);
        } else {
          const warnBadge = document.createElement("span");
          warnBadge.className = "badge-warn-visual";
          warnBadge.textContent = "⚠ Won't match from photos";
          tdVisual.appendChild(warnBadge);

          const hint = document.createElement("div");
          hint.style.fontSize = "0.75rem";
          hint.style.color = "var(--text-muted)";
          hint.style.marginTop = "4px";
          hint.textContent = "(No visual description)";
          tdVisual.appendChild(hint);
        }
        tr.appendChild(tdVisual);

        // 4. Created
        const tdCreated = document.createElement("td");
        tdCreated.style.fontSize = "0.8rem";
        tdCreated.style.color = "var(--text-secondary)";
        if (item.created_at) {
          try {
            tdCreated.textContent = new Date(item.created_at).toLocaleDateString([], {
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            });
          } catch {
            tdCreated.textContent = item.created_at;
          }
        } else {
          tdCreated.textContent = "-";
        }
        tr.appendChild(tdCreated);

        // 5. Delete Action Button
        const tdAction = document.createElement("td");
        tdAction.style.textAlign = "right";
        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "btn btn-danger";
        delBtn.style.padding = "4px 10px";
        delBtn.style.fontSize = "0.75rem";
        delBtn.textContent = "Delete";

        delBtn.addEventListener("click", async () => {
          if (!confirm(`Are you sure you want to delete demo service "${item.title}"?`)) {
            return;
          }
          try {
            delBtn.disabled = true;
            const resp = await fetch(`/v1/services/demo/${item.id}`, {
              method: "DELETE",
              headers: getHeaders(),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
              alert(formatErrorMessage(data, "Failed to delete demo service."));
              delBtn.disabled = false;
              return;
            }
            showToast(`Deleted demo service "${item.title}"`);
            fetchDemoServices();
            fetchCatalogServices();
          } catch (err) {
            console.error("Delete demo service error:", err);
            alert("Network error while deleting demo service.");
            delBtn.disabled = false;
          }
        });

        tdAction.appendChild(delBtn);
        tr.appendChild(tdAction);

        els.demoServicesTbody.appendChild(tr);
      });
    } catch (err) {
      console.warn("Error fetching demo services:", err);
    }
  }

  // Hook Delete All Demo Services button
  if (els.deleteAllDemosBtn) {
    els.deleteAllDemosBtn.addEventListener("click", async () => {
      const count = state.demoServices.length;
      if (!confirm(`Are you sure you want to delete ALL ${count} demo service(s)? This cannot be undone.`)) {
        return;
      }
      try {
        els.deleteAllDemosBtn.disabled = true;
        const resp = await fetch("/v1/services/demo", {
          method: "DELETE",
          headers: getHeaders(),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          alert(formatErrorMessage(data, "Failed to delete demo services."));
          els.deleteAllDemosBtn.disabled = false;
          return;
        }
        showToast(data.message || `Deleted ${count} demo services`);
        fetchDemoServices();
        fetchCatalogServices();
      } catch (err) {
        console.error("Delete all demo services error:", err);
        alert("Network error while deleting demo services.");
        els.deleteAllDemosBtn.disabled = false;
      }
    });
  }

  if (els.refreshDemosBtn) {
    els.refreshDemosBtn.addEventListener("click", () => {
      fetchDemoServices();
    });
  }

  // App Initialization
  function initAddServiceToggle() {
    const panel = document.getElementById("screen-add-service");
    if (!els.toggleAddServiceBtn || !panel) return;
    els.toggleAddServiceBtn.addEventListener("click", () => {
      const open = panel.style.display === "none";
      panel.style.display = open ? "block" : "none";
      els.toggleAddServiceBtn.setAttribute("aria-expanded", String(open));
      els.toggleAddServiceBtn.textContent = open ? "Close Form" : "+ Add Demo Service";
      if (open && els.svcTitle) els.svcTitle.focus();
    });
  }

  function init() {
    initAddServiceToggle();
    initAuth();
    initAddService();
    updateSubmitButtonState();
    fetchCatalogServices();
    fetchDemoServices();
    fetchHistory();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

