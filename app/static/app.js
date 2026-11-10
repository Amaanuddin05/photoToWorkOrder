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
    selectedServiceId: null,
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
    candidateCards: document.getElementById("candidate-cards"),
    serviceSearchFilter: document.getElementById("service-search-filter"),
    anotherServiceDropdown: document.getElementById("another-service-dropdown"),
    confirmBtn: document.getElementById("confirm-btn"),
    confirmSpinner: document.getElementById("confirm-spinner"),
    confirmBtnText: document.getElementById("confirm-btn-text"),
    nextPostBtn: document.getElementById("next-post-btn"),
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
    if (intentEl && intentEl.value) {
      formData.append("intent_enabled", intentEl.value);  // test-only per-request override
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

  // Render Screen 2 (Result View)
  function renderResultScreen(decision) {
    clearResultError();
    state.selectedServiceId = null;
    els.confirmBtn.disabled = true;
    els.confirmBtn.style.display = "inline-flex";
    els.nextPostBtn.style.display = "none";

    // 1. Action Badge & Shadow Mode Note
    els.actionBadge.className = "action-badge";
    if (decision.action === "auto_create") {
      els.actionBadge.textContent = "Auto-create";
      els.actionBadge.classList.add("badge-auto-create");
      els.shadowNote.style.display = "none";
    } else if (decision.action === "no_action") {
      els.actionBadge.textContent = "No action";
      els.actionBadge.classList.add("badge-no-action");
      els.shadowNote.style.display = "none";
    } else {
      els.actionBadge.textContent = "Draft";
      els.actionBadge.classList.add("badge-draft");
      if (decision.would_be_action === "auto_create" || decision.would_auto || decision.shadow) {
        els.shadowNote.textContent = "Would be: Auto-create (Shadow Mode)";
        els.shadowNote.style.display = "inline";
      } else {
        els.shadowNote.style.display = "none";
      }
    }

    // 2. Agree Indicator
    if (decision.agree === true) {
      els.agreeBadge.style.display = "inline-flex";
      els.agreeBadge.className = "agree-badge agree-true";
      els.agreeBadge.textContent = "✓ Text and photo agree";
    } else if (decision.agree === false) {
      els.agreeBadge.style.display = "inline-flex";
      els.agreeBadge.className = "agree-badge agree-false";
      els.agreeBadge.textContent = "⚠ Text and photo differ";
      if (decision.agree_mode && decision.agree_mode !== "strict") {
        els.agreeBadge.textContent += ` (strictness: ${decision.agree_mode})`;
      }
    } else {
      els.agreeBadge.style.display = "none";
    }

    // 3. Reason in Plain Words
    els.reasonText.textContent = formatReason(decision.reason);

    // 4. Left Column: Submitted Input
    if (state.selectedFileUrl) {
      els.submittedImg.src = state.selectedFileUrl;
      els.submittedImgWrapper.style.display = "block";
      els.submittedNoPhoto.style.display = "none";
    } else {
      els.submittedImgWrapper.style.display = "none";
      els.submittedNoPhoto.style.display = "block";
    }
    els.submittedDescText.textContent = state.submittedDesc || "(No description provided)";

    if (state.submittedLoc) {
      els.submittedLocTitle.style.display = "block";
      els.submittedLocText.style.display = "block";
      els.submittedLocText.textContent = state.submittedLoc;
    } else {
      els.submittedLocTitle.style.display = "none";
      els.submittedLocText.style.display = "none";
    }

    // 5. Right Column: Candidate Cards
    while (els.candidateCards.firstChild) {
      els.candidateCards.removeChild(els.candidateCards.firstChild);
    }

    const candidates = decision.top_candidates || [];
    candidates.slice(0, 3).forEach((cand, idx) => {
      const card = document.createElement("div");
      card.className = "candidate-card";
      card.tabIndex = 0;
      card.setAttribute("role", "radio");
      card.setAttribute("aria-checked", "false");

      const sId = cand.service_id != null ? cand.service_id : cand.id;
      const sTitle = cand.title || cand.service || `Service #${sId}`;
      card.dataset.serviceId = String(sId);
      card.dataset.rank = String(idx + 1);

      // Top Row (Rank badge + Emergency warning)
      const topRow = document.createElement("div");
      topRow.className = "card-top-row";

      const rankBadge = document.createElement("span");
      rankBadge.className = "card-rank-badge";
      rankBadge.textContent = `#${idx + 1}`;
      topRow.appendChild(rankBadge);

      const isEmergency =
        EMERGENCY_SERVICES.includes(sTitle) ||
        sTitle.toLowerCase().startsWith("emergency") ||
        (decision.reason && decision.reason.includes("always_draft_services"));

      if (isEmergency) {
        const emergencyBadge = document.createElement("span");
        emergencyBadge.className = "emergency-badge";
        emergencyBadge.textContent = "Needs human sign-off";
        topRow.appendChild(emergencyBadge);
      }

      card.appendChild(topRow);

      // Service Title
      const titleEl = document.createElement("div");
      titleEl.className = "card-title";
      titleEl.textContent = sTitle;
      card.appendChild(titleEl);

      // Score Row
      const scoreRow = document.createElement("div");
      scoreRow.className = "card-score-row";

      const barBg = document.createElement("div");
      barBg.className = "score-bar-bg";

      const barFill = document.createElement("div");
      barFill.className = "score-bar-fill";
      const scoreVal = typeof cand.confidence === "number" ? cand.confidence : (typeof cand.score === "number" ? cand.score : 0);
      const pct = Math.max(0, Math.min(100, Math.round(scoreVal * 1000) / 10));
      barFill.style.width = `${pct}%`;
      barBg.appendChild(barFill);
      scoreRow.appendChild(barBg);

      const pctLabel = document.createElement("span");
      pctLabel.className = "score-pct-label";
      pctLabel.textContent = cand.confidence_pct || (typeof cand.confidence === "number" ? `${pct.toFixed(1)}%` : "N/A");
      scoreRow.appendChild(pctLabel);

      card.appendChild(scoreRow);

      // Per-stage ranks metadata if available
      const metaRow = document.createElement("div");
      metaRow.className = "card-meta-row";
      const metaParts = [];
      if (cand.text_rank != null) metaParts.push(`Text rank: #${cand.text_rank}`);
      if (cand.image_rank != null) metaParts.push(`Image rank: #${cand.image_rank}`);
      if (cand.pool_rank != null) metaParts.push(`Pool rank: #${cand.pool_rank}`);
      if (metaParts.length > 0) {
        metaRow.textContent = metaParts.join(" • ");
        card.appendChild(metaRow);
      }

      // Selection Handlers
      card.addEventListener("click", () => {
        selectService(sId, card);
      });

      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectService(sId, card);
        }
      });

      els.candidateCards.appendChild(card);
    });

    // Reset Dropdown
    els.anotherServiceDropdown.value = "";
    els.serviceSearchFilter.value = "";
    filterCatalogOptions("");

    // Toggle Screen Views
    els.screenNewPost.style.display = "none";
    els.screenResult.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectService(serviceId, activeCardEl) {
    state.selectedServiceId = Number(serviceId);
    clearResultError();

    const allCards = els.candidateCards.querySelectorAll(".candidate-card");
    allCards.forEach((c) => {
      if (Number(c.dataset.serviceId) === state.selectedServiceId) {
        c.classList.add("selected");
        c.setAttribute("aria-checked", "true");
      } else {
        c.classList.remove("selected");
        c.setAttribute("aria-checked", "false");
      }
    });

    if (activeCardEl) {
      els.anotherServiceDropdown.value = "";
    } else {
      els.anotherServiceDropdown.value = String(serviceId);
    }

    els.confirmBtn.disabled = false;
  }

  // Keyboard shortcut listener (1, 2, 3) for candidate cards
  window.addEventListener("keydown", (e) => {
    if (els.screenResult.style.display !== "block") return;

    const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    if (e.key === "1" || e.key === "2" || e.key === "3") {
      const rankIdx = parseInt(e.key, 10);
      const targetCard = els.candidateCards.querySelector(`.candidate-card[data-rank="${rankIdx}"]`);
      if (targetCard) {
        e.preventDefault();
        targetCard.click();
      }
    }
  });

  // Dropdown filter and selection
  function filterCatalogOptions(filterText) {
    const term = filterText.toLowerCase().trim();
    while (els.anotherServiceDropdown.firstChild) {
      els.anotherServiceDropdown.removeChild(els.anotherServiceDropdown.firstChild);
    }

    const defaultOpt = document.createElement("option");
    defaultOpt.value = "";
    defaultOpt.textContent = "-- Choose catalog service --";
    els.anotherServiceDropdown.appendChild(defaultOpt);

    state.catalogServices.forEach((svc) => {
      if (!term || svc.title.toLowerCase().includes(term)) {
        const opt = document.createElement("option");
        opt.value = String(svc.id);
        opt.textContent = svc.title;
        els.anotherServiceDropdown.appendChild(opt);
      }
    });

    if (state.selectedServiceId) {
      els.anotherServiceDropdown.value = String(state.selectedServiceId);
    }
  }

  els.serviceSearchFilter.addEventListener("input", (e) => {
    filterCatalogOptions(e.target.value);
  });

  els.anotherServiceDropdown.addEventListener("change", (e) => {
    const val = e.target.value;
    if (val) {
      selectService(Number(val), null);
    } else {
      state.selectedServiceId = null;
      els.confirmBtn.disabled = true;
      const allCards = els.candidateCards.querySelectorAll(".candidate-card");
      allCards.forEach((c) => {
        c.classList.remove("selected");
        c.setAttribute("aria-checked", "false");
      });
    }
  });

  // Confirm Final Service -> POST /v1/decisions/{decision_id}/feedback
  els.confirmBtn.addEventListener("click", async () => {
    if (!state.currentDecision || !state.currentDecision.decision_id || !state.selectedServiceId) {
      return;
    }

    clearResultError();
    els.confirmBtn.disabled = true;
    els.confirmSpinner.style.display = "inline-block";
    els.confirmBtnText.textContent = "Saving...";

    const decId = state.currentDecision.decision_id;
    const payload = { service_id: state.selectedServiceId };

    try {
      const resp = await fetch(`/v1/decisions/${decId}/feedback`, {
        method: "POST",
        headers: getHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        let errMessage = `Error saving decision (${resp.status})`;
        try {
          const errData = await resp.json();
          errMessage = formatErrorMessage(errData, errMessage);
        } catch (_) {}
        throw new Error(errMessage);
      }

      showToast("Saved");
      els.confirmBtn.style.display = "none";
      els.nextPostBtn.style.display = "inline-flex";

      fetchHistory();
    } catch (err) {
      showResultError(err.message || "Failed to confirm service. Please try again.");
      els.confirmBtn.disabled = false;
    } finally {
      els.confirmSpinner.style.display = "none";
      els.confirmBtnText.textContent = "Confirm Final Service";
    }
  });

  // Reset & Next Post
  function resetFormAndGoToScreen1() {
    clearPhotoInput();
    els.descInput.value = "";
    els.locationInput.value = "";
    state.submittedDesc = "";
    state.submittedLoc = "";
    state.currentDecision = null;
    state.selectedServiceId = null;
    clearFormError();
    clearResultError();

    updateSubmitButtonState();
    els.screenResult.style.display = "none";
    els.screenNewPost.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  els.nextPostBtn.addEventListener("click", resetFormAndGoToScreen1);
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

      // 4. Final Service
      const tdFinal = document.createElement("td");
      const finalTitle = row.final_service_title || row.finalServiceTitle;
      const finalId = row.final_service_id != null ? row.final_service_id : row.finalServiceId;
      if (finalTitle) {
        tdFinal.textContent = finalTitle;
      } else if (finalId) {
        tdFinal.textContent = `Service #${finalId}`;
      } else {
        tdFinal.className = "pending-label";
        tdFinal.textContent = "Pending";
      }
      tr.appendChild(tdFinal);

      // 5. Match / Mismatch Icon
      const tdMatch = document.createElement("td");
      if (finalId != null) {
        const iconSpan = document.createElement("span");
        if (finalId === topId) {
          iconSpan.className = "match-icon match-yes";
          iconSpan.textContent = "✓";
          iconSpan.title = "Match: final selection matches suggestion";
        } else {
          iconSpan.className = "match-icon match-no";
          iconSpan.textContent = "≠";
          iconSpan.title = "Mismatch: operator chose different service";
        }
        tdMatch.appendChild(iconSpan);
      } else {
        tdMatch.textContent = "-";
      }
      tr.appendChild(tdMatch);

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
      filterCatalogOptions("");
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
  function init() {
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

