/* Radio channel demo: chat-style message feed backed by /v1/radio/*. Plain JS, textContent only. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const feed = $("feed"), empty = $("empty"), err = $("err");
  const fileInput = $("file"), textInput = $("text"), sendBtn = $("send");
  let attached = null;
  let lastIds = "";
  let stick = true;
  feed.addEventListener("scroll", () => { stick = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60; });
  const pending = new Set();
  const open = new Set();   // message ids whose details panel is open

  function headers() {
    const k = sessionStorage.getItem("siglip_api_key") || "";
    return k ? { Authorization: `Bearer ${k}` } : {};
  }
  try {
    $("sender").value = localStorage.getItem("radio_sender") || "Worker";
    $("llm").checked = localStorage.getItem("radio_llm") === "1";
    $("llm-model").value = localStorage.getItem("radio_llm_model") || "claude-sonnet-5-5";
  } catch (_) {}
  const remember = () => {
    try {
      localStorage.setItem("radio_sender", $("sender").value);
      localStorage.setItem("radio_llm", $("llm").checked ? "1" : "0");
      localStorage.setItem("radio_llm_model", $("llm-model").value);
    } catch (_) {}
  };
  ["sender", "llm", "llm-model"].forEach((id) => $(id).addEventListener("change", remember));

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text != null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function woIcon(m) {
    if (pending.has(m.id)) {
      const p = el("span", "…", "wo-icon wo-pending");
      p.title = "Processing";
      return p;
    }
    if (m.action !== "auto_create" && m.action !== "draft") return null;
    const auto = m.action === "auto_create";
    const ic = el("button", auto ? "✓" : "✎", `wo-icon ${auto ? "wo-auto" : "wo-draft"}`);
    ic.type = "button";
    ic.title = `${auto ? "Work order auto-created" : "Draft work order"}: ${m.service || "?"} (click for details)`;
    ic.setAttribute("aria-expanded", String(open.has(m.id)));
    ic.addEventListener("click", () => {
      if (open.has(m.id)) open.delete(m.id); else open.add(m.id);
      lastIds = "";
      render(cache);
    });
    return ic;
  }

  const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");
  const num = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");

  function detailsPanel(m) {
    const d = m.details || {};
    const box = el("div", null, "wo-details");
    const t = document.createElement("table");
    const add = (k, v, cls) => {
      const tr = document.createElement("tr");
      tr.appendChild(el("th", k));
      tr.appendChild(el("td", v, cls));
      t.appendChild(tr);
    };
    const auto = m.action === "auto_create";
    add("Status", `${auto ? "Work order auto-created" : "Draft: needs review"}${m.decision_id ? ` (decision #${m.decision_id})` : ""}`, auto ? "ok" : "");
    add("Service", m.service || "—");
    if (typeof d.confidence === "number") add("Confidence", `${pct(d.confidence)} (auto needs ≥ 40%)`, d.confidence >= 0.4 ? "ok" : "bad");
    if (typeof d.gap === "number") add("Gap to #2", `${pct(d.gap)} (auto needs ≥ 10%)`, d.gap >= 0.1 ? "ok" : "bad");
    if (d.text_guard && typeof d.text_guard.score === "number") {
      const tg = d.text_guard;
      add("Text match", `${num(tg.score)} (auto ≥ ${tg.auto_min}, floor ${tg.no_action_floor})`, tg.score >= tg.auto_min ? "ok" : "bad");
    }
    if (d.text_top1 || d.image_top1) {
      add("Text #1 / Photo #1", `${d.text_top1 || "—"} / ${d.image_top1 || "—"}`);
      add("Agreement", `${d.agree === true ? "yes" : d.agree === false ? "no" : "—"} · strictness ${d.agree_mode || "strict"}`, d.agree === true ? "ok" : d.agree === false ? "bad" : "muted");
    }
    if (d.intent) {
      const names = { issue: "issue report", done: "work completed", other: "not maintenance" };
      add("Intent", `${names[d.intent.label] || d.intent.label} (${pct(d.intent.prob)})`);
    }
    if (Array.isArray(d.top) && d.top.length) {
      const pick = m.llm && !m.llm.error ? m.llm.service : null;
      const ranked = d.top.slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
      add("Top candidates (pipeline score)", ranked.map((c, i) => `${i + 1}. ${c.title} ${pct(c.confidence)}${c.title === pick ? "  ← LLM pick" : ""}`).join("\n"));
    }
    const l = m.llm;
    if (l && l.used) {
      add("LLM", `${l.served_model || l.model}${l.trigger ? ` · trigger: ${l.trigger}` : ""}`);
      if (l.error) {
        add("LLM result", `failed: ${l.error}`, "bad");
      } else {
        add("LLM verdict", `intent ${l.intent} · ${l.service} (${l.confidence}) · photo ${l.photo_matches ? "matches" : "does not match"}${l.promoted ? " · promoted to auto" : ""}`);
        if (l.reason) add("LLM reasoning", l.reason);
      }
      add("LLM usage", `${l.input_tokens || 0} in / ${l.output_tokens || 0} out tokens · ≈ $${(l.cost_usd || 0).toFixed(6)} · ${((l.latency_ms || 0) / 1000).toFixed(1)} s`);
    } else if (d.llm_skip) {
      add("LLM", `not used: ${d.llm_skip}`, "muted");
    }
    add("Reason", m.reason || "—", "muted");
    if (typeof d.latency_ms === "number") add("Processing time", `${(d.latency_ms / 1000).toFixed(2)} s`);
    t.querySelectorAll("td").forEach((td) => { td.style.whiteSpace = "pre-line"; });
    box.appendChild(t);
    return box;
  }

  function renderMessage(m) {
    const me = m.sender === $("sender").value;
    const row = el("div", null, `msg${me ? " me" : ""}`);
    row.dataset.id = m.id;
    const b = el("div", null, "bubble");
    const meta = el("div", null, "meta");
    meta.appendChild(el("b", m.sender || "Worker"));
    const d = m.created_at ? new Date(m.created_at) : null;
    meta.appendChild(el("span", d ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""));
    b.appendChild(meta);
    if (m.photo_url) {
      const img = document.createElement("img");
      img.src = m.photo_url;
      img.alt = "Photo";
      img.addEventListener("load", () => { if (stick) feed.scrollTop = feed.scrollHeight; });
      img.addEventListener("click", () => window.open(m.photo_url, "_blank"));
      b.appendChild(img);
    }
    if (m.text) b.appendChild(el("div", m.text, "txt"));
    if (m.action === "auto_create" || m.action === "draft") {
      const line = el("div", `${m.action === "auto_create" ? "Work order" : "Draft"} #${m.decision_id}: ${m.service || "?"}`, "wo-line");
      if (m.llm && m.llm.used) {
        line.appendChild(document.createTextNode(" "));
        const lb = el("span", m.llm.error ? "LLM err" : `LLM $${(m.llm.cost_usd || 0).toFixed(4)}`, "llm-badge");
        lb.title = m.llm.error || `${m.llm.served_model || m.llm.model}: ${m.llm.input_tokens} in / ${m.llm.output_tokens} out tokens`;
        line.appendChild(lb);
      }
      b.appendChild(line);
      if (open.has(m.id)) b.appendChild(detailsPanel(m));
    }
    row.appendChild(b);
    const ic = woIcon(m);
    if (ic) row.appendChild(ic);
    return row;
  }

  function render(msgs) {
    const key = msgs.map((m) => `${m.id}:${m.action || ""}:${pending.has(m.id) ? "p" : ""}:${open.has(m.id) ? "o" : ""}`).join(",");
    if (key === lastIds) return;
    lastIds = key;
    const first = !feed.dataset.loaded;
    feed.dataset.loaded = "1";
    const atBottom = first || feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
    feed.replaceChildren();
    if (!msgs.length) feed.appendChild(empty);
    msgs.forEach((m) => feed.appendChild(renderMessage(m)));
    stick = atBottom;
    if (atBottom) feed.scrollTop = feed.scrollHeight;
  }

  let cache = [];
  async function load() {
    try {
      const r = await fetch("/v1/radio/messages?limit=200", { headers: headers() });
      if (!r.ok) {
        err.textContent = r.status === 401 ? "API key required: save it on the Operator UI page." : `Could not load messages (${r.status})`;
        return;
      }
      err.textContent = "";
      cache = await r.json();
      render(cache);
    } catch (e) {
      err.textContent = "Server not reachable";
    }
  }

  function setAttachment(f) {
    attached = f;
    const box = $("attach-preview");
    if (!f) { box.style.display = "none"; fileInput.value = ""; return; }
    $("attach-img").src = URL.createObjectURL(f);
    $("attach-name").textContent = f.name;
    box.style.display = "flex";
  }

  $("attach").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => setAttachment(fileInput.files[0] || null));
  $("attach-remove").addEventListener("click", () => setAttachment(null));
  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
  });

  $("composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = textInput.value.trim();
    if (!text && !attached) return;
    const fd = new FormData();
    fd.append("sender", $("sender").value || "Worker");
    if (text) fd.append("text", text);
    if (attached) fd.append("image", attached, attached.name);
    fd.append("llm_enabled", $("llm").checked ? "on" : "off");
    fd.append("llm_model", $("llm-model").value);

    // Optimistic bubble while the pipeline runs
    const tempId = -Date.now();
    pending.add(tempId);
    cache = cache.concat([{ id: tempId, sender: $("sender").value, text, created_at: new Date().toISOString(),
      photo_url: attached ? URL.createObjectURL(attached) : null }]);
    render(cache);
    feed.scrollTop = feed.scrollHeight;
    textInput.value = "";
    setAttachment(null);
    sendBtn.disabled = true;
    err.textContent = "";
    try {
      const r = await fetch("/v1/radio/messages", { method: "POST", headers: headers(), body: fd });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        err.textContent = (j.detail && (j.detail.message || j.detail)) || `Send failed (${r.status})`;
      }
    } catch (_) {
      err.textContent = "Send failed: server not reachable";
    } finally {
      pending.delete(tempId);
      sendBtn.disabled = false;
      await load();
      feed.scrollTop = feed.scrollHeight;
    }
  });

  $("clear").addEventListener("click", async () => {
    if (!confirm("Delete all messages in this demo channel? Work orders stay in the decision log.")) return;
    await fetch("/v1/radio/messages", { method: "DELETE", headers: headers() });
    lastIds = "";
    load();
  });

  load();
  setInterval(load, 5000);
})();
