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
    const ic = el("span", auto ? "✓" : "✎", `wo-icon ${auto ? "wo-auto" : "wo-draft"}`);
    ic.title = `${auto ? "Work order auto-created" : "Draft work order"}: ${m.service || "?"}\n${m.reason || ""}`;
    return ic;
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
    }
    row.appendChild(b);
    const ic = woIcon(m);
    if (ic) row.appendChild(ic);
    return row;
  }

  function render(msgs) {
    const key = msgs.map((m) => `${m.id}:${m.action || ""}:${pending.has(m.id) ? "p" : ""}`).join(",");
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
