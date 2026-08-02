// Delivery Operations RCA — frontend logic
// Implements the contract and UX rules in 03_UI_HANDOFF.md.

(() => {
  "use strict";

  const API_BASE = "";
  const SESSION_KEY = "rca_session_id";
  const CLIENT_TIMEOUT_MS = 45000;

  // ---------------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------------
  const chatLog = document.getElementById("chat-log");
  const emptyState = document.getElementById("empty-state");
  const composer = document.getElementById("composer");
  const input = document.getElementById("composer-input");
  const sendBtn = document.getElementById("composer-send");
  const newConvBtn = document.getElementById("new-conversation");
  const statusDot = document.getElementById("status-dot");
  const statusLabel = document.getElementById("status-label");
  const contextStrip = document.getElementById("context-strip");
  const contextStripToggle = document.getElementById("context-strip-toggle");
  const contextStripText = document.getElementById("context-strip-text");

  // ---------------------------------------------------------------
  // State
  // ---------------------------------------------------------------
  let sessionId = localStorage.getItem(SESSION_KEY) || null;
  let inFlight = false;
  let prevContext = {};
  let currentContext = {};
  let entities = null;

  // ---------------------------------------------------------------
  // Markdown rendering
  // ---------------------------------------------------------------
  marked.setOptions({ gfm: true, breaks: false });

  function renderMarkdown(md) {
    const rawHtml = marked.parse(md || "");
    const clean = DOMPurify.sanitize(rawHtml, { ALLOWED_ATTR: ["href", "align", "target", "rel"] });
    const wrapper = document.createElement("div");
    wrapper.className = "md";
    wrapper.innerHTML = clean;
    postProcessMarkdown(wrapper);
    return wrapper;
  }

  function postProcessMarkdown(root) {
    // Wrap tables for horizontal scroll.
    root.querySelectorAll("table").forEach((table) => {
      const scroll = document.createElement("div");
      scroll.className = "table-scroll";
      table.parentNode.insertBefore(scroll, table);
      scroll.appendChild(table);
    });

    // Colour breach-rate style percentage cells by threshold.
    root.querySelectorAll("td").forEach((td) => {
      const text = td.textContent.trim();
      const m = text.match(/^(\d+(?:\.\d+)?)%$/);
      if (!m) return;
      const val = parseFloat(m[1]);
      if (val >= 50) td.classList.add("breach-critical");
      else if (val >= 20) td.classList.add("breach-warning");
      else td.classList.add("breach-ok");
    });

    // Monospace STORE_### codes anywhere in text (outside code/pre).
    wrapStoreCodesInTextNodes(root);

    // Style truncation / "show all hours" notices: a paragraph that is
    // entirely italic and mentions "show me all hours".
    root.querySelectorAll("p").forEach((p) => {
      const onlyEm =
        p.children.length === 1 &&
        p.children[0].tagName === "EM" &&
        p.textContent.trim() === p.children[0].textContent.trim();
      if (!onlyEm) return;
      p.classList.add("truncation-notice");
      const em = p.children[0];
      const html = em.innerHTML.replace(
        /(show me all hours)/i,
        '<a class="show-all-link" data-msg="show me all hours">$1</a>'
      );
      em.innerHTML = html;
    });
  }

  function wrapStoreCodesInTextNodes(root) {
    const skip = new Set(["CODE", "PRE", "A"]);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (skip.has(node.parentNode.tagName)) return NodeFilter.FILTER_REJECT;
        if (node.parentNode.classList && node.parentNode.classList.contains("store-code")) {
          return NodeFilter.FILTER_REJECT;
        }
        return /STORE_\d{3}/.test(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    });
    const nodes = [];
    let n;
    while ((n = walker.nextNode())) nodes.push(n);

    nodes.forEach((node) => {
      const parts = node.nodeValue.split(/(STORE_\d{3})/);
      if (parts.length === 1) return;
      const frag = document.createDocumentFragment();
      parts.forEach((part) => {
        if (/^STORE_\d{3}$/.test(part)) {
          const span = document.createElement("span");
          span.className = "store-code";
          span.textContent = part;
          frag.appendChild(span);
        } else if (part) {
          frag.appendChild(document.createTextNode(part));
        }
      });
      node.parentNode.replaceChild(frag, node);
    });
  }

  // ---------------------------------------------------------------
  // Chat log rendering
  // ---------------------------------------------------------------
  function hideEmptyState() {
    if (emptyState && emptyState.parentNode) emptyState.remove();
  }

  function addUserMessage(text) {
    hideEmptyState();
    const el = document.createElement("div");
    el.className = "message message--user";
    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    bubble.textContent = text;
    el.appendChild(bubble);
    chatLog.appendChild(el);
    scrollToBottom();
    return el;
  }

  function addTypingIndicator() {
    hideEmptyState();
    const el = document.createElement("div");
    el.className = "message message--assistant";
    el.dataset.typing = "1";
    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    bubble.innerHTML = `
      <div class="typing-indicator">
        <span class="label">Analysing…</span>
        <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
      </div>`;
    el.appendChild(bubble);
    chatLog.appendChild(el);
    scrollToBottom();
    return el;
  }

  function addAssistantMessage(data) {
    hideEmptyState();
    const el = document.createElement("div");
    el.className = `message message--assistant provenance-${data.provenance}`;

    const bubble = document.createElement("div");
    bubble.className = "message__bubble";

    if (data.provenance === "exploratory") {
      const banner = document.createElement("div");
      banner.className = "exploratory-banner";
      banner.innerHTML =
        "<span>⚠</span><span>Exploratory result — not from the validated analysis path. Verify before acting on it.</span>";
      bubble.appendChild(banner);
    }

    const mdBody = renderMarkdown(data.reply);
    bubble.appendChild(mdBody);

    if (Array.isArray(data.suggestions) && data.suggestions.length) {
      const chipRow = document.createElement("div");
      chipRow.className = "suggestion-chips";
      data.suggestions.forEach((s) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "suggestion-chip";
        chip.textContent = s;
        chip.addEventListener("click", () => sendMessage(s));
        chipRow.appendChild(chip);
      });
      bubble.appendChild(chipRow);
    }

    const actions = document.createElement("div");
    actions.className = "message-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => copyText(data.reply, copyBtn));
    actions.appendChild(copyBtn);
    bubble.appendChild(actions);

    // Clickable "show me all hours" link inside truncation notices.
    bubble.querySelectorAll(".show-all-link").forEach((link) => {
      link.addEventListener("click", () => sendMessage(link.dataset.msg));
    });

    el.appendChild(bubble);
    chatLog.appendChild(el);
    scrollAssistantIntoView(el);
    return el;
  }

  function addConnError(retryText) {
    hideEmptyState();
    const el = document.createElement("div");
    el.className = "message message--assistant";
    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    bubble.innerHTML = "";
    const box = document.createElement("div");
    box.className = "conn-error";
    box.innerHTML = `<span>⚠ Can't reach the agent.</span>`;
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.textContent = "Retry";
    retryBtn.addEventListener("click", () => {
      el.remove();
      sendMessage(retryText);
    });
    box.appendChild(retryBtn);
    bubble.appendChild(box);
    el.appendChild(bubble);
    chatLog.appendChild(el);
    scrollToBottom();
    return el;
  }

  function copyText(text, btn) {
    navigator.clipboard?.writeText(text).then(() => {
      btn.textContent = "Copied";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.classList.remove("copied");
      }, 1500);
    });
  }

  function scrollToBottom() {
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function scrollAssistantIntoView(el) {
    // Anchor to the TOP of the new assistant message, not the bottom —
    // long replies (up to 168 lines) must not dump the user at the end.
    requestAnimationFrame(() => {
      el.scrollIntoView({ block: "start", behavior: "smooth" });
    });
  }

  // ---------------------------------------------------------------
  // Context panel / strip
  // ---------------------------------------------------------------
  function formatHours(hours) {
    if (!hours || !hours.length) return null;
    const sorted = [...hours].sort((a, b) => a - b);
    const runs = [];
    let start = sorted[0];
    let prev = sorted[0];
    for (let i = 1; i <= sorted.length; i++) {
      const cur = sorted[i];
      if (cur === prev + 1) {
        prev = cur;
        continue;
      }
      runs.push(start === prev ? `${start}` : `${start}–${prev}`);
      start = cur;
      prev = cur;
    }
    return runs.join(", ");
  }

  function fieldDisplay(context, field) {
    switch (field) {
      case "store":
        return context.store || null;
      case "city":
        return context.city || null;
      case "date":
        return context.date || null;
      case "hours":
        return formatHours(context.hours) || (context.hour_label ? context.hour_label : null);
      default:
        return null;
    }
  }

  function updateContextPanel(context) {
    prevContext = currentContext;
    currentContext = context || {};
    const inherited = new Set(currentContext.inherited || []);

    ["store", "city", "date", "hours"].forEach((field) => {
      const fieldEl = document.querySelector(`.context-field[data-field="${field}"]`);
      const valueEl = fieldEl.querySelector(".context-field__value");
      const labelEl = fieldEl.querySelector(".context-field__label");
      const display = fieldDisplay(currentContext, field);
      const prevDisplay = fieldDisplay(prevContext, field);

      valueEl.textContent = display || "—";
      valueEl.classList.toggle("placeholder", !display);

      // Inherited glyph
      let glyph = labelEl.querySelector(".inherited-glyph");
      if (inherited.has(field) && display) {
        if (!glyph) {
          glyph = document.createElement("span");
          glyph.className = "inherited-glyph";
          glyph.textContent = "↩";
          glyph.title = "carried from earlier turn";
          glyph.setAttribute("aria-label", "carried from earlier turn");
          labelEl.appendChild(glyph);
        }
      } else if (glyph) {
        glyph.remove();
      }

      // Resolution method tooltip (low priority, high charm).
      const method = (currentContext.resolution_methods || {})[field];
      valueEl.title = method ? `${field}: ${method}` : "";

      // Flash animation on change.
      if (display && display !== prevDisplay) {
        fieldEl.classList.remove("flash");
        // Force reflow so the animation can restart.
        void fieldEl.offsetWidth;
        fieldEl.classList.add("flash");
        setTimeout(() => fieldEl.classList.remove("flash"), 650);
      }
    });

    const intentEl = document.querySelector('[data-value="intent"]');
    intentEl.textContent = currentContext.intent || "—";
    intentEl.classList.toggle("placeholder", !currentContext.intent);

    updateContextStrip(currentContext);
  }

  function updateContextStrip(context) {
    const parts = [];
    const inherited = new Set(context.inherited || []);
    const push = (field, label) => {
      const v = fieldDisplay(context, field);
      if (!v) return;
      parts.push(`${v}${inherited.has(field) ? " ↩" : ""}`);
    };
    push("store");
    push("city");
    push("date");
    push("hours");
    contextStripText.textContent = parts.length ? parts.join(" · ") : "No context yet";
  }

  contextStripToggle.addEventListener("click", () => {
    contextStrip.classList.toggle("expanded");
  });

  // ---------------------------------------------------------------
  // Status indicator
  // ---------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) throw new Error("unhealthy");
      statusDot.className = "status-dot live";
      statusLabel.textContent = "live";
    } catch {
      statusDot.className = "status-dot offline";
      statusLabel.textContent = "offline";
    }
  }

  // ---------------------------------------------------------------
  // Entities (starter data — cities/stores for future autocomplete)
  // ---------------------------------------------------------------
  async function loadEntities() {
    try {
      const res = await fetch(`${API_BASE}/entities`);
      if (res.ok) entities = await res.json();
    } catch {
      /* non-critical */
    }
  }

  // ---------------------------------------------------------------
  // Session restore
  // ---------------------------------------------------------------
  async function restoreSession() {
    if (!sessionId) return;
    try {
      const res = await fetch(`${API_BASE}/session/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();
      if (!data.turn_count) return;
      hideEmptyState();
      (data.turns || []).forEach((turn) => {
        addUserMessage(turn.user);
        addAssistantMessage({
          reply: turn.reply,
          provenance: turn.provenance || "validated",
          suggestions: [],
        });
      });
      const cc = data.current_context || {};
      updateContextPanel({
        city: cc.city,
        store: cc.store,
        date: cc.date,
        hours: cc.hours,
        intent: cc.last_intent,
        inherited: [],
        resolution_methods: {},
      });
    } catch {
      /* start fresh if restore fails */
    }
  }

  // ---------------------------------------------------------------
  // Send flow
  // ---------------------------------------------------------------
  function setInFlight(state) {
    inFlight = state;
    input.disabled = state;
    sendBtn.disabled = state || !input.value.trim();
  }

  async function sendMessage(text) {
    const message = (text ?? input.value).trim();
    if (!message || inFlight) return;

    addUserMessage(message);
    input.value = "";
    autoGrow();
    setInFlight(true);

    const typingEl = addTypingIndicator();

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: sessionId }),
        signal: AbortSignal.timeout(CLIENT_TIMEOUT_MS),
      });

      typingEl.remove();

      if (!res.ok) {
        addConnError(message);
        input.value = message;
        autoGrow();
        return;
      }

      const data = await res.json();
      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);

      addAssistantMessage(data);
      updateContextPanel(data.context);
    } catch (err) {
      typingEl.remove();
      addConnError(message);
      input.value = message;
      autoGrow();
    } finally {
      setInFlight(false);
      input.focus();
    }
  }

  // ---------------------------------------------------------------
  // Composer
  // ---------------------------------------------------------------
  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
    sendBtn.disabled = inFlight || !input.value.trim();
  }

  input.addEventListener("input", autoGrow);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });

  document.querySelectorAll(".starter-chip").forEach((chip) => {
    chip.addEventListener("click", () => sendMessage(chip.dataset.msg));
  });

  newConvBtn.addEventListener("click", async () => {
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/session/${sessionId}/reset`, { method: "POST" });
      } catch {
        /* best-effort */
      }
    }
    sessionId = null;
    localStorage.removeItem(SESSION_KEY);
    chatLog.innerHTML = "";
    chatLog.appendChild(emptyState);
    updateContextPanel({});
    input.value = "";
    autoGrow();
    input.focus();
  });

  // ---------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------
  (async function init() {
    updateContextPanel({});
    checkHealth();
    setInterval(checkHealth, 30000);
    await loadEntities();
    await restoreSession();
    input.focus();
  })();
})();
