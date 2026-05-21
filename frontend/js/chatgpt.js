/* ChatGPT-like frontend UI.
 * Requirements covered:
 * - Sidebar history (search/new/rename/delete)
 * - Chat bubbles + markdown + code blocks
 * - Auto-scroll, typing indicator + skeleton + streaming reveal
 * - Dark mode toggle
 * - Per-conversation state cached in localStorage
 */

// ----------------------------
// DOM refs
// ----------------------------
const sidebarEl = document.getElementById("sidebar");
const convoListEl = document.getElementById("convo-list");
const chatSearchEl = document.getElementById("chat-search");
const newChatBtn = document.getElementById("new-chat-btn");

const conversationTitleEl = document.getElementById("conversation-title");
const messagesEl = document.getElementById("messages");
const typingIndicatorEl = document.getElementById("typing-indicator");

const composerEl = document.getElementById("composer");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const fileInputEl = document.getElementById("file-input");

const loginStateEl = document.getElementById("login-state");
const loginLinkEl = document.getElementById("login-link");
const registerLinkEl = document.getElementById("register-link");
const logoutBtn = document.getElementById("logout-btn");

const themeToggleBtn = document.getElementById("theme-toggle");
const mobileSidebarToggleBtn = document.getElementById(
  "mobile-sidebar-toggle"
);

const accountToggleBtn = document.getElementById("account-toggle-btn");
const accountDropdownEl = document.getElementById("account-dropdown");
const changeNameBtnEl = document.getElementById("change-name-btn");
const changePasswordBtnEl = document.getElementById("change-password-btn");
const deleteAccountBtnEl = document.getElementById("delete-account-btn");
const accountToggleNameEl = document.getElementById("account-toggle-name");

// ----------------------------
// Local state
// ----------------------------
const LS = {
  theme: "legalrag.theme",
  titles: "legalrag.conversationTitles",
  messages: "legalrag.messagesByConversationId",
};

const state = {
  token: null,
  username: null,
  theme: "light",
  // current conversation
  currentConversationId: null, // null means "new chat" (not created yet)
  // sidebar data
  conversations: [], // { conversationId, historyId, title }
  // cached data
  titles: {},
  messagesByCid: {},
  // UI
  isLoading: false,
  /** @type {Record<string, { id: number, filename: string }|undefined>} */
  attachedByCid: {},
  /** @type {Record<string, { sessionId: number, templateId: string, title?: string }|undefined>} */
  procedureByCid: {},
  /** @type {{ loaded: boolean, templates: any[] }} */
  procedureTemplates: { loaded: false, templates: [] },
};

let lastPreviewObjectUrl = null;

const WELCOME_TEXT =
  "Xin chào! Tôi có thể hỗ trợ bạn tra cứu thông tin pháp luật. Bạn hãy nhập câu hỏi ở ô phía dưới.";

// ----------------------------
// Helpers
// ----------------------------
function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function attachConversationKey() {
  return state.currentConversationId === null ? "draft" : String(state.currentConversationId);
}

function getAttachedDocument() {
  return state.attachedByCid[attachConversationKey()] ?? null;
}

function setAttachedDocument(doc) {
  const key = attachConversationKey();
  if (doc && doc.id != null) {
    state.attachedByCid[key] = { id: doc.id, filename: doc.filename };
  } else {
    delete state.attachedByCid[key];
  }
}

function getProcedureSession() {
  const key = attachConversationKey();
  return state.procedureByCid[key] ?? null;
}

function setProcedureSession(sess) {
  const key = attachConversationKey();
  if (sess && sess.sessionId != null) {
    state.procedureByCid[key] = {
      sessionId: sess.sessionId,
      templateId: String(sess.templateId || ""),
      title: sess.title || "",
    };
  } else {
    delete state.procedureByCid[key];
  }
}

async function loadProcedureTemplates(token) {
  if (state.procedureTemplates.loaded) return state.procedureTemplates.templates;
  const data = await apiRequest("/procedures/templates", "GET", null, token);
  const templates = Array.isArray(data?.templates) ? data.templates : [];
  state.procedureTemplates.loaded = true;
  state.procedureTemplates.templates = templates;
  return templates;
}

function normalizeVietnamese(s) {
  return String(s ?? "")
    .toLowerCase()
    .replace(/đ/g, "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function matchProcedureTemplateId(userText, templates) {
  const q = normalizeVietnamese(userText);
  const has = (kw) => q.includes(normalizeVietnamese(kw));

  // Explicit command: /thu_tuc <id>
  const cmd = q.match(/\/(thu_tuc|thutuc|procedure)\s+([a-z0-9_\\-]+)/i);
  if (cmd && cmd[2]) return cmd[2];

  // Divorce
  if (has("ly hon") || has("ly hôn")) {
    if (has("don phuong") || has("đơn phương")) return "ly_hon_don_phuong";
    if (has("thuan tinh") || has("thuận tình")) return "ly_hon_thuan_tinh";
    if (has("thay doi nguoi nuoi con") || has("thay đổi người nuôi con")) {
      return "thay_doi_nguoi_nuoi_con_sau_ly_hon";
    }
    return "ly_hon_thuan_tinh";
  }

  // Land
  if (has("dat dai") || has("đất đai") || has("so do") || has("sổ đỏ") || has("gcn")) {
    if (has("cap doi") || has("cấp đổi")) return "cap_doi_gcn_qsd_dat";
    if (has("lan dau") || has("lần đầu")) return "cap_gcn_qsd_dat_lan_dau";
    if (has("bien dong") || has("biến động")) return "dang_ky_bien_dong_qsd_dat";
    if (has("cung cap thong tin") || has("cung cấp thông tin")) return "cung_cap_thong_tin_dat_dai";
    // default
    return "dang_ky_bien_dong_qsd_dat";
  }

  // Civil status / hộ tịch
  if (has("ho tich") || has("hộ tịch")) {
    if (has("cai chinh") || has("cải chính")) return "cai_chinh_ho_tich";
    if (has("khai sinh") || has("kết hôn") || has("ket hon")) return has("khai sinh") ? "dang_ky_khai_sinh" : "dang_ky_ket_hon";
    return "cai_chinh_ho_tich";
  }
  if (has("khai sinh")) return "dang_ky_khai_sinh";
  if (has("ket hon") || has("kết hôn")) return "dang_ky_ket_hon";

  // Residence
  if (has("tam tru") || has("tạm trú")) return "dang_ky_tam_tru";
  if (has("cu tru") || has("cư trú") || has("xac nhan cu tru") || has("xác nhận cư trú")) return "xac_nhan_thong_tin_cu_tru";

  // Fallback: attempt match by template title keywords
  const safeTemplates = Array.isArray(templates) ? templates : [];
  for (const t of safeTemplates) {
    const tid = String(t?.id ?? "");
    const title = String(t?.title ?? "");
    if (tid && title && q.includes(normalizeVietnamese(title).slice(0, 12))) {
      return tid;
    }
  }
  return null;
}

function isProcedureStartIntent(text) {
  const q = normalizeVietnamese(text);
  return (
    q.includes("lam thu tuc") ||
    q.includes("thuc tuc") ||
    q.includes("thu tuc") ||
    q.includes("lam don") ||
    q.includes("tao don")
  );
}

function isProcedureCancelIntent(text) {
  const q = normalizeVietnamese(text);
  return q === "/huy" || q.includes("huy thu tuc") || q.includes("thoat thu tuc") || q.includes("huy bo thu tuc");
}

function isGeneralLegalQuestion(text) {
  const q = normalizeVietnamese(text);
  // Heuristic: legal Q&A queries (should go to /query), even if a procedure is active.
  return (
    q.includes("?") ||
    q.startsWith("tai sao") ||
    q.startsWith("vi sao") ||
    q.includes("tai sao ") ||
    q.includes("vi sao ") ||
    q.includes("cong van") ||
    q.includes("nghi dinh") ||
    q.includes("thong tu") ||
    q.includes("quyet dinh") ||
    q.includes("van ban") ||
    q.includes("dieu ") ||
    q.includes("khoan ") ||
    q.includes("bo luat") ||
    q.includes("luat ")
  );
}


function getApiBaseUrl() {
  return (
    window.API_BASE_URL ||
    `http://${window.location.hostname || "localhost"}:8002`
  );
}

async function openDocumentPreview(documentId, filename) {
  const token = state.token;
  if (!token) return;
  const modal = document.getElementById("doc-preview-modal");
  const iframe = document.getElementById("doc-preview-iframe");
  const fallback = document.getElementById("doc-preview-fallback");
  const titleEl = document.getElementById("doc-preview-title");
  if (!modal || !iframe) return;
  if (lastPreviewObjectUrl) {
    URL.revokeObjectURL(lastPreviewObjectUrl);
    lastPreviewObjectUrl = null;
  }
  if (fallback) {
    fallback.classList.add("d-none");
    fallback.innerHTML = "";
  }
  iframe.classList.remove("d-none");
  iframe.removeAttribute("src");
  if (titleEl) titleEl.textContent = filename || "Tài liệu";
  try {
    const res = await fetch(`${getApiBaseUrl()}/documents/${documentId}/file`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || res.statusText);
    }
    const blob = await res.blob();
    const mime = blob.type || "application/octet-stream";
    lastPreviewObjectUrl = URL.createObjectURL(blob);
    if (mime.includes("pdf") || mime === "application/octet-stream") {
      iframe.src = lastPreviewObjectUrl;
    } else if (mime.startsWith("image/")) {
      iframe.src = lastPreviewObjectUrl;
    } else {
      iframe.classList.add("d-none");
      if (fallback) {
        fallback.classList.remove("d-none");
        const safeName = escapeHtml(filename || "file");
        fallback.innerHTML = `<p class="doc-preview-fallback-msg">Trình duyệt không xem trực tiếp định dạng này.</p>
          <p><a class="doc-preview-download" href="${lastPreviewObjectUrl}" download="${safeName}">Tải xuống</a></p>`;
      }
    }
    modal.classList.remove("d-none");
  } catch (e) {
    appendInlineError(`Không mở được file: ${e?.message ?? e}`);
  }
}

function closeDocumentPreview() {
  const modal = document.getElementById("doc-preview-modal");
  const iframe = document.getElementById("doc-preview-iframe");
  if (lastPreviewObjectUrl) {
    URL.revokeObjectURL(lastPreviewObjectUrl);
    lastPreviewObjectUrl = null;
  }
  if (iframe) iframe.removeAttribute("src");
  modal?.classList.add("d-none");
}

function appendUserAttachmentMessage(attachment) {
  const msg = {
    role: "user",
    content: "",
    attachment: {
      id: attachment.id,
      filename: attachment.filename,
      mime: attachment.mime || "application/pdf",
    },
  };
  const cid = state.currentConversationId;
  const list = getCachedMessages(cid);
  const next = [...list, msg];
  setCachedMessages(cid, next);
  appendMessageToDom(msg);
}

function bindDocPreviewModal() {
  document.getElementById("doc-preview-close")?.addEventListener("click", closeDocumentPreview);
  document.getElementById("doc-preview-backdrop")?.addEventListener("click", closeDocumentPreview);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDocumentPreview();
  });
}


function setTheme(theme) {
  state.theme = theme;
  document.body.classList.toggle("dark", theme === "dark");
  try {
    localStorage.setItem(LS.theme, theme);
  } catch {
    // ignore
  }
  if (themeToggleBtn) {
    const sunSvg =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 18C15.3137 18 18 15.3137 18 12C18 8.68629 15.3137 6 12 6C8.68629 6 6 8.68629 6 12C6 15.3137 8.68629 18 12 18Z" stroke="currentColor" stroke-width="2"/><path d="M12 2V4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M12 20V22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M4 12H2" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M22 12H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M19.07 4.93L17.66 6.34" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M6.34 17.66L4.93 19.07" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M19.07 19.07L17.66 17.66" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M6.34 6.34L4.93 4.93" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
    const moonSvg =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M21 12.6C19.8 13.2 18.4 13.5 17 13.5C12.6 13.5 9 9.9 9 5.5C9 4.1 9.3 2.7 9.9 1.5C5.4 2.7 2 6.8 2 11.6C2 17.3 6.7 22 12.4 22C17.2 22 21.3 18.6 22.5 14.1C22.1 14.3 21.6 14.5 21 14.8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

    // If currently dark -> show "sun" icon to indicate switching to light.
    themeToggleBtn.innerHTML = theme === "dark" ? sunSvg : moonSvg;
    themeToggleBtn.title = theme === "dark" ? "Switch to light" : "Switch to dark";
  }
}

function loadTheme() {
  let stored = null;
  try {
    stored = localStorage.getItem(LS.theme);
  } catch {
    // ignore
  }
  if (stored === "dark" || stored === "light") {
    setTheme(stored);
    return;
  }
  const prefersDark =
    window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(prefersDark ? "dark" : "light");
}

function loadTitles() {
  try {
    const raw = localStorage.getItem(LS.titles);
    state.titles = raw ? JSON.parse(raw) : {};
  } catch {
    state.titles = {};
  }
}

function loadMessagesCache() {
  try {
    const raw = localStorage.getItem(LS.messages);
    state.messagesByCid = raw ? JSON.parse(raw) : {};
  } catch {
    state.messagesByCid = {};
  }
}

function persistTitles() {
  try {
    localStorage.setItem(LS.titles, JSON.stringify(state.titles));
  } catch {
    // ignore
  }
}

function persistMessages() {
  try {
    localStorage.setItem(LS.messages, JSON.stringify(state.messagesByCid));
  } catch {
    // ignore
  }
}

function getConversationKey(cid) {
  // LocalStorage keys cannot be "null", so we normalize.
  return cid === null ? "draft" : String(cid);
}

function getCachedMessages(cid) {
  const key = getConversationKey(cid);
  return Array.isArray(state.messagesByCid[key]) ? state.messagesByCid[key] : [];
}

function setCachedMessages(cid, messages) {
  const key = getConversationKey(cid);
  state.messagesByCid[key] = Array.isArray(messages) ? messages : [];
  persistMessages();
}

function updateAuthStateUI() {
  const token = state.token;
  if (!loginStateEl) return;

  if (token) {
    loginStateEl.textContent = "Đã đăng nhập";
    // Show account actions
    changeNameBtnEl?.classList.remove("d-none");
    changePasswordBtnEl?.classList.remove("d-none");
    deleteAccountBtnEl?.classList.remove("d-none");
    logoutBtn?.classList.remove("d-none");
    
    // Hide auth links
    loginLinkEl?.classList.add("d-none");
    registerLinkEl?.classList.add("d-none");
    if (loginLinkEl) loginLinkEl.style.display = "none";
    if (registerLinkEl) registerLinkEl.style.display = "none";
    
    // Force display style
    if (changeNameBtnEl) changeNameBtnEl.style.display = "flex";
    if (changePasswordBtnEl) changePasswordBtnEl.style.display = "flex";
    if (deleteAccountBtnEl) deleteAccountBtnEl.style.display = "flex";
    if (logoutBtn) logoutBtn.style.display = "flex";

    if (accountToggleNameEl) {
      accountToggleNameEl.textContent = state.username ?? "Tài khoản";
    }
  } else {
    loginStateEl.textContent = "Chưa đăng nhập";
    logoutBtn?.classList.add("d-none");
    loginLinkEl?.classList.remove("d-none");
    registerLinkEl?.classList.remove("d-none");
    if (loginLinkEl) loginLinkEl.style.display = "flex";
    if (registerLinkEl) registerLinkEl.style.display = "flex";
    
    // Hide account actions
    changeNameBtnEl?.classList.add("d-none");
    changePasswordBtnEl?.classList.add("d-none");
    deleteAccountBtnEl?.classList.add("d-none");
    
    if (changeNameBtnEl) changeNameBtnEl.style.display = "none";
    if (changePasswordBtnEl) changePasswordBtnEl.style.display = "none";
    if (deleteAccountBtnEl) deleteAccountBtnEl.style.display = "none";
    if (logoutBtn) logoutBtn.style.display = "none";
    if (accountToggleNameEl) {
      accountToggleNameEl.textContent = "Tài khoản";
    }
  }
}

function normalizeTitle(rawTitle) {
  const t = String(rawTitle ?? "").trim();
  if (!t) return "Cuộc trò chuyện";
  // Keep it short-ish like ChatGPT.
  return t.length > 64 ? `${t.slice(0, 64)}...` : t;
}

function computeSidebarTitle(conversationId, fallbackTitle) {
  const renamed = state.titles[`${conversationId}`];
  return normalizeTitle(renamed ?? fallbackTitle);
}

function scrollMessagesToBottom() {
  messagesEl?.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}

function showTyping(show) {
  if (!typingIndicatorEl) return;
  typingIndicatorEl.classList.toggle("d-none", !show);
}

function setSendEnabled(enabled) {
  if (!sendBtn) return;
  sendBtn.disabled = !enabled;
}

function autosizeTextarea(el) {
  if (!el) return;
  el.style.height = "0px";
  const next = Math.min(el.scrollHeight, 180);
  el.style.height = `${next}px`;
}

function parseSourcesToReferences(sourcesText) {
  if (!sourcesText) return [];
  if (Array.isArray(sourcesText)) return sourcesText;
  const raw = String(sourcesText);
  const lines = raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  // Heuristic: in this app, sourcesText is a text block; show each line as a "law" row.
  return lines.slice(0, 10).map((line) => ({ law: line, article: "" }));
}

// ----------------------------
// Markdown rendering (safe: escapes HTML first)
// ----------------------------
function markdownToHtml(md) {
  const text = String(md ?? "");

  // Split into code blocks and normal text blocks.
  const fenceRegex = /```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;
  let result = "";
  let lastIndex = 0;
  let match = null;

  function renderInlineSegment(seg) {
    // Escape first.
    let escaped = escapeHtml(seg);

    // Inline code: `code`
    escaped = escaped.replace(/`([^`]+)`/g, (_m, code) => {
      return `<code class="inline-code">${code}</code>`;
    });

    // Bold: **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, (_m, bold) => {
      return `<strong>${bold}</strong>`;
    });

    // Newlines
    escaped = escaped.replace(/\n/g, "<br/>");
    return escaped;
  }

  while ((match = fenceRegex.exec(text)) !== null) {
    const before = text.slice(lastIndex, match.index);
    result += renderInlineSegment(before);

    const code = match[2] ?? "";
    result += `<pre><code>${escapeHtml(code)}</code></pre>`;

    lastIndex = fenceRegex.lastIndex;
  }

  const tail = text.slice(lastIndex);
  result += renderInlineSegment(tail);
  return result;
}

function markdownToPlainHtmlForTyping(md) {
  // Lightweight render while streaming (no markdown parsing yet).
  const escaped = escapeHtml(md);
  return escaped.replace(/\n/g, "<br/>");
}

// ----------------------------
// Rendering messages
// ----------------------------
function createMessageElement(message) {
  const row = document.createElement("div");
  row.className = `message-row ${message.role === "user" ? "user" : "ai"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = message.role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  if (message.attachment && message.role === "user") {
    bubble.classList.add("message-bubble-with-attachment");
  }

  const copyBtn = document.createElement("button");
  copyBtn.className = "copy-btn";
  copyBtn.type = "button";
  copyBtn.textContent = "⧉";
  copyBtn.title = "Copy";
  copyBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(message.content ?? "");
    } catch {
      const ta = document.createElement("textarea");
      ta.value = message.content ?? "";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    copyBtn.textContent = "✓";
    setTimeout(() => (copyBtn.textContent = "⧉"), 900);
  });

  const hasText = String(message.content ?? "").trim().length > 0;
  if (!message.isSkeleton && message.role === "user" && !hasText && message.attachment) {
    copyBtn.classList.add("d-none");
  }

  const contentEl = document.createElement("div");
  contentEl.className = "content";

  if (message.isSkeleton) {
    contentEl.innerHTML = `
      <div class="skeleton-line" style="width: 80%"></div>
      <div class="skeleton-line" style="width: 65%"></div>
      <div class="skeleton-line" style="width: 92%"></div>
    `;
  } else if (!hasText && message.attachment && message.role === "user") {
    contentEl.style.display = "none";
    contentEl.innerHTML = "";
  } else if (message.isHtml) {
    contentEl.innerHTML = message.content ?? "";
  } else {
    contentEl.innerHTML = markdownToHtml(message.content ?? "");
  }

  bubble.appendChild(copyBtn);

  if (!message.isSkeleton && message.attachment && message.role === "user") {
    const att = message.attachment;
    const card = document.createElement("button");
    card.type = "button";
    card.className = "chat-attachment-card";
    const fn = escapeHtml(att.filename);
    const mime = String(att.mime || "");
    const typeLabel = mime.includes("pdf")
      ? "PDF"
      : mime.includes("word") || mime.includes("document")
        ? "Word"
        : mime.includes("image")
          ? "Ảnh"
          : "Tài liệu";
    card.innerHTML = `
      <span class="chat-attachment-icon" aria-hidden="true"></span>
      <span class="chat-attachment-meta">
        <span class="chat-attachment-name">${fn}</span>
        <span class="chat-attachment-type">${typeLabel}</span>
      </span>`;
    card.addEventListener("click", () => {
      openDocumentPreview(att.id, att.filename);
    });
    bubble.appendChild(card);
  }

  bubble.appendChild(contentEl);

  const refs = Array.isArray(message.references) ? message.references : [];
  if (!message.isSkeleton && message.role !== "user" && refs.length > 0) {
    const refBox = document.createElement("div");
    refBox.className = "reference-box";

    const refTitle = document.createElement("div");
    refTitle.className = "reference-title";
    refTitle.textContent = "Căn cứ pháp lý";
    refBox.appendChild(refTitle);

    const ul = document.createElement("ul");
    ul.className = "reference-list";
    refs.forEach((ref) => {
      const li = document.createElement("li");
      li.textContent = ref?.law ?? "";
      ul.appendChild(li);
    });
    refBox.appendChild(ul);
    bubble.appendChild(refBox);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  return { row, contentEl };
}

function clearMessages() {
  if (!messagesEl) return;
  messagesEl.innerHTML = "";
}

function renderConversationMessages(messages, opts = {}) {
  const { animate = true } = opts;
  if (!messagesEl) return;

  clearMessages();

  const safe = Array.isArray(messages) ? messages : [];
  for (const msg of safe) {
    const { row } = createMessageElement(msg);
    messagesEl.appendChild(row);
  }

  if (animate) {
    messagesEl.classList.remove("chat-fade-in");
    // force reflow
    // eslint-disable-next-line no-unused-expressions
    messagesEl.offsetHeight;
    messagesEl.classList.add("chat-fade-in");
  }

  scrollMessagesToBottom();
}

function appendMessageToDom(message) {
  const { row } = createMessageElement(message);
  messagesEl.appendChild(row);
  scrollMessagesToBottom();
  return row;
}

// ----------------------------
// Sidebar / history integration
// ----------------------------
function renderConversationsList() {
  if (!convoListEl) return;

  const q = String(chatSearchEl?.value ?? "").trim().toLowerCase();

  const filtered = state.conversations.filter((c) => {
    if (!q) return true;
    return (c.title ?? "").toLowerCase().includes(q);
  });

  const itemsHtml = filtered
    .map((c) => {
      const active = state.currentConversationId === c.conversationId;
      return `
        <div class="chat-item ${active ? "active" : ""}" data-cid="${
          c.conversationId ?? ""
        }" data-hid="${c.historyId ?? ""}">
          <div class="title">${escapeHtml(c.title ?? "")}</div>
          <div class="actions" aria-hidden="true">
            <button class="icon-btn convo-rename" type="button" title="Rename">✎</button>
            <button class="icon-btn convo-delete" type="button" title="Delete">🗑</button>
          </div>
        </div>
      `;
    })
    .join("");

  convoListEl.innerHTML = itemsHtml;

  // Bind click/select + actions
  const nodes = convoListEl.querySelectorAll(".chat-item");
  nodes.forEach((node) => {
    const cid = Number(node.getAttribute("data-cid"));
    const hidRaw = node.getAttribute("data-hid");
    const historyId = Number(hidRaw);

    node.addEventListener("click", () => {
      if (!Number.isFinite(cid)) return;
      selectConversation(cid);
    });

    const renameBtn = node.querySelector(".convo-rename");
    const deleteBtn = node.querySelector(".convo-delete");

    renameBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      renameConversation(cid);
    });

    deleteBtn?.addEventListener("click", async (e) => {
      e.stopPropagation();
      deleteConversation(historyId, cid);
    });
  });
}

async function fetchConversationsFromBackend() {
  if (!state.token) return [];

  const items = await apiRequest(
    "/history?skip=0&limit=100&group_by_conversation=true",
    "GET",
    null,
    state.token
  );

  const safeItems = Array.isArray(items) ? items : [];
  // items include: id (history id of first message), conversation_id, query_text
  return safeItems
    .filter((x) => x && x.conversation_id != null)
    .map((x) => ({
      conversationId: x.conversation_id,
      historyId: x.id,
      title: computeSidebarTitle(x.conversation_id, x.query_text),
    }));
}

async function loadConversationMessages(cid) {
  const token = state.token;
  if (!token) return [];

  const items = await apiRequest(
    `/history/conversation/${cid}`,
    "GET",
    null,
    token
  );
  const safe = Array.isArray(items) ? items : [];

  // Convert to {role, content, references}
  const messages = [];
  for (const item of safe) {
    messages.push({
      role: "user",
      content: item?.query_text ?? "",
    });
    messages.push({
      role: "ai",
      content: item?.answer_text ?? "",
      references: parseSourcesToReferences(item?.sources),
    });
  }

  return messages;
}

async function reloadSidebar() {
  try {
    state.conversations = await fetchConversationsFromBackend();
    renderConversationsList();
  } catch (e) {
    // if token invalid, just show empty list
    state.conversations = [];
    renderConversationsList();
  }
}

async function selectConversation(cid) {
  state.currentConversationId = cid;
  const current = state.conversations.find((c) => c.conversationId === cid);
  conversationTitleEl.textContent = current?.title ?? "Cuộc trò chuyện";

  // Cached render first
  const cached = getCachedMessages(cid);
  renderConversationMessages(cached, { animate: false });

  // Fetch fresh from backend
  try {
    showTyping(false);
    const messages = await loadConversationMessages(cid);
    // Cache and render
    setCachedMessages(cid, messages);
    renderConversationMessages(messages, { animate: true });
  } catch (e) {
    // fallback: cached already shown
    console.warn("Failed to load conversation", e);
  }
}

function startNewChat() {
  state.currentConversationId = null;
  conversationTitleEl.textContent = "Cuộc trò chuyện mới";
  clearMessages();
  const draftMessages = getCachedMessages(null);
  // For draft, we keep it empty (new chat).
  setCachedMessages(null, []);
  const welcome = { role: "ai", content: WELCOME_TEXT };
  renderConversationMessages([welcome], { animate: true });
  // Sidebar highlight
  renderConversationsList();
  delete state.attachedByCid["draft"];
}

async function renameConversation(cid) {
  const current = state.conversations.find((c) => c.conversationId === cid);
  const currentTitle = current?.title ?? "Cuộc trò chuyện";
  const nextTitle = window.prompt("Nhập tiêu đề chat:", currentTitle);
  if (!nextTitle) return;

  state.titles[String(cid)] = nextTitle;
  persistTitles();

  // Update in-memory titles
  if (current) current.title = computeSidebarTitle(cid, currentTitle);
  conversationTitleEl.textContent = state.currentConversationId === cid ? nextTitle : conversationTitleEl.textContent;

  renderConversationsList();
}

async function deleteConversation(historyId, cid) {
  if (!Number.isFinite(historyId)) return;
  const ok = window.confirm("Xóa cuộc hội thoại này? Thao tác không thể hoàn tác.");
  if (!ok) return;

  try {
    await apiRequest(
      `/history/${historyId}?delete_conversation=true`,
      "DELETE",
      null,
      state.token
    );

    // Clear local cache/title
    delete state.titles[String(cid)];
    persistTitles();
    delete state.messagesByCid[String(cid)];
    persistMessages();

    // If deleting current chat, go to new chat
    if (state.currentConversationId === cid) {
      startNewChat();
    }

    // Reload sidebar
    await reloadSidebar();
  } catch (e) {
    alert(`Không thể xóa: ${e.message}`);
  }
}

// ----------------------------
// Sending / streaming
// ----------------------------
function addUserAndSkeleton(question) {
  const userMessage = { role: "user", content: question };
  const aiSkeleton = { role: "ai", content: "", isSkeleton: true };

  // Update DOM
  const messagesSoFar = getCachedMessages(state.currentConversationId);
  const nextMessages = [...messagesSoFar, userMessage, aiSkeleton];

  // If current cid null (draft), cache under draft.
  const cidKey = state.currentConversationId;
  setCachedMessages(cidKey, nextMessages);

  renderConversationMessages(nextMessages, { animate: false });
  return { userMessage, aiSkeleton };
}

function updateSkeletonToText(messageElement, text, { final = false } = {}) {
  const contentEl = messageElement.querySelector(".content");
  if (!contentEl) return;
  if (final) {
    contentEl.innerHTML = markdownToHtml(text);
  } else {
    contentEl.innerHTML = markdownToPlainHtmlForTyping(text);
  }
}

async function revealStreamingText(messageEl, fullText) {
  const text = String(fullText ?? "");
  const total = text.length;
  if (total === 0) {
    updateSkeletonToText(messageEl, "", { final: true });
    return;
  }

  // Chunk sizes tuned for readability/perf
  const step = Math.max(10, Math.floor(total / 80));
  let revealed = 0;

  return await new Promise((resolve) => {
    const tick = () => {
      revealed = Math.min(total, revealed + step);
      const part = text.slice(0, revealed);
      updateSkeletonToText(messageEl, part, { final: false });
      if (revealed >= total) {
        updateSkeletonToText(messageEl, text, { final: true });
        resolve();
        return;
      }
      setTimeout(tick, 18);
    };
    tick();
  });
}

function upsertReferencesInBubble(bubbleEl, refs) {
  if (!bubbleEl) return;
  const safeRefs = Array.isArray(refs) ? refs : [];

  const existing = bubbleEl.querySelector(".reference-box");
  if (existing) existing.remove();

  if (safeRefs.length === 0) return;

  const refBox = document.createElement("div");
  refBox.className = "reference-box";

  const refTitle = document.createElement("div");
  refTitle.className = "reference-title";
  refTitle.textContent = "Căn cứ pháp lý";
  refBox.appendChild(refTitle);

  const ul = document.createElement("ul");
  ul.className = "reference-list";
  safeRefs.forEach((ref) => {
    const li = document.createElement("li");
    li.textContent = ref?.law ?? "";
    ul.appendChild(li);
  });
  refBox.appendChild(ul);
  bubbleEl.appendChild(refBox);
}

async function downloadWithAuth(url, filenameFallback) {
  const token = state.token;
  if (!token) {
    alert("Bạn cần đăng nhập để tải file.");
    return;
  }
  const res = await fetch(url, {
    method: "GET",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    let msg = "Tải file thất bại";
    try {
      const j = await res.json();
      msg = j?.detail ? String(j.detail) : JSON.stringify(j);
    } catch {
      msg = await res.text();
    }
    alert(msg || "Tải file thất bại");
    return;
  }
  const blob = await res.blob();
  let filename = filenameFallback || "download";
  const cd = res.headers.get("content-disposition") || "";
  const m = cd.match(/filename\\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?/i);
  const raw = decodeURIComponent((m && (m[1] || m[2])) || "").trim();
  if (raw) filename = raw;

  const a = document.createElement("a");
  const objectUrl = URL.createObjectURL(blob);
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
}

function upsertProcedureDownloadActions(bubbleEl, sessionId, templateId) {
  if (!bubbleEl) return;
  const existing = bubbleEl.querySelector(".procedure-download-actions");
  if (existing) existing.remove();
  if (!sessionId) return;

  const box = document.createElement("div");
  box.className = "procedure-download-actions";

  const mkBtn = (label, typeClass) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `procedure-btn ${typeClass}`;
    b.textContent = label;
    return b;
  };

  const base = `${getApiBaseUrl()}/procedures/sessions/${sessionId}/export`;
  const docxBtn = mkBtn("Tải DOCX", "docx");
  docxBtn.addEventListener("click", () =>
    downloadWithAuth(`${base}?format=docx`, `procedure_${sessionId}.docx`)
  );

  box.appendChild(docxBtn);

  if (templateId) {
    const blankBtn = mkBtn("Tải biểu mẫu trống", "blank");
    blankBtn.addEventListener("click", () =>
      downloadWithAuth(
        `${getApiBaseUrl()}/procedures/templates/${templateId}/download-blank`,
        `${templateId}_trong.docx`
      )
    );
    box.appendChild(blankBtn);
  }

  bubbleEl.appendChild(box);
}

const CATEGORY_MAP = {
  giao_thong: {
    label: "🚗 Giao thông",
    templates: [
      { id: "de_nghi_giai_quyet_tai_nan_giao_thong", title: "Đơn đề nghị giải quyết tai nạn giao thông" },
      { id: "khoi_kien_gay_tai_nan_giao_thong", title: "Đơn khởi kiện vụ án giao thông" }
    ]
  },
  dan_su: {
    label: "⚖️ Dân sự",
    templates: [
      { id: "don_khieu_nai", title: "Đơn khiếu nại (Mẫu số 01)" },
      { id: "don_xin_tam_hoan_nvqs", title: "Đơn xin tạm hoãn nghĩa vụ quân sự" },
      { id: "mau_don_to_cao", title: "Đơn tố cáo hành vi lừa đảo" },
      { id: "don_to_giac_toi_pham", title: "Đơn tố giác tội phạm" }
    ]
  },
  lao_dong: {
    label: "💼 Lao động",
    templates: [
      { id: "khoi_kien_tranh_chap_lao_dong", title: "Đơn khởi kiện tranh chấp lao động" },
      { id: "de_nghi_ki_tiep_hop_dong_lao_dong", title: "Đơn đề nghị ký tiếp hợp đồng lao động" },
      { id: "de_nghi_huong_tro_cap_that_nghiep", title: "Đơn đề nghị hưởng trợ cấp thất nghiệp" }
    ]
  },
  khac: {
    label: "📁 Khác",
    templates: [
      { id: "don_thuan_tinh_ly_hon_mau", title: "Đơn thuận tình ly hôn chính thức" },
      { id: "don_xin_hoc_them", title: "Đơn xin học lớp bồi dưỡng kiến thức" },
      { id: "don_xin_xac_nhan_gia_dinh_kho_khan", title: "Đơn xin xác nhận hoàn cảnh khó khăn" },
      { id: "don_ly_hon_don_phuong_mau", title: "Đơn ly hôn đơn phương chính thức" }
    ]
  }
};

function renderCategoryMenuHtml() {
  return `
    <div style="margin-bottom: 8px;">Chào bạn! Vui lòng chọn nhóm thủ tục hành chính bạn cần thực hiện dưới đây:</div>
    <div class="procedure-menu-box">
      <button type="button" class="procedure-cat-btn" data-cat="giao_thong">🚗 Giao thông</button>
      <button type="button" class="procedure-cat-btn" data-cat="dan_su">⚖️ Dân sự</button>
      <button type="button" class="procedure-cat-btn" data-cat="lao_dong">💼 Lao động</button>
      <button type="button" class="procedure-cat-btn" data-cat="khac">📁 Khác</button>
    </div>
  `;
}

function renderTemplateMenuHtml(catKey) {
  const cat = CATEGORY_MAP[catKey];
  if (!cat) return renderCategoryMenuHtml();
  
  const buttonsHtml = cat.templates.map(t => {
    return `<button type="button" class="procedure-tmpl-btn" data-tid="${t.id}">📄 ${t.title}</button>`;
  }).join("");
  
  return `
    <div style="margin-bottom: 8px;">Bạn đã chọn lĩnh vực <strong>${cat.label}</strong>. Vui lòng chọn biểu mẫu cần lập:</div>
    <div class="procedure-menu-box">
      ${buttonsHtml}
      <button type="button" class="procedure-back-btn">⬅️ Quay lại</button>
    </div>
  `;
}

function updateLastAiMessage(html) {
  const cid = state.currentConversationId;
  const messages = getCachedMessages(cid);
  if (messages.length === 0) return;
  
  let lastAiIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "ai") {
      lastAiIdx = i;
      break;
    }
  }
  
  if (lastAiIdx !== -1) {
    messages[lastAiIdx].content = html;
    messages[lastAiIdx].isHtml = true;
    setCachedMessages(cid, messages);
    
    const aiRows = messagesEl.querySelectorAll(".message-row.ai");
    if (aiRows.length > 0) {
      const lastAiRow = aiRows[aiRows.length - 1];
      const bubbleEl = lastAiRow.querySelector(".message-bubble");
      const contentEl = bubbleEl.querySelector(".content");
      if (contentEl) {
        contentEl.innerHTML = html;
      }
    }
  }
}

async function startProcedureFromMenu(tid) {
  const token = state.token;
  if (!token) {
    appendInlineError("Bạn cần đăng nhập để làm thủ tục.");
    return;
  }

  state.isLoading = true;
  setSendEnabled(false);
  showTyping(true);

  try {
    const templates = await loadProcedureTemplates(token);
    const selectedTemplate = templates.find(x => x.id === tid) || { title: tid };

    updateLastAiMessage(`Đã chọn biểu mẫu: <strong>${selectedTemplate.title}</strong>`);

    const started = await apiRequest(
      "/procedures/sessions",
      "POST",
      JSON.stringify({ template_id: tid }),
      token,
      "application/json"
    );

    setProcedureSession({
      sessionId: started?.session_id,
      templateId: started?.template_id ?? tid,
      title: selectedTemplate.title,
    });

    const firstQuestion = started?.question || "Câu hỏi đầu tiên chưa sẵn sàng.";
    const answer = `Bắt đầu thủ tục: **${selectedTemplate.title}**\n\nMình sẽ hỏi lần lượt để bạn điền thông tin. (Gõ "hủy" để dừng)\n\n${firstQuestion}`;
    
    const cid = state.currentConversationId;
    const messages = getCachedMessages(cid);
    const newAiMsg = { role: "ai", content: answer, references: [] };
    const next = [...messages, newAiMsg];
    
    setCachedMessages(cid, next);
    showTyping(false);
    
    const row = appendMessageToDom(newAiMsg);
    const bubbleEl = row.querySelector(".message-bubble");
    if (bubbleEl) {
      await revealStreamingText(bubbleEl, answer);
    }
  } catch (e) {
    appendInlineError(`Không thể khởi tạo thủ tục: ${e?.message ?? e}`);
  } finally {
    state.isLoading = false;
    setSendEnabled(true);
    showTyping(false);
  }
}

async function handleSend(question) {
  if (state.isLoading) return;
  const token = state.token;
  if (!token) {
    appendInlineError("Bạn cần đăng nhập trước khi sử dụng chức năng hỏi đáp.");
    return;
  }

  const trimmed = String(question ?? "").trim();
  if (!trimmed) return;

  state.isLoading = true;
  setSendEnabled(false);
  showTyping(true);

  // UX: autosize + disable quickly
  composerEl?.focus();

  addUserAndSkeleton(trimmed);

  // Identify the last AI bubble skeleton element for streaming update.
  const aiBubble = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
  const aiMessageEl = aiBubble?.querySelector(".message-bubble")?.parentElement
    ? aiBubble.querySelector(".message-bubble")
    : aiBubble;

  try {
    const prevCid = state.currentConversationId;
    const att = getAttachedDocument();
    const activeProc = getProcedureSession();

    // Start or Switch procedure intent:
    if (isProcedureStartIntent(trimmed)) {
      setProcedureSession(null); // Clear active session

      const answer = renderCategoryMenuHtml();
      const beforeMessages = getCachedMessages(prevCid);
      const next = beforeMessages.map((m, idx) => {
        const isLastAiSkeleton = m.role === "ai" && m.isSkeleton;
        if (idx === beforeMessages.length - 1 && isLastAiSkeleton) {
          return { role: "ai", content: answer, isHtml: true, references: [] };
        }
        return m;
      });
      setCachedMessages(state.currentConversationId, next);
      showTyping(false);
      const lastAiRow = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
      const lastBubbleEl = lastAiRow?.querySelector(".message-bubble");
      if (lastBubbleEl) {
        updateSkeletonToText(lastBubbleEl, answer, { final: true });
        const contentEl = lastBubbleEl.querySelector(".content");
        if (contentEl) contentEl.innerHTML = answer;
      }
      await reloadSidebar();
      if (state.currentConversationId != null) renderConversationsList();
      state.isLoading = false;
      setSendEnabled(true);
      showTyping(false);
      return;
    }

    // Cancel procedure flow
    if (activeProc && isProcedureCancelIntent(trimmed)) {
      setProcedureSession(null);
      const answer = "Đã hủy phiên làm thủ tục. Bạn có thể hỏi pháp luật hoặc bắt đầu thủ tục khác.";
      const references = [];
      const beforeMessages = getCachedMessages(prevCid);
      const next = beforeMessages.map((m, idx) => {
        const isLastAiSkeleton = m.role === "ai" && m.isSkeleton;
        if (idx === beforeMessages.length - 1 && isLastAiSkeleton) {
          return { role: "ai", content: answer, references: references };
        }
        return m;
      });
      setCachedMessages(state.currentConversationId, next);
      showTyping(false);
      const lastAiRow = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
      const lastBubbleEl = lastAiRow?.querySelector(".message-bubble");
      if (lastBubbleEl) {
        await revealStreamingText(lastBubbleEl, answer);
        upsertReferencesInBubble(lastBubbleEl, references);
      }
      await reloadSidebar();
      if (state.currentConversationId != null) renderConversationsList();
      state.isLoading = false;
      setSendEnabled(true);
      showTyping(false);
      return;
    }

    // Procedure wizard flow: if we have an active procedure session, send message to procedure endpoint.
    // BUT: if the user asks a general legal question, route to /query instead.
    if (activeProc && activeProc.sessionId && !isGeneralLegalQuestion(trimmed)) {
      const body = await apiRequest(
        `/procedures/sessions/${activeProc.sessionId}/message`,
        "POST",
        JSON.stringify({ message: trimmed }),
        token,
        "application/json"
      );
      const done = !!body?.complete;
      const question = body?.question;
      const preview = body?.preview_text;
      const answer = done
        ? `Đã thu thập đủ thông tin.\n\nBản nháp:\n${preview || ""}\n\nBạn có thể tải file ở các nút bên dưới.`
        : String(question || "Bạn vui lòng cung cấp thêm thông tin.");
      const references = [];
      const beforeMessages = getCachedMessages(prevCid);
      const next = beforeMessages.map((m, idx) => {
        const isLastAiSkeleton = m.role === "ai" && m.isSkeleton;
        if (idx === beforeMessages.length - 1 && isLastAiSkeleton) {
          return { role: "ai", content: answer, references: references };
        }
        return m;
      });
      setCachedMessages(state.currentConversationId, next);
      showTyping(false);
      const lastAiRow = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
      const lastBubbleEl = lastAiRow?.querySelector(".message-bubble");
      if (lastBubbleEl) {
        await revealStreamingText(lastBubbleEl, answer);
        upsertReferencesInBubble(lastBubbleEl, references);
        if (done) {
          upsertProcedureDownloadActions(lastBubbleEl, activeProc.sessionId, activeProc.templateId);
          // Auto-exit procedure mode after completion so subsequent messages go to /query.
          setProcedureSession(null);
        }
      }
      await reloadSidebar();
      if (state.currentConversationId != null) renderConversationsList();
      state.isLoading = false;
      setSendEnabled(true);
      showTyping(false);
      return;
    }



    const payload = {
      query: trimmed,
      conversation_id: state.currentConversationId,
      include_user_documents: true,
      user_documents_only: !!(att && att.id),
      user_document_id: att && att.id != null ? att.id : null,
    };

    const data = await apiRequest("/query", "POST", JSON.stringify(payload), token, "application/json");

    const newCid = data?.conversation_id ?? state.currentConversationId;
    if (prevCid === null && newCid != null) {
      const draftAtt = state.attachedByCid["draft"];
      if (draftAtt) {
        state.attachedByCid[String(newCid)] = draftAtt;
        delete state.attachedByCid["draft"];
      }
    }
    state.currentConversationId = newCid;
    conversationTitleEl.textContent = computeSidebarTitle(
      newCid,
      trimmed
    );

    // Build final message
    const answer = data?.response?.answer ?? "";
    const references = parseSourcesToReferences(data?.response?.sources);

    // Cache update:
    const beforeMessages = getCachedMessages(prevCid);
    // Replace the skeleton with final AI content.
    // Find last message index (ai skeleton)
    const next = beforeMessages.map((m, idx) => {
      const isLastAiSkeleton = m.role === "ai" && m.isSkeleton;
      if (idx === beforeMessages.length - 1 && isLastAiSkeleton) {
        return { role: "ai", content: answer, references: references };
      }
      return m;
    });

    // If conversation id was created now, migrate cache from draft to cid.
    if (prevCid === null && newCid != null) {
      setCachedMessages(newCid, next);
      setCachedMessages(null, []); // clear draft
    } else {
      setCachedMessages(newCid, next);
    }

    showTyping(false);

    // Streaming reveal into the last AI bubble
    // Locate last ai bubble DOM again after re-render.
    const lastAiRow = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
    const lastBubbleEl = lastAiRow?.querySelector(".message-bubble");
    if (lastBubbleEl) {
      await revealStreamingText(lastBubbleEl, answer);
      upsertReferencesInBubble(lastBubbleEl, references);
    } else {
      // fallback: re-render final markdown
      renderConversationMessages(next, { animate: false });
    }

    // Refresh sidebar when first conversation created or when errors could have saved history.
    await reloadSidebar();
    if (state.currentConversationId != null) {
      renderConversationsList();
    }
  } catch (e) {
    // Backend might have stored history; ensure sidebar is refreshed for UX.
    try {
      await reloadSidebar();
      renderConversationsList();
    } catch {
      // ignore
    }

    const message = e?.message ?? String(e);
    showTyping(false);
    setSendEnabled(true);
    state.isLoading = false;

    // Replace skeleton with error
    const lastAiRow = Array.from(messagesEl.querySelectorAll(".message-row.ai")).pop();
    const lastBubbleEl = lastAiRow?.querySelector(".message-bubble");
    if (lastBubbleEl) {
      updateSkeletonToText(lastBubbleEl, message, { final: true });
    }

    // If we were still on a "draft" chat (conversation_id was not returned),
    // backend may have created a new conversation during ValueError handling.
    // Switch UI to the latest conversation so the user sees the "new chat" entry.
    if (state.currentConversationId === null && state.conversations.length > 0) {
      const latest = state.conversations[0];
      state.currentConversationId = latest.conversationId;
      conversationTitleEl.textContent = latest.title;
      try {
        await selectConversation(latest.conversationId);
      } catch {
        // ignore: DOM fallback is already present
      }
    }

    state.isLoading = false;
    setSendEnabled(true);
    scrollMessagesToBottom();
    return;
  }

  state.isLoading = false;
  setSendEnabled(true);
  showTyping(false);
  scrollMessagesToBottom();
}

function appendInlineError(text) {
  const errorMsg = { role: "ai", content: text };
  appendMessageToDom(errorMsg);
}

// Helper: convert backend error JSON detail into readable message
// (apiRequest already returns `errorMessage = errorData.detail || ...`)

// ----------------------------
// Event binding
// ----------------------------
function bindUI() {
  // Bind click listener for interactive procedure menu
  messagesEl?.addEventListener("click", async (e) => {
    const catBtn = e.target.closest(".procedure-cat-btn");
    const tmplBtn = e.target.closest(".procedure-tmpl-btn");
    const backBtn = e.target.closest(".procedure-back-btn");

    if (catBtn) {
      e.stopPropagation();
      const cat = catBtn.getAttribute("data-cat");
      const html = renderTemplateMenuHtml(cat);
      updateLastAiMessage(html);
      return;
    }

    if (backBtn) {
      e.stopPropagation();
      const html = renderCategoryMenuHtml();
      updateLastAiMessage(html);
      return;
    }

    if (tmplBtn) {
      e.stopPropagation();
      const tid = tmplBtn.getAttribute("data-tid");
      await startProcedureFromMenu(tid);
      return;
    }
  });

  themeToggleBtn?.addEventListener("click", () => {
    const next = state.theme === "dark" ? "light" : "dark";
    setTheme(next);
  });

  mobileSidebarToggleBtn?.addEventListener("click", () => {
    if (!sidebarEl) return;
    const isHidden = sidebarEl.style.display === "none";
    sidebarEl.style.display = isHidden ? "flex" : "none";
  });

  chatSearchEl?.addEventListener("input", () => {
    renderConversationsList();
  });

  newChatBtn?.addEventListener("click", () => {
    startNewChat();
  });

  logoutBtn?.addEventListener("click", () => {
    localStorage.removeItem("token");
    state.token = null;
    state.username = null;
    updateAuthStateUI();
    accountDropdownEl?.classList.add("d-none");
    startNewChat();
  });

  // Account dropdown
  accountToggleBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    accountDropdownEl?.classList.toggle("d-none");
  });
  document.addEventListener("click", () => {
    accountDropdownEl?.classList.add("d-none");
  });


  // Sidebar suggestions (2-3 questions)
  const sidebarSuggestions =
    document.querySelectorAll("#sidebar-suggestions .suggestion-btn") || [];
  sidebarSuggestions.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const q = btn.textContent.trim();
      if (!q) return;
      composerEl.value = q;
      autosizeTextarea(composerEl);
      composerEl.value = "";
      autosizeTextarea(composerEl);
      await handleSend(q);
      accountDropdownEl?.classList.add("d-none");
    });
  });

  // textarea auto resize
  composerEl?.addEventListener("input", () => autosizeTextarea(composerEl));

  // Enter to send, Shift+Enter newline
  composerEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const q = composerEl.value;
      composerEl.value = "";
      autosizeTextarea(composerEl);
      void handleSend(q);
    }
  });

  // Send button
  sendBtn?.addEventListener("click", () => {
    const q = composerEl.value;
    if (!q.trim()) return;
    composerEl.value = "";
    autosizeTextarea(composerEl);
    void handleSend(q);
  });

  uploadBtn?.addEventListener("click", () => fileInputEl?.click());
  fileInputEl?.addEventListener("change", async () => {
    if (!fileInputEl.files || fileInputEl.files.length === 0) return;
    const token = state.token;
    if (!token) {
      appendInlineError("Bạn cần đăng nhập để tải tài liệu lên.");
      fileInputEl.value = "";
      return;
    }
    const f = fileInputEl.files[0];
    fileInputEl.value = "";
    try {
      const res = await apiUploadFile("/documents/upload", f, token);
      setAttachedDocument({
        id: res.id,
        filename: res.filename,
      });
      appendUserAttachmentMessage({
        id: res.id,
        filename: res.filename,
        mime: res.mime_type,
      });
      appendInlineError(
        `Đã tải lên: ${res.filename} (${res.chunks_indexed ?? 0} đoạn đã lập chỉ mục). Nhấn vào thẻ file trong khung chat để xem trước.`
      );
    } catch (e) {
      appendInlineError(String(e?.message ?? e));
    }
  });
}

// ----------------------------
// Auth + init
// ----------------------------
async function verifyToken() {
  const token = localStorage.getItem("token");
  state.token = token;
  if (!token) {
    updateAuthStateUI();
    return;
  }

  try {
    const verify = await apiRequest("/auth/verify", "GET", null, token);
    state.username = verify?.username ?? null;
    state.token = token;
    updateAuthStateUI();
  } catch {
    localStorage.removeItem("token");
    state.token = null;
    state.username = null;
    updateAuthStateUI();
  }
}

async function init() {
  loadTheme();
  loadTitles();
  loadMessagesCache();
  updateAuthStateUI();

  bindUI();
  bindDocPreviewModal();
  setSendEnabled(true);

  // Default UI
  conversationTitleEl.textContent = "Cuộc trò chuyện mới";
  composerEl && autosizeTextarea(composerEl);

  await verifyToken();

  // Load sidebar + draft
  if (state.token) {
    await reloadSidebar();
    // Select the latest conversation if exists
    if (state.conversations.length > 0) {
      const latest = state.conversations[0];
      // Conversation list ordering should be newest first (backend sorts by created_at desc).
      state.currentConversationId = latest.conversationId;
      conversationTitleEl.textContent = latest.title;
      // Try cached immediately
      const cached = getCachedMessages(latest.conversationId);
      renderConversationMessages(cached, { animate: false });
      // Then load fresh
      try {
        const msgs = await loadConversationMessages(latest.conversationId);
        setCachedMessages(latest.conversationId, msgs);
        renderConversationMessages(msgs, { animate: true });
      } catch {
        // keep cache
      }
    } else {
      startNewChat();
    }
    renderConversationsList();
  } else {
    startNewChat();
  }
}

init().catch((e) => {
  console.error(e);
  showTyping(false);
  setSendEnabled(true);
});

// ----------------------------
// Settings Modal Logic
// ----------------------------
const settingsModal = document.getElementById("settings-modal");
const settingsClose = document.getElementById("settings-close");
const settingsBackdrop = document.getElementById("settings-backdrop");

function closeSettings() {
  settingsModal?.classList.add("d-none");
  // Clear inputs
  document.getElementById("new-username-input").value = "";
  document.getElementById("old-password-input").value = "";
  document.getElementById("new-password-input").value = "";
  document.getElementById("delete-password-input").value = "";
}

function openSettings(sectionId) {
  // Hide all sections
  document.querySelectorAll(".settings-section").forEach((s) => s.classList.add("d-none"));
  // Show target section
  document.getElementById(sectionId)?.classList.remove("d-none");
  // Update title
  const titles = {
    "change-name-section": "Đổi tên tài khoản",
    "change-password-section": "Đổi mật khẩu",
    "delete-account-section": "Xóa tài khoản",
  };
  const titleEl = document.getElementById("settings-title");
  if (titleEl) titleEl.textContent = titles[sectionId] || "Cài đặt";

  settingsModal?.classList.remove("d-none");
  accountDropdownEl?.classList.add("d-none");
}

settingsClose?.addEventListener("click", closeSettings);
settingsBackdrop?.addEventListener("click", closeSettings);

changeNameBtnEl?.addEventListener("click", () => openSettings("change-name-section"));
changePasswordBtnEl?.addEventListener("click", () => openSettings("change-password-section"));
deleteAccountBtnEl?.addEventListener("click", () => openSettings("delete-account-section"));

document.getElementById("submit-change-name")?.addEventListener("click", async () => {
  const newUsername = document.getElementById("new-username-input").value.trim();
  if (!newUsername) return alert("Vui lòng nhập tên mới");

  try {
    const data = await apiRequest("/users/update-username", "POST", JSON.stringify({ new_username: newUsername }), state.token);
    state.username = data.new_username;
    updateAuthStateUI();
    alert("Đổi tên thành công");
    closeSettings();
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById("submit-change-password")?.addEventListener("click", async () => {
  const oldPassword = document.getElementById("old-password-input").value;
  const newPassword = document.getElementById("new-password-input").value;
  if (!oldPassword || !newPassword) return alert("Vui lòng nhập đầy đủ thông tin");

  try {
    await apiRequest("/users/change-password", "POST", JSON.stringify({ old_password: oldPassword, new_password: newPassword }), state.token);
    alert("Đổi mật khẩu thành công");
    closeSettings();
  } catch (e) {
    alert(e.message);
  }
});

document.getElementById("submit-delete-account")?.addEventListener("click", async () => {
  const password = document.getElementById("delete-password-input").value;
  if (!password) return alert("Vui lòng nhập mật khẩu để xác nhận");

  if (!confirm("Bạn có chắc chắn muốn xóa tài khoản? Hành động này không thể hoàn tác.")) return;

  try {
    await apiRequest("/users/delete-account", "POST", JSON.stringify({ password }), state.token);
    alert("Tài khoản đã được xóa");
    localStorage.removeItem("token");
    window.location.reload();
  } catch (e) {
    alert(e.message);
  }
});
