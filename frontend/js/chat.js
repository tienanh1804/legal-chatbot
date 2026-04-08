const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const chatBox = document.getElementById("chat-box");
const typingIndicator = document.getElementById("typing-indicator");
const suggestionButtons = document.querySelectorAll(".suggestion-btn");
const logoutBtn = document.getElementById("logout-btn");
const loginState = document.getElementById("login-state");
const loginLink = document.getElementById("login-link");
const registerLink = document.getElementById("register-link");
const sendBtn = document.getElementById("send-btn");
const historyList = document.getElementById("history-list");
const newChatBtn = document.getElementById("new-chat-btn");
const chatSessionTitle = document.getElementById("chat-session-title");

let currentConversationId = null;
const DEFAULT_CHAT_TITLE = "Phiên hỏi đáp pháp luật";
const DEFAULT_ASSISTANT_MESSAGE =
  "Xin chào. Tôi có thể hỗ trợ bạn tra cứu thông tin pháp luật. Bạn hãy nhập câu hỏi ở ô phía dưới.";

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function parseSourcesToReferences(sources) {
  if (!sources) return [];
  if (Array.isArray(sources)) return sources;

  return String(sources)
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => ({ law: line, article: "" }));
}

function truncateText(text, maxLength = 60) {
  const cleanText = String(text || "").replace(/\s+/g, " ").trim();
  if (cleanText.length <= maxLength) return cleanText;
  return `${cleanText.slice(0, maxLength).trim()}...`;
}

function updateSessionTitle(title = DEFAULT_CHAT_TITLE) {
  if (chatSessionTitle) {
    chatSessionTitle.textContent = title;
  }
}

function resetChatBox() {
  if (!chatBox) return;
  chatBox.innerHTML = `
    <div class="message ai-message">
      <div class="message-role">Trợ lý AI</div>
      <div class="message-content">${escapeHtml(DEFAULT_ASSISTANT_MESSAGE)}</div>
    </div>
  `;
}

function setActiveHistoryItem(conversationId = null) {
  if (!historyList) return;
  const items = historyList.querySelectorAll(".history-item");
  items.forEach((item) => {
    const isActive = conversationId !== null && Number(item.dataset.conversationId) === Number(conversationId);
    item.classList.toggle("active", isActive);
  });
}

function updateAuthState() {
  const token = localStorage.getItem("token");
  if (!loginState) return;
  if (token) {
    loginState.textContent = "Đã đăng nhập";
    logoutBtn?.classList.remove("d-none");
    loginLink?.classList.add("d-none");
    registerLink?.classList.add("d-none");
  } else {
    loginState.textContent = "Chưa đăng nhập";
    logoutBtn?.classList.add("d-none");
    loginLink?.classList.remove("d-none");
    registerLink?.classList.remove("d-none");
  }
}

function appendMessage(role, content, references = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user-message" : "ai-message"}`;

  let html = `
    <div class="message-role">${role === "user" ? "Người dùng" : "Trợ lý AI"}</div>
    <div class="message-content">${escapeHtml(content).replace(/\n/g, "<br>")}</div>
  `;

  if (references.length > 0) {
    const refsHtml = references.map((ref) => `
      <li>
        <strong>${escapeHtml(ref.law || ref.title || "Văn bản pháp luật")}</strong>
        ${ref.article ? ` - ${escapeHtml(ref.article)}` : ""}
        ${ref.url ? `<div><a href="${escapeHtml(ref.url)}" target="_blank" rel="noopener noreferrer">Mở nguồn</a></div>` : ""}
      </li>
    `).join("");

    html += `
      <div class="reference-box">
        <div class="reference-title">Căn cứ pháp lý</div>
        <ul class="mb-0">${refsHtml}</ul>
      </div>
    `;
  }

  wrapper.innerHTML = html;
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function toggleTyping(show) {
  typingIndicator.classList.toggle("d-none", !show);
  if (sendBtn) sendBtn.disabled = show;
  if (questionInput) questionInput.disabled = show;
  if (newChatBtn) newChatBtn.disabled = show;
}

function createHistoryButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "list-group-item list-group-item-action history-item";
  button.dataset.conversationId = item.conversation_id;
  button.innerHTML = `
    <div class="history-item-title">${escapeHtml(truncateText(item.query_text, 56))}</div>
    <div class="history-item-meta">${new Date(item.created_at).toLocaleString("vi-VN")}</div>
  `;
  button.addEventListener("click", () => loadConversation(item.conversation_id));
  return button;
}

async function loadHistoryList() {
  const token = localStorage.getItem("token");
  if (!historyList) return;

  historyList.innerHTML = "";

  if (!token) {
    const emptyState = document.createElement("div");
    emptyState.className = "history-empty text-muted small";
    emptyState.textContent = "Đăng nhập để xem lịch sử hội thoại.";
    historyList.appendChild(emptyState);
    return;
  }

  try {
    const historyItems = await apiRequest("/history", "GET", null, token, null);

    if (!Array.isArray(historyItems) || historyItems.length === 0) {
      const emptyState = document.createElement("div");
      emptyState.className = "history-empty text-muted small";
      emptyState.textContent = "Chưa có hội thoại nào. Hãy bắt đầu một cuộc trò chuyện mới.";
      historyList.appendChild(emptyState);
      return;
    }

    historyItems.forEach((item) => {
      historyList.appendChild(createHistoryButton(item));
    });
  } catch (error) {
    const errorState = document.createElement("div");
    errorState.className = "history-empty text-danger small";
    errorState.textContent = `Không tải được lịch sử: ${error.message}`;
    historyList.appendChild(errorState);
  }
}

function startNewConversation({ focusInput = true } = {}) {
  currentConversationId = null;
  resetChatBox();
  updateSessionTitle(DEFAULT_CHAT_TITLE);
  setActiveHistoryItem(null);
  if (focusInput) {
    questionInput?.focus();
  }
}

async function loadConversation(conversationId) {
  const token = localStorage.getItem("token");
  if (!token || !conversationId) return;

  toggleTyping(true);

  try {
    const conversation = await apiRequest(
      `/history/conversation/${conversationId}`,
      "GET",
      null,
      token,
      null
    );

    if (!Array.isArray(conversation) || conversation.length === 0) {
      throw new Error("Không tìm thấy nội dung hội thoại.");
    }

    currentConversationId = Number(conversationId);
    chatBox.innerHTML = "";

    conversation.forEach((item) => {
      appendMessage("user", item.query_text);
      appendMessage("ai", item.answer_text, parseSourcesToReferences(item.sources));
    });

    updateSessionTitle(truncateText(conversation[0]?.query_text || DEFAULT_CHAT_TITLE, 70));
    setActiveHistoryItem(currentConversationId);
  } catch (error) {
    startNewConversation({ focusInput: false });
    appendMessage("ai", `Không thể tải hội thoại: ${error.message}`);
  } finally {
    toggleTyping(false);
    questionInput?.focus();
  }
}

async function handleSendQuestion(question) {
  const token = localStorage.getItem("token");

  if (!token) {
    appendMessage("ai", "Bạn cần đăng nhập trước khi sử dụng chức năng hỏi đáp.");
    return;
  }

  const isFirstMessageInConversation = currentConversationId === null;

  appendMessage("user", question);
  if (isFirstMessageInConversation) {
    updateSessionTitle(truncateText(question, 70));
  }
  toggleTyping(true);

  try {
    const data = await apiRequest(
      "/query",
      "POST",
      JSON.stringify({
        query: question,
        conversation_id: currentConversationId
      }),
      token,
      "application/json"
    );

    currentConversationId = data.conversation_id ?? currentConversationId;

    const answer = data?.response?.answer || "Không nhận được câu trả lời từ hệ thống.";
    const references = parseSourcesToReferences(data?.response?.sources);

    appendMessage("ai", answer, references);
    await loadHistoryList();
    setActiveHistoryItem(currentConversationId);
  } catch (error) {
    appendMessage("ai", `Không thể xử lý yêu cầu: ${error.message}`);
  } finally {
    toggleTyping(false);
    questionInput?.focus();
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", async function (event) {
    event.preventDefault();
    const question = questionInput.value.trim();

    if (!question) return;

    questionInput.value = "";
    await handleSendQuestion(question);
  });
}

suggestionButtons.forEach((button) => {
  button.addEventListener("click", async function () {
    const question = button.textContent.trim();
    questionInput.value = "";
    await handleSendQuestion(question);
  });
});

newChatBtn?.addEventListener("click", () => {
  startNewConversation();
});

logoutBtn?.addEventListener("click", async () => {
  localStorage.removeItem("token");
  currentConversationId = null;
  updateAuthState();
  await loadHistoryList();
  startNewConversation({ focusInput: false });
  appendMessage("ai", "Bạn đã đăng xuất.");
});

async function initializeChatPage() {
  updateAuthState();
  startNewConversation({ focusInput: false });
  await loadHistoryList();
}

initializeChatPage();
