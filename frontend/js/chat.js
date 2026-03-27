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

let currentConversationId = null;

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
}

async function handleSendQuestion(question) {
  const token = localStorage.getItem("token");

  if (!token) {
    appendMessage("ai", "Bạn cần đăng nhập trước khi sử dụng chức năng hỏi đáp.");
    return;
  }

  appendMessage("user", question);
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

logoutBtn?.addEventListener("click", () => {
  localStorage.removeItem("token");
  updateAuthState();
  appendMessage("ai", "Bạn đã đăng xuất.");
});

updateAuthState();
