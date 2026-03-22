const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const chatBox = document.getElementById("chat-box");
const typingIndicator = document.getElementById("typing-indicator");
const suggestionButtons = document.querySelectorAll(".suggestion-btn");

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
