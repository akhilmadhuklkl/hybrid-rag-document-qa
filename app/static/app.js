const elements = {
  browse: document.querySelector("#browse-button"),
  fileInput: document.querySelector("#file-input"),
  dropZone: document.querySelector("#drop-zone"),
  progress: document.querySelector("#upload-progress"),
  documentList: document.querySelector("#document-list"),
  documentTotal: document.querySelector("#document-total"),
  conversationTotal: document.querySelector("#conversation-total"),
  conversationList: document.querySelector("#conversation-list"),
  refresh: document.querySelector("#refresh-documents"),
  status: document.querySelector("#api-status"),
  statusDot: document.querySelector("#status-dot"),
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome-state"),
  form: document.querySelector("#query-form"),
  input: document.querySelector("#question-input"),
  send: document.querySelector("#send-button"),
  toast: document.querySelector("#toast"),
};

const showToast = (message) => {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.setTimeout(() => elements.toast.classList.remove("visible"), 3200);
};

const api = async (path, options = {}) => {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) { /* no JSON */ }
    throw new Error(message);
  }
  return response;
};

const checkHealth = async () => {
  try {
    await api("/health");
    elements.status.textContent = "API connected";
    elements.statusDot.classList.add("online");
  } catch (_) {
    elements.status.textContent = "API unavailable";
    elements.statusDot.classList.remove("online");
  }
};

const fileType = (source) => source.split(".").pop().slice(0, 4).toUpperCase();

const renderDocuments = (payload) => {
  elements.documentTotal.textContent = `${payload.total_documents} document${payload.total_documents === 1 ? "" : "s"}`;
  elements.documentList.replaceChildren();
  if (!payload.documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-library";
    empty.textContent = "Your library is empty. Upload a company document to begin asking grounded questions.";
    elements.documentList.append(empty);
    return;
  }
  payload.documents.forEach((documentInfo) => {
    const item = document.createElement("div");
    item.className = "document-item";

    const type = document.createElement("div");
    type.className = "document-type";
    type.textContent = fileType(documentInfo.source);

    const meta = document.createElement("div");
    meta.className = "document-meta";
    const name = document.createElement("strong");
    name.textContent = documentInfo.title || documentInfo.source;
    name.title = documentInfo.source;
    const details = document.createElement("span");
    details.textContent = `${documentInfo.chunk_count} chunks${documentInfo.page_count ? ` · ${documentInfo.page_count} pages` : ""}`;
    meta.append(name, details);

    const remove = document.createElement("button");
    remove.className = "delete-document";
    remove.type = "button";
    remove.title = `Delete ${documentInfo.source}`;
    remove.setAttribute("aria-label", `Delete ${documentInfo.source}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteDocument(documentInfo));
    item.append(type, meta, remove);
    elements.documentList.append(item);
  });
};

const loadDocuments = async () => {
  try {
    const response = await api("/documents");
    renderDocuments(await response.json());
  } catch (error) { showToast(error.message); }
};

const loadConversations = async () => {
  try {
    const response = await api("/conversations");
    const conversations = await response.json();
    elements.conversationTotal.textContent = `${conversations.length} conversation${conversations.length === 1 ? "" : "s"}`;
    elements.conversationList.replaceChildren();
    conversations.slice(0, 5).forEach((conversation) => {
      const button = document.createElement("button");
      button.className = "conversation-item";
      button.type = "button";
      button.textContent = conversation.title;
      button.title = conversation.title;
      button.addEventListener("click", () => loadConversation(conversation.conversation_id));
      elements.conversationList.append(button);
    });
  } catch (_) { /* history is optional if the database is unavailable */ }
};

const loadConversation = async (conversationId) => {
  try {
    const response = await api(`/conversations/${encodeURIComponent(conversationId)}`);
    const conversation = await response.json();
    elements.conversation.replaceChildren();
    conversation.messages.forEach((message) => {
      if (message.role === "user") addUserMessage(message.content);
      else {
        const assistant = addAssistantMessage();
        assistant.answer.classList.remove("typing");
        assistant.answer.textContent = message.content;
        renderCompletion(assistant.bubble, { ...conversation, ...message, grounded: true, generation_mode: "history", reranked: false });
      }
    });
  } catch (error) { showToast(error.message); }
};

const deleteDocument = async (documentInfo) => {
  if (!window.confirm(`Delete “${documentInfo.title || documentInfo.source}” and all indexed chunks?`)) return;
  try {
    await api(`/documents/${encodeURIComponent(documentInfo.document_id)}`, { method: "DELETE" });
    showToast("Document removed");
    await loadDocuments();
  } catch (error) { showToast(error.message); }
};

const uploadFiles = async (files) => {
  if (!files.length) return;
  const form = new FormData();
  [...files].forEach((file) => form.append("files", file));
  elements.progress.hidden = false;
  try {
    const response = await api("/documents/upload/async", { method: "POST", body: form });
    const result = await response.json();
    const duplicateText = result.duplicates_skipped.length ? ` · ${result.duplicates_skipped.length} duplicate skipped` : "";
    if (!result.jobs.length) showToast(`No new files to index${duplicateText}`);
    else { showToast(`Indexing started${duplicateText}`); await pollJobs(result.jobs.map((job) => job.job_id)); }
    await loadDocuments();
  } catch (error) { showToast(error.message); }
  finally { elements.progress.hidden = true; elements.fileInput.value = ""; }
};

const pollJobs = async (jobIds) => {
  let remaining = new Set(jobIds);
  while (remaining.size) {
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    const statuses = await Promise.all([...remaining].map(async (id) => (await api(`/jobs/${id}`)).json()));
    statuses.forEach((job) => { if (["completed", "failed"].includes(job.status)) remaining.delete(job.job_id); });
    const active = statuses.find((job) => ["processing", "queued"].includes(job.status));
    if (active) showToast(`Indexing ${active.status}…`);
    const failed = statuses.find((job) => job.status === "failed");
    if (failed) throw new Error(failed.error || "Document indexing failed");
  }
  showToast("Documents indexed successfully");
};

const addUserMessage = (text) => {
  elements.welcome?.remove();
  const message = document.createElement("div");
  message.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  message.append(bubble);
  elements.conversation.append(message);
};

const addAssistantMessage = () => {
  const message = document.createElement("div");
  message.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  const label = document.createElement("div");
  label.className = "assistant-label";
  label.textContent = "Atlas answer";
  const answer = document.createElement("div");
  answer.className = "answer-text typing";
  bubble.append(label, answer);
  message.append(bubble);
  elements.conversation.append(message);
  return { bubble, answer };
};

const renderCompletion = (bubble, metadata) => {
  const grid = document.createElement("div");
  grid.className = "citation-grid";
  metadata.citations.forEach((citation) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    const head = document.createElement("div");
    head.className = "citation-head";
    const index = document.createElement("span");
    index.className = "citation-index";
    index.textContent = citation.index;
    const title = document.createElement("a");
    title.href = citation.document_id ? `/documents/${encodeURIComponent(citation.document_id)}/file` : "#";
    title.target = "_blank";
    title.rel = "noreferrer";
    title.textContent = `${citation.title || citation.source}${citation.page ? ` · p.${citation.page}` : ""}`;
    const score = document.createElement("span");
    score.className = "evidence";
    score.textContent = `${Math.round(citation.evidence_score * 100)}%`;
    head.append(index, title, score);
    const excerpt = document.createElement("p");
    excerpt.textContent = citation.text;
    card.append(head, excerpt);
    grid.append(card);
  });
  if (metadata.citations.length) bubble.append(grid);

  const meta = document.createElement("div");
  meta.className = "answer-meta";
  const grounded = document.createElement("span");
  grounded.textContent = metadata.grounded ? "✓ Grounded" : "Unverified";
  const mode = document.createElement("span");
  mode.textContent = metadata.generation_mode.replaceAll("_", " ");
  meta.append(grounded, mode);
  if (metadata.reranked) {
    const reranked = document.createElement("span");
    reranked.textContent = "Reranked";
    meta.append(reranked);
  }
  bubble.append(meta);
};

const parseEvent = (block) => {
  const type = block.match(/^event: (.+)$/m)?.[1];
  const data = block.match(/^data: (.+)$/m)?.[1];
  return type && data ? { type, data: JSON.parse(data) } : null;
};

const askQuestion = async (question) => {
  addUserMessage(question);
  const assistant = addAssistantMessage();
  elements.send.disabled = true;
  elements.input.disabled = true;
  try {
    const response = await api("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      blocks.map(parseEvent).filter(Boolean).forEach((event) => {
        if (event.type === "token") assistant.answer.textContent += event.data.text;
        if (event.type === "complete") {
          assistant.answer.classList.remove("typing");
          renderCompletion(assistant.bubble, event.data);
          loadConversations();
        }
      });
      if (done) break;
    }
  } catch (error) {
    assistant.answer.classList.remove("typing");
    assistant.answer.textContent = `I couldn't complete that request. ${error.message}`;
  } finally {
    elements.send.disabled = false;
    elements.input.disabled = false;
    elements.input.focus();
    elements.conversation.scrollTo({ top: elements.conversation.scrollHeight, behavior: "smooth" });
  }
};

elements.browse.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => uploadFiles(elements.fileInput.files));
elements.refresh.addEventListener("click", loadDocuments);
["dragenter", "dragover"].forEach((eventName) => elements.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); elements.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => elements.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault(); elements.dropZone.classList.remove("dragging");
}));
elements.dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.input.value.trim();
  if (!question) return;
  elements.input.value = "";
  elements.input.style.height = "auto";
  askQuestion(question);
});
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); }
});
elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
});
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => askQuestion(button.dataset.prompt)));

checkHealth();
loadDocuments();
loadConversations();
