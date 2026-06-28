"use strict";
// Personal AI (TurboVec) — single-page UI logic.
const API = "/api/v1";
let folders = [];
let activeFolder = null;
let conversationId = null;

const $ = (id) => document.getElementById(id);
const chat = $("chat");

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || res.statusText);
  return body.data;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.style.display = "block";
  setTimeout(() => (t.style.display = "none"), 2600);
}

// ---- Folders ----
async function loadFolders() {
  folders = await api("/folders");
  const box = $("folders");
  box.innerHTML = "";
  folders.forEach((f) => {
    const el = document.createElement("div");
    el.className = "folder" + (activeFolder === f.id ? " active" : "");
    el.textContent = "📁 " + f.name;
    el.onclick = () => selectFolder(f.id);
    box.appendChild(el);
  });
}

async function selectFolder(id) {
  activeFolder = id;
  await loadFolders();
  const f = folders.find((x) => x.id === id);
  $("filesTitle").textContent = "Files · " + (f ? f.name : "");
  await loadFiles();
}

$("addFolder").onclick = async () => {
  const name = $("folderName").value.trim();
  if (!name) return;
  const f = await api("/folders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  $("folderName").value = "";
  await selectFolder(f.id);
};

// ---- Files ----
async function loadFiles() {
  if (!activeFolder) return;
  const files = await api(`/folders/${activeFolder}/files`);
  const box = $("files");
  box.innerHTML = "";
  files.forEach((file) => {
    const el = document.createElement("div");
    el.className = "file";
    el.innerHTML = `<span class="name">📄 ${file.name}</span>
      <span class="badge ${file.status}">${file.status}</span>`;
    el.querySelector(".name").onclick = () => preview(file.id);
    box.appendChild(el);
  });
}

async function preview(fileId) {
  const data = await api(`/files/${fileId}/preview`);
  $("modalTitle").textContent = data.name;
  $("modalBody").textContent = data.text || "(no extracted text)";
  $("modal").classList.add("show");
}

async function uploadFiles(fileList) {
  if (!activeFolder) return toast("Select or create a folder first");
  for (const file of fileList) {
    toast(`Indexing ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    try {
      await api(`/folders/${activeFolder}/files`, { method: "POST", body: form });
    } catch (e) {
      toast(`Failed: ${e.message}`);
    }
    await loadFiles();
  }
  toast("Done indexing");
}

const drop = $("drop");
drop.onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => uploadFiles(e.target.files);
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); })
);
drop.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

// ---- Conversations ----
async function loadConversations() {
  const list = await api("/chat/conversations");
  const box = $("conversations");
  box.innerHTML = "";
  list.forEach((c) => {
    const el = document.createElement("div");
    el.className = "convo" + (conversationId === c.id ? " active" : "");
    el.textContent = "💬 " + (c.title || "Untitled");
    el.onclick = () => loadConversation(c.id);
    box.appendChild(el);
  });
}

async function loadConversation(id) {
  conversationId = id;
  const messages = await api(`/chat/conversations/${id}/messages`);
  chat.innerHTML = "";
  messages.forEach((m) => {
    if (m.role === "user") {
      addMessage("user").textContent = m.content;
    } else {
      const asst = addMessage("assistant");
      asst.querySelector(".body").textContent = m.content;
    }
  });
  chat.scrollTop = chat.scrollHeight;
  await loadConversations();
}

// ---- Usage / cost meter ----
async function refreshMeter() {
  try {
    const u = await api("/usage/totals");
    $("meter").textContent = `$${u.cost_usd.toFixed(6)} · ${u.total_tokens.toLocaleString()} tok`;
  } catch (_) { /* ignore */ }
}

// ---- Chat ----
function clearHint() {
  const hint = chat.querySelector(".hint");
  if (hint) hint.remove();
}

function addMessage(role) {
  clearHint();
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  if (role === "assistant") {
    msg.innerHTML = `<div class="body"></div><div class="sources"></div>`;
  }
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function renderSources(container, sources) {
  container.innerHTML = "";
  sources.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "src";
    chip.innerHTML = `<span class="lbl">${s.label}</span> ${s.file_name}`;
    chip.title = s.heading_path || s.snippet || "";
    chip.onclick = () => preview(s.file_id);
    container.appendChild(chip);
  });
}

async function send() {
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  const userMsg = addMessage("user");
  userMsg.textContent = text;

  const asst = addMessage("assistant");
  const body = asst.querySelector(".body");
  const srcBox = asst.querySelector(".sources");
  body.textContent = "…";

  const payload = { message: text, conversation_id: conversationId };
  if ($("scopeFolder").checked && activeFolder) payload.folder_id = activeFolder;

  const res = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      const evt = JSON.parse(part.slice(6));
      if (evt.type === "start") conversationId = evt.conversation_id;
      else if (evt.type === "sources") renderSources(srcBox, evt.sources);
      else if (evt.type === "token") {
        if (answer === "") body.textContent = "";
        answer += evt.text;
        body.textContent = answer;
        chat.scrollTop = chat.scrollHeight;
      }
    }
  }
  await loadConversations();
  await refreshMeter();
}

$("send").onclick = send;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("btnNew").onclick = () => {
  conversationId = null;
  chat.innerHTML = `<div class="hint"><h2>New chat</h2>Ask anything about your files.</div>`;
};

// ---- Second-brain actions ----
$("btnSummarise").onclick = async () => {
  const topic = prompt("Summarise everything about…");
  if (!topic) return;
  const userMsg = addMessage("user");
  userMsg.textContent = `Summarise: ${topic}`;
  const asst = addMessage("assistant");
  asst.querySelector(".body").textContent = "Summarising…";
  const payload = { query: topic };
  if ($("scopeFolder").checked && activeFolder) payload.folder_id = activeFolder;
  const data = await api("/search/summarise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  asst.querySelector(".body").textContent = data.summary;
  renderSources(asst.querySelector(".sources"),
    (data.sources || []).map((s, i) => ({ ...s, label: "S" + (i + 1) })));
};

$("btnMentions").onclick = async () => {
  const topic = prompt("Which files mention…");
  if (!topic) return;
  const rows = await api(`/search/mentions?q=${encodeURIComponent(topic)}`);
  const asst = addMessage("assistant");
  const body = asst.querySelector(".body");
  if (!rows.length) { body.textContent = "No files mention that."; return; }
  body.textContent = `Files mentioning “${topic}”:\n` +
    rows.map((r) => `• ${r.file_name}  (${r.hits} hit${r.hits > 1 ? "s" : ""})`).join("\n");
};

// ---- Modal ----
$("modalClose").onclick = () => $("modal").classList.remove("show");
$("modal").onclick = (e) => { if (e.target.id === "modal") $("modal").classList.remove("show"); };

// Refresh the cost meter after second-brain actions too.
["btnSummarise", "btnMentions"].forEach((id) => {
  const orig = $(id).onclick;
  $(id).onclick = async () => { await orig(); refreshMeter(); };
});

loadFolders();
loadConversations();
refreshMeter();
