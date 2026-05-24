// chat.js

if (!requireAuth()) { /* redirected */ }

document.getElementById('app').insertAdjacentHTML('afterbegin', buildSidebar('chat'));

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId       = null;
let sessionStarting = false;   // BUG FIX: guard against double session creation
let mediaRecorder   = null;
let audioChunks     = [];
let isRecording     = false;

const topicId       = getParam('topic_id');
const topicName     = getParam('topic_name');
const resumeSession = getParam('session_id');

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    document.getElementById('mic-btn').style.display = 'none';
  }

  if (topicName) {
    document.getElementById('chat-title').textContent = topicName;
  }

  if (resumeSession) {
    sessionId = resumeSession;
    document.getElementById('chat-subtitle').textContent = 'Session resumed';
    loadHistory(resumeSession);
  } else {
    startSession();
  }
})();

// ── Session management ────────────────────────────────────────────────────────
async function startSession() {
  if (sessionStarting) return;
  sessionStarting = true;

  try {
    const res = await apiFetch('/tutor/start', {
      method: 'POST',
      body: JSON.stringify({ topic_id: topicId || null })
    });
    if (!res || !res.ok) { showToast('Failed to start session', 'error'); return; }
    const data = await res.json();
    sessionId = data.session_id;
    document.getElementById('chat-subtitle').textContent = 'Session active';
  } catch {
    showToast('Failed to start session', 'error');
  } finally {
    sessionStarting = false;
  }
}

async function loadHistory(sid) {
  try {
    const res = await apiFetch(`/tutor/history?session_id=${sid}`);
    if (!res || !res.ok) return;
    const messages = await res.json();
    if (messages.length) {
      document.getElementById('chat-empty').style.display = 'none';
      messages.forEach(m =>
        appendMessage(m.role === 'user' ? 'user' : 'assistant', m.content, m.agent_type, null, null, sid)
      );
    }
  } catch { /* silent */ }
}

async function endSession() {
  if (!sessionId) { window.location.href = 'dashboard.html'; return; }
  try {
    await apiFetch(`/tutor/end?session_id=${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId })
    });
  } finally {
    window.location.href = 'dashboard.html';
  }
}

// ── Messaging ─────────────────────────────────────────────────────────────────
// Guard: true while a response is in-flight — blocks double-submit
let _messageInFlight = false;

async function sendMessage() {
  const input   = document.getElementById('msg-input');
  const message = input.value.trim();

  if (!message) return;
  if (!sessionId) {
    showToast('Session is starting, please wait…', 'warning');
    return;
  }
  // Frontend in-flight guard — prevents double-submit on slow responses
  if (_messageInFlight) {
    showToast('Please wait — a response is already on its way.', 'warning');
    return;
  }

  _messageInFlight = true;
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  appendMessage('user', message);
  showTyping();

  try {
    const res = await apiFetch('/tutor/text', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, message })
    });
    removeTyping();

    if (!res) return;

    if (res.status === 404) {
      showToast('Session has ended.', 'warning');
      setTimeout(() => window.location.href = 'dashboard.html', 1500);
      return;
    }

    // 409 = server-side in-flight lock: another request is already processing
    if (res.status === 409) {
      showToast('Already generating a response — please wait.', 'warning');
      return;
    }

    if (!res.ok) {
      showToast('Failed to send message. Please try again.', 'error');
      return;
    }

    const data = await res.json();
    appendMessage('assistant', data.reply, data.agent_type, data.message_id, data.trace_id, String(sessionId));

    if (data.quiz_redirect) {
      const tId   = data.topic_id || topicId || null;
      const tName = data.topic_name || topicName || '';

      if (tId) {
        // Known curriculum topic UUID \u2014 redirect directly to quiz page
        setTimeout(() => {
          window.location.href =
            `quiz.html?topic_id=${encodeURIComponent(tId)}` +
            `&session_id=${encodeURIComponent(sessionId)}` +
            `&topic_name=${encodeURIComponent(tName)}`;
        }, 1500);
      } else if (tName) {
        // Ad-hoc topic extracted from message \u2014 go to quiz picker pre-filled
        setTimeout(() => {
          window.location.href =
            `quiz.html?session_id=${encodeURIComponent(sessionId)}` +
            `&adhoc_topic=${encodeURIComponent(tName)}`;
        }, 1500);
      } else {
        showQuizTopicPicker();
      }
    }
  } catch {
    removeTyping();
    showToast('Network error. Please try again.', 'error');
  } finally {
    _messageInFlight = false;
    document.getElementById('send-btn').disabled = false;
  }
}

// ── Message rendering ─────────────────────────────────────────────────────────
function appendMessage(role, content, agentType, messageId, traceId, sessionIdForFeedback) {
  const area  = document.getElementById('messages-area');
  const empty = document.getElementById('chat-empty');
  if (empty) empty.style.display = 'none';

  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const badge = (agentType && role === 'assistant')
    ? `<div class="msg-meta"><span class="badge badge-${agentType}">${agentType}</span></div>`
    : '';

  if (role === 'assistant') {
    // Build feedback buttons — only shown on assistant messages
    const fbId = messageId || '';
    const fbTrace = traceId || '';
    const fbSession = sessionIdForFeedback || sessionId || '';
    row.innerHTML = `
      <div class="msg-avatar">🎓</div>
      <div class="msg-bubble-wrap">
        ${badge}
        <div class="msg-bubble">${escHtml(content)}</div>
        <div class="msg-feedback" data-message-id="${escHtml(fbId)}" data-trace-id="${escHtml(fbTrace)}" data-session-id="${escHtml(fbSession)}">
          <button class="fb-btn fb-up"   title="Good response"  onclick="submitFeedback(this, 5)">👍</button>
          <button class="fb-btn fb-down" title="Bad response"   onclick="submitFeedback(this, 1)">👎</button>
        </div>
      </div>`;
  } else {
    row.innerHTML = `
      <div class="msg-bubble-wrap">
        <div class="msg-bubble">${escHtml(content)}</div>
      </div>`;
  }

  area.appendChild(row);
  area.scrollTop = area.scrollHeight;
}

// ── Feedback submission ───────────────────────────────────────────────────────
async function submitFeedback(btn, rating) {
  const wrap      = btn.closest('.msg-feedback');
  const messageId = wrap.dataset.messageId;
  const traceId   = wrap.dataset.traceId;
  const sid       = wrap.dataset.sessionId;

  if (!sid) return;   // no session yet — silently ignore

  // Lock both buttons immediately so the student can't double-submit
  wrap.querySelectorAll('.fb-btn').forEach(b => b.disabled = true);

  // Visual state before the network call
  if (rating === 5) btn.classList.add('fb-up-active');
  else              btn.classList.add('fb-down-active');

  try {
    const body = {
      session_id: sid,
      rating,
      langsmith_trace_id: traceId || '',
      comment: '',
    };
    if (messageId) body.message_id = messageId;

    const res = await apiFetch('/tutor/feedback', {
      method: 'POST',
      body: JSON.stringify(body),
    });

    if (!res || !res.ok) {
      // Re-enable on failure so student can retry
      wrap.querySelectorAll('.fb-btn').forEach(b => b.disabled = false);
      btn.classList.remove('fb-up-active', 'fb-down-active');
      showToast('Could not save feedback. Please try again.', 'error');
      return;
    }

    if (rating === 1) {
      showToast('Thanks for the feedback — we\'ll use this to improve.', 'info');
    }
  } catch {
    wrap.querySelectorAll('.fb-btn').forEach(b => b.disabled = false);
    btn.classList.remove('fb-up-active', 'fb-down-active');
  }
}

function showTyping() {
  const area = document.getElementById('messages-area');
  const row  = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = 'typing-row';
  row.innerHTML = `
    <div class="msg-avatar">🎓</div>
    <div class="msg-bubble-wrap">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
      <span class="typing-label">Processing…</span>
    </div>`;
  area.appendChild(row);
  area.scrollTop = area.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing-row');
  if (t) t.remove();
}

// ── Voice input ───────────────────────────────────────────────────────────────
function toggleVoice() {
  if (isRecording) stopRecording();
  else startRecording();
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks   = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop          = sendVoice;
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('mic-btn').classList.add('recording');
    document.getElementById('mic-btn').title = 'Stop recording';
    document.getElementById('voice-status').classList.add('visible');
  } catch {
    showToast('Microphone access denied.', 'error');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  isRecording = false;
  document.getElementById('mic-btn').classList.remove('recording');
  document.getElementById('mic-btn').title = 'Voice input';
  document.getElementById('voice-status').classList.remove('visible');
}

async function sendVoice() {
  if (!audioChunks.length || !sessionId) return;

  const blob = new Blob(audioChunks, { type: 'audio/webm' });

  // FIX: Show the user's own recorded audio in the chat immediately
  appendVoiceMessage('user', blob);

  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('audio', blob, 'recording.webm');
  form.append('language', 'en');

  showTyping();
  try {
    const res = await apiFetch('/tutor/voice', { method: 'POST', body: form });
    if (!res || !res.ok) { removeTyping(); showToast('Voice upload failed', 'error'); return; }

    const { job_id } = await res.json();

    // Poll up to 450 attempts x 3s = 22.5 min — covers slow local Ollama on CPU.
    // Every 3s we update the typing bubble with elapsed time so the user
    // knows it is still running and not frozen.
    const voiceStart = Date.now();
    const _typingTimer = setInterval(() => {
      const sec  = Math.round((Date.now() - voiceStart) / 1000);
      const mins = Math.floor(sec / 60);
      const secs = sec % 60;
      const label = mins > 0 ? `Processing… ${mins}m ${secs}s` : `Processing… ${secs}s`;
      const lbl = document.querySelector('#typing-row .typing-label');
      if (lbl) lbl.textContent = label;
    }, 3000);

    await poll(async () => {
      const r = await apiFetch(`/tutor/job/${job_id}/status`);
      if (!r || !r.ok) return false;
      const d = await r.json();
      if (d.status === 'done') {
        clearInterval(_typingTimer);
        removeTyping();
        const text     = d.result?.text      || 'Done.';
        const audioB64 = d.result?.audio_b64 || '';
        const agent    = d.result?.agent_type || '';
        appendVoiceMessage('assistant', null, text, audioB64, agent);
        return true;
      }
      if (d.status === 'failed') {
        clearInterval(_typingTimer);
        removeTyping();
        showToast('Voice processing failed: ' + (d.error || 'unknown error'), 'error');
        return true;
      }
      return false;
    }, 3000, 450).catch(() => {
      clearInterval(_typingTimer);
      removeTyping();
      showToast('Voice processing timed out after 22 minutes. Please try again.', 'error');
    });
  } catch {
    removeTyping();
    showToast('Voice processing failed', 'error');
  }
}

// ── Voice message rendering ───────────────────────────────────────────────────
// Renders user audio blobs and assistant text+audio replies as chat bubbles.
function appendVoiceMessage(role, blob, text, audioB64, agentType) {
  const area  = document.getElementById('messages-area');
  const empty = document.getElementById('chat-empty');
  if (empty) empty.style.display = 'none';

  const row = document.createElement('div');
  row.className = 'message-row ' + role;

  if (role === 'user') {
    const objectUrl = URL.createObjectURL(blob);
    row.innerHTML =
      '<div class="msg-bubble-wrap">' +
        '<div class="msg-bubble voice-bubble">' +
          '<span class="voice-label">\u{1F3A4} Voice message</span>' +
          '<audio class="voice-audio" controls src="' + objectUrl + '"></audio>' +
        '</div>' +
      '</div>';
    row.querySelector('audio').addEventListener('emptied',
      () => URL.revokeObjectURL(objectUrl), { once: true });

  } else {
    const badge = agentType
      ? '<div class="msg-meta"><span class="badge badge-' + escHtml(agentType) + '">' + escHtml(agentType) + '</span></div>'
      : '';

    let audioEl = '';
    if (audioB64) {
      audioEl = '<audio class="voice-audio" controls src="data:audio/wav;base64,' + audioB64 + '"></audio>';
    }

    row.innerHTML =
      '<div class="msg-avatar">\u{1F393}</div>' +
      '<div class="msg-bubble-wrap">' +
        badge +
        '<div class="msg-bubble voice-bubble">' +
          (text ? '<p class="voice-text">' + escHtml(text) + '</p>' : '') +
          audioEl +
        '</div>' +
      '</div>';

    if (audioB64) {
      const audio = row.querySelector('audio');
      audio.play().catch(() => { /* autoplay blocked — controls let user play manually */ });
    }
  }

  area.appendChild(row);
  area.scrollTop = area.scrollHeight;
}

// ── Document upload ───────────────────────────────────────────────────────────
async function uploadDoc(input) {
  const file = input.files[0];
  if (!file) return;

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await apiFetch('/tutor/upload-doc', { method: 'POST', body: form });
    if (!res || !res.ok) { showToast('Upload failed', 'error'); return; }
    const data = await res.json();
    showDocPill(data.doc_id, file.name);
  } catch {
    showToast('Upload failed', 'error');
  }
  input.value = '';
}

// ── Doc processing pill ───────────────────────────────────────────────────────
function showDocPill(docId, filename) {
  const existing = document.getElementById(`doc-pill-${docId}`);
  if (existing) existing.remove();

  const pill = document.createElement('div');
  pill.id = `doc-pill-${docId}`;
  pill.className = 'doc-pill';
  pill.innerHTML = `
    <span class="doc-pill-icon">📄</span>
    <span class="doc-pill-name">${escHtml(filename)}</span>
    <span class="doc-pill-status">⏳ Indexing…</span>`;

  // Insert between topbar and messages area
  const chatMain    = document.querySelector('.chat-main');
  const messagesArea = document.getElementById('messages-area');
  chatMain.insertBefore(pill, messagesArea);

  const intervalId = setInterval(async () => {
    try {
      const r = await apiFetch(`/tutor/doc-status/${docId}`);
      if (!r || !r.ok) return;
      const d = await r.json();
      const statusEl = pill.querySelector('.doc-pill-status');

      if (d.status === 'ready') {
        clearInterval(intervalId);
        statusEl.textContent = '✅ Ready — you can now ask about this document';
        pill.classList.add('doc-pill-done');
        setTimeout(() => {
          pill.classList.add('doc-pill-fade');
          setTimeout(() => pill.remove(), 500);
        }, 2500);
      } else if (d.status === 'failed') {
        clearInterval(intervalId);
        statusEl.textContent = '❌ Indexing failed';
        pill.classList.add('doc-pill-error');
        setTimeout(() => pill.remove(), 3500);
      } else {
        // Show live progress message from backend if provided (e.g. "Embedding chunk 3/12…")
        if (d.message) {
          statusEl.textContent = `⏳ ${d.message}`;
        }
      }
    } catch { /* silent */ }
  }, 2500);
}

// ── Quiz topic picker modal ───────────────────────────────────────────────────
async function showQuizTopicPicker() {
  const modal = document.getElementById('topic-modal');
  const list  = document.getElementById('topic-pick-list');
  list.innerHTML = '<div class="spinner" style="margin:12px auto"></div>';
  modal.style.display = 'flex';

  try {
    const res    = await apiFetch('/curriculum/topics');
    const topics = await res.json();
    list.innerHTML = topics.map(t => `
      <div class="topic-pick-item"
           onclick="goToQuiz('${escHtml(t.id)}','${escHtml(t.name)}')">
        ${escHtml(t.name)}
      </div>`).join('');
  } catch {
    list.innerHTML = '<p style="color:var(--color-error);font-size:13px">Failed to load topics.</p>';
  }
}

function goToQuiz(tid, tname) {
  document.getElementById('topic-modal').style.display = 'none';
  window.location.href =
    `quiz.html?topic_id=${encodeURIComponent(tid)}` +
    `&session_id=${encodeURIComponent(sessionId)}` +
    `&topic_name=${encodeURIComponent(tname)}`;
}

// ── Input helpers ─────────────────────────────────────────────────────────────
function handleInputKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}