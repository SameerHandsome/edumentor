// dashboard.js

if (!requireAuth()) { /* redirected */ }

document.getElementById('app').insertAdjacentHTML('afterbegin', buildSidebar('dashboard'));

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  await Promise.all([loadProfile(), loadTopics(), loadSessions()]);
}

// ── Profile / Welcome banner ──────────────────────────────────────────────────
async function loadProfile() {
  try {
    const res = await apiFetch('/user/profile');
    if (!res || !res.ok) return;
    const data = await res.json();

    document.getElementById('welcome-name').textContent = `Welcome back, ${data.full_name} 👋`;

    const avatarEl = document.getElementById('welcome-avatar');
    if (data.avatar_url) {
      avatarEl.innerHTML = `<img src="${escHtml(data.avatar_url)}" alt="Avatar" onerror="this.style.display='none'" />`;
    } else {
      avatarEl.textContent = (data.full_name || '?')[0].toUpperCase();
    }
  } catch { /* silent */ }
}

// ── Topics grid ───────────────────────────────────────────────────────────────
async function loadTopics() {
  const container = document.getElementById('topics-container');
  try {
    const res = await apiFetch('/curriculum/topics');
    if (!res || !res.ok) {
      container.innerHTML = emptyState('📭', 'No topics found', 'Check back later for available topics.');
      return;
    }
    const topics = await res.json();
    document.getElementById('stat-topics').textContent = topics.length;

    if (!topics.length) {
      container.innerHTML = emptyState('📭', 'No topics available', 'Topics will appear here once added.');
      return;
    }

    container.className = 'topics-grid';
    container.innerHTML = topics.map(t => `
      <div class="topic-card">
        <div class="topic-card-header">
          <span class="topic-name">${escHtml(t.name)}</span>
          ${t.grade_level ? `<span class="topic-grade">Grade ${escHtml(String(t.grade_level))}</span>` : ''}
        </div>
        <p class="topic-desc">${escHtml(t.description || 'No description available.')}</p>
        <div class="topic-actions">
          <a href="chat.html?topic_id=${encodeURIComponent(t.id)}&topic_name=${encodeURIComponent(t.name)}"
             class="btn btn-secondary btn-sm">💬 Study</a>
          <a href="quiz.html?topic_id=${encodeURIComponent(t.id)}&topic_name=${encodeURIComponent(t.name)}"
             class="btn btn-ghost btn-sm">📝 Quiz Me</a>
        </div>
      </div>
    `).join('');
  } catch {
    container.innerHTML = emptyState('⚠️', 'Failed to load topics', '');
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function emptyState(icon, title, desc) {
  return `<div class="empty-state">
    <div class="empty-icon">${icon}</div>
    <h3>${title}</h3>
    ${desc ? `<p>${desc}</p>` : ''}
  </div>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  } catch { return iso; }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadDashboard();

// ── Recent sessions ───────────────────────────────────────────────────────────
async function loadSessions() {
  const container = document.getElementById('sessions-container');
  try {
    const res = await apiFetch('/tutor/sessions');
    if (!res || !res.ok) {
      container.innerHTML = emptyState('📭', 'No sessions yet', 'Start a new chat to begin learning.');
      return;
    }
    const sessions = await res.json();
    document.getElementById('stat-sessions').textContent = sessions.length;
    document.getElementById('stat-active').textContent = sessions.filter(s => s.is_active).length;

    const recent = sessions.slice(0, 5);
    if (!recent.length) {
      container.innerHTML = emptyState('💬', 'No sessions yet', 'Start your first chat session!');
      return;
    }

    container.innerHTML = `<div class="sessions-list">${recent.map(s => {
      const label = s.display_name || (s.topic_id ? 'Topic Session' : 'General Session');
      return `
      <div class="session-item-wrap" data-session-id="${escHtml(s.id)}">
        <a href="chat.html?session_id=${encodeURIComponent(s.id)}" class="session-item">
          <div class="session-info">
            <div class="session-topic">${escHtml(label)}</div>
            <div class="session-time">${formatDate(s.started_at)}</div>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <span class="badge ${s.is_active ? 'badge-active' : 'badge-ended'}">
              ${s.is_active ? '● Live' : 'Ended'}
            </span>
            <span class="session-arrow">›</span>
          </div>
        </a>
        <button class="session-menu-btn" title="Options" onclick="toggleMenu(event,'${escHtml(s.id)}')">⋯</button>
        <div class="session-menu" id="menu-${escHtml(s.id)}">
          <button onclick="renameSession('${escHtml(s.id)}')">✏️ Rename</button>
          <button onclick="deleteSession('${escHtml(s.id)}')" class="danger">🗑️ Delete</button>
        </div>
      </div>`;
    }).join('')}</div>`;

    document.addEventListener('click', closeAllMenus);
  } catch {
    container.innerHTML = emptyState('⚠️', 'Failed to load sessions', '');
  }
}

function toggleMenu(e, sessionId) {
  e.preventDefault();
  e.stopPropagation();
  const menu = document.getElementById(`menu-${sessionId}`);
  const isOpen = menu.classList.contains('open');
  closeAllMenus();
  if (!isOpen) menu.classList.add('open');
}

function closeAllMenus() {
  document.querySelectorAll('.session-menu.open').forEach(m => m.classList.remove('open'));
}

async function renameSession(sessionId) {
  closeAllMenus();
  const wrap = document.querySelector(`[data-session-id="${sessionId}"]`);
  const currentName = wrap?.querySelector('.session-topic')?.textContent || '';
  const newName = prompt('Rename session:', currentName);
  if (!newName || !newName.trim()) return;
  try {
    const res = await apiFetch(`/tutor/sessions/${sessionId}/rename`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName.trim() }),
    });
    if (res && res.ok) {
      if (wrap) wrap.querySelector('.session-topic').textContent = newName.trim();
    } else {
      alert('Rename failed. Please try again.');
    }
  } catch {
    alert('Network error.');
  }
}

async function deleteSession(sessionId) {
  closeAllMenus();
  if (!confirm('Delete this session and all its messages? This cannot be undone.')) return;
  try {
    const res = await apiFetch(`/tutor/sessions/${sessionId}`, { method: 'DELETE' });
    if (res && res.ok) {
      const wrap = document.querySelector(`[data-session-id="${sessionId}"]`);
      if (wrap) wrap.remove();
    } else {
      alert('Delete failed. Please try again.');
    }
  } catch {
    alert('Network error.');
  }
}