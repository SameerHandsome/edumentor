// api.js — Shared fetch utility with JWT auth + refresh logic

const BASE = 'http://localhost:8000';

async function apiFetch(path, options = {}) {
  const token = localStorage.getItem('access_token');
  const isFormData = options.body instanceof FormData;
  const headers = { ...options.headers };
  if (!isFormData) headers['Content-Type'] = 'application/json';
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let res = await fetch(BASE + path, { ...options, headers });

  if (res.status === 401) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      headers['Authorization'] = `Bearer ${localStorage.getItem('access_token')}`;
      res = await fetch(BASE + path, { ...options, headers });
    } else {
      window.location.href = '/login.html';
      return null;
    }
  }

  if (res.status === 429) {
    let msg = 'Too many requests. Please try again later.';
    try {
      const data = await res.clone().json();
      if (data.retry_after) msg = `Too many requests. Try again in ${data.retry_after} seconds.`;
    } catch {}
    showToast(msg, 'warning');
    return res;
  }

  return res;
}

async function tryRefresh() {
  const rt = localStorage.getItem('refresh_token');
  if (!rt) return false;
  try {
    const res = await fetch(BASE + '/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: rt })
    });
    if (!res.ok) return false;
    const data = await res.json();
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

function showToast(msg, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const icons = { info: 'ℹ️', error: '❌', success: '✅', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || ''}</span><span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}

function poll(fn, intervalMs, maxAttempts = 20) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    let inflight = false; // prevent overlapping async ticks
    let settled  = false; // prevent resolving/rejecting twice
    const id = setInterval(async () => {
      if (inflight || settled) return;
      inflight = true;
      attempts++;
      try {
        const result = await fn();
        if (result) {
          settled = true;
          clearInterval(id);
          resolve(result);
        } else if (attempts >= maxAttempts) {
          settled = true;
          clearInterval(id);
          reject(new Error('Max poll attempts reached'));
        }
      } catch (err) {
        settled = true;
        clearInterval(id);
        reject(err);
      } finally {
        inflight = false;
      }
    }, intervalMs);
  });
}

function storeTokens(data) {
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);
}

function requireAuth() {
  if (!localStorage.getItem('access_token')) {
    window.location.href = '/login.html';
    return false;
  }
  return true;
}

// BUG FIX: old logout() just cleared localStorage — the refresh token stayed
// valid on the server for 7 days. Now we call POST /auth/logout first to
// blacklist the refresh token in Redis before clearing local storage.
async function logout() {
  const rt = localStorage.getItem('refresh_token');
  if (rt) {
    await fetch(BASE + '/auth/logout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: rt })
    }).catch(() => {}); // best-effort — don't block redirect on network failure
  }
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  window.location.href = '/login.html';
}

function getParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function buildSidebar(active) {
  return `
  <aside class="sidebar">
    <div class="sidebar-logo">
      <a href="dashboard.html" class="logo-mark">
        <div class="logo-icon">🎓</div>
        <span class="logo-text">EduMentor</span>
      </a>
    </div>
    <nav class="sidebar-nav">
      <a href="dashboard.html" class="nav-link ${active === 'dashboard' ? 'active' : ''}">
        <span class="nav-icon">🏠</span> Dashboard
      </a>
      <a href="chat.html" class="nav-link ${active === 'chat' ? 'active' : ''}">
        <span class="nav-icon">💬</span> Chat
      </a>
      <a href="quiz.html" class="nav-link ${active === 'quiz' ? 'active' : ''}">
        <span class="nav-icon">📝</span> Quiz
      </a>
      <a href="profile.html" class="nav-link ${active === 'profile' ? 'active' : ''}">
        <span class="nav-icon">👤</span> Profile
      </a>
    </nav>
    <div class="sidebar-footer">
      <button onclick="logout()" class="nav-link logout">
        <span class="nav-icon">🚪</span> Logout
      </button>
    </div>
  </aside>`;
}