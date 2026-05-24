// login.js — Authentication logic

// Redirect if already authenticated
if (localStorage.getItem('access_token')) {
  window.location.href = 'dashboard.html';
}

// ── GitHub OAuth callback handler ─────────────────────────────────────────────
(function handleGithubCallback() {
  const params = new URLSearchParams(window.location.search);

  const errorParam = params.get('error');
  if (errorParam) {
    const desc = params.get('error_description') || errorParam;
    showError('login', `GitHub login failed: ${desc}`);
    window.history.replaceState({}, '', window.location.pathname);
    return;
  }

  const code = params.get('code');
  const state = params.get('state');
  if (!code) return;

  document.getElementById('loading-overlay').classList.add('visible');

  fetch(`http://localhost:8000/auth/github/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state || '')}`)
    .then(r => r.json())
    .then(data => {
      if (data.access_token) {
        storeTokens(data);
        window.location.href = 'dashboard.html';
      } else {
        document.getElementById('loading-overlay').classList.remove('visible');
        showError('login', 'GitHub authentication failed. Please try again.');
      }
    })
    .catch(() => {
      document.getElementById('loading-overlay').classList.remove('visible');
      showError('login', 'GitHub authentication failed. Please try again.');
    });
})();

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.querySelectorAll('.tab-btn')[tab === 'login' ? 0 : 1].classList.add('active');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showError(form, msg) {
  const el = document.getElementById(`${form}-error`);
  el.textContent = msg;
  el.classList.add('visible');
}

function clearError(form) {
  document.getElementById(`${form}-error`).classList.remove('visible');
}

function setBtnLoading(btnId, loading, defaultLabel) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  btn.textContent = loading ? 'Please wait…' : defaultLabel;
}

// ── Login ─────────────────────────────────────────────────────────────────────
async function doLogin() {
  const email    = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;
  clearError('login');

  if (!email || !password) {
    showError('login', 'Please fill in all fields.');
    return;
  }

  setBtnLoading('login-btn', true, 'Sign In');
  try {
    const res  = await fetch('http://localhost:8000/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();

    if (res.ok) {
      storeTokens(data);
      window.location.href = 'dashboard.html';
    } else if (res.status === 401) {
      showError('login', 'Invalid email or password.');
    } else {
      showError('login', data.detail || 'Login failed. Please try again.');
    }
  } catch {
    showError('login', 'Could not connect to server.');
  } finally {
    setBtnLoading('login-btn', false, 'Sign In');
  }
}

// ── Signup ────────────────────────────────────────────────────────────────────
async function doSignup() {
  const full_name = document.getElementById('signup-name').value.trim();
  const email     = document.getElementById('signup-email').value.trim();
  const password  = document.getElementById('signup-password').value;
  clearError('signup');

  if (!full_name || !email || !password) {
    showError('signup', 'Please fill in all fields.');
    return;
  }
  if (password.length < 8) {
    showError('signup', 'Password must be at least 8 characters.');
    return;
  }

  setBtnLoading('signup-btn', true, 'Create Account');
  try {
    const res  = await fetch('http://localhost:8000/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name })
    });
    const data = await res.json();

    if (res.status === 201 || res.ok) {
      storeTokens(data);
      window.location.href = 'dashboard.html';
    } else if (res.status === 409) {
      showError('signup', 'Email already registered. Try signing in.');
    } else {
      showError('signup', data.detail || 'Signup failed. Please try again.');
    }
  } catch {
    showError('signup', 'Could not connect to server.');
  } finally {
    setBtnLoading('signup-btn', false, 'Create Account');
  }
}

// ── GitHub OAuth initiation ───────────────────────────────────────────────────
async function doGithubLogin() {
  try {
    const res  = await fetch('http://localhost:8000/auth/github/login');
    const data = await res.json();
    if (data.url) {
      window.location.href = data.url;
    } else {
      showToast('GitHub login unavailable.', 'error');
    }
  } catch {
    showToast('Could not connect to server.', 'error');
  }
}

// ── Enter key support ─────────────────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const loginActive = document.getElementById('tab-login').classList.contains('active');
  if (loginActive) doLogin();
  else doSignup();
});