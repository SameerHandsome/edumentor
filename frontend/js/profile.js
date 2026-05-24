// profile.js

if (!requireAuth()) { /* redirected */ }

document.getElementById('app').insertAdjacentHTML('afterbegin', buildSidebar('profile'));

// ── Load profile form ─────────────────────────────────────────────────────────
async function loadProfile() {
  try {
    const res = await apiFetch('/user/profile');
    if (!res || !res.ok) return;
    const data = await res.json();

    document.getElementById('field-name').value  = data.full_name || '';
    document.getElementById('field-email').value = data.email || '';
    document.getElementById('field-style').value = data.explanation_style || 'step_by_step';
    document.getElementById('field-goal').value  = data.session_goal || '';
    document.getElementById('field-weak').value  =
      Array.isArray(data.weak_topics)
        ? data.weak_topics.join(', ')
        : (data.weak_topics || '');

    document.getElementById('display-name').textContent  = data.full_name || 'User';
    document.getElementById('display-email').textContent = data.email || '';

    const avatarEl = document.getElementById('profile-avatar');
    if (data.avatar_url) {
      avatarEl.innerHTML =
        `<img src="${escHtml(data.avatar_url)}" alt="Avatar" onerror="this.style.display='none'" />`;
    } else {
      avatarEl.textContent = (data.full_name || '?')[0].toUpperCase();
    }
  } catch {
    showToast('Failed to load profile', 'error');
  }
}

// ── Save profile ──────────────────────────────────────────────────────────────
async function saveProfile() {
  const full_name        = document.getElementById('field-name').value.trim();
  const explanation_style = document.getElementById('field-style').value;
  const session_goal     = document.getElementById('field-goal').value.trim();
  const weakRaw          = document.getElementById('field-weak').value;
  const weak_topics      = weakRaw.split(',').map(s => s.trim()).filter(Boolean);

  const saveBtn     = document.getElementById('save-btn');
  saveBtn.disabled  = true;
  saveBtn.textContent = 'Saving…';

  try {
    const res = await apiFetch('/user/profile', {
      method: 'PATCH',
      body: JSON.stringify({ full_name, explanation_style, session_goal, weak_topics })
    });

    if (res && res.ok) {
      document.getElementById('display-name').textContent = full_name;

      const fb = document.getElementById('save-feedback');
      fb.classList.add('visible');
      setTimeout(() => fb.classList.remove('visible'), 3000);

      showToast('Profile updated', 'success');
    } else {
      showToast('Failed to save profile', 'error');
    }
  } catch {
    showToast('Network error', 'error');
  } finally {
    saveBtn.disabled    = false;
    saveBtn.textContent = 'Save Changes';
  }
}

// ── Load mastery table ────────────────────────────────────────────────────────
async function loadMastery() {
  const loadingEl = document.getElementById('mastery-loading');
  const tableEl   = document.getElementById('mastery-table');
  const emptyEl   = document.getElementById('mastery-empty');

  try {
    const res = await apiFetch('/user/mastery');
    loadingEl.style.display = 'none';

    if (!res || !res.ok) {
      emptyEl.style.display = 'block';
      return;
    }

    const data = await res.json();

    if (!data || !data.length) {
      emptyEl.style.display = 'block';
      return;
    }

    document.getElementById('mastery-body').innerHTML = data.map(m => {
      const theta    = typeof m.theta === 'number' ? m.theta : 0;
      const thetaPct = Math.min(Math.max((theta / 3) * 100, 0), 100);
      const acc      = typeof m.accuracy === 'number' ? m.accuracy : 0;
      const accPct   = Math.round(acc * 100);
      const accClass = accPct >= 75 ? 'accuracy-high'
                     : accPct >= 50 ? 'accuracy-mid'
                     : 'accuracy-low';
      return `
        <tr>
          <td><strong>${escHtml(m.topic_name || '—')}</strong></td>
          <td>
            <div class="theta-bar-wrap">
              <div class="theta-bar-bg">
                <div class="theta-bar-fill" style="width:${thetaPct}%"></div>
              </div>
              <span class="theta-value">${theta.toFixed(1)}</span>
            </div>
          </td>
          <td>${m.attempts ?? '—'}</td>
          <td>${m.correct  ?? '—'}</td>
          <td><span class="accuracy-pill ${accClass}">${accPct}%</span></td>
        </tr>`;
    }).join('');

    tableEl.style.display = 'table';
  } catch {
    loadingEl.style.display = 'none';
    emptyEl.style.display   = 'block';
    emptyEl.textContent     = 'Failed to load mastery data.';
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadProfile();
loadMastery();