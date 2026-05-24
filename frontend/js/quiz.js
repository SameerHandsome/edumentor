// quiz.js

if (!requireAuth()) { /* redirected */ }

document.getElementById('app').insertAdjacentHTML('afterbegin', buildSidebar('quiz'));

// ── URL params ────────────────────────────────────────────────────────────────
const topicId    = getParam('topic_id');
const topicName  = getParam('topic_name') || 'Quiz';
const adhocTopic = getParam('adhoc_topic'); // from chat.js redirect when topic has no UUID

// BUG FIX: quiz.html can be opened directly from dashboard without a session_id
// (e.g. clicking "Quiz Me" on a topic card). The old code passed an empty string
// to /quiz/generate?session_id= which caused a 422 Unprocessable Entity from
// FastAPI because session_id is a required UUID query param.
// Fix: start a real tutor session first if no session_id is in the URL.
let sessionId   = getParam('session_id');

// ── State ─────────────────────────────────────────────────────────────────────
let questions     = [];
let currentIndex  = 0;
let timerInterval = null;
let timerSeconds  = 30;
let timerStart    = null;
const results     = [];   // { question, is_correct }

// ── Bootstrap ─────────────────────────────────────────────────────────────────
if (topicId) {
  // Flow 1 — UUID topic in URL (dashboard / curriculum picker)
  ensureSessionThenLoad();
} else if (adhocTopic) {
  // Flow 2 — chat redirect: ?adhoc_topic=Machine+Learning (no UUID)
  window._pickerTopicId   = adhocTopic;
  window._pickerTopicName = adhocTopic;
  startPickerQuiz();
} else {
  // No topic at all — show picker so user can choose
  showTopicPicker();
}

// BUG FIX: if no session_id came from the URL (user clicked "Quiz Me" directly
// from the dashboard topic card), create a session first so /quiz/generate
// gets a valid UUID instead of an empty string.
async function ensureSessionThenLoad() {
  if (!sessionId) {
    try {
      const res = await apiFetch('/tutor/start', {
        method: 'POST',
        body: JSON.stringify({ topic_id: topicId })
      });
      if (res && res.ok) {
        const data = await res.json();
        sessionId = data.session_id;
      }
    } catch { /* fall through — loadQuiz will show no-questions state */ }
  }
  loadQuiz();
}

// ── Topic picker (shown when quiz.html opened with no topic_id) ───────────────
async function showTopicPicker() {
  showState('picker');

  // Populate curriculum topics
  const topicList = document.getElementById('picker-topic-list');
  try {
    const res = await apiFetch('/curriculum/topics');
    if (res && res.ok) {
      const topics = await res.json();
      if (topics && topics.length) {
        topicList.innerHTML = topics.map(t => `
          <button class="choice-btn picker-topic-btn"
                  onclick="selectPickerTopic('${escHtml(t.id)}', '${escHtml(t.name)}')">
            <span class="choice-label">📚</span>
            <span>${escHtml(t.name)}</span>
          </button>`).join('');
      } else {
        topicList.innerHTML = '<p class="picker-empty">No curriculum topics found.</p>';
      }
    } else {
      topicList.innerHTML = '<p class="picker-empty">Could not load topics.</p>';
    }
  } catch {
    topicList.innerHTML = '<p class="picker-empty">Could not load topics.</p>';
  }
}

function selectPickerTopic(tid, tname) {
  // Set globals so loadQuiz works normally
  window._pickerTopicId   = tid;
  window._pickerTopicName = tname;
  const _nameEl = document.getElementById('picker-topic-name');
  if (_nameEl) _nameEl.textContent = tname;
  startPickerQuiz();
}

async function startPickerQuiz() {
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const _inputEl = document.getElementById('custom-topic-input');
  const rawInput  = window._pickerTopicId || (_inputEl ? _inputEl.value.trim() : '');
  const tname    = window._pickerTopicName || rawInput;

  if (!rawInput) {
    showToast('Please select a topic or enter a custom topic.', 'warning');
    return;
  }

  const _nameEl = document.getElementById('picker-topic-name');
  if (_nameEl) _nameEl.textContent = tname;
  showState('loading');

  // If rawInput is a topic name (not a UUID), resolve/create via the backend.
  // POST /tutor/start accepts topic_id as null — the topic UUID will be resolved
  // on the quiz side via get_or_create_topic().
  let resolvedTopicId = rawInput;

  if (!UUID_RE.test(rawInput)) {
    // Custom text topic — create an ad-hoc session without a topic_id,
    // then call /quiz/generate with topic_name so the backend auto-creates the topic.
    try {
      const res = await apiFetch('/tutor/start', {
        method: 'POST',
        body: JSON.stringify({ topic_id: null })
      });
      if (res && res.ok) {
        const data = await res.json();
        sessionId = data.session_id;
      }
    } catch { /* fall through */ }

    questions    = [];
    currentIndex = 0;
    showState('loading');
    try {
      const url = `/quiz/generate` +
        `?topic_name=${encodeURIComponent(rawInput)}` +
        `&session_id=${encodeURIComponent(sessionId || '')}` +
        `&num_questions=10`;

      const r = await apiFetch(url);
      if (!r || r.status === 404 || !r.ok) { showNoQuestions(); return; }

      questions = await r.json();
      if (!questions || !questions.length) { showNoQuestions(); return; }
      window._activeTopicName = tname;
      currentIndex = 0;
      showQuestion();
    } catch {
      showToast('Failed to load quiz questions.', 'error');
    }
    return;
  }

  // UUID path — topic already resolved from curriculum picker
  try {
    const res = await apiFetch('/tutor/start', {
      method: 'POST',
      body: JSON.stringify({ topic_id: resolvedTopicId })
    });
    if (res && res.ok) {
      const data = await res.json();
      sessionId = data.session_id;
    }
  } catch { /* fall through */ }

  questions    = [];
  currentIndex = 0;
  showState('loading');
  try {
    const url = `/quiz/generate` +
      `?topic_id=${encodeURIComponent(resolvedTopicId)}` +
      `&session_id=${encodeURIComponent(sessionId || '')}` +
      `&num_questions=10`;

    const r = await apiFetch(url);
    if (!r || r.status === 404 || !r.ok) { showNoQuestions(); return; }

    questions = await r.json();
    if (!questions || !questions.length) { showNoQuestions(); return; }
    window._activeTopicName = tname;
    currentIndex = 0;
    showQuestion();
  } catch {
    showToast('Failed to load quiz questions.', 'error');
  }
}

// ── Load quiz ─────────────────────────────────────────────────────────────────
async function loadQuiz() {
  showState('loading');

  // Guard: topic_id must be a UUID. If it's a name string (e.g. from a direct link
  // or a mis-configured redirect), show the topic picker instead of sending a 422.
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!topicId || !UUID_RE.test(topicId)) {
    showTopicPicker();
    return;
  }

  // Guard: session_id must also be a non-empty UUID.
  if (!sessionId || !UUID_RE.test(sessionId)) {
    // Try to create a session first, then reload
    try {
      const sr = await apiFetch('/tutor/start', {
        method: 'POST',
        body: JSON.stringify({ topic_id: topicId })
      });
      if (sr && sr.ok) {
        const sd = await sr.json();
        sessionId = sd.session_id;
      }
    } catch { /* fall through — will 422 and show no-questions */ }
  }

  try {
    const url = `/quiz/generate` +
      `?topic_id=${encodeURIComponent(topicId)}` +
      `&session_id=${encodeURIComponent(sessionId || '')}` +
      `&num_questions=10`;

    const res = await apiFetch(url);

    if (!res || res.status === 404 || !res.ok) {
      showNoQuestions();
      return;
    }

    questions = await res.json();

    if (!questions || !questions.length) {
      showNoQuestions();
      return;
    }

    currentIndex = 0;
    showQuestion();
  } catch {
    showToast('Failed to load quiz questions.', 'error');
  }
}

function showNoQuestions() {
  showState('loading');
  document.querySelector('#state-loading h3').textContent = 'No questions available';
  document.querySelector('#state-loading p').textContent  =
    'No questions available for this topic yet. Try another topic.';
  document.querySelector('#state-loading .spinner').style.display = 'none';

  const back      = document.createElement('a');
  back.href       = 'dashboard.html';
  back.className  = 'btn btn-secondary';
  back.textContent = '← Back to Dashboard';
  back.style.marginTop = '16px';
  document.getElementById('state-loading').appendChild(back);
}

// ── Show question ─────────────────────────────────────────────────────────────
function showQuestion() {
  showState('question');
  const q = questions[currentIndex];

  document.getElementById('q-topic').textContent    = window._activeTopicName || topicName;
  document.getElementById('q-progress').textContent =
    `Question ${currentIndex + 1} of ${questions.length}`;
  document.getElementById('difficulty-label').textContent =
    `Difficulty: ${q.difficulty_b?.toFixed(2) ?? '—'}`;
  document.getElementById('question-text').textContent = q.question_text;

  const resultSection = document.getElementById('result-section');
  resultSection.style.display = 'none';
  resultSection.innerHTML     = '';

  // Render choice buttons
  const grid = document.getElementById('choices-grid');
  grid.innerHTML = '';
  Object.entries(q.choices).forEach(([key, val]) => {
    const btn       = document.createElement('button');
    btn.className   = 'choice-btn';
    btn.dataset.key = key;
    btn.innerHTML   =
      `<span class="choice-label">${key}</span><span>${escHtml(val)}</span>`;
    btn.onclick = () => submitAnswer(key);
    grid.appendChild(btn);
  });

  startTimer();
}

// ── Timer ─────────────────────────────────────────────────────────────────────
function startTimer() {
  clearTimer();
  timerSeconds = 30;
  timerStart   = Date.now();
  updateTimerDisplay(30);

  timerInterval = setInterval(() => {
    timerSeconds--;
    updateTimerDisplay(timerSeconds);
    if (timerSeconds <= 0) {
      clearTimer();
      submitAnswer('', true);
    }
  }, 1000);
}

function clearTimer() {
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function updateTimerDisplay(s) {
  const numEl = document.getElementById('timer-num');
  const barEl = document.getElementById('timer-bar');
  numEl.textContent = s;
  numEl.className   = `timer-number${s <= 8 ? ' urgent' : ''}`;
  barEl.style.width = `${(s / 30) * 100}%`;
  barEl.className   = `timer-bar-fill${s <= 8 ? ' urgent' : ''}`;
}

function skipQuestion() {
  clearTimer();
  submitAnswer('', true);
}

// ── Submit answer ─────────────────────────────────────────────────────────────
async function submitAnswer(selectedKey) {
  clearTimer();

  const elapsed   = timerStart ? Math.round((Date.now() - timerStart) / 1000) : 30;
  const timeTaken = Math.min(elapsed, 30);

  // Disable buttons & highlight selected
  document.querySelectorAll('.choice-btn').forEach(b => {
    b.disabled = true;
    if (b.dataset.key === selectedKey) b.classList.add('selected');
  });

  // Show scoring indicator
  const resultSection = document.getElementById('result-section');
  resultSection.style.display = 'block';
  resultSection.innerHTML =
    `<div class="result-loading"><div class="spinner spinner-sm"></div> Scoring your answer…</div>`;

  const q      = questions[currentIndex];
  let   result = null;

  try {
    const res = await apiFetch('/quiz/submit', {
      method: 'POST',
      body: JSON.stringify({
        session_id:         sessionId || '',
        question_id:        q.id,
        selected_answer:    selectedKey,
        time_taken_seconds: timeTaken
      })
    });

    if (!res || !res.ok) throw new Error('Submit failed');

    const { job_id } = await res.json();

    result = await poll(async () => {
      const r = await apiFetch(`/quiz/result/${job_id}`);
      if (!r || !r.ok) return false;
      const d = await r.json();
      if (d.status === 'done')   return d.result;
      if (d.status === 'failed') return '__failed__';
      return false;
    }, 1500);
  } catch {
    result = '__failed__';
  }

  // ── Handle failed scoring ──────────────────────────────────────────────────
  if (result === '__failed__' || result === null) {
    resultSection.innerHTML = `
      <div class="result-verdict">
        <span class="verdict-icon">⚠️</span>
        <div class="verdict-text">
          <h3>Scoring failed</h3>
          <p>Your answer was recorded. Moving on…</p>
        </div>
      </div>`;
    results.push({ question: q.question_text, is_correct: null });
    renderNextButton(resultSection);
    return;
  }

  // ── Colour answer buttons ──────────────────────────────────────────────────
  document.querySelectorAll('.choice-btn').forEach(b => {
    if (b.dataset.key === result.correct_answer)             b.classList.add('correct');
    else if (b.dataset.key === selectedKey && !result.is_correct) b.classList.add('wrong');
  });

  // ── Show result details ────────────────────────────────────────────────────
  const delta      = result.theta_after - result.theta_before;
  const arrow      = delta >= 0 ? '↑' : '↓';
  const arrowColor = delta >= 0 ? 'var(--color-success)' : 'var(--color-error)';
  const verdictColor = result.is_correct ? 'var(--color-success)' : 'var(--color-error)';

  resultSection.innerHTML = `
    <div class="result-verdict">
      <span class="verdict-icon">${result.is_correct ? '✅' : '❌'}</span>
      <div class="verdict-text">
        <h3 style="color:${verdictColor}">${result.is_correct ? 'Correct!' : 'Incorrect'}</h3>
        <p>Correct answer: <strong>${escHtml(result.correct_answer)}</strong></p>
      </div>
    </div>
    <div class="theta-change">
      <span>Skill level:</span>
      <span style="font-family:var(--font-mono)">${result.theta_before?.toFixed(2)}</span>
      <span>→</span>
      <span style="font-family:var(--font-mono);color:${arrowColor}">
        ${result.theta_after?.toFixed(2)} ${arrow}
      </span>
    </div>
    <div class="explanation-box">${escHtml(result.explanation || '')}</div>`;

  results.push({ question: q.question_text, is_correct: result.is_correct });
  renderNextButton(resultSection);
}

// ── Next button ───────────────────────────────────────────────────────────────
function renderNextButton(container) {
  const btn       = document.createElement('button');
  btn.className   = 'btn btn-primary';
  btn.textContent = currentIndex + 1 < questions.length ? 'Next Question →' : 'See Results';
  btn.onclick     = () => {
    currentIndex++;
    if (currentIndex < questions.length) showQuestion();
    else showFinished();
  };
  container.appendChild(btn);
}

// ── Finished screen ───────────────────────────────────────────────────────────
function showFinished() {
  showState('finished');

  const correct  = results.filter(r => r.is_correct === true).length;
  const total    = results.length;
  const accuracy = total ? Math.round((correct / total) * 100) : 0;

  document.getElementById('finish-trophy').textContent    =
    accuracy >= 80 ? '🏆' : accuracy >= 60 ? '🎓' : '📚';
  document.getElementById('finish-score').textContent     = `${correct}/${total}`;
  document.getElementById('finish-accuracy').textContent  = `${accuracy}%`;
  document.getElementById('finish-total').textContent     = total;
  document.getElementById('finish-subtitle').textContent  =
    accuracy >= 80 ? 'Excellent work! Keep it up.' :
    accuracy >= 60 ? 'Good effort! Keep practicing.' :
    'Keep going — practice makes perfect.';

  document.getElementById('question-summary').innerHTML = results.map((r, i) => `
    <div class="summary-row">
      <span class="summary-icon">
        ${r.is_correct === true ? '✅' : r.is_correct === false ? '❌' : '⚠️'}
      </span>
      <span class="summary-text">
        Q${i + 1}: ${escHtml((questions[i]?.question_text || '—').slice(0, 60))}…
      </span>
    </div>`).join('');
}

// ── Utility ───────────────────────────────────────────────────────────────────
function showState(state) {
  ['loading', 'question', 'finished', 'picker'].forEach(s => {
    const el = document.getElementById(`state-${s}`);
    if (el) el.style.display = s === state ? '' : 'none';
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}