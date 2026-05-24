# EduMentor Test Suite

```
tests/
├── conftest.py                          # Shared fixtures (settings, base_state)
│
├── unit/                                # Fast, isolated, no I/O
│   ├── test_intent_classifier.py        # Intent routing + fallback
│   ├── test_socratic_agent.py           # Socratic agent state updates
│   ├── test_quiz_agent.py               # Groq quiz generation, IRT keys
│   ├── test_prompt_assembly.py          # 7-layer prompt, char budget
│   ├── test_circuit_breaker.py          # CB state machine (CLOSED/OPEN/HALF_OPEN)
│   └── test_irt_scoring.py              # IRT maths: P(correct), window, theta update
│
├── integration/                         # Component interaction, mocked infra
│   ├── conftest.py                      # FastAPI client, service mocks
│   ├── test_agent_graph.py              # Full LangGraph graph end-to-end
│   ├── test_api_endpoints.py            # REST endpoints (auth, tutor, quiz)
│   └── test_cache_layer.py             # Redis multi-layer cache read/write
│
└── evaluation/                          # LLM-as-Judge via Groq
    ├── conftest.py                      # Registers 'eval' marker
    ├── llm_judge.py                     # Judge module (all metrics, Groq only)
    ├── test_hallucination_correctness.py
    ├── test_socratic_relevance.py
    ├── test_audio_clarity.py
    └── test_eval_suite.py               # Full suite + summary table
```

---

## Running tests

### All unit tests (fast, no API keys)
```bash
pytest tests/unit/ -v
```

### All integration tests
```bash
pytest tests/integration/ -v
```

### Evaluation tests (mocked Groq — no real API key needed)
```bash
pytest tests/evaluation/ -m eval -v
```

### Skip evaluation in CI
```bash
pytest -m "not eval"
```

### Full suite + coverage
```bash
pytest --cov=app --cov-report=term-missing -v
```

---

## Evaluation metrics

All evaluation uses **Groq** as the LLM judge (`llama-3.3-70b-versatile`).
**No OpenAI, no Anthropic** — only Groq.

| Metric              | What it measures                                           | Pass threshold |
|---------------------|------------------------------------------------------------|---------------|
| `hallucination`     | Response stays grounded in the provided context            | ≥ 0.70        |
| `correctness`       | Response matches the reference answer                      | ≥ 0.70        |
| `relevance`         | Response is on-topic / answers the question                | ≥ 0.60        |
| `socratic_quality`  | Response guides rather than answers directly               | ≥ 0.70        |
| `audio_clarity`     | Response is suitable for TTS (no markdown/LaTeX)           | ≥ 0.70        |

### Live Groq evaluation (real API calls)
Set `GROQ_API_KEY` in `.env` and run:
```bash
pytest tests/evaluation/ -m eval -v -s
```
The `-s` flag prints the summary table to stdout.

---

## Environment variables required for tests

```env
GROQ_API_KEY=gsk_...            # Only needed for live eval runs
DATABASE_URL=...                # Mocked in CI
REDIS_URL=...                   # Mocked in CI
QDRANT_URL=...                  # Mocked in CI
JWT_SECRET=...                  # Any 32+ char string for CI
```

The root `conftest.py` injects safe placeholder values automatically via
`monkeypatch.setenv` so tests run in CI without a real `.env`.
