# EduMentor

EduMentor is a production-grade AI tutoring backend that uses a multi-agent LangGraph pipeline to deliver Socratic, adaptive learning experiences over text and voice. It is aimed at students who want guided, personalised instruction rather than direct answers: the system classifies each student message, routes it to one of three specialist agents (Socratic guide, concept explainer, or quiz generator), and assembles every response from a 7-layer prompt that injects curriculum context retrieved from a vector store, the student's current IRT-estimated ability level, their conversational preferences, and recent session history. The platform tracks student mastery using a 3PL Item Response Theory model, supports GitHub OAuth as well as password authentication, and exposes a full MLOps loop — thumbs-down feedback triggers DPO training-data export, a weekly Celery beat job runs Evidently drift detection, and shadow testing lets a candidate fine-tuned model run in parallel with production without serving its output to users.

---

## Tech Stack

**Backend**
- Python 3.11
- FastAPI 0.136 with Uvicorn (ASGI)
- SQLAlchemy 2.0 (async) with asyncpg
- Alembic (schema migrations)
- Pydantic v2 / pydantic-settings
- python-jose (JWT), bcrypt / passlib (password hashing)
- structlog (structured logging)
- httpx (async HTTP client)

**Frontend**
- Vanilla HTML/CSS/JavaScript (no framework)
- Six pages: `index.html`, `login.html`, `dashboard.html`, `chat.html`, `quiz.html`, `profile.html`
- A lightweight `serve.py` development server (Python `http.server`)

**Database**
- Neon PostgreSQL (cloud-hosted Postgres) — users, sessions, messages, mastery profiles, quiz attempts, topics, feedback, jobs, user documents
- Upstash Redis — conversation history, rate limiting, in-flight deduplication locks, document ingestion status, multi-layer application cache (L2)
- Qdrant — two collections: `curriculum_docs` (chunked subject PDFs) and `user_memory` (per-user long-term memory summaries)

**ML & AI**
- Ollama (local) — serves the primary fine-tuned Phi-3.5-mini (`edumentor:latest` via `Modelfile`) and a fallback model; used by all agents except quiz
- Groq (cloud) — quiz agent only; model `meta-llama/llama-4-scout-17b-16e-instruct`
- DSPy 2.5 with `optimize_quiz_dspy.py` — few-shot quiz prompt optimisation, output saved to `dspy_optimized_quiz.json`
- sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — cross-encoder reranker for RAG results
- `nomic-embed-text` via Ollama — dense embeddings for Qdrant ingestion and retrieval
- Custom BM25 encoder — sparse vectors for hybrid retrieval
- HyDE (Hypothetical Document Embeddings) — query transform for curriculum retrieval
- CRAG (Corrective RAG) with Tavily web search fallback — triggered when local retrieval scores are below threshold
- Self-RAG reflection tokens — post-generation quality gate
- Multi-query transform — for user-memory retrieval
- RRF (Reciprocal Rank Fusion) — fuses dense and sparse Qdrant result lists
- faster-whisper — local STT for voice input
- Coqui XTTS v2 — local TTS for voice SSE stream output
- IRT 3PL (Item Response Theory) — adaptive student ability estimation (θ)
- Evidently AI — data drift detection
- RAGAS — RAG evaluation metrics
- Weights & Biases — experiment tracking
- LangSmith — LangGraph trace observability

**Infrastructure & DevOps**
- Docker (multi-stage Dockerfile: builder → runtime)
- Docker Compose — orchestrates: nginx, api, celery-worker, celery-beat, celery-flower, prometheus, grafana
- Nginx — reverse proxy; HTTP on port 80, proxies to `api:8000`; SSE-aware config for `/tutor/stream`
- Celery 5 with CloudAMQP (AMQP broker) — four task queues: `voice`, `quiz`, `session`, `mlops`; dead-letter queues for each
- Prometheus + Grafana — metrics scraping and dashboards
- Helm chart in `k8s/` — Kubernetes deployment with HPA (horizontal pod autoscaler)
- GitHub Actions implied (`.gitignore` patterns, LangSmith tracing hooks)

**Testing**
- pytest 8.3 with pytest-asyncio (async test support)
- pytest-cov (coverage)
- aiosqlite (in-memory SQLite for integration tests)
- RAGAS — RAG pipeline evaluation
- Groq LLM-as-Judge (`llama-3.3-70b-versatile`) for eval suite metrics

---

## Project Structure

### `app/`
The entire FastAPI application. Every subdirectory here is a Python package.

#### `app/main.py`
Application entry point. Defines the `lifespan` context manager that runs at startup: initialises Qdrant collections, fits BM25 from the actual ingested corpus (falls back to a hardcoded seed if empty), pre-warms the cross-encoder reranker, injects RAG dependencies into LangGraph, loads MCP tools, and starts the Prometheus side-server on port 9090. `DEBUG=true` in `.env` is required for `/docs` (Swagger) and `/redoc` to be served.

#### `app/agents/`
The LangGraph multi-agent orchestrator and all agent node implementations.

- **`graph.py`** — builds and compiles the `StateGraph`. The flow is always `START → retrieval → intent_classify → [socratic_agent | explainer_agent | quiz_agent] → END`. Intent routing is a conditional edge based on `state.intent`.
- **`state.py`** — `EduMentorState` (Pydantic model): the shared mutable object passed between all graph nodes. Contains session metadata, student profile fields (`theta`, `student_level`, `weak_topics`), RAG chunks, conversation history, quiz state, and error propagation.
- **`intent_classifier.py`** — classifies the student's message into one of five intents: `socratic`, `explain`, `quiz`, `feedback`, `meta`. Unknown intents fall back to `socratic`.
- **`socratic_agent.py`** — the main conversational agent. Calls `routed_chat` (or `routed_chat_with_shadow` if shadow testing is enabled) through the 3-tier cascade model router. Never gives direct answers — guides with questions.
- **`explainer_agent.py`** — used when intent is `explain`. Uses analogy-first, chain-of-thought structure.
- **`quiz_agent.py`** — the only agent that uses Groq (not Ollama). Generates MCQ questions using the 3PL IRT b-parameter as a difficulty target. Loads optimised few-shot demonstrations from `dspy_optimized_quiz.json` if present.
- **`retrieval_node.py`** — the first node in every graph run. Calls the hybrid retriever for both `curriculum_docs` and `user_memory` collections, populates `state.rag_chunks` and `state.session_summary`.
- **`ollama_client.py`** — thin async wrapper around the Ollama HTTP API; used directly by some agents for non-chat calls.
- **`tools/mcp_tools.py`** — at startup, wraps the arXiv LangChain community tool as a `BaseTool`. `MCP_SERVERS` in `.env` can configure additional stdio/SSE MCP servers; the tool list is returned via `get_mcp_tools()`.
- **`tools/web_search_tool.py`** — Tavily web search tool, used by CRAG when local retrieval quality is low.

#### `app/api/routes/`
All HTTP route handlers. Every router uses `Depends(get_current_user_id)` from `deps.py` for auth.

- **`auth.py`** — `POST /auth/signup`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`, `GET /auth/github/login`, `GET /auth/github/callback`. Both password and GitHub OAuth flows emit the same JWT access/refresh token pair.
- **`tutor.py`** — the largest route file. Handles session lifecycle (`POST /tutor/start`, `POST /tutor/end`), `POST /tutor/text` (the main chat endpoint, invokes LangGraph), `POST /tutor/voice` (202 async job pattern), `GET /tutor/stream` (SSE text + TTS audio), feedback, document upload/delete/status, job status polling, session list/rename/delete, history.
- **`quiz.py`** — `POST /quiz/generate`, `POST /quiz/submit`, quiz history.
- **`curriculum.py`** — `GET /curriculum/topics`, topic-level mastery data.
- **`user.py`** — user profile read/update, preferences.
- **`mlops.py`** — admin-only endpoints: `POST /mlops/reload-model`, `GET /mlops/drift-report`, `POST /mlops/export-dpo`, `GET /mlops/shadow-stats`.
- **`deps.py`** — `get_current_user_id` dependency: decodes the `Authorization: Bearer` JWT and returns the `user_id` string.

#### `app/core/`
Cross-cutting infrastructure utilities.

- **`config.py`** — `Settings` (pydantic-settings, reads `.env`). All environment variables are defined here with types and defaults. `get_settings()` is `lru_cache`-wrapped; `settings` is a module-level singleton.
- **`database.py`** — async SQLAlchemy engine and `get_db` dependency (yields `AsyncSession`).
- **`redis_client.py`** — `redis_client()` async context manager; all Redis key templates and helper functions (`get_history`, `push_message`, `check_rate_limit`, `acquire_inflight_lock`, `release_inflight_lock`).
- **`model_router.py`** — 3-tier cascade router. Tier 1: fine-tuned Phi-3.5 (primary Ollama). Tier 2: fallback Ollama model. Tier 3: hardcoded stub responses per agent type. Never raises — always returns a string.
- **`circuit_breaker.py`** — `CircuitBreaker` class implementing CLOSED / OPEN / HALF_OPEN state machine. Two instances: `_ollama_primary_cb`, `_ollama_secondary_cb`.
- **`multi_layer_cache.py`** — L1 (in-process `cachetools.TTLCache`) + L2 (Redis) two-layer cache. Pre-configured helpers for topics, mastery, job status, text responses, quiz responses. L1 is protected by a `threading.Lock` (safe for single-threaded asyncio).
- **`metrics.py`** — Prometheus `Counter` and `Histogram` definitions for request counts and latency; `metrics_middleware` wraps every HTTP handler.
- **`middleware.py`** — `metrics_middleware` function registered as an HTTP middleware on the FastAPI app.
- **`security.py`** — `hash_password`, `verify_password`, `create_access_token`, `create_refresh_token`, `decode_token`.
- **`shadow_testing.py`** — `shadow_call` fires the candidate model in a background task (non-blocking), logs both live and shadow responses to LangSmith. Activated by `SHADOW_MODEL_ENABLED=true`.

#### `app/models/`
SQLAlchemy ORM models (one file per table): `user.py` (User, UserPreference), `session.py` (Session, Message), `mastery.py` (MasteryProfile), `quiz.py` (QuizQuestion, QuizAttempt), `topic.py` (Topic), `feedback.py` (Feedback), `job.py` (Job), `user_document.py` (UserDocument).

#### `app/schemas/`
Pydantic v2 request/response schemas: `auth.py`, `tutor.py`, `quiz.py`, `topic.py`, `user.py`.

#### `app/prompts/assembly.py`
The 7-layer prompt builder used by all agents. Layers in strict order: system prompt → user preferences → session summary (from `user_memory` Qdrant) → RAG chunks (from `curriculum_docs` Qdrant) → chain-of-thought instruction → last 5 messages (Redis → PostgreSQL fallback) → current user query. Hard character budgets prevent context overflow.

#### `app/rag/`
All retrieval-augmented generation logic.

- **`retriever.py`** — hybrid retriever; for `curriculum_docs` uses HyDE + BM25 + Qdrant hybrid search + RRF fusion + cross-encoder rerank. For `user_memory` uses multi-query transform + BM25 + hybrid search + RRF + rerank.
- **`embeddings.py`** — `embed_text` / `embed_batch` using `nomic-embed-text` via Ollama HTTP API.
- **`bm25.py`** — custom `BM25Encoder`; fitted at startup from the actual Qdrant corpus.
- **`reranker.py`** — cross-encoder reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2`; `prewarm_reranker()` loads the model at startup.
- **`collections.py`** — Qdrant collection schemas, `init_collections()`, `get_qdrant_client()`.
- **`hyde.py`** — generates a hypothetical document for a query, embeds it; used for curriculum retrieval.
- **`multi_query.py`** — generates N paraphrases of a query; used for user-memory retrieval.
- **`crag.py`** — CRAG: runs local retrieval, scores results with the cross-encoder; if score < `_LOW_THRESHOLD` falls back to Tavily web search.
- **`self_rag.py`** — Self-RAG: three binary reflection tokens (`ISREL`, `ISSUP`, `ISUSE`) evaluated by the local LLM as a quality gate.
- **`filters.py`** — builds Qdrant `Filter` objects (`build_curriculum_filter`, `build_user_memory_filter`).
- **`ingestion.py`** — batch ingestion of curriculum PDFs from `data/`; called by `scripts/ingest.py`.
- **`user_docs_ingestion.py`** — ingestion and deletion of user-uploaded documents into the `user_docs` Qdrant collection.
- **`check.py`** — utility to inspect collection stats.

#### `app/services/`
Business logic that does not belong in route handlers.

- **`irt.py`** — 3PL IRT implementation: `p3pl`, `update_theta` (Newton-Raphson), `theta_to_level`. Updates `MasteryProfile` in PostgreSQL after each quiz attempt.
- **`session_service.py`** — `create_session`, `end_session`, `save_message`, `get_user_mastery`, `get_user_preferences`, `get_session_history_from_db`.
- **`memory_service.py`** — long-term memory consolidation: summarises session messages and upserts into the `user_memory` Qdrant collection. Called from Celery session tasks, not from graph nodes.
- **`stt.py`** — faster-whisper speech-to-text transcription.
- **`tts.py`** — Coqui XTTS v2 streaming TTS; `synthesize_stream` is an async generator yielding audio chunks for the SSE endpoint.

#### `app/tasks/`
All Celery task definitions.

- **`celery_app.py`** — Celery factory: CloudAMQP broker, no result backend (job status lives in PostgreSQL + app cache). Four durable queues with dead-letter exchange: `voice`, `session`, `mlops`, `doc`.
- **`voice_tasks.py`** — `process_voice`: transcribes audio (faster-whisper), runs LangGraph, synthesises TTS, writes result to the `jobs` table.
- **`session_tasks.py`** — `summarize_session`: called when a session ends; runs memory consolidation via `memory_service`.
- **`mlops_tasks.py`** — `export_dpo_pairs`: exports thumbs-down rows as JSONL for DPO fine-tuning. `run_drift_detection`: runs Evidently data drift check. Both tasks can be triggered on-demand or by the weekly beat schedule.
- **`doc_tasks.py`** — `ingest_document`: background task that chunks and embeds user-uploaded documents into Qdrant; updates the `user_documents` table on completion.

---

### `alembic/`
Alembic migration environment. `alembic.ini` at the root points here. Seven migration versions in `versions/`:
1. Initial schema (all core tables)
2. OAuth fields on users
3. `user_documents` table
4. `is_admin` on users
5. Backfill `mastery.theta`
6. Backfill `mastery.attempts` / `correct`
7. `session_id` on `user_documents`

---

### `frontend/`
Static web client — no build step, no framework. Six HTML pages correspond to the six UI screens. JavaScript is split by page: `api.js` contains all `fetch` wrappers (auth headers, base URL); `chat.js` drives the SSE stream and document upload UI; `quiz.js` handles MCQ rendering and IRT-adaptive submission. `serve.py` is a one-liner development HTTP server; in production, Nginx serves these files directly.

---

### `tests/`
Three test tiers (see section 6 for commands).

- **`unit/`** — 6 files; no I/O; tests IRT maths, circuit breaker state machine, prompt assembly character budgets, intent classifier routing, and agent state updates.
- **`integration/`** — 3 files; uses an in-process FastAPI test client with SQLite (aiosqlite); mocks Redis, Qdrant, and LLM calls. Tests the full LangGraph graph, REST endpoints, and the multi-layer cache.
- **`evaluation/`** — LLM-as-Judge eval suite using Groq `llama-3.3-70b-versatile`. Five metrics: `hallucination`, `correctness`, `relevance`, `socratic_quality`, `audio_clarity`. Runs against real or mocked Groq depending on whether `GROQ_API_KEY` is set.
- **`conftest.py`** (root) — injects safe placeholder env vars via `monkeypatch.setenv` so the unit suite runs without a `.env` file.

---

### `scripts/`
Operational utilities run directly (not imported by the app).

- **`ingest.py`** — reads PDFs from `data/`, chunks them with PyMuPDF, embeds with `nomic-embed-text`, upserts into `curriculum_docs` Qdrant collection. **Must be run before the app will retrieve meaningful curriculum context.**
- **`seed_topics.py`** — seeds the `topics` table in PostgreSQL with the five subject areas (Mathematics, Physics, Chemistry, Biology, Coding).
- **`fix_topics.py`** — repairs topic slug collisions or missing grade levels.
- **`optimize_quiz_dspy.py`** — runs DSPy bootstrap few-shot optimisation on the quiz agent; writes `app/agents/dspy_optimized_quiz.json`.

---

### `data/`
Seed PDF documents for the curriculum RAG pipeline. Five subject folders: `biology`, `chemistry`, `coding`, `mathematics`, `physics`. Each contains one reference PDF. These are the documents `scripts/ingest.py` reads. `data/README.txt` describes the expected directory layout.

---

### `k8s/`
Helm chart for Kubernetes deployment. `Chart.yaml` defines the chart metadata. `values.yaml` is the configuration surface. Templates: `deployment-api.yaml`, `deployment-celery.yaml`, `service.yaml`, `ingress.yaml`, `secrets.yaml`, `hpa.yaml` (Horizontal Pod Autoscaler).

---

### Root-level files

| File | Purpose |
|------|---------|
| `Dockerfile` | Two-stage build: `builder` compiles wheels; `runtime` copies wheels and app, runs as non-root `edumentor` user |
| `docker-compose.yml` | Local stack: nginx, api, celery-worker, celery-beat, celery-flower (127.0.0.1:5555), prometheus (127.0.0.1:9090), grafana (127.0.0.1:3000) |
| `nginx.conf` | HTTP reverse proxy; special SSE config for `/tutor/stream` (buffering off, 3600s timeout) |
| `prometheus.yml` | Scrape config pointing at the FastAPI Prometheus side-server |
| `Modelfile` | Ollama model definition for the fine-tuned Phi-3.5-mini; references `./models/phi-3.5-mini-instruct.Q4_K_M.gguf` |
| `alembic.ini` | Alembic config; `sqlalchemy.url` is overridden at runtime by `env.py` from `SYNC_DATABASE_URL` |
| `requirements.txt` | Pinned Python dependencies |
| `pyproject.toml` | ruff linting config |
| `pytest.ini` | Sets `pythonpath = .` and `asyncio_mode = auto` |

---

## How the System Works — End to End

The following traces a student sending a text message: `"Explain how photosynthesis works"`.

**1. Authentication** — The frontend attaches `Authorization: Bearer <jwt>` to every request. `deps.py` decodes the JWT with python-jose, extracts `user_id`, and injects it as a dependency.

**2. `POST /tutor/text`** — `tutor.py` receives the request. It first checks the Redis rate limiter (sliding window, `TEXT_RATE_LIMIT` requests per minute). It then reads the student's mastery profile: first from the L1/L2 multi-layer cache, falling back to a PostgreSQL query if cold. Conversation history is fetched from Redis (last 5 turns), falling back to the `messages` table. A per-user, per-session, per-query-hash Redis lock is acquired (`SET NX EX`) to prevent duplicate in-flight submissions.

**3. Cache check** — The MD5 of the message is checked against the text response cache (L1 + L2). On a hit the cached reply is returned immediately without touching LangGraph.

**4. LangGraph invocation** — An `EduMentorState` is constructed with all the student context (theta, level, history, preferences, topic info, `has_user_docs` flag). `graph.ainvoke(state)` is called, running the compiled `StateGraph`.

**5. Retrieval node** — The first graph node. For `curriculum_docs`: the raw query is transformed via HyDE (the LLM generates a hypothetical answer which is then embedded), BM25 encodes the query as a sparse vector, both are sent to Qdrant as a hybrid search, the results are RRF-fused, and the top-20 candidates are reranked by the cross-encoder to produce top-3 chunks. CRAG may trigger a Tavily web search if the cross-encoder scores are all below threshold. For `user_memory`: multi-query expansion produces N paraphrases, each is embedded and searched with a mandatory `user_id` filter, results are RRF-fused and reranked. Both result sets are written into `state.rag_chunks` and `state.session_summary`.

**6. Intent classification** — The classifier looks at `state.user_query` and returns one of: `socratic`, `explain`, `quiz`, `feedback`, `meta`. For this message it returns `explain`.

**7. Explainer agent** — The conditional edge routes to `explainer_agent`. It calls `build_prompt` from `app/prompts/assembly.py`, which assembles the 7-layer prompt: system prompt (with `student_level`, `theta`, `explanation_style` interpolated) → user preference instructions → session summary from memory → RAG curriculum chunks → CoT scaffold → last 5 history messages → current query. The assembled message list is passed to `routed_chat` in the model router.

**8. Model router** — Tier 1: checks `_ollama_primary_cb` (circuit breaker). If CLOSED, calls `ChatOllama` with `edumentor:latest` (the fine-tuned Phi-3.5-mini). On success, returns. On failure, the circuit breaker records the error. Tier 2: tries the fallback Ollama model. Tier 3: returns a hardcoded stub if both circuit breakers are OPEN.

**9. Response handling** — The route handler sanity-checks the reply for state leakage (if it starts with `{` and contains `user_query` etc., something went wrong). It saves the user message and the assistant message to the `messages` table (with `agent_type="explain"` and a LangSmith `trace_id`). The in-flight lock is released. Both messages are pushed to the Redis conversation list. The response payload is cached in L1 and L2.

**10. Response** — `TextResponse` is returned to the frontend: `{ session_id, reply, agent_type: "explain", trace_id, message_id }`.

**Voice path** — `POST /tutor/voice` is different: it returns a `202` immediately with a `job_id`, queues `process_voice` to the `voice` Celery queue, and the frontend polls `GET /tutor/job/{job_id}/status`. The worker runs STT (faster-whisper), then runs the same LangGraph pipeline, then calls Coqui XTTS and stores the result. For streaming audio, `GET /tutor/stream` is an SSE endpoint that sends the last assistant text first, then streams TTS audio chunks as base64.

**Quiz path** — When intent is `quiz`, the route redirects the frontend with `quiz_redirect: true` and a `topic_id`. The frontend calls `POST /quiz/generate`, which invokes the Groq-backed `quiz_agent` and returns an MCQ. The student submits via `POST /quiz/submit`; `irt.py` computes the new `theta` using a Newton-Raphson update and writes it to `mastery_profiles`.

**Async workers** — When a session ends, `summarize_session` runs in Celery (`session` queue): it summarises the conversation with the LLM and upserts the summary as a vector into `user_memory` Qdrant. When 5 students each give 3 thumbs-down, `export_dpo_pairs` runs in Celery (`mlops` queue): it queries thumbed-down feedback rows, exports `(prompt, rejected)` pairs as JSONL, and marks them so they are never exported twice.

---

## How to Run This Project Locally

### Prerequisites

- Git
- Docker and Docker Compose
- The fine-tuned GGUF model file (see Gotchas below)
- Accounts and API keys for: Neon PostgreSQL, Upstash Redis, Qdrant Cloud (or self-hosted Qdrant), CloudAMQP (free tier works)

### Environment Variables

Create a `.env` file in the project root. Every variable listed below is required unless marked optional.

```env
# PostgreSQL (Neon)
DATABASE_URL=postgresql+asyncpg://user:password@host/dbname
SYNC_DATABASE_URL=postgresql+psycopg2://user:password@host/dbname

# Redis (Upstash)
# CRITICAL: use rediss:// (TLS), not https://
REDIS_URL=rediss://default:password@host:6380
REDIS_TOKEN=your-upstash-token

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Celery (CloudAMQP)
CELERY_BROKER_URL=amqps://user:password@host/vhost

# JWT
JWT_SECRET=a-random-string-at-least-32-chars-long

# Groq (required for quiz agent and eval tests)
GROQ_API_KEY=gsk_...

# Ollama (if using the default localhost Ollama)
# Leave as default unless Ollama is on a different host
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=edumentor:latest
OLLAMA_FALLBACK_MODEL=qwen3.5:0.8b

# Required for /docs (Swagger UI) to be served
DEBUG=true

# GitHub OAuth (optional — only needed if you want GitHub login)
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
GITHUB_REDIRECT_URI=http://localhost/auth/github/callback
FRONTEND_URL=http://localhost:3000

# Observability (optional)
LANGSMITH_API_KEY=
WANDB_API_KEY=
TAVILY_API_KEY=
```

### Steps

```bash
# 1. Clone the repo
git clone <repo-url>
cd edumentor

# 2. Create the .env file (see above)

# 3. Start Ollama on your host machine (not in Docker) and pull models
ollama pull nomic-embed-text
ollama pull qwen3.5:0.8b   # fallback model

# 4. (Optional but recommended) Load the fine-tuned model
#    Place the GGUF file at ./models/phi-3.5-mini-instruct.Q4_K_M.gguf
#    then register it with Ollama:
ollama create edumentor -f Modelfile
#    If you skip this, set OLLAMA_MODEL=qwen3.5:0.8b in .env to use the fallback as primary

# 5. Run database migrations (requires SYNC_DATABASE_URL to be set)
docker compose run --rm api alembic upgrade head

# 6. Seed curriculum topics into PostgreSQL
docker compose run --rm api python scripts/seed_topics.py

# 7. Ingest curriculum PDFs into Qdrant
docker compose run --rm api python scripts/ingest.py

# 8. Start the full stack
docker compose up --build -d

# 9. Verify health
curl http://localhost/health
# Expected: {"status":"ok"}
```

### Verify it is working

- `GET http://localhost/health` → `{"status": "ok"}`
- `GET http://localhost/docs` → Swagger UI (requires `DEBUG=true`)
- `GET http://localhost:9090` → Prometheus (bound to 127.0.0.1; access from the host directly)
- `GET http://localhost:3000` → Grafana login (default password: `edumentor`)
- `GET http://localhost:5555` → Celery Flower (bound to 127.0.0.1)

---

## How to Run the Tests

Install dependencies in a virtualenv first:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The root `conftest.py` injects placeholder env vars automatically, so unit and integration tests do not require a real `.env` file.

### Unit tests (no I/O, no API keys needed)

```bash
pytest tests/unit/ -v
```

Covers: IRT 3PL maths (`p3pl`, `update_theta`, `theta_to_level`), circuit breaker state machine transitions, 7-layer prompt assembly and character budget enforcement, intent classifier routing and fallback, socratic and quiz agent state mutations.

### Integration tests (mocked infra)

```bash
pytest tests/integration/ -v
```

Covers: the full LangGraph graph end-to-end with mocked Ollama/Groq, REST endpoint auth flows and response shapes (using FastAPI `TestClient` backed by aiosqlite), multi-layer cache read/write/invalidation. No real Redis, PostgreSQL, or Qdrant connections needed.

### Evaluation tests (LLM-as-Judge)

```bash
# Mocked Groq (no API key needed, uses synthetic responses)
pytest tests/evaluation/ -m eval -v

# Live Groq (real API calls — set GROQ_API_KEY first)
pytest tests/evaluation/ -m eval -v -s
```

The `-s` flag prints the evaluation summary table. Covers: hallucination (≥ 0.70), correctness (≥ 0.70), relevance (≥ 0.60), socratic quality (≥ 0.70), audio clarity / TTS suitability (≥ 0.70).

### Full suite with coverage

```bash
pytest --cov=app --cov-report=term-missing -v
```

### Skip evaluation in CI

```bash
pytest -m "not eval"
```

---

## Gotchas & Non-Obvious Things

**The fine-tuned GGUF is not in the repo.** `Modelfile` references `./models/phi-3.5-mini-instruct.Q4_K_M.gguf` which is a local file that must be placed on the host running Ollama. Without it, set `OLLAMA_MODEL=qwen3.5:0.8b` (or any installed Ollama model) to use the fallback as primary — the system will work but without the fine-tuned behaviour.

**`DEBUG=true` is required for `/docs`.** `app/main.py` passes `docs_url="/docs" if settings.DEBUG else None`. If `DEBUG` is absent from `.env` (or is `false`), `/docs` returns 404. This is not a bug; it is intentional for production.

**Redis URL must be `rediss://` (TLS).** Upstash Redis requires the `rediss://` scheme. Using `https://` or `redis://` will cause connection failures at startup.

**Upstash free-tier databases pause after inactivity.** If you hit Redis connection errors after not using the app for a while, the Upstash DB has likely paused. Create a new database instance from the Upstash dashboard and update `REDIS_URL` and `REDIS_TOKEN`.

**Celery broker must be CloudAMQP (AMQP), not Redis.** The Celery config uses `amqps://` for `CELERY_BROKER_URL`. Using Upstash Redis as the Celery broker will fail because Upstash blocks port 6380 for non-Redis protocols. CloudAMQP free tier (Lemur plan) is sufficient.

**Ingest and seed must run before the app is useful.** If `scripts/ingest.py` has not been run, the `curriculum_docs` Qdrant collection is empty. At startup, BM25 will fall back to its hardcoded seed corpus and RAG will return no chunks. The app starts cleanly in this state, but agent responses will lack curriculum context.

**`scripts/seed_topics.py` must run before quiz generation.** The quiz agent queries the `topics` table to resolve topic UUIDs. An empty topics table causes quiz topic auto-creation for every request, which works but pollutes the table.

**Prometheus port conflict on uvicorn reload.** `start_http_server(settings.PROMETHEUS_PORT)` binds a socket in the lifespan. On hot reload (uvicorn `--reload`), the lifespan runs again and the port is already bound, raising `OSError`. The code catches this and logs a warning — the app continues but only the first-bound server serves metrics. In production this is not an issue (single-worker, no reload).

**LangSmith tracing is optional but highly recommended for debugging.** Set `LANGSMITH_API_KEY` and the `LANGCHAIN_TRACING_V2=true` env var is set automatically. Without it, `state.langsmith_trace_id` will be empty and trace-based debugging is unavailable.

**The `quiz` intent path in `/tutor/text` does not call the quiz agent.** When `intent == "quiz"`, the route handler returns early with a `quiz_redirect: true` response and a `topic_id`. The actual quiz question is generated only when the frontend explicitly calls `POST /quiz/generate`. This is by design — the chat endpoint is not intended to return quiz JSON inline.

**DPO export produces a `chosen`-null JSONL.** `mlops_tasks.export_dpo_pairs` writes `{ prompt, rejected, chosen: null }`. A separate Colab notebook (not included in this repo) is expected to generate `chosen` responses via Groq before running DPO fine-tuning with Unsloth.

**Shadow testing is disabled by default.** Set `SHADOW_MODEL_ENABLED=true` and `SHADOW_MODEL_NAME=<any Ollama model name>` to activate it. Shadow calls are fire-and-forget; failures are logged but never surface to the user.

**K8s Helm chart assumes secrets are pre-created.** `k8s/templates/secrets.yaml` contains a `Secret` manifest. Values are expected to be base64-encoded strings from `k8s/values.yaml`. The chart is not production-hardened (no external secrets operator, no vault integration).

**`ollama_client.py` vs `model_router.py`** — `ollama_client.py` is a thin direct HTTP client used for non-chat calls (embedding checks, one-off completions). `model_router.py` uses LangChain's `ChatOllama` and is the path used by all agent nodes for chat completions. They are not interchangeable.

**Celery result backend is intentionally disabled.** `backend=None` in `celery_app.py`. Task status is tracked via the `jobs` PostgreSQL table and the multi-layer app cache. Do not set `CELERY_RESULT_BACKEND` expecting it to work as standard Celery result storage — the config ignores it.