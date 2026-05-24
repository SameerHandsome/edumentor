"""
Centralized Prometheus metrics with p50/p95/p99 quantile histograms.
All latency metrics define explicit quantile buckets so Grafana can
compute p50, p95, p99 with histogram_quantile().
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Latency buckets: covers p50 (~0.5s) to p99 (~10s) for LLM workloads ─────
_LLM_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 30.0]
_FAST_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

# ── HTTP ─────────────────────────────────────────────────────────────────────
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency — use histogram_quantile(0.99, ...) for p99",
    ["method", "endpoint", "status_code"],
    buckets=_FAST_BUCKETS,
)
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)
HTTP_ERRORS_TOTAL = Counter("http_errors_total", "Total HTTP 5xx errors", ["endpoint"])

# ── LLM / Agent ───────────────────────────────────────────────────────────────
LLM_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "LLM inference latency per tier — p50/p95/p99 via histogram_quantile",
    ["agent_type", "tier"],
    buckets=_LLM_BUCKETS,
)
LLM_TOKENS = Counter("llm_tokens_generated_total", "Approximate tokens generated", ["agent_type"])

# ── STT / TTS ─────────────────────────────────────────────────────────────────
STT_LATENCY = Histogram(
    "stt_latency_seconds",
    "Whisper transcription latency",
    buckets=_LLM_BUCKETS,
)
TTS_LATENCY = Histogram(
    "tts_latency_seconds",
    "Coqui TTS synthesis latency",
    buckets=_LLM_BUCKETS,
)
VOICE_RTT = Histogram(
    "voice_round_trip_seconds",
    "Full voice pipeline RTT: upload → STT → LLM → TTS — critical p95/p99",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0],
)

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "End-to-end RAG retrieval latency",
    ["collection", "strategy"],  # strategy: crag_correct|crag_corrected|self_rag_skipped
    buckets=_FAST_BUCKETS,
)
CRAG_STATUS = Counter(
    "crag_retrieval_status_total",
    "CRAG retrieval quality status",
    ["status"],  # correct|ambiguous|corrected|fallback|empty
)
RERANKER_LATENCY = Histogram(
    "reranker_latency_seconds",
    "Cross-encoder reranker latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
)

# ── Celery ────────────────────────────────────────────────────────────────────
CELERY_QUEUE_LENGTH = Gauge("celery_queue_length", "Current Celery queue depth", ["queue"])
CELERY_TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Celery task execution time",
    ["task_name"],
    buckets=_LLM_BUCKETS,
)
CELERY_TASK_FAILURES = Counter("celery_task_failures_total", "Celery task failures", ["task_name"])

# ── Circuit Breaker ───────────────────────────────────────────────────────────
CB_STATE_GAUGE = Gauge(
    "circuit_breaker_state", "CB state: 0=closed, 1=open, 2=half-open", ["service"]
)
CB_TRIPS_COUNTER = Counter("circuit_breaker_trips_total", "Circuit breaker trips", ["service"])

# ── IRT / Learning ────────────────────────────────────────────────────────────
THETA_GAUGE = Gauge("student_theta_average", "Average IRT theta across all active students")
QUIZ_ATTEMPTS = Counter("quiz_attempts_total", "Quiz attempts", ["correct"])
DRIFT_SCORE_GAUGE = Gauge("evidently_drift_score", "Latest Evidently AI drift score")
