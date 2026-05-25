"""
Centralized Prometheus metrics with p50/p95/p99 quantile histograms.
All latency metrics define explicit quantile buckets so Grafana can
compute p50, p95, p99 with histogram_quantile().
"""

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# ── Latency buckets: covers p50 (~0.5s) to p99 (~10s) for LLM workloads ─────
_LLM_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 30.0]
_FAST_BUCKETS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


# ── Get-or-create helpers ─────────────────────────────────────────────────────
# Python caches modules, so in a single pytest process this module body only
# runs once per interpreter — but _clear_prometheus_registry() in the test
# fixture unregisters collectors while the module stays cached.  The next
# fixture setup re-imports the already-cached module (no-op), yet the
# collectors are gone from the registry, so any code path that re-registers
# them (e.g. a second `from app.core.metrics import ...`) raises ValueError.
# Wrapping every definition in a get-or-create call makes registration
# idempotent whether the module is fresh or being re-entered after a clear.

def _histogram(name: str, doc: str, labelnames=(), buckets=None) -> Histogram:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    kwargs: dict = {}
    if labelnames:
        kwargs["labelnames"] = list(labelnames)
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, **kwargs)


def _counter(name: str, doc: str, labelnames=()) -> Counter:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    kwargs: dict = {}
    if labelnames:
        kwargs["labelnames"] = list(labelnames)
    return Counter(name, doc, **kwargs)


def _gauge(name: str, doc: str, labelnames=()) -> Gauge:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    kwargs: dict = {}
    if labelnames:
        kwargs["labelnames"] = list(labelnames)
    return Gauge(name, doc, **kwargs)


# ── HTTP ─────────────────────────────────────────────────────────────────────
HTTP_REQUEST_DURATION = _histogram(
    "http_request_duration_seconds",
    "HTTP request latency — use histogram_quantile(0.99, ...) for p99",
    labelnames=["method", "endpoint", "status_code"],
    buckets=_FAST_BUCKETS,
)
HTTP_REQUESTS_TOTAL = _counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"],
)
HTTP_ERRORS_TOTAL = _counter(
    "http_errors_total",
    "Total HTTP 5xx errors",
    labelnames=["endpoint"],
)

# ── LLM / Agent ───────────────────────────────────────────────────────────────
LLM_LATENCY = _histogram(
    "llm_inference_latency_seconds",
    "LLM inference latency per tier — p50/p95/p99 via histogram_quantile",
    labelnames=["agent_type", "tier"],
    buckets=_LLM_BUCKETS,
)
LLM_TOKENS = _counter(
    "llm_tokens_generated_total",
    "Approximate tokens generated",
    labelnames=["agent_type"],
)

# ── STT / TTS ─────────────────────────────────────────────────────────────────
STT_LATENCY = _histogram(
    "stt_latency_seconds",
    "Whisper transcription latency",
    buckets=_LLM_BUCKETS,
)
TTS_LATENCY = _histogram(
    "tts_latency_seconds",
    "Coqui TTS synthesis latency",
    buckets=_LLM_BUCKETS,
)
VOICE_RTT = _histogram(
    "voice_round_trip_seconds",
    "Full voice pipeline RTT: upload → STT → LLM → TTS — critical p95/p99",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0],
)

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_RETRIEVAL_LATENCY = _histogram(
    "rag_retrieval_latency_seconds",
    "End-to-end RAG retrieval latency",
    labelnames=["collection", "strategy"],  # strategy: crag_correct|crag_corrected|self_rag_skipped
    buckets=_FAST_BUCKETS,
)
CRAG_STATUS = _counter(
    "crag_retrieval_status_total",
    "CRAG retrieval quality status",
    labelnames=["status"],  # correct|ambiguous|corrected|fallback|empty
)
RERANKER_LATENCY = _histogram(
    "reranker_latency_seconds",
    "Cross-encoder reranker latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
)

# ── Celery ────────────────────────────────────────────────────────────────────
CELERY_QUEUE_LENGTH = _gauge(
    "celery_queue_length",
    "Current Celery queue depth",
    labelnames=["queue"],
)
CELERY_TASK_DURATION = _histogram(
    "celery_task_duration_seconds",
    "Celery task execution time",
    labelnames=["task_name"],
    buckets=_LLM_BUCKETS,
)
CELERY_TASK_FAILURES = _counter(
    "celery_task_failures_total",
    "Celery task failures",
    labelnames=["task_name"],
)

# ── Circuit Breaker ───────────────────────────────────────────────────────────
CB_STATE_GAUGE = _gauge(
    "circuit_breaker_state",
    "CB state: 0=closed, 1=open, 2=half-open",
    labelnames=["service"],
)
CB_TRIPS_COUNTER = _counter(
    "circuit_breaker_trips_total",
    "Circuit breaker trips",
    labelnames=["service"],
)

# ── IRT / Learning ────────────────────────────────────────────────────────────
THETA_GAUGE = _gauge(
    "student_theta_average",
    "Average IRT theta across all active students",
)
QUIZ_ATTEMPTS = _counter(
    "quiz_attempts_total",
    "Quiz attempts",
    labelnames=["correct"],
)
DRIFT_SCORE_GAUGE = _gauge(
    "evidently_drift_score",
    "Latest Evidently AI drift score",
)
