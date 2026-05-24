# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — builder
# Compiles wheels for packages that need gcc/build-essential (bcrypt, numpy,
# fastembed, etc.). The compiled wheels are copied to the final stage; the
# build tools are left behind.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — runtime
# Starts from the same slim base. Installs only the OS libs the app needs at
# runtime (ffmpeg for audio, libsndfile1 for WAV), then copies in the
# pre-built wheels and the application source.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Runtime-only OS deps — no build-essential, no gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install wheels compiled in the builder stage (no internet, no compiler)
COPY --from=builder /build/wheels /tmp/wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels -r requirements.txt \
 && rm -rf /tmp/wheels

# Copy application source — this layer changes most often so it comes last
COPY alembic/       ./alembic/
COPY alembic.ini    .
COPY app/           ./app/

# Non-root user — created AFTER pip so site-packages are owned by root
# (read-only for the app user is intentional)
RUN useradd -m -u 1000 -s /sbin/nologin edumentor \
 && chown -R edumentor:edumentor /app
USER edumentor

EXPOSE 8000

# Single worker here; scale horizontally via K8s replicas
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]