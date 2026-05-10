# =============================================================
# Heart Disease MLOps Pipeline
# File    : Dockerfile
# Purpose : Build a production-ready container image for the
#           FastAPI model-serving application.
#
# Build   : docker build -t heart-disease-api:v1 .
# Run     : docker run -d -p 8000:8000 --name heart-api heart-disease-api:v1
# Test    : curl http://localhost:8000/health
# =============================================================

# ── Base image ────────────────────────────────────────────────
# python:3.9-slim keeps the image small (~125 MB base)
FROM python:3.9-slim

# ── Metadata labels ───────────────────────────────────────────
LABEL maintainer="BITS Pilani WILP Student"
LABEL description="Heart Disease Prediction API — MLOps Assignment"
LABEL version="1.0.0"

# ── Working directory inside the container ───────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────
# gcc is required to compile some Python packages (e.g. numpy wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If requirements.txt doesn't change, pip install is skipped on rebuild.
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────
COPY src/     ./src/
COPY models/  ./models/

# ── Environment variables ─────────────────────────────────────
ENV MODEL_PATH=/app/models/best_model.pkl
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Expose the FastAPI port ───────────────────────────────────
EXPOSE 8000

# ── Health-check (Docker will mark container unhealthy if fails)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start the application ─────────────────────────────────────
# --host 0.0.0.0  → listen on all interfaces (required inside Docker)
# --workers 1     → single worker is fine for this assignment
CMD ["uvicorn", "src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]