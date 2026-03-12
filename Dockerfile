FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY qppg_service/ ./qppg_service/

# Install with server extras (no torch/QARA in base image)
RUN pip install --no-cache-dir ".[server]"

# Data directory for SQLite
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

ENV QPPG_DB=/data/chains.db
ENV QPPG_DOMAIN=default
ENV QPPG_HOST=0.0.0.0
ENV QPPG_PORT=8000

CMD llm-guard-kit serve \
    --domain "$QPPG_DOMAIN" \
    --host "$QPPG_HOST" \
    --port "$QPPG_PORT" \
    --db "$QPPG_DB" \
    --dashboard
