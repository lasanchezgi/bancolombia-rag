FROM python:3.12.10-slim-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --no-cache-dir

WORKDIR /app

ENV UV_CACHE_DIR=/app/.uv-cache

COPY pyproject.toml .
COPY uv.lock* .

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/eval_results.json ./data/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p data/raw data/processed .chroma .memory .uv-cache \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser

CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
