FROM python:3.12-slim AS base

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install uv

WORKDIR /app

ENV UV_CACHE_DIR=/app/.uv-cache

# Copiar archivos de dependencias primero (cache de Docker)
COPY pyproject.toml .
COPY uv.lock* .

# Instalar dependencias de producción (reproducible, sin dev)
RUN uv sync --frozen --no-dev

# Copiar código fuente
COPY src/ ./src/
COPY scripts/ ./scripts/

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Crear directorios necesarios y usuario sin privilegios
RUN mkdir -p data/raw data/processed .chroma .memory .uv-cache \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

# Comando por defecto — cada servicio lo sobreescribe en docker-compose.yml
CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
