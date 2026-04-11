FROM python:3.12.10-slim-bookworm AS base

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install uv --no-cache-dir

WORKDIR /app

# Copiar archivos de dependencias primero (cache de Docker)
COPY pyproject.toml .
COPY uv.lock* .

# Instalar dependencias de producción (reproducible, sin dev)
RUN uv sync --frozen --no-dev

# Copiar código fuente
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copiar datos de evaluación estáticos (leídos por pages/evaluation.py)
COPY data/eval_results.json ./data/

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Crear directorios necesarios y usuario no-root para seguridad
RUN mkdir -p data/raw data/processed .chroma .memory \
    && addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app

USER appuser

# Comando por defecto — cada servicio lo sobreescribe en docker-compose.yml
CMD ["uv", "run", "streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
