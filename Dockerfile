FROM python:3.12-slim AS base

WORKDIR /app

# Instalar dependencias del sistema necesarias para httpx y beautifulsoup4
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv como gestor de dependencias
RUN pip install uv

# Copiar archivos de dependencias primero para aprovechar el cache de capas Docker
COPY pyproject.toml uv.lock ./

# Instalar dependencias de producción usando uv (frozen = reproducible)
RUN uv sync --frozen

# Copiar código fuente y scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Hacer que los imports de src funcionen desde cualquier punto de entrada
ENV PYTHONPATH=/app

# Comando por defecto; cada servicio lo sobreescribe en docker-compose.yml
CMD ["uv", "run", "python", "-m", "src.mcp_server.server"]
