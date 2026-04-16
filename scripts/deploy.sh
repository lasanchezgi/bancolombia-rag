#!/bin/bash
set -e

# Cargar variables del .env
if [ ! -f .env ]; then
  echo "❌ Error: no se encontró el archivo .env en el directorio actual."
  echo "   Ejecuta este script desde la raíz del proyecto."
  exit 1
fi

source .env

# Verificar variables requeridas
missing=()
[ -z "$DOCKER_IMAGE" ] && missing+=("DOCKER_IMAGE")
[ -z "$EC2_HOST" ] && missing+=("EC2_HOST")
[ -z "$EC2_USER" ] && missing+=("EC2_USER")
[ -z "$EC2_KEY_PATH" ] && missing+=("EC2_KEY_PATH")

if [ ${#missing[@]} -gt 0 ]; then
  echo "❌ Error: faltan variables en .env:"
  for var in "${missing[@]}"; do
    echo "   - $var"
  done
  echo ""
  echo "   Agrega estas variables a tu .env (ver .env.example)"
  exit 1
fi

EC2_KEY_PATH="${EC2_KEY_PATH/#\~/$HOME}"

if [ ! -f "$EC2_KEY_PATH" ]; then
  echo "❌ Error: no se encontró el archivo .pem en: $EC2_KEY_PATH"
  exit 1
fi

echo "🔨 Construyendo imagen multi-arch..."
docker rmi "$DOCKER_IMAGE" 2>/dev/null || true
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t "$DOCKER_IMAGE" \
  --push .
echo "✅ Imagen subida a Docker Hub"

echo "🚀 Desplegando en EC2..."
ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=no \
  "$EC2_USER@$EC2_HOST" << 'REMOTE'
    set -e
    cd ~/bancolombia-rag

    echo "→ Limpiando imágenes antiguas..."
    docker system prune -a -f

    echo "→ Descargando nueva imagen..."
    docker-compose pull frontend

    echo "→ Levantando contenedores..."
    docker-compose up -d

    echo "→ Esperando que ChromaDB inicie..."
    sleep 20

    echo "→ Repoblando ChromaDB..."
    docker exec bancolombia-rag-frontend-1 \
      /app/.venv/bin/python scripts/run_pipeline.py

    echo "→ Arreglando permisos..."
    docker exec -u root bancolombia-rag-frontend-1 \
      bash -c "chown -R appuser:appgroup /app/data /app/.memory"

    echo "✅ Deploy completado en EC2"
REMOTE

echo ""
echo "🎉 Deploy exitoso!"
echo "   Chat:       http://$EC2_HOST:8501"
echo "   Monitoreo:  http://$EC2_HOST:8501/Monitoreo"
echo "   MCP SSE:    http://$EC2_HOST:8000/sse"
