# bancolombia-rag

Sistema RAG (Retrieval-Augmented Generation) end-to-end para responder preguntas
sobre la documentación pública de Bancolombia.

---

## Descripción

`bancolombia-rag` indexa páginas del sitio web de Bancolombia, las convierte en
embeddings vectoriales y las almacena en ChromaDB. Un agente conversacional
basado en OpenAI utiliza esos vectores (vía un servidor MCP) para responder
preguntas con contexto real y actualizado, sin alucinaciones.

Características principales:

- Scraping asíncrono con `httpx` + `BeautifulSoup4`
- Embeddings con `text-embedding-3-small` de OpenAI
- Vector store `ChromaDB` self-hosted en Docker
- Servidor MCP (`FastMCP`) que expone herramientas de búsqueda
- Agente conversacional con `openai` SDK nativo (sin LangChain)
- Interfaz de chat con `Streamlit`

---

## Arquitectura

```bash
Bancolombia Website
        │
   [src/scraper]        → httpx crawler + BeautifulSoup parser
        │                 guarda JSON en data/raw/
   [src/pipeline]       → limpieza de texto + chunking por ventana deslizante
        │                 guarda chunks en data/processed/
   [src/embeddings]     → OpenAI text-embedding-3-small (batch)
        │
   [src/vector_store]   → ChromaDB HttpClient (adaptador del puerto abstracto)
        │
   [src/mcp_server]     → FastMCP: search_documents, get_document_by_id
        │
   [src/agent]          → OpenAI gpt-4o-mini + ConversationMemory
        │
   [src/frontend]       → Streamlit chat UI
```

La capa `vector_store` expone una ABC (`VectorStoreRepository`) que desacopla
la lógica de negocio del backend concreto: cambiar de ChromaDB a Pinecone o
Qdrant solo requiere implementar un nuevo adaptador.

---

## Instalación

### Prerequisitos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — gestor de dependencias y entornos
- Docker + Docker Compose (para ChromaDB y los servicios containerizados)

### Setup local

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd bancolombia-rag

# Crear el entorno virtual e instalar dependencias
uv sync --dev

# Copiar variables de entorno y completar los valores
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY y demás configuraciones
```

---

## Variables de entorno

| Variable | Tipo | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | string | Clave de API de OpenAI |
| `CHROMA_HOST` | string | Host del servidor ChromaDB (default: `localhost`) |
| `CHROMA_PORT` | int | Puerto HTTP de ChromaDB (default: `8000`) |
| `MCP_SERVER_HOST` | string | Host del servidor MCP (default: `localhost`) |
| `MCP_SERVER_PORT` | int | Puerto del servidor MCP (default: `8080`) |
| `BANCOLOMBIA_BASE_URL` | string | URL base para el scraper |
| `MAX_PAGES` | int | Límite de páginas a crawlear |
| `CHUNK_SIZE` | int | Tamaño de cada chunk en caracteres |
| `CHUNK_OVERLAP` | int | Solapamiento entre chunks en caracteres |
| `TOP_K_RESULTS` | int | Documentos a recuperar por consulta |

---

## Cómo ejecutar

### Modo local (paso a paso)

```bash
# 1. Levantar ChromaDB
docker run -p 8000:8000 chromadb/chroma:latest

# 2. Ejecutar el scraper (guarda páginas en data/raw/)
uv run python scripts/run_scraper.py

# 3. Ejecutar el pipeline (limpia, chunking, embeddings → ChromaDB)
uv run python scripts/run_pipeline.py

# 4. Levantar el servidor MCP
uv run python -m src.mcp_server.server

# 5. Abrir la interfaz de chat
uv run streamlit run src/frontend/app.py
```

### Modo Docker (todos los servicios)

```bash
# Generar uv.lock si aún no existe
uv lock

# Construir y levantar todos los servicios
docker-compose up --build

# Acceder a la UI en http://localhost:8501
```

---

## Decisiones técnicas

| Decisión | Alternativa considerada | Razón |
|---|---|---|
| `httpx` para scraping | `requests` | Soporte async nativo, mejor para crawling concurrente |
| `FastMCP` para servidor MCP | SDK MCP oficial (Node.js) | No requiere runtime Node.js, nativo en Python |
| `ChromaDB` como vector store | Pinecone, Qdrant | Zero-infraestructura en dev local, embeddable |
| ABC `VectorStoreRepository` | Dependencia directa a ChromaDB | Permite cambiar backend sin tocar lógica de negocio |
| OpenAI SDK nativo | LangChain | Menos abstracción, control total del agentic loop |
| `uv` como gestor de dependencias | `pip` + `venv` | Resolución determinista, 10-100x más rápido |
| `ruff` para linting | `flake8` + `isort` | Un solo binario, misma cobertura de reglas |
| `line-length = 120` | 88 (PEP 8 estricto) | Los strings de system prompt y mensajes de error son inherentemente largos; forzar 88 produce roturas artificiales que dañan la legibilidad |

---

## Limitaciones conocidas

- El scraper no implementa backoff exponencial ante errores 429 (rate limiting).
- Los embeddings se generan de forma síncrona; llamadas batch planificadas para v2.
- La memoria conversacional del agente es in-memory: no persiste entre reinicios.
- ChromaDB en modo HTTP no soporta multi-tenancy en la versión actual.
- El scraper respeta `robots.txt` manualmente (no hay integración automática).
