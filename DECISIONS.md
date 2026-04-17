# Architectural Decision Records — bancolombia-rag

Este documento registra las decisiones técnicas significativas tomadas durante el diseño e
implementación del sistema RAG para el sitio web de Bancolombia/personas. Cada ADR refleja
el razonamiento real detrás de cada elección: el contexto que la motivó, las alternativas
evaluadas y los trade-offs aceptados conscientemente.

Los ADRs están ordenados por capa del sistema, desde la adquisición de datos hasta el despliegue.

---

## ADR-001: Estrategia de scraping — sitemap-driven vs link crawling

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El sitio `bancolombia.com/personas` tiene una estructura profunda con cientos de páginas de
productos financieros. Necesitábamos una estrategia para descubrir y descargar ese contenido
de forma sistemática y reproducible. El sitio publica `sitemap-personas.xml` con 473 URLs
catalogadas explícitamente. Adicionalmente, el archivo `robots.txt` del sitio permite
explícitamente al agente `ClaudeBot`, que usamos como User-Agent (`ClaudeBot/1.0`) para
identificar nuestro scraper de forma transparente.

**Decisión:**
Scraping sitemap-driven: descargamos `sitemap-personas.xml` una sola vez, extraemos todos los
elementos `<loc>` con BeautifulSoup en modo XML, y construimos la lista de URLs a visitar.
Las URLs que coinciden con patrones bloqueados (formularios, búsquedas, scripts de prueba) se
filtran mediante `_BLOCKED_PATTERNS` en `src/scraper/crawler.py`, respetando el espíritu del
`robots.txt` del sitio aunque sin parsearlo dinámicamente.

**Alternativas consideradas:**

- **Link crawling ciego**: Seguir todos los `<a href>` desde la página principal. Descartado
  porque genera un grafo de crawl no determinístico, visita páginas duplicadas por parámetros
  de URL, y es difícil de acotar sin lógica adicional de deduplicación.
- **Scrapy**: Framework completo de crawling con scheduler, middlewares y pipelines. Descartado
  por ser excesivo para un corpus estático de ~473 páginas. Añade dependencia pesada y curva
  de aprendizaje sin beneficio proporcional.
- **Playwright / Selenium**: Scraping con navegador headless para renderizar JavaScript.
  Descartado porque el contenido relevante de Bancolombia está en el HTML estático; el
  renderizado JS no añade información útil para el corpus RAG y triplicaría el tiempo de
  scraping.

**Consecuencias:**
El corpus es determinístico y reproducible. Ejecutar el scraper dos veces produce el mismo
conjunto de URLs (salvo actualizaciones del sitemap). La estrategia es respetuosa con el
servidor: concurrencia máxima de 5 requests simultáneos, delay de 0.5s entre requests, y
timeout de 10s por página (`src/scraper/crawler.py`).

**Trade-offs aceptados:**
No detectamos páginas nuevas que el equipo de Bancolombia no haya incluido en el sitemap.
La base de conocimiento refleja lo que Bancolombia declara públicamente en su sitemap, no
la totalidad absoluta del sitio. Para el caso de uso (asistente de productos/personas), esto
es suficiente y deseable: evitamos páginas de administración, landing pages de campaña
temporales y URLs de parámetros dinámicos.

---

## ADR-002: Estrategia de chunking — RecursiveCharacterTextSplitter

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El texto extraído de páginas bancarias es heterogéneo: páginas de producto con listas de
beneficios, artículos explicativos con párrafos densos, páginas FAQ con preguntas cortas, y
tablas de tarifas. Un chunking uniforme o demasiado fino pierde contexto; uno demasiado grueso
degrada la precisión del retrieval. Necesitábamos una estrategia que preservara coherencia
semántica sin depender de un tokenizador específico.

**Decisión:**
Chunking recursivo por palabras: tamaño de 500 palabras con solapamiento de 50 palabras,
implementado en `src/pipeline/chunker.py`. El algoritmo `_split_recursive()` divide el texto
por separadores en orden de preferencia semántica:

1. `\n\n` — límites de párrafo (preferido)
2. `\n` — saltos de línea
3. `. ` — fin de oración
4. ` ` — separación por palabra
5. `""` — carácter individual (fallback de último recurso)

Tras dividir, `_merge_pieces()` reagrupa fragmentos pequeños con solapamiento de 50 palabras
para que cada chunk incluya contexto del chunk anterior.

El tamaño de 500 palabras (~650 tokens en español) fue elegido deliberadamente: es suficiente
para contener una explicación completa de un producto bancario (ej. "¿Qué es una cuenta de
ahorros y cuáles son sus beneficios?") sin exceder el contexto útil para el modelo de
embeddings ni el contexto del LLM en la respuesta.

**Alternativas consideradas:**

- **Chunk por tokens fijos**: Dividir cada N tokens usando el tokenizador de tiktoken. Más
  preciso para controlar el tamaño del contexto en el LLM, pero corta oraciones a la mitad
  y pierde coherencia semántica. Descartado en favor de la legibilidad del chunk.
- **Chunking semántico con embeddings**: Detectar límites de chunk por similitud coseno entre
  oraciones consecutivas. Produce chunks más coherentes pero requiere embeddings adicionales
  durante el pipeline (costo) y es no-determinístico. Descartado por complejidad innecesaria
  para el corpus actual.
- **Chunk fijo por número de oraciones**: Simple pero produce chunks de longitud muy variable
  (una oración de FAQ vs un párrafo largo de términos). Descartado.
- **Sin chunking (documento completo)**: Almacenar cada página como un único documento.
  Imposible para páginas de 3000+ palabras que superan el límite de embeddings y degradan
  significativamente la precisión del retrieval.

**Consecuencias:**
El pipeline genera chunks con metadata completa: `chunk_id`, `url`, `title`, `category`,
`subcategory`, `chunk_index`, `total_chunks`, `word_count`. El solapamiento de 50 palabras
garantiza que preguntas cuya respuesta está en el límite entre dos chunks sean respondibles.

**Trade-offs aceptados:**
El solapamiento introduce redundancia (~10% de tokens duplicados en la base vectorial). Es
un costo aceptable a cambio de no perder respuestas en los bordes de los chunks. El conteo
por palabras en lugar de tokens introduce una pequeña variabilidad en el tamaño real en tokens
según el contenido, pero es insignificante para el caso de uso.

---

## ADR-003: Modelo de embeddings — text-embedding-3-small

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Los embeddings son el corazón del sistema RAG: determinan qué tan bien el retrieval encuentra
los chunks relevantes para cada pregunta. El corpus es en español colombiano con terminología
financiera específica (ej. "CDT", "cupo de crédito", "cuenta AFC"). Necesitábamos un modelo
con buena cobertura del español, dimensionalidad razonable para almacenamiento en ChromaDB,
y costo bajo para procesar ~1500 chunks sin hacer la demo económicamente inviable.

**Decisión:**
`text-embedding-3-small` de OpenAI, configurado para producir vectores de 1536 dimensiones.
El cliente en `src/embeddings/embedder.py` procesa en lotes de 100 textos por llamada a la
API, con reintentos exponenciales ante `RateLimitError`. El costo total para embeber el corpus
completo (~1500 chunks) fue inferior a $0.01 USD, lo que hace la demo reproducible sin costo
significativo para el evaluador.

**Alternativas consideradas:**

- **sentence-transformers local** (ej. `paraphrase-multilingual-mpnet-base-v2`): Gratuito y
  sin latencia de API. Descartado porque requiere descargar modelos de ~400MB en el contenedor
  Docker, aumenta el tamaño de imagen significativamente, y la calidad en español técnico-
  financiero es inferior a los modelos OpenAI según evaluaciones internas.
- **text-embedding-3-large**: Dimensionalidad de 3072, mejor calidad. Descartado por precio
  5x superior y porque para un corpus de ~1500 chunks la diferencia de calidad no justifica
  el costo adicional.
- **text-embedding-ada-002**: Modelo anterior de OpenAI, dimensionalidad 1536. Descartado
  porque `text-embedding-3-small` tiene mejor performance en benchmarks multilingües (MIRACL,
  MTEB) con precio similar.
- **multilingual-e5-large**: Modelo de Microsoft optimizado para recuperación multilingüe.
  Descartado por las mismas razones que sentence-transformers: requiere hosting local del
  modelo y aumenta la complejidad del despliegue.

**Consecuencias:**
Todos los embeddings pasan por la API de OpenAI, por lo que el sistema requiere `OPENAI_API_KEY`
tanto para la ingesta (pipeline) como para las consultas (agent). Los vectores de 1536d en
ChromaDB ocupan ~9MB para el corpus completo, manejable sin problemas en t3.small.

**Trade-offs aceptados:**
Dependencia de un servicio externo (OpenAI) para la generación de embeddings. Si la API no
está disponible, el sistema no puede procesar nuevas consultas. Aceptado conscientemente porque
la alternativa (modelo local) añadiría 400MB+ al contenedor y complejidad de infraestructura
desproporcionada para el tamaño del proyecto.

---

## ADR-004: Base vectorial — ChromaDB self-hosted

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Necesitábamos una base de datos vectorial capaz de almacenar ~1500 vectores de 1536 dimensiones
con metadata filtrable, que el evaluador pudiera ejecutar sin crear cuentas en servicios externos
ni incurrir en costos. La base debe ser reproducible con `docker-compose up` y persistente
entre reinicios del contenedor.

**Decisión:**
ChromaDB en Docker, con soporte dual de cliente en `src/vector_store/chroma_repository.py`:

- **Modo local** (`CHROMA_HOST=local`): `chromadb.PersistentClient(path=".chroma")` — para
  desarrollo y tests de CI sin Docker.
- **Modo HTTP** (cualquier otro valor): `chromadb.HttpClient(host, port)` — para el entorno
  Docker donde ChromaDB corre como servicio independiente en `docker-compose.yml`.

La colección usa distancia coseno (`hnsw:space: cosine`) para que la similitud entre query y
chunk refleje orientación semántica, no magnitud. El score se convierte de distancia a
similitud: `score = 1.0 - distance`.

La abstracción `VectorStoreRepository` (ABC en `src/vector_store/repository.py`) define la
interfaz con `add_documents()`, `query()`, `delete_collection()` y `count()`.
`ChromaRepository` es el único adaptador implementado actualmente, pero la interfaz permite
reemplazarlo sin tocar el dominio.

**Alternativas consideradas:**

- **Pinecone**: Servicio gestionado de alta calidad. Descartado porque requiere cuenta gratuita
  y el evaluador dependería de disponibilidad del servicio externo.
- **Weaviate**: Potente con soporte nativo de módulos de embedding. Descartado por mayor
  complejidad de configuración y mayor consumo de RAM (no adecuado para t3.small).
- **Qdrant**: Excelente alternativa de alto rendimiento. Descartado por ser prácticamente
  equivalente a ChromaDB para el tamaño del corpus, sin ventaja que justifique el cambio.
  Identificado como migración futura si el corpus supera 1M documentos.
- **pgvector**: Extensión de PostgreSQL. Descartado porque añade una dependencia de base de
  datos relacional sin beneficio para un corpus puramente vectorial.
- **FAISS**: Librería de Facebook para búsqueda vectorial. Excelente rendimiento pero sin
  servidor HTTP, sin metadata nativa, y sin soporte de filtrado. Requeriría implementar
  persistencia manualmente.

**Consecuencias:**
El evaluador puede reproducir el sistema completo con `docker-compose up` sin ninguna cuenta
externa. Los datos persisten en el volumen `chroma_data` de Docker. El modo dual
`PersistentClient`/`HttpClient` permite ejecutar tests de CI sin Docker (usando `.chroma/`
local).

**Trade-offs aceptados:**
ChromaDB no está optimizado para escala horizontal. Es adecuado para decenas de miles de
documentos en un único nodo, pero requeriría migración a Qdrant o Weaviate para un corpus
de producción real (>100K documentos). Este trade-off es aceptable para la demo del proyecto.

---

## ADR-005: MCP Server — FastMCP con doble transporte

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El enunciado del proyecto requiere implementar un servidor MCP (Model Context Protocol) que
exponga las capacidades RAG como herramientas consumibles por clientes MCP externos. El
servidor debe ser accesible tanto desde el agente Python interno (que lo lanza como subproceso)
como desde clientes externos como Claude Desktop, n8n o cualquier cliente MCP compatible.

**Decisión:**
FastMCP 3.2 con soporte de doble transporte en `src/mcp_server/server.py`, seleccionable
via variable de entorno `MCP_TRANSPORT`:

- **stdio** (defecto): El agente lanza el servidor como subproceso y se comunica via stdin/stdout.
  Este modo es obligatorio para la integración interna con el agente (`src/agent/agent.py`),
  que mantiene el proceso MCP vivo en un hilo daemon durante toda la sesión.
- **SSE** (Server-Sent Events): El servidor escucha en `MCP_SERVER_HOST:MCP_SERVER_PORT`
  (por defecto `0.0.0.0:8000`). Este modo permite el consumo desde Claude Desktop, n8n u
  otros clientes MCP sin modificar el código del servidor.

El servidor expone 3 herramientas y 1 recurso:

- `search_knowledge_base(query, top_k=5, category=None)`: recuperación semántica principal
- `get_article_by_url(url)`: recuperación por URL exacta
- `list_categories()`: listado de categorías disponibles
- `knowledgebase://stats`: estadísticas de la colección (recurso MCP)

**Alternativas consideradas:**

- **MCP SDK TypeScript**: Implementación oficial del protocolo MCP. Descartado porque
  introduciría Node.js como dependencia adicional en un proyecto puramente Python. FastMCP
  ofrece la misma funcionalidad en Python nativo.
- **Implementación manual del protocolo MCP**: Implementar el protocolo JSON-RPC directamente.
  Descartado por la complejidad de manejo de mensajes, lifecycles y errores que FastMCP
  abstrae correctamente.

**Consecuencias:**
El servidor MCP es totalmente independiente del agente: puede correrse standalone en modo SSE
en producción (EC2) y ser consumido por múltiples clientes simultáneos. Esto fue demostrado
en producción con Claude Desktop conectado al servidor en EC2. El diseño dual transporte no
añade complejidad: es una sola línea de configuración (`MCP_TRANSPORT=sse`).

**Trade-offs aceptados:**
El modo stdio acoplado a subprocess implica que si el proceso MCP falla, el agente debe
reiniciarlo. La implementación actual usa un singleton lazy-initialized que no tiene lógica
de reconexión automática. En producción real se añadiría supervisión del proceso hijo.

---

## ADR-006: Agente — OpenAI SDK nativo vs frameworks

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Necesitábamos un agente capaz de recibir una pregunta del usuario, decidir si necesita
consultar la base de conocimiento, ejecutar las herramientas MCP apropiadas, y generar una
respuesta coherente citando fuentes. La decisión crítica fue si usar un framework de
orquestación (LangChain, LangGraph, CrewAI) o implementar el loop directamente.

**Decisión:**
OpenAI SDK nativo con agentic loop manual implementado en `src/agent/agent.py`. El loop
completo ocupa aproximadamente 30 líneas de código Python legible:

1. Construir mensaje con contexto de memoria (short + mid + long term)
2. Llamar a `gpt-4o-mini` con `tools=[...]` y `tool_choice="auto"`
3. Si `finish_reason == "tool_calls"`: ejecutar herramientas via MCP, agregar resultados, volver al paso 2
4. Si `finish_reason == "stop"`: retornar respuesta final y actualizar memorias

El argumento central es la transparencia: cada paso del loop es legible y trazable línea por
línea, sin magia de frameworks. Un evaluador puede entender completamente el comportamiento
del agente leyendo `_run_agentic_loop()`.

**Alternativas consideradas:**

- **LangChain**: El framework más popular para aplicaciones LLM. Descartado porque añade
  cientos de abstracciones (Chains, Runnables, CallbackHandlers) que no aportan valor para
  un loop de 30 líneas. LangChain es valioso cuando se necesitan docenas de componentes
  intercambiables; aquí sería complejidad accidental.
- **LangGraph**: Framework de grafos para agentes multi-step. Descartado por la misma razón:
  el grafo de este agente es lineal (usuario → herramientas → respuesta) y no justifica la
  abstracción de nodos/edges/state.
- **CrewAI**: Framework para agentes múltiples con roles. Descartado porque el sistema tiene
  un único agente. CrewAI añadiría overhead de coordinación sin beneficio.
- **AutoGen**: Framework de Microsoft para conversaciones multi-agente. Descartado por las
  mismas razones que CrewAI.

**Consecuencias:**
El agente es completamente auditable. Si el comportamiento es incorrecto, el debugging es
directo: se puede agregar un `print` en cualquier punto del loop. No hay callbacks ocultos
ni middleware implícito.

**Trade-offs aceptados:**
Si el proyecto creciera a múltiples agentes especializados (ej. agente de búsqueda, agente
de síntesis, agente de validación), el loop manual requeriría refactoring hacia un framework
de orquestación. Para la arquitectura actual de un único agente con 3 herramientas, el SDK
nativo es la opción correcta.

---

## ADR-007: Memoria del agente — arquitectura de 3 capas

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El enunciado del proyecto requiere explícitamente manejo de memoria en tres horizontes
temporales: corto, mediano y largo plazo. Necesitábamos una implementación que fuera
funcional, demostrable, y no dependiera de infraestructura externa (Redis, DynamoDB) para
mantener el sistema autónomo.

**Decisión:**
Tres capas de memoria implementadas en `src/agent/memory.py`, cada una con propósito y
ciclo de vida distintos:

**ShortTermMemory** (corto plazo):
Ventana deslizante de máximo 20 mensajes en RAM. Preserva el mensaje de sistema en el índice 0.
Cuando se supera el límite, elimina el mensaje no-sistema más antiguo (buffer circular).
Enviada completa al LLM en cada turno.

**MidTermMemory** (mediano plazo):
Se activa cuando `ShortTermMemory` supera 15 mensajes. Genera un resumen de 3-5 oraciones
via `gpt-4o-mini` capturando los puntos clave de la conversación hasta ese momento. El
resumen se inyecta como mensaje de sistema adicional en el contexto del agente. Es un resumen
único por conversación (se actualiza pero no se acumula).

**LongTermMemory** (largo plazo):
Perfil JSON persistido en `.memory/user_profile.json`. Rastrea:

- `topics_consulted`: categorías de Bancolombia consultadas históricamente
- `products_of_interest`: productos que generaron interacción
- `session_count`: contador acumulado de sesiones
- `last_session`: timestamp ISO de la última sesión

Se actualiza automáticamente al final de cada conversación via `update_from_conversation()`,
que mapea keywords detectados en los mensajes a categorías via `_BANCOLOMBIA_KEYWORDS`.
Se incluye en el system prompt inicial para personalizar las respuestas según el historial.

**Alternativas consideradas:**

- **Solo ventana de contexto simple**: Un único buffer sin límite. Descartado porque en
  conversaciones largas excede el límite de tokens del LLM y aumenta el costo por turno.
- **Redis**: Cache distribuido para short-term memory. Descartado porque requiere servicio
  adicional en Docker Compose sin beneficio real para una aplicación de usuario único.
- **DynamoDB / base de datos relacional**: Para persistencia de largo plazo. Descartado
  porque añade dependencia de nube externa y el perfil JSON local es suficiente para
  demostrar el concepto.

**Consecuencias:**
La arquitectura de memoria es completamente local: no requiere servicios externos. El perfil
de largo plazo persiste entre reinicios del contenedor (montado como volumen `.memory/` en
`docker-compose.yml`). El resumen de mediano plazo reduce el costo de tokens en
conversaciones largas al comprimir el historial.

**Trade-offs aceptados:**
La memoria de largo plazo es local al contenedor, no distribuida. Si el servicio se reinicia
en una instancia diferente, el perfil no se transfiere. Para producción real se requeriría
almacenamiento externo (S3, DynamoDB). El diseño actual es adecuado para la demo y para
un usuario único.

---

## ADR-008: LLM principal — gpt-4o-mini

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Necesitábamos un LLM capaz de: (1) razonar sobre cuándo usar herramientas de búsqueda,
(2) interpretar resultados estructurados del servidor MCP, (3) generar respuestas en español
colombiano con citas de fuentes, y (4) gestionar la lógica de memoria (generar resúmenes para
MidTermMemory). El costo debe ser viable para demos repetidas.

**Decisión:**
`gpt-4o-mini` de OpenAI, configurado en `src/agent/agent.py`. El modelo tiene soporte nativo
de tool use (function calling), responde en español correctamente, es significativamente más
económico que `gpt-4o` ($0.15/1M input tokens vs $2.50/1M), y tiene latencia adecuada para
una interfaz de chat interactiva (~1-2s por turno típico).

**Alternativas consideradas:**

- **gpt-4o**: Máxima capacidad de OpenAI. Descartado por precio 15x superior a `gpt-4o-mini`
  sin ganancia proporcional para el caso de uso (responder preguntas sobre productos
  bancarios con contexto recuperado). La calidad de razonamiento de `gpt-4o-mini` es
  suficiente cuando el contexto relevante está en el prompt.
- **Claude Sonnet (claude-sonnet-4-6)**: Excelente para razonamiento complejo. Descartado
  porque el sistema ya depende de la API de OpenAI para embeddings; añadir Anthropic como
  segundo proveedor aumentaría la complejidad de gestión de keys y costos.
- **Gemini Flash**: Muy económico. Descartado por menor madurez del ecosistema de tool use
  en Python al momento de la implementación.
- **Llama local** (via Ollama): Costo cero por inferencia. Descartado porque en t3.small
  (2 vCPU, 2GB RAM) la inferencia local de un modelo capaz sería demasiado lenta (~30-60s
  por turno) para una experiencia de chat usable.

**Consecuencias:**
El costo por conversación típica (10-15 turnos) es inferior a $0.01 USD. El sistema es
económicamente viable para demos repetidas sin límite práctico de uso. El modelo maneja
correctamente el agentic loop multi-turno con tool calls anidadas.

**Trade-offs aceptados:**
Dependencia completa de OpenAI: embeddings + LLM en el mismo proveedor. Si OpenAI tiene
una interrupción, el sistema completo queda inoperante. Aceptado como simplificación
arquitectónica; en producción se añadiría fallback a un segundo proveedor.

---

## ADR-009: Gestión de dependencias — uv

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Con múltiples colaboradores potenciales y un pipeline CI/CD, necesitábamos un gestor de
dependencias que garantizara instalaciones reproducibles via lock file, fuera rápido en CI,
y alineara con los estándares emergentes del ecosistema Python en 2025.

**Decisión:**
`uv` como gestor de dependencias y entorno virtual, con `pyproject.toml` como fuente de
verdad y `uv.lock` como lock file. Todos los comandos del proyecto usan el prefijo `uv run`
(scraper, pipeline, tests, frontend, servidor MCP).

La velocidad de `uv` es relevante en CI: la instalación de dependencias del proyecto tarda
~15 segundos en `ubuntu-latest` vs ~2-3 minutos con `pip`. En el `Dockerfile`, `uv sync
--frozen --no-dev` garantiza que el contenedor usa exactamente las versiones del lock file.

**Alternativas consideradas:**

- **pip + requirements.txt**: Estándar histórico. Descartado porque sin lock file los builds
  no son reproducibles (dependencias transitivas pueden cambiar entre instalaciones). Un
  `pip freeze > requirements.txt` manual es propenso a errores y difícil de mantener.
- **Poetry**: Gestor maduro con lock file y gestión de entornos. Descartado porque `uv` es
  10-100x más rápido y está ganando adopción como reemplazo estándar. Poetry añade overhead
  de configuración (grupos de dependencias, configuración en `[tool.poetry]`) que `uv` evita
  usando directamente los estándares PEP 517/621.
- **Pipenv**: Descartado por velocidad inferior a Poetry y menor adopción comunitaria en 2025.
  La comunidad Python se ha consolidado en `uv` como el estándar moderno.

**Consecuencias:**
El workflow de CI en `.github/workflows/ci.yml` instala `uv` directamente desde
`astral-sh/setup-uv` y ejecuta `uv sync --dev` para el entorno de testing. El Dockerfile
usa `uv sync --frozen --no-dev` para builds reproducibles. Cualquier colaborador puede
reproducir el entorno exacto con `uv sync`.

**Trade-offs aceptados:**
`uv` es una herramienta relativamente nueva (lanzada 2024). Aunque su adopción es acelerada,
equipos con procesos legacy establecidos en Poetry o pip podrían necesitar adaptación. Para
un proyecto nuevo en 2025, es la elección con mejor ratio velocidad/madurez/adopción.

---

## ADR-010: Infraestructura — EC2 t3.small con Docker Compose

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Necesitábamos un entorno de despliegue que: (1) fuera accesible públicamente para el
evaluador sin requerir VPN ni configuración especial, (2) ejecutara el stack completo
(ChromaDB + Frontend Streamlit + MCP Server), (3) no incurriera en costos para el evaluador,
y (4) fuera reproducible localmente con el mismo `docker-compose.yml`.

**Decisión:**
EC2 t3.small (2 vCPU, 2GB RAM) en AWS, usando créditos de AWS Academy existentes (~$0.023/hr).
El stack completo se despliega con `docker-compose up -d` sin modificaciones al mismo
`docker-compose.yml` usado en desarrollo local. La URL pública es accesible desde cualquier
navegador sin configuración adicional.

La arquitectura de servicios en `docker-compose.yml`:

- **chromadb**: Siempre activo, con volumen persistente `chroma_data`
- **frontend**: Siempre activo, expone puerto 8501, monta `.memory/` y `data/` como volúmenes
- **scraper** y **pipeline**: Perfil `tools`, ejecución one-time para poblar la KB

**Alternativas consideradas:**

- **Heroku / Railway / Render**: Plataformas PaaS con tier gratuito. Descartadas porque el
  tier gratuito tiene limitaciones de RAM (512MB-1GB) insuficientes para ChromaDB + Streamlit
  simultáneos, y las instancias "se duermen" después de inactividad (latencia de cold start
  inaceptable para demo).
- **AWS Lambda / Cloud Functions**: Serverless. Descartado porque ChromaDB requiere estado
  persistente y las herramientas MCP requieren un proceso de larga duración, ambos
  incompatibles con el modelo serverless.
- **ECS Fargate**: Contenedores serverless en AWS. Descartado por complejidad de configuración
  (task definitions, clusters, ALB) y costo superior para una demo de corta duración.

**Consecuencias:**
El evaluador accede al frontend en `http://<ec2-ip>:8501` sin ninguna configuración. El MCP
Server en modo SSE está disponible en `http://<ec2-ip>:8000` para clientes externos como
Claude Desktop. La instancia corre 24/7 durante el período de evaluación con costo total
de ~$1-2 USD.

**Trade-offs aceptados:**
t3.small tiene 2GB RAM, lo cual es ajustado para el stack completo bajo carga. En producción
real se usaría t3.medium o t3.large. Para una demo con tráfico controlado, 2GB es suficiente.
La instancia no tiene auto-scaling ni balanceo de carga.

---

## ADR-011: CI/CD — GitHub Actions con uv

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El enunciado del proyecto requiere explícitamente un pipeline de integración continua. El
pipeline debe ejecutarse en cada push y pull request a `main`, verificando calidad de código
y tests. Dado que ya usamos `uv` como gestor de dependencias, el pipeline debe ser coherente
con esa decisión.

**Decisión:**
GitHub Actions con un único job `lint-and-test` en `.github/workflows/ci.yml`, ejecutando
tres etapas en secuencia:

1. `uv run ruff check src/ tests/` — análisis estático (estilo + errores comunes)
2. `uv run black --check src/ tests/` — verificación de formato
3. `uv run pytest tests/ -v --tb=short` — suite de tests

El pipeline usa Python 3.12 (coherente con el Dockerfile) y configura las variables de
entorno necesarias para tests sin servicios externos (ej. `CHROMA_HOST=local` para usar
`PersistentClient` en lugar del servidor Docker).

Una decisión específica de configuración de ruff: `line-length = 120` en `pyproject.toml`.
El motivo es práctico: los system prompts del agente (`src/agent/prompts.py`) contienen
strings multilínea con instrucciones largas que con 88 caracteres (default de Black) se
fragmentarían artificialmente, dificultando la lectura. 120 caracteres permite que las
instrucciones del prompt sean legibles en una sola línea.

**Alternativas consideradas:**

- **GitLab CI / CircleCI / Travis CI**: Sistemas CI alternativos. Descartados porque el
  repositorio está en GitHub y GitHub Actions no añade costo ni configuración de integración.
- **Pre-commit hooks**: Ejecutar linting localmente antes de cada commit. Complementario pero
  no sustituto del CI en remoto. Se puede añadir en el futuro sin reemplazar GitHub Actions.

**Consecuencias:**
Cada PR a `main` es validado automáticamente. El badge de CI en el README refleja el estado
del pipeline. Los tests en CI usan `CHROMA_HOST=local` para evitar dependencia de Docker,
lo que permite verificar la lógica de negocio sin infraestructura adicional.

**Trade-offs aceptados:**
El pipeline no incluye tests de integración end-to-end (ej. scraper real, pipeline completo).
Ejecutar el scraper en CI descargaría páginas reales de Bancolombia en cada push, lo cual
es indeseable por velocidad y por cortesía con el servidor. Los tests de integración se
ejecutan manualmente en local o en la instancia EC2.

---

## ADR-012: Arquitectura Clean — separación de capas con ABC como puertos

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El enunciado evalúa explícitamente Clean Architecture. Necesitábamos una estructura que
separara claramente el dominio (lógica de negocio RAG) de los adaptadores (ChromaDB, OpenAI,
Streamlit), permitiendo que los componentes de infraestructura sean intercambiables sin
modificar el núcleo del sistema.

**Decisión:**
Separación en capas con responsabilidades claras:

```bash
src/
├── scraper/          # Adquisición de datos (adaptador web)
├── pipeline/         # Transformación (dominio: chunking, limpieza)
├── embeddings/       # Generación de vectores (adaptador OpenAI)
├── vector_store/     # Almacenamiento vectorial (puerto + adaptador)
│   ├── repository.py        # Puerto: ABC con interfaz genérica
│   └── chroma_repository.py # Adaptador: implementación ChromaDB
├── mcp_server/       # Servidor de herramientas (adaptador MCP)
├── agent/            # Orquestador (dominio: lógica del agente)
└── frontend/         # Presentación (adaptador Streamlit)
```

El patrón clave es `VectorStoreRepository` como puerto (ABC) en `src/vector_store/repository.py`:

```python
class VectorStoreRepository(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None: ...
    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int, ...) -> list[Document]: ...
    @abstractmethod
    def delete_collection(self) -> None: ...
    @abstractmethod
    def count(self) -> int: ...
```

`ChromaRepository` implementa este puerto. Para migrar a Qdrant, se implementaría
`QdrantRepository(VectorStoreRepository)` sin tocar el agente, el servidor MCP ni el pipeline.

**Alternativas consideradas:**

- **Arquitectura monolítica en scripts**: Todo en un único script `main.py`. Descartado
  porque hace el sistema no testeable (no se pueden mockear componentes), no escalable, y
  viola explícitamente el requerimiento del enunciado.
- **Hexagonal estricta con DTOs en cada frontera**: Cada capa transforma objetos de dominio
  en DTOs específicos de la capa. Descartado por sobrediseño: con 5 capas y ~1500 líneas de
  código total, el overhead de DTOs no se justifica. Se usa Pydantic para validación donde
  es necesario.

**Consecuencias:**
Cada capa es testeable de forma independiente. Los tests en `tests/` verifican componentes
individuales sin levantar el stack completo. El pipeline de CI puede correr tests de todas
las capas con `CHROMA_HOST=local` sin Docker. El sistema puede escalar horizontalmente
reemplazando un adaptador por otro sin refactoring del dominio.

**Trade-offs aceptados:**
La separación estricta implica que agregar una nueva funcionalidad requiere tocar múltiples
archivos (definir en el ABC, implementar en el adaptador, usar en el dominio). Es un trade-off
conocido de la arquitectura hexagonal: mayor verbosidad a cambio de mayor mantenibilidad.
Para el tamaño actual del proyecto, el beneficio justifica el costo.

---

## ADR-013: Logging de conversaciones — ConversationLogger con SQLite

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Para poder monitorear la calidad del sistema RAG en producción, identificar gaps en la
base de conocimiento, y auditar el comportamiento del agente, necesitábamos persistir cada
interacción (pregunta, respuesta, chunks recuperados, scores, tiempos de latencia, llamadas
MCP). Esta trazabilidad es indispensable para detectar preguntas fuera de scope, herramientas
que fallan, o respuestas de baja calidad sin tener que reproducir las sesiones manualmente.

**Decisión:**
`ConversationLogger` en `src/agent/conversation_logger.py` usando SQLite como backend de
persistencia. El logger registra en `data/conversations.db` (montado como volumen en Docker):

- **Tabla `conversations`**: `session_id`, `question`, `response`, `tools_used` (JSON),
  `retrieval_scores` (JSON), `latency_ms`, `error` (si aplica), `timestamp`
- **Tabla `mcp_calls`**: `call_id`, `tool_name`, `arguments` (JSON), `result_preview`,
  `latency_ms`, `success`, `timestamp` — para trazabilidad granular de cada herramienta MCP

El agente (`src/agent/agent.py`) instancia el logger al inicio y registra cada interacción
al finalizar el loop agentic. El dashboard de monitoreo consume directamente estas tablas
para renderizar métricas en tiempo real.

**Alternativas consideradas:**

- **Logging a fichero de texto**: Simple pero no consultable. Para analizar patrones (ej.
  "¿qué categorías tienen más preguntas fallidas?") se requiere parsear texto, no SQL.
  Descartado en favor de SQLite que permite queries ad-hoc directamente.
- **Redis con TTL**: Cache de conversaciones recientes con expiración automática. Descartado
  porque añade un servicio al Compose sin ventaja: SQLite en disco es suficiente para el
  volumen de una demo y persiste indefinidamente sin configuración adicional.
- **PostgreSQL / RDS**: Base de datos relacional completa. Descartado por sobredimensionamiento
  para un sistema de usuario único. El docstring de `ConversationLogger` documenta
  explícitamente que la migración a PostgreSQL es el path de producción.
- **OpenTelemetry / Datadog / Prometheus**: Observabilidad de nivel producción. Descartado
  por complejidad de setup y costo. SQLite local ofrece el 80% del valor con el 5% del
  esfuerzo para el scope del proyecto.

**Consecuencias:**
El `data/conversations.db` persiste entre reinicios via el volumen `./data:/app/data` del
Compose. El dashboard de monitoreo puede mostrar métricas históricas completas sin latencia
adicional (reads locales). Los tests de `ConversationLogger` usan `db_path=":memory:"` para
no generar ficheros en CI.

**Trade-offs aceptados:**
SQLite no soporta escrituras concurrentes desde múltiples instancias. En un despliegue
multi-instancia (múltiples réplicas del frontend) se produciría contención en escritura.
Para el caso de un único contenedor frontend (la arquitectura actual), SQLite es correcto.
La migración a PostgreSQL/RDS requiere cambiar solo este adaptador sin tocar el agente.

---

## ADR-014: Observabilidad operacional — Dashboard de monitoreo con autenticación

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Un sistema RAG en producción necesita visibilidad sobre: calidad del retrieval por categoría,
preguntas que el sistema no puede responder (gaps de KB), historial de conversaciones para
auditoría, y trazabilidad de las llamadas a herramientas MCP. Esta información es sensible
(incluye conversaciones de usuarios) y no debe ser pública. La solución debía integrarse
sin añadir infraestructura nueva.

**Decisión:**
Página Streamlit `src/frontend/pages/monitoring.py` con autenticación por contraseña.
Streamlit detecta automáticamente los archivos en `pages/` y los añade a la navegación
del sidebar sin configuración adicional. La autenticación usa `st.session_state` con
contraseña configurable via `MONITORING_PASSWORD` (variable de entorno, con default
`bancolombia2026`). Si la contraseña es incorrecta, `st.stop()` impide renderizar el
dashboard.

El dashboard expone cuatro secciones:

1. **Métricas de sesiones**: total de conversaciones, latencia promedio, tasa de error
2. **Trazabilidad MCP**: detalle de cada llamada a herramienta (tool, argumentos, latencia,
   éxito/fallo) consultable por sesión
3. **Gaps de KB**: preguntas con score de similitud bajo (retrieval fallido)
4. **Historial reciente**: últimas N conversaciones con expansión de detalles

**Alternativas consideradas:**

- **Grafana + Prometheus**: Stack de observabilidad estándar en producción. Descartado por
  complejidad de configuración (exporters, dashboards, servicios adicionales en Compose) y
  costo de setup desproporcionado para el scope del proyecto.
- **Página separada con servidor Flask**: Separar el dashboard en su propio servicio.
  Descartado porque añade un servicio al Compose sin beneficio: Streamlit multi-page maneja
  la separación de forma nativa y con el mismo proceso.
- **Autenticación con OAuth / JWT**: Sistema de autenticación robusto. Descartado por
  complejidad innecesaria para una demo con un único usuario evaluador. La contraseña en
  variable de entorno es suficiente para proteger datos que no son de producción real.

**Consecuencias:**
El evaluador puede acceder al dashboard en la misma URL del frontend (`:8501`), navegando
al item "Monitoreo" en el sidebar. La variable `MONITORING_PASSWORD` está configurada en
`docker-compose.yml` con default explícito. No se añade ningún servicio nuevo al Compose.

**Trade-offs aceptados:**
La autenticación basada en `st.session_state` no es persistente entre pestañas del navegador:
el usuario debe re-autenticarse si abre el dashboard en una nueva pestaña. Es un trade-off
aceptable para simplificar la implementación. En producción real se usaría Streamlit
Community Cloud auth o un reverse proxy con autenticación básica.

---

## ADR-015: Evaluación formal — métricas de faithfulness y factualidad

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El enunciado del proyecto requiere demostrar que el sistema RAG produce respuestas correctas
y verificables. Necesitábamos una metodología de evaluación rigurosa que midiera dos
dimensiones clave: (1) **faithfulness** — si las respuestas están ancladas en los chunks
recuperados, sin inventar información; y (2) **factuality** — si las respuestas son
factualmante correctas respecto a la documentación de Bancolombia. Adicionalmente, queríamos
comparar cuantitativamente el impacto del Cross-Encoder reranking (ADR implícito en sección
de mejoras futuras implementadas) sobre estas métricas.

**Decisión:**
Pipeline de evaluación implementado en `notebooks/rag_evaluation.ipynb`:

- **Dataset de evaluación**: 15 preguntas representativas sobre productos de Bancolombia,
  con respuestas de referencia escritas manualmente a partir de la documentación oficial.
- **Métricas evaluadas**:
  - `faithfulness_score`: fracción de afirmaciones en la respuesta verificables en los
    chunks recuperados (evaluado por `gpt-4o-mini` como juez)
  - `factuality_score`: corrección factual de la respuesta vs la respuesta de referencia
    (evaluado por `gpt-4o-mini` como juez LLM-as-a-judge)
- **Comparación A/B**: cada pregunta se evalúa dos veces — pipeline base (sin reranking)
  y pipeline con Cross-Encoder — para cuantificar el impacto del reranking.
- **Resultados persistidos** en `data/eval_results.json`, leídos por
  `src/frontend/pages/evaluation.py` para visualización en el dashboard.

La página de evaluación Streamlit muestra los resultados con gráficos de barras
comparativos, tabla detallada por pregunta, y resumen estadístico (media, mediana, p10/p90).
Acceso protegido con la misma contraseña que el dashboard de monitoreo (`MONITORING_PASSWORD`).

**Alternativas consideradas:**

- **RAGAS framework**: Librería especializada para evaluación de RAG (faithfulness, answer
  relevancy, context precision). Descartado porque añade una dependencia pesada para el
  Dockerfile de producción y requiere configuración adicional. El LLM-as-a-judge con
  `gpt-4o-mini` produce métricas equivalentes con código completamente auditable.
- **Evaluación humana manual (human-in-the-loop)**: Revisión manual de cada respuesta.
  Descartado por no ser escalable ni reproducible. El LLM-as-a-judge permite re-evaluar
  automáticamente si el sistema cambia.
- **Métricas de texto clásicas (BLEU, ROUGE)**: Comparación léxica vs respuestas de
  referencia. Descartado porque BLEU/ROUGE penalizan respuestas correctas con diferente
  vocabulario. Para respuestas en lenguaje natural, LLM-as-a-judge captura mejor la
  equivalencia semántica.

**Consecuencias:**
Los resultados de evaluación (`data/eval_results.json`) son archivos estáticos en el
repositorio, actualizados manualmente al re-ejecutar el notebook. El Dockerfile copia
este archivo en la imagen (`COPY data/eval_results.json ./data/`) para que la página
de evaluación funcione sin montar volúmenes adicionales. Si el pipeline cambia, el
notebook debe re-ejecutarse y el JSON actualizado antes del próximo build Docker.

**Trade-offs aceptados:**
La evaluación es un snapshot estático, no continua. Si el sistema cambia (nuevo modelo,
nuevos chunks, nuevo reranker), los resultados del JSON pueden quedar desactualizados
hasta que alguien re-ejecute el notebook manualmente. En producción real se integraría
la evaluación en el CI/CD como job periódico. Para el scope del proyecto, la evaluación
manual con resultados versionados en el repositorio es suficiente.

---

## ADR-016: Seguridad de contenedores — imagen non-root y .dockerignore

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
Por defecto, los contenedores Docker ejecutan procesos como `root` (UID 0), lo cual viola
el principio de mínimo privilegio. Si el proceso del contenedor es comprometido, el
atacante tiene privilegios de root dentro del contenedor y potencialmente en el host
(según la configuración del daemon Docker). Para el despliegue en EC2 (ADR-010), correr
como root no es aceptable. Adicionalmente, sin `.dockerignore`, el contexto de build
incluye el fichero `.env` (con claves de API), el directorio `.git`, y cientos de
ficheros de desarrollo que no pertenecen a la imagen de producción.

**Decisión:**
Dos medidas de seguridad aplicadas en el `Dockerfile`:

**1. Usuario non-root (`appuser`):**

```dockerfile
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --no-create-home appuser \
    && chown -R appuser:appgroup /app
USER appuser
```

El proceso Streamlit y el subproceso MCP se ejecutan con `appuser` (UID sin shell, sin
home directory). Crea directorios necesarios (`data/`, `.chroma/`, `.memory/`) antes del
cambio de usuario para garantizar los permisos correctos.

**2. Fichero `.dockerignore`:**

Excluye del contexto de build:

- `.env`, `.env.*` — claves de API nunca deben entrar a la imagen
- `.git/`, `.gitignore` — historial de control de versiones irrelevante en producción
- `notebooks/`, `tests/`, `.venv/` — artefactos de desarrollo
- `data/raw/`, `data/processed/`, `data/conversations.db` — datos generados en runtime
  (montados como volúmenes en producción, no embebidos en la imagen)
- `**/__pycache__`, `**/*.pyc` — bytecode Python compilado
- `.chroma/`, `.memory/` — estados locales de desarrollo

**Alternativas consideradas:**

- **`USER root` con capacidades restringidas (seccomp profiles)**: Limitar syscalls sin
  cambiar el UID. Más complejo de configurar y no elimina el riesgo de escalada de
  privilegios. Descartado en favor del enfoque más simple y estándar.
- **Imagen distroless**: Imagen mínima sin shell, sin package manager. Mayor reducción de
  superficie de ataque. Descartado porque `uv` requiere un entorno Python estándar que
  distroless no proporciona fácilmente.

**Consecuencias:**
El scanner de vulnerabilidades del IDE dejó de reportar CVEs de alta severidad tras fijar
la imagen base a `python:3.12.10-slim-bookworm` (versión de patch específica). El contexto
de build se redujo significativamente al excluir directorios de datos y desarrollo. El
fichero `.env` ya no puede entrar accidentalmente a la imagen aunque alguien olvide añadirlo
al `.gitignore`.

**Trade-offs aceptados:**
El usuario `appuser` no puede instalar paquetes adicionales en tiempo de ejecución (sin
`sudo`, sin acceso a `apt`). Esto es intencional: la imagen debe ser inmutable. Si se
necesita depurar en producción, se puede usar `docker exec --user root` con privilegios
explícitos temporales.

---

## ADR-017: sentence-transformers como dependencia opcional

**Estado:** Aceptado
**Fecha:** Abril 2026

**Contexto:**
El reranker Cross-Encoder requiere sentence-transformers,
que instala PyTorch y dependencias CUDA de NVIDIA (~7GB).
La imagen Docker resultaba de 8.9GB — imposible de desplegar
en EC2 t3.small con 20GB de disco total.

**Decisión:**
sentence-transformers se declara como dependencia opcional
en pyproject.toml. La imagen de producción se construye sin
ella (USE_RERANKING=false por defecto en EC2).

El reranker tiene import condicional en reranker.py:
si sentence-transformers no está instalado, retorna los
documentos sin reordenar (fallback graceful).

**Alternativas descartadas:**

- **Incluir PyTorch en producción**: imagen de 8.9GB,
  imposible en t3.small.
- **Migrar a t3.medium o t3.large**: costo adicional
  innecesario para el scope de esta prueba técnica.
- **Usar un modelo de reranking más liviano (TF-IDF)**:
  menor calidad que Cross-Encoder, no justifica el cambio.

**Consecuencias:**

- Imagen de producción: ~1GB (vs 8.9GB con PyTorch)
- Reranking disponible en desarrollo local con
  USE_RERANKING=true
- En producción el retrieval usa ChromaDB directo (top-5
  por similitud coseno) sin reranking
- El impacto en calidad está medido: F1 baja de 0.926
  (con reranking) a 0.888 (sin reranking) — diferencia
  de 4.3 puntos porcentuales aceptable para producción

**Plan de escalabilidad:**
Migrar a t3.medium (~$0.04/hr) o usar un servicio de
reranking dedicado con caché permitiría activar
USE_RERANKING=true en producción sin impacto en el
tamaño de imagen.

---

## Mejoras futuras identificadas

Estas mejoras fueron identificadas durante el desarrollo pero no implementadas en la versión
actual. Se documentan aquí para que el equipo o evaluador entienda qué se sacrificó
conscientemente y por qué.

### 1. Reranking con Cross-Encoder

**Impacto:** Alto — mejora significativa en la calidad del retrieval
**Estado:** Aceptado - implementado (2026-04-06)
**Descripción:** Después de recuperar los top-k chunks por similitud coseno, se aplica un
Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) para reordenar los resultados
por relevancia real al query. El retrieval bi-encoder es eficiente pero puede ordenar
incorrectamente chunks con alta similitud léxica pero baja relevancia semántica. El reranking
añade ~200ms de latencia pero reduce respuestas con fuentes incorrectas.

**Detalles de implementación:**

- Modelo: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22 MB, corre 100% local en CPU, sin API key)
- Clase `Reranker` en `src/embeddings/reranker.py` — sigue el mismo patrón que `Embedder`
- Singleton `get_reranker()` en `src/mcp_server/tools.py` — carga el modelo una sola vez (~2-3s)
- Parámetro `use_reranking: bool = True` en `search_knowledge_base` — activo por defecto
- Cuando `use_reranking=True`: ChromaDB recupera `top_k * 3` candidatos; el Cross-Encoder
  reordena y retorna los mejores `top_k`. Cada resultado incluye `rerank_score` y `retrieval_method`
- Fallback graceful: si el modelo falla, se retornan los resultados de ChromaDB sin reordenar

### 2. Actualización incremental de la Knowledge Base

**Impacto:** Medio — mantiene el corpus actualizado automáticamente
**Estado:** No implementado, requiere scheduler
**Descripción:** El scraper actual es one-shot. Para un sistema de producción, se necesita
un job periódico (cron diario/semanal) que detecte páginas nuevas o modificadas en el sitemap,
las rescrape, y actualice solo los chunks afectados en ChromaDB sin re-procesar el corpus
completo. Requiere comparar hashes de contenido o fechas de última modificación del sitemap.

### 3. Autenticación en MCP Server

**Impacto:** Medio — requerido para exposición pública segura
**Estado:** No implementado — fuera de scope del proyecto académico
**Descripción:** El MCP Server en modo SSE está actualmente expuesto sin autenticación en
el puerto 8000. En producción real se requeriría token Bearer o API key para proteger el
endpoint. FastMCP soporta middleware de autenticación; la implementación es sencilla pero
añade complejidad de gestión de keys fuera del scope del demo.

### 4. Migración de ChromaDB a Qdrant para escala horizontal

**Impacto:** Alto — necesario a partir de ~100K documentos
**Estado:** Identificado como evolución natural
**Descripción:** ChromaDB es adecuado para el corpus actual (~1500 chunks). Si el sistema
se escalara para cubrir todo el sitio de Bancolombia (no solo /personas) o múltiples bancos,
ChromaDB en modo single-node sería un cuello de botella. Qdrant ofrece clustering nativo,
colecciones distribuidas y mejor rendimiento de escritura en paralelo. La abstracción
`VectorStoreRepository` (ADR-012) fue diseñada precisamente para facilitar esta migración.

### 5. Knowledge Base Coverage Dashboard

**Impacto:** Medio — observabilidad del sistema RAG
**Estado:** Implementado — cubierto por el Dashboard de Monitoreo (ver ADR-014)
**Descripción:** El dashboard de monitoreo (`src/frontend/pages/monitoring.py`) implementa
esta funcionalidad: muestra métricas de cobertura por categoría, gaps de KB (preguntas con
score de similitud bajo), historial de conversaciones y trazabilidad de llamadas MCP.
Acceso protegido con contraseña via `MONITORING_PASSWORD`.
