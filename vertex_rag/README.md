# vertex_rag

Vertex AI RAG Engine integration for BestBrain: use Google Cloud Vertex AI RAG to **upsert files**, **ask questions**, **manage status**, **retrieve files**, and more. All logic lives in separate, focused modules under this directory.

## All capabilities (summary)

- **Configuration** – project, location, corpus ID; env-based defaults (`config.py`).
- **Client init** – initialize Vertex AI and resolve corpus name (`client.py`).
- **Corpus management** – create RAG corpus; list corpora (`corpus.py`).
- **Upsert files** – upload local file; import remote files (GCS/Drive) with chunking (`upsert.py`).
- **Ask questions** – retrieval + optional LLM answer over corpus (`query.py`).
- **Retrieval** – list corpus files; semantic retrieval query (`retrieval.py`).
- **Retrieve / delete files** – get one RAG file by name; delete RAG file (`files.py`).
- **Status management** – file status, corpus status (file count + list); success/error response helpers (`status.py`).
- **High-level engine** – single entry point: upload, import, list_files, retrieval_query, create RAG Tool for Gemini (`engine.py`).

---

## Directory layout

```
vertex_rag/
├── README.md         # This file – capabilities index and usage
├── __init__.py       # Package exports (VertexRagConfig, VertexRagEngine, create_tool)
├── config.py         # Configuration: project, location, corpus ID (env + VertexRagConfig)
├── client.py         # Client bootstrap: init_vertexai, init_vertex_rag, get_rag
├── corpus.py         # Corpus management: create_corpus, list_corpora
├── upsert.py         # Upsert: upload_local_file, import_remote_files (GCS/Drive)
├── retrieval.py      # Retrieval: list_corpus_files, retrieval_query (semantic search)
├── query.py          # Ask questions: ask_question (retrieval + optional LLM answer)
├── files.py          # File get/delete: get_file, delete_file
├── status.py         # Status: get_file_status, get_corpus_status, success/error response helpers
└── engine.py         # High-level engine: VertexRagEngine, create_tool (RAG Tool for Gemini)
```

---

## Capabilities index

All public entry points by module. Use this table to find the right file and function for each capability.

| Capability | File | Function / class | Description |
|------------|------|------------------|-------------|
| **Configuration** | `config.py` | `VertexRagConfig` | Dataclass: project_id, location, rag_corpus_id, corpus_name. |
| | `config.py` | `get_default_config()` | Build config from env (GOOGLE_CLOUD_PROJECT, VERTEX_AI_LOCATION, VERTEX_RAG_CORPUS_ID). |
| | `config.py` | `get_corpus_name(corpus_id)` | Return full RAG corpus resource name. |
| **Client init** | `client.py` | `init_vertexai(project_id, location)` | Initialize Vertex AI for project/location. |
| | `client.py` | `init_vertex_rag(config)` | Init Vertex AI from config; ensure corpus_name set. |
| | `client.py` | `get_rag()` | Return `vertexai.rag` module. |
| | `client.py` | `get_default_corpus_name(corpus_id)` | Default RAG corpus resource name. |
| **Corpus management** | `corpus.py` | `create_corpus(display_name, description, config)` | Create a new RAG corpus (embedding model config). |
| | `corpus.py` | `list_corpora(config)` | List all RAG corpora in project/location. |
| **Upsert files** | `upsert.py` | `upload_local_file(path, display_name, description, config, corpus_name)` | Upload one local file into a corpus. |
| | `upsert.py` | `import_remote_files(paths, chunk_size, chunk_overlap, ...)` | Import remote files (GCS/Drive) with chunking. |
| **Ask questions** | `query.py` | `ask_question(question, top_k, rag_file_ids, vector_distance_threshold, generate_answer, model, config, corpus_name)` | Retrieval + optional LLM answer over corpus. |
| **Retrieval** | `retrieval.py` | `list_corpus_files(config, corpus_name)` | List all RAG files in a corpus. |
| | `retrieval.py` | `retrieval_query(text, top_k, rag_file_ids, vector_distance_threshold, config, corpus_name)` | Semantic retrieval; returns contexts. |
| **Retrieve / delete files** | `files.py` | `get_file(name, config)` | Get one RAG file metadata by resource name. |
| | `files.py` | `delete_file(name, force_delete, config)` | Delete a RAG file by resource name. |
| **Status management** | `status.py` | `get_file_status(file_name, config)` | File metadata wrapped in success/error response. |
| | `status.py` | `get_corpus_status(config, corpus_name)` | Corpus summary: file count + list of files. |
| | `status.py` | `build_success_response(case, data)` | Standard success payload (type, status, data). |
| | `status.py` | `build_error_response(case, error)` | Standard error payload (type, status, data). |
| **High-level engine** | `engine.py` | `VertexRagEngine(corpus_id, config)` | Single entry point: upload, import, list_files, retrieval_query, create_tool. |
| | `engine.py` | `VertexRagEngine.upload_local_file(...)` | Upload one local file (delegates to upsert). |
| | `engine.py` | `VertexRagEngine.import_remote_files(...)` | Import remote files (delegates to upsert). |
| | `engine.py` | `VertexRagEngine.list_files()` | List corpus files (delegates to retrieval). |
| | `engine.py` | `VertexRagEngine.retrieval_query(...)` | Semantic retrieval (delegates to retrieval). |
| | `engine.py` | `VertexRagEngine.create_tool(...)` | Build RAG Tool for Gemini. |
| | `engine.py` | `create_tool(corpus_id, output_pattern, ...)` | Standalone: create RAG Tool for a corpus. |

---

## Quick start

1. **Environment**  
   Set `GOOGLE_CLOUD_PROJECT`; optionally `VERTEX_AI_LOCATION` (default `us-central1`), `VERTEX_RAG_CORPUS_ID`.

2. **Corpus**  
   Create once: `from vertex_rag.corpus import create_corpus` → `create_corpus("My Corpus", "Description")`, or use an existing corpus ID.

3. **Upsert**  
   - Local: `from vertex_rag.upsert import upload_local_file` → `upload_local_file("/path/to/file.pdf", "Doc", "Description")`.  
   - Remote: `from vertex_rag.upsert import import_remote_files` → `import_remote_files(["gs://bucket/file.pdf"], chunk_size=512)`.

4. **Ask / retrieve**  
   - Question + answer: `from vertex_rag.query import ask_question` → `ask_question("Your question?", generate_answer=True)`.  
   - Context only: `from vertex_rag.retrieval import retrieval_query` → `retrieval_query("query", top_k=10)`.

5. **Files & status**  
   - List files: `from vertex_rag.retrieval import list_corpus_files` → `list_corpus_files()`.  
   - One file: `from vertex_rag.files import get_file` → `get_file(name)`.  
   - Delete: `from vertex_rag.files import delete_file` → `delete_file(name)`.  
   - Corpus status: `from vertex_rag.status import get_corpus_status` → `get_corpus_status()`.

6. **Single entry point**  
   `from vertex_rag import VertexRagEngine`  
   `engine = VertexRagEngine(corpus_id="...")`  
   Then: `engine.upload_local_file(...)`, `engine.import_remote_files(...)`, `engine.list_files()`, `engine.retrieval_query(...)`, `engine.create_tool(...)`.

---

## Dependencies

- `google-cloud-aiplatform` (Vertex AI SDK, including `vertexai.rag` and generative models).
