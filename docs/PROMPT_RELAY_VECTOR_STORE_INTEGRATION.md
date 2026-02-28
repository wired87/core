# Prompt: Integrate VectorStore into Relay for Semantic Case Classification

## Objective

Replace or augment the current AIChatClassifier with a VectorStore-based semantic classifier. At Relay init, embed all relay case descriptions and store them in DuckDB. On each incoming query (when `type` is missing or CHAT), use similarity search to classify intent, retrieve the matched case struct, and run its callable.

## Current Flow

1. **Relay.connect()**: Creates OrchestratorManager with `relay_cases` (from RELAY_CASES_CONFIG). Creates AIChatClassifier with same case struct.
2. **Relay.receive()**: Forwards payload to `orchestrator.handle_relay_payload()`.
3. **Orchestrator.handle_relay_payload()**: If `type` is missing or CHAT, calls `_ensure_data_type_from_classifier()` which uses `chat_classifier.main(user_id, msg)` to get a case name. Then `_resolve_case(data_type)` returns the case dict with `func`. The handler dispatches via `_dispatch_relay_handler()`.

## Target Flow

1. **Relay.connect()**: After orchestrator is created:
   - Create `VectorStore(store_name="relay_cases", db_path=...)` (e.g. `relay_vector_store.duckdb` in project root).
   - Call `vector_store.create_store()`.
   - For each relay case (from `self.relay_cases`):
     - Build text: `f"{case_name} {desc}".strip()` (same as AIChatClassifier `_case_items`).
     - Embed the text using an embedding model (e.g. QBrainTableManager `_generate_embedding` or Gem/GenAI).
     - Build metadata: `{"case": case_name, "desc": desc, "req_struct": req_struct, "out_struct": out_struct}`. Do NOT store `func` (callable) — it is not JSON-serializable.
     - `vector_store.upsert_vectors(ids=[case_name], vectors=[embedding], metadata=[metadata])`.
   - Store `self._vector_store = vector_store` and `self._relay_case_embedder` (or embedding fn ref).

2. **Orchestrator._ensure_data_type_from_classifier()** (or new path):
   - When `type` is missing or CHAT and `msg` is non-empty:
     - Embed the user message `msg` using the same embedding model.
     - Call `vector_store.similarity_search(query_vector=embedding, top_k=1)`.
     - If results and `score >= min_similarity`:
       - `data_type = results[0]["metadata"]["case"]`.
       - Set `payload["type"] = data_type`.
     - Else: fall back to CHAT or existing classifier.

3. **Dispatch**: The orchestrator already uses `_resolve_case(data_type)` to get the case dict (including `func`) from `self.cases`. The `func` is the callable from the original relay_cases. Run it via `_dispatch_relay_handler()` as today.

## Implementation Details

### 1. Relay Init (connect)

- **Where**: Inside `Relay.connect()`, after `self.orchestrator` is created and `self.relay_cases` is finalized.
- **VectorStore config**: `store_name="relay_cases"`, `db_path` from env (e.g. `RELAY_VECTOR_DB_PATH`) or default `relay_vector_store.duckdb`, `normalize_embeddings=True`.
- **Embedding**: Use a shared embedding function. Options:
  - `QBrainTableManager._generate_embedding(text)` (GenAI text-embedding-004).
  - Or inject an `embed_fn: Callable[[str], List[float]]` into Relay/Orchestrator.
- **Case struct serialization**: For each case, extract `case`, `desc`, `req_struct`, `out_struct`. Omit `func`. Serialize `req_struct`/`out_struct` with `_sanitize_req_struct_for_json` (or equivalent) so type hints become strings.
- **Idempotency**: Use `upsert_vectors` so reconnects or case updates overwrite existing rows. Case id = `case_name` (e.g. `"SET_ENV"`).

### 2. Classification Path

- **When**: In `_ensure_data_type_from_classifier()` when `data_type` is None or CHAT and `msg` is non-empty.
- **Flow**:
  1. Embed `msg` → `query_vector`.
  2. `results = vector_store.similarity_search(query_vector, top_k=1)`.
  3. If `results` and `results[0]["score"] >= min_similarity`:
     - `data_type = results[0]["metadata"]["case"]`.
     - `payload["type"] = data_type`.
  4. Else: keep current fallback (CHAT or None).
- **min_similarity**: Configurable (env `AI_CHAT_MIN_SIMILARITY` or default 0.5).

### 3. Callable Resolution

- The VectorStore metadata does NOT contain the callable. The orchestrator continues to use `self.cases` (relay_cases) to resolve the case by name and get `func`.
- `_resolve_case(data_type)` already does: `for c in self.cases: if c.get("case") == data_type: return c` (with `func`). No change needed.

### 4. Shared VectorStore Access

- The VectorStore is created in Relay and must be reachable from the orchestrator during classification.
- **Option A**: Pass `vector_store` into OrchestratorManager: `OrchestratorManager(cases, user_id, relay=self, vector_store=self._vector_store)`.
- **Option B**: Orchestrator gets it from relay: `vector_store = getattr(self.relay, "_vector_store", None)` when classifying.
- Prefer Option A for explicit dependency injection.

### 5. Embedding Model Consistency

- Use the SAME embedding model for both indexing (case descriptions) and query (user message). Otherwise similarity scores are meaningless.
- Store the model name or embedder ref so it can be reused.

### 6. Lifecycle

- **Create**: In `Relay.connect()` after relay_cases are ready.
- **Close**: In `Relay.disconnect()` call `self._vector_store.close()`.
- **Persistence**: DuckDB file persists across connections. On connect, `upsert_vectors` refreshes case embeddings (handles new/updated cases).

## File Changes Summary

| File | Change |
|------|--------|
| `relay_station.py` | In `connect()`: create VectorStore, embed relay_cases, upsert; store `_vector_store`. In `disconnect()`: close vector_store. Pass `vector_store` to OrchestratorManager. |
| `core/orchestrator_manager/orchestrator.py` | Accept `vector_store` in `__init__`. In `_ensure_data_type_from_classifier()`: when using vector classification, embed msg, call `vector_store.similarity_search`, set `payload["type"]` from top result. Fallback to existing classifier if no vector_store or no match. |
| `core/qbrain_manager/__init__.py` or new `utils/embedding.py` | Expose `embed_text(text: str) -> List[float]` for shared use (or use existing `_generate_embedding`). |

## Edge Cases

- **Empty relay_cases**: Skip vector store init; fall back to existing classifier.
- **Embedding failure**: Catch, log, fall back to existing classifier.
- **No vector_store**: Orchestrator falls back to `chat_classifier.main()` as today.
- **Duplicate case names**: Use case name as id; last upsert wins.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RELAY_VECTOR_DB_PATH` | Path to DuckDB file for relay case vectors. Default: `relay_vector_store.duckdb`. |
| `RELAY_USE_VECTOR_CLASSIFIER` | If `true`, use VectorStore for classification when available. |
| `AI_CHAT_MIN_SIMILARITY` | Minimum cosine similarity to accept a classification. Default: 0.5. |

## Testing

1. Unit test: Create Relay with mock relay_cases, verify VectorStore is populated with correct ids and metadata.
2. Integration test: Send a message without `type`, verify classification returns correct case and handler runs.
3. Verify fallback: Disable vector store or use message with no match, ensure CHAT/fallback path still works.

---

## Implementation Prompt (Copy-Paste)

Implement the following in the BestBrain codebase:

1. **Relay.connect()**: After creating the orchestrator, initialize a VectorStore (`_db.vector_store.VectorStore`) with `store_name="relay_cases"` and `db_path` from `RELAY_VECTOR_DB_PATH` or default `relay_vector_store.duckdb`. Call `create_store()`. For each item in `self.relay_cases`, extract `case` and `desc`, build `text = f"{case} {desc}".strip()`, embed `text` using an embedding function (e.g. from QBrainTableManager or a shared `embed_text` utility), and call `vector_store.upsert_vectors(ids=[case], vectors=[embedding], metadata=[{case, desc, req_struct, out_struct}])`. Do not store `func` in metadata. Store `self._vector_store = vector_store`. Pass `vector_store` to OrchestratorManager. In `disconnect()`, call `self._vector_store.close()`.

2. **OrchestratorManager**: Accept optional `vector_store` in `__init__`. In `_ensure_data_type_from_classifier()`, when `data_type` is None or CHAT and `msg` is non-empty and `vector_store` is set: embed `msg`, call `vector_store.similarity_search(query_vector, top_k=1)`, and if the top result has `score >= min_similarity`, set `data_type = results[0]["metadata"]["case"]` and `payload["type"] = data_type`. Otherwise fall back to existing `chat_classifier.main()`.

3. **Embedding**: Use the same embedding model for case descriptions and user messages. Expose `embed_text(text: str) -> List[float]` (e.g. via QBrainTableManager or a new `utils/embedding.py`) and use it in both Relay and Orchestrator.

4. **Callable execution**: No change. The orchestrator already uses `_resolve_case(data_type)` to get the case dict (including `func`) from `self.cases`. The `func` is the callable from the original relay_cases. Run it via `_dispatch_relay_handler()` as today.
