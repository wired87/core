# Prompt: Production-Ready Brain Graph With DuckDB Hybrid Classification

## Objective

Implement a modular `Brain` system in the `graph` package that:

- Inherits from `GUtils` using `nx.MultiGraph`
- Hydrates user-scoped long-term memory from DuckDB/QBrain tables
- Maintains short-term QA memory
- Classifies user input into `RELAY_CASES_CONFIG` goals with a hybrid strategy
- Fills required payload fields from memory
- Executes the selected case callable only when required data is complete

The design must be production-ready, low-latency, deterministic, and easy to evolve.

## Non-Goals

- Do not replace Relay/Orchestrator transport flow in this task
- Do not hard-couple to one embedding provider only
- Do not put giant payload blobs in graph nodes (store compact refs)

## Existing Contracts To Respect

- Relay cases come from `predefined_case.RELAY_CASES_CONFIG`
- Case schema keys: `case`, `desc`, `req_struct`, `out_struct`, `func`
- DB manager and qbrain manager are already available (`_db.manager`, `core.qbrain_manager`)
- Graph base class is `graph.local_graph_utils.GUtils`

## Required Package Layout

Create/update:

- `graph/brain.py` (main orchestrator class)
- `graph/brain_schema.py` (node/edge schema constants and typed helpers)
- `graph/brain_hydrator.py` (DuckDB user memory ingestion)
- `graph/brain_classifier.py` (hybrid goal classification)
- `graph/brain_executor.py` (callable gating and execution)
- `graph/brain_workers.py` (thread/offload helpers)

## Graph Schema Contract

### Node Types

- `USER`
- `GOAL`
- `SUB_GOAL`
- `SHORT_TERM_STORAGE`
- `LONG_TERM_STORAGE`
- `CONTENT`

### Edge Types

- `derived_from`
- `requires`
- `satisfies`
- `references_table_row`
- `follows`
- `parent_of`

### Node Payload Rules

- `LONG_TERM_STORAGE`: compact reference only:
  - `table_name`, `row_id`, `description`, `updated_at`, `user_id`
- `SHORT_TERM_STORAGE`: `role`, `message`, `created_at`, `request_id`
- `GOAL/SUB_GOAL`: selected case + resolved/missing field details
- `CONTENT`: chunk refs and retrieval metadata (avoid huge blobs)

## Brain Workflow

1. **Init**
   - Accept `user_id`, optional `QBrainTableManager`/`DBManager`, optional relay case list
   - Initialize graph as `nx.MultiGraph`
   - Build case index for hybrid classification

2. **Hydration**
   - Read all user-scoped rows from tables derived from `MANAGERS_INFO`
   - Build `LONG_TERM_STORAGE` nodes and `references_table_row` edges
   - Index long-term references for retrieval

3. **Input ingestion**
   - Add latest user query as `SHORT_TERM_STORAGE` node
   - Keep last 30 short-term nodes in active memory window
   - For file input: chunk using processor layer and link `CONTENT` nodes

4. **Hybrid goal classification**
   - Stage 1: exact/rule score (`case`, aliases, obvious tokens)
   - Stage 2: vector similarity over case descriptions (+ relevant refs)
   - Stage 3: LLM fallback only for unresolved tie/low confidence
   - Return structured decision: `case`, `confidence`, `source`

5. **Required-data filling**
   - Flatten required keys from selected `req_struct`
   - Resolve values from:
     - direct payload
     - short-term window
     - long-term references
   - Build `resolved`, `missing` maps

6. **Execute-or-ask**
   - If no missing keys: execute selected case callable with assembled payload
   - Else: emit explicit missing-fields message and create `SUB_GOAL` node

## Hybrid Classification Policy

- Prefer deterministic rule match when confidence is high
- Use vector retrieval to improve semantic recall
- Use LLM fallback only when required (cost/latency guard)
- Always include provenance in output (`rule`, `vector`, `llm`)

## Concurrency & Performance

- Offload embedding and hydration with bounded `ThreadPoolExecutor`
- Keep graph mutation serialized per `user_id` to avoid race conditions
- Upsert vector entries in batches
- Reuse long-lived vector store connection

## Debug Logging Contract

For each public and key private method:

- Print at method start: `method_name...`
- Print at method end: `method_name... done`
- Print important checkpoints between logical steps

## Callable Execution Contract

- Never call `func` unless all required fields are present
- Support both sync and async callables
- Wrap execution with safe error handling and structured response
- Return:
  - `status`: `executed` | `need_data` | `error`
  - `goal_case`
  - `resolved_fields`
  - `missing_fields`
  - `next_message`

## Testing Requirements

1. Unit: hydration builds long-term nodes from sample DuckDB rows
2. Unit: classifier returns expected case for representative queries
3. Unit: required field filling from mixed short/long memory
4. Unit: callable gating logic (`need_data` vs `executed`)
5. Integration: complete query flow from input -> goal -> execute/ask

## Implementation Prompt (Copy/Paste)

Implement a `Brain` architecture in the `graph` package with `Brain(GUtils)` using `nx.MultiGraph`. It must hydrate all user-scoped DuckDB/QBrain references into `LONG_TERM_STORAGE` nodes, maintain the last 30 QA messages as `SHORT_TERM_STORAGE`, classify user intent into `RELAY_CASES_CONFIG` goals using hybrid rule + vector + LLM fallback, fill selected case `req_struct` keys from short-term and long-term memory, and execute the case callable only when all required keys are present. If fields are missing, return a structured missing-data prompt and create a `SUB_GOAL` node.

Use modular files: `brain.py`, `brain_schema.py`, `brain_hydrator.py`, `brain_classifier.py`, `brain_executor.py`, `brain_workers.py`. Keep node payloads compact. Add deterministic debug prints (`method_name...` at start and `method_name... done` at end). Ensure thread-safe offload for heavy retrieval/embedding work and deterministic behavior under repeated execution.
