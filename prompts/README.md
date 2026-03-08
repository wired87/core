# Prompts Directory

Collected prompts from across the BestBrain codebase, organized in separate `.md` files.

## Architecture & Graph

| File | Source | Description |
|------|--------|-------------|
| `brain_graph_duckdb_hybrid.md` | `qbrain/docs/PROMPT_BRAIN_GRAPH_DUCKDB_HYBRID.md` | Brain graph with DuckDB hybrid classification |
| `relay_vector_store_integration.md` | `qbrain/docs/PROMPT_RELAY_VECTOR_STORE_INTEGRATION.md` | VectorStore integration for Relay case classification |
| `graph_processor_knowledge_graph.md` | `qbrain/docs/PROMPT_GRAPH_PROCESSOR_KNOWLEDGE_GRAPH.md` | Knowledge graph from chunked file content |

## Manager & Extraction

| File | Source | Description |
|------|--------|-------------|
| `extract_params.md` | `qbrain/core/param_manager/extraction_prompt.py` | JAX param extraction from LaTeX |
| `extract_fields.md` | `qbrain/core/fields_manager/prompt.py` | Field extraction |
| `extract_methods.md` | `qbrain/core/method_manager/xtrct_prompt.py` | Method/equation extraction for JAX |

## Session & Research

| File | Source | Description |
|------|--------|-------------|
| `session_manager_research_files.md` | `qbrain/core/session_manager/prompts.md` | Research files column in session manager |
| `researcher_extend_case.md` | `qbrain/core/researcher2/researcher2/prompt.md` | Extend researcher with case.py and Gemini deep research |

## BigQuery Toolbox

| File | Source | Description |
|------|--------|-------------|
| `bigquery_toolbox_classification.md` | `qbrain/_bigquery_toolbox/prompts.py` | Intent classification |
| `bigquery_toolbox_table_filter.md` | | Table selection for queries |
| `bigquery_toolbox_sql_generation.md` | | SQL query generation |
| `bigquery_toolbox_query_expansion.md` | | Search query expansion |
| `bigquery_toolbox_natural_answer.md` | | Natural language answer from SQL results |
| `bigquery_toolbox_platform_help.md` | | Platform assistant |
| `bigquery_toolbox_query_rewrite.md` | | Query rewrite with conversation history |
| `bigquery_toolbox_upload_instructions.md` | | Ingest/upload instructions |

## Module Manager (Physics Extraction)

| File | Source | Description |
|------|--------|-------------|
| `module_manager_graph_link.md` | `qbrain/core/module_manager/utils/prompts.py` | Graph link extraction |
| `module_manager_full_extraction.md` | | Full physics paper extraction |
| `module_manager_center_field.md` | | Center field identification |
| `module_manager_parameter.md` | | Parameter extraction |
| `module_manager_equation.md` | | Equation extraction |

## Other

| File | Source | Description |
|------|--------|-------------|
| `create_video_demo.md` | `prompts/create/create video.md` | QDash demo video recording |
| `explanation_bestbrain.md` | `qbrain/chat_manger/prompts/explanation_bestbrain.py` | BestBrain simulation engine overview |
