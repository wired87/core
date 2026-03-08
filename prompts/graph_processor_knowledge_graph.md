# Prompt: Production-Ready Knowledge Graph from Chunked File Content

## Objective

Adapt the `graph.processor` to build a knowledge graph from chunked file content. Create `type=CONTENT` nodes for each chunk, link them modularly (parent-child, sequential, optionally file-level), and persist the graph via GUtils. The processor remains format-agnostic (PDF, CSV, text, images); the graph layer consumes chunk rows and produces a traversable knowledge graph.

## Current Flow

1. **FileProcessorFacade** ([graph/processor/main.py](graph/processor/main.py)): Routes by extension to PdfProcessor, TableProcessor, TextProcessor, or ImageProcessor.
2. **BaseProcessor** ([graph/processor/base.py](graph/processor/base.py)): Loads documents, splits into large (1000) and small (200) chunks via `RecursiveCharacterTextSplitter`, wraps in `KnowledgeNode` (from `..models`), returns `List[Dict]` via `to_dict()`.
3. **Output**: Flat list of chunk rows. No graph construction, no GUtils integration.
4. **KnowledgeNode**: Imported from `graph.models`; model must provide `id`, `content`, `source_file`, `chunk_type`, `parent_id`, `page`, `category`, `tags`, and `to_dict()`.

## Target Flow

1. **Processor**: Unchanged chunking logic. Returns `List[Dict]` (KnowledgeNode rows).
2. **GraphBuilder**: New component that accepts rows + GUtils, adds CONTENT nodes, adds edges (parent_of, follows, optionally part_of_file).
3. **FileProcessorFacade**: New method `process_to_graph(file_path, g: GUtils)` or `process_file` returns rows; caller optionally passes rows + g to GraphBuilder.
4. **GUtils**: Receives CONTENT nodes and edges. `local_batch_loader` persists to schemas for downstream DB push.

## Node Schema: CONTENT

| Field | Type | Description |
|-------|------|-------------|
| id | str | Unique id, e.g. `{filename}_p{i}` or `{filename}_p{i}_c{j}` |
| type | str | Always `"CONTENT"` |
| content | str | Chunk text |
| source_file | str | Basename of source file |
| chunk_type | str | `"large"` or `"small"` |
| parent_id | str \| None | Parent large chunk id; None for large chunks |
| page | int | Page number (0 if N/A) |
| category | str | `"Document"` \| `"Data"` \| `"Code"` |
| tags | List[str] | e.g. `["pdf"]`, `["child"]` |

## Edge Schema: Modular Links

| rel | src_layer | trgt_layer | Description |
|-----|-----------|------------|-------------|
| parent_of | CONTENT | CONTENT | Large chunk → small chunk |
| follows | CONTENT | CONTENT | Sequential chunk i → chunk i+1 (same parent) |
| part_of_file | CONTENT | FILE | Chunk → source file node (optional) |

## Implementation Prompt (Copy-Paste)

Implement the following in the BestBrain codebase:

1. **graph/models.py**: Create a `KnowledgeNode` dataclass with fields `id`, `content`, `source_file`, `chunk_type`, `parent_id`, `page`, `category`, `tags`. Implement `to_dict()` returning a dict with `type="CONTENT"` included.

2. **graph/processor/graph_builder.py**: Create `build_graph(rows: List[Dict], g: GUtils, add_file_nodes: bool = False) -> int`. For each row, add a CONTENT node via `g.add_node(attrs={**row, "type": "CONTENT"})`. For rows with `parent_id`, add edge `(parent_id, row["id"], rel="parent_of", src_layer="CONTENT", trgt_layer="CONTENT")`. For sequential small chunks under the same parent, add `follows` edges. If `add_file_nodes`, create FILE nodes per `source_file` and add `part_of_file` edges. Return the number of nodes added.

3. **graph/processor/base.py**: Ensure `from graph.models import KnowledgeNode` (or `from ..models import KnowledgeNode` if `graph/models.py` exists and is in package).

4. **graph/processor/main.py**: Add `process_to_graph(self, file_path: str, g: GUtils) -> int` that calls `process_file(file_path)` to get rows, then `GraphBuilder.build_graph(rows, g)`, and returns the node count.

5. **Modular linking**: Follow the edge schema (parent_of, follows, part_of_file). Keep processor and GraphBuilder separate: processor produces rows, GraphBuilder consumes rows and populates the graph.
