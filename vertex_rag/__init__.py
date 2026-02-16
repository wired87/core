"""
High-level helpers for working with Vertex AI RAG Engine inside BestBrain.

This package is intentionally split into focused modules:

- config:     Configuration and corpus naming helpers.
- client:     Vertex AI initialization helpers.
- corpus:     Create RAG corpora.
- upsert:     Upload/import files into a RAG corpus.
- retrieval:  List corpus files and run retrieval_query.
- status:     File/corpus status, success/error response helpers.
- engine:     Main VertexRagEngine class + create_tool helper.
"""

from .config import VertexRagConfig, get_default_config
from .engine import VertexRagEngine, create_tool
from .corpus import create_corpus

__all__ = [
    "VertexRagConfig",
    "get_default_config",
    "VertexRagEngine",
    "create_tool",
    "create_corpus",
]

