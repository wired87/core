"""
High-level helpers for working with Vertex AI RAG Engine inside BestBrain.

This package is intentionally split into focused modules:

- config:     Configuration and corpus naming helpers.
- client:     Vertex AI initialization helpers.
- upsert:     Upload/import files into a RAG corpus.
- retrieval:  Retrieve context from the RAG corpus.
- status:     Simple status/utility helpers.
- engine:     Main VertexRagEngine class + create_tool helper.
"""

from .config import VertexRagConfig, get_default_config
from .engine import VertexRagEngine, create_tool

__all__ = [
    "VertexRagConfig",
    "get_default_config",
    "VertexRagEngine",
    "create_tool",
]

