"""
Vertex AI RAG Engine – file retrieval and deletion.
Get a single RAG file by name; delete a RAG file by resource name.
"""
from typing import Any, Dict, Optional

from vertexai import rag

from .client import init_vertex_rag
from .config import VertexRagConfig


def get_file(
    name: str,
    config: Optional[VertexRagConfig] = None,
) -> Dict[str, Any]:
    """
    Retrieve a single RAG file by its full resource name or ID.
    Returns file metadata (name, display_name, description, etc.).
    """
    init_vertex_rag(config)
    rag_file = rag.get_file(name=name)
    return {
        "name": getattr(rag_file, "name", None),
        "display_name": getattr(rag_file, "display_name", None),
        "description": getattr(rag_file, "description", None),
    }


def delete_file(
    name: str,
    force_delete: bool = False,
    config: Optional[VertexRagConfig] = None,
) -> Dict[str, Any]:
    """
    Delete a RAG file by its full resource name.
    name: e.g. projects/{project}/locations/{loc}/ragCorpora/{corpus}/ragFiles/{file_id}
    force_delete: if True, ignore external DB errors (see Vertex AI RAG API).
    Returns operation metadata or raises on failure.
    """
    init_vertex_rag(config)
    try:
        try:
        operation = rag.delete_file(name=name, force_delete=force_delete)
    except TypeError:
        operation = rag.delete_file(name=name)
    except TypeError:
        operation = rag.delete_file(name=name)
    return {
        "name": getattr(operation, "name", None),
        "done": getattr(operation, "done", None),
    }
