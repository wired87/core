from typing import Any, Dict, Optional

from vertexai import rag

from .client import init_vertex_rag
from .config import VertexRagConfig, get_default_config


def _build_default_backend_config() -> rag.RagVectorDbConfig:
    """
    Build a default RagVectorDbConfig using the recommended text-embedding-005 model.

    This is a thin wrapper around the official sample config so that all
    per-session corpora use a consistent, managed backing store.
    """
    return rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )
    )


def create_corpus(
    display_name: str,
    description: str = "",
    *,
    config: Optional[VertexRagConfig] = None,
) -> Dict[str, Any]:
    """
    Create a new Vertex AI RAG corpus and return its identifiers.

    Returns a dict with:
        - name:       Full corpus resource name.
        - corpus_id:  Short corpus id (last path segment).
        - display_name, description: Echoed metadata (where available).
    """
    cfg = init_vertex_rag(config or get_default_config())

    backend_config = _build_default_backend_config()

    corpus = rag.create_corpus(
        display_name=display_name,
        description=description,
        backend_config=backend_config,
    )

    name = getattr(corpus, "name", None)
    corpus_id = ""
    if isinstance(name, str) and name:
        corpus_id = name.rsplit("/", 1)[-1]

    return {
        "name": name,
        "corpus_id": corpus_id,
        "display_name": getattr(corpus, "display_name", None),
        "description": getattr(corpus, "description", None),
    }

"""
Vertex AI RAG Engine – corpus management.
Create and list RAG corpora (indexes).
"""
from typing import Any, Dict, List, Optional

from vertexai import rag

from .client import init_vertex_rag
from .config import VertexRagConfig


def _ensure_init(config: Optional[VertexRagConfig] = None) -> VertexRagConfig:
    return init_vertex_rag(config)


def create_corpus(
    display_name: str,
    description: str = "",
    config: Optional[VertexRagConfig] = None,
) -> Dict[str, Any]:
    """
    Create a new RAG corpus with the given display name and description.
    Uses default embedding model (text-embedding-005). Returns corpus metadata.
    """
    cfg = _ensure_init(config)
    backend_config = rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )
    )
    corpus = rag.create_corpus(
        display_name=display_name,
        description=description,
        backend_config=backend_config,
    )
    return {
        "name": getattr(corpus, "name", None),
        "display_name": getattr(corpus, "display_name", None),
        "description": getattr(corpus, "description", None),
    }


def list_corpora(config: Optional[VertexRagConfig] = None) -> List[Dict[str, Any]]:
    """
    List all RAG corpora in the project/location.
    """
    cfg = _ensure_init(config)
    corpora = rag.list_corpora()
    result = []
    for c in corpora:
        result.append({
            "name": getattr(c, "name", None),
            "display_name": getattr(c, "display_name", None),
            "description": getattr(c, "description", None),
        })
    return result
