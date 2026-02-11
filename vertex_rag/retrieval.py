from typing import Any, Dict, List, Optional

from vertexai import rag

from .client import init_vertex_rag
from .config import VertexRagConfig


def _resolve_corpus_name(
    cfg: VertexRagConfig,
    corpus_name: Optional[str] = None,
) -> str:
    final_corpus = corpus_name or cfg.corpus_name
    if not final_corpus:
        raise ValueError(
            "No corpus_name provided and no rag_corpus_id configured. "
            "Either pass corpus_name explicitly, or set VERTEX_RAG_CORPUS_ID."
        )
    return final_corpus


def list_corpus_files(
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all RAG files in the configured corpus using rag.list_files.
    """
    cfg = init_vertex_rag(config)
    final_corpus = _resolve_corpus_name(cfg, corpus_name)

    files = rag.list_files(corpus_name=final_corpus)
    return [
        {
            "name": getattr(f, "name", None),
            "display_name": getattr(f, "display_name", None),
            "description": getattr(f, "description", None),
        }
        for f in files
    ]


def retrieval_query(
    text: str,
    top_k: int = 10,
    rag_file_ids: Optional[List[str]] = None,
    vector_distance_threshold: Optional[float] = None,
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a pure retrieval query against the RAG Engine and return matched contexts.
    """
    cfg = init_vertex_rag(config)
    final_corpus = _resolve_corpus_name(cfg, corpus_name)

    rag_resource = rag.RagResource(
        rag_corpus=final_corpus,
        rag_file_ids=rag_file_ids or [],
    )

    filter_obj = None
    if vector_distance_threshold is not None:
        # See official docs for Filter options; we expose a minimal wrapper here.
        filter_obj = rag.utils.resources.Filter(
            vector_distance_threshold=vector_distance_threshold
        )

    retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,
        filter=filter_obj,
    )

    response = rag.retrieval_query(
        rag_resources=[rag_resource],
        text=text,
        rag_retrieval_config=retrieval_config,
    )

    # Normalise response into a dict; exact shape depends on SDK,
    # so we keep only high-signal, JSON-safe attributes.
    contexts: List[Dict[str, Any]] = []
    for ctx in getattr(response, "contexts", []) or []:
        contexts.append(
            {
                "source": getattr(ctx, "source", None),
                "score": getattr(ctx, "score", None),
                "chunk": getattr(ctx, "chunk", None),
                "rag_file": getattr(ctx, "rag_file", None),
            }
        )

    return {
        "corpus_name": final_corpus,
        "contexts": contexts,
    }

