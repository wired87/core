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


def upload_local_file(
    path: str,
    display_name: str,
    description: str = "",
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload a single local file into the RAG corpus using rag.upload_file.
    """
    cfg = init_vertex_rag(config)
    final_corpus = _resolve_corpus_name(cfg, corpus_name)

    rag_file = rag.upload_file(
        corpus_name=final_corpus,
        path=path,
        display_name=display_name,
        description=description,
    )

    # Normalise into a simple serialisable dict
    return {
        "name": getattr(rag_file, "name", None),
        "display_name": getattr(rag_file, "display_name", None),
        "description": getattr(rag_file, "description", None),
    }


def import_remote_files(
    paths: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    import_result_sink: Optional[str] = None,
    max_embedding_requests_per_min: Optional[int] = None,
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Import a set of remote files (GCS / Drive URLs) into the RAG corpus using rag.import_files.
    """
    cfg = init_vertex_rag(config)
    final_corpus = _resolve_corpus_name(cfg, corpus_name)

    transformation_config = rag.TransformationConfig(
        rag.ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    response = rag.import_files(
        corpus_name=final_corpus,
        paths=paths,
        transformation_config=transformation_config,
        import_result_sink=import_result_sink,
        max_embedding_requests_per_min=max_embedding_requests_per_min,
    )

    return {
        "imported_rag_files_count": getattr(response, "imported_rag_files_count", None),
        "corpus_name": final_corpus,
    }

