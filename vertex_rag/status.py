from typing import Any, Dict, List, Optional

from .config import VertexRagConfig
from . import files as rag_files
from . import retrieval


def get_file_status(
    file_name: str,
    config: Optional[VertexRagConfig] = None,
) -> Dict[str, Any]:
    """
    Get status/metadata for a single RAG file by name.
    Returns file metadata (name, display_name, description) or error info.
    """
    try:
        meta = rag_files.get_file(name=file_name, config=config)
        return build_success_response("RAG_FILE_STATUS", meta)
    except Exception as e:
        return build_error_response("RAG_FILE_STATUS", e)


def build_success_response(case: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper to standardise Relay-style success responses for vertex_rag.
    """
    return {
        "type": case,
        "status": {
            "state": "success",
            "code": 200,
            "msg": "",
        },
        "data": data,
    }


def build_error_response(case: str, error: Exception) -> Dict[str, Any]:
    """
    Helper to standardise Relay-style error responses for vertex_rag.
    """
    return {
        "type": case,
        "status": {
            "state": "error",
            "code": 500,
            "msg": str(error),
        },
        "data": {},
    }


def get_corpus_status(
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get status summary for a RAG corpus: corpus name and list of files (count + metadata).
    Uses list_corpus_files under the hood. Returns a success-shaped response.
    """
    try:
        files = retrieval.list_corpus_files(config=config, corpus_name=corpus_name)
        data = {
            "corpus_name": corpus_name or (config.corpus_name if config else None),
            "file_count": len(files),
            "files": files,
        }
        return build_success_response("RAG_CORPUS_STATUS", data)
    except Exception as e:
        return build_error_response("RAG_CORPUS_STATUS", e)

