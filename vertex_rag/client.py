from typing import Optional

import vertexai

from .config import VertexRagConfig, get_default_config


def init_vertex_rag(config: Optional[VertexRagConfig] = None) -> VertexRagConfig:
    """
    Initialize the Vertex AI client for RAG operations.

    Returns the effective config used for initialization.
    """
    cfg = config or get_default_config()
    if not cfg.project_id:
        raise ValueError(
            "VertexRagConfig.project_id is required. "
            "Ensure GCP_PROJECT_ID / GCP_ID / VERTEX_PROJECT_ID is set."
        )

    vertexai.init(project=cfg.project_id, location=cfg.location)
    return cfg

