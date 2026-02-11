import os
from dataclasses import dataclass
from typing import Optional

from core.app_utils import GCP_ID


@dataclass
class VertexRagConfig:
    """
    Configuration for Vertex AI RAG Engine usage.

    - project_id:    GCP project id (defaults from existing BestBrain config).
    - location:      Vertex AI location/region.
    - rag_corpus_id: ID of the RAG corpus (not the full resource name).
    """

    project_id: str
    location: str = "us-central1"
    rag_corpus_id: Optional[str] = None

    @property
    def corpus_name(self) -> Optional[str]:
        """
        Full resource name for the RAG corpus, if rag_corpus_id is set.

        Example:
        projects/{PROJECT_ID}/locations/{location}/ragCorpora/{rag_corpus_id}
        """
        if not self.rag_corpus_id:
            return None
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/ragCorpora/{self.rag_corpus_id}"
        )


def get_default_config() -> VertexRagConfig:
    """
    Build a default VertexRagConfig from environment and existing app config.

    Priority for project_id:
    1. core.app_utils.GCP_ID (which itself reads GCP_PROJECT_ID)
    2. env VERTEX_PROJECT_ID
    3. env GCP_ID
    """
    project_id = GCP_ID or os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GCP_ID")
    if not project_id:
        raise ValueError(
            "VertexRagConfig.project_id is not set. "
            "Set GCP_PROJECT_ID or VERTEX_PROJECT_ID or GCP_ID in the environment."
        )

    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    rag_corpus_id = os.environ.get("VERTEX_RAG_CORPUS_ID")

    return VertexRagConfig(
        project_id=project_id,
        location=location,
        rag_corpus_id=rag_corpus_id,
    )

