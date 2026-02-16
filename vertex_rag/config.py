"""
Vertex AI RAG Engine – configuration.
Reads project, location, and optional corpus ID from environment.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

# Project and region (required for Vertex AI RAG)
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID") or ""
LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")

# Optional: pre-created RAG corpus resource name
# Format: projects/{project}/locations/{location}/ragCorpora/{corpus_id}
CORPUS_ID = os.environ.get("VERTEX_RAG_CORPUS_ID", "")
CORPUS_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{CORPUS_ID}"
    if (PROJECT_ID and CORPUS_ID) else ""
)


def get_corpus_name(corpus_id: str = None) -> str:
    """Return full RAG corpus resource name. Uses CORPUS_ID env if corpus_id not provided."""
    cid = corpus_id or CORPUS_ID
    if not PROJECT_ID or not cid:
        return ""
    return f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{cid}"


@dataclass
class VertexRagConfig:
    """Configuration for Vertex AI RAG (project, location, corpus)."""
    project_id: str = field(default_factory=lambda: PROJECT_ID)
    location: str = field(default_factory=lambda: LOCATION)
    rag_corpus_id: str = field(default_factory=lambda: CORPUS_ID)
    corpus_name: str = field(default="", init=False)

    def __post_init__(self):
        if self.project_id and self.rag_corpus_id:
            self.corpus_name = (
                f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.rag_corpus_id}"
            )
        else:
            self.corpus_name = ""


def get_default_config() -> VertexRagConfig:
    """Build a VertexRagConfig from environment variables."""
    return VertexRagConfig()
