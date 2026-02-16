"""
Vertex AI RAG Engine – client bootstrap.
Initializes Vertex AI and exposes the RAG module for use by other modules.
"""
import vertexai
from vertexai import rag

from auth.load_sa_creds import load_service_account_credentials
from .config import PROJECT_ID, LOCATION, get_corpus_name, VertexRagConfig


def init_vertexai(project_id: str = None, location: str = None):
    """Initialize Vertex AI for the given project and location."""
    pid = project_id or PROJECT_ID
    loc = location or LOCATION
    if not pid:
        raise ValueError("Vertex AI project_id is required (set GOOGLE_CLOUD_PROJECT or pass project_id)")
    vertexai.init(project=pid, location=loc)
    return vertexai


def init_vertex_rag(config: VertexRagConfig = None) -> VertexRagConfig:
    """
    Initialize Vertex AI from config and ensure corpus_name is set.
    Returns the same config object (with corpus_name populated if needed).
    """
    from .config import get_default_config
    cfg = config or get_default_config()
    if not cfg.project_id:
        raise ValueError("Vertex AI project_id is required (set GOOGLE_CLOUD_PROJECT or pass config)")
    vertexai.init(
        credentials=load_service_account_credentials()
    )
    return cfg


def get_rag():
    """Return the vertexai.rag module (call init_vertexai or init_vertex_rag first if needed)."""
    return rag


def get_default_corpus_name(corpus_id: str = None) -> str:
    """Return the default RAG corpus resource name."""
    return get_corpus_name(corpus_id)
