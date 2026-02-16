from typing import Any, Dict, List, Optional

#pip install --upgrade google-cloud-aiplatform
from vertexai import rag
from vertexai.generative_models import Tool

from .config import VertexRagConfig, get_default_config
from .client import init_vertex_rag
from . import upsert, retrieval


def create_tool(
    corpus_id: str,
    output_pattern: Dict[str, Any],
    *,
    config: Optional[VertexRagConfig] = None,
    top_k: int = 10,
    vector_distance_threshold: Optional[float] = 0.5,
    rag_file_ids: Optional[List[str]] = None,
) -> Tool:
    """
    Create a ready-to-use Vertex AI RAG retrieval Tool for a given corpus.

    Args:
        corpus_id:          The RAG corpus id (not full resource name).
        output_pattern:     Dict describing the expected output shape; attached
                            to the returned Tool as `output_pattern` metadata.
        config:             Optional base VertexRagConfig. If omitted, uses
                            environment-based defaults.
        top_k:              Number of chunks to retrieve per query.
        vector_distance_threshold:
                            Optional distance threshold filter.
        rag_file_ids:       Optional list of specific rag_file IDs to restrict retrieval to.
    """
    cfg = config or get_default_config()
    # Override corpus id for this tool instance
    cfg.rag_corpus_id = corpus_id
    cfg = init_vertex_rag(cfg)

    if not cfg.corpus_name:
        raise ValueError("Failed to construct corpus_name for Vertex RAG Tool.")

    rag_resource = rag.RagResource(
        rag_corpus=cfg.corpus_name,
        rag_file_ids=rag_file_ids or [],
    )

    filter_obj = None
    if vector_distance_threshold is not None:
        filter_obj = rag.utils.resources.Filter(
            vector_distance_threshold=vector_distance_threshold
        )

    retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,
        filter=filter_obj,
    )

    retrieval_spec = rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag_resource],
        ),
        rag_retrieval_config=retrieval_config,
    )

    tool = Tool.from_retrieval(retrieval=retrieval_spec)
    # Attach the desired output pattern as simple metadata for orchestration.
    setattr(tool, "output_pattern", output_pattern)
    return tool


class VertexRagEngine:
    """
    High-level main class for Vertex AI RAG Engine usage in BestBrain.

    Wraps upload/import, retrieval, and Tool creation around a single config.
    """

    def __init__(
        self,
        corpus_id: Optional[str] = None,
        config: Optional[VertexRagConfig] = None,
    ):
        base_cfg = config or get_default_config()
        if corpus_id:
            base_cfg.rag_corpus_id = corpus_id
        self.config = init_vertex_rag(base_cfg)

    # --- Upsert / import ---

    def upload_local_file(
        self,
        path: str,
        display_name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        return upsert.upload_local_file(
            path=path,
            display_name=display_name,
            description=description,
            config=self.config,
            corpus_name=self.config.corpus_name,
        )

    def import_remote_files(
        self,
        paths: List[str],
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        import_result_sink: Optional[str] = None,
        max_embedding_requests_per_min: Optional[int] = None,
    ) -> Dict[str, Any]:
        return upsert.import_remote_files(
            paths=paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            import_result_sink=import_result_sink,
            max_embedding_requests_per_min=max_embedding_requests_per_min,
            config=self.config,
            corpus_name=self.config.corpus_name,
        )

    # --- Retrieval ---

    def list_files(self) -> List[Dict[str, Any]]:
        return retrieval.list_corpus_files(
            config=self.config,
            corpus_name=self.config.corpus_name,
        )

    def retrieval_query(
        self,
        text: str,
        top_k: int = 10,
        rag_file_ids: Optional[List[str]] = None,
        vector_distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        return retrieval.retrieval_query(
            text=text,
            top_k=top_k,
            rag_file_ids=rag_file_ids,
            vector_distance_threshold=vector_distance_threshold,
            config=self.config,
            corpus_name=self.config.corpus_name,
        )

    # --- Tool helper ---

    def create_tool(
        self,
        output_pattern: Dict[str, Any],
        *,
        top_k: int = 10,
        vector_distance_threshold: Optional[float] = 0.5,
        rag_file_ids: Optional[List[str]] = None,
    ) -> Tool:
        """
        Convenience wrapper around the module-level create_tool using this engine's corpus.
        """
        if not self.config.rag_corpus_id:
            raise ValueError(
                "VertexRagEngine.config.rag_corpus_id must be set to create a Tool. "
                "Provide corpus_id when constructing the engine or set VERTEX_RAG_CORPUS_ID."
            )

        return create_tool(
            corpus_id=self.config.rag_corpus_id,
            output_pattern=output_pattern,
            config=self.config,
            top_k=top_k,
            vector_distance_threshold=vector_distance_threshold,
            rag_file_ids=rag_file_ids,
        )

