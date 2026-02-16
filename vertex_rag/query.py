"""
Vertex AI RAG Engine – ask questions.
Run retrieval over the corpus and optionally generate an LLM answer using retrieved context.
"""
from typing import Any, Dict, List, Optional

from vertexai import rag
from vertexai.generative_models import GenerativeModel

from .client import init_vertex_rag
from .config import VertexRagConfig
from . import retrieval


def _resolve_corpus(
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> str:
    cfg = init_vertex_rag(config)
    final = corpus_name or cfg.corpus_name
    if not final:
        raise ValueError(
            "No corpus_name provided and no rag_corpus_id configured. "
            "Set VERTEX_RAG_CORPUS_ID or pass corpus_name."
        )
    return final


def ask_question(
    question: str,
    *,
    top_k: int = 10,
    rag_file_ids: Optional[List[str]] = None,
    vector_distance_threshold: Optional[float] = None,
    generate_answer: bool = True,
    model: str = "gemini-2.0-flash",
    config: Optional[VertexRagConfig] = None,
    corpus_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask a question over the RAG corpus: run retrieval, then optionally generate an LLM answer.

    - If generate_answer is False, returns only retrieval contexts.
    - If generate_answer is True, builds a prompt with the retrieved contexts and returns
      both contexts and the model's answer (text).
    """
    final_corpus = _resolve_corpus(config, corpus_name)
    out = retrieval.retrieval_query(
        text=question,
        top_k=top_k,
        rag_file_ids=rag_file_ids,
        vector_distance_threshold=vector_distance_threshold,
        config=config,
        corpus_name=final_corpus,
    )
    if not generate_answer:
        return out

    contexts = out.get("contexts") or []
    parts = []
    for c in contexts:
        if isinstance(c, dict):
            parts.append(c.get("chunk", str(c)))
        else:
            parts.append(getattr(c, "chunk", str(c)))
    context_block = "\n\n".join(parts)
    prompt = f"Use the following context to answer the question. If the context does not contain enough information, say so.\n\nContext:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    gen_model = GenerativeModel(model)
    response = gen_model.generate_content(prompt)
    answer = response.text if response and hasattr(response, "text") else ""
    out["answer"] = answer
    return out
