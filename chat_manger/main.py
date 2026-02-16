"""
AIChatClassifier: classifies user intent to relay case names using local embeddings + similarity.
No Gem/Gemma; embeds the user query and each case label, returns the case with highest similarity.
"""
from __future__ import annotations

import os
from typing import Optional

from gem_core.gem import Gem

# Default model: runs locally, no API key. Small and fast.
#DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class AIChatClassifier:
    """
    Classifies user intent to one of the relay case names by embedding the query
    and the case labels locally, then picking the case with highest similarity score.
    No API key or cloud calls.
    """
    def __init__(
        self,
        case_struct: list,
        cfg_creator=None,
        embedding_model: str = None,
        min_similarity: float = None,
        gem=None
    ):
        self.cfg_creator = cfg_creator
        self.case_struct = case_struct or []
        #self.embedding_model_name = embedding_model or os.environ.get("AI_CHAT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.min_similarity = min_similarity if min_similarity is not None else float(os.environ.get("AI_CHAT_MIN_SIMILARITY", "0.0"))
        self.gem=gem or Gem()

        # Build (case_name, text_for_embedding) e.g. "SET_FIELD", "SET_FIELD Set Field"
        self._case_items: dict[str, str] = {}
        self.filtered_case: dict[str, str] = {}
        for item in self.case_struct:
            case_name = item.get("case") if isinstance(item, dict) else getattr(item, "case", None)
            desc = (item.get("desc") or "") if isinstance(item, dict) else (getattr(item, "description", None) or "")
            if case_name:
                self.filtered_case[case_name] = desc
                text = f"{case_name} {desc}".strip()
                self._case_items[case_name] = text

        self.user_history: dict[str, list] = {}
        self.goal: Optional[str] = None


    def main(self, user_id: str, user_input: str) -> Optional[str]:
        print("ai_chat_classifier.main...")
        self._update_history(user_id, user_input)
        classification = self._classify_input(user_input)
        print("ai_chat_classifier.main... done")
        return classification

    def _update_history(self, user_id: str, message: str) -> None:
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id   ].append(message)

    def _classify_input(self, user_input: str) -> Optional[str]:
        """Classify by embedding user_input and returning the case with highest similarity to its label."""
        print("[AIChatClassifier] _classify_input: start")
        if not self._case_items:
            print("[AIChatClassifier] _classify_input: no cases configured")
            return None

        query = (user_input or "").strip()

        if not query:
            print("[AIChatClassifier] _classify_input: empty user input")
            return None
        try:
            """
            import numpy as np
            q_emb = np.array(embed(query), dtype=np.float64).flatten()
            case_embeddings = list(self._case_items.values())
            # Cosine similarity: dot product of normalized vectors (don't use embedder.similarity - it's lru_cached and ndarray is unhashable)
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
            case_matrix = np.vstack(case_embeddings)
            case_matrix = case_matrix / (np.linalg.norm(case_matrix, axis=1, keepdims=True) + 1e-12)

            scores = case_matrix @ q_norm
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score < self.min_similarity:
                print(f"[AIChatClassifier] _classify_input: best score {best_score:.4f} below min {self.min_similarity}")
            return None
            """

            case_name = self.gem.ask(
                f"""
                Classifc the user input to one of theprovided cases based on semantic meaning and its dscription.
                return jus the single case (in front of anything)
                
                CASE:DESCRIPTION:
                {self._case_items}
                
                
                USER INPUT:
                {user_input}
                
                
                retunr just eh single stinrg no explanation or anything else
                """
            )
            return case_name
        except Exception as e:
            print(f"[AIChatClassifier] _classify_input: error={e}")
            import traceback
            traceback.print_exc()
        return None
