from __future__ import annotations

import numpy as np

from audit import audit_print, category_log


class AnswerSimilarityService:
    """Computes semantic similarity between two texts using the shared SBERT model.

    Similarity is calculated as cosine similarity between sentence embeddings,
    resulting in a value between -1 and 1 (in practice 0 to 1 for natural language).

    The same SBERT model used for indexing (all-MiniLM-L6-v2) is reused here
    to ensure consistency in embedding space.
    """

    def __init__(self, base) -> None:
        self._base = base

    def compute(self, text_a: str, text_b: str) -> float:
        """Return cosine similarity between embeddings of text_a and text_b."""
        audit_print(
            "answer_similarity.compute.start",
            text_a_len=len(text_a),
            text_b_len=len(text_b),
        )
        embeddings = self._base.encode_batch([text_a, text_b])
        a: np.ndarray = embeddings[0]
        b: np.ndarray = embeddings[1]
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            audit_print("answer_similarity.compute.zero_norm", norm_a=norm_a, norm_b=norm_b)
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        audit_print("answer_similarity.compute.completed", similarity=round(similarity, 4))
        return similarity
