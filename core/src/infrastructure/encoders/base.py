from __future__ import annotations

from typing import Any

import numpy as np

from audit import audit_print, category_log
from domain.exceptions import ValidationError

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

# Module-level cache so all pipeline encoders share a single loaded model.
_MODEL_CACHE: dict[str, Any] = {}


class SharedSbertBase:
    """Loads and caches a single SentenceTransformer model shared by all pipeline encoders.

    Returns raw (un-normalized) BERT embeddings of shape [N, base_dim].
    Normalization is always the responsibility of each pipeline encoder at its final step.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_dim: int = 384  # updated after first load

    def _load(self) -> Any:
        global _MODEL_CACHE
        if self.model_name not in _MODEL_CACHE:
            audit_print("encoder.base.load_model.start", model_name=self.model_name)
            if SentenceTransformer is None:
                raise ValidationError(
                    "Base encoder unavailable: sentence-transformers is not installed."
                )
            try:
                model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise ValidationError(
                    f"Failed to load base encoder model '{self.model_name}': {exc}"
                ) from exc
            inferred_dim = int(model.get_sentence_embedding_dimension())
            self.base_dim = inferred_dim
            _MODEL_CACHE[self.model_name] = model
            audit_print(
                "encoder.base.load_model.completed",
                model_name=self.model_name,
                base_dim=self.base_dim,
            )
        return _MODEL_CACHE[self.model_name]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return raw SBERT embeddings for a batch of texts. Shape: [N, base_dim].

        normalize_embeddings=False to preserve variance for PCA fitting.
        """
        model = self._load()
        audit_print(
            "encoder.base.encode_batch.start",
            model_name=self.model_name,
            batch_size=len(texts),
        )
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        self.base_dim = vectors.shape[1]
        audit_print(
            "encoder.base.encode_batch.completed",
            model_name=self.model_name,
            batch_size=len(texts),
            embedding_dim=self.base_dim,
        )
        category_log("BASE", embedding_dim=self.base_dim)
        return vectors.astype(np.float64)

    def encode_single(self, text: str) -> np.ndarray:
        """Return raw SBERT embedding for a single text. Shape: [base_dim]."""
        return self.encode_batch([text])[0]
