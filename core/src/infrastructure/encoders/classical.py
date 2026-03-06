from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA  # type: ignore

from audit import audit_print, category_log, preview_text, preview_vector
from domain.exceptions import ValidationError
from domain.ir import l2_normalize
from infrastructure.encoders.base import SharedSbertBase


class ClassicalPipelineEncoder:
    """Pipeline 1 – Classical Semantic (Baseline).

    text
    → tokenizer BERT
    → SentenceTransformer embedding (384-dim)
    → PCA dimensionality reduction (→ vector_dim)
    → L2 normalization
    → stored in pgvector (embedding_vector column)

    Requires fit() to be called with corpus embeddings before encode().
    """

    def __init__(self, base: SharedSbertBase, dim: int = 64, seed: int = 42) -> None:
        self.base = base
        self.dim = dim
        self.seed = seed
        self._pca: PCA | None = None
        self.is_fitted: bool = False

    def fit(self, raw_embeddings: np.ndarray) -> None:
        """Fit PCA on corpus raw SBERT embeddings. Shape: [N, base_dim]."""
        audit_print(
            "encoder.classical.fit.start",
            n_samples=raw_embeddings.shape[0],
            input_dim=raw_embeddings.shape[1],
            output_dim=self.dim,
        )
        self._pca = PCA(n_components=self.dim, random_state=self.seed)
        self._pca.fit(raw_embeddings)
        self.is_fitted = True
        explained = float(np.sum(self._pca.explained_variance_ratio_))
        audit_print(
            "encoder.classical.fit.completed",
            output_dim=self.dim,
            explained_variance_ratio=round(explained, 4),
        )
        category_log("PCA", input_dim=raw_embeddings.shape[1], output_dim=self.dim, pipeline="classical")

    def transform(self, raw_embedding: np.ndarray) -> list[float]:
        """Apply PCA transform to a single raw embedding. Requires fit()."""
        if not self.is_fitted or self._pca is None:
            raise ValidationError(
                "[PIPELINE classical] Encoder not fitted. Index a dataset first."
            )
        reduced = self._pca.transform(raw_embedding.reshape(1, -1))[0]
        return l2_normalize(reduced.tolist())

    def encode(self, text: str) -> list[float]:
        """Full pipeline: text → SBERT → PCA → L2 normalize."""
        audit_print(
            "encoder.classical.encode.start",
            pipeline="classical",
            text=preview_text(text),
        )
        raw = self.base.encode_single(text)
        audit_print(
            "encoder.classical.encode.embedding_dim",
            pipeline="classical",
            embedding_dim=raw.shape[0],
        )
        result = self.transform(raw)
        audit_print(
            "encoder.classical.encode.completed",
            pipeline="classical",
            pca_dim=self.dim,
            vector=preview_vector(result),
        )
        category_log("PIPELINE classical", final_vector_dim=self.dim)
        category_log("NORMALIZE", vector_norm=1.0)
        return result

    def save_state(self, path: str) -> None:
        """Persist fitted PCA to disk so state survives container restarts."""
        import os
        import joblib  # type: ignore
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pca": self._pca}, path)
        audit_print("encoder.classical.save_state", path=path)

    def load_state(self, path: str) -> bool:
        """Load fitted PCA from disk. Returns True if successful."""
        import os
        import joblib  # type: ignore
        if not os.path.exists(path):
            return False
        state = joblib.load(path)
        self._pca = state["pca"]
        self.is_fitted = True
        audit_print("encoder.classical.load_state", path=path)
        return True

    def encode_batch_transform(self, raw_embeddings: np.ndarray) -> list[list[float]]:
        """Batch transform pre-computed raw embeddings (used during indexing)."""
        if not self.is_fitted or self._pca is None:
            raise ValidationError(
                "[PIPELINE classical] Encoder not fitted. Call fit() first."
            )
        reduced = self._pca.transform(raw_embeddings)
        return [l2_normalize(row.tolist()) for row in reduced]
