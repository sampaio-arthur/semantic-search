from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD  # type: ignore

from audit import audit_print, preview_text, preview_vector
from domain.exceptions import ValidationError
from domain.ir import l2_normalize
from infrastructure.encoders.base import SharedSbertBase


class StatisticalPipelineEncoder:
    """Pipeline 3 – Statistical Semantic (Matrix Factorization).

    text
    → tokenizer BERT
    → SentenceTransformer embedding (384-dim)
    → PCA dimensionality reduction (→ pca_intermediate_dim)
    → TruncatedSVD (→ vector_dim)
    → L2 normalization
    → stored in pgvector (statistical_vector column)

    PCA centers the embedding space; TruncatedSVD further factorizes the centered
    embedding matrix into a lower-dimensional representation without re-centering.
    This two-stage linear transformation is the distinguishing characteristic of
    this pipeline relative to the classical (PCA only) baseline.

    Requires fit() to be called with corpus embeddings before encode().
    """

    def __init__(
        self,
        base: SharedSbertBase,
        dim: int = 64,
        pca_intermediate_dim: int = 128,
        seed: int = 42,
    ) -> None:
        self.base = base
        self.dim = dim
        self.pca_intermediate_dim = pca_intermediate_dim
        self.seed = seed
        self._pca: PCA | None = None
        self._svd: TruncatedSVD | None = None
        self.is_fitted: bool = False

    def fit(self, raw_embeddings: np.ndarray) -> None:
        """Fit PCA then TruncatedSVD on corpus raw SBERT embeddings. Shape: [N, base_dim]."""
        audit_print(
            "encoder.statistical.fit.start",
            n_samples=raw_embeddings.shape[0],
            input_dim=raw_embeddings.shape[1],
            pca_intermediate_dim=self.pca_intermediate_dim,
            svd_output_dim=self.dim,
        )
        # Step 1: PCA to center and reduce to intermediate dimensionality
        self._pca = PCA(n_components=self.pca_intermediate_dim, random_state=self.seed)
        pca_out = self._pca.fit_transform(raw_embeddings)
        pca_explained = float(np.sum(self._pca.explained_variance_ratio_))
        audit_print(
            "encoder.statistical.fit.pca_completed",
            pca_intermediate_dim=self.pca_intermediate_dim,
            pca_explained_variance_ratio=round(pca_explained, 4),
        )

        # Step 2: TruncatedSVD on PCA-transformed data for final factorization
        self._svd = TruncatedSVD(n_components=self.dim, random_state=self.seed)
        self._svd.fit(pca_out)
        svd_explained = float(np.sum(self._svd.explained_variance_ratio_))
        self.is_fitted = True
        audit_print(
            "encoder.statistical.fit.completed",
            pca_intermediate_dim=self.pca_intermediate_dim,
            svd_output_dim=self.dim,
            svd_explained_variance_ratio=round(svd_explained, 4),
        )

    def transform(self, raw_embedding: np.ndarray) -> list[float]:
        """Apply PCA → TruncatedSVD → L2 normalize to a single raw embedding."""
        if not self.is_fitted or self._pca is None or self._svd is None:
            raise ValidationError(
                "[PIPELINE statistical] Encoder not fitted. Index a dataset first."
            )
        pca_out = self._pca.transform(raw_embedding.reshape(1, -1))
        svd_out = self._svd.transform(pca_out)[0]
        return l2_normalize(svd_out.tolist())

    def encode(self, text: str) -> list[float]:
        """Full pipeline: text → SBERT → PCA → TruncatedSVD → L2 normalize."""
        audit_print(
            "encoder.statistical.encode.start",
            pipeline="statistical",
            text=preview_text(text),
        )
        raw = self.base.encode_single(text)
        audit_print(
            "encoder.statistical.encode.embedding_dim",
            pipeline="statistical",
            embedding_dim=raw.shape[0],
            pca_dim=self.pca_intermediate_dim,
        )
        result = self.transform(raw)
        audit_print(
            "encoder.statistical.encode.completed",
            pipeline="statistical",
            svd_dim=self.dim,
            vector=preview_vector(result),
        )
        return result

    def save_state(self, path: str) -> None:
        """Persist fitted PCA and TruncatedSVD to disk."""
        import os
        import joblib  # type: ignore
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pca": self._pca, "svd": self._svd}, path)
        audit_print("encoder.statistical.save_state", path=path)

    def load_state(self, path: str) -> bool:
        """Load fitted PCA and TruncatedSVD from disk. Returns True if successful."""
        import os
        import joblib  # type: ignore
        if not os.path.exists(path):
            return False
        state = joblib.load(path)
        self._pca = state["pca"]
        self._svd = state["svd"]
        self.is_fitted = True
        audit_print("encoder.statistical.load_state", path=path)
        return True

    def encode_batch_transform(self, raw_embeddings: np.ndarray) -> list[list[float]]:
        """Batch transform pre-computed raw embeddings (used during indexing)."""
        if not self.is_fitted or self._pca is None or self._svd is None:
            raise ValidationError(
                "[PIPELINE statistical] Encoder not fitted. Call fit() first."
            )
        pca_out = self._pca.transform(raw_embeddings)
        svd_out = self._svd.transform(pca_out)
        return [l2_normalize(row.tolist()) for row in svd_out]
