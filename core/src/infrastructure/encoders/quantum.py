from __future__ import annotations

import threading
from typing import Any

import numpy as np
from sklearn.decomposition import PCA  # type: ignore

from audit import audit_print, category_log, preview_text, preview_vector
from domain.exceptions import ValidationError
from domain.ir import l2_normalize
from infrastructure.encoders.base import SharedSbertBase

try:
    import pennylane as qml  # type: ignore
except Exception:  # pragma: no cover
    qml = None


class QuantumPipelineEncoder:
    """Pipeline 2 – Quantum Semantic (Residual Quantum Feature Map).

    text
    → SentenceTransformer embedding (384-dim)
    → PCA_base (384 → dim=64) → base_vector_64
    → PCA_angles (64 → n_qubits=6) → angle_normalize [0, π]
    → quantum circuit: AngleEmbedding + StronglyEntanglingLayers
    → measurement probabilities (2^n_qubits = 64-dim) → Hellinger transform
    → concat(base_vector_64, quantum_vector_64) → vector_128
    → PCA_final (128 → 64)
    → L2 normalization
    → stored in pgvector (quantum_vector column)

    The residual connection (concatenating the base vector) preserves the
    semantic geometry of the SBERT embedding space while the quantum circuit
    adds a non-linear transformation component.

    The circuit is compiled once at instantiation with fixed random weights (seeded).
    Requires fit() to be called with corpus embeddings before encode().
    """

    def __init__(self, base: SharedSbertBase, n_qubits: int = 6, dim: int = 64, seed: int = 42) -> None:
        if qml is None:
            raise ValidationError(
                "Quantum encoder unavailable: PennyLane is not installed."
            )
        self.base = base
        self.n_qubits = n_qubits
        self.circuit_output_dim = 2 ** n_qubits  # 64 for n_qubits=6
        self.dim = dim  # final output dim after PCA_final
        self.seed = seed
        self._pca_base: PCA | None = None    # 384 → dim
        self._pca_angles: PCA | None = None  # dim → n_qubits (for angle embedding)
        self._pca_final: PCA | None = None   # (dim + circuit_output_dim) → dim
        self._angle_min: np.ndarray | None = None
        self._angle_max: np.ndarray | None = None
        self.is_fitted: bool = False
        self._lock = threading.RLock()

        # Fixed random weights for StronglyEntanglingLayers — seeded for reproducibility.
        rng = np.random.default_rng(seed)
        n_layers = 2
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self._weights = rng.uniform(0.0, 2.0 * np.pi, weight_shape)

        # Compile the circuit once.
        self._qnode = self._build_qnode()
        audit_print(
            "encoder.quantum.init.completed",
            n_qubits=n_qubits,
            circuit_output_dim=self.circuit_output_dim,
            final_dim=dim,
            n_layers=n_layers,
            seed=seed,
        )

    def _build_qnode(self) -> Any:
        dev = qml.device("default.qubit", wires=self.n_qubits)
        weights_ref = self._weights  # captured by closure

        @qml.qnode(dev)
        def circuit(angles: np.ndarray) -> Any:
            qml.AngleEmbedding(angles, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights_ref, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def fit(self, raw_embeddings: np.ndarray) -> None:
        """Fit all three PCA transformations on corpus raw SBERT embeddings.

        Three-stage fitting:
        1. PCA_base: raw_dim → dim  (same as classical pipeline)
        2. PCA_angles: dim → n_qubits  (reduces base vector to circuit angle count)
        3. Run circuit on all corpus documents to build residual concat vectors
        4. PCA_final: (dim + circuit_output_dim) → dim
        """
        audit_print(
            "encoder.quantum.fit.start",
            n_samples=raw_embeddings.shape[0],
            input_dim=raw_embeddings.shape[1],
            base_pca_output_dim=self.dim,
            angle_pca_output_dim=self.n_qubits,
            circuit_output_dim=self.circuit_output_dim,
            final_dim=self.dim,
        )

        # Step 1: PCA_base 384 → dim=64
        pca_base = PCA(n_components=self.dim, random_state=self.seed)
        base_vectors = pca_base.fit_transform(raw_embeddings)  # [N, dim]
        base_explained = float(np.sum(pca_base.explained_variance_ratio_))
        audit_print(
            "encoder.quantum.fit.pca_base_completed",
            output_dim=self.dim,
            explained_variance_ratio=round(base_explained, 4),
        )

        # Step 2: PCA_angles dim=64 → n_qubits=6 (for angle embedding)
        pca_angles = PCA(n_components=self.n_qubits, random_state=self.seed)
        angle_vectors = pca_angles.fit_transform(base_vectors)  # [N, n_qubits]
        angles_explained = float(np.sum(pca_angles.explained_variance_ratio_))
        # Per-component min/max for normalization → [0, π]
        angle_min = angle_vectors.min(axis=0)
        angle_max = angle_vectors.max(axis=0)
        audit_print(
            "encoder.quantum.fit.pca_angles_completed",
            output_dim=self.n_qubits,
            explained_variance_ratio=round(angles_explained, 4),
        )

        # Step 3: Run circuit on all corpus documents to collect quantum vectors
        audit_print(
            "encoder.quantum.fit.circuit_pass.start",
            n_samples=raw_embeddings.shape[0],
        )
        span = np.where(angle_max - angle_min == 0, 1.0, angle_max - angle_min)
        quantum_vectors = []
        for angles in angle_vectors:
            angles_norm = np.pi * (angles - angle_min) / span
            quantum_vectors.append(self._run_circuit(angles_norm))
        quantum_arr = np.array(quantum_vectors, dtype=np.float64)  # [N, circuit_output_dim]
        audit_print(
            "encoder.quantum.fit.circuit_pass.completed",
            n_samples=raw_embeddings.shape[0],
            quantum_dim=self.circuit_output_dim,
        )

        # Step 4: Concat [base_64, quantum_64] → 128, fit PCA_final → 64
        concat_vectors = np.concatenate([base_vectors, quantum_arr], axis=1)  # [N, 128]
        pca_final = PCA(n_components=self.dim, random_state=self.seed)
        pca_final.fit(concat_vectors)
        final_explained = float(np.sum(pca_final.explained_variance_ratio_))
        # Atomic state swap: all components are ready, swap together
        with self._lock:
            self._pca_base = pca_base
            self._pca_angles = pca_angles
            self._angle_min = angle_min
            self._angle_max = angle_max
            self._pca_final = pca_final
            self.is_fitted = True
        audit_print(
            "encoder.quantum.fit.pca_final_completed",
            input_dim=self.dim + self.circuit_output_dim,
            output_dim=self.dim,
            explained_variance_ratio=round(final_explained, 4),
        )
        audit_print(
            "encoder.quantum.fit.completed",
            final_dim=self.dim,
        )

    def _run_circuit(self, angles: np.ndarray) -> list[float]:
        """Run the quantum circuit and return Hellinger-transformed probabilities."""
        try:
            probs = self._qnode(angles)
        except Exception as exc:
            raise RuntimeError("Quantum feature map failed") from exc
        # Hellinger transform: sqrt(probabilities)
        hellinger = np.sqrt(np.abs(np.array(probs, dtype=np.float64)))
        return hellinger.tolist()

    def transform(self, raw_embedding: np.ndarray) -> list[float]:
        """Apply Residual Quantum Feature Map to a single raw embedding.

        text → PCA_base(64) → PCA_angles(6) → circuit → Hellinger(64)
             → concat(base_64, quantum_64=128) → PCA_final(64) → L2 normalize
        """
        with self._lock:
            pca_base = self._pca_base
            pca_angles = self._pca_angles
            pca_final = self._pca_final
            angle_min = self._angle_min
            angle_max = self._angle_max
            is_fitted = self.is_fitted
        if not is_fitted or pca_base is None or pca_angles is None or pca_final is None:
            raise ValidationError(
                "[PIPELINE quantum] Encoder not fitted. Index a dataset first."
            )
        # PCA_base: 384 → 64
        base_vec = pca_base.transform(raw_embedding.reshape(1, -1))[0]
        # PCA_angles: 64 → 6 → normalize [0, π]
        angles = pca_angles.transform(base_vec.reshape(1, -1))[0]
        span = np.where(angle_max - angle_min == 0, 1.0, angle_max - angle_min)
        angles_norm = np.pi * (angles - angle_min) / span
        # Circuit → Hellinger → quantum_64
        quantum_vec = self._run_circuit(angles_norm)
        # Concat [base_64, quantum_64] → 128 → PCA_final → 64 → L2
        concat = np.concatenate([base_vec, np.array(quantum_vec, dtype=np.float64)])
        final = pca_final.transform(concat.reshape(1, -1))[0]
        return l2_normalize(final.tolist())

    def encode(self, text: str) -> list[float]:
        """Full pipeline: text → SBERT → Residual Quantum Feature Map → L2 normalize."""
        audit_print(
            "encoder.quantum.encode.start",
            pipeline="quantum",
            qubits=self.n_qubits,
            circuit_output_dim=self.circuit_output_dim,
            final_dim=self.dim,
            text=preview_text(text),
        )
        raw = self.base.encode_single(text)
        audit_print(
            "encoder.quantum.encode.embedding_dim",
            pipeline="quantum",
            embedding_dim=raw.shape[0],
        )
        result = self.transform(raw)
        audit_print(
            "encoder.quantum.encode.completed",
            pipeline="quantum",
            final_dim=self.dim,
            vector=preview_vector(result),
        )
        category_log("PIPELINE quantum", base_vector_dim=self.dim, quantum_vector_dim=self.circuit_output_dim, concat_dim=self.dim + self.circuit_output_dim, final_vector_dim=self.dim)
        category_log("NORMALIZE", vector_norm=1.0)
        return result

    def save_state(self, path: str) -> None:
        """Persist fitted PCAs and angle normalization params to disk."""
        import os
        import joblib  # type: ignore
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "pca_base": self._pca_base,
            "pca_angles": self._pca_angles,
            "pca_final": self._pca_final,
            "angle_min": self._angle_min,
            "angle_max": self._angle_max,
        }, path)
        audit_print("encoder.quantum.save_state", path=path)

    def load_state(self, path: str) -> bool:
        """Load fitted PCAs and angle normalization params from disk. Returns True if successful."""
        import os
        import joblib  # type: ignore
        if not os.path.exists(path):
            return False
        state = joblib.load(path)
        with self._lock:
            self._pca_base = state["pca_base"]
            self._pca_angles = state["pca_angles"]
            self._pca_final = state["pca_final"]
            self._angle_min = state["angle_min"]
            self._angle_max = state["angle_max"]
            self.is_fitted = True
        audit_print("encoder.quantum.load_state", path=path)
        return True

    def encode_batch_transform(self, raw_embeddings: np.ndarray) -> list[list[float]]:
        """Batch transform pre-computed raw embeddings (used during indexing)."""
        with self._lock:
            pca_base = self._pca_base
            pca_angles = self._pca_angles
            pca_final = self._pca_final
            angle_min = self._angle_min
            angle_max = self._angle_max
            is_fitted = self.is_fitted
        if not is_fitted or pca_base is None or pca_angles is None or pca_final is None:
            raise ValidationError(
                "[PIPELINE quantum] Encoder not fitted. Call fit() first."
            )
        base_vecs = pca_base.transform(raw_embeddings)        # [N, dim]
        angle_vecs = pca_angles.transform(base_vecs)          # [N, n_qubits]
        span = np.where(angle_max - angle_min == 0, 1.0, angle_max - angle_min)
        quantum_vecs = []
        for angles in angle_vecs:
            angles_norm = np.pi * (angles - angle_min) / span
            quantum_vecs.append(self._run_circuit(angles_norm))
        quantum_arr = np.array(quantum_vecs, dtype=np.float64)       # [N, circuit_output_dim]
        concat = np.concatenate([base_vecs, quantum_arr], axis=1)    # [N, dim+circuit_output_dim]
        finals = pca_final.transform(concat)                         # [N, dim]
        return [l2_normalize(row.tolist()) for row in finals]
