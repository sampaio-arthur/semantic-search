from __future__ import annotations

import hashlib
import math

import numpy as np

from audit import audit_print, preview_text, preview_vector
from domain.exceptions import ValidationError
from domain.ir import l2_normalize

try:
    import pennylane as qml  # type: ignore
except Exception:  # pragma: no cover
    qml = None


class PennyLaneQuantumEncoder:
    """Deterministic quantum-inspired encoder using a simulated feature map.

    Text -> hashed angles -> AngleEmbedding + entangling layers -> measurement probabilities.
    Output is real float32, fixed dimension = 2**n_qubits, L2-normalized.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._dev = qml.device("default.qubit", wires=n_qubits) if qml else None
        self._qnode = self._build_qnode() if qml else None

    def _angles_from_text(self, text: str) -> np.ndarray:
        values = np.zeros(self.n_qubits, dtype=np.float64)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.n_qubits):
                values[i] += (digest[i] / 255.0) * math.pi
                values[i] += ((digest[i + self.n_qubits] / 255.0) - 0.5) * 0.25
        values = np.mod(values, 2 * math.pi)
        return values

    def _build_qnode(self):
        if not qml:
            return None

        @qml.qnode(self._dev)
        def circuit(angles):
            qml.AngleEmbedding(angles, wires=range(self.n_qubits), rotation="Y")
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RZ(angles[i] / 2.0, wires=i)
            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def encode(self, text: str) -> list[float]:
        if self._qnode is None:
            raise ValidationError("Quantum encoder unavailable: PennyLane is not installed/loaded.")
        audit_print(
            "encoder.quantum.encode.start",
            n_qubits=self.n_qubits,
            dim=self.dim,
            text=preview_text(text),
        )
        angles = self._angles_from_text(text)
        audit_print(
            "encoder.quantum.angles.computed",
            n_qubits=self.n_qubits,
            angles=[round(float(value), 6) for value in angles.tolist()],
        )
        try:
            probs = [float(x) for x in self._qnode(angles).tolist()]
        except Exception as exc:
            raise ValidationError(f"Quantum encoder failed to encode text: {exc}") from exc
        normalized = l2_normalize([float(np.float32(x)) for x in probs])
        audit_print(
            "encoder.quantum.encode.completed",
            n_qubits=self.n_qubits,
            vector=preview_vector(normalized),
        )
        return normalized
