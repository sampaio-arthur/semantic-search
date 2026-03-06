from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover
    class BaseSettings(BaseModel):  # type: ignore
        def __init__(self, **data):
            env_values = {}
            model_fields = getattr(self.__class__, "model_fields", {})
            for field_name in model_fields:
                env_name = field_name.upper()
                raw = os.getenv(env_name)
                if raw is not None:
                    env_values[field_name] = raw
            env_values.update(data)
            super().__init__(**env_values)

    def SettingsConfigDict(**kwargs):  # type: ignore
        return kwargs


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    app_name: str = "quantum-semantic-search"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    database_url: str = ""
    db_scheme: str = "postgresql+psycopg"
    db_host: str = "db"
    db_port: int = 5432
    db_name: str = "tcc"
    db_user: str = "tcc"
    db_password: str = "tcc"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 60 * 24 * 7

    # Shared final vector dimensionality for all three pipelines.
    # Classical:    BERT(384) -> PCA(vector_dim)
    # Quantum:      BERT(384) -> PCA_base(vector_dim) -> PCA_angles(quantum_n_qubits)
    #               -> circuit -> probs(2^quantum_n_qubits == vector_dim) -> Hellinger
    #               -> concat(2*vector_dim) -> PCA_final(vector_dim)
    # Statistical:  BERT(384) -> PCA(pca_intermediate_dim) -> TruncatedSVD(vector_dim)
    vector_dim: int = 64

    # Number of qubits for quantum pipeline.
    # Must satisfy 2^quantum_n_qubits == vector_dim (circuit output dim == final dim).
    quantum_n_qubits: int = 6  # 2^6 = 64

    # Intermediate PCA dimension for statistical pipeline before TruncatedSVD.
    pca_intermediate_dim: int = 128

    classical_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42

    password_reset_expire_minutes: int = 30

    require_auth_for_indexing: bool = True
    require_admin_for_indexing: bool = False

    # Directory where fitted encoder state (PCA/SVD) is persisted across restarts.
    # Inside the container /app maps to ./core on the host (bind mount).
    encoder_state_dir: str = "/app/data/encoder_state"

    def __init__(self, **data):
        super().__init__(**data)

        if (self.database_url or "").strip():
            resolved_database_url = self.database_url
        else:
            user = quote_plus(self.db_user)
            password = quote_plus(self.db_password)
            resolved_database_url = f"{self.db_scheme}://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"

        self.database_url = resolved_database_url
        if not self.database_url.lower().startswith("postgresql"):
            raise ValueError("Only PostgreSQL is supported. Configure DATABASE_URL or DB_* with a postgresql URL.")
        if self.app_env.lower() not in {"dev", "development", "test"} and self.jwt_secret.strip() == "change-me":
            raise ValueError("JWT_SECRET must be configured in non-development environments.")
        if 2 ** self.quantum_n_qubits != self.vector_dim:
            raise ValueError(
                f"quantum_n_qubits={self.quantum_n_qubits} inconsistent with vector_dim={self.vector_dim}. "
                f"Require 2^quantum_n_qubits == vector_dim."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
