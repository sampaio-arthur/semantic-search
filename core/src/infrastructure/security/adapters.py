from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext

from domain.exceptions import UnauthorizedError
from infrastructure.config import Settings


class BcryptPasswordHasher:
    def __init__(self) -> None:
        # bcrypt_sha256 avoids bcrypt's 72-byte input limit while still using bcrypt as backend.
        self._ctx = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")

    def hash(self, password: str) -> str:
        return self._ctx.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        return self._ctx.verify(password, password_hash)


class JoseJwtProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _encode(self, subject: str, expires_minutes: int, token_type: str, extra_claims: dict[str, str] | None = None) -> str:
        payload = {"sub": subject, "type": token_type, "exp": datetime.now(UTC) + timedelta(minutes=expires_minutes)}
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, self.settings.jwt_secret, algorithm=self.settings.jwt_algorithm)

    def create_access_token(self, subject: str, extra_claims: dict[str, str] | None = None) -> str:
        return self._encode(subject, self.settings.access_token_expire_minutes, "access", extra_claims)

    def create_refresh_token(self, subject: str) -> str:
        return self._encode(subject, self.settings.refresh_token_expire_minutes, "refresh")

    def decode_token(self, token: str) -> dict[str, str]:
        try:
            return jwt.decode(token, self.settings.jwt_secret, algorithms=[self.settings.jwt_algorithm])
        except JWTError as exc:
            raise UnauthorizedError("Invalid token.") from exc


class Sha256ResetTokenGenerator:
    def generate(self) -> str:
        return secrets.token_urlsafe(32)

    def hash(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
