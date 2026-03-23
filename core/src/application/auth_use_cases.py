from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from domain.entities import User
from domain.exceptions import ConflictError, UnauthorizedError, ValidationError
from domain.ports import (
    JwtProviderPort,
    NotificationPort,
    PasswordHasherPort,
    PasswordResetRepositoryPort,
    ResetTokenGeneratorPort,
    UserRepositoryPort,
)


@dataclass(slots=True)
class AuthTokens:
    access_token: str
    token_type: str = "bearer"
    refresh_token: str | None = None


class SignUpUseCase:
    def __init__(self, users: UserRepositoryPort, hasher: PasswordHasherPort) -> None:
        self.users = users
        self.hasher = hasher

    def execute(self, email: str, password: str, name: str | None = None) -> User:
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters.")
        if self.users.get_by_email(email):
            raise ConflictError("Email already registered.")
        user = User(id=None, email=email.lower().strip(), password_hash=self.hasher.hash(password), name=name)
        return self.users.create(user)


class SignInUseCase:
    def __init__(self, users: UserRepositoryPort, hasher: PasswordHasherPort, jwt: JwtProviderPort) -> None:
        self.users = users
        self.hasher = hasher
        self.jwt = jwt

    def execute(self, email: str, password: str, issue_refresh: bool = True) -> AuthTokens:
        user = self.users.get_by_email(email.lower().strip())
        if not user or not self.hasher.verify(password, user.password_hash):
            raise UnauthorizedError("Invalid credentials.")
        access = self.jwt.create_access_token(str(user.id), {"email": user.email})
        refresh = self.jwt.create_refresh_token(str(user.id)) if issue_refresh else None
        return AuthTokens(access_token=access, refresh_token=refresh)


class RefreshTokenUseCase:
    def __init__(self, jwt: JwtProviderPort) -> None:
        self.jwt = jwt

    def execute(self, refresh_token: str) -> AuthTokens:
        claims = self.jwt.decode_token(refresh_token)
        if claims.get("type") != "refresh":
            raise UnauthorizedError("Invalid refresh token.")
        sub = claims.get("sub")
        if not sub:
            raise UnauthorizedError("Invalid refresh token.")
        return AuthTokens(access_token=self.jwt.create_access_token(sub))


class RequestPasswordResetUseCase:
    def __init__(
        self,
        users: UserRepositoryPort,
        resets: PasswordResetRepositoryPort,
        tokens: ResetTokenGeneratorPort,
        notifier: NotificationPort,
        expiry_minutes: int = 30,
    ) -> None:
        self.users = users
        self.resets = resets
        self.tokens = tokens
        self.notifier = notifier
        self.expiry_minutes = expiry_minutes

    def execute(self, email: str) -> None:
        user = self.users.get_by_email(email.lower().strip())
        if not user:
            return
        token = self.tokens.generate()
        token_hash = self.tokens.hash(token)
        expires_at = datetime.now(UTC) + timedelta(minutes=self.expiry_minutes)
        self.resets.create(user.id or 0, token_hash, expires_at)
        self.notifier.send_password_reset(user.email, token)


class ConfirmPasswordResetUseCase:
    def __init__(
        self,
        users: UserRepositoryPort,
        resets: PasswordResetRepositoryPort,
        tokens: ResetTokenGeneratorPort,
        hasher: PasswordHasherPort,
    ) -> None:
        self.users = users
        self.resets = resets
        self.tokens = tokens
        self.hasher = hasher

    def execute(self, token: str, new_password: str) -> None:
        if len(new_password) < 8:
            raise ValidationError("Password must be at least 8 characters.")
        user_id = self.resets.consume_valid(self.tokens.hash(token), datetime.now(UTC))
        if not user_id:
            raise UnauthorizedError("Invalid or expired reset token.")
        self.users.update_password_hash(user_id, self.hasher.hash(new_password))
