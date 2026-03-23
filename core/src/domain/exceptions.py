class DomainError(Exception):
    """Base domain/application error."""


class NotFoundError(DomainError):
    pass


class UnauthorizedError(DomainError):
    pass


class ConflictError(DomainError):
    pass


class ValidationError(DomainError):
    pass
