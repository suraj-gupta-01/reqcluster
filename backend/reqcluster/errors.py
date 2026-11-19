"""Application errors mapped to HTTP status and machine code."""


class AppError(Exception):
    status: int = 500
    code: str = "internal"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(AppError):
    status, code = 404, "not_found"


class Invalid(AppError):
    status, code = 400, "invalid"


class TooLarge(AppError):
    status, code = 413, "too_large"


class Conflict(AppError):
    status, code = 409, "conflict"


class Stale(AppError):
    status, code = 409, "stale"


class InsufficientData(AppError):
    status, code = 422, "insufficient_data"


class Unavailable(AppError):
    status, code = 503, "unavailable"


class Cancelled(AppError):
    """Used by the job runner only."""

    status, code = 499, "cancelled"


class Unauthorized(AppError):
    status, code = 401, "unauthorized"
