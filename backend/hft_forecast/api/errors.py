"""One structured error format for every response: ``{code, message, details}``.

Stack traces are never returned; unexpected errors are logged server-side.
"""

from __future__ import annotations

from typing import Any, Final

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from hft_forecast.core.errors import (
    BackboneError,
    CompatibilityError,
    ConfigError,
    DataSourceUnavailableError,
    DataValidationError,
    NotFoundError,
    PluginNotFoundError,
)

log = structlog.get_logger(__name__)

HTTP_BAD_REQUEST: Final = 400
HTTP_NOT_FOUND: Final = 404
HTTP_UNPROCESSABLE: Final = 422
HTTP_INTERNAL: Final = 500
HTTP_UNAVAILABLE: Final = 503


class ErrorBody(BaseModel):
    """Error response body."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


def status_for(exc: BackboneError) -> int:
    """HTTP status code for a :class:`BackboneError`."""
    if isinstance(exc, NotFoundError | PluginNotFoundError):
        return HTTP_NOT_FOUND
    if isinstance(exc, CompatibilityError | DataValidationError):
        return HTTP_UNPROCESSABLE
    if isinstance(exc, DataSourceUnavailableError):
        return HTTP_UNAVAILABLE
    if isinstance(exc, ConfigError):
        return HTTP_BAD_REQUEST
    return HTTP_BAD_REQUEST


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def install_error_handlers(app: FastAPI) -> None:
    """Register exception handlers producing :class:`ErrorBody` responses."""

    @app.exception_handler(BackboneError)
    async def backbone_error(_: Request, exc: BackboneError) -> JSONResponse:
        body = ErrorBody(code=exc.code, message=exc.message, details=_json_safe(exc.details))
        return JSONResponse(body.model_dump(), status_code=status_for(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": [str(x) for x in e.get("loc", ())], "msg": str(e.get("msg", ""))}
            for e in exc.errors()
        ]
        body = ErrorBody(
            code="validation_error", message="Invalid request", details={"errors": errors}
        )
        return JSONResponse(body.model_dump(), status_code=HTTP_UNPROCESSABLE)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        body = ErrorBody(code=f"http_{exc.status_code}", message=str(exc.detail))
        return JSONResponse(body.model_dump(), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", error=str(exc))
        body = ErrorBody(code="internal_error", message="Internal server error")
        return JSONResponse(body.model_dump(), status_code=HTTP_INTERNAL)
