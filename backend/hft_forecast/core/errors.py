"""Exception hierarchy. Every error raised by Backbone derives from :class:`BackboneError`."""

from __future__ import annotations

from typing import Any


class BackboneError(Exception):
    """Base class for all errors raised by this package.

    Attributes:
        code: Stable machine-readable error code, surfaced by the API.
        message: Human-readable description.
        details: Optional structured context (never secrets).
    """

    code: str = "backbone_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})


class ConfigError(BackboneError):
    """Invalid configuration or run specification."""

    code = "config_error"


class PluginError(BackboneError):
    """Plugin registration or lookup failure."""

    code = "plugin_error"


class DuplicatePluginError(PluginError):
    """Two plugins of the same kind share a name."""

    code = "duplicate_plugin"


class PluginNotFoundError(PluginError):
    """A requested plugin is not registered."""

    code = "plugin_not_found"


class DataError(BackboneError):
    """Data could not be fetched, parsed or found."""

    code = "data_error"


class DataValidationError(DataError):
    """Data failed validation on ingest."""

    code = "data_validation_error"


class DataSourceUnavailableError(DataError):
    """A data source cannot be reached or is not configured."""

    code = "data_source_unavailable"


class LookaheadError(BackboneError):
    """A strategy tried to access data beyond the current bar."""

    code = "lookahead"


class CompatibilityError(ConfigError):
    """A run configuration combines incompatible plugins."""

    code = "incompatible_config"


class EngineError(BackboneError):
    """The backtest engine failed."""

    code = "engine_error"


class NotFoundError(BackboneError):
    """A stored entity (run, preset, job, dataset) was not found."""

    code = "not_found"


class JobCancelledError(BackboneError):
    """A background job was cancelled."""

    code = "job_cancelled"
