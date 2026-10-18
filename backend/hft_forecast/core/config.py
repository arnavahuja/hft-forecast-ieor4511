"""Application settings loaded from environment variables and ``.env``.

Secrets (WRDS passwords) are never stored here; WRDS uses ``~/.pgpass``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
LOCALHOST = "127.0.0.1"


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        data_dir: Root for ``cache/``, ``imports/`` and ``runs/``.
        user_plugins_dir: Directory scanned for user plugins.
        host: Bind address. Always localhost.
        port: API port.
        frontend_origin: Allowed CORS origin for the local front end.
        dev_mode: Enables hot reload of user plugins.
        log_level: Log level name.
        log_json: Emit JSON logs instead of console logs.
        wrds_username: WRDS user name (password comes from ``~/.pgpass``).
        default_calendar: Exchange calendar code for annualization and validation.
        max_workers: Background job worker processes.
        default_benchmark: Default benchmark instrument symbol.
    """

    model_config = SettingsConfigDict(
        env_prefix="HFTF_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    user_plugins_dir: Path = Path("user_plugins")
    host: str = LOCALHOST
    port: int = DEFAULT_PORT
    frontend_origin: str = f"http://{LOCALHOST}:{DEFAULT_FRONTEND_PORT}"
    dev_mode: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    wrds_username: str | None = Field(
        default=None, validation_alias=AliasChoices("WRDS_USERNAME", "HFTF_WRDS_USERNAME")
    )
    default_calendar: str = "XNYS"
    max_workers: int = 2
    default_benchmark: str = "SPY"

    @property
    def cache_dir(self) -> Path:
        """Parquet cache directory."""
        return self.data_dir / "cache"

    @property
    def imports_dir(self) -> Path:
        """Imported local datasets directory."""
        return self.data_dir / "imports"

    @property
    def runs_dir(self) -> Path:
        """Run results directory."""
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        """Create data directories if missing."""
        for path in (self.cache_dir, self.imports_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)
