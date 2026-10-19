"""Shared WRDS connection handling for the CRSP and TAQ sources.

Credentials: the WRDS user name comes from ``WRDS_USERNAME`` in ``.env``; the password is read
by the ``wrds`` library from ``~/.pgpass``. Nothing secret is ever stored by this package.
"""

from __future__ import annotations

import os
from abc import ABC
from pathlib import Path
from typing import Any

import pandas as pd

from hft_forecast.core.errors import DataSourceUnavailableError
from hft_forecast.core.interfaces import DataSource


def _pgpass_path() -> Path:
    return Path(os.environ.get("PGPASSFILE", Path.home() / ".pgpass"))


class WrdsConnection(DataSource, ABC):
    """A lazily opened WRDS session, with the credential check the Settings page uses."""

    _connection: Any = None

    def available(self) -> tuple[bool, str]:
        """WRDS needs the ``wrds`` package, a user name and ``~/.pgpass``."""
        try:
            import wrds  # noqa: F401
        except ImportError:
            return False, "Install the optional extra: pip install -e '.[wrds]'"
        if not self.env.wrds_username:
            return False, "Set WRDS_USERNAME in .env"
        if not _pgpass_path().exists():
            return False, (
                '~/.pgpass not found: run `python -c "import wrds; '
                'wrds.Connection()"` once in a terminal to create it'
            )
        return True, ""

    def _connect(self) -> Any:
        ok, reason = self.available()
        if not ok:
            raise DataSourceUnavailableError(reason)
        if self._connection is None:
            import wrds

            self._connection = wrds.Connection(wrds_username=self.env.wrds_username)
        return self._connection

    def _query(
        self, sql: str, params: dict[str, Any] | None = None, date_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Run SQL on WRDS (isolated so tests can substitute recorded responses)."""
        frame: pd.DataFrame = self._connect().raw_sql(sql, params=params, date_cols=date_cols)
        return frame

    def test_connection(self) -> dict[str, Any]:
        """Connection check used by the Settings page."""
        try:
            self._query("select 1 as ok")
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "message": f"Connected as {self.env.wrds_username}"}
