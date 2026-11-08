"""Run `hftf ...` with ONE WRDS connection per process instead of one per month.

Why: the shared source opens a new `wrds.Connection` for every month it fetches. WRDS accounts
on Duo push get a push for every PostgreSQL connection, so an unattended pull fails with
"PAM authentication failed" as soon as a push is not approved. This wrapper patches the
connection to be opened once and reused, and reopens it (one push) only if it drops.
It also makes a failed login raise at once instead of waiting on an interactive prompt.

Usage (same arguments as hftf):
    python backend/hft_forecast/ml/horizons/h5m/scripts/pull_taq_onecon.py data pull-taq ...

The shared code is not modified; the patch lives only in this process.
"""

from __future__ import annotations

import builtins
import getpass
import sys
from typing import Any

from hft_forecast.core.errors import DataSourceUnavailableError
from hft_forecast.data.sources import wrds_connection as wc

_shared: dict[str, Any] = {}


def _no_prompt(*_args: Any, **_kwargs: Any) -> str:
    raise DataSourceUnavailableError(
        "WRDS login was rejected and asked for interactive input "
        "(Duo push not approved, or pgpass wrong)"
    )


# the wrds library falls back to input()/getpass() when a login fails; never block on that
builtins.input = _no_prompt
getpass.getpass = _no_prompt


def _connect(self: wc.WrdsConnection) -> Any:
    ok, reason = self.available()
    if not ok:
        raise DataSourceUnavailableError(reason)
    conn = _shared.get("conn")
    if conn is None:
        import wrds

        try:
            conn = wrds.Connection(wrds_username=self.env.wrds_username)
        except Exception as exc:
            # stop the whole process: retrying month by month would only send more Duo pushes;
            # resume_pull.ps1 restarts it after a pause
            print(f"WRDS LOGIN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            sys.stderr.flush()
            import os

            os._exit(3)
        _shared["conn"] = conn
    return conn


_DROPPED = ("server closed", "connection already closed", "terminating connection",
            "could not receive data", "ssl syscall", "connection reset", "eof detected")


def _query(self: wc.WrdsConnection, sql: str, params: dict[str, Any] | None = None,
           date_cols: list[str] | None = None) -> Any:
    try:
        return self._connect().raw_sql(sql, params=params, date_cols=date_cols)
    except Exception as exc:  # reopen once if the shared connection died
        if not any(s in str(exc).lower() for s in _DROPPED):
            raise
        try:
            _shared.pop("conn").close()
        except Exception:
            pass
        return self._connect().raw_sql(sql, params=params, date_cols=date_cols)


wc.WrdsConnection._connect = _connect  # type: ignore[method-assign]
wc.WrdsConnection._query = _query  # type: ignore[method-assign]

if __name__ == "__main__":
    from hft_forecast.cli import app

    sys.argv = ["hftf", *sys.argv[1:]]
    app()
