"""WRDS login helper: store credentials once, connect anywhere without typing them.

One-time setup (run in a terminal; it prompts for your WRDS username and password):
    python src/wrds_login.py

Then in any script / notebook:
    from wrds_login import connect
    db = connect()
    df = db.raw_sql("select * from crsp.dsf limit 5")

`connect()` returns a small built-in client (SQLAlchemy 2 + psycopg2) with the
same method names as the official `wrds` package (raw_sql, get_table,
list_libraries, list_tables, describe_table). The official package is not used
because its current releases pin pandas<2.3, while this project uses pandas 3.

Credentials live in the standard PostgreSQL password file (pgpass), which the
WRDS server connection reads automatically:
    Windows: %APPDATA%\\postgresql\\pgpass.conf
    macOS/Linux: ~/.pgpass
Nothing is stored in your project folder, so it is safe to share/commit code.
"""

import getpass
import os
import sys
from pathlib import Path

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"


def pgpass_path() -> Path:
    if os.name == "nt":
        return Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def _escape(field: str) -> str:
    # pgpass format: ':' and '\' inside a field must be backslash-escaped
    return field.replace("\\", "\\\\").replace(":", "\\:")


def _split(line: str) -> list[str]:
    fields, cur, i = [], "", 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            cur += line[i + 1]
            i += 2
            continue
        if c == ":":
            fields.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    fields.append(cur)
    return fields


def _is_wrds_line(line: str) -> bool:
    f = _split(line.rstrip("\r\n"))
    return len(f) == 5 and f[0] == WRDS_HOST and f[1] == str(WRDS_PORT) and f[2] == WRDS_DB


def saved_username() -> str | None:
    """Return the WRDS username stored in pgpass, or None if not set up."""
    p = pgpass_path()
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if _is_wrds_line(line):
            return _split(line)[3]
    return None


def save_credentials(username: str, password: str) -> Path:
    """Write/replace the WRDS entry in pgpass, keeping any other entries."""
    p = pgpass_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    others = []
    if p.exists():
        others = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not _is_wrds_line(l)]
    entry = ":".join([WRDS_HOST, str(WRDS_PORT), WRDS_DB, _escape(username), _escape(password)])
    p.write_text("\n".join(others + [entry]) + "\n", encoding="utf-8")
    if os.name != "nt":
        p.chmod(0o600)  # libpq ignores the file unless it is private
    return p


_LIBRARY_QUERY = """
WITH pgobjs AS (
    SELECT oid, relnamespace, relkind
    FROM pg_class
    WHERE relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"])
),
schemas AS (
    SELECT nspname AS schemaname, pg_namespace.oid, array_agg(DISTINCT relkind) AS relkind_a
    FROM pg_namespace
    JOIN pgobjs ON pg_namespace.oid = relnamespace
    WHERE nspname !~ '(^pg_)|(_old$)|(_new$)|(information_schema)'
        AND has_schema_privilege(nspname, 'USAGE') = TRUE
    GROUP BY nspname, pg_namespace.oid
)
SELECT schemaname FROM schemas WHERE relkind_a != ARRAY['v'::"char"]
UNION
SELECT nv.schemaname
FROM schemas nv
JOIN pgobjs v ON nv.oid = v.relnamespace AND v.relkind = 'v'::"char"
JOIN pg_depend dv ON v.oid = dv.refobjid AND dv.refclassid = 'pg_class'::regclass::oid
    AND dv.classid = 'pg_rewrite'::regclass::oid AND dv.deptype = 'i'::"char"
JOIN pg_depend dt ON dv.objid = dt.objid AND dv.refobjid <> dt.refobjid
    AND dt.classid = 'pg_rewrite'::regclass::oid AND dt.refclassid = 'pg_class'::regclass::oid
JOIN pgobjs t ON dt.refobjid = t.oid
    AND (t.relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'f'::"char", 'p'::"char"]))
JOIN schemas nt ON t.relnamespace = nt.oid
GROUP BY nv.schemaname
ORDER BY 1
"""


class WRDS:
    """Minimal WRDS client; method names mirror the official `wrds.Connection`."""

    def __init__(self, username: str):
        import sqlalchemy as sa

        self.username = username
        url = sa.engine.URL.create(
            "postgresql+psycopg2", username=username, host=WRDS_HOST, port=WRDS_PORT, database=WRDS_DB
        )  # no password in the URL: libpq reads it from pgpass
        self.engine = sa.create_engine(
            url,
            connect_args={"sslmode": "require", "application_name": f"{sys.platform} python wrds_login"},
        )
        # autocommit so one failed query doesn't poison later ones
        self.connection = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        self._libraries = None

    # -- discovery -----------------------------------------------------------
    def list_libraries(self) -> list[str]:
        if self._libraries is None:
            self._libraries = self.raw_sql(_LIBRARY_QUERY)["schemaname"].tolist()
        return self._libraries

    def list_tables(self, library: str) -> list[str]:
        q = "SELECT table_name FROM information_schema.tables WHERE table_schema = %(lib)s ORDER BY 1"
        return self.raw_sql(q, params={"lib": library})["table_name"].tolist()

    def describe_table(self, library: str, table: str):
        q = """SELECT column_name AS name, data_type AS type, is_nullable AS nullable
               FROM information_schema.columns
               WHERE table_schema = %(lib)s AND table_name = %(tbl)s ORDER BY ordinal_position"""
        return self.raw_sql(q, params={"lib": library, "tbl": table})

    # -- data ----------------------------------------------------------------
    def raw_sql(self, sql: str, params=None, date_cols=None, index_col=None, chunksize=None, **kwargs):
        """Run a SQL query and return a DataFrame.

        Use %(name)s placeholders with `params={"name": value}`.
        With `chunksize`, returns an iterator of DataFrames streamed from the server
        (use this for big TAQ pulls so memory stays bounded).
        """
        import pandas as pd

        kw = dict(params=params, parse_dates=date_cols, index_col=index_col, **kwargs)
        if chunksize:
            return self._stream(sql, chunksize, kw)
        return pd.read_sql_query(sql, self.connection, **kw)

    def _stream(self, sql, chunksize, kw):
        import pandas as pd

        # server-side cursor needs its own transactional (non-autocommit) connection
        with self.engine.connect().execution_options(stream_results=True) as conn:
            yield from pd.read_sql_query(sql, conn, chunksize=chunksize, **kw)

    def get_table(self, library: str, table: str, columns=None, obs=None, offset=None, date_cols=None, **kwargs):
        """SELECT columns FROM library.table [LIMIT obs] [OFFSET offset]."""
        q = lambda s: '"' + s.replace('"', '""') + '"'
        cols = ", ".join(q(c) for c in columns) if columns else "*"
        sql = f"SELECT {cols} FROM {q(library)}.{q(table)}"
        if obs is not None:
            sql += f" LIMIT {int(obs)}"
        if offset:
            sql += f" OFFSET {int(offset)}"
        return self.raw_sql(sql, date_cols=date_cols, **kwargs)

    def close(self):
        self.connection.close()
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(username: str | None = None) -> WRDS:
    """Connect to WRDS using the saved credentials (no prompts)."""
    username = username or saved_username()
    if username is None:
        raise RuntimeError(
            f"No WRDS credentials found in {pgpass_path()}.\n"
            f"Run once in a terminal:  python {Path(__file__).resolve()}"
        )
    return WRDS(username)


def _setup():
    current = saved_username()
    prompt = f"WRDS username [{current}]: " if current else "WRDS username: "
    username = input(prompt).strip() or current
    if not username:
        sys.exit("Username is required.")
    keep = current is not None and username == current
    password = getpass.getpass("WRDS password (hidden" + (", Enter = keep saved" if keep else "") + "): ")
    if password:
        print(f"Saved credentials to {save_credentials(username, password)}")
    elif not keep:
        sys.exit("Password is required.")

    print("Testing connection (approve the Duo push on your phone if one arrives)...")
    db = connect()
    libs = db.list_libraries()
    print(f"Connected as {username}. You can access {len(libs)} libraries.")
    taq = sorted(l for l in libs if l.startswith("taq"))
    print("TAQ libraries:", ", ".join(taq[:15]) + (" ..." if len(taq) > 15 else "") if taq else "none found")
    db.close()


if __name__ == "__main__":
    _setup()
