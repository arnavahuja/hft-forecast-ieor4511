"""Run store: metadata in SQLite (``runs/hft_forecast.db``), bulk results as Parquet.

Stored per run: config, config hash, git commit, plugin versions, data cache keys, timings,
logs, status, metrics and results. Runs can be named, tagged, annotated, grouped into
experiments, cloned, archived and deleted. Presets store reusable configs. The number of
runs per experiment feeds the deflated Sharpe ratio.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core.errors import NotFoundError
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.specs import MetricValue
from hft_forecast.services.result_io import META, load_result, save_result

DB_FILE: Final = "hft_forecast.db"
METRICS_FILE: Final = "metrics.json"
ML_METRICS_FILE: Final = "ml_metrics.json"
LOG_FILE: Final = "log.txt"
RESEARCH_FILE: Final = "research.json"
FACTORS_FILE: Final = "factors.parquet"

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    strategy TEXT NOT NULL,
    engine TEXT NOT NULL,
    git_commit TEXT,
    plugin_versions_json TEXT NOT NULL DEFAULT '{}',
    data_refs_json TEXT NOT NULL DEFAULT '{}',
    timings_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    experiment TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    headline_json TEXT NOT NULL DEFAULT '{}',
    parent_id TEXT,
    job_id TEXT,
    kind TEXT NOT NULL DEFAULT 'backtest'
);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE TABLE IF NOT EXISTS experiments (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS research_trials (
    experiment TEXT NOT NULL,
    source_id TEXT NOT NULL,
    params_json TEXT NOT NULL,
    sharpe_per_period REAL
);
CREATE TABLE IF NOT EXISTS test_unlocks (
    experiment TEXT NOT NULL,
    unlocked_at TEXT NOT NULL,
    run_id TEXT
);
"""

HEADLINE_KEYS: Final = (
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "turnover_ann",
    "n_trades",
    "sharpe_per_period",
)


class RunStatus(StrEnum):
    """Lifecycle of a run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRecord(BaseModel):
    """Stored run metadata."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    name: str = ""
    status: RunStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    config: dict[str, Any]
    config_hash: str
    strategy: str
    engine: str
    git_commit: str | None = None
    plugin_versions: dict[str, str] = Field(default_factory=dict)
    data_refs: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    experiment: str | None = None
    archived: bool = False
    error: str | None = None
    headline: dict[str, float | None] = Field(default_factory=dict)
    parent_id: str | None = None
    job_id: str | None = None
    kind: str = "backtest"


class Preset(BaseModel):
    """A saved configuration."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    name: str
    config: dict[str, Any]
    created_at: str
    updated_at: str
    notes: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        name=row["name"],
        status=RunStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        config=json.loads(row["config_json"]),
        config_hash=row["config_hash"],
        strategy=row["strategy"],
        engine=row["engine"],
        git_commit=row["git_commit"],
        plugin_versions=json.loads(row["plugin_versions_json"]),
        data_refs=json.loads(row["data_refs_json"]),
        timings=json.loads(row["timings_json"]),
        tags=json.loads(row["tags_json"]),
        notes=row["notes"],
        experiment=row["experiment"],
        archived=bool(row["archived"]),
        error=row["error"],
        headline=json.loads(row["headline_json"]),
        parent_id=row["parent_id"],
        job_id=row["job_id"],
        kind=row["kind"],
    )


class RunStore:
    """SQLite + Parquet run store.

    Args:
        root: The ``runs`` directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.root / DB_FILE, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def run_dir(self, run_id: str) -> Path:
        """Directory holding a run's results."""
        return self.root / run_id

    # ---- create / update

    def create(
        self,
        config: BacktestConfig,
        *,
        parent_id: str | None = None,
        job_id: str | None = None,
        kind: str = "backtest",
    ) -> RunRecord:
        """Insert a queued run."""
        run_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._connect() as con:
            if config.experiment:
                con.execute(
                    "INSERT OR IGNORE INTO experiments(name, created_at) VALUES (?, ?)",
                    (config.experiment, now),
                )
            con.execute(
                "INSERT INTO runs(id, name, status, created_at, config_json, config_hash, "
                "strategy, engine, tags_json, notes, experiment, parent_id, job_id, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    config.name,
                    RunStatus.QUEUED.value,
                    now,
                    config.model_dump_json(),
                    config.config_hash(),
                    config.strategy.name,
                    config.execution.engine,
                    json.dumps(config.tags),
                    config.notes,
                    config.experiment,
                    parent_id,
                    job_id,
                    kind,
                ),
            )
        return self.get(run_id)

    def create_ml(
        self,
        *,
        name: str,
        config: dict[str, Any],
        config_hash: str,
        experiment: str,
        model: str,
        tags: list[str],
        notes: str = "",
    ) -> RunRecord:
        """Insert a queued ML run. Same table as a backtest, ``kind='ml'``.

        An ML run has no ``BacktestConfig``, so it cannot go through :meth:`create`; the
        economic test (which does have one) is registered as a separate backtest run.
        """
        run_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO experiments(name, created_at) VALUES (?, ?)",
                (experiment, now),
            )
            con.execute(
                "INSERT INTO runs(id, name, status, created_at, config_json, config_hash, "
                "strategy, engine, tags_json, notes, experiment, kind) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ml')",
                (
                    run_id,
                    name,
                    RunStatus.QUEUED.value,
                    now,
                    json.dumps(config, default=str),
                    config_hash,
                    model,
                    "none",
                    json.dumps(tags),
                    notes,
                    experiment,
                ),
            )
        return self.get(run_id)

    def complete_ml(
        self,
        run_id: str,
        metrics: dict[str, Any],
        manifest_path: Path,
        *,
        headline: dict[str, float | None] | None = None,
        git_commit: str | None = None,
        timings: dict[str, float] | None = None,
    ) -> RunRecord:
        """Finish an ML run: metrics without a ``BacktestResult``.

        The run's result slot stays empty until the economic test fills it; faking an empty
        result so :meth:`complete` could be reused would put a meaningless equity curve in
        the UI.
        """
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ML_METRICS_FILE).write_text(json.dumps(metrics, indent=1, default=str))
        flat = headline if headline is not None else _scalars(metrics)
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, finished_at=?, git_commit=?, timings_json=?, "
                "headline_json=?, data_refs_json=?, error=NULL WHERE id=?",
                (
                    RunStatus.COMPLETED.value,
                    _now(),
                    git_commit,
                    json.dumps(timings or {}),
                    json.dumps(flat, default=str),
                    json.dumps({"manifest": str(manifest_path)}),
                    run_id,
                ),
            )
        return self.get(run_id)

    def mark_running(self, run_id: str) -> None:
        """Set status to running."""
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, started_at=? WHERE id=?",
                (RunStatus.RUNNING.value, _now(), run_id),
            )

    def mark_failed(self, run_id: str, error: str, status: RunStatus = RunStatus.FAILED) -> None:
        """Set status to failed (or cancelled) with an error message."""
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, finished_at=?, error=? WHERE id=?",
                (status.value, _now(), error, run_id),
            )

    def complete(
        self,
        run_id: str,
        result: BacktestResult,
        metrics: dict[str, MetricValue],
        *,
        git_commit: str | None,
        plugin_versions: dict[str, str],
        data_refs: dict[str, Any],
        timings: dict[str, float],
        factors: pl.DataFrame | None = None,
    ) -> RunRecord:
        """Persist results (and factor returns, if any) and mark the run completed."""
        directory = self.run_dir(run_id)
        save_result(result, directory)
        if factors is not None:
            factors.write_parquet(directory / FACTORS_FILE)
        (directory / METRICS_FILE).write_text(
            json.dumps({k: v.model_dump(mode="json") for k, v in metrics.items()}, indent=1)
        )
        headline = {k: metrics[k].value for k in HEADLINE_KEYS if k in metrics}
        headline["sharpe_per_period"] = _per_period_sharpe(
            headline.get("sharpe"), result.periods_per_year
        )
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, finished_at=?, git_commit=?, plugin_versions_json=?, "
                "data_refs_json=?, timings_json=?, headline_json=?, error=NULL WHERE id=?",
                (
                    RunStatus.COMPLETED.value,
                    _now(),
                    git_commit,
                    json.dumps(plugin_versions),
                    json.dumps(data_refs, default=str),
                    json.dumps(timings),
                    json.dumps(headline),
                    run_id,
                ),
            )
        return self.get(run_id)

    def ml_metrics(self, run_id: str) -> dict[str, Any]:
        """Statistical metrics of an ML run (not the ``MetricValue`` map of a backtest).

        Raises:
            NotFoundError: If the run has none.
        """
        path = self.run_dir(run_id) / ML_METRICS_FILE
        if not path.exists():
            raise NotFoundError(f"Run '{run_id}' has no ML metrics")
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def update_labels(
        self,
        run_id: str,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        experiment: str | None = None,
        archived: bool | None = None,
    ) -> RunRecord:
        """Rename, tag, annotate, move to an experiment or archive a run."""
        self.get(run_id)
        sets: list[str] = []
        params: list[Any] = []
        for col, value in (("name", name), ("notes", notes), ("experiment", experiment)):
            if value is not None:
                sets.append(f"{col}=?")
                params.append(value)
        if tags is not None:
            sets.append("tags_json=?")
            params.append(json.dumps(tags))
        if archived is not None:
            sets.append("archived=?")
            params.append(int(archived))
        if sets:
            with self._connect() as con:
                con.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", (*params, run_id))
        return self.get(run_id)

    def append_log(self, run_id: str, line: str) -> None:
        """Append a line to the run log."""
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / LOG_FILE).open("a") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def logs(self, run_id: str) -> str:
        """Run log text."""
        path = self.run_dir(run_id) / LOG_FILE
        return path.read_text() if path.exists() else ""

    def delete(self, run_id: str) -> None:
        """Delete a run and its results."""
        self.get(run_id)
        with self._connect() as con:
            con.execute("DELETE FROM runs WHERE id=?", (run_id,))
        shutil.rmtree(self.run_dir(run_id), ignore_errors=True)

    # ---- read

    def get(self, run_id: str) -> RunRecord:
        """Fetch one run."""
        with self._connect() as con:
            row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Run '{run_id}' not found")
        return _row_to_record(row)

    def list_runs(
        self,
        *,
        search: str | None = None,
        experiment: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        include_archived: bool = False,
        kind: str | None = "backtest",
        limit: int = 500,
    ) -> list[RunRecord]:
        """List runs, newest first, with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            clauses.append("archived=0")
        if experiment:
            clauses.append("experiment=?")
            params.append(experiment)
        if status:
            clauses.append("status=?")
            params.append(status)
        if kind == "research":
            clauses.append("kind<>'backtest'")
        elif kind:
            clauses.append("kind=?")
            params.append(kind)
        if tag:
            clauses.append("tags_json LIKE ?")
            params.append(f'%"{tag}"%')
        if search:
            clauses.append("(name LIKE ? OR strategy LIKE ? OR notes LIKE ? OR id LIKE ?)")
            params.extend([f"%{search}%"] * 4)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT ?", (*params, limit)
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def result(self, run_id: str) -> BacktestResult:
        """Load a completed run's result."""
        record = self.get(run_id)
        directory = self.run_dir(run_id)
        # an ML run completes with metrics and no result, so the file has to be checked too
        if record.status is not RunStatus.COMPLETED or not (directory / META).exists():
            raise NotFoundError(f"Run '{run_id}' has no results (status {record.status})")
        return load_result(directory)

    def factors(self, run_id: str) -> pl.DataFrame | None:
        """Factor returns stored with a run (``None`` if the run had none)."""
        path = self.run_dir(run_id) / FACTORS_FILE
        return pl.read_parquet(path) if path.exists() else None

    def metrics(self, run_id: str) -> dict[str, MetricValue]:
        """Stored metrics of a completed run."""
        path = self.run_dir(run_id) / METRICS_FILE
        if not path.exists():
            raise NotFoundError(f"Run '{run_id}' has no metrics")
        raw = json.loads(path.read_text())
        return {k: MetricValue.model_validate(v) for k, v in raw.items()}

    # ---- research results

    def complete_research(
        self, run_id: str, payload: dict[str, Any], headline: dict[str, float | None]
    ) -> RunRecord:
        """Store a research result (JSON) and mark it completed."""
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RESEARCH_FILE).write_text(json.dumps(payload, default=str))
        with self._connect() as con:
            con.execute(
                "UPDATE runs SET status=?, finished_at=?, headline_json=?, error=NULL WHERE id=?",
                (RunStatus.COMPLETED.value, _now(), json.dumps(headline), run_id),
            )
        return self.get(run_id)

    def research(self, run_id: str) -> dict[str, Any]:
        """Stored research result."""
        path = self.run_dir(run_id) / RESEARCH_FILE
        if not path.exists():
            raise NotFoundError(f"No research result for '{run_id}'")
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    # ---- experiments and trials

    def experiments(self) -> list[dict[str, Any]]:
        """Experiments with run counts."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT e.name, e.created_at, e.notes, COUNT(r.id) AS runs FROM experiments e "
                "LEFT JOIN runs r ON r.experiment = e.name GROUP BY e.name ORDER BY e.created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def add_trials(
        self, experiment: str, source_id: str, trials: list[tuple[dict[str, Any], float | None]]
    ) -> None:
        """Record research trials (e.g. sweep points) so deflated Sharpe counts them."""
        with self._connect() as con:
            con.executemany(
                "INSERT INTO research_trials VALUES (?, ?, ?, ?)",
                [(experiment, source_id, json.dumps(p, default=str), s) for p, s in trials],
            )

    def count_trials(self, experiment: str) -> int:
        """Exact number of trials (backtests plus research trials) in an experiment."""
        with self._connect() as con:
            research = con.execute(
                "SELECT COUNT(*) FROM research_trials WHERE experiment=?", (experiment,)
            ).fetchone()[0]
            runs = con.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment=? AND kind IN ('backtest', 'ml')",
                (experiment,),
            ).fetchone()[0]
        return int(research) + int(runs)

    def trials(self, experiment: str | None) -> tuple[int, list[float]]:
        """Trial count and per-period Sharpe ratios (backtests plus research trials)."""
        if not experiment:
            return 1, []
        with self._connect() as con:
            extra = con.execute(
                "SELECT sharpe_per_period FROM research_trials WHERE experiment=?", (experiment,)
            ).fetchall()
        research = [float(r[0]) for r in extra if r[0] is not None]
        n_research = len(extra)
        # ML runs count as trials too: every evaluated configuration is one, which is the
        # whole point of trial counting
        runs = [
            r
            for kind in ("backtest", "ml")
            for r in self.list_runs(experiment=experiment, include_archived=True, kind=kind)
        ]
        sharpes = [
            float(s)
            for r in runs
            if r.status is RunStatus.COMPLETED
            and (s := r.headline.get("sharpe_per_period")) is not None
        ]
        return max(len(runs) + n_research, 1), sharpes + research

    def record_test_unlock(self, experiment: str, run_id: str | None) -> None:
        """Record that the locked test period was unlocked."""
        with self._connect() as con:
            con.execute("INSERT INTO test_unlocks VALUES (?, ?, ?)", (experiment, _now(), run_id))

    def test_unlocks(self, experiment: str) -> list[dict[str, Any]]:
        """When the test period of an experiment was unlocked."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM test_unlocks WHERE experiment=?", (experiment,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- presets

    def save_preset(self, name: str, config: dict[str, Any], notes: str = "") -> Preset:
        """Create or replace a preset by name."""
        now = _now()
        with self._connect() as con:
            existing = con.execute("SELECT id FROM presets WHERE name=?", (name,)).fetchone()
            if existing:
                con.execute(
                    "UPDATE presets SET config_json=?, updated_at=?, notes=? WHERE id=?",
                    (json.dumps(config), now, notes, existing["id"]),
                )
                preset_id = existing["id"]
            else:
                preset_id = uuid.uuid4().hex[:10]
                con.execute(
                    "INSERT INTO presets VALUES (?, ?, ?, ?, ?, ?)",
                    (preset_id, name, json.dumps(config), now, now, notes),
                )
        return self.preset(preset_id)

    def preset(self, preset_id: str) -> Preset:
        """Fetch a preset."""
        with self._connect() as con:
            row = con.execute("SELECT * FROM presets WHERE id=?", (preset_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Preset '{preset_id}' not found")
        return Preset(
            id=row["id"],
            name=row["name"],
            config=json.loads(row["config_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            notes=row["notes"],
        )

    def presets(self) -> list[Preset]:
        """All presets by name."""
        with self._connect() as con:
            ids = [r["id"] for r in con.execute("SELECT id FROM presets ORDER BY name")]
        return [self.preset(i) for i in ids]

    def delete_preset(self, preset_id: str) -> None:
        """Delete a preset."""
        self.preset(preset_id)
        with self._connect() as con:
            con.execute("DELETE FROM presets WHERE id=?", (preset_id,))


def _scalars(metrics: dict[str, Any]) -> dict[str, float | None]:
    """The top-level numbers of an ML metrics blob, for the run list."""
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def _per_period_sharpe(annual: float | None, periods_per_year: float) -> float | None:
    if annual is None or periods_per_year <= 0:
        return None
    return float(annual / periods_per_year**0.5)
