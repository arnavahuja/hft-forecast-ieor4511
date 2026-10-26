"""Command line interface: ``hftf run``, ``hftf data pull``, ``hftf serve``, ..."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from hft_forecast.core import columns as C
from hft_forecast.core.config import Settings
from hft_forecast.core.errors import BackboneError
from hft_forecast.core.logging import configure_logging
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.types import Adjustment, DataRequest, Frequency
from hft_forecast.ml.train import MANIFEST_FILE

app = typer.Typer(
    help="hft-forecast: short-horizon forecasting research and backtesting.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Data commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _services() -> Any:
    from hft_forecast.services.container import Services

    settings = Settings()
    configure_logging(settings.log_level, json=settings.log_json)
    return Services.create(settings)


def load_config(path: Path) -> BacktestConfig:
    """Load a run config from YAML or JSON."""
    raw = yaml.safe_load(path.read_text())
    return BacktestConfig.model_validate(raw)


def _fmt(value: float | None, pct: bool) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if pct else f"{value:.3f}"


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="YAML/JSON run config", exists=True)],
    name: Annotated[str | None, typer.Option(help="Override the run name")] = None,
) -> None:
    """Run a backtest from a config file and store the result."""
    services = _services()
    cfg = load_config(config)
    if name:
        cfg = cfg.model_copy(update={"name": name})
    try:
        record = services.execute_run(cfg)
    except BackboneError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        if exc.details:
            typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"run {record.id} completed ({record.timings.get('total_seconds')}s)")
    pct = {"total_return", "cagr", "ann_vol", "max_drawdown", "turnover_ann"}
    for key, value in record.headline.items():
        if key == "sharpe_per_period":
            continue
        typer.echo(f"  {key:<16} {_fmt(value, key in pct)}")
    typer.echo(f"results: {services.store.run_dir(record.id)}")


@app.command()
def plugins(kind: Annotated[str | None, typer.Option(help="Plugin kind")] = None) -> None:
    """List registered plugins and import failures."""
    services = _services()
    for spec in services.plugins.specs(kind):
        typer.echo(
            f"{spec.kind:<22} {spec.name:<24} {spec.version:<8} {spec.origin:<8} "
            f"{spec.description[:60]}"
        )
    for failure in services.plugins.report.failures:
        typer.echo(f"FAILED {failure.module}: {failure.error}", err=True)


@app.command("check-lookahead")
def check_lookahead_cmd(
    config: Annotated[Path, typer.Argument(help="Run config", exists=True)],
) -> None:
    """Run the truncation lookahead check on the config's strategy and data."""
    from hft_forecast.engine.lookahead import check_lookahead

    services = _services()
    cfg = load_config(config)
    prepared = services.runner.prepare(cfg)
    data = prepared.data
    if prepared.options.strategy_instruments is not None:
        data = data.select_instruments(prepared.options.strategy_instruments)
    report = check_lookahead(prepared.strategy, data)
    typer.echo(report.message)
    for m in report.mismatches:
        typer.echo(f"  {m}")
    raise typer.Exit(0 if report.passed else 1)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address (localhost only)")] = None,
    port: Annotated[int | None, typer.Option(help="Port")] = None,
    reload: Annotated[bool, typer.Option(help="Auto-reload backend code")] = False,
) -> None:
    """Start the API server (bound to 127.0.0.1)."""
    import uvicorn

    settings = Settings()
    configure_logging(settings.log_level, json=settings.log_json)
    bind = host or settings.host
    if bind not in ("127.0.0.1", "localhost", "::1"):
        typer.echo("Refusing to bind to a non-local address", err=True)
        raise typer.Exit(2)
    uvicorn.run(
        "hft_forecast.api.app:create_app",
        factory=True,
        host=bind,
        port=port or settings.port,
        reload=reload,
        log_level="info",
    )


@app.command()
def openapi(
    out: Annotated[Path, typer.Option(help="Output file")] = Path("frontend/openapi.json"),
) -> None:
    """Write the OpenAPI schema (source for the generated TypeScript client)."""
    from hft_forecast.api.app import create_app

    schema = create_app(Settings()).openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n")
    typer.echo(f"wrote {out}")


@data_app.command("pull")
def data_pull(
    instruments: Annotated[list[str], typer.Argument(help="Tickers / ids")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    source: Annotated[str, typer.Option()] = "yahoo",
    dataset: Annotated[str, typer.Option()] = "daily",
    frequency: Annotated[Frequency, typer.Option()] = Frequency.D1,
    adjustment: Annotated[Adjustment, typer.Option()] = Adjustment.SPLIT,
    refresh: Annotated[bool, typer.Option(help="Force a re-pull")] = False,
) -> None:
    """Pull data into the cache and catalog."""
    services = _services()
    request = DataRequest(
        source=source,
        dataset=dataset,
        instruments=tuple(instruments),
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        frequency=frequency,
        adjustment=adjustment,
    )
    data = services.data.fetch(request, refresh=refresh)
    typer.echo(f"{len(data)} rows, {len(data.instruments)} instruments, fields {data.fields}")


@data_app.command("pull-taq")
def data_pull_taq(
    manifest: Annotated[str, typer.Option(help="Manifest name, e.g. sp50_2024h1_5s")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    symbols: Annotated[list[str] | None, typer.Argument(help="Ticker symbols")] = None,
    symbols_file: Annotated[
        Path | None, typer.Option(help="File with one ticker per line", exists=True)
    ] = None,
    grid: Annotated[Frequency, typer.Option(help="Bar size")] = Frequency.S5,
    trades: Annotated[bool, typer.Option(help="Also pull taq_trade_bars")] = True,
) -> None:
    """Pull TAQ bars one instrument-month at a time and record them in a manifest.

    Re-running the same command resumes: cached months are served from disk and the manifest
    is merged, so a multi-hour pull survives an interruption.
    """
    from hft_forecast.data.manifest import Manifest, ManifestEntry

    services = _services()
    tickers = sorted({*(symbols or []), *_read_symbols(symbols_file)})
    if not tickers:
        typer.echo("error: give ticker symbols or --symbols-file", err=True)
        raise typer.Exit(2)
    data = services.data
    directory = data.paths.manifests_dir
    book = (
        Manifest.load(directory, manifest)
        if (directory / f"{manifest}.yaml").exists()
        else Manifest(name=manifest, source="wrds_taq", frequency=grid.value)
    )
    datasets = ["taq_nbbo_bars", *(["taq_trade_bars"] if trades else [])]
    totals = {"error": 0, "warning": 0, "info": 0}
    for dataset in datasets:
        request = DataRequest(
            source="wrds_taq",
            dataset=dataset,
            instruments=tuple(tickers),
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            frequency=grid,
        )
        for part in data.partitions(request):
            label = f"{part.instruments[0]} {part.start:%Y-%m} {dataset}"
            try:
                fetched = data.fetch(part)
            except BackboneError as exc:
                typer.echo(f"  {label}: skipped ({exc.message})", err=True)
                continue
            path = data.cached_file(part)
            book = book.merged(
                [
                    ManifestEntry(
                        instrument=part.instruments[0],
                        dataset=dataset,
                        start=part.start.isoformat(),
                        end=part.end.isoformat(),
                        rows=len(fetched),
                        path=str(path or ""),
                    )
                ]
            )
            book.save(directory)
            record = data.catalog.get(part.key_without_dates())
            for severity, count in record.quality.items():
                totals[severity] = totals.get(severity, 0) + count
            typer.echo(f"  {label}: {len(fetched)} rows")
    typer.echo(f"manifest {book.save(directory)} ({len(book.entries)} partitions)")
    typer.echo("quality: " + ", ".join(f"{k} {v}" for k, v in totals.items()))


def _read_symbols(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]


@data_app.command("import")
def data_import(
    path: Annotated[Path, typer.Argument(exists=True)],
    name: Annotated[str | None, typer.Option()] = None,
    symbol: Annotated[str | None, typer.Option(help="Instrument id for single-asset files")] = None,
    frequency: Annotated[Frequency, typer.Option()] = Frequency.D1,
    timezone: Annotated[str, typer.Option()] = "UTC",
) -> None:
    """Import a local file using the guessed column mapping."""
    from hft_forecast.data.importer import guess_mapping, read_any

    services = _services()
    mapping = guess_mapping(read_any(path, n_rows=1000), path.stem)
    update: dict[str, Any] = {"frequency": frequency, "timezone": timezone}
    if name:
        update["name"] = name
    if symbol:
        update["fixed_symbol"] = symbol
    mapping = mapping.model_copy(update=update)
    typer.echo(f"mapping: {mapping.model_dump_json()}")
    record = services.data.import_file(path, mapping)
    typer.echo(f"imported {record.id}: {record.rows} rows, instruments {record.instruments[:10]}")


@data_app.command("catalog")
def data_catalog() -> None:
    """List datasets in the catalog."""
    services = _services()
    for rec in services.data.catalog.records():
        typer.echo(
            f"{rec.id:<28} {rec.kind:<7} {rec.source:<11} {rec.frequency:<5} "
            f"{rec.start} → {rec.end}  {rec.rows} rows  {len(rec.instruments)} instr."
        )


@app.command()
def new(
    kind: Annotated[
        str,
        typer.Argument(
            help="strategy, overlay, metric, chart, data_source, cost_model, "
            "portfolio_constructor, model or feature_set"
        ),
    ],
    name: Annotated[str, typer.Argument(help="snake_case plugin name")],
    directory: Annotated[Path, typer.Option(help="Target directory")] = Path("user_plugins"),
) -> None:
    """Scaffold a new plugin (and a test) from a template."""
    from hft_forecast.scaffold import scaffold

    try:
        created = scaffold(kind, name, directory)
    except BackboneError as exc:
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(1) from exc
    for path in created:
        typer.echo(f"created {path}")
    typer.echo(
        "Restart the server (or keep it running with HFTF_DEV_MODE=true for hot "
        "reload); the plugin appears in the UI automatically."
    )


ml_app = typer.Typer(help="Forecasting experiments.", no_args_is_help=True)
app.add_typer(ml_app, name="ml")


def _experiment(path: Path, model: str | None, seed: int | None) -> Any:
    from hft_forecast.core.run_config import PluginRef
    from hft_forecast.ml.config import ExperimentConfig

    config = ExperimentConfig.load(path)
    update: dict[str, Any] = {}
    if model:
        update["model"] = PluginRef(name=model)
    if seed is not None:
        update["seed"] = seed
    return config.model_copy(update=update) if update else config


@ml_app.command("build-dataset")
def ml_build_dataset(
    config: Annotated[Path, typer.Argument(help="Experiment config", exists=True)],
) -> None:
    """Build (or reuse) the features and labels of an experiment."""
    from hft_forecast.ml.datasets import build_dataset

    services = _services()
    cfg = _experiment(config, None, None)
    directory = build_dataset(cfg, services.data)
    files = sorted(directory.glob("*.parquet"))
    typer.echo(f"{directory} ({len(files)} instruments)")
    for path in files:
        typer.echo(f"  {path.stem:<10} {path.stat().st_size / 1e6:.1f} MB")


@ml_app.command("run")
def ml_run(
    config: Annotated[Path, typer.Argument(help="Experiment config", exists=True)],
    model: Annotated[str | None, typer.Option(help="Override the model")] = None,
    seed: Annotated[int | None, typer.Option(help="Override the seed")] = None,
) -> None:
    """Train, predict out of sample and register the run."""
    from hft_forecast.ml.evaluate import evaluate
    from hft_forecast.ml.train import load_predictions, run_experiment
    from hft_forecast.services.runner import git_commit

    services = _services()
    cfg = _experiment(config, model, seed)
    record = services.store.create_ml(
        name=cfg.name or f"{cfg.experiment} {cfg.model.name}",
        config=cfg.model_dump(mode="json"),
        config_hash=cfg.config_hash(),
        experiment=cfg.experiment,
        model=cfg.model.name,
        tags=["ml", cfg.horizon.name, cfg.model.name, *cfg.tags],
        notes=cfg.notes,
    )
    services.store.mark_running(record.id)
    directory = services.store.run_dir(record.id)
    try:
        manifest = run_experiment(cfg, services.data, directory, git_commit())
    except BackboneError as exc:
        services.store.mark_failed(record.id, exc.message)
        typer.echo(f"error: {exc.message}", err=True)
        raise typer.Exit(1) from exc
    metrics = evaluate(load_predictions(directory), cfg.seed)
    pooled = metrics["models"][manifest.model]["pooled"]
    services.store.complete_ml(
        record.id,
        {**metrics, "n_eff": manifest.n_eff},
        directory / MANIFEST_FILE,
        headline={
            "accuracy": pooled["accuracy"],
            "ic": pooled["ic"],
            "n_eff": manifest.n_eff,
        },
        git_commit=manifest.git_commit,
        timings=manifest.timings,
    )
    typer.echo(f"run {record.id}  {manifest.model}  {len(manifest.folds)} folds")
    typer.echo(f"  rows {manifest.n_rows:,}   n_eff {manifest.n_eff:,.0f}")
    for name, values in metrics["models"].items():
        p = values["pooled"]
        typer.echo(f"  {name:<22} accuracy {p['accuracy']:.4f}  IC {p['ic']:.4f}")
    typer.echo(f"results: {directory}")


@ml_app.command("econ")
def ml_econ(
    run_id: Annotated[str, typer.Argument(help="An ml run id")],
    cross_sectional: Annotated[bool, typer.Option(help="Run the cross-sectional book")] = False,
    variant: Annotated[str, typer.Option(help="Cost variant")] = "quoted_spread",
) -> None:
    """Run the economic test for a trained model through the backtest engine."""
    import polars as pl

    from hft_forecast.ml.config import ExperimentConfig
    from hft_forecast.ml.economic import (
        RETURN_COLUMN,
        cross_sectional_config,
        per_name_configs,
        pooled_returns,
    )
    from hft_forecast.ml.train import PREDICTIONS_FILE

    services = _services()
    record = services.store.get(run_id)
    cfg = ExperimentConfig.model_validate(record.config)
    directory = services.store.run_dir(run_id)
    predictions = directory / PREDICTIONS_FILE
    label = str(json.loads((directory / MANIFEST_FILE).read_text())["model"])
    configs = (
        [cross_sectional_config(cfg, predictions, label, variant)]
        if cross_sectional
        else per_name_configs(cfg, predictions, label, variant)
    )
    books: dict[str, pl.DataFrame] = {}
    for backtest in configs:
        done = services.execute_run(backtest)
        result = services.store.result(done.id)
        name = backtest.data.instruments[0] if backtest.data.instruments else "book"
        books[name] = pl.DataFrame({C.TIMESTAMP: result.timestamps, RETURN_COLUMN: result.returns})
        sharpe = done.headline.get("sharpe")
        typer.echo(
            f"  {name:<8} {done.id}  net Sharpe {sharpe if sharpe is None else f'{sharpe:.2f}'}"
        )
    pooled = pooled_returns(books)
    if pooled.height:
        series = pooled["pooled_return"].to_numpy()
        typer.echo(f"pooled per-bar mean {series.mean():.3e}  t-stat {_t_stat(series):.2f}")


def _t_stat(values: Any) -> float:
    from hft_forecast.analytics import stats as S

    return float(S.mean_t_stat(values))


@ml_app.command("compare")
def ml_compare(
    experiment: Annotated[list[str], typer.Option(help="Experiment names")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("report"),
) -> None:
    """Write the one comparison table across horizons."""
    from hft_forecast.ml.report import entry_from_metrics, write_report
    from hft_forecast.services.run_store import RunStatus

    services = _services()
    entries = []
    for name in experiment:
        runs = [
            r
            for r in services.store.list_runs(experiment=name, kind="ml")
            if r.status is RunStatus.COMPLETED
        ]
        if not runs:
            typer.echo(f"no completed ml runs for '{name}'", err=True)
            continue
        metrics = services.store.ml_metrics(runs[0].id)
        for model in sorted(metrics.get("models", {})):
            entries.append(entry_from_metrics(name, model, metrics))
    if not entries:
        typer.echo("nothing to compare", err=True)
        raise typer.Exit(1)
    csv_path, md_path = write_report(entries, out)
    typer.echo(f"wrote {csv_path} and {md_path}")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
