"""Command line interface: ``hftf run``, ``hftf data pull``, ``hftf serve``, ..."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from hft_forecast.core.config import Settings
from hft_forecast.core.errors import BackboneError
from hft_forecast.core.logging import configure_logging
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.types import Adjustment, DataRequest, Frequency

app = typer.Typer(
    help="hft-forecast: short-horizon forecasting research and backtesting.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Data commands.", no_args_is_help=True)
app.add_typer(data_app, name="data")
preset_app = typer.Typer(help="Preset commands.", no_args_is_help=True)
app.add_typer(preset_app, name="preset")


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


@preset_app.command("import")
def preset_import(
    configs: Annotated[list[Path], typer.Argument(help="YAML/JSON run configs", exists=True)],
) -> None:
    """Save run configs as presets (named after each config's name, else the file name)."""
    services = _services()
    for path in configs:
        cfg = load_config(path)
        name = cfg.name or path.stem
        preset = services.store.save_preset(name, cfg.model_dump(mode="json"))
        typer.echo(f"preset {preset.id}  {name}")


@app.command()
def new(
    kind: Annotated[
        str,
        typer.Argument(
            help="strategy, overlay, metric, chart, data_source, "
            "cost_model or portfolio_constructor"
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


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
