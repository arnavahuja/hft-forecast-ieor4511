"""Self-contained HTML tearsheet (inline SVG charts, no external assets).

Printing the page from a browser produces the PDF version.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Final

import numpy as np

from hft_forecast.analytics.stats import aggregate_returns, drawdown_series
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.specs import MetricDescriptor, MetricFormat, MetricKind, MetricValue
from hft_forecast.core.types import FloatArray
from hft_forecast.services.run_store import RunRecord

WIDTH: Final = 900
HEIGHT: Final = 220
PAD: Final = 36
ACCENT: Final = "#14B8A6"
BENCH: Final = "#8A8A8A"
NEG: Final = "#EF4444"
POS: Final = "#22C55E"
MAX_POINTS: Final = 900

CSS: Final = """
:root { color-scheme: dark; }
body { background:#0A0A0A; color:#EDEDED; font:14px/1.45 Inter, system-ui, sans-serif;
       margin:0; padding:32px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:15px; color:#A3A3A3; margin:28px 0 8px;
     text-transform:uppercase; letter-spacing:.06em; }
.sub { color:#A3A3A3; margin-bottom:20px; }
.card { background:#121212; border:1px solid #262626; border-radius:8px; padding:16px;
        margin-bottom:16px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
td, th { padding:4px 8px; border-bottom:1px solid #262626; }
th { text-align:left; color:#A3A3A3; font-weight:500; }
td.num { text-align:right; font-family: "JetBrains Mono", ui-monospace, monospace; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }
.pos { color:#22C55E; } .neg { color:#EF4444; } .muted { color:#6B6B6B; }
@media print { body { background:#fff; color:#111; } .card { border-color:#ddd;
  background:#fff; } }
"""


def fmt_value(value: float | None, fmt: MetricFormat) -> str:
    """Format a metric value for display (the single formatting utility on the backend)."""
    if value is None or not np.isfinite(value):
        return "—"
    if fmt is MetricFormat.PERCENT:
        return f"{value:.2%}"
    if fmt is MetricFormat.CURRENCY:
        return f"{value:,.0f}"
    if fmt in (MetricFormat.INTEGER, MetricFormat.BARS, MetricFormat.DAYS):
        return f"{value:,.0f}"
    return f"{value:.3f}"


def _thin(values: FloatArray) -> FloatArray:
    step = max(len(values) // MAX_POINTS, 1)
    return values[::step]


def _path(values: FloatArray, lo: float, hi: float) -> str:
    n = len(values)
    if n < 2 or hi <= lo:
        return ""
    xs = PAD + np.arange(n) / (n - 1) * (WIDTH - 2 * PAD)
    ys = HEIGHT - PAD - (values - lo) / (hi - lo) * (HEIGHT - 2 * PAD)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True) if np.isfinite(y))
    return pts


def svg_lines(series: Sequence[tuple[FloatArray, str, bool]], label: str) -> str:
    """Inline SVG line chart; each series is ``(values, color, dashed)``."""
    thinned = [(_thin(v), c, d) for v, c, d in series]
    finite = np.concatenate([v[np.isfinite(v)] for v, _, _ in thinned]) if thinned else []
    if len(finite) == 0:
        return ""
    lo, hi = float(np.min(finite)), float(np.max(finite))
    lines = []
    for values, color, dashed in thinned:
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5"{dash} '
            f'points="{_path(values, lo, hi)}"/>'
        )
    axis = (
        f'<text x="{PAD}" y="16" fill="#A3A3A3" font-size="11">{html.escape(label)}</text>'
        f'<text x="4" y="{PAD + 4}" fill="#6B6B6B" font-size="10">{hi:.2f}</text>'
        f'<text x="4" y="{HEIGHT - PAD}" fill="#6B6B6B" font-size="10">{lo:.2f}</text>'
    )
    return (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="100%" role="img" '
        f'aria-label="{html.escape(label)}">{axis}{"".join(lines)}</svg>'
    )


def render_tearsheet(
    record: RunRecord,
    result: BacktestResult,
    metrics: dict[str, MetricValue],
    descriptors: Sequence[MetricDescriptor],
) -> str:
    """Render the tearsheet HTML."""
    title = html.escape(record.name or f"{record.strategy} run {record.id}")
    wealth = np.cumprod(1 + result.returns)
    series = [(wealth, ACCENT, False)]
    if result.benchmark_returns is not None:
        series.append((np.cumprod(1 + result.benchmark_returns), BENCH, True))
    equity_svg = svg_lines(series, "Growth of 1")
    dd_svg = svg_lines([(drawdown_series(result.returns), NEG, False)], "Drawdown")

    groups: dict[str, list[str]] = {}
    for d in descriptors:
        mv = metrics.get(d.key)
        if mv is None or d.kind is MetricKind.TABLE:
            continue
        bench = fmt_value(mv.benchmark, d.format) if d.benchmark else ""
        groups.setdefault(d.group, []).append(
            f"<tr><td>{html.escape(d.label)}</td><td class='num'>"
            f"{fmt_value(mv.value, d.format)}</td><td class='num muted'>{bench}</td></tr>"
        )
    cards = "".join(
        f"<div class='card'><h2>{html.escape(g)}</h2><table><tr><th>Metric</th>"
        f"<th style='text-align:right'>Strategy</th><th style='text-align:right'>Benchmark"
        f"</th></tr>{''.join(rows)}</table></div>"
        for g, rows in groups.items()
    )
    codes, yearly = aggregate_returns(result.returns, result.timestamps, "Y")
    years = "".join(
        f"<tr><td>{int(c) + 1970}</td><td class='num {'pos' if v >= 0 else 'neg'}'>"
        f"{v:.2%}</td></tr>"
        for c, v in zip(codes, yearly, strict=True)
    )
    cfg = record.config
    sub = (
        f"{html.escape(cfg['strategy']['name'])} · {cfg['data']['source']} · "
        f"{cfg['data']['start']} → {cfg['data']['end']} · engine "
        f"{cfg['execution']['engine']} · config {record.config_hash}"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{title} · Tearsheet</title><style>{CSS}</style></head><body>
<h1>{title}</h1><div class="sub">{sub}</div>
<div class="card">{equity_svg}</div><div class="card">{dd_svg}</div>
<div class="grid">{cards}<div class='card'><h2>Annual returns</h2><table>{years}</table></div>
</div></body></html>"""
