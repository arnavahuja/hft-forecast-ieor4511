"""The one comparison table across horizons (§7.5): CSV and Markdown, nothing else.

The markdown table is what goes into the report and the presentation. No PDF, no slides.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

COMPARISON_CSV: Final = "comparison.csv"
COMPARISON_MD: Final = "comparison.md"
BEST_BY: Final = "net_sharpe_quoted_spread"
"""The best model per horizon is the one with the highest out-of-sample net Sharpe
under the quoted-spread cost variant."""

ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("n_rows", "rows"),
    ("n_eff", "n_eff"),
    ("n_sessions", "sessions"),
    ("n_folds", "folds"),
    ("accuracy", "accuracy [CI]"),
    ("auc_up", "AUC up"),
    ("auc_down", "AUC down"),
    ("ic", "IC [CI]"),
    ("ic_t_stat", "IC t-stat"),
    ("mae_bps", "MAE (bps)"),
    ("r2_vs_zero", "R2 vs zero"),
    ("net_sharpe_none", "net Sharpe @ none"),
    ("net_sharpe_quoted_spread", "net Sharpe @ quoted_spread"),
    ("net_sharpe_quoted_spread_stressed", "net Sharpe @ stressed"),
    ("breakeven_cost_multiple", "breakeven cost multiple"),
    ("turnover_per_session", "turnover / session"),
    ("dsr", "DSR"),
    ("pbo", "PBO"),
    ("n_trials", "trials"),
    ("cross_sectional_net_sharpe", "cross-sectional net Sharpe"),
)
"""Table rows, in order. Adding one is a change to the frozen interface (§8.1)."""


@dataclass
class Entry:
    """One ``(horizon, model)`` column of the table."""

    horizon: str
    model: str
    values: dict[str, float | None] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Column header."""
        return f"{self.horizon} / {self.model}"


def entry_from_metrics(
    horizon: str, model: str, metrics: dict[str, Any], economic: dict[str, Any] | None = None
) -> Entry:
    """Flatten one run's metrics (and its economic results, if it has any) into a column."""
    pooled = metrics.get("models", {}).get(model, {}).get("pooled", {})
    entry = Entry(horizon=horizon, model=model)
    entry.values = {
        "n_rows": _number(metrics.get("n_rows")),
        "n_sessions": _number(metrics.get("n_sessions")),
        "n_folds": _number(metrics.get("n_folds")),
        **{
            k: _number(pooled.get(k))
            for k in ("accuracy", "auc_up", "auc_down", "ic", "ic_t_stat", "mae_bps", "r2_vs_zero")
        },
        **{k: _number(v) for k, v in (economic or {}).items()},
    }
    for name, bounds in pooled.get("ci", {}).items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            entry.intervals[name] = (float(bounds[0]), float(bounds[1]))
    return entry


def comparison_table(entries: list[Entry]) -> list[list[str]]:
    """The table as rows of strings, header first."""
    best = _best_per_horizon(entries)
    header = ["metric", *(e.key for e in entries)]
    out = [header]
    for key, label in ROWS:
        row = [label]
        for entry in entries:
            text = _format(entry, key)
            row.append(f"**{text}**" if entry.key in best and key == BEST_BY else text)
        out.append(row)
    return out


def write_report(entries: list[Entry], directory: Path) -> tuple[Path, Path]:
    """Write ``comparison.csv`` and ``comparison.md``; return both paths."""
    directory.mkdir(parents=True, exist_ok=True)
    table = comparison_table(entries)
    csv_path = directory / COMPARISON_CSV
    with csv_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(table)
    md_path = directory / COMPARISON_MD
    md_path.write_text(to_markdown(table))
    return csv_path, md_path


def to_markdown(table: list[list[str]]) -> str:
    """GitHub-flavoured markdown table."""
    header, *body = table
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines) + "\n"


def _best_per_horizon(entries: list[Entry]) -> set[str]:
    best: dict[str, Entry] = {}
    for entry in entries:
        value = entry.values.get(BEST_BY)
        if value is None or not math.isfinite(value):
            continue
        current = best.get(entry.horizon)
        if current is None or value > (current.values.get(BEST_BY) or -math.inf):
            best[entry.horizon] = entry
    return {e.key for e in best.values()}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format(entry: Entry, key: str) -> str:
    value = entry.values.get(key)
    if value is None:
        return "—"
    text = f"{value:,.0f}" if abs(value) >= 1000 or key.startswith("n_") else f"{value:.3f}"
    interval = entry.intervals.get(key)
    if interval:
        text += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
    return text
