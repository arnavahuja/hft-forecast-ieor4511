"""PNG charts for the comparison report (matplotlib, dark theme, no purple)."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from hft_forecast.analytics import stats as S

if TYPE_CHECKING:
    from hft_forecast.services.report import ReportRequest, RunSeries

BG: Final = "#0f1419"
PANEL: Final = "#161c23"
TEXT: Final = "#e6e8eb"
MUTED: Final = "#8a949e"
GRID: Final = "#2a323b"
NEGATIVE: Final = "#e57373"
POSITIVE: Final = "#4fc3f7"
PALETTE: Final = (
    "#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#fff176", "#4db6ac",
    "#90a4ae", "#ff8a65", "#aed581", "#64b5f6", "#f0b27a", "#26a69a",
)  # fmt: skip
DPI: Final = 150

Scaler = Callable[["RunSeries"], "pd.Series[float]"]


def _style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler  # type: ignore[import-untyped]

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "axes.prop_cycle": cycler(color=PALETTE),
            "font.size": 10,
        }
    )
    return plt


def _png(plt: Any, fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    return buf.getvalue()


def _cumulative(plt: Any, runs: list[RunSeries], req: ReportRequest, scale: Scaler) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 6))
    for rs in runs:
        series = scale(rs) if rs.scaled else rs.net
        label = f"{rs.name} (vol-scaled {req.vol_target:.0%})" if rs.scaled else rs.name
        ax.plot(series.index, S.wealth(series.to_numpy()), label=label, linewidth=1.4)
    ax.set_yscale("log")
    suffix = " - market-neutral series vol-scaled" if any(rs.scaled for rs in runs) else ""
    ax.set_title(f"Cumulative net return (log scale){suffix}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(fontsize=8, loc="upper left")
    return _png(plt, fig)


def _underwater(plt: Any, runs: list[RunSeries]) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 5))
    for rs in runs:
        dd = S.drawdown_series(rs.net.to_numpy()) * 100
        ax.plot(rs.net.index, dd, label=rs.name, linewidth=1.2)
    ax.set_title("Underwater plot (net, unscaled)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8, loc="lower left")
    return _png(plt, fig)


def _rolling_sharpe(plt: Any, runs: list[RunSeries], window: int) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 5))
    for rs in runs:
        ex = (rs.net - rs.rf).iloc[1:]
        roll = ex.rolling(window).mean() / ex.rolling(window).std() * math.sqrt(rs.ppy)
        ax.plot(roll.index, roll, label=rs.name, linewidth=1.2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(f"Rolling {window}-period Sharpe ratio (net, excess of risk-free)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (annualized)")
    ax.legend(fontsize=8, loc="upper left")
    return _png(plt, fig)


def _heatmap(plt: Any, corr: pd.DataFrame) -> bytes:
    from matplotlib.colors import LinearSegmentedColormap

    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.2 * n + 4, 1.0 * n + 3))
    cmap = LinearSegmentedColormap.from_list("neg_pos", [NEGATIVE, PANEL, POSITIVE])
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            text = f"{corr.iat[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation of net returns")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Strategy")
    return _png(plt, fig)


def _sharpe_drawdown(plt: Any, metrics: pd.DataFrame) -> bytes:
    net = metrics[metrics["variant"] == "net"]
    names = net["strategy"].tolist()
    colors = [PALETTE[k % len(PALETTE)] for k in range(len(names))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.bar(names, net["sharpe"], color=colors)
    a1.set_title("Net Sharpe ratio")
    a1.set_ylabel("Sharpe")
    a2.bar(names, net["max_drawdown"] * 100, color=colors)
    a2.set_title("Maximum drawdown (net)")
    a2.set_ylabel("Drawdown (%)")
    for a in (a1, a2):
        a.set_xlabel("Strategy")
        a.tick_params(axis="x", rotation=45)
        for lbl in a.get_xticklabels():
            lbl.set_ha("right")
    return _png(plt, fig)


def _cost_lines(plt: Any, costs: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, grp in costs.groupby("strategy", sort=False):
        ax.plot(grp["one_way_bps"], grp["net_sharpe"], marker="o", label=str(name))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Cost sensitivity: net Sharpe vs one-way trading cost")
    ax.set_xlabel("One-way trading cost (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.legend(fontsize=8)
    return _png(plt, fig)


def render_charts(
    runs: list[RunSeries], tables: dict[str, pd.DataFrame], req: ReportRequest, scale: Scaler
) -> dict[str, bytes]:
    """All report charts as ``{file name: PNG bytes}``."""
    plt = _style()
    out = {
        "01_cumulative_net_log.png": _cumulative(plt, runs, req, scale),
        "02_underwater.png": _underwater(plt, runs),
        "03_rolling_sharpe.png": _rolling_sharpe(plt, runs, req.rolling_window),
        "04_correlation_heatmap.png": _heatmap(plt, tables["correlations"]),
        "05_sharpe_and_drawdown.png": _sharpe_drawdown(plt, tables["metrics"]),
    }
    costs = tables.get("cost_sensitivity")
    if costs is not None and len(costs):
        out["06_cost_sensitivity.png"] = _cost_lines(plt, costs)
    return out
