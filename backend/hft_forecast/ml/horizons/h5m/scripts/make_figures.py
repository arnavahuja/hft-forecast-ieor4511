"""Final h5m figures, from the saved reports only (no data loading, no fitting).

usage (repository root, environment active):
    python backend/hft_forecast/ml/horizons/h5m/scripts/make_figures.py

Reads   E:/4511_5min/wrds_taq/reports      (research half, 2024H1)
        E:/4511_5min/wrds_taq_h2/reports   (holdout, 2024H2)
Writes  backend/hft_forecast/ml/horizons/h5m/figures/final_*.png  (4 figures)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

H1 = Path(r"E:\4511_5min\wrds_taq\reports")
H2 = Path(r"E:\4511_5min\wrds_taq_h2\reports")
OUT = Path(__file__).resolve().parents[1] / "figures"
RUNS_H1 = {"T6": "cac38dec1522", "T1": "a569afa2eba8"}
RUN_H6 = "d970798b2c45"
MODELS = ["T3", "T4", "T5", "T6", "T7"]
NAMES = {"T3": "Linear\n(basic)", "T4": "Linear\n+ flow", "T5": "LightGBM\n+ flow",
         "T6": "MLP\n+ flow", "T7": "GRU\n+ flow + seq"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def md_table(path: Path) -> list[dict[str, str]]:
    """Rows of the first markdown table in a file, keyed by its header."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == len(head):
            rows.append(dict(zip(head, cells)))
    return rows


def num(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pick(rows: list[dict[str, str]], trial: str, head: str, col: str) -> float:
    for r in rows:
        if r["trial"] == trial and r["head"] == head:
            return num(r[col])
    return float("nan")


def se(rows, trial, head):
    ic, t = pick(rows, trial, head, "ic_cs"), pick(rows, trial, head, "t_ic_cs")
    return abs(ic / t) if t else 0.0


def fig_ladder(h1, h2) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, head, title in zip(axes, ("y_hat", "p_up-p_down"), ("Return head (primary)", "Direction head")):
        x = range(len(MODELS))
        a = [pick(h1, m, head, "ic_cs") for m in MODELS]
        b = [pick(h2, "H" + m[1:], head, "ic_cs") for m in MODELS]
        ea = [se(h1, m, head) for m in MODELS]
        eb = [se(h2, "H" + m[1:], head) for m in MODELS]
        ax.bar([i - 0.2 for i in x], a, 0.4, yerr=ea, capsize=3, label="2024H1 (research)", color="#2a7f62")
        ax.bar([i + 0.2 for i in x], b, 0.4, yerr=eb, capsize=3, label="2024H2 (holdout)", color="#d08c3a")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x), [NAMES[m] for m in MODELS], fontsize=8.5)
        ax.set_title(title)
    axes[0].set_ylabel("Cross-sectional IC, tradeable target\n(fill at t+1, hold 5 min); bars = 1 s.e.")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Model ladder: every model lands near IC 0.01, in both halves of 2024", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "final_model_ladder.png", dpi=200)
    plt.close(fig)


def fig_economics(h1, h2) -> None:
    labels, gross, cost = [], [], []
    for m in MODELS:
        for tag, rows, t in (("H1", h1, m), ("H2", h2, "H" + m[1:])):
            labels.append(f"{m} {tag}")
            gross.append(pick(rows, t, "y_hat", "ls_bps"))
            cost.append(pick(rows, t, "y_hat", "ls_cost"))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], gross, 0.4, label="Gross return of the long-short pair", color="#2a7f62")
    ax.bar([i + 0.2 for i in x], cost, 0.4, label="Round-trip quoted spread, both legs", color="#b5482e")
    ax.set_xticks(list(x), labels, rotation=45, fontsize=8)
    ax.set_ylabel("bps per 5-minute position")
    ax.legend(frameon=False)
    ax.set_title("Long top 10% / short bottom 10% each minute: costs are 25-80 times the gross edge")
    fig.tight_layout()
    fig.savefig(OUT / "final_economics.png", dpi=200)
    plt.close(fig)


def fig_where(h1_het, h2_het) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    groups = ["low price", "mid price", "high price"]
    for rows, off, lab, col in ((h1_het, -0.2, "2024H1 (T6)", "#2a7f62"), (h2_het, 0.2, "2024H2 (H6)", "#d08c3a")):
        v = [num(next(r["ic_cs"] for r in rows if r["group"] == g and r["head"] == "y_hat")) for g in groups]
        axes[0].bar([i + off for i in range(3)], v, 0.4, label=lab, color=col)
        tod = [r for r in rows if r["dimension"] == "time_of_day" and r["head"] == "y_hat"]
        axes[1].plot([r["group"][:5] for r in tod], [num(r["ic_cs"]) for r in tod], marker="o", label=lab, color=col)
    axes[0].set_xticks(range(3), groups)
    axes[0].set_ylabel("Cross-sectional IC within the group (MLP, y_hat)")
    axes[0].set_title("By price level (tercile of median price)")
    axes[1].set_title("By time of day (New York, 30-min windows)")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Where the signal is: high-priced stocks and the afternoon, in both halves", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "final_where.png", dpi=200)
    plt.close(fig)


def curve(path: Path, var: str) -> list[float]:
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["var"] == var and r["hold"] == "5"]
    return [num(r["ic_cs"]) for r in sorted(rows, key=lambda r: int(r["lag"]))]


def fig_latency() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lags = list(range(6))
    t6 = H1 / f"latency_curve_{RUNS_H1['T6']}.csv"
    series = [
        ("Quote imbalance (H1)", curve(t6, "f_imbalance"), "#4c72b0", "-"),
        ("Reversal, -past 5-min return (H1)", [-v for v in curve(t6, "f_logret_5")], "#55a868", "-"),
        ("T1: model trained on t -> t+5 (H1)", curve(H1 / "latency_curve_T1.csv", "y_hat_logret"), "#999999", "--"),
        ("T6: MLP trained on t+1 -> t+6 (H1)", curve(t6, "y_hat_logret"), "#2a7f62", "-"),
        ("H6: same MLP, holdout (H2)", curve(H2 / f"latency_curve_{RUN_H6}.csv", "y_hat_logret"), "#d08c3a", "-"),
    ]
    for name, ys, col, ls in series:
        ax.plot(lags, ys, marker="o", color=col, ls=ls, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Order delay (minutes between decision and fill)")
    ax.set_ylabel("Cross-sectional IC, hold 5 minutes")
    ax.set_title("Imbalance falls off a cliff; reversal and the retrained model decay slowly")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_latency.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    h1 = md_table(H1 / "eval_h5m_T3-T7.md")
    h2 = md_table(H2 / "eval_h5m_H3-H8.md")
    fig_ladder(h1, h2)
    fig_economics(h1, h2)
    fig_where(md_table(H1 / f"hetero_{RUNS_H1['T6']}.md"), md_table(H2 / f"hetero_{RUN_H6}.md"))
    fig_latency()
    for p in sorted(OUT.glob("final_*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
