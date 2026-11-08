"""Plot IC and gross PnL by execution latency from latency_curve.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_CSV = Path(r"E:\4511_5min\wrds_taq\reports\latency_curve.csv")
DEFAULT_HALF_SPREAD_BPS = 1.17
VARIABLE_LABELS = {
    "f_imbalance": "Quote Imbalance",
    "f_microprice_bps": "Microprice",
    "f_logret_5": "Reversal (\u2212f_logret_5)",
    "y_hat_logret": "Logit Ridge Model",
}
EXPECTED_LAGS = tuple(range(6))


def read_hold_rows(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    """Load hold=5 records grouped by variable."""
    grouped: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["hold"]) != 5:
                continue
            grouped.setdefault(row["var"], []).append(
                {
                    "lag": int(row["lag"]),
                    "ic_cs": float(row["ic_cs"]),
                    "gross_bps_per_trade": float(row["gross_bps_per_trade"]),
                }
            )

    if not grouped:
        raise ValueError(f"No hold=5 records found in {csv_path}")

    for var, rows in grouped.items():
        rows.sort(key=lambda row: row["lag"])
        lags = tuple(int(row["lag"]) for row in rows)
        if lags != EXPECTED_LAGS:
            raise ValueError(f"{var} must have exactly lags 0-5; found {lags}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--half-spread-bps",
        type=float,
        default=DEFAULT_HALF_SPREAD_BPS,
        help="Median half-spread in bps; round-trip cost is twice this value",
    )
    args = parser.parse_args()

    grouped = read_hold_rows(args.csv_path)
    output_dir = Path(__file__).resolve().parents[1] / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Hold 5 min \u00b7 47 large caps \u00b7 2024H1 \u00b7 1-min NBBO bars",
        fontsize=11,
    )
    for var, rows in grouped.items():
        sign = -1 if var == "f_logret_5" else 1
        ax.plot(
            [row["lag"] for row in rows],
            [sign * row["ic_cs"] for row in rows],
            marker="o",
            label=VARIABLE_LABELS.get(var, var),
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(EXPECTED_LAGS)
    ax.set_xlabel("Order latency (minutes)")
    ax.set_ylabel("Cross-sectional IC")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "latency_ic.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Hold 5 min \u00b7 47 large caps \u00b7 2024H1 \u00b7 1-min NBBO bars",
        fontsize=11,
    )
    for var, rows in grouped.items():
        sign = -1 if var == "f_logret_5" else 1
        ax.plot(
            [row["lag"] for row in rows],
            [sign * row["gross_bps_per_trade"] for row in rows],
            marker="o",
            label=VARIABLE_LABELS.get(var, var),
        )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(
        2 * args.half_spread_bps,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Taker round-trip cost",
    )
    ax.set_xticks(EXPECTED_LAGS)
    ax.set_xlabel("Order latency (minutes)")
    ax.set_ylabel("Gross PnL per trade (bps)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(output_dir / "latency_pnl.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
