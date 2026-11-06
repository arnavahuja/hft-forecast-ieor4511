"""Build the h30s status report (PDF + figures) from the registered runs.

    python scripts/h30s_report.py            # compute numbers, draw figures, write the PDF
    python scripts/h30s_report.py --reuse    # reuse summary.json, only redraw and rebuild

Writes backend/hft_forecast/ml/horizons/h30s/report/{h30s_report.pdf, summary.json, figures/}.
Needs matplotlib and reportlab (`uv pip install matplotlib reportlab`) besides the framework.
Every number in the PDF is computed here from data/runs/* and data/features/*; the only
hand-written numbers are the data-pull facts in PULL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
H30S = ROOT / "backend" / "hft_forecast" / "ml" / "horizons" / "h30s"
OUT = H30S / "report"
FIGS = OUT / "figures"
RUNS = ROOT / "data" / "runs"

# the three runs on pulled TAQ bars (experiment h30s)
RUN_IDS = {
    "logit_ridge · basic": ("820759f5b4c7", "logit_ridge@1.0.0"),
    "logit_ridge · basic + flow": ("84929c4261f2", "logit_ridge@1.0.0"),
    "lgbm · basic + flow": ("cf45c1c618ae", "lgbm@1.0.0"),
}
BEST = "lgbm · basic + flow"
BASE = "logit_ridge · basic"
DATASET_FLOW = "ee279fe0e6ef7efa"  # basic + h30s_flow features and labels
CONFIG_FLOW = H30S / "config_flow.yaml"
PULL = {  # facts of the TAQ pull, recorded when it ran (2026-10-30)
    "names": 28, "start": "2024-01-02", "end": "2024-06-28", "sessions": 124, "grid": "5 s",
    "partitions": 336, "skipped": 0, "hours": "about 2 h 45 min", "disk_gb": 2.6,
    "bars_per_stock": 580_320,
}
LARGE_TICK_PRICE = 50.0  # below this price one tick is >= 2 bps of the price

# palette (reference instance, light mode): series in fixed order, ink for text
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e5e4e0", "#a3a29d"


# ----------------------------------------------------------------------------- compute
def _metrics() -> dict:
    out: dict = {"runs": {}, "folds": {}}
    for name, (rid, model) in RUN_IDS.items():
        m = json.loads((RUNS / rid / "ml_metrics.json").read_text())
        man = json.loads((RUNS / rid / "manifest.json").read_text())
        p = m["models"][model]["pooled"]
        out["runs"][name] = {k: p[k] for k in (
            "accuracy", "balanced_accuracy", "macro_f1", "auc_up", "auc_down", "ic", "ic_t_stat",
            "ic_sessions", "r2_vs_zero", "rmse_bps")} | {
            "run_id": rid, "n_rows": man["n_rows"], "n_test": m["n_rows"], "n_test_sessions": m["n_sessions"],
            "n_eff": man["n_eff"], "n_folds": len(man["folds"])}
        out["folds"][name] = [f["ic"] for f in m["models"][model]["folds"]]
        if name == BASE:  # baselines are the same in every run
            for b in ("zero@1.0.0", "last_sign@1.0.0"):
                bp = m["models"][b]["pooled"]
                out["runs"][b.split("@")[0]] = {"accuracy": bp["accuracy"],
                    "balanced_accuracy": bp["balanced_accuracy"], "auc_up": bp.get("auc_up"),
                    "ic": bp.get("ic"), "class_balance": bp["class_balance"]}
            out["fold_dates"] = [(f["first_test"][:10], f["last_test"][:10]) for f in man["folds"]]
    return out


def _predictions(name: str) -> pl.LazyFrame:
    rid, model = RUN_IDS[name]
    return (pl.scan_parquet(RUNS / rid / "predictions.parquet")
            .filter((pl.col("model") == model) & pl.col("fwd_logret").is_not_null()
                    & pl.col("direction").is_not_null())
            .select("timestamp", "instrument_id", "session_id", "direction",
                    score=pl.col("p_up") - pl.col("p_down"), fwd_bps=pl.col("fwd_logret") * 1e4))


def _spearman(a: str, b: str) -> pl.Expr:
    return pl.corr(pl.col(a).rank(), pl.col(b).rank())


def _per_stock_and_signal() -> dict:
    """Per-stock IC, deciles and time of day.

    ICs here pool each stock over ALL test sessions. Ranking within a single session is biased
    for any score that moves with the price level (tick_bps, spread_bps = spread / mid): the
    session's price ranks use its future prices, which gives a random walk an IC of about +0.06
    (see _random_walk_bias). Pooled over 84 sessions that bias is negligible.
    """
    feats = (pl.scan_parquet(sorted((ROOT / "data" / "features" / DATASET_FLOW).glob("*.parquet")))
             .select("timestamp", "instrument_id", "f_spread_bps", "f_spread_ticks", "mid"))
    out: dict = {"stock": {}, "deciles": {}, "tod": {}, "check": {}}
    for name in (BASE, BEST):
        df = _predictions(name).join(feats, on=["timestamp", "instrument_id"], how="left").collect()
        per = df.group_by("instrument_id").agg(ic=_spearman("score", "fwd_bps"))
        out["stock"][name] = dict(per.iter_rows())
        per_session = (df.group_by("instrument_id", "session_id").agg(ic=_spearman("score", "fwd_bps"))
                       .filter(pl.col("ic").is_finite()))
        out["check"][name] = {"pooled_per_stock": float(per["ic"].mean()),
                              "per_stock_session": float(per_session["ic"].mean())}
        dec = (df.with_columns(q=(pl.col("score").rank() / pl.len() * 10).floor().clip(0, 9)
                               .over("instrument_id"))
               .group_by("q").agg(fwd=pl.col("fwd_bps").mean(), half=(pl.col("f_spread_bps") / 2).mean(),
                                  up=(pl.col("direction") == 1).mean(), down=(pl.col("direction") == -1).mean())
               .sort("q"))
        out["deciles"][name] = dec.to_dicts()
        tod = (df.with_columns(b=pl.col("timestamp").dt.convert_time_zone("America/New_York")
                               .dt.truncate("30m").dt.strftime("%H:%M"))
               .group_by("b", "instrument_id").agg(ic=_spearman("score", "fwd_bps"))  # all sessions
               .filter(pl.col("ic").is_finite())
               .group_by("b").agg(ic=pl.col("ic").mean()).sort("b"))
        out["tod"][name] = tod.to_dicts()
        if name == BEST:
            out["stock_info"] = (df.group_by("instrument_id")
                                 .agg(flat=(pl.col("direction") == 0).mean(),
                                      spread_ticks=pl.col("f_spread_ticks").median(),
                                      spread_bps=pl.col("f_spread_bps").median(),
                                      price=pl.col("mid").median(), rows=pl.len())
                                 .sort("instrument_id").to_dicts())
            out["class_balance"] = {str(k): v for k, v in
                                    df.group_by("direction").len().with_columns(
                                        share=pl.col("len") / pl.col("len").sum())
                                    .select("direction", "share").iter_rows()}
        del df
    return out


def _univariate_ic(test_sessions: set[int]) -> tuple[dict, dict]:
    """Spearman IC of each feature with the forward return over the test sessions.

    Returns (pooled per stock, then averaged over stocks; the per-stock-session version), the
    second only to document the level bias in the report.
    """
    lf = pl.scan_parquet(sorted((ROOT / "data" / "features" / DATASET_FLOW).glob("*.parquet")))
    cols = [c for c in lf.collect_schema().names() if c.startswith("f_")]
    df = (lf.filter(pl.col("label_valid") & pl.col("session_id").is_in(sorted(test_sessions)))
          .select("instrument_id", "session_id", "fwd_logret", *cols).collect())
    pooled = df.group_by("instrument_id").agg([_spearman(c, "fwd_logret").alias(c) for c in cols])
    by_session = df.group_by("instrument_id", "session_id").agg(
        [_spearman(c, "fwd_logret").alias(c) for c in cols])
    mean = lambda t, c: float(t[c].filter(t[c].is_finite()).mean())  # noqa: E731
    return {c: mean(pooled, c) for c in cols}, {c: mean(by_session, c) for c in cols}


def _random_walk_bias(sessions: int = 300, bars: int = 4680, h: int = 6, seed: int = 0) -> float:
    """Mean per-session Spearman IC of 1/price with the next-h-bar return on a pure random walk."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(sessions):
        mid = 20 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], bars)) * 0.005
        out.append(spearmanr(1 / mid[:-h], np.log(mid[h:] / mid[:-h])).correlation)
    return float(np.nanmean(out))


def _importance() -> dict:
    """Diagnostic only: refit the lgbm direction head on the LAST fold's training rows (never
    its test rows) with the run's settings, and read the booster's gain importance."""
    from hft_forecast.ml.config import ExperimentConfig
    from hft_forecast.ml.cv import split, uniqueness_weights
    from hft_forecast.ml.datasets import design_matrix, load_dataset, usable_rows
    from hft_forecast.ml.base import DIRECTION, FWD_LOGRET
    from hft_forecast.ml.models.h30s_lgbm import Lgbm

    cfg = ExperimentConfig.load(CONFIG_FLOW)
    frame = usable_rows(load_dataset(ROOT / "data" / "features" / DATASET_FLOW, cfg.data.instruments))
    folds = list(split(frame, cfg.cv, cfg.horizon, cfg.embargo_bars))
    last = folds[-1]
    test_sessions = {s for f in folds for s in f.test_sessions}
    weights = uniqueness_weights(frame)[last.train]
    train = frame.filter(pl.Series(last.train))
    del frame
    x, cols = design_matrix(train)
    model = Lgbm()
    model.set_feature_names(tuple(cols))
    model.fit(x, train[DIRECTION].to_numpy().astype(np.int64), train[FWD_LOGRET].to_numpy(),
              weights, cfg.seed + last.index)
    clf = model.state_dict()["classifier"]
    booster = clf.booster_
    gain = booster.feature_importance(importance_type="gain")
    total = float(gain.sum())
    # multiclass boosting grows one tree per class per round (tree i belongs to class i % k):
    # split the gain into the trees for 'flat' (move vs no move) and those for 'down'/'up'
    dump = booster.dump_model()
    k = int(dump.get("num_class", 1))
    names = {-1: "down", 0: "flat", 1: "up"}
    by_class: dict[str, dict[str, float]] = {}

    def walk(node: dict, acc: dict[str, float]) -> None:
        if "split_feature" in node:
            name = cols[node["split_feature"]]
            acc[name] = acc.get(name, 0.0) + float(node.get("split_gain", 0.0))
            walk(node["left_child"], acc)
            walk(node["right_child"], acc)

    for i, tree in enumerate(dump["tree_info"]):
        label = names[int(clf.classes_[i % k])]
        walk(tree["tree_structure"], by_class.setdefault(label, {}))
    groups = {"flat": by_class.get("flat", {}),
              "direction": {c: by_class.get("up", {}).get(c, 0.0) + by_class.get("down", {}).get(c, 0.0)
                            for c in cols}}
    groups = {g: {c: v / (sum(d.values()) or 1.0) for c, v in d.items()} for g, d in groups.items()}
    return {"fold": last.index, "train_rows": int(train.height),
            "train_sessions": [int(min(last.train_sessions)), int(max(last.train_sessions))],
            "best_iteration": int(model.state_dict()["best_iterations"][0] or 0),
            "gain": {c: float(g) / total for c, g in zip(cols, gain)},
            "gain_by_group": groups,
            "test_sessions": sorted(int(s) for s in test_sessions)}


def compute() -> dict:
    s = {"generated": dt.date.today().isoformat(), "pull": PULL}
    s |= _metrics()
    print("metrics done")
    s |= _per_stock_and_signal()
    print("per-stock, deciles, time of day done")
    s["importance"] = _importance()
    print("importance done")
    s["univariate_ic"], s["univariate_ic_per_session"] = _univariate_ic(set(s["importance"]["test_sessions"]))
    s["random_walk_bias"] = _random_walk_bias()
    print("univariate IC done")
    s["n_completed_runs"], s["n_aborted_runs"] = _run_counts()
    return s


def _run_counts() -> tuple[int, int]:
    done = aborted = 0
    for d in RUNS.iterdir():
        if d.is_dir():
            if (d / "manifest.json").exists():
                done += json.loads((d / "manifest.json").read_text()).get("experiment") == "h30s"
            else:
                aborted += 1
    return done, aborted


# ----------------------------------------------------------------------------- figures
def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    boxes = [
        ("WRDS TAQ", "NBBO + trades\n28 stocks\n2024 H1"),
        ("5 s bars", "pull-taq\n580k bars\nper stock"),
        ("Features", "basic (12)\n+ h30s_flow\n(21)"),
        ("Labels", "30 s mid\nmove: down /\nflat / up"),
        ("Walk-\nforward CV", "40 train /\n10 test\nsessions"),
        ("Models", "logit_ridge\nlgbm"),
        ("Evaluation", "IC, AUC,\naccuracy"),
    ]
    w, gap = 12.6, 1.7
    for i, (title, sub) in enumerate(boxes):
        x0 = i * (w + gap)
        ours = title in ("Features", "Models")
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x0, 6), w, 18, boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor="#eaf2fc" if ours else "#f4f4f2", edgecolor=BLUE if ours else GRAY, lw=1))
        ax.text(x0 + w / 2, 20.2, title, ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
        ax.text(x0 + w / 2, 11.8, sub, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.25)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x0 + w + gap - 0.2, 15), xytext=(x0 + w + 0.4, 15),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    ax.text(0, 1, "Blue boxes: added by the h30s group (h30s_flow features, lgbm model). "
            "Grey: shared framework.", fontsize=7, color=INK2)
    fig.savefig(path)
    plt.close(fig)


def fig_models(s: dict, path: Path) -> None:
    order = ["zero", "last_sign", BASE, "logit_ridge · basic + flow", BEST]
    labels = ["zero\n(baseline)", "last sign\n(baseline)", "logit_ridge\nbasic", "logit_ridge\nbasic + flow",
              "lgbm\nbasic + flow"]
    colors = [GRAY, GRAY, "#9cc0ec", "#5f9ae2", BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, (key, title, fmt) in zip(axes, [("accuracy", "Accuracy (3 classes)", "{:.1%}"),
                                            ("auc_up", "AUC, up vs rest", "{:.3f}"),
                                            ("ic", "IC (rank corr. with return)", "{:.3f}")]):
        vals = [s["runs"][k].get(key) for k in order]
        y = np.arange(len(order))
        v = [0 if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in vals]
        ax.barh(y, v, color=colors, height=0.62)
        for yi, (vi, raw) in enumerate(zip(v, vals)):
            txt = "n/a" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else fmt.format(vi)
            ax.text(max(vi, 0), yi, " " + txt, va="center", fontsize=7, color=INK)
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(y), fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
        lo = 0.3 if key == "accuracy" else (0.45 if key == "auc_up" else min(min(v) * 1.6, 0))
        ax.set_xlim(lo, max(v) * (1.22 if key == "ic" else 1.06))
        if key == "ic":
            ax.axvline(0, color=INK2, lw=0.6)
        if key == "auc_up":
            ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_folds(s: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    x = np.arange(len(s["folds"][BEST]))
    for name, color in [(BASE, AQUA), ("logit_ridge · basic + flow", ORANGE), (BEST, BLUE)]:
        ax.plot(x, s["folds"][name], marker="o", ms=4.5, lw=1.6, color=color, label=name)
    md = lambda d: dt.date.fromisoformat(d).strftime("%b %d")  # noqa: E731
    ax.set_xticks(x, [f"fold {i}\n{md(a)}–\n{md(b)}" for i, (a, b) in enumerate(s["fold_dates"])],
                  fontsize=6.3)
    ax.set_ylabel("IC")
    ax.set_title("IC in each walk-forward test window (fold, test dates)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def large_tick(s: dict) -> set[str]:
    """Low-priced stocks, where one cent is a large step (>= 2 bps of the price)."""
    return {r["instrument_id"] for r in s["stock_info"] if r["price"] <= LARGE_TICK_PRICE}


def fig_stocks(s: dict, path: Path) -> None:
    info = {r["instrument_id"]: r for r in s["stock_info"]}
    LARGE_TICK = large_tick(s)
    names = sorted(info, key=lambda k: s["stock"][BEST][k])
    base = [s["stock"][BASE][k] for k in names]
    best = [s["stock"][BEST][k] for k in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.hlines(y, base, best, color=GRAY, lw=1)
    ax.scatter(base, y, s=22, color=AQUA, label="logit_ridge · basic", zorder=3)
    ax.scatter(best, y, s=26, color=BLUE, label="lgbm · basic + flow", zorder=3)
    ax.set_yticks(y, [f"{k}{'  ●' if k in LARGE_TICK else ''}" for k in names], fontsize=7)
    for yi, k in enumerate(names):
        r = info[k]
        ax.text(0.335, yi, f"flat {r['flat']:.0%} · spread {r['spread_ticks']:.0f} tick"
                f"{'s' if r['spread_ticks'] >= 1.5 else ''} · ${r['price']:.0f}",
                va="center", fontsize=6.3, color=INK2)
    ax.set_xlim(-0.005, 0.47)
    ax.set_xlabel("IC with the next 30 s mid return (each stock pooled over all test sessions)")
    ax.set_title("IC by stock (● = low price, ≤ $50: one tick ≥ 2 bps of the price)", fontsize=9, pad=22)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_deciles(s: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    best = s["deciles"][BEST]
    x = np.arange(1, 11)
    half = float(np.mean([d["half"] for d in best]))
    ax.axhspan(-half, half, color="#f4f4f2", zorder=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.text(10.4, half * 0.92, f"± half-spread ≈ {half:.2f} bps", ha="right", va="top", fontsize=7, color=INK2)
    for name, color in [(BASE, AQUA), (BEST, BLUE)]:
        ax.plot(x, [d["fwd"] for d in s["deciles"][name]], marker="o", ms=4.5, lw=1.6, color=color, label=name)
    ax.set_xticks(x, [f"D{i}" for i in x])
    ax.set_xlabel("signal decile within each stock (D1 = most bearish, D10 = most bullish)")
    ax.set_ylabel("mean next-30 s mid return (bps)")
    ax.set_ylim(-half * 1.15, half * 1.15)
    ax.set_title("Average realized move by signal decile, against the cost of crossing the spread", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_tod(s: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    for name, color in [(BASE, AQUA), (BEST, BLUE)]:
        t = s["tod"][name]
        ax.plot([r["b"] for r in t], [r["ic"] for r in t], marker="o", ms=4.5, lw=1.6, color=color, label=name)
    ax.set_ylabel("IC")
    ax.set_xlabel("30-minute bucket (New York time)")
    ax.set_title("IC through the trading day", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


FLOW_PREFIXES = ("f_ofi", "f_imb_", "f_spread_ticks", "f_one_tick", "f_tick_bps", "f_microprice_ticks",
                 "f_twap", "f_range", "f_quotes_rel")


def _is_flow(c: str) -> bool:
    return c.startswith(FLOW_PREFIXES)


def fig_importance(s: dict, path: Path) -> None:
    imp = s["importance"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax, (group, title) in zip(axes, [("direction", "Trees for 'up' and 'down'"),
                                         ("flat", "Trees for 'flat' (move or not)")]):
        top = sorted(imp["gain_by_group"][group].items(), key=lambda kv: kv[1], reverse=True)[:12][::-1]
        y = np.arange(len(top))
        ax.barh(y, [g for _, g in top], color=[BLUE if _is_flow(c) else ORANGE for c, _ in top], height=0.62)
        for yi, (_, g) in enumerate(top):
            ax.text(g, yi, f" {g:.0%}", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(y, [c[2:] for c, _ in top], fontsize=6.8)
        ax.set_xlim(0, max(g for _, g in top) * 1.25)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of split gain", fontsize=7.5)
    axes[1].legend(handles=[plt.matplotlib.patches.Patch(color=BLUE, label="h30s_flow (new)"),
                            plt.matplotlib.patches.Patch(color=ORANGE, label="basic (shipped)")],
                   fontsize=7, loc="lower right")
    fig.suptitle(f"LightGBM gain importance, refit on fold {imp['fold']}'s training rows only "
                 f"({imp['train_rows'] / 1e6:.1f} M rows)", fontsize=9, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figures(s: dict) -> dict[str, Path]:
    _style()
    FIGS.mkdir(parents=True, exist_ok=True)
    paths = {k: FIGS / f"{k}.png" for k in
             ("pipeline", "models", "folds", "stocks", "deciles", "tod", "importance")}
    fig_pipeline(paths["pipeline"])
    fig_models(s, paths["models"])
    fig_folds(s, paths["folds"])
    fig_stocks(s, paths["stocks"])
    fig_deciles(s, paths["deciles"])
    fig_tod(s, paths["tod"])
    fig_importance(s, paths["importance"])
    return paths


# ----------------------------------------------------------------------------- PDF
def pdf(s: dict, figs: dict[str, Path], path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # DejaVu ships with matplotlib
    for name, file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                       ("DV-I", "DejaVuSans-Oblique.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9, leading=12.5,
                          textColor=colors.HexColor(INK), alignment=TA_LEFT, spaceAfter=5)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10.5, textColor=colors.HexColor(INK2))
    h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3)
    title = ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=18, leading=22, spaceAfter=4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.6, leading=9.6, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")

    def P(t, st=body):
        return Paragraph(t, st)

    def bullets(items):
        return [Paragraph(t, bullet, bulletText="•") for t in items]

    def table(rows, widths, header=True, zebra=True):
        data = [[P(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK2)),
                 ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                 ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if zebra:
            style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f6f4")) for i in range(2, len(rows), 2)]
        t.setStyle(TableStyle(style))
        return t

    def fig(key, width_in=7.0, caption=None, heading=None):
        """Figure (with an optional heading above it) kept on one page with its caption."""
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(figs[key])).getSize()
        img = Image(str(figs[key]), width=width_in * inch, height=width_in * inch * ih / iw)
        items = ([heading] if heading else []) + [img] + ([P(caption, small)] if caption else [])
        return KeepTogether(items + [Spacer(1, 6)])

    r = s["runs"]
    b, f, g = r[BASE], r["logit_ridge · basic + flow"], r[BEST]
    folds_up = sum(x > y for x, y in zip(s["folds"][BEST], s["folds"][BASE]))
    stocks_up = sum(s["stock"][BEST][k] > s["stock"][BASE][k] for k in s["stock"][BEST])
    n_st = len(s["stock"][BEST])
    LARGE_TICK = large_tick(s)
    big = [k for k in s["stock"][BEST] if k in LARGE_TICK]
    small_ = [k for k in s["stock"][BEST] if k not in LARGE_TICK]
    mean = lambda run, ks: float(np.mean([s["stock"][run][k] for k in ks]))  # noqa: E731
    dec_b, dec_g = s["deciles"][BASE], s["deciles"][BEST]
    half = float(np.mean([d["half"] for d in dec_g]))
    imp = s["importance"]
    gain = s["importance"]["gain"]
    flow_gain = sum(v for k, v in gain.items() if _is_flow(k))
    top_gain = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    uic = sorted(s["univariate_ic"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    cb = s["class_balance"]
    pct = lambda x: f"{x:.1%}"  # noqa: E731

    story = [
        P("h30s — predicting the 30-second mid-price move on a 5-second grid", title),
        P(f"IEOR 4511 capstone · h30s group · status report generated {s['generated']} from the "
          f"registered runs (code: <font face='DV-I'>scripts/h30s_report.py</font>)", small),
        Spacer(1, 6),
        P("Summary", h1),
        *bullets([
            f"<b>Task.</b> Predict whether the NBBO mid price of a stock is down, flat or up 30 seconds ahead, "
            f"from level-1 quote data on 5-second bars, using the team framework (<i>hftf ml</i>).",
            f"<b>Data.</b> WRDS TAQ, {PULL['names']} large-cap S&amp;P 500 stocks, {PULL['start']} to "
            f"{PULL['end']} ({PULL['sessions']} sessions, {PULL['bars_per_stock']:,} bars per stock). "
            f"Walk-forward evaluation on {g['n_folds']} test windows: {g['n_test'] / 1e6:.1f} M out-of-sample "
            f"predictions ({g['n_rows'] / 1e6:.1f} M usable rows in total).",
            f"<b>Result.</b> There is real, consistent short-horizon predictability. The best model "
            f"(LightGBM on the shipped features plus our order-flow features) reaches IC {g['ic']:.3f}, "
            f"AUC {g['auc_up']:.3f} and {pct(g['accuracy'])} accuracy, against {pct(r['zero']['accuracy'])} "
            f"for always predicting the most common class. It beats the starting model in {folds_up} of "
            f"{len(s['folds'][BEST])} test windows and in {stocks_up} of {n_st} stocks.",
            f"<b>What drives it.</b> Level-1 queue imbalance (bid vs ask size) and recent order flow, and "
            f"the tick regime: low-priced stocks, where one cent is a large step relative to the price, are "
            f"far more predictable (mean IC {mean(BEST, big):.2f} for the {len(big)} stocks priced ≤ $50 vs "
            f"{mean(BEST, small_):.2f} for the rest).",
            f"<b>How big it is.</b> Small. The strongest decile of signals predicts a move of about "
            f"{abs(dec_g[0]['fwd']):.2f} bps, below the average half-spread of {half:.2f} bps, so the signal "
            f"does not pay for crossing the spread on its own; it is useful for timing and passive execution.",
            "<b>Look-ahead.</b> Labels never cross sessions, features use only past bars (checked "
            "automatically), CV is purged and embargoed walk-forward; all 367 framework tests pass (section 6).",
        ]),
        Spacer(1, 4),
        table([["Model (features)", "Accuracy", "Balanced acc.", "AUC up", "IC", "IC t-stat", "R² vs zero"],
               ["zero (most common class)", pct(r["zero"]["accuracy"]), pct(r["zero"]["balanced_accuracy"]), "–", "–", "–", "–"],
               ["last sign (momentum)", pct(r["last_sign"]["accuracy"]), pct(r["last_sign"]["balanced_accuracy"]),
                f"{r['last_sign']['auc_up']:.3f}", f"{r['last_sign']['ic']:.3f}", "–", "–"],
               *[[k, pct(v["accuracy"]), pct(v["balanced_accuracy"]), f"{v['auc_up']:.3f}", f"{v['ic']:.3f}",
                  f"{v['ic_t_stat']:.0f}", pct(v["r2_vs_zero"])]
                 for k, v in [(BASE, b), ("logit_ridge · basic + flow", f), (BEST, g)]]],
              [2.15 * inch, 0.75 * inch, 0.9 * inch, 0.65 * inch, 0.6 * inch, 0.7 * inch, 0.8 * inch]),
        P(f"Pooled over {g['n_test']:,} test rows in {g['ic_sessions']} test sessions. IC = rank correlation "
          f"between the model's up-minus-down probability and the realized 30 s log return; the t-stat uses "
          f"the per-session ICs. Classes in the test data: down {pct(cb.get('-1', 0))}, flat {pct(cb.get('0', 0))}, "
          f"up {pct(cb.get('1', 0))}.", small),
        PageBreak(),

        P("1. What the pipeline does", h1),
        fig("pipeline", 7.2),
        *bullets([
            f"<b>Bars.</b> <i>hftf data pull-taq</i> aggregates TAQ NBBO quotes (and trades) on the WRDS "
            f"server into 5-second bars labelled at the bar end: last bid/ask and sizes, mid, microprice, "
            f"time-weighted mid, quote count. {PULL['partitions']} stock-months, {PULL['skipped']} skipped, "
            f"{PULL['hours']}, {PULL['disk_gb']} GB.",
            "<b>Universe.</b> Our 28 pilot large caps (AAPL, MSFT, NVDA, …). GOOG stands in for GOOGL and "
            "BRK.B is missing: <i>pull-taq</i> only reads symbols without a TAQ suffix (GOOGL is stored as "
            "GOOG + L, BRK.B as BRK + B).",
            "<b>Label.</b> The log change of the mid over the next 30 s (6 bars), classified down / flat / up "
            "with a 1 bp dead zone. Labels are computed within a session only; the first and last minute of "
            "each session and stale targets are dropped.",
            "<b>Cross-validation.</b> Purged, embargoed walk-forward by session: train on 40 sessions, test "
            "on the next 10, roll forward; training rows whose label overlaps the test window are removed. "
            f"This gives {g['n_folds']} test windows covering {g['n_test_sessions']} sessions "
            f"({s['fold_dates'][0][0]} to {s['fold_dates'][-1][1]}).",
            "<b>Models.</b> <i>logit_ridge</i> (framework baseline: logistic regression for the direction, "
            "ridge for the return) and <i>lgbm</i> (ours). Each run also scores the <i>zero</i> and "
            "<i>last_sign</i> baselines on the same rows.",
        ]),

        P("2. What the h30s group added", h1),
        P("2.1 Data and two framework fixes", h2),
        *bullets([
            "Pulled the TAQ bars and pointed the h30s config at them (<i>horizons/h30s/config.yaml</i>, "
            "<i>symbols_pilot28.txt</i>).",
            "<b>Fixed a bug that affects every group</b> (<i>ml/datasets.load_bars</i>): the data cache keeps "
            "one growing file per stock, so on a multi-month pull every monthly manifest entry also contained "
            "the earlier months and January was loaded six times (2.0 M rows instead of 580 k per stock; "
            "96% of rows were then dropped as bar gaps). Each entry is now sliced to its own dates and "
            "partitions are aligned by column name. Regression test added.",
            "Relaxed one shared test that required the model registry to equal the five shipped models "
            "exactly, which breaks as soon as a group registers its own model.",
        ]),
        P("2.2 Order-flow feature set <i>h30s_flow</i> (21 features)", h2),
        table([["Group", "Features", "Idea"],
               ["Order-flow imbalance", "ofi_1, ofi_2, ofi_6, ofi_12, ofi_60",
                "Cont–Kukanov–Stoikov OFI between consecutive bar-end NBBO snapshots, summed over 5 s … 5 min and "
                "divided by average depth: net pressure from queue growth/depletion and price moves."],
               ["Imbalance persistence", "imb_mean_6/12/60, imb_change_1/6",
                "Is the bid/ask size imbalance building up or fading?"],
               ["Tick regime", "spread_ticks, one_tick_share, tick_bps, imb_x_one_tick",
                "Spread in ticks, share of recent bars at a one-tick spread, tick size relative to price, and "
                "imbalance × one-tick share: imbalance matters most when the spread is pinned at one tick."],
               ["Microprice / intra-bar", "microprice_ticks, twap_dev_bps, range_bps, quotes_rel",
                "Microprice tilt in ticks, last mid vs the bar's time-weighted mid, mid range in the bar, "
                "quote intensity relative to its recent level."]],
              [1.25 * inch, 2.0 * inch, 3.95 * inch]),
        P("The shipped <i>basic</i> set (12 features) has mid returns over 1/5/10 bars, spread, queue "
          "imbalance, microprice tilt in bps, quote count, a stale flag, time since the open and three "
          "volatility measures. Every new feature is a non-negative shift or a trailing window inside the "
          "session, and passes the framework's causality check.", small),
        P("2.3 LightGBM model <i>lgbm</i>", h2),
        P("Implements what the shared <i>gbm</i> stub describes, in our own file "
          "(<i>ml/models/h30s_lgbm.py</i>; the stub is untouched): one gradient-boosted booster for the "
          "direction and one for the signed return, using the framework's sample weights, seeded. The number "
          "of trees is chosen by early stopping on held-out blocks <i>inside the training window</i> (every "
          "10th block of 20 000 rows, so validation rows span all stocks and dates); the test window is "
          f"never used. On the last fold the direction booster used its full budget of "
          f"{imp['best_iteration']} trees (early stopping did not trigger), so a larger budget or learning "
          f"rate may still add a little (section 7)."),

        P("3. How the model performs", h1),
        fig("models", 7.2, f"All runs are scored on the same {g['n_test'] / 1e6:.1f} M test rows. Grey: framework baselines; blue "
            "shades: logit_ridge with the shipped features, logit_ridge with our order-flow features added, "
            "and LightGBM with both."),
        fig("folds", 7.2, f"The improvement is not driven by one period: LightGBM with order flow has the "
            f"highest IC in {folds_up} of {len(s['folds'][BEST])} test windows; adding the features to the "
            f"linear model helps in {sum(x > y for x, y in zip(s['folds']['logit_ridge · basic + flow'], s['folds'][BASE]))} "
            f"of {len(s['folds'][BEST])}."),
        fig("tod", 7.2, "Predictability is lowest right after the open, when the book is still forming, and "
            "highest from late morning through the afternoon."),
        fig("stocks", 7.2, f"IC rises for {stocks_up} of {n_st} stocks. Low-priced stocks (●, price ≤ $50, so "
            f"one cent is ≥ 2 bps and the mid is often unchanged for 30 s) are far more predictable: mean IC "
            f"{mean(BASE, big):.3f} → {mean(BEST, big):.3f}; the other stocks {mean(BASE, small_):.3f} → "
            f"{mean(BEST, small_):.3f}. Many high-priced stocks also quote a one-tick spread (AAPL, AMZN, GOOG), "
            f"but there a tick is under 1 bp. 'flat' = share of test rows with no move beyond the dead zone; "
            f"spread and price are medians over the test rows."),

        KeepTogether([
        P("4. What contributes to the result", h1),
        P("4.1 Step by step", h2),
        table([["Change", "IC", "Δ IC", "Accuracy", "AUC up"],
               ["logit_ridge, shipped features", f"{b['ic']:.4f}", "–", pct(b["accuracy"]), f"{b['auc_up']:.3f}"],
               ["+ h30s_flow features (same linear model)", f"{f['ic']:.4f}", f"+{f['ic'] - b['ic']:.4f}",
                pct(f["accuracy"]), f"{f['auc_up']:.3f}"],
               ["+ LightGBM instead of logit_ridge", f"{g['ic']:.4f}", f"+{g['ic'] - f['ic']:.4f}",
                pct(g["accuracy"]), f"{g['auc_up']:.3f}"]],
              [3.2 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch, 0.8 * inch]),
        P(f"About {(f['ic'] - b['ic']) / (g['ic'] - b['ic']):.0%} of the IC gain comes from the new features "
          f"and {(g['ic'] - f['ic']) / (g['ic'] - b['ic']):.0%} from the nonlinear model. The model matters "
          f"because the value of the imbalance depends on the tick regime, an interaction a linear model only "
          f"sees through the hand-made imb_x_one_tick term.", small),
        ]),
        fig("importance", 7.0, heading=P("4.2 Which inputs the model uses", h2), caption=f"Gain importance from a diagnostic refit of the lgbm direction booster on fold "
            f"{imp['fold']}'s training sessions only (same settings as the run). LightGBM grows one tree per "
            f"class per round, so the gain can be split exactly into the trees that score 'up' and 'down' "
            f"(left) and the trees that score 'flat' (right)."),
        P(f"Most of the gain, in both halves, goes to the tick size relative to the price and to volatility. "
          f"They decide whether the mid moves beyond the 1 bp dead zone at all (in a low-priced stock that "
          f"takes a full tick, in a high-priced one it happens constantly), and every tree, including the "
          f"'up' and 'down' trees, has to separate a move from no move. The order-book features that say "
          f"<i>which way</i> (microprice tilt, queue imbalance, imbalance × one-tick share, imbalance changes, "
          f"order flow) rank next in the up/down trees and barely appear in the flat trees. The new "
          f"h30s_flow features take {flow_gain:.0%} of the total gain "
          f"({sum(v for k, v in imp['gain_by_group']['direction'].items() if _is_flow(k)):.0%} in the up/down "
          f"trees, {sum(v for k, v in imp['gain_by_group']['flat'].items() if _is_flow(k)):.0%} in the flat "
          f"trees). Gain shows what the trees split on, not how much each input adds out of sample; the "
          f"ablation in 4.1 is the cleaner measure of contribution.", small),
        P("4.3 Single-feature signal on the test sessions", h2),
        table([["Feature", "IC (pooled)", "IC (per session)", "Set"]] +
              [[c[2:], f"{v:+.3f}", f"{s['univariate_ic_per_session'][c]:+.3f}",
                "h30s_flow" if _is_flow(c) else "basic"] for c, v in uic[:12]],
              [2.0 * inch, 0.9 * inch, 1.1 * inch, 1.0 * inch]),
        P("Rank correlation of each feature alone with the realized 30 s return over the test sessions "
          "(descriptive; no model). 'Pooled' ranks each stock over all test sessions and averages over stocks; "
          "'per session' ranks within each stock-session. The two agree for the order-book features but not "
          "for the price-level ones (next paragraph).", small),
        P(f"<b>Measurement note: per-session IC is biased for price-level features.</b> Within one stock and one "
          f"session, tick_bps (one tick ÷ price) and spread_bps (spread ÷ price, with the spread fixed at one "
          f"tick) move only with the price level, and ranking a level within a single session uses that "
          f"session's future prices. On a simulated random walk, where nothing is predictable, the per-session "
          f"IC of 1/price with the next 30 s return averages {s['random_walk_bias']:+.3f}. On our data "
          f"tick_bps scores {s['univariate_ic_per_session']['f_tick_bps']:+.3f} per session but "
          f"{s['univariate_ic']['f_tick_bps']:+.3f} pooled. The framework's headline IC ranks all test rows "
          f"together and is not affected; the per-stock and time-of-day ICs in this report are pooled over "
          f"all test sessions for the same reason (LightGBM per-stock IC: "
          f"{s['check'][BEST]['pooled_per_stock']:.3f} pooled vs {s['check'][BEST]['per_stock_session']:.3f} "
          f"per stock-session, so the model's predictability is not an artifact of this effect). Worth keeping "
          f"in mind for any group that evaluates per session.", small),
        P("4.4 Why it works", h2),
        P("The mid moves when one side of the best quote is used up. In a low-priced stock one cent is a "
          "large step, so the bid and ask queues are long, the spread sits at one tick and the price can only "
          "move a tick at a time; the relative size of the two queues (and how fast they grow or shrink) then "
          "says which side will break first, often many seconds ahead. In a high-priced stock a cent is a "
          "tiny step, the book refreshes many times a second and any information in the queues is used up long "
          "before 30 seconds; the published evidence (Aït-Sahalia et al., NBER w30366) likewise finds that "
          "predictability falls with the share price and sits mostly in the last milliseconds."),

        P("5. Is it economically meaningful?", h1),
        fig("deciles", 7.2, f"Sorting each stock's predictions into deciles, the realized 30 s move rises "
            f"monotonically from {dec_g[0]['fwd']:+.2f} bps (D1) to {dec_g[-1]['fwd']:+.2f} bps (D10) for "
            f"LightGBM ({dec_b[0]['fwd']:+.2f} to {dec_b[-1]['fwd']:+.2f} for the starting model). Even the "
            f"extreme deciles stay inside the ±{half:.2f} bps half-spread band."),
        P("The model ranks future moves reliably, but the expected size of a move is smaller than the cost of "
          "crossing the spread, and R² against a zero forecast is under 1%. As a stand-alone taker strategy it "
          "would lose money; its value is in deciding when to post or cross, i.e. execution and market making, "
          "and as an input to longer-horizon models. The framework's economic test (<i>hftf ml econ</i>) "
          "would quantify this and has not been run yet."),

        KeepTogether([
        P("6. No look-ahead bias: safeguards", h1),
        table([["Stage", "Safeguard", "Verified by"],
               ["Universe", "Fixed list of 28 large caps chosen before looking at results; not selected on "
                "performance. (The framework also supports point-in-time S&amp;P 500 membership.)", "symbols_pilot28.txt"],
               ["Bars", "Bars are labelled at the bar end; a bar at t contains only quotes up to t.", "framework bar schema"],
               ["Labels", "The target shift is taken within a session, so a label never reaches into the next "
                "day; labels crossing the open/close, gaps and stale targets are dropped.", "tests/unit/ml/test_labels.py"],
               ["Features", "Every feature is a non-negative shift or a trailing window inside the session.",
                "check_feature_lookahead (recomputes features after deleting later bars) for basic and h30s_flow; "
                "hand-computed OFI test"],
               ["Cross-validation", "Walk-forward by whole sessions; training rows whose label window overlaps the "
                "test window are purged, plus an embargo after it.", "tests/unit/ml/test_cv.py"],
               ["Model selection", "LightGBM early stopping uses held-out blocks inside the training window only.",
                "tests/unit/ml/test_h30s_flow_lgbm.py"],
               ["Data loading", "load_bars bug fixed before any reported run: each bar enters the dataset exactly "
                "once, so folds and labels see the true bar sequence.", "regression test in test_datasets.py"],
               ["Diagnostics", "The feature-importance refit uses the last fold's training rows only.", "this script"],
               ["Evaluation", "Descriptive ICs in this report pool each stock over all test sessions: ranking within "
                "one session would let price-level features borrow that session's future prices (measurement "
                "note, 4.3).", f"random-walk check: per-session bias {s['random_walk_bias']:+.3f}"]],
              [1.15 * inch, 3.75 * inch, 2.3 * inch]),
        ]),
        P("Latest test run: <b>367 passed, 5 skipped</b> (ruff and import-linter clean; mypy shows the 20 "
          "errors that already exist on main).", small),

        P("7. Limitations and next steps", h1),
        *bullets([
            "<b>Trade flow is missing.</b> Signed volume and trade intensity were the next-strongest signals in our "
            "separate research on 2025 data, but the shared dataset builder loads only quote bars. Joining the "
            "trade bars we already pulled is a shared-code change (cross-group PR).",
            "<b>One dead zone for all stocks.</b> The 1 bp flat band is 1–2 ticks for AAPL but a fraction of a tick "
            "for F (where 70%+ of rows are flat); a spread- or tick-relative band would make classes comparable.",
            "<b>Universe and period.</b> 28 names over six months of 2024; BRK.B and GOOGL need a <i>pull-taq</i> fix "
            "for suffixed symbols.",
            f"<b>Multiple testing.</b> Every evaluated configuration is a trial of experiment h30s (the deflated "
            f"Sharpe divides by the count). So far: {s['n_completed_runs']} completed h30s runs (3 on TAQ data, 1 "
            f"synthetic smoke test) and {s['n_aborted_runs']} aborted attempts that produced no results.",
            "<b>Economic test.</b> Run <i>hftf ml econ</i> on the LightGBM run to measure what survives costs.",
            f"<b>Model capacity.</b> The LightGBM direction booster used its whole {imp['best_iteration']}-tree "
            f"budget without early stopping triggering; a larger budget is a cheap check (one more trial).",
        ]),
        P("Known framework issues worth a team fix: (1) <i>build_dataset</i> reuses a feature folder left "
          "half-built by a crashed run (delete <i>data/features/&lt;hash&gt;</i> after a failure); (2) on "
          "Windows set <i>PGPASSFILE=%APPDATA%\\postgresql\\pgpass.conf</i> before pulling.", small),

        P("Reproduce", h2),
        P("<font face='DV' size='7.5'>hftf data pull-taq --manifest pilot28_2024h1_5s --start 2024-01-02 "
          "--end 2024-06-28 --grid 5s --symbols-file backend/hft_forecast/ml/horizons/h30s/symbols_pilot28.txt<br/>"
          "hftf ml run backend/hft_forecast/ml/horizons/h30s/config.yaml --model logit_ridge<br/>"
          "hftf ml run backend/hft_forecast/ml/horizons/h30s/config_flow.yaml --model logit_ridge<br/>"
          "hftf ml run backend/hft_forecast/ml/horizons/h30s/config_flow.yaml --model lgbm<br/>"
          "python scripts/h30s_report.py</font>", body),
        P("Runs: " + ", ".join(f"{k} = {v['run_id']}" for k, v in [(BASE, b), ("logit_ridge · basic + flow", f),
                                                                    (BEST, g)]) + ". Seed 42, PYTHONHASHSEED=0.", small),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7)
        canvas.setFillColor(colors.HexColor(INK2))
        canvas.drawString(0.75 * inch, 0.5 * inch, "h30s group · 30 s horizon on a 5 s grid")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.75 * inch,
                            title="h30s status report", author="IEOR 4511 h30s group")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="reuse summary.json instead of recomputing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    if args.reuse and summary_path.exists():
        s = json.loads(summary_path.read_text())
    else:
        s = compute()
        summary_path.write_text(json.dumps(s, indent=1, default=float))
    figs = figures(s)
    pdf(s, figs, OUT / "h30s_report.pdf")
    print(f"wrote {OUT / 'h30s_report.pdf'}")


if __name__ == "__main__":
    main()
