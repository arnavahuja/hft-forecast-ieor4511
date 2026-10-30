"""Shared report sections. Every results report includes the look-ahead safeguards section."""

import subprocess
import sys

from hft.config import ROOT

# (stage, safeguard, how it is verified). Add a row whenever a pipeline stage is added (Steps 8-10).
LOOKAHEAD_SAFEGUARDS = [
    ("Universe", "S&P 500 membership taken as of each date from CRSP (no end-of-sample constituent list, "
                 "so no survivorship bias); TAQ symbols linked by that day's CUSIP.",
     "`build_universe` per date; members and links logged per day."),
    ("Cleaning", "Filters use only each row's own fields (prices, sizes, condition codes); no filter "
                 "looks at later data.", "`tests/test_clean.py`"),
    ("Trade signing", "Lee–Ready uses the last NBBO strictly *before* the trade (as-of join on trade time − 1 ns), "
                      "so a quote at the same nanosecond, possibly caused by the trade, is never used.",
     "`tests/test_signing.py` (same-timestamp quote case)"),
    ("Features", "Every feature at grid time t uses only events with timestamp ≤ t (backward as-of joins); "
                 "windows are exact (t − w, t]; windows near the open are truncated at 09:30 rather than "
                 "filled with later data.",
     "Shift test (perturbing all events after t leaves features at t unchanged) and a brute-force "
     "recomputation of window features (`tests/test_features_labels.py`)."),
    ("Labels", "Target = mid-to-mid return from the last mid ≤ t to the last mid ≤ t + h; never trade prices; "
               "labels that would cross 16:00 are dropped.", "Label alignment test on a known mid path."),
    ("Day boundaries", "Features and labels use only the same day's 09:30–16:00 events.",
     "Day-boundary test (adding the neighbouring days' and after-close events changes nothing)."),
    ("Harness check", "A deliberately leaked feature set (features from t + 30 s) must look absurdly good.",
     "Real-data placebo (`tests/test_placebo_real.py`); leaked R² ≈ 100% vs small R² for correct features."),
    ("Normalization, splits, tuning (Steps 8–10)", "Planned: per-stock scaling and winsorizing fitted on the "
     "training window only; walk-forward by whole days (no train/test day overlap); hyper-parameters tuned on "
     "a validation day inside the training window; volume weights known as of the prior day.",
     "To be added with Step 8 (assertions on split dates and scaler fit ranges)."),
]


def run_tests() -> str:
    """Run the test suite and return pytest's one-line summary."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    return lines[-1] if lines else "pytest produced no output"


def lookahead_section(test_summary: str | None = None, extra: list[str] | None = None) -> list[str]:
    lines = ["## No look-ahead bias: safeguards", "",
             "Every stage of the pipeline is built so that a prediction at time t uses only information "
             "available at t, and each safeguard is verified by an automated test or check:", "",
             "| Stage | Safeguard | Verified by |", "|---|---|---|"]
    lines += [f"| {s} | {g} | {v} |" for s, g, v in LOOKAHEAD_SAFEGUARDS]
    lines += [""]
    if test_summary:
        lines += [f"Latest test run: **{test_summary}**", ""]
    return lines + (extra or [])
