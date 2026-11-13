#!/usr/bin/env python
"""Run the whole h1m pipeline: feature study, training, economic test, impact regression.

    python run_pipeline.py

Edit the SETTINGS block below to change instruments or dates. It overrides the group's
config.yaml for this run only; the file on disk is not touched. The derived config is
written next to the results so a run can be reproduced exactly.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import yaml

# ----------------------------------------------------------------- SETTINGS

CONFIG = Path("backend/hft_forecast/ml/horizons/h1m/config.yaml")

# The universe to pull, in symbols.txt. The run itself uses whatever of it the manifest
# actually holds (see instruments()), so a pull that is still going does not break a run.
# No suffixed class (BRK-B is sym_root BRK + sym_suffix B and GOOGL is GOOG + L, which the TAQ
# query excludes) and no split inside the window: TAQ bars are unadjusted, so NVDA (10:1,
# 2024-06-10) and AVGO (10:1, 2024-07-15) would each contribute a -90% bar.
SYMBOLS = Path("symbols.txt")
START = "2024-01-02"
END = "2024-06-28"  # a trading day; a weekend end date fetches nothing
# Half of 2024 for 20 names is ~11.7 M bars. The full year does not fit: the design matrix is
# float64, so 23 M bars x ~50 features is over the 16 GB on this machine, and it is also the
# window the group's Colab notebook used, which makes the two sets of numbers comparable.
MANIFEST = "sp_2024_5s"  # must match the --manifest you pulled with; None pulls inline instead

TRAIN_SESSIONS = 40  # sessions per training fold
TEST_SESSIONS = 10  # sessions per test fold

# Every model is fitted on identical folds. `last_sign` and `zero` come along with each run,
# so one entry here is enough to get the baselines.
MODELS = [
    {"name": "logit_ridge"},
    {"name": "torch_seq", "params": {"encoder": "gru"}},
    {"name": "torch_seq", "params": {"encoder": "lstm"}},
    {"name": "torch_seq", "params": {"encoder": "tcn"}},
]

RUN_IMPACT = True  # the Cont, Kukanov and Stoikov regression
EDA_ALL_SESSIONS = False  # True studies the test folds too, which leaks

# -------------------------------------------------------------- end SETTINGS

DERIVED = Path("data/pipeline_config.yaml")
RUN_ID = re.compile(r"^run ([0-9a-f]{12})\b", re.MULTILINE)
# "  AAPL     <id>  net Sharpe ..." — the backtest the economic test creates. The ml run
# itself has no equity curve, so the Runs page does not list it; this is the one to open.
BOOK_ID = re.compile(r"^\s+\S+\s+([0-9a-f]{12})\s+net Sharpe", re.MULTILINE)


def log(message: str) -> None:
    """One timestamped line."""
    print(f"\n\033[36m[{time.strftime('%H:%M:%S')}] {message}\033[0m", flush=True)


def step(name: str, *args: str) -> str:
    """Run one hftf command, streaming its output, and return what it printed.

    Raises:
        SystemExit: If the command fails, with its exit code.
    """
    log(f"{name}  ·  hftf {' '.join(args)}")
    started = time.time()
    # stderr is merged in so the structlog lines stay in order with the output
    process = subprocess.Popen(
        [sys.executable, "-m", "hft_forecast.cli", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    elapsed = time.time() - started
    if code != 0:
        log(f"FAILED after {elapsed:.0f}s — {name}")
        raise SystemExit(code)
    log(f"done in {elapsed:.0f}s")
    return "".join(lines)


def instruments() -> list[str]:
    """The symbols that are both wanted and pulled, in symbols.txt order.

    Raises:
        SystemExit: If the manifest holds none of them.
    """
    wanted = [s.strip() for s in SYMBOLS.read_text().splitlines() if s.strip()]
    if not MANIFEST:
        return wanted
    manifest = yaml.safe_load(Path(f"data/manifests/{MANIFEST}.yaml").read_text())
    entries = manifest if isinstance(manifest, list) else manifest["entries"]
    have = {e["instrument"] for e in entries if e["dataset"] == "taq_nbbo_bars"}
    ready = [s for s in wanted if s in have]
    missing = [s for s in wanted if s not in have]
    if not ready:
        raise SystemExit(f"manifest {MANIFEST} holds none of {wanted}")
    if missing:
        log(f"not pulled yet, skipping: {', '.join(missing)}")
    return ready


def write_config(model: dict[str, object], suffix: str = "") -> Path:
    """Apply the settings and one model to the group's config, and write the derived copy.

    Raises:
        SystemExit: If a date is not a real calendar date.
    """
    for name, value in (("START", START), ("END", END)):
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise SystemExit(f"{name} = {value!r} is not a real date ({exc})") from exc
    config = yaml.safe_load(CONFIG.read_text())
    config["data"] |= {"instruments": instruments(), "start": START, "end": END}
    # without a manifest fetch_bars goes through the cache, which pulls the whole range in one
    # WRDS query per name: no resume, so keep it for short ranges and use pull-taq for the rest
    config["manifest"] = MANIFEST
    if not MANIFEST:
        del config["manifest"]
    config["cv"] |= {"train_sessions": TRAIN_SESSIONS, "test_sessions": TEST_SESSIONS}
    config["model"] = model
    config["name"] = f"h1m {label_of(model)}"
    out = DERIVED.with_name(f"{DERIVED.stem}{suffix}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def label_of(model: dict[str, object]) -> str:
    """The label a run gets in the report: torch_seq gru, or plain logit_ridge."""
    params = model.get("params") or {}
    encoder = params.get("encoder") if isinstance(params, dict) else None
    return f"{model['name']} {encoder}" if encoder else str(model["name"])


def main() -> None:
    """Study the features once, then train every model on the same folds and price each."""
    first = write_config(MODELS[0])
    names = yaml.safe_load(first.read_text())["data"]["instruments"]
    log(f"{len(names)} names: {', '.join(names)}   {START} to {END}   manifest {MANIFEST}")

    eda = ["ml", "eda", str(first)]
    if EDA_ALL_SESSIONS:
        eda.append("--all-sessions")
    step("feature study", *eda)

    books: list[tuple[str, str]] = []
    runs: list[tuple[str, str]] = []
    for number, model in enumerate(MODELS, start=1):
        label = label_of(model)
        config = write_config(model, f"_{label.replace(' ', '_')}")
        output = step(f"model {number}/{len(MODELS)}: train {label}", "ml", "run", str(config))
        match = RUN_ID.search(output)
        if not match:
            log(f"no run id in the output for {label}; skipping its economic test")
            continue
        runs.append((label, match.group(1)))
        econ = step(
            f"model {number}/{len(MODELS)}: economic test {label}", "ml", "econ", match.group(1)
        )
        books += [(label, book) for book in BOOK_ID.findall(econ)]

    if RUN_IMPACT:
        step("price impact", "ml", "impact", str(first))

    log(f"all done: {len(runs)} models, {len(books)} books")
    links = "\n".join(f"  {label:20} http://127.0.0.1:5173/runs/{book}" for label, book in books)
    files = "\n".join(f"  {label:20} data/runs/{run}/" for label, run in runs)
    print(
        f"""
  feature study   http://127.0.0.1:5173/eda
  every run       http://127.0.0.1:5173/runs   (set the Kind filter to "All kinds")

  books with an equity curve:
{links}

  predictions and metrics:
{files}

  (start the UI with `make run` if it is not already up)
""",
        flush=True,
    )


if __name__ == "__main__":
    main()
