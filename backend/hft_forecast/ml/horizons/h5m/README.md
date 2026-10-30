# h5m - 300s horizon on a 1min grid (5 bars)

Owned by the h5m group.

## Current baseline

The checked-in configs use deterministic synthetic quotes so the complete pipeline can be
run without WRDS credentials or a downloaded TAQ manifest. They are smoke/research baselines,
not evidence of predictive performance on real market data.

- `config.yaml`: `logit_ridge` with the shared `basic` features and 15/30-bar volatility
  windows.
- `config_last_sign5.yaml`: `last_sign` using `f_logret_5`, the five-bar return on the 1-minute
  grid.
- `config_taq.yaml`: the same `logit_ridge` baseline configured for the local
  `sp50_2024h1_1min` WRDS TAQ manifest and a fixed 50-name point-in-time universe.

The synthetic configs deliberately share their data, labels, features and CV settings, so their
`dataset_hash` is identical. The TAQ config is separate and requires the complete local
manifest; it is not expected to work until the download finishes. Its universe is selected
from historical S&P 500 members on 2024-01-02 and ranked by CRSP market capitalization on that
date. This fixed point-in-time sample does not include constituents added later in the period.

No deep-learning model or feature research is included in this branch foundation.

## What you own
- `config.yaml` here (horizon grid, dead zone, CV windows, model choice)
- your models in `hft_forecast/ml/models/` or `user_plugins/`
- your feature sets in `hft_forecast/ml/features/` or `user_plugins/`
- anything else in this folder (notebooks, notes)

## What you do not change without a cross-group PR
The frozen interface: the bar schema and `bar_label = end`, `hftf data pull-taq` and
manifests, `HorizonSpec`/`LabelParams`/`make_labels`, the `FeatureSet` protocol and its
causality test, `CVSpec`/`split`/`label_uniqueness`, the `Forecaster` protocol and the
predictions schema, `run_experiment`'s outputs, `ml/economic.py`, `prediction_signal`,
`quoted_spread`, and the metric and table names in `ml/evaluate.py` and `ml/report.py`.

## Running

From the repository root in the project Python environment:

```powershell
$env:PYTHONUTF8 = "1"
$env:HFTF_DATA_DIR = Join-Path $env:TEMP "h5m-data-smoke"
python -m pip install -e ".[dev]"
hftf ml build-dataset backend/hft_forecast/ml/horizons/h5m/config.yaml
hftf ml run backend/hft_forecast/ml/horizons/h5m/config.yaml
```

The synthetic dataset cache and runs go under a temporary directory, separate from the default
`data/` research store. The run writes out-of-sample predictions and reports metrics for
`logit_ridge`, `zero` and `last_sign` on identical rows. To explicitly run the five-bar
momentum config instead, pass `config_last_sign5.yaml`; that is a separate trial in experiment
`h5m`. Run `econ` only after a successful run and retain all trial records.

## Running with local WRDS TAQ data

The TAQ cache, manifest, and raw downloaded data are local research data, not Git files. The
checked-in TAQ config records only the experiment settings and selected tickers. In PowerShell,
point the project at the same external data directory used by `pull-taq`. The WRDS library may
create its password file under `%APPDATA%\postgresql\pgpass.conf` on Windows, so set
`PGPASSFILE` for each shell if that is where the file was created:

```powershell
$env:PYTHONUTF8 = "1"
$env:HFTF_DATA_DIR = "E:\4511_5min\wrds_taq"
$env:PGPASSFILE = Join-Path $env:APPDATA "postgresql\pgpass.conf"
hftf ml build-dataset backend/hft_forecast/ml/horizons/h5m/config_taq.yaml
hftf ml run backend/hft_forecast/ml/horizons/h5m/config_taq.yaml
```

The manifest must be complete before building the dataset. On another computer, first obtain
the data through its authorized WRDS account (or arrange an institution-approved data transfer);
do not push Parquet data, credentials, or machine-specific manifest paths to GitHub.

Install dependencies once per environment; skip the install command above if already installed.
The project requires Python 3.12 or newer.

```powershell
hftf ml econ <run_id>
hftf ml compare --experiment h5m --out (Join-Path $env:TEMP "h5m-report-smoke")
```

The run id is printed by `ml run`. Compare reports include completed trials registered in the
configured data directory.

Every configuration you evaluate is one trial of experiment `h5m`, and the deflated
Sharpe divides by that count. Do not delete runs to make the number look better.
