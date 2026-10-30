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

Both configs deliberately share their data, labels, features and CV settings, so their
`dataset_hash` is identical. Their configured `min_train_rows` is suitable for the synthetic
baseline; revisit it when switching to the actual TAQ universe. Do not uncomment or invent a
manifest until `sp50_2024h1_1min` is available locally and its instrument list is confirmed.

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

The dataset cache and runs go under a temporary directory, separate from the default `data/`
research store. The run writes out-of-sample predictions and reports metrics for
`logit_ridge`, `zero` and `last_sign` on identical rows. To explicitly run the five-bar
momentum config instead, pass `config_last_sign5.yaml`; that is a separate trial in experiment
`h5m`. Run `econ` only after a successful run and retain all trial records.

When the shared TAQ manifest and its symbols are available, prepare a separate real-data config
or update these configs with the verified instrument list and manifest. Keep an offline config
available for repeatable development.

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
