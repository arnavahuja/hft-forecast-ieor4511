# h1m — 60s horizon on a 5s grid (12 bars)

Owned by the h1m group.

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
```bash
hftf ml build-dataset backend/hft_forecast/ml/horizons/h1m/config.yaml
hftf ml eda           backend/hft_forecast/ml/horizons/h1m/config.yaml   # feature research
hftf ml run           backend/hft_forecast/ml/horizons/h1m/config.yaml
hftf ml impact        backend/hft_forecast/ml/horizons/h1m/config.yaml   # Cont et al. (2014)
```

## Why 5s bars
The grid is the finest the WRDS TAQ pull produces, and the order flow imbalance of
Cont, Kukanov and Stoikov is only visible between quote updates: a coarser bar nets
out the updates inside it and reads as a lower bound on the fit. The horizon is 60s,
so `steps = 12`.
