# h30s — 30s horizon on a 5s grid (6 bars)

Owned by the h30s group.

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
hftf ml build-dataset backend/hft_forecast/ml/horizons/h30s/config.yaml
hftf ml run backend/hft_forecast/ml/horizons/h30s/config.yaml --model logit_ridge
hftf ml econ <run_id>
hftf ml compare --experiment h30s --out report/
```

Every configuration you evaluate is one trial of experiment `h30s`, and the deflated
Sharpe divides by that count. Do not delete runs to make the number look better.
