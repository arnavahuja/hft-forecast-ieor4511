# ADR 0011: Feature research is a run, not a notebook

Status: accepted

Extends ADR 0008 (frozen interface) and ADR 0010 (prune to the forecast comparison).

## Context
The first real feature work on this project was a Colab notebook: WRDS TAQ level-1 quotes on
a 5 s grid, 27 candidate features, an information-coefficient ranking, quantile profiles, a
regime split and a redundancy check against a 60 s forward return. It was good work and it
answered the right question — *which features lead the label?* — but it lived outside the
repository, which has three consequences this project cannot afford:

1. **It is not comparable.** Three groups forecast at three horizons. A ranking computed in
   one group's notebook, on that group's grid, with that group's own IC code, cannot be read
   against another group's.
2. **It drifts.** The features in a notebook are a copy of the features in the repo until
   someone edits one of them.
3. **It leaks quietly.** A notebook that ranks features over the whole sample and then hands
   the winners to a model has already chosen on the test folds. The model's own
   cross-validation will not show it, because by then the choice is upstream.

A fourth consideration is practical: ADR 0010 removed the Strategy Lab and the research
service, so there is no longer anywhere in the UI to *look* at anything but a finished
backtest. Feature work that nobody can see is feature work nobody checks.

## Decision
Feature research is a module in the repository (`ml/eda.py`, `ml/eda_charts.py`,
`services/eda.py`), run by `hftf ml eda <config.yaml>`, and stored as a run with
`kind='eda'`.

- **It reads a built dataset.** The study sees whatever `features:`, `horizon:` and `labels:`
  the config declares, so it describes the features that will actually be trained on, at the
  group's own grid, with no second implementation to drift.
- **Every statistic is per instrument-session, then aggregated.** The block count travels
  with every number. Pooling across the panel would let one volatile name carry a ranking.
- **Training sessions only by default** (`cv.train_sessions`). `--all-sessions` exists, and
  the UI badges a study that used it.
- **Nothing is imputed, and nothing undefined is faked.** Small blocks are skipped; a feature
  that never moves is absent from the ranking rather than present and blank.
- **It is read-only in the UI.** The Features page renders stored `ChartSpec`s and tables and
  can show two studies side by side. The API cannot launch a study, consistent with ADR 0008:
  the front end views, the CLI produces.
- **A study is not a trial.** `count_trials` stays `kind IN ('backtest', 'ml')`, so looking
  at a feature does not deflate anyone's Sharpe ratio. Fitting a model on it does.

The 60 s horizon the notebook used is added as a fourth horizon, `h1m` (60 s on a 5 s grid),
with its own package under `ml/horizons/`. The notebook's genuinely new features — spread
change, size log ratio, notional, imbalance tilt, size changes and the two queue
replenishment proxies — become one registered feature set, `quote_dynamics`. The other
twenty of its twenty-seven were already `basic` and `ofi` under different names, and are not
duplicated.

## Consequences
- A group's feature study is citable by run id and comparable across horizons.
- Adding a feature set makes it appear in the study with no further code.
- The study is one more thing to run before a model, and it is slower than a notebook cell
  because it builds the full dataset first. The dataset is cached by hash, so the cost is
  paid once and `hftf ml run` reuses it.
- `ml/eda.py` deliberately computes forward returns from future bars. That is measurement of
  a label, not a feature, and nothing in the module feeds a model — but it means the
  lookahead check in `ml/features/registry.py` does not apply to it, and the module must stay
  out of the feature path.
