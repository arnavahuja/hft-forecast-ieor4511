# Feature research (the EDA module)

Before fitting anything, answer: does this feature lead the label, and does it do so
reliably? `hftf ml eda` answers it from the same config the model will use, so the features
studied are exactly the features trained on.

```bash
hftf ml eda backend/hft_forecast/ml/horizons/h1m/config.yaml
```

The study is registered as a run with `kind='eda'` and appears under **Features** in the UI,
where two studies can be read side by side.

## What it computes

| Section | Question | Table |
|---|---|---|
| Coverage | How much usable data does each name contribute? | `coverage` |
| IC ranking | Which features correlate with the forward return, and how consistently? | `ranking` |
| Monotonicity | Is the relationship a steady tilt or one extreme bucket? | `buckets` |
| Signal decay | At which horizon is the feature strongest? | `decay` |
| Redundancy | Which features are the same feature twice? | `correlation` |
| Missingness | What does each feature cost in rows? | `missing` |

Everything lands in the run directory: `eda.json` (what the API serves) and `eda/*.csv`
(one per table, every feature, for a write-up). `eda/blocks.csv` has the per-stock-day
correlations the ranking aggregates.

## Three rules it follows

**Training sessions only.** The default `train_sessions` is the config's `cv.train_sessions`.
Ranking features on sessions that later serve as test folds leaks, and the model's own
cross-validation cannot detect that leak — the features were already chosen. `--all-sessions`
turns it off and the UI badges the study as including test folds.

**Per stock-day, then aggregated.** Every statistic is computed inside one instrument-session
block and then averaged over blocks, so one volatile name or one wild day cannot carry a
ranking. The `blocks` count sits next to every number.

**Nothing is imputed.** Blocks with fewer than `--min-block-rows` usable rows are skipped,
not filled. A feature that never moves inside a block has no correlation there (polars
reports NaN), so it is absent from the ranking rather than sitting in it as a blank row —
`features_ranked` versus `features` in the header is that gap.

## Reading the numbers

- **Mean IC** is a Spearman rank correlation with the forward return. In microstructure,
  0.02 is a real signal and 0.2 means something leaked.
- **t-stat** is over blocks, not rows: "is the edge consistent across stock-days", not "is it
  significant", a question that answers yes to almost anything at a million rows. Blocks
  co-move, so it ranks rather than tests.
- **Buckets** are ranked *within* a stock-day and exist only for this diagnostic. A whole-day
  rank is not knowable at prediction time and must never become a feature.
- **Decay** re-measures the same features against forward returns at `0.25x, 0.5x, 1x, 2x,
  4x` the configured horizon. A feature that is strong at one bar and gone at twenty is not
  weak — it belongs to a shorter-horizon group.

## Adding a feature set to a study

There is nothing to add: the study reads whatever `features:` the config lists, so a new
feature set shows up in the ranking as soon as it is in the config. See
[Add a chart](05-add-a-chart.md) for the plugin pattern and
[`ml/features/quote_dynamics.py`](../../backend/hft_forecast/ml/features/quote_dynamics.py)
for a feature set written against level-1 quotes.

## What it is not

It does not fit, score or select anything. A ranking is a reason to *try* a feature, and the
only honest test is the out-of-sample one `hftf ml run` performs on folds the study never
read. Running a study, picking the top five features and reporting the model's CV score as
though the features were chosen blind is the exact mistake the training-session default
exists to make harder.
