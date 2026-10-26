# Bring your own bars (level 1 or level 2)

Three ways data enters the repo. All three end in the same canonical bar schema, so the
labels, features, CV and economic test do not care which one you used.

| Where it comes from | How |
|---|---|
| WRDS TAQ NBBO (level 1) | `hftf data pull-taq --manifest <name> --grid 1min --start … --end …` |
| A file from a venue or a vendor (level 1 or level 2) | Data page → *Import a local file*, or `hftf data import <file>` |
| Nothing yet (offline work) | source `synthetic`, dataset `quotes` |

## The canonical bar

One row per `(timestamp, instrument_id)`. Timestamps are UTC and **labelled at the bar end**:
the row at `t` summarizes `(t − Δ, t]`. The fields the pipeline uses:

```
close mid bid ask bid_size ask_size microprice spread spread_bps n_quotes is_stale
bid_px_1 bid_sz_1 ask_px_1 ask_sz_1 bid_px_2 …        # level 2 only
```

## Importing a file

Map the columns your file actually has; the importer derives the rest and never overwrites
something you mapped:

| You map | You get for free |
|---|---|
| `bid_px_1`, `bid_sz_1`, `ask_px_1`, `ask_sz_1` (book levels) | `bid`, `ask`, `bid_size`, `ask_size` |
| `bid`, `ask` | `mid`, `spread`, `spread_bps` |
| …and the sizes | `microprice` |
| `mid` | `close` (and the dataset is flagged `price_basis = mid`) |

Level 1 of the book **is** the best quote, so a file that carries both must agree on them;
the import refuses the file if they do not. Set **Book levels** to how deep you want to go
(0 for a level-1 file). Set **Timestamps mark the bar** to `start` if the file labels bars at
their opening — the import shifts them by one bar so everything on disk is end-labelled.

`n_quotes` and `is_stale` are not derivable from a snapshot. The shipped `basic` feature set
requires them; if your file has neither, map what you have and write a feature set that uses
it (that is the group's job anyway).

## Pointing an experiment at it

The import shows up in the catalog with an id. Use it as the source and dataset:

```yaml
horizon: {name: h5m, seconds: 300, grid: 1min}
data:
  source: local_file
  dataset: my-book-a1b2c3     # the catalog id, or a manifest name below
  instruments: [AAPL, MSFT]
  start: 2024-01-02
  end: 2024-06-28
  frequency: 1min             # must equal the horizon grid
features:
  - name: basic
```

For pulled TAQ bars use `manifest: sp50_2024h1_1min` instead of `data.source`.

## Choosing the grid

`HorizonSpec` takes any horizon that is a whole number of bars: 30 s on a 5 s grid, 5 min on
a 1 min grid, 30 min on a 1 min grid, and so on. Two things are counted in **bars**, not
seconds, and need attention when you change grid:

- `basic`'s `vol_windows` (default `[20, 100]`). On a 5 min grid a session is only 78 bars,
  so a 100-bar window is null everywhere; shrink it (`params: {vol_windows: [5, 20]}`).
- `cv.train_sessions` / `test_sessions` are in sessions, so they are grid-independent.

If a dataset turns out empty, the error lists the share of rows each feature is missing — the
column at 100 % is the one that emptied it.
