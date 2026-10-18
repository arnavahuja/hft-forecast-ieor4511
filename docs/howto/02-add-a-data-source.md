# Add a data source

Template: [`templates/example_data_source.py`](../../templates/example_data_source.py)

```bash
hftf new data_source my_vendor
```

A data source implements three methods:

| Method | Returns |
|---|---|
| `describe()` | `DatasetInfo` for each dataset: frequencies, asset classes, fields, quality notes |
| `search_instruments(query)` | `Instrument`s for the instrument picker |
| `fetch(request)` | `MarketData` in the canonical long format |

The canonical format is a polars frame with `timestamp` (UTC), `instrument_id` and one
column per field (`open, high, low, close, volume`, plus anything extra). Run it through
`hft_forecast.data.normalize.normalize_frame` to cast types, sort and remove duplicates. Pass
`Instrument` metadata (multiplier, asset class) to `MarketData` so engines size positions
correctly.

`DataService` handles caching (Parquet, versioned), validation and the catalog. Your source
only fetches. Credentials come from `.env` through `self.env`, never from params. Override
`available()` to return `(False, reason)` when the source is not configured. Set
`cacheable = False` for sources that are already local.

If a source can supply a point-in-time universe, return a `universe_member` field when
`request.universe` is set; the `index_membership` universe then uses it.
