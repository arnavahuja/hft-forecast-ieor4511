# Add a metric

Template: [`templates/example_metric.py`](../../templates/example_metric.py)

```bash
hftf new metric my_metrics
```

A metric plugin is a group of related outputs. Each output has a `MetricDescriptor`:

| Field | Meaning |
|---|---|
| `key` | Unique across all metrics |
| `label`, `group`, `description` | Display |
| `format` | `percent`, `ratio`, `number`, `integer`, `currency`, `bars`, `days` |
| `higher_is_better` | Drives best/worst highlighting on the Compare page (`None` = neutral) |
| `kind` | `scalar` or `table` (a table is a list of row dicts) |
| `benchmark` | Also compute it on the benchmark |

`compute(ctx)` returns `{key: value}`. `ctx.returns` holds the returns being measured (the
benchmark's on the benchmark pass). `ctx.result` holds the full `BacktestResult`, and
`ctx.periods_per_year` is the calendar-derived annualization (never hard-code 252).
Return `None` when a value is undefined.

New metrics show up in Run Detail, Compare and the tearsheet automatically. Date-range
brushing recomputes them for the selected window.
