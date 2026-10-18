# Add a chart

Template: [`templates/example_chart.py`](../../templates/example_chart.py)

```bash
hftf new chart my_chart
```

A chart builder returns a library-neutral `ChartSpec`:

- `type`: one of `line, area, stacked_area, bar, histogram, scatter, heatmap, boxplot, radar`
- `x_axis`, `y_axes` (`time`, `value`, `log`, `category`; optional `format`)
- `series`: `data` rows are `[x, y]`; `role` (`series`, `benchmark`, `positive`, `negative`,
  `reference`) picks theme colors; set `run_id` so a run keeps its color on the Compare page
- `annotations`: shaded bands and reference lines
- `time_series=True` joins synchronized zoom and date-range brushing

Set `group` (the Run Detail tab it appears on) and `scopes` (`single` for Run Detail,
`compare` for the Compare page). Override `applicable(runs)` to hide the chart when it has
nothing to show.

**A chart using an existing type needs backend code only.** A new chart *type* also needs
one renderer in `frontend/src/charts/registry.ts` (map the type to an ECharts option).
