import clsx from 'clsx'
import { Fragment, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  useCombine,
  useCompareChart,
  useCompareCharts,
  useCompareMetrics,
  useRuns,
  type ChartInfo,
} from '@/api/hooks'
import { ChartPanel } from '@/components/ChartPanel'
import { ReportExport } from '@/components/ReportExport'
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Input,
  Label,
  PageHeader,
  Select,
  Skeleton,
  Stat,
  Toggle,
} from '@/components/ui'
import { formatNumber, type NumberFormat } from '@/lib/format'
import { runColorMap } from '@/lib/runColors'
import { useCompareStore } from '@/store/compare'

function CompareChart({
  chart,
  runIds,
  align,
  colors,
  params,
}: {
  chart: ChartInfo
  runIds: string[]
  align: boolean
  colors: Record<string, string>
  params?: Record<string, unknown>
}) {
  const query = useCompareChart(chart.name, runIds, align, params)
  return (
    <ChartPanel title={chart.title} query={query} colors={colors} group="compare" height={320} />
  )
}

function CombinedPortfolio({
  runIds,
  labels,
}: {
  runIds: string[]
  labels: Record<string, string>
}) {
  const combine = useCombine()
  const [weights, setWeights] = useState<Record<string, string>>({})
  const w = runIds.map((id) => Number(weights[id] ?? 1 / runIds.length))
  const metrics = combine.data?.metrics ?? {}
  const tiles: { key: string; label: string; format: NumberFormat }[] = [
    { key: 'cagr', label: 'CAGR', format: 'percent' },
    { key: 'ann_vol', label: 'Volatility', format: 'percent' },
    { key: 'sharpe', label: 'Sharpe', format: 'ratio' },
    { key: 'max_drawdown', label: 'Max DD', format: 'percent' },
  ]
  return (
    <Card title="Combined portfolio builder">
      <div className="grid gap-3 md:grid-cols-4">
        {runIds.map((id) => (
          <div key={id}>
            <Label htmlFor={`w-${id}`}>{labels[id] ?? id}</Label>
            <Input
              id={`w-${id}`}
              type="number"
              step="0.05"
              value={weights[id] ?? (1 / runIds.length).toFixed(3)}
              onChange={(e) => setWeights({ ...weights, [id]: e.target.value })}
            />
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center justify-between">
        <span className="text-2xs text-muted">
          Weights are rebalanced every bar over the common period. Sum:{' '}
          {formatNumber(
            w.reduce((a, b) => a + b, 0),
            'ratio',
          )}
        </span>
        <Button
          variant="primary"
          loading={combine.isPending}
          onClick={() => combine.mutate({ run_ids: runIds, weights: w })}
        >
          Combine
        </Button>
      </div>
      {combine.isError && <ErrorState error={combine.error} />}
      {combine.data && (
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-6">
          {tiles.map((t) => (
            <Stat
              key={t.key}
              label={t.label}
              value={formatNumber(metrics[t.key]?.value, t.format)}
            />
          ))}
          <Stat
            label="Diversification ratio"
            value={formatNumber(combine.data.diversification_ratio, 'ratio')}
          />
          <Stat label="Vol reduction" value={formatNumber(combine.data.vol_reduction, 'percent')} />
        </div>
      )}
    </Card>
  )
}

export function ComparePage() {
  const { selected: stored, toggle, setSelected, clear } = useCompareStore()
  const [align, setAlign] = useState(false)
  const [metricKey, setMetricKey] = useState('sharpe')
  const allRuns = useRuns({ status: 'completed', limit: 1000 })
  const charts = useCompareCharts()
  // the selection is persisted in the browser, so it outlives the runs it names: drop the
  // ones the store no longer has instead of asking the API for a deleted run
  const selected = useMemo(() => {
    if (!allRuns.data) return stored
    const known = new Set(allRuns.data.map((r) => r.id))
    return stored.filter((id) => known.has(id))
  }, [stored, allRuns.data])
  useEffect(() => {
    if (selected.length !== stored.length) setSelected(selected)
  }, [selected, stored, setSelected])
  const table = useCompareMetrics(selected, align)
  const colors = useMemo(() => runColorMap(selected), [selected])
  const labels = useMemo(
    () =>
      Object.fromEntries(
        (allRuns.data ?? []).map((r) => [r.id, r.name || `${r.strategy} ${r.id.slice(0, 6)}`]),
      ),
    [allRuns.data],
  )
  const [adding, setAdding] = useState('')

  return (
    <div className="space-y-4">
      <PageHeader
        title="Compare"
        subtitle="Pick 2 or more runs. Each run keeps the same color in every chart."
        actions={
          <>
            <Toggle
              id="align"
              checked={align}
              onChange={setAlign}
              label="Align dates to the common period"
            />
            <Button variant="ghost" onClick={clear}>
              Clear
            </Button>
          </>
        }
      />
      <Card title="Runs">
        <div className="flex flex-wrap items-center gap-2">
          {selected.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-2 rounded-full border border-border bg-elevated px-3 py-1 text-sm"
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colors[id] }}
                aria-hidden
              />
              <Link to={`/runs/${id}`} className="hover:text-accent">
                {labels[id] ?? id}
              </Link>
              <button
                type="button"
                aria-label={`Remove ${labels[id] ?? id}`}
                className="text-muted hover:text-primary"
                onClick={() => toggle(id)}
              >
                ✕
              </button>
            </span>
          ))}
          <Select
            aria-label="Add run"
            className="w-72"
            value={adding}
            onChange={(e) => setAdding(e.target.value)}
          >
            <option value="">Add a completed run…</option>
            {allRuns.data
              ?.filter((r) => !selected.includes(r.id))
              .map((r) => (
                <option key={r.id} value={r.id}>
                  {labels[r.id]} · {r.id}
                </option>
              ))}
          </Select>
          <Button
            disabled={!adding}
            onClick={() => {
              setSelected([...selected, adding])
              setAdding('')
            }}
          >
            Add
          </Button>
        </div>
      </Card>

      {selected.length < 2 ? (
        <EmptyState title="Select at least two runs to compare">
          Use the Runs page (select rows, then Compare) or add runs above.
        </EmptyState>
      ) : (
        <>
          {table.data?.warnings.map((w) => (
            <div
              key={w}
              className="rounded-md border border-warning/40 bg-warning/5 px-3 py-2 text-sm text-warning"
            >
              {w}
            </div>
          ))}
          <Card title="Metrics (best in green, worst in red per row)" padded={false}>
            {table.isPending ? (
              <div className="p-4">
                <Skeleton height={300} />
              </div>
            ) : table.isError ? (
              <div className="p-4">
                <ErrorState error={table.error} />
              </div>
            ) : (
              <div className="max-h-[560px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface">
                    <tr className="border-b border-border text-2xs uppercase tracking-wide text-muted">
                      <th className="px-3 py-2 text-left font-medium">Metric</th>
                      {table.data.runs.map((r) => (
                        <th key={r.id} className="px-3 py-2 text-right font-medium">
                          <span className="inline-flex items-center gap-1.5">
                            <span
                              className="h-2 w-2 rounded-full"
                              style={{ backgroundColor: colors[r.id] }}
                              aria-hidden
                            />
                            {r.name || r.strategy}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.data.rows.map((row, i) => {
                      const newGroup = i === 0 || table.data.rows[i - 1]?.group !== row.group
                      return (
                        <Fragment key={row.key}>
                          {newGroup && (
                            <tr key={`${row.group}-h`} className="bg-canvas">
                              <td
                                colSpan={table.data.runs.length + 1}
                                className="px-3 py-1.5 text-2xs font-medium uppercase tracking-wide text-muted"
                              >
                                {row.group}
                              </td>
                            </tr>
                          )}
                          <tr key={row.key} className="border-b border-border/60">
                            <td className="px-3 py-1.5 text-secondary">{row.label}</td>
                            {table.data.runs.map((r) => (
                              <td
                                key={r.id}
                                className={clsx(
                                  'px-3 py-1.5 text-right font-mono tabular-nums',
                                  row.best === r.id && 'bg-positive/10 text-positive',
                                  row.worst === r.id && 'bg-negative/10 text-negative',
                                )}
                              >
                                {formatNumber(row.values[r.id], row.format as NumberFormat)}
                              </td>
                            ))}
                          </tr>
                        </Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <div className="flex items-center gap-2">
            <Label htmlFor="bar-metric">Metric for bar chart</Label>
            <Select
              id="bar-metric"
              className="w-56"
              value={metricKey}
              onChange={(e) => setMetricKey(e.target.value)}
            >
              {table.data?.rows.map((r) => (
                <option key={r.key} value={r.key}>
                  {r.label}
                </option>
              ))}
            </Select>
            <Badge>{selected.length} runs</Badge>
          </div>
          {charts.isError && <ErrorState error={charts.error} />}
          <div className="grid gap-4 2xl:grid-cols-2">
            {charts.data?.map((c) => (
              <CompareChart
                key={c.name}
                chart={c}
                runIds={selected}
                align={align}
                colors={colors}
                params={c.name === 'compare_metric_bars' ? { metric: metricKey } : undefined}
              />
            ))}
          </div>
          <CombinedPortfolio runIds={selected} labels={labels} />
          <ReportExport runIds={selected} />
        </>
      )}
    </div>
  )
}
