import type { ColumnDef } from '@tanstack/react-table'
import { useCallback, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  useCancelRun,
  useCloneRun,
  useRun,
  useRunChart,
  useRunCharts,
  useRunLogs,
  useRunMetrics,
  useRunOverlays,
  useRunPositions,
  useRunTrades,
  useUpdateRun,
  type BacktestConfig,
  type ChartInfo,
  type Window,
} from '@/api/hooks'
import { ChartPanel } from '@/components/ChartPanel'
import type { ZoomWindow } from '@/components/ChartRenderer'
import { DataTable } from '@/components/DataTable'
import { MetricRowsTable, MetricTable } from '@/components/MetricTable'
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  Input,
  PageHeader,
  Skeleton,
  Stat,
  StatusBadge,
  Tabs,
} from '@/components/ui'
import {
  formatDate,
  formatDateTime,
  formatDuration,
  formatNumber,
  signClass,
  type NumberFormat,
} from '@/lib/format'
import { useCompareStore } from '@/store/compare'

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'performance', label: 'Performance' },
  { id: 'risk', label: 'Risk' },
  { id: 'positions', label: 'Positions' },
  { id: 'trades', label: 'Trades' },
  { id: 'costs', label: 'Costs' },
  { id: 'factors', label: 'Factors' },
  { id: 'overlays', label: 'Overlays' },
  { id: 'config', label: 'Config' },
  { id: 'logs', label: 'Logs' },
] as const
type TabId = (typeof TABS)[number]['id']

/** Which chart groups and metric groups appear on each tab (unknown groups go to Performance). */
const TAB_CHART_GROUPS: Partial<Record<TabId, string[]>> = {
  performance: ['Performance', 'Distribution'],
  risk: ['Drawdown', 'Rolling'],
  positions: ['Positions'],
  trades: ['Trades'],
  costs: ['Costs'],
  factors: ['Factor'],
  overlays: ['Overlays'],
}
const TAB_METRIC_GROUPS: Partial<Record<TabId, string[]>> = {
  overview: ['Return', 'Risk-adjusted', 'Drawdown', 'Benchmark-relative'],
  performance: ['Return', 'Statistical'],
  risk: ['Risk', 'Drawdown', 'Benchmark-relative'],
  positions: ['Portfolio'],
  trades: ['Trades'],
  costs: ['Costs'],
  factors: ['Factor'],
}
const HEADLINE: { key: string; label: string; format: NumberFormat; signed?: boolean }[] = [
  { key: 'total_return', label: 'Total return', format: 'percent', signed: true },
  { key: 'cagr', label: 'CAGR', format: 'percent', signed: true },
  { key: 'sharpe', label: 'Sharpe', format: 'ratio' },
  { key: 'ann_vol', label: 'Volatility', format: 'percent' },
  { key: 'max_drawdown', label: 'Max drawdown', format: 'percent', signed: true },
  { key: 'n_trades', label: 'Trades', format: 'integer' },
]
const OVERVIEW_CHARTS = ['equity_curve', 'drawdown_underwater']

function knownGroups(): Set<string> {
  return new Set(Object.values(TAB_CHART_GROUPS).flat())
}

function chartsForTab(tab: TabId, charts: ChartInfo[]): ChartInfo[] {
  if (tab === 'overview') {
    return OVERVIEW_CHARTS.flatMap((name) => charts.filter((c) => c.name === name))
  }
  const groups = TAB_CHART_GROUPS[tab]
  if (!groups) return []
  const known = knownGroups()
  return charts.filter(
    (c) =>
      c.applicable && (groups.includes(c.group) || (tab === 'performance' && !known.has(c.group))),
  )
}

function RunChart({
  runId,
  chart,
  window,
  onZoom,
  group,
}: {
  runId: string
  chart: ChartInfo
  window: Window
  onZoom: (w: ZoomWindow | null) => void
  group: string
}) {
  const query = useRunChart(runId, chart.name, window)
  return <ChartPanel title={chart.title} query={query} group={group} onZoom={onZoom} />
}

function TradesTable({ runId }: { runId: string }) {
  const [offset, setOffset] = useState(0)
  const limit = 200
  const trades = useRunTrades(runId, offset, limit)
  const columns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () => [
      { accessorKey: 'instrument_id', header: 'Instrument' },
      { accessorKey: 'direction', header: 'Side' },
      {
        accessorKey: 'entry_time',
        header: 'Entry',
        cell: (c) => formatDate(c.getValue() as string),
      },
      { accessorKey: 'exit_time', header: 'Exit', cell: (c) => formatDate(c.getValue() as string) },
      {
        accessorKey: 'entry_price',
        header: 'Entry px',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
      },
      {
        accessorKey: 'exit_price',
        header: 'Exit px',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
      },
      { accessorKey: 'bars_held', header: 'Bars', meta: { numeric: true } },
      {
        accessorKey: 'pnl',
        header: 'PnL',
        meta: { numeric: true },
        cell: (c) => (
          <span className={signClass(c.getValue() as number)}>
            {formatNumber(c.getValue() as number, 'currency')}
          </span>
        ),
      },
      {
        accessorKey: 'return',
        header: 'Return',
        meta: { numeric: true },
        cell: (c) => (
          <span className={signClass(c.getValue() as number)}>
            {formatNumber(c.getValue() as number, 'percent')}
          </span>
        ),
      },
      {
        accessorKey: 'mae',
        header: 'MAE',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'percent'),
      },
      {
        accessorKey: 'mfe',
        header: 'MFE',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'percent'),
      },
      {
        accessorKey: 'is_open',
        header: 'Open',
        cell: (c) => (c.getValue() ? <Badge tone="info">open</Badge> : ''),
      },
    ],
    [],
  )
  if (trades.isPending) return <Skeleton height={200} />
  if (trades.isError) return <ErrorState error={trades.error} />
  const total = trades.data.total
  return (
    <Card
      title={`Trades (${total})`}
      padded={false}
      actions={
        total > limit && (
          <>
            <Button
              size="sm"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              Prev
            </Button>
            <span className="text-2xs text-muted">
              {offset + 1}–{Math.min(offset + limit, total)}
            </span>
            <Button
              size="sm"
              disabled={offset + limit >= total}
              onClick={() => setOffset(offset + limit)}
            >
              Next
            </Button>
          </>
        )
      }
    >
      <DataTable data={trades.data.rows} columns={columns} dense empty="No trades" />
    </Card>
  )
}

function PositionsTable({ runId }: { runId: string }) {
  const positions = useRunPositions(runId)
  const columns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () => [
      { accessorKey: 'instrument_id', header: 'Instrument' },
      {
        accessorKey: 'weight',
        header: 'Weight',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'percent'),
      },
      {
        accessorKey: 'units',
        header: 'Units',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
      },
      {
        accessorKey: 'price',
        header: 'Price',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
      },
      {
        accessorKey: 'value',
        header: 'Value',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'currency'),
      },
    ],
    [],
  )
  if (positions.isPending) return <Skeleton height={120} />
  if (positions.isError) return <ErrorState error={positions.error} />
  return (
    <Card
      title={`Positions at ${formatDate(positions.data.timestamp)} · equity ${formatNumber(positions.data.equity, 'currency')}`}
      padded={false}
    >
      <DataTable data={positions.data.rows} columns={columns} dense empty="Flat (no positions)" />
    </Card>
  )
}

function OverlayStats({ runId }: { runId: string }) {
  const overlays = useRunOverlays(runId)
  if (overlays.isPending) return <Skeleton height={120} />
  if (overlays.isError) return <ErrorState error={overlays.error} />
  if (!overlays.data.length) return <EmptyState title="No overlays in this run" />
  const metrics: { key: string; label: string; format: NumberFormat }[] = [
    { key: 'cagr', label: 'CAGR', format: 'percent' },
    { key: 'sharpe', label: 'Sharpe', format: 'ratio' },
    { key: 'max_drawdown', label: 'Max DD', format: 'percent' },
    { key: 'avg_gross', label: 'Avg gross', format: 'percent' },
  ]
  return (
    <Card title="Before vs after each overlay (zero-cost simulation)" padded={false}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-2xs uppercase tracking-wide text-muted">
            <th className="px-3 py-2 text-left font-medium">Overlay</th>
            {metrics.map((m) => (
              <th key={m.key} className="px-3 py-2 text-right font-medium">
                {m.label} before → after
              </th>
            ))}
            <th className="px-3 py-2 text-right font-medium">Δ CAGR</th>
            <th className="px-3 py-2 text-right font-medium">Cells changed</th>
          </tr>
        </thead>
        <tbody>
          {overlays.data.map((o) => (
            <tr key={o.position} className="border-b border-border/60 last:border-0">
              <td className="px-3 py-1.5">
                {o.position + 1}. {o.name}
                {o.notes.length > 0 && (
                  <div className="text-2xs text-muted">{o.notes.join(' · ')}</div>
                )}
              </td>
              {metrics.map((m) => (
                <td key={m.key} className="px-3 py-1.5 text-right font-mono tabular-nums">
                  {formatNumber(o.before[m.key], m.format)} →{' '}
                  {formatNumber(o.after[m.key], m.format)}
                </td>
              ))}
              <td
                className={`px-3 py-1.5 text-right font-mono tabular-nums ${signClass(o.delta_cagr)}`}
              >
                {formatNumber(o.delta_cagr, 'percent')}
              </td>
              <td className="px-3 py-1.5 text-right font-mono tabular-nums">
                {formatNumber(o.cells_changed, 'integer')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

function Labels({
  runId,
  name,
  tags,
  notes,
}: {
  runId: string
  name: string
  tags: string[]
  notes: string
}) {
  const update = useUpdateRun(runId)
  const [draft, setDraft] = useState({ name, tags: tags.join(', '), notes })
  return (
    <Card title="Labels">
      <div className="grid gap-2 md:grid-cols-3">
        <Input
          aria-label="Run name"
          value={draft.name}
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
          placeholder="Name"
        />
        <Input
          aria-label="Tags"
          value={draft.tags}
          onChange={(e) => setDraft({ ...draft, tags: e.target.value })}
          placeholder="tags"
        />
        <Input
          aria-label="Notes"
          value={draft.notes}
          onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
          placeholder="notes"
        />
      </div>
      <div className="mt-2 flex justify-end">
        <Button
          size="sm"
          loading={update.isPending}
          onClick={() =>
            update.mutate({
              name: draft.name,
              notes: draft.notes,
              tags: draft.tags
                .split(',')
                .map((t) => t.trim())
                .filter(Boolean),
            })
          }
        >
          Save labels
        </Button>
      </div>
    </Card>
  )
}

export function RunDetailPage() {
  const { runId = '' } = useParams()
  const navigate = useNavigate()
  const run = useRun(runId)
  const completed = run.data?.status === 'completed'
  const [tab, setTab] = useState<TabId>('overview')
  const [window, setWindow] = useState<Window>({})
  const metrics = useRunMetrics(runId, window, completed)
  const charts = useRunCharts(runId, completed)
  const logs = useRunLogs(runId, !completed && run.data?.status !== 'failed')
  const cancel = useCancelRun()
  const clone = useCloneRun()
  const toggleCompare = useCompareStore((s) => s.toggle)
  const inCompare = useCompareStore((s) => s.selected.includes(runId))
  const onZoom = useCallback(
    (w: ZoomWindow | null) => setWindow(w ? { start: w.start, end: w.end } : {}),
    [],
  )

  if (run.isPending) return <Skeleton height={400} />
  if (run.isError) return <ErrorState error={run.error} onRetry={() => void run.refetch()} />
  const r = run.data
  const config = r.config as unknown as BacktestConfig
  const values = metrics.data?.values ?? {}
  const tabCharts = chartsForTab(tab, charts.data ?? [])
  const group = `run-${runId}`

  return (
    <div>
      <PageHeader
        title={
          <span className="flex items-center gap-2">
            {r.name || `${r.strategy} · ${r.id}`} <StatusBadge status={r.status} />
          </span>
        }
        subtitle={`${r.strategy} · ${config.data.source}:${config.data.dataset} · ${config.data.start} → ${config.data.end} · ${r.engine} · created ${formatDateTime(r.created_at)}`}
        actions={
          <>
            {!completed && r.status !== 'failed' && r.status !== 'cancelled' && (
              <Button
                variant="danger"
                onClick={() => cancel.mutate(runId)}
                loading={cancel.isPending}
              >
                Cancel
              </Button>
            )}
            <Button onClick={() => toggleCompare(runId)}>
              {inCompare ? 'Remove from compare' : 'Add to compare'}
            </Button>
            <Button
              loading={clone.isPending}
              onClick={() =>
                clone.mutate(
                  { id: runId, changes: {}, run: true },
                  { onSuccess: (res) => res.run && navigate(`/runs/${res.run.id}`) },
                )
              }
            >
              Re-run
            </Button>
            {completed && (
              <>
                <a
                  className="text-sm text-accent hover:underline"
                  href={`/api/v1/runs/${runId}/export?format=csv`}
                >
                  CSV
                </a>
                <a
                  className="text-sm text-accent hover:underline"
                  href={`/api/v1/runs/${runId}/export?format=parquet`}
                >
                  Parquet
                </a>
                <a
                  className="text-sm text-accent hover:underline"
                  target="_blank"
                  rel="noreferrer"
                  href={`/api/v1/runs/${runId}/export?format=html`}
                >
                  Tearsheet
                </a>
              </>
            )}
          </>
        }
      />
      {r.status === 'failed' && <ErrorState error={new Error(r.error ?? 'Run failed')} />}
      {!completed && r.status !== 'failed' && (
        <Card title="Running…">
          <pre className="max-h-64 overflow-auto font-mono text-2xs text-secondary">
            {logs.data}
          </pre>
        </Card>
      )}
      {completed && (
        <>
          <Tabs tabs={TABS} active={tab} onChange={setTab} />
          {(window.start ?? window.end) && (
            <div className="mb-3 flex items-center gap-3 rounded-md border border-accent/40 bg-accent/5 px-3 py-2 text-sm">
              <span>
                Window {formatDate(window.start)} → {formatDate(window.end)}: metrics recalculated
                for the selection.
              </span>
              <Button size="sm" onClick={() => setWindow({})}>
                Reset
              </Button>
            </div>
          )}
          {tab === 'overview' && (
            <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
              {HEADLINE.map((h) =>
                metrics.isPending ? (
                  <Skeleton key={h.key} height={62} />
                ) : (
                  <Stat
                    key={h.key}
                    label={h.label}
                    value={formatNumber(values[h.key]?.value, h.format)}
                    className={h.signed ? signClass(values[h.key]?.value) : undefined}
                  />
                ),
              )}
            </div>
          )}
          {!['config', 'logs'].includes(tab) && (
            <div className="space-y-4">
              {charts.isError && <ErrorState error={charts.error} />}
              <div className="grid gap-4 2xl:grid-cols-2">
                {tabCharts.map((c) => (
                  <RunChart
                    key={c.name}
                    runId={runId}
                    chart={c}
                    window={window}
                    onZoom={onZoom}
                    group={group}
                  />
                ))}
              </div>
              {charts.isSuccess && tabCharts.length === 0 && TAB_CHART_GROUPS[tab] && (
                <EmptyState title="No charts apply to this run">
                  For example, factor charts need factor data.
                </EmptyState>
              )}
              {tab === 'positions' && <PositionsTable runId={runId} />}
              {tab === 'trades' && <TradesTable runId={runId} />}
              {tab === 'overlays' && <OverlayStats runId={runId} />}
              {TAB_METRIC_GROUPS[tab] &&
                (metrics.isPending ? (
                  <Skeleton height={200} />
                ) : metrics.isError ? (
                  <ErrorState error={metrics.error} />
                ) : (
                  <MetricTable
                    descriptors={metrics.data.descriptors}
                    values={values}
                    groups={TAB_METRIC_GROUPS[tab]}
                  />
                ))}
              {metrics.data?.descriptors
                .filter((d) => d.kind === 'table' && TAB_METRIC_GROUPS[tab]?.includes(d.group))
                .map((d) => {
                  const rows = values[d.key]?.table
                  return rows && rows.length ? (
                    <Card key={d.key} title={d.label} padded={false}>
                      <MetricRowsTable rows={rows} />
                    </Card>
                  ) : null
                })}
            </div>
          )}
          {tab === 'config' && (
            <div className="space-y-4">
              <Labels runId={runId} name={r.name} tags={r.tags} notes={r.notes} />
              <Card title="Reproducibility">
                <dl className="grid gap-x-6 gap-y-1 text-sm md:grid-cols-2">
                  <dt className="text-muted">Config hash</dt>
                  <dd className="font-mono">{r.config_hash}</dd>
                  <dt className="text-muted">Git commit</dt>
                  <dd className="font-mono">{r.git_commit ?? '—'}</dd>
                  <dt className="text-muted">Plugin versions</dt>
                  <dd className="font-mono text-2xs">
                    {Object.entries(r.plugin_versions)
                      .map(([k, v]) => `${k}@${v}`)
                      .join(', ')}
                  </dd>
                  <dt className="text-muted">Data snapshot</dt>
                  <dd className="font-mono text-2xs">{JSON.stringify(r.data_refs)}</dd>
                  <dt className="text-muted">Timings</dt>
                  <dd className="font-mono text-2xs">{formatDuration(r.timings.total_seconds)}</dd>
                  {r.parent_id && (
                    <>
                      <dt className="text-muted">Cloned from</dt>
                      <dd>
                        <Link className="text-accent hover:underline" to={`/runs/${r.parent_id}`}>
                          {r.parent_id}
                        </Link>
                      </dd>
                    </>
                  )}
                </dl>
              </Card>
              <Card title="Config">
                <pre className="max-h-[480px] overflow-auto font-mono text-2xs text-secondary">
                  {JSON.stringify(r.config, null, 2)}
                </pre>
              </Card>
            </div>
          )}
          {tab === 'logs' && (
            <Card title="Logs">
              <pre className="max-h-[600px] overflow-auto font-mono text-2xs text-secondary">
                {logs.data || 'No logs'}
              </pre>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
