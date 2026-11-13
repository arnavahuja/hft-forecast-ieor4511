import type { ColumnDef } from '@tanstack/react-table'
import { useEffect, useMemo, useState } from 'react'
import { useEdaStudies, useEdaStudy, type EdaStudy } from '@/api/hooks'
import { ChartRenderer } from '@/components/ChartRenderer'
import { DataTable } from '@/components/DataTable'
import {
  Badge,
  Card,
  EmptyState,
  ErrorState,
  Label,
  PageHeader,
  Select,
  Skeleton,
  Stat,
} from '@/components/ui'
import { formatDateTime, formatNumber, signClass } from '@/lib/format'

/**
 * Feature research: what a study found before any model was fitted.
 *
 * Read-only, like the rest of the app. A study is produced by `hftf ml eda` and stored with
 * the run it is registered as; this page lists them and renders what was stored, with an
 * optional second study beside the first so two horizons can be read against each other.
 */
export function EdaPage() {
  const studies = useEdaStudies()
  const [leftId, setLeftId] = useState('')
  const [rightId, setRightId] = useState('')

  const rows = useMemo(() => studies.data ?? [], [studies.data])
  useEffect(() => {
    const completed = rows.filter((r) => r.status === 'completed')
    const first = completed[0]
    if (!leftId && first) setLeftId(first.id)
  }, [rows, leftId])

  const left = useEdaStudy(leftId || undefined)
  const right = useEdaStudy(rightId || undefined)

  if (studies.isPending) return <Skeleton className="w-full" height={320} />
  if (studies.isError)
    return <ErrorState error={studies.error} onRetry={() => void studies.refetch()} />

  const options = rows
    .filter((r) => r.status === 'completed')
    .map((r) => ({
      value: r.id,
      label: `${r.name || r.experiment || r.id} · ${formatDateTime(r.created_at)}`,
    }))

  return (
    <div className="space-y-5">
      <PageHeader
        title="Feature research"
        subtitle="Which features lead the label, how reliably, and how fast the edge decays"
      />

      {options.length === 0 ? (
        <EmptyState title="No feature studies yet">
          Run <code className="font-mono">hftf ml eda &lt;config.yaml&gt;</code> for a horizon, then
          reload. Studies are computed on the CV training sessions only.
        </EmptyState>
      ) : (
        <>
          <Card>
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-64 flex-1">
                <Label htmlFor="eda-left">Study</Label>
                <Select
                  id="eda-left"
                  value={leftId}
                  onChange={(e) => setLeftId(e.target.value)}
                  aria-label="Study"
                >
                  {options.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="min-w-64 flex-1">
                <Label htmlFor="eda-right">Compare with</Label>
                <Select
                  id="eda-right"
                  value={rightId}
                  onChange={(e) => setRightId(e.target.value)}
                  aria-label="Compare with"
                >
                  <option value="">None</option>
                  {options
                    .filter((o) => o.value !== leftId)
                    .map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                </Select>
              </div>
            </div>
          </Card>

          {left.isError && <ErrorState error={left.error} onRetry={() => void left.refetch()} />}
          {left.isPending && leftId && <Skeleton className="w-full" height={320} />}
          {left.data && (
            <div className={rightId && right.data ? 'grid gap-5 2xl:grid-cols-2' : 'space-y-5'}>
              <StudyView study={left.data} />
              {rightId && right.data && <StudyView study={right.data} compact />}
            </div>
          )}
        </>
      )}
    </div>
  )
}

/** One study: header numbers, its charts, and the full ranking table. */
function StudyView({ study, compact = false }: { study: EdaStudy; compact?: boolean }) {
  const s = study.summary as Record<string, number | string | null>
  const featureNames = (study.features as { name?: string }[])
    .map((f) => f.name)
    .filter(Boolean)
    .join(', ')
  return (
    <div className="space-y-5">
      <Card
        title={`${study.horizon} · ${study.horizon_seconds}s ahead on ${study.bar_seconds}s bars`}
      >
        <div className="mb-3 flex flex-wrap gap-1">
          <Badge>{study.experiment}</Badge>
          {featureNames && <Badge tone="info">{featureNames}</Badge>}
          <Badge tone={s.train_sessions === null ? 'warning' : 'neutral'}>
            {s.train_sessions === null
              ? 'all sessions (includes test folds)'
              : `first ${String(s.train_sessions)} sessions`}
          </Badge>
        </div>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Stat label="Instruments" value={formatNumber(num(s.instruments), 'integer')} />
          <Stat label="Sessions" value={formatNumber(num(s.sessions), 'integer')} />
          <Stat label="Labelled bars" value={formatNumber(num(s.labelled_bars), 'integer')} />
          <Stat
            label="Features ranked"
            value={`${formatNumber(num(s.features_ranked), 'integer')} / ${formatNumber(num(s.features), 'integer')}`}
          />
          <Stat label="Strongest" value={<span className="text-sm">{str(s.top_feature)}</span>} />
          <Stat
            label="Its mean IC"
            value={formatNumber(num(s.top_mean_ic))}
            className={signClass(num(s.top_mean_ic))}
          />
        </div>
        <p className="mt-3 text-2xs text-muted">
          The information coefficient is a rank correlation with the forward return, computed per
          instrument-session and averaged over {formatNumber(num(s.sessions), 'integer')} sessions.
          Blocks are not independent, so read the t-statistic as a ranking device, not as a test.
        </p>
      </Card>

      {study.charts.map((spec) => (
        <Card key={spec.id} title={spec.title}>
          <ChartRenderer spec={spec} height={compact ? 260 : 320} />
        </Card>
      ))}

      <Card title="Ranking (every feature)">
        <RankingTable rows={study.tables.ranking ?? []} />
      </Card>
    </div>
  )
}

type Row = Record<string, unknown>

const RANKING_COLUMNS: ColumnDef<Row>[] = [
  { accessorKey: 'feature', header: 'Feature' },
  {
    accessorKey: 'mean_ic',
    header: 'Mean IC',
    meta: { numeric: true },
    cell: (c) => (
      <span className={signClass(c.getValue() as number)}>
        {formatNumber(c.getValue() as number)}
      </span>
    ),
  },
  {
    accessorKey: 't_stat',
    header: 't-stat',
    meta: { numeric: true },
    cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
  },
  {
    accessorKey: 'median_ic',
    header: 'Median IC',
    meta: { numeric: true },
    cell: (c) => formatNumber(c.getValue() as number),
  },
  {
    accessorKey: 'share_positive',
    header: 'Blocks positive',
    meta: { numeric: true },
    cell: (c) => formatNumber(c.getValue() as number, 'percent'),
  },
  {
    accessorKey: 'blocks',
    header: 'Blocks',
    meta: { numeric: true },
    cell: (c) => formatNumber(c.getValue() as number, 'integer'),
  },
  {
    accessorKey: 'rows',
    header: 'Rows',
    meta: { numeric: true },
    cell: (c) => formatNumber(c.getValue() as number, 'integer'),
  },
]

function RankingTable({ rows }: { rows: Row[] }) {
  return (
    <DataTable
      data={rows}
      columns={RANKING_COLUMNS}
      dense
      empty="No feature produced a correlation."
    />
  )
}

function num(value: unknown): number | null {
  return typeof value === 'number' ? value : null
}

function str(value: unknown): string {
  return typeof value === 'string' && value ? value : '—'
}
