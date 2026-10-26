import type { ColumnDef, RowSelectionState } from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, unwrap } from '@/api/client'
import { useDeleteRuns, useExperiments, useRuns, type RunRecord } from '@/api/hooks'
import { DataTable } from '@/components/DataTable'
import {
  Badge,
  Button,
  Card,
  ErrorState,
  Input,
  PageHeader,
  Select,
  Skeleton,
  StatusBadge,
  Toggle,
} from '@/components/ui'
import { formatDateTime, formatNumber, signClass } from '@/lib/format'
import { useCompareStore } from '@/store/compare'
import { useQueryClient } from '@tanstack/react-query'

export function RunsPage() {
  const [search, setSearch] = useState('')
  const [experiment, setExperiment] = useState('')
  const [status, setStatus] = useState('')
  const [tag, setTag] = useState('')
  const [archived, setArchived] = useState(false)
  const [selection, setSelection] = useState<RowSelectionState>({})
  const runs = useRuns({
    search: search || undefined,
    experiment: experiment || undefined,
    status: status || undefined,
    tag: tag || undefined,
    include_archived: archived,
  })
  const experiments = useExperiments()
  const remove = useDeleteRuns()
  const setCompare = useCompareStore((s) => s.setSelected)
  const navigate = useNavigate()
  const qc = useQueryClient()
  const selectedIds = Object.keys(selection).filter((k) => selection[k])

  const archive = async (value: boolean) => {
    for (const id of selectedIds) {
      unwrap(
        await api.PATCH('/api/v1/runs/{run_id}', {
          params: { path: { run_id: id } },
          body: { archived: value },
        }),
      )
    }
    setSelection({})
    void qc.invalidateQueries({ queryKey: ['runs'] })
  }

  const columns = useMemo<ColumnDef<RunRecord>[]>(
    () => [
      {
        accessorKey: 'name',
        header: 'Run',
        cell: (c) => (
          <Link
            to={`/runs/${c.row.original.id}`}
            className="hover:text-accent"
            onClick={(e) => e.stopPropagation()}
          >
            {c.row.original.name || c.row.original.strategy}
            <div className="font-mono text-2xs text-muted">{c.row.original.id}</div>
          </Link>
        ),
      },
      { accessorKey: 'strategy', header: 'Strategy' },
      {
        accessorKey: 'status',
        header: 'Status',
        cell: (c) => <StatusBadge status={String(c.getValue())} />,
      },
      { accessorKey: 'experiment', header: 'Experiment' },
      {
        accessorKey: 'tags',
        header: 'Tags',
        enableSorting: false,
        cell: (c) => (
          <span className="flex flex-wrap gap-1">
            {(c.getValue() as string[]).map((t) => (
              <Badge key={t}>{t}</Badge>
            ))}
            {c.row.original.archived && <Badge tone="warning">archived</Badge>}
          </span>
        ),
      },
      {
        id: 'cagr',
        header: 'CAGR',
        meta: { numeric: true },
        accessorFn: (r) => r.headline.cagr ?? null,
        cell: (c) => (
          <span className={signClass(c.getValue() as number)}>
            {formatNumber(c.getValue() as number, 'percent')}
          </span>
        ),
      },
      {
        id: 'sharpe',
        header: 'Sharpe',
        meta: { numeric: true },
        accessorFn: (r) => r.headline.sharpe ?? null,
        cell: (c) => formatNumber(c.getValue() as number, 'ratio'),
      },
      {
        id: 'vol',
        header: 'Vol',
        meta: { numeric: true },
        accessorFn: (r) => r.headline.ann_vol ?? null,
        cell: (c) => formatNumber(c.getValue() as number, 'percent'),
      },
      {
        id: 'mdd',
        header: 'Max DD',
        meta: { numeric: true },
        accessorFn: (r) => r.headline.max_drawdown ?? null,
        cell: (c) => formatNumber(c.getValue() as number, 'percent'),
      },
      {
        accessorKey: 'created_at',
        header: 'Created',
        cell: (c) => formatDateTime(c.getValue() as string),
      },
    ],
    [],
  )

  return (
    <div>
      <PageHeader
        title="Runs"
        subtitle="Searchable history with tags, experiments and bulk actions."
      />
      <Card
        padded={false}
        title={
          <div className="flex flex-wrap items-center gap-2">
            <Input
              aria-label="Search runs"
              placeholder="Search name, strategy, notes…"
              className="w-64"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <Select
              aria-label="Experiment"
              className="w-44"
              value={experiment}
              onChange={(e) => setExperiment(e.target.value)}
            >
              <option value="">All experiments</option>
              {experiments.data?.map((x) => (
                <option key={x.name} value={x.name}>
                  {x.name} ({x.runs})
                </option>
              ))}
            </Select>
            <Select
              aria-label="Status"
              className="w-36"
              value={status}
              onChange={(e) => setStatus(e.target.value)}
            >
              <option value="">Any status</option>
              {['queued', 'running', 'completed', 'failed', 'cancelled'].map((s) => (
                <option key={s}>{s}</option>
              ))}
            </Select>
            <Input
              aria-label="Tag"
              placeholder="tag"
              className="w-28"
              value={tag}
              onChange={(e) => setTag(e.target.value)}
            />
            <Toggle id="archived" checked={archived} onChange={setArchived} label="Archived" />
          </div>
        }
        actions={
          selectedIds.length > 0 && (
            <>
              <span className="text-2xs text-muted">{selectedIds.length} selected</span>
              <Button
                size="sm"
                variant="primary"
                onClick={() => {
                  setCompare(selectedIds)
                  navigate('/compare')
                }}
              >
                Compare
              </Button>
              <Button size="sm" onClick={() => void archive(true)}>
                Archive
              </Button>
              <Button size="sm" onClick={() => void archive(false)}>
                Unarchive
              </Button>
              <Button
                size="sm"
                variant="danger"
                loading={remove.isPending}
                onClick={() => {
                  if (window.confirm(`Delete ${selectedIds.length} runs?`))
                    remove.mutate(selectedIds, { onSuccess: () => setSelection({}) })
                }}
              >
                Delete
              </Button>
            </>
          )
        }
      >
        {runs.isPending ? (
          <div className="p-4">
            <Skeleton height={240} />
          </div>
        ) : runs.isError ? (
          <div className="p-4">
            <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
          </div>
        ) : (
          <DataTable
            data={runs.data}
            columns={columns}
            getRowId={(r) => r.id}
            selectable
            selection={selection}
            onSelectionChange={setSelection}
            onRowClick={(r) => navigate(`/runs/${r.id}`)}
            empty="No runs match the filters"
          />
        )}
      </Card>
    </div>
  )
}
