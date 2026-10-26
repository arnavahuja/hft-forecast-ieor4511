import { Link } from 'react-router-dom'
import { useJobs, usePluginHealth, useRuns } from '@/api/hooks'
import {
  Badge,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  ProgressBar,
  Skeleton,
  StatusBadge,
} from '@/components/ui'
import { formatDateTime, formatNumber, signClass } from '@/lib/format'

export function DashboardPage() {
  const runs = useRuns({ limit: 8 })
  const jobs = useJobs()
  const health = usePluginHealth()
  const active = (jobs.data ?? []).filter((j) => j.status === 'queued' || j.status === 'running')
  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Recent runs, running jobs and plugin health."
        actions={
          <>
            <Link
              to="/data"
              className="rounded-md bg-accent px-3.5 py-2 text-sm font-medium text-canvas hover:bg-accent-strong"
            >
              Get data
            </Link>
            <Link
              to="/compare"
              className="rounded-md border border-border bg-elevated px-3.5 py-2 text-sm hover:border-muted"
            >
              Compare
            </Link>
          </>
        }
      />
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Recent runs" className="xl:col-span-2" padded={false}>
          {runs.isPending ? (
            <div className="p-4">
              <Skeleton height={200} />
            </div>
          ) : runs.isError ? (
            <div className="p-4">
              <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
            </div>
          ) : runs.data.length === 0 ? (
            <div className="p-4">
              <EmptyState title="No runs yet">
                Train a model with <code>hftf ml run</code>, then run the economic test with{' '}
                <code>hftf ml econ</code>.
              </EmptyState>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-2xs uppercase tracking-wide text-muted">
                  <th className="px-3 py-2 font-medium">Run</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 text-right font-medium">CAGR</th>
                  <th className="px-3 py-2 text-right font-medium">Sharpe</th>
                  <th className="px-3 py-2 text-right font-medium">Max DD</th>
                  <th className="px-3 py-2 font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr
                    key={r.id}
                    className="border-b border-border/60 last:border-0 hover:bg-elevated"
                  >
                    <td className="px-3 py-2">
                      <Link to={`/runs/${r.id}`} className="hover:text-accent">
                        {r.name || r.strategy}
                      </Link>
                      <div className="text-2xs text-muted">{r.strategy}</div>
                    </td>
                    <td className="px-3 py-2">
                      <StatusBadge status={r.status} />
                    </td>
                    <td
                      className={`px-3 py-2 text-right font-mono tabular-nums ${signClass(r.headline.cagr)}`}
                    >
                      {formatNumber(r.headline.cagr, 'percent')}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {formatNumber(r.headline.sharpe, 'ratio')}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums text-negative">
                      {formatNumber(r.headline.max_drawdown, 'percent')}
                    </td>
                    <td className="px-3 py-2 text-secondary">{formatDateTime(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <div className="space-y-4">
          <Card title={`Running jobs (${active.length})`}>
            {jobs.isPending ? (
              <Skeleton height={60} />
            ) : active.length === 0 ? (
              <p className="text-sm text-muted">No jobs running.</p>
            ) : (
              <ul className="space-y-3">
                {active.map((j) => (
                  <li key={j.id} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span>
                        {j.kind}{' '}
                        {j.run_id && (
                          <Link className="text-accent" to={`/runs/${j.run_id}`}>
                            {j.run_id}
                          </Link>
                        )}
                      </span>
                      <StatusBadge status={j.status} />
                    </div>
                    <ProgressBar value={j.progress} />
                    <div className="text-2xs text-muted">{j.message}</div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
          <Card title="Plugin health">
            {health.isPending ? (
              <Skeleton height={100} />
            ) : health.isError ? (
              <ErrorState error={health.error} />
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-1 text-sm">
                  {Object.entries(health.data.counts).map(([kind, n]) => (
                    <div key={kind} className="flex justify-between">
                      <span className="text-secondary">{kind}</span>
                      <span className="font-mono tabular-nums">{n}</span>
                    </div>
                  ))}
                </div>
                {health.data.failures.length === 0 ? (
                  <Badge tone="positive">all plugins loaded</Badge>
                ) : (
                  <ul className="space-y-2">
                    {health.data.failures.map((f) => (
                      <li key={f.module} className="rounded border border-negative/40 p-2 text-2xs">
                        <div className="font-mono text-negative">{f.module}</div>
                        <div className="text-secondary">{f.error}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}
