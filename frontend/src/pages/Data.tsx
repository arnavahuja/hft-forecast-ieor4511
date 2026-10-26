import type { ColumnDef } from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import { API_BASE, unwrap, type Schemas } from '@/api/client'
import {
  useCatalog,
  useDatasetPreview,
  useDatasetQuality,
  useDeleteDataset,
  useImportData,
  usePullData,
  useRefreshDataset,
  useSources,
  type DatasetRecord,
  type ImportMapping,
} from '@/api/hooks'
import { DataTable } from '@/components/DataTable'
import { JobProgress } from '@/components/JobProgress'
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
  Toggle,
} from '@/components/ui'
import { formatDate, formatDateTime, formatNumber, toText } from '@/lib/format'

const CANONICAL = [
  'open',
  'high',
  'low',
  'close',
  'adj_close',
  'volume',
  'dividend',
  'split_ratio',
] as const
// Level-1 quote fields. Whatever is left unmapped is derived on import where it follows from
// what is there: level 1 of the book is the quote, the quotes give the mid and the spread.
const QUOTE = [
  'bid',
  'ask',
  'bid_size',
  'ask_size',
  'mid',
  'microprice',
  'spread',
  'spread_bps',
  'n_quotes',
  'is_stale',
] as const
const FREQUENCIES = [
  '1s',
  '5s',
  '10s',
  '30s',
  '1min',
  '5min',
  '15min',
  '1h',
  '1d',
  '1w',
  '1mo',
] as const

type Preview = Schemas['ImportPreviewOut']

function QualityBadges({ quality }: { quality: Record<string, number> }) {
  return (
    <span className="flex gap-1">
      {(quality.error ?? 0) > 0 && <Badge tone="negative">{quality.error} errors</Badge>}
      {(quality.warning ?? 0) > 0 && <Badge tone="warning">{quality.warning} warnings</Badge>}
      {!quality.error && !quality.warning && <Badge tone="positive">ok</Badge>}
    </span>
  )
}

function PullForm() {
  const sources = useSources()
  const pull = usePullData()
  const [form, setForm] = useState({
    source: 'yahoo',
    dataset: 'daily',
    instruments: 'SPY, QQQ, TLT',
    start: '2010-01-01',
    end: new Date().toISOString().slice(0, 10),
    frequency: '1d',
    adjustment: 'split',
    refresh: false,
  })
  const source = sources.data?.find((s) => s.name === form.source)
  return (
    <Card title="Pull from a source">
      <div className="grid gap-3 md:grid-cols-4">
        <div>
          <Label htmlFor="pull-source">Source</Label>
          <Select
            id="pull-source"
            value={form.source}
            onChange={(e) => {
              const next = sources.data?.find((s) => s.name === e.target.value)
              setForm({
                ...form,
                source: e.target.value,
                dataset: toText(next?.datasets[0]?.dataset, ''),
              })
            }}
          >
            {sources.data
              ?.filter((s) => s.name !== 'local_file')
              .map((s) => (
                <option key={s.name} value={s.name} disabled={!s.available}>
                  {s.name}
                  {!s.available ? ` (${s.reason || 'unavailable'})` : ''}
                </option>
              ))}
          </Select>
        </div>
        <div>
          <Label htmlFor="pull-dataset">Dataset</Label>
          <Select
            id="pull-dataset"
            value={form.dataset}
            onChange={(e) => setForm({ ...form, dataset: e.target.value })}
          >
            {source?.datasets.map((d) => (
              <option key={toText(d.dataset)} value={toText(d.dataset)}>
                {toText(d.dataset)}
              </option>
            ))}
          </Select>
        </div>
        <div>
          <Label htmlFor="pull-freq">Frequency</Label>
          <Select
            id="pull-freq"
            value={form.frequency}
            onChange={(e) => setForm({ ...form, frequency: e.target.value })}
          >
            {FREQUENCIES.map((f) => (
              <option key={f}>{f}</option>
            ))}
          </Select>
        </div>
        <div>
          <Label htmlFor="pull-adj">Adjustment</Label>
          <Select
            id="pull-adj"
            value={form.adjustment}
            onChange={(e) => setForm({ ...form, adjustment: e.target.value })}
          >
            <option value="split">split-adjusted</option>
            <option value="total_return">total return</option>
            <option value="raw">raw</option>
          </Select>
        </div>
        <div className="md:col-span-2">
          <Label htmlFor="pull-instruments">Instruments</Label>
          <Input
            id="pull-instruments"
            value={form.instruments}
            onChange={(e) => setForm({ ...form, instruments: e.target.value })}
          />
        </div>
        <div>
          <Label htmlFor="pull-start">Start</Label>
          <Input
            id="pull-start"
            type="date"
            value={form.start}
            onChange={(e) => setForm({ ...form, start: e.target.value })}
          />
        </div>
        <div>
          <Label htmlFor="pull-end">End</Label>
          <Input
            id="pull-end"
            type="date"
            value={form.end}
            onChange={(e) => setForm({ ...form, end: e.target.value })}
          />
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between">
        <Toggle
          id="pull-refresh"
          checked={form.refresh}
          onChange={(v) => setForm({ ...form, refresh: v })}
          label="Force refresh"
        />
        <Button
          variant="primary"
          loading={pull.isPending}
          onClick={() =>
            pull.mutate({
              ...form,
              instruments: form.instruments.split(/[\s,;]+/).filter(Boolean),
              frequency: form.frequency as Schemas['Frequency'],
              adjustment: form.adjustment as Schemas['Adjustment'],
            })
          }
        >
          Pull data
        </Button>
      </div>
      {pull.isError && <ErrorState error={pull.error} />}
      {pull.data && (
        <div className="mt-3">
          <JobProgress jobId={pull.data.id} compact />
        </div>
      )}
    </Card>
  )
}

function FieldSelect({
  id,
  label,
  value,
  columns,
  onChange,
}: {
  id: string
  label: string
  value: string | undefined
  columns: string[]
  onChange: (value: string) => void
}) {
  return (
    <div>
      <Label htmlFor={id}>{label}</Label>
      <Select id={id} value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
        <option value="">\u2014</option>
        {columns.map((c) => (
          <option key={c}>{c}</option>
        ))}
      </Select>
    </div>
  )
}

function GroupHeading({ children }: { children: string }) {
  return (
    <h4 className="md:col-span-4 text-xs font-medium uppercase tracking-wide text-muted">
      {children}
    </h4>
  )
}

function MappingEditor({
  preview,
  mapping,
  onChange,
}: {
  preview: Preview
  mapping: ImportMapping
  onChange: (m: ImportMapping) => void
}) {
  const columns = preview.columns
  const book = mapping.book_columns ?? {}
  const detectedLevels = Math.floor(Object.keys(book).length / 4)
  const used = new Set([
    mapping.timestamp,
    mapping.symbol,
    ...Object.values(mapping.columns ?? {}),
    ...Object.values(book),
  ])
  const setColumn = (field: string, source: string) => {
    const rest = Object.fromEntries(
      Object.entries(mapping.columns ?? {}).filter(([k]) => k !== field),
    )
    onChange({ ...mapping, columns: source ? { ...rest, [field]: source } : rest })
  }
  return (
    <div className="grid gap-3 md:grid-cols-4">
      <div>
        <Label htmlFor="map-ts">Timestamp column</Label>
        <Select
          id="map-ts"
          value={mapping.timestamp}
          onChange={(e) => onChange({ ...mapping, timestamp: e.target.value })}
        >
          {columns.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </Select>
      </div>
      <div>
        <Label htmlFor="map-symbol">Symbol column</Label>
        <Select
          id="map-symbol"
          value={mapping.symbol ?? ''}
          onChange={(e) => onChange({ ...mapping, symbol: e.target.value || null })}
        >
          <option value="">(single instrument)</option>
          {columns.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </Select>
      </div>
      {!mapping.symbol && (
        <div>
          <Label htmlFor="map-fixed">Instrument id</Label>
          <Input
            id="map-fixed"
            value={mapping.fixed_symbol ?? ''}
            onChange={(e) => onChange({ ...mapping, fixed_symbol: e.target.value })}
          />
        </div>
      )}
      <div>
        <Label htmlFor="map-name">Dataset name</Label>
        <Input
          id="map-name"
          value={mapping.name ?? ''}
          onChange={(e) => onChange({ ...mapping, name: e.target.value })}
        />
      </div>
      <GroupHeading>Bars</GroupHeading>
      {CANONICAL.map((field) => (
        <FieldSelect
          key={field}
          id={`map-${field}`}
          label={field}
          value={mapping.columns?.[field]}
          columns={columns}
          onChange={(source) => setColumn(field, source)}
        />
      ))}
      <GroupHeading>Quotes (level 1)</GroupHeading>
      {QUOTE.map((field) => (
        <FieldSelect
          key={field}
          id={`map-${field}`}
          label={field}
          value={mapping.columns?.[field]}
          columns={columns}
          onChange={(source) => setColumn(field, source)}
        />
      ))}
      <GroupHeading>Order book</GroupHeading>
      <div>
        <Label htmlFor="map-book-levels">Book levels to import</Label>
        <Input
          id="map-book-levels"
          type="number"
          min={0}
          value={mapping.book_levels ?? 0}
          onChange={(e) => onChange({ ...mapping, book_levels: Number(e.target.value) || 0 })}
        />
      </div>
      <p className="self-end pb-2 text-xs text-muted md:col-span-3">
        {detectedLevels > 0
          ? `${detectedLevels} level(s) found in the file. Level 1 fills bid, ask and the sizes when they are not mapped above.`
          : 'No bid_px_1 / bid_sz_1 style columns found, so this file is level 1 at best.'}
      </p>
      <GroupHeading>How to read the file</GroupHeading>
      <div>
        <Label htmlFor="map-freq">Frequency</Label>
        <Select
          id="map-freq"
          value={mapping.frequency ?? '1d'}
          onChange={(e) =>
            onChange({ ...mapping, frequency: e.target.value as Schemas['Frequency'] })
          }
        >
          {FREQUENCIES.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </Select>
      </div>
      <div>
        <Label htmlFor="map-bar-label">Timestamps mark the bar</Label>
        <Select
          id="map-bar-label"
          value={mapping.bar_label ?? 'end'}
          onChange={(e) => onChange({ ...mapping, bar_label: e.target.value })}
        >
          <option value="end">end (right-closed)</option>
          <option value="start">start (shifted to the end on import)</option>
        </Select>
      </div>
      <div>
        <Label htmlFor="map-tz">Timezone of timestamps</Label>
        <Input
          id="map-tz"
          value={mapping.timezone ?? 'UTC'}
          onChange={(e) => onChange({ ...mapping, timezone: e.target.value })}
        />
      </div>
      <div>
        <Label htmlFor="map-fmt">Datetime format (optional)</Label>
        <Input
          id="map-fmt"
          placeholder="%Y-%m-%d"
          value={mapping.datetime_format ?? ''}
          onChange={(e) => onChange({ ...mapping, datetime_format: e.target.value || null })}
        />
      </div>
      <div>
        <Label htmlFor="map-adj">Prices are</Label>
        <Select
          id="map-adj"
          value={mapping.adjustment ?? 'split'}
          onChange={(e) =>
            onChange({ ...mapping, adjustment: e.target.value as Schemas['Adjustment'] })
          }
        >
          <option value="split">split-adjusted</option>
          <option value="total_return">total-return adjusted</option>
          <option value="raw">raw (unadjusted)</option>
        </Select>
      </div>
      <div className="md:col-span-4">
        <Label>Extra columns kept as fields</Label>
        <div className="flex flex-wrap gap-3">
          {columns
            .filter((c) => !used.has(c) || (mapping.extra ?? []).includes(c))
            .map((c) => (
              <Toggle
                key={c}
                id={`extra-${c}`}
                checked={(mapping.extra ?? []).includes(c)}
                onChange={(on) =>
                  onChange({
                    ...mapping,
                    extra: on
                      ? [...(mapping.extra ?? []), c]
                      : (mapping.extra ?? []).filter((x) => x !== c),
                  })
                }
                label={c}
              />
            ))}
        </div>
      </div>
    </div>
  )
}

function ImportWizard() {
  const [preview, setPreview] = useState<Preview | null>(null)
  const [mapping, setMapping] = useState<ImportMapping | null>(null)
  const [path, setPath] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const importer = useImportData()

  const load = async (form: FormData) => {
    setLoading(true)
    setError(null)
    try {
      // multipart upload: plain fetch, response typed by the generated schema
      const response = await fetch(`${API_BASE}/data/import/preview`, {
        method: 'POST',
        body: form,
      })
      const body: unknown = await response.json()
      const data = unwrap<Preview>({
        data: body as Preview,
        error: response.ok ? undefined : body,
        response,
      })
      setPreview(data)
      setMapping(data.mapping)
    } catch (e) {
      setError(e)
    } finally {
      setLoading(false)
    }
  }

  const previewColumns = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () =>
      (preview?.columns ?? []).map((c) => ({
        accessorKey: c,
        header: c,
        cell: (x) => toText(x.getValue(), ''),
      })),
    [preview],
  )

  return (
    <Card title="Import a local file (CSV, Parquet, Excel, Feather)">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <Label htmlFor="upload">Upload</Label>
          <input
            id="upload"
            type="file"
            accept=".csv,.txt,.parquet,.pq,.xlsx,.xls,.feather,.arrow,.ipc"
            className="text-sm text-secondary file:mr-3 file:rounded file:border file:border-border file:bg-elevated file:px-3 file:py-1.5 file:text-primary"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (!file) return
              const form = new FormData()
              form.append('file', file)
              void load(form)
            }}
          />
        </div>
        <span className="pb-2 text-sm text-muted">or</span>
        <div className="min-w-72 flex-1">
          <Label htmlFor="path">Local path</Label>
          <Input
            id="path"
            value={path}
            placeholder="/path/to/data.csv"
            onChange={(e) => setPath(e.target.value)}
          />
        </div>
        <Button
          disabled={!path}
          loading={loading}
          onClick={() => {
            const form = new FormData()
            form.append('path', path)
            void load(form)
          }}
        >
          Preview
        </Button>
      </div>
      {error ? (
        <div className="mt-3">
          <ErrorState error={error} />
        </div>
      ) : null}
      {preview && mapping && (
        <div className="mt-4 space-y-4">
          <div className="rounded-md border border-border">
            <DataTable data={preview.rows.slice(0, 8)} columns={previewColumns} dense />
          </div>
          <MappingEditor preview={preview} mapping={mapping} onChange={setMapping} />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setPreview(null)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              loading={importer.isPending}
              onClick={() =>
                importer.mutate(
                  { token: preview.token, mapping },
                  { onSuccess: () => setPreview(null) },
                )
              }
            >
              Import
            </Button>
          </div>
          {importer.isError && <ErrorState error={importer.error} />}
        </div>
      )}
      {importer.isSuccess && !preview && (
        <p className="mt-3 text-sm text-positive">
          Imported {importer.data.name} ({formatNumber(importer.data.rows, 'integer')} rows).
        </p>
      )}
    </Card>
  )
}

function DatasetDetail({ dataset, onClose }: { dataset: DatasetRecord; onClose: () => void }) {
  const preview = useDatasetPreview(dataset.id)
  const quality = useDatasetQuality(dataset.id)
  const refresh = useRefreshDataset()
  const remove = useDeleteDataset()
  const cols = useMemo<ColumnDef<Record<string, unknown>>[]>(
    () =>
      Object.keys(preview.data?.rows[0] ?? {}).map((c) => ({
        accessorKey: c,
        header: c,
        meta: { numeric: typeof preview.data?.rows[0]?.[c] === 'number' },
        cell: (x) => {
          const v = x.getValue()
          return typeof v === 'number' ? formatNumber(v, 'ratio') : toText(v, '')
        },
      })),
    [preview.data],
  )
  return (
    <Card
      title={`${dataset.name} · ${dataset.id}`}
      actions={
        <>
          {dataset.kind === 'cache' && (
            <Button
              size="sm"
              loading={refresh.isPending}
              onClick={() => refresh.mutate(dataset.id)}
            >
              Refresh
            </Button>
          )}
          <Button
            size="sm"
            variant="danger"
            loading={remove.isPending}
            onClick={() => {
              if (window.confirm(`Delete dataset ${dataset.name}?`))
                remove.mutate(dataset.id, { onSuccess: onClose })
            }}
          >
            Delete
          </Button>
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        </>
      }
    >
      <div className="grid gap-4 xl:grid-cols-2">
        <div>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
            Data quality
          </h4>
          {quality.isPending ? (
            <Skeleton height={80} />
          ) : quality.isError ? (
            <ErrorState error={quality.error} />
          ) : quality.data.issues.length === 0 ? (
            <p className="text-sm text-positive">No issues found.</p>
          ) : (
            <ul className="space-y-1.5 text-sm">
              {quality.data.issues.map((i, k) => (
                <li key={k} className="flex gap-2">
                  <Badge
                    tone={
                      i.severity === 'error'
                        ? 'negative'
                        : i.severity === 'warning'
                          ? 'warning'
                          : 'info'
                    }
                  >
                    {i.severity}
                  </Badge>
                  <span className="text-secondary">
                    {i.instrument && (
                      <span className="font-mono text-2xs text-muted">{i.instrument} </span>
                    )}
                    {i.message}
                    {i.examples.length > 0 && (
                      <span className="text-muted"> e.g. {i.examples.slice(0, 3).join(', ')}</span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
            Instruments
          </h4>
          {preview.data ? (
            <div className="max-h-48 overflow-auto text-sm">
              <table className="w-full">
                <tbody>
                  {(preview.data.stats.instruments as Record<string, unknown>[]).map((s) => (
                    <tr key={String(s.instrument_id)} className="border-b border-border/60">
                      <td className="py-1">{String(s.instrument_id)}</td>
                      <td className="py-1 text-right font-mono tabular-nums">
                        {formatNumber(Number(s.rows), 'integer')}
                      </td>
                      <td className="py-1 text-right text-muted">
                        {formatDate(String(s.start))} → {formatDate(String(s.end))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Skeleton height={80} />
          )}
        </div>
      </div>
      <h4 className="mb-2 mt-4 text-xs font-medium uppercase tracking-wide text-muted">Preview</h4>
      {preview.isPending ? (
        <Skeleton height={160} />
      ) : preview.isError ? (
        <ErrorState error={preview.error} />
      ) : (
        <div className="max-h-80 overflow-auto rounded-md border border-border">
          <DataTable data={preview.data.rows} columns={cols} dense />
        </div>
      )}
    </Card>
  )
}

export function DataPage() {
  const catalog = useCatalog()
  const [selected, setSelected] = useState<DatasetRecord | null>(null)
  const columns = useMemo<ColumnDef<DatasetRecord>[]>(
    () => [
      { accessorKey: 'name', header: 'Name' },
      { accessorKey: 'kind', header: 'Kind', cell: (c) => <Badge>{String(c.getValue())}</Badge> },
      { accessorKey: 'source', header: 'Source' },
      { accessorKey: 'frequency', header: 'Freq' },
      { accessorKey: 'start', header: 'Start', cell: (c) => formatDate(c.getValue() as string) },
      { accessorKey: 'end', header: 'End', cell: (c) => formatDate(c.getValue() as string) },
      {
        id: 'n',
        header: 'Instruments',
        meta: { numeric: true },
        accessorFn: (r) => r.instruments.length,
      },
      {
        accessorKey: 'rows',
        header: 'Rows',
        meta: { numeric: true },
        cell: (c) => formatNumber(c.getValue() as number, 'integer'),
      },
      {
        accessorKey: 'quality_label',
        header: 'Quality',
        cell: (c) => (
          <span className="flex items-center gap-1">
            <Badge tone={c.getValue() === 'lower' ? 'warning' : 'neutral'}>
              {String(c.getValue())}
            </Badge>
            <QualityBadges quality={c.row.original.quality} />
          </span>
        ),
      },
      {
        accessorKey: 'survivorship_bias_free',
        header: 'Survivorship-free',
        cell: (c) => (c.getValue() ? 'yes' : 'no'),
      },
      {
        accessorKey: 'last_refresh',
        header: 'Refreshed',
        cell: (c) => formatDateTime(c.getValue() as string),
      },
    ],
    [],
  )
  return (
    <div className="space-y-4">
      <PageHeader
        title="Data"
        subtitle="Catalog of cached and imported datasets. Every fetch goes through the Parquet cache."
      />
      <Card title="Catalog" padded={false}>
        {catalog.isPending ? (
          <div className="p-4">
            <Skeleton height={120} />
          </div>
        ) : catalog.isError ? (
          <div className="p-4">
            <ErrorState error={catalog.error} onRetry={() => void catalog.refetch()} />
          </div>
        ) : catalog.data.length === 0 ? (
          <div className="p-4">
            <EmptyState title="No datasets yet">
              Pull from a source or import a file below.
            </EmptyState>
          </div>
        ) : (
          <DataTable
            data={catalog.data}
            columns={columns}
            getRowId={(r) => r.id}
            onRowClick={setSelected}
          />
        )}
      </Card>
      {selected && <DatasetDetail dataset={selected} onClose={() => setSelected(null)} />}
      <div className="grid gap-4 xl:grid-cols-2">
        <PullForm />
        <ImportWizard />
      </div>
    </div>
  )
}
