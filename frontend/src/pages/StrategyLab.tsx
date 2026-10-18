import { DndContext, closestCenter, type DragEndEvent } from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useEffect, useMemo, useState } from 'react'
import {
  useLaunchRun,
  useLookaheadCheck,
  usePlugins,
  usePresets,
  useSavePreset,
  useSources,
  useValidateRun,
  type Issue,
  type Plugin,
} from '@/api/hooks'
import { JobProgress } from '@/components/JobProgress'
import { PluginPicker, type PluginChoice } from '@/components/PluginPicker'
import { SchemaForm } from '@/components/SchemaForm'
import {
  Badge,
  Button,
  Card,
  ErrorState,
  Input,
  Label,
  PageHeader,
  Select,
  Skeleton,
  Toggle,
} from '@/components/ui'
import { configToDraft, draftToConfig } from '@/lib/config'
import { toText } from '@/lib/format'
import { newKey, useLabStore, type LabDraft, type PluginRefDraft } from '@/store/lab'

const VALIDATE_DEBOUNCE_MS = 400

function choice(ref: PluginRefDraft | null): PluginChoice | null {
  return ref ? { name: ref.name, params: ref.params } : null
}

function toRef(key: string, c: PluginChoice | null): PluginRefDraft | null {
  return c ? { key, name: c.name, params: c.params } : null
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent/15 text-2xs text-accent">
            {n}
          </span>
          {title}
        </span>
      }
    >
      {children}
    </Card>
  )
}

function IssueList({ issues }: { issues: Issue[] }) {
  if (!issues.length) {
    return <p className="text-sm text-positive">Configuration is valid.</p>
  }
  return (
    <ul className="space-y-1.5 text-sm">
      {issues.map((i, k) => (
        <li key={`${i.field}-${k}`} className="flex gap-2">
          <Badge tone={i.level === 'error' ? 'negative' : 'warning'}>{i.level}</Badge>
          <span>
            <span className="font-mono text-2xs text-muted">{i.field}</span>{' '}
            <span className="text-secondary">{i.message}</span>
          </span>
        </li>
      ))}
    </ul>
  )
}

function SortableOverlay({
  item,
  plugin,
  onChange,
  onRemove,
  index,
}: {
  item: PluginRefDraft
  plugin: Plugin | undefined
  onChange: (params: Record<string, unknown>) => void
  onRemove: () => void
  index: number
}) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: item.key })
  return (
    <div
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className="rounded-md border border-border bg-canvas p-3"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="cursor-grab rounded px-1 text-muted hover:text-primary"
            aria-label={`Drag to reorder ${item.name}`}
            {...attributes}
            {...listeners}
          >
            ⋮⋮
          </button>
          <span className="text-xs text-muted">{index + 1}.</span>
          <span className="text-sm font-medium">{item.name}</span>
          {plugin?.implements.map((c) => (
            <Badge key={c} tone="accent">
              {c}
            </Badge>
          ))}
        </div>
        <Button size="sm" variant="ghost" onClick={onRemove} aria-label={`Remove ${item.name}`}>
          Remove
        </Button>
      </div>
      {plugin ? (
        <SchemaForm
          schema={plugin.params_schema}
          value={item.params}
          onChange={(v) => onChange(v)}
        />
      ) : (
        <p className="text-sm text-negative">Plugin not installed.</p>
      )}
    </div>
  )
}

function PluginList({
  label,
  kind,
  plugins,
  items,
  onChange,
  sortable,
}: {
  label: string
  kind: string
  plugins: Plugin[]
  items: PluginRefDraft[]
  onChange: (items: PluginRefDraft[]) => void
  sortable: boolean
}) {
  const [adding, setAdding] = useState('')
  const byName = new Map(plugins.map((p) => [p.name, p]))
  const onDragEnd = (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return
    const from = items.findIndex((i) => i.key === e.active.id)
    const to = items.findIndex((i) => i.key === e.over?.id)
    onChange(arrayMove(items, from, to))
  }
  const list = items.map((item, index) => (
    <SortableOverlay
      key={item.key}
      item={item}
      index={index}
      plugin={byName.get(item.name)}
      onChange={(params) => onChange(items.map((i) => (i.key === item.key ? { ...i, params } : i)))}
      onRemove={() => onChange(items.filter((i) => i.key !== item.key))}
    />
  ))
  return (
    <div className="space-y-2">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <Label htmlFor={`add-${kind}`}>{label}</Label>
          <Select id={`add-${kind}`} value={adding} onChange={(e) => setAdding(e.target.value)}>
            <option value="">Add…</option>
            {plugins.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </Select>
        </div>
        <Button
          disabled={!adding}
          onClick={() => {
            onChange([...items, { key: newKey(kind), name: adding, params: {} }])
            setAdding('')
          }}
        >
          Add
        </Button>
      </div>
      {sortable ? (
        <DndContext collisionDetection={closestCenter} onDragEnd={onDragEnd}>
          <SortableContext items={items.map((i) => i.key)} strategy={verticalListSortingStrategy}>
            <div className="space-y-2">{list}</div>
          </SortableContext>
        </DndContext>
      ) : (
        <div className="space-y-2">{list}</div>
      )}
    </div>
  )
}

export function StrategyLabPage() {
  const { draft, set, replace, reset } = useLabStore()
  const plugins = usePlugins()
  const sources = useSources()
  const presets = usePresets()
  const validate = useValidateRun()
  const launch = useLaunchRun()
  const lookahead = useLookaheadCheck()
  const savePreset = useSavePreset()
  const [presetName, setPresetName] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)

  const config = useMemo(() => draftToConfig(draft), [draft])
  const configKey = JSON.stringify(config)
  const { mutate: runValidation } = validate

  useEffect(() => {
    const t = window.setTimeout(() => runValidation(config), VALIDATE_DEBOUNCE_MS)
    return () => window.clearTimeout(t)
    // configKey captures every change to the config
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configKey, runValidation])

  const ofKind = (kind: string) => (plugins.data ?? []).filter((p) => p.kind === kind)
  const source = sources.data?.find((s) => s.name === draft.source)
  const datasets = source?.datasets ?? []
  const dataset = datasets.find((d) => d.dataset === draft.dataset)
  const frequencies = (dataset?.frequencies as string[] | undefined) ?? ['1d']
  const issues = validate.data?.issues ?? []
  const valid = validate.data?.ok ?? false

  const upd =
    <K extends keyof LabDraft>(key: K) =>
    (value: LabDraft[K]) => {
      const patch: Partial<LabDraft> = {}
      patch[key] = value
      set(patch)
    }

  if (plugins.isPending || sources.isPending) {
    return (
      <div className="space-y-4">
        <Skeleton height={40} />
        <Skeleton height={300} />
      </div>
    )
  }
  if (plugins.isError)
    return <ErrorState error={plugins.error} onRetry={() => void plugins.refetch()} />

  return (
    <div>
      <PageHeader
        title="Strategy Lab"
        subtitle="Build a run step by step. Forms are generated from each plugin's parameter schema."
        actions={
          <>
            <Select
              aria-label="Load preset"
              className="w-48"
              value=""
              onChange={(e) => {
                const p = presets.data?.find((x) => x.id === e.target.value)
                if (p) replace(configToDraft(p.config as Parameters<typeof configToDraft>[0]))
              }}
            >
              <option value="">Load preset…</option>
              {presets.data?.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </Select>
            <Button variant="ghost" onClick={reset}>
              Reset
            </Button>
          </>
        }
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-4">
          <Step n={1} title="Data and universe">
            <div className="grid gap-3 md:grid-cols-3">
              <div>
                <Label htmlFor="source">Source</Label>
                <Select
                  id="source"
                  value={draft.source}
                  onChange={(e) => {
                    const next = sources.data?.find((s) => s.name === e.target.value)
                    set({
                      source: e.target.value,
                      dataset: toText(next?.datasets[0]?.dataset, ''),
                    })
                  }}
                >
                  {sources.data?.map((s) => (
                    <option key={s.name} value={s.name} disabled={!s.available}>
                      {s.name}
                      {s.available ? '' : ' (unavailable)'}
                    </option>
                  ))}
                </Select>
              </div>
              <div>
                <Label htmlFor="dataset">Dataset</Label>
                <Select
                  id="dataset"
                  value={draft.dataset}
                  onChange={(e) => set({ dataset: e.target.value })}
                >
                  {datasets.map((d) => (
                    <option key={toText(d.dataset)} value={toText(d.dataset)}>
                      {toText(d.dataset)}
                    </option>
                  ))}
                </Select>
              </div>
              <div>
                <Label htmlFor="freq">Frequency</Label>
                <Select
                  id="freq"
                  value={draft.frequency}
                  onChange={(e) => set({ frequency: e.target.value })}
                >
                  {frequencies.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="md:col-span-3">
                <Label htmlFor="instruments">Instruments (comma separated)</Label>
                <Input
                  id="instruments"
                  value={draft.instruments}
                  onChange={(e) => set({ instruments: e.target.value })}
                  placeholder="SPY, TLT, GLD"
                />
                {dataset?.notes ? (
                  <p className="mt-1 text-2xs text-warning">{toText(dataset.notes)}</p>
                ) : null}
              </div>
              <div>
                <Label htmlFor="start">Start</Label>
                <Input
                  id="start"
                  type="date"
                  value={draft.start}
                  onChange={(e) => set({ start: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="end">End</Label>
                <Input
                  id="end"
                  type="date"
                  value={draft.end}
                  onChange={(e) => set({ end: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="adj">Adjustment</Label>
                <Select
                  id="adj"
                  value={draft.adjustment}
                  onChange={(e) => set({ adjustment: e.target.value })}
                >
                  <option value="split">split-adjusted</option>
                  <option value="total_return">total return</option>
                  <option value="raw">raw</option>
                </Select>
              </div>
            </div>
            <div className="mt-4">
              <PluginPicker
                id="universe"
                label="Universe (point-in-time)"
                plugins={ofKind('universe')}
                allowNone
                noneLabel="static (all instruments)"
                value={choice(draft.universe)}
                onChange={(c) => set({ universe: toRef('universe', c) })}
              />
            </div>
          </Step>

          <Step n={2} title="Strategy">
            <PluginPicker
              id="strategy"
              label="Strategy"
              plugins={ofKind('strategy')}
              value={choice(draft.strategy)}
              onChange={(c) => set({ strategy: toRef('strategy', c) })}
            />
          </Step>

          <Step n={3} title="Portfolio construction">
            <PluginPicker
              id="constructor"
              label="Portfolio constructor"
              plugins={ofKind('portfolio_constructor')}
              allowNone
              noneLabel="None (strategy emits weights)"
              value={choice(draft.portfolio)}
              onChange={(c) => set({ portfolio: toRef('constructor', c) })}
            />
          </Step>

          <Step n={4} title="Overlays (applied in order; drag to reorder)">
            <PluginList
              label="Overlay"
              kind="overlay"
              plugins={ofKind('overlay')}
              items={draft.overlays}
              onChange={upd('overlays')}
              sortable
            />
          </Step>

          <Step n={5} title="Engine and costs">
            <div className="grid gap-3 md:grid-cols-4">
              <div>
                <Label htmlFor="engine">Engine</Label>
                <Select
                  id="engine"
                  value={draft.engine}
                  onChange={(e) => set({ engine: e.target.value })}
                >
                  <option value="vectorized">vectorized</option>
                  <option value="event">event-driven</option>
                </Select>
              </div>
              <div>
                <Label htmlFor="lag">Execution lag (bars)</Label>
                <Input
                  id="lag"
                  type="number"
                  min={1}
                  value={draft.lagBars}
                  onChange={(e) => set({ lagBars: Number(e.target.value) })}
                />
              </div>
              <div>
                <Label htmlFor="price">Trade at</Label>
                <Select
                  id="price"
                  value={draft.price}
                  onChange={(e) => set({ price: e.target.value })}
                >
                  <option value="close">close</option>
                  <option value="open">open</option>
                </Select>
              </div>
              <div>
                <Label htmlFor="rebalance">Rebalance</Label>
                <Select
                  id="rebalance"
                  value={draft.rebalance}
                  onChange={(e) => set({ rebalance: e.target.value })}
                >
                  {['on_change', 'every_bar', 'weekly', 'monthly', 'quarterly', 'every_n'].map(
                    (r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ),
                  )}
                </Select>
              </div>
              {draft.rebalance === 'every_n' && (
                <div>
                  <Label htmlFor="every-n">Every N bars</Label>
                  <Input
                    id="every-n"
                    type="number"
                    min={1}
                    value={draft.rebalanceEveryN}
                    onChange={(e) => set({ rebalanceEveryN: Number(e.target.value) })}
                  />
                </div>
              )}
              <div>
                <Label htmlFor="cash">Cash rate (annual)</Label>
                <Input
                  id="cash"
                  type="number"
                  step="0.001"
                  value={draft.cashRate}
                  onChange={(e) => set({ cashRate: Number(e.target.value) })}
                />
              </div>
              {draft.engine === 'event' && (
                <>
                  <div>
                    <Label htmlFor="participation">Max participation</Label>
                    <Input
                      id="participation"
                      type="number"
                      step="0.01"
                      min={0.01}
                      max={1}
                      value={draft.maxParticipation}
                      onChange={(e) => set({ maxParticipation: Number(e.target.value) })}
                    />
                  </div>
                  <div>
                    <Label htmlFor="latency">Latency (bars)</Label>
                    <Input
                      id="latency"
                      type="number"
                      min={0}
                      value={draft.latencyBars}
                      onChange={(e) => set({ latencyBars: Number(e.target.value) })}
                    />
                  </div>
                </>
              )}
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <PluginList
                label="Cost models"
                kind="cost"
                plugins={ofKind('cost_model')}
                items={draft.costs}
                onChange={upd('costs')}
                sortable={false}
              />
              <div className="space-y-4">
                <PluginPicker
                  id="slippage"
                  label="Slippage model"
                  plugins={ofKind('slippage_model')}
                  allowNone
                  value={choice(draft.slippage)}
                  onChange={(c) => set({ slippage: toRef('slippage', c) })}
                />
                {draft.engine === 'event' && (
                  <PluginPicker
                    id="fill"
                    label="Fill model"
                    plugins={ofKind('fill_model')}
                    allowNone
                    noneLabel="Default"
                    value={choice(draft.fillModel)}
                    onChange={(c) => set({ fillModel: toRef('fill', c) })}
                  />
                )}
              </div>
            </div>
          </Step>

          <Step n={6} title="Benchmark, capital and research split">
            <div className="grid gap-3 md:grid-cols-3">
              <div>
                <Label htmlFor="bench">Benchmark</Label>
                <Input
                  id="bench"
                  value={draft.benchmark}
                  placeholder="SPY, source:SYMBOL or source:dataset:SYMBOL"
                  onChange={(e) => set({ benchmark: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="riskfree">Risk-free (cash earns it, Sharpe uses excess)</Label>
                <Input
                  id="riskfree"
                  value={draft.riskFree}
                  placeholder="source:dataset:field"
                  onChange={(e) => set({ riskFree: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="capital">Initial capital</Label>
                <Input
                  id="capital"
                  type="number"
                  value={draft.initialCapital}
                  onChange={(e) => set({ initialCapital: Number(e.target.value) })}
                />
              </div>
              <div>
                <Label htmlFor="seed">Random seed</Label>
                <Input
                  id="seed"
                  type="number"
                  value={draft.seed}
                  onChange={(e) => set({ seed: Number(e.target.value) })}
                />
              </div>
              <div>
                <Label htmlFor="experiment">Experiment</Label>
                <Input
                  id="experiment"
                  value={draft.experiment}
                  onChange={(e) => set({ experiment: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="train-end">Train end</Label>
                <Input
                  id="train-end"
                  type="date"
                  value={draft.trainEnd}
                  onChange={(e) => set({ trainEnd: e.target.value })}
                />
              </div>
              <div>
                <Label htmlFor="val-end">Validation end (test locked after)</Label>
                <Input
                  id="val-end"
                  type="date"
                  value={draft.validationEnd}
                  onChange={(e) => set({ validationEnd: e.target.value })}
                />
              </div>
              <div className="flex items-end pb-2">
                <Toggle
                  id="unlock"
                  checked={draft.testUnlocked}
                  onChange={(v) => set({ testUnlocked: v })}
                  label="Unlock test period (recorded)"
                />
              </div>
            </div>
          </Step>
        </div>

        <div className="space-y-4 xl:sticky xl:top-6 xl:self-start">
          <Card title="Run">
            <div className="space-y-3">
              <div>
                <Label htmlFor="run-name">Name</Label>
                <Input
                  id="run-name"
                  value={draft.name}
                  onChange={(e) => set({ name: e.target.value })}
                  placeholder="optional"
                />
              </div>
              <div>
                <Label htmlFor="tags">Tags</Label>
                <Input
                  id="tags"
                  value={draft.tags}
                  onChange={(e) => set({ tags: e.target.value })}
                  placeholder="comma separated"
                />
              </div>
              <div>
                <Label htmlFor="notes">Notes</Label>
                <textarea
                  id="notes"
                  value={draft.notes}
                  onChange={(e) => set({ notes: e.target.value })}
                  className="h-16 w-full rounded-md border border-border bg-canvas p-2 text-sm text-primary focus:border-accent focus:outline-none"
                />
              </div>
              <Button
                variant="primary"
                className="w-full"
                disabled={!valid}
                loading={launch.isPending}
                onClick={() => launch.mutate(config, { onSuccess: (res) => setJobId(res.job.id) })}
              >
                Run backtest
              </Button>
              {launch.isError && <ErrorState error={launch.error} />}
              {jobId && <JobProgress jobId={jobId} />}
            </div>
          </Card>
          <Card title="Validation">
            {validate.isPending && !validate.data ? (
              <Skeleton height={40} />
            ) : (
              <IssueList issues={issues} />
            )}
            {validate.isError && <ErrorState error={validate.error} />}
          </Card>
          <Card title="Checks and presets">
            <div className="space-y-3">
              <Button
                className="w-full"
                loading={lookahead.isPending}
                disabled={!valid}
                onClick={() => lookahead.mutate(config)}
              >
                Run lookahead check
              </Button>
              {lookahead.data && (
                <div className="text-sm">
                  <Badge tone={lookahead.data.passed ? 'positive' : 'negative'}>
                    {lookahead.data.passed ? 'passed' : 'failed'}
                  </Badge>{' '}
                  <span className="text-secondary">{lookahead.data.message}</span>
                  {lookahead.data.mismatches.slice(0, 3).map((m, i) => (
                    <pre
                      key={i}
                      className="mt-1 overflow-auto rounded bg-canvas p-1 text-2xs text-muted"
                    >
                      {JSON.stringify(m)}
                    </pre>
                  ))}
                </div>
              )}
              {lookahead.isError && <ErrorState error={lookahead.error} />}
              <div className="flex gap-2">
                <Input
                  aria-label="Preset name"
                  placeholder="Preset name"
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                />
                <Button
                  disabled={!presetName}
                  loading={savePreset.isPending}
                  onClick={() =>
                    savePreset.mutate({
                      name: presetName,
                      config: config,
                      notes: '',
                    })
                  }
                >
                  Save
                </Button>
              </div>
              {savePreset.isSuccess && <p className="text-2xs text-positive">Preset saved.</p>}
              {savePreset.isError && <ErrorState error={savePreset.error} />}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
